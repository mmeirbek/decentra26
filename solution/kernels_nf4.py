"""NF4 weights with double-quantized scales -> fp16, in one pass.

bitsandbytes stores every frozen projection as 4-bit NF4 codes (two per byte, high nibble first),
one uint8 scale per 64 weights, and those scales quantized once more: a float32 scale per 256 of
them, a 256-entry code book and a float32 offset. bitsandbytes rebuilds the fp32 scales in one
kernel and the weights in a second one, before every matmul and again in backward. Here each
thread reads 32-bit words of codes (8 weights) and writes the 8 fp16 weights next to each other,
scales included, in one pass.

The arithmetic is the one bitsandbytes does (code book lookup, multiply, add the offset, multiply
by the NF4 level, round to fp16), with fused multiply-add switched off, so the weights come out bit
for bit the same.
"""

import torch
import triton
import triton.language as tl

WORDS_PER_PROGRAM = 256  # 32-bit words of packed codes, 8 weights each (fastest on a T4)
NF4_WARPS = 8
WEIGHTS_PER_SCALE = 64
SCALES_PER_SCALE2 = 256


@triton.jit(do_not_specialize=["n_words"])
def nf4_dequant_kernel(words_ptr, scale_q_ptr, scale2_ptr, scale_code_ptr, offset_ptr, nf4_ptr,
                       out_ptr, n_words, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    word_idx = pid * BLOCK + tl.arange(0, BLOCK)
    live = word_idx < n_words
    word = tl.load(words_ptr + word_idx, mask=live, other=0)
    group = word_idx // 8  # 64 weights = 8 words share one scale
    scale_q = tl.load(scale_q_ptr + group, mask=live, other=0).to(tl.int32)
    scale = tl.load(scale_code_ptr + scale_q, mask=live, other=0.0) * \
        tl.load(scale2_ptr + group // 256, mask=live, other=0.0)
    scale = scale + tl.load(offset_ptr)
    # Weight k of a little-endian word sits in byte k // 2, high nibble first.
    k = tl.arange(0, 8)
    shift = (k // 2) * 8 + (1 - k % 2) * 4
    code = (word[:, None] >> shift[None, :]) & 15
    weights = tl.load(nf4_ptr + code) * scale[:, None]
    out_offs = word_idx.to(tl.int64)[:, None] * 8 + k[None, :]
    tl.store(out_ptr + out_offs, weights.to(out_ptr.dtype.element_ty), mask=live[:, None])


class NF4Weight:
    """The tensors of one bitsandbytes Params4bit that the kernel reads, looked up once."""

    def __init__(self, param):
        qs = param.quant_state
        if qs.quant_type != "nf4" or not qs.nested or qs.blocksize != WEIGHTS_PER_SCALE \
                or qs.state2.blocksize != SCALES_PER_SCALE2 or param.data.dtype != torch.uint8:
            raise ValueError("needs NF4 weights with nested 8-bit scales and blocksize 64")
        self.shape = tuple(qs.shape)
        self.numel = self.shape[0] * self.shape[1]
        if self.numel % 8:
            raise ValueError("needs whole 32-bit words of packed codes")
        self.words = param.data.view(-1).view(torch.int32)
        self.scale_q = qs.absmax
        self.scale2 = qs.state2.absmax
        self.scale_code = qs.state2.code
        self.offset = qs.offset if torch.is_tensor(qs.offset) else \
            torch.tensor(float(qs.offset), dtype=torch.float32, device=self.words.device)
        self.nf4 = qs.code
        self.dtype = qs.dtype

    def dequant(self, out=None):
        """The weight as a dense (out_features, in_features) tensor in the compute dtype."""
        if out is None:
            out = torch.empty(self.shape, dtype=self.dtype, device=self.words.device)
        n_words = self.numel // 8
        grid = (triton.cdiv(n_words, WORDS_PER_PROGRAM),)
        nf4_dequant_kernel[grid](self.words, self.scale_q, self.scale2, self.scale_code,
                                 self.offset, self.nf4, out, n_words, BLOCK=WORDS_PER_PROGRAM,
                                 num_warps=NF4_WARPS, enable_fp_fusion=False)
        return out


def dequant_rows(weights, dtype, device, workspace=None):
    """Several weights with the same in_features, stacked by rows into one dense tensor (one
    matmul then serves all of them). With `workspace`, the result is a view of it: every dequant
    of a step writes the same buffer, which the matmul reading it has finished with by then (one
    CUDA stream), so the allocator never sees these 5-55 MB temporaries."""
    rows = sum(w.shape[0] for w in weights)
    cols = weights[0].shape[1]
    if workspace is not None:
        out = workspace[:rows * cols].view(rows, cols)
    else:
        out = torch.empty((rows, cols), dtype=dtype, device=device)
    start = 0
    for w in weights:
        w.dequant(out[start:start + w.shape[0]])
        start += w.shape[0]
    return out
