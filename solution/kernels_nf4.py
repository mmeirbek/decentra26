"""NF4 weights with double-quantized scales -> fp16, in one pass.

bitsandbytes stores every frozen projection as 4-bit NF4 codes (two per byte, high nibble first),
one uint8 scale per 64 weights, and those scales quantized once more: a float32 scale per 256 of
them, a 256-entry code book and a float32 offset. bitsandbytes rebuilds the fp32 scales in one
kernel and the weights in a second one, before every matmul and again in backward. This kernel
reads each packed byte once and writes the two fp16 weights it holds.

The arithmetic is the one bitsandbytes does (code book lookup, multiply, add the offset, multiply
by the NF4 level, round to fp16), with fused multiply-add switched off, so the weights come out bit
for bit the same.
"""

import torch
import triton
import triton.language as tl

BYTES_PER_PROGRAM = 1024
WEIGHTS_PER_SCALE = 64
SCALES_PER_SCALE2 = 256


@triton.jit(do_not_specialize=["n_bytes"])
def nf4_dequant_kernel(packed_ptr, scale_q_ptr, scale2_ptr, scale_code_ptr, offset_ptr, nf4_ptr,
                       out_ptr, n_bytes, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    live = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=live, other=0).to(tl.int32)
    group = offs // 32  # 64 weights share one scale, two weights per byte
    scale_q = tl.load(scale_q_ptr + group, mask=live, other=0).to(tl.int32)
    scale = tl.load(scale_code_ptr + scale_q, mask=live, other=0.0) * \
        tl.load(scale2_ptr + group // 256, mask=live, other=0.0)
    scale = scale + tl.load(offset_ptr)
    high = tl.load(nf4_ptr + (byte >> 4), mask=live, other=0.0) * scale
    low = tl.load(nf4_ptr + (byte & 15), mask=live, other=0.0) * scale
    pair = tl.interleave(high, low)
    out_offs = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offs, pair.to(out_ptr.dtype.element_ty), mask=out_offs < 2 * n_bytes)


class NF4Weight:
    """The tensors of one bitsandbytes Params4bit that the kernel reads, looked up once."""

    def __init__(self, param):
        qs = param.quant_state
        if qs.quant_type != "nf4" or not qs.nested or qs.blocksize != WEIGHTS_PER_SCALE \
                or qs.state2.blocksize != SCALES_PER_SCALE2 or param.data.dtype != torch.uint8:
            raise ValueError("needs NF4 weights with nested 8-bit scales and blocksize 64")
        self.shape = tuple(qs.shape)
        self.numel = self.shape[0] * self.shape[1]
        self.packed = param.data.view(-1)
        self.scale_q = qs.absmax
        self.scale2 = qs.state2.absmax
        self.scale_code = qs.state2.code
        self.offset = qs.offset if torch.is_tensor(qs.offset) else \
            torch.tensor(float(qs.offset), dtype=torch.float32, device=self.packed.device)
        self.nf4 = qs.code
        self.dtype = qs.dtype

    def dequant(self, out=None):
        """The weight as a dense (out_features, in_features) tensor in the compute dtype."""
        if out is None:
            out = torch.empty(self.shape, dtype=self.dtype, device=self.packed.device)
        n_bytes = self.numel // 2
        grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
        nf4_dequant_kernel[grid](self.packed, self.scale_q, self.scale2, self.scale_code,
                                 self.offset, self.nf4, out, n_bytes, BLOCK=BYTES_PER_PROGRAM,
                                 num_warps=4, enable_fp_fusion=False)
        return out


def dequant_rows(weights, dtype, device):
    """Several weights with the same in_features, stacked by rows into one dense tensor (one
    matmul then serves all of them)."""
    rows = sum(w.shape[0] for w in weights)
    out = torch.empty((rows, weights[0].shape[1]), dtype=dtype, device=device)
    start = 0
    for w in weights:
        w.dequant(out[start:start + w.shape[0]])
        start += w.shape[0]
    return out
