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


@triton.jit
def nf4_dequant_kernel(words_ptr, scale_q_ptr, scale2_ptr, scale_code_ptr, nf4_ptr, out_ptr,
                       prog_weight_ptr, word_start_ptr, scale_q_start_ptr, scale2_start_ptr,
                       offset_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # The weights of a group sit back to back; each program stays inside one of them.
    w = tl.load(prog_weight_ptr + pid)
    word_idx = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    word = tl.load(words_ptr + word_idx)
    group = (word_idx - tl.load(word_start_ptr + w)) // 8  # 64 weights = 8 words share one scale
    scale_q = tl.load(scale_q_ptr + tl.load(scale_q_start_ptr + w) + group).to(tl.int32)
    scale = tl.load(scale_code_ptr + scale_q) * \
        tl.load(scale2_ptr + tl.load(scale2_start_ptr + w) + group // 256)
    scale = scale + tl.load(offset_ptr + w)
    # Weight k of a little-endian word sits in byte k // 2, high nibble first.
    k = tl.arange(0, 8)
    shift = (k // 2) * 8 + (1 - k % 2) * 4
    code = (word[:, None] >> shift[None, :]) & 15
    weights = tl.load(nf4_ptr + code) * scale[:, None]
    tl.store(out_ptr + word_idx[:, None] * 8 + k[None, :], weights.to(out_ptr.dtype.element_ty))


def _check(param):
    qs = param.quant_state
    if qs.quant_type != "nf4" or not qs.nested or qs.blocksize != WEIGHTS_PER_SCALE \
            or qs.state2.blocksize != SCALES_PER_SCALE2 or param.data.dtype != torch.uint8:
        raise ValueError("needs NF4 weights with nested 8-bit scales and blocksize 64")
    rows, cols = qs.shape
    if (rows * cols) % (8 * WORDS_PER_PROGRAM):
        raise ValueError("weight size is not a whole number of dequant programs")
    return qs


class NF4Group:
    """bitsandbytes Params4bit weights with the same in_features, dequantized by one launch into
    one (sum of out_features, in_features) tensor, so one matmul serves them all (q|k|v, gate|up).

    With `share=True` the packed codes and the 8-bit scales of the group are moved into one
    buffer each and every Params4bit (and its quant state) is pointed at its slice of them:
    the same values, no second copy, and bitsandbytes reads them as before.

    With `host` (offload.HostWeights, codes of `layer` placed back to back in its block) the codes
    stay on the host: `words_start` / `n_words` locate them in the layer's block, and each
    dequant reads them from the GPU slot the caller passes (offload.WeightStream)."""

    def __init__(self, params, share=False, host=None, layer=0, names=()):
        states = [_check(p) for p in params]
        cols = states[0].shape[1]
        if any(qs.shape[1] != cols or qs.dtype != states[0].dtype for qs in states):
            raise ValueError("a group needs one in_features and one dtype")
        if any(not torch.equal(qs.code, states[0].code)
               or not torch.equal(qs.state2.code, states[0].state2.code) for qs in states[1:]):
            raise ValueError("a group needs one NF4 table and one scale code book")
        dev = states[0].absmax.device  # the scales stay on the GPU when the codes do not
        self.shape = (sum(qs.shape[0] for qs in states), cols)
        self.numel = self.shape[0] * cols
        self.dtype = states[0].dtype
        self.nf4, self.scale_code = states[0].code, states[0].state2.code
        words = [p.data.view(-1).view(torch.int32) for p in params]
        scale_q = [qs.absmax for qs in states]
        scale2 = [qs.state2.absmax for qs in states]
        self.words_start = self.n_words = None
        if host is not None:
            starts = [host.offsets[name][0] for name in names]
            if any(not host.is_placed(p, layer, name) for p, name in zip(params, names)) or any(
                    a + p.data.numel() != b for a, b, p in zip(starts, starts[1:], params)):
                raise ValueError("the group's codes are not back to back in the host block")
            self.words_start, self.n_words = starts[0] // 4, sum(w.numel() for w in words)
        if len(params) == 1:
            self.words, self.scale_q, self.scale2 = words[0], scale_q[0], scale2[0]
        elif share:
            (self.scale_q, shared_q), (self.scale2, shared_2) = (
                self._share(parts) for parts in (scale_q, scale2))
            if host is None:
                self.words, shared_words = self._share(words)
                for p, wv in zip(params, shared_words):
                    p.data = wv.view(torch.uint8).view(p.data.shape)
            else:  # the codes already sit back to back in the host block
                self.words = None
            for qs, sq, s2 in zip(states, shared_q, shared_2):
                qs.absmax, qs.state2.absmax = sq, s2
        else:  # a copy for the kernel next to bitsandbytes' own tensors (tests)
            self.words, self.scale_q, self.scale2 = (torch.cat(parts)
                                                     for parts in (words, scale_q, scale2))
        offsets = [float(qs.offset) for qs in states]

        def starts(parts):
            out, at = [], 0
            for part in parts:
                out.append(at)
                at += part.numel()
            return out

        programs = [n.numel() // WORDS_PER_PROGRAM for n in words]
        self.prog_weight = torch.repeat_interleave(torch.arange(len(params), dtype=torch.int32),
                                                   torch.tensor(programs)).to(dev)
        self.word_start = torch.tensor(starts(words), dtype=torch.int64, device=dev)
        self.scale_q_start = torch.tensor(starts(scale_q), dtype=torch.int64, device=dev)
        self.scale2_start = torch.tensor(starts(scale2), dtype=torch.int64, device=dev)
        self.offset = torch.tensor(offsets, dtype=torch.float32, device=dev)
        self.programs = sum(programs)

    @staticmethod
    def _share(parts):
        """One buffer holding `parts` back to back, and views of it to use in their place (the
        originals are then dropped, so the group costs no memory of its own)."""
        whole = torch.cat([t.reshape(-1) for t in parts])
        views, at = [], 0
        for t in parts:
            views.append(whole[at:at + t.numel()].view(t.shape))
            at += t.numel()
        return whole, views

    def dequant(self, out=None, words=None):
        """The weights as one dense (sum of out_features, in_features) tensor in compute dtype;
        `words`: the group's codes on the GPU when they live on the host."""
        words = self.words if words is None else words
        if words is None or not words.is_cuda and self.scale_q.is_cuda:
            raise RuntimeError("the codes of this group are on the host: pass their GPU copy")
        if out is None:
            out = torch.empty(self.shape, dtype=self.dtype, device=self.scale_q.device)
        nf4_dequant_kernel[(self.programs,)](
            words, self.scale_q, self.scale2, self.scale_code, self.nf4, out,
            self.prog_weight, self.word_start, self.scale_q_start, self.scale2_start, self.offset,
            BLOCK=WORDS_PER_PROGRAM, num_warps=NF4_WARPS, enable_fp_fusion=False)
        return out


def dequant_group(group, workspace=None, words=None):
    """`group.dequant()` into a view of `workspace` when given: every dequant of a step writes
    the same buffer, which the matmul reading it has finished with by then (one CUDA stream), so
    the allocator never sees these 5-55 MB temporaries."""
    if workspace is None:
        return group.dequant(words=words)
    rows, cols = group.shape
    return group.dequant(workspace[:rows * cols].view(rows, cols), words)
