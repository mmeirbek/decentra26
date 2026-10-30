"""The gated MLP activation of Qwen2, on the fused gate|up matmul output, forward and backward.

gate_proj and up_proj share their input, so one matmul against both weights stacked by rows gives
a (tokens, 2 * intermediate) fp16 buffer: gate in the first half of each row, up in the second.
Base Soup keeps gate, SiLU(gate) and up for backward in fp32 (three 8,960-wide tensors per token per
layer, 5.9 GB at the peak); this keeps only the fp16 buffer, which holds the very values the
base casts to fp32.

Forward writes act = SiLU(gate) * up in fp16 for down_proj. Backward reads the gradient of act and
overwrites the buffer in place with the gradients of gate and up, so backward allocates nothing
8,960 wide. The arithmetic is torch's: SiLU(g) = g / (1 + exp(-g)), and
dgate = (dact * up) * sigmoid(g) * (1 + g * (1 - sigmoid(g))), dup = dact * SiLU(g).
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["backward"])
def gated_act_kernel(gu_ptr, act_ptr, backward, I: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = cols < I
    gate_at = gu_ptr + tok.to(tl.int64) * (2 * I) + cols
    up_at = gate_at + I
    act_at = act_ptr + tok.to(tl.int64) * I + cols
    g = tl.load(gate_at, mask=live, other=0.0).to(tl.float32)
    u = tl.load(up_at, mask=live, other=0.0).to(tl.float32)
    denom = 1.0 + tl.exp(-g)
    silu = g / denom
    if backward != 0:
        dact = tl.load(act_at, mask=live, other=0.0).to(tl.float32)
        sig = 1.0 / denom
        dsilu = dact * u
        tl.store(gate_at, (dsilu * sig * (1.0 + g * (1.0 - sig))).to(tl.float16), mask=live)
        tl.store(up_at, (dact * silu).to(tl.float16), mask=live)
    else:
        tl.store(act_at, (silu * u).to(tl.float16), mask=live)
