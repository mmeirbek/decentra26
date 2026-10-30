"""Residual add + RMSNorm, forward and backward, one program per token.

Base Soup keeps the residual stream, the RMSNorm output and everything between the matmuls in fp32
(peft upcasts the norm weights, bitsandbytes returns its output in the input's dtype), yet every
matmul casts its input back to fp16. These kernels keep the residual in fp32 exactly as the base
does and hand the matmuls the fp16 they would have received anyway:

    h = residual + delta                 (fp32, as `residual + hidden_states`)
    xhat = h * rsqrt(mean(h^2) + eps);  x = weight * xhat   -> fp16 for the next matmul

For backward only xhat (fp16) and 1/rms are kept, not the fp32 residual: that is all the RMSNorm
gradient reads.

Before attention (with_lora) the same pass also writes the LoRA inputs of q_proj and v_proj,
x * dropout_mask * dropout_scale in fp16, with the masks drawn by torch's own dropout (so the
randomness is the base run's), and keeps those masks for backward as bits of the real tokens only.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["with_lora"])
def add_rmsnorm_fwd_kernel(res_ptr, delta_ptr, w_ptr, h_ptr, rstd_ptr, x_ptr, xhat_ptr,
                           rowmap_ptr, keep_q_ptr, keep_v_ptr, drop_scale_ptr, xq_ptr, xv_ptr,
                           bits_q_ptr, bits_v_ptr, eps, with_lora,
                           H: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    live = cols < H
    offs = tok.to(tl.int64) * H + cols
    h = tl.load(res_ptr + offs, mask=live, other=0.0) + \
        tl.load(delta_ptr + offs, mask=live, other=0.0).to(tl.float32)
    tl.store(h_ptr + offs, h, mask=live)
    rstd = tl.math.rsqrt(tl.sum(h * h, axis=0) / H + eps)
    tl.store(rstd_ptr + tok, rstd)
    xhat = h * rstd
    tl.store(xhat_ptr + offs, xhat.to(tl.float16), mask=live)
    y = tl.load(w_ptr + cols, mask=live, other=0.0) * xhat
    tl.store(x_ptr + offs, y.to(tl.float16), mask=live)
    if with_lora != 0:
        pos = tl.load(rowmap_ptr + tok).to(tl.int64) * H + cols
        drop_scale = tl.load(drop_scale_ptr)
        keep_q = tl.load(keep_q_ptr + pos, mask=live, other=0) != 0
        keep_v = tl.load(keep_v_ptr + pos, mask=live, other=0) != 0
        tl.store(xq_ptr + offs, tl.where(keep_q, y * drop_scale, 0.0).to(tl.float16), mask=live)
        tl.store(xv_ptr + offs, tl.where(keep_v, y * drop_scale, 0.0).to(tl.float16), mask=live)
        # eight mask entries per byte, entry 8 * j + b in bit b of byte j
        bit = tl.arange(0, 8)[None, :]
        byte_cols = tl.arange(0, BLOCK // 8)
        byte_offs = tok.to(tl.int64) * (H // 8) + byte_cols
        byte_live = byte_cols < H // 8
        packed_q = tl.sum(tl.reshape(keep_q.to(tl.int32), (BLOCK // 8, 8)) << bit, axis=1)
        packed_v = tl.sum(tl.reshape(keep_v.to(tl.int32), (BLOCK // 8, 8)) << bit, axis=1)
        tl.store(bits_q_ptr + byte_offs, packed_q.to(tl.uint8), mask=byte_live)
        tl.store(bits_v_ptr + byte_offs, packed_v.to(tl.uint8), mask=byte_live)


@triton.jit(do_not_specialize=["with_lora"])
def add_rmsnorm_bwd_kernel(dx_ptr, xhat_ptr, rstd_ptr, w_ptr, dres_out_ptr, dres_ptr, ddelta_ptr,
                           bits_q_ptr, bits_v_ptr, drop_scale_ptr, dxq_ptr, dxv_ptr, with_lora,
                           H: tl.constexpr, BLOCK: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    live = cols < H
    offs = tok.to(tl.int64) * H + cols
    dy = tl.load(dx_ptr + offs, mask=live, other=0.0).to(tl.float32)
    if with_lora != 0:
        bit = tl.arange(0, 8)[None, :]
        byte_cols = tl.arange(0, BLOCK // 8)
        byte_offs = tok.to(tl.int64) * (H // 8) + byte_cols
        byte_live = byte_cols < H // 8
        packed_q = tl.load(bits_q_ptr + byte_offs, mask=byte_live, other=0).to(tl.int32)
        packed_v = tl.load(bits_v_ptr + byte_offs, mask=byte_live, other=0).to(tl.int32)
        keep_q = tl.reshape((packed_q[:, None] >> bit) & 1, (BLOCK,)) != 0
        keep_v = tl.reshape((packed_v[:, None] >> bit) & 1, (BLOCK,)) != 0
        drop_scale = tl.load(drop_scale_ptr)
        dq = tl.load(dxq_ptr + offs, mask=live, other=0.0).to(tl.float32)
        dv = tl.load(dxv_ptr + offs, mask=live, other=0.0).to(tl.float32)
        dy = dy + tl.where(keep_q, dq * drop_scale, 0.0) + tl.where(keep_v, dv * drop_scale, 0.0)
    rstd = tl.load(rstd_ptr + tok)
    xhat = tl.load(xhat_ptr + offs, mask=live, other=0.0).to(tl.float32)
    dxhat = dy * tl.load(w_ptr + cols, mask=live, other=0.0)
    proj = tl.sum(dxhat * xhat, axis=0) / H
    dres = rstd * (dxhat - xhat * proj) + tl.load(dres_out_ptr + offs, mask=live, other=0.0)
    tl.store(dres_ptr + offs, dres, mask=live)
    tl.store(ddelta_ptr + offs, dres.to(tl.float16), mask=live)


@triton.jit
def lora_inputs_kernel(xhat_ptr, w_ptr, bits_q_ptr, bits_v_ptr, drop_scale_ptr, xq_ptr, xv_ptr,
                       H: tl.constexpr, BLOCK: tl.constexpr):
    """Backward's copy of the q_proj / v_proj LoRA inputs, x * mask * dropout_scale, from the kept
    xhat and mask bits (the forward pass does not keep them)."""
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    live = cols < H
    offs = tok.to(tl.int64) * H + cols
    y = tl.load(w_ptr + cols, mask=live, other=0.0) * \
        tl.load(xhat_ptr + offs, mask=live, other=0.0).to(tl.float32)
    bit = tl.arange(0, 8)[None, :]
    byte_cols = tl.arange(0, BLOCK // 8)
    byte_offs = tok.to(tl.int64) * (H // 8) + byte_cols
    byte_live = byte_cols < H // 8
    packed_q = tl.load(bits_q_ptr + byte_offs, mask=byte_live, other=0).to(tl.int32)
    packed_v = tl.load(bits_v_ptr + byte_offs, mask=byte_live, other=0).to(tl.int32)
    keep_q = tl.reshape((packed_q[:, None] >> bit) & 1, (BLOCK,)) != 0
    keep_v = tl.reshape((packed_v[:, None] >> bit) & 1, (BLOCK,)) != 0
    drop_scale = tl.load(drop_scale_ptr)
    tl.store(xq_ptr + offs, tl.where(keep_q, y * drop_scale, 0.0).to(tl.float16), mask=live)
    tl.store(xv_ptr + offs, tl.where(keep_v, y * drop_scale, 0.0).to(tl.float16), mask=live)
