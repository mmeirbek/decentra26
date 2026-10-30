"""Residual add + RMSNorm, forward and backward, one program per token.

Base Soup keeps the residual stream, the RMSNorm output and everything between the matmuls in fp32
(peft upcasts the norm weights, bitsandbytes returns its output in the input's dtype), yet every
matmul casts its input back to fp16. These kernels keep the residual in fp32 exactly as the base
does and hand the matmuls the fp16 they would have received anyway:

    h = residual + delta                 (fp32, as `residual + hidden_states`)
    x = weight * (h * rsqrt(mean(h^2) + eps))   -> fp16 for the q/k/v or gate/up matmul

Before attention (WITH_LORA) the same pass also writes the LoRA inputs of q_proj and v_proj:
x * dropout_mask * dropout_scale in fp16, with the masks drawn by torch's own dropout, so the
randomness is the base run's.
"""

import triton
import triton.language as tl


@triton.jit
def add_rmsnorm_fwd_kernel(res_ptr, delta_ptr, w_ptr, h_ptr, rstd_ptr, x_ptr,
                           rowmap_ptr, keep_q_ptr, keep_v_ptr, drop_scale_ptr, xq_ptr, xv_ptr,
                           eps, H: tl.constexpr, BLOCK: tl.constexpr, WITH_LORA: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    live = cols < H
    offs = tok.to(tl.int64) * H + cols
    h = tl.load(res_ptr + offs, mask=live, other=0.0) + \
        tl.load(delta_ptr + offs, mask=live, other=0.0).to(tl.float32)
    tl.store(h_ptr + offs, h, mask=live)
    rstd = tl.math.rsqrt(tl.sum(h * h, axis=0) / H + eps)
    tl.store(rstd_ptr + tok, rstd)
    y = tl.load(w_ptr + cols, mask=live, other=0.0) * (h * rstd)
    tl.store(x_ptr + offs, y.to(tl.float16), mask=live)
    if WITH_LORA:
        pos = tl.load(rowmap_ptr + tok).to(tl.int64) * H + cols
        drop_scale = tl.load(drop_scale_ptr)
        keep_q = tl.load(keep_q_ptr + pos, mask=live, other=0) != 0
        keep_v = tl.load(keep_v_ptr + pos, mask=live, other=0) != 0
        tl.store(xq_ptr + offs, tl.where(keep_q, y * drop_scale, 0.0).to(tl.float16), mask=live)
        tl.store(xv_ptr + offs, tl.where(keep_v, y * drop_scale, 0.0).to(tl.float16), mask=live)


@triton.jit
def add_rmsnorm_bwd_kernel(dx_ptr, h_ptr, rstd_ptr, w_ptr, dres_out_ptr, dres_ptr, ddelta_ptr,
                           rowmap_ptr, keep_q_ptr, keep_v_ptr, drop_scale_ptr, dxq_ptr, dxv_ptr,
                           H: tl.constexpr, BLOCK: tl.constexpr, WITH_LORA: tl.constexpr):
    tok = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    live = cols < H
    offs = tok.to(tl.int64) * H + cols
    dy = tl.load(dx_ptr + offs, mask=live, other=0.0).to(tl.float32)
    if WITH_LORA:
        pos = tl.load(rowmap_ptr + tok).to(tl.int64) * H + cols
        drop_scale = tl.load(drop_scale_ptr)
        keep_q = tl.load(keep_q_ptr + pos, mask=live, other=0) != 0
        keep_v = tl.load(keep_v_ptr + pos, mask=live, other=0) != 0
        dq = tl.load(dxq_ptr + offs, mask=live, other=0.0).to(tl.float32)
        dv = tl.load(dxv_ptr + offs, mask=live, other=0.0).to(tl.float32)
        dy = dy + tl.where(keep_q, dq * drop_scale, 0.0) + tl.where(keep_v, dv * drop_scale, 0.0)
    rstd = tl.load(rstd_ptr + tok)
    xhat = tl.load(h_ptr + offs, mask=live, other=0.0) * rstd
    dxhat = dy * tl.load(w_ptr + cols, mask=live, other=0.0)
    proj = tl.sum(dxhat * xhat, axis=0) / H
    dres = rstd * (dxhat - xhat * proj) + tl.load(dres_out_ptr + offs, mask=live, other=0.0)
    tl.store(dres_ptr + offs, dres, mask=live)
    tl.store(ddelta_ptr + offs, dres.to(tl.float16), mask=live)
