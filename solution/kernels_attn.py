"""The tail of q_proj / k_proj / v_proj: LoRA add, RoPE, layout for attention. Forward and backward.

In base Soup each projection's fp16 output is cast to fp32, cloned by peft, added to the LoRA
branch, rotated by RoPE through slice / neg / cat, then cast back to fp16 inside SDPA: about ten
passes over q, k and v. Here one program per (token, head) reads the fused q|k|v matmul output and
the LoRA outputs and writes the rotated fp16 head straight into the padded (batch, seq, head, dim)
layout SDPA takes, with the same fp32 arithmetic:

    q = fp32(base) + fp32(fp16(lora * scaling));  q_rot = q * cos + rotate_half(q) * sin
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seq_len"])
def qkv_rope_fwd_kernel(qkv_ptr, lora_q_ptr, lora_v_ptr, cos_ptr, sin_ptr, rowmap_ptr,
                        q_ptr, k_ptr, v_ptr, scaling_q, scaling_v, seq_len,
                        NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr):
    tok = tl.program_id(0)
    head = tl.program_id(1)  # NQ query heads, then NKV key heads, then NKV value heads
    HALF: tl.constexpr = D // 2
    WIDTH: tl.constexpr = (NQ + 2 * NKV) * D
    pos = tl.load(rowmap_ptr + tok).to(tl.int64)
    t = pos % seq_len
    lo = tl.arange(0, HALF)
    src = qkv_ptr + tok.to(tl.int64) * WIDTH + head * D
    if head < NQ + NKV:
        x1 = tl.load(src + lo).to(tl.float32)
        x2 = tl.load(src + HALF + lo).to(tl.float32)
        if head < NQ:
            lora = lora_q_ptr + tok.to(tl.int64) * (NQ * D) + head * D
            x1 = x1 + (tl.load(lora + lo).to(tl.float32) * scaling_q).to(tl.float16).to(tl.float32)
            x2 = x2 + (tl.load(lora + HALF + lo).to(tl.float32) * scaling_q).to(tl.float16).to(tl.float32)
            dst = q_ptr + (pos * NQ + head) * D
        else:
            dst = k_ptr + (pos * NKV + head - NQ) * D
        c1 = tl.load(cos_ptr + t * D + lo)
        c2 = tl.load(cos_ptr + t * D + HALF + lo)
        s1 = tl.load(sin_ptr + t * D + lo)
        s2 = tl.load(sin_ptr + t * D + HALF + lo)
        tl.store(dst + lo, (x1 * c1 + (-x2) * s1).to(tl.float16))
        tl.store(dst + HALF + lo, (x2 * c2 + x1 * s2).to(tl.float16))
    else:
        vh = head - NQ - NKV
        full = tl.arange(0, D)
        x = tl.load(src + full).to(tl.float32)
        lora = lora_v_ptr + tok.to(tl.int64) * (NKV * D) + vh * D
        x = x + (tl.load(lora + full).to(tl.float32) * scaling_v).to(tl.float16).to(tl.float32)
        tl.store(v_ptr + (pos * NKV + vh) * D + full, x.to(tl.float16))


@triton.jit(do_not_specialize=["seq_len"])
def qkv_rope_bwd_kernel(dq_ptr, dk_ptr, dv_ptr, cos_ptr, sin_ptr, rowmap_ptr,
                        dqkv_ptr, dlora_q_ptr, dlora_v_ptr, scaling_q, scaling_v, seq_len,
                        NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    WIDTH: tl.constexpr = (NQ + 2 * NKV) * D
    pos = tl.load(rowmap_ptr + tok).to(tl.int64)
    t = pos % seq_len
    lo = tl.arange(0, HALF)
    dst = dqkv_ptr + tok.to(tl.int64) * WIDTH + head * D
    if head < NQ + NKV:
        if head < NQ:
            src = dq_ptr + (pos * NQ + head) * D
        else:
            src = dk_ptr + (pos * NKV + head - NQ) * D
        g1 = tl.load(src + lo).to(tl.float32)
        g2 = tl.load(src + HALF + lo).to(tl.float32)
        c1 = tl.load(cos_ptr + t * D + lo)
        c2 = tl.load(cos_ptr + t * D + HALF + lo)
        s1 = tl.load(sin_ptr + t * D + lo)
        s2 = tl.load(sin_ptr + t * D + HALF + lo)
        d1 = (g1 * c1 + g2 * s2).to(tl.float16)
        d2 = (g2 * c2 + (-(g1 * s1))).to(tl.float16)
        tl.store(dst + lo, d1)
        tl.store(dst + HALF + lo, d2)
        if head < NQ:
            lora = dlora_q_ptr + tok.to(tl.int64) * (NQ * D) + head * D
            tl.store(lora + lo, (d1.to(tl.float32) * scaling_q).to(tl.float16))
            tl.store(lora + HALF + lo, (d2.to(tl.float32) * scaling_q).to(tl.float16))
    else:
        vh = head - NQ - NKV
        full = tl.arange(0, D)
        d = tl.load(dv_ptr + (pos * NKV + vh) * D + full).to(tl.float16)
        tl.store(dst + full, d)
        lora = dlora_v_ptr + tok.to(tl.int64) * (NKV * D) + vh * D
        tl.store(lora + full, (d.to(tl.float32) * scaling_v).to(tl.float16))
