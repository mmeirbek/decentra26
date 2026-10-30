"""Dev-only: time our kernels on a T4 for several block sizes and warp counts, on tensors of the
real sizes (1,000 tokens of Qwen2.5-1.5B). dev/kaggle_push.py --tune ships it as solution/_tune.py
and runs `python -m solution._tune`; the winners become constants in solution/. Never part of the
submitted solution.
"""

import itertools

import torch
import triton

from solution.kernels_attn import qkv_rope_bwd_kernel, qkv_rope_fwd_kernel
from solution.kernels_loss import lm_loss_kernel
from solution.kernels_mlp import gated_act_kernel
from solution.kernels_nf4 import nf4_dequant_kernel
from solution.kernels_norm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel

T, H, I, V, NQ, NKV, D = 1000, 1536, 8960, 151936, 12, 2, 128
dev = "cuda"


def bench(fn, reps=30):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def report(name, rows):
    rows.sort(key=lambda r: r[1])
    best = rows[0][1]
    print(f"\n[tune] {name}")
    for cfg, ms, gbs in rows:
        print(f"[tune]   {cfg:32s} {ms * 1000:9.1f} us  {gbs:6.0f} GB/s  {'<- best' if ms == best else ''}")


def nf4():
    n = I * H
    words = torch.randint(-2**31, 2**31 - 1, (n // 8,), dtype=torch.int32, device=dev)
    scale_q = torch.randint(0, 255, (n // 64,), dtype=torch.uint8, device=dev)
    scale2 = torch.rand(n // 64 // 256 + 1, device=dev)
    code = torch.rand(256, device=dev)
    offset = torch.rand(1, device=dev)
    nf4 = torch.rand(16, device=dev)
    out = torch.empty(n, dtype=torch.float16, device=dev)
    rows = []
    zero = torch.zeros(1, dtype=torch.int64, device=dev)  # one weight: every start at 0
    for block, warps in itertools.product((256, 512, 1024, 2048), (2, 4, 8)):
        prog_weight = torch.zeros(n // 8 // block, dtype=torch.int32, device=dev)
        ms = bench(lambda: nf4_dequant_kernel[(n // 8 // block,)](
            words, scale_q, scale2, code, nf4, out, prog_weight, zero, zero, zero, offset,
            BLOCK=block, num_warps=warps, enable_fp_fusion=False))
        rows.append((f"WORDS={block} warps={warps}", ms, (n // 2 + 2 * n) / ms / 1e6))
    report("nf4_dequant (gate_proj weight)", rows)


def gated():
    gu = torch.randn(T, 2 * I, device=dev).half()
    act = torch.empty(T, I, dtype=torch.float16, device=dev)
    for backward in (False, True):
        rows = []
        for block, warps in itertools.product((512, 1024, 2048, 4096), (2, 4, 8)):
            ms = bench(lambda: gated_act_kernel[(T, triton.cdiv(I, block))](
                gu, act, int(backward), I=I, BLOCK=block, num_warps=warps))
            moved = T * I * 2 * (5 if backward else 3)
            rows.append((f"BLOCK={block} warps={warps}", ms, moved / ms / 1e6))
        report(f"gated_act {'backward' if backward else 'forward'}", rows)


def norms():
    res = torch.randn(T, H, device=dev)
    delta = torch.randn(T, H, device=dev).half()
    w = torch.rand(H, device=dev)
    rowmap = torch.arange(T, device=dev)
    keep = torch.rand(T, H, device=dev) > 0.05
    scale = torch.ones(1, device=dev)
    h = torch.empty_like(res)
    rstd = torch.empty(T, device=dev)
    x, xhat, xq, xv = (torch.empty(T, H, dtype=torch.float16, device=dev) for _ in range(4))
    bq, bv = (torch.empty(T, H // 8, dtype=torch.uint8, device=dev) for _ in range(2))
    dres, dres_out = torch.empty_like(res), torch.randn(T, H, device=dev)
    for with_lora in (True, False):
        rows = []
        for warps in (2, 4, 8, 16):
            ms = bench(lambda: add_rmsnorm_fwd_kernel[(T,)](
                res, delta, w, h, rstd, x, xhat, rowmap, keep, keep, scale, xq, xv, bq, bv, 1e-6,
                int(with_lora), H=H, BLOCK=2048, num_warps=warps))
            rows.append((f"fwd warps={warps}", ms, T * H * (4 + 2 + 4 + 2 + 2 + (5 if with_lora else 0)) / ms / 1e6))
            ms = bench(lambda: add_rmsnorm_bwd_kernel[(T,)](
                x, xhat, rstd, w, dres_out, dres, xv, bq, bv, scale, xq, xv, int(with_lora),
                H=H, BLOCK=2048, num_warps=warps))
            rows.append((f"bwd warps={warps}", ms, T * H * (2 + 2 + 4 + 4 + 2 + (4 if with_lora else 0)) / ms / 1e6))
        report(f"add_rmsnorm with_lora={with_lora}", rows)


def loss():
    rows_n = 256
    logits = torch.randn(rows_n, V, device=dev).half()
    target = torch.randint(0, V, (rows_n,), device=dev)
    gscale = torch.ones(1, device=dev)
    out = [torch.empty(rows_n, device=dev) for _ in range(2)] + \
        [torch.empty(rows_n, dtype=torch.int32, device=dev)]
    rows = []
    for block, warps in itertools.product((2048, 4096, 8192), (4, 8, 16)):
        work = logits.clone()
        ms = bench(lambda: lm_loss_kernel[(rows_n,)](work, target, gscale, *out, V, 1, V=V,
                                                     BLOCK=block, num_warps=warps))
        rows.append((f"BLOCK={block} warps={warps}", ms, rows_n * V * 6 / ms / 1e6))
    report("lm_loss (256 tokens, with grad)", rows)


def rope():
    qkv = torch.randn(T, (NQ + 2 * NKV) * D, device=dev).half()
    lq, lv = torch.randn(T, NQ * D, device=dev).half(), torch.randn(T, NKV * D, device=dev).half()
    cos, sin = torch.rand(512, D, device=dev), torch.rand(512, D, device=dev)
    rowmap = torch.arange(T, device=dev)
    q = torch.empty(T, NQ, D, dtype=torch.float16, device=dev)
    k, v = (torch.empty(T, NKV, D, dtype=torch.float16, device=dev) for _ in range(2))
    rows = []
    for warps in (1, 2, 4):
        ms = bench(lambda: qkv_rope_fwd_kernel[(T, NQ + 2 * NKV)](
            qkv, lq, lv, cos, sin, rowmap, q, k, v, 2.0, 2.0, 512, NQ=NQ, NKV=NKV, D=D,
            num_warps=warps))
        rows.append((f"fwd warps={warps}", ms, T * (NQ + 2 * NKV) * D * 2 * 3 / ms / 1e6))
        ms = bench(lambda: qkv_rope_bwd_kernel[(T, NQ + 2 * NKV)](
            q, k, v, cos, sin, rowmap, qkv, lq, lv, 2.0, 2.0, 512, NQ=NQ, NKV=NKV, D=D,
            num_warps=warps))
        rows.append((f"bwd warps={warps}", ms, T * (NQ + 2 * NKV) * D * 2 * 3 / ms / 1e6))
    report("qkv_rope", rows)


if __name__ == "__main__":
    print(f"[tune] {torch.cuda.get_device_name(0)}")
    for part in (nf4, gated, norms, loss, rope):
        part()
