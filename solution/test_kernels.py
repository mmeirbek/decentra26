"""Each kernel against plain PyTorch, forward and backward.

On a GPU: `python -m pytest solution -q`. Without one, the same tests run on the CPU under
Triton's interpreter: `TRITON_INTERPRET=1 python -m pytest solution -q`.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    DEV = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEV = "cpu"
else:
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

from solution import kernels_loss  # noqa: E402
from solution.blocks import ACT_BLOCK  # noqa: E402
from solution.kernels_attn import qkv_rope_bwd_kernel, qkv_rope_fwd_kernel  # noqa: E402
from solution.kernels_mlp import gated_act_kernel  # noqa: E402
from solution.kernels_nf4 import NF4Weight  # noqa: E402
from solution.kernels_norm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel  # noqa: E402


def close(a, b, tol):
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("shape", [(320, 128), (256, 1536)])
def test_nf4_dequant_is_bitsandbytes_bit_for_bit(shape):
    import bitsandbytes.functional as bnbf

    torch.manual_seed(0)
    w = torch.randn(shape, dtype=torch.float16, device=DEV) * 0.05
    packed, qs = bnbf.quantize_4bit(w, blocksize=64, compress_statistics=True, quant_type="nf4")
    ref = bnbf.dequantize_4bit(packed, qs)
    ours = NF4Weight(SimpleNamespace(data=packed, quant_state=qs)).dequant()
    assert ours.dtype == ref.dtype and ours.shape == ref.shape
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("chunk", [16, 512])
def test_fused_loss_matches_cross_entropy_entropy_accuracy_and_grad(monkeypatch, chunk):
    monkeypatch.setattr(kernels_loss, "CHUNK", chunk)
    torch.manual_seed(1)
    tokens, hidden = 45, 64
    vocab = 151936 if DEV == "cuda" else 1000
    h = (torch.randn(tokens, hidden, device=DEV) * 0.5).half().requires_grad_()
    w = (torch.randn(vocab, hidden, device=DEV) * 0.2).half()
    targets = torch.randint(0, vocab, (tokens,), device=DEV)
    targets[::4] = -100
    denom = (targets != -100).sum()
    scale = torch.tensor([1024.0], device=DEV)
    loss, token_loss, entropy, hit = kernels_loss.FusedLMLoss.apply(h, w, targets, denom, scale)
    (loss * scale).sum().backward()

    logits = (h.detach() @ w.t()).float()
    ref_tok = F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")
    logp = F.log_softmax(logits, -1)
    ref_ent = -(logp.exp() * logp).sum(-1)
    labelled = targets != -100
    ref_hit = (logits.argmax(-1) == targets) & labelled
    close(token_loss, ref_tok, 2e-4)
    close(loss, ref_tok.sum() / denom, 2e-4)
    close(entropy, ref_ent, 2e-4)
    assert torch.equal(hit.bool(), ref_hit)
    onehot = F.one_hot(targets.clamp(min=0), vocab).float()
    dlogits = ((logp.exp() - onehot) * (scale / denom) * labelled[:, None]).half()
    close(h.grad, dlogits @ w, 2e-2)


def _norm_reference(res, delta, w, eps, keep_q, keep_v, drop_scale):
    h = res + delta.float()
    y = w * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps))
    return h, y, y * keep_q * drop_scale, y * keep_v * drop_scale


@pytest.mark.parametrize("with_lora", [True, False])
def test_add_rmsnorm_forward_and_backward(with_lora):
    torch.manual_seed(2)
    tokens, hidden, eps = 13, 1536, 1e-6
    rowmap = torch.randperm(40, device=DEV)[:tokens].sort().values
    res = torch.randn(tokens, hidden, device=DEV) * 3
    delta = torch.randn(tokens, hidden, device=DEV).half()
    w = torch.rand(hidden, device=DEV) + 0.5
    keep_q_pad = torch.rand(40, hidden, device=DEV) > 0.05
    keep_v_pad = torch.rand(40, hidden, device=DEV) > 0.05
    drop_scale = torch.tensor([1.0 / 0.95], device=DEV)
    kq, kv = keep_q_pad[rowmap].float(), keep_v_pad[rowmap].float()
    if not with_lora:
        kq = kv = torch.zeros_like(kq)

    h = torch.empty_like(res)
    rstd = torch.empty(tokens, device=DEV)
    x = torch.empty(tokens, hidden, dtype=torch.float16, device=DEV)
    xq, xv = torch.empty_like(x), torch.empty_like(x)
    add_rmsnorm_fwd_kernel[(tokens,)](res, delta, w, h, rstd, x, rowmap, keep_q_pad, keep_v_pad,
                                      drop_scale, xq, xv, eps, H=hidden, BLOCK=2048,
                                      WITH_LORA=with_lora, num_warps=8)
    res_r = res.clone().requires_grad_()
    h_ref, y, yq, yv = _norm_reference(res_r, delta, w, eps, kq, kv, drop_scale)
    close(h, h_ref, 1e-6)
    close(x, y.half(), 1e-3)
    if with_lora:
        close(xq, yq.half(), 1e-3)
        close(xv, yv.half(), 1e-3)

    dx = torch.randn(tokens, hidden, device=DEV).half()
    dxq = torch.randn(tokens, hidden, device=DEV).half()
    dxv = torch.randn(tokens, hidden, device=DEV).half()
    dres_out = torch.randn(tokens, hidden, device=DEV)
    dres = torch.empty_like(res)
    ddelta = torch.empty_like(x)
    add_rmsnorm_bwd_kernel[(tokens,)](dx, h, rstd, w, dres_out, dres, ddelta, rowmap, keep_q_pad,
                                      keep_v_pad, drop_scale, dxq, dxv, H=hidden, BLOCK=2048,
                                      WITH_LORA=with_lora, num_warps=8)
    upstream = (y * dx.float()).sum() + (h_ref * dres_out).sum()
    if with_lora:
        upstream = upstream + (yq * dxq.float()).sum() + (yv * dxv.float()).sum()
    upstream.backward()
    close(dres, res_r.grad, 1e-4)
    close(ddelta, res_r.grad.half(), 1e-3)


def _rope(x, cos, sin):
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def test_qkv_rope_forward_and_backward():
    torch.manual_seed(3)
    batch, seq, nq, nkv, d = 3, 11, 4, 2, 32
    lengths = [11, 7, 3]
    mask = torch.zeros(batch, seq, dtype=torch.bool, device=DEV)
    for b, n in enumerate(lengths):
        mask[b, :n] = True
    rowmap = mask.reshape(-1).nonzero().squeeze(1)
    tokens = rowmap.numel()
    width = (nq + 2 * nkv) * d
    qkv = torch.randn(tokens, width, device=DEV).half()
    lq = torch.randn(tokens, nq * d, device=DEV).half()
    lv = torch.randn(tokens, nkv * d, device=DEV).half()
    ang = torch.rand(seq, d // 2, device=DEV) * 6
    cos = torch.cat([ang.cos(), ang.cos()], -1).contiguous()
    sin = torch.cat([ang.sin(), ang.sin()], -1).contiguous()
    sq, sv = 2.0, 1.5
    q = torch.empty(tokens, nq, d, dtype=torch.float16, device=DEV)
    k = torch.empty(tokens, nkv, d, dtype=torch.float16, device=DEV)
    v = torch.empty_like(k)
    qkv_rope_fwd_kernel[(tokens, nq + 2 * nkv)](qkv, lq, lv, cos, sin, rowmap, q, k, v, sq, sv,
                                                 seq, NQ=nq, NKV=nkv, D=d, num_warps=1)

    pos = rowmap % seq
    c, s = cos[pos][:, None, :], sin[pos][:, None, :]
    base = qkv.float().requires_grad_()
    lq_r = lq.float().requires_grad_()
    lv_r = lv.float().requires_grad_()
    q_ref = base[:, :nq * d].view(tokens, nq, d) + (lq_r * sq).half().float().view(tokens, nq, d)
    k_ref = base[:, nq * d:(nq + nkv) * d].view(tokens, nkv, d)
    v_ref = base[:, (nq + nkv) * d:].view(tokens, nkv, d) + \
        (lv_r * sv).half().float().view(tokens, nkv, d)
    q_ref, k_ref = _rope(q_ref, c, s), _rope(k_ref, c, s)
    close(q, q_ref.half(), 2e-3)
    close(k, k_ref.half(), 2e-3)
    close(v, v_ref.half(), 2e-3)

    dq = torch.randn_like(q)
    dk = torch.randn_like(k)
    dv = torch.randn_like(v)
    dqkv = torch.empty_like(qkv)
    dlq, dlv = torch.empty_like(lq), torch.empty_like(lv)
    qkv_rope_bwd_kernel[(tokens, nq + 2 * nkv)](dq, dk, dv, cos, sin, rowmap, dqkv, dlq, dlv, sq,
                                                 sv, seq, NQ=nq, NKV=nkv, D=d, num_warps=1)
    ((q_ref * dq.float()).sum() + (k_ref * dk.float()).sum() + (v_ref * dv.float()).sum()).backward()
    close(dqkv, base.grad, 2e-3)
    close(dlq, (base.grad[:, :nq * d].half().float() * sq), 2e-3)
    close(dlv, (base.grad[:, (nq + nkv) * d:].half().float() * sv), 2e-3)


def test_gated_act_forward_and_backward():
    torch.manual_seed(4)
    tokens, inter = 9, 8960 if DEV == "cuda" else 320
    gu = torch.randn(tokens, 2 * inter, device=DEV).half()
    act = torch.empty(tokens, inter, dtype=torch.float16, device=DEV)
    grid = (tokens, -(-inter // ACT_BLOCK))
    gated_act_kernel[grid](gu, act, I=inter, BLOCK=ACT_BLOCK, BACKWARD=False, num_warps=4)
    g = gu[:, :inter].float().requires_grad_()
    u = gu[:, inter:].float().requires_grad_()
    ref = F.silu(g) * u
    close(act, ref.half(), 2e-3)
    dact = torch.randn(tokens, inter, device=DEV).half()
    ref.backward(dact.float())
    buf = gu.clone()
    gated_act_kernel[grid](buf, dact, I=inter, BLOCK=ACT_BLOCK, BACKWARD=True, num_warps=4)
    close(buf[:, :inter], g.grad.half(), 2e-3)
    close(buf[:, inter:], u.grad.half(), 2e-3)


@pytest.mark.skipif(DEV != "cuda", reason="the CUDA dropout kernel is what the base run uses")
def test_dropout_mask_is_the_one_torch_dropout_draws():
    x = torch.randn(4, 77, 1536, device="cuda")
    ones = torch.ones_like(x)
    torch.cuda.manual_seed(123)
    ref = F.dropout(x, 0.05, training=True)
    state_ref = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(123)
    out, keep = torch.native_dropout(ones, 0.05, True)
    assert torch.equal(torch.cuda.get_rng_state(), state_ref)
    scale = out.amax()
    assert torch.equal(ref, torch.where(keep, x * scale, torch.zeros_like(x)))
