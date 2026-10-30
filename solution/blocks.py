"""Autograd functions of the fused Qwen2 decoder: what runs between the kernels and what is kept.

A decoder layer becomes  AttnPrologue -> SDPA (torch, unchanged) -> PostBlock, and the model ends
with FinalNorm -> FusedLMLoss. Tokens are packed: only the real (non-pad) positions of the batch go
through the matmuls and kernels; `rowmap` holds each one's flat index in the padded batch, which
SDPA, the dropout masks and the outputs for TRL / rewind still use.

Kept for backward per token and layer: each norm's normalized input (fp16) and 1/rms, the LoRA
inputs (as the base keeps them) and their dropout masks as bits, what SDPA keeps, and the fp16
gate|up buffer (or only the MLP input when `recompute_mlp` is on). Weights are dequantized from NF4 into
a temporary right before each matmul, in forward and again in backward, as bitsandbytes does.
"""

import torch
import triton

from .kernels_attn import qkv_rope_bwd_kernel, qkv_rope_fwd_kernel
from .kernels_mlp import gated_act_kernel
from .kernels_nf4 import NF4Weight, dequant_rows
from .kernels_norm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel

ACT_BLOCK_FWD, ACT_BLOCK_BWD, ACT_WARPS = 1024, 512, 8  # fastest on a T4
HALF = torch.float16


def _base_layer(module):
    return getattr(module, "base_layer", module)


class LoraSide:
    """The single active LoRA adapter on one projection."""

    def __init__(self, module):
        names = list(getattr(module, "active_adapters", []) or [])
        if names != ["default"] or list(module.lora_A.keys()) != ["default"]:
            raise ValueError("needs exactly one active LoRA adapter")
        if getattr(module, "merged", False) or getattr(module, "disable_adapters", False):
            raise ValueError("LoRA adapter is merged or disabled")
        if getattr(module, "use_dora", {}).get("default") or "default" in getattr(module, "lora_variant", {}):
            raise ValueError("DoRA and other LoRA variants are not supported")
        self.a = module.lora_A["default"].weight
        self.b = module.lora_B["default"].weight
        if module.lora_B["default"].bias is not None:
            raise ValueError("LoRA bias is not supported")
        drop = module.lora_dropout["default"]
        self.dropout = drop.p if isinstance(drop, torch.nn.Dropout) else 0.0
        self.scaling = float(module.scaling["default"])


class LayerPlan:
    """What the fused path of one decoder layer reads, looked up once before training."""

    def __init__(self, layer, recompute_mlp):
        attn, mlp = layer.self_attn, layer.mlp
        if getattr(attn, "attention_dropout", 0.0) or getattr(attn, "sliding_window", None):
            raise ValueError("attention dropout and sliding windows are not supported")
        for name in ("k_proj", "o_proj"):
            if hasattr(getattr(attn, name), "lora_A"):
                raise ValueError(f"LoRA on {name} is not supported")
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(getattr(mlp, name), "lora_A"):
                raise ValueError(f"LoRA on {name} is not supported")
        self.attn = attn
        self.lora_q = LoraSide(attn.q_proj)
        self.lora_v = LoraSide(attn.v_proj)
        if self.lora_q.dropout != self.lora_v.dropout:
            raise ValueError("q_proj and v_proj LoRA dropout differ")
        q, k, v = (_base_layer(m) for m in (attn.q_proj, attn.k_proj, attn.v_proj))
        self.qkv = [NF4Weight(m.weight) for m in (q, k, v)]
        biases = [m.bias for m in (q, k, v)]
        if any(b is None for b in biases):
            raise ValueError("Qwen2 q/k/v projections carry a bias")
        self.qkv_bias = torch.cat([b.detach().to(HALF) for b in biases])
        self.o = [NF4Weight(_base_layer(attn.o_proj).weight)]
        self.gate_up = [NF4Weight(_base_layer(mlp.gate_proj).weight),
                        NF4Weight(_base_layer(mlp.up_proj).weight)]
        self.down = [NF4Weight(_base_layer(mlp.down_proj).weight)]
        for w in self.qkv + self.o + self.gate_up + self.down:
            if w.dtype != HALF:
                raise ValueError("needs fp16 compute (the T4 path)")
        self.ln_in = layer.input_layernorm
        self.ln_post = layer.post_attention_layernorm
        self.n_heads = attn.config.num_attention_heads
        self.n_kv = attn.config.num_key_value_heads
        self.head_dim = attn.head_dim
        self.hidden = self.ln_in.weight.shape[0]
        self.inter = self.gate_up[0].shape[0]
        self.recompute_mlp = recompute_mlp
        if self.n_heads * self.head_dim != self.qkv[0].shape[0] or \
                self.n_kv * self.head_dim != self.qkv[1].shape[0]:
            raise ValueError("unexpected q/k/v shapes")
        if self.head_dim & (self.head_dim - 1):
            raise ValueError("head_dim must be a power of two")


class StepState:
    """Per-forward facts shared by every layer."""

    def __init__(self, rowmap, batch, seq_len, cos, sin, segments):
        dev = rowmap.device
        self.rowmap, self.batch, self.seq_len = rowmap, batch, seq_len
        self.segments = segments  # [(start, stop)] of each row's tokens, or None
        self.cos, self.sin = cos, sin
        # placeholders for the pointers a norm kernel without LoRA inputs does not read
        self.no_keep = torch.ones(1, dtype=torch.bool, device=dev)
        self.no_bits = torch.zeros(1, dtype=torch.uint8, device=dev)
        self.unit = torch.ones(1, dtype=torch.float32, device=dev)


class Normed:
    """What one residual add + RMSNorm produces: the new residual h (fp32, for the next block), x
    (fp16, for the matmul), and for backward xhat (fp16) and 1/rms; with LoRA inputs also xq and xv
    (fp16) and the two dropout masks as bits of the real tokens."""

    def __init__(self, **parts):
        self.__dict__.update(parts)


def _norm_launch(res, delta, weight, eps, step, keep_q=None, keep_v=None, drop_scale=None):
    tokens, hidden = res.shape
    dev = res.device
    h = torch.empty_like(res)
    rstd = torch.empty(tokens, dtype=torch.float32, device=dev)
    x = torch.empty((tokens, hidden), dtype=HALF, device=dev)
    xhat = torch.empty_like(x)
    with_lora = keep_q is not None
    xq = torch.empty_like(x) if with_lora else x
    xv = torch.empty_like(x) if with_lora else x
    bits_q = torch.empty((tokens, hidden // 8), dtype=torch.uint8, device=dev) if with_lora \
        else step.no_bits
    bits_v = torch.empty_like(bits_q) if with_lora else step.no_bits
    add_rmsnorm_fwd_kernel[(tokens,)](
        res, delta, weight, h, rstd, x, xhat, step.rowmap,
        keep_q if with_lora else step.no_keep, keep_v if with_lora else step.no_keep,
        drop_scale if with_lora else step.unit, xq, xv, bits_q, bits_v, eps,
        H=hidden, BLOCK=triton.next_power_of_2(hidden), WITH_LORA=with_lora, num_warps=8)
    return Normed(h=h, rstd=rstd, x=x, xhat=xhat, xq=xq, xv=xv, bits_q=bits_q, bits_v=bits_v)


def _norm_backward(dx, xhat, rstd, weight, dres_out, step, bits_q=None, bits_v=None,
                   drop_scale=None, dxq=None, dxv=None):
    tokens, hidden = xhat.shape
    dres = torch.empty((tokens, hidden), dtype=torch.float32, device=xhat.device)
    ddelta = torch.empty((tokens, hidden), dtype=HALF, device=xhat.device)
    with_lora = bits_q is not None
    add_rmsnorm_bwd_kernel[(tokens,)](
        dx, xhat, rstd, weight, dres_out, dres, ddelta,
        bits_q if with_lora else step.no_bits, bits_v if with_lora else step.no_bits,
        drop_scale if with_lora else step.unit, dxq if with_lora else dx, dxv if with_lora else dx,
        H=hidden, BLOCK=triton.next_power_of_2(hidden), WITH_LORA=with_lora, num_warps=8)
    return dres, ddelta


def _no_autocast(tensor):
    """Our matmuls already get the fp16 operands autocast would make; keep autocast out of them
    (and out of the `out=` matmuls it does not handle)."""
    return torch.autocast(device_type=tensor.device.type, enabled=False)


def _zeros_if_none(grad, like, shape=None, dtype=None):
    if grad is not None:
        return grad.contiguous()
    return torch.zeros(shape if shape is not None else like.shape,
                       dtype=dtype or like.dtype, device=like.device)


class AttnPrologue(torch.autograd.Function):
    """residual + delta -> RMSNorm -> q|k|v matmul (+ LoRA on q and v) -> RoPE -> packed q, k, v
    of shape (tokens, heads, head_dim)."""

    @staticmethod
    def forward(ctx, res, delta, keep_q, keep_v, drop_scale, aq, bq, av, bv, plan, step):
        with _no_autocast(res):
            return AttnPrologue._forward(ctx, res, delta, keep_q, keep_v, drop_scale, aq, bq, av,
                                         bv, plan, step)

    @staticmethod
    def _forward(ctx, res, delta, keep_q, keep_v, drop_scale, aq, bq, av, bv, plan, step):
        dev = res.device
        n = _norm_launch(res, delta, plan.ln_in.weight, plan.ln_in.variance_epsilon, step, keep_q,
                         keep_v, drop_scale)
        xq, xv = n.xq, n.xv
        w = dequant_rows(plan.qkv, HALF, dev)
        qkv = torch.addmm(plan.qkv_bias, n.x, w.t())
        del w
        lora_aq = xq @ aq.to(HALF).t()
        lora_q = lora_aq @ bq.to(HALF).t()
        lora_av = xv @ av.to(HALF).t()
        lora_v = lora_av @ bv.to(HALF).t()
        tokens = res.shape[0]
        q = torch.empty((tokens, plan.n_heads, plan.head_dim), dtype=HALF, device=dev)
        k = torch.empty((tokens, plan.n_kv, plan.head_dim), dtype=HALF, device=dev)
        v = torch.empty_like(k)
        qkv_rope_fwd_kernel[(tokens, plan.n_heads + 2 * plan.n_kv)](
            qkv, lora_q, lora_v, step.cos, step.sin, step.rowmap, q, k, v,
            plan.lora_q.scaling, plan.lora_v.scaling, step.seq_len,
            NQ=plan.n_heads, NKV=plan.n_kv, D=plan.head_dim, num_warps=1)
        ctx.save_for_backward(n.xhat, n.rstd, xq, xv, lora_aq, lora_av, n.bits_q, n.bits_v,
                              drop_scale, aq, bq, av, bv)
        ctx.plan, ctx.step = plan, step
        return n.h, q, k, v

    @staticmethod
    def backward(ctx, dh, dq, dk, dv):
        xhat, rstd, xq, xv, lora_aq, lora_av, bits_q, bits_v, drop_scale, aq, bq, av, bv = \
            ctx.saved_tensors
        plan, step = ctx.plan, ctx.step
        tokens, hidden = xhat.shape
        dev = xhat.device
        dq = _zeros_if_none(dq, xhat, (tokens, plan.n_heads, plan.head_dim), HALF)
        dk = _zeros_if_none(dk, xhat, (tokens, plan.n_kv, plan.head_dim), HALF)
        dv = _zeros_if_none(dv, xhat, (tokens, plan.n_kv, plan.head_dim), HALF)
        width = (plan.n_heads + 2 * plan.n_kv) * plan.head_dim
        dqkv = torch.empty((tokens, width), dtype=HALF, device=dev)
        dlora_q = torch.empty((tokens, plan.n_heads * plan.head_dim), dtype=HALF, device=dev)
        dlora_v = torch.empty((tokens, plan.n_kv * plan.head_dim), dtype=HALF, device=dev)
        qkv_rope_bwd_kernel[(tokens, plan.n_heads + 2 * plan.n_kv)](
            dq, dk, dv, step.cos, step.sin,
            step.rowmap, dqkv, dlora_q, dlora_v, plan.lora_q.scaling, plan.lora_v.scaling,
            step.seq_len, NQ=plan.n_heads, NKV=plan.n_kv, D=plan.head_dim, num_warps=1)
        w = dequant_rows(plan.qkv, HALF, dev)
        dx = dqkv @ w
        del w, dqkv
        aq16, bq16, av16, bv16 = (t.to(HALF) for t in (aq, bq, av, bv))
        d_bq = (dlora_q.t() @ lora_aq).float()
        d_lora_aq = dlora_q @ bq16
        d_aq = (d_lora_aq.t() @ xq).float()
        dxq = d_lora_aq @ aq16
        d_bv = (dlora_v.t() @ lora_av).float()
        d_lora_av = dlora_v @ bv16
        d_av = (d_lora_av.t() @ xv).float()
        dxv = d_lora_av @ av16
        dres = ddelta = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            dres, ddelta = _norm_backward(
                dx, xhat, rstd, plan.ln_in.weight,
                _zeros_if_none(dh, xhat, (tokens, hidden), torch.float32), step, bits_q, bits_v,
                drop_scale, dxq, dxv)
        return dres, ddelta, None, None, None, d_aq, d_bq, d_av, d_bv, None, None


class PostBlock(torch.autograd.Function):
    """packed attention output (tokens, hidden) -> o_proj -> + residual -> RMSNorm -> gate|up ->
    SiLU * up -> down_proj."""

    @staticmethod
    def forward(ctx, res, attn, plan, step):
        with _no_autocast(res):
            return PostBlock._forward(ctx, res, attn, plan, step)

    @staticmethod
    def _forward(ctx, res, attn, plan, step):
        dev = res.device
        w = dequant_rows(plan.o, HALF, dev)
        o = attn @ w.t()
        del w
        n = _norm_launch(res, o, plan.ln_post.weight, plan.ln_post.variance_epsilon, step)
        del o
        w = dequant_rows(plan.gate_up, HALF, dev)
        gu = n.x @ w.t()
        del w
        act = torch.empty((res.shape[0], plan.inter), dtype=HALF, device=dev)
        gated_act_kernel[(res.shape[0], triton.cdiv(plan.inter, ACT_BLOCK_FWD))](
            gu, act, I=plan.inter, BLOCK=ACT_BLOCK_FWD, BACKWARD=False, num_warps=ACT_WARPS)
        w = dequant_rows(plan.down, HALF, dev)
        out = act @ w.t()
        del w, act
        ctx.save_for_backward(n.xhat, n.rstd, n.x if plan.recompute_mlp else gu)
        ctx.plan, ctx.step = plan, step
        return n.h, out

    @staticmethod
    def backward(ctx, dh, dout):
        xhat, rstd, kept = ctx.saved_tensors
        plan, step = ctx.plan, ctx.step
        dev = xhat.device
        tokens, hidden = xhat.shape
        dout = _zeros_if_none(dout, xhat, (tokens, hidden), HALF)
        w = dequant_rows(plan.down, HALF, dev)
        dact = dout @ w
        del w
        w_gu = None
        if plan.recompute_mlp:
            w_gu = dequant_rows(plan.gate_up, HALF, dev)
            gu = kept @ w_gu.t()
        else:
            gu = kept
        gated_act_kernel[(tokens, triton.cdiv(plan.inter, ACT_BLOCK_BWD))](
            gu, dact, I=plan.inter, BLOCK=ACT_BLOCK_BWD, BACKWARD=True, num_warps=ACT_WARPS)
        del dact
        if w_gu is None:
            w_gu = dequant_rows(plan.gate_up, HALF, dev)
        dx = gu @ w_gu
        del w_gu, gu
        dres, do = _norm_backward(dx, xhat, rstd, plan.ln_post.weight,
                                  _zeros_if_none(dh, xhat, (tokens, hidden), torch.float32), step)
        del dx
        w = dequant_rows(plan.o, HALF, dev)
        dattn = do @ w
        del w, do
        return dres, dattn, None, None


def attend(lp, q, k, v, step, attention_fn, mask):
    """SDPA on packed q, k, v; returns the packed output (tokens, heads * head_dim).

    When each row's real tokens are contiguous (`step.segments`, always the case for TRL's padded
    batches), every row runs on its own real length with the causal flag: the same keys and the
    same softmax as the base's padded call with its mask, without the pad positions' work. Other
    masks take the padded call itself."""
    tokens = q.shape[0]
    if step.segments is not None:
        outs = []
        for start, stop in step.segments:
            qi = q[start:stop].unsqueeze(0).transpose(1, 2)
            ki = k[start:stop].unsqueeze(0).transpose(1, 2)
            vi = v[start:stop].unsqueeze(0).transpose(1, 2)
            ki = repeat_kv(ki, lp.n_heads // lp.n_kv)
            vi = repeat_kv(vi, lp.n_heads // lp.n_kv)
            out = torch.nn.functional.scaled_dot_product_attention(
                qi, ki, vi, dropout_p=0.0, is_causal=stop - start > 1, scale=lp.attn.scaling)
            outs.append(out.transpose(1, 2).reshape(stop - start, -1))
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
    slots = step.batch * step.seq_len
    shape = (step.batch, step.seq_len)
    q_pad = q.new_zeros((slots,) + q.shape[1:]).index_copy(0, step.rowmap, q)
    k_pad = k.new_zeros((slots,) + k.shape[1:]).index_copy(0, step.rowmap, k)
    v_pad = v.new_zeros((slots,) + v.shape[1:]).index_copy(0, step.rowmap, v)
    out, _ = attention_fn(lp.attn, q_pad.view(*shape, *q.shape[1:]).transpose(1, 2),
                          k_pad.view(*shape, *k.shape[1:]).transpose(1, 2),
                          v_pad.view(*shape, *v.shape[1:]).transpose(1, 2), mask, dropout=0.0,
                          scaling=lp.attn.scaling, sliding_window=lp.attn.sliding_window)
    return out.reshape(slots, -1).index_select(0, step.rowmap).view(tokens, -1)


def repeat_kv(x, groups):
    """(1, kv_heads, n, d) -> (1, kv_heads * groups, n, d), as transformers' repeat_kv."""
    if groups == 1:
        return x
    b, heads, n, d = x.shape
    return x[:, :, None].expand(b, heads, groups, n, d).reshape(b, heads * groups, n, d)


class FinalNorm(torch.autograd.Function):
    """residual + last MLP output -> final RMSNorm -> fp16 input of the output head."""

    @staticmethod
    def forward(ctx, res, delta, norm, step):
        with _no_autocast(res):
            n = _norm_launch(res, delta, norm.weight, norm.variance_epsilon, step)
        ctx.save_for_backward(n.xhat, n.rstd)
        ctx.norm, ctx.step = norm, step
        return n.x

    @staticmethod
    def backward(ctx, dx):
        xhat, rstd = ctx.saved_tensors
        dres, ddelta = _norm_backward(dx.contiguous(), xhat, rstd, ctx.norm.weight,
                                      torch.zeros(xhat.shape, dtype=torch.float32,
                                                  device=xhat.device), ctx.step)
        return dres, ddelta, None, None
