"""One training step of base Soup's path and of solution/'s fused path, measured on the CPU.

    docker run --rm -e TRITON_INTERPRET=1 -v "$PWD":/repo soupk-cpu \
        python dev/compare_cpu.py --side base --layers 2 --out runs_cpu/base_2.json

The model has the real sizes of Qwen2.5-1.5B-Instruct (hidden 1536, MLP 8960, 12/2 heads, vocab
151,936, tied head), NF4 weights with nested scales, fp32 norms and embedding as after
prepare_model_for_kbit_training, LoRA r=16 / alpha=32 / dropout 0.05 on q_proj and v_proj, and
`--layers` decoder layers (random weights: sizes are what is measured here, not quality). The batch
is a peak one: 4 rows padded to 512, 1,000 real tokens.

"base" runs what a base step does on the T4: the model under fp16 autocast (peft and SDPA are told
autocast is on, as on CUDA), accelerate's fp32 copy of the logits, TRL's entropy and accuracy, Soup's
per-row losses, backward. "ours" runs solution/'s fused path, which computes all of those itself.

Measured, the same way for both sides:
- bytes autograd keeps for backward (saved_tensors_hooks), and live tensor bytes when the step's
  forward work is done (the base run's peak moment) and at the highest point seen;
- every operation run: torch ops by kind, and our Triton launches;
- bytes read + written by everything that is not a matmul (what bounds those ops on a GPU), and
  matmul FLOPs.
Speed itself cannot be measured here: Triton runs under its CPU interpreter.
"""

import argparse
import json
import os
import sys
import time
import weakref
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MATMUL = {"mm", "addmm", "bmm", "baddbmm", "matmul", "linear"}
QUANT_MATMUL = {"gemm_4bit"}
DEQUANT = {"dequantize_4bit", "dequantize_blockwise"}
ATTENTION_PREFIX = ("_scaled_dot_product", "scaled_dot_product")


def nbytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


class Meter(TorchDispatchMode):
    """Live tensor bytes (by storage), op counts, non-matmul traffic and matmul FLOPs."""

    def __init__(self):
        super().__init__()
        self.holders, self.size = {}, {}
        self.live = self.peak = 0
        self.ops = Counter()
        self.traffic = Counter()
        self.flops = 0
        self.paused = 0  # inside a Triton launch: the CPU interpreter's own copies are not counted

    def _track(self, t):
        if not isinstance(t, torch.Tensor) or t.device.type != "cpu":
            return
        storage = t.untyped_storage()
        key, size = storage.data_ptr(), storage.nbytes()
        if not size:
            return
        if key not in self.holders:
            self.holders[key], self.size[key] = 0, size
            self.live += size
            self.peak = max(self.peak, self.live)
        self.holders[key] += 1
        weakref.finalize(t, self._release, key)

    def _release(self, key):
        if key in self.holders:
            self.holders[key] -= 1
            if not self.holders[key]:
                self.live -= self.size.pop(key)
                del self.holders[key]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.paused:
            return out
        name = func._schema.name.split("::")[-1]
        returns_alias = any(r.alias_info is not None and not r.alias_info.is_write
                            for r in func._schema.returns)
        if name in MATMUL:
            self.ops["matmul"] += 1
            a, b = [x for x in tree_leaves(args) if isinstance(x, torch.Tensor)][-2:]
            out_t = [x for x in tree_leaves(out) if isinstance(x, torch.Tensor)][0]
            k = a.shape[-1]
            self.flops += 2 * out_t.numel() * k
        elif name in QUANT_MATMUL:
            self.ops["matmul"] += 1
            a = args[0]
            n_out = args[2][0]
            self.flops += 2 * a.numel() // a.shape[-1] * n_out * a.shape[-1]
            n = args[2][0] * args[2][1]
            self.traffic["nf4 dequant"] += n // 2 + 2 * n  # read packed, write fp16 weight
        elif name in DEQUANT:
            self.ops["nf4 dequant"] += 1
            self.traffic["nf4 dequant"] += sum(nbytes(x) for x in tree_leaves(args)) + \
                sum(nbytes(x) for x in tree_leaves(out))
        elif name.startswith(ATTENTION_PREFIX):
            self.ops["attention"] += 1
        elif not returns_alias:
            self.ops["other torch ops"] += 1
            self.traffic["torch elementwise / copies / reductions"] += \
                sum(nbytes(x) for x in tree_leaves(args)) + sum(nbytes(x) for x in tree_leaves(out))
        for t in tree_leaves(out):
            self._track(t)
        return out


def watch_triton(meter):
    """Count our kernel launches and the bytes of the tensors each one is given."""
    from solution import kernels_attn, kernels_loss, kernels_mlp, kernels_nf4, kernels_norm

    kernels = {}
    for mod in (kernels_attn, kernels_loss, kernels_mlp, kernels_nf4, kernels_norm):
        for name in dir(mod):
            if name.endswith("_kernel"):
                kernels[name] = getattr(mod, name)
    for kernel_cls in {type(k) for k in kernels.values()}:
        original = kernel_cls.run

        def run(self, *args, _original=original, **kwargs):
            if not kwargs.get("warmup"):
                fn = getattr(self, "fn", None)
                name = getattr(fn, "__name__", "?")
                meter.ops[f"triton {name}"] += 1
                meter.traffic["our triton kernels (upper bound)"] += \
                    sum(nbytes(x) for x in args)
            meter.paused += 1
            try:
                return _original(self, *args, **kwargs)
            finally:
                meter.paused -= 1

        kernel_cls.run = run


def build(layers):
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=1536, intermediate_size=8960,
                      num_hidden_layers=layers, num_attention_heads=12, num_key_value_heads=2,
                      max_position_embeddings=32768, tie_word_embeddings=True, rms_norm_eps=1e-6,
                      rope_theta=1000000.0)
    cfg._attn_implementation = "sdpa"
    with torch.device("cpu"):
        model = Qwen2ForCausalLM(cfg).to(torch.float16)
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                lin = getattr(parent, name)
                q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                      compute_dtype=torch.float16, compress_statistics=True,
                                      quant_type="nf4")
                q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False,
                                             compress_statistics=True, quant_type="nf4",
                                             blocksize=64)
                if lin.bias is not None:
                    q.bias = torch.nn.Parameter(lin.bias.data.clone(), requires_grad=False)
                setattr(parent, name, q.to("cpu"))
    for param in model.parameters():  # prepare_model_for_kbit_training
        param.requires_grad = False
        if param.dtype == torch.float16 and param.__class__.__name__ != "Params4bit":
            param.data = param.data.float()
    peft_model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                  target_modules=["q_proj", "v_proj"],
                                                  task_type="CAUSAL_LM"))
    peft_model.train()
    return peft_model


def peak_batch():
    torch.manual_seed(1)
    lengths, seq = [512, 220, 140, 128], 512
    ids = torch.randint(0, 151936, (4, seq))
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
        labels[b, n // 3:n] = ids[b, n // 3:n]
    return ids, mask, labels


def base_step(model, ids, mask, labels, meter):
    from soup_cli.trainer.rewind_hf import row_losses
    from trl.trainer.utils import entropy_from_logits

    original_enabled = torch.is_autocast_enabled
    original_sdpa = F.scaled_dot_product_attention

    def enabled(device_type=None):  # peft asks without a device; on CUDA that is autocast's state
        return original_enabled(device_type or "cpu")

    def sdpa(q, k, v, *args, **kwargs):  # CUDA autocast runs SDPA in fp16
        return original_sdpa(q.half(), k.half(), v.half(), *args, **kwargs)

    torch.is_autocast_enabled = enabled
    F.scaled_dot_product_attention = sdpa
    try:
        with torch.autocast("cpu", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
        logits = out.logits.float()  # accelerate's convert_outputs_to_fp32
        out["logits"] = logits
        with torch.no_grad():
            per_token = entropy_from_logits(logits)
            (per_token * mask).sum() / mask.sum()
            shift_logits = logits[..., :-1, :].contiguous()  # TRL's accuracy
            shift_logits.argmax(dim=-1)
            del shift_logits, per_token
        row_losses(logits, labels)  # Soup's rewind
        at_forward_end = meter.live
        loss = out.loss
        del out, logits
        loss.backward()
    finally:
        torch.is_autocast_enabled = original_enabled
        F.scaled_dot_product_attention = original_sdpa
    return at_forward_end


def ours_step(model, ids, mask, labels, meter):
    from solution.model import install_plan, patch_qwen2

    patch_qwen2()
    plan = install_plan(model, require_cuda=False)
    plan.activate()
    with torch.autocast("cpu", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    assert out.logits is None and plan.last_step is not None
    at_forward_end = meter.live
    out.loss.backward()
    return at_forward_end


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--side", choices=("base", "ours"), required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = build(args.layers)
    ids, mask, labels = peak_batch()
    meter = Meter()
    if args.side == "ours":
        watch_triton(meter)
    saved = {}

    def pack(t):  # unique storages: two ops saving one tensor keep it once
        storage = t.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return t

    started = time.time()
    with meter, torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        step = base_step if args.side == "base" else ours_step
        at_forward_end = step(model, ids, mask, labels, meter)
    result = {"side": args.side, "layers": args.layers, "real_tokens": int(mask.sum()),
              "padded_tokens": int(mask.numel()),
              "saved_for_backward_mib": sum(saved.values()) / 2**20,
              "live_at_forward_end_mib": at_forward_end / 2**20,
              "live_peak_seen_mib": meter.peak / 2**20,
              "ops": dict(meter.ops), "traffic_mib": {k: v / 2**20 for k, v in meter.traffic.items()},
              "matmul_gflop": meter.flops / 1e9, "cpu_seconds": round(time.time() - started, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
