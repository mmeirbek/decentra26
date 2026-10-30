"""Turn the four dev/compare_cpu.py results into whole-model numbers.

    python dev/compare_report.py runs_cpu

One layer's share is the 2-layer run minus the 1-layer run; the rest (embedding, output head, loss,
metrics) is the 1-layer run minus that. The full model has 28 layers. Weights are not tensors the
step allocates, so they are added from their sizes; the CUDA context is not included at all.
Times are estimates for a T4 from the measured bytes and FLOPs (~250 GB/s for memory-bound work,
~27 TFLOPS for fp16 matmuls, the rate base Soup's profile implies); only Kaggle measures speed.
"""

import json
import sys
from pathlib import Path

LAYERS = 28
GBPS, TFLOPS = 250e9, 27e12
WEIGHTS_MIB = {"base": 890 + 645 + 35, "ours": 445 + 645 + 35}  # embedding, NF4 layers, LoRA+Adam
# Ours: on the CPU, torch.mm(h, W.t()) with an fp16 W makes a dense copy of the transposed
# 151,936 x 1,536 head (445 MiB) that cuBLAS never makes, so its CPU peak is not a GPU number.
# Its GPU peak is the end of forward (every layer's kept tensors alive) plus the loss stage's own
# buffers: one 512-token chunk of fp16 logits (148 MiB) and the fp16 hidden-state gradient (3 MiB).
OURS_LOSS_BUFFERS_MIB = 148 + 3


def load(folder, side, layers):
    return json.loads((Path(folder) / f"{side}_{layers}.json").read_text())


def full(one, two, pick):
    per_layer = pick(two) - pick(one)
    return pick(one) - per_layer + LAYERS * per_layer, per_layer


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "runs_cpu"
    rows = {}
    for side in ("base", "ours"):
        one, two = load(folder, side, 1), load(folder, side, 2)
        fwd_end, act_layer = full(one, two, lambda r: r["live_at_forward_end_mib"])
        if side == "base":  # its peak is TRL's accuracy copy, measured as is
            peak = one["live_peak_seen_mib"] + (LAYERS - 1) * act_layer
        else:
            peak = fwd_end + OURS_LOSS_BUFFERS_MIB
        saved, saved_layer = full(one, two, lambda r: r["saved_for_backward_mib"])
        traffic, traffic_layer = full(one, two, lambda r: sum(r["traffic_mib"].values()))
        flops, flops_layer = full(one, two, lambda r: r["matmul_gflop"])
        ops, ops_layer = full(one, two, lambda r: sum(r["ops"].values()))
        rows[side] = {
            "kept per layer, MiB": act_layer,
            "kept per token per layer, KiB": act_layer * 1024 / one["padded_tokens"],
            "step tensors at peak, MiB": peak,
            "peak incl. weights, MiB": peak + WEIGHTS_MIB[side],
            "ops per layer": ops_layer,
            "ops per step": ops,
            "non-matmul traffic per step, GB": traffic * 2**20 / 1e9,
            "matmul per step, TFLOP": flops / 1e3,
            "T4 estimate: memory-bound part, s": traffic * 2**20 / GBPS,
            "T4 estimate: matmul part, s": flops * 1e9 / TFLOPS,
        }
    width = max(len(k) for k in rows["base"])
    print(f"{'':{width}}  {'base':>12}  {'ours':>12}  {'base/ours':>9}")
    for key in rows["base"]:
        b, o = rows["base"][key], rows["ours"][key]
        print(f"{key:{width}}  {b:12.1f}  {o:12.1f}  {b / o if o else float('inf'):9.2f}")


if __name__ == "__main__":
    main()
