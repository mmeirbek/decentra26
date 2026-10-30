"""Summarize a Kaggle run downloaded with `kaggle kernels output mmeirbek/soup-kernel-dev -p DIR`.

    python3 dev/kaggle_report.py runs_kaggle/out_1
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = [  # (group, regex over the CUDA kernel name), first match wins
    ("ours: triton", r"^(nf4_dequant|add_rmsnorm|qkv_rope|gated_act|lm_loss)"),
    ("matmul (cuBLAS / cutlass)", r"gemm|xmma|cutlass|sm75_|turing_|Kernel2|splitK|cublas"),
    ("attention (SDPA)", r"fmha|attention|flash|mem_eff|efficient"),
    ("bitsandbytes dequant", r"kDequantize|dequantize"),
    ("softmax / log-softmax / CE", r"softmax|nll|cross_entropy|log_softmax"),
    ("reductions", r"reduce|Reduce|argmax|sum|norm"),
    ("copies / casts", r"copy|Copy|cat|index|gather|scatter|fill|Fill|transpose"),
    ("elementwise", r"elementwise|vectorized|unrolled|pointwise"),
]


def group_of(name):
    for group, pattern in GROUPS:
        if re.search(pattern, name):
            return group
    return "other"


def main():
    out = Path(sys.argv[1])
    base = out / "out" if (out / "out").exists() else out
    result = json.loads((base / "result.json").read_text()) if (base / "result.json").exists() else {}
    print("== summary")
    print(json.dumps(result.get("summary"), indent=1))
    for side in ("base", "solution"):
        for run in result.get("runs", {}).get(side, []):
            m = run["metrics"]
            metrics_file = base / f"{run['run']}_metrics.json"
            steps = json.loads(metrics_file.read_text()).get("step_s", []) if metrics_file.exists() else []
            print(f"\n{run['run']}: exit {run['exit_code']} wall {run['wall_s']} s peak {run['gpu_peak_mib']} MiB "
                  f"apply {m.get('apply_s')} s loop {m.get('train_loop_s')} s median step {m.get('median_step_s')} s")
            print("  first steps", steps[:4], "last", steps[-3:])
            own = m.get("own_kernel_ms") or {}
            if own:
                total = sum(own.values())
                print(f"  our kernels {total:.0f} ms over steps 3+ ({100 * total / 1000 / m['timed_loop_s']:.1f}% of "
                      f"{m['timed_loop_s']} s):", {k: round(v) for k, v in sorted(own.items(), key=lambda x: -x[1])})
            compiled = base / f"{run['run']}_compiled.json"
            if compiled.exists():
                print("  compiled:", json.loads(compiled.read_text()))
            log = base / f"{run['run']}_train.log"
            if log.exists():
                text = log.read_text(errors="replace")
                for line in text.splitlines():
                    if "[solution]" in line or "Traceback" in line or re.search(r"\bError\b", line):
                        print("  log:", line.strip()[:200])
    checks = base / "checks.json"
    if checks.exists():
        print("\n== checks")
        for check in json.loads(checks.read_text())["checks"]:
            print(("PASS " if check["ok"] else "FAIL ") + check["check"] + " :: " + str(check["detail"])[:300])
    profile = base / "solution_profile_profile.json"
    if profile.exists():
        prof = json.loads(profile.read_text())
        steps = prof["active_steps"]
        groups = defaultdict(float)
        for row in prof["kernels"]:
            groups[group_of(row["name"])] += row["time_us"]
        total = sum(groups.values())
        print(f"\n== GPU time per step by group (profile, {steps} steps): {total / steps / 1000:.1f} ms")
        for group, us in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  {group:30s} {us / steps / 1000:8.1f} ms  {100 * us / total:5.1f}%")
        print("  top kernels:")
        for row in prof["kernels"][:25]:
            print(f"    {row['time_us'] / steps / 1000:7.2f} ms  x{row['count'] // steps:<4} {row['name'][:110]}")


if __name__ == "__main__":
    main()
