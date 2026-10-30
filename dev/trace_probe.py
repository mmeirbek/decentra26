"""Dev-only: where the GPU waits for the CPU in a training step. dev/kaggle_push.py --probe-trace
ships it as solution/_trace.py and calls install() from apply(). Never part of the submitted
solution.

Profiles steps FIRST..LAST (0-based) with CPU, CUDA and Python stacks, writes the Chrome trace
to /kaggle/working/out/trace.json.gz and prints, from it:
  - wall time of the window, GPU busy time, GPU idle time;
  - every GPU idle gap longer than GAP_US, attributed to the innermost Python frames of the main
    thread that ran while the GPU was idle (summed over gaps), so the CPU work that leaves the GPU
    waiting is named by file and line.
    python3 dev/trace_probe.py trace.json.gz   re-runs the analysis on a downloaded trace.
"""

import gzip
import json
import os
import sys
from collections import defaultdict

FIRST, LAST = 6, 8
GAP_US = 5.0
OUT = "/kaggle/working/out"


def analyze(trace_path, top=45):
    opener = gzip.open if trace_path.endswith(".gz") else open
    with opener(trace_path, "rt") as f:
        events = json.load(f)["traceEvents"]
    gpu = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in events
                 if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    steps = sorted((e["ts"], e["ts"] + e["dur"]) for e in events
                   if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep#"))
    if not gpu:
        print("[trace] no GPU events")
        return
    lo = steps[0][0] if steps else gpu[0][0]
    hi = steps[-1][1] if steps else gpu[-1][1]
    busy, gaps, end = 0.0, [], lo
    for start, stop in gpu:
        if stop <= lo or start >= hi:
            continue
        start, stop = max(start, lo), min(stop, hi)
        if start > end:
            if start - end >= GAP_US:
                gaps.append((end, start))
        busy += max(0.0, stop - max(start, end))
        end = max(end, stop)
    if hi > end and hi - end >= GAP_US:
        gaps.append((end, hi))
    wall = hi - lo
    idle = sum(b - a for a, b in gaps)
    print(f"[trace] window {wall / 1000:.1f} ms over {len(steps)} steps: GPU busy "
          f"{busy / 1000:.1f} ms ({100 * busy / wall:.1f}%), idle in gaps >= {GAP_US:.0f} us "
          f"{idle / 1000:.1f} ms in {len(gaps)} gaps", flush=True)

    # the main thread: the one that runs ProfilerStep
    tid = next((e["tid"] for e in events if str(e.get("name", "")).startswith("ProfilerStep#")),
               None)
    frames = sorted(((e["ts"], e["ts"] + e.get("dur", 0), e["name"]) for e in events
                     if e.get("ph") == "X" and e.get("cat") == "python_function"
                     and e.get("tid") == tid), key=lambda x: (x[0], -x[1]))
    ops = sorted(((e["ts"], e["ts"] + e.get("dur", 0), e["name"]) for e in events
                  if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "cuda_runtime")
                  and e.get("tid") == tid), key=lambda x: (x[0], -x[1]))

    def ours(name):
        return any(k in name for k in ("solution/", "soup_cli/", "trl/", "transformers/",
                                       "accelerate/", "peft/", "torch/optim", "torch/amp",
                                       "torch/nn/utils", "torch/autograd", "_train.py"))

    def blame(intervals, keyfn):
        out = defaultdict(float)
        j = 0
        for a, b in gaps:
            while j < len(intervals) and intervals[j][1] < a - 1e5:
                j += 1
            # deepest (shortest) covering frames: give each gap microsecond to the innermost
            covering = [(s, t, n) for s, t, n in intervals[j:j + 4000] if s < b and t > a]
            if not covering:
                out["(no frame)"] += b - a
                continue
            points = sorted({a, b, *[max(a, s) for s, _, _ in covering],
                             *[min(b, t) for _, t, _ in covering]})
            for p, q in zip(points, points[1:]):
                mid = (p + q) / 2
                live = [(t - s, n) for s, t, n in covering if s <= mid < t and keyfn(n)]
                name = min(live)[1] if live else "(none)"
                out[name] += q - p
        return out

    by_frame = blame(frames, ours)
    print(f"[trace] GPU idle by innermost library / solution Python frame (ms over the window):")
    for name, us in sorted(by_frame.items(), key=lambda kv: -kv[1])[:top]:
        print(f"[trace]   {us / 1000:8.2f}  {name[:150]}", flush=True)
    by_op = blame(ops, lambda n: True)
    print(f"[trace] GPU idle by innermost CPU op / CUDA runtime call (ms over the window):")
    for name, us in sorted(by_op.items(), key=lambda kv: -kv[1])[:25]:
        print(f"[trace]   {us / 1000:8.2f}  {name[:150]}", flush=True)
    longest = sorted(gaps, key=lambda g: g[0] - g[1])[:12]
    print("[trace] longest gaps (ms from window start, length ms):")
    for a, b in longest:
        print(f"[trace]   at {(a - lo) / 1000:8.2f}  {(b - a) / 1000:6.2f}", flush=True)


def install():
    import torch
    import transformers
    from torch.profiler import ProfilerActivity, profile

    state = {"prof": None}

    class TraceCallback(transformers.TrainerCallback):
        def on_step_begin(self, args, st, control, **kwargs):
            if st.global_step == FIRST and state["prof"] is None:
                torch.cuda.synchronize()
                state["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                        with_stack=True, record_shapes=False)
                state["prof"].__enter__()
            if state["prof"] is not None:
                state["step"] = torch.profiler.record_function(f"ProfilerStep#{st.global_step}")
                state["step"].__enter__()

        def on_step_end(self, args, st, control, **kwargs):
            prof = state["prof"]
            if prof is None or "step" not in state:
                return
            torch.cuda.synchronize()
            state.pop("step").__exit__(None, None, None)
            if st.global_step - 1 == LAST:
                prof.__exit__(None, None, None)
                state["prof"] = False
                os.makedirs(OUT, exist_ok=True)
                path = os.path.join(OUT, "trace.json")
                prof.export_chrome_trace(path)
                with open(path, "rb") as f, gzip.open(path + ".gz", "wb") as g:
                    g.write(f.read())
                os.remove(path)
                analyze(path + ".gz")

    original_init = transformers.Trainer.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        self.add_callback(TraceCallback())

    transformers.Trainer.__init__ = init


if __name__ == "__main__":
    analyze(sys.argv[1])
