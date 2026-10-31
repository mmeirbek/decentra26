"""Dev-only: where the wall time of a solution run goes outside the training steps.
dev/kaggle_push.py --probe-timeline ships it as solution/_timeline.py and calls install() from
apply(). Never part of the submitted solution.

A sampler thread reads the main thread's stack every 10 ms. At exit it prints:
  - seconds from process start to apply(), and the run's span from apply() to exit;
  - a coarse timeline: for every 0.5 s, the outermost frame of a library of interest;
  - outside the training steps, the time per innermost frame of a library of interest.
"""

import atexit
import collections
import os
import sys
import threading
import time

INTERVAL = 0.01
LIBS = ("soup_cli/", "transformers/", "peft/", "trl/", "datasets/", "accelerate/",
        "bitsandbytes/", "safetensors/", "huggingface_hub/", "torch/", "triton/", "solution/",
        "importlib", "tokenizers", "_train.py", "rich/", "pydantic")


def _since_process_start():
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except Exception:  # noqa: BLE001
        return float("nan")


def _label(frame):
    code = frame.f_code
    path = code.co_filename
    for lib in LIBS:
        at = path.find(lib)
        if at >= 0:
            return f"{path[at:]}:{code.co_name}"
    return None


def install():
    main = threading.main_thread().ident
    t0 = time.perf_counter()
    print(f"[timeline] apply() at {_since_process_start():.2f} s after process start", flush=True)
    samples = []  # (t, outermost label, innermost label, in_step)
    state = {"stop": False}

    def sample():
        while not state["stop"]:
            frame = sys._current_frames().get(main)
            labels, in_step = [], False
            while frame is not None:
                if frame.f_code.co_name in ("training_step", "_inner_training_loop_step"):
                    in_step = True
                label = _label(frame)
                if label:
                    labels.append(label)
                frame = frame.f_back
            if labels:
                outer = next((lb for lb in reversed(labels) if not lb.startswith("_train.py")),
                             labels[-1])
                samples.append((time.perf_counter() - t0, outer, labels[0], in_step,
                                tuple(labels[:6])))
            time.sleep(INTERVAL)

    thread = threading.Thread(target=sample, name="timeline", daemon=True)
    thread.start()

    def report():
        state["stop"] = True
        end = time.perf_counter() - t0
        print(f"[timeline] apply() to exit: {end:.2f} s, {len(samples)} samples", flush=True)
        buckets = collections.OrderedDict()
        for t, outer, inner, in_step, _ in samples:
            buckets.setdefault(int(t / 0.5), collections.Counter())[
                "STEP" if in_step else outer] += 1
        for b, counter in buckets.items():
            print(f"[timeline] {b * 0.5:6.1f}s  {counter.most_common(1)[0][0][:110]}", flush=True)
        outside = collections.Counter()
        stacks = collections.Counter()
        for _, _, inner, in_step, stack in samples:
            if not in_step:
                outside[inner] += INTERVAL
                stacks[" <- ".join(stack[:4])] += INTERVAL
        print("[timeline] outside the training steps, by innermost library frame (s):", flush=True)
        for label, s in outside.most_common(40):
            print(f"[timeline]   {s:6.2f}  {label[:140]}", flush=True)
        print("[timeline] ... by short stack (s):", flush=True)
        for label, s in stacks.most_common(30):
            print(f"[timeline]   {s:6.2f}  {label[:220]}", flush=True)

    atexit.register(report)
