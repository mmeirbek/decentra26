"""Dev-only timing of the first training steps; dev/kaggle_push.py --probe-first-step adds it to the
uploaded solution/ as solution/_probe.py and calls install() from apply(). Never part of the
submitted solution.

Prints, for steps 1-3, the fused forward and the whole training step (forward, backward,
optimizer), each between two CUDA synchronizations, and the time of the first launch of every
kernel of ours in the main thread.
"""

import time

import torch


def install():
    import transformers

    from . import model

    calls = {"forward": 0, "step": 0}
    original_forward = model.fused_forward

    def fused_forward(*args, **kwargs):
        calls["forward"] += 1
        if calls["forward"] > 3:
            return original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = original_forward(*args, **kwargs)
        torch.cuda.synchronize()
        print(f"[probe] step {calls['forward']}: fused forward {time.perf_counter() - started:.3f} s",
              flush=True)
        return out

    model.fused_forward = fused_forward

    original_step = transformers.Trainer.training_step

    def training_step(self, *args, **kwargs):
        calls["step"] += 1
        if calls["step"] > 3:
            return original_step(self, *args, **kwargs)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = original_step(self, *args, **kwargs)
        torch.cuda.synchronize()
        print(f"[probe] step {calls['step']}: training_step {time.perf_counter() - started:.3f} s",
              flush=True)
        return out

    transformers.Trainer.training_step = training_step

    from triton.runtime.jit import JITFunction

    seen = set()
    original_run = JITFunction.run

    def run(self, *args, **kwargs):
        name = getattr(getattr(self, "fn", None), "__name__", "?")
        if kwargs.get("warmup") or name in seen:
            return original_run(self, *args, **kwargs)
        seen.add(name)
        torch.cuda.synchronize()
        started = time.perf_counter()
        out = original_run(self, *args, **kwargs)
        torch.cuda.synchronize()
        print(f"[probe] first launch of {name}: {1000 * (time.perf_counter() - started):.1f} ms",
              flush=True)
        return out

    JITFunction.run = run
