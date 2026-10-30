"""Dev-only timing of the first training steps; dev/kaggle_push.py --probe-first-step adds it to the
uploaded solution/ as solution/_probe.py and calls install() from apply(). Never part of the
submitted solution.

Prints, for steps 1-3, the fused forward and the whole training step (forward, backward,
optimizer), each between two CUDA synchronizations, and the time of the first launch of every
kernel of ours in the main thread.
"""

import atexit
import os
import time

import torch


def age():
    """Seconds since this process started (Linux /proc), the clock the harness's wall time uses."""
    try:
        with open("/proc/self/stat") as f:
            start_ticks = int(f.read().rsplit(")", 1)[1].split()[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return uptime - start_ticks / os.sysconf("SC_CLK_TCK")
    except OSError:
        return float("nan")


def mark(what):
    print(f"[probe] t={age():6.2f} s  {what}", flush=True)


def install():
    import transformers

    from . import model
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    mark("apply() done")
    atexit.register(mark, "interpreter exit")

    def around(owner, name, label):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            mark(f"{label} start")
            try:
                return original(*args, **kwargs)
            finally:
                mark(f"{label} end")

        setattr(owner, name, wrapped)

    around(SFTTrainerWrapper, "_setup_transformers", "model load + LoRA")
    around(SFTTrainerWrapper, "train", "Soup train()")
    around(transformers.Trainer, "train", "HF Trainer.train()")
    around(transformers.Trainer, "save_model", "save_model")
    original_init = Qwen2ForCausalLM.__init__

    def init(self, *args, **kwargs):
        mark("Qwen2ForCausalLM built (weights load next)")
        original_init(self, *args, **kwargs)

    Qwen2ForCausalLM.__init__ = init

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
