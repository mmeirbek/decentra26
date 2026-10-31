"""Dev-only: what is alive on the GPU at the end of the forward pass of step 3 (the peak), grouped by
the line that allocated it. dev/kaggle_push.py --probe-memory ships it as solution/_memprobe.py and
calls install() from apply(). Never part of the submitted solution.
"""

from collections import defaultdict

import torch


def _where(frames):
    """The first frame in our code, else the first in torch / transformers / peft / bnb / soup."""
    for frame in frames:
        name = frame.get("filename", "")
        if "/solution/" in name:
            return f"solution/{name.split('/solution/')[1]}:{frame.get('line')} {frame.get('name')}"
    for frame in frames:
        name = frame.get("filename", "")
        for lib in ("transformers", "peft", "bitsandbytes", "soup_cli", "trl", "accelerate", "torch"):
            if f"/{lib}/" in name:
                return f"{lib}/{name.split(f'/{lib}/')[1]}:{frame.get('line')} {frame.get('name')}"
    return "?"


def report(tag):
    snap = torch.cuda.memory._snapshot()
    sizes, counts, example = defaultdict(int), defaultdict(int), {}
    for seg in snap["segments"]:
        for block in seg["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = block.get("frames") or []
            key = _where(frames)
            sizes[key] += block["size"]
            counts[key] += 1
            if block["size"] > example.get(key, (0, None))[0]:
                example[key] = (block["size"], frames)
    total = sum(sizes.values())
    print(f"[memprobe] {tag}: allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB "
          f"(snapshot {total / 2**20:.0f} MiB), reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB,"
          f" peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
    for key, size in sorted(sizes.items(), key=lambda kv: -kv[1])[:30]:
        print(f"[memprobe]   {size / 2**20:8.1f} MiB  x{counts[key]:<5d} {key}", flush=True)
    for key, size in sorted(sizes.items(), key=lambda kv: -kv[1])[:2]:
        print(f"[memprobe]   stack of the largest block of {key}:", flush=True)
        shown = 0
        for frame in example[key][1] or []:
            name = frame.get("filename", "")
            if name.endswith(".py") and shown < 30:
                shown += 1
                print(f"[memprobe]       {name.split('-packages/')[-1]}:{frame.get('line')} "
                      f"{frame.get('name')}", flush=True)


def phase(tag):
    """Peak allocated / reserved since the previous phase mark, then a fresh peak."""
    print(f"[memprobe] phase peak up to '{tag}': allocated "
          f"{torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, reserved "
          f"{torch.cuda.max_memory_reserved() / 2**20:.0f} MiB (now {torch.cuda.memory_allocated() / 2**20:.0f}"
          f" / {torch.cuda.memory_reserved() / 2**20:.0f})", flush=True)
    torch.cuda.reset_peak_memory_stats()


def install():
    import peft
    import transformers

    from . import kernels_loss, model, trainer

    torch.cuda.memory._record_memory_history(max_entries=200000)
    phase("apply")

    prepare = peft.prepare_model_for_kbit_training

    def prepare_model_for_kbit_training(*a, **k):
        phase("model loaded (before peft's k-bit preparation)")
        out = prepare(*a, **k)
        phase("peft's k-bit preparation")
        return out

    peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    original_install = trainer.install_plan

    def install_plan(*a, **k):
        phase("get_peft_model and the rest of Soup's setup")
        out = original_install(*a, **k)
        phase("install_plan")
        return out

    trainer.install_plan = install_plan

    class StepPeaks(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step <= 8 or state.global_step % 16 == 0:
                phase(f"step {state.global_step - 1}")

        def on_train_end(self, args, state, control, **kwargs):
            phase("last steps")

    original_init = transformers.Trainer.__init__

    def init(self, *a, **k):
        original_init(self, *a, **k)
        self.add_callback(StepPeaks())

    transformers.Trainer.__init__ = init
    calls = {"n": 0}
    original = kernels_loss.FusedLMLoss.forward

    def forward(ctx, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            report("step 3, before the loss (end of the layers' forward)")
        out = original(ctx, *args, **kwargs)
        if calls["n"] == 3:
            report("step 3, after the loss forward")
        return out

    kernels_loss.FusedLMLoss.forward = staticmethod(forward)

    original_activate = model.ModelPlan.activate

    def activate(self):
        phase("trainer setup")
        report("before plan.activate (model loaded, before training)")
        original_activate(self)
        report("after plan.activate")
        phase("plan.activate")

    model.ModelPlan.activate = activate
