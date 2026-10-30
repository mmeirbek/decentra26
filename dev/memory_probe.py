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
    sizes, counts = defaultdict(int), defaultdict(int)
    for seg in snap["segments"]:
        for block in seg["blocks"]:
            if block["state"] != "active_allocated":
                continue
            frames = block.get("frames") or []
            key = _where(frames)
            sizes[key] += block["size"]
            counts[key] += 1
    total = sum(sizes.values())
    print(f"[memprobe] {tag}: allocated {torch.cuda.memory_allocated() / 2**20:.0f} MiB "
          f"(snapshot {total / 2**20:.0f} MiB), reserved {torch.cuda.memory_reserved() / 2**20:.0f} MiB,"
          f" peak allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
    for key, size in sorted(sizes.items(), key=lambda kv: -kv[1])[:30]:
        print(f"[memprobe]   {size / 2**20:8.1f} MiB  x{counts[key]:<5d} {key}", flush=True)


def install():
    from . import kernels_loss, model

    torch.cuda.memory._record_memory_history(max_entries=200000)
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
        report("before plan.activate (model loaded, before training)")
        original_activate(self)
        report("after plan.activate")

    model.ModelPlan.activate = activate
