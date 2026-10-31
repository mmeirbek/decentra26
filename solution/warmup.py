"""Compile our Triton kernels while Soup loads the model.

Every measured run starts with empty compiler caches, so each kernel variant costs a compile (one to
a few seconds on Kaggle's CPU) before its first launch. The compiles start in a background thread of
this same process as soon as `apply()` runs, with the sizes of the task's model
(Qwen2.5-1.5B-Instruct), and overlap the imports, Soup's setup and the model load; a model of other
sizes compiles its own variants at their first launch instead. Each compiled kernel is also made ready to
launch there: Triton builds a small C launcher for it and loads the binary on its first launch,
about 0.3 s per kernel on Kaggle that would otherwise fall into the first training step. The
kernels compile in the order of their first use: the first fused forward waits for the forward
ones only, the first backward for the rest. Nothing here launches a kernel on the GPU
(`warmup=True` compiles only), nothing is read from disk, and nothing is cached across runs.
"""

import threading
import time
import types

import torch

QWEN25_15B = types.SimpleNamespace(model_type="qwen2", hidden_size=1536, intermediate_size=8960,
                                   num_attention_heads=12, num_key_value_heads=2, head_dim=128,
                                   vocab_size=151936, num_hidden_layers=28)

_thread = None
_forward_ready = threading.Event()  # set once the kernels the forward pass launches are ready
_waited = {"forward": False, "backward": False}


def start(config=QWEN25_15B):
    """Start compiling for a Qwen2 config, once per process, when a GPU is there."""
    global _thread
    if _thread is not None or not torch.cuda.is_available():
        return
    if getattr(config, "model_type", None) != "qwen2":
        return
    _thread = threading.Thread(target=_compile_all, args=(config,), name="solution-compile",
                               daemon=True)
    _thread.start()


def wait(backward=False):
    """Block until the kernels of the forward pass (or, with `backward`, all of them) are ready."""
    which = "backward" if backward else "forward"
    if _thread is None or _waited[which]:
        return
    started = time.perf_counter()
    if backward:
        _thread.join()
    else:
        _forward_ready.wait()
    _waited[which] = True
    print(f"[solution] first fused {which} waited {time.perf_counter() - started:.2f} s for the "
          "background compile", flush=True)


def _compile_all(config):
    import triton

    from .blocks import ACT_BLOCK, ACT_WARPS
    from .kernels_attn import qkv_rope_bwd_kernel, qkv_rope_fwd_kernel
    from .kernels_loss import VOCAB_BLOCK, lm_loss_kernel
    from .kernels_mlp import gated_act_kernel
    from .kernels_nf4 import NF4_WARPS, WORDS_PER_PROGRAM, nf4_dequant_kernel
    from .kernels_norm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel, lora_inputs_kernel

    started = time.perf_counter()
    try:
        dev = torch.device("cuda", torch.cuda.current_device())

        def t(dtype, n=16):
            return torch.zeros(n, dtype=dtype, device=dev)

        def ready(kernel):  # build the launcher and load the binary now, not at the first launch
            init = getattr(kernel, "_init_handles", None)
            if callable(init):
                init()

        f16, f32, i64, u8 = torch.float16, torch.float32, torch.int64, torch.uint8
        hidden, vocab = config.hidden_size, config.vocab_size
        n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden // n_q
        block = triton.next_power_of_2(hidden)
        grid = (1,)
        ready(nf4_dequant_kernel.warmup(t(torch.int32), t(u8), t(f32), t(f32), t(f32), t(f16),
                                        t(torch.int32), t(i64), t(i64), t(i64), t(f32),
                                        BLOCK=WORDS_PER_PROGRAM, num_warps=NF4_WARPS,
                                        enable_fp_fusion=False, grid=grid))
        ready(add_rmsnorm_fwd_kernel.warmup(
            t(f32), t(f16), t(f32), t(f32), t(f32), t(f16), t(f16), t(i64), t(torch.bool),
            t(torch.bool), t(f32), t(f16), t(f16), t(u8), t(u8), 1e-6, 0, H=hidden, BLOCK=block,
            num_warps=8, grid=grid))

        def rope(kernel):
            ready(kernel.warmup(t(f16), t(f16), t(f16), t(f32), t(f32), t(i64), t(f16), t(f16),
                                t(f16), 2.0, 2.0, 512, NQ=n_q, NKV=n_kv, D=head_dim, num_warps=1,
                                grid=grid))

        rope(qkv_rope_fwd_kernel)
        ready(gated_act_kernel.warmup(t(f16), t(f16), 0, I=config.intermediate_size,
                                      BLOCK=ACT_BLOCK, num_warps=ACT_WARPS, grid=grid))
        ready(lm_loss_kernel.warmup(t(f16), t(i64), t(f32), t(f32), t(f32), t(torch.int32), vocab,
                                    0, V=vocab, BLOCK=VOCAB_BLOCK, num_warps=8, grid=grid))
        _forward_ready.set()
        ready(add_rmsnorm_bwd_kernel.warmup(
            t(f16), t(f16), t(f32), t(f32), t(f32), t(f32), t(f16), t(u8), t(u8), t(f32), t(f16),
            t(f16), 0, H=hidden, BLOCK=block, num_warps=8, grid=grid))
        ready(lora_inputs_kernel.warmup(t(f16), t(f32), t(u8), t(u8), t(f32), t(f16), t(f16),
                                        H=hidden, BLOCK=block, num_warps=8, grid=grid))
        rope(qkv_rope_bwd_kernel)
        print(f"[solution] background compile of 8 kernel variants took "
              f"{time.perf_counter() - started:.1f} s", flush=True)
    except Exception as exc:  # noqa: BLE001 - a failed warm-up only means compiling at first use
        print(f"[solution] background compile skipped: {exc!r}", flush=True)
    finally:
        _forward_ready.set()
