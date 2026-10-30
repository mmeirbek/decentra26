"""Compile our Triton kernels while Soup loads the model.

Every measured run starts with empty compiler caches, so each kernel variant costs a compile (one to
a few seconds on Kaggle's CPU) before its first launch. `from_pretrained` builds the Qwen2 module
from its config first and then spends most of its time reading and quantizing the weights, so the
compiles start in a background thread of this same process as soon as the module exists, with the
sizes from that config (model.patch_qwen2 calls `start`). The first fused forward waits for the
thread. Nothing here launches a kernel on the GPU (`warmup=True` compiles only), nothing is read
from disk, and nothing is cached across runs.
"""

import threading
import time

import torch

_thread = None
_waited = False


def start(config):
    """Start compiling for a Qwen2 config, once per process, when a GPU is there."""
    global _thread
    if _thread is not None or not torch.cuda.is_available():
        return
    if getattr(config, "model_type", None) != "qwen2":
        return
    _thread = threading.Thread(target=_compile_all, args=(config,), name="solution-compile",
                               daemon=True)
    _thread.start()


def wait():
    global _waited
    if _thread is None or _waited:
        return
    started = time.perf_counter()
    _thread.join()
    _waited = True
    print(f"[solution] first fused step waited {time.perf_counter() - started:.1f} s for the "
          "background compile", flush=True)


def _compile_all(config):
    import triton

    from .blocks import ACT_BLOCK
    from .kernels_attn import qkv_rope_bwd_kernel, qkv_rope_fwd_kernel
    from .kernels_loss import VOCAB_BLOCK, lm_loss_kernel
    from .kernels_mlp import gated_act_kernel
    from .kernels_nf4 import WORDS_PER_PROGRAM, nf4_dequant_kernel
    from .kernels_norm import add_rmsnorm_bwd_kernel, add_rmsnorm_fwd_kernel

    started = time.perf_counter()
    try:
        dev = torch.device("cuda", torch.cuda.current_device())

        def t(dtype, n=16):
            return torch.zeros(n, dtype=dtype, device=dev)

        f16, f32, i64, u8 = torch.float16, torch.float32, torch.int64, torch.uint8
        hidden, vocab = config.hidden_size, config.vocab_size
        n_q, n_kv = config.num_attention_heads, config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden // n_q
        block = triton.next_power_of_2(hidden)
        grid = (1,)
        nf4_dequant_kernel.warmup(t(torch.int32), t(u8), t(f32), t(f32), t(f32), t(f32), t(f16),
                                  1024, BLOCK=WORDS_PER_PROGRAM, num_warps=4,
                                  enable_fp_fusion=False, grid=grid)
        for with_lora in (True, False):
            add_rmsnorm_fwd_kernel.warmup(t(f32), t(f16), t(f32), t(f32), t(f32), t(f16), t(f16),
                                          t(i64), t(torch.bool), t(torch.bool), t(f32), t(f16),
                                          t(f16), t(u8), t(u8), 1e-6, H=hidden, BLOCK=block,
                                          WITH_LORA=with_lora, num_warps=8, grid=grid)
            add_rmsnorm_bwd_kernel.warmup(t(f16), t(f16), t(f32), t(f32), t(f32), t(f32), t(f16),
                                          t(u8), t(u8), t(f32), t(f16), t(f16), H=hidden,
                                          BLOCK=block, WITH_LORA=with_lora, num_warps=8, grid=grid)
        for kernel in (qkv_rope_fwd_kernel, qkv_rope_bwd_kernel):
            kernel.warmup(t(f16), t(f16), t(f16), t(f32), t(f32), t(i64), t(f16), t(f16), t(f16),
                          2.0, 2.0, 512, NQ=n_q, NKV=n_kv, D=head_dim, num_warps=1, grid=grid)
        for backward in (False, True):
            gated_act_kernel.warmup(t(f16), t(f16), I=config.intermediate_size, BLOCK=ACT_BLOCK,
                                    BACKWARD=backward, num_warps=4, grid=grid)
        lm_loss_kernel.warmup(t(f16), t(i64), t(f32), t(f32), t(f32), t(torch.int32), vocab,
                              V=vocab, BLOCK=VOCAB_BLOCK, WITH_GRAD=True, num_warps=8, grid=grid)
        print(f"[solution] background compile of 10 kernel variants took "
              f"{time.perf_counter() - started:.1f} s", flush=True)
    except Exception as exc:  # noqa: BLE001 - a failed warm-up only means compiling at first use
        print(f"[solution] background compile skipped: {exc!r}", flush=True)
