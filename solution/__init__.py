"""Fused Triton path for Soup's 4-bit LoRA SFT of Qwen2 on a T4.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` starts compiling our kernels and pinning
host memory in background threads, then installs narrow patches; each one falls back to the
original code unless it sees the real 4-bit Qwen2 PEFT model on CUDA:

- `peft.prepare_model_for_kbit_training`: the tied embedding / output head stays fp16 as loaded
  instead of being widened to fp32, which the fused path never reads (solution/trainer.py);
- `SFTTrainerWrapper._setup_transformers` and `.train`: attach the fused plan to the finished
  model and keep it active only while Soup trains (solution/model.py);
- `Qwen2ForCausalLM.forward`: training steps with labels run the fused forward;
- `Bnb4bitQuantize.convert` and `caching_allocator_warmup` (transformers): while the checkpoint
  loads, each projection's 4-bit codes move to pinned host memory, and no GPU block the size of
  the whole model is reserved first (solution/offload.py);
- `trl.SFTTrainer.compute_loss`: logs TRL's metrics and feeds Soup's rewind log from the fused
  step, since it returns no logits (solution/trainer.py);
- three that cut setup work outside training and change no result (solution/startup.py).

Our kernels: NF4 dequantization (kernels_nf4.py), residual add + RMSNorm (kernels_norm.py), the
q/k/v tail with LoRA and RoPE (kernels_attn.py), the gated MLP activation (kernels_mlp.py) and the
loss over the vocabulary (kernels_loss.py). The codes and what backward keeps stream between host
and GPU around them (offload.py).
"""


def apply():
    from . import warmup

    warmup.start()  # first: the compiles overlap the imports below, Soup's setup and the load
    from . import offload

    offload.prepare(warmup.QWEN25_15B)  # the pinned host memory too, before any weight copy
    from .startup import cache_nf4_table, patch_dataset_map, share_package_map

    share_package_map()
    cache_nf4_table()
    from .model import patch_qwen2
    from .offload import patch_bnb_loading, skip_allocator_warmup
    from .trainer import patch_kbit_preparation, patch_soup_wrapper, patch_trl_compute_loss

    patch_qwen2()
    patch_bnb_loading()
    skip_allocator_warmup()
    patch_kbit_preparation()
    patch_trl_compute_loss()
    patch_soup_wrapper()
    patch_dataset_map()
