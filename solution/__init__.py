"""Fused Triton path for Soup's 4-bit LoRA SFT of Qwen2 on a T4.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. `apply()` installs four narrow patches; each one
falls back to the original code unless it sees the real 4-bit Qwen2 PEFT model on CUDA:

- `peft.prepare_model_for_kbit_training`: the tied embedding / output head stays fp16 as loaded
  instead of being widened to fp32, which the fused path never reads (solution/trainer.py);
- `SFTTrainerWrapper._setup_transformers` and `.train`: attach the fused plan to the finished
  model and keep it active only while Soup trains (solution/model.py);
- `Qwen2ForCausalLM.forward` and `__init__`: training steps with labels run the fused forward; once
  the module exists, our kernels compile in a background thread while its weights load
  (solution/warmup.py);
- `trl.SFTTrainer.compute_loss`: logs TRL's metrics and feeds Soup's rewind log from the fused
  step, since it returns no logits (solution/trainer.py).

Our kernels: NF4 dequantization (kernels_nf4.py), residual add + RMSNorm (kernels_norm.py), the
q/k/v tail with LoRA and RoPE (kernels_attn.py), the gated MLP activation (kernels_mlp.py) and the
loss over the vocabulary (kernels_loss.py).
"""


def apply():
    from .model import patch_qwen2
    from .trainer import patch_kbit_preparation, patch_soup_wrapper, patch_trl_compute_loss

    patch_qwen2()
    patch_kbit_preparation()
    patch_trl_compute_loss()
    patch_soup_wrapper()
