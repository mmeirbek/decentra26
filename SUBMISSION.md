# Submission: TEAM_NAME

## Links
- Repository, final commit: https://github.com/mmeirbek/decentra26/commit/c9a28dd411830b6fe8ac3f35051a3d555ea06634
  (after it only README.md and SUBMISSION.md changed)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  our README): https://www.kaggle.com/code/mmeirbek/soup-kernel-challenge-decentra26 (version 1)
- Demo video (up to 3 minutes): VIDEO_LINK
- Presentation: SLIDES_LINK

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| CAPTAIN_NAME | kernels, measurements, write-up | mmeirbek | mmeirbek | CAPTAIN_CONTACT |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4, public notebook) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels_nf4.py` | bitsandbytes `dequantize_blockwise` + `dequantize_4bit` (NF4 with nested scales), one launch per q\|k\|v and gate\|up group | 7.57% |
| `gated_act_kernel` | `solution/kernels_mlp.py` | fp32 gate / up, SiLU and product, and their backward (in place on the gate\|up buffer) | 3.35% |
| `add_rmsnorm_fwd_kernel` | `solution/kernels_norm.py` | residual add + `Qwen2RMSNorm` in fp32, LoRA dropout inputs, mask packing | 1.41% |
| `add_rmsnorm_bwd_kernel` | `solution/kernels_norm.py` | backward of the above, with the LoRA input gradients | 1.19% |
| `lm_loss_kernel` | `solution/kernels_loss.py` | `ForCausalLMLoss`, fp32 logit copies, TRL entropy / accuracy, Soup's per-row losses | 0.86% |
| `qkv_rope_fwd_kernel` | `solution/kernels_attn.py` | peft clone + LoRA add, casts, RoPE via slice / neg / cat | 0.47% |
| `qkv_rope_bwd_kernel` | `solution/kernels_attn.py` | backward of the above | 0.38% |
| `lora_inputs_kernel` | `solution/kernels_norm.py` | the stored LoRA dropout inputs (rebuilt in backward from `xhat` and mask bits) | 0.27% |
| **all** | | | **15.5%** (limit 5%) |

Ideas taken from other kernels, with links (we wrote the code ourselves):
- Fused large-vocabulary loss without full logits: Liger Kernel `FusedLinearCrossEntropy`
  (https://github.com/linkedin/Liger-Kernel), Unsloth fast cross entropy
  (https://github.com/unslothai/unsloth), Wijmans et al., *Cut Your Losses in Large-Vocabulary
  Language Models* (https://arxiv.org/abs/2411.09009).
- Online softmax: Milakov & Gimelshein (https://arxiv.org/abs/1805.02867); FlashAttention
  (https://arxiv.org/abs/2205.14135).
- RMSNorm / SwiGLU / RoPE in one pass: Liger Kernel, Unsloth, the Triton *Layer Normalization*
  tutorial (https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html).
- NF4 and double quantization: QLoRA (https://arxiv.org/abs/2305.14314); the dequantization
  formulas follow bitsandbytes 0.50.2.
- Activation recomputation: Chen et al., *Training Deep Nets with Sublinear Memory Cost*
  (https://arxiv.org/abs/1604.06174).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 86.5 | 52.4 | speedup 1.653x |
| of it the 64 training steps, s | 55.3 | 27.3 | 2.02x |
| Peak GPU memory, MiB | 14 345 | 2 143 | base/ours 6.694x |
| Eval loss (public rows) | 2.04963 | 2.04964 | +0.000% vs base |

Our own run with `--repeats 2` on another Kaggle machine: speedup 1.590x, memory 6.418x (the time
depends on the machine by about ±10%).

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (fp32 activation chain, 3–4 full fp32 copies of the logits,
padding, elementwise traffic, NF4 dequantization), how each kernel works, results against the base
(speed, memory, quality, the version history with A/B on one machine), limits and next steps.

## Limitations
- The fused path is written for this case: Qwen2, NF4 with nested scales, fp16 on a T4, LoRA on
  q_proj and v_proj, SDPA. Anything else runs Soup's own path unchanged.
- About 25 s of the 52 s are outside the training loop (imports, CUDA, model loading, saving; the
  base spends about 31 s there); the training loop itself is about 2.0x faster.
- Matmuls stay on cuBLAS (~73% of the GPU time, ~29 TFLOPS); NF4 dequantization is a separate
  kernel at the T4's memory bandwidth. A fused Triton "dequant + matmul" for Turing is the next
  step, as is our own GQA-aware attention (the memory-efficient kernel needs k / v repeated to 12
  heads, ~7% of the GPU time).
- Recomputing the MLP in backward is a deliberate trade: about −1.1 GB at the peak for about
  +10–13% per step.
- The peak follows the longest batch of the data (its real tokens); 4 rows of 512 tokens would
  raise it.
