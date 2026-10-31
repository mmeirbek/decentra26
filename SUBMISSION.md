# Submission: Angels

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-2ab274f7-angels/commit/c3225300bc765fcc9f40478c6a4862ba64351969
  (after it only README.md and SUBMISSION.md changed; public mirror with the same history and
  hashes, which the Kaggle notebook clones: https://github.com/mmeirbek/decentra26)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  our README): https://www.kaggle.com/code/mmeirbek/soup-kernel-challenge-decentra26?scriptVersionId=364184757 (version 4)
- Demo video (up to 3 minutes) and presentation, one Google Drive folder:
  https://drive.google.com/drive/folders/1pr8es_StRZTiHbMatobY_ha1orPD0qFi

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Нурдаулет Мейрбек (Nurdaulet Meirbek) | captain; kernels, offload, measurements, write-up | [mmeirbek](https://github.com/mmeirbek) | [mmeirbek](https://www.kaggle.com/mmeirbek) | Telegram [@mmeirbek](https://t.me/mmeirbek) |
## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4, public notebook) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/kernels_nf4.py` | bitsandbytes `dequantize_blockwise` + `dequantize_4bit` (NF4 with nested scales), one launch per q\|k\|v and gate\|up group | 9.15% |
| `gated_act_kernel` | `solution/kernels_mlp.py` | fp32 gate / up, SiLU and product, and their backward (in place on the gate\|up buffer) | 3.90% |
| `add_rmsnorm_fwd_kernel` | `solution/kernels_norm.py` | residual add + `Qwen2RMSNorm` in fp32, LoRA dropout inputs, mask packing | 1.77% |
| `add_rmsnorm_bwd_kernel` | `solution/kernels_norm.py` | backward of the above, with the LoRA input gradients | 1.45% |
| `lm_loss_kernel` | `solution/kernels_loss.py` | `ForCausalLMLoss`, fp32 logit copies, TRL entropy / accuracy, Soup's per-row losses | 0.99% |
| `qkv_rope_fwd_kernel` | `solution/kernels_attn.py` | peft clone + LoRA add, casts, RoPE via slice / neg / cat | 0.57% |
| `qkv_rope_bwd_kernel` | `solution/kernels_attn.py` | backward of the above | 0.44% |
| `lora_inputs_kernel` | `solution/kernels_norm.py` | the stored LoRA dropout inputs (rebuilt in backward from `xhat` and mask bits) | 0.34% |
| **all** | | | **18.6%** (limit 5%) |

Around the kernels (not kernels; solution/offload.py, solution/startup.py): the 4-bit codes of every
layer and what each layer keeps for its backward live in pinned host memory and stream over PCIe on
side streams next to the compute (two GPU slots for the codes, the kept tensors one layer ahead);
setup work outside training is cut (TRL's package map, in-memory `datasets.map` fingerprints,
bitsandbytes' NF4 table, the bf16 checkpoint converted to fp16 on the GPU, not the CPU). Nothing of it changes a value.

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
- Offloading weights and activations with copies overlapped with compute: Rhu et al., *vDNN*
  (https://arxiv.org/abs/1602.08124); Ren et al., *ZeRO-Offload* (https://arxiv.org/abs/2101.06840);
  our teammate's parallel solution (two weight slots, activation offload without `record_stream`,
  the two setup patches).

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json` of the notebook above)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 90.2 | 45.2 | speedup 1.996x |
| of it the 64 training steps, s | 56.1 | 24.2 | 2.32x |
| Peak GPU memory, MiB | 14 345 | 1 061 | base/ours 13.520x |
| Eval loss (public rows) | 2.04960 | 2.04964 | +0.002% vs base |

The time depends on the Kaggle machine: the base took 90.2 s in this run and 79-82 s in our A/B runs on
one machine, where the speedup is 1.68-1.72x (ours 46.4-47.9 s).

- [x] All 10 checks of `harness/check_rules.py` passed
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (fp32 activation chain, 3–4 full fp32 copies of the logits,
padding, elementwise traffic, NF4 dequantization), how each kernel works, results against the base
(speed, memory, quality, the version history with A/B on one machine), limits and next steps.

## Limitations
- The fused path is written for this case: Qwen2, NF4 with nested scales, fp16 on a T4, LoRA on
  q_proj and v_proj, SDPA. Anything else runs Soup's own path unchanged.
- About 21 s of the 45 s are outside the training loop; ~16 s of that run before
  `apply()` (Python, torch, CUDA, transformers: the harness's own, shared with the base).
- Matmuls stay on cuBLAS (~73% of the GPU time, ~29 TFLOPS); NF4 dequantization is a separate
  kernel at the T4's memory bandwidth. A fused Triton "dequant + matmul" for Turing is the next
  step, as is our own GQA-aware attention (the memory-efficient kernel needs k / v repeated to 12
  heads, ~7% of the GPU time).
- The run keeps 3.5 GiB of pinned host memory and moves ~1.3 GB of codes and up to ~2 GB of kept
  tensors over PCIe per step; a machine with a slower bus would wait on those copies longer.
- The tied embedding / output head (445 MiB) stays on the GPU: the head reads all of it twice a
  step, and copying it over PCIe would cost more than a second per run.
- The peak follows the longest batch of the data (its real tokens); 4 rows of 512 tokens would
  raise it.
