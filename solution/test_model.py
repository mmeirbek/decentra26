"""The fused forward/backward of a small 4-bit LoRA Qwen2 against transformers + peft themselves.

The model is built the way Soup builds the real one (fp16 weights quantized to NF4 with nested
scales, the rest widened to fp32, LoRA on q_proj and v_proj), only small. The reference is the
original Qwen2ForCausalLM.forward with peft's LoRA layers; the fused path must give the same loss,
the same TRL metrics and per-row losses, and the same LoRA gradients up to fp16 rounding.
"""

import os

import pytest
import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    DEV = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEV = "cpu"
else:
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

from solution.model import install_plan, patch_qwen2, row_segments  # noqa: E402


def tiny_4bit_model():
    """As from_pretrained leaves it: NF4 linears, everything else fp16."""
    import bitsandbytes as bnb
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=1000, hidden_size=128, intermediate_size=320,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=True, rms_norm_eps=1e-6)
    cfg._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(cfg).to(torch.float16)
    for layer in model.model.layers:
        for parent, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                              (layer.mlp, ("gate_proj", "up_proj", "down_proj"))):
            for name in names:
                lin = getattr(parent, name)
                q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                      compute_dtype=torch.float16, compress_statistics=True,
                                      quant_type="nf4")
                q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False,
                                             compress_statistics=True, quant_type="nf4",
                                             blocksize=64)
                if lin.bias is not None:
                    q.bias = torch.nn.Parameter(lin.bias.data.clone(), requires_grad=False)
                setattr(parent, name, q.to(DEV))
    model.to(DEV)
    model.is_loaded_in_4bit = True  # as from_pretrained marks it: peft then wraps with its bnb LoRA
    return model


def tiny_qlora_model(lora_dropout=0.0):
    from peft import LoraConfig, get_peft_model

    model = tiny_4bit_model()
    for param in model.parameters():  # what prepare_model_for_kbit_training does
        param.requires_grad = False
        if param.dtype == torch.float16 and param.__class__.__name__ != "Params4bit":
            param.data = param.data.float()
    peft_model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=lora_dropout,
                                                  target_modules=["q_proj", "v_proj"],
                                                  task_type="CAUSAL_LM"))
    for name, param in peft_model.named_parameters():
        if "lora_B" in name:
            param.data.normal_(0, 0.05)
    peft_model.train()
    return peft_model


def batch(vocab=1000):
    torch.manual_seed(5)
    lengths, seq = [24, 17, 9], 24
    ids = torch.randint(0, vocab, (len(lengths), seq), device=DEV)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
        labels[b, n // 3:n] = ids[b, n // 3:n]
    return ids, mask, labels


def batch_with(lengths, seq, holes=(), vocab=1000):
    torch.manual_seed(6)
    ids = torch.randint(0, vocab, (len(lengths), seq), device=DEV)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for b, n in enumerate(lengths):
        mask[b, :n] = 1
        labels[b, n // 3:n] = ids[b, n // 3:n]
    for b, t in holes:
        mask[b, t] = 0
        labels[b, t] = -100
    return ids, mask, labels


def reference(peft_model, ids, mask, labels):
    out = peft_model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    out.loss.backward()
    logits = out.logits.float().detach()
    logp = F.log_softmax(logits, -1)
    entropy = (-(logp.exp() * logp).sum(-1) * mask).sum() / mask.sum()
    shift = labels[:, 1:]
    hit = ((logits[:, :-1].argmax(-1) == shift) & (shift != -100)).sum()
    tok = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), shift.reshape(-1),
                          ignore_index=-100, reduction="none").view(shift.shape)
    rows = tok.sum(1) / (shift != -100).sum(1)
    grads = {n: p.grad.clone() for n, p in peft_model.named_parameters() if p.grad is not None}
    return out.loss.detach(), entropy, hit, rows, grads


@pytest.mark.parametrize("case", ["padded", "no padding", "hole in a row"])
def test_fused_path_matches_transformers_and_peft(case):
    patch_qwen2()
    peft_model = tiny_qlora_model()
    if case == "padded":
        ids, mask, labels = batch()
    elif case == "no padding":
        ids, mask, labels = batch_with([16, 16], 16)
    else:  # not contiguous: the fused path falls back to padded attention with the mask
        ids, mask, labels = batch_with([20, 12], 20, holes=[(0, 2)])
    ref_loss, ref_entropy, ref_hit, ref_rows, ref_grads = reference(peft_model, ids, mask, labels)
    peft_model.zero_grad(set_to_none=True)

    plan = install_plan(peft_model, require_cuda=False)
    assert plan is not None
    plan.activate()
    out = peft_model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    assert out.logits is None
    out.loss.backward()
    step = plan.last_step
    torch.testing.assert_close(out.loss.detach(), ref_loss, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(step.entropy_sum / mask.sum(), ref_entropy, rtol=1e-2, atol=1e-2)
    assert abs(int(step.correct) - int(ref_hit)) <= 1
    shift = labels[:, 1:]
    rows = step.token_loss.sum(1) / (shift != -100).sum(1)
    torch.testing.assert_close(rows, ref_rows, rtol=1e-2, atol=1e-2)
    grads = {n: p.grad for n, p in peft_model.named_parameters() if p.grad is not None}
    assert grads.keys() == ref_grads.keys() and len(grads) == 8
    for name, ref in ref_grads.items():
        cos = F.cosine_similarity(grads[name].flatten(), ref.flatten(), dim=0)
        rel = (grads[name] - ref).norm() / ref.norm()
        assert cos > 0.999 and rel < 0.05, (name, float(cos), float(rel))


def test_fused_step_with_lora_dropout_under_autocast():
    patch_qwen2()
    peft_model = tiny_qlora_model(lora_dropout=0.05)
    ids, mask, labels = batch()
    plan = install_plan(peft_model, require_cuda=False)
    plan.activate()
    with torch.autocast(device_type=DEV, dtype=torch.float16, enabled=DEV == "cuda"):
        out = peft_model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    grads = [p.grad for n, p in peft_model.named_parameters() if "lora_" in n]
    assert len(grads) == 8 and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert plan.last_step is not None


def test_inactive_plan_and_calls_without_labels_keep_the_original_path():
    patch_qwen2()
    peft_model = tiny_qlora_model()
    ids, mask, labels = batch()
    plan = install_plan(peft_model, require_cuda=False)
    out = peft_model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    assert out.logits is not None  # installed but not active: Soup's own path
    with torch.no_grad():
        out = peft_model(input_ids=ids, attention_mask=mask, use_cache=False)
    assert out.logits is not None and out.logits.shape[:2] == ids.shape
    assert plan.last_step is None


def test_embedding_is_restored_bit_for_bit_after_training():
    peft_model = tiny_qlora_model()
    plan = install_plan(peft_model, require_cuda=False)
    before = plan.embed.weight.detach().clone()
    plan.activate()
    assert plan.embed.weight.dtype == torch.float16
    plan.deactivate()
    assert plan.embed.weight.dtype == before.dtype and torch.equal(plan.embed.weight, before)


def test_kbit_preparation_keeps_the_tied_embedding_as_loaded():
    import peft
    from solution.trainer import patch_kbit_preparation

    patch_kbit_preparation()
    widened = tiny_qlora_model().get_base_model().model.embed_tokens.weight
    model = tiny_4bit_model()
    weight = model.model.embed_tokens.weight
    storage = weight.data_ptr()
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert weight.dtype == torch.float16 and weight.data_ptr() == storage  # not even copied
    assert torch.equal(weight.float(), widened)  # the values peft's fp32 copy holds
    assert model.model.norm.weight.dtype == torch.float32  # everything else as peft makes it
    assert not any(p.requires_grad for p in model.parameters())

    model.config.hidden_act = "gelu"  # a model the fused path does not take: Soup's own path
    assert install_plan(model, require_cuda=False) is None  # gets the embedding peft would give
    assert weight.dtype == torch.float32 and torch.equal(weight, widened)


def test_row_segments():
    rowmap = torch.tensor([0, 1, 2, 5, 6, 10])  # seq_len 5: rows 0 (0-2), 1 (0-1), 2 (0)
    assert row_segments(rowmap, 5) == [(0, 3), (3, 5), (5, 6)]
    assert row_segments(torch.tensor([0, 2, 5]), 5) is None  # a hole in row 0


def test_a_masked_position_with_a_label_goes_through_soups_path():
    patch_qwen2()
    peft_model = tiny_qlora_model()
    ids, mask, labels = batch_with([20, 12], 20, holes=[(0, 9)])  # label at 10 is set
    plan = install_plan(peft_model, require_cuda=False)
    plan.activate()
    with torch.autocast(device_type=DEV, dtype=torch.float16):  # as accelerate runs training
        out = peft_model(input_ids=ids, attention_mask=mask, labels=labels, use_cache=False)
    assert out.logits is not None and plan.last_step is None
