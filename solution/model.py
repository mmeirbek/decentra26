"""The fused training forward of Qwen2ForCausalLM and the plan it runs on.

`install_plan(model)` runs once Soup has built the 4-bit PEFT model. It checks that the model is
the case this path is written for (Qwen2, NF4 with nested scales, fp16 compute, LoRA on q_proj and
v_proj, SDPA) and attaches a `ModelPlan`. Anything else keeps Soup's own path. The plan is active
only while Soup trains (trainer.py). The tied embedding / output head stays in fp16 as loaded
(`fp16_embedding_to_keep`; its values came from an fp16 checkpoint, so nothing changes), and if
peft widened it anyway, it is narrowed for training and restored afterwards. When the 4-bit
codes were moved to pinned host memory while loading (offload.patch_bnb_loading), q_proj and
v_proj follow here, and the plan streams the codes and stashes what backward keeps (offload.py).

`fused_forward` replaces Qwen2ForCausalLM.forward for training steps with labels:
  - pad positions are dropped right away (`rowmap` = flat indices of the real tokens); attention
    runs causally over each row's own tokens (blocks.attend), which is the base's padded attention
    with its mask without the pad positions; a mask that is not one contiguous run per row falls
    back to the padded call with the mask transformers builds;
  - the LoRA dropout masks come from torch's own dropout run on a tensor of the base run's shape,
    dtype and order (q then v, layer by layer), so they are the masks the base run draws;
  - the output carries the loss and no logits. Entropy, accuracy and each token's loss are left in
    `plan.last_step` for the trainer (see trainer.py), which logs them the way TRL and Soup do.
"""

import torch

from .blocks import AttnPrologue, FinalNorm, LayerPlan, PostBlock, StepState, attend
from .kernels_loss import IGNORE_INDEX, FusedLMLoss, labelled_first, loss_scale_of
from .offload import PROJECTIONS, ActivationStash, WeightStream
from . import warmup

# True: keep no MLP activations and redo gate|up in backward (-1.1 GB at the peak for 10-13% more
# time per step when everything stays on the GPU). With the kept tensors in host memory, gate|up
# goes there too and costs no GPU memory, so it is kept: faster on a T4 than recomputing it in all
# layers or in every other one (A/B on one machine).
RECOMPUTE_MLP = False


class TokenStats:
    """What one fused step leaves for TRL's metrics and Soup's rewind log."""

    def __init__(self, token_loss, entropy_sum, correct, labelled):
        self.token_loss = token_loss  # (batch, seq) fp32: CE of labels[:, t + 1] at t, else 0
        self.entropy_sum = entropy_sum
        self.correct = correct
        self.labelled = labelled


class ModelPlan:
    def __init__(self, causal_lm, recompute_mlp, host=None):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        config = causal_lm.config
        if getattr(config, "hidden_act", None) != "silu":
            raise ValueError("needs SiLU")
        if config._attn_implementation != "sdpa":
            raise ValueError("needs SDPA attention")
        if any(kind != "full_attention" for kind in getattr(config, "layer_types", [])):
            raise ValueError("sliding-window layers are not supported")
        embed = causal_lm.model.embed_tokens
        if causal_lm.lm_head.weight is not embed.weight or causal_lm.lm_head.bias is not None:
            raise ValueError("needs the output head tied to the embedding")
        self.attention = ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,
                                                               sdpa_attention_forward)
        self.causal_lm = causal_lm
        self.host = host
        self.layers = [LayerPlan(layer, recompute_mlp, host, i)
                       for i, layer in enumerate(causal_lm.model.layers[:config.num_hidden_layers])]
        self.weights = self.stash = None
        self.embed = embed
        self.norm = causal_lm.model.norm
        self.scaler = None
        self.last_step = None
        self.active = False
        self.fused_steps = 0
        self.workspace = None
        self._ones = None
        self._drop_scale = {}

    def activate(self):
        """For the training run: the tied embedding / output head is stored in fp16 (usually it
        already is, see `fp16_embedding_to_keep`). Its values are fp16 numbers (the checkpoint
        loads in fp16 on a T4; peft only widens them), so this frees half of it and the per-step
        fp16 copy autocast made for the output head, and changes no value."""
        weight = self.embed.weight
        if weight.dtype != torch.float16:
            self._embed_dtype = weight.dtype
            weight.data = weight.data.to(torch.float16)
        largest = max(group.numel for lp in self.layers
                      for group in (lp.qkv, lp.o, lp.gate_up, lp.down))
        self.workspace = torch.empty(largest, dtype=torch.float16, device=weight.device)
        if self.host is not None:
            self.weights = WeightStream(self.host, weight.device)
            self.stash = ActivationStash(self.host, weight.device)
        self.active = True

    def deactivate(self):
        """After training: the embedding goes back to the dtype Soup gave it, value for value."""
        if self.active:
            print(f"[solution] fused forward ran for {self.fused_steps} training steps", flush=True)
        self.active = False
        self.workspace = self.weights = self.stash = None  # the codes stay on the host: frozen,
        # not in the adapter, and a projection that runs its own forward takes them back first
        dtype = getattr(self, "_embed_dtype", None)
        if dtype is not None:
            self.embed.weight.data = self.embed.weight.data.to(dtype)
            self._embed_dtype = None

    def dropout_input(self, shape, device):
        """A float32 tensor of ones with the shape of the base run's dropout input."""
        numel = shape[0] * shape[1] * shape[2]
        if self._ones is None or self._ones.numel() < numel:
            self._ones = torch.ones(numel, dtype=torch.float32, device=device)
        return self._ones[:numel].view(shape)

    def keep_mask(self, ones, p):
        """The keep mask torch's dropout draws for a tensor like `ones`, and its 1/(1-p) scale as
        torch computes it (read off the output, so no host sync and no guessing the rounding)."""
        if p <= 0.0:
            return torch.ones(ones.shape, dtype=torch.bool, device=ones.device), \
                torch.ones(1, dtype=torch.float32, device=ones.device)
        out, keep = torch.native_dropout(ones, p, True)
        if p not in self._drop_scale:
            self._drop_scale[p] = out.amax().reshape(1)
        return keep, self._drop_scale[p]


def fp16_embedding_to_keep(model):
    """The tied embedding / output head weight of a 4-bit Qwen2ForCausalLM that is still fp16 as
    loaded, else None. peft.prepare_model_for_kbit_training widens it to fp32: 890 MB next to the
    445 MB it copies, the largest moment of the whole run on a T4, to hold the same fp16 values that
    the fused path reads. trainer.patch_kbit_preparation keeps this one as loaded; should the
    fused path not take the model after all, install_plan widens it as peft would have."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    if not isinstance(model, Qwen2ForCausalLM) or not getattr(model, "is_loaded_in_4bit", False):
        return None
    weight = model.model.embed_tokens.weight
    if model.lm_head.weight is not weight or weight.dtype != torch.float16:
        return None
    return weight


def _widen_kept_embedding(causal_lm):
    """Soup's own path runs after all: the embedding gets the fp32 that peft would have given it."""
    if causal_lm.__dict__.pop("_soupk_fp16_embedding", False):
        weight = causal_lm.model.embed_tokens.weight
        weight.data = weight.data.to(torch.float32)


def _codes_to_host(causal_lm, host):
    """Every projection's codes in the host block (q_proj and v_proj only now: peft has put
    their LoRA on the GPU); False if any does not fit, so nothing is streamed."""
    host.wait()
    for i, layer in enumerate(causal_lm.model.layers[:host.n_layers]):
        for name in PROJECTIONS:
            module = layer.get_submodule(name)
            module = getattr(module, "base_layer", module)
            if not host.is_placed(module.weight, i, name) and not host.place(module, i, name):
                return False
    host.settle()
    return True


def install_plan(peft_model, recompute_mlp=None, require_cuda=True):
    """Attach the fused path to the Qwen2 inside `peft_model`; returns the plan, or None (with the
    reason printed) when the model is not the case it is written for."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    if not isinstance(peft_model, torch.nn.Module):
        return None
    get_base = getattr(peft_model, "get_base_model", None)
    causal_lm = get_base() if callable(get_base) else peft_model
    if not isinstance(causal_lm, Qwen2ForCausalLM):
        return None
    host = causal_lm.__dict__.pop("_soupk_host", None)
    recompute = RECOMPUTE_MLP if recompute_mlp is None else recompute_mlp
    try:
        if require_cuda and causal_lm.model.embed_tokens.weight.device.type != "cuda":
            raise ValueError("needs a CUDA device")
        if host is not None and not _codes_to_host(causal_lm, host):
            print("[solution] 4-bit codes stay on the GPU: the host blocks do not fit them",
                  flush=True)
            host.restore(causal_lm)
            host = None
        plan = ModelPlan(causal_lm, recompute, host)
    except (ValueError, AttributeError, KeyError) as exc:
        print(f"[solution] fused path off, Soup's own path runs: {exc}", flush=True)
        if host is not None:
            host.restore(causal_lm)
        _widen_kept_embedding(causal_lm)
        return None
    causal_lm.__dict__["_soupk_plan"] = plan
    if host is not None and getattr(causal_lm, "hf_device_map", None) is None:
        # transformers 5 records no device map, so accelerate.prepare_model calls model.to(device),
        # which would copy every code back from the host. This is the map a model placed on one
        # device carries (Soup's own layer streaming sets it for the same reason).
        causal_lm.hf_device_map = {"": causal_lm.model.embed_tokens.weight.device.index}
    where = ", 4-bit codes and kept tensors in pinned host memory" if host is not None else ""
    print(f"[solution] fused path on: {len(plan.layers)} layers, packed tokens, fused loss{where}",
          flush=True)
    return plan


def plan_of(model):
    """The plan behind a (possibly PEFT-wrapped) model, or None."""
    if not isinstance(model, torch.nn.Module):
        return None
    get_base = getattr(model, "get_base_model", None)
    base = get_base() if callable(get_base) else model
    return base.__dict__.get("_soupk_plan") if isinstance(base, torch.nn.Module) else None


def row_segments(rowmap, seq_len):
    """[(start, stop)] of each row's tokens in packed order, or None when some row's real tokens
    are not one contiguous run of positions (then per-row causal attention would not be the base's
    masked attention). One small host copy; `nonzero` has already synchronized."""
    pos = rowmap.tolist()
    segments, start = [], 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] // seq_len != pos[i - 1] // seq_len:
            if pos[i - 1] - pos[start] != i - 1 - start:
                return None
            segments.append((start, i))
            start = i
    return segments


def labels_only_on_real_tokens(attention_mask, labels):
    """True when every position the loss scores (next label not -100) is a real token. Base Soup
    also scores masked positions whose next label is set; TRL's batches never have one (pads end
    the row and carry -100), and a batch that does goes through Soup's own path."""
    if labels.shape != attention_mask.shape:
        return False
    target = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
    return not bool(((attention_mask == 0) & (target != IGNORE_INDEX)).any())


def fusable_call(model, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                 use_cache, logits_to_keep, kwargs, labels):
    return (model.training and torch.is_grad_enabled() and input_ids is not None
            and input_ids.dim() == 2 and attention_mask is not None and attention_mask.dim() == 2
            and inputs_embeds is None and past_key_values is None and position_ids is None
            and not use_cache and not logits_to_keep
            and not kwargs.get("output_attentions") and not kwargs.get("output_hidden_states")
            and "shift_labels" not in kwargs
            and labels_only_on_real_tokens(attention_mask, labels))


def fused_forward(model, plan, input_ids, attention_mask, labels, num_items_in_batch=None):
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    warmup.wait()
    if plan.fused_steps == 0 and plan.host is not None:
        moved = plan.host.reclaim()
        if moved:
            print(f"[solution] {moved} projections' codes had been copied back to the GPU; they "
                  "are read from the host again", flush=True)
    plan.fused_steps += 1
    dev = input_ids.device
    batch, seq_len = input_ids.shape
    hidden = plan.embed.weight.shape[1]
    rowmap = attention_mask.reshape(-1).nonzero().squeeze(1)
    tokens = rowmap.numel()
    ids = input_ids.reshape(-1).index_select(0, rowmap)
    res = torch.nn.functional.embedding(ids, plan.embed.weight).float()
    targets = torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[:, 1:]
    targets = targets.reshape(-1).index_select(0, rowmap)
    order, n_labelled = labelled_first(targets)  # synchronizes here, while the GPU is idle anyway

    # What Qwen2Model.forward builds, from the padded batch: positions and RoPE tables; the padded
    # mask only when some row's tokens are not contiguous (then attention runs padded, as the base).
    position_ids = torch.arange(seq_len, device=dev).unsqueeze(0)
    like = res.new_empty(()).expand(batch, seq_len, hidden)
    cos, sin = model.model.rotary_emb(like, position_ids)
    segments = row_segments(rowmap, seq_len)
    mask = None
    if segments is None:
        mask = modeling_qwen2.create_causal_mask(config=model.config, inputs_embeds=like,
                                                 attention_mask=attention_mask,
                                                 past_key_values=None, position_ids=position_ids)
    step = StepState(rowmap, batch, seq_len, cos[0].contiguous(), sin[0].contiguous(), segments,
                     plan.workspace, plan.weights, plan.stash)
    if plan.stash is not None:
        plan.stash.begin_step()

    delta = torch.zeros((tokens, hidden), dtype=torch.float16, device=dev)
    ones = plan.dropout_input((batch, seq_len, hidden), dev)
    for lp in plan.layers:
        keep_q, drop_scale = plan.keep_mask(ones, lp.lora_q.dropout)
        keep_v, _ = plan.keep_mask(ones, lp.lora_v.dropout)
        res, q, k, v = AttnPrologue.apply(res, delta, keep_q, keep_v, drop_scale, lp.lora_q.a,
                                          lp.lora_q.b, lp.lora_v.a, lp.lora_v.b, lp, step)
        attn = attend(lp, q, k, v, step, plan.attention, mask)
        res, delta = PostBlock.apply(res, attn, lp, step)
    if plan.stash is not None:
        plan.stash.flush()
    x = FinalNorm.apply(res, delta, plan.norm, step)

    if num_items_in_batch is None:
        denominator = (targets != IGNORE_INDEX).sum()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(dev)
    else:
        denominator = num_items_in_batch
    loss, token_loss, entropy, hit = FusedLMLoss.apply(x, plan.embed.weight, targets, denominator,
                                                       loss_scale_of(plan.scaler, dev), order,
                                                       n_labelled)
    padded_loss = torch.zeros(batch * seq_len, dtype=torch.float32, device=dev)
    padded_loss.index_copy_(0, rowmap, token_loss)
    plan.last_step = TokenStats(padded_loss.view(batch, seq_len), entropy.sum(), hit.sum(),
                                (targets != IGNORE_INDEX).sum())
    return CausalLMOutputWithPast(loss=loss)


def patch_qwen2():
    """Route Qwen2ForCausalLM.forward through the fused path when a plan is attached."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    original = Qwen2ForCausalLM.forward
    if getattr(original, "_soupk", False):
        return

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        plan = self.__dict__.get("_soupk_plan")
        if plan is not None and plan.active and labels is not None and fusable_call(
                self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
                use_cache, logits_to_keep, kwargs, labels):
            return fused_forward(self, plan, input_ids, attention_mask, labels,
                                 kwargs.get("num_items_in_batch"))
        return original(self, input_ids=input_ids, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                        logits_to_keep=logits_to_keep, **kwargs)

    forward._soupk = True
    forward.__wrapped__ = original
    Qwen2ForCausalLM.forward = forward
