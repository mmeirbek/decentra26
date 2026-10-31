"""Hooks into Soup and TRL around the fused model forward.

- `peft.prepare_model_for_kbit_training` (Soup calls it while building the model): for the
  model the fused path is written for, the tied embedding / output head stays fp16 as loaded
  instead of being widened to fp32 (model.fp16_embedding_to_keep).
- `SFTTrainerWrapper._setup_transformers`: after it, the model is the final 4-bit PEFT model, so
  the fused plan is attached (model.install_plan).
- `SFTTrainerWrapper.train`: the plan is active (fp16 embedding, fused forward) only while Soup
  trains; afterwards the model is exactly what Soup built.
- `trl.SFTTrainer.compute_loss`: when the fused path runs, the model returns the loss without
  logits. TRL would read `outputs.logits` for `entropy` and `mean_token_accuracy`, so this logs both,
  and `num_tokens`, from the kernel's per-token results with TRL's own formulas, and hands each
  token's loss to Soup's rewind recorder through the attribute it reads for steps without logits
  (rewind_hf.py, `_soup_chunked_step`, added in Soup for #1693). Every other case runs TRL's
  compute_loss untouched.
"""

import torch

from .model import fp16_embedding_to_keep, install_plan, labels_only_on_real_tokens, plan_of

try:
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR
except ImportError:  # the attribute name rewind_hf.py reads
    STEP_ATTR = "_soup_chunked_step"


def _fused_step_applies(trainer, model, inputs, plan):
    args = trainer.args
    return (plan is not None and plan.active and model.training and "labels" in inputs
            and "attention_mask" in inputs and "shift_labels" not in inputs
            and "position_ids" not in inputs and not getattr(args, "use_liger_kernel", False)
            and getattr(trainer, "num_virtual_tokens", 0) == 0
            and not getattr(trainer, "aux_loss_enabled", False)
            and getattr(trainer, "label_smoother", None) is None
            and getattr(trainer, "compute_loss_func", None) is None
            and getattr(trainer, "model_accepts_loss_kwargs", True)
            and labels_only_on_real_tokens(inputs["attention_mask"], inputs["labels"]))


def _patch(owner, name, make):
    original = getattr(owner, name)
    if getattr(original, "_soupk", False):
        return
    patched = make(original)
    patched._soupk = True
    patched.__wrapped__ = original
    setattr(owner, name, patched)


def patch_trl_compute_loss():
    from transformers import Trainer
    from trl import SFTTrainer

    def make(original):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            plan = plan_of(model)
            if not _fused_step_applies(self, model, inputs, plan):
                return original(self, model, inputs, return_outputs=return_outputs,
                                num_items_in_batch=num_items_in_batch)
            mode = "train" if self.model.training else "eval"
            inputs.pop("_prediction_loss_only", None)
            inputs["use_cache"] = False
            plan.scaler = getattr(self.accelerator, "scaler", None)
            plan.last_step = None
            loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                                 num_items_in_batch=num_items_in_batch)
            step, plan.last_step = plan.last_step, None
            if step is None:
                raise RuntimeError("the fused forward did not run for a step it was set up for")
            gather = self.accelerator.gather_for_metrics
            attention_mask = inputs["attention_mask"]
            # TRL: sum(per_token_entropy * attention_mask) / attention_mask.sum()
            entropy = step.entropy_sum / attention_mask.sum()
            self._metrics[mode]["entropy"].append(gather(entropy).mean().item())
            if mode == "train":
                self._total_train_tokens += gather(attention_mask.sum()).sum().item()
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]
            correct, total = gather(step.correct), gather(step.labelled)
            total_sum = total.sum()
            accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
            setattr(self, STEP_ATTR, step)
            return (loss, outputs) if return_outputs else loss

        return compute_loss

    _patch(SFTTrainer, "compute_loss", make)


def patch_kbit_preparation():
    import peft

    def make(original):
        def prepare_model_for_kbit_training(model, *args, **kwargs):
            weight = fp16_embedding_to_keep(model)
            if weight is None:
                return original(model, *args, **kwargs)
            loaded = weight.data
            weight.data = loaded.new_empty(0, dtype=torch.float32)  # not fp16: peft leaves it be
            try:
                return original(model, *args, **kwargs)
            finally:
                weight.data = loaded
                model.__dict__["_soupk_fp16_embedding"] = True

        return prepare_model_for_kbit_training

    _patch(peft, "prepare_model_for_kbit_training", make)


def _last_step_unlocks_host_memory():
    from transformers import TrainerCallback

    from .offload import unlock_host_memory

    class Unlock(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.max_steps and state.global_step >= state.max_steps:
                torch.cuda.synchronize()  # the last step's copies are done
                unlock_host_memory()

    return Unlock()


def patch_soup_wrapper():
    from soup_cli.trainer.sft import SFTTrainerWrapper

    def make_setup(original):
        def _setup_transformers(self, cfg, tcfg):
            result = original(self, cfg, tcfg)
            model = getattr(self, "model", None)
            if isinstance(model, torch.nn.Module):
                install_plan(model)
            return result

        return _setup_transformers

    def make_train(original):
        def train(self, *args, **kwargs):
            plan = plan_of(getattr(self, "model", None))
            if plan is None:
                return original(self, *args, **kwargs)
            trainer = getattr(self, "trainer", None)
            if plan.host is not None and hasattr(trainer, "add_callback"):
                trainer.add_callback(_last_step_unlocks_host_memory())
            plan.activate()
            try:
                return original(self, *args, **kwargs)
            finally:
                plan.deactivate()

        return train

    _patch(SFTTrainerWrapper, "_setup_transformers", make_setup)
    _patch(SFTTrainerWrapper, "train", make_train)
