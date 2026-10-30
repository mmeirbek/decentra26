"""Loss over the 151,936-token vocabulary without full logits.

Base Soup builds the logits of the whole batch, and from them: ForCausalLMLoss's fp32 copy and
saved log-softmax, accelerate's fp32 copy of the model output, TRL's entropy (128 rows at a time)
and its accuracy (one more full fp32 copy), and Soup's per-row losses for rewind.jsonl. That is
about 4 GB at the peak and several passes over 1.2 GB per step.

Here the output head runs on `CHUNK` tokens at a time (cuBLAS, fp16 as under autocast). One Triton
program per token walks its row of logits twice:
  1. online max / sum of exp / sum of exp * logit, the first index of the maximum and the label's
     logit: cross-entropy, entropy and the argmax hit for accuracy, all in fp32;
  2. writes the gradient (softmax - one_hot) * scale over the logits it read, in fp16, where
     `scale` is the GradScaler's current loss scale over the number of labelled tokens. That is the
     value base Soup's backward rounds to fp16, so small probabilities survive the same way.
The gradient with respect to the hidden states is then one more matmul per chunk, taken during
the forward pass; backward only hands it on. Only one chunk of logits exists at a time.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK = 128
VOCAB_BLOCK = 4096


@triton.jit(do_not_specialize=["with_grad"])
def lm_loss_kernel(logits_ptr, target_ptr, gscale_ptr, loss_ptr, entropy_ptr, hit_ptr, row_stride,
                   with_grad, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * row_stride
    target = tl.load(target_ptr + row)
    labelled = target != -100
    target = tl.where(labelled, target, 0)

    run_max = tl.full([], float("-inf"), tl.float32)
    sum_exp = tl.zeros([], tl.float32)
    sum_exp_x = tl.zeros([], tl.float32)
    best = tl.full([], float("-inf"), tl.float32)
    best_idx = tl.zeros([], tl.int32)
    for start in range(0, V, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        live = cols < V
        x = tl.load(base + cols, mask=live, other=float("-inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_idx = tl.argmax(x, axis=0)
        best_idx = tl.where(block_max > best, start + block_idx, best_idx)
        best = tl.maximum(best, block_max)
        new_max = tl.maximum(run_max, block_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.where(live, tl.exp(x - new_max), 0.0)
        sum_exp = sum_exp * rescale + tl.sum(e, axis=0)
        sum_exp_x = sum_exp_x * rescale + tl.sum(e * tl.where(live, x, 0.0), axis=0)
        run_max = new_max

    lse = run_max + tl.log(sum_exp)
    label_logit = tl.load(base + target).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - label_logit, 0.0))
    tl.store(entropy_ptr + row, lse - sum_exp_x / sum_exp)
    tl.store(hit_ptr + row, (labelled & (best_idx == target)).to(tl.int32))

    if with_grad != 0:
        g = tl.where(labelled, tl.load(gscale_ptr), 0.0)
        for start in range(0, V, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            live = cols < V
            x = tl.load(base + cols, mask=live, other=0.0).to(tl.float32)
            grad = (tl.exp(x - lse) - tl.where(cols == target, 1.0, 0.0)) * g
            tl.store(base + cols, grad.to(tl.float16), mask=live)


def loss_scale_of(scaler, device):
    """The loss scale the GradScaler will multiply this step's loss by, as a 1-element tensor on
    the device (no host sync). 1.0 without a scaler."""
    scale = getattr(scaler, "_scale", None) if scaler is not None else None
    if scale is not None and getattr(scaler, "is_enabled", lambda: True)():
        return scale.reshape(1).to(device=device, dtype=torch.float32)
    init = getattr(scaler, "_init_scale", None) if scaler is not None else None
    enabled = scaler is not None and getattr(scaler, "is_enabled", lambda: True)()
    value = float(init) if (enabled and init is not None) else 1.0
    return torch.full((1,), value, dtype=torch.float32, device=device)


class FusedLMLoss(torch.autograd.Function):
    """sum over labelled tokens of cross-entropy(hidden @ weight.T, target) / denominator.

    Also returns, without gradients, each token's cross-entropy, entropy and argmax hit (TRL's
    entropy covers every real token, so every token's logits are built). Only labelled tokens
    have a gradient: tokens go labelled first (`order`, `n_labelled` from the caller), the second
    pass of the kernel and the gradient matmul run on those only, and the rest get exact zeros as
    before, without the work.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, denominator, loss_scale, order, n_labelled):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return FusedLMLoss._forward(ctx, hidden, weight, targets, denominator, loss_scale,
                                        order, n_labelled)

    @staticmethod
    def _forward(ctx, hidden, weight, targets, denominator, loss_scale, order, n_labelled):
        n_tokens, vocab = hidden.shape[0], weight.shape[0]
        dev = hidden.device
        with_grad = hidden.requires_grad
        hidden = hidden.index_select(0, order)
        targets = targets.index_select(0, order)
        token_loss = torch.empty(n_tokens, dtype=torch.float32, device=dev)
        entropy = torch.empty_like(token_loss)
        hit = torch.empty(n_tokens, dtype=torch.int32, device=dev)
        grad = torch.zeros_like(hidden) if with_grad else None
        gscale = (loss_scale / denominator).to(torch.float32)
        logits = torch.empty((min(CHUNK, n_tokens), vocab), dtype=hidden.dtype, device=dev)
        for start in range(0, n_tokens, CHUNK):
            stop = min(start + CHUNK, n_tokens)
            labelled_stop = min(stop, n_labelled) if with_grad else start
            chunk_logits = logits[:stop - start]
            torch.mm(hidden[start:stop], weight.t(), out=chunk_logits)
            lm_loss_kernel[(stop - start,)](
                chunk_logits, targets[start:stop], gscale, token_loss[start:stop],
                entropy[start:stop], hit[start:stop], vocab, int(labelled_stop > start), V=vocab,
                BLOCK=VOCAB_BLOCK, num_warps=8)
            if labelled_stop > start:
                torch.mm(chunk_logits[:labelled_stop - start], weight,
                         out=grad[start:labelled_stop])
        del logits, hidden
        back = torch.empty_like(order)
        back[order] = torch.arange(n_tokens, device=dev)
        token_loss, entropy, hit = (t.index_select(0, back) for t in (token_loss, entropy, hit))
        loss = token_loss.sum() / denominator
        if with_grad:
            ctx.save_for_backward(grad.index_select(0, back), loss_scale)
        ctx.mark_non_differentiable(token_loss, entropy, hit)
        return loss, token_loss, entropy, hit

    @staticmethod
    def backward(ctx, grad_loss, _token_loss, _entropy, _hit):
        grad_hidden, loss_scale = ctx.saved_tensors
        # grad_hidden already carries the loss scale it was rounded at; grad_loss is that scale
        # (the GradScaler's), so the ratio is exactly 1 and the multiply changes nothing.
        ratio = (grad_loss.reshape(1).float() / loss_scale).to(grad_hidden.dtype)
        return grad_hidden * ratio, None, None, None, None, None, None


def labelled_first(targets):
    """A stable order with the labelled tokens first, and how many there are (one host sync)."""
    unlabelled = (targets == IGNORE_INDEX).to(torch.int8)
    order = torch.argsort(unlabelled, stable=True)
    return order, int(targets.numel() - unlabelled.sum())
