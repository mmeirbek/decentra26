"""The frozen 4-bit weights and what the backward keeps live in pinned host memory; the GPU holds
only the layer at work and the next one.

PCIe on the Kaggle T4 moves 12.3 GB/s to the GPU and 13.1 GB/s back, on copy engines that run
next to the compute: a 680 MB copy beside a matmul loop leaves the loop's time unchanged. So:

- `HostWeights`: the NF4 codes of every decoder layer (q|k|v|o|gate|up|down, 22.3 MiB a layer on
  Qwen2.5-1.5B) get a pinned host block, allocated in a background thread as soon as the model's
  config is known. While transformers loads and quantizes the checkpoint, each projection's codes
  move there right after quantization (`patch_bnb_loading`), except q_proj and v_proj: peft puts
  the LoRA weights on the device of their base weight, so those two follow once peft has built
  the model (`install_plan`). The scales stay on the GPU (1/32 of the codes). A projection that
  runs its own forward later gets its codes back on the GPU first (a forward pre-hook), so Soup's
  own path works with any of it.
- `WeightStream`: two GPU slots that the layers take in turn; while layer i computes, a copy
  stream fills the other slot with the layer that comes next (i + 1 in forward, i - 1 in
  backward). Events order each copy after the last use of the slot and each use after its copy.
- `ActivationStash`: what each layer keeps for its backward is copied to a pinned host arena on a
  side stream right after the layer's forward made it, and copied back one layer ahead in the
  backward. The GPU sources stay referenced until the compute stream has waited for their copy,
  so their memory goes back to the compute stream's own pool (no `record_stream`). The last layer
  keeps its tensors on the GPU: the backward starts with it.

Nothing here changes a value: the same bytes are read by the same kernels, only from elsewhere.
"""

import threading
import time

import torch

PROJECTIONS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
AFTER_PEFT = ("self_attn.q_proj", "self_attn.v_proj")  # LoRA targets: placed after get_peft_model
ALIGN = 256
MAX_STEP_TOKENS = 4 * 512  # batch 4, max_length 512: the most real tokens a step can have
SMALL = 64 << 10  # tensors under 64 KB stay on the GPU (rstd, LoRA weights, scales)
KEEP_LAST = 1  # layers at the end whose saved tensors stay on the GPU


def _aligned(n):
    return -(-n // ALIGN) * ALIGN


def _pinned(nbytes):
    """`nbytes` of page-locked host memory. Registered with cudaHostRegister at their exact size:
    PyTorch's pinned allocator rounds every block up to a power of two (4.35 GB pinned for our
    3.5 GiB), and each pinned byte costs time to lock and again to unlock when the process ends
    (on Kaggle 3.5 GiB: 1.9 s and 0.65 s that way, 1.5 s and 0.26 s registered)."""
    buf = torch.empty(nbytes + 4096, dtype=torch.uint8)
    view = buf[(-buf.data_ptr()) % 4096:][:nbytes]  # page-aligned; the view keeps buf alive
    try:
        cudart = torch.cuda.cudart()
        if cudart.cudaHostRegister(view.data_ptr(), nbytes, 0) == cudart.cudaError.success \
                and view.is_pinned():
            _registered.append(view.data_ptr())
            return view
    except Exception:  # noqa: BLE001 - PyTorch's pinned allocator instead
        pass
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


_registered = []  # host addresses this process registered


def unlock_host_memory():
    """After the last training step: unlock the registered host memory in a background thread,
    while Soup saves the checkpoint and the adapter, instead of when the process ends (the
    memory stays valid, only pageable again: nothing copies from it asynchronously any more)."""
    def unlock(addresses):
        cudart = torch.cuda.cudart()
        for address in addresses:
            cudart.cudaHostUnregister(address)

    addresses, _registered[:] = list(_registered), []
    if addresses:
        threading.Thread(target=unlock, args=(addresses,), name="solution-unpin",
                         daemon=True).start()


class HostWeights:
    """Pinned host blocks for the NF4 codes of a Qwen2 model's decoder layers."""

    def __init__(self, config, stash_bytes_per_token=0):
        self.layout = _layout(config, stash_bytes_per_token)
        hidden, inter = config.hidden_size, config.intermediate_size
        heads, kv = config.num_attention_heads, config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden // heads
        shapes = {"self_attn.q_proj": (heads * head_dim, hidden),
                  "self_attn.k_proj": (kv * head_dim, hidden),
                  "self_attn.v_proj": (kv * head_dim, hidden),
                  "self_attn.o_proj": (hidden, heads * head_dim),
                  "mlp.gate_proj": (inter, hidden), "mlp.up_proj": (inter, hidden),
                  "mlp.down_proj": (hidden, inter)}
        self.offsets, at = {}, 0
        for name in PROJECTIONS:
            rows, cols = shapes[name]
            self.offsets[name] = (at, rows * cols // 2)  # two 4-bit codes per byte
            at += rows * cols // 2
        self.layer_bytes = at
        self.n_layers = config.num_hidden_layers
        self.blocks = [None] * self.n_layers
        self.arena_bytes = _aligned(MAX_STEP_TOKENS * stash_bytes_per_token) + 32 * ALIGN \
            if stash_bytes_per_token else 0
        self.arenas = [None] * self.n_layers
        self.placed = {}  # id(param) -> (param, module, gpu device, layer, projection)
        self._copies = []  # (event, GPU codes) of copies to the host still under way
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._allocate, name="solution-pin", daemon=True)
        self._thread.start()

    def _allocate(self):
        started = time.perf_counter()
        try:
            for i in range(self.n_layers):
                block = _pinned(self.layer_bytes)
                with self._lock:
                    self.blocks[i] = block
            if self.arena_bytes:
                for i in range(self.n_layers - KEEP_LAST):
                    arena = _pinned(self.arena_bytes)
                    with self._lock:
                        self.arenas[i] = arena
            total = self.n_layers * self.layer_bytes + \
                (self.n_layers - KEEP_LAST) * self.arena_bytes
            print(f"[solution] pinned host memory: {total / 2 ** 30:.2f} GiB in "
                  f"{time.perf_counter() - started:.1f} s (background)", flush=True)
        except Exception as exc:  # noqa: BLE001 - whatever is missing stays on the GPU
            print(f"[solution] pinned host memory: {exc!r}", flush=True)
        finally:
            self._done.set()

    def wait(self):
        self._done.wait()

    def view(self, layer, name):
        start, nbytes = self.offsets[name]
        return self.blocks[layer][start:start + nbytes]

    def is_placed(self, param, layer, name):
        block = self.blocks[layer]
        return block is not None and param.data.device.type == "cpu" and \
            param.data.data_ptr() == block.data_ptr() + self.offsets[name][0]

    def place(self, module, layer, name, param=None):
        """Move one projection's codes (`param`, else the module's weight) into its slot of the
        layer's block; False when the block is not there yet or the codes do not fit it."""
        param = module.weight if param is None else param
        with self._lock:
            block = self.blocks[layer]
        if block is None or param.data.dtype != torch.uint8 or not param.data.is_cuda \
                or param.data.numel() != self.offsets[name][1]:
            return False
        device = param.data.device
        host = self.view(layer, name)
        gpu = param.data
        host.copy_(gpu.view(-1), non_blocking=True)  # the GPU copy lives until this is done
        done = torch.cuda.Event()
        done.record()
        self._copies.append((done, gpu))
        while self._copies and self._copies[0][0].query():
            self._copies.pop(0)
        param.data = host.view(gpu.shape)
        if id(param) not in self.placed:
            module.register_forward_pre_hook(_codes_back_on_gpu)
        self.placed[id(param)] = (param, module, device, layer, name)
        return True

    def reclaim(self):
        """Codes something copied back to the GPU (a `.to(device)` of the model) go back to
        their host block, which still holds the same bytes; how many did."""
        moved = 0
        for param, module, device, layer, name in self.placed.values():
            if param.data.is_cuda:
                param.data = self.view(layer, name).view(param.data.shape)
                moved += 1
        return moved

    def settle(self):
        """The host blocks hold every placed projection's codes from here on."""
        self.wait()
        for done, _ in self._copies:
            done.synchronize()
        self._copies = []

    def restore(self, model):
        """Every placed projection back on its GPU, and any LoRA weights peft put on the host
        next to them (Soup's own path runs after all)."""
        self.settle()
        device = None
        for param, module, device, _, _ in self.placed.values():
            if param.data.device.type == "cpu":
                param.data = param.data.to(device)
        self.placed = {}
        if device is None:
            return
        for module in model.modules():
            for name in ("lora_A", "lora_B"):
                for sub in getattr(module, name, {}).values() if hasattr(module, name) else ():
                    if any(p.device.type == "cpu" for p in sub.parameters()):
                        sub.to(device)


def _codes_back_on_gpu(module, args):
    param = module.weight
    if param.data.device.type == "cpu":
        param.data = param.data.to(param.quant_state.absmax.device)


def stash_bytes_per_token(config, gate_up=True, lora_rank=64):
    """What one layer keeps for its backward per real token, with room: the two norms' fp16
    inputs, the dropout mask bits, the LoRA activations, q / k / v and the attention output with
    its log-sum-exp, and the fp16 gate|up output unless the MLP is recomputed."""
    hidden, heads = config.hidden_size, config.num_attention_heads
    kv = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", None) or hidden // heads
    per = 2 * hidden * 2 + 2 * (hidden // 8) + 2 * 2 * lora_rank + 2 * 4 \
        + 2 * heads * head_dim * 2 + 2 * 2 * kv * head_dim + 4 * heads
    if gate_up:
        per += 2 * 2 * config.intermediate_size
    return per * 9 // 8


def wants_host_weights(model, quantizer):
    """The models the fused path takes: a Qwen2ForCausalLM loading NF4 with nested scales in fp16
    on CUDA."""
    if type(model).__name__ != "Qwen2ForCausalLM" or not torch.cuda.is_available():
        return False
    qc = getattr(quantizer, "quantization_config", None)
    return (getattr(qc, "load_in_4bit", False) and getattr(qc, "bnb_4bit_quant_type", "") == "nf4"
            and getattr(qc, "bnb_4bit_use_double_quant", False)
            and getattr(qc, "bnb_4bit_compute_dtype", None) == torch.float16)


_early = {}  # the HostWeights `prepare` started for the task's model, until a load takes it


def prepare(config):
    """Start pinning the host memory for a model of `config` now (at apply(): it then overlaps
    the imports, which make no CUDA calls; pinning beside the weight loading would slow every
    copy of it, as cudaHostAlloc holds the driver's lock)."""
    if torch.cuda.is_available() and not _early:
        from .model import RECOMPUTE_MLP

        _early["host"] = HostWeights(config, stash_bytes_per_token(config,
                                                                    gate_up=not RECOMPUTE_MLP))


def _host_for(config):
    from .model import RECOMPUTE_MLP

    stash = stash_bytes_per_token(config, gate_up=not RECOMPUTE_MLP)
    host = _early.pop("host", None)
    if host is not None and host.layout == _layout(config, stash):
        return host
    return HostWeights(config, stash)


def _layout(config, stash_bytes_per_token):
    heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
    return (config.hidden_size, config.intermediate_size, heads, config.num_key_value_heads,
            head_dim, config.num_hidden_layers, stash_bytes_per_token)


def skip_allocator_warmup():
    """transformers reserves the whole model's size on the GPU in one block before it loads the
    weights (caching_allocator_warmup, 1.07 GB here), so later copies need no cudaMalloc. With the
    codes on the host only half of that is ever on the GPU, and the reserve would set the run's
    memory peak, so the model this path takes loads without it; the values are the same."""
    try:
        from transformers import modeling_utils
    except Exception:  # noqa: BLE001 - nothing to patch
        return
    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if original is None or getattr(original, "_soupk", False):
        return

    def caching_allocator_warmup(model, expanded_device_map, hf_quantizer, *args, **kwargs):
        if wants_host_weights(model, hf_quantizer):
            from .startup import cache_nf4_table

            _loading["cast_on_gpu"] = True  # this load's tensors convert on the GPU
            cache_nf4_table()  # bitsandbytes is imported by now
            return None
        return original(model, expanded_device_map, hf_quantizer, *args, **kwargs)

    caching_allocator_warmup._soupk = True
    caching_allocator_warmup.__wrapped__ = original
    modeling_utils.caching_allocator_warmup = caching_allocator_warmup


_loading = {"cast_on_gpu": False}
CAST_CHUNK = 32 << 20


def patch_materialize():
    """transformers loads each checkpoint tensor with `tensor.to(device=cuda, dtype=float16)`; for
    the bf16 checkpoint PyTorch converts on the CPU first (its rule for copies between CPU and
    GPU), 1.8 s of the load on Kaggle's CPU. For the model this path takes, each tensor goes to the
    GPU as stored and converts there, 0.8 s, in pieces of at most 32 MB so the 467 MB embedding
    needs no second copy beside it. Same rounding: all 1,543,714,304 values of the task's
    checkpoint come out bit for bit the same (our probe; only a NaN's payload bits could differ)."""
    try:
        from transformers import core_model_loading as loading
    except Exception:  # noqa: BLE001 - nothing to patch
        return
    original = getattr(loading, "_materialize_copy", None)
    if original is None or getattr(original, "_soupk", False):
        return

    def _materialize_copy(tensor, device=None, dtype=None):
        if not _loading["cast_on_gpu"] or device is None or dtype is None:
            return original(tensor, device, dtype)
        target = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        tensor = tensor[...]  # what loads it from the safetensors slice
        if target.type != "cuda" or tensor.device.type != "cpu" or tensor.dtype == dtype \
                or not tensor.is_floating_point() or not dtype.is_floating_point:
            return original(tensor, device, dtype)
        return cast_on_gpu(tensor, target, dtype)

    _materialize_copy._soupk = True
    _materialize_copy.__wrapped__ = original
    loading._materialize_copy = _materialize_copy


def cast_on_gpu(tensor, device, dtype):
    """`tensor.to(device=device, dtype=dtype)`, converted on the GPU, in pieces of rows."""
    row_bytes = tensor[0].numel() * tensor.element_size() if tensor.dim() else 0
    if tensor.dim() == 0 or tensor.numel() * tensor.element_size() <= CAST_CHUNK:
        return tensor.to(device).to(dtype)
    rows = max(1, CAST_CHUNK // row_bytes)
    out = torch.empty(tensor.shape, dtype=dtype, device=device)
    for start in range(0, tensor.shape[0], rows):
        out[start:start + rows].copy_(tensor[start:start + rows].to(device))
    return out


def layer_and_projection(name, n_layers):
    """(layer, projection) of a checkpoint name like model.layers.3.self_attn.k_proj.weight."""
    parts = name.split(".")
    if len(parts) != 6 or parts[0] != "model" or parts[1] != "layers" or parts[5] != "weight":
        return None
    if not parts[2].isdigit() or int(parts[2]) >= n_layers:
        return None
    projection = parts[3] + "." + parts[4]
    return (int(parts[2]), projection) if projection in PROJECTIONS else None


def patch_bnb_loading():
    """transformers quantizes each Linear weight of a 4-bit checkpoint on the GPU as it loads it
    (Bnb4bitQuantize.convert); right after, the codes of a model with HostWeights move to the
    host. q_proj and v_proj stay until peft has put their LoRA next to them."""
    try:
        from transformers.integrations import bitsandbytes as hf_bnb
    except Exception:  # noqa: BLE001 - nothing to patch
        return
    ops = getattr(hf_bnb, "Bnb4bitQuantize", None)
    original = getattr(ops, "convert", None)
    if original is None or getattr(original, "_soupk", False):
        return

    def convert(self, input_dict, full_layer_name=None, model=None, **kwargs):
        result = original(self, input_dict, full_layer_name=full_layer_name, model=model,
                          **kwargs)
        if model is None or not isinstance(model, torch.nn.Module):
            return result
        host = model.__dict__.get("_soupk_host")
        if host is None and "_soupk_host" not in model.__dict__:
            host = None
            if wants_host_weights(model, getattr(self, "hf_quantizer", None)):
                host = _host_for(model.config)
            model.__dict__["_soupk_host"] = host
        if host is not None:
            for name, param in result.items():
                where = layer_and_projection(name, host.n_layers)
                if where is not None and where[1] not in AFTER_PEFT:
                    # the loader sets `param` on the module right after this returns
                    module = model.get_submodule(name.rsplit(".", 1)[0])
                    host.place(module, *where, param=param)
        return result

    convert._soupk = True
    convert.__wrapped__ = original
    ops.convert = convert


class WeightStream:
    """Two GPU slots for the NF4 codes of the layer at work and the next one."""

    def __init__(self, host, device):
        self.host = host
        self.n = host.n_layers
        self.slots = [torch.empty(host.layer_bytes, dtype=torch.uint8, device=device)
                      for _ in range(2)]
        self.words32 = [slot.view(torch.int32) for slot in self.slots]
        self.resident = [None, None]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.free = [torch.cuda.Event(), torch.cuda.Event()]
        self.copy = torch.cuda.Stream(device)

    def _fetch(self, layer):
        slot = layer % 2
        if self.resident[slot] == layer:
            return
        self.copy.wait_event(self.free[slot])  # the slot's last reader is done (no-op at first)
        with torch.cuda.stream(self.copy):
            self.slots[slot].copy_(self.host.blocks[layer], non_blocking=True)
        self.ready[slot].record(self.copy)
        self.resident[slot] = layer

    def enter(self, layer, ahead):
        """`layer`'s codes ready for the compute stream; the copy of `ahead` under way."""
        self._fetch(layer)
        torch.cuda.current_stream().wait_event(self.ready[layer % 2])
        if 0 <= ahead < self.n:
            self._fetch(ahead)

    def leave(self, layer):
        self.free[layer % 2].record(torch.cuda.current_stream())

    def words(self, layer, start, count):
        return self.words32[layer % 2][start:start + count]


class Held:
    """One autograd function's saved tensors: on the GPU, or host views with their GPU copies to
    come back."""

    __slots__ = ("items", "gpu", "event")

    def __init__(self, items):
        self.items = items
        self.gpu = None
        self.event = None


class ActivationStash:
    def __init__(self, host, device):
        self.host = host
        self.n = host.n_layers
        self.device = device
        self.d2h = torch.cuda.Stream(device)
        self.h2d = torch.cuda.Stream(device)
        self.used = [0] * self.n
        self.held = [[] for _ in range(self.n)]
        self.inflight = []  # (event, GPU sources) of copies the compute stream has not waited on

    def begin_step(self):
        """A new forward rewrites the arenas: the last backward's copies out of them go first."""
        self.used = [0] * self.n
        self.held = [[] for _ in range(self.n)]
        self.d2h.wait_stream(self.h2d)

    def _retire(self, keep):
        compute = torch.cuda.current_stream()
        while len(self.inflight) > keep:
            event, _ = self.inflight.pop(0)
            compute.wait_event(event)  # the sources are freed after this, in compute order

    def push(self, layer, tensors):
        arena = self.host.arenas[layer] if layer < self.n - KEEP_LAST else None
        if arena is None:
            return Held(list(tensors))
        items, sources = [], []
        compute = torch.cuda.current_stream()
        self.d2h.wait_stream(compute)
        with torch.cuda.stream(self.d2h):
            for t in tensors:
                nbytes = t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0
                at = self.used[layer]
                if nbytes < SMALL or not t.is_cuda or not t.is_contiguous() \
                        or at + nbytes > arena.numel():
                    items.append(t)
                    continue
                self.used[layer] = at + _aligned(nbytes)
                view = arena[at:at + nbytes].view(t.dtype).view(t.shape)
                view.copy_(t, non_blocking=True)
                items.append(view)
                sources.append(t)
        event = torch.cuda.Event()
        event.record(self.d2h)
        self.inflight.append((event, sources))
        self._retire(keep=3)  # a whole layer (three pushes) in flight
        held = Held(items)
        self.held[layer].append(held)
        return held

    def flush(self):
        self._retire(keep=0)

    def prefetch(self, layer):
        """Start copying `layer`'s saved tensors back to the GPU."""
        if not 0 <= layer < self.n:
            return
        for held in self.held[layer]:
            self._pull(held)

    def _pull(self, held):
        if held.gpu is not None or all(not isinstance(t, torch.Tensor) or t.is_cuda
                                       for t in held.items):
            return
        compute = torch.cuda.current_stream()
        gpu = [torch.empty(t.shape, dtype=t.dtype, device=self.device)
               if isinstance(t, torch.Tensor) and not t.is_cuda else t for t in held.items]
        self.h2d.wait_stream(compute)  # allocated in compute order: earlier users are done
        with torch.cuda.stream(self.h2d):
            for t, g in zip(held.items, gpu):
                if g is not t:
                    g.copy_(t, non_blocking=True)
        held.event = torch.cuda.Event()
        held.event.record(self.h2d)
        held.gpu = gpu

    def get(self, held):
        if held.gpu is None:
            self._pull(held)
        if held.gpu is None:  # nothing went to the host
            items, held.items = held.items, None
            return items
        torch.cuda.current_stream().wait_event(held.event)
        items, held.gpu, held.items = held.gpu, None, None
        return items
