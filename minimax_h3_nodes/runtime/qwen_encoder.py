"""Qwen3-VL layer-50 text encoder used by MiniMax-H3.

H3 consumes the unnormalised output immediately after language layer 49.  It
does not consume the final language-model norm or LM head.  The configuration
is therefore trimmed *before* ``from_pretrained`` so layers 50+ are never
materialised.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import logging

from .components import H3ComponentError, model_root_path, resolve_component
from .h3_settings import (
    ALLOW_PARTIAL_OFFLOAD_INT8,
    INT8_TE_DIRNAME,
    QUANT_KEY_SUFFIXES,
    TE_GPU_HEADROOM,
    TE_VISUAL_ON_CPU,
    TEXT_ENCODER_SELECTED_LAYERS,
)

LOGGER = logging.getLogger(__name__)

SELECTED_LAYERS = TEXT_ENCODER_SELECTED_LAYERS
HIDDEN_SIZE = 5120

_LATER_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_TEXT_CONFIG_CONTRACT = {
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
_LINEAR_LEAVES = frozenset(
    {"weight", "bias", "weight_scale", "weight_scale_2", "comfy_quant", "input_scale"}
)
_STRIP_PREFIXES = ("model.",)  # checkpoint 相对 causal_lm.model 的键


def _torch_dtype(value: str):
    import torch

    normalized = str(value).lower().replace("torch.", "")
    values = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return values[normalized]
    except KeyError as exc:
        raise H3ComponentError(f"Unsupported text-encoder dtype: {value!r}") from exc


def _resolve_device(value: str):
    import torch

    if value != "auto":
        return torch.device(value)
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except ImportError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _qwen_causal_lm_class():
    """Return the class matching the released checkpoint key namespace.

    The H3 snapshot advertises ``Qwen3VLForConditionalGeneration`` and stores
    weights under ``model.visual.*`` / ``model.language_model.*`` plus
    ``lm_head.weight``.  Loading that snapshot directly into the bare
    ``Qwen3VLModel`` is version-dependent because the bare class uses a
    different base-model prefix.  Load the declared wrapper, then retain its
    ``.model`` backbone.
    """

    try:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    except ImportError:
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration,
            )

            return Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise H3ComponentError(
                "Transformers with Qwen3-VL support is required. "
                "Install this node package's requirements.txt."
            ) from exc


def _validate_qwen_config(config: Any, component: Path) -> Any:
    """Validate the concrete Qwen3-VL-32B config shipped with H3."""

    if getattr(config, "model_type", None) != "qwen3_vl":
        raise H3ComponentError(
            f"{component}/config.json must have model_type='qwen3_vl', got "
            f"{getattr(config, 'model_type', None)!r}"
        )
    architectures = getattr(config, "architectures", None)
    if not isinstance(architectures, (list, tuple)) or not any(
        str(name) == "Qwen3VLForConditionalGeneration" for name in architectures
    ):
        raise H3ComponentError(
            f"{component}/config.json must advertise "
            "Qwen3VLForConditionalGeneration; got "
            f"architectures={architectures!r}"
        )
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        raise H3ComponentError(
            f"{component}/config.json is not a Qwen3-VL multimodal configuration"
        )
    mismatches: list[str] = []
    for name, expected in _TEXT_CONFIG_CONTRACT.items():
        actual = getattr(text_config, name, None)
        try:
            actual_int = int(actual)
        except (TypeError, ValueError):
            actual_int = None
        if actual_int != expected:
            mismatches.append(f"{name}={actual!r} (expected {expected})")
    if mismatches:
        raise H3ComponentError(
            "H3 requires its Qwen3-VL-32B text architecture; "
            + ", ".join(mismatches)
        )
    available_layers = int(getattr(text_config, "num_hidden_layers", -1))
    if available_layers < SELECTED_LAYERS:
        raise H3ComponentError(
            f"H3 needs at least {SELECTED_LAYERS} Qwen layers, got "
            f"{available_layers}"
        )
    return text_config


def _validate_loading_info(loading_info: Any) -> None:
    """Reject accidental partial loads while allowing intentionally cut layers."""

    if not isinstance(loading_info, dict):
        return
    missing = [
        str(key)
        for key in (loading_info.get("missing_keys") or ())
        if "rotary_emb.inv_freq" not in str(key)
    ]
    mismatched = list(loading_info.get("mismatched_keys") or ())
    unexpected: list[str] = []
    for key in loading_info.get("unexpected_keys") or ():
        match = _LATER_LAYER_KEY.match(str(key))
        if match and int(match.group(1)) >= SELECTED_LAYERS:
            continue
        # Some Transformers versions report deterministic, non-persistent
        # rotary buffers when reading older snapshots.
        if "rotary_emb.inv_freq" in str(key):
            continue
        unexpected.append(str(key))
    if missing or mismatched or unexpected:
        raise H3ComponentError(
            "Qwen3-VL checkpoint did not load cleanly after the intentional "
            f"layer-{SELECTED_LAYERS} cut: missing={missing[:12]!r}, "
            f"mismatched={mismatched[:12]!r}, unexpected={unexpected[:12]!r}"
        )


def _physical_module_bytes(module: Any) -> int:
    """Return serialized storage bytes, not a tensor subclass's logical size.

    Comfy ``QuantizedTensor`` reports the original BF16 shape/dtype through the
    Tensor API.  ``numel() * element_size()`` therefore over-counts each INT8
    Linear by roughly 2x.  Its module state dict contains the real qdata,
    scales, marker and bias, which is the storage that ModelPatcher moves.
    """

    return sum(int(tensor.nbytes) for tensor in module.state_dict().values())


def _direct_tensor_slots(roots: list[Any], excluded_module_ids: set[int]):
    """Yield direct parameter/buffer slots outside streamable Linear modules."""

    visited: set[int] = set()
    for root in roots:
        for module in root.modules():
            module_id = id(module)
            if module_id in visited or module_id in excluded_module_ids:
                continue
            visited.add(module_id)
            for collection_name in ("_parameters", "_buffers"):
                collection = getattr(module, collection_name, {})
                for name, tensor in tuple(collection.items()):
                    if tensor is not None:
                        yield module, collection_name, name, tensor


def _direct_tensor_bytes(roots: list[Any], excluded_module_ids: set[int]) -> int:
    """Count unique non-Linear tensors used by the text-only forward path."""

    seen: set[int] = set()
    total = 0
    for _module, _collection, _name, tensor in _direct_tensor_slots(
        roots, excluded_module_ids
    ):
        tensor_id = id(tensor)
        if tensor_id not in seen:
            total += int(tensor.nbytes)
            seen.add(tensor_id)
    return total


def _move_direct_tensors(
    roots: list[Any], excluded_module_ids: set[int], device: Any
) -> None:
    """Move only direct static tensors, preserving shared-parameter aliases.

    Calling ``language_model.to(device)`` would recursively move the 350
    quantized Linear modules and bypass ModelPatcher's low-VRAM bookkeeping.
    """

    import torch

    target_device = torch.device(device)
    moved: dict[int, Any] = {}
    for module, collection_name, name, tensor in _direct_tensor_slots(
        roots, excluded_module_ids
    ):
        tensor_id = id(tensor)
        replacement = moved.get(tensor_id)
        if replacement is None:
            if tensor.device == target_device:
                replacement = tensor
            else:
                value = tensor.to(device=target_device)
                if collection_name == "_parameters":
                    replacement = torch.nn.Parameter(
                        value, requires_grad=bool(tensor.requires_grad)
                    )
                else:
                    replacement = value
            moved[tensor_id] = replacement
        getattr(module, collection_name)[name] = replacement


def _quantized_language_linears(roots: list[Any]) -> list[Any]:
    """Collect each complete Comfy quantized Linear exactly once."""

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        return []

    linears: list[Any] = []
    seen: set[int] = set()
    for root in roots:
        for module in root.modules():
            module_id = id(module)
            if module_id in seen:
                continue
            weight = getattr(module, "weight", None)
            if hasattr(module, "comfy_cast_weights") and isinstance(
                weight, QuantizedTensor
            ):
                seen.add(module_id)
                linears.append(module)
    return linears


def _make_quantized_linear_patcher(
    linears: list[Any], *, load_device: Any, offload_device: Any
):
    """Create a no-copy ModelPatcher view over the Qwen language Linears."""

    import torch
    from comfy.model_patcher import ModelPatcher

    class _H3TextLinearBank(torch.nn.Module):
        def __init__(self, modules: list[Any]) -> None:
            super().__init__()
            # PyTorch permits shared module registration.  These are the exact
            # objects referenced by Qwen's decoder layers; no weights are copied.
            self.linears = torch.nn.ModuleList(modules)

        def get_dtype(self):
            for linear in self.linears:
                weight = getattr(linear, "weight", None)
                if weight is not None:
                    return weight.dtype
            return None

    bank = _H3TextLinearBank(linears)
    storage_bytes = _physical_module_bytes(bank)
    patcher = ModelPatcher(
        bank,
        load_device=load_device,
        offload_device=offload_device,
        size=storage_bytes,
    )
    return bank, patcher, storage_bytes


class MiniMaxH3TextEncoder:
    """Resident/offload-aware wrapper around the trimmed Qwen3-VL backbone."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        component_path: Path,
        load_device: Any,
        offload_device: Any,
        quantized: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.component_path = component_path
        self.load_device = load_device
        self.offload_device = offload_device
        self.quantized = bool(quantized)
        self._lock = threading.RLock()
        self._compute_device = None
        self._inference_active = False
        self._linear_bank = None
        self._linear_patcher = None
        self._streaming_linears: tuple[Any, ...] = ()
        self._linear_storage_bytes = 0
        self._static_storage_bytes = 0

    def _movable(self):  # 文本编码实际参与的直接子模块（可选跳过 visual 塔）
        return [m for n, m in self.model.named_children() if not (TE_VISUAL_ON_CPU and n == "visual")]

    def _actual_device(self):
        for m in self._movable():
            for p in m.parameters():
                return p.device
            for b in m.buffers():
                return b.device
        return next(self.model.parameters()).device

    @property
    def device(self):
        # Partial load intentionally leaves many Linear weights on CPU.  The
        # execution device must follow activations, never the first parameter.
        return self._compute_device or self._actual_device()

    def _set_compute_device(self, device: Any | None) -> None:
        import torch

        self._compute_device = None if device is None else torch.device(device)
        if self._compute_device is None:
            if hasattr(self.model, "_h3_compute_device"):
                delattr(self.model, "_h3_compute_device")
        else:
            object.__setattr__(
                self.model, "_h3_compute_device", self._compute_device
            )

    def _ensure_linear_patcher(self):
        if self._linear_patcher is not None:
            return self._linear_patcher
        roots = self._movable()
        linears = _quantized_language_linears(roots)
        expected = SELECTED_LAYERS * 7
        if len(linears) != expected:
            raise H3ComponentError(
                "H3 INT8 text streaming requires exactly "
                f"{expected} complete quantized Linear modules, got {len(linears)}"
            )
        bank, patcher, linear_bytes = _make_quantized_linear_patcher(
            linears,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )
        self._linear_bank = bank
        self._linear_patcher = patcher
        self._streaming_linears = tuple(linears)
        excluded = {id(module) for module in linears}
        self._linear_storage_bytes = linear_bytes
        self._static_storage_bytes = _direct_tensor_bytes(roots, excluded)
        LOGGER.info(
            "H3 text_encoder physical storage %.2fGB "
            "(INT8 Linear %.2fGB + static %.2fGB); visual stays on CPU",
            (linear_bytes + self._static_storage_bytes) / 2**30,
            linear_bytes / 2**30,
            self._static_storage_bytes / 2**30,
        )
        return patcher

    def _move_static_tensors(self, device: Any) -> None:
        excluded = {id(module) for module in self._streaming_linears}
        _move_direct_tensors(self._movable(), excluded, device)

    def _unload_linear_patcher(self) -> None:
        patcher = self._linear_patcher
        if patcher is None:
            return
        try:
            import comfy.model_management as mm

            unload = getattr(mm, "unload_model_and_clones", None)
            if callable(unload):
                unload(patcher)
            else:
                patcher.detach()
        finally:
            # If load_models_gpu failed before registering the LoadedModel, the
            # global unload helper cannot see it.  Detach any remaining load.
            loaded_size = getattr(patcher, "loaded_size", lambda: 0)()
            bank = getattr(patcher, "model", None)
            if (
                loaded_size
                or bool(getattr(bank, "model_lowvram", False))
                or bool(getattr(patcher, "pinned", ()))
            ):
                patcher.detach()

    def load_for_inference(self) -> "MiniMaxH3TextEncoder":
        import torch

        if self.load_device.type != "cuda":
            return self
        if self._inference_active and self.device == self.load_device:
            return self

        if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
            patcher = self._ensure_linear_patcher()
            reserve = TE_GPU_HEADROOM + self._static_storage_bytes
            try:
                import comfy.model_management as mm

                # Reserve static embedding/norm/RoPE bytes plus encode
                # workspace.  ModelPatcher uses the remainder for resident
                # INT8 Linears and streams all others on Comfy's CUDA streams.
                mm.load_models_gpu([patcher], memory_required=reserve)
                self._move_static_tensors(self.load_device)
            except torch.OutOfMemoryError:
                LOGGER.warning(
                    "text_encoder INT8 partial load OOM; cleaned up and "
                    "falling back to CPU encode"
                )
                self._unload_linear_patcher()
                self._move_static_tensors(self.offload_device)
                self._set_compute_device(None)
                self._inference_active = False
                try:
                    mm.soft_empty_cache()
                except (ImportError, UnboundLocalError):
                    torch.cuda.empty_cache()
                return self
            self._set_compute_device(self.load_device)
            self._inference_active = True
            loaded = int(getattr(patcher, "loaded_size", lambda: 0)())
            LOGGER.info(
                "H3 text_encoder CUDA streaming ready: static %.2fGB, "
                "Linear resident %.2f/%.2fGB, workspace reserve %.2fGB",
                self._static_storage_bytes / 2**30,
                loaded / 2**30,
                self._linear_storage_bytes / 2**30,
                TE_GPU_HEADROOM / 2**30,
            )
            return self

        if self._actual_device() == self.load_device:
            self._set_compute_device(self.load_device)
            self._inference_active = True
            return self
        mods = self._movable()
        need = sum(_physical_module_bytes(module) for module in mods)
        free = torch.cuda.mem_get_info(self.load_device)[0] if torch.cuda.is_available() else 0
        if free < need + TE_GPU_HEADROOM:
            LOGGER.warning("text_encoder 需 %.1fGB + %.1fGB 工作区 > 空闲 %.1fGB，回退 CPU encode",
                           need / 2**30, TE_GPU_HEADROOM / 2**30, free / 2**30)
            return self
        try:
            for m in mods:
                m.to(self.load_device)
        except torch.OutOfMemoryError:
            LOGGER.warning("text_encoder 上卡 OOM，回滚 CPU encode")
            for m in mods:
                m.to(self.offload_device)
            torch.cuda.empty_cache()
            self._set_compute_device(None)
            self._inference_active = False
            return self
        self._set_compute_device(self.load_device)
        self._inference_active = True
        return self

    def offload_after_inference(self) -> None:
        import torch

        with self._lock:
            if self._linear_patcher is not None:
                self._unload_linear_patcher()
                self._move_static_tensors(self.offload_device)
            elif self._actual_device() != self.offload_device:
                for module in self._movable():
                    module.to(self.offload_device)
            self._set_compute_device(None)
            self._inference_active = False
            try:
                import comfy.model_management as mm

                mm.soft_empty_cache()
            except ImportError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def encode_prompt(self, prompt: str):
        """Return H3's positive layer-50 features as CPU ``[L, 5120]`` BF16."""

        import torch

        if not isinstance(prompt, str) or not prompt:
            raise H3ComponentError("MiniMax-H3 prompt must be a non-empty string")
        with self._lock, torch.inference_mode():
            self.load_for_inference()
            # Match the native H3 encoder recipe.  This is independent of the
            # DiT attention backend and materially reduces Qwen's long-sequence
            # attention workspace on supported CUDA builds.
            # 临时打开 cudnn SDP，退出前必须恢复，避免污染全局 torch.backends
            restore_cudnn_sdp = None
            if self.device.type == "cuda":
                enable_cudnn_sdp = getattr(
                    torch.backends.cuda, "enable_cudnn_sdp", None
                )
                cudnn_sdp_enabled = getattr(
                    torch.backends.cuda, "cudnn_sdp_enabled", None
                )
                if callable(enable_cudnn_sdp) and callable(cudnn_sdp_enabled):
                    previous = bool(cudnn_sdp_enabled())
                    enable_cudnn_sdp(True)

                    def restore_cudnn_sdp() -> None:
                        enable_cudnn_sdp(previous)

                elif callable(enable_cudnn_sdp):
                    enable_cudnn_sdp(True)
            try:
                encoded = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device, dtype=torch.long)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, dtype=torch.long)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = outputs.last_hidden_state[0].to(
                    dtype=torch.bfloat16, device="cpu"
                )
                expected = (int(input_ids.shape[1]), HIDDEN_SIZE)
                if tuple(hidden.shape) != expected:
                    raise H3ComponentError(
                        f"Unexpected Qwen3-VL feature shape {tuple(hidden.shape)}; "
                        f"expected {expected}"
                    )
                return hidden.contiguous()
            finally:
                if restore_cudnn_sdp is not None:
                    restore_cudnn_sdp()

    def encode_conditioning(self, prompt: str) -> dict[str, Any]:
        import torch

        hidden = self.encode_prompt(prompt)
        return {
            "prompt": prompt,
            "hidden_states": hidden,
            "text_len": int(hidden.shape[0]),
            "text_token_tags": torch.ones(int(hidden.shape[0]), dtype=torch.long),
            "cfg_distilled": True,
        }


def _component_is_quantized(component: Path) -> bool:
    idx = component / "model.safetensors.index.json"
    if idx.is_file():
        import json
        wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {}
        return any(str(k).endswith(QUANT_KEY_SUFFIXES) for k in wm)
    from safetensors import safe_open
    files = sorted(component.glob("*.safetensors"))
    if not files:
        return False
    with safe_open(str(files[0]), framework="pt") as reader:
        return any(k.endswith(QUANT_KEY_SUFFIXES) for k in reader.keys())


def _swap_lang_linears(layers, LinearCls, *, dtype) -> int:
    """把 language_model.layers 内 nn.Linear 换成 comfy ops.Linear（meta）。"""
    import torch.nn as nn

    n = 0
    for layer in layers:
        for name, mod in list(layer.named_modules()):
            if not isinstance(mod, nn.Linear) or type(mod) is not nn.Linear:
                continue
            parent = layer
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf = parts[-1]
            new = LinearCls(
                mod.in_features, mod.out_features,
                bias=mod.bias is not None, device="meta", dtype=dtype,
            )
            setattr(parent, leaf, new)
            n += 1
    return n


def _assign_param(module, name: str, value) -> None:
    import torch
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() and hasattr(parent, "__getitem__") else getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        prev = parent._parameters[leaf]
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=bool(prev.requires_grad) if prev is not None else False)
        return
    if leaf in parent._buffers:
        parent._buffers[leaf] = value
        return
    raise H3ComponentError(f"not a parameter/buffer: {name}")


def _flush_linear_bag(module, prefix: str, bag: dict, device) -> bool:
    """Use the same strict meta-to-QuantizedTensor path as the DiT loader."""

    from .model_loader import _flush_linear

    return _flush_linear(module, prefix, bag, device)


def _stream_load_quantized_backbone(model, component: Path, *, offload_device) -> None:
    """流式写入 int8_convrot + 透传 bf16（visual/embed/norm）。"""
    import json
    from safetensors import safe_open
    from .model_loader import _checkpoint_index  # 复用分片索引

    # _checkpoint_index expects component dir; may raise — fallback local
    try:
        shards, weight_map = _checkpoint_index(component)
    except Exception:
        shards, weight_map = [], None
        idx = component / "model.safetensors.index.json"
        if idx.is_file():
            wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {}
            shards = sorted({component / str(v) for v in wm.values()})
            weight_map = {str(k): str(v) for k, v in wm.items()}
        else:
            shards = sorted(component.glob("*.safetensors"))
            weight_map = None
            from safetensors import safe_open as _so
            weight_map = {}
            for sp in shards:
                with _so(str(sp), framework="pt") as r:
                    for k in r.keys():
                        weight_map[k] = sp.name

    linears = {
        f"{name}.": mod for name, mod in model.named_modules()
        if name and hasattr(mod, "in_features") and hasattr(mod, "out_features")
    }
    expected = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
    needed: dict[str, set[str]] = defaultdict(set)
    for ck in weight_map:
        local = ck
        for pref in _STRIP_PREFIXES:
            if local.startswith(pref):
                local = local[len(pref):]
                break
        for lp, mod in linears.items():
            if local.startswith(lp) and local[len(lp):] in _LINEAR_LEAVES:
                needed[lp].add(local[len(lp):])
                break

    pending: dict[str, dict] = defaultdict(dict)
    loaded: set[str] = set()
    expected_quantized = sum(
        "comfy_quant" in leaves for leaves in needed.values()
    )
    loaded_quantized = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for ck in reader.keys():
                local = ck
                for pref in _STRIP_PREFIXES:
                    if local.startswith(pref):
                        local = local[len(pref):]
                        break
                matched = None
                for lp in linears:
                    if local.startswith(lp) and local[len(lp):] in _LINEAR_LEAVES:
                        matched = (lp, local[len(lp):])
                        break
                tensor = reader.get_tensor(ck)
                if matched:
                    lp, leaf = matched
                    pending[lp][leaf] = tensor
                    if needed[lp] <= set(pending[lp]):
                        loaded_quantized += int(
                            _flush_linear_bag(
                                linears[lp], lp, pending.pop(lp), offload_device
                            )
                        )
                        loaded.update(f"{lp}{x}" for x in needed[lp])
                    continue
                if local not in expected:
                    continue  # lm_head / layer>=50 等已裁剪键
                target = expected[local]
                if (
                    tensor.is_floating_point()
                    and getattr(target, "is_floating_point", lambda: False)()
                    and tensor.dtype != target.dtype
                ):
                    tensor = tensor.to(dtype=target.dtype)
                _assign_param(model, local, tensor.to(device=offload_device))
                loaded.add(local)
    for lp, bag in list(pending.items()):
        loaded_quantized += int(
            _flush_linear_bag(linears[lp], lp, bag, offload_device)
        )
        loaded.update(f"{lp}{x}" for x in bag)
    if expected_quantized <= 0 or loaded_quantized != expected_quantized:
        raise H3ComponentError(
            "H3 text_encoder quantized Linear contract failed: "
            f"materialized={loaded_quantized}, expected={expected_quantized}"
        )
    LOGGER.info(
        "H3 text_encoder materialized %d complete INT8/convrot QuantizedTensor layers",
        loaded_quantized,
    )
    meta_left = [
        n for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if getattr(t, "device", None) is not None and t.device.type == "meta"
    ]
    if meta_left:
        raise H3ComponentError(f"int8 text_encoder 仍有 meta 张量: {meta_left[:12]!r}")


def _load_quantized_text_encoder(
    component: Path,
    *,
    config,
    model_dtype,
    offload_device,
    attention_backend: str,
):
    """meta 构建 → language Linear 换 comfy ops → 流式装 int8_convrot。"""
    import torch
    from .model_loader import _require_int8_ops

    model_cls = _qwen_causal_lm_class()
    ops = _require_int8_ops(model_dtype)
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise H3ComponentError("int8 text_encoder 需要 accelerate（init_empty_weights）") from exc

    with init_empty_weights():
        try:
            causal_lm = model_cls._from_config(  # type: ignore[attr-defined]
                config, attn_implementation=attention_backend
            )
        except Exception:
            causal_lm = model_cls(config)
    model = getattr(causal_lm, "model", None)
    if model is None:
        raise H3ComponentError("Qwen3VL backbone missing .model")
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None) if language_model is not None else None
    if layers is None or len(layers) != SELECTED_LAYERS:
        raise H3ComponentError(
            f"int8 path expects {SELECTED_LAYERS} layers, got "
            f"{None if layers is None else len(layers)}"
        )
    n_swap = _swap_lang_linears(layers, ops.Linear, dtype=model_dtype)
    if n_swap <= 0:
        raise H3ComponentError("未能替换任何 language Linear 为 comfy ops")
    _stream_load_quantized_backbone(model, component, offload_device=offload_device)
    language_model.norm = torch.nn.Identity()
    if hasattr(language_model, "config"):
        language_model.config.num_hidden_layers = SELECTED_LAYERS
        language_model.config.output_hidden_states = False
        language_model.config.use_cache = False
    del causal_lm
    model.requires_grad_(False).eval()
    return model


def load_h3_text_encoder(
    model_root: str,
    *,
    text_encoder_path: str | None = None,
    tokenizer_path: str | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    offload_device: str = "cpu",
    attention_backend: str = "sdpa",
) -> MiniMaxH3TextEncoder:
    """Load the local Qwen3-VL component with no Hub or remote-code fallback."""

    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not text_encoder_path:  # 未显式指定时自动优先 int8 量化目录（26GB vs BF16 62GB）
        auto = model_root_path(model_root) / INT8_TE_DIRNAME
        if auto.is_dir() and (auto / "config.json").is_file():
            text_encoder_path = str(auto)
            LOGGER.info("text_encoder_path 未指定，自动选用量化目录 %s", INT8_TE_DIRNAME)
    component = resolve_component(
        model_root,
        ("text_encoder", "qwen3vl", "qwen"),
        explicit=text_encoder_path,
        required_files=("config.json",),
    )
    tokenizer_component = resolve_component(
        model_root,
        ("tokenizer", "processor", "text_encoder"),
        explicit=tokenizer_path,
    )
    model_dtype = _torch_dtype(dtype)
    load_device = _resolve_device(device)
    target_offload_device = torch.device(offload_device)

    config = AutoConfig.from_pretrained(
        str(component),
        local_files_only=True,
        trust_remote_code=False,
    )
    text_config = _validate_qwen_config(config, component)
    text_config.num_hidden_layers = SELECTED_LAYERS
    text_config.output_hidden_states = False
    text_config.use_cache = False
    config.output_hidden_states = False
    config.use_cache = False

    quantized = _component_is_quantized(component)
    if quantized:
        if not ALLOW_PARTIAL_OFFLOAD_INT8:
            raise H3ComponentError("int8 text_encoder 需要 ALLOW_PARTIAL_OFFLOAD_INT8")
        model = _load_quantized_text_encoder(
            component, config=config, model_dtype=model_dtype,
            offload_device=target_offload_device, attention_backend=attention_backend,
        )
    else:
        model_cls = _qwen_causal_lm_class()
        load_kwargs = {
            "config": config,
            "local_files_only": True,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "attn_implementation": attention_backend,
            "output_loading_info": True,
        }
        # Construct on CPU.  The native H3 lifecycle never lets this ~64 GB
        # encoder become GPU-resident during component loading; encode_prompt()
        # moves it to the selected Comfy device only for the actual encode, then
        # immediately offloads it again.
        try:
            loaded = model_cls.from_pretrained(
                str(component),
                dtype=model_dtype,
                **load_kwargs,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise
            loaded = model_cls.from_pretrained(
                str(component),
                torch_dtype=model_dtype,
                **load_kwargs,
            )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise H3ComponentError(
                "Transformers did not return (model, loading_info) while loading "
                "the Qwen3-VL checkpoint"
            )
        causal_lm, loading_info = loaded
        _validate_loading_info(loading_info)
        model = getattr(causal_lm, "model", None)
        if model is None:
            raise H3ComponentError(
                "Qwen3VLForConditionalGeneration checkpoint has no .model backbone"
            )
        language_model = getattr(model, "language_model", None)
        if language_model is None or not hasattr(language_model, "norm"):
            raise H3ComponentError("Loaded Qwen3-VL model has no language_model.norm")
        layers = getattr(language_model, "layers", None)
        if layers is None or len(layers) != SELECTED_LAYERS:
            raise H3ComponentError(
                "Qwen3-VL backbone was not trimmed to exactly "
                f"{SELECTED_LAYERS} layers; got "
                f"{None if layers is None else len(layers)}"
            )
        language_model.norm = torch.nn.Identity()
        if hasattr(language_model, "config"):
            language_model.config.num_hidden_layers = SELECTED_LAYERS
            language_model.config.output_hidden_states = False
            language_model.config.use_cache = False
        del causal_lm
        model.requires_grad_(False).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_component),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    return MiniMaxH3TextEncoder(
        model=model,
        tokenizer=tokenizer,
        component_path=component,
        load_device=load_device,
        offload_device=target_offload_device,
        quantized=quantized,
    )


__all__ = [
    "HIDDEN_SIZE",
    "SELECTED_LAYERS",
    "MiniMaxH3TextEncoder",
    "load_h3_text_encoder",
]
