"""Qwen3-VL layer-50 text encoder used by MiniMax-H3.

H3 consumes the unnormalised output immediately after language layer 49.  It
does not consume the final language-model norm or LM head.  The configuration
is therefore trimmed *before* ``from_pretrained`` so layers 50+ are never
materialised.
"""

from __future__ import annotations

import re
import sys
import threading
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import logging

from .backend_state import TORCH_BACKEND_STATE_LOCK
from .components import (
    H3ComponentError,
    model_root_path,
    read_json,
    resolve_component,
)
from .h3_settings import (
    ALLOW_PARTIAL_OFFLOAD_INT8,
    INT8_FORMAT,
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
_LATER_LAYER_LOCAL_KEY = re.compile(r"^language_model\.layers\.(\d+)\.")
_TEXT_CONFIG_CONTRACT = {
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
}
_VISION_CONFIG_CONTRACT = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "out_hidden_size": 5120,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "in_channels": 3,
}
_VISION_DEEPSTACK_INDEXES = (8, 16, 24)
_LINEAR_LEAVES = frozenset(
    {"weight", "bias", "weight_scale", "weight_scale_2", "comfy_quant", "input_scale"}
)
_STRIP_PREFIXES = ("model.",)  # checkpoint 相对 causal_lm.model 的键


@contextmanager
def _scoped_cudnn_sdp(active: bool):
    """Temporarily enable cuDNN SDP without leaking process-global state."""

    if not active:
        yield
        return

    import torch

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    enable = getattr(cuda_backend, "enable_cudnn_sdp", None)
    enabled = getattr(cuda_backend, "cudnn_sdp_enabled", None)
    if not (callable(enable) and callable(enabled)):
        yield
        return

    with TORCH_BACKEND_STATE_LOCK:
        previous = bool(enabled())
        try:
            enable(True)
            yield
        finally:
            enable(previous)


def minimax_h3_mm_token_type_ids(
    input_ids: Any,
    *,
    image_token_id: int,
    video_token_id: int,
):
    """Build Qwen3-VL M-RoPE types: text=0, image=1, video=2."""

    import torch

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim not in (1, 2):
        raise H3ComponentError(
            "Qwen input_ids must be a 1-D or 2-D tensor for mm token types"
        )
    types = torch.zeros_like(input_ids, dtype=torch.int32)
    types[input_ids == int(image_token_id)] = 1
    types[input_ids == int(video_token_id)] = 2
    return types


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
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        raise H3ComponentError(
            f"{component}/config.json has no Qwen3-VL vision_config"
        )
    vision_mismatches: list[str] = []
    for name, expected in _VISION_CONFIG_CONTRACT.items():
        actual = getattr(vision_config, name, None)
        try:
            actual_int = int(actual)
        except (TypeError, ValueError):
            actual_int = None
        if actual_int != expected:
            vision_mismatches.append(
                f"{name}={actual!r} (expected {expected})"
            )
    raw_deepstack = getattr(vision_config, "deepstack_visual_indexes", None)
    try:
        deepstack = tuple(int(value) for value in raw_deepstack)
    except (TypeError, ValueError):
        deepstack = ()
    if deepstack != _VISION_DEEPSTACK_INDEXES:
        vision_mismatches.append(
            "deepstack_visual_indexes="
            f"{raw_deepstack!r} (expected {list(_VISION_DEEPSTACK_INDEXES)!r})"
        )
    if vision_mismatches:
        raise H3ComponentError(
            "H3 requires the released Qwen3-VL-32B vision architecture; "
            + ", ".join(vision_mismatches)
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
        processor: Any | None = None,
        component_path: Path,
        load_device: Any,
        offload_device: Any,
        quantized: bool = False,
        tokenizer_component_path: Path | None = None,
        processor_component_path: Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.component_path = component_path
        self.load_device = load_device
        self.offload_device = offload_device
        self.quantized = bool(quantized)
        self.tokenizer_component_path = (
            tokenizer_component_path.resolve()
            if tokenizer_component_path is not None
            else component_path.resolve()
        )
        self.processor_component_path = (
            processor_component_path.resolve()
            if processor_component_path is not None
            else None
        )
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

    def _visual_module(self):
        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise H3ComponentError(
                "Loaded Qwen3-VL backbone has no visual tower; multimodal "
                "FL2VA/Ref2VA encoding is unavailable"
            )
        return visual

    def _offload_visual_after_staged(
        self, visual: Any, *, preserve_primary_error: bool
    ) -> bool:
        """Return the visual tower to CPU without masking a primary failure."""

        try:
            visual.to(self.offload_device)
        except BaseException:
            if not preserve_primary_error:
                raise
            LOGGER.exception(
                "Failed to offload Qwen visual tower while preserving the "
                "original staged-encode error"
            )
            return False
        return True

    def _require_processor(self, *, video: bool = False):
        processor = self.processor
        if processor is None or not callable(
            getattr(processor, "image_processor", None)
        ):
            # HF image processors are callable objects rather than functions.
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is None or not callable(image_processor):
                raise H3ComponentError(
                    "MiniMax-H3 multimodal Qwen encoding requires the local "
                    "AutoProcessor component with an image_processor"
                )
        if video:
            video_processor = getattr(processor, "video_processor", None)
            if video_processor is None or not callable(video_processor):
                raise H3ComponentError(
                    "MiniMax-H3 Ref2VA video encoding requires the local "
                    "AutoProcessor component with a video_processor"
                )
        return processor

    def _token_id(self, name: str) -> int:
        config = getattr(self.model, "config", None)
        value = getattr(config, name, None)
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise H3ComponentError(
                f"Qwen3-VL config is missing required {name}"
            ) from exc
        if token_id < 0:
            raise H3ComponentError(f"Qwen3-VL config has invalid {name}={token_id}")
        return token_id

    @staticmethod
    def _output_hidden(output: Any):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, dict):
            hidden = output.get("last_hidden_state")
        if hidden is None:
            raise H3ComponentError(
                "Qwen3-VL language_model did not return last_hidden_state"
            )
        return hidden

    @staticmethod
    def _vision_output_to_cpu(output: Any, *, context: str) -> dict[str, Any]:
        """Retain every DeepStack tensor needed by the later LM phase."""

        import torch

        pooled = getattr(output, "pooler_output", None)
        deepstack = getattr(output, "deepstack_features", None)
        if pooled is None or deepstack is None:
            raise H3ComponentError(
                f"Qwen3-VL {context} visual output lacks pooler_output or "
                "deepstack_features; refusing an incorrect pooled-only encode"
            )
        if isinstance(pooled, torch.Tensor):
            pooled_items = [pooled]
        elif isinstance(pooled, (tuple, list)) and pooled:
            pooled_items = list(pooled)
        else:
            raise H3ComponentError(
                f"Qwen3-VL {context} pooler_output has unsupported type "
                f"{type(pooled).__name__}"
            )
        if not isinstance(deepstack, (tuple, list)) or not deepstack:
            raise H3ComponentError(
                f"Qwen3-VL {context} returned no DeepStack features"
            )
        if not all(isinstance(item, torch.Tensor) for item in pooled_items):
            raise H3ComponentError(
                f"Qwen3-VL {context} returned non-tensor pooled features"
            )
        if not all(isinstance(item, torch.Tensor) for item in deepstack):
            raise H3ComponentError(
                f"Qwen3-VL {context} returned non-tensor DeepStack features"
            )
        pooled_tensor = torch.cat(pooled_items, dim=0).detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        deepstack_tensors = tuple(
            item.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            for item in deepstack
        )
        if pooled_tensor.ndim != 2:
            raise H3ComponentError(
                f"Qwen3-VL {context} pooled feature shape must be [N,D], got "
                f"{tuple(pooled_tensor.shape)}"
            )
        if any(
            item.ndim != 2
            or int(item.shape[0]) != int(pooled_tensor.shape[0])
            or int(item.shape[1]) != int(pooled_tensor.shape[1])
            for item in deepstack_tensors
        ):
            raise H3ComponentError(
                f"Qwen3-VL {context} DeepStack tensors do not align with "
                f"pooled shape {tuple(pooled_tensor.shape)}"
            )
        return {"pooled": pooled_tensor, "deepstack": deepstack_tensors}

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

    def _rollback_failed_quantized_load(self) -> bool:
        """Best-effort, repeatable rollback for a partial INT8 acquisition."""

        clean = True
        try:
            self._unload_linear_patcher()
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to detach Qwen INT8 Linear patcher during load rollback"
            )
        try:
            self._move_static_tensors(self.offload_device)
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to return Qwen static tensors to the offload device "
                "during load rollback"
            )
        try:
            self._set_compute_device(None)
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to clear Qwen compute-device marker during load rollback"
            )
            # Keep the wrapper state conservative even if a foreign model
            # object rejects marker removal.  Repeating the rollback remains
            # safe and will retry all physical cleanup actions.
            self._compute_device = None
        self._inference_active = False
        return clean

    def _rollback_failed_full_load(self, modules: Sequence[Any]) -> bool:
        """Best-effort rollback after a BF16/FP16 module move fails."""

        clean = True
        for module in modules:
            try:
                module.to(self.offload_device)
            except BaseException:
                clean = False
                LOGGER.exception(
                    "Failed to return a Qwen module to the offload device "
                    "during full-load rollback"
                )
        try:
            self._set_compute_device(None)
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to clear Qwen compute-device marker during "
                "full-load rollback"
            )
            self._compute_device = None
        self._inference_active = False
        return clean

    def load_for_inference(
        self, *, extra_headroom: int = 0
    ) -> "MiniMaxH3TextEncoder":
        import torch

        extra_headroom = max(0, int(extra_headroom))

        if self.load_device.type != "cuda":
            return self
        if self._inference_active and self.device == self.load_device:
            return self

        if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
            patcher = self._ensure_linear_patcher()
            reserve = (
                TE_GPU_HEADROOM
                + self._static_storage_bytes
                + extra_headroom
            )
            try:
                import comfy.model_management as mm

                # Reserve static embedding/norm/RoPE bytes plus encode
                # workspace.  ModelPatcher uses the remainder for resident
                # INT8 Linears and streams all others on Comfy's CUDA streams.
                mm.load_models_gpu([patcher], memory_required=reserve)
                self._move_static_tensors(self.load_device)
            except BaseException as exc:
                rollback_clean = self._rollback_failed_quantized_load()
                if isinstance(exc, torch.OutOfMemoryError) and rollback_clean:
                    LOGGER.warning(
                        "text_encoder INT8 partial load OOM; cleaned up and "
                        "falling back to CPU encode"
                    )
                    try:
                        mm.soft_empty_cache()
                    except BaseException:
                        try:
                            torch.cuda.empty_cache()
                        except BaseException:
                            LOGGER.exception(
                                "Failed to empty CUDA cache after Qwen INT8 OOM"
                            )
                    return self
                # Non-OOM errors must retain their original type, instance and
                # traceback.  An OOM whose rollback was incomplete is also
                # re-raised rather than running a mixed-device CPU fallback.
                raise
            self._set_compute_device(self.load_device)
            self._inference_active = True
            loaded = int(getattr(patcher, "loaded_size", lambda: 0)())
            LOGGER.info(
                "H3 text_encoder CUDA streaming ready: static %.2fGB, "
                "Linear resident %.2f/%.2fGB, workspace reserve %.2fGB, "
                "multimodal reserve %.2fGB",
                self._static_storage_bytes / 2**30,
                loaded / 2**30,
                self._linear_storage_bytes / 2**30,
                TE_GPU_HEADROOM / 2**30,
                extra_headroom / 2**30,
            )
            return self

        if self._actual_device() == self.load_device:
            self._set_compute_device(self.load_device)
            self._inference_active = True
            return self
        mods = self._movable()
        need = sum(_physical_module_bytes(module) for module in mods)
        free = torch.cuda.mem_get_info(self.load_device)[0] if torch.cuda.is_available() else 0
        if free < need + TE_GPU_HEADROOM + extra_headroom:
            LOGGER.warning("text_encoder 需 %.1fGB + %.1fGB 工作区 > 空闲 %.1fGB，回退 CPU encode",
                           need / 2**30, TE_GPU_HEADROOM / 2**30, free / 2**30)
            return self
        try:
            for m in mods:
                m.to(self.load_device)
        except BaseException as exc:
            rollback_clean = self._rollback_failed_full_load(mods)
            if isinstance(exc, torch.OutOfMemoryError) and rollback_clean:
                LOGGER.warning("text_encoder 上卡 OOM，回滚 CPU encode")
                try:
                    torch.cuda.empty_cache()
                except BaseException:
                    LOGGER.exception(
                        "Failed to empty CUDA cache after Qwen full-load OOM"
                    )
                return self
            raise
        self._set_compute_device(self.load_device)
        self._inference_active = True
        return self

    def offload_after_inference(self) -> None:
        import torch

        with self._lock:
            cleanup_error: BaseException | None = None

            def cleanup(action, message: str) -> None:
                nonlocal cleanup_error
                try:
                    action()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        LOGGER.exception(message)

            if self._linear_patcher is not None:
                cleanup(
                    self._unload_linear_patcher,
                    "Additional Qwen patcher-offload failure",
                )
                cleanup(
                    lambda: self._move_static_tensors(self.offload_device),
                    "Additional Qwen static-tensor offload failure",
                )
            else:
                try:
                    needs_offload = self._actual_device() != self.offload_device
                except BaseException as exc:
                    cleanup_error = exc
                    needs_offload = True
                if needs_offload:
                    for module in self._movable():
                        cleanup(
                            lambda module=module: module.to(self.offload_device),
                            "Additional Qwen module offload failure",
                        )
            cleanup(
                lambda: self._set_compute_device(None),
                "Additional Qwen compute-marker cleanup failure",
            )
            if hasattr(self.model, "_h3_compute_device"):
                cleanup(
                    lambda: delattr(self.model, "_h3_compute_device"),
                    "Additional Qwen model-marker cleanup failure",
                )
            # Logical residency state is session-scoped and must never survive
            # a failed physical cleanup.  A later idempotent offload call can
            # retry any tensor move which failed above.
            self._compute_device = None
            self._inference_active = False
            try:
                import comfy.model_management as mm

                mm.soft_empty_cache()
            except BaseException:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except BaseException:
                    LOGGER.exception("Failed to empty cache after Qwen offload")
            if cleanup_error is not None:
                raise cleanup_error

    def _encode_visual_features_staged(
        self,
        *,
        pixel_values: Any | None,
        image_grid_thw: Any | None,
        pixel_values_videos: Any | None,
        video_grid_thw: Any | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Run only the visual tower, retain pooled + DeepStack on CPU.

        The 7GB visual tower and the streamed 32B language model must not be
        resident on a 24GB card at the same time.  Calling the full Qwen model
        cannot provide that guarantee, so this phase explicitly uses
        ``get_image_features``/``get_video_features`` and offloads visual
        before the language phase starts.
        """

        import torch

        if not TE_VISUAL_ON_CPU:
            raise H3ComponentError(
                "Staged multimodal Qwen encoding requires TE_VISUAL_ON_CPU=True; "
                "otherwise load_for_inference could co-reside the visual tower "
                "with the language model"
            )

        # This is deliberately unconditional: a previous direct caller may
        # have left the language patcher resident after encode_prompt().
        self.offload_after_inference()
        visual = self._visual_module()
        visual_device = self.load_device
        if visual_device.type == "cuda":
            try:
                import comfy.model_management as mm

                mm.free_memory(
                    _physical_module_bytes(visual) + TE_GPU_HEADROOM,
                    visual_device,
                )
            except ImportError:
                # Plain-PyTorch environments have no other Comfy-managed
                # models to evict.  The explicit OOM path below remains safe.
                pass
        try:
            visual.to(visual_device)
        except BaseException as exc:
            self._offload_visual_after_staged(
                visual, preserve_primary_error=True
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if isinstance(exc, torch.OutOfMemoryError):
                raise H3ComponentError(
                    "Qwen3-VL visual tower does not fit after language-model "
                    "offload; staged multimodal encoding cannot continue safely"
                ) from exc
            raise

        image_features = None
        video_features = None
        try:
            parameter = next(visual.parameters(), None)
            visual_dtype = (
                parameter.dtype
                if parameter is not None and parameter.is_floating_point()
                else torch.bfloat16
            )
            if pixel_values is not None:
                getter = getattr(self.model, "get_image_features", None)
                if not callable(getter):
                    raise H3ComponentError(
                        "Installed Transformers Qwen3-VL lacks "
                        "get_image_features; refusing an unverified staged path"
                    )
                output = getter(
                    pixel_values.to(device=visual_device, dtype=visual_dtype),
                    image_grid_thw.to(device=visual_device, dtype=torch.long),
                    return_dict=True,
                )
                image_features = self._vision_output_to_cpu(
                    output, context="image"
                )
                del output
            if pixel_values_videos is not None:
                getter = getattr(self.model, "get_video_features", None)
                if not callable(getter):
                    raise H3ComponentError(
                        "Installed Transformers Qwen3-VL lacks "
                        "get_video_features; refusing an unverified staged path"
                    )
                output = getter(
                    pixel_values_videos.to(
                        device=visual_device, dtype=visual_dtype
                    ),
                    video_grid_thw.to(device=visual_device, dtype=torch.long),
                    return_dict=True,
                )
                video_features = self._vision_output_to_cpu(
                    output, context="video"
                )
                del output
        except torch.OutOfMemoryError as exc:
            raise H3ComponentError(
                "Qwen3-VL visual forward exhausted device memory after "
                "staged eviction; reduce reference count/resolution"
            ) from exc
        finally:
            self._offload_visual_after_staged(
                visual, preserve_primary_error=sys.exc_info()[0] is not None
            )
            if torch.cuda.is_available():
                try:
                    import comfy.model_management as mm

                    mm.soft_empty_cache()
                except ImportError:
                    torch.cuda.empty_cache()
        return image_features, video_features

    def _encode_staged_language(
        self,
        input_ids: Any,
        *,
        image_grid_thw: Any | None,
        video_grid_thw: Any | None,
        image_features: dict[str, Any] | None,
        video_features: dict[str, Any] | None,
    ):
        """Inject cached visual features and run the trimmed language model."""

        import torch

        feature_bytes = 0
        for feature_set in (image_features, video_features):
            if feature_set is None:
                continue
            feature_bytes += int(feature_set["pooled"].nbytes)
            feature_bytes += sum(
                int(item.nbytes) for item in feature_set["deepstack"]
            )
        # When image and video coexist, HF builds a modality-ordered joint
        # DeepStack list before the LM forward.  Reserve that second copy too.
        if image_features is not None and video_features is not None:
            feature_bytes *= 2
        self.load_for_inference(extra_headroom=feature_bytes)
        device = self.device
        visual = self._visual_module()
        if device.type == "cuda" and any(
            parameter.device.type == "cuda" for parameter in visual.parameters()
        ):
            raise H3ComponentError(
                "Qwen staged-residency invariant violated: visual tower is "
                "still on CUDA while loading the language model"
            )

        ids = input_ids.to(device=device, dtype=torch.long)[None]
        attention_mask = torch.ones_like(ids, dtype=torch.long)
        image_token_id = self._token_id("image_token_id")
        video_token_id = self._token_id("video_token_id")
        mm_types = minimax_h3_mm_token_type_ids(
            ids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )

        embedding = getattr(self.model, "get_input_embeddings", None)
        if not callable(embedding):
            raise H3ComponentError(
                "Qwen3-VL backbone lacks get_input_embeddings()"
            )
        inputs_embeds = embedding()(ids)

        image_positions = ids == image_token_id
        video_positions = ids == video_token_id
        image_mask = None
        video_mask = None
        image_deepstack = None
        video_deepstack = None

        if image_features is not None:
            pooled = image_features["pooled"].to(
                device=device, dtype=inputs_embeds.dtype
            )
            if int(image_positions.sum().item()) != int(pooled.shape[0]):
                raise H3ComponentError(
                    "Qwen image placeholder/features mismatch: "
                    f"tokens={int(image_positions.sum().item())}, "
                    f"features={int(pooled.shape[0])}"
                )
            image_mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, pooled)
            image_deepstack = tuple(
                item.to(device=device, dtype=inputs_embeds.dtype)
                for item in image_features["deepstack"]
            )
        elif bool(image_positions.any().item()):
            raise H3ComponentError(
                "Qwen presentation contains image pads but no image features"
            )

        if video_features is not None:
            pooled = video_features["pooled"].to(
                device=device, dtype=inputs_embeds.dtype
            )
            if int(video_positions.sum().item()) != int(pooled.shape[0]):
                raise H3ComponentError(
                    "Qwen video placeholder/features mismatch: "
                    f"tokens={int(video_positions.sum().item())}, "
                    f"features={int(pooled.shape[0])}"
                )
            video_mask = video_positions.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, pooled)
            video_deepstack = tuple(
                item.to(device=device, dtype=inputs_embeds.dtype)
                for item in video_features["deepstack"]
            )
        elif bool(video_positions.any().item()):
            raise H3ComponentError(
                "Qwen presentation contains video pads but no video features"
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_positions = image_mask[..., 0]
            video_positions = video_mask[..., 0]
            visual_pos_masks = image_positions | video_positions
            if image_deepstack is None or video_deepstack is None or len(
                image_deepstack
            ) != len(video_deepstack):
                raise H3ComponentError(
                    "Qwen image/video DeepStack layer counts do not match"
                )
            image_joint = image_positions[visual_pos_masks]
            video_joint = video_positions[visual_pos_masks]
            combined = []
            for image_item, video_item in zip(
                image_deepstack, video_deepstack
            ):
                joint = image_item.new_zeros(
                    int(visual_pos_masks.sum().item()), image_item.shape[-1]
                )
                joint[image_joint] = image_item
                joint[video_joint] = video_item
                combined.append(joint)
            deepstack_visual_embeds = tuple(combined)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = image_deepstack
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = video_deepstack

        if visual_pos_masks is None or not deepstack_visual_embeds:
            raise H3ComponentError(
                "Multimodal Qwen encode produced no DeepStack injection payload"
            )
        language_model = getattr(self.model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if language_model is None or layers is None:
            raise H3ComponentError(
                "Qwen3-VL backbone lacks language_model.layers"
            )
        if len(deepstack_visual_embeds) > len(layers):
            raise H3ComponentError(
                "Qwen visual DeepStack has more injection layers than the "
                "trimmed language model"
            )

        rope = getattr(self.model, "get_rope_index", None)
        if not callable(rope):
            raise H3ComponentError(
                "Installed Transformers Qwen3-VL lacks get_rope_index(); "
                "multimodal M-RoPE cannot be reproduced safely"
            )
        position_ids, _rope_delta = rope(
            ids,
            mm_token_type_ids=mm_types,
            image_grid_thw=(
                None
                if image_grid_thw is None
                else image_grid_thw.to(device=device, dtype=torch.long)
            ),
            video_grid_thw=(
                None
                if video_grid_thw is None
                else video_grid_thw.to(device=device, dtype=torch.long)
            ),
            attention_mask=attention_mask,
        )
        try:
            outputs = language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=list(deepstack_visual_embeds),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        except TypeError as exc:
            raise H3ComponentError(
                "Installed Transformers Qwen3-VL language model does not "
                "support verified DeepStack injection; refusing fallback"
            ) from exc
        hidden = self._output_hidden(outputs)[0].to(
            device="cpu", dtype=torch.bfloat16
        )
        expected = (int(ids.shape[1]), HIDDEN_SIZE)
        if tuple(hidden.shape) != expected:
            raise H3ComponentError(
                f"Unexpected Qwen3-VL feature shape {tuple(hidden.shape)}; "
                f"expected {expected}"
            )
        return hidden.contiguous()

    def encode_ids(
        self,
        input_ids: Any,
        *,
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_grid_thw: Any | None = None,
    ):
        """Encode official H3 presentation ids into CPU BF16 layer-50 states.

        Multimodal calls use a strict two-phase DeepStack path.  Text-only
        callers are routed through the existing T2VA implementation.
        """

        import torch

        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            shape = getattr(input_ids, "shape", None)
            raise H3ComponentError(
                f"Qwen input_ids must be a 1-D torch.Tensor, got {shape!r}"
            )
        if int(input_ids.numel()) <= 0:
            raise H3ComponentError("Qwen input_ids must be non-empty")
        if (pixel_values is None) != (image_grid_thw is None):
            raise H3ComponentError(
                "pixel_values and image_grid_thw must be supplied together"
            )
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise H3ComponentError(
                "pixel_values_videos and video_grid_thw must be supplied together"
            )
        has_multimodal = (
            pixel_values is not None or pixel_values_videos is not None
        )
        if not has_multimodal:
            # Keep text-only semantics centralized in encode_prompt.  Decode
            # ids back is not lossless, so run the same backbone directly.
            with self._lock, torch.inference_mode():
                self.load_for_inference()
                with _scoped_cudnn_sdp(self.device.type == "cuda"):
                    ids = input_ids.to(self.device, dtype=torch.long)[None]
                    outputs = self.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
                    hidden = self._output_hidden(outputs)[0].to(
                        device="cpu", dtype=torch.bfloat16
                    )
                    expected = (int(ids.shape[1]), HIDDEN_SIZE)
                    if tuple(hidden.shape) != expected:
                        raise H3ComponentError(
                            f"Unexpected Qwen3-VL feature shape "
                            f"{tuple(hidden.shape)}; expected {expected}"
                        )
                    return hidden.contiguous()

        with self._lock, torch.inference_mode():
            with _scoped_cudnn_sdp(self.load_device.type == "cuda"):
                image_features, video_features = (
                    self._encode_visual_features_staged(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                    )
                )
                return self._encode_staged_language(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_features=image_features,
                    video_features=video_features,
                )

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
            with _scoped_cudnn_sdp(self.device.type == "cuda"):
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

    @staticmethod
    def _conditioning_payload(
        *,
        prompt: str,
        hidden: Any,
        token_tags: Any,
        input_ids: Any,
        **metadata: Any,
    ) -> dict[str, Any]:
        import torch

        hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        token_tags = token_tags.detach().to(device="cpu", dtype=torch.long).contiguous()
        input_ids = input_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        if hidden.ndim != 2 or int(hidden.shape[1]) != HIDDEN_SIZE:
            raise H3ComponentError(
                f"Qwen conditioning must be [L,{HIDDEN_SIZE}], got "
                f"{tuple(hidden.shape)}"
            )
        if token_tags.ndim != 1 or int(token_tags.shape[0]) != int(
            hidden.shape[0]
        ):
            raise H3ComponentError(
                "Qwen text_token_tags must be [L] and align with prompt_embeds"
            )
        if input_ids.ndim != 1 or int(input_ids.shape[0]) != int(hidden.shape[0]):
            raise H3ComponentError(
                "Qwen presentation input_ids must align with prompt_embeds"
            )
        output = {
            "prompt": prompt,
            "prompt_embeds": hidden,
            # Compatibility alias used by the original T2VA runtime.
            "hidden_states": hidden,
            "text_len": int(hidden.shape[0]),
            "text_token_tags": token_tags,
            "presentation_input_ids": input_ids,
            "cfg_distilled": True,
        }
        output.update(metadata)
        return output

    def encode_fl2va_conditioning(
        self,
        prompt: str,
        images: Sequence[Any],
    ) -> dict[str, Any]:
        """Encode one/two already-prepared FL target-canvas keyframes.

        The caller owns FL canvas semantics.  The exact same prepared images
        must also be passed to the Video-VAE keyframe encoder.
        """

        from .presentation import (
            minimax_h3_multi_image_presentation_ids,
            minimax_h3_multi_image_presentation_token_tags,
        )

        if not isinstance(prompt, str) or not prompt:
            raise H3ComponentError("MiniMax-H3 prompt must be a non-empty string")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise H3ComponentError("FL2VA images must be an ordered sequence")
        images = list(images)
        if len(images) not in (1, 2):
            raise H3ComponentError(
                "FL2VA Qwen presentation requires exactly one or two keyframes"
            )
        processor = self._require_processor()
        vision = processor.image_processor(
            images=images,
            return_tensors="pt",
        )
        try:
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
        except (KeyError, TypeError) as exc:
            raise H3ComponentError(
                "Qwen image_processor must return pixel_values and image_grid_thw"
            ) from exc
        if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape[1:]) != (3,):
            raise H3ComponentError(
                f"Qwen image_grid_thw must be [N,3], got "
                f"{tuple(image_grid_thw.shape)}"
            )
        if int(image_grid_thw.shape[0]) != len(images):
            raise H3ComponentError(
                f"Qwen processor returned {int(image_grid_thw.shape[0])} grids "
                f"for {len(images)} FL2VA images"
            )
        merge_size = int(getattr(processor.image_processor, "merge_size", 0))
        if merge_size <= 0:
            raise H3ComponentError(
                "Qwen image_processor has no positive spatial merge_size"
            )
        merge_area = merge_size**2
        image_token_counts = [
            int(image_grid_thw[index].prod().item()) // merge_area
            for index in range(len(images))
        ]
        if any(count <= 0 for count in image_token_counts):
            raise H3ComponentError("Qwen processor produced an empty image grid")
        input_ids = minimax_h3_multi_image_presentation_ids(
            self.tokenizer,
            prompt=prompt,
            image_token_counts=image_token_counts,
        )
        token_tags = minimax_h3_multi_image_presentation_token_tags(
            self.tokenizer,
            prompt=prompt,
            image_token_counts=image_token_counts,
        )
        hidden = self.encode_ids(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return self._conditioning_payload(
            prompt=prompt,
            hidden=hidden,
            token_tags=token_tags,
            input_ids=input_ids,
            presentation="fl2va_multi_image_v1",
            image_token_counts=tuple(image_token_counts),
        )

    def encode_ref2va_conditioning(
        self,
        prompt: str,
        condition_labels: Sequence[tuple[str, int]],
        *,
        images: Sequence[Any] = (),
        videos: Sequence[Any] = (),
    ) -> dict[str, Any]:
        """Encode ordered Ref2VA references.

        ``videos`` contains one array/list of frames per Video label.  Frames
        must already be sampled from the prepared reference video at exactly
        2fps; this method disables processor-side sampling and derives the
        official temporal-patch timestamps from that sampled stream.
        """

        import numpy as np
        import torch

        from .presentation import (
            QWEN_VIDEO_SAMPLE_FPS,
            minimax_h3_qwen_video_sample_plan,
            minimax_h3_ref2va_video_presentation,
        )

        if not isinstance(prompt, str) or not prompt:
            raise H3ComponentError("MiniMax-H3 prompt must be a non-empty string")
        labels = [(str(kind), int(ordinal)) for kind, ordinal in condition_labels]
        if not labels:
            raise H3ComponentError("Ref2VA requires at least one condition label")
        counters = {"image": 0, "audio": 0, "video": 0}
        for kind, ordinal in labels:
            if kind not in counters:
                raise H3ComponentError(
                    f"Unsupported Ref2VA Qwen condition label {kind!r}"
                )
            counters[kind] += 1
            if ordinal != counters[kind]:
                raise H3ComponentError(
                    f"Ref2VA {kind} ordinals must be consecutive and 1-based; "
                    f"expected {counters[kind]}, got {ordinal}"
                )
        images = list(images)
        videos = list(videos)
        if len(images) != counters["image"]:
            raise H3ComponentError(
                f"Ref2VA labels require {counters['image']} images, got "
                f"{len(images)}"
            )
        if len(videos) != counters["video"]:
            raise H3ComponentError(
                f"Ref2VA labels require {counters['video']} sampled videos, got "
                f"{len(videos)}"
            )
        processor = self._require_processor(video=bool(videos))
        merge_size = int(getattr(processor.image_processor, "merge_size", 0))
        if merge_size <= 0:
            raise H3ComponentError(
                "Qwen image_processor has no positive spatial merge_size"
            )
        merge_area = merge_size**2

        pixel_values = None
        image_grid_thw = None
        image_token_counts: list[int] = []
        if images:
            image_output = processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            try:
                pixel_values = image_output["pixel_values"]
                image_grid_thw = image_output["image_grid_thw"]
            except (KeyError, TypeError) as exc:
                raise H3ComponentError(
                    "Qwen image_processor must return pixel_values and "
                    "image_grid_thw"
                ) from exc
            if int(image_grid_thw.shape[0]) != len(images):
                raise H3ComponentError(
                    "Qwen image grid count does not match Ref2VA image count"
                )
            image_token_counts = [
                int(image_grid_thw[index].prod().item()) // merge_area
                for index in range(len(images))
            ]

        pixel_values_videos = None
        video_grid_thw = None
        video_block_token_counts: list[list[int]] = []
        video_block_timestamps: list[list[float]] = []
        if videos:
            stacked_videos = []
            sampled_counts = []
            for index, video in enumerate(videos):
                if isinstance(video, torch.Tensor):
                    array = video.detach().cpu().numpy()
                elif isinstance(video, np.ndarray):
                    array = video
                else:
                    try:
                        array = np.stack(list(video))
                    except Exception as exc:
                        raise H3ComponentError(
                            f"Ref2VA sampled video {index} cannot be stacked"
                        ) from exc
                if array.ndim != 4 or int(array.shape[0]) <= 0 or int(
                    array.shape[-1]
                ) != 3:
                    raise H3ComponentError(
                        "Ref2VA sampled videos must have shape [T,H,W,3], got "
                        f"{tuple(array.shape)} for video {index}"
                    )
                stacked_videos.append(array)
                sampled_counts.append(int(array.shape[0]))
            video_output = processor.video_processor(
                videos=stacked_videos,
                do_sample_frames=False,
                return_tensors="pt",
            )
            try:
                pixel_values_videos = video_output["pixel_values_videos"]
                video_grid_thw = video_output["video_grid_thw"]
            except (KeyError, TypeError) as exc:
                raise H3ComponentError(
                    "Qwen video_processor must return pixel_values_videos and "
                    "video_grid_thw"
                ) from exc
            if int(video_grid_thw.shape[0]) != len(videos):
                raise H3ComponentError(
                    "Qwen video grid count does not match Ref2VA video count"
                )
            for index, sample_count in enumerate(sampled_counts):
                _indices, timestamps = minimax_h3_qwen_video_sample_plan(
                    sample_count,
                    source_fps=QWEN_VIDEO_SAMPLE_FPS,
                    sample_fps=QWEN_VIDEO_SAMPLE_FPS,
                )
                temporal_blocks = int(video_grid_thw[index, 0].item())
                if temporal_blocks != len(timestamps):
                    raise H3ComponentError(
                        "Qwen temporal-patch mismatch for Ref2VA video "
                        f"{index}: processor={temporal_blocks}, "
                        f"timestamps={len(timestamps)}"
                    )
                per_block = (
                    int(video_grid_thw[index, 1].item())
                    * int(video_grid_thw[index, 2].item())
                    // merge_area
                )
                if per_block <= 0:
                    raise H3ComponentError(
                        f"Qwen processor produced empty video grid {index}"
                    )
                video_block_token_counts.append(
                    [per_block] * temporal_blocks
                )
                video_block_timestamps.append(timestamps)

        input_ids, token_tags = minimax_h3_ref2va_video_presentation(
            self.tokenizer,
            prompt=prompt,
            condition_labels=labels,
            image_token_count=image_token_counts or None,
            video_block_token_counts=video_block_token_counts or None,
            video_block_timestamps=video_block_timestamps or None,
        )
        hidden = self.encode_ids(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        return self._conditioning_payload(
            prompt=prompt,
            hidden=hidden,
            token_tags=token_tags,
            input_ids=input_ids,
            presentation="ref2va_ordered_v1",
            condition_labels=tuple(labels),
            image_token_counts=tuple(image_token_counts),
            video_block_token_counts=tuple(
                tuple(item) for item in video_block_token_counts
            ),
            video_block_timestamps=tuple(
                tuple(item) for item in video_block_timestamps
            ),
            video_sample_fps=QWEN_VIDEO_SAMPLE_FPS,
        )

    def encode_conditioning(self, prompt: str) -> dict[str, Any]:
        import torch

        hidden = self.encode_prompt(prompt)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        return self._conditioning_payload(
            prompt=prompt,
            hidden=hidden,
            token_tags=torch.ones(int(hidden.shape[0]), dtype=torch.long),
            input_ids=input_ids,
            presentation="t2va_text_only_v1",
        )


def _component_is_quantized(
    component: Path,
    *,
    shards: Sequence[Path] | None = None,
    weight_map: dict[str, str] | None = None,
) -> bool:
    from .model_loader import _checkpoint_index
    from safetensors import safe_open

    if shards is None:
        shards, weight_map = _checkpoint_index(component)
    if weight_map:
        return any(str(key).endswith(QUANT_KEY_SUFFIXES) for key in weight_map)
    for shard in shards:
        with safe_open(str(shard), framework="pt") as reader:
            if any(key.endswith(QUANT_KEY_SUFFIXES) for key in reader.keys()):
                return True
    return False


def _validate_text_encoder_quant_metadata(
    component: Path, *, partition: str | None
) -> dict[str, Any]:
    """Validate the manifest for an INT8 Qwen checkpoint.

    Older converter output did not record a partition.  It remains usable
    because the released Qwen component is shared by FL2VA and Ref2VA, but it
    cannot itself prove partition provenance.  New output records ``shared``;
    a concrete partition, when declared, must match the loader request.
    """

    path = component / "quant_meta.json"
    if not path.is_file():
        raise H3ComponentError(
            f"INT8 Qwen checkpoint requires {path} with format and convrot metadata"
        )
    metadata = read_json(path)
    if metadata.get("format") != INT8_FORMAT:
        raise H3ComponentError(
            f"{path}.format must be {INT8_FORMAT!r} for an INT8 H3 text encoder"
        )
    if metadata.get("convrot") is not True:
        raise H3ComponentError(
            f"{path}.convrot must be true for an INT8 H3 text encoder"
        )
    if "arch" in metadata and metadata.get("arch") != "qwen3_vl_text_encoder":
        raise H3ComponentError(
            f"{path}.arch must be 'qwen3_vl_text_encoder', got "
            f"{metadata.get('arch')!r}"
        )
    optional_integer_contract = {
        "selected_layers": SELECTED_LAYERS,
        "quantized_linears": SELECTED_LAYERS * 7,
    }
    for name, expected in optional_integer_contract.items():
        if name not in metadata:
            continue
        try:
            actual = int(metadata[name])
        except (TypeError, ValueError):
            actual = None
        if actual != expected:
            raise H3ComponentError(
                f"{path}.{name} must be {expected}, got {metadata[name]!r}"
            )

    declared = metadata.get("partition")
    if declared is None:
        LOGGER.warning(
            "Legacy Qwen INT8 manifest %s has no partition provenance; "
            "accepting it only as the architecture-shared FL2VA/Ref2VA text "
            "encoder",
            path,
        )
        return metadata
    if not isinstance(declared, str) or not declared.strip():
        raise H3ComponentError(
            f"{path}.partition must be 'shared', 'FL2VA', or 'Ref2VA'"
        )
    normalized_declared = declared.strip().lower()
    if normalized_declared not in {"shared", "fl2va", "ref2va"}:
        raise H3ComponentError(
            f"{path}.partition must be 'shared', 'FL2VA', or 'Ref2VA', got "
            f"{declared!r}"
        )
    if normalized_declared == "shared":
        return metadata
    if partition is None:
        raise H3ComponentError(
            f"{path} declares partition {declared!r}; the text-encoder loader "
            "must receive an explicit partition to validate it"
        )
    normalized_requested = str(partition).strip().lower()
    if normalized_requested not in {"fl2va", "ref2va"}:
        raise H3ComponentError(
            f"Unknown text-encoder partition {partition!r}; expected 'fl2va' "
            "or 'ref2va'"
        )
    if normalized_declared != normalized_requested:
        raise H3ComponentError(
            "INT8 Qwen checkpoint partition mismatch: requested "
            f"{normalized_requested!r}, but {path} declares {declared!r}"
        )
    return metadata


def _validate_text_encoder_quant_marker(
    prefix: str,
    marker_config: dict[str, Any],
    quant_metadata: dict[str, Any],
) -> None:
    """Require each Linear marker to agree with the component manifest."""

    for name in ("format", "convrot"):
        expected = quant_metadata.get(name)
        actual = marker_config.get(name)
        if actual != expected:
            raise H3ComponentError(
                f"Qwen quantization metadata mismatch for {prefix}: "
                f"marker {name}={actual!r}, quant_meta.json={expected!r}"
            )


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


def _intentional_qwen_cut_key(local: str) -> bool:
    """Return whether a checkpoint key belongs to deliberately removed state."""

    if local == "lm_head.weight":
        return True
    later = _LATER_LAYER_LOCAL_KEY.match(local)
    if later and int(later.group(1)) >= SELECTED_LAYERS:
        return True
    # Some Transformers versions serialized this deterministic, now
    # non-persistent buffer.  Restrict the exception to the language backbone.
    return (
        local.startswith(("language_model.", "visual."))
        and local.endswith("rotary_emb.inv_freq")
    )


def _validate_qwen_tensor_shape(name: str, tensor, target) -> None:
    if tuple(tensor.shape) != tuple(target.shape):
        raise H3ComponentError(
            f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
            f"vs model {tuple(target.shape)}"
        )


def _flush_linear_bag(
    module,
    prefix: str,
    bag: dict,
    device,
    *,
    quant_metadata: dict[str, Any] | None = None,
) -> bool:
    """Use the same strict meta-to-QuantizedTensor path as the DiT loader."""

    from .model_loader import _flush_linear, _quantized_linear_config

    if quant_metadata is not None:
        marker_config = _quantized_linear_config(prefix, bag)
        if marker_config is not None:
            _validate_text_encoder_quant_marker(
                prefix, marker_config, quant_metadata
            )

    return _flush_linear(module, prefix, bag, device)


def _stream_load_quantized_backbone(
    model,
    component: Path,
    *,
    offload_device,
    quant_metadata: dict[str, Any],
) -> None:
    """流式写入 int8_convrot + 透传 bf16（visual/embed/norm）。"""
    from safetensors import safe_open
    from .model_loader import _checkpoint_index  # 复用分片索引

    # All index and shard paths pass through the same canonical containment
    # gate as the DiT loader.  Never fall back after a malformed/escaping index.
    shards, weight_map = _checkpoint_index(component)
    indexed_weight_map = weight_map
    if weight_map is None:
        weight_map = {}
        for shard in shards:
            with safe_open(str(shard), framework="pt") as reader:
                for key in reader.keys():
                    if key in weight_map:
                        raise H3ComponentError(
                            f"Duplicate checkpoint tensor {key!r} across shards"
                        )
                    weight_map[key] = shard.name

    linears = {
        f"{name}.": mod for name, mod in model.named_modules()
        if name and hasattr(mod, "in_features") and hasattr(mod, "out_features")
    }
    expected = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
    # state_dict omits deterministic non-persistent buffers.  Keep them in
    # ``expected`` so older snapshots may supply them, but do not require them.
    required = set(model.state_dict())
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
    seen_local: set[str] = set()
    seen_checkpoint_keys: set[str] = set()
    unexpected: list[str] = []
    expected_quantized = sum(
        "comfy_quant" in leaves for leaves in needed.values()
    )
    loaded_quantized = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for ck in reader.keys():
                if indexed_weight_map is not None:
                    declared_shard = indexed_weight_map.get(ck)
                    if declared_shard is None:
                        raise H3ComponentError(
                            f"Checkpoint shard {shard} contains unindexed tensor {ck!r}"
                        )
                    if (component / declared_shard).resolve() != shard.resolve():
                        raise H3ComponentError(
                            f"Checkpoint index maps {ck!r} to {declared_shard!r}, "
                            f"but it was found in {shard}"
                        )
                seen_checkpoint_keys.add(ck)
                local = ck
                for pref in _STRIP_PREFIXES:
                    if local.startswith(pref):
                        local = local[len(pref):]
                        break
                if local in seen_local:
                    raise H3ComponentError(
                        f"Duplicate checkpoint tensor after prefix removal: {local!r}"
                    )
                seen_local.add(local)
                matched = None
                for lp in linears:
                    if local.startswith(lp) and local[len(lp):] in _LINEAR_LEAVES:
                        matched = (lp, local[len(lp):])
                        break
                if matched is None and local not in expected:
                    if not _intentional_qwen_cut_key(local):
                        unexpected.append(ck)
                    continue
                tensor = reader.get_tensor(ck)
                if matched:
                    lp, leaf = matched
                    if local in loaded or leaf in pending.get(lp, {}):
                        raise H3ComponentError(
                            f"Duplicate checkpoint tensor {local!r} in {shard}"
                        )
                    pending[lp][leaf] = tensor
                    if needed[lp] <= set(pending[lp]):
                        loaded_quantized += int(
                            _flush_linear_bag(
                                linears[lp],
                                lp,
                                pending.pop(lp),
                                offload_device,
                                quant_metadata=quant_metadata,
                            )
                        )
                        loaded.update(f"{lp}{x}" for x in needed[lp])
                    continue
                target = expected[local]
                _validate_qwen_tensor_shape(local, tensor, target)
                if (
                    tensor.is_floating_point()
                    and getattr(target, "is_floating_point", lambda: False)()
                    and tensor.dtype != target.dtype
                ):
                    tensor = tensor.to(dtype=target.dtype)
                _assign_param(model, local, tensor.to(device=offload_device))
                loaded.add(local)

    if indexed_weight_map is not None:
        absent_from_shards = sorted(set(indexed_weight_map) - seen_checkpoint_keys)
        if absent_from_shards:
            raise H3ComponentError(
                "Qwen checkpoint index contains tensors absent from its shards: "
                f"{absent_from_shards[:12]!r}"
            )
    for lp, bag in list(pending.items()):
        loaded_quantized += int(
            _flush_linear_bag(
                linears[lp],
                lp,
                bag,
                offload_device,
                quant_metadata=quant_metadata,
            )
        )
        loaded.update(f"{lp}{x}" for x in bag)
    required_quantized = SELECTED_LAYERS * 7
    if not (
        expected_quantized == loaded_quantized == required_quantized
    ):
        raise H3ComponentError(
            "H3 text_encoder quantized Linear contract failed: "
            f"materialized={loaded_quantized}, checkpoint={expected_quantized}, "
            f"required={required_quantized}"
        )
    LOGGER.info(
        "H3 text_encoder materialized %d complete INT8/convrot QuantizedTensor layers",
        loaded_quantized,
    )
    missing = sorted(
        name
        for name in required - loaded
        if not _intentional_qwen_cut_key(name)
    )
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:12]!r} (total {len(missing)})")
        if unexpected:
            details.append(
                f"unexpected={unexpected[:12]!r} (total {len(unexpected)})"
            )
        raise H3ComponentError(
            "H3 text_encoder checkpoint contract failed: " + "; ".join(details)
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
    quant_metadata: dict[str, Any],
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
    expected_swaps = SELECTED_LAYERS * 7
    if n_swap != expected_swaps:
        raise H3ComponentError(
            "Qwen INT8 architecture must expose exactly "
            f"{expected_swaps} quantized language Linears, got {n_swap}"
        )
    _stream_load_quantized_backbone(
        model,
        component,
        offload_device=offload_device,
        quant_metadata=quant_metadata,
    )
    language_model.norm = torch.nn.Identity()
    if hasattr(language_model, "config"):
        language_model.config.num_hidden_layers = SELECTED_LAYERS
        language_model.config.output_hidden_states = False
        language_model.config.use_cache = False
    del causal_lm
    model.requires_grad_(False).eval()
    return model


def _load_qwen_bf16_checkpoint(
    model_cls: Any,
    component: Path,
    *,
    model_dtype: Any,
    load_kwargs: dict[str, Any],
):
    """Load only safetensors, retaining the Transformers dtype compatibility path."""

    try:
        return model_cls.from_pretrained(
            str(component),
            dtype=model_dtype,
            **load_kwargs,
        )
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return model_cls.from_pretrained(
            str(component),
            torch_dtype=model_dtype,
            **load_kwargs,
        )


def load_h3_text_encoder(
    model_root: str,
    *,
    partition: str | None = None,
    require_multimodal_processor: bool = False,
    text_encoder_path: str | None = None,
    tokenizer_path: str | None = None,
    processor_path: str | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    offload_device: str = "cpu",
    attention_backend: str = "sdpa",
) -> MiniMaxH3TextEncoder:
    """Load the local Qwen3-VL component with no Hub or remote-code fallback."""

    import torch
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

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
    processor_components: list[Path] = []
    if processor_path:
        processor_components.append(
            resolve_component(
                model_root,
                ("processor", "text_encoder", "tokenizer"),
                explicit=processor_path,
            )
        )
    else:
        # Official releases normally own a processor slot, while repacked INT8
        # releases sometimes co-locate preprocessor_config.json with the text
        # encoder or tokenizer.  Probe every supported local layout and never
        # download/fall back to remote code.
        for required_name in (
            "preprocessor_config.json",
            "processor_config.json",
        ):
            try:
                processor_components.append(
                    resolve_component(
                        model_root,
                        ("processor", "text_encoder", "tokenizer"),
                        required_files=(required_name,),
                    )
                )
            except H3ComponentError:
                pass
        for candidate_path in (component, tokenizer_component):
            if any(
                (candidate_path / name).is_file()
                for name in (
                    "preprocessor_config.json",
                    "processor_config.json",
                )
            ):
                processor_components.append(candidate_path)
        # Preserve a final AutoProcessor inference attempt for releases whose
        # processor class is declared only through config.json.
        try:
            processor_components.append(
                resolve_component(
                    model_root,
                    ("processor", "text_encoder", "tokenizer"),
                )
            )
        except H3ComponentError:
            pass
    processor_components = list(dict.fromkeys(processor_components))
    # Admit all small tokenizer/processor artifacts before constructing the
    # ~64GB Qwen model.  FL2VA/Ref2VA loaders can therefore fail closed without
    # paying the model allocation cost when their processor is absent.
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_component),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    processor = None
    loaded_processor_component = None
    processor_errors: list[str] = []
    for processor_component in processor_components:
        try:
            candidate = AutoProcessor.from_pretrained(
                str(processor_component),
                local_files_only=True,
                trust_remote_code=False,
            )
            if getattr(candidate, "image_processor", None) is None:
                raise H3ComponentError(
                    f"AutoProcessor at {processor_component} has no image_processor"
                )
            processor = candidate
            loaded_processor_component = processor_component
            break
        except Exception as exc:
            processor_errors.append(f"{processor_component}: {exc}")
    if processor is None:
        if processor_path or require_multimodal_processor:
            details = "; ".join(processor_errors) or "no processor candidate"
            raise H3ComponentError(
                "Could not load the required local Qwen3-VL multimodal "
                f"processor: {details}"
            )
        LOGGER.warning(
            "Could not load a local Qwen3-VL AutoProcessor from candidates %s; "
            "T2VA remains available but multimodal tasks will fail closed. %s",
            [str(path) for path in processor_components],
            "; ".join(processor_errors),
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

    from .model_loader import _checkpoint_index

    checkpoint_shards, checkpoint_weight_map = _checkpoint_index(component)
    quantized = _component_is_quantized(
        component,
        shards=checkpoint_shards,
        weight_map=checkpoint_weight_map,
    )
    if quantized:
        quant_metadata = _validate_text_encoder_quant_metadata(
            component, partition=partition
        )
        if not ALLOW_PARTIAL_OFFLOAD_INT8:
            raise H3ComponentError("int8 text_encoder 需要 ALLOW_PARTIAL_OFFLOAD_INT8")
        model = _load_quantized_text_encoder(
            component, config=config, model_dtype=model_dtype,
            offload_device=target_offload_device, attention_backend=attention_backend,
            quant_metadata=quant_metadata,
        )
    else:
        stale_quant_meta = component / "quant_meta.json"
        if stale_quant_meta.is_file():
            raise H3ComponentError(
                f"{stale_quant_meta} declares a quantized text encoder, but "
                "the checkpoint contains no quantized Linear markers"
            )
        model_cls = _qwen_causal_lm_class()
        load_kwargs = {
            "config": config,
            "local_files_only": True,
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
            "attn_implementation": attention_backend,
            "output_loading_info": True,
        }
        # Construct on CPU.  The native H3 lifecycle never lets this ~64 GB
        # encoder become GPU-resident during component loading; encode_prompt()
        # moves it to the selected Comfy device only for the actual encode, then
        # immediately offloads it again.
        loaded = _load_qwen_bf16_checkpoint(
            model_cls,
            component,
            model_dtype=model_dtype,
            load_kwargs=load_kwargs,
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

    return MiniMaxH3TextEncoder(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        component_path=component,
        load_device=load_device,
        offload_device=target_offload_device,
        quantized=quantized,
        tokenizer_component_path=tokenizer_component,
        processor_component_path=loaded_processor_component,
    )


__all__ = [
    "HIDDEN_SIZE",
    "SELECTED_LAYERS",
    "MiniMaxH3TextEncoder",
    "load_h3_text_encoder",
    "minimax_h3_mm_token_type_ids",
]
