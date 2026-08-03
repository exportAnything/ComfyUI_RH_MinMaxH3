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

from ..backend_state import TORCH_BACKEND_STATE_LOCK
from ..components import (
    H3ComponentError,
    model_root_path,
    read_json,
    resolve_component,
    validate_weight_partition,
)
from ..h3_settings import (
    ALLOW_PARTIAL_OFFLOAD_INT8,
    INT8_FORMAT,
    INT8_TE_DIRNAME,
    QUANT_KEY_SUFFIXES,
    TE_GPU_HEADROOM,
    TE_VISUAL_ON_CPU,
    TEXT_ENCODER_SELECTED_LAYERS,
    TRANSFORMERS_MAX_VERSION,
    TRANSFORMERS_MIN_VERSION,
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
# MiniMax-H3 processor 实测契约；防指错目录静默降级
_PROCESSOR_IMAGE_CONTRACT = {
    "shortest_edge": 65536, "longest_edge": 16777216,
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
    "image_mean": (0.5, 0.5, 0.5), "image_std": (0.5, 0.5, 0.5),
}
_PROCESSOR_VIDEO_CONTRACT = {
    "shortest_edge": 4096, "longest_edge": 25165824,
    "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
    "image_mean": (0.5, 0.5, 0.5), "image_std": (0.5, 0.5, 0.5),
}
_LINEAR_LEAVES = frozenset(
    {"weight", "bias", "weight_scale", "weight_scale_2", "comfy_quant", "input_scale"}
)
_STRIP_PREFIXES = ("model.",)  # checkpoint 相对 causal_lm.model 的键

def _norm_mean_std(value: Any) -> tuple[float, ...] | None:
    try:
        return tuple(round(float(x), 6) for x in value)
    except (TypeError, ValueError):
        return None

def _size_edges(size: Any) -> tuple[int | None, int | None]:
    """兼容 size={shortest_edge,longest_edge} 或旧 min_pixels/max_pixels。"""
    if not isinstance(size, dict):
        return None, None
    short = size.get("shortest_edge", size.get("min_pixels"))
    long = size.get("longest_edge", size.get("max_pixels"))
    try:
        return (int(short) if short is not None else None,
                int(long) if long is not None else None)
    except (TypeError, ValueError):
        return None, None

def _validate_processor_json(path: Path, contract: dict[str, Any], *, label: str) -> list[str]:
    """校验 preprocessor JSON；缺文件返回空（由调用方决定是否强制）。"""
    if not path.is_file():
        return []
    try:
        data = read_json(path)
    except Exception as exc:
        return [f"{label}: 无法读取 {path.name}: {exc}"]
    bad: list[str] = []
    short, long = _size_edges(data.get("size") or data)
    if short != contract["shortest_edge"] or long != contract["longest_edge"]:
        bad.append(
            f"{label}.size shortest/longest={short}/{long} "
            f"(expected {contract['shortest_edge']}/{contract['longest_edge']})"
        )
    for key in ("patch_size", "temporal_patch_size", "merge_size"):
        actual = data.get(key)
        try:
            actual_i = int(actual) if actual is not None else None
        except (TypeError, ValueError):
            actual_i = None
        if actual_i != contract[key]:
            bad.append(f"{label}.{key}={actual!r} (expected {contract[key]})")
    for key in ("image_mean", "image_std"):
        got = _norm_mean_std(data.get(key))
        if got != contract[key]:
            bad.append(f"{label}.{key}={data.get(key)!r} (expected {list(contract[key])})")
    return bad

def validate_h3_processor_contract(processor_dir: Path | str, *, require_video: bool = True) -> None:
    """校验 image/video preprocessor；失败 fail-closed。"""
    root = Path(processor_dir)
    bad = _validate_processor_json(
        root / "preprocessor_config.json", _PROCESSOR_IMAGE_CONTRACT, label="image",
    )
    video_path = root / "video_preprocessor_config.json"
    if require_video or video_path.is_file():
        if not video_path.is_file():
            bad.append("缺少 video_preprocessor_config.json")
        else:
            bad.extend(_validate_processor_json(
                video_path, _PROCESSOR_VIDEO_CONTRACT, label="video",
            ))
    if bad:
        raise H3ComponentError(
            "H3 processor 与 MiniMax-H3 发布配置不一致（勿混用通用 Qwen3-VL "
            "processor）：" + "; ".join(bad)
        )

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

__all__ = [n for n in list(globals()) if not n.startswith("__")]
