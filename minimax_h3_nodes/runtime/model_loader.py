"""Streaming local checkpoint loader for the direct H3 DiT."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .components import (
    H3ComponentError,
    model_root_path,
    read_json,
    release_metadata,
    resolve_component,
    validate_t2va_partition,
)
import logging

from .h3_settings import (
    ALLOW_PARTIAL_OFFLOAD_INT8,
    DIT_INFERENCE_RESERVE,
    FORCE_FULL_LOAD_BF16,
    INT8_DIT_DIRNAME,
    INT8_FORMAT,
    QUANT_KEY_SUFFIXES,
)

LOGGER = logging.getLogger(__name__)

_TRANSFORMER_CONFIG_FIELDS = frozenset(
    {
        "hidden_size",
        "num_layers",
        "token_refiner_num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "ffn_hidden_size",
        "latents_dim",
        "audio_latents_dim",
        "patch_size",
        "text_dim",
        "timestep_input_dim",
        "time_embed_hidden_size",
        "time_embed_dim",
        "adaln_out_features",
        "final_adaln_out_features",
        "rope_inv_freq_len",
        "norm_eps",
        "qk_norm_eps",
        "final_norm_eps",
    }
)
_LINEAR_LEAVES = frozenset(
    {"weight", "bias", "weight_scale", "weight_scale_2", "comfy_quant", "input_scale"}
)
_OUTER_PREFIXES = (
    "model.diffusion_model.",
    "model.transformer.",
    "diffusion_model.",
    "transformer.",
    "module.",
    "model.",
)
_AUX_QUANT_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
)


def _validate_transformer_config(raw: dict, path: Path) -> None:
    """Validate the concrete Diffusers-style config shipped with H3."""

    class_name = raw.get("_class_name")
    if class_name not in (None, "MiniMaxH3DiTModel"):
        raise H3ComponentError(
            f"{path} declares unsupported _class_name={class_name!r}; "
            "expected 'MiniMaxH3DiTModel'"
        )
    missing = sorted(_TRANSFORMER_CONFIG_FIELDS - set(raw))
    if missing:
        raise H3ComponentError(
            f"{path} is missing H3 transformer config fields: {missing!r}"
        )


def _dtype(value: str):
    import torch

    normalized = str(value).lower().replace("torch.", "")
    choices = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if normalized not in choices:
        raise H3ComponentError(
            "H3 DiT supports bfloat16 or float16 base weights; "
            "the patch/time/final projections remain float32"
        )
    return choices[normalized]


def _devices(device: str, offload_device: str):
    import torch

    if device == "auto":
        try:
            import comfy.model_management as mm

            load = mm.get_torch_device()
        except ImportError:
            load = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        load = torch.device(device)
    if offload_device == "auto":
        try:
            import comfy.model_management as mm

            offload = mm.unet_offload_device()
        except ImportError:
            offload = torch.device("cpu")
    else:
        offload = torch.device(offload_device)
    return load, offload


def _checkpoint_index(component: Path) -> tuple[list[Path], dict[str, str] | None]:
    """返回 (分片路径列表, weight_map 或 None)。"""
    index_candidates = (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
        "transformer.safetensors.index.json",
    )
    for name in index_candidates:
        path = component / name
        if not path.is_file():
            continue
        value = read_json(path)
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise H3ComponentError(f"{path} has no non-empty weight_map")
        files = sorted({component / str(item) for item in weight_map.values()})
        missing = [item for item in files if not item.is_file()]
        if missing:
            raise H3ComponentError(
                f"Checkpoint index references missing shards: {missing!r}"
            )
        return files, {str(k): str(v) for k, v in weight_map.items()}

    patterns = (
        "diffusion_pytorch_model*.safetensors",
        "model*.safetensors",
        "transformer*.safetensors",
        "*.safetensors",
    )
    for pattern in patterns:
        files = sorted(component.glob(pattern))
        if files:
            return files, None
    raise H3ComponentError(
        f"No safetensors checkpoint or shard index found below {component}"
    )


def _is_quantized_map(weight_map: dict[str, str] | None, shards: list[Path]) -> bool:
    if weight_map:
        return any(k.endswith(QUANT_KEY_SUFFIXES) for k in weight_map)
    from safetensors import safe_open

    with safe_open(str(shards[0]), framework="pt") as reader:
        return any(k.endswith(QUANT_KEY_SUFFIXES) for k in reader.keys())


def _require_int8_ops(compute_dtype):
    """构建 mixed_precision_ops；Comfy 过旧则明确报错。"""
    import inspect

    try:
        import comfy.ops as cops
    except ImportError as exc:
        raise H3ComponentError(
            "INT8/convrot checkpoint 需要 ComfyUI（comfy.ops）"
        ) from exc
    src = inspect.getsource(cops._load_quantized_module)
    if INT8_FORMAT not in src:
        raise H3ComponentError(
            f"当前 ComfyUI 的 comfy.ops 不支持 {INT8_FORMAT}/convrot，请升级 ComfyUI"
        )
    return cops.mixed_precision_ops({}, compute_dtype=compute_dtype)


def _tensor_nbytes(tensor) -> int:
    try:
        from comfy.quant_ops import QuantizedTensor

        if isinstance(tensor, QuantizedTensor):
            q = tensor._qdata
            n = int(q.numel()) * int(q.element_size())
            params = getattr(tensor, "_params", None)
            if params is not None:
                for attr in ("scale", "block_scale"):
                    scale = getattr(params, attr, None)
                    if scale is not None and hasattr(scale, "numel"):
                        n += int(scale.numel()) * int(scale.element_size())
            return n
    except ImportError:
        pass
    return int(tensor.numel()) * int(tensor.element_size())


def _model_nbytes(model) -> int:
    return sum(_tensor_nbytes(t) for t in model.state_dict().values())


def _linear_modules(model) -> dict[str, Any]:
    return {
        f"{name}.": module
        for name, module in model.named_modules()
        if name
        and hasattr(module, "in_features")
        and hasattr(module, "out_features")
    }


def _strip_outer(key: str) -> list[str]:
    out = [key]
    for prefix in _OUTER_PREFIXES:
        if key.startswith(prefix):
            out.append(key[len(prefix) :])
    return out


def _match_linear(local: str, linears: dict[str, Any]) -> tuple[str, str] | None:
    for pref in linears:
        if local.startswith(pref):
            leaf = local[len(pref) :]
            if leaf in _LINEAR_LEAVES:
                return pref, leaf
    return None


def _assign_tensor(module, name: str, value) -> None:
    """Replace one meta parameter/buffer without walking the full state dict."""

    import torch

    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        if part.isdigit() and hasattr(parent, "__getitem__"):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf in parent._parameters:
        previous = parent._parameters[leaf]
        requires_grad = bool(previous.requires_grad) if previous is not None else False
        parent._parameters[leaf] = torch.nn.Parameter(
            value,
            requires_grad=requires_grad,
        )
        return
    if leaf in parent._buffers:
        parent._buffers[leaf] = value
        return
    raise H3ComponentError(f"Checkpoint key is not a parameter or buffer: {name}")


def _flush_linear(module, prefix: str, bag: dict[str, Any], device) -> None:
    import inspect
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        for leaf, tensor in bag.items():
            if leaf not in {"weight", "bias"}:
                continue
            target = getattr(module, leaf)
            if target is not None and tuple(tensor.shape) != tuple(target.shape):
                raise H3ComponentError(
                    f"Shape mismatch for {prefix}{leaf}: checkpoint "
                    f"{tuple(tensor.shape)} vs model {tuple(target.shape)}"
                )
            _assign_tensor(module, leaf, tensor.to(device=device))
        return
    state = {f"{prefix}{leaf}": tensor.to(device=device) for leaf, tensor in bag.items()}
    missing: list[str] = []
    unexpected: list[str] = []
    errors: list[str] = []
    # meta 模块必须 assign，否则 copy_ 静默 no-op（comfy.ops 认 assign_to_params_buffers）
    meta = {"assign_to_params_buffers": True}
    kwargs = {}
    if "assign" in inspect.signature(module._load_from_state_dict).parameters:
        kwargs["assign"] = True
    module._load_from_state_dict(
        state, prefix, meta, True, missing, unexpected, errors, **kwargs
    )
    if errors:
        raise H3ComponentError(
            f"Quantized load failed for {prefix}: " + "; ".join(errors[:8])
        )
    for leaf, tensor in bag.items():  # 兜底：仍 meta 的 bias/weight 强制替换
        if leaf not in {"weight", "bias"}:
            continue
        cur = getattr(module, leaf, None)
        if cur is not None and getattr(cur, "device", None) is not None and cur.device.type == "meta":
            _assign_tensor(module, leaf, tensor.to(device=device))


def _default_rope(config, *, device):
    import torch

    axis_dim = int(config.rope_inv_freq_len) * 2
    return 1.0 / (
        10000.0
        ** (
            torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)
            / axis_dim
        )
    )


@dataclass
class H3ModelHandle:
    """Object passed through the custom ``H3_MODEL`` Comfy socket."""

    model: Any
    model_patcher: Any
    component_path: Path
    load_device: Any
    offload_device: Any
    dtype: Any
    metadata: dict
    checkpoint_files: tuple[Path, ...]
    quantized: bool = False

    @property
    def transformer(self):
        return self.model

    @property
    def patcher(self):
        return self.model_patcher

    def load_for_inference(self):
        if self.model_patcher is not None:
            import comfy.model_management as mm

            force_full = FORCE_FULL_LOAD_BF16 and not (
                self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8
            )
            import torch

            try:
                if force_full:
                    mm.load_models_gpu(
                        [self.model_patcher],
                        force_full_load=True,
                    )
                else:  # int8 partial：给采样激活留出 reserve，避免权重塞满后采样必 OOM
                    try:
                        mm.load_models_gpu([self.model_patcher], memory_required=DIT_INFERENCE_RESERVE)
                    except torch.OutOfMemoryError:  # 兜底：清卡后加倍预留重试一次
                        LOGGER.warning("DiT partial load OOM，清卡后以 2x reserve 重试")
                        mm.unload_all_models()
                        mm.soft_empty_cache()
                        mm.load_models_gpu([self.model_patcher], memory_required=2 * DIT_INFERENCE_RESERVE)
            except TypeError:
                mm.load_models_gpu([self.model_patcher])
        else:
            self.model.to(self.load_device)
        # Partial INT8 offload intentionally leaves some (often the first)
        # parameters on CPU.  Sampling activations must still be created on
        # ComfyUI's load device so mixed_precision_ops can stream/cast the
        # resident and offloaded weights against CUDA inputs.
        object.__setattr__(
            self.model,
            "_h3_compute_device",
            torch.device(self.load_device),
        )
        if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
            return self.model  # MixedPrecisionOps 允许 CPU/GPU 混驻
        misplaced: list[str] = []
        tensors = (
            *self.model.named_parameters(),
            *self.model.named_buffers(),
        )
        for name, tensor in tensors:
            if tensor.device != self.load_device:
                misplaced.append(f"{name}={tensor.device}")
                if len(misplaced) == 8:
                    break
        if misplaced:
            raise RuntimeError(
                "ComfyUI did not fully load the H3 DiT onto "
                f"{self.load_device}; first misplaced tensors: "
                + ", ".join(misplaced)
                + ". Update ComfyUI or use a memory mode/device with enough VRAM."
            )
        return self.model

    def offload_after_inference(self) -> None:
        if self.model_patcher is None:
            self.model.to(self.offload_device)


def _comfy_patcher(model, load_device, offload_device, size: int):
    try:
        from comfy.model_patcher import ModelPatcher
    except ImportError:
        return None
    return ModelPatcher(
        model,
        load_device=load_device,
        offload_device=offload_device,
        size=size,
    )


def load_h3_model(
    model_root: str,
    partition: str = "fl2va",
    *,
    transformer_path: str | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    offload_device: str = "auto",
    attention_backend: str = "sdpa",
    qkv_layout: str = "grouped",
) -> H3ModelHandle:
    """Construct on meta and stream H3 DiT shards to the offload device.

    自动识别 BF16 与 int8_convrot（``.comfy_quant`` / ``.weight_scale``）checkpoint。
    """

    import torch
    from safetensors import safe_open

    from .dit import (
        MiniMaxH3DiTConfig,
        MiniMaxH3DiTModel,
        prepare_checkpoint_tensor,
    )

    normalized_partition = str(partition).strip().lower()
    if normalized_partition not in {"fl2va", "ref2va"}:
        raise H3ComponentError("partition must be 'fl2va' or 'ref2va'")
    if normalized_partition != "fl2va":
        raise H3ComponentError(
            "The direct v0.2 sampler currently supports T2VA only; "
            "T2VA requires the FL2VA checkpoint partition"
        )

    metadata = release_metadata(model_root)
    validate_t2va_partition(metadata)
    declared_partition = metadata.get("partition") if metadata else None
    if declared_partition and declared_partition != normalized_partition:
        raise H3ComponentError(
            f"Requested {normalized_partition!r} weights, but model_index declares "
            f"{declared_partition!r}"
        )
    if not transformer_path:  # 未显式指定时自动优先 int8 量化目录，避免 BF16 整模上卡 OOM
        auto = model_root_path(model_root) / INT8_DIT_DIRNAME
        if auto.is_dir() and (auto / "config.json").is_file():
            transformer_path = str(auto)
            LOGGER.info("transformer_path 未指定，自动选用量化目录 %s", INT8_DIT_DIRNAME)
    component = resolve_component(
        model_root,
        ("transformer", "dit"),
        explicit=transformer_path,
        required_files=("config.json",),
    )
    config_path = component / "config.json"
    config_raw = read_json(config_path)
    _validate_transformer_config(config_raw, config_path)
    config = MiniMaxH3DiTConfig.from_dict(config_raw)
    model_dtype = _dtype(dtype)
    load_device, target_offload = _devices(device, offload_device)
    shards, weight_map = _checkpoint_index(component)
    quantized = _is_quantized_map(weight_map, shards)
    operations = _require_int8_ops(model_dtype) if quantized else None

    model = MiniMaxH3DiTModel.from_config(
        config,
        device=torch.device("meta"),
        dtype=model_dtype,
        attention_backend=attention_backend,
        operations=operations,
    )
    linears = _linear_modules(model)
    expected_tensors = {
        **dict(model.named_parameters(remove_duplicate=False)),
        **dict(model.named_buffers(remove_duplicate=False)),
    }
    expected = set(expected_tensors)
    needed: dict[str, set[str]] = defaultdict(set)
    linear_keys: set[str] = set()
    key_source = dict(weight_map or {})
    if not key_source:
        for shard in shards:
            with safe_open(str(shard), framework="pt") as reader:
                for ck in reader.keys():
                    key_source[ck] = shard.name
    for ck in key_source:
        for cand in _strip_outer(ck):
            matched = _match_linear(cand, linears)
            if matched:
                pref, leaf = matched
                needed[pref].add(leaf)
                linear_keys.add(f"{pref}{leaf}")
                break

    loaded: set[str] = set()
    unexpected: list[str] = []
    pending: dict[str, dict[str, Any]] = defaultdict(dict)

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for checkpoint_key in reader.keys():
                local = None
                matched = None
                for cand in _strip_outer(checkpoint_key):
                    matched = _match_linear(cand, linears)
                    if matched:
                        local = f"{matched[0]}{matched[1]}"
                        break
                    if cand in expected:
                        local = cand
                        break
                if local is None:
                    unexpected.append(checkpoint_key)
                    continue
                if local in loaded or (
                    matched and matched[1] in pending.get(matched[0], {})
                ):
                    raise H3ComponentError(
                        f"Duplicate checkpoint tensor {local!r} in {shard}"
                    )
                tensor = reader.get_tensor(checkpoint_key)
                tensor = prepare_checkpoint_tensor(
                    local,
                    tensor,
                    config=config,
                    qkv_layout=qkv_layout,
                )
                if matched is not None:
                    pref, leaf = matched
                    pending[pref][leaf] = tensor
                    if needed[pref] <= set(pending[pref]):
                        _flush_linear(
                            linears[pref],
                            pref,
                            pending.pop(pref),
                            target_offload,
                        )
                        loaded.update(f"{pref}{x}" for x in needed[pref])
                    continue
                target = expected_tensors[local]
                if tensor.is_floating_point() and tensor.dtype != target.dtype:
                    tensor = tensor.to(dtype=target.dtype)
                if tuple(tensor.shape) != tuple(target.shape):
                    raise H3ComponentError(
                        f"Shape mismatch for {local}: checkpoint "
                        f"{tuple(tensor.shape)} vs model {tuple(target.shape)}"
                    )
                tensor = tensor.to(device=target_offload)
                _assign_tensor(model, local, tensor)
                loaded.add(local)

    for pref, bag in list(pending.items()):
        _flush_linear(linears[pref], pref, bag, target_offload)
        loaded.update(f"{pref}{x}" for x in bag)

    if "rope.inv_freq" not in loaded and "rope.inv_freq" in expected:
        _assign_tensor(
            model,
            "rope.inv_freq",
            _default_rope(config, device=target_offload),
        )
        loaded.add("rope.inv_freq")

    missing = sorted(
        k
        for k in (expected | linear_keys) - loaded
        if not k.endswith(_AUX_QUANT_SUFFIXES)
    )
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:20]!r} (total {len(missing)})")
        if unexpected:
            details.append(
                f"unexpected={unexpected[:20]!r} (total {len(unexpected)})"
            )
        raise H3ComponentError(
            "H3 DiT checkpoint contract failed: " + "; ".join(details)
        )

    model.requires_grad_(False).eval()
    model.post_load_weights()
    size = _model_nbytes(model)
    patcher = _comfy_patcher(model, load_device, target_offload, size)
    return H3ModelHandle(
        model=model,
        model_patcher=patcher,
        component_path=component,
        load_device=load_device,
        offload_device=target_offload,
        dtype=model_dtype,
        metadata=metadata,
        checkpoint_files=tuple(shards),
        quantized=quantized,
    )


__all__ = ["H3ModelHandle", "load_h3_model"]
