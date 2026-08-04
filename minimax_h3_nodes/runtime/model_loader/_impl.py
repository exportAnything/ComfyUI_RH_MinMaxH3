"""Streaming local checkpoint loader for the direct H3 DiT."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..components import (
    H3ComponentError,
    read_json,
    release_metadata,
    resolve_component,
    resolve_partition_root,
    validate_task_partition,
    validate_weight_partition,
)
import logging

from ..h3_settings import (
    ADALN_CURVE_TABLE_KEY,
    ALLOW_PARTIAL_OFFLOAD_INT8,
    DIT_INFERENCE_RESERVE,
    ENABLE_DIT_LAYERWISE_OFFLOAD,
    FORCE_FULL_LOAD_BF16,
    INT8_DIT_DIRNAME,
    INT8_FORMAT,
    OPT_DYNAMIC_ACTIVATION_RESERVE,
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

# The public H3 releases all use this one DiT architecture.  Keep this gate in
# the loader (rather than MiniMaxH3DiTConfig itself) so tiny configurations
# remain useful for unit tests while an untrusted release config cannot make us
# allocate an arbitrarily large meta-module graph before checkpoint validation.
_OFFICIAL_H3_DIT_ARCHITECTURE = {
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "hidden_size": 5376,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 18 * 5376,
    "final_adaln_out_features": 2 * 5376,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}
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


def _config_architecture(raw: dict) -> dict:
    """Mirror MiniMaxH3DiTConfig.from_dict: unwrap the architecture mapping.

    Published configs may wrap the architecture while Diffusers-style configs
    keep these fields at top level.  Validate the exact mapping that will reach
    the constructor.
    """

    for wrapper_key in ("arch_config", "transformer_config", "dit_config"):
        nested = raw.get(wrapper_key)
        if isinstance(nested, dict):
            return nested
    return raw


def _validate_transformer_config(raw: dict, path: Path) -> None:
    """Validate the concrete Diffusers-style config shipped with H3.

    This stage runs before checkpoint discovery (cheap and allocation-free).
    ``time_embed_dim`` is the only deferred field: in the curve-table variant,
    adaLN input width is the basis rank rather than upstream 2688, and the correct
    value is known only after reading the checkpoint. See
    :func:`_validate_adaln_curve_contract`.
    """

    class_name = raw.get("_class_name")
    if class_name not in (None, "MiniMaxH3DiTModel"):
        raise H3ComponentError(
            f"{path} declares unsupported _class_name={class_name!r}; "
            "expected 'MiniMaxH3DiTModel'"
        )

    architecture = _config_architecture(raw)
    required = set(_TRANSFORMER_CONFIG_FIELDS)
    expected_architecture = {
        field: value
        for field, value in _OFFICIAL_H3_DIT_ARCHITECTURE.items()
        if field != "time_embed_dim"
    }
    if architecture.get("adaln_curve_grid") is not None:
        # These fields describe only the time embedder replaced by the sample table.
        # A curve-table config may omit them, but if present they are still validated
        # against upstream values to catch a wrong directory.
        for optional in ("timestep_input_dim", "time_embed_hidden_size"):
            required.discard(optional)
            if optional not in architecture:
                expected_architecture.pop(optional, None)

    missing = sorted(required - set(architecture))
    if missing:
        raise H3ComponentError(
            f"{path} is missing H3 transformer config fields: {missing!r}"
        )

    mismatches: list[str] = []
    for field, expected in expected_architecture.items():
        actual = architecture[field]
        if field == "patch_size":
            if (
                not isinstance(actual, (list, tuple))
                or any(type(value) is not int for value in actual)
                or tuple(actual) != expected
            ):
                mismatches.append(f"{field}={actual!r} (expected {expected!r})")
            continue
        if type(expected) is int:
            matches = type(actual) is int and actual == expected
        else:
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and float(actual) == expected
            )
        if not matches:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")

    if mismatches:
        raise H3ComponentError(
            f"{path} does not match the official MiniMax-H3 DiT architecture: "
            + "; ".join(mismatches)
        )


def _validate_adaln_curve_contract(
    raw: dict, path: Path, curve: tuple[int, int] | None
) -> None:
    """Cross-check the config's adaLN width against the checkpoint's actual form.

    ``curve`` is the measured checkpoint ``(grid, rank)``; None means the original monolithic form.
    """

    architecture = _config_architecture(raw)
    declared_dim = architecture["time_embed_dim"]
    declared_grid = architecture.get("adaln_curve_grid")
    official_dim = _OFFICIAL_H3_DIT_ARCHITECTURE["time_embed_dim"]

    if curve is None:
        if declared_grid is not None:
            raise H3ComponentError(
                f"{path} declares adaln_curve_grid, but the checkpoint has no "
                f"{ADALN_CURVE_TABLE_KEY}; confirm that the weights and config belong together"
            )
        if type(declared_dim) is not int or declared_dim != official_dim:
            raise H3ComponentError(
                f"{path} does not match the official MiniMax-H3 DiT "
                f"architecture: time_embed_dim={declared_dim!r} "
                f"(expected {official_dim!r})"
            )
        return

    grid, rank = curve
    if declared_grid is not None and int(declared_grid) != grid:
        raise H3ComponentError(
            f"{path}.adaln_curve_grid={declared_grid!r} does not match checkpoint "
            f"{ADALN_CURVE_TABLE_KEY} row count {grid}"
        )
    if type(declared_dim) is not int or declared_dim != rank:
        raise H3ComponentError(
            f"{path}.time_embed_dim={declared_dim!r} does not match curve-table checkpoint rank "
            f"{rank} (the curve-table variant's adaLN input width is the basis rank)"
        )


def _validate_quant_meta_partition(
    component: Path,
    partition: str,
    *,
    required: bool = False,
) -> dict:
    """Validate the converter manifest for an INT8/convrot DiT.

    A BF16 component legitimately has no ``quant_meta.json``.  Once checkpoint
    inspection identifies quantized tensors, however, this manifest is the
    partition proof and is mandatory.
    """

    path = component / "quant_meta.json"
    if not path.is_file():
        if required:
            raise H3ComponentError(
                f"INT8/convrot checkpoint requires {path} with format, "
                "convrot, and partition metadata"
            )
        return {}
    metadata = read_json(path)
    if metadata.get("format") != INT8_FORMAT:
        raise H3ComponentError(
            f"{path}.format must be {INT8_FORMAT!r} for an INT8 H3 DiT"
        )
    if metadata.get("convrot") is not True:
        raise H3ComponentError(
            f"{path}.convrot must be true for an INT8 H3 DiT"
        )
    declared = metadata.get("partition")
    if not isinstance(declared, str) or not declared.strip():
        raise H3ComponentError(
            f"{path}.partition must be a non-empty string for an INT8 H3 DiT"
        )
    normalized = declared.strip().lower()
    if normalized not in {"fl2va", "ref2va"}:
        raise H3ComponentError(
            f"{path}.partition must be 'FL2VA' or 'Ref2VA', got {declared!r}"
        )
    if normalized != partition:
        raise H3ComponentError(
            f"INT8 checkpoint partition mismatch: requested {partition!r}, "
            f"but {path} declares {declared!r}"
        )
    return metadata


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
    """Return (list of shard paths, weight_map or None)."""

    component = component.resolve()

    def contained_checkpoint(raw: str | Path, *, label: str) -> Path:
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise H3ComponentError(f"{label} must be a non-empty relative path")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise H3ComponentError(
                f"{label} must stay below {component}; got {str(raw)!r}"
            )
        candidate = (component / relative).resolve()
        try:
            candidate.relative_to(component)
        except ValueError as exc:
            raise H3ComponentError(
                f"{label} resolves outside checkpoint component {component}: "
                f"{candidate}"
            ) from exc
        return candidate

    index_candidates = (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
        "transformer.safetensors.index.json",
    )
    for name in index_candidates:
        path = contained_checkpoint(name, label="Checkpoint index")
        if not path.is_file():
            continue
        value = read_json(path)
        weight_map = value.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise H3ComponentError(f"{path} has no non-empty weight_map")
        files = sorted(
            {
                contained_checkpoint(
                    item,
                    label=f"Checkpoint shard referenced by {path}",
                )
                for item in weight_map.values()
            }
        )
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
        files = sorted(
            {
                contained_checkpoint(
                    path.relative_to(component),
                    label="Checkpoint shard",
                )
                for path in component.glob(pattern)
            }
        )
        if files:
            return files, None
    raise H3ComponentError(
        f"No safetensors checkpoint or shard index found below {component}"
    )


def _detect_adaln_curve(
    weight_map: dict[str, str] | None, shards: list[Path]
) -> tuple[int, int] | None:
    """Detect a curve-table checkpoint: return ``(grid, rank)`` or None for the original checkpoint.

    This must be decided before model construction: curve-table mode has no
    ``time_embedder`` and adds an ``adaln_t_table`` buffer, so the two forms have
    different state_dict key sets.
    """

    from safetensors import safe_open

    def _is_table(key: str) -> bool:
        return any(cand == ADALN_CURVE_TABLE_KEY for cand in _strip_outer(key))

    candidates: list[tuple[Path, str]] = []
    if weight_map:
        for key, shard_name in weight_map.items():
            if _is_table(key):
                shard = next((s for s in shards if s.name == shard_name), None)
                if shard is None:
                    raise H3ComponentError(
                        f"weight_map points {key} to unknown shard {shard_name!r}"
                    )
                candidates.append((shard, key))
    else:
        for shard in shards:
            with safe_open(str(shard), framework="pt") as reader:
                candidates.extend(
                    (shard, key) for key in reader.keys() if _is_table(key)
                )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise H3ComponentError(
            f"checkpoint contains multiple {ADALN_CURVE_TABLE_KEY} entries: "
            f"{[k for _, k in candidates]!r}"
        )
    shard, key = candidates[0]
    with safe_open(str(shard), framework="pt") as reader:
        shape = tuple(reader.get_slice(key).get_shape())
    if len(shape) != 2:
        raise H3ComponentError(
            f"{ADALN_CURVE_TABLE_KEY} must be [grid, rank]; got {shape}"
        )
    grid, rank = int(shape[0]), int(shape[1])
    if grid < 2 or rank < 1:
        raise H3ComponentError(
            f"{ADALN_CURVE_TABLE_KEY} has invalid shape: grid={grid}, rank={rank}"
        )
    return grid, rank


def _is_quantized_map(weight_map: dict[str, str] | None, shards: list[Path]) -> bool:
    if weight_map:
        return any(k.endswith(QUANT_KEY_SUFFIXES) for k in weight_map)
    from safetensors import safe_open

    for shard in shards:
        with safe_open(str(shard), framework="pt") as reader:
            if any(k.endswith(QUANT_KEY_SUFFIXES) for k in reader.keys()):
                return True
    return False


def _comfy_version_label() -> str:
    for mod_name in ("comfyui_version", "comfy"):
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", None)
            if ver: return str(ver)
        except ImportError: pass
    return ""


def _int8_ops_supported(cops) -> bool:
    """Check QUANT_ALGOS first, then fall back to source probing for older Comfy versions."""
    import inspect
    for mod_name in ("comfy.quant_ops", "comfy.ops"):
        try:
            mod = __import__(mod_name, fromlist=["QUANT_ALGOS"])
            algos = getattr(mod, "QUANT_ALGOS", None)
            if isinstance(algos, dict) and INT8_FORMAT in algos: return True
            if isinstance(algos, (set, list, tuple)) and INT8_FORMAT in algos: return True
        except ImportError: pass
    loader = getattr(cops, "_load_quantized_module", None)
    if not callable(loader): return False
    try: return INT8_FORMAT in inspect.getsource(loader)
    except (OSError, TypeError): return False


def _require_int8_ops(compute_dtype):
    """Build mixed_precision_ops; when unsupported, instruct the user to use BF16 rather than silently pretending to load."""
    try:
        import comfy.ops as cops
    except ImportError as exc:
        raise H3ComponentError(
            "INT8/convrot checkpoints require ComfyUI (comfy.ops); alternatively use BF16 DiT"
        ) from exc
    if not _int8_ops_supported(cops):
        ver = _comfy_version_label()
        raise H3ComponentError(
            f"Current ComfyUI{f' {ver}' if ver else ''} does not support {INT8_FORMAT}/convrot. "
            "Upgrade ComfyUI or use the BF16 transformer (automatic layerwise offload)."
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


def _device_free_bytes(device) -> int | None:  # Prefer Comfy, then torch, then None.
    try:
        import comfy.model_management as mm
        return int(mm.get_free_memory(device))
    except Exception:
        pass
    try:
        import torch
        d = torch.device(device)
        if d.type != "cuda" or not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(d)
        return int(free)
    except Exception:
        return None


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


def _quantized_linear_config(prefix: str, bag: dict[str, Any]) -> dict | None:
    """Validate and decode one Comfy ``int8_tensorwise`` marker."""

    import json
    import torch

    marker = bag.get("comfy_quant")
    quant_aux = {
        leaf for leaf in ("weight_scale", "weight_scale_2", "input_scale")
        if leaf in bag
    }
    weight = bag.get("weight")
    if marker is None:
        if quant_aux or (weight is not None and weight.dtype == torch.int8):
            raise H3ComponentError(
                f"Incomplete quantized checkpoint entry for {prefix}: "
                "int8/scale data is present without comfy_quant"
            )
        return None
    missing = sorted({"weight", "weight_scale"} - set(bag))
    if missing:
        raise H3ComponentError(
            f"Incomplete quantized checkpoint entry for {prefix}: missing={missing!r}"
        )
    if weight.dtype != torch.int8:
        raise H3ComponentError(
            f"Quantized checkpoint entry {prefix}weight must be int8, got {weight.dtype}"
        )
    try:
        config = json.loads(bytes(marker.detach().cpu().tolist()))
    except Exception as exc:
        raise H3ComponentError(
            f"Invalid comfy_quant marker for {prefix}"
        ) from exc
    if not isinstance(config, dict) or config.get("format") != INT8_FORMAT:
        raise H3ComponentError(
            f"Unsupported comfy_quant marker for {prefix}: {config!r}"
        )
    if config.get("convrot") is not True:
        raise H3ComponentError(
            f"H3 INT8 checkpoint requires convrot=true for {prefix}"
        )
    return config


def _validate_quantized_linear(module, prefix: str, bag: dict[str, Any]) -> None:
    """Reject the dangerous raw-int8 fallback after a quantized load."""

    import torch

    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError as exc:
        raise H3ComponentError(
            f"Cannot validate quantized Linear {prefix}: comfy.quant_ops is unavailable"
        ) from exc

    weight = getattr(module, "weight", None)
    if not isinstance(weight, QuantizedTensor):
        raise H3ComponentError(
            f"Quantized Linear {prefix} did not materialize as QuantizedTensor; "
            "refusing to use a raw int8 weight"
        )
    qdata = getattr(weight, "_qdata", None)
    params = getattr(weight, "_params", None)
    scale = getattr(params, "scale", None) if params is not None else None
    problems: list[str] = []
    if getattr(module, "quant_format", None) != INT8_FORMAT:
        problems.append(
            f"quant_format={getattr(module, 'quant_format', None)!r}"
        )
    if qdata is None or qdata.dtype != torch.int8:
        problems.append("missing int8 qdata")
    elif qdata.device.type == "meta":
        problems.append("qdata is still meta")
    elif tuple(qdata.shape) != tuple(bag["weight"].shape):
        problems.append(
            f"qdata shape {tuple(qdata.shape)} != {tuple(bag['weight'].shape)}"
        )
    if scale is None:
        problems.append("missing scale")
    elif scale.device.type == "meta":
        problems.append("scale is still meta")
    elif tuple(scale.shape) != tuple(bag["weight_scale"].shape):
        problems.append(
            f"scale shape {tuple(scale.shape)} != {tuple(bag['weight_scale'].shape)}"
        )
    if getattr(params, "convrot", None) is not True:
        problems.append("convrot is not enabled")
    if problems:
        raise H3ComponentError(
            f"Quantized Linear {prefix} materialization failed: " + "; ".join(problems)
        )


def _flush_linear(module, prefix: str, bag: dict[str, Any], device) -> bool:
    """Materialize one Linear bag; return whether it is a full quantized layer."""

    import inspect
    import torch
    import torch.nn as nn

    # QuantizedTensor materialization validates the encoded qdata against the
    # checkpoint bag, but it cannot infer the architecture's intended Linear
    # dimensions after replacement.  Check the original meta slots first so a
    # corrupt INT8 matrix cannot silently replace a differently-shaped layer.
    for leaf in ("weight", "bias"):
        tensor = bag.get(leaf)
        target = getattr(module, leaf, None)
        if tensor is not None and target is not None:
            if tuple(tensor.shape) != tuple(target.shape):
                raise H3ComponentError(
                    f"Shape mismatch for {prefix}{leaf}: checkpoint "
                    f"{tuple(tensor.shape)} vs model {tuple(target.shape)}"
                )

    quant_config = _quantized_linear_config(prefix, bag)
    if type(module) is nn.Linear:
        if quant_config is not None:
            raise H3ComponentError(
                f"Quantized checkpoint entry {prefix} was matched to plain nn.Linear"
            )
        for leaf, tensor in bag.items():
            if leaf not in {"weight", "bias"}:
                continue
            target = getattr(module, leaf)
            value = tensor
            if (
                target is not None
                and tensor.is_floating_point()
                and target.is_floating_point()
                and tensor.dtype != target.dtype
            ):
                value = tensor.to(dtype=target.dtype)
            _assign_tensor(module, leaf, value.to(device=device))
        return False

    # Comfy's quantized loader allocates qdata and scale on factory_kwargs.device.
    # The module was intentionally constructed on meta, so leaving this untouched
    # silently creates a meta QuantizedTensor.  The old raw-weight fallback then
    # discarded weight_scale/comfy_quant/convrot and produced snow output.
    factory_kwargs = getattr(module, "factory_kwargs", None)
    if isinstance(factory_kwargs, dict):
        factory_kwargs["device"] = torch.device(device)
    elif quant_config is not None:
        raise H3ComponentError(
            f"Quantized Linear {prefix} has no mutable factory_kwargs device"
        )

    state = {
        f"{prefix}{leaf}": tensor.to(
            device=torch.device("cpu") if leaf == "comfy_quant" else device
        )
        for leaf, tensor in bag.items()
    }
    missing: list[str] = []
    unexpected: list[str] = []
    errors: list[str] = []
    # Meta modules require assign; otherwise copy_ is a silent no-op
    # (comfy.ops recognizes assign_to_params_buffers).
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
    if quant_config is not None:
        _validate_quantized_linear(module, prefix, bag)
        return True
    for leaf, tensor in bag.items():  # Fallback: forcibly replace any bias/weight still on meta.
        if leaf not in {"weight", "bias"}:
            continue
        cur = getattr(module, leaf, None)
        if cur is not None and getattr(cur, "device", None) is not None and cur.device.type == "meta":
            _assign_tensor(module, leaf, tensor.to(device=device))
    return False


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


def _nonstreamable_tensor_slots(model):
    """Yield direct tensor slots that Comfy cannot stream through cast ops."""

    for module_name, module in model.named_modules():
        if hasattr(module, "comfy_cast_weights"):
            continue
        for collection_name in ("_parameters", "_buffers"):
            collection = getattr(module, collection_name, {})
            for tensor_name, tensor in tuple(collection.items()):
                if tensor is None:
                    continue
                name = f"{module_name}.{tensor_name}" if module_name else tensor_name
                yield name, module, collection_name, tensor_name, tensor


def _nonstreamable_tensor_items(model):
    """Yield unique tensors that Comfy cannot stream through cast-weight ops."""

    seen: set[int] = set()
    for name, _module, _collection, _tensor_name, tensor in (
        _nonstreamable_tensor_slots(model)
    ):
        tensor_id = id(tensor)
        if tensor_id in seen:
            continue
        seen.add(tensor_id)
        yield name, tensor


def _move_nonstreamable_tensors(model, device) -> None:
    """Move native tensors without recursively touching streamable Linears."""

    import torch

    target = torch.device(device)
    moved: dict[int, Any] = {}
    for _name, module, collection_name, tensor_name, tensor in (
        _nonstreamable_tensor_slots(model)
    ):
        tensor_id = id(tensor)
        replacement = moved.get(tensor_id)
        if replacement is None:
            if tensor.device == target:
                replacement = tensor
            else:
                value = tensor.to(device=target)
                if collection_name == "_parameters":
                    replacement = torch.nn.Parameter(
                        value,
                        requires_grad=bool(tensor.requires_grad),
                    )
                else:
                    replacement = value
            moved[tensor_id] = replacement
        getattr(module, collection_name)[tensor_name] = replacement


def _nonstreamable_tensor_bytes(model) -> int:
    return sum(_tensor_nbytes(tensor) for _name, tensor in _nonstreamable_tensor_items(model))


def _misplaced_nonstreamable_tensors(model, device, *, limit: int = 8) -> list[str]:
    import torch

    target = torch.device(device)
    misplaced: list[str] = []
    for name, tensor in _nonstreamable_tensor_items(model):
        if tensor.device != target:
            misplaced.append(f"{name}={tensor.device}")
            if len(misplaced) >= int(limit):
                break
    return misplaced


def _unload_model_patcher(patcher) -> None:
    """Release a Comfy ModelPatcher, detaching if manager cleanup is unavailable."""

    manager_error: Exception | None = None
    try:
        import comfy.model_management as mm

        unload = getattr(mm, "unload_model_and_clones", None)
        if callable(unload):
            unload(patcher)
        else:
            manager_error = RuntimeError(
                "comfy.model_management.unload_model_and_clones is unavailable"
            )
    except Exception as exc:
        manager_error = exc

    loaded_size = getattr(patcher, "loaded_size", lambda: 0)()
    bank = getattr(patcher, "model", None)
    still_loaded = bool(
        loaded_size
        or getattr(bank, "model_lowvram", False)
        or getattr(patcher, "pinned", ())
    )
    if manager_error is not None or still_loaded:
        detach = getattr(patcher, "detach", None)
        if not callable(detach):
            if manager_error is not None:
                raise manager_error
            raise RuntimeError("H3 ModelPatcher remains loaded and cannot detach")
        detach()
        if manager_error is not None:
            LOGGER.warning(
                "Comfy ModelPatcher manager unload failed; detached directly: %s",
                manager_error,
            )


def _clear_compute_device_marker(model) -> None:
    """Remove the session marker safely even when cleanup runs more than once."""

    try:
        delattr(model, "_h3_compute_device")
    except AttributeError:
        pass


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
    activation_hint: dict | None = None  # P0-6 canvas/sequence hint.
    residency_mode: str = "unknown"  # full | layerwise | partial | reject

    @property
    def transformer(self):
        return self.model

    @property
    def patcher(self):
        return self.model_patcher

    def set_activation_hint(self, **hint: Any) -> None:
        self.activation_hint = dict(hint)

    def _activation_reserve_bytes(self) -> int:
        if not OPT_DYNAMIC_ACTIVATION_RESERVE or not self.activation_hint:
            return int(DIT_INFERENCE_RESERVE)
        from ..vram_budget import resolve_activation_reserve
        h = self.activation_hint
        return resolve_activation_reserve(
            seq_len=int(h.get("seq_len", 0) or 0),
            video_latent_t=int(h.get("video_latent_t", 0) or 0),
            video_latent_h=int(h.get("video_latent_h", 0) or 0),
            video_latent_w=int(h.get("video_latent_w", 0) or 0),
            ref_rows=int(h.get("ref_rows", 0) or 0),
            task=str(h.get("task", "") or ""),
            dtype="int8" if self.quantized else "bf16",
        )

    def _full_reside_budget_bytes(self) -> int:  # Full-model residency requirement: weights + activation reserve.
        return int(_model_nbytes(self.model)) + int(self._activation_reserve_bytes())

    def _decide_residency(self) -> str:
        from ..vram_budget import residency_tier
        free = _device_free_bytes(self.load_device)
        weights = int(_model_nbytes(self.model))
        act = int(self._activation_reserve_bytes())
        tier = residency_tier(free_bytes=free, weight_bytes=weights, activation_bytes=act)
        LOGGER.info(
            "DiT residency tier=%s free=%s weights=%s activation=%s quantized=%s",
            tier, free, weights, act, self.quantized,
        )
        return tier

    def _vram_can_full_reside(self) -> bool:
        return self._decide_residency() == "full"

    def _use_layerwise(self) -> bool:
        device = str(self.load_device)
        if not (
            ENABLE_DIT_LAYERWISE_OFFLOAD
            and not self.quantized
            and device.startswith("cuda")
        ):
            return False
        tier = self._decide_residency()
        if tier == "reject":
            from ..downscale import format_downscale_hint
            hint = self.activation_hint or {}
            w = int(hint.get("width", 1344) or 1344); h = int(hint.get("height", 768) or 768)
            raise RuntimeError(
                "Insufficient VRAM for layerwise BF16 DiT (one layer plus activations); "
                f"reduce the canvas (suggested {format_downscale_hint(w, h)}) or use INT8"
            )
        return tier != "full"  # Disable automatically when the full model fits.

    def park_after_inference(self) -> str:
        """P1-1 soft residency: layerwise→layerwise-warm; full model→gpu-resident."""
        from ..layerwise_offload import get_layerwise_offload
        mgr = get_layerwise_offload(self.model)
        if mgr is not None and getattr(mgr, "_enabled", False):
            mgr.disable()  # Leave blocks in pinned CPU memory.
            self.residency_mode = "layerwise-warm"
            object.__setattr__(self.model, "_h3_residency_mode", "layerwise-warm")
            _clear_compute_device_marker(self.model)
            return "layerwise-warm"
        if getattr(self.model, "_h3_compute_device", None) is not None:
            mode = "partial-warm" if self.quantized else "gpu-resident"
            self.residency_mode = mode
            object.__setattr__(self.model, "_h3_residency_mode", mode)
            return mode
        self.offload_after_inference()
        return "cold"

    def load_for_inference(self):
        import torch

        try:
            warm = getattr(self.model, "_h3_residency_mode", None)
            if warm == "gpu-resident" and getattr(self.model, "_h3_compute_device", None) is not None:
                self.residency_mode = "gpu-resident"
                LOGGER.info("DiT residency warm hit: gpu-resident")
                return self.model

            if self._use_layerwise():  # BF16: prefetch by layer to avoid full-model force_full_load.
                if self.model_patcher is not None:
                    try:
                        _unload_model_patcher(self.model_patcher)
                    except Exception as exc:
                        LOGGER.warning("Failed to unload ModelPatcher before enabling layerwise mode: %s", exc)
                from ..layerwise_offload import attach_layerwise_offload

                attach_layerwise_offload(self.model, device=self.load_device).enable()
                self.residency_mode = "layerwise"
                object.__setattr__(
                    self.model, "_h3_compute_device", torch.device(self.load_device)
                )
                object.__setattr__(self.model, "_h3_residency_mode", "layerwise")
                return self.model

            if self.model_patcher is not None:
                import comfy.model_management as mm

                force_full = FORCE_FULL_LOAD_BF16 and not (
                    self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8
                )
                nonstreamable_bytes = (
                    _nonstreamable_tensor_bytes(self.model)
                    if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8
                    else 0
                )
                try:
                    if force_full:
                        mm.load_models_gpu(
                            [self.model_patcher],
                            force_full_load=True,
                        )
                        self.residency_mode = "full"
                    else:  # INT8 partial: reserve space for sampling activations and nonstreamable layers.
                        act = self._activation_reserve_bytes()
                        reserve = act + nonstreamable_bytes
                        try:
                            mm.load_models_gpu(
                                [self.model_patcher],
                                memory_required=reserve,
                            )
                        except torch.OutOfMemoryError:  # Clear unrelated models, then retry with 2× dynamic activation reserve.
                            LOGGER.warning(
                                "DiT partial load OOM; cleared the device and retrying with 2× activation_reserve"
                            )
                            mm.unload_all_models()
                            mm.soft_empty_cache()
                            mm.load_models_gpu(
                                [self.model_patcher],
                                memory_required=(2 * act + nonstreamable_bytes),
                            )
                        self.residency_mode = "partial"
                except TypeError:
                    mm.load_models_gpu([self.model_patcher])
                    if not self.residency_mode or self.residency_mode == "unknown":
                        self.residency_mode = "partial" if self.quantized else "full"

                if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
                    # ModelPatcher can stream comfy-cast Linears, but native
                    # FP32 projections, norms and RoPE buffers must be placed
                    # explicitly. Their bytes were reserved above so this move
                    # preserves DIT_INFERENCE_RESERVE for sampler activations.
                    _move_nonstreamable_tensors(self.model, self.load_device)
                    misplaced_static = _misplaced_nonstreamable_tensors(
                        self.model,
                        self.load_device,
                    )
                    if misplaced_static:
                        raise RuntimeError(
                            "ComfyUI left non-streamable H3 DiT tensors off the "
                            f"compute device {self.load_device}: "
                            + ", ".join(misplaced_static)
                        )
            else:
                self.model.to(self.load_device)
                self.residency_mode = "full"

            # Partial INT8 offload intentionally leaves streamable weights on
            # CPU. Activations still belong on Comfy's load device.
            object.__setattr__(
                self.model,
                "_h3_compute_device",
                torch.device(self.load_device),
            )
            object.__setattr__(self.model, "_h3_residency_mode", self.residency_mode)
            if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
                return self.model

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
        except Exception:
            from ..layerwise_offload import get_layerwise_offload

            mgr = get_layerwise_offload(self.model)
            if mgr is not None:
                try:
                    mgr.disable()
                except Exception as cleanup_exc:
                    LOGGER.error("layerwise disable after load error: %s", cleanup_exc)
            _clear_compute_device_marker(self.model)
            if self.model_patcher is not None:
                try:
                    _unload_model_patcher(self.model_patcher)
                except Exception as cleanup_exc:
                    LOGGER.error(
                        "Failed to clean up H3 ModelPatcher after load error: %s",
                        cleanup_exc,
                    )
                if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
                    try:
                        _move_nonstreamable_tensors(
                            self.model,
                            self.offload_device,
                        )
                    except Exception as cleanup_exc:
                        LOGGER.error(
                            "Failed to offload non-streamable H3 DiT tensors "
                            "after load error: %s",
                            cleanup_exc,
                        )
            else:
                try:
                    self.model.to(self.offload_device)
                except Exception as cleanup_exc:
                    LOGGER.error(
                        "Failed to offload raw H3 DiT after load error: %s",
                        cleanup_exc,
                    )
            raise

    def offload_after_inference(self) -> None:
        from ..layerwise_offload import get_layerwise_offload

        cleanup_error: Exception | None = None
        try:
            mgr = get_layerwise_offload(self.model)
            if mgr is not None and getattr(mgr, "_enabled", False):
                try:
                    mgr.disable()
                except Exception as exc:
                    cleanup_error = exc
            elif self.model_patcher is None:
                self.model.to(self.offload_device)
            else:
                try:
                    _unload_model_patcher(self.model_patcher)
                except Exception as exc:
                    cleanup_error = exc
                if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
                    try:
                        _move_nonstreamable_tensors(
                            self.model,
                            self.offload_device,
                        )
                    except Exception as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
                        else:
                            LOGGER.exception(
                                "Additional non-streamable H3 DiT offload failure"
                            )
        finally:
            _clear_compute_device_marker(self.model)
        if cleanup_error is not None:
            raise cleanup_error


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
    task: str = "t2va",
    transformer_path: str | None = None,
    transformer_weights: str | Path | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    offload_device: str = "auto",
    attention_backend: str = "sdpa",
    qkv_layout: str = "grouped",
) -> H3ModelHandle:
    """Construct on meta and stream H3 DiT shards to the offload device.

    Automatically detect BF16 and int8_convrot (``.comfy_quant`` /
    ``.weight_scale``) checkpoints. ``transformer_weights`` points to flat
    single-file weights under ``models/MiniMax-H3``; structure and configuration
    still come from the ``transformer_path`` component directory, while only
    tensors come from that file.
    """

    normalized_partition = str(partition).strip().lower()
    if normalized_partition not in {"fl2va", "ref2va"}:
        raise H3ComponentError("partition must be 'fl2va' or 'ref2va'")

    # Resolve the requested release child first.  All explicit components are
    # subsequently constrained to this concrete partition root.
    root = resolve_partition_root(model_root, normalized_partition)
    metadata = release_metadata(root)
    validate_task_partition(metadata, task, normalized_partition)
    if not transformer_path:  # Prefer the int8 directory when unspecified to avoid moving the full BF16 model to GPU and OOM.
        auto = root / INT8_DIT_DIRNAME
        if auto.is_dir() and (auto / "config.json").is_file():
            transformer_path = str(auto)
            LOGGER.info("transformer_path not specified; automatically selected quantized directory %s", INT8_DIT_DIRNAME)
    component = resolve_component(
        root,
        ("transformer", "dit"),
        explicit=transformer_path,
        required_files=("config.json",),
    )
    quant_metadata = _validate_quant_meta_partition(
        component, normalized_partition
    )
    config_path = component / "config.json"
    config_raw = read_json(config_path)
    _validate_transformer_config(config_raw, config_path)

    # Curve-table detection must happen before model construction: this variant has
    # no time_embedder and adds an adaln_t_table buffer, so the two forms have
    # different state_dict key sets. Shard discovery is likewise cheap and allocation-free.
    external_weights = (
        validate_weight_partition(
            transformer_weights, normalized_partition, kind="transformer"
        )
        if transformer_weights
        else None
    )
    if external_weights is not None:
        if not external_weights.is_file():
            raise H3ComponentError(
                f"MiniMax-H3 DiT weight file not found: {external_weights}"
            )
        shards, weight_map = [external_weights], None
        LOGGER.info(
            "DiT weights come from flat single file %s; structure/configuration still come from %s",
            external_weights.name,
            component,
        )
    else:
        shards, weight_map = _checkpoint_index(component)
    curve = _detect_adaln_curve(weight_map, shards)
    _validate_adaln_curve_contract(config_raw, config_path, curve)

    # Imports stay below all cheap partition checks so a mixed FL2VA/Ref2VA
    # release fails before model allocation/materialization.
    import torch
    from safetensors import safe_open

    from ..dit import (
        MiniMaxH3DiTConfig,
        MiniMaxH3DiTModel,
        prepare_checkpoint_tensor,
    )

    config = MiniMaxH3DiTConfig.from_dict(config_raw)
    if curve is not None:
        grid, rank = curve
        config = replace(config, adaln_curve_grid=grid, time_embed_dim=rank)
        LOGGER.info(
            "H3 DiT curve-table checkpoint: %d-dimensional adaLN input (grid=%d), no time embedder",
            rank, grid,
        )
    model_dtype = _dtype(dtype)
    load_device, target_offload = _devices(device, offload_device)
    quantized = _is_quantized_map(weight_map, shards)
    if quantized:
        # Flat single-file weights have no quant_meta.json; the filename is the
        # partition evidence and was already checked exactly in validate_weight_partition.
        if not quant_metadata and external_weights is None:
            quant_metadata = _validate_quant_meta_partition(
                component,
                normalized_partition,
                required=True,
            )
    elif quant_metadata:
        raise H3ComponentError(
            f"{component / 'quant_meta.json'} declares INT8/convrot, but the "
            "checkpoint contains no quantized tensor markers"
        )
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
    expected_quantized = sum(
        "comfy_quant" in leaves for leaves in needed.values()
    )
    loaded_quantized = 0

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
                        loaded_quantized += int(_flush_linear(
                            linears[pref],
                            pref,
                            pending.pop(pref),
                            target_offload,
                        ))
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
        loaded_quantized += int(
            _flush_linear(linears[pref], pref, bag, target_offload)
        )
        loaded.update(f"{pref}{x}" for x in bag)

    if quantized:
        if expected_quantized <= 0 or loaded_quantized != expected_quantized:
            raise H3ComponentError(
                "H3 DiT quantized Linear contract failed: "
                f"materialized={loaded_quantized}, expected={expected_quantized}"
            )
        LOGGER.info(
            "H3 DiT materialized %d complete INT8/convrot QuantizedTensor layers",
            loaded_quantized,
        )

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
