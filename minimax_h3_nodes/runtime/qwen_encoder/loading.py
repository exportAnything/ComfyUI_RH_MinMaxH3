from __future__ import annotations
from .helpers import *  # noqa: F403

def _qwen_causal_lm_class():
    """Return the class matching the released checkpoint key namespace.

    The H3 snapshot advertises ``Qwen3VLForConditionalGeneration`` and stores
    weights under ``model.visual.*`` / ``model.language_model.*`` plus
    ``lm_head.weight``.  Loading that snapshot directly into the bare
    ``Qwen3VLModel`` is version-dependent because the bare class uses a
    different base-model prefix.  Load the declared wrapper, then retain its
    ``.model`` backbone.
    """

    def _parse_ver(text: str) -> tuple[int, ...]:
        parts = []
        for piece in str(text).split("+")[0].split(".")[:3]:
            digits = "".join(ch for ch in piece if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts + [0] * (3 - len(parts)))

    try:
        import transformers
        ver = getattr(transformers, "__version__", "0")
        cur, lo, hi = _parse_ver(ver), _parse_ver(TRANSFORMERS_MIN_VERSION), _parse_ver(TRANSFORMERS_MAX_VERSION)
        if cur < lo or cur > hi:
            raise H3ComponentError(
                f"transformers {ver} is outside the supported range "
                f"[{TRANSFORMERS_MIN_VERSION}, {TRANSFORMERS_MAX_VERSION}]; "
                "use Transformers <=5.8.1 for the legacy loader, or select the native NVFP4 checkpoint"
            )
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    except H3ComponentError:
        raise
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

def _component_is_quantized(
    component: Path,
    *,
    shards: Sequence[Path] | None = None,
    weight_map: dict[str, str] | None = None,
) -> bool:
    from ..model_loader import _checkpoint_index
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


_NATIVE_NVFP4_FORMAT = "nvfp4"
_NATIVE_LANGUAGE_PREFIX = "model.layers."
_NATIVE_EMBED_PREFIX = "model.embed_tokens."
_NATIVE_VISUAL_SIGNAL = "visual.deepstack_merger_list.0.norm.weight"
_NATIVE_NVFP4_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _decode_comfy_quant_marker(reader: Any, key: str) -> dict[str, Any]:
    """Decode one tiny Comfy quant marker without materialising its weight."""

    import json

    try:
        marker = reader.get_tensor(key)
        value = json.loads(bytes(marker.detach().cpu().tolist()))
    except Exception as exc:
        raise H3ComponentError(
            f"Invalid comfy_quant marker {key!r} in a Qwen checkpoint"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("format"), str):
        raise H3ComponentError(
            f"Invalid comfy_quant marker {key!r}: expected an object with format"
        )
    return value


def _text_encoder_checkpoint_profile(shards: Sequence[Path]) -> dict[str, Any]:
    """Inspect keys and tiny marker tensors only; never read model weight payloads.

    The native Comfy MiniMax checkpoint uses ``model.layers.*`` for the
    truncated language model and ``visual.*`` for Qwen3-VL.  The older H3
    runtime path consumes Hugging Face ``model.language_model.*`` checkpoints,
    so the key namespace is also the safest dispatch signal.
    """

    from safetensors import safe_open

    keys: set[str] = set()
    formats: dict[str, int] = defaultdict(int)
    marker_configs: dict[str, dict[str, Any]] = {}
    pre_quant_scale_count = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in keys:
                    raise H3ComponentError(
                        f"Duplicate Qwen checkpoint tensor {key!r} across shards"
                    )
                keys.add(key)
                if key.endswith(".pre_quant_scale"):
                    pre_quant_scale_count += 1
                if key.endswith(".comfy_quant"):
                    config = _decode_comfy_quant_marker(reader, key)
                    marker_configs[key] = config
                    formats[str(config["format"])] += 1

    layer_marker_counts: dict[int, int] = defaultdict(int)
    for key, config in marker_configs.items():
        if config.get("format") != _NATIVE_NVFP4_FORMAT:
            continue
        match = re.match(r"^model\.layers\.(\d+)\..+\.comfy_quant$", key)
        if match:
            layer_marker_counts[int(match.group(1))] += 1

    return {
        "keys": frozenset(keys),
        "formats": dict(formats),
        "marker_configs": marker_configs,
        "layer_marker_counts": dict(layer_marker_counts),
        "pre_quant_scale_count": pre_quant_scale_count,
    }


def _validate_native_nvfp4_profile(
    profile: dict[str, Any], *, checkpoint: Path
) -> dict[str, Any]:
    """Fail closed unless a file is the native truncated MiniMax Qwen3-VL."""

    keys = profile["keys"]
    formats = profile["formats"]
    unsupported = sorted(set(formats) - {_NATIVE_NVFP4_FORMAT, INT8_FORMAT})
    if unsupported:
        raise H3ComponentError(
            f"{checkpoint.name} mixes unsupported Qwen quant formats {unsupported!r}"
        )
    if formats.get(_NATIVE_NVFP4_FORMAT) != SELECTED_LAYERS * 7:
        raise H3ComponentError(
            f"{checkpoint.name} must contain exactly {SELECTED_LAYERS * 7} "
            f"NVFP4 language Linears, got {formats.get(_NATIVE_NVFP4_FORMAT, 0)}"
        )
    expected_layers = set(range(SELECTED_LAYERS))
    layer_counts = profile["layer_marker_counts"]
    if set(layer_counts) != expected_layers or any(
        layer_counts[index] != 7 for index in expected_layers
    ):
        raise H3ComponentError(
            f"{checkpoint.name} does not contain seven NVFP4 projections in "
            f"each Qwen language layer 0..{SELECTED_LAYERS - 1}"
        )
    expected_markers = {
        f"{_NATIVE_LANGUAGE_PREFIX}{layer}.{projection}.comfy_quant"
        for layer in expected_layers
        for projection in _NATIVE_NVFP4_PROJECTIONS
    }
    actual_markers = {
        key
        for key, config in profile["marker_configs"].items()
        if config.get("format") == _NATIVE_NVFP4_FORMAT
    }
    if actual_markers != expected_markers:
        raise H3ComponentError(
            f"{checkpoint.name} has an invalid NVFP4 projection set: "
            f"missing={sorted(expected_markers - actual_markers)[:8]!r}, "
            f"unexpected={sorted(actual_markers - expected_markers)[:8]!r}"
        )
    incomplete = []
    for marker in expected_markers:
        prefix = marker[: -len("comfy_quant")]
        missing_leaves = [
            leaf
            for leaf in ("weight", "weight_scale", "weight_scale_2")
            if f"{prefix}{leaf}" not in keys
        ]
        if missing_leaves:
            incomplete.append((prefix, missing_leaves))
    if incomplete:
        raise H3ComponentError(
            f"{checkpoint.name} contains incomplete NVFP4 projections: "
            f"{incomplete[:8]!r}"
        )
    required = {
        _NATIVE_VISUAL_SIGNAL,
        f"{_NATIVE_EMBED_PREFIX}weight",
        f"{_NATIVE_EMBED_PREFIX}weight_scale",
        f"{_NATIVE_EMBED_PREFIX}comfy_quant",
        f"{_NATIVE_LANGUAGE_PREFIX}{SELECTED_LAYERS - 1}.self_attn.q_proj.weight",
    }
    missing = sorted(required - keys)
    if missing:
        raise H3ComponentError(
            f"{checkpoint.name} is missing native MiniMax Qwen3-VL tensors "
            f"{missing!r}"
        )
    embed_config = profile["marker_configs"].get(
        f"{_NATIVE_EMBED_PREFIX}comfy_quant", {}
    )
    if embed_config.get("format") != INT8_FORMAT:
        raise H3ComponentError(
            f"{checkpoint.name} must use {INT8_FORMAT!r} for model.embed_tokens"
        )
    if formats.get(INT8_FORMAT) != 1:
        raise H3ComponentError(
            f"{checkpoint.name} must contain only the one INT8 embedding marker, "
            f"got {formats.get(INT8_FORMAT, 0)}"
        )
    return profile


def _native_nvfp4_checkpoint_profile(
    shards: Sequence[Path], *, checkpoint: Path
) -> dict[str, Any] | None:
    """Return a validated native profile, or ``None`` for legacy/BF16 files."""

    profile = _text_encoder_checkpoint_profile(shards)
    if _NATIVE_NVFP4_FORMAT not in profile["formats"]:
        return None
    return _validate_native_nvfp4_profile(profile, checkpoint=checkpoint)


def _require_native_minimax_comfy(profile: dict[str, Any]) -> None:
    """Gate the native path before Comfy attempts to allocate the 32B model."""

    try:
        import comfy.quant_ops as quant_ops
        import comfy.sd as comfy_sd
        import comfy.text_encoders.minimax  # noqa: F401
    except ImportError as exc:
        raise H3ComponentError(
            "NVFP4/AWQ MiniMax-H3 text encoders require a current ComfyUI "
            "with native comfy.text_encoders.minimax support"
        ) from exc

    clip_type = getattr(getattr(comfy_sd, "CLIPType", None), "MINIMAX", None)
    if clip_type is None:
        raise H3ComponentError(
            "This ComfyUI build has no CLIPType.MINIMAX; update ComfyUI before "
            "loading the NVFP4/AWQ Qwen3-VL encoder"
        )
    algos = getattr(quant_ops, "QUANT_ALGOS", {})
    nvfp4 = algos.get(_NATIVE_NVFP4_FORMAT) if isinstance(algos, dict) else None
    parameters = nvfp4.get("parameters", set()) if isinstance(nvfp4, dict) else set()
    required = {"weight_scale", "weight_scale_2"}
    if profile.get("pre_quant_scale_count"):
        required.add("pre_quant_scale")
    missing = sorted(required - set(parameters))
    if missing:
        raise H3ComponentError(
            "This ComfyUI build cannot load the NVFP4/AWQ parameter set; "
            f"missing quant parameters {missing!r}. Update ComfyUI."
        )


def _load_native_minimax_text_encoder(
    weights: Path,
    *,
    component: Path,
    model_dtype: Any,
    load_device: Any,
    offload_device: Any,
    profile: dict[str, Any],
):
    """Load Comfy's native MiniMax CLIP and expose the H3 runtime adapter."""

    _require_native_minimax_comfy(profile)
    import comfy.sd as comfy_sd

    clip_type = comfy_sd.CLIPType.MINIMAX
    model_options = {
        "dtype": model_dtype,
        "load_device": load_device,
        "offload_device": offload_device,
    }
    clip = comfy_sd.load_clip(
        [str(weights)],
        clip_type=clip_type,
        model_options=model_options,
    )
    if not callable(getattr(clip, "tokenize", None)) or not callable(
        getattr(clip, "encode_from_tokens", None)
    ):
        raise H3ComponentError(
            "ComfyUI's native MiniMax text encoder returned an invalid CLIP handle"
        )

    from .encoder import MiniMaxH3ComfyTextEncoder

    return MiniMaxH3ComfyTextEncoder(
        clip=clip,
        component_path=component,
        weights_path=weights,
        quant_format=_NATIVE_NVFP4_FORMAT,
    )

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
    """Replace nn.Linear inside language_model.layers with comfy ops.Linear (meta)."""
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

    from ..model_loader import _flush_linear, _quantized_linear_config

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
    shards: list[Path] | None = None,
    weight_map: dict[str, str] | None = None,
) -> None:
    """Stream int8_convrot values while passing through bf16 (visual/embed/norm)."""
    from safetensors import safe_open
    from ..model_loader import _checkpoint_index  # Reuse the shard index.

    # All index and shard paths pass through the same canonical containment
    # gate as the DiT loader.  Never fall back after a malformed/escaping index.
    if shards is None:
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
        raise H3ComponentError(f"int8 text_encoder still contains meta tensors: {meta_left[:12]!r}")

def _load_quantized_text_encoder(
    component: Path,
    *,
    config,
    model_dtype,
    offload_device,
    attention_backend: str,
    quant_metadata: dict[str, Any],
    shards: list[Path] | None = None,
    weight_map: dict[str, str] | None = None,
):
    """Build on meta → replace language Linear with comfy ops → stream int8_convrot."""
    import torch
    from ..model_loader import _require_int8_ops

    model_cls = _qwen_causal_lm_class()
    ops = _require_int8_ops(model_dtype)
    try:
        from accelerate import init_empty_weights
    except ImportError as exc:
        raise H3ComponentError("int8 text_encoder requires accelerate (init_empty_weights)") from exc

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
        shards=shards,
        weight_map=weight_map,
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
    text_encoder_weights: str | Path | None = None,
    tokenizer_path: str | None = None,
    processor_path: str | None = None,
    dtype: str = "bfloat16",
    device: str = "auto",
    offload_device: str = "cpu",
    attention_backend: str = "sdpa",
) -> MiniMaxH3TextEncoder:
    """Load the local Qwen3-VL component with no Hub or remote-code fallback."""

    import torch

    # Native Comfy MiniMax checkpoints have a content-proven architecture/type
    # but deliberately use Comfy's regular text_encoders filename convention.
    # Inspect only their safetensors header and tiny comfy_quant markers before
    # applying the older converter-filename admission rule.
    external_weights: Path | None = None
    native_profile: dict[str, Any] | None = None
    if text_encoder_weights:
        candidate = Path(text_encoder_weights).expanduser().resolve()
        if not candidate.is_file():
            raise H3ComponentError(
                f"MiniMax-H3 text_encoder weight file not found: {candidate}"
            )
        native_profile = _native_nvfp4_checkpoint_profile(
            [candidate], checkpoint=candidate
        )
        if native_profile is not None:
            # The validated key/quant contract proves this is the shared,
            # layer-50 MiniMax Qwen3-VL encoder; no partition-bearing filename
            # is necessary for a component shared by FL2VA and Ref2VA.
            external_weights = candidate
        else:
            external_weights = validate_weight_partition(
                candidate, partition or "fl2va", kind="text_encoder"
            )
    if not text_encoder_path:  # Prefer the int8 quantized directory when unspecified (26GB vs BF16 62GB).
        auto = model_root_path(model_root) / INT8_TE_DIRNAME
        if auto.is_dir() and (auto / "config.json").is_file():
            text_encoder_path = str(auto)
            LOGGER.info("text_encoder_path not specified; automatically selected quantized directory %s", INT8_TE_DIRNAME)
    try:
        component = resolve_component(
            model_root,
            ("text_encoder", "qwen3vl", "qwen"),
            explicit=text_encoder_path,
            required_files=("config.json",),
        )
    except H3ComponentError:
        if native_profile is None or external_weights is None:
            raise
        # The native Comfy architecture and tokenizer are code-defined and do
        # not need a Transformers config directory.  Preserve a stable
        # component identity for callers that only have the standalone file.
        component = external_weights

    model_dtype = _torch_dtype(dtype)
    load_device = _resolve_device(device)
    target_offload_device = torch.device(offload_device)
    if native_profile is not None:
        assert external_weights is not None
        LOGGER.info(
            "Loading native MiniMax Qwen3-VL %s checkpoint %s (%d AWQ pre-scales)",
            _NATIVE_NVFP4_FORMAT,
            external_weights,
            int(native_profile.get("pre_quant_scale_count", 0)),
        )
        return _load_native_minimax_text_encoder(
            external_weights,
            component=component,
            model_dtype=model_dtype,
            load_device=load_device,
            offload_device=target_offload_device,
            profile=native_profile,
        )

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

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
            # Upstream size/mean contract: reject incorrect hard-coded values from
            # generic Qwen3-VL or experimental branches.
            validate_h3_processor_contract(
                processor_component,
                require_video=bool(require_multimodal_processor),
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

    from ..model_loader import _checkpoint_index

    if external_weights is not None:
        checkpoint_shards, checkpoint_weight_map = [external_weights], None
        LOGGER.info(
            "text_encoder weights come from flat single file %s; structure/configuration still come from %s",
            external_weights.name,
            component,
        )
    else:
        checkpoint_shards, checkpoint_weight_map = _checkpoint_index(component)
    quantized = _component_is_quantized(
        component,
        shards=checkpoint_shards,
        weight_map=checkpoint_weight_map,
    )
    if quantized:
        # Flat single-file weights have no quant_meta.json; use the conversion contract
        # as the comparison baseline for each Linear marker. It is partition-independent
        # because the Qwen component is shared by both partitions.
        quant_metadata = (
            {"format": INT8_FORMAT, "convrot": True}
            if external_weights is not None
            else _validate_text_encoder_quant_metadata(component, partition=partition)
        )
        if not ALLOW_PARTIAL_OFFLOAD_INT8:
            raise H3ComponentError("int8 text_encoder requires ALLOW_PARTIAL_OFFLOAD_INT8")
        model = _load_quantized_text_encoder(
            component, config=config, model_dtype=model_dtype,
            offload_device=target_offload_device, attention_backend=attention_backend,
            quant_metadata=quant_metadata,
            shards=checkpoint_shards,
            weight_map=checkpoint_weight_map,
        )
    elif external_weights is not None:
        raise H3ComponentError(
            f"{external_weights.name} is not a quantized checkpoint; for a BF16 text encoder, select "
            "the sharded component from the release because Transformers requires the complete component directory"
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

    # ``encoder`` imports several loader helpers at module import time.  Keep
    # this import deferred so that the two modules do not form a partial
    # circular import, while ensuring the wrapper class is in this module's
    # scope when the loader is actually called.
    from .encoder import MiniMaxH3TextEncoder

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

__all__ = [n for n in list(globals()) if not n.startswith("__")]
