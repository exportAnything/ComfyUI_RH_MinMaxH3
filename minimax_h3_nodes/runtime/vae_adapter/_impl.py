# SPDX-License-Identifier: Apache-2.0
"""Direct, in-process MiniMax H3 video/audio VAE adapters.

This module intentionally has no SGLang or Diffusers dependency.  The released
inference-only VAE implementations are vendored next to the custom node and
their distributed collectives are reduced to single-process operations.

Checkpoint contract
-------------------
* video VAE: ``video_vae/source/model.safetensors`` (canonical release layout)
* audio VAE: one or more ``*.safetensors`` under ``audio_vae``
* standard Comfy single-file VAEs: configuration embedded under the
  ``minimax_h3_video_vae`` or ``minimax_h3_audio_vae`` safetensors metadata key
* both component ``config.json`` files must carry per-channel
  ``latents_mean`` and ``latents_std`` values
* the video wrapper config is paired with
  ``video_vae/source/config.json``; model construction uses and validates that
  inner ``AutoencoderKLLegacy`` architecture instead of guessing from the
  outer remote-code wrapper

The diffusion model predicts *normalized* H3 latents.  Adapter ``decode``
methods therefore reverse the release normalization before invoking a VAE;
the encode helpers apply the forward normalization.
"""

from __future__ import annotations

import inspect
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ...vendor.minimax_h3_audio_vae import DacAudioVAE  # vendor lives at the top level of minimax_h3_nodes.
from ...vendor.minimax_h3_video_vae import AutoencoderKLLegacy
from ..backend_state import TORCH_BACKEND_STATE_LOCK


LOGGER = logging.getLogger(__name__)

# Both release condition encoders temporarily mutate shared state: Video VAE
# encoding changes the model dtype/tiling mode and forks the default generators,
# while Audio VAE encoding changes CUDA/cuDNN/SDP backend flags.  Decode/model
# moves must not observe those transient states, so every adapter model action
# shares one re-entrant critical section.
_VAE_CONDITION_ENCODE_LOCK = threading.RLock()

VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_SAMPLE_RATE = 32_000
AUDIO_OUTPUT_CHANNELS = 2

_SINGLE_FILE_METADATA_KEYS = {
    "video_vae": "minimax_h3_video_vae",
    "audio_vae": "minimax_h3_audio_vae",
}
_SINGLE_FILE_CONFIG_TENSOR_KEYS = frozenset({"latents_mean", "latents_std"})


class H3VAEError(RuntimeError):
    """Raised for an invalid component layout, checkpoint, or tensor contract."""


@dataclass(frozen=True)
class _H3SingleFileMetadata:
    config: dict[str, Any]
    weight_dtype: torch.dtype


@dataclass(frozen=True)
class H3LatentStats:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        expected_channels: int,
        component_name: str,
    ) -> "H3LatentStats":
        mean = _config_value(config, "latents_mean")
        std = _config_value(config, "latents_std")
        if not isinstance(mean, Sequence) or isinstance(mean, (str, bytes)):
            raise H3VAEError(
                f"{component_name}/config.json is missing numeric latents_mean"
            )
        if not isinstance(std, Sequence) or isinstance(std, (str, bytes)):
            raise H3VAEError(
                f"{component_name}/config.json is missing numeric latents_std"
            )
        try:
            mean_tuple = tuple(float(value) for value in mean)
            std_tuple = tuple(float(value) for value in std)
        except (TypeError, ValueError) as exc:
            raise H3VAEError(
                f"{component_name} latent statistics must contain numbers"
            ) from exc
        if len(mean_tuple) != expected_channels or len(std_tuple) != expected_channels:
            raise H3VAEError(
                f"{component_name} latent statistics must contain "
                f"{expected_channels} values; got mean={len(mean_tuple)}, "
                f"std={len(std_tuple)}"
            )
        if any(not torch.isfinite(torch.tensor(value)) for value in mean_tuple):
            raise H3VAEError(f"{component_name} latents_mean contains non-finite data")
        if any(
            not torch.isfinite(torch.tensor(value)) or value <= 0
            for value in std_tuple
        ):
            raise H3VAEError(
                f"{component_name} latents_std must be finite and greater than zero"
            )
        return cls(mean=mean_tuple, std=std_tuple)

    def tensors_for(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.std, device=value.device, dtype=value.dtype)
        shape = [1] * value.ndim
        shape[1] = len(self.mean)
        return mean.view(shape), std.view(shape)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors_for(value)
        return (value - mean) / std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        mean, std = self.tensors_for(value)
        return value * std + mean


@dataclass
class H3VAEBundle:
    """Pair of direct VAE adapters used by H3 sampling/conditioning nodes."""

    video_vae: "MiniMaxH3VideoVAEAdapter"
    audio_vae: "MiniMaxH3AudioVAEAdapter"

    def to(self, device: str | torch.device) -> "H3VAEBundle":
        self.video_vae.to(device)
        self.audio_vae.to(device)
        return self

    def offload(self) -> "H3VAEBundle":
        return self.to("cpu")


def _config_value(config: Mapping[str, Any], name: str, default: Any = None) -> Any:
    if name in config:
        return config[name]
    for nested_name in ("arch_config", "vae_config", "config"):
        nested = config.get(nested_name)
        if isinstance(nested, Mapping) and name in nested:
            return nested[name]
    return default


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise H3VAEError(f"Missing component config: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise H3VAEError(f"Cannot read component config: {path}") from exc
    if not isinstance(value, dict):
        raise H3VAEError(f"Expected a JSON object in {path}")
    return value


def _resolve_within_root(root: Path, candidate: Path, *, label: str) -> Path:
    """Resolve symlinks while keeping a VAE artifact below its selected root."""

    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise H3VAEError(
            f"{label} resolves outside selected VAE root {resolved_root}: "
            f"{resolved_candidate}"
        ) from exc
    return resolved_candidate


def _safe_relative_path(raw_value: Any, *, label: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise H3VAEError(f"{label} must be a non-empty relative path")
    relative = Path(raw_value.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise H3VAEError(
            f"{label} must be relative and must not contain '..': {raw_value!r}"
        )
    return relative


def _component_config(
    component_dir: Path,
    component_name: str,
) -> dict[str, Any]:
    config_path = _resolve_within_root(
        component_dir,
        component_dir / "config.json",
        label=f"{component_name} config",
    )
    return _read_json(config_path)


def _single_file_metadata_config(
    checkpoint: Path,
    component_name: str,
    *,
    required: bool,
) -> _H3SingleFileMetadata | None:
    """Read one standard Comfy H3 VAE's embedded JSON configuration.

    RunningHub release checkpoints predate this metadata convention.  When
    ``required`` is false, a checkpoint with no H3 metadata therefore keeps
    using its external ``config.json`` unchanged.  A checkpoint declaring the
    opposite H3 component is always rejected instead of being treated as a
    legacy file.
    """

    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.suffix.lower() != ".safetensors":
        if required:
            raise H3VAEError(
                f"Standalone {component_name} checkpoint must be a .safetensors file: "
                f"{checkpoint}"
            )
        return None
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise H3VAEError(
            "Loading MiniMax H3 weights requires the safetensors package"
        ) from exc
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            dtype_names = {
                str(handle.get_slice(key).get_dtype())
                for key in handle.keys()
                if key not in _SINGLE_FILE_CONFIG_TENSOR_KEYS
            }
    except Exception as exc:
        raise H3VAEError(
            f"Cannot read safetensors metadata from {checkpoint}"
        ) from exc

    metadata_key = _SINGLE_FILE_METADATA_KEYS[component_name]
    raw_config = metadata.get(metadata_key)
    if raw_config is None:
        other_keys = sorted(
            key
            for name, key in _SINGLE_FILE_METADATA_KEYS.items()
            if name != component_name and key in metadata
        )
        if other_keys:
            raise H3VAEError(
                f"{checkpoint} declares {other_keys[0]!r}, expected "
                f"{metadata_key!r}"
            )
        if required:
            raise H3VAEError(
                f"Standalone {component_name} checkpoint {checkpoint} is missing "
                f"safetensors metadata key {metadata_key!r}"
            )
        return None
    try:
        config = json.loads(raw_config)
    except (TypeError, json.JSONDecodeError) as exc:
        raise H3VAEError(
            f"Safetensors metadata {metadata_key!r} in {checkpoint} is not valid JSON"
        ) from exc
    if not isinstance(config, dict):
        raise H3VAEError(
            f"Safetensors metadata {metadata_key!r} in {checkpoint} must be a JSON object"
        )
    dtype_map = {
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
    }
    floating_dtypes = {dtype_map[name] for name in dtype_names if name in dtype_map}
    if len(floating_dtypes) != 1:
        found = ", ".join(sorted(dtype_names)) or "none"
        raise H3VAEError(
            f"Standard Comfy {component_name} checkpoint must use one floating "
            f"weight dtype; found: {found}"
        )
    return _H3SingleFileMetadata(
        config=config,
        weight_dtype=next(iter(floating_dtypes)),
    )


def _looks_like_component(path: Path, component_name: str) -> bool:
    config_path = _resolve_within_root(
        path,
        path / "config.json",
        label=f"{component_name} config",
    )
    if not config_path.is_file():
        return False
    try:
        config = _read_json(config_path)
    except H3VAEError:
        return False
    class_name = str(config.get("_class_name", "")).lower()
    if component_name == "video_vae":
        return (
            _config_value(config, "latent_channels") == VIDEO_LATENT_CHANNELS
            or "video" in class_name
            or (path / "source" / "model.safetensors").is_file()
        )
    return (
        _config_value(config, "latent_channels") == AUDIO_LATENT_CHANNELS
        or "audio" in class_name
    )


def resolve_h3_component_dir(
    model_or_component_path: str | Path,
    component_name: str,
) -> Path:
    """Resolve a release root or a direct component directory."""

    if component_name not in {"video_vae", "audio_vae"}:
        raise ValueError(f"Unsupported H3 VAE component {component_name!r}")
    root = Path(model_or_component_path).expanduser().resolve()
    if not root.exists():
        raise H3VAEError(f"Model path does not exist: {root}")
    if root.is_file():
        root = root.parent

    candidates: list[Path] = []
    if root.name == component_name or _looks_like_component(root, component_name):
        candidates.append(root)
    candidates.extend(
        [
            root / component_name,
            root / ("vae" if component_name == "video_vae" else component_name),
        ]
    )

    model_index = _resolve_within_root(
        root,
        root / "model_index.json",
        label="VAE model_index",
    )
    if model_index.is_file():
        index = _read_json(model_index)
        entry = index.get(component_name)
        if isinstance(entry, Mapping):
            relative = entry.get("path") or entry.get("subfolder")
            if isinstance(relative, str):
                candidates.insert(
                    0,
                    root
                    / _safe_relative_path(
                        relative,
                        label=f"model_index {component_name} path",
                    ),
                )
        # Diffusers-style entries are [library, class]; their subfolder is the
        # key itself and is already covered by root/component_name.

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = _resolve_within_root(
            root,
            candidate,
            label=f"{component_name} component",
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        if _looks_like_component(candidate, component_name):
            return candidate
    searched = ", ".join(str(path) for path in seen)
    raise H3VAEError(
        f"Could not resolve MiniMax H3 {component_name}; searched: {searched}"
    )


def _files_from_safetensors_index(
    index_path: Path,
    *,
    component_root: Path | None = None,
) -> list[Path]:
    root = (component_root or index_path.parent).resolve()
    index_path = _resolve_within_root(
        root,
        index_path,
        label="Safetensors index",
    )
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise H3VAEError(f"Invalid safetensors index: {index_path}")
    paths = sorted(
        {
            _resolve_within_root(
                root,
                index_path.parent
                / _safe_relative_path(
                    name,
                    label=f"Safetensors shard in {index_path}",
                ),
                label=f"Safetensors shard in {index_path}",
            )
            for name in weight_map.values()
        }
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise H3VAEError(
            "Safetensors index references missing files: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _safe_component_relative_path(
    component_dir: Path,
    raw_value: Any,
    *,
    field_name: str,
) -> Path:
    relative = _safe_relative_path(raw_value, label=field_name)
    return _resolve_within_root(
        component_dir,
        component_dir / relative,
        label=field_name,
    )


def _declared_weight_files(
    component_dir: Path,
    component_name: str,
    config: Mapping[str, Any],
) -> list[Path] | None:
    """Resolve the release's ``source_safetensors_path`` declaration.

    The real H3 bundle uses two different bases for the same basename:
    ``video_vae/source/model.safetensors`` and
    ``audio_vae/model.safetensors``.  The outer config stores only
    ``model.safetensors``, so component type determines the preferred base.
    """

    declared = config.get("source_safetensors_path")
    if declared is None:
        return None
    relative = _safe_component_relative_path(
        component_dir,
        declared,
        field_name=f"{component_name}.source_safetensors_path",
    ).relative_to(component_dir.resolve())
    bases = (
        (component_dir / "source", component_dir)
        if component_name == "video_vae"
        else (component_dir, component_dir / "source")
    )
    attempted: list[Path] = []
    for base in bases:
        candidate = (base / relative).resolve()
        try:
            candidate.relative_to(component_dir.resolve())
        except ValueError:
            continue
        attempted.append(candidate)
        if candidate.is_file():
            if candidate.name.endswith(".safetensors.index.json"):
                return _files_from_safetensors_index(
                    candidate,
                    component_root=component_dir,
                )
            return [candidate]
        index_candidate = candidate.with_name(
            candidate.name + ".index.json"
        )
        attempted.append(index_candidate)
        if index_candidate.is_file():
            return _files_from_safetensors_index(
                index_candidate,
                component_root=component_dir,
            )
    raise H3VAEError(
        f"Declared {component_name} checkpoint does not exist; checked: "
        + ", ".join(str(path) for path in attempted)
    )


def _explicit_weight_files(
    weight_files: Iterable[str | Path] | None,
    component_name: str,
) -> list[Path]:
    """Validate caller-supplied checkpoint files for one VAE component.

    These come from the flat ``models/MiniMax-H3`` weights root and therefore
    live outside the release component directory; only existence is enforced
    here, the filename type gate runs in the node layer.
    """

    if not weight_files:
        return []
    out: list[Path] = []
    for raw in weight_files:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise H3VAEError(
                f"{component_name} checkpoint file not found: {path}"
            )
        out.append(path)
    return out


def _component_weight_files(
    component_dir: Path,
    component_name: str,
    config: Mapping[str, Any],
) -> list[Path]:
    declared = _declared_weight_files(component_dir, component_name, config)
    if declared is not None:
        return declared

    if component_name == "video_vae":
        canonical = _resolve_within_root(
            component_dir,
            component_dir / "source" / "model.safetensors",
            label="video_vae canonical checkpoint",
        )
        if canonical.is_file():
            return [canonical]

    index_candidates = sorted(component_dir.glob("*.safetensors.index.json"))
    index_candidates.extend(
        sorted((component_dir / "source").glob("*.safetensors.index.json"))
        if (component_dir / "source").is_dir()
        else []
    )
    if index_candidates:
        return _files_from_safetensors_index(
            index_candidates[0],
            component_root=component_dir,
        )

    top_level = sorted(
        {
            _resolve_within_root(
                component_dir,
                path,
                label=f"{component_name} checkpoint",
            )
            for path in component_dir.glob("*.safetensors")
        }
    )
    top_level = [path for path in top_level if path.is_file()]
    if top_level:
        return top_level
    recursive = sorted(
        {
            _resolve_within_root(
                component_dir,
                path,
                label=f"{component_name} checkpoint",
            )
            for path in component_dir.rglob("*.safetensors")
        }
    )
    recursive = [path for path in recursive if path.is_file()]
    if recursive:
        return recursive
    raise H3VAEError(f"No safetensors checkpoint found under {component_dir}")


def _component_load_inputs(
    model_or_component_path: str | Path,
    component_name: str,
    weight_files: Sequence[str | Path] | None,
) -> tuple[Path, dict[str, Any], list[Path], bool, torch.dtype | None]:
    """Resolve configuration and weights for release or Comfy-native layouts.

    A direct file path requires the matching embedded metadata.  An external
    weight override keeps the selected release/descriptor ``config.json`` as a
    fallback, but matching embedded metadata takes precedence for architecture
    and latent statistics.  Auto-discovered RunningHub directory weights are
    deliberately left on the historical external-config path.
    """

    selected = Path(model_or_component_path).expanduser().resolve()
    explicit_weights = _explicit_weight_files(weight_files, component_name)
    if selected.is_file():
        if explicit_weights and explicit_weights != [selected]:
            raise H3VAEError(
                f"A standalone {component_name} checkpoint cannot be combined "
                "with a different weight_files override"
            )
        embedded = _single_file_metadata_config(
            selected,
            component_name,
            required=True,
        )
        assert embedded is not None
        return (
            selected.parent,
            embedded.config,
            [selected],
            True,
            embedded.weight_dtype,
        )

    component_dir = resolve_h3_component_dir(selected, component_name)
    config = _component_config(component_dir, component_name)
    if explicit_weights:
        embedded = None
        if len(explicit_weights) == 1:
            embedded = _single_file_metadata_config(
                explicit_weights[0],
                component_name,
                required=False,
            )
        if embedded is not None:
            merged_config = dict(config)
            merged_config.update(embedded.config)
            config = merged_config
        return (
            component_dir,
            config,
            explicit_weights,
            embedded is not None,
            embedded.weight_dtype if embedded is not None else None,
        )

    return (
        component_dir,
        config,
        _component_weight_files(component_dir, component_name, config),
        False,
        None,
    )


def _resolve_dtype(value: Any, *, default: torch.dtype) -> torch.dtype:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower().replace("torch.", "")
    mapping = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {value!r}")
    return mapping[normalized]


@contextmanager
def _default_dtype(dtype: torch.dtype):
    with TORCH_BACKEND_STATE_LOCK:
        previous = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


def _can_assign_state_dict() -> bool:
    try:
        return "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters
    except (TypeError, ValueError):
        return False


def _best_key_prefix(
    keys: Iterable[str],
    model_keys: set[str],
    component_name: str,
) -> str:
    source_keys = tuple(keys)
    prefixes = (
        "",
        "module.",
        "model.",
        f"{component_name}.",
        f"model.{component_name}.",
        f"module.{component_name}.",
        "vae.",
        "first_stage_model.",
    )
    best_prefix = ""
    best_score = (-1, -len(source_keys), 0)
    for prefix in prefixes:
        # Never partially strip a checkpoint.  The native release is
        # unprefixed; wrapper prefixes are accepted only when every tensor uses
        # that same namespace, avoiding key collisions and accidental remaps.
        if prefix and not all(key.startswith(prefix) for key in source_keys):
            continue
        mapped = (
            tuple(key[len(prefix) :] for key in source_keys)
            if prefix
            else source_keys
        )
        if len(set(mapped)) != len(mapped):
            continue
        hits = sum(key in model_keys for key in mapped)
        unexpected = len(mapped) - hits
        # Prefer exact/unprefixed keys when scores tie, matching the upstream
        # native loader's strict state-dict contract.
        score = (hits, -unexpected, 1 if not prefix else 0)
        if score > best_score:
            best_prefix, best_score = prefix, score
    return best_prefix


def _normalize_checkpoint_keys(
    state: Mapping[str, torch.Tensor],
    model_keys: set[str],
    component_name: str,
) -> dict[str, torch.Tensor]:
    prefix = _best_key_prefix(state.keys(), model_keys, component_name)
    if prefix:
        LOGGER.info(
            "Stripping uniform %s checkpoint prefix %r",
            component_name,
            prefix,
        )
        normalized = {key[len(prefix) :]: value for key, value in state.items()}
    else:
        normalized = dict(state)
    if len(normalized) != len(state):
        raise H3VAEError(
            f"{component_name} checkpoint prefix {prefix!r} creates key collisions"
        )

    # PyTorch's legacy weight_norm checkpoints use ``weight_g``/``weight_v``.
    # The parametrizations API exposes the same tensors as ``original0`` and
    # ``original1``.  Convert only when the legacy key is not itself expected
    # and the precise modern target exists in this concrete model.
    weight_norm_suffixes = (
        (".weight_g", ".parametrizations.weight.original0"),
        (".weight_v", ".parametrizations.weight.original1"),
    )
    remapped: dict[str, torch.Tensor] = {}
    remap_count = 0
    for key, value in normalized.items():
        target = key
        if key not in model_keys:
            for legacy_suffix, current_suffix in weight_norm_suffixes:
                if not key.endswith(legacy_suffix):
                    continue
                candidate = key[: -len(legacy_suffix)] + current_suffix
                if candidate in model_keys:
                    target = candidate
                    remap_count += 1
                break
        if target in remapped:
            raise H3VAEError(
                f"{component_name} checkpoint key conversion creates duplicate "
                f"target {target!r}"
            )
        remapped[target] = value
    if remap_count:
        LOGGER.info(
            "Remapped %d legacy weight_norm keys for %s",
            remap_count,
            component_name,
        )
    return remapped


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise H3VAEError(
            "Loading MiniMax H3 weights requires the safetensors package"
        ) from exc
    return load_file(str(path), device="cpu")


def _construct_and_load(
    factory,
    weight_files: Sequence[Path],
    *,
    component_name: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    strict: bool,
    low_memory: bool,
    ignored_checkpoint_keys: Iterable[str] = (),
) -> torch.nn.Module:
    use_meta = bool(low_memory and _can_assign_state_dict())
    with _default_dtype(weight_dtype):
        if use_meta:
            with torch.device("meta"):
                model = factory()
        else:
            model = factory()

    model_keys = set(model.state_dict().keys())
    ignored_keys = set(ignored_checkpoint_keys)
    loaded_keys: set[str] = set()
    unexpected: set[str] = set()
    for weight_file in weight_files:
        LOGGER.info("Loading MiniMax H3 %s weights: %s", component_name, weight_file)
        state = _normalize_checkpoint_keys(
            _load_safetensors(weight_file), model_keys, component_name
        )
        ignored_present = ignored_keys.intersection(state).difference(model_keys)
        if ignored_present:
            LOGGER.info(
                "Using embedded metadata instead of %s checkpoint tensors: %s",
                component_name,
                ", ".join(sorted(ignored_present)),
            )
            for key in ignored_present:
                state.pop(key)
        unexpected.update(set(state) - model_keys)
        usable = {key: value for key, value in state.items() if key in model_keys}
        loaded_keys.update(usable)
        if use_meta:
            model.load_state_dict(usable, strict=False, assign=True)
        else:
            model.load_state_dict(usable, strict=False)
        del state, usable

    missing = model_keys - loaded_keys
    if strict and (missing or unexpected):
        missing_preview = ", ".join(sorted(missing)[:12])
        unexpected_preview = ", ".join(sorted(unexpected)[:12])
        raise H3VAEError(
            f"{component_name} checkpoint mismatch: "
            f"missing={len(missing)} [{missing_preview}], "
            f"unexpected={len(unexpected)} [{unexpected_preview}]"
        )
    if use_meta:
        meta_names = [
            name
            for name, tensor in (
                *model.named_parameters(),
                *model.named_buffers(),
            )
            if tensor.device.type == "meta"
        ]
        if meta_names:
            raise H3VAEError(
                f"{component_name} checkpoint left {len(meta_names)} meta tensors; "
                f"first entries: {', '.join(meta_names[:12])}"
            )

    model = model.to(device=device, dtype=weight_dtype)
    model.eval()
    model.requires_grad_(False)
    return model


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _extract_samples(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value.get("samples")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor or a mapping containing 'samples'")
    return value


@contextmanager
def _autocast_for(device: torch.device, dtype: torch.dtype):
    enabled = device.type in {"cuda", "xpu"} and dtype in {
        torch.float16,
        torch.bfloat16,
    }
    if not enabled:
        yield
        return
    with torch.autocast(device_type=device.type, dtype=dtype):
        yield


def _fork_rng(device: torch.device, seed: int):
    devices: list[int] = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    return torch.random.fork_rng(devices=devices, enabled=True)


@contextmanager
def _audio_vae_determinism():
    """Match the released deterministic reference-audio encode recipe."""

    with TORCH_BACKEND_STATE_LOCK:
        backends = torch.backends
        saved = (
            backends.cuda.matmul.allow_tf32,
            backends.cudnn.allow_tf32,
            backends.cudnn.benchmark,
            backends.cudnn.deterministic,
            backends.cudnn.enabled,
            backends.cuda.flash_sdp_enabled(),
            backends.cuda.mem_efficient_sdp_enabled(),
            backends.cuda.math_sdp_enabled(),
        )
        backends.cuda.matmul.allow_tf32 = False
        backends.cudnn.allow_tf32 = False
        backends.cudnn.benchmark = False
        backends.cudnn.deterministic = True
        backends.cudnn.enabled = False
        backends.cuda.enable_flash_sdp(False)
        backends.cuda.enable_mem_efficient_sdp(False)
        backends.cuda.enable_math_sdp(True)
        try:
            yield
        finally:
            (
                backends.cuda.matmul.allow_tf32,
                backends.cudnn.allow_tf32,
                backends.cudnn.benchmark,
                backends.cudnn.deterministic,
                backends.cudnn.enabled,
                flash,
                memory_efficient,
                math_sdp,
            ) = saved
            backends.cuda.enable_flash_sdp(flash)
            backends.cuda.enable_mem_efficient_sdp(memory_efficient)
            backends.cuda.enable_math_sdp(math_sdp)


class MiniMaxH3VideoVAEAdapter:
    """Direct visual VAE with H3 normalization and Comfy tensor conversion."""

    latent_channels = VIDEO_LATENT_CHANNELS
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4

    def __init__(
        self,
        model: AutoencoderKLLegacy,
        stats: H3LatentStats,
        *,
        component_dir: Path,
        compute_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model = model
        self.stats = stats
        self.component_dir = component_dir
        self.compute_dtype = compute_dtype

    @property
    def device(self) -> torch.device:
        return _model_device(self.model)

    @property
    def raw_model(self) -> AutoencoderKLLegacy:
        return self.model

    @property
    def processor(self):
        return self.model.processor

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def eval(self) -> "MiniMaxH3VideoVAEAdapter":
        self.model.eval()
        return self

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | str | None = None,
    ) -> "MiniMaxH3VideoVAEAdapter":
        kwargs: dict[str, Any] = {"device": torch.device(device)}
        if dtype is not None:
            kwargs["dtype"] = _resolve_dtype(dtype, default=torch.float32)
        with _VAE_CONDITION_ENCODE_LOCK:
            self.model.to(**kwargs)
        return self

    def offload(self) -> "MiniMaxH3VideoVAEAdapter":
        return self.to("cpu")

    @torch.inference_mode()
    def decode_base(
        self,
        raw_latents: torch.Tensor,
        *,
        frame_count: int | None = None,
    ) -> torch.Tensor:
        """Decode unnormalized latents to model-native normalized pixels."""

        with _VAE_CONDITION_ENCODE_LOCK:
            device = self.device
            raw_latents = raw_latents.to(device=device, dtype=torch.float32)
            with _autocast_for(device, self.compute_dtype):
                return self.model.decode_base(raw_latents, frame_num=frame_count)

    @torch.inference_mode()
    def decode_bcthw(
        self,
        normalized_latents: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        frame_count: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        """Return decoded frames as float32 ``[B,3,T,H,W]`` in ``[0,1]``."""

        latents = _extract_samples(normalized_latents, name="video latents")
        if latents.ndim != 5 or int(latents.shape[1]) != self.latent_channels:
            raise H3VAEError(
                "MiniMax H3 video latents must be [B,24,T,H,W], got "
                f"{tuple(latents.shape)}"
            )
        with _VAE_CONDITION_ENCODE_LOCK:
            device = self.device
            latents = latents.to(device=device, dtype=torch.float32)
            raw = self.stats.denormalize(latents)
            with _autocast_for(device, self.compute_dtype):
                frames = self.model.decode_base(raw, frame_num=frame_count)
                frames = self.model.processor.revert_tensor(frames)
        if frames.ndim != 5 or int(frames.shape[1]) != 3:
            raise H3VAEError(
                "MiniMax H3 video VAE returned an invalid tensor: "
                f"{tuple(frames.shape)}"
            )
        if target_height is not None:
            if int(frames.shape[-2]) < int(target_height):
                raise H3VAEError("Decoded video is shorter than target_height")
            frames = frames[..., : int(target_height), :]
        if target_width is not None:
            if int(frames.shape[-1]) < int(target_width):
                raise H3VAEError("Decoded video is narrower than target_width")
            frames = frames[..., : int(target_width)]
        return frames.float().contiguous()

    @torch.inference_mode()
    def decode(
        self,
        normalized_latents: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        frame_count: int | None = None,
        target_height: int | None = None,
        target_width: int | None = None,
    ) -> torch.Tensor:
        """Return a ComfyUI ``IMAGE`` batch as ``[B*T,H,W,3]``."""

        frames = self.decode_bcthw(
            normalized_latents,
            frame_count=frame_count,
            target_height=target_height,
            target_width=target_width,
        )
        return (
            frames.permute(0, 2, 3, 4, 1)
            .reshape(-1, frames.shape[-2], frames.shape[-1], 3)
            .contiguous()
        )

    def _canonical_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 3 and pixels.shape[-1] == 3:
            pixels = pixels.unsqueeze(0).unsqueeze(0)  # [B=1,T=1,H,W,C]
        elif pixels.ndim == 4 and pixels.shape[-1] == 3:
            pixels = pixels.unsqueeze(1)  # [B,T=1,H,W,C]
        elif pixels.ndim == 5 and pixels.shape[-1] == 3:
            pass  # [B,T,H,W,C]
        elif pixels.ndim == 5 and pixels.shape[1] == 3:
            return pixels
        else:
            raise H3VAEError(
                "Pixels must be [H,W,3], [B,H,W,3], [B,T,H,W,3], or "
                f"[B,3,T,H,W]; got {tuple(pixels.shape)}"
            )
        return pixels.permute(0, 4, 1, 2, 3).contiguous()

    @torch.inference_mode()
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        process_image: bool | None = None,
        seed: int = 42,
        use_fp16_latent: bool = True,
        parallel_tiling: bool | None = False,
    ) -> torch.Tensor:
        """Encode ``[0,1]`` pixels and return normalized ``[B,24,T,H,W]``.

        The release keyframe/reference-image recipe samples the VAE posterior
        under seed 42, hence sampling (rather than posterior mean) is retained.
        The official condition path also disables parallel tiling, runs the
        VAE in fp32, casts the sampled latent to fp16, then converts back to
        fp32 for normalization.  That fp16 round-trip is numerically material
        and therefore enabled by default.  Multi-frame inputs are trimmed with
        the vendored VAE processor's native temporal alignment rule.
        """

        if not isinstance(pixels, torch.Tensor):
            raise TypeError("Video VAE pixels must be a torch.Tensor")
        if process_image is None:
            process_image = pixels.ndim in (3, 4)
        pixels = self._canonical_pixels(pixels)
        batch, channels, frames, height, width = pixels.shape
        if channels != 3:
            raise H3VAEError("Video VAE input must contain three RGB channels")
        if height % 16 or width % 16:
            raise H3VAEError(
                f"Video VAE input must be divisible by 16, got {height}x{width}"
            )

        with _VAE_CONDITION_ENCODE_LOCK:
            device = self.device
            pixels = pixels.to(device=device, dtype=torch.float32)
            flat = pixels.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, channels, height, width
            )
            flat = self.model.transform(flat)
            transformed = flat.reshape(batch, frames, channels, height, width)
            transformed = transformed.permute(0, 2, 1, 3, 4).contiguous()

            # Match encode_images/encode_videos: align to the complete spatial
            # patch and, for video, keep the longest supported leading chunk.
            processor = self.model.processor
            if process_image:
                if frames != 1:
                    raise H3VAEError(
                        "process_image=True requires exactly one frame per batch item"
                    )
            else:
                used_frames = int(processor.get_suitable_video_length(frames))
                transformed = transformed[:, :, :used_frames]
            new_height, new_width = processor._align_to_total_patch_size(
                height, width
            )
            if process_image:
                transformed = processor._crop_to_align(
                    transformed[:, :, 0],
                    new_height,
                    new_width,
                    is_video=False,
                ).unsqueeze(2)
            else:
                transformed = processor._crop_to_align(
                    transformed,
                    new_height,
                    new_width,
                    is_video=True,
                )

            parameter = next(self.model.parameters())
            previous_dtype = parameter.dtype
            missing_tiling = object()
            previous_parallel_tiling = getattr(
                self.model, "parallel_tiling", missing_tiling
            )
            if (
                parallel_tiling is not None
                and previous_parallel_tiling is not missing_tiling
            ):
                self.model.parallel_tiling = bool(parallel_tiling)
            # The release contract explicitly encodes keyframes/references with
            # fp32 VAE weights, even though visual decode uses fp16 autocast.
            if previous_dtype != torch.float32:
                self.model.to(dtype=torch.float32)
            try:
                with _fork_rng(device, seed):
                    torch.default_generator.manual_seed(seed)
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(seed)
                    raw = self.model.encode_base(
                        transformed,
                        process_image=process_image,
                    )
            finally:
                if previous_parallel_tiling is not missing_tiling:
                    self.model.parallel_tiling = previous_parallel_tiling
                if previous_dtype != torch.float32:
                    self.model.to(dtype=previous_dtype)
            if raw.ndim == 4:
                raw = raw.unsqueeze(2)
            if use_fp16_latent:
                raw = raw.to(torch.float16)
            # Official conditioning normalizes fp32 latents on CPU after the
            # numerically significant fp16 round-trip.
            raw_cpu = raw.float().cpu()
        return self.stats.normalize(raw_cpu).contiguous()


class MiniMaxH3AudioVAEAdapter:
    """Direct DAC/BigVGAN audio VAE with H3 latent normalization."""

    latent_channels = AUDIO_LATENT_CHANNELS
    sample_rate = AUDIO_SAMPLE_RATE
    output_channels = AUDIO_OUTPUT_CHANNELS
    latent_rate = 40

    def __init__(
        self,
        model: DacAudioVAE,
        stats: H3LatentStats,
        *,
        component_dir: Path,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.model = model
        self.stats = stats
        self.component_dir = component_dir
        self.compute_dtype = compute_dtype

    @property
    def device(self) -> torch.device:
        return _model_device(self.model)

    @property
    def raw_model(self) -> DacAudioVAE:
        return self.model

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def eval(self) -> "MiniMaxH3AudioVAEAdapter":
        self.model.eval()
        return self

    def to(
        self,
        device: str | torch.device,
        dtype: torch.dtype | str | None = None,
    ) -> "MiniMaxH3AudioVAEAdapter":
        kwargs: dict[str, Any] = {"device": torch.device(device)}
        if dtype is not None:
            kwargs["dtype"] = _resolve_dtype(dtype, default=torch.float32)
        with _VAE_CONDITION_ENCODE_LOCK:
            self.model.to(**kwargs)
        return self

    def offload(self) -> "MiniMaxH3AudioVAEAdapter":
        return self.to("cpu")

    @torch.inference_mode()
    def decode_native(
        self,
        raw_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode unnormalized ``[C,32,T]`` to native ``[C,1,L]``."""

        with _VAE_CONDITION_ENCODE_LOCK:
            device = self.device
            raw_latents = raw_latents.to(device=device, dtype=torch.float32)
            with _autocast_for(device, self.compute_dtype):
                return self.model.decode(raw_latents)

    @torch.inference_mode()
    def decode(
        self,
        normalized_latents: torch.Tensor | Mapping[str, torch.Tensor],
        *,
        sample_count: int | None = None,
    ) -> torch.Tensor:
        """Return a ComfyUI waveform tensor ``[1,2,L]``."""

        latents = _extract_samples(normalized_latents, name="audio latents")
        if latents.ndim == 4:
            if int(latents.shape[0]) != 1:
                raise H3VAEError("MiniMax H3 audio decode currently supports batch=1")
            latents = latents.squeeze(0)
        if latents.ndim != 3 or int(latents.shape[1]) != self.latent_channels:
            raise H3VAEError(
                "MiniMax H3 audio latents must be [2,32,T], got "
                f"{tuple(latents.shape)}"
            )
        if int(latents.shape[0]) != self.output_channels:
            raise H3VAEError(
                "MiniMax H3 audio latents must contain two channel rows, got "
                f"{int(latents.shape[0])}"
            )
        with _VAE_CONDITION_ENCODE_LOCK:
            latents = latents.to(device=self.device, dtype=torch.float32)
            raw = self.stats.denormalize(latents)
            waveform = self.decode_native(raw)
        if waveform.ndim != 3 or int(waveform.shape[1]) != 1:
            raise H3VAEError(
                "MiniMax H3 audio VAE returned an invalid tensor: "
                f"{tuple(waveform.shape)}"
            )
        waveform = waveform.permute(1, 0, 2).float().contiguous()
        if sample_count is not None:
            waveform = waveform[..., : int(sample_count)]
        return waveform

    @torch.inference_mode()
    def encode(
        self,
        waveform: torch.Tensor | Mapping[str, Any],
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Encode audio to normalized ``[2,32,T]`` posterior-mean latents."""

        if isinstance(waveform, Mapping):
            if sample_rate is None:
                sample_rate = int(waveform.get("sample_rate", self.sample_rate))
            waveform = waveform.get("waveform")
        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Audio input must be a Tensor or Comfy AUDIO mapping")
        sample_rate = self.sample_rate if sample_rate is None else int(sample_rate)
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3 or int(waveform.shape[0]) != 1:
            raise H3VAEError(
                "MiniMax H3 audio encode expects [1,C,L] or [C,L], got "
                f"{tuple(waveform.shape)}"
            )
        if sample_rate != self.sample_rate:
            try:
                import torchaudio
            except ImportError as exc:
                raise H3VAEError(
                    "Resampling reference audio requires torchaudio, or provide 32 kHz"
                ) from exc
            waveform = torchaudio.transforms.Resample(
                sample_rate, self.sample_rate
            )(waveform.float())

        waveform = waveform.squeeze(0).float()
        channels = int(waveform.shape[0])
        if channels < 1:
            raise H3VAEError("Reference audio contains no channels")
        if channels == 1:
            waveform = waveform.repeat(2, 1)
        elif channels > self.output_channels:  # No layout: average mono→stereo.
            import logging
            logging.getLogger(__name__).warning(
                "audio VAE received %s channels with no layout; averaged down to stereo", channels
            )
            waveform = waveform.mean(dim=0, keepdim=True).expand(
                self.output_channels, -1
            ).contiguous()

        # The released audio VAE treats stereo channels as its batch dimension.
        with _VAE_CONDITION_ENCODE_LOCK, _audio_vae_determinism():
            device = self.device
            waveform = waveform.to(device)
            audio_data = self.model.preprocess(
                waveform.unsqueeze(1), self.sample_rate
            )
            with _autocast_for(device, self.compute_dtype):
                encoded = self.model.encoder(audio_data)
                if bool(getattr(self.model, "attn_proj", False)):
                    encoded = self.model.pre_block(
                        encoded.transpose(1, 2)
                    ).transpose(1, 2)
                raw = self.model.mean_proj(encoded)
            # Match the official posterior-mean path: normalize fp32 rows on
            # CPU, then keep their channel-major layout for packing.
            raw_cpu = raw.float().cpu()
        if raw_cpu.ndim != 3:
            raise H3VAEError(
                "MiniMax H3 audio VAE mean_proj must return a 3D latent, got "
                f"{tuple(raw_cpu.shape)}"
            )
        if int(raw_cpu.shape[0]) != self.output_channels:
            raise H3VAEError(
                "MiniMax H3 audio VAE mean_proj must return two channel rows, got "
                f"{int(raw_cpu.shape[0])}"
            )
        if int(raw_cpu.shape[1]) == self.latent_channels:
            pass  # [2, 32, T], the adapter's native contract.
        elif int(raw_cpu.shape[2]) == self.latent_channels:
            raw_cpu = raw_cpu.transpose(1, 2).contiguous()  # [2,T,32] -> [2,32,T]
        else:
            raise H3VAEError(
                "MiniMax H3 audio VAE cannot canonicalize mean_proj latent "
                f"{tuple(raw_cpu.shape)}; expected [2,32,T] or [2,T,32]"
            )
        return self.stats.normalize(raw_cpu).contiguous()


_VIDEO_SOURCE_CONTRACT: dict[str, Any] = {
    "_class_name": "AutoencoderKLLegacy",
    "causal_decoder": False,
    "causal_encoder": True,
    "ch": 128,
    "ch_mult": [1, 2, 2, 4, 4, 8],
    "embed_dim": 24,
    "in_channels": 3,
    "num_res_blocks": 2,
    "num_res_blocks_decoder": None,
    "out_ch": 3,
    "padding_mode": "reflect",
    "padding_mode_t": None,
    "pixel_norm_type": "imagenet",
    "scaling_factor": 1.0,
    "shift_factor": 0.0,
    "space_down": [2, 2, 2, 2, 1, 1],
    "space_up": [1, 2, 2, 2, 2, 1],
    "time_down": [1, 2, 2, 1, 1, 1],
    "time_up": None,
    "use_3d_conv": True,
    "use_t_isolated_gn": True,
    "use_vit_decoder": True,
    "vae_ratio": 16,
    "vae_ratio_t": 4,
    "vit_decoder_kwargs": {
        "dim_head": 64,
        "ffn_activation_fn": "silu",
        "ffn_use_gated": True,
        "heads": 32,
        "norm_affine": True,
        "norm_type": "rms_norm",
        "num_layers": 36,
        "qk_norm_affine": False,
        "qk_norm_type": "rms_norm",
        "rope_dim_ratio": 0.75,
        "rope_theta": 100.0,
    },
    "z_channels": 24,
    "zq_ch_decoder": None,
    "zq_ch_encoder": None,
}


def _video_source_config(
    component_dir: Path,
    outer_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and verify the real bundle's inner visual-VAE config."""

    embedded_source = outer_config.get("source_config")
    if embedded_source is not None:
        if not isinstance(embedded_source, Mapping):
            raise H3VAEError(
                "video_vae safetensors source_config must be a JSON object"
            )
        source = dict(embedded_source)
        source_label = "video_vae safetensors source_config"
    else:
        source_path = _resolve_within_root(
            component_dir,
            component_dir / "source" / "config.json",
            label="video_vae source config",
        )
        source = _read_json(source_path)
        source_label = "video_vae/source/config.json"
    outer_source_class = outer_config.get("source_class_name")
    if outer_source_class not in (None, "AutoencoderKLLegacy"):
        raise H3VAEError(
            "video_vae.source_class_name must be AutoencoderKLLegacy, got "
            f"{outer_source_class!r}"
        )
    mismatches: list[str] = []
    for key, expected in _VIDEO_SOURCE_CONTRACT.items():
        if key not in source:
            mismatches.append(f"{key}=<missing> (expected {expected!r})")
        elif source[key] != expected:
            mismatches.append(f"{key}={source[key]!r} (expected {expected!r})")
    if mismatches:
        raise H3VAEError(
            f"{source_label} does not match the MiniMax H3 "
            "release architecture: "
            + "; ".join(mismatches[:12])
        )
    return source


def _video_vae_factory(
    outer_config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    *,
    use_tiling: bool,
):
    clip_length = int(_config_value(outer_config, "vae_clip_length", 17))
    token_drop = int(_config_value(outer_config, "vae_token_drop", 3))
    encoder_tiling = bool(
        _config_value(outer_config, "vae_encoder_tiling", 1)
    )
    tile_size = int(_config_value(outer_config, "vae_tile_size", 256))
    tile_overlap = int(
        _config_value(outer_config, "vae_tile_overlap_min", 64)
    )
    chunk_dim = int(_config_value(outer_config, "vae_chunk_dim", -1))
    if chunk_dim != -1:
        raise H3VAEError(
            "ComfyUI single-process video VAE requires vae_chunk_dim=-1"
        )
    if bool(_config_value(outer_config, "vae_encoder_parallel", 0)):
        raise H3VAEError("video_vae requests unsupported encoder parallelism")
    if bool(_config_value(outer_config, "vae_decoder_parallel", 0)):
        raise H3VAEError("video_vae requests unsupported decoder parallelism")

    def factory():
        return AutoencoderKLLegacy(
            in_channels=int(source_config["in_channels"]),
            out_ch=int(source_config["out_ch"]),
            ch=int(source_config["ch"]),
            embed_dim=int(source_config["embed_dim"]),
            z_channels=int(source_config["z_channels"]),
            use_3d_conv=bool(source_config["use_3d_conv"]),
            zq_ch_encoder=source_config["zq_ch_encoder"],
            zq_ch_decoder=source_config["zq_ch_decoder"],
            num_res_blocks=int(source_config["num_res_blocks"]),
            num_res_blocks_decoder=source_config["num_res_blocks_decoder"],
            ch_mult=list(source_config["ch_mult"]),
            space_down=list(source_config["space_down"]),
            space_up=list(source_config["space_up"]),
            time_down=list(source_config["time_down"]),
            time_up=source_config["time_up"],
            padding_mode=str(source_config["padding_mode"]),
            padding_mode_t=source_config["padding_mode_t"],
            use_t_isolated_gn=bool(source_config["use_t_isolated_gn"]),
            causal_encoder=bool(source_config["causal_encoder"]),
            causal_decoder=bool(source_config["causal_decoder"]),
            use_vit_decoder=bool(source_config["use_vit_decoder"]),
            vit_decoder_kwargs=dict(source_config["vit_decoder_kwargs"]),
            shift_factor=float(source_config["shift_factor"]),
            scaling_factor=float(source_config["scaling_factor"]),
            pixel_norm_type=str(source_config["pixel_norm_type"]),
            clip_length=clip_length,
            token_drop=token_drop,
            encoder_tiling=encoder_tiling,
            decoder_tiling=bool(use_tiling),
            parallel_tiling=False,
            tile_size=tile_size,
            tile_overlap_min=tile_overlap,
            encoder_parallel=False,
            decoder_parallel=False,
            chunk_dim=chunk_dim,
        )

    return factory


def _remove_weight_norm_parametrizations(model: torch.nn.Module) -> int:
    """Fold the vendored audio model's weight norm into plain weights.

    Standard Comfy H3 audio checkpoints are distributed after this conversion,
    whereas RunningHub checkpoints retain weight-norm parameters.  Returning
    the count lets the native path fail loudly if the vendored architecture
    unexpectedly stops exposing those parametrizations.
    """

    from torch.nn.utils import parametrize

    removed = 0
    for module in model.modules():
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(
                module,
                "weight",
                leave_parametrized=True,
            )
            removed += 1
    return removed


def _audio_vae_factory(*, folded_weight_norm: bool = False):
    def factory():
        model = DacAudioVAE(
            encoder_dim=64,
            encoder_rates=[2, 4, 4, 5, 5],
            latent_dim=2048,
            decoder_dim=1024,
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            sample_rate=AUDIO_SAMPLE_RATE,
            vae_latent_channels=AUDIO_LATENT_CHANNELS,
            attn_proj=True,
            decoder_type="bigvgan",
        )
        if folded_weight_norm:
            removed = _remove_weight_norm_parametrizations(model)
            if removed == 0:
                raise H3VAEError(
                    "Standard Comfy audio VAE expected weight-norm "
                    "parametrizations in the vendored architecture"
                )
        return model

    return factory


def load_video_vae(
    model_or_component_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    weight_dtype: torch.dtype | str | None = None,
    compute_dtype: torch.dtype | str | None = None,
    use_tiling: bool = True,
    strict: bool = True,
    low_memory: bool = True,
    weight_files: Sequence[str | Path] | None = None,
) -> MiniMaxH3VideoVAEAdapter:
    """Load the released visual VAE directly into this ComfyUI process.

    ``weight_files`` overrides checkpoint discovery with an explicit file, which
    is how the flat ``models/MiniMax-H3`` weights root is consumed: the config
    comes from embedded metadata when the file declares
    ``minimax_h3_video_vae``, otherwise from the release component directory.
    ``model_or_component_path`` may itself be a standard Comfy single-file VAE.
    """

    (
        component_dir,
        config,
        resolved_weights,
        native_single_file,
        native_weight_dtype,
    ) = (
        _component_load_inputs(
            model_or_component_path,
            "video_vae",
            weight_files,
        )
    )
    class_name = config.get("_class_name")
    if class_name not in (None, "MiniMaxH3VideoVAE"):
        raise H3VAEError(
            f"video_vae._class_name must be MiniMaxH3VideoVAE, got {class_name!r}"
        )
    channels = int(_config_value(config, "latent_channels", VIDEO_LATENT_CHANNELS))
    if channels != VIDEO_LATENT_CHANNELS:
        raise H3VAEError(f"video_vae latent_channels must be 24, got {channels}")
    source_config = _video_source_config(component_dir, config)
    stats = H3LatentStats.from_config(
        config,
        expected_channels=VIDEO_LATENT_CHANNELS,
        component_name="video_vae",
    )
    weight_dtype = _resolve_dtype(
        weight_dtype,
        default=native_weight_dtype or torch.float32,
    )
    default_compute_dtype = (
        native_weight_dtype
        if native_single_file and native_weight_dtype is not None
        else (
            torch.float16
            if torch.device(device).type == "cuda"
            else torch.float32
        )
    )
    compute_dtype = _resolve_dtype(
        compute_dtype,
        default=default_compute_dtype,
    )
    model = _construct_and_load(
        _video_vae_factory(
            config,
            source_config,
            use_tiling=use_tiling,
        ),
        resolved_weights,
        component_name="video_vae",
        device=torch.device(device),
        weight_dtype=weight_dtype,
        strict=strict,
        low_memory=low_memory,
        ignored_checkpoint_keys=(
            _SINGLE_FILE_CONFIG_TENSOR_KEYS if native_single_file else ()
        ),
    )
    return MiniMaxH3VideoVAEAdapter(
        model,
        stats,
        component_dir=component_dir,
        compute_dtype=compute_dtype,
    )


def load_audio_vae(
    model_or_component_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    weight_dtype: torch.dtype | str | None = None,
    compute_dtype: torch.dtype | str = torch.float32,
    strict: bool = True,
    low_memory: bool = True,
    weight_files: Sequence[str | Path] | None = None,
) -> MiniMaxH3AudioVAEAdapter:
    """Load the released DAC/BigVGAN VAE directly into this ComfyUI process.

    See :func:`load_video_vae` for ``weight_files`` and direct single-file
    checkpoint behavior.
    """

    (
        component_dir,
        config,
        resolved_weights,
        native_single_file,
        native_weight_dtype,
    ) = (
        _component_load_inputs(
            model_or_component_path,
            "audio_vae",
            weight_files,
        )
    )
    class_name = config.get("_class_name")
    if class_name not in (None, "MiniMaxH3AudioVAE"):
        raise H3VAEError(
            f"audio_vae._class_name must be MiniMaxH3AudioVAE, got {class_name!r}"
        )
    channels = int(_config_value(config, "latent_channels", AUDIO_LATENT_CHANNELS))
    if channels != AUDIO_LATENT_CHANNELS:
        raise H3VAEError(f"audio_vae latent_channels must be 32, got {channels}")
    sample_rate = int(_config_value(config, "sample_rate", AUDIO_SAMPLE_RATE))
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise H3VAEError(f"audio_vae sample_rate must be 32000, got {sample_rate}")
    output_channels = int(
        _config_value(config, "output_channel", AUDIO_OUTPUT_CHANNELS)
    )
    if output_channels != AUDIO_OUTPUT_CHANNELS:
        raise H3VAEError(
            f"audio_vae output_channel must be 2, got {output_channels}"
        )
    stats = H3LatentStats.from_config(
        config,
        expected_channels=AUDIO_LATENT_CHANNELS,
        component_name="audio_vae",
    )
    weight_dtype = _resolve_dtype(
        weight_dtype,
        default=native_weight_dtype or torch.float32,
    )
    compute_dtype = _resolve_dtype(compute_dtype, default=torch.float32)
    model = _construct_and_load(
        _audio_vae_factory(folded_weight_norm=native_single_file),
        resolved_weights,
        component_name="audio_vae",
        device=torch.device(device),
        weight_dtype=weight_dtype,
        strict=strict,
        low_memory=low_memory,
        ignored_checkpoint_keys=(
            _SINGLE_FILE_CONFIG_TENSOR_KEYS if native_single_file else ()
        ),
    )
    return MiniMaxH3AudioVAEAdapter(
        model,
        stats,
        component_dir=component_dir,
        compute_dtype=compute_dtype,
    )


def _selected_component_root(
    selected_root: Path,
    value: str | Path | None,
    *,
    label: str,
) -> Path | None:
    """Resolve one explicitly selected component below ``selected_root``."""

    if value is None:
        return None
    selected = Path(value).expanduser()
    if not selected.is_absolute() and ".." in selected.parts:
        raise H3VAEError(
            f"{label} must not traverse outside model_root: {value!r}"
        )
    candidate = selected if selected.is_absolute() else selected_root / selected
    return _resolve_within_root(selected_root, candidate, label=label)


def load_h3_vae_bundle(
    model_root: str | Path,
    *,
    vae_path: str | Path | None = None,
    video_vae_path: str | Path | None = None,
    audio_vae_path: str | Path | None = None,
    video_vae_weights: str | Path | None = None,
    audio_vae_weights: str | Path | None = None,
    device: str | torch.device = "cpu",
    video_compute_dtype: torch.dtype | str | None = None,
    audio_compute_dtype: torch.dtype | str = torch.float32,
    use_video_tiling: bool = True,
    strict: bool = True,
    low_memory: bool = True,
) -> H3VAEBundle:
    """Load both native H3 VAEs.

    ``video_vae_path``/``audio_vae_path`` select the two components
    independently, which is what the dual VAE loader node emits.  ``vae_path``
    remains supported for a merged package that carries both, and is used as
    the fallback root for whichever side was not named explicitly.
    """

    selected_root = Path(model_root).expanduser().resolve()
    component_root = (
        _selected_component_root(
            selected_root, vae_path, label="Selected merged VAE"
        )
        or selected_root
    )
    video_root = (
        _selected_component_root(
            selected_root, video_vae_path, label="Selected video VAE"
        )
        or component_root
    )
    audio_root = (
        _selected_component_root(
            selected_root, audio_vae_path, label="Selected audio VAE"
        )
        or component_root
    )

    video = load_video_vae(
        video_root,
        device=device,
        compute_dtype=video_compute_dtype,
        use_tiling=use_video_tiling,
        strict=strict,
        low_memory=low_memory,
        weight_files=(video_vae_weights,) if video_vae_weights else None,
    )
    try:
        audio = load_audio_vae(
            audio_root,
            device=device,
            compute_dtype=audio_compute_dtype,
            strict=strict,
            low_memory=low_memory,
            weight_files=(audio_vae_weights,) if audio_vae_weights else None,
        )
    except Exception:
        # Avoid leaving the much larger visual decoder resident after a failed
        # second-component load.
        video.offload()
        raise
    return H3VAEBundle(video_vae=video, audio_vae=audio)


__all__ = [
    "AUDIO_LATENT_CHANNELS",
    "AUDIO_OUTPUT_CHANNELS",
    "AUDIO_SAMPLE_RATE",
    "H3LatentStats",
    "H3VAEBundle",
    "H3VAEError",
    "MiniMaxH3AudioVAEAdapter",
    "MiniMaxH3VideoVAEAdapter",
    "VIDEO_LATENT_CHANNELS",
    "load_audio_vae",
    "load_h3_vae_bundle",
    "load_video_vae",
    "resolve_h3_component_dir",
]
