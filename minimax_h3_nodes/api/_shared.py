"""Shared node helpers."""
from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import tempfile
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import torch
except ImportError:  # Keep import errors actionable during Comfy start-up.
    torch = None  # type: ignore[assignment]

try:
    import comfy.model_management as model_management
    from comfy.utils import ProgressBar
except ImportError:
    model_management = None
    ProgressBar = None

from ..contracts import (
    H3_ASPECT_RATIOS,
    H3_AUDIO_CHANNELS,
    H3_AV_LATENT_SCHEMA,
    H3_AV_LATENT_SCHEMA_V2,
    H3_CONDITIONING_SCHEMA,
    H3_CONDITIONING_SCHEMA_V2,
    H3_DEFAULT_AUDIO_SHIFT,
    H3_DEFAULT_SIGMA_POINTS,
    H3_DEFAULT_VIDEO_SHIFT,
    H3_FINITE_ASPECT_RATIOS,
    H3_MODEL_SCHEMA,
    H3_MODEL_SCHEMA_V2,
    H3_REF2VA_PARTITION,
    H3_REFERENCE_LIST_SCHEMA_V2,
    H3_TASK_FL2VA,
    H3_TASK_REF2VA,
    H3_TASK_T2VA,
    H3_TARGET_SCHEMA,
    H3_TARGET_SCHEMA_V2,
    H3_TEXT_ENCODER_SCHEMA,
    H3_TEXT_ENCODER_SCHEMA_V2,
    H3_T2VA_PARTITION,
    H3_VAE_SCHEMA,
    H3_VAE_SCHEMA_V2,
    H3TaskNotImplementedError,
    append_ref2va_reference,
    compute_component_fingerprint,
    compute_release_fingerprint,
    condition_order_fingerprint,
    make_conditioning_v2,
    make_fl2va_keyframe,
    make_ref2va_reference,
    make_t2va_conditioning,
    normalize_task,
    partition_for_task,
    require_t2va,
    resolve_fl2va_target_v2,
    resolve_ref2va_target_v2,
    resolve_t2va_target,
    validate_av_latent_v2,
    validate_av_latent,
    validate_component_compatibility,
    validate_component_for_task,
    validate_conditioning_v2,
    validate_fl2va_keyframes,
    validate_ref2va_references,
    validate_target_v2,
    validate_target,
)
from ..runtime.h3_settings import (
    ACCEL_MODE_CHOICES,
    ACCEL_OFF,
    AUDIO_VAE_DIRNAME,
    AUDIO_VAE_FILENAME,
    AUDIO_VAE_MODEL_NAME,
    BF16_TE_MODEL_NAME,
    CACHE_DIT_MC,
    CACHE_DIT_RDT_COOKBOOK,
    CACHE_DIT_WARMUP,
    H3_COMPONENT_KINDS,
    H3_REFERENCE_IMAGE_SIZE_MATCH,
    H3_REFERENCE_IMAGE_SIZE_MODES,
    INT8_DIT_DIRNAME,
    INT8_TE_DIRNAME,
    INT8_TE_FILENAME,
    NATIVE_AUDIO_VAE_FILENAME,
    NATIVE_FL2VA_DIT_FILENAME,
    NATIVE_REF2VA_DIT_FILENAME,
    NATIVE_TE_FILENAME,
    NATIVE_VIDEO_VAE_FILENAME,
    SAMPLER_MODE_CHOICES,
    SAMPLER_MODE_EULER,
    VAE_MERGED_DIRNAME,
    VAE_MERGED_MODEL_NAME,
    VELOCITY_STRIDE,
    VIDEO_VAE_DIRNAME,
    VIDEO_VAE_FILENAME,
    VIDEO_VAE_MODEL_NAME,
    bf16_dit_model_name,
    int8_dit_filename,
    is_comfy_native_weight_filename,
)
from ..sampling import sample_h3

CATEGORY = "RunningHub/MiniMax H3"
NATIVE_MODEL_ROOT_LABEL = "Native Comfy model folders"
MODEL_TYPE = "MINIMAX_H3_DIRECT_MODEL"
TEXT_ENCODER_TYPE = "MINIMAX_H3_TEXT_ENCODER"
VAE_BUNDLE_TYPE = "MINIMAX_H3_VAE_BUNDLE"
TARGET_TYPE = "MINIMAX_H3_TARGET"
CONDITIONING_TYPE = "MINIMAX_H3_CONDITIONING"
AV_LATENT_TYPE = "MINIMAX_H3_AV_LATENT"
FL_KEYFRAMES_TYPE = "MINIMAX_H3_FL_KEYFRAMES"
REFERENCES_TYPE = "MINIMAX_H3_REFERENCES"


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "MiniMax-H3 Direct requires the PyTorch bundled with ComfyUI; torch is not installed in this Python environment"
        )
    return torch


def _require_comfy():
    if model_management is None:
        raise RuntimeError(
            "This node can run only inside the ComfyUI process; comfy.model_management was not found"
        )


def _h3_temporary_directory(prefix: str):
    """Allocate media scratch space under ComfyUI's managed temp root."""

    _require_comfy()
    try:
        import folder_paths
    except ImportError as exc:
        raise RuntimeError("ComfyUI folder_paths is unavailable, so the H3 temporary directory cannot be created") from exc
    temp_root = Path(folder_paths.get_temp_directory()).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    safe_prefix = str(prefix).strip() or "h3_"
    if not safe_prefix.startswith("h3_"):
        safe_prefix = f"h3_{safe_prefix}"
    return tempfile.TemporaryDirectory(prefix=safe_prefix, dir=str(temp_root))


def _new_progress(total: int):
    return ProgressBar(int(total)) if ProgressBar is not None else None


def _advance_progress(progress: Any, current: int, total: int) -> None:
    if progress is not None:
        progress.update_absolute(int(current), int(total))


def _check_interrupted() -> None:
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


def _runtime_module(name: str):
    """Import a sibling runtime module without any SGLang fallback."""
    # Comfy loads this as custom_nodes.<plugin_name>.minimax_h3_nodes.api, so the
    # top-level minimax_h3_nodes package name cannot be hard-coded.
    errs: list[BaseException] = []
    for modname, package in (
        (f"..runtime.{name}", __package__),  # api -> runtime
        (f".runtime.{name}", (__package__ or "").rsplit(".", 1)[0] or None),
        (f"minimax_h3_nodes.runtime.{name}", None),
    ):
        if not package and modname.startswith("."):
            continue
        try:
            return (
                importlib.import_module(modname, package=package)
                if package
                else importlib.import_module(modname)
            )
        except ImportError as exc:
            errs.append(exc)
    raise RuntimeError(
        f"MiniMax-H3 Direct runtime.{name} is incomplete: {errs[-1] if errs else 'unknown'}. "
        "Confirm that the custom node directory contains the native runtime code, not the legacy server-side node package."
    ) from (errs[-1] if errs else None)


def _model_root_choices() -> list[str]:
    """Return Loader COMBO options, falling back to a placeholder so object_info can still be generated."""
    try:
        module = _runtime_module("components")
        choices: list[str] = []
        lister = getattr(module, "list_h3_model_roots", None)
        if callable(lister):
            choices = list(lister())
        weight_lister = getattr(module, "list_h3_weight_files", None)
        if callable(weight_lister):
            native_available = any(
                is_comfy_native_weight_filename(path.name, kind)
                for kind in H3_COMPONENT_KINDS
                for path in weight_lister(kind)
            )
            if native_available:
                choices = [NATIVE_MODEL_ROOT_LABEL] + [
                    value for value in choices if value != NATIVE_MODEL_ROOT_LABEL
                ]
        if choices:
            return choices
        logging.getLogger(__name__).warning(
            "runtime.components does not expose list_h3_model_roots; INPUT_TYPES is using the MiniMax-H3 placeholder"
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "list_h3_model_roots failed; INPUT_TYPES is using the MiniMax-H3 placeholder: %s", exc
        )
    return ["MiniMax-H3"]


_COMPONENT_KIND_LABELS = {
    "transformer": "DiT",
    "text_encoder": "text encoder",
    VIDEO_VAE_DIRNAME: "Video VAE",
    AUDIO_VAE_DIRNAME: "Audio VAE",
}


def _normalized_kind(kind: str) -> str:
    normalized = str(kind).strip().lower()
    if normalized not in H3_COMPONENT_KINDS:
        raise ValueError(
            f"Unknown MiniMax-H3 component type {kind!r}; choices: {H3_COMPONENT_KINDS!r}"
        )
    return normalized


def _legacy_partition_roots(root: Path, partition_name: str) -> tuple[Path, ...]:
    """Partition candidates inside one legacy release root, by name only."""

    basename = "FL2VA" if partition_name == H3_T2VA_PARTITION else "Ref2VA"
    desired_children = (root / basename, root / partition_name)
    known_children = tuple(
        root / name for name in ("FL2VA", "fl2va", "Ref2VA", "ref2va")
    )
    if root.name.lower() == partition_name and root.is_dir():
        return (root,)
    existing_desired = tuple(child for child in desired_children if child.is_dir())
    if existing_desired:
        return existing_desired
    if not any(child.is_dir() for child in known_children):
        # Metadata-free single-partition development bundle.
        return (root,) if root.is_dir() else ()
    return ()


def _partition_roots(partition: str = H3_T2VA_PARTITION) -> list[Path]:
    """Return release partition candidates using directory names only.

    ``INPUT_TYPES`` runs during object-definition/manifest generation.  It
    must not call the execution-time release resolver, which may parse
    ``model_index.json``.  Strict metadata admission remains in ``load()``.
    """

    partition_name = str(partition).strip().lower()
    if partition_name not in {H3_T2VA_PARTITION, H3_REF2VA_PARTITION}:
        return []
    module = _runtime_module("components")
    lister = getattr(module, "list_h3_model_root_paths", None)
    roots = list(lister()) if callable(lister) else []
    out: list[Path] = []
    seen: set[str] = set()
    for raw_root in roots:
        for candidate in _legacy_partition_roots(
            Path(raw_root).expanduser(), partition_name
        ):
            resolved = candidate.resolve()
            if str(resolved) not in seen:
                seen.add(str(resolved))
                out.append(resolved)
    return out


def _weight_file_choices(
    kind: str,
    partition: str = H3_T2VA_PARTITION,
) -> list[Path]:
    """Single-file weights from dedicated and standard Comfy model folders."""

    module = _runtime_module("components")
    lister = getattr(module, "list_h3_weight_files", None)
    if not callable(lister):
        return []
    try:
        return list(lister(_normalized_kind(kind), partition))
    except Exception:
        return []


def _partition_label(partition: str = H3_T2VA_PARTITION) -> str:
    return "FL2VA" if str(partition).strip().lower() == H3_T2VA_PARTITION else "Ref2VA"


def _default_dit_model_name(partition: str = H3_T2VA_PARTITION) -> str:
    return (
        NATIVE_FL2VA_DIT_FILENAME
        if str(partition).strip().lower() == H3_T2VA_PARTITION
        else NATIVE_REF2VA_DIT_FILENAME
    )


def _default_te_model_name() -> str:
    return NATIVE_TE_FILENAME


def _default_vae_model_name() -> str:
    return VAE_MERGED_MODEL_NAME


def _default_video_vae_model_name() -> str:
    return NATIVE_VIDEO_VAE_FILENAME


def _default_audio_vae_model_name() -> str:
    return NATIVE_AUDIO_VAE_FILENAME


def _default_model_name(kind: str, partition: str = H3_T2VA_PARTITION) -> str:
    return {
        "transformer": lambda: _default_dit_model_name(partition),
        "text_encoder": _default_te_model_name,
        VIDEO_VAE_DIRNAME: _default_video_vae_model_name,
        AUDIO_VAE_DIRNAME: _default_audio_vae_model_name,
    }[_normalized_kind(kind)]()


def _is_sharded_weight(name: str) -> bool:
    return "-of-" in name or name.endswith(".index.json")


def _single_weight_names(component_dir: Path) -> list[str]:
    """Non-sharded weight filenames, including the video VAE's ``source/``."""

    names: list[str] = []
    for base in (component_dir, component_dir / "source"):
        try:
            names.extend(
                path.name
                for path in base.glob("*.safetensors")
                if path.is_file() and not _is_sharded_weight(path.name)
            )
        except OSError:
            continue
    return sorted(set(names))


def _read_component_config(path: Path) -> dict[str, Any] | None:
    """Read one component ``config.json`` without raising during INPUT_TYPES."""

    try:
        config_path = path / "config.json"
        if not config_path.is_file():
            return None
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _component_kind_of(path: Path) -> str | None:
    """Classify a component directory by its config, not by its parent name.

    The dedicated root lets a user name a component directory anything, so the
    COMBO must not infer the type from a ``transformer``/``text_encoder``
    prefix.  Classification mirrors the runtime loaders' own admission checks.
    """

    if not path.is_dir():
        return None
    config = _read_component_config(path)
    if config is None:
        return None
    architecture = config
    for wrapper in ("arch_config", "transformer_config", "dit_config"):
        nested = config.get(wrapper)
        if isinstance(nested, dict):
            architecture = nested
            break
    class_name = str(config.get("_class_name") or "")
    if class_name == "MiniMaxH3DiTModel":
        return "transformer"
    if class_name == "MiniMaxH3VideoVAE":
        return VIDEO_VAE_DIRNAME
    if class_name == "MiniMaxH3AudioVAE":
        return AUDIO_VAE_DIRNAME
    if str(config.get("model_type") or "") == "qwen3_vl":
        return "text_encoder"
    architectures = config.get("architectures")
    if isinstance(architectures, (list, tuple)) and any(
        str(name) == "Qwen3VLForConditionalGeneration" for name in architectures
    ):
        return "text_encoder"
    # When the upstream DiT config lacks _class_name, identify it from structural
    # fields (same source as model_loader's _TRANSFORMER_CONFIG_FIELDS, using only
    # the three most discriminating fields).
    if all(
        field in architecture
        for field in ("latents_dim", "audio_latents_dim", "adaln_out_features")
    ):
        return "transformer"
    lowered = path.name.strip().lower()
    if lowered == VIDEO_VAE_DIRNAME or (path / "source" / "config.json").is_file():
        return VIDEO_VAE_DIRNAME
    if lowered == AUDIO_VAE_DIRNAME:
        return AUDIO_VAE_DIRNAME
    return None


def _is_valid_component_dir(path: Path, kind: str) -> bool:
    return _component_kind_of(path) == _normalized_kind(kind)


def _model_name_for_component_dir(
    component_dir: Path, kind: str, partition: str
) -> str:
    """Map a component directory to the COMBO model name."""

    kind = _normalized_kind(kind)
    part = _partition_label(partition)
    dirname = component_dir.name
    singles = _single_weight_names(component_dir)
    # Generic filenames carry no type information; fall back to the logical name
    # so the video/audio dropdowns do not contain identical entries.
    distinctive = [name for name in singles if name != "model.safetensors"]
    if kind == "transformer":
        preferred = int8_dit_filename(part)
        if (component_dir / preferred).is_file():
            return preferred
        if distinctive:
            return distinctive[0]
        if dirname == INT8_DIT_DIRNAME:
            return preferred
        return bf16_dit_model_name(part) if dirname == "transformer" else dirname
    if kind == "text_encoder":
        if (component_dir / INT8_TE_FILENAME).is_file():
            return INT8_TE_FILENAME
        if distinctive:
            return distinctive[0]
        if dirname == INT8_TE_DIRNAME:
            return INT8_TE_FILENAME
        return BF16_TE_MODEL_NAME if dirname == "text_encoder" else dirname
    if distinctive:
        return distinctive[0]
    return (
        VIDEO_VAE_MODEL_NAME if kind == VIDEO_VAE_DIRNAME else AUDIO_VAE_MODEL_NAME
    )


def _model_name_for_vae_dir(component_dir: Path) -> str:
    if component_dir.name == VAE_MERGED_DIRNAME:
        return VAE_MERGED_MODEL_NAME
    return component_dir.name


def _selector_alias_map(kind: str, partition: str) -> dict[str, str]:
    """model name / legacy dir name → relative component directory."""

    part = _partition_label(partition)
    if kind == "transformer":
        return {
            bf16_dit_model_name(part): "transformer",
            int8_dit_filename(part): INT8_DIT_DIRNAME,
            "transformer": "transformer",
            INT8_DIT_DIRNAME: INT8_DIT_DIRNAME,
        }
    if kind == "text_encoder":
        return {
            BF16_TE_MODEL_NAME: "text_encoder",
            INT8_TE_FILENAME: INT8_TE_DIRNAME,
            "text_encoder": "text_encoder",
            INT8_TE_DIRNAME: INT8_TE_DIRNAME,
        }
    if kind == VIDEO_VAE_DIRNAME:
        return {
            VIDEO_VAE_MODEL_NAME: VIDEO_VAE_DIRNAME,
            VIDEO_VAE_FILENAME: VIDEO_VAE_DIRNAME,
            VIDEO_VAE_DIRNAME: VIDEO_VAE_DIRNAME,
        }
    if kind == AUDIO_VAE_DIRNAME:
        return {
            AUDIO_VAE_MODEL_NAME: AUDIO_VAE_DIRNAME,
            AUDIO_VAE_FILENAME: AUDIO_VAE_DIRNAME,
            AUDIO_VAE_DIRNAME: AUDIO_VAE_DIRNAME,
        }
    if kind == "vae":  # Combined dual-VAE package (legacy workflows).
        return {
            VAE_MERGED_MODEL_NAME: VAE_MERGED_DIRNAME,
            VAE_MERGED_DIRNAME: VAE_MERGED_DIRNAME,
        }
    return {}


# Component directories in the upstream release that carry config/tokenizer/processor
# files, ordered by priority. Flat single-file weights have no sidecar and obtain their
# configuration here.
_KIND_CONFIG_DIRNAMES = {
    "transformer": ("transformer",),
    "text_encoder": ("text_encoder",),
    VIDEO_VAE_DIRNAME: (
        VIDEO_VAE_DIRNAME,
        f"{VAE_MERGED_DIRNAME}/{VIDEO_VAE_DIRNAME}",
    ),
    AUDIO_VAE_DIRNAME: (
        AUDIO_VAE_DIRNAME,
        f"{VAE_MERGED_DIRNAME}/{AUDIO_VAE_DIRNAME}",
    ),
}


def _component_candidates(
    kind: str,
    partition: str = H3_T2VA_PARTITION,
) -> list[Path]:
    """Return release component directories of the requested kind.

    VAE components may sit one level deeper inside a merged ``vae/`` package,
    so that layer is scanned as well.  Everything else stays at depth 1 to keep
    ``INPUT_TYPES`` cheap.
    """

    kind = _normalized_kind(kind)
    nested = kind in (VIDEO_VAE_DIRNAME, AUDIO_VAE_DIRNAME)
    out: list[Path] = []
    seen: set[str] = set()

    def _offer(candidate: Path) -> None:
        if not _is_valid_component_dir(candidate, kind):
            return
        resolved = candidate.resolve()
        if str(resolved) not in seen:
            seen.add(str(resolved))
            out.append(resolved)

    for search_root in _partition_roots(partition):
        try:
            children = sorted(
                child for child in search_root.iterdir() if child.is_dir()
            )
        except OSError:
            continue
        for child in children:
            _offer(child)
            if not nested:
                continue
            try:
                grandchildren = sorted(
                    item for item in child.iterdir() if item.is_dir()
                )
            except OSError:
                continue
            for grandchild in grandchildren:
                _offer(grandchild)
    return out


def _is_below(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _prefer_selected_root(
    candidates: list[Path],
    model_root: str | Path | None,
) -> list[Path]:
    """Order candidates so the release the user actually picked resolves first."""

    if not model_root:
        return candidates
    selected = Path(str(model_root)).expanduser()
    if not selected.is_dir():
        return candidates
    selected = selected.resolve()
    preferred = [path for path in candidates if _is_below(selected, path)]
    return preferred + [path for path in candidates if path not in preferred]


def _config_component_dirname(
    kind: str,
    partition: str = H3_T2VA_PARTITION,
    model_root: str | Path | None = None,
) -> str:
    """Release-relative config directory backing a flat single-file weight."""

    kind = _normalized_kind(kind)
    options = _KIND_CONFIG_DIRNAMES[kind]
    roots = _prefer_selected_root(_partition_roots(partition), model_root)
    for root in roots:
        for relative in options:
            if (root / relative / "config.json").is_file():
                return relative
    return options[0]


def _selector_weight_file(
    value: str,
    kind: str,
    partition: str = H3_T2VA_PARTITION,
) -> Path | None:
    """Return the flat weight file a selection names, if it is one."""

    text = str(value or "").strip()
    if not text:
        return None
    module = _runtime_module("components")
    choice_name = getattr(module, "weight_file_choice_name", None)
    for candidate in _weight_file_choices(kind, partition):
        display = (
            str(choice_name(candidate, kind))
            if callable(choice_name)
            else candidate.name
        )
        if display == text or candidate.name == text:
            return candidate
    return None


def _native_component_selection(
    value: str,
    kind: str,
    partition: str,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, float]] | None:
    """Resolve a standard Comfy single-file checkpoint and its local descriptor.

    Native FP8/NVFP4/VAE files intentionally live in ComfyUI's normal model
    folders rather than a copied Diffusers release tree.  Small package-owned
    descriptors provide the component paths required by the H3 wrapper
    contract; the selected safetensors file remains the external identity and
    source of weights.
    """

    normalized_kind = _normalized_kind(kind)
    normalized_partition = str(partition).strip().lower()
    if not is_comfy_native_weight_filename(value, normalized_kind):
        return None
    module = _runtime_module("components")
    resolver = _find_callable(
        module,
        ("resolve_weight_file",),
        _COMPONENT_KIND_LABELS[normalized_kind],
    )
    gate = _find_callable(module, ("validate_weight_partition",), "weight partition")
    weights = Path(
        gate(
            resolver(value, normalized_kind, normalized_partition),
            normalized_partition,
            kind=normalized_kind,
        )
    ).resolve()
    root = (
        Path(__file__).resolve().parents[1]
        / "runtime"
        / "native_components"
        / normalized_partition
    ).resolve()
    component = (root / normalized_kind).resolve()
    if not (component / "config.json").is_file():
        raise RuntimeError(
            f"Native MiniMax-H3 descriptor is missing for {normalized_kind}: {component}"
        )
    return (
        root,
        component,
        weights,
        {},
        {
            "video": float(H3_DEFAULT_VIDEO_SHIFT),
            "audio": float(H3_DEFAULT_AUDIO_SHIFT),
        },
    )


def _selector_to_component_dirname(
    value: str,
    prefix: str,
    partition: str = H3_T2VA_PARTITION,
    model_root: str | Path | None = None,
) -> str:
    """Resolve a COMBO selection to the release-relative component directory.

    A flat single-file weight carries no config, so it maps to the release
    directory that does; a release component maps to itself.
    """

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{prefix} model name cannot be empty")
    if prefix == "vae":
        # Combined dual-VAE package: retain only the directory-name resolution path
        # for legacy workflows.
        for root in _prefer_selected_root(_partition_roots(partition), model_root):
            if (root / text / "video_vae" / "config.json").is_file():
                return text
    else:
        kind = _normalized_kind(prefix)
        if _selector_weight_file(text, kind, partition) is not None:
            return _config_component_dirname(kind, partition, model_root)
        for component_dir in _prefer_selected_root(
            _component_candidates(kind, partition), model_root
        ):
            if (
                component_dir.name == text
                or (component_dir / text).is_file()
                or (component_dir / "source" / text).is_file()
                or _model_name_for_component_dir(component_dir, kind, partition)
                == text
            ):
                for root in _partition_roots(partition):
                    if _is_below(root, component_dir):
                        return component_dir.relative_to(root).as_posix()
    aliases = _selector_alias_map(prefix, partition)
    if text in aliases:
        return aliases[text]
    if prefix in ("transformer", "text_encoder") and text.startswith(prefix):
        return text
    if prefix == "vae" and text == VAE_MERGED_DIRNAME:
        return text
    label = _COMPONENT_KIND_LABELS.get(prefix, prefix)
    raise ValueError(f"Cannot resolve the {label} model name: {text}")


def _component_model_choices(
    prefix: str,
    partition: str = H3_T2VA_PARTITION,
) -> list[str]:
    """List selectable model names of one kind; never expose an auto mode.

    Three sources feed one dropdown: native files in standard Comfy model
    folders, RunningHub files below ``models/MiniMax-H3`` (both classified by
    filename), and sharded release components below ``models/diffusers/...``.
    """

    kind = _normalized_kind(prefix)
    preferred = _default_model_name(kind, partition)
    names: set[str] = set()
    try:
        module = _runtime_module("components")
        choice_name = getattr(module, "weight_file_choice_name", None)
        names.update(
            str(choice_name(path, kind)) if callable(choice_name) else path.name
            for path in _weight_file_choices(kind, partition)
        )
        for component_dir in _component_candidates(kind, partition):
            names.add(_model_name_for_component_dir(component_dir, kind, partition))
    except Exception:
        pass
    if not names:
        return [preferred]
    unquantized = {
        bf16_dit_model_name(_partition_label(partition)),
        BF16_TE_MODEL_NAME,
        VIDEO_VAE_MODEL_NAME,
        AUDIO_VAE_MODEL_NAME,
    }
    return sorted(
        names,
        key=lambda name: (name != preferred, name in unquantized, name),
    )


def _component_dir_input(
    prefix: str,
    label: str,
    partition: str = H3_T2VA_PARTITION,
):
    choices = _component_model_choices(prefix, partition)
    return (
        choices,
        {
            "default": choices[0],
            "tooltip": (
                f"You must explicitly select a {label} model name (weight filename or logical name); "
                "native FP8/NVFP4, RunningHub INT8-CONVROT, and BF16 weights are never switched automatically."
            ),
        },
    )


def _video_vae_input(partition: str = H3_T2VA_PARTITION):
    choices = _component_model_choices(VIDEO_VAE_DIRNAME, partition)
    return (
        choices,
        {
            "default": choices[0],
            "tooltip": (
                "24-channel video VAE weights from Comfy's normal VAE folder or a RunningHub release; "
                f"legacy names are {VIDEO_VAE_FILENAME} and {VIDEO_VAE_MODEL_NAME}."
            ),
        },
    )


def _audio_vae_input(partition: str = H3_T2VA_PARTITION):
    choices = _component_model_choices(AUDIO_VAE_DIRNAME, partition)
    return (
        choices,
        {
            "default": choices[0],
            "tooltip": (
                "32-channel audio VAE weights from Comfy's normal VAE folder or a RunningHub release; "
                f"legacy names are {AUDIO_VAE_FILENAME} and {AUDIO_VAE_MODEL_NAME}."
            ),
        },
    )


def _model_root_input(partition: str | None = None):
    partition_hint = (
        f"; this node always resolves the {partition.upper()} partition"
        if partition is not None
        else ""
    )
    return (
        _model_root_choices(),
        {
            "tooltip": (
                "Select the MiniMax-H3 release root for RunningHub INT8-CONVROT or BF16 shards. "
                "When you select native FP8/NVFP4/VAE single files from Comfy's standard model folders, "
                f"this legacy root value is ignored{partition_hint}. All three components must declare the same partition."
            ),
        },
    )


def _validate_model_root_input(model_root="MiniMax-H3", **kwargs):
    # RunningHub/workflows may inject a real directory name that is absent from the
    # local COMBO list; allow it and resolve it at execution time.
    if model_root is None:
        return True
    if not str(model_root).strip():
        return "model_root cannot be empty"
    return True


def _validate_explicit_component_input(value, label: str):
    # A dynamically connected input cannot be validated until execution.
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return f"{label} must explicitly select a model; auto and empty values are not allowed"
    return True


def _existing_directory(
    value: str,
    label: str = "model_root",
    *,
    partition: str | None = None,
    required_component: str | Path | None = None,
    required_files: tuple[str, ...] = (),
) -> Path:
    path = Path(str(value or "").strip()).expanduser()
    if not str(value or "").strip():
        raise ValueError(f"{label} cannot be empty")
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")
    if path.is_absolute() and not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")

    # Keep the node UI aligned with runtime.components.model_root_path(): a
    # A simple folder name below models/diffusers or models/minimax_h3 is also valid.
    module = _runtime_module("components")
    resolver = getattr(module, "model_root_path", None)
    if callable(resolver):
        try:
            return Path(
                resolver(
                    value,
                    partition=partition,
                    required_component=required_component,
                    required_files=required_files,
                )
            ).resolve()
        except (ValueError, OSError) as exc:
            component_error = getattr(module, "H3ComponentError", None)
            if component_error is not None and isinstance(exc, component_error):
                raise
            raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if path.is_dir():
        return path.resolve()
    raise FileNotFoundError(f"{label} does not exist: {path}")


def _resolve_t2va_release(
    value: str,
    *,
    required_component: str | Path | None = None,
    required_files: tuple[str, ...] = (),
) -> tuple[Path, dict[str, Any], dict[str, float]]:
    """Resolve/validate the FL2VA release before any large component loads."""

    root = _existing_directory(
        value,
        partition=H3_T2VA_PARTITION,
        required_component=required_component,
        required_files=required_files,
    )
    module = _runtime_module("components")
    resolve_partition = _find_callable(
        module,
        ("resolve_partition_root",),
        "partition root",
    )
    root = Path(
        resolve_partition(
            root,
            H3_T2VA_PARTITION,
            required_component=required_component,
            required_files=required_files,
        )
    ).resolve()
    metadata_reader = _find_callable(
        module,
        ("release_metadata",),
        "release metadata",
    )
    validator = _find_callable(
        module,
        ("validate_t2va_partition",),
        "T2VA partition validator",
    )
    sigma_reader = _find_callable(
        module,
        ("release_sigma_shift_scales",),
        "sigma-shift metadata parser",
    )
    metadata = metadata_reader(root)
    validator(metadata)
    declared_scales = sigma_reader(metadata)
    sigma_scales = declared_scales or {
        "video": float(H3_DEFAULT_VIDEO_SHIFT),
        "audio": float(H3_DEFAULT_AUDIO_SHIFT),
    }
    return root, dict(metadata), dict(sigma_scales)


def _resolve_release(
    value: str,
    *,
    task: str,
    required_component: str | Path | None = None,
    required_files: tuple[str, ...] = (),
) -> tuple[Path, str, dict[str, Any], dict[str, float]]:
    """Resolve one official task partition before opening checkpoint tensors."""

    normalized_task = normalize_task(task)
    partition = partition_for_task(normalized_task)
    root = _existing_directory(
        value,
        partition=partition,
        required_component=required_component,
        required_files=required_files,
    )
    module = _runtime_module("components")
    root = Path(
        module.resolve_partition_root(
            root,
            partition,
            required_component=required_component,
            required_files=required_files,
        )
    ).resolve()
    metadata = dict(module.release_metadata(root))
    # task and partition are correctness-critical.  Do not route this through
    # _call_supported(), which is intentionally allowed to drop optional args.
    module.validate_task_partition(metadata, normalized_task, partition)
    declared_scales = module.release_sigma_shift_scales(metadata)
    sigma_scales = declared_scales or {
        "video": float(H3_DEFAULT_VIDEO_SHIFT),
        "audio": float(H3_DEFAULT_AUDIO_SHIFT),
    }
    return root, partition, metadata, dict(sigma_scales)


def _release_fingerprint(
    root: str | Path,
    partition: str,
    metadata: Mapping[str, Any],
) -> str:
    """Stable in-process identity for cross-node release compatibility checks."""

    return compute_release_fingerprint(
        str(Path(root).resolve()),
        str(partition).strip().lower(),
        metadata,
    )


def _component_fingerprint(
    release_fingerprint: str,
    component_kind: str,
    component_path: str | Path,
    *,
    related_paths: Mapping[str, str | Path] | None = None,
) -> str:
    return compute_component_fingerprint(
        release_fingerprint,
        component_kind,
        str(Path(component_path).resolve()),
        related_paths=related_paths,
    )


def _wrapper_release_fingerprint(wrapper: Mapping[str, Any]) -> str:
    root = wrapper.get("model_root")
    partition = wrapper.get("partition")
    metadata = wrapper.get("release_metadata", {})
    if root is None or partition is None or not isinstance(metadata, Mapping):
        raise ValueError("MiniMax-H3 component wrapper is missing its release identity")
    computed = _release_fingerprint(root, str(partition), metadata)
    declared = wrapper.get("release_fingerprint")
    if declared is not None and declared != computed:
        raise ValueError("MiniMax-H3 component release_fingerprint was modified or is stale")
    return computed


# Must match the union of contracts._impl's _H3_COMPONENT_RELATED_PATH_FIELDS and
# _H3_COMPONENT_EXTERNAL_PATH_FIELDS: component fingerprints computed on both sides
# are compared exactly. Keys are fingerprint kinds (values of contracts'
# _H3_COMPONENT_FINGERPRINT_KINDS), not contract component_kind values: DiT is named
# "transformer" here rather than "model".
_WRAPPER_RELATED_PATH_FIELDS = {
    "transformer": ("transformer_weights_path",),
    "text_encoder": (
        "tokenizer_path",
        "processor_path",
        "text_encoder_weights_path",
    ),
    "vae": (
        "audio_vae_path",
        "video_vae_weights_path",
        "audio_vae_weights_path",
    ),
}


def _wrapper_component_fingerprint(
    wrapper: Mapping[str, Any],
    *,
    component_kind: str,
    path_key: str,
    fingerprint_key: str,
) -> str:
    path = wrapper.get(path_key)
    if not isinstance(path, str) or not path:
        raise ValueError(f"MiniMax-H3 {component_kind} wrapper is missing {path_key}")
    computed = _component_fingerprint(
        _wrapper_release_fingerprint(wrapper),
        component_kind,
        path,
        related_paths={
            key.removesuffix("_path"): value
            for key in _WRAPPER_RELATED_PATH_FIELDS.get(component_kind, ())
            if isinstance((value := wrapper.get(key)), str) and value
        }
        or None,
    )
    declared = wrapper.get(fingerprint_key, wrapper.get("component_fingerprint"))
    if declared is not None and declared != computed:
        raise ValueError(
            f"MiniMax-H3 {component_kind} component fingerprint was modified or is stale"
        )
    return computed


def _validate_component_wrapper(
    wrapper: Any,
    *,
    task: str,
    label: str,
    schemas: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(wrapper, Mapping) or wrapper.get("schema") not in tuple(schemas):
        raise TypeError(
            f"{label} port schema is invalid; expected {', '.join(repr(item) for item in schemas)}"
        )
    normalized_task = normalize_task(task)
    kind_by_label = {
        "h3_model": "model",
        "h3_text_encoder": "text_encoder",
        "h3_vae_bundle": "vae",
    }
    try:
        component_kind = kind_by_label[label]
    except KeyError as exc:
        raise ValueError(f"Unknown MiniMax-H3 component label {label!r}") from exc
    clean = validate_component_for_task(
        wrapper,
        component_kind=component_kind,
        task=normalized_task,
    )
    fingerprint = _wrapper_release_fingerprint(clean)
    declared = clean.get("release_fingerprint")
    if declared is not None and declared != fingerprint:
        raise ValueError(f"{label}.release_fingerprint does not match the component paths/metadata")
    clean["release_fingerprint"] = fingerprint
    return clean


def _require_same_release(*components: tuple[str, Mapping[str, Any]]) -> str:
    fingerprints = {
        label: _wrapper_release_fingerprint(wrapper)
        for label, wrapper in components
    }
    if len(set(fingerprints.values())) != 1:
        detail = ", ".join(f"{label}={value[:12]}" for label, value in fingerprints.items())
        raise ValueError(f"MiniMax-H3 components are not from the same release: {detail}")
    return next(iter(fingerprints.values()))


def _resolve_selected_component(
    root: Path,
    value: str,
    *,
    keys: tuple[str, ...],
    label: str,
    partition: str = H3_T2VA_PARTITION,
    required_files: tuple[str, ...] = (),
    kind: str | None = None,
) -> Path:
    validation = _validate_explicit_component_input(value, label)
    if validation is not True:
        raise ValueError(validation)
    prefix = kind or ("vae" if VAE_MERGED_DIRNAME in keys else str(keys[0]))
    dirname = _selector_to_component_dirname(value, prefix, partition)
    module = _runtime_module("components")
    resolver = _find_callable(module, ("resolve_component",), label)
    resolved = Path(
        resolver(
            root,
            keys,
            explicit=dirname,
            required_files=required_files,
        )
    ).resolve()
    weights: Path | None = None
    if prefix != "vae":
        weights = _selector_weight_file(value, prefix, partition)
        if weights is not None:
            # Flat roots have no quant_meta.json; the filename is the partition evidence.
            gate = _find_callable(
                module, ("validate_weight_partition",), "weight partition"
            )
            weights = Path(gate(weights, partition, kind=prefix)).resolve()
    return resolved, weights


def _resolve_selected_vae_component(
    root: Path,
    value: str,
    *,
    kind: str,
    partition: str = H3_T2VA_PARTITION,
) -> tuple[Path, Path | None]:
    return _resolve_selected_component(
        root,
        value,
        keys=(kind,),
        label=_COMPONENT_KIND_LABELS[kind],
        partition=partition,
        required_files=("config.json",),
        kind=kind,
    )


def _resolve_selected_vae(
    root: Path, value: str, *, partition: str = H3_T2VA_PARTITION
) -> Path:
    """Resolve a legacy merged dual-VAE package directory."""

    component, _weights = _resolve_selected_component(
        root,
        value,
        keys=(VAE_MERGED_DIRNAME,),
        label="VAE",
        partition=partition,
        required_files=(
            "video_vae/config.json",
            "audio_vae/config.json",
        ),
    )
    return component


def _runtime_dtype_name(name: str, *, allow_float32: bool = True) -> str:
    """Normalise UI dtype values to the local runtime loader contract."""

    normalized = str(name).strip().lower()
    if normalized == "auto":
        # H3's released base weights are BF16.  Individual FP32 layers are
        # selected by the architecture/loader and must not be inferred here.
        return "bfloat16"
    allowed = {"bfloat16", "float16"}
    if allow_float32:
        allowed.add("float32")
    if normalized not in allowed:
        supported = ", ".join(sorted(allowed))
        raise ValueError(f"dtype {name!r} is unsupported; choices: auto, {supported}")
    return normalized


def _optional_path(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def _weights_field(name: str, value: Path | None) -> dict[str, str]:
    return {name: str(value)} if value is not None else {}


def _require_parameter(function, name: str, label: str) -> None:
    """Fail loudly when a runtime cannot accept an explicit weights override.

    ``_call_supported`` is allowed to drop optional arguments, but silently
    dropping the selected checkpoint would load the release's own weights
    instead of the file the user picked.
    """

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return
    if name not in signature.parameters:
        raise RuntimeError(
            f"The current runtime's {label} loader does not support {name}, so it cannot load "
            "single-file weights under models/MiniMax-H3; update the custom node runtime code"
        )


def _call_supported(function, **kwargs):
    """Call a runtime entry point while allowing harmless API growth."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return function(**accepted)


def _find_callable(module: Any, names: tuple[str, ...], component: str):
    for name in names:
        function = getattr(module, name, None)
        if callable(function):
            return function
    raise RuntimeError(
        f"runtime.{module.__name__.rsplit('.', 1)[-1]} has no {component} loader; "
        f"expected one of these entry points: {', '.join(names)}"
    )


def _devices() -> tuple[Any, Any]:
    _require_comfy()
    load_device = model_management.get_torch_device()
    offload_getter = getattr(model_management, "unet_offload_device", None)
    offload_device = (
        offload_getter() if callable(offload_getter) else _require_torch().device("cpu")
    )
    return load_device, offload_device


def _wrapper_value(wrapper: Any, *names: str) -> Any | None:
    if isinstance(wrapper, Mapping):
        for name in names:
            if name in wrapper and wrapper[name] is not None:
                return wrapper[name]
    for name in names:
        value = getattr(wrapper, name, None)
        if value is not None:
            return value
    return None


def _unwrap_runtime_handle(
    wrapper: Any,
    expected_schema: str | Sequence[str],
    label: str,
):
    schemas = (
        (expected_schema,)
        if isinstance(expected_schema, str)
        else tuple(expected_schema)
    )
    if not isinstance(wrapper, Mapping) or wrapper.get("schema") not in schemas:
        raise TypeError(
            f"The {label} port is not an output from a MiniMax-H3 Direct loader"
        )
    handle = wrapper.get("handle")
    if handle is None:
        raise RuntimeError(f"The {label} wrapper is missing its runtime handle")
    return handle


def _normalise_prompt_embeddings(value: Any):
    t = _require_torch()
    if isinstance(value, Mapping):
        for key in ("prompt_embeds", "text_embeddings", "last_hidden_state"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError("The Qwen encoder returned an empty sequence")
        value = value[0]
    if not isinstance(value, t.Tensor):
        raise TypeError(
            "The Qwen encoder must return a torch.Tensor or a mapping containing prompt_embeds"
        )
    if value.ndim == 3:
        if int(value.shape[0]) != 1:
            raise ValueError("MiniMax-H3 Direct v0 supports batch=1 only")
        value = value[0]
    if value.ndim != 2:
        raise ValueError(
            f"Qwen encoder output must be [L,5120]; got shape={tuple(value.shape)}"
        )
    # The 64 GB text encoder must not remain co-resident with the 62 GB DiT.
    return value.detach().to(device="cpu").contiguous()


def _encode_prompt(handle: Any, prompt: str):
    encode = _wrapper_value(handle, "encode_prompt", "encode_conditioning")
    if not callable(encode):
        raise RuntimeError(
            "The text encoder handle is missing encode_prompt(prompt); "
            "update minimax_h3_nodes/runtime/qwen_encoder.py"
        )
    try:
        try:
            result = _call_supported(
                encode,
                prompt=prompt,
                text=prompt,
                task=H3_TASK_T2VA,
                conditions=[],
            )
        except TypeError:
            # Keep the minimal, documented handle contract friendly to simple
            # implementations that use one positional-only prompt.
            result = encode(prompt)
        return _normalise_prompt_embeddings(result)
    finally:
        # The source runtime never co-resides the ~64 GB Qwen encoder with the
        # ~62 GB DiT.  Enforce that boundary even if encode_prompt raises.
        offload = _wrapper_value(
            handle,
            "offload_after_inference",
            "offload",
            "release_from_device",
        )
        if callable(offload):
            offload()


def _build_t2va_packed(prompt_embeds: Any, target: Mapping[str, Any]):
    module = _runtime_module("packing")
    build = _find_callable(
        module,
        (
            "build_t2va_packed_conditioning",
            "build_t2va_packed",
            "minimax_h3_packed_sequence",
        ),
        "T2VA packing",
    )
    packed = _call_supported(
        build,
        prompt_embeds=prompt_embeds,
        text_embeddings=prompt_embeds,
        text_len=int(prompt_embeds.shape[0]),
        latent_t=int(target["video_latent_t"]),
        latent_h=int(target["video_latent_h"]),
        latent_w=int(target["video_latent_w"]),
        audio_t=int(target["audio_latent_t"]),
        audio_channel=H3_AUDIO_CHANNELS,
        include_keyframe_cond=False,
        keyframe_frame_indices=None,
        frame_count=int(target["frame_count"]),
    )
    if not isinstance(packed, Mapping):
        raise TypeError("runtime.packing must return a packed conditioning mapping")
    # Some adapters return {"packed": structural_fields}; accept that without
    # weakening the strict structural checks in H3DenoiseBranch.
    nested = packed.get("packed")
    if isinstance(nested, Mapping):
        packed = nested
    return dict(packed)


def _build_v2_packed(
    conditioning: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    clean = validate_conditioning_v2(conditioning)
    clean_target = validate_target_v2(
        target, expected_task=str(clean["task"]), require_resolved=True
    )
    declared_target = clean.get("target")
    if declared_target is not None:
        encoded_target = validate_target_v2(
            declared_target,
            expected_task=str(clean["task"]),
            require_resolved=True,
        )
        if _target_fingerprint(encoded_target) != _target_fingerprint(clean_target):
            raise ValueError("The conditioning target does not match the Empty AV Latent target")
    declared_fingerprint = clean.get("target_fingerprint")
    actual_fingerprint = _target_fingerprint(clean_target)
    if declared_fingerprint is not None and declared_fingerprint != actual_fingerprint:
        raise ValueError("conditioning.target_fingerprint does not match the target contents")
    blocks = clean.get("condition_blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError("FL2VA/Ref2VA conditioning is missing ordered condition_blocks")
    module = _runtime_module("packing")
    common = {
        "latent_t": int(clean_target["video_latent_t"]),
        "latent_h": int(clean_target["video_latent_h"]),
        "latent_w": int(clean_target["video_latent_w"]),
        "audio_t": int(clean_target["audio_latent_t"]),
        "audio_channel": H3_AUDIO_CHANNELS,
        "text_token_tags": clean.get("text_token_tags"),
    }
    if clean["task"] == H3_TASK_FL2VA:
        packed = module.build_fl2va_packed_conditioning(
            clean["prompt_embeds"],
            frame_count=int(clean_target["frame_count"]),
            condition_blocks=blocks,
            **common,
        )
    elif clean["task"] == H3_TASK_REF2VA:
        packed = module.build_ref2va_packed_conditioning(
            clean["prompt_embeds"],
            condition_blocks=blocks,
            **common,
        )
    else:
        raise ValueError(f"The v2 packed builder does not support task={clean['task']!r}")
    if not isinstance(packed, Mapping):
        raise TypeError("The runtime.packing v2 builder must return a mapping")
    return dict(packed)


def _load_model_patcher_if_present(handle: Any) -> Any | None:
    patcher = _wrapper_value(handle, "model_patcher", "patcher")
    if patcher is None:
        return None
    _require_comfy()
    model_management.load_models_gpu([patcher])
    return patcher


def _resolve_transformer(handle: Any) -> Any:
    """Materialise a transformer through a runtime handle/ModelPatcher."""

    prepare = _wrapper_value(
        handle,
        "load_for_inference",
        "prepare_for_inference",
        "get_model",
    )
    prepared = prepare() if callable(prepare) else None
    if prepared is not None:
        candidate = _wrapper_value(prepared, "transformer", "model") or prepared
        if callable(candidate):
            return candidate

    patcher = _load_model_patcher_if_present(handle)
    if patcher is not None:
        candidate = _wrapper_value(handle, "transformer")
        if candidate is None:
            candidate = getattr(patcher, "model", None)
        # A BaseModel-style patcher may wrap the actual diffusion module.
        nested = _wrapper_value(
            candidate,
            "diffusion_model",
            "transformer",
            "model",
        )
        if callable(nested):
            candidate = nested
        if callable(candidate):
            return candidate

    candidate = _wrapper_value(handle, "transformer", "model", "module")
    if candidate is None and callable(handle):
        candidate = handle
    if not callable(candidate):
        raise RuntimeError(
            "Could not obtain MiniMaxH3DiTModel from the model handle; "
            "the handle must expose transformer/model or load_for_inference()"
        )

    # A simple runtime implementation may return an offloaded raw nn.Module.
    # Move device only; never cast the whole module because H3 deliberately
    # keeps selected projections/heads/RoPE state in fp32.
    try:
        parameter = next(candidate.parameters())
    except (AttributeError, StopIteration):
        parameter = None
    if parameter is not None and parameter.device.type == "cpu":
        load_device, _ = _devices()
        candidate.to(device=load_device)
    eval_method = getattr(candidate, "eval", None)
    if callable(eval_method):
        eval_method()
    return candidate


def _activation_hint_from_packed(
    packed: Mapping[str, Any], target: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Extract P0-6 activation-budget hints from packed/target."""
    target = target or {}
    seq_len = int(packed.get("seq_len", 0) or 0)
    update = packed.get("update_mask")
    ref_rows = 0
    if update is not None and hasattr(update, "numel"):
        try:
            ref_rows = int((~update.view(-1).bool()).sum().item())
        except Exception:
            ref_rows = 0
    return {
        "task": str(target.get("task") or packed.get("task") or ""),
        "seq_len": seq_len,
        "video_latent_t": int(target.get("video_latent_t", 0) or 0),
        "video_latent_h": int(target.get("video_latent_h", 0) or 0),
        "video_latent_w": int(target.get("video_latent_w", 0) or 0),
        "ref_rows": ref_rows,
    }


@contextmanager
def _transformer_session(
    model_wrapper: Mapping[str, Any],
    *,
    activation_hint: Mapping[str, Any] | None = None,
):
    handle = _unwrap_runtime_handle(
        model_wrapper, (H3_MODEL_SCHEMA, H3_MODEL_SCHEMA_V2), "h3_model"
    )
    try:
        # Acquisition can materialize a ModelPatcher or move a raw module to
        # GPU.  Keep it inside the cleanup boundary so partial load failures
        # cannot leave the huge DiT resident.
        setter = getattr(handle, "set_activation_hint", None)
        if callable(setter) and activation_hint:
            hint = dict(activation_hint)
            # The downgrade hint needs pixel dimensions.
            target_w = hint.get("width"); target_h = hint.get("height")
            if not target_w or not target_h:
                # Rough estimate: latent * 16.
                lt, lh, lw = hint.get("video_latent_t"), hint.get("video_latent_h"), hint.get("video_latent_w")
                if lh and lw:
                    hint["width"] = int(lw) * 16; hint["height"] = int(lh) * 16
            setter(**hint)
        try:
            from ..runtime.residency import get_residency_manager
            get_residency_manager().acquire(handle, kind="dit")
        except Exception:
            pass
        transformer = _resolve_transformer(handle)
        yield transformer, handle
    finally:
        try:
            from ..runtime.residency import get_residency_manager
            from ..runtime.h3_settings import OPT_RESIDENCY_LEASE
            if OPT_RESIDENCY_LEASE and hasattr(handle, "park_after_inference"):
                get_residency_manager().release(handle, kind="dit")
            else:
                offload = _wrapper_value(
                    handle, "offload_after_inference", "offload", "release_from_device",
                )
                if callable(offload):
                    offload()
        except Exception:
            offload = _wrapper_value(
                handle, "offload_after_inference", "offload", "release_from_device",
            )
            if callable(offload):
                offload()


@contextmanager
def _vae_device_session(adapter: Any, load_device: Any):
    """Keep partial VAE moves inside an unconditional cleanup boundary.

    OPT_VAE_RESIDENCY: after offloading to CPU, skip soft_empty_cache so consecutive
    tasks on the same worker can reload quickly; serial sessions still prevent video
    and audio from reaching peak residency simultaneously.
    """
    from ..runtime.h3_settings import OPT_VAE_RESIDENCY
    try:
        adapter.to(load_device)
        yield adapter
    finally:
        try:
            adapter.offload()
        finally:
            if not OPT_VAE_RESIDENCY:
                model_management.soft_empty_cache()


def _decode_video(adapter: Any, latent: Any, target: Mapping[str, Any]):
    t = _require_torch()
    decode = getattr(adapter, "decode", None)
    if not callable(decode):
        raise RuntimeError("The video VAE adapter is missing decode(latents)")
    frames = _call_supported(
        decode,
        normalized_latents=latent,
        latents=latent,
        latent=latent,
        frame_count=int(target["frame_count"]),
        target_height=int(target["height"]),
        target_width=int(target["width"]),
    )
    if isinstance(frames, Mapping):
        frames = frames.get("images", frames.get("frames"))
    if isinstance(frames, (tuple, list)):
        frames = frames[0] if frames else None
    if not isinstance(frames, t.Tensor):
        raise TypeError("video VAE decode must return a torch.Tensor")

    if frames.ndim == 5:
        if int(frames.shape[1]) in (3, 4):  # [B,C,T,H,W]
            frames = frames.permute(0, 2, 3, 4, 1)
        elif int(frames.shape[-1]) not in (3, 4):  # not [B,T,H,W,C]
            raise ValueError(
                f"Unrecognized video decode shape={tuple(frames.shape)}"
            )
        frames = frames.reshape(-1, *frames.shape[-3:])
    elif frames.ndim == 4:
        if int(frames.shape[-1]) in (3, 4):  # Comfy IMAGE
            pass
        elif int(frames.shape[1]) in (3, 4):  # [T,C,H,W]
            frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"Unrecognized video decode shape={tuple(frames.shape)}"
            )
    else:
        raise ValueError(
            f"video decode must return a rank-4/5 tensor; got {tuple(frames.shape)}"
        )
    frames = frames[..., :3].to(dtype=t.float32)
    height, width = int(target["height"]), int(target["width"])
    if int(frames.shape[1]) < height or int(frames.shape[2]) < width:
        raise ValueError(
            f"decoded frames={int(frames.shape[2])}x{int(frames.shape[1])} "
            f"is smaller than target={width}x{height}"
        )
    # The native VAE's tile padding is bottom/right; match the source crop.
    return (
        frames[:, :height, :width, :]
        .to(device="cpu")
        .contiguous()
    )


def _decode_audio(
    adapter: Any,
    latent: Any,
    *,
    sample_count: int | None = None,
) -> dict[str, Any]:
    t = _require_torch()
    decode = getattr(adapter, "decode", None)
    if not callable(decode):
        raise RuntimeError("The audio VAE adapter is missing decode(latents)")
    output = _call_supported(
        decode,
        normalized_latents=latent,
        latents=latent,
        latent=latent,
        sample_count=(int(sample_count) if sample_count is not None else None),
    )
    if isinstance(output, Mapping):
        waveform = output.get("waveform")
        sample_rate = int(
            output.get(
                "sample_rate",
                getattr(adapter, "sample_rate", 32000),
            )
        )
    else:
        waveform = output[0] if isinstance(output, (tuple, list)) else output
        sample_rate = int(getattr(adapter, "sample_rate", 32000))
    if not isinstance(waveform, t.Tensor):
        raise TypeError("audio VAE decode must return a waveform tensor")
    if waveform.ndim == 2:  # [C,L]
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # Native H3 audio VAE returns [C,1,L]; Comfy AUDIO is [B,C,L].
        if (
            int(waveform.shape[0]) in (1, 2)
            and int(waveform.shape[1]) == 1
            and int(waveform.shape[0]) != 1
        ):
            waveform = waveform.permute(1, 0, 2)
    else:
        raise ValueError(
            f"audio waveform must be [C,L], [C,1,L], or [B,C,L]; got {tuple(waveform.shape)}"
        )
    if waveform.ndim != 3:
        raise ValueError("Cannot convert the audio waveform to Comfy AUDIO [B,C,L]")
    return {
        "waveform": waveform.to(device="cpu", dtype=t.float32).contiguous(),
        "sample_rate": sample_rate,
    }


def _single_image_dimensions(image: Any, label: str) -> tuple[int, int]:
    t = _require_torch()
    if not isinstance(image, t.Tensor) or image.ndim != 4:
        raise TypeError(f"{label} must be a ComfyUI IMAGE [B,H,W,C]")
    if int(image.shape[0]) != 1:
        raise ValueError(f"{label} supports batch=1 only; got {int(image.shape[0])}")
    if int(image.shape[-1]) != 3:
        raise ValueError(f"{label} must be an RGB IMAGE; got channel count={int(image.shape[-1])}")
    height, width = int(image.shape[1]), int(image.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} dimensions must be positive")
    return width, height


def _audio_metadata(audio: Any, label: str) -> tuple[Any, int, float]:
    t = _require_torch()
    if not isinstance(audio, Mapping):
        raise TypeError(f"{label} must be a ComfyUI AUDIO mapping")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, t.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{label}.waveform must be a [B,C,L] tensor")
    if int(waveform.shape[0]) != 1:
        raise ValueError(f"{label} supports batch=1 only")
    if int(waveform.shape[1]) <= 0 or int(waveform.shape[2]) <= 0:
        raise ValueError(f"{label}.waveform cannot be empty")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{label}.sample_rate must be a positive integer")
    duration = int(waveform.shape[2]) / int(sample_rate)
    return waveform, int(sample_rate), float(duration)


def _downmix_comfy_audio(audio: Mapping[str, Any], label: str = "audio") -> Mapping[str, Any]:
    """For Comfy AUDIO with >2 channels and no layout, average to mono then expand to stereo (matching the ffmpeg -ac 2 fallback)."""
    waveform, sample_rate, _ = _audio_metadata(audio, label)
    channels = int(waveform.shape[1])
    if channels <= H3_AUDIO_CHANNELS:
        return audio
    logging.getLogger(__name__).warning(
        "%s has %s channels and no channel layout; averaged down to stereo", label, channels
    )
    stereo = waveform.mean(dim=1, keepdim=True).expand(-1, H3_AUDIO_CHANNELS, -1).contiguous()
    out = dict(audio); out["waveform"] = stereo; out["sample_rate"] = sample_rate
    return out


def _video_dimensions(video: Any, label: str) -> tuple[int, int]:
    getter = getattr(video, "get_dimensions", None)
    components_getter = getattr(video, "get_components", None)
    if not callable(getter) or not callable(components_getter):
        raise TypeError(
            f"{label} must be a ComfyUI VIDEO (requires get_dimensions/get_components)"
        )
    dimensions = getter()
    if (
        not isinstance(dimensions, Sequence)
        or isinstance(dimensions, (str, bytes))
        or len(dimensions) != 2
    ):
        raise ValueError(f"{label}.get_dimensions() must return (width,height)")
    width, height = int(dimensions[0]), int(dimensions[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} dimensions must be positive")
    return width, height


def _probe_video_audio(video: Any, label: str) -> tuple[bool, float | None]:
    """Probe the active Comfy VIDEO without guessing from its visual duration."""

    components = video.get_components()
    audio = getattr(components, "audio", None)
    if audio is None:
        return False, None
    _, _, duration = _audio_metadata(audio, f"{label}.audio")
    return True, duration


def _target_shape_info(target: Mapping[str, Any]) -> str:
    fields = (
        "task",
        "partition",
        "width",
        "height",
        "frame_count",
        "duration_seconds",
        "fps",
        "video_latent_t",
        "video_latent_h",
        "video_latent_w",
        "audio_latent_t",
        "keyframe_signature",
    )
    return json.dumps(
        {key: target[key] for key in fields if key in target},
        ensure_ascii=False,
        indent=2,
    )


def _image_tensor_to_pil(image: Any, label: str):
    from PIL import Image

    _single_image_dimensions(image, label)
    array = (
        image[0]
        .detach()
        .to(device="cpu", dtype=_require_torch().float32)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=_require_torch().uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _pil_to_image_tensor(image: Any):
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return _require_torch().from_numpy(array).to(dtype=_require_torch().float32).div_(255.0).unsqueeze(0)


def _multimodal_qwen_encode(
    handle: Any,
    *,
    task: str,
    prompt: str,
    images: Sequence[Any],
    condition_labels: Sequence[tuple[str, int]] = (),
    videos: Sequence[Any] = (),
    cache_parts: Sequence[Any] = (),
    te_fingerprint: str = "",
) -> dict[str, Any]:
    from ..runtime.encode_cache import get_encode_cache, visual_cache_key
    from ..runtime.h3_settings import OPT_ENCODE_CACHE
    if task == H3_TASK_FL2VA:
        encode = getattr(handle, "encode_fl2va_conditioning", None)
        if not callable(encode):
            raise RuntimeError(
                "The Qwen handle is missing encode_fl2va_conditioning(prompt, images)"
            )
    elif task == H3_TASK_REF2VA:
        encode = getattr(handle, "encode_ref2va_conditioning", None)
        if not callable(encode):
            raise RuntimeError(
                "The Qwen handle is missing encode_ref2va_conditioning(prompt, labels, images, videos)"
            )
    else:
        raise ValueError(f"Multimodal Qwen does not support task={task!r}")
    cache_key = visual_cache_key(
        material_fp="|".join(str(p) for p in cache_parts),
        plan=f"{task}|{prompt}|{tuple(condition_labels)}",
        te_fingerprint=str(te_fingerprint or ""),
    )
    if OPT_ENCODE_CACHE:
        hit = get_encode_cache().get(cache_key)
        if isinstance(hit, Mapping) and "prompt_embeds" in hit and "text_token_tags" in hit:
            return {"prompt_embeds": hit["prompt_embeds"], "text_token_tags": hit["text_token_tags"]}
    try:
        if task == H3_TASK_FL2VA:
            result = encode(prompt, list(images))
        else:
            result = encode(
                prompt,
                list(condition_labels),
                images=list(images),
                videos=list(videos),
            )
        if not isinstance(result, Mapping):
            raise TypeError("The Qwen multimodal encoder must return a mapping")
        prompt_embeds = _normalise_prompt_embeddings(result)
        tags = result.get("text_token_tags")
        if not isinstance(tags, _require_torch().Tensor) or tags.ndim != 1:
            raise ValueError("The Qwen multimodal encoder is missing rank-1 text_token_tags")
        if int(tags.shape[0]) != int(prompt_embeds.shape[0]):
            raise ValueError("Qwen prompt_embeds and text_token_tags lengths do not match")
        # Do not retain the encoder's ``hidden_states`` compatibility alias:
        # it points at the original device tensor and would keep a large Qwen
        # activation resident while the Video VAE is loaded next.  The node
        # contract only needs these two canonical CPU tensors.
        out = {
            "prompt_embeds": prompt_embeds,
            "text_token_tags": tags.detach().to(device="cpu").contiguous(),
        }
        if OPT_ENCODE_CACHE:
            get_encode_cache().put(cache_key, out)
        return out
    finally:
        offload = getattr(handle, "offload_after_inference", None)
        if callable(offload):
            offload()


def _vae_bundle_components(wrapper: Mapping[str, Any]) -> tuple[Any, Any]:
    bundle = wrapper.get("bundle")
    video_vae = _wrapper_value(bundle, "video_vae")
    audio_vae = _wrapper_value(bundle, "audio_vae")
    if video_vae is None or audio_vae is None:
        raise RuntimeError("The VAE bundle must contain video_vae and audio_vae")
    return video_vae, audio_vae


def _video_source_path(video: Any, workdir: Path) -> Path:
    active_trim = getattr(video, "get_active_trim_window", None)
    start, duration = active_trim() if callable(active_trim) else (0.0, 0.0)
    save_to = getattr(video, "save_to", None)
    if float(start) != 0.0 or float(duration) != 0.0:
        if not callable(save_to):
            raise TypeError("A trimmed VIDEO must provide save_to()")
        path = workdir / "reference_input_trimmed.mp4"
        save_to(str(path))
        return path
    stream_getter = getattr(video, "get_stream_source", None)
    source = None
    if callable(stream_getter):
        try:
            source = stream_getter()
        except (AttributeError, NotImplementedError):
            source = None
    if isinstance(source, (str, Path)):
        return Path(source).expanduser().resolve()
    path = workdir / "reference_input.mp4"
    if isinstance(source, io.BytesIO):
        source.seek(0)
        path.write_bytes(source.read())
        source.seek(0)
        return path
    # Component-backed/older standard VIDEO implementations may expose only
    # save_to().  Materialize them under the already managed request workdir.
    if callable(save_to):
        save_to(str(path))
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError("VIDEO.save_to() did not produce a non-empty video file")
        return path
    raise TypeError(
        "VIDEO must provide get_stream_source() (file path/BytesIO) or save_to()"
    )


def _first_finite_rotation(values: Sequence[Any]) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            rotation = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rotation):
            return rotation % 360.0
    return None


def _probe_video_display_metadata(path: str | Path, stream: Any) -> tuple[Any, float]:
    """Read DAR and display-matrix rotation without relying on one PyAV ABI.

    PyAV exposes DAR on the video codec context, but stream side-data support
    differs by release and was removed in PyAV 14.  ffprobe is already part of
    the required ffmpeg media toolchain, so it is the authoritative fallback
    for ``AV_PKT_DATA_DISPLAYMATRIX`` rotation, matching the official H3 probe.
    """

    display_aspect_ratio: Any = None
    for owner in (stream, getattr(stream, "codec_context", None)):
        if owner is None:
            continue
        try:
            candidate = getattr(owner, "display_aspect_ratio", None)
        except Exception:
            continue
        if candidate not in (None, ""):
            display_aspect_ratio = candidate
            break

    metadata = getattr(stream, "metadata", None)
    rotation_values: list[Any] = []
    try:
        side_data = getattr(stream, "side_data", None)
    except Exception:
        side_data = None
    if side_data is not None:
        try:
            rotation_values.extend(getattr(item, "rotation", None) for item in side_data)
        except TypeError:
            pass
    if isinstance(metadata, Mapping):
        rotation_values.append(metadata.get("rotate"))
    rotation = _first_finite_rotation(rotation_values)

    import subprocess

    try:
        media = _runtime_module("media_conditioning")
        result = media._run_cancellable_process(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=display_aspect_ratio:stream_tags=rotate:stream_side_data=rotation",
                "-of",
                "json",
                str(path),
            ],
            interrupt_check=_check_interrupted,
            timeout_seconds=60.0,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams") or []
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"ffprobe could not resolve the reference video's DAR/rotation: {path}"
        ) from exc
    if not streams or not isinstance(streams[0], Mapping):
        raise ValueError(f"The ffprobe result for the reference video has no video stream: {path}")
    probed = streams[0]
    ffprobe_dar = probed.get("display_aspect_ratio")
    if ffprobe_dar not in (None, "", "N/A", "0:1", "0/1"):
        display_aspect_ratio = ffprobe_dar
    ffprobe_rotations = [
        item.get("rotation")
        for item in (probed.get("side_data_list") or [])
        if isinstance(item, Mapping)
    ]
    tags = probed.get("tags")
    if isinstance(tags, Mapping):
        ffprobe_rotations.append(tags.get("rotate"))
    probed_rotation = _first_finite_rotation(ffprobe_rotations)
    if probed_rotation is not None:
        rotation = probed_rotation
    return display_aspect_ratio, 0.0 if rotation is None else rotation


def _probe_video_path(path: str | Path):
    import av

    media = _runtime_module("media_conditioning")
    _check_interrupted()
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"The reference video has no video stream: {path}")
        stream = container.streams.video[0]
        fps_value = stream.average_rate or stream.base_rate
        if fps_value is None:
            raise ValueError(f"Could not resolve fps for the reference video: {path}")
        fps = float(fps_value)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and stream.duration is not None and stream.time_base is not None:
            frame_count = int(round(float(stream.duration * stream.time_base) * fps))
        if frame_count <= 0:
            frame_count = 0
            for _ in container.decode(video=0):
                _check_interrupted()
                frame_count += 1
        sar = str(getattr(stream, "sample_aspect_ratio", None) or "1:1")
        display_aspect_ratio, rotation = _probe_video_display_metadata(path, stream)
        # PyAV reports coded geometry.  The official preprocessing policy
        # computes display geometry from SAR and rotation before choosing the
        # normalized canvas; constructing the dataclass directly would treat
        # coded width/height as display dimensions.
        return media.ReferenceVideoMetadata.from_coded(
            width=int(stream.width),
            height=int(stream.height),
            fps=fps,
            frame_count=frame_count,
            has_audio=bool(container.streams.audio),
            sample_aspect_ratio=sar,
            display_aspect_ratio=display_aspect_ratio,
            rotation_degrees=rotation,
        )


def _decode_video_file(path: str | Path):
    import av
    import numpy as np

    frames: list[Any] = []
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(video=0):
            _check_interrupted()
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"The prepared reference video contains no decodable frames: {path}")
    array = np.stack(frames, axis=0)
    return _require_torch().from_numpy(array).to(dtype=_require_torch().float32).div_(255.0)


def _load_pcm_wav(path: str | Path) -> dict[str, Any]:
    import numpy as np

    with wave.open(str(path), "rb") as reader:
        channels = int(reader.getnchannels())
        sample_rate = int(reader.getframerate())
        sample_width = int(reader.getsampwidth())
        frames = int(reader.getnframes())
        raw = reader.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"The reference WAV must be PCM s16; got sample_width={sample_width}")
    array = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).T.copy()
    waveform = _require_torch().from_numpy(array).to(dtype=_require_torch().float32).div_(32768.0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


def _target_fingerprint(target: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(target),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _preflight_target_conditions(
    *,
    task: str,
    target: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
) -> None:
    """Reject crossed target/material branches before Qwen or a VAE is loaded."""

    if task == H3_TASK_FL2VA:
        semantic = [int(item["frame_index"]) for item in conditions]
        resolved = [int(item["resolved_frame_index"]) for item in conditions]
        if semantic != list(target.get("keyframe_signature", ())):
            raise ValueError("FL2VA target.keyframe_signature does not match the current keyframes")
        if semantic != list(target.get("semantic_frame_indices", ())):
            raise ValueError("FL2VA target.semantic_frame_indices is cross-wired with the keyframes")
        if resolved != list(target.get("pixel_frame_indices", ())):
            raise ValueError("FL2VA target.pixel_frame_indices is cross-wired with the keyframes")
        return

    if task != H3_TASK_REF2VA:
        raise ValueError(f"Preflight does not support task={task!r}")
    actual_order = condition_order_fingerprint(task, conditions)
    declared_order = target.get("reference_order_fingerprint")
    if declared_order != actual_order:
        raise ValueError("The Ref2VA target is cross-wired with the current ordered references")
    source_index = target.get("duration_source_condition_index")
    if source_index is None:
        return
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or not 0 <= source_index < len(conditions)
    ):
        raise ValueError("Ref2VA duration_source_condition_index is invalid")
    source = conditions[source_index]
    if source.get("type") not in {"audio", "video", "video_audio"}:
        raise ValueError("The Ref2VA automatic duration source is not an audio-bearing reference")
    if source.get("has_audio") is False:
        raise ValueError("The Ref2VA automatic duration source is a silent reference")
    expected_duration = target.get("audio_reference_duration_seconds")
    source_duration = source.get("audio_duration_seconds")
    if expected_duration is not None and (
        source_duration is None
        or abs(float(expected_duration) - float(source_duration)) > 1e-9
    ):
        raise ValueError("The Ref2VA automatic-duration target does not match the reference audio-track duration")




__all__ = [n for n in list(globals()) if not n.startswith("__")]
