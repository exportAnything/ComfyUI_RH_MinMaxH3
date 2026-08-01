"""ComfyUI nodes for in-process MiniMax-H3 inference.

No node in this module opens a socket, submits a job, or imports SGLang.  The
runtime components are ordinary Python/PyTorch modules loaded into the same
process as ComfyUI.

T2VA keeps its original v1 wire contract for existing workflows.  FL2VA and
Ref2VA use the task-aware v2 contract and are admitted only after their media,
presentation, packed-layout, and checkpoint-partition invariants agree.
"""

from __future__ import annotations

import importlib
import hashlib
import io
import inspect
import json
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

from .contracts import (
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
from .runtime.h3_settings import (
    BF16_TE_MODEL_NAME,
    CACHE_DIT_MC,
    CACHE_DIT_MODE_CHOICES,
    CACHE_DIT_MODE_OFF,
    CACHE_DIT_RDT_COOKBOOK,
    CACHE_DIT_WARMUP,
    INT8_DIT_DIRNAME,
    INT8_TE_DIRNAME,
    INT8_TE_FILENAME,
    VAE_MERGED_DIRNAME,
    VAE_MERGED_MODEL_NAME,
    bf16_dit_model_name,
    int8_dit_filename,
)
from .sampling import sample_h3, sample_t2va

CATEGORY = "MiniMax H3 Direct"
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
            "MiniMax-H3 Direct 需要 ComfyUI 自带的 PyTorch；当前 Python 环境未安装 torch"
        )
    return torch


def _require_comfy():
    if model_management is None:
        raise RuntimeError(
            "该节点只能在 ComfyUI 进程中执行；未找到 comfy.model_management"
        )


def _h3_temporary_directory(prefix: str):
    """Allocate media scratch space under ComfyUI's managed temp root."""

    _require_comfy()
    try:
        import folder_paths
    except ImportError as exc:
        raise RuntimeError("ComfyUI folder_paths 不可用，无法创建 H3 临时目录") from exc
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

    try:
        return importlib.import_module(f".runtime.{name}", package=__package__)
    except ImportError as exc:
        raise RuntimeError(
            f"MiniMax-H3 Direct runtime.{name} 未安装完整：{exc}. "
            "请确认 custom node 目录包含原生 runtime 代码，而不是旧版服务端节点包。"
        ) from exc


def _model_root_choices() -> list[str]:
    """Loader COMBO 选项；失败时回退占位，保证 object_info 可生成。"""

    module = _runtime_module("components")
    lister = getattr(module, "list_h3_model_roots", None)
    if callable(lister):
        try:
            choices = list(lister())
            if choices:
                return choices
        except Exception:
            pass
    return ["MiniMax-H3"]


def _partition_roots(partition: str = H3_T2VA_PARTITION) -> list[Path]:
    """Return partition candidates using directory names only.

    ``INPUT_TYPES`` runs during object-definition/manifest generation.  It
    must not call the execution-time release resolver, which may parse
    ``model_index.json``.  Strict metadata admission remains in ``load()``.
    """

    partition_name = str(partition).strip().lower()
    if partition_name not in {H3_T2VA_PARTITION, H3_REF2VA_PARTITION}:
        return []
    basename = (
        "FL2VA" if partition_name == H3_T2VA_PARTITION else "Ref2VA"
    )
    module = _runtime_module("components")
    lister = getattr(module, "list_h3_model_root_paths", None)
    roots = list(lister()) if callable(lister) else []
    out: list[Path] = []
    seen: set[str] = set()
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        desired_children = (root / basename, root / partition_name)
        known_children = tuple(
            root / name for name in ("FL2VA", "fl2va", "Ref2VA", "ref2va")
        )
        if root.name.lower() == partition_name and root.is_dir():
            candidates = (root,)
        else:
            existing_desired = tuple(
                child for child in desired_children if child.is_dir()
            )
            if existing_desired:
                candidates = existing_desired
            elif not any(child.is_dir() for child in known_children):
                # Metadata-free single-partition development bundle.
                candidates = (root,) if root.is_dir() else ()
            else:
                candidates = ()
        for candidate in candidates:
            partition_root = candidate.resolve()
            key = str(partition_root)
            if key not in seen:
                seen.add(key)
                out.append(partition_root)
    return out


def _partition_label(partition: str = H3_T2VA_PARTITION) -> str:
    return "FL2VA" if str(partition).strip().lower() == H3_T2VA_PARTITION else "Ref2VA"


def _default_dit_model_name(partition: str = H3_T2VA_PARTITION) -> str:
    return int8_dit_filename(_partition_label(partition))


def _default_te_model_name() -> str:
    return INT8_TE_FILENAME


def _default_vae_model_name() -> str:
    return VAE_MERGED_MODEL_NAME


def _is_sharded_weight(name: str) -> bool:
    return "-of-" in name or name.endswith(".index.json")


def _single_weight_names(component_dir: Path) -> list[str]:
    try:
        return sorted(
            p.name
            for p in component_dir.glob("*.safetensors")
            if p.is_file() and not _is_sharded_weight(p.name)
        )
    except OSError:
        return []


def _model_name_for_component_dir(
    component_dir: Path, prefix: str, partition: str
) -> str:
    """Map a component directory to the COMBO model name."""

    part = _partition_label(partition)
    dirname = component_dir.name
    if prefix == "transformer":
        preferred = int8_dit_filename(part)
        if (component_dir / preferred).is_file():
            return preferred
        singles = _single_weight_names(component_dir)
        if singles:
            return singles[0]
        if dirname == "transformer":
            return bf16_dit_model_name(part)
        if dirname == INT8_DIT_DIRNAME:
            return preferred
        return dirname
    if prefix == "text_encoder":
        if (component_dir / INT8_TE_FILENAME).is_file():
            return INT8_TE_FILENAME
        singles = _single_weight_names(component_dir)
        if singles:
            return singles[0]
        if dirname == "text_encoder":
            return BF16_TE_MODEL_NAME
        if dirname == INT8_TE_DIRNAME:
            return INT8_TE_FILENAME
        return dirname
    return dirname


def _model_name_for_vae_dir(component_dir: Path) -> str:
    if component_dir.name == VAE_MERGED_DIRNAME:
        return VAE_MERGED_MODEL_NAME
    return component_dir.name


def _selector_alias_map(prefix: str, partition: str) -> dict[str, str]:
    """model name / legacy dir name → relative component directory."""

    part = _partition_label(partition)
    if prefix == "transformer":
        return {
            bf16_dit_model_name(part): "transformer",
            int8_dit_filename(part): INT8_DIT_DIRNAME,
            "transformer": "transformer",
            INT8_DIT_DIRNAME: INT8_DIT_DIRNAME,
        }
    if prefix == "text_encoder":
        return {
            BF16_TE_MODEL_NAME: "text_encoder",
            INT8_TE_FILENAME: INT8_TE_DIRNAME,
            "text_encoder": "text_encoder",
            INT8_TE_DIRNAME: INT8_TE_DIRNAME,
        }
    if prefix == "vae":
        return {
            VAE_MERGED_MODEL_NAME: VAE_MERGED_DIRNAME,
            VAE_MERGED_DIRNAME: VAE_MERGED_DIRNAME,
        }
    return {}


def _is_valid_component_dir(path: Path, prefix: str) -> bool:
    if not path.is_dir():
        return False
    if prefix == "vae":
        return (
            (path / "video_vae" / "config.json").is_file()
            and (path / "audio_vae" / "config.json").is_file()
        )
    return path.name.startswith(prefix) and (path / "config.json").is_file()


def _selector_to_component_dirname(
    value: str,
    prefix: str,
    partition: str = H3_T2VA_PARTITION,
) -> str:
    """Resolve COMBO model name or legacy directory name to a relative dir."""

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{prefix} 模型名不能为空")
    aliases = _selector_alias_map(prefix, partition)
    if text in aliases:
        return aliases[text]
    for root in _partition_roots(partition):
        direct = root / text
        if _is_valid_component_dir(direct, prefix):
            return text
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not _is_valid_component_dir(child, prefix):
                continue
            if (child / text).is_file():
                return child.name
            if prefix == "vae":
                for nested in (
                    child / "video_vae" / text,
                    child / "audio_vae" / text,
                ):
                    if nested.is_file():
                        return child.name
    if prefix != "vae" and text.startswith(prefix):
        return text
    if prefix == "vae" and text == VAE_MERGED_DIRNAME:
        return text
    raise ValueError(f"无法解析{prefix}模型名：{text}")


def _component_model_choices(
    prefix: str,
    partition: str = H3_T2VA_PARTITION,
) -> list[str]:
    """List selectable model names; never expose an auto mode."""

    preferred = (
        _default_dit_model_name(partition)
        if prefix == "transformer"
        else _default_te_model_name()
    )
    names: set[str] = set()
    try:
        for root in _partition_roots(partition):
            for child in root.iterdir():
                if _is_valid_component_dir(child, prefix):
                    names.add(_model_name_for_component_dir(child, prefix, partition))
    except Exception:
        pass
    if not names:
        return [preferred]
    return sorted(
        names,
        key=lambda name: (name != preferred, name == bf16_dit_model_name(_partition_label(partition)) or name == BF16_TE_MODEL_NAME, name),
    )


def _vae_model_choices(partition: str = H3_T2VA_PARTITION) -> list[str]:
    names: set[str] = set()
    try:
        for root in _partition_roots(partition):
            for child in root.iterdir():
                if _is_valid_component_dir(child, "vae"):
                    names.add(_model_name_for_vae_dir(child))
    except Exception:
        pass
    preferred = _default_vae_model_name()
    return sorted(names, key=lambda name: (name != preferred, name)) or [preferred]


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
                f"必须明确选择{label}模型名（权重文件名或逻辑名）；"
                "不会再自动切换量化/BF16 权重。"
            ),
        },
    )


def _vae_dir_input(partition: str = H3_T2VA_PARTITION):
    choices = _vae_model_choices(partition)
    return (
        choices,
        {
            "default": choices[0],
            "tooltip": (
                "必须明确选择合并双 VAE 模型名；官方逻辑名为 "
                f"{VAE_MERGED_MODEL_NAME}。"
            ),
        },
    )


def _model_root_input(partition: str | None = None):
    partition_hint = (
        f"；该节点固定解析 {partition.upper()} 分区"
        if partition is not None
        else ""
    )
    return (
        _model_root_choices(),
        {
            "tooltip": (
                "从 models/diffusers 或 models/minimax_h3 下选择 MiniMax-H3 "
                f"权重根目录（可含 FL2VA/Ref2VA 子目录）{partition_hint}。"
            ),
        },
    )


def _validate_model_root_input(model_root="MiniMax-H3", **kwargs):
    # RH / 工作流可能注入不在本机 COMBO 列表中的真实目录名，放行后由执行期解析。
    if model_root is None:
        return True
    if not str(model_root).strip():
        return "model_root 不能为空"
    return True


def _validate_explicit_component_input(value, label: str):
    # A dynamically connected input cannot be validated until execution.
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return f"{label} 必须显式选择模型，不能使用 auto 或空值"
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
        raise ValueError(f"{label} 不能为空")
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{label} 不是目录：{path}")
    if path.is_absolute() and not path.is_dir():
        raise FileNotFoundError(f"{label} 不存在：{path}")

    # Keep the node UI aligned with runtime.components.model_root_path(): a
    # simple folder name below models/diffusers 或 models/minimax_h3 也有效。
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
            raise FileNotFoundError(f"{label} 不存在：{path}") from exc
    if path.is_dir():
        return path.resolve()
    raise FileNotFoundError(f"{label} 不存在：{path}")


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
        raise ValueError("MiniMax-H3 component wrapper 缺少 release identity")
    computed = _release_fingerprint(root, str(partition), metadata)
    declared = wrapper.get("release_fingerprint")
    if declared is not None and declared != computed:
        raise ValueError("MiniMax-H3 component release_fingerprint 被篡改或已过期")
    return computed


def _wrapper_component_fingerprint(
    wrapper: Mapping[str, Any],
    *,
    component_kind: str,
    path_key: str,
    fingerprint_key: str,
) -> str:
    path = wrapper.get(path_key)
    if not isinstance(path, str) or not path:
        raise ValueError(f"MiniMax-H3 {component_kind} wrapper 缺少 {path_key}")
    computed = _component_fingerprint(
        _wrapper_release_fingerprint(wrapper),
        component_kind,
        path,
        related_paths=(
            {
                key.removesuffix("_path"): value
                for key in ("tokenizer_path", "processor_path")
                if isinstance((value := wrapper.get(key)), str) and value
            }
            if component_kind == "text_encoder"
            else None
        ),
    )
    declared = wrapper.get(fingerprint_key, wrapper.get("component_fingerprint"))
    if declared is not None and declared != computed:
        raise ValueError(
            f"MiniMax-H3 {component_kind} component fingerprint 被篡改或已过期"
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
            f"{label} 端口 schema 无效；期望 {', '.join(repr(item) for item in schemas)}"
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
        raise ValueError(f"未知 MiniMax-H3 component label {label!r}") from exc
    clean = validate_component_for_task(
        wrapper,
        component_kind=component_kind,
        task=normalized_task,
    )
    fingerprint = _wrapper_release_fingerprint(clean)
    declared = clean.get("release_fingerprint")
    if declared is not None and declared != fingerprint:
        raise ValueError(f"{label}.release_fingerprint 与组件路径/metadata 不一致")
    clean["release_fingerprint"] = fingerprint
    return clean


def _require_same_release(*components: tuple[str, Mapping[str, Any]]) -> str:
    fingerprints = {
        label: _wrapper_release_fingerprint(wrapper)
        for label, wrapper in components
    }
    if len(set(fingerprints.values())) != 1:
        detail = ", ".join(f"{label}={value[:12]}" for label, value in fingerprints.items())
        raise ValueError(f"MiniMax-H3 组件不是同一 release：{detail}")
    return next(iter(fingerprints.values()))


def _resolve_selected_component(
    root: Path,
    value: str,
    *,
    keys: tuple[str, ...],
    label: str,
    partition: str = H3_T2VA_PARTITION,
    required_files: tuple[str, ...] = (),
) -> Path:
    validation = _validate_explicit_component_input(value, label)
    if validation is not True:
        raise ValueError(validation)
    prefix = "vae" if VAE_MERGED_DIRNAME in keys else str(keys[0])
    dirname = _selector_to_component_dirname(value, prefix, partition)
    module = _runtime_module("components")
    resolver = _find_callable(module, ("resolve_component",), label)
    return Path(
        resolver(
            root,
            keys,
            explicit=dirname,
            required_files=required_files,
        )
    ).resolve()


def _resolve_selected_vae(
    root: Path, value: str, *, partition: str = H3_T2VA_PARTITION
) -> Path:
    return _resolve_selected_component(
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
        supported = "、".join(sorted(allowed))
        raise ValueError(f"dtype {name!r} 不受支持；可选：auto、{supported}")
    return normalized


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
        f"runtime.{module.__name__.rsplit('.', 1)[-1]} 没有 {component} loader；"
        f"期望其中一个入口：{', '.join(names)}"
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
            f"{label} 端口不是 MiniMax-H3 Direct loader 的输出"
        )
    handle = wrapper.get("handle")
    if handle is None:
        raise RuntimeError(f"{label} wrapper 缺少 runtime handle")
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
            raise RuntimeError("Qwen encoder 返回了空序列")
        value = value[0]
    if not isinstance(value, t.Tensor):
        raise TypeError(
            "Qwen encoder 必须返回 torch.Tensor 或包含 prompt_embeds 的 mapping"
        )
    if value.ndim == 3:
        if int(value.shape[0]) != 1:
            raise ValueError("MiniMax-H3 Direct v0 只支持 batch=1")
        value = value[0]
    if value.ndim != 2:
        raise ValueError(
            f"Qwen encoder 输出必须为 [L,5120]，实际 shape={tuple(value.shape)}"
        )
    # The 64 GB text encoder must not remain co-resident with the 62 GB DiT.
    return value.detach().to(device="cpu").contiguous()


def _encode_prompt(handle: Any, prompt: str):
    encode = _wrapper_value(handle, "encode_prompt", "encode_conditioning")
    if not callable(encode):
        raise RuntimeError(
            "text encoder handle 缺少 encode_prompt(prompt)；"
            "请更新 minimax_h3_nodes/runtime/qwen_encoder.py"
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
        raise TypeError("runtime.packing 必须返回 packed conditioning mapping")
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
            raise ValueError("conditioning target 与 Empty AV Latent target 不一致")
    declared_fingerprint = clean.get("target_fingerprint")
    actual_fingerprint = _target_fingerprint(clean_target)
    if declared_fingerprint is not None and declared_fingerprint != actual_fingerprint:
        raise ValueError("conditioning.target_fingerprint 与 target 内容不一致")
    blocks = clean.get("condition_blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError("FL2VA/Ref2VA conditioning 缺少有序 condition_blocks")
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
        raise ValueError(f"v2 packed builder 不支持 task={clean['task']!r}")
    if not isinstance(packed, Mapping):
        raise TypeError("runtime.packing v2 builder 必须返回 mapping")
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
            "无法从 model handle 取得 MiniMaxH3DiTModel；"
            "handle 应暴露 transformer/model，或 load_for_inference()"
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


@contextmanager
def _transformer_session(model_wrapper: Mapping[str, Any]):
    handle = _unwrap_runtime_handle(
        model_wrapper, (H3_MODEL_SCHEMA, H3_MODEL_SCHEMA_V2), "h3_model"
    )
    try:
        # Acquisition can materialize a ModelPatcher or move a raw module to
        # GPU.  Keep it inside the cleanup boundary so partial load failures
        # cannot leave the huge DiT resident.
        transformer = _resolve_transformer(handle)
        yield transformer
    finally:
        offload = _wrapper_value(
            handle,
            "offload_after_inference",
            "offload",
            "release_from_device",
        )
        if callable(offload):
            offload()


@contextmanager
def _vae_device_session(adapter: Any, load_device: Any):
    """Keep partial VAE moves inside an unconditional cleanup boundary."""

    try:
        adapter.to(load_device)
        yield adapter
    finally:
        try:
            adapter.offload()
        finally:
            model_management.soft_empty_cache()


def _decode_video(adapter: Any, latent: Any, target: Mapping[str, Any]):
    t = _require_torch()
    decode = getattr(adapter, "decode", None)
    if not callable(decode):
        raise RuntimeError("video VAE adapter 缺少 decode(latents)")
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
        raise TypeError("video VAE decode 必须返回 torch.Tensor")

    if frames.ndim == 5:
        if int(frames.shape[1]) in (3, 4):  # [B,C,T,H,W]
            frames = frames.permute(0, 2, 3, 4, 1)
        elif int(frames.shape[-1]) not in (3, 4):  # not [B,T,H,W,C]
            raise ValueError(
                f"无法识别 video decode shape={tuple(frames.shape)}"
            )
        frames = frames.reshape(-1, *frames.shape[-3:])
    elif frames.ndim == 4:
        if int(frames.shape[-1]) in (3, 4):  # Comfy IMAGE
            pass
        elif int(frames.shape[1]) in (3, 4):  # [T,C,H,W]
            frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"无法识别 video decode shape={tuple(frames.shape)}"
            )
    else:
        raise ValueError(
            f"video decode 应返回 rank-4/5 tensor，实际 {tuple(frames.shape)}"
        )
    frames = frames[..., :3].to(dtype=t.float32)
    height, width = int(target["height"]), int(target["width"])
    if int(frames.shape[1]) < height or int(frames.shape[2]) < width:
        raise ValueError(
            f"decoded frames={int(frames.shape[2])}x{int(frames.shape[1])} "
            f"小于 target={width}x{height}"
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
        raise RuntimeError("audio VAE adapter 缺少 decode(latents)")
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
        raise TypeError("audio VAE decode 必须返回 waveform tensor")
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
            f"audio waveform 必须为 [C,L]、[C,1,L] 或 [B,C,L]，实际 {tuple(waveform.shape)}"
        )
    if waveform.ndim != 3:
        raise ValueError("无法把 audio waveform 转换为 Comfy AUDIO [B,C,L]")
    return {
        "waveform": waveform.to(device="cpu", dtype=t.float32).contiguous(),
        "sample_rate": sample_rate,
    }


class MiniMaxH3DirectModelLoader:
    """Load the native H3 DiT without a server or Diffusers pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "dtype": (
                    ["auto", "bfloat16", "float16"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto/bfloat16 推荐；runtime 会保留 H3 指定的 fp32 层。"
                        ),
                    },
                ),
                "transformer_path": _component_dir_input("transformer", "DiT"),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        transformer_path=None,
        **kwargs,
    ):
        if transformer_path is None:
            transformer_path = _default_dit_model_name(H3_T2VA_PARTITION)
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(transformer_path, "DiT")

    def load(
        self,
        model_root: str,
        dtype: str,
        transformer_path: str = "",
    ):
        selector = transformer_path or _default_dit_model_name(H3_T2VA_PARTITION)
        component_dir = _selector_to_component_dirname(
            selector, "transformer", H3_T2VA_PARTITION
        )
        root, release_info, sigma_scales = _resolve_t2va_release(
            model_root,
            required_component=component_dir,
            required_files=("config.json",),
        )
        selected_transformer = _resolve_selected_component(
            root,
            selector,
            keys=("transformer", "dit"),
            label="DiT",
            partition=H3_T2VA_PARTITION,
            required_files=("config.json",),
        )
        module = _runtime_module("model_loader")
        loader = _find_callable(
            module,
            (
                "load_h3_model",
                "load_h3_transformer",
                "load_minimax_h3_model",
            ),
            "DiT",
        )
        handle = _call_supported(
            loader,
            model_root=str(root),
            root=str(root),
            partition=H3_T2VA_PARTITION,
            task=H3_TASK_T2VA,
            dtype=_runtime_dtype_name(dtype, allow_float32=False),
            dtype_name=_runtime_dtype_name(dtype, allow_float32=False),
            device="auto",
            offload_device="auto",
            transformer_path=str(selected_transformer),
        )
        if handle is None:
            raise RuntimeError("runtime.model_loader 返回了 None")
        return (
            {
                "schema": H3_MODEL_SCHEMA,
                "handle": handle,
                "model_root": str(root),
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "dtype": dtype,
                "transformer_path": str(selected_transformer),
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


class MiniMaxH3DirectTextEncoderLoader:
    """Load the Qwen3-VL-32B layer-50 conditioning encoder in process."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "text_encoder_path": _component_dir_input("text_encoder", "文本编码器"),
            },
        }

    RETURN_TYPES = (TEXT_ENCODER_TYPE,)
    RETURN_NAMES = ("h3_text_encoder",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        text_encoder_path=None,
        **kwargs,
    ):
        if text_encoder_path is None:
            text_encoder_path = _default_te_model_name()
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(
            text_encoder_path, "Text Encoder"
        )

    def load(
        self,
        model_root: str,
        dtype: str,
        text_encoder_path: str = "",
    ):
        selector = text_encoder_path or _default_te_model_name()
        component_dir = _selector_to_component_dirname(
            selector, "text_encoder", H3_T2VA_PARTITION
        )
        root, release_info, sigma_scales = _resolve_t2va_release(
            model_root,
            required_component=component_dir,
            required_files=("config.json",),
        )
        selected_text_encoder = _resolve_selected_component(
            root,
            selector,
            keys=("text_encoder", "qwen3vl", "qwen"),
            label="Text Encoder",
            partition=H3_T2VA_PARTITION,
            required_files=("config.json",),
        )
        module = _runtime_module("qwen_encoder")
        loader = _find_callable(
            module,
            (
                "load_h3_text_encoder",
                "load_h3_qwen_encoder",
                "load_qwen_encoder",
            ),
            "Qwen3-VL",
        )
        handle = _call_supported(
            loader,
            model_root=str(root),
            root=str(root),
            partition=H3_T2VA_PARTITION,
            dtype=_runtime_dtype_name(dtype),
            dtype_name=_runtime_dtype_name(dtype),
            device="auto",
            offload_device="cpu",
            text_encoder_path=str(selected_text_encoder),
        )
        if handle is None:
            raise RuntimeError("runtime.qwen_encoder 返回了 None")
        return (
            {
                "schema": H3_TEXT_ENCODER_SCHEMA,
                "handle": handle,
                "model_root": str(root),
                "dtype": dtype,
                "text_encoder_path": str(selected_text_encoder),
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


class MiniMaxH3DirectVAELoader:
    """Load the native 24-channel video and 32-channel audio VAEs.

    Both released VAE weight sets stay FP32.  The video adapter independently
    applies FP16 autocast to decode operations where the source runtime does;
    exposing a generic weight dtype here would incorrectly suggest that BF16
    VAE weights are part of the H3 contract.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "vae_path": _vae_dir_input(),
            }
        }

    RETURN_TYPES = (VAE_BUNDLE_TYPE,)
    RETURN_NAMES = ("h3_vae_bundle",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        vae_path=None,
        **kwargs,
    ):
        if vae_path is None:
            vae_path = _default_vae_model_name()
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(vae_path, "VAE")

    def load(self, model_root: str, vae_path: str = ""):
        selector = vae_path or _default_vae_model_name()
        component_dir = _selector_to_component_dirname(
            selector, "vae", H3_T2VA_PARTITION
        )
        root, release_info, sigma_scales = _resolve_t2va_release(
            model_root,
            required_component=component_dir,
            required_files=(
                "video_vae/config.json",
                "audio_vae/config.json",
            ),
        )
        selected_vae = _resolve_selected_vae(
            root, selector, partition=H3_T2VA_PARTITION
        )
        module = _runtime_module("vae_adapter")
        loader = _find_callable(
            module,
            ("load_h3_vae_bundle", "load_minimax_h3_vae_bundle"),
            "dual VAE",
        )
        bundle = _call_supported(
            loader,
            model_root=str(root),
            root=str(root),
            vae_path=str(selected_vae),
            device="cpu",
            offload_device="cpu",
            video_compute_dtype="float16",
            audio_compute_dtype="float32",
        )
        if bundle is None:
            raise RuntimeError("runtime.vae_adapter 返回了 None")
        return (
            {
                "schema": H3_VAE_SCHEMA,
                "bundle": bundle,
                "model_root": str(root),
                "vae_path": str(selected_vae),
                "weight_dtype": "float32",
                "video_decode_compute": "float16_autocast",
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


def _supported_tasks_for_release(
    metadata: Mapping[str, Any],
    partition: str,
) -> tuple[str, ...]:
    declared = metadata.get("tasks")
    if isinstance(declared, list) and declared and all(
        isinstance(item, str) and item for item in declared
    ):
        return tuple(str(item).strip().lower() for item in declared)
    return (
        (H3_TASK_T2VA, H3_TASK_FL2VA)
        if partition == H3_T2VA_PARTITION
        else (H3_TASK_REF2VA,)
    )


class _MiniMaxH3ExplicitModelLoader:
    """Partition-explicit v2 DiT loader shared by FL2VA and Ref2VA."""

    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        partition = partition_for_task(cls.TASK)
        return {
            "required": {
                "model_root": _model_root_input(partition),
                "dtype": (
                    ["auto", "bfloat16", "float16"],
                    {
                        "default": "auto",
                        "tooltip": "auto/bfloat16 推荐；runtime 保留 H3 指定的 fp32 层。",
                    },
                ),
                "transformer_path": _component_dir_input(
                    "transformer", "DiT", partition
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        transformer_path=None,
        **kwargs,
    ):
        if transformer_path is None:
            transformer_path = _default_dit_model_name(partition_for_task(cls.TASK))
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(transformer_path, "DiT")

    def load(
        self,
        model_root: str,
        dtype: str,
        transformer_path: str = "",
    ):
        task = normalize_task(self.TASK)
        partition_hint = partition_for_task(task)
        selector = transformer_path or _default_dit_model_name(partition_hint)
        component_dir = _selector_to_component_dirname(
            selector, "transformer", partition_hint
        )
        root, partition, release_info, sigma_scales = _resolve_release(
            model_root,
            task=task,
            required_component=component_dir,
            required_files=("config.json",),
        )
        selected_transformer = _resolve_selected_component(
            root,
            selector,
            keys=("transformer", "dit"),
            label="DiT",
            partition=partition,
            required_files=("config.json",),
        )
        loader = _find_callable(
            _runtime_module("model_loader"),
            ("load_h3_model",),
            "DiT",
        )
        # task/partition/explicit path are critical.  A runtime without this
        # exact contract must fail instead of silently loading FL weights.
        handle = loader(
            model_root=str(root),
            partition=partition,
            task=task,
            transformer_path=str(selected_transformer),
            dtype=_runtime_dtype_name(dtype, allow_float32=False),
            device="auto",
            offload_device="auto",
        )
        if handle is None:
            raise RuntimeError("runtime.model_loader 返回了 None")
        release_fingerprint = _release_fingerprint(root, partition, release_info)
        component_fingerprint = _component_fingerprint(
            release_fingerprint, "transformer", selected_transformer
        )
        return (
            {
                "schema": H3_MODEL_SCHEMA_V2,
                "handle": handle,
                "model_root": str(root),
                "partition": partition,
                "task": task,
                "tasks": _supported_tasks_for_release(release_info, partition),
                "dtype": dtype,
                "transformer_path": str(selected_transformer),
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
                "release_fingerprint": release_fingerprint,
                "component_fingerprint": component_fingerprint,
                "transformer_fingerprint": component_fingerprint,
            },
        )


class MiniMaxH3FL2VAModelLoader(_MiniMaxH3ExplicitModelLoader):
    TASK = H3_TASK_FL2VA


class MiniMaxH3Ref2VAModelLoader(_MiniMaxH3ExplicitModelLoader):
    TASK = H3_TASK_REF2VA


class _MiniMaxH3ExplicitTextEncoderLoader:
    """Partition-explicit v2 Qwen3-VL loader."""

    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        partition = partition_for_task(cls.TASK)
        return {
            "required": {
                "model_root": _model_root_input(partition),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "text_encoder_path": _component_dir_input(
                    "text_encoder", "文本编码器", partition
                ),
            }
        }

    RETURN_TYPES = (TEXT_ENCODER_TYPE,)
    RETURN_NAMES = ("h3_text_encoder",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        text_encoder_path=None,
        **kwargs,
    ):
        if text_encoder_path is None:
            text_encoder_path = _default_te_model_name()
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(
            text_encoder_path, "Text Encoder"
        )

    def load(
        self,
        model_root: str,
        dtype: str,
        text_encoder_path: str = "",
    ):
        task = normalize_task(self.TASK)
        partition_hint = partition_for_task(task)
        selector = text_encoder_path or _default_te_model_name()
        component_dir = _selector_to_component_dirname(
            selector, "text_encoder", partition_hint
        )
        root, partition, release_info, sigma_scales = _resolve_release(
            model_root,
            task=task,
            required_component=component_dir,
            required_files=("config.json",),
        )
        selected_text_encoder = _resolve_selected_component(
            root,
            selector,
            keys=("text_encoder", "qwen3vl", "qwen"),
            label="Text Encoder",
            partition=partition,
            required_files=("config.json",),
        )
        loader = _find_callable(
            _runtime_module("qwen_encoder"),
            ("load_h3_text_encoder",),
            "Qwen3-VL",
        )
        handle = loader(
            model_root=str(root),
            partition=partition,
            require_multimodal_processor=True,
            text_encoder_path=str(selected_text_encoder),
            dtype=_runtime_dtype_name(dtype),
            device="auto",
            offload_device="cpu",
        )
        if handle is None:
            raise RuntimeError("runtime.qwen_encoder 返回了 None")
        tokenizer_component = Path(
            getattr(handle, "tokenizer_component_path", selected_text_encoder)
        ).resolve()
        processor_value = getattr(handle, "processor_component_path", None)
        processor_component = (
            Path(processor_value).resolve()
            if processor_value is not None
            else None
        )
        related_paths = {"tokenizer": tokenizer_component}
        if processor_component is not None:
            related_paths["processor"] = processor_component
        release_fingerprint = _release_fingerprint(root, partition, release_info)
        component_fingerprint = _component_fingerprint(
            release_fingerprint,
            "text_encoder",
            selected_text_encoder,
            related_paths=related_paths,
        )
        return (
            {
                "schema": H3_TEXT_ENCODER_SCHEMA_V2,
                "handle": handle,
                "model_root": str(root),
                "partition": partition,
                "task": task,
                "tasks": _supported_tasks_for_release(release_info, partition),
                "dtype": dtype,
                "text_encoder_path": str(selected_text_encoder),
                "tokenizer_path": str(tokenizer_component),
                **(
                    {"processor_path": str(processor_component)}
                    if processor_component is not None
                    else {}
                ),
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
                "release_fingerprint": release_fingerprint,
                "component_fingerprint": component_fingerprint,
                "text_encoder_fingerprint": component_fingerprint,
            },
        )


class MiniMaxH3FL2VATextEncoderLoader(_MiniMaxH3ExplicitTextEncoderLoader):
    TASK = H3_TASK_FL2VA


class MiniMaxH3Ref2VATextEncoderLoader(_MiniMaxH3ExplicitTextEncoderLoader):
    TASK = H3_TASK_REF2VA


class _MiniMaxH3ExplicitVAELoader:
    """Partition-explicit v2 dual VAE loader."""

    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        partition = partition_for_task(cls.TASK)
        return {
            "required": {
                "model_root": _model_root_input(partition),
                "vae_path": _vae_dir_input(partition),
            }
        }

    RETURN_TYPES = (VAE_BUNDLE_TYPE,)
    RETURN_NAMES = ("h3_vae_bundle",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        vae_path=None,
        **kwargs,
    ):
        if vae_path is None:
            vae_path = _default_vae_model_name()
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(vae_path, "VAE")

    def load(self, model_root: str, vae_path: str = ""):
        task = normalize_task(self.TASK)
        partition_hint = partition_for_task(task)
        selector = vae_path or _default_vae_model_name()
        component_dir = _selector_to_component_dirname(
            selector, "vae", partition_hint
        )
        root, partition, release_info, sigma_scales = _resolve_release(
            model_root,
            task=task,
            required_component=component_dir,
            required_files=(
                "video_vae/config.json",
                "audio_vae/config.json",
            ),
        )
        selected_vae = _resolve_selected_vae(
            root, selector, partition=partition
        )
        loader = _find_callable(
            _runtime_module("vae_adapter"),
            ("load_h3_vae_bundle",),
            "dual VAE",
        )
        bundle = loader(
            model_root=str(root),
            vae_path=str(selected_vae),
            device="cpu",
            video_compute_dtype="float16",
            audio_compute_dtype="float32",
        )
        if bundle is None:
            raise RuntimeError("runtime.vae_adapter 返回了 None")
        release_fingerprint = _release_fingerprint(root, partition, release_info)
        component_fingerprint = _component_fingerprint(
            release_fingerprint, "vae", selected_vae
        )
        return (
            {
                "schema": H3_VAE_SCHEMA_V2,
                "bundle": bundle,
                "model_root": str(root),
                "partition": partition,
                "task": task,
                "tasks": _supported_tasks_for_release(release_info, partition),
                "vae_path": str(selected_vae),
                "weight_dtype": "float32",
                "video_decode_compute": "float16_autocast",
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
                "release_fingerprint": release_fingerprint,
                "component_fingerprint": component_fingerprint,
                "vae_fingerprint": component_fingerprint,
            },
        )


class MiniMaxH3FL2VAVAELoader(_MiniMaxH3ExplicitVAELoader):
    TASK = H3_TASK_FL2VA


class MiniMaxH3Ref2VAVAELoader(_MiniMaxH3ExplicitVAELoader):
    TASK = H3_TASK_REF2VA


def _single_image_dimensions(image: Any, label: str) -> tuple[int, int]:
    t = _require_torch()
    if not isinstance(image, t.Tensor) or image.ndim != 4:
        raise TypeError(f"{label} 必须是 ComfyUI IMAGE [B,H,W,C]")
    if int(image.shape[0]) != 1:
        raise ValueError(f"{label} 只支持 batch=1，实际为 {int(image.shape[0])}")
    if int(image.shape[-1]) != 3:
        raise ValueError(f"{label} 必须是 RGB IMAGE，实际通道数={int(image.shape[-1])}")
    height, width = int(image.shape[1]), int(image.shape[2])
    if height <= 0 or width <= 0:
        raise ValueError(f"{label} 尺寸必须为正数")
    return width, height


def _audio_metadata(audio: Any, label: str) -> tuple[Any, int, float]:
    t = _require_torch()
    if not isinstance(audio, Mapping):
        raise TypeError(f"{label} 必须是 ComfyUI AUDIO mapping")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not isinstance(waveform, t.Tensor) or waveform.ndim != 3:
        raise ValueError(f"{label}.waveform 必须是 [B,C,L] tensor")
    if int(waveform.shape[0]) != 1:
        raise ValueError(f"{label} 只支持 batch=1")
    if int(waveform.shape[1]) <= 0 or int(waveform.shape[2]) <= 0:
        raise ValueError(f"{label}.waveform 不能为空")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"{label}.sample_rate 必须是正整数")
    duration = int(waveform.shape[2]) / int(sample_rate)
    return waveform, int(sample_rate), float(duration)


def _video_dimensions(video: Any, label: str) -> tuple[int, int]:
    getter = getattr(video, "get_dimensions", None)
    components_getter = getattr(video, "get_components", None)
    if not callable(getter) or not callable(components_getter):
        raise TypeError(
            f"{label} 必须是 ComfyUI VIDEO（需要 get_dimensions/get_components）"
        )
    dimensions = getter()
    if (
        not isinstance(dimensions, Sequence)
        or isinstance(dimensions, (str, bytes))
        or len(dimensions) != 2
    ):
        raise ValueError(f"{label}.get_dimensions() 必须返回 (width,height)")
    width, height = int(dimensions[0]), int(dimensions[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} 尺寸必须为正数")
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
) -> dict[str, Any]:
    if task == H3_TASK_FL2VA:
        encode = getattr(handle, "encode_fl2va_conditioning", None)
        if not callable(encode):
            raise RuntimeError(
                "Qwen handle 缺少 encode_fl2va_conditioning(prompt, images)"
            )
    elif task == H3_TASK_REF2VA:
        encode = getattr(handle, "encode_ref2va_conditioning", None)
        if not callable(encode):
            raise RuntimeError(
                "Qwen handle 缺少 encode_ref2va_conditioning(prompt, labels, images, videos)"
            )
    else:
        raise ValueError(f"multimodal Qwen 不支持 task={task!r}")
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
            raise TypeError("Qwen multimodal encoder 必须返回 mapping")
        prompt_embeds = _normalise_prompt_embeddings(result)
        tags = result.get("text_token_tags")
        if not isinstance(tags, _require_torch().Tensor) or tags.ndim != 1:
            raise ValueError("Qwen multimodal encoder 缺少 rank-1 text_token_tags")
        if int(tags.shape[0]) != int(prompt_embeds.shape[0]):
            raise ValueError("Qwen prompt_embeds 与 text_token_tags 长度不一致")
        # Do not retain the encoder's ``hidden_states`` compatibility alias:
        # it points at the original device tensor and would keep a large Qwen
        # activation resident while the Video VAE is loaded next.  The node
        # contract only needs these two canonical CPU tensors.
        return {
            "prompt_embeds": prompt_embeds,
            "text_token_tags": tags.detach().to(device="cpu").contiguous(),
        }
    finally:
        offload = getattr(handle, "offload_after_inference", None)
        if callable(offload):
            offload()


def _vae_bundle_components(wrapper: Mapping[str, Any]) -> tuple[Any, Any]:
    bundle = wrapper.get("bundle")
    video_vae = _wrapper_value(bundle, "video_vae")
    audio_vae = _wrapper_value(bundle, "audio_vae")
    if video_vae is None or audio_vae is None:
        raise RuntimeError("VAE bundle 必须包含 video_vae 和 audio_vae")
    return video_vae, audio_vae


def _video_source_path(video: Any, workdir: Path) -> Path:
    active_trim = getattr(video, "get_active_trim_window", None)
    start, duration = active_trim() if callable(active_trim) else (0.0, 0.0)
    save_to = getattr(video, "save_to", None)
    if float(start) != 0.0 or float(duration) != 0.0:
        if not callable(save_to):
            raise TypeError("带 trim 的 VIDEO 必须提供 save_to()")
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
            raise ValueError("VIDEO.save_to() 没有生成非空视频文件")
        return path
    raise TypeError(
        "VIDEO 必须提供 get_stream_source()（文件路径/BytesIO）或 save_to()"
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
            f"reference video 无法通过 ffprobe 解析 DAR/旋转：{path}"
        ) from exc
    if not streams or not isinstance(streams[0], Mapping):
        raise ValueError(f"reference video 的 ffprobe 结果缺少视频流：{path}")
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
            raise ValueError(f"reference video 没有视频流：{path}")
        stream = container.streams.video[0]
        fps_value = stream.average_rate or stream.base_rate
        if fps_value is None:
            raise ValueError(f"reference video 无法解析 fps：{path}")
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
        raise ValueError(f"prepared reference video 没有可解码帧：{path}")
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
        raise ValueError(f"reference WAV 必须是 PCM s16，实际 sample_width={sample_width}")
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
            raise ValueError("FL2VA target.keyframe_signature 与当前 keyframes 不一致")
        if semantic != list(target.get("semantic_frame_indices", ())):
            raise ValueError("FL2VA target.semantic_frame_indices 与 keyframes 串线")
        if resolved != list(target.get("pixel_frame_indices", ())):
            raise ValueError("FL2VA target.pixel_frame_indices 与 keyframes 串线")
        return

    if task != H3_TASK_REF2VA:
        raise ValueError(f"preflight 不支持 task={task!r}")
    actual_order = condition_order_fingerprint(task, conditions)
    declared_order = target.get("reference_order_fingerprint")
    if declared_order != actual_order:
        raise ValueError("Ref2VA target 与当前 ordered references 串线")
    source_index = target.get("duration_source_condition_index")
    if source_index is None:
        return
    if (
        isinstance(source_index, bool)
        or not isinstance(source_index, int)
        or not 0 <= source_index < len(conditions)
    ):
        raise ValueError("Ref2VA duration_source_condition_index 无效")
    source = conditions[source_index]
    if source.get("type") not in {"audio", "video", "video_audio"}:
        raise ValueError("Ref2VA 自动时长来源不是 audio-bearing reference")
    if source.get("has_audio") is False:
        raise ValueError("Ref2VA 自动时长来源是静音 reference")
    expected_duration = target.get("audio_reference_duration_seconds")
    source_duration = source.get("audio_duration_seconds")
    if expected_duration is not None and (
        source_duration is None
        or abs(float(expected_duration) - float(source_duration)) > 1e-9
    ):
        raise ValueError("Ref2VA 自动时长 target 与 reference 音轨时长不一致")


class MiniMaxH3T2VATarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    list(H3_ASPECT_RATIOS),
                    {"default": "16:9"},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 5.0,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": "时长秒数，必须在 5.0–15.0（执行期再次校验）。",
                    },
                ),
            },
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "显式输出宽度；0 表示按 aspect_ratio 自动计算。需与 height 同时填写并按 32 对齐。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "显式输出高度；0 表示按 aspect_ratio 自动计算。需与 width 同时填写并按 32 对齐。",
                    },
                ),
            },
        }

    RETURN_TYPES = (TARGET_TYPE, "STRING")
    RETURN_NAMES = ("target", "shape_info")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/conditioning"

    @classmethod
    def VALIDATE_INPUTS(
        cls, duration_seconds=5.0, width=0, height=0, **kwargs
    ):
        # 连接外部节点时可能绕过 UI min/max，这里先拦一层
        if duration_seconds is None:
            return True
        try:
            value = float(duration_seconds)
        except (TypeError, ValueError):
            return f"Invalid duration_seconds: {duration_seconds!r}"
        from .contracts import H3_MAX_DURATION_SECONDS, H3_MIN_DURATION_SECONDS

        if not (H3_MIN_DURATION_SECONDS <= value <= H3_MAX_DURATION_SECONDS):
            return (
                f"Invalid duration_seconds: {value}. "
                f"Allowed: [{H3_MIN_DURATION_SECONDS}, {H3_MAX_DURATION_SECONDS}]"
            )
        # 显式尺寸同样在提交期拦截；动态连接无法静态取值时留给执行期合同校验。
        if width is None or height is None:
            return True
        try:
            resolve_t2va_target(
                aspect_ratio=str(kwargs.get("aspect_ratio", "16:9")),
                duration_seconds=value,
                width=width,
                height=height,
            )
        except (TypeError, ValueError) as exc:
            return str(exc)
        return True

    def build(
        self,
        aspect_ratio: str,
        duration_seconds: float,
        width: int = 0,
        height: int = 0,
    ):
        target = resolve_t2va_target(
            aspect_ratio=aspect_ratio,
            duration_seconds=float(duration_seconds),
            width=width,
            height=height,
        )
        info = {
            key: target[key]
            for key in (
                "width",
                "height",
                "frame_count",
                "duration_seconds",
                "fps",
                "video_latent_t",
                "video_latent_h",
                "video_latent_w",
                "audio_latent_t",
            )
        }
        return (target, json.dumps(info, ensure_ascii=False, indent=2))


class MiniMaxH3FL2VAFirstFrameCondition:
    """Create the only valid first-only or first+last FL2VA signatures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"first_frame": ("IMAGE",)},
            "optional": {"last_frame": ("IMAGE",)},
        }

    RETURN_TYPES = (FL_KEYFRAMES_TYPE,)
    RETURN_NAMES = ("keyframes",)
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/fl2va"

    def build(self, first_frame: Any, last_frame: Any | None = None):
        first_width, first_height = _single_image_dimensions(
            first_frame, "first_frame"
        )
        items = [
            make_fl2va_keyframe(
                first_frame,
                0,
                display_width=first_width,
                display_height=first_height,
            )
        ]
        if last_frame is not None:
            last_width, last_height = _single_image_dimensions(
                last_frame, "last_frame"
            )
            items.append(
                make_fl2va_keyframe(
                    last_frame,
                    -1,
                    display_width=last_width,
                    display_height=last_height,
                )
            )
        return (validate_fl2va_keyframes(items),)


class MiniMaxH3FL2VALastFrameCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"last_frame": ("IMAGE",)}}

    RETURN_TYPES = (FL_KEYFRAMES_TYPE,)
    RETURN_NAMES = ("keyframes",)
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/fl2va"

    def build(self, last_frame: Any):
        width, height = _single_image_dimensions(last_frame, "last_frame")
        item = make_fl2va_keyframe(
            last_frame,
            -1,
            display_width=width,
            display_height=height,
        )
        return (validate_fl2va_keyframes([item]),)


class MiniMaxH3FL2VATarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframes": (FL_KEYFRAMES_TYPE,),
                "aspect_ratio": (list(H3_ASPECT_RATIOS), {"default": "auto"}),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 5.0, "max": 15.0, "step": 0.1},
                ),
            },
            "optional": {
                "width": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32}),
            },
        }

    RETURN_TYPES = (TARGET_TYPE, "STRING")
    RETURN_NAMES = ("target", "shape_info")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/fl2va"

    def build(
        self,
        keyframes: Any,
        aspect_ratio: str,
        duration_seconds: float,
        width: int = 0,
        height: int = 0,
    ):
        clean_keyframes = validate_fl2va_keyframes(keyframes)
        target = resolve_fl2va_target_v2(
            aspect_ratio=str(aspect_ratio),
            duration_seconds=float(duration_seconds),
            keyframes=clean_keyframes,
            width=width,
            height=height,
        )
        clean = validate_target_v2(
            target, expected_task=H3_TASK_FL2VA, require_resolved=True
        )
        return (clean, _target_shape_info(clean))


class MiniMaxH3Ref2VAImageReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"references": (REFERENCES_TYPE,)},
        }

    RETURN_TYPES = (REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/ref2va"

    def append(self, image: Any, references: Mapping[str, Any] | None = None):
        width, height = _single_image_dimensions(image, "image")
        reference = make_ref2va_reference(
            "image",
            image,
            display_width=width,
            display_height=height,
        )
        return (append_ref2va_reference(references, reference),)


class MiniMaxH3Ref2VAAudioReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {"references": (REFERENCES_TYPE,)},
        }

    RETURN_TYPES = (REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/ref2va"

    def append(self, audio: Any, references: Mapping[str, Any] | None = None):
        waveform, _, duration = _audio_metadata(audio, "audio")
        if int(waveform.shape[1]) > H3_AUDIO_CHANNELS:
            raise ValueError(
                "audio 超过双声道且 Comfy AUDIO 不含 channel layout；请先转换为 "
                "mono/stereo，或改用会通过 ffmpeg '-ac 2' 下混的文件/视频引用路径"
            )
        reference = make_ref2va_reference(
            "audio",
            audio,
            has_audio=True,
            audio_duration_seconds=duration,
        )
        return (append_ref2va_reference(references, reference),)


class MiniMaxH3Ref2VAVideoReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "reference_type": (
                    ["video", "video_audio"],
                    {"default": "video"},
                ),
            },
            "optional": {"references": (REFERENCES_TYPE,)},
        }

    RETURN_TYPES = (REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/ref2va"

    def append(
        self,
        video: Any,
        reference_type: str,
        references: Mapping[str, Any] | None = None,
    ):
        width, height = _video_dimensions(video, "video")
        has_audio, audio_duration = _probe_video_audio(video, "video")
        if reference_type == "video_audio" and not has_audio:
            raise ValueError("video_audio reference 必须包含音轨")
        reference = make_ref2va_reference(
            str(reference_type),
            video,
            display_width=width,
            display_height=height,
            has_audio=has_audio,
            audio_duration_seconds=audio_duration,
        )
        return (append_ref2va_reference(references, reference),)


class MiniMaxH3Ref2VATarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": (REFERENCES_TYPE,),
                "aspect_ratio": (list(H3_ASPECT_RATIOS), {"default": "auto"}),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": "0 表示从唯一的实际音频 reference 推导时长。",
                    },
                ),
            },
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "与 height 同时设为 0 时按 aspect_ratio；否则使用手动输出宽度。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "与 width 同时设为 0 时按 aspect_ratio；否则使用手动输出高度。",
                    },
                ),
            },
        }

    RETURN_TYPES = (TARGET_TYPE, "STRING")
    RETURN_NAMES = ("target", "shape_info")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/ref2va"

    def build(
        self,
        references: Mapping[str, Any],
        aspect_ratio: str,
        duration_seconds: float,
        width: int = 0,
        height: int = 0,
    ):
        clean_references = validate_ref2va_references(references)
        requested_duration = (
            None if float(duration_seconds) == 0.0 else float(duration_seconds)
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio=str(aspect_ratio),
            duration_seconds=requested_duration,
            references=clean_references,
            width=width,
            height=height,
        )
        clean = validate_target_v2(
            target, expected_task=H3_TASK_REF2VA, require_resolved=True
        )
        clean["reference_order_fingerprint"] = condition_order_fingerprint(
            H3_TASK_REF2VA, clean_references
        )
        return (clean, _target_shape_info(clean))


class MiniMaxH3FL2VAEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "target": (TARGET_TYPE,),
                "keyframes": (FL_KEYFRAMES_TYPE,),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/fl2va"

    def encode(
        self,
        h3_text_encoder: Mapping[str, Any],
        h3_vae_bundle: Mapping[str, Any],
        target: Mapping[str, Any],
        keyframes: Any,
        prompt: str,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        clean_target = validate_target_v2(
            target, expected_task=H3_TASK_FL2VA, require_resolved=True
        )
        clean_keyframes = validate_fl2va_keyframes(
            keyframes, frame_count=int(clean_target["frame_count"])
        )
        progress_total = 3 + 2 * len(clean_keyframes)
        progress_current = 0
        progress = _new_progress(progress_total)
        _preflight_target_conditions(
            task=H3_TASK_FL2VA,
            target=clean_target,
            conditions=clean_keyframes,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        compatibility = validate_component_compatibility(
            task=H3_TASK_FL2VA,
            text_encoder=h3_text_encoder,
            vae=h3_vae_bundle,
            target=clean_target,
        )
        text_wrapper = compatibility["text_encoder"]
        vae_wrapper = compatibility["vae"]
        release_fingerprint = _require_same_release(
            ("text_encoder", text_wrapper), ("vae", vae_wrapper)
        )
        prepared_images = []
        media = _runtime_module("media_conditioning")
        for ordinal, item in enumerate(clean_keyframes):
            _check_interrupted()
            source = _image_tensor_to_pil(item["media"], f"keyframes[{ordinal}]")
            prepared_images.append(
                media.prepare_fl_keyframe_canvas(
                    source,
                    target_width=int(clean_target["width"]),
                    target_height=int(clean_target["height"]),
                    keyframe_ordinal=ordinal,
                )
            )
            progress_current += 1
            _advance_progress(progress, progress_current, progress_total)

        _check_interrupted()
        text_handle = _unwrap_runtime_handle(
            text_wrapper, H3_TEXT_ENCODER_SCHEMA_V2, "h3_text_encoder"
        )
        encoded = _multimodal_qwen_encode(
            text_handle,
            task=H3_TASK_FL2VA,
            prompt=prompt,
            images=prepared_images,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)

        video_vae, _ = _vae_bundle_components(vae_wrapper)
        _require_comfy()
        model_management.unload_all_models()
        load_device = model_management.get_torch_device()
        condition_blocks = []
        with _vae_device_session(video_vae, load_device):
            for index, (item, prepared) in enumerate(
                zip(clean_keyframes, prepared_images, strict=True)
            ):
                _check_interrupted()
                pixels = _pil_to_image_tensor(prepared)
                condition_blocks.append(
                    media.encode_visual_condition_rows(
                        video_vae,
                        pixels,
                        condition_index=index,
                        kind="image",
                        process_image=True,
                        semantic_frame_index=int(item["frame_index"]),
                        resolved_frame_index=int(item["resolved_frame_index"]),
                    )
                )
                progress_current += 1
                _advance_progress(progress, progress_current, progress_total)

        conditioning = make_conditioning_v2(
            H3_TASK_FL2VA,
            prompt,
            encoded["prompt_embeds"],
            conditions=clean_keyframes,
            text_token_tags=encoded["text_token_tags"],
        )
        conditioning.update(
            {
                "condition_blocks": condition_blocks,
                "target": clean_target,
                "target_fingerprint": _target_fingerprint(clean_target),
                "release_fingerprint": release_fingerprint,
                "text_encoder_fingerprint": _wrapper_component_fingerprint(
                    text_wrapper,
                    component_kind="text_encoder",
                    path_key="text_encoder_path",
                    fingerprint_key="text_encoder_fingerprint",
                ),
                "vae_fingerprint": _wrapper_component_fingerprint(
                    vae_wrapper,
                    component_kind="vae",
                    path_key="vae_path",
                    fingerprint_key="vae_fingerprint",
                ),
            }
        )
        clean_conditioning = validate_conditioning_v2(
            conditioning, expected_task=H3_TASK_FL2VA
        )
        # Repeat the complete cross-value gate after the expensive encoders:
        # validators added by newer wire-contract versions must see the final
        # condition rows and fingerprints before anything reaches sampling.
        validate_component_compatibility(
            task=H3_TASK_FL2VA,
            text_encoder=text_wrapper,
            vae=vae_wrapper,
            target=clean_target,
            conditioning=clean_conditioning,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        return (clean_conditioning,)


class MiniMaxH3Ref2VAEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "target": (TARGET_TYPE,),
                "references": (REFERENCES_TYPE,),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/ref2va"

    def encode(
        self,
        h3_text_encoder: Mapping[str, Any],
        h3_vae_bundle: Mapping[str, Any],
        target: Mapping[str, Any],
        references: Mapping[str, Any],
        prompt: str,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        clean_target = validate_target_v2(
            target, expected_task=H3_TASK_REF2VA, require_resolved=True
        )
        clean_references = validate_ref2va_references(references)
        progress_total = 3 + 3 * len(clean_references)
        progress_current = 0
        progress = _new_progress(progress_total)
        _preflight_target_conditions(
            task=H3_TASK_REF2VA,
            target=clean_target,
            conditions=clean_references,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        compatibility = validate_component_compatibility(
            task=H3_TASK_REF2VA,
            text_encoder=h3_text_encoder,
            vae=h3_vae_bundle,
            target=clean_target,
        )
        text_wrapper = compatibility["text_encoder"]
        vae_wrapper = compatibility["vae"]
        release_fingerprint = _require_same_release(
            ("text_encoder", text_wrapper), ("vae", vae_wrapper)
        )
        media = _runtime_module("media_conditioning")
        prepared: list[dict[str, Any]] = []
        qwen_images: list[Any] = []
        qwen_videos: list[Any] = []
        condition_labels: list[tuple[str, int]] = []
        ordinals = {"image": 0, "audio": 0, "video": 0}

        with _h3_temporary_directory("h3_ref2va_") as workdir_value:
            workdir = Path(workdir_value)
            for index, item in enumerate(clean_references):
                _check_interrupted()
                kind = str(item["type"])
                record: dict[str, Any] = {
                    "condition_index": index,
                    "kind": kind,
                    "material_fingerprint": str(item["material_fingerprint"]),
                }
                if kind == "image":
                    source = _image_tensor_to_pil(item["media"], f"references[{index}]")
                    image = media.prepare_reference_image(source)
                    record["visual_pixels"] = _pil_to_image_tensor(image)
                    qwen_images.append(image)
                    ordinals["image"] += 1
                    condition_labels.append(("image", ordinals["image"]))
                elif kind == "audio":
                    waveform, sample_rate, _ = _audio_metadata(
                        item["media"], f"references[{index}]"
                    )
                    media.build_reference_audio_plan(
                        kind="audio",
                        input_has_audio=True,
                        source_channels=int(waveform.shape[1]),
                        source_sample_rate=sample_rate,
                    )
                    record.update(
                        {"audio_waveform": waveform, "audio_sample_rate": sample_rate}
                    )
                    ordinals["audio"] += 1
                    condition_labels.append(("audio", ordinals["audio"]))
                else:
                    video = item["media"]
                    item_dir = workdir / f"condition_{index}"
                    item_dir.mkdir(parents=True, exist_ok=True)
                    source_path = _video_source_path(video, item_dir)
                    source_meta = _probe_video_path(source_path)
                    plan = media.build_reference_video_plan(
                        source_meta,
                        target_frame_count=int(clean_target["frame_count"]),
                        kind=kind,
                    )
                    materialized = media.execute_reference_video_plan(
                        source_path,
                        plan,
                        workdir=item_dir,
                        probe=_probe_video_path,
                        interrupt_check=_check_interrupted,
                    )
                    prepared_meta = _probe_video_path(materialized["prepared_path"])
                    presentation = _runtime_module("presentation")
                    sample_indices, _timestamps = (
                        presentation.minimax_h3_qwen_video_sample_plan(
                            int(prepared_meta.frame_count),
                            source_fps=float(prepared_meta.fps),
                        )
                    )
                    qwen_videos.append(
                        media.decode_reference_video_samples(
                            materialized["prepared_path"],
                            sample_indices,
                            interrupt_check=_check_interrupted,
                        )
                    )
                    frames = _decode_video_file(materialized["prepared_path"])
                    record["visual_pixels"] = frames.unsqueeze(0)
                    has_audio = bool(materialized["input_has_audio"])
                    if has_audio:
                        audio_plan = media.build_reference_audio_plan(
                            kind=kind,
                            input_has_audio=True,
                        )
                        extracted = media.execute_reference_audio_plan(
                            materialized["original_path"],
                            audio_plan,
                            workdir=item_dir,
                            interrupt_check=_check_interrupted,
                        )
                        audio = _load_pcm_wav(extracted["audio_path"])
                        waveform, sample_rate, _ = _audio_metadata(
                            audio, f"references[{index}].soundtrack"
                        )
                        record.update(
                            {
                                "audio_waveform": waveform,
                                "audio_sample_rate": sample_rate,
                            }
                        )
                        ordinals["audio"] += 1
                        condition_labels.append(("audio", ordinals["audio"]))
                    elif kind == "video_audio":
                        raise ValueError("video_audio reference 必须包含音轨")
                    ordinals["video"] += 1
                    condition_labels.append(("video", ordinals["video"]))
                prepared.append(record)
                progress_current += 1
                _advance_progress(progress, progress_current, progress_total)

            _check_interrupted()
            text_handle = _unwrap_runtime_handle(
                text_wrapper, H3_TEXT_ENCODER_SCHEMA_V2, "h3_text_encoder"
            )
            encoded = _multimodal_qwen_encode(
                text_handle,
                task=H3_TASK_REF2VA,
                prompt=prompt,
                images=qwen_images,
                condition_labels=condition_labels,
                videos=qwen_videos,
            )
            progress_current += 1
            _advance_progress(progress, progress_current, progress_total)

            video_vae, audio_vae = _vae_bundle_components(vae_wrapper)
            _require_comfy()
            model_management.unload_all_models()
            load_device = model_management.get_torch_device()
            visual_blocks: dict[int, Mapping[str, Any]] = {}
            audio_blocks: dict[int, Mapping[str, Any]] = {}
            with _vae_device_session(video_vae, load_device):
                for record in prepared:
                    _check_interrupted()
                    pixels = record.get("visual_pixels")
                    if pixels is not None:
                        index = int(record["condition_index"])
                        kind = str(record["kind"])
                        visual_blocks[index] = media.encode_visual_condition_rows(
                            video_vae,
                            pixels,
                            condition_index=index,
                            kind=kind,
                            process_image=(kind == "image"),
                            material_fingerprint=str(
                                record["material_fingerprint"]
                            ),
                        )
                    progress_current += 1
                    _advance_progress(progress, progress_current, progress_total)

            with _vae_device_session(audio_vae, load_device):
                for record in prepared:
                    _check_interrupted()
                    waveform = record.get("audio_waveform")
                    if waveform is not None:
                        index = int(record["condition_index"])
                        audio_blocks[index] = media.encode_audio_condition_rows(
                            audio_vae,
                            waveform,
                            condition_index=index,
                            kind=str(record["kind"]),
                            sample_rate=int(record["audio_sample_rate"]),
                            material_fingerprint=str(
                                record["material_fingerprint"]
                            ),
                        )
                    progress_current += 1
                    _advance_progress(progress, progress_current, progress_total)

        condition_blocks = [
            media.merge_condition_blocks(
                visual_blocks.get(index), audio_blocks.get(index)
            )
            for index in range(len(clean_references))
        ]
        conditioning = make_conditioning_v2(
            H3_TASK_REF2VA,
            prompt,
            encoded["prompt_embeds"],
            conditions=clean_references,
            text_token_tags=encoded["text_token_tags"],
        )
        conditioning.update(
            {
                "condition_blocks": condition_blocks,
                "condition_labels": condition_labels,
                "target": clean_target,
                "target_fingerprint": _target_fingerprint(clean_target),
                "release_fingerprint": release_fingerprint,
                "text_encoder_fingerprint": _wrapper_component_fingerprint(
                    text_wrapper,
                    component_kind="text_encoder",
                    path_key="text_encoder_path",
                    fingerprint_key="text_encoder_fingerprint",
                ),
                "vae_fingerprint": _wrapper_component_fingerprint(
                    vae_wrapper,
                    component_kind="vae",
                    path_key="vae_path",
                    fingerprint_key="vae_fingerprint",
                ),
            }
        )
        clean_conditioning = validate_conditioning_v2(
            conditioning, expected_task=H3_TASK_REF2VA
        )
        validate_component_compatibility(
            task=H3_TASK_REF2VA,
            text_encoder=text_wrapper,
            vae=vae_wrapper,
            target=clean_target,
            conditioning=clean_conditioning,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        return (clean_conditioning,)


class MiniMaxH3T2VATextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/conditioning"

    def encode(self, h3_text_encoder: Mapping[str, Any], prompt: str):
        handle = _unwrap_runtime_handle(
            h3_text_encoder, H3_TEXT_ENCODER_SCHEMA, "h3_text_encoder"
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        prompt_embeds = _encode_prompt(handle, prompt)
        return (make_t2va_conditioning(prompt, prompt_embeds),)


class MiniMaxH3UnsupportedConditioning:
    """Legacy fail-closed placeholder retained for old serialized workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (
                    [H3_TASK_FL2VA, H3_TASK_REF2VA],
                    {"default": H3_TASK_FL2VA},
                )
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "fail"
    CATEGORY = f"{CATEGORY}/conditioning"

    def fail(self, task: str):
        normalized = normalize_task(task)
        destination = (
            "MiniMaxH3FL2VAEncode"
            if normalized == H3_TASK_FL2VA
            else "MiniMaxH3Ref2VAEncode"
        )
        raise H3TaskNotImplementedError(
            "MiniMaxH3UnsupportedConditioning 是旧工作流的迁移错误占位节点；"
            f"请删除它并连接新的 {destination} 节点。"
        )


class MiniMaxH3EmptyAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"target": (TARGET_TYPE,)}}

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY}/latent"

    def generate(self, target: Mapping[str, Any]):
        t = _require_torch()
        is_v2 = (
            isinstance(target, Mapping)
            and target.get("schema") == H3_TARGET_SCHEMA_V2
        )
        clean_target = (
            validate_target_v2(target, require_resolved=True)
            if is_v2
            else validate_target(target)
        )
        # Shape carrier only.  The sampler replaces these zeros with the two
        # independently-seeded source-compatible noise streams.
        video = t.zeros(
            1,
            24,
            int(clean_target["video_latent_t"]),
            int(clean_target["video_latent_h"]),
            int(clean_target["video_latent_w"]),
            dtype=t.float32,
            device="cpu",
        )
        audio = t.zeros(
            2,
            32,
            int(clean_target["audio_latent_t"]),
            dtype=t.float32,
            device="cpu",
        )
        return (
            {
                "schema": H3_AV_LATENT_SCHEMA_V2 if is_v2 else H3_AV_LATENT_SCHEMA,
                "task": str(clean_target["task"]),
                **(
                    {
                        "partition": str(clean_target["partition"]),
                        "target_fingerprint": _target_fingerprint(clean_target),
                    }
                    if is_v2
                    else {}
                ),
                "target": clean_target,
                "video": video,
                "audio": audio,
                "sampled": False,
            },
        )


class MiniMaxH3DualSigmaSampler:
    """Task-aware dual-sigma sampler; standard KSampler cannot express H3 state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_model": (MODEL_TYPE,),
                "conditioning": (CONDITIONING_TYPE,),
                "av_latent": (AV_LATENT_TYPE,),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": (1 << 63) - 1,
                    },
                ),
                "sigma_points": (
                    "INT",
                    {
                        "default": H3_DEFAULT_SIGMA_POINTS,
                        "min": 2,
                        "max": 1000,
                        "tooltip": (
                            "遵循原仓库语义：50 个 sigma 点产生 49 次 DiT forward。"
                        ),
                    },
                ),
                "video_shift": (
                    "FLOAT",
                    {
                        "default": H3_DEFAULT_VIDEO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "audio_shift": (
                    "FLOAT",
                    {
                        "default": H3_DEFAULT_AUDIO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "cache_dit": (
                    list(CACHE_DIT_MODE_CHOICES),
                    {
                        "default": CACHE_DIT_MODE_OFF,
                        "tooltip": (
                            "官方 Cache-DiT 近似加速。off=关闭；auto=命中已验证 "
                            "1344×768/124f/50steps 合同才启用；minimax-h3-cache-v1="
                            "强制该 profile（不匹配则报错）；manual=cookbook 旋钮。"
                            "需 pip install 'cache-dit>=1.3.0'。不可作 consistency GT。"
                        ),
                    },
                ),
            },
            "optional": {
                "cache_dit_rdt": (
                    "FLOAT",
                    {
                        "default": CACHE_DIT_RDT_COOKBOOK,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "仅 cache_dit=manual 生效；残差阈值，越大越快越糙。",
                    },
                ),
                "cache_dit_mc": (
                    "INT",
                    {
                        "default": CACHE_DIT_MC,
                        "min": 1,
                        "max": 32,
                        "tooltip": "仅 cache_dit=manual：最大连续缓存步数。",
                    },
                ),
                "cache_dit_warmup": (
                    "INT",
                    {
                        "default": CACHE_DIT_WARMUP,
                        "min": 0,
                        "max": 64,
                        "tooltip": "仅 cache_dit=manual：缓存前 warmup 步数。",
                    },
                ),
            },
        }

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("sampled_av_latent",)
    FUNCTION = "sample"
    CATEGORY = f"{CATEGORY}/sampling"

    def sample(
        self,
        h3_model: Mapping[str, Any],
        conditioning: Mapping[str, Any],
        av_latent: Mapping[str, Any],
        seed: int,
        sigma_points: int,
        video_shift: float,
        audio_shift: float,
        cache_dit: str = CACHE_DIT_MODE_OFF,
        cache_dit_rdt: float = CACHE_DIT_RDT_COOKBOOK,
        cache_dit_mc: int = CACHE_DIT_MC,
        cache_dit_warmup: int = CACHE_DIT_WARMUP,
    ):
        is_v2 = (
            isinstance(conditioning, Mapping)
            and conditioning.get("schema") == H3_CONDITIONING_SCHEMA_V2
        )
        progress_offset = 1 if is_v2 else 0
        progress_total = max(1, int(sigma_points) - 1) + progress_offset
        progress_bar = (
            ProgressBar(progress_total) if ProgressBar is not None else None
        )
        if is_v2:
            clean_conditioning = validate_conditioning_v2(conditioning)
            task = str(clean_conditioning["task"])
            clean_latent = validate_av_latent_v2(av_latent, expected_task=task)
            compatibility = validate_component_compatibility(
                task=task,
                model=h3_model,
                target=clean_latent["target"],
                conditioning=clean_conditioning,
                av_latent=clean_latent,
            )
            model_wrapper = compatibility["model"]
            release_fingerprint = _wrapper_release_fingerprint(model_wrapper)
            _wrapper_component_fingerprint(
                model_wrapper,
                component_kind="transformer",
                path_key="transformer_path",
                fingerprint_key="transformer_fingerprint",
            )
            encoded_release = clean_conditioning.get("release_fingerprint")
            if encoded_release != release_fingerprint:
                raise ValueError(
                    "conditioning 的 TE/VAE release 与 sampler DiT release 不一致"
                )
            _check_interrupted()
            packed = _build_v2_packed(
                clean_conditioning,
                clean_latent["target"],
            )
            _advance_progress(progress_bar, 1, progress_total)
            sampler = sample_h3
        else:
            clean_conditioning = conditioning
            clean_latent = validate_av_latent(av_latent)
            model_wrapper = h3_model
            packed = _build_t2va_packed(
                conditioning["prompt_embeds"],
                clean_latent["target"],
            )
            sampler = sample_t2va

        def update_progress(current: int, total: int):
            if progress_bar is not None:
                progress_bar.update_absolute(
                    int(current) + progress_offset,
                    progress_total,
                )

        def check_cancelled():
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()

        with _transformer_session(model_wrapper) as transformer:
            output = sampler(
                transformer=transformer,
                conditioning=clean_conditioning,
                av_latent=clean_latent,
                packed=packed,
                seed=int(seed),
                sigma_points=int(sigma_points),
                video_shift=float(video_shift),
                audio_shift=float(audio_shift),
                progress=update_progress,
                check_cancelled=check_cancelled,
                cache_dit=str(cache_dit),
                cache_dit_rdt=float(cache_dit_rdt),
                cache_dit_mc=int(cache_dit_mc),
                cache_dit_warmup=int(cache_dit_warmup),
            )
        if is_v2:
            output.update(
                {
                    "schema": H3_AV_LATENT_SCHEMA_V2,
                    "partition": partition_for_task(str(clean_conditioning["task"])),
                    "release_fingerprint": release_fingerprint,
                    "target_fingerprint": _target_fingerprint(clean_latent["target"]),
                    "vae_fingerprint": clean_conditioning.get("vae_fingerprint"),
                }
            )
            validate_av_latent_v2(output, expected_task=str(clean_conditioning["task"]))
        return (output,)


class MiniMaxH3DecodeAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "sampled_av_latent": (AV_LATENT_TYPE,),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "decode"
    CATEGORY = f"{CATEGORY}/decode"

    def decode(
        self,
        h3_vae_bundle: Mapping[str, Any],
        sampled_av_latent: Mapping[str, Any],
    ):
        is_v2 = (
            isinstance(sampled_av_latent, Mapping)
            and sampled_av_latent.get("schema") == H3_AV_LATENT_SCHEMA_V2
        )
        if is_v2:
            latent = validate_av_latent_v2(sampled_av_latent)
            task = str(latent["task"])
            compatibility = validate_component_compatibility(
                task=task,
                vae=h3_vae_bundle,
                target=latent["target"],
                av_latent=latent,
            )
            vae_wrapper = compatibility["vae"]
            expected_release = latent.get("release_fingerprint")
            actual_release = _wrapper_release_fingerprint(vae_wrapper)
            if expected_release != actual_release:
                raise ValueError("sampled latent 与 Decode VAE 不属于同一 release")
            expected_vae = latent.get("vae_fingerprint")
            actual_vae = _wrapper_component_fingerprint(
                vae_wrapper,
                component_kind="vae",
                path_key="vae_path",
                fingerprint_key="vae_fingerprint",
            )
            if not expected_vae or expected_vae != actual_vae:
                raise ValueError("sampled latent 与 Decode VAE 的 component fingerprint 不一致")
        else:
            if (
                not isinstance(h3_vae_bundle, Mapping)
                or h3_vae_bundle.get("schema") != H3_VAE_SCHEMA
            ):
                raise TypeError(
                    "h3_vae_bundle 端口不是 MiniMax H3 Direct VAE Loader 的输出"
                )
            vae_wrapper = h3_vae_bundle
            latent = validate_av_latent(sampled_av_latent)
        if not latent.get("sampled", False):
            raise ValueError(
                "输入仍是空 latent；请先连接 MiniMax H3 Dual Sigma Sampler"
            )
        bundle = vae_wrapper.get("bundle")
        video_vae = _wrapper_value(bundle, "video_vae")
        audio_vae = _wrapper_value(bundle, "audio_vae")
        if video_vae is None or audio_vae is None:
            raise RuntimeError(
                "VAE bundle 必须同时包含 video_vae 和 audio_vae"
            )
        _require_comfy()
        # Decode components sequentially.  Keeping either the 62 GB DiT or
        # both VAEs resident defeats ComfyUI's memory manager on real H3
        # hardware.
        model_management.unload_all_models()
        load_device = model_management.get_torch_device()
        with _vae_device_session(video_vae, load_device):
            frames = _decode_video(
                video_vae, latent["video"], latent["target"]
            )

        with _vae_device_session(audio_vae, load_device):
            audio = _decode_audio(
                audio_vae,
                latent["audio"],
                sample_count=int(round(float(latent["target"]["duration_seconds"]) * 32000))
                if latent["target"].get("duration_seconds") is not None
                else None,
            )
        return (frames, audio)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DirectModelLoader": MiniMaxH3DirectModelLoader,
    "MiniMaxH3DirectTextEncoderLoader": MiniMaxH3DirectTextEncoderLoader,
    "MiniMaxH3DirectVAELoader": MiniMaxH3DirectVAELoader,
    "MiniMaxH3FL2VAModelLoader": MiniMaxH3FL2VAModelLoader,
    "MiniMaxH3FL2VATextEncoderLoader": MiniMaxH3FL2VATextEncoderLoader,
    "MiniMaxH3FL2VAVAELoader": MiniMaxH3FL2VAVAELoader,
    "MiniMaxH3Ref2VAModelLoader": MiniMaxH3Ref2VAModelLoader,
    "MiniMaxH3Ref2VATextEncoderLoader": MiniMaxH3Ref2VATextEncoderLoader,
    "MiniMaxH3Ref2VAVAELoader": MiniMaxH3Ref2VAVAELoader,
    "MiniMaxH3T2VATarget": MiniMaxH3T2VATarget,
    "MiniMaxH3T2VATextEncode": MiniMaxH3T2VATextEncode,
    "MiniMaxH3FL2VAFirstFrameCondition": MiniMaxH3FL2VAFirstFrameCondition,
    "MiniMaxH3FL2VALastFrameCondition": MiniMaxH3FL2VALastFrameCondition,
    "MiniMaxH3FL2VATarget": MiniMaxH3FL2VATarget,
    "MiniMaxH3FL2VAEncode": MiniMaxH3FL2VAEncode,
    "MiniMaxH3Ref2VAImageReference": MiniMaxH3Ref2VAImageReference,
    "MiniMaxH3Ref2VAAudioReference": MiniMaxH3Ref2VAAudioReference,
    "MiniMaxH3Ref2VAVideoReference": MiniMaxH3Ref2VAVideoReference,
    "MiniMaxH3Ref2VATarget": MiniMaxH3Ref2VATarget,
    "MiniMaxH3Ref2VAEncode": MiniMaxH3Ref2VAEncode,
    "MiniMaxH3UnsupportedConditioning": MiniMaxH3UnsupportedConditioning,
    "MiniMaxH3EmptyAVLatent": MiniMaxH3EmptyAVLatent,
    "MiniMaxH3DualSigmaSampler": MiniMaxH3DualSigmaSampler,
    "MiniMaxH3DecodeAV": MiniMaxH3DecodeAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectModelLoader": "MiniMax H3 Model Loader (Direct)",
    "MiniMaxH3DirectTextEncoderLoader": "MiniMax H3 Qwen3-VL Loader (Direct)",
    "MiniMaxH3DirectVAELoader": "MiniMax H3 Dual VAE Loader (Direct)",
    "MiniMaxH3FL2VAModelLoader": "MiniMax H3 FL2VA Model Loader (Direct)",
    "MiniMaxH3FL2VATextEncoderLoader": "MiniMax H3 FL2VA Qwen3-VL Loader (Direct)",
    "MiniMaxH3FL2VAVAELoader": "MiniMax H3 FL2VA Dual VAE Loader (Direct)",
    "MiniMaxH3Ref2VAModelLoader": "MiniMax H3 Ref2VA Model Loader (Direct)",
    "MiniMaxH3Ref2VATextEncoderLoader": "MiniMax H3 Ref2VA Qwen3-VL Loader (Direct)",
    "MiniMaxH3Ref2VAVAELoader": "MiniMax H3 Ref2VA Dual VAE Loader (Direct)",
    "MiniMaxH3T2VATarget": "MiniMax H3 T2VA Target",
    "MiniMaxH3T2VATextEncode": "MiniMax H3 T2VA Text Encode",
    "MiniMaxH3FL2VAFirstFrameCondition": "MiniMax H3 FL2VA First / First+Last",
    "MiniMaxH3FL2VALastFrameCondition": "MiniMax H3 FL2VA Last Only",
    "MiniMaxH3FL2VATarget": "MiniMax H3 FL2VA Target",
    "MiniMaxH3FL2VAEncode": "MiniMax H3 FL2VA Encode",
    "MiniMaxH3Ref2VAImageReference": "MiniMax H3 Ref2VA Image Reference",
    "MiniMaxH3Ref2VAAudioReference": "MiniMax H3 Ref2VA Audio Reference",
    "MiniMaxH3Ref2VAVideoReference": "MiniMax H3 Ref2VA Video Reference",
    "MiniMaxH3Ref2VATarget": "MiniMax H3 Ref2VA Target",
    "MiniMaxH3Ref2VAEncode": "MiniMax H3 Ref2VA Encode",
    "MiniMaxH3UnsupportedConditioning": (
        "MiniMax H3 Legacy Unsupported Conditioning (Migration Error)"
    ),
    "MiniMaxH3EmptyAVLatent": "MiniMax H3 Empty AV Latent",
    "MiniMaxH3DualSigmaSampler": "MiniMax H3 Dual Sigma Sampler",
    "MiniMaxH3DecodeAV": "MiniMax H3 Decode Video + Audio",
}


__all__ = [
    "AV_LATENT_TYPE",
    "CONDITIONING_TYPE",
    "FL_KEYFRAMES_TYPE",
    "MODEL_TYPE",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "REFERENCES_TYPE",
    "TARGET_TYPE",
    "TEXT_ENCODER_TYPE",
    "VAE_BUNDLE_TYPE",
]
