"""Local MiniMax-H3 component discovery.

The released H3 bundle follows a Diffusers-style ``model_index.json`` but the
actual component entry may be a conventional ``[library, class]`` pair, a
subfolder string, or an object containing ``path``/``subfolder``.  In the first
case the component's key is also its directory name.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

from .h3_settings import INT8_DIT_DIRNAME, INT8_TE_DIRNAME, VAE_MERGED_DIRNAME


class H3ComponentError(ValueError):
    """Raised when a local H3 checkpoint does not satisfy the node contract."""


H3_PARTITIONS = ("fl2va", "ref2va")
RH_GLOBAL_MODELS_DIR = Path("/data/ComfyUI/global/models")
H3_MODEL_ROOTS_ENV = "MINIMAX_H3_MODEL_ROOTS"


def read_json(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise H3ComponentError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise H3ComponentError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise H3ComponentError(f"Expected a JSON object in {path}")
    return value


def _looks_like_h3_root(path: Path) -> bool:
    """轻量目录特征判断：只检查路径存在，不读取模型文件内容。

    不能仅凭 ``model_index.json`` 判定——``models/diffusers`` 下大量无关
    模型也有该文件。
    """

    if not path.is_dir():
        return False
    # 正式 release：父目录下挂 FL2VA / Ref2VA
    if (path / "FL2VA").is_dir() or (path / "Ref2VA").is_dir():
        return True
    # 单分区根：同时具备 DiT + Qwen + 至少一种 VAE 目录
    has_core = (path / "transformer").is_dir() and (path / "text_encoder").is_dir()
    has_vae = (path / "video_vae").is_dir() or (path / "audio_vae").is_dir()
    if has_core and has_vae:
        return True
    # 名称兜底（目录尚未下全时仍希望出现在 COMBO）
    lowered = path.name.lower().replace("_", "-")
    return "minimax" in lowered and "h3" in lowered


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.expanduser())
        if key in seen:
            continue
        seen.add(key)
        out.append(path.expanduser())
    return out


def _model_bucket_paths(folder_paths_module=None) -> list[Path]:
    """Return configured model buckets plus RunningHub's global NFS buckets."""

    module = folder_paths_module
    if module is None:
        try:
            import folder_paths as module
        except ImportError:
            module = None

    buckets: list[Path] = []
    if module is not None:
        getter = getattr(module, "get_folder_paths", None)
        if callable(getter):
            try:
                buckets.extend(Path(item) for item in getter("diffusers"))
            except (KeyError, OSError, TypeError):
                pass
        models_dir = Path(getattr(module, "models_dir", "") or "")
        if str(models_dir):
            buckets.extend(
                (models_dir / "diffusers", models_dir / "minimax_h3")
            )

    global_models = Path(
        os.environ.get("MINIMAX_H3_GLOBAL_MODELS_DIR", str(RH_GLOBAL_MODELS_DIR))
    ).expanduser()
    buckets.extend(
        (global_models / "diffusers", global_models / "minimax_h3")
    )
    return _unique_paths(buckets)


def _h3_release_score(path: Path) -> int:
    """Prefer a release that contains explicit INT8 and merged-VAE artifacts."""

    partition = path / "FL2VA" if (path / "FL2VA").is_dir() else path
    score = 0
    if (partition / INT8_DIT_DIRNAME / "config.json").is_file():
        score += 16
    if (partition / INT8_TE_DIRNAME / "config.json").is_file():
        score += 8
    vae = partition / VAE_MERGED_DIRNAME
    if (
        (vae / "video_vae" / "config.json").is_file()
        and (vae / "audio_vae" / "config.json").is_file()
    ):
        score += 4
    if (partition / "transformer" / "config.json").is_file():
        score += 2
    if (partition / "text_encoder" / "config.json").is_file():
        score += 1
    return score


def list_h3_model_root_paths() -> list[Path]:
    """Enumerate concrete H3 roots, preferring complete persistent releases."""

    candidates: list[Path] = []
    raw_extra = os.environ.get(H3_MODEL_ROOTS_ENV, "")
    if raw_extra:
        candidates.extend(
            Path(item) for item in raw_extra.split(os.pathsep) if item.strip()
        )

    for base in _model_bucket_paths():
        # The official bundle name is known.  Probe it directly so object_info
        # generation never scans the entire RunningHub global NFS catalogue.
        candidates.append(base / "MiniMax-H3")
        if base == RH_GLOBAL_MODELS_DIR / "diffusers" or base == RH_GLOBAL_MODELS_DIR / "minimax_h3":
            continue
        if not base.is_dir():
            continue
        try:
            candidates.extend(child for child in base.iterdir() if child.is_dir())
        except OSError:
            continue

    roots = [
        path.resolve()
        for path in _unique_paths(candidates)
        if _looks_like_h3_root(path)
    ]
    return sorted(roots, key=lambda path: (-_h3_release_score(path), str(path)))


def list_h3_model_roots() -> list[str]:
    """枚举可用的 MiniMax-H3 根目录名，供 loader COMBO 使用。

    只扫描 ``models/diffusers`` 与 ``models/minimax_h3`` 下一层目录名，
    不打开 safetensors / 不解析 JSON，避免 Manifest 占位阶段读模型内容。
    目录为空时回退占位项，保证节点仍可注册。
    """

    fallback = ["MiniMax-H3"]
    names: list[str] = []
    seen: set[str] = set()
    for root in list_h3_model_root_paths():
        name = root.name
        if not name or name.startswith(".") or name in seen:
            continue
        seen.add(name)
        names.append(name)

    return names or list(fallback)


def model_root_path(value: str | Path) -> Path:
    """Resolve an explicit path or a folder under ComfyUI ``models/``.

    相对名搜索顺序（命中第一个存在的目录即返回）：
    1. ``models/<value>``
    2. ``models/diffusers/<value>``（RH 常见布局）
    3. ``models/minimax_h3/<value>``
    4. 当 ``value`` 为 MiniMax-H3 别名时，再试 ``models/diffusers/MiniMax-H3``
    5. ``models/diffusers/MiniMax-H3/<value>``（可直接填 ``FL2VA``）
    6. 当前工作目录下的 ``<value>``
    """

    raw = Path(value).expanduser()
    if raw.is_dir():
        return raw.resolve()

    try:
        import folder_paths
    except ImportError:
        folder_paths = None

    text = str(value or "").strip().strip("/\\")
    candidates: list[Path] = []
    if folder_paths is not None and text:
        models_dir = Path(folder_paths.models_dir)
        rel = Path(text)
        candidates.extend(
            [
                models_dir / rel,
                models_dir / "diffusers" / rel,
                models_dir / "minimax_h3" / rel,
            ]
        )
        alias = text.lower().replace("_", "-")
        if alias in {"minimax-h3", "minimaxh3", "h3"}:
            candidates.append(models_dir / "diffusers" / "MiniMax-H3")
            candidates.append(models_dir / "minimax_h3" / "MiniMax-H3")
        candidates.append(models_dir / "diffusers" / "MiniMax-H3" / rel)
        candidates.append(models_dir / "minimax_h3" / "MiniMax-H3" / rel)
    if text:
        rel = Path(text)
        for base in _model_bucket_paths(folder_paths):
            candidates.append(base / rel)
            candidates.append(base / "MiniMax-H3" / rel)
        if len(rel.parts) == 1:
            candidates.extend(
                root for root in list_h3_model_root_paths() if root.name == text
            )
    if text:
        candidates.append(Path.cwd() / Path(text))
    seen: set[str] = set()
    unique: list[Path] = []
    existing: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if candidate.is_dir():
            existing.append(candidate.resolve())
    if existing:
        # A RunningHub worker may retain an older local cache while the clean
        # release lives on the 11.31 global NFS mount.  Prefer the root that has
        # all explicitly selectable artifacts; an absolute path still returned
        # above and therefore always remains authoritative.
        return sorted(
            _unique_paths(existing),
            key=lambda path: (-_h3_release_score(path), str(path)),
        )[0]
    searched = ", ".join(str(item) for item in [raw, *unique])
    raise H3ComponentError(f"MiniMax-H3 model directory not found; searched: {searched}")


def _entry_subfolder(index: dict, key: str) -> str | None:
    entry = index.get(key)
    if isinstance(entry, str) and entry:
        return entry
    if isinstance(entry, dict):
        candidate = entry.get("path") or entry.get("subfolder")
        return candidate if isinstance(candidate, str) and candidate else key
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        # Standard Diffusers model-index entries name the library/class only.
        return key
    if entry is not None:
        return key
    return None


def resolve_component(
    model_root: str | Path,
    keys: Iterable[str],
    *,
    explicit: str | Path | None = None,
    required_files: tuple[str, ...] = (),
) -> Path:
    """Resolve one local component directory without downloading anything."""

    root = model_root_path(model_root)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = root / path
        if not path.is_dir():
            raise H3ComponentError(f"Component directory not found: {path}")
        missing = [item for item in required_files if not (path / item).exists()]
        if missing:
            raise H3ComponentError(
                f"Component directory {path} is missing required files: {missing}"
            )
        return path.resolve()

    index_path = root / "model_index.json"
    index = read_json(index_path) if index_path.is_file() else {}
    attempted: list[Path] = []
    for key in keys:
        subfolder = _entry_subfolder(index, key)
        if not subfolder:
            continue
        candidate = root / subfolder
        attempted.append(candidate)
        if candidate.is_dir() and all((candidate / item).exists() for item in required_files):
            return candidate.resolve()

    for key in keys:
        candidate = root / key
        attempted.append(candidate)
        if candidate.is_dir() and all((candidate / item).exists() for item in required_files):
            return candidate.resolve()

    if all((root / item).exists() for item in required_files):
        return root
    attempted_text = ", ".join(str(item) for item in attempted) or str(root)
    raise H3ComponentError(
        f"Could not resolve component {tuple(keys)!r} below {root}; "
        f"checked: {attempted_text}"
    )


def release_metadata(model_root: str | Path) -> dict:
    root = model_root_path(model_root)
    index_path = root / "model_index.json"
    if not index_path.is_file():
        return {}
    index = read_json(index_path)
    value = index.get("_minimax_h3")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise H3ComponentError("model_index.json._minimax_h3 must be an object")
    return dict(value)


def resolve_partition_root(
    model_root: str | Path,
    partition: str,
) -> Path:
    """Resolve either a release root or its immediate partition child.

    MiniMax H3 is commonly downloaded as a parent directory containing sibling
    ``FL2VA`` and ``Ref2VA`` releases.  Each child owns its own
    ``model_index.json``.  A root that already has a model index is never
    redirected to a sibling: its declared partition remains authoritative.
    """

    root = model_root_path(model_root)
    normalized = str(partition).strip().lower()
    if normalized not in H3_PARTITIONS:
        raise H3ComponentError(
            f"Unknown MiniMax H3 partition {partition!r}; expected one of "
            f"{H3_PARTITIONS!r}"
        )

    if (root / "model_index.json").is_file():
        metadata = release_metadata(root)
        declared = metadata.get("partition")
        if declared is not None:
            if not isinstance(declared, str):
                raise H3ComponentError(
                    "model_index.json._minimax_h3.partition must be a string"
                )
            if declared.strip().lower() != normalized:
                raise H3ComponentError(
                    f"Requested {normalized!r} weights, but {root / 'model_index.json'} "
                    f"declares partition {declared!r}"
                )
        return root

    matches: list[Path] = []
    inspected: list[Path] = []
    try:
        children = sorted(
            (child for child in root.iterdir() if child.is_dir()),
            key=lambda child: child.name.lower(),
        )
    except OSError as exc:
        raise H3ComponentError(f"Cannot inspect MiniMax H3 root {root}: {exc}") from exc

    for child in children:
        index_path = child / "model_index.json"
        if not index_path.is_file():
            continue
        inspected.append(index_path)
        metadata = release_metadata(child)
        declared = metadata.get("partition")
        if declared is not None and not isinstance(declared, str):
            raise H3ComponentError(
                f"{index_path}._minimax_h3.partition must be a string"
            )
        declared_match = (
            isinstance(declared, str)
            and declared.strip().lower() == normalized
        )
        name = child.name.strip().lower()
        named_match = name == normalized or name.endswith(f"-{normalized}")
        if declared_match or (declared is None and named_match):
            matches.append(child.resolve())

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise H3ComponentError(
            f"Multiple {normalized!r} release roots found below {root}: "
            + ", ".join(str(path) for path in matches)
        )

    # Preserve support for pre-release, metadata-free roots that contain the
    # components directly rather than a model_index.json.
    if any(
        (root / name).is_dir()
        for name in ("transformer", "dit", "text_encoder", "video_vae", "audio_vae")
    ):
        return root

    inspected_text = ", ".join(str(path) for path in inspected) or "no child model_index.json files"
    raise H3ComponentError(
        f"Could not find MiniMax H3 partition {normalized!r} below {root}; "
        f"inspected {inspected_text}"
    )


def release_sigma_shift_scales(metadata: dict) -> dict[str, float] | None:
    """Return validated per-modality release sigma shifts when declared."""

    raw = metadata.get("sigma_shift_scales")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise H3ComponentError(
            "model_index.json._minimax_h3.sigma_shift_scales must be an object"
        )
    missing = [name for name in ("video", "audio") if name not in raw]
    if missing:
        raise H3ComponentError(
            "model_index.json._minimax_h3.sigma_shift_scales is missing "
            + ", ".join(missing)
        )
    out: dict[str, float] = {}
    for name in ("video", "audio"):
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise H3ComponentError(
                f"sigma_shift_scales.{name} must be a positive finite number"
            )
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise H3ComponentError(
                f"sigma_shift_scales.{name} must be a positive finite number"
            )
        out[name] = value
    return out


def validate_t2va_partition(metadata: dict) -> None:
    """Reject a ref-only partition while allowing pre-release bundles."""

    if not metadata:
        return
    partition = metadata.get("partition")
    if partition is not None:
        if not isinstance(partition, str):
            raise H3ComponentError(
                "model_index.json._minimax_h3.partition must be a string"
            )
        if partition.strip().lower() != "fl2va":
            raise H3ComponentError(
                "T2VA requires the FL2VA checkpoint partition; "
                f"model_index declares {partition!r}"
            )
    tasks = metadata.get("tasks")
    if tasks is not None:
        if not isinstance(tasks, list) or any(
            not isinstance(task, str) for task in tasks
        ):
            raise H3ComponentError(
                "model_index.json._minimax_h3.tasks must be a list of strings"
            )
        normalized_tasks = {task.strip().lower() for task in tasks}
        if "t2va" not in normalized_tasks:
            raise H3ComponentError(
                f"This checkpoint partition does not advertise t2va; tasks={tasks!r}"
            )
    release_sigma_shift_scales(metadata)


__all__ = [
    "H3ComponentError",
    "H3_PARTITIONS",
    "list_h3_model_root_paths",
    "list_h3_model_roots",
    "model_root_path",
    "read_json",
    "release_metadata",
    "release_sigma_shift_scales",
    "resolve_component",
    "resolve_partition_root",
    "validate_t2va_partition",
]
