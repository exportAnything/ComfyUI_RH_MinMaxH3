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
H3_TASK_PARTITIONS = {
    "t2va": "fl2va",
    "fl2va": "fl2va",
    "ref2va": "ref2va",
}
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


def _partition_release_score(path: Path) -> int:
    """Score completeness within one already-admitted partition.

    BF16 base directories and alternate/quantized directories carry equal
    weight.  This score must never reintroduce the old FL2VA+INT8 bias when a
    caller has supplied task/component hints.
    """

    root = path.resolve()
    score = 1 if (root / "model_index.json").is_file() else 0
    for prefix in ("transformer", "text_encoder"):
        has_base = (root / prefix / "config.json").is_file()
        try:
            has_alternate = any(
                child.is_dir()
                and child.name.startswith(f"{prefix}_")
                and (child / "config.json").is_file()
                for child in root.iterdir()
            )
        except OSError:
            has_alternate = False
        # Completeness is variant-neutral: a release containing both BF16 and
        # INT8 copies must not outrank an otherwise equivalent BF16-only root
        # when the caller explicitly selected the BF16 component directory.
        if has_base or has_alternate:
            score += 8
    merged_vae = root / VAE_MERGED_DIRNAME
    has_merged_vae = (
        (merged_vae / "video_vae" / "config.json").is_file()
        and (merged_vae / "audio_vae" / "config.json").is_file()
    )
    has_direct_vae = (
        (root / "video_vae" / "config.json").is_file()
        and (root / "audio_vae" / "config.json").is_file()
    )
    if has_merged_vae or has_direct_vae:
        score += 8
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
        if base in {
            RH_GLOBAL_MODELS_DIR / "diffusers",
            RH_GLOBAL_MODELS_DIR / "minimax_h3",
        }:
            continue
        if not base.is_dir():
            continue
        try:
            candidates.extend(child for child in base.iterdir() if child.is_dir())
        except OSError:
            continue

    roots = _unique_paths(
        path.resolve()
        for path in _unique_paths(candidates)
        if _looks_like_h3_root(path)
    )
    return sorted(roots, key=lambda path: (-_h3_release_score(path), str(path)))


def list_h3_model_roots() -> list[str]:
    """枚举可用的 MiniMax-H3 根目录名，供 loader COMBO 使用。

    只扫描 ``models/diffusers`` 与 ``models/minimax_h3`` 下一层目录名，
    不打开 safetensors / 不解析 JSON，避免 Manifest 占位阶段读模型内容。
    目录为空时回退占位项，保证节点仍可注册。
    """

    fallback = ["MiniMax-H3"]
    roots = _unique_paths(
        path.resolve() for path in list_h3_model_root_paths()
    )
    counts: dict[str, int] = {}
    for root in roots:
        if root.name and not root.name.startswith("."):
            counts[root.name] = counts.get(root.name, 0) + 1
    choices: list[str] = []
    seen: set[str] = set()
    for root in roots:
        name = root.name
        if not name or name.startswith("."):
            continue
        choice = str(root.resolve()) if counts.get(name, 0) > 1 else name
        if choice in seen:
            continue
        seen.add(choice)
        choices.append(choice)

    return choices or list(fallback)


def model_root_path(
    value: str | Path,
    *,
    partition: str | None = None,
    required_component: str | Path | None = None,
    required_files: tuple[str, ...] = (),
) -> Path:
    """Resolve an explicit path or a folder under ComfyUI ``models/``.

    相对名搜索顺序（命中第一个存在的目录即返回）：
    1. ``models/<value>``
    2. ``models/diffusers/<value>``（RH 常见布局）
    3. ``models/minimax_h3/<value>``
    4. 当 ``value`` 为 MiniMax-H3 别名时，再试 ``models/diffusers/MiniMax-H3``
    5. ``models/diffusers/MiniMax-H3/<value>``（可直接填 ``FL2VA``）
    6. 当前工作目录下的 ``<value>``
    """

    raw_text = str(value or "").strip()
    raw = Path(raw_text).expanduser()
    has_hints = (
        partition is not None
        or required_component is not None
        or bool(required_files)
    )
    bare_name = (
        bool(raw_text)
        and not raw.is_absolute()
        and "/" not in raw_text
        and "\\" not in raw_text
        and raw_text not in (".", "..")
    )
    if raw.is_dir() and (not has_hints or not bare_name):
        resolved = raw.resolve()
        if not has_hints:
            return resolved
        # An explicit path is authoritative, but hints are still admission
        # checks.  Treat it as the sole candidate so a missing component fails
        # locally instead of falling back to another same-named release.
        return _select_hinted_model_root(
            raw_text,
            (resolved,),
            partition=partition,
            required_component=required_component,
            required_files=required_files,
        )
    if raw.is_absolute():
        raise H3ComponentError(f"MiniMax-H3 model directory not found: {raw}")

    try:
        import folder_paths
    except ImportError:
        folder_paths = None

    text = raw_text.strip("/\\")
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
        concrete = _unique_paths(path.resolve() for path in existing)
        if has_hints:
            return _select_hinted_model_root(
                text,
                concrete,
                partition=partition,
                required_component=required_component,
                required_files=required_files,
            )
        # A worker may retain an older local cache while the clean release
        # lives on a shared global model mount.  Prefer the root that has
        # all explicitly selectable artifacts; an absolute path still returned
        # above and therefore always remains authoritative.
        return sorted(
            concrete,
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


def _relative_component_path(value: str | Path, *, label: str) -> Path:
    """Parse an untrusted model-index/key path without allowing traversal."""

    text = str(value).strip()
    path = Path(text)
    if not text or path == Path(".") or path.is_absolute() or ".." in path.parts:
        raise H3ComponentError(
            f"{label} must be a non-empty relative path without '.' or '..': "
            f"{value!r}"
        )
    return path


def _resolve_contained_path(
    root: Path,
    candidate: Path,
    *,
    label: str,
    allow_root: bool = True,
) -> Path:
    """Resolve symlinks and require ``candidate`` to remain below ``root``."""

    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        relative = resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise H3ComponentError(
            f"{label} {resolved_candidate} resolves outside the selected "
            f"MiniMax H3 root {resolved_root}"
        ) from exc
    if not allow_root and relative == Path("."):
        raise H3ComponentError(
            f"{label} {resolved_candidate} must be a child of the selected "
            f"MiniMax H3 root {resolved_root}"
        )
    return resolved_candidate


def _missing_required_files(
    component_root: Path,
    required_files: Iterable[str | Path],
) -> list[str]:
    """Validate required files as contained regular files."""

    resolved_root = component_root.resolve()
    missing: list[str] = []
    for raw_item in required_files:
        item = _relative_component_path(
            raw_item,
            label="required component file",
        )
        candidate = _resolve_contained_path(
            resolved_root,
            resolved_root / item,
            label="Required component file",
        )
        if not candidate.is_file():
            missing.append(str(item))
    return missing


def _select_hinted_model_root(
    value: str,
    candidates: Iterable[Path],
    *,
    partition: str | None,
    required_component: str | Path | None,
    required_files: tuple[str, ...],
) -> Path:
    """Choose a concrete same-basename root using execution-time hints."""

    normalized_partition: str | None = None
    if partition is not None:
        normalized_partition = str(partition).strip().lower()
        if normalized_partition not in H3_PARTITIONS:
            raise H3ComponentError(
                f"Unknown MiniMax H3 partition hint {partition!r}; expected "
                f"one of {H3_PARTITIONS!r}"
            )

    component_hint: Path | None = None
    if required_component is not None:
        component_text = str(required_component).strip()
        if not component_text:
            raise H3ComponentError("required_component hint cannot be empty")
        component_hint = Path(component_text).expanduser()
        if not component_hint.is_absolute():
            component_hint = _relative_component_path(
                component_text,
                label="required_component hint",
            )
    file_hints = tuple(
        _relative_component_path(item, label="required_files hint")
        for item in required_files
    )

    viable: list[tuple[int, Path]] = []
    rejected: list[str] = []
    for raw_candidate in candidates:
        candidate = raw_candidate.resolve()
        try:
            partition_root = (
                _admit_partition_root(candidate, normalized_partition)
                if normalized_partition is not None
                else candidate
            )
            component_root = partition_root
            if component_hint is not None:
                if component_hint.is_absolute():
                    component_root = _resolve_contained_path(
                        partition_root,
                        component_hint,
                        label="Required component",
                    )
                else:
                    component_root = _resolve_contained_path(
                        partition_root,
                        partition_root / component_hint,
                        label="Required component",
                    )
                if not component_root.is_dir():
                    raise H3ComponentError(
                        f"required component directory is missing: {component_root}"
                    )
            missing = _missing_required_files(component_root, file_hints)
            if missing:
                raise H3ComponentError(
                    f"required component {component_root} is missing files {missing!r}"
                )
            score = _partition_release_score(partition_root)
            viable.append((score, candidate))
        except (H3ComponentError, OSError) as exc:
            rejected.append(f"{candidate}: {exc}")

    if not viable:
        detail = "; ".join(rejected) or "no existing candidates"
        raise H3ComponentError(
            f"No MiniMax-H3 root named {value!r} satisfies partition="
            f"{normalized_partition!r}, component={str(required_component)!r}: "
            f"{detail}"
        )
    best_score = max(score for score, _candidate in viable)
    winners = sorted(
        candidate for score, candidate in viable if score == best_score
    )
    if len(winners) > 1:
        raise H3ComponentError(
            f"Ambiguous MiniMax-H3 root {value!r} for partition="
            f"{normalized_partition!r}, component={str(required_component)!r}; "
            f"score={best_score}, candidates="
            + ", ".join(str(candidate) for candidate in winners)
            + ". Select one of these absolute paths explicitly."
        )
    return winners[0]


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
        resolved_root = root.resolve()
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise H3ComponentError(
                f"Explicit component {resolved_path} is outside the selected "
                f"MiniMax H3 partition root {resolved_root}; cross-partition "
                "component paths are not allowed"
            ) from exc
        missing = _missing_required_files(resolved_path, required_files)
        if missing:
            raise H3ComponentError(
                f"Component directory {path} is missing required files: {missing}"
            )
        return resolved_path

    index_path = root / "model_index.json"
    index = read_json(index_path) if index_path.is_file() else {}
    attempted: list[Path] = []
    for key in keys:
        subfolder = _entry_subfolder(index, key)
        if not subfolder:
            continue
        relative = _relative_component_path(
            subfolder,
            label=f"model_index component {key!r}",
        )
        candidate = root / relative
        attempted.append(candidate)
        resolved_candidate = _resolve_contained_path(
            root,
            candidate,
            label=f"Component {key!r}",
        )
        if resolved_candidate.is_dir() and not _missing_required_files(
            resolved_candidate,
            required_files,
        ):
            return resolved_candidate

    for key in keys:
        relative = _relative_component_path(
            str(key),
            label=f"component key {key!r}",
        )
        candidate = root / relative
        attempted.append(candidate)
        resolved_candidate = _resolve_contained_path(
            root,
            candidate,
            label=f"Component key {key!r}",
        )
        if resolved_candidate.is_dir() and not _missing_required_files(
            resolved_candidate,
            required_files,
        ):
            return resolved_candidate

    if not _missing_required_files(root, required_files):
        return root
    attempted_text = ", ".join(str(item) for item in attempted) or str(root)
    raise H3ComponentError(
        f"Could not resolve component {tuple(keys)!r} below {root}; "
        f"checked: {attempted_text}"
    )


def _release_metadata_at_root(root: Path) -> dict:
    """Read release metadata from one already-resolved concrete directory."""

    index_path = root / "model_index.json"
    if not index_path.is_file():
        return {}
    index_path = _resolve_contained_path(
        root,
        index_path,
        label="Release model_index.json",
    )
    index = read_json(index_path)
    value = index.get("_minimax_h3")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise H3ComponentError("model_index.json._minimax_h3 must be an object")
    return dict(value)


def release_metadata(model_root: str | Path) -> dict:
    root = model_root_path(model_root)
    return _release_metadata_at_root(root)


def _enforce_named_partition_signal(root: Path, requested: str) -> None:
    """Reject relabelling a metadata-free root with an explicit partition name."""

    partition_signals: set[str] = set()
    for candidate in (root.name, root.parent.name):
        name = candidate.strip().lower()
        for known_partition in H3_PARTITIONS:
            if name == known_partition or name.endswith(f"-{known_partition}"):
                partition_signals.add(known_partition)
    if len(partition_signals) > 1:
        raise H3ComponentError(
            f"Conflicting partition names in metadata-free root {root}: "
            f"{sorted(partition_signals)!r}"
        )
    if partition_signals and requested not in partition_signals:
        declared = next(iter(partition_signals))
        raise H3ComponentError(
            f"Requested {requested!r} weights, but metadata-free root "
            f"{root} is explicitly named for partition {declared!r}"
        )


def _normalize_partition(partition: str) -> str:
    normalized = str(partition).strip().lower()
    if normalized not in H3_PARTITIONS:
        raise H3ComponentError(
            f"Unknown MiniMax H3 partition {partition!r}; expected one of "
            f"{H3_PARTITIONS!r}"
        )
    return normalized


def _has_direct_component_signal(root: Path) -> bool:
    """Return whether a metadata-free directory resembles an H3 partition."""

    direct_names = {
        "transformer",
        "dit",
        "text_encoder",
        "video_vae",
        "audio_vae",
        VAE_MERGED_DIRNAME,
    }
    try:
        return any(
            child.is_dir()
            and (
                child.name in direct_names
                or child.name.startswith("transformer_")
                or child.name.startswith("text_encoder_")
            )
            for child in root.iterdir()
        )
    except OSError:
        return False


def _admit_partition_root(root: Path, normalized: str) -> Path:
    """Admit a partition below one concrete root without resolving aliases.

    This private path-only layer intentionally never calls ``model_root_path``,
    ``release_metadata`` or the public ``resolve_partition_root``.  The hinted
    same-basename selector can therefore use it without recursive resolution.
    """

    normalized = _normalize_partition(normalized)
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise H3ComponentError(f"MiniMax-H3 model directory not found: {root}")

    if (root / "model_index.json").is_file():
        metadata = _release_metadata_at_root(root)
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
        else:
            _enforce_named_partition_signal(root, normalized)
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

    resolved_root = root.resolve()
    for child in children:
        index_path = child / "model_index.json"
        name = child.name.strip().lower()
        named_match = name == normalized or name.endswith(f"-{normalized}")
        if not index_path.is_file() and not named_match:
            continue
        resolved_child = _resolve_contained_path(
            resolved_root,
            child,
            label="Partition child",
            allow_root=False,
        )
        if not index_path.is_file():
            if named_match and _has_direct_component_signal(resolved_child):
                matches.append(resolved_child)
            continue

        inspected.append(index_path)
        metadata = _release_metadata_at_root(resolved_child)
        declared = metadata.get("partition")
        if declared is not None:
            if not isinstance(declared, str):
                raise H3ComponentError(
                    f"{index_path}._minimax_h3.partition must be a string"
                )
            declared_match = declared.strip().lower() == normalized
        else:
            declared_match = False
        if declared_match or (declared is None and named_match):
            matches.append(resolved_child)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise H3ComponentError(
            f"Multiple {normalized!r} release roots found below {root}: "
            + ", ".join(str(path) for path in matches)
        )

    # Preserve support for pre-release, metadata-free roots that contain the
    # components directly rather than a model_index.json.  A directory name
    # that explicitly advertises the opposite partition remains authoritative:
    # otherwise ``.../Ref2VA`` could be relabelled as FL2VA simply because its
    # old release predates model_index metadata.
    if _has_direct_component_signal(root):
        _enforce_named_partition_signal(root, normalized)
        return root

    inspected_text = (
        ", ".join(str(path) for path in inspected)
        or "no child model_index.json files"
    )
    raise H3ComponentError(
        f"Could not find MiniMax H3 partition {normalized!r} below {root}; "
        f"inspected {inspected_text}"
    )


def resolve_partition_root(
    model_root: str | Path,
    partition: str,
    *,
    required_component: str | Path | None = None,
    required_files: tuple[str, ...] = (),
) -> Path:
    """Resolve either a release root or its immediate partition child.

    MiniMax H3 is commonly downloaded as a parent directory containing sibling
    ``FL2VA`` and ``Ref2VA`` releases.  Each child owns its own
    ``model_index.json``.  A root that already has a model index is never
    redirected to a sibling: its declared partition remains authoritative.
    """

    normalized = _normalize_partition(partition)
    root = model_root_path(
        model_root,
        partition=normalized,
        required_component=required_component,
        required_files=required_files,
    )
    return _admit_partition_root(root, normalized)


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


def partition_for_task(task: str) -> str:
    """Return the official checkpoint partition serving an H3 task."""

    if not isinstance(task, str) or not task.strip():
        raise H3ComponentError("MiniMax H3 task must be a non-empty string")
    normalized = task.strip().lower()
    try:
        return H3_TASK_PARTITIONS[normalized]
    except KeyError as exc:
        raise H3ComponentError(
            f"Unsupported MiniMax H3 task {task!r}; expected one of "
            f"{tuple(H3_TASK_PARTITIONS)!r}"
        ) from exc


def validate_release_metadata(metadata: dict) -> dict:
    """Validate the official ``model_index.json._minimax_h3`` contract.

    Empty metadata remains accepted for old, metadata-free development roots.
    Once a release declares ``_minimax_h3``, however, it must satisfy the v1
    public contract in full.  This prevents a partially copied or hand-edited
    release from silently selecting the wrong DiT partition or sigma schedule.
    """

    if not isinstance(metadata, dict):
        raise H3ComponentError("model_index.json._minimax_h3 must be an object")
    if not metadata:
        return {}

    schema_version = metadata.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != 1:
        raise H3ComponentError(
            "model_index.json._minimax_h3.schema_version must be 1"
        )

    partition = metadata.get("partition")
    if partition not in H3_PARTITIONS:
        raise H3ComponentError(
            "model_index.json._minimax_h3.partition must be one of "
            f"{H3_PARTITIONS!r}"
        )

    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise H3ComponentError(
            "model_index.json._minimax_h3.tasks must be a non-empty list"
        )
    if any(not isinstance(task, str) or not task for task in tasks):
        raise H3ComponentError(
            "model_index.json._minimax_h3.tasks must contain non-empty strings"
        )
    if len(set(tasks)) != len(tasks):
        raise H3ComponentError(
            "model_index.json._minimax_h3.tasks must not contain duplicates"
        )
    for task in tasks:
        if task != task.strip().lower() or task not in H3_TASK_PARTITIONS:
            raise H3ComponentError(
                f"tasks must contain canonical MiniMax H3 task names, got {task!r}"
            )
        expected_partition = partition_for_task(task)
        if expected_partition != partition:
            raise H3ComponentError(
                f"task {task!r} does not belong to partition {partition!r}"
            )

    aliases = metadata.get("task_aliases", {})
    if not isinstance(aliases, dict) or any(
        not isinstance(alias, str)
        or not alias
        or not isinstance(target, str)
        or not target
        for alias, target in aliases.items()
    ):
        raise H3ComponentError(
            "model_index.json._minimax_h3.task_aliases must map strings to strings"
        )
    for alias, target in aliases.items():
        if target not in tasks:
            raise H3ComponentError(
                f"task alias {alias!r} targets undeclared task {target!r}"
            )
        # H3 v1 publishes no aliases.  The official canonicalizer is identity,
        # so only an identity entry can satisfy its mapping contract.
        if alias.strip().lower() != target:
            raise H3ComponentError(
                f"unsupported task alias mapping {alias!r} -> {target!r}"
            )

    scales = release_sigma_shift_scales(metadata)
    if scales is None:
        raise H3ComponentError(
            "model_index.json._minimax_h3.sigma_shift_scales must be an object"
        )
    return {
        **metadata,
        "partition": partition,
        "tasks": list(tasks),
        "task_aliases": dict(aliases),
        "sigma_shift_scales": scales,
    }


def validate_task_partition(
    metadata: dict,
    task: str,
    partition: str | None = None,
) -> str:
    """Validate task, requested partition, and release metadata together.

    Returns the canonical task name.  This function is deliberately cheap and
    is intended to run before checkpoint indexes or tensors are opened.
    """

    expected_partition = partition_for_task(task)
    canonical_task = task.strip().lower()
    requested_partition = (
        expected_partition if partition is None else str(partition).strip().lower()
    )
    if requested_partition not in H3_PARTITIONS:
        raise H3ComponentError(
            f"Unknown MiniMax H3 partition {partition!r}; expected one of "
            f"{H3_PARTITIONS!r}"
        )
    if requested_partition != expected_partition:
        raise H3ComponentError(
            f"MiniMax H3 task {canonical_task!r} requires partition "
            f"{expected_partition!r}, not {requested_partition!r}"
        )

    validated = validate_release_metadata(metadata)
    if not validated:
        return canonical_task
    declared_partition = validated["partition"]
    if declared_partition != requested_partition:
        raise H3ComponentError(
            f"Requested {requested_partition!r} weights, but model_index declares "
            f"{declared_partition!r}"
        )
    aliases = validated.get("task_aliases", {})
    canonical_task = aliases.get(canonical_task, canonical_task)
    if canonical_task not in validated["tasks"]:
        raise H3ComponentError(
            f"Task {task!r} is not served by MiniMax H3 partition "
            f"{declared_partition!r}; supported tasks: {validated['tasks']!r}"
        )
    if partition_for_task(canonical_task) != declared_partition:
        raise H3ComponentError(
            f"Task {task!r} resolves outside partition {declared_partition!r}"
        )
    return canonical_task


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
    validate_task_partition(metadata, "t2va", "fl2va")


__all__ = [
    "H3ComponentError",
    "H3_PARTITIONS",
    "H3_TASK_PARTITIONS",
    "list_h3_model_root_paths",
    "list_h3_model_roots",
    "model_root_path",
    "read_json",
    "release_metadata",
    "release_sigma_shift_scales",
    "partition_for_task",
    "resolve_component",
    "resolve_partition_root",
    "validate_release_metadata",
    "validate_task_partition",
    "validate_t2va_partition",
]
