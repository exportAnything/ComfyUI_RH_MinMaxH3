"""Data contracts shared by the direct MiniMax-H3 ComfyUI nodes.

This module intentionally has no torch or ComfyUI imports.  Apart from making
the geometry rules easy to test, that keeps an incomplete/broken torch install
from hiding useful model-package validation errors during ComfyUI start-up.

The v1 direct node path implements only T2VA.  Its public constants and
validators are intentionally kept stable below.  The parallel v2 contract is
task-aware and describes the FL2VA/Ref2VA media, target and component
compatibility rules without importing torch or ComfyUI.  This lets later node
ports share one fail-fast protocol while existing T2VA workflows keep their
original wire format.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

H3_TASK_T2VA = "t2va"
H3_TASK_FL2VA = "fl2va"
H3_TASK_REF2VA = "ref2va"
H3_TASKS = (H3_TASK_T2VA, H3_TASK_FL2VA, H3_TASK_REF2VA)

H3_T2VA_PARTITION = "fl2va"
H3_REF2VA_PARTITION = "ref2va"
H3_PARTITIONS = (H3_T2VA_PARTITION, H3_REF2VA_PARTITION)
H3_TASK_PARTITIONS = MappingProxyType(
    {
        H3_TASK_T2VA: H3_T2VA_PARTITION,
        H3_TASK_FL2VA: H3_T2VA_PARTITION,
        H3_TASK_REF2VA: H3_REF2VA_PARTITION,
    }
)

H3_TARGET_SCHEMA = "minimax_h3_target/v1"
H3_CONDITIONING_SCHEMA = "minimax_h3_conditioning/v1"
H3_AV_LATENT_SCHEMA = "minimax_h3_av_latent/v1"
H3_MODEL_SCHEMA = "minimax_h3_model/v1"
H3_TEXT_ENCODER_SCHEMA = "minimax_h3_text_encoder/v1"
H3_VAE_SCHEMA = "minimax_h3_vae_bundle/v1"

# V2 schemas are separate rather than aliases.  A v1 value always means the
# already-shipped T2VA-only contract; accepting it as multimodal conditioning
# would make stale workflows silently run with the wrong presentation/packing.
H3_TARGET_SCHEMA_V2 = "minimax_h3_target/v2"
H3_CONDITIONING_SCHEMA_V2 = "minimax_h3_conditioning/v2"
H3_AV_LATENT_SCHEMA_V2 = "minimax_h3_av_latent/v2"
H3_MODEL_SCHEMA_V2 = "minimax_h3_model/v2"
H3_TEXT_ENCODER_SCHEMA_V2 = "minimax_h3_text_encoder/v2"
H3_VAE_SCHEMA_V2 = "minimax_h3_vae_bundle/v2"
H3_CONDITION_MATERIAL_SCHEMA_V2 = "minimax_h3_condition_material/v2"
H3_REFERENCE_LIST_SCHEMA_V2 = "minimax_h3_references/v2"

H3_CONDITION_ROLE_KEYFRAME = "keyframe"
H3_CONDITION_ROLE_REFERENCE = "reference"
H3_FL2VA_KEYFRAME_SIGNATURES = ((0,), (-1,), (0, -1))
H3_REF2VA_REFERENCE_TYPES = ("image", "audio", "video", "video_audio")
H3_REF2VA_AUDIO_BEARING_TYPES = ("audio", "video", "video_audio")

H3_COMPONENT_SCHEMAS_V2 = MappingProxyType(
    {
        "model": H3_MODEL_SCHEMA_V2,
        "text_encoder": H3_TEXT_ENCODER_SCHEMA_V2,
        "vae": H3_VAE_SCHEMA_V2,
    }
)

H3_FPS = 24
H3_SHORT_EDGE = 768
H3_MAX_PIXELS = H3_SHORT_EDGE * 1344
H3_CANVAS_MULTIPLE = 32
H3_MIN_DURATION_SECONDS = 5.0
H3_MAX_DURATION_SECONDS = 15.0
H3_MIN_ASPECT_RATIO = 1.0 / 4.0
H3_MAX_ASPECT_RATIO = 4.0

H3_VIDEO_CHANNELS = 24
H3_AUDIO_LATENT_CHANNELS = 32
H3_AUDIO_CHANNELS = 2
H3_VIDEO_VAE_SPATIAL_RATIO = 16
H3_VIDEO_PATCH_SIZE = (1, 2, 2)
H3_VIDEO_ROW_WIDTH = 96
H3_AUDIO_ROW_WIDTH = 32
H3_TEXT_WIDTH = 5120

H3_DEFAULT_SIGMA_POINTS = 50
H3_DEFAULT_VIDEO_SHIFT = 12.0
H3_DEFAULT_AUDIO_SHIFT = 3.0
H3_IMGVID_COND_TIMESTEP = 0.999
H3_AUDIO_REF_COND_TIMESTEP = 1.0

H3_FINITE_ASPECT_RATIOS = (
    "21:9",
    "16:9",
    "4:3",
    "1:1",
    "3:4",
    "9:16",
)
H3_ASPECT_RATIOS = ("auto", *H3_FINITE_ASPECT_RATIOS)


class H3ConditionMaterialV2(TypedDict, total=False):
    """In-memory condition material passed between ComfyUI nodes."""

    schema: str
    task: str
    role: str
    type: str
    media: Any
    frame_index: int
    resolved_frame_index: int
    condition_index: int
    display_width: int
    display_height: int
    has_audio: bool
    audio_duration_seconds: float
    material_id: str
    material_fingerprint: str


class H3ReferenceListV2(TypedDict):
    """Ordered Ref2VA material chain; order is prompt/packing semantics."""

    schema: str
    task: str
    partition: str
    materials: list[H3ConditionMaterialV2]


class H3TargetV2(TypedDict, total=False):
    schema: str
    task: str
    partition: str
    requested_aspect_ratio: str
    requested_duration_seconds: float | None
    audio_reference_duration_seconds: float
    resolution_stage: str
    fps: int
    geometry: str
    temporal: str
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    video_latent_t: int
    video_latent_h: int
    video_latent_w: int
    audio_latent_t: int
    reference_order_fingerprint: str


class H3ConditioningV2(TypedDict, total=False):
    schema: str
    task: str
    partition: str
    prompt: str
    prompt_embeds: Any
    text_token_tags: Any
    conditions: list[H3ConditionMaterialV2]
    condition_blocks: list[Any]
    condition_order_fingerprint: str
    target: H3TargetV2
    target_fingerprint: str
    release_fingerprint: str
    text_encoder_fingerprint: str
    vae_fingerprint: str
    cfg_distilled: bool
    guidance_scale: float


class H3ComponentDescriptorV2(TypedDict, total=False):
    schema: str
    task: str
    tasks: list[str] | tuple[str, ...]
    partition: str
    model_root: str
    release_metadata: Mapping[str, Any]


class H3ContractError(ValueError):
    """Raised when values connected in a workflow violate the H3 contract."""


class H3TaskNotImplementedError(NotImplementedError):
    """Raised for a known H3 task that the first direct node path cannot run."""


def normalize_task(task: Any) -> str:
    """Return one canonical public task name."""

    if not isinstance(task, str) or not task.strip():
        raise H3ContractError("MiniMax-H3 task 必须是非空字符串")
    normalized = task.strip().lower()
    if normalized not in H3_TASK_PARTITIONS:
        raise H3ContractError(
            f"未知 MiniMax-H3 task {task!r}；可识别值为 {', '.join(H3_TASKS)}"
        )
    return normalized


def partition_for_task(task: Any) -> str:
    """Return the only checkpoint partition that may serve ``task``."""

    return H3_TASK_PARTITIONS[normalize_task(task)]


def validate_task_partition(task: Any, partition: Any) -> str:
    """Fail before loading weights when task and release partition disagree."""

    normalized_task = normalize_task(task)
    if not isinstance(partition, str) or not partition.strip():
        raise H3ContractError("MiniMax-H3 partition 必须是非空字符串")
    normalized_partition = partition.strip().lower()
    if normalized_partition not in H3_PARTITIONS:
        raise H3ContractError(
            f"未知 MiniMax-H3 partition {partition!r}；可识别值为 "
            f"{', '.join(H3_PARTITIONS)}"
        )
    expected = partition_for_task(normalized_task)
    if normalized_partition != expected:
        raise H3ContractError(
            f"task {normalized_task!r} 必须使用 {expected!r} 权重分区，"
            f"实际为 {normalized_partition!r}"
        )
    return normalized_partition


def require_t2va(task: str) -> str:
    """Validate the legacy v1 wire contract, which is T2VA-only."""

    normalized = str(task or "").strip().lower()
    if normalized == H3_TASK_T2VA:
        return normalized
    if normalized in (H3_TASK_FL2VA, H3_TASK_REF2VA):
        raise H3TaskNotImplementedError(
            "旧版 MiniMax-H3 v1 数据合同只接受 T2VA；"
            f"{normalized.upper()} 请使用对应的 v2 Target、Encode、"
            "Empty AV Latent 和 Dual Sigma Sampler 节点。"
        )
    raise H3ContractError(
        f"未知 MiniMax-H3 task {task!r}；可识别值为 {', '.join(H3_TASKS)}"
    )


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise H3ContractError(f"{name} 必须是数字")
    out = float(value)
    if not math.isfinite(out):
        raise H3ContractError(f"{name} 必须是有限数字")
    if positive and out <= 0.0:
        raise H3ContractError(f"{name} 必须大于 0")
    return out


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise H3ContractError(f"{name} 必须是正整数")
    return int(value)


def align_frame_count(frame_count: int) -> int:
    """Snap a frame count upward to the model's ``17n+5`` boundary."""

    frame_count = _positive_int(frame_count, "frame_count")
    out = frame_count
    while out % 17 != 5:
        out += 1
    return out


def video_latent_t(frame_count: int) -> int:
    frame_count = _positive_int(frame_count, "frame_count")
    if frame_count <= 5:
        return 2
    return ((frame_count - 5) // 17) * 5 + 2


def frame_count_from_video_latent_t(latent_t: int) -> int:
    latent_t = _positive_int(latent_t, "video latent T")
    if latent_t == 1:
        return 1
    if latent_t < 2 or (latent_t - 2) % 5:
        raise H3ContractError("MiniMax-H3 video latent T 必须为 1 或满足 5n+2")
    return 17 * ((latent_t - 2) // 5) + 5


def audio_latent_t(duration_seconds: float) -> int:
    duration = _finite_float(
        duration_seconds, "duration_seconds", positive=True
    )
    return int(round(duration * 40.0))


def parse_aspect_ratio(value: str) -> tuple[int, int]:
    if value == "auto":
        # The original T2VA task profile resolves auto to the 16:9 policy
        # default.  FL2VA auto is material-derived and is intentionally not
        # handled by this T2VA-only path.
        return (16, 9)
    if value not in H3_FINITE_ASPECT_RATIOS:
        raise H3ContractError(
            f"不支持的 aspect_ratio {value!r}；可选值：{', '.join(H3_ASPECT_RATIOS)}"
        )
    width, height = value.split(":", 1)
    return int(width), int(height)


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def resolve_spatial_shape(
    width: int | float,
    height: int | float,
) -> dict[str, Any]:
    """Port of H3 ``adapt_shape_v1`` target geometry resolution."""

    source_width = _finite_float(width, "aspect width", positive=True)
    source_height = _finite_float(height, "aspect height", positive=True)
    ratio = source_width / source_height
    if not H3_MIN_ASPECT_RATIO <= ratio <= H3_MAX_ASPECT_RATIO:
        raise H3ContractError(
            "MiniMax-H3 宽高比必须在 1:4 到 4:1 之间，"
            f"实际为 {source_width:g}:{source_height:g}"
        )

    if ratio >= 1.0:
        nominal_width = H3_SHORT_EDGE * ratio
        nominal_height = float(H3_SHORT_EDGE)
    else:
        nominal_width = float(H3_SHORT_EDGE)
        nominal_height = H3_SHORT_EDGE / ratio

    nominal_area = nominal_width * nominal_height
    if nominal_area > H3_MAX_PIXELS:
        size_mode = "area"
        scale = math.sqrt(H3_MAX_PIXELS / nominal_area)
        nominal_width *= scale
        nominal_height *= scale
    else:
        size_mode = "short_edge"

    resolved_width = _nearest_multiple(nominal_width, H3_CANVAS_MULTIPLE)
    resolved_height = _nearest_multiple(nominal_height, H3_CANVAS_MULTIPLE)
    return {
        "geometry": "resolved_v2",
        "shape_policy_version": "adapt_shape_v1",
        "base_short_edge": H3_SHORT_EDGE,
        "effective_short_edge": min(resolved_width, resolved_height),
        "size_mode": size_mode,
        "max_pixels": H3_MAX_PIXELS,
        "multiple": H3_CANVAS_MULTIPLE,
        "rounding": "nearest",
        "width": resolved_width,
        "height": resolved_height,
    }


def resolve_explicit_spatial_shape(width: int, height: int) -> dict[str, Any]:
    """Validate and preserve an explicitly requested H3 canvas."""

    resolved_width = _positive_int(width, "width")
    resolved_height = _positive_int(height, "height")
    if (
        resolved_width % H3_CANVAS_MULTIPLE
        or resolved_height % H3_CANVAS_MULTIPLE
    ):
        raise H3ContractError(
            f"width/height 必须按 {H3_CANVAS_MULTIPLE} 对齐，"
            f"实际为 {resolved_width}x{resolved_height}"
        )
    ratio = resolved_width / resolved_height
    if not H3_MIN_ASPECT_RATIO <= ratio <= H3_MAX_ASPECT_RATIO:
        raise H3ContractError(
            "MiniMax-H3 宽高比必须在 1:4 到 4:1 之间，"
            f"实际为 {resolved_width}:{resolved_height}"
        )
    pixels = resolved_width * resolved_height
    if pixels > H3_MAX_PIXELS:
        raise H3ContractError(
            f"显式分辨率最多允许 {H3_MAX_PIXELS} 像素，"
            f"实际为 {resolved_width}x{resolved_height}={pixels}"
        )
    return {
        "geometry": "explicit_v1",
        "shape_policy_version": "explicit_v1",
        "base_short_edge": H3_SHORT_EDGE,
        "effective_short_edge": min(resolved_width, resolved_height),
        "size_mode": "explicit",
        "max_pixels": H3_MAX_PIXELS,
        "multiple": H3_CANVAS_MULTIPLE,
        "rounding": "none",
        "width": resolved_width,
        "height": resolved_height,
    }


def resolve_t2va_target(
    *,
    aspect_ratio: str,
    duration_seconds: float,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Resolve public T2VA target fields into the exact latent geometry."""

    duration = _finite_float(
        duration_seconds, "duration_seconds", positive=True
    )
    if not H3_MIN_DURATION_SECONDS <= duration <= H3_MAX_DURATION_SECONDS:
        raise H3ContractError(
            "duration_seconds 必须在 "
            f"{H3_MIN_DURATION_SECONDS:g} 到 {H3_MAX_DURATION_SECONDS:g} 秒之间"
        )
    frame_count = align_frame_count(int(round(duration * H3_FPS)))
    aligned_duration = frame_count / H3_FPS
    explicit_width = None if width in (None, 0) else width
    explicit_height = None if height in (None, 0) else height
    if (explicit_width is None) != (explicit_height is None):
        raise H3ContractError("显式 width 和 height 必须同时填写，或同时设为 0 使用宽高比")
    if explicit_width is None:
        ratio_width, ratio_height = parse_aspect_ratio(str(aspect_ratio))
        shape = resolve_spatial_shape(ratio_width, ratio_height)
    else:
        shape = resolve_explicit_spatial_shape(explicit_width, explicit_height)
    shape.update(
        {
            "schema": H3_TARGET_SCHEMA,
            "task": H3_TASK_T2VA,
            "requested_aspect_ratio": str(aspect_ratio),
            "requested_width": int(width or 0),
            "requested_height": int(height or 0),
            "fps": H3_FPS,
            "requested_duration_seconds": duration,
            "duration_seconds": aligned_duration,
            "frame_count": frame_count,
            "video_latent_t": video_latent_t(frame_count),
            "audio_latent_t": audio_latent_t(aligned_duration),
        }
    )
    shape["video_latent_h"] = int(shape["height"]) // H3_VIDEO_VAE_SPATIAL_RATIO
    shape["video_latent_w"] = int(shape["width"]) // H3_VIDEO_VAE_SPATIAL_RATIO
    return shape


def _validate_duration_seconds(value: Any, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    duration = _finite_float(value, "duration_seconds", positive=True)
    if not H3_MIN_DURATION_SECONDS <= duration <= H3_MAX_DURATION_SECONDS:
        raise H3ContractError(
            "duration_seconds 必须在 "
            f"{H3_MIN_DURATION_SECONDS:g} 到 {H3_MAX_DURATION_SECONDS:g} 秒之间"
        )
    return duration


def _temporal_shape_v2(duration_seconds: Any) -> dict[str, Any]:
    requested = _validate_duration_seconds(duration_seconds)
    assert requested is not None
    frame_count = align_frame_count(int(round(requested * H3_FPS)))
    aligned_duration = frame_count / H3_FPS
    return {
        "temporal": "resolved",
        "resolution_stage": "resolved",
        "requested_duration_seconds": requested,
        "duration_seconds": aligned_duration,
        "fps": H3_FPS,
        "frame_count": frame_count,
        "video_latent_t": video_latent_t(frame_count),
        "audio_latent_t": audio_latent_t(aligned_duration),
    }


def _apply_video_spatial_latents(shape: dict[str, Any]) -> dict[str, Any]:
    if shape.get("width") is not None and shape.get("height") is not None:
        shape["video_latent_h"] = (
            int(shape["height"]) // H3_VIDEO_VAE_SPATIAL_RATIO
        )
        shape["video_latent_w"] = (
            int(shape["width"]) // H3_VIDEO_VAE_SPATIAL_RATIO
        )
    return shape


def _parse_flexible_aspect_ratio(value: Any) -> tuple[int, int]:
    if not isinstance(value, str) or not value:
        raise H3ContractError("aspect_ratio 必须是非空字符串")
    parts = value.split(":")
    if len(parts) != 2:
        raise H3ContractError(
            f"FL2VA aspect_ratio 必须是 'W:H' 或 'auto'，实际为 {value!r}"
        )
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise H3ContractError(
            f"FL2VA aspect_ratio 必须使用整数 W:H，实际为 {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise H3ContractError(
            f"FL2VA aspect_ratio 两项都必须大于 0，实际为 {value!r}"
        )
    return width, height


def _display_metadata(
    *,
    display_width: Any = None,
    display_height: Any = None,
) -> dict[str, int]:
    if display_width is None and display_height is None:
        return {}
    if display_width is None or display_height is None:
        raise H3ContractError("display_width/display_height 必须同时提供")
    return {
        "display_width": _positive_int(display_width, "display_width"),
        "display_height": _positive_int(display_height, "display_height"),
    }


def _material_display_shape(material: Mapping[str, Any]) -> tuple[int, int] | None:
    width = material.get("display_width")
    height = material.get("display_height")
    if width is not None or height is not None:
        metadata = _display_metadata(
            display_width=width,
            display_height=height,
        )
        return metadata["display_width"], metadata["display_height"]

    media = material.get("media")
    size = getattr(media, "size", None)
    if (
        isinstance(size, Sequence)
        and not isinstance(size, (str, bytes))
        and len(size) == 2
    ):
        try:
            return (
                _positive_int(int(size[0]), "media width"),
                _positive_int(int(size[1]), "media height"),
            )
        except (TypeError, ValueError, H3ContractError):
            pass

    shape = getattr(media, "shape", None)
    if shape is not None and len(shape) >= 3:
        # Standard ComfyUI IMAGE is [B,H,W,C].  This intentionally only
        # inspects shape metadata and never imports torch.
        try:
            return (
                _positive_int(int(shape[-2]), "media width"),
                _positive_int(int(shape[-3]), "media height"),
            )
        except (TypeError, ValueError, H3ContractError):
            pass
    return None


def make_fl2va_keyframe(
    image: Any,
    frame_index: int,
    *,
    display_width: int | None = None,
    display_height: int | None = None,
) -> H3ConditionMaterialV2:
    """Create one FL2VA endpoint condition without resolving ``-1`` early."""

    if image is None:
        raise H3ContractError("FL2VA keyframe image 不能为空")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        raise H3ContractError("FL2VA frame_index 必须是整数")
    if frame_index not in (0, -1):
        raise H3ContractError("FL2VA frame_index 只允许 0（首帧）或 -1（尾帧）")
    material: H3ConditionMaterialV2 = {
        "schema": H3_CONDITION_MATERIAL_SCHEMA_V2,
        "task": H3_TASK_FL2VA,
        "role": H3_CONDITION_ROLE_KEYFRAME,
        "type": "image",
        "media": image,
        "frame_index": frame_index,
    }
    material.update(
        _display_metadata(
            display_width=display_width,
            display_height=display_height,
        )
    )
    return material


def validate_fl2va_keyframes(
    keyframes: Any,
    *,
    frame_count: int | None = None,
) -> list[H3ConditionMaterialV2]:
    """Validate the three official endpoint signatures, preserving order."""

    if not isinstance(keyframes, Sequence) or isinstance(keyframes, (str, bytes)):
        raise H3ContractError("FL2VA keyframes 必须是列表")
    if not 1 <= len(keyframes) <= 2:
        raise H3ContractError("FL2VA 必须提供 1 或 2 张 keyframe")
    clean: list[H3ConditionMaterialV2] = []
    for index, raw in enumerate(keyframes):
        path = f"keyframes[{index}]"
        if not isinstance(raw, Mapping):
            raise H3ContractError(f"{path} 必须是对象")
        if raw.get("schema") != H3_CONDITION_MATERIAL_SCHEMA_V2:
            raise H3ContractError(
                f"{path}.schema 必须为 {H3_CONDITION_MATERIAL_SCHEMA_V2!r}"
            )
        if normalize_task(raw.get("task")) != H3_TASK_FL2VA:
            raise H3ContractError(f"{path}.task 必须为 'fl2va'")
        if raw.get("role") != H3_CONDITION_ROLE_KEYFRAME or raw.get("type") != "image":
            raise H3ContractError(f"{path} 必须是 image/keyframe")
        if raw.get("media") is None:
            raise H3ContractError(f"{path}.media 不能为空")
        frame_index = raw.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise H3ContractError(f"{path}.frame_index 必须是整数")
        entry = dict(raw)
        entry["condition_index"] = index
        clean.append(entry)

    signature = tuple(int(item["frame_index"]) for item in clean)
    if signature not in H3_FL2VA_KEYFRAME_SIGNATURES:
        raise H3ContractError(
            "FL2VA keyframe 顺序只允许 [0]、[-1] 或 [0,-1]，"
            f"实际为 {list(signature)!r}"
        )
    if frame_count is not None:
        aligned_frames = _positive_int(frame_count, "frame_count")
        if aligned_frames % 17 != 5:
            raise H3ContractError("FL2VA frame_count 必须已经对齐到 17n+5")
        for entry in clean:
            semantic = int(entry["frame_index"])
            entry["resolved_frame_index"] = (
                aligned_frames - 1 if semantic == -1 else semantic
            )
    return clean


def _validate_material_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise H3ContractError("material_id 必须是 32 位小写十六进制字符串")
    return value


def material_compatibility_fingerprint(material: Mapping[str, Any]) -> str:
    """Return the stable identity of one Ref2VA descriptor instance."""

    if not isinstance(material, Mapping):
        raise H3ContractError("Ref2VA material 必须是对象")
    if material.get("schema") != H3_CONDITION_MATERIAL_SCHEMA_V2:
        raise H3ContractError("Ref2VA material schema 无效")
    task = normalize_task(material.get("task"))
    if task != H3_TASK_REF2VA:
        raise H3ContractError("material fingerprint 仅用于 Ref2VA reference")
    role = material.get("role")
    if role != H3_CONDITION_ROLE_REFERENCE:
        raise H3ContractError("Ref2VA material role 必须为 reference")
    material_type = material.get("type")
    if material_type not in H3_REF2VA_REFERENCE_TYPES:
        raise H3ContractError("Ref2VA material type 无效")
    material_id = _validate_material_id(material.get("material_id"))
    payload = json.dumps(
        {
            "schema": H3_CONDITION_MATERIAL_SCHEMA_V2,
            "task": task,
            "role": role,
            "type": material_type,
            "material_id": material_id,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_ref2va_reference(
    reference_type: str,
    media: Any,
    *,
    display_width: int | None = None,
    display_height: int | None = None,
    has_audio: bool | None = None,
    audio_duration_seconds: float | None = None,
) -> H3ConditionMaterialV2:
    """Create one ordered Ref2VA material descriptor."""

    if reference_type not in H3_REF2VA_REFERENCE_TYPES:
        raise H3ContractError(
            f"Ref2VA reference type 必须是 {H3_REF2VA_REFERENCE_TYPES!r}，"
            f"实际为 {reference_type!r}"
        )
    if media is None:
        raise H3ContractError("Ref2VA reference media 不能为空")
    if has_audio is not None and not isinstance(has_audio, bool):
        raise H3ContractError("has_audio 必须是 bool 或 None")
    if reference_type == "image" and (
        has_audio is not None or audio_duration_seconds is not None
    ):
        raise H3ContractError("Ref2VA image reference 不能携带音频元数据")
    if reference_type == "audio" and has_audio is False:
        raise H3ContractError("Ref2VA audio reference 必须包含音频")
    if reference_type == "video_audio" and has_audio is False:
        raise H3ContractError("Ref2VA video_audio reference 必须包含音轨")

    material: H3ConditionMaterialV2 = {
        "schema": H3_CONDITION_MATERIAL_SCHEMA_V2,
        "task": H3_TASK_REF2VA,
        "role": H3_CONDITION_ROLE_REFERENCE,
        "type": reference_type,
        "media": media,
        "material_id": uuid.uuid4().hex,
    }
    material["material_fingerprint"] = material_compatibility_fingerprint(
        material
    )
    if reference_type in ("image", "video", "video_audio"):
        material.update(
            _display_metadata(
                display_width=display_width,
                display_height=display_height,
            )
        )
    elif display_width is not None or display_height is not None:
        raise H3ContractError("Ref2VA audio reference 不能携带画面尺寸")
    if reference_type == "audio":
        material["has_audio"] = True
    elif has_audio is not None:
        material["has_audio"] = has_audio
    if audio_duration_seconds is not None:
        duration = _finite_float(
            audio_duration_seconds,
            "audio_duration_seconds",
            positive=True,
        )
        if material.get("has_audio") is False:
            raise H3ContractError("静音 reference 不能携带 audio_duration_seconds")
        material["audio_duration_seconds"] = duration
    return material


def validate_ref2va_references(references: Any) -> list[H3ConditionMaterialV2]:
    """Validate and copy an ordered Ref2VA material stream (no artificial max)."""

    if isinstance(references, Mapping):
        if references.get("schema") != H3_REFERENCE_LIST_SCHEMA_V2:
            raise H3ContractError(
                f"references.schema 必须为 {H3_REFERENCE_LIST_SCHEMA_V2!r}"
            )
        if normalize_task(references.get("task")) != H3_TASK_REF2VA:
            raise H3ContractError("references.task 必须为 'ref2va'")
        validate_task_partition(H3_TASK_REF2VA, references.get("partition"))
        materials = references.get("materials")
    else:
        materials = references
    if not isinstance(materials, Sequence) or isinstance(materials, (str, bytes)):
        raise H3ContractError("Ref2VA references 必须是有序列表")
    if len(materials) == 0:
        raise H3ContractError("Ref2VA 至少需要一个 reference")

    clean: list[H3ConditionMaterialV2] = []
    seen_material_fingerprints: set[str] = set()
    for index, raw in enumerate(materials):
        path = f"references[{index}]"
        if not isinstance(raw, Mapping):
            raise H3ContractError(f"{path} 必须是对象")
        if raw.get("schema") != H3_CONDITION_MATERIAL_SCHEMA_V2:
            raise H3ContractError(
                f"{path}.schema 必须为 {H3_CONDITION_MATERIAL_SCHEMA_V2!r}"
            )
        if normalize_task(raw.get("task")) != H3_TASK_REF2VA:
            raise H3ContractError(f"{path}.task 必须为 'ref2va'")
        if raw.get("role") != H3_CONDITION_ROLE_REFERENCE:
            raise H3ContractError(f"{path}.role 必须为 'reference'")
        reference_type = raw.get("type")
        if reference_type not in H3_REF2VA_REFERENCE_TYPES:
            raise H3ContractError(
                f"{path}.type 必须是 {H3_REF2VA_REFERENCE_TYPES!r}"
            )
        if raw.get("media") is None:
            raise H3ContractError(f"{path}.media 不能为空")
        expected_material_fingerprint = material_compatibility_fingerprint(raw)
        if raw.get("material_fingerprint") != expected_material_fingerprint:
            raise H3ContractError(
                f"{path}.material_fingerprint 与 descriptor identity 不一致"
            )
        if expected_material_fingerprint in seen_material_fingerprints:
            raise H3ContractError(
                "Ref2VA references 不能重复使用同一 material identity"
            )
        seen_material_fingerprints.add(expected_material_fingerprint)
        if raw.get("frame_index") is not None:
            raise H3ContractError(f"{path}.frame_index 不允许用于 Ref2VA")
        has_audio = raw.get("has_audio")
        if has_audio is not None and not isinstance(has_audio, bool):
            raise H3ContractError(f"{path}.has_audio 必须是 bool")
        if reference_type == "image" and (
            has_audio is not None or raw.get("audio_duration_seconds") is not None
        ):
            raise H3ContractError(f"{path} image reference 不能携带音频元数据")
        if reference_type == "audio" and has_audio is False:
            raise H3ContractError(f"{path} audio reference 必须包含音频")
        if reference_type == "video_audio" and has_audio is False:
            raise H3ContractError(f"{path} video_audio reference 必须包含音轨")
        entry = dict(raw)
        entry["condition_index"] = index
        if reference_type == "audio":
            entry["has_audio"] = True
        if entry.get("audio_duration_seconds") is not None:
            entry["audio_duration_seconds"] = _finite_float(
                entry["audio_duration_seconds"],
                f"{path}.audio_duration_seconds",
                positive=True,
            )
            if entry.get("has_audio") is False:
                raise H3ContractError(f"{path} 静音 reference 不能携带音频时长")
        clean.append(entry)
    return clean


def make_ref2va_references(references: Any) -> H3ReferenceListV2:
    return {
        "schema": H3_REFERENCE_LIST_SCHEMA_V2,
        "task": H3_TASK_REF2VA,
        "partition": H3_REF2VA_PARTITION,
        "materials": validate_ref2va_references(references),
    }


def append_ref2va_reference(
    references: Mapping[str, Any] | None,
    reference: Mapping[str, Any],
) -> H3ReferenceListV2:
    """Return a new list so branch reuse cannot mutate an upstream node value."""

    existing: list[H3ConditionMaterialV2] = []
    if references is not None:
        existing = validate_ref2va_references(references)
    combined = [*existing, dict(reference)]
    return make_ref2va_references(combined)


def resolve_t2va_target_v2(
    *,
    aspect_ratio: str,
    duration_seconds: float,
    width: int | None = None,
    height: int | None = None,
) -> H3TargetV2:
    """V2 projection of the shipped T2VA target, with identical geometry."""

    target = resolve_t2va_target(
        aspect_ratio=aspect_ratio,
        duration_seconds=duration_seconds,
        width=width,
        height=height,
    )
    target.update(
        {
            "schema": H3_TARGET_SCHEMA_V2,
            "partition": H3_T2VA_PARTITION,
            "temporal": "resolved",
            "resolution_stage": "resolved",
        }
    )
    return target


def resolve_fl2va_target_v2(
    *,
    aspect_ratio: str,
    duration_seconds: float,
    keyframes: Any = None,
    width: int | None = None,
    height: int | None = None,
) -> H3TargetV2:
    """Resolve FL2VA target geometry or mark ``auto`` geometry deferred.

    ``auto`` selects the semantic first-frame anchor.  With a last-only
    signature, that one image is the geometry source.  When image dimensions
    are not known yet the returned target is deliberately deferred rather
    than falling back to 16:9.
    """

    temporal = _temporal_shape_v2(duration_seconds)
    frame_count = int(temporal["frame_count"])
    clean_keyframes = (
        validate_fl2va_keyframes(keyframes, frame_count=frame_count)
        if keyframes is not None
        else []
    )
    target: H3TargetV2 = {
        "schema": H3_TARGET_SCHEMA_V2,
        "task": H3_TASK_FL2VA,
        "partition": H3_T2VA_PARTITION,
        "requested_aspect_ratio": str(aspect_ratio),
        **temporal,
    }
    if clean_keyframes:
        target["keyframe_signature"] = [
            int(item["frame_index"]) for item in clean_keyframes
        ]
        target["semantic_frame_indices"] = list(target["keyframe_signature"])
        target["pixel_frame_indices"] = [
            int(item["resolved_frame_index"]) for item in clean_keyframes
        ]

    explicit_width = None if width in (None, 0) else width
    explicit_height = None if height in (None, 0) else height
    if (explicit_width is None) != (explicit_height is None):
        raise H3ContractError("显式 width 和 height 必须同时填写，或同时设为 0")
    if explicit_width is not None:
        shape = resolve_explicit_spatial_shape(explicit_width, explicit_height)
        shape["geometry_source"] = "explicit_target"
        shape["requested_width"] = int(explicit_width)
        shape["requested_height"] = int(explicit_height)
    elif aspect_ratio == "auto":
        source: Mapping[str, Any] | None = None
        for item in clean_keyframes:
            if int(item["frame_index"]) == 0:
                source = item
                break
        if source is None and clean_keyframes:
            source = clean_keyframes[0]
        display_shape = _material_display_shape(source) if source is not None else None
        if display_shape is None:
            shape = {
                "geometry": "deferred",
                "geometry_source": "first_keyframe",
                "shape_policy_version": "adapt_shape_v1",
                "base_short_edge": H3_SHORT_EDGE,
                "size_mode": "deferred",
                "resolution_stage": "pre_media",
            }
        else:
            shape = resolve_spatial_shape(*display_shape)
            semantic = int(source["frame_index"])
            shape["geometry_source"] = (
                "first_keyframe" if semantic == 0 else "last_keyframe"
            )
            shape["geometry_source_condition_index"] = int(
                source["condition_index"]
            )
            shape["geometry_source_frame_index"] = semantic
            shape["geometry_source_width"] = int(display_shape[0])
            shape["geometry_source_height"] = int(display_shape[1])
    else:
        ratio_width, ratio_height = _parse_flexible_aspect_ratio(aspect_ratio)
        shape = resolve_spatial_shape(ratio_width, ratio_height)
        shape["geometry_source"] = "explicit_target"
    target.update(shape)
    return _apply_video_spatial_latents(target)


def _ref2va_duration_source(
    references: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    sources = [
        item
        for item in references
        if item.get("type") == "audio"
        or (
            item.get("type") in {"video", "video_audio"}
            and item.get("has_audio") is True
        )
    ]
    if not sources:
        raise H3ContractError(
            "Ref2VA 自动时长需要且只允许一个 audio-bearing reference"
        )
    if len(sources) > 1:
        raise H3ContractError(
            "Ref2VA 存在多个 audio-bearing references 时必须显式填写时长"
        )
    source = sources[0]
    if source.get("has_audio") is False:
        raise H3ContractError("静音 video 不能提供 Ref2VA 自动目标时长")
    return source


def resolve_ref2va_target_v2(
    *,
    aspect_ratio: str,
    duration_seconds: float | None,
    references: Any = None,
    width: int | None = None,
    height: int | None = None,
) -> H3TargetV2:
    """Resolve Ref2VA target geometry and optional audio-derived duration.

    ``width`` and ``height`` form an all-or-nothing explicit canvas override.
    Leaving both unset (or zero through the node UI) preserves the official
    aspect-ratio bucket policy used by existing workflows.
    """

    if aspect_ratio not in H3_ASPECT_RATIOS:
        raise H3ContractError(
            "Ref2VA aspect_ratio 只允许 'auto' 或官方六个比例桶："
            f"{', '.join(H3_FINITE_ASPECT_RATIOS)}"
        )
    explicit_width = None if width in (None, 0) else width
    explicit_height = None if height in (None, 0) else height
    if (explicit_width is None) != (explicit_height is None):
        raise H3ContractError(
            "width 和 height 必须同时填写，或同时设为 0 使用 aspect_ratio"
        )
    if explicit_width is None:
        ratio = "16:9" if aspect_ratio == "auto" else aspect_ratio
        ratio_width, ratio_height = parse_aspect_ratio(ratio)
        shape = resolve_spatial_shape(ratio_width, ratio_height)
        shape["geometry_source"] = (
            "policy_default" if aspect_ratio == "auto" else "explicit_target"
        )
    else:
        shape = resolve_explicit_spatial_shape(
            explicit_width, explicit_height
        )
        shape["geometry_source"] = "explicit_target"
        shape["requested_width"] = int(explicit_width)
        shape["requested_height"] = int(explicit_height)
    clean_references = (
        validate_ref2va_references(references) if references is not None else []
    )
    target: H3TargetV2 = {
        "schema": H3_TARGET_SCHEMA_V2,
        "task": H3_TASK_REF2VA,
        "partition": H3_REF2VA_PARTITION,
        "requested_aspect_ratio": aspect_ratio,
        **shape,
    }
    if clean_references:
        target["reference_order_fingerprint"] = condition_order_fingerprint(
            H3_TASK_REF2VA,
            clean_references,
        )
    if duration_seconds is None:
        if not clean_references:
            raise H3ContractError(
                "Ref2VA 省略 duration_seconds 时必须提供 references"
            )
        source = _ref2va_duration_source(clean_references)
        target["requested_duration_seconds"] = None
        target["duration_source_condition_index"] = int(source["condition_index"])
        source_duration = source.get("audio_duration_seconds")
        if source_duration is None:
            target.update(
                {
                    "temporal": "deferred_from_audio_reference",
                    "resolution_stage": "pre_media",
                    "fps": H3_FPS,
                }
            )
        else:
            target.update(_temporal_shape_v2(source_duration))
            target["requested_duration_seconds"] = None
            target["audio_reference_duration_seconds"] = float(source_duration)
            target["temporal"] = "resolved_from_audio_reference"
    else:
        target.update(_temporal_shape_v2(duration_seconds))
    return _apply_video_spatial_latents(target)


def resolve_deferred_target_v2(
    target: Mapping[str, Any],
    conditions: Any,
) -> H3TargetV2:
    """Resolve media-derived FL geometry or Ref audio duration exactly once."""

    clean = validate_target_v2(target, require_resolved=False)
    task = clean["task"]
    if task == H3_TASK_FL2VA and clean.get("geometry") == "deferred":
        return resolve_fl2va_target_v2(
            aspect_ratio=str(clean["requested_aspect_ratio"]),
            duration_seconds=float(clean["requested_duration_seconds"]),
            keyframes=conditions,
        )
    if (
        task == H3_TASK_REF2VA
        and clean.get("temporal") == "deferred_from_audio_reference"
    ):
        explicit_width = (
            int(clean["requested_width"])
            if clean.get("geometry") == "explicit_v1"
            else None
        )
        explicit_height = (
            int(clean["requested_height"])
            if clean.get("geometry") == "explicit_v1"
            else None
        )
        return resolve_ref2va_target_v2(
            aspect_ratio=str(clean["requested_aspect_ratio"]),
            duration_seconds=None,
            references=conditions,
            width=explicit_width,
            height=explicit_height,
        )
    return clean


def validate_target_v2(
    target: Any,
    *,
    expected_task: str | None = None,
    require_resolved: bool = True,
) -> H3TargetV2:
    """Validate target provenance, exact 17n+5 timing, and bounded geometry."""

    if not isinstance(target, Mapping):
        raise H3ContractError("target v2 必须是对象")
    if target.get("schema") != H3_TARGET_SCHEMA_V2:
        raise H3ContractError(
            f"target schema 不匹配：期望 {H3_TARGET_SCHEMA_V2!r}"
        )
    clean: H3TargetV2 = dict(target)
    task = normalize_task(clean.get("task"))
    if expected_task is not None and task != normalize_task(expected_task):
        raise H3ContractError(
            f"target.task={task!r} 与工作流 task={normalize_task(expected_task)!r} 不一致"
        )
    clean["task"] = task
    clean["partition"] = validate_task_partition(task, clean.get("partition"))
    fps = _positive_int(clean.get("fps"), "target.fps")
    if fps != H3_FPS:
        raise H3ContractError(f"target.fps 必须为 {H3_FPS}")
    clean["fps"] = fps

    aspect_ratio = clean.get("requested_aspect_ratio")
    if not isinstance(aspect_ratio, str) or not aspect_ratio:
        raise H3ContractError("target.requested_aspect_ratio 必须是非空字符串")
    if task in (H3_TASK_T2VA, H3_TASK_REF2VA):
        if aspect_ratio not in H3_ASPECT_RATIOS:
            raise H3ContractError(
                f"{task} aspect_ratio 只允许 {H3_ASPECT_RATIOS!r}"
            )
    elif aspect_ratio != "auto":
        ratio_width, ratio_height = _parse_flexible_aspect_ratio(aspect_ratio)
        resolve_spatial_shape(ratio_width, ratio_height)

    geometry = clean.get("geometry")
    spatial_deferred = geometry == "deferred"
    if spatial_deferred:
        if task != H3_TASK_FL2VA or aspect_ratio != "auto":
            raise H3ContractError("只有 FL2VA auto target 可以延迟空间解析")
        forbidden_spatial = (
            "width",
            "height",
            "video_latent_h",
            "video_latent_w",
        )
        if any(clean.get(field) is not None for field in forbidden_spatial):
            raise H3ContractError("deferred FL2VA target 不能伪装已解析空间字段")
        if require_resolved:
            raise H3ContractError("FL2VA target 空间尺寸仍未从 keyframe 解析")
    elif geometry in ("resolved_v2", "explicit_v1"):
        width = _positive_int(clean.get("width"), "target.width")
        height = _positive_int(clean.get("height"), "target.height")
        if width % H3_CANVAS_MULTIPLE or height % H3_CANVAS_MULTIPLE:
            raise H3ContractError("target width/height 必须按 32 对齐")
        ratio = width / height
        if not H3_MIN_ASPECT_RATIO <= ratio <= H3_MAX_ASPECT_RATIO:
            raise H3ContractError("target 宽高比必须在 1:4 到 4:1 之间")
        if geometry == "explicit_v1":
            requested_width = _positive_int(
                clean.get("requested_width"), "target.requested_width"
            )
            requested_height = _positive_int(
                clean.get("requested_height"), "target.requested_height"
            )
            if (width, height) != (requested_width, requested_height):
                raise H3ContractError("显式 target 尺寸与 requested_width/height 不一致")
            if width * height > H3_MAX_PIXELS:
                raise H3ContractError(
                    f"显式 target 最多允许 {H3_MAX_PIXELS} 像素"
                )
        else:
            if task == H3_TASK_FL2VA and aspect_ratio == "auto":
                source_width = _positive_int(
                    clean.get("geometry_source_width"),
                    "target.geometry_source_width",
                )
                source_height = _positive_int(
                    clean.get("geometry_source_height"),
                    "target.geometry_source_height",
                )
                expected_spatial = resolve_spatial_shape(
                    source_width, source_height
                )
            else:
                declared_ratio = "16:9" if aspect_ratio == "auto" else aspect_ratio
                if task == H3_TASK_FL2VA:
                    ratio_width, ratio_height = _parse_flexible_aspect_ratio(
                        declared_ratio
                    )
                else:
                    ratio_width, ratio_height = parse_aspect_ratio(declared_ratio)
                expected_spatial = resolve_spatial_shape(
                    ratio_width, ratio_height
                )
            if (width, height) != (
                int(expected_spatial["width"]),
                int(expected_spatial["height"]),
            ):
                raise H3ContractError(
                    "target resolved geometry 与声明比例/素材比例不一致"
                )
        clean["width"] = width
        clean["height"] = height
        latent_h = _positive_int(clean.get("video_latent_h"), "target.video_latent_h")
        latent_w = _positive_int(clean.get("video_latent_w"), "target.video_latent_w")
        if (
            latent_h != height // H3_VIDEO_VAE_SPATIAL_RATIO
            or latent_w != width // H3_VIDEO_VAE_SPATIAL_RATIO
        ):
            raise H3ContractError("target 的像素尺寸与 video latent 尺寸不一致")
        clean["video_latent_h"] = latent_h
        clean["video_latent_w"] = latent_w
    else:
        raise H3ContractError(f"未知 target.geometry {geometry!r}")

    temporal = clean.get("temporal")
    temporal_deferred = temporal == "deferred_from_audio_reference"
    if temporal_deferred:
        if task != H3_TASK_REF2VA:
            raise H3ContractError("只有 Ref2VA 可以从 reference audio 延迟解析时长")
        if clean.get("requested_duration_seconds") is not None:
            raise H3ContractError("延迟 Ref2VA target 不应包含显式目标时长")
        forbidden_temporal = (
            "duration_seconds",
            "frame_count",
            "video_latent_t",
            "audio_latent_t",
        )
        if any(clean.get(field) is not None for field in forbidden_temporal):
            raise H3ContractError("deferred Ref2VA target 不能伪装已解析时间字段")
        if require_resolved:
            raise H3ContractError("Ref2VA target 时长仍未从 reference audio 解析")
    elif temporal in ("resolved", "resolved_from_audio_reference"):
        if temporal == "resolved_from_audio_reference":
            if task != H3_TASK_REF2VA:
                raise H3ContractError("只有 Ref2VA 能使用 audio-derived duration")
            if clean.get("requested_duration_seconds") is not None:
                raise H3ContractError("audio-derived target 不应声明请求时长")
            source_duration = _validate_duration_seconds(
                clean.get("audio_reference_duration_seconds")
            )
        else:
            source_duration = _validate_duration_seconds(
                clean.get("requested_duration_seconds")
            )
        assert source_duration is not None
        expected_frames = align_frame_count(int(round(source_duration * fps)))
        frame_count = _positive_int(clean.get("frame_count"), "target.frame_count")
        if frame_count != expected_frames:
            raise H3ContractError(
                "target.frame_count 与 requested/source duration 的 17n+5 对齐结果不一致"
            )
        expected_duration = expected_frames / fps
        duration = _finite_float(
            clean.get("duration_seconds"),
            "target.duration_seconds",
            positive=True,
        )
        if not math.isclose(duration, expected_duration, rel_tol=0.0, abs_tol=1e-9):
            raise H3ContractError(
                "target.duration_seconds 与 requested/source duration 对齐结果不一致"
            )
        latent_t = _positive_int(clean.get("video_latent_t"), "target.video_latent_t")
        if latent_t != video_latent_t(expected_frames):
            raise H3ContractError("target.frame_count 与 video_latent_t 不一致")
        audio_t = _positive_int(clean.get("audio_latent_t"), "target.audio_latent_t")
        if audio_t != audio_latent_t(expected_duration):
            raise H3ContractError("target.frame_count 与 audio_latent_t 不一致")
        clean["frame_count"] = frame_count
        clean["video_latent_t"] = latent_t
        clean["audio_latent_t"] = audio_t
        clean["duration_seconds"] = duration
    else:
        raise H3ContractError(f"未知 target.temporal {temporal!r}")

    expected_stage = (
        "pre_media" if spatial_deferred or temporal_deferred else "resolved"
    )
    if clean.get("resolution_stage") != expected_stage:
        raise H3ContractError(
            f"target.resolution_stage 必须为 {expected_stage!r}"
        )
    if task == H3_TASK_FL2VA and not spatial_deferred:
        signature_raw = clean.get("keyframe_signature")
        semantic_raw = clean.get("semantic_frame_indices")
        pixels_raw = clean.get("pixel_frame_indices")
        try:
            signature = tuple(int(value) for value in signature_raw)
            semantic = tuple(int(value) for value in semantic_raw)
            pixels = tuple(int(value) for value in pixels_raw)
        except (TypeError, ValueError) as exc:
            raise H3ContractError("FL2VA target 缺少完整 keyframe index 契约") from exc
        if signature not in H3_FL2VA_KEYFRAME_SIGNATURES or semantic != signature:
            raise H3ContractError("FL2VA target keyframe signature 非法或语义索引不一致")
        expected_pixels = tuple(
            int(clean["frame_count"]) - 1 if value == -1 else value
            for value in semantic
        )
        if pixels != expected_pixels:
            raise H3ContractError("FL2VA target resolved keyframe indices 不一致")
        clean["keyframe_signature"] = list(signature)
        clean["semantic_frame_indices"] = list(semantic)
        clean["pixel_frame_indices"] = list(pixels)
    if task == H3_TASK_REF2VA:
        reference_fingerprint = clean.get("reference_order_fingerprint")
        if not isinstance(reference_fingerprint, str) or not reference_fingerprint:
            raise H3ContractError(
                "Ref2VA target 缺少 reference_order_fingerprint"
            )
    return clean


def validate_target(target: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(target, Mapping):
        raise H3ContractError(
            "target 端口内容无效；请连接 MiniMax H3 T2VA Target 节点"
        )
    if target.get("schema") != H3_TARGET_SCHEMA:
        raise H3ContractError(
            f"target schema 不匹配：期望 {H3_TARGET_SCHEMA!r}"
        )
    require_t2va(str(target.get("task", "")))
    required = (
        "width",
        "height",
        "frame_count",
        "video_latent_t",
        "video_latent_h",
        "video_latent_w",
        "audio_latent_t",
    )
    missing = [key for key in required if key not in target]
    if missing:
        raise H3ContractError(f"target 缺少字段：{', '.join(missing)}")
    clean = dict(target)
    for key in required:
        clean[key] = _positive_int(clean[key], f"target.{key}")
    if clean["width"] % H3_CANVAS_MULTIPLE or clean["height"] % H3_CANVAS_MULTIPLE:
        raise H3ContractError("target width/height 必须按 32 对齐")
    if (
        clean["video_latent_h"] != clean["height"] // H3_VIDEO_VAE_SPATIAL_RATIO
        or clean["video_latent_w"]
        != clean["width"] // H3_VIDEO_VAE_SPATIAL_RATIO
    ):
        raise H3ContractError("target 的像素尺寸与 video latent 尺寸不一致")
    return clean


def make_t2va_conditioning(prompt: str, prompt_embeds: Any) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise H3ContractError("prompt 不能为空")
    shape = getattr(prompt_embeds, "shape", None)
    if shape is None or len(shape) not in (2, 3):
        raise H3ContractError(
            "Qwen 编码结果必须是 [L,5120] 或 [1,L,5120] tensor"
        )
    if len(shape) == 3 and int(shape[0]) != 1:
        raise H3ContractError("MiniMax-H3 Direct v0 只支持 batch=1")
    if int(shape[-1]) != H3_TEXT_WIDTH:
        raise H3ContractError(
            f"Qwen hidden width 必须为 {H3_TEXT_WIDTH}，实际为 {int(shape[-1])}"
        )
    return {
        "schema": H3_CONDITIONING_SCHEMA,
        "task": H3_TASK_T2VA,
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,
        "cfg_distilled": True,
        "guidance_scale": 1.0,
    }


def validate_conditioning(conditioning: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(conditioning, Mapping):
        raise H3ContractError(
            "conditioning 端口内容无效；请连接 MiniMax H3 T2VA Text Encode"
        )
    if conditioning.get("schema") != H3_CONDITIONING_SCHEMA:
        raise H3ContractError(
            f"conditioning schema 不匹配：期望 {H3_CONDITIONING_SCHEMA!r}"
        )
    require_t2va(str(conditioning.get("task", "")))
    embeds = conditioning.get("prompt_embeds")
    shape = getattr(embeds, "shape", None)
    if shape is None:
        raise H3ContractError("conditioning 缺少 prompt_embeds tensor")
    if len(shape) == 3 and int(shape[0]) == 1:
        pass
    elif len(shape) != 2:
        raise H3ContractError("prompt_embeds 必须是 [L,5120] 或 [1,L,5120]")
    if int(shape[-1]) != H3_TEXT_WIDTH:
        raise H3ContractError(
            f"prompt_embeds 最后一维必须为 {H3_TEXT_WIDTH}"
        )
    return dict(conditioning)


def _validate_prompt_embeds(prompt_embeds: Any, *, path: str) -> Any:
    shape = getattr(prompt_embeds, "shape", None)
    if shape is None or len(shape) not in (2, 3):
        raise H3ContractError(f"{path} 必须是 [L,5120] 或 [1,L,5120] tensor")
    if len(shape) == 3 and int(shape[0]) != 1:
        raise H3ContractError("MiniMax-H3 当前只支持 batch=1")
    if int(shape[-1]) != H3_TEXT_WIDTH:
        raise H3ContractError(
            f"{path} 最后一维必须为 {H3_TEXT_WIDTH}，实际为 {int(shape[-1])}"
        )
    return prompt_embeds


def _prompt_sequence_length(prompt_embeds: Any) -> int:
    _validate_prompt_embeds(prompt_embeds, path="prompt_embeds")
    shape = prompt_embeds.shape
    return int(shape[-2])


def _normalise_text_token_tags(
    value: Any,
    *,
    expected_length: int,
    required: bool,
) -> Any:
    if value is None:
        if required:
            raise H3ContractError(
                "FL2VA/Ref2VA conditioning 必须包含官方 presentation text_token_tags"
            )
        return None
    shape = getattr(value, "shape", None)
    values: list[Any] | None = None
    if shape is not None:
        if len(shape) != 1:
            raise H3ContractError("text_token_tags 必须是 rank-1")
        length = int(shape[0])
        candidate = value
        for method_name in ("detach", "cpu"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                candidate = method()
        tolist = getattr(candidate, "tolist", None)
        if callable(tolist):
            candidate_values = tolist()
            if isinstance(candidate_values, list):
                values = candidate_values
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        length = len(value)
        values = list(value)
    else:
        raise H3ContractError("text_token_tags 必须是一维 tensor 或整数列表")
    if length != expected_length:
        raise H3ContractError(
            f"text_token_tags 长度必须等于 prompt_embeds L={expected_length}，"
            f"实际为 {length}"
        )
    if values is not None:
        for index, tag in enumerate(values):
            if isinstance(tag, bool) or not isinstance(tag, (int, float)):
                raise H3ContractError(f"text_token_tags[{index}] 必须是整数 0 或 1")
            if int(tag) != tag or int(tag) not in (0, 1):
                raise H3ContractError(f"text_token_tags[{index}] 只允许 0 或 1")
    return value


def target_compatibility_fingerprint(target: Mapping[str, Any]) -> str:
    """Match the node-level stable target fingerprint without media state."""

    payload = json.dumps(
        dict(target),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _condition_order_payload(
    task: str,
    conditions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        entry: dict[str, Any] = {
            "condition_index": index,
            "kind": str(condition["type"]),
        }
        if task == H3_TASK_FL2VA:
            entry["semantic_frame_index"] = int(condition["frame_index"])
            if condition.get("resolved_frame_index") is not None:
                entry["resolved_frame_index"] = int(
                    condition["resolved_frame_index"]
                )
        elif task == H3_TASK_REF2VA:
            entry["material_fingerprint"] = str(
                condition["material_fingerprint"]
            )
        payload.append(entry)
    return payload


def condition_order_fingerprint(
    task: str,
    conditions: Sequence[Mapping[str, Any]],
) -> str:
    payload = json.dumps(
        _condition_order_payload(normalize_task(task), conditions),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_condition_blocks(
    *,
    task: str,
    conditions: Sequence[Mapping[str, Any]],
    blocks: Any,
) -> list[dict[str, Any]]:
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise H3ContractError("conditioning.condition_blocks 必须是有序列表")
    if len(blocks) != len(conditions):
        raise H3ContractError(
            "conditioning.conditions 与 condition_blocks 数量必须一一对应"
        )
    clean_blocks: list[dict[str, Any]] = []
    for index, (condition, raw_block) in enumerate(
        zip(conditions, blocks, strict=True)
    ):
        path = f"condition_blocks[{index}]"
        if not isinstance(raw_block, Mapping):
            raise H3ContractError(f"{path} 必须是对象")
        block = dict(raw_block)
        condition_index = block.get("condition_index")
        if (
            isinstance(condition_index, bool)
            or not isinstance(condition_index, int)
            or condition_index != index
        ):
            raise H3ContractError(
                f"{path}.condition_index 必须严格等于原始顺序 {index}"
            )
        kind = block.get("kind", block.get("type"))
        if not isinstance(kind, str):
            raise H3ContractError(f"{path}.kind 必须是字符串")
        kind = kind.strip().lower()
        expected_kind = str(condition["type"])
        if task == H3_TASK_FL2VA:
            if kind not in ("image", "keyframe"):
                raise H3ContractError(f"{path}.kind 必须为 image/keyframe")
            semantic = block.get(
                "semantic_frame_index", block.get("frame_index")
            )
            resolved = block.get("resolved_frame_index")
            if semantic != condition.get("frame_index"):
                raise H3ContractError(f"{path} 的 semantic_frame_index 串线")
            if resolved != condition.get("resolved_frame_index"):
                raise H3ContractError(f"{path} 的 resolved_frame_index 串线")
        else:
            if kind != expected_kind:
                raise H3ContractError(
                    f"{path}.kind={kind!r} 与 references[{index}].type="
                    f"{expected_kind!r} 不一致"
                )
            expected_fingerprint = condition.get("material_fingerprint")
            if block.get("material_fingerprint") != expected_fingerprint:
                raise H3ContractError(
                    f"{path}.material_fingerprint 与 references[{index}] 串线"
                )
        block["condition_index"] = index
        block["kind"] = kind
        clean_blocks.append(block)
    return clean_blocks


def make_conditioning_v2(
    task: str,
    prompt: str,
    prompt_embeds: Any,
    *,
    conditions: Any = None,
    text_token_tags: Any = None,
    token_tags: Any = None,
) -> H3ConditioningV2:
    """Create task-aware Qwen conditioning before packed-sequence assembly."""

    normalized_task = normalize_task(task)
    if not isinstance(prompt, str) or not prompt.strip():
        raise H3ContractError("prompt 不能为空")
    _validate_prompt_embeds(prompt_embeds, path="prompt_embeds")
    if text_token_tags is not None and token_tags is not None:
        raise H3ContractError(
            "只能传 text_token_tags；token_tags 仅作为旧调用兼容别名"
        )
    # Explicit legacy compatibility: read the old spelling, but canonicalize
    # the returned v2 payload to text_token_tags only.
    canonical_tags = text_token_tags if text_token_tags is not None else token_tags
    canonical_tags = _normalise_text_token_tags(
        canonical_tags,
        expected_length=_prompt_sequence_length(prompt_embeds),
        required=normalized_task in (H3_TASK_FL2VA, H3_TASK_REF2VA),
    )
    if normalized_task == H3_TASK_T2VA:
        if conditions not in (None, (), []):
            raise H3ContractError("T2VA conditioning 不允许条件素材")
        clean_conditions: list[H3ConditionMaterialV2] = []
    elif normalized_task == H3_TASK_FL2VA:
        clean_conditions = validate_fl2va_keyframes(conditions)
    else:
        clean_conditions = validate_ref2va_references(conditions)
    conditioning: H3ConditioningV2 = {
        "schema": H3_CONDITIONING_SCHEMA_V2,
        "task": normalized_task,
        "partition": partition_for_task(normalized_task),
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,
        "conditions": clean_conditions,
        "cfg_distilled": True,
        "guidance_scale": 1.0,
    }
    if canonical_tags is not None:
        conditioning["text_token_tags"] = canonical_tags
    if clean_conditions:
        conditioning["condition_order_fingerprint"] = condition_order_fingerprint(
            normalized_task, clean_conditions
        )
    return conditioning


def validate_conditioning_v2(
    conditioning: Any,
    *,
    expected_task: str | None = None,
) -> H3ConditioningV2:
    if not isinstance(conditioning, Mapping):
        raise H3ContractError("conditioning v2 必须是对象")
    if conditioning.get("schema") != H3_CONDITIONING_SCHEMA_V2:
        raise H3ContractError(
            f"conditioning schema 不匹配：期望 {H3_CONDITIONING_SCHEMA_V2!r}"
        )
    clean: H3ConditioningV2 = dict(conditioning)
    task = normalize_task(clean.get("task"))
    if expected_task is not None and task != normalize_task(expected_task):
        raise H3ContractError(
            f"conditioning.task={task!r} 与工作流 task="
            f"{normalize_task(expected_task)!r} 不一致"
        )
    clean["task"] = task
    clean["partition"] = validate_task_partition(task, clean.get("partition"))
    prompt = clean.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise H3ContractError("conditioning.prompt 不能为空")
    prompt_embeds = clean.get("prompt_embeds")
    _validate_prompt_embeds(prompt_embeds, path="conditioning.prompt_embeds")
    if clean.get("text_token_tags") is not None and clean.get("token_tags") is not None:
        raise H3ContractError(
            "conditioning 不能同时包含 text_token_tags 和旧 token_tags"
        )
    legacy_tags = clean.pop("token_tags", None)
    text_token_tags = (
        clean.get("text_token_tags")
        if clean.get("text_token_tags") is not None
        else legacy_tags
    )
    text_token_tags = _normalise_text_token_tags(
        text_token_tags,
        expected_length=_prompt_sequence_length(prompt_embeds),
        required=task in (H3_TASK_FL2VA, H3_TASK_REF2VA),
    )
    if text_token_tags is not None:
        clean["text_token_tags"] = text_token_tags
    if clean.get("cfg_distilled") is not True:
        raise H3ContractError("MiniMax-H3 v2 只允许单个 CFG-distilled 正向分支")
    guidance = _finite_float(clean.get("guidance_scale"), "guidance_scale")
    if guidance != 1.0:
        raise H3ContractError("MiniMax-H3 v2 guidance_scale 必须为 1.0")
    raw_conditions = clean.get("conditions", [])
    if task == H3_TASK_T2VA:
        if raw_conditions not in (None, (), []):
            raise H3ContractError("T2VA conditioning 不允许条件素材")
        clean["conditions"] = []
        if clean.get("condition_blocks") not in (None, (), []):
            raise H3ContractError("T2VA conditioning 不允许 condition_blocks")
    elif task == H3_TASK_FL2VA:
        target = validate_target_v2(
            clean.get("target"),
            expected_task=task,
            require_resolved=True,
        )
        clean["target"] = target
        clean["conditions"] = validate_fl2va_keyframes(
            raw_conditions,
            frame_count=int(target["frame_count"]),
        )
    else:
        target = validate_target_v2(
            clean.get("target"),
            expected_task=task,
            require_resolved=True,
        )
        clean["target"] = target
        clean["conditions"] = validate_ref2va_references(raw_conditions)
    if task in (H3_TASK_FL2VA, H3_TASK_REF2VA):
        expected_order = condition_order_fingerprint(task, clean["conditions"])
        if clean.get("condition_order_fingerprint") != expected_order:
            raise H3ContractError(
                "conditioning.condition_order_fingerprint 与原始条件顺序不一致"
            )
        clean["condition_blocks"] = _validate_condition_blocks(
            task=task,
            conditions=clean["conditions"],
            blocks=clean.get("condition_blocks"),
        )
        expected_target_fingerprint = target_compatibility_fingerprint(clean["target"])
        if clean.get("target_fingerprint") != expected_target_fingerprint:
            raise H3ContractError(
                "conditioning.target_fingerprint 与 target 内容不一致"
            )
        if task == H3_TASK_FL2VA:
            semantic = [
                int(item["frame_index"]) for item in clean["conditions"]
            ]
            resolved = [
                int(item["resolved_frame_index"]) for item in clean["conditions"]
            ]
            if semantic != list(clean["target"]["semantic_frame_indices"]):
                raise H3ContractError("FL2VA conditioning 与 target 语义关键帧串线")
            if resolved != list(clean["target"]["pixel_frame_indices"]):
                raise H3ContractError("FL2VA conditioning 与 target 像素关键帧串线")
        elif clean["target"].get("reference_order_fingerprint") != expected_order:
            raise H3ContractError(
                "Ref2VA conditioning 与 target reference 顺序指纹不一致"
            )
        for field in (
            "release_fingerprint",
            "text_encoder_fingerprint",
            "vae_fingerprint",
        ):
            value = clean.get(field)
            if not isinstance(value, str) or not value:
                raise H3ContractError(f"conditioning.{field} 必须是非空字符串")
    return clean


def validate_av_latent(latent: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(latent, Mapping):
        raise H3ContractError(
            "av_latent 端口内容无效；请连接 MiniMax H3 Empty AV Latent"
        )
    if latent.get("schema") != H3_AV_LATENT_SCHEMA:
        raise H3ContractError(
            f"av_latent schema 不匹配：期望 {H3_AV_LATENT_SCHEMA!r}"
        )
    require_t2va(str(latent.get("task", "")))
    target = validate_target(latent.get("target", {}))
    video = latent.get("video")
    audio = latent.get("audio")
    video_shape = getattr(video, "shape", None)
    audio_shape = getattr(audio, "shape", None)
    expected_video = (
        1,
        H3_VIDEO_CHANNELS,
        target["video_latent_t"],
        target["video_latent_h"],
        target["video_latent_w"],
    )
    expected_audio = (
        H3_AUDIO_CHANNELS,
        H3_AUDIO_LATENT_CHANNELS,
        target["audio_latent_t"],
    )
    if video_shape is None or tuple(int(x) for x in video_shape) != expected_video:
        raise H3ContractError(
            f"video latent shape 必须为 {expected_video}，实际为 {video_shape}"
        )
    if audio_shape is None or tuple(int(x) for x in audio_shape) != expected_audio:
        raise H3ContractError(
            f"audio latent shape 必须为 {expected_audio}，实际为 {audio_shape}"
        )
    return dict(latent)


def validate_av_latent_v2(
    latent: Any,
    *,
    expected_task: str | None = None,
) -> dict[str, Any]:
    if not isinstance(latent, Mapping):
        raise H3ContractError("av_latent v2 必须是对象")
    if latent.get("schema") != H3_AV_LATENT_SCHEMA_V2:
        raise H3ContractError(
            f"av_latent schema 不匹配：期望 {H3_AV_LATENT_SCHEMA_V2!r}"
        )
    clean = dict(latent)
    task = normalize_task(clean.get("task"))
    if expected_task is not None and task != normalize_task(expected_task):
        raise H3ContractError(
            f"av_latent.task={task!r} 与工作流 task="
            f"{normalize_task(expected_task)!r} 不一致"
        )
    clean["task"] = task
    clean["partition"] = validate_task_partition(task, clean.get("partition"))
    target = validate_target_v2(
        clean.get("target"),
        expected_task=task,
        require_resolved=True,
    )
    clean["target"] = target
    declared_target_fingerprint = clean.get("target_fingerprint")
    if declared_target_fingerprint is not None:
        expected_target_fingerprint = target_compatibility_fingerprint(target)
        if declared_target_fingerprint != expected_target_fingerprint:
            raise H3ContractError(
                "av_latent.target_fingerprint 与 av_latent.target 不一致"
            )
    video = clean.get("video")
    audio = clean.get("audio")
    video_shape = getattr(video, "shape", None)
    audio_shape = getattr(audio, "shape", None)
    expected_video = (
        1,
        H3_VIDEO_CHANNELS,
        target["video_latent_t"],
        target["video_latent_h"],
        target["video_latent_w"],
    )
    expected_audio = (
        H3_AUDIO_CHANNELS,
        H3_AUDIO_LATENT_CHANNELS,
        target["audio_latent_t"],
    )
    if video_shape is None or tuple(int(value) for value in video_shape) != expected_video:
        raise H3ContractError(
            f"video latent shape 必须为 {expected_video}，实际为 {video_shape}"
        )
    if audio_shape is None or tuple(int(value) for value in audio_shape) != expected_audio:
        raise H3ContractError(
            f"audio latent shape 必须为 {expected_audio}，实际为 {audio_shape}"
        )
    if clean.get("sampled") not in (True, False):
        raise H3ContractError("av_latent.sampled 必须是 bool")
    return clean


def validate_release_for_task(
    metadata: Any,
    task: str,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Validate the official ``model_index._minimax_h3`` admission record."""

    normalized_task = normalize_task(task)
    if metadata in (None, {}):
        if allow_missing:
            return {}
        raise H3ContractError("缺少 model_index.json._minimax_h3 release metadata")
    if not isinstance(metadata, Mapping):
        raise H3ContractError("release_metadata 必须是对象")
    raw = metadata.get("_minimax_h3", metadata)
    if not isinstance(raw, Mapping):
        raise H3ContractError("model_index.json._minimax_h3 必须是对象")
    if raw.get("schema_version") != 1:
        raise H3ContractError("release_metadata.schema_version 必须为 1")
    partition = validate_task_partition(normalized_task, raw.get("partition"))
    tasks = raw.get("tasks")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not tasks:
        raise H3ContractError("release_metadata.tasks 必须是非空字符串列表")
    normalized_tasks: list[str] = []
    for index, value in enumerate(tasks):
        try:
            declared_task = normalize_task(value)
        except H3ContractError as exc:
            raise H3ContractError(
                f"release_metadata.tasks[{index}] 不是受支持任务"
            ) from exc
        validate_task_partition(declared_task, partition)
        normalized_tasks.append(declared_task)
    if len(set(normalized_tasks)) != len(normalized_tasks):
        raise H3ContractError("release_metadata.tasks 不能包含重复项")
    if normalized_task not in normalized_tasks:
        raise H3ContractError(
            f"release partition {partition!r} 未声明 task {normalized_task!r}；"
            f"tasks={normalized_tasks!r}"
        )
    aliases = raw.get("task_aliases", {})
    if not isinstance(aliases, Mapping):
        raise H3ContractError("release_metadata.task_aliases 必须是对象")
    for alias, target in aliases.items():
        if not isinstance(alias, str) or not alias or not isinstance(target, str):
            raise H3ContractError("release_metadata.task_aliases 必须映射非空字符串")
        if target not in normalized_tasks:
            raise H3ContractError(
                f"release task alias {alias!r} 指向未声明任务 {target!r}"
            )
        # The official v1 dispatcher defines no aliases.  Identity entries are
        # harmless, but a new public spelling must not be guessed here.
        if alias != target:
            raise H3ContractError(
                f"当前 MiniMax-H3 协议不支持 task alias {alias!r}->{target!r}"
            )
    scales = raw.get("sigma_shift_scales")
    if not isinstance(scales, Mapping):
        raise H3ContractError("release_metadata.sigma_shift_scales 必须是对象")
    normalized_scales = {
        "video": _finite_float(
            scales.get("video"),
            "sigma_shift_scales.video",
            positive=True,
        ),
        "audio": _finite_float(
            scales.get("audio"),
            "sigma_shift_scales.audio",
            positive=True,
        ),
    }
    clean = dict(raw)
    clean.update(
        {
            "schema_version": 1,
            "partition": partition,
            "tasks": normalized_tasks,
            "task_aliases": dict(aliases),
            "sigma_shift_scales": normalized_scales,
        }
    )
    return clean


_H3_COMPONENT_SCHEMA_OPTIONS = MappingProxyType(
    {
        "model": frozenset((H3_MODEL_SCHEMA, H3_MODEL_SCHEMA_V2)),
        "text_encoder": frozenset(
            (H3_TEXT_ENCODER_SCHEMA, H3_TEXT_ENCODER_SCHEMA_V2)
        ),
        "vae": frozenset((H3_VAE_SCHEMA, H3_VAE_SCHEMA_V2)),
    }
)

_H3_COMPONENT_PATH_FIELDS = MappingProxyType(
    {
        "model": "transformer_path",
        "text_encoder": "text_encoder_path",
        "vae": "vae_path",
    }
)

_H3_COMPONENT_FINGERPRINT_FIELDS = MappingProxyType(
    {
        "model": "transformer_fingerprint",
        "text_encoder": "text_encoder_fingerprint",
        "vae": "vae_fingerprint",
    }
)

_H3_COMPONENT_FINGERPRINT_KINDS = MappingProxyType(
    {
        "model": "transformer",
        "text_encoder": "text_encoder",
        "vae": "vae",
    }
)

_H3_COMPONENT_RELATED_PATH_FIELDS = MappingProxyType(
    {
        "model": (),
        "text_encoder": ("tokenizer_path", "processor_path"),
        "vae": (),
    }
)


_H3_IDENTITY_JSON_NAMES = frozenset(
    {
        "config.json",
        "model_index.json",
        "quant_meta.json",
        "merge_meta.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "chat_template.json",
        "added_tokens.json",
    }
)
_H3_IDENTITY_STAT_NAMES = frozenset(
    {"tokenizer.json", "vocab.json", "merges.txt", "vocab.txt"}
)


def _canonical_json_digest(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise H3ContractError(
            f"MiniMax-H3 identity JSON cannot be read: {path}: {exc}"
        ) from exc
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _component_artifact_snapshot(component_path: str | Path) -> dict[str, Any]:
    """Cheap local snapshot identity; this is not a content-integrity proof."""

    root = Path(component_path).expanduser().resolve()
    snapshot: dict[str, Any] = {"root": str(root), "exists": root.exists()}
    if not root.is_dir():
        return snapshot

    json_files: list[dict[str, Any]] = []
    storage_files: list[dict[str, Any]] = []
    try:
        candidates = sorted(root.rglob("*"), key=lambda path: str(path))
    except OSError as exc:
        raise H3ContractError(
            f"Cannot enumerate MiniMax-H3 component identity below {root}: {exc}"
        ) from exc
    for logical_path in candidates:
        resolved_path = logical_path.resolve()
        try:
            logical_relative = logical_path.relative_to(root)
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise H3ContractError(
                f"MiniMax-H3 identity artifact escapes component root {root}: "
                f"{logical_path} -> {resolved_path}"
            ) from exc
        if not resolved_path.is_file():
            continue
        name = logical_path.name
        relative_text = logical_relative.as_posix()
        if name in _H3_IDENTITY_JSON_NAMES or name.endswith(
            ".safetensors.index.json"
        ):
            json_files.append(
                {
                    "path": relative_text,
                    "resolved": str(resolved_path),
                    "sha256": _canonical_json_digest(resolved_path),
                }
            )
            continue
        if name.endswith(".safetensors") or name in _H3_IDENTITY_STAT_NAMES:
            try:
                stat = resolved_path.stat()
            except OSError as exc:
                raise H3ContractError(
                    f"Cannot stat MiniMax-H3 identity artifact {resolved_path}: {exc}"
                ) from exc
            storage_files.append(
                {
                    "path": relative_text,
                    "resolved": str(resolved_path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    snapshot["json"] = json_files
    snapshot["storage"] = storage_files
    return snapshot


def compute_release_fingerprint(
    model_root: str,
    partition: str,
    metadata: Mapping[str, Any],
) -> str:
    root = Path(model_root).resolve()
    value: dict[str, Any] = {
        "root": str(root),
        "partition": partition,
        "metadata": dict(metadata),
    }
    model_index = root / "model_index.json"
    if model_index.is_file():
        resolved_index = model_index.resolve()
        try:
            resolved_index.relative_to(root)
        except ValueError as exc:
            raise H3ContractError(
                f"MiniMax-H3 model_index escapes release root {root}: "
                f"{resolved_index}"
            ) from exc
        value["model_index"] = {
            "resolved": str(resolved_index),
            "sha256": _canonical_json_digest(resolved_index),
        }
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_component_fingerprint(
    release_fingerprint: str,
    component_kind: str,
    component_path: str,
    *,
    related_paths: Mapping[str, str | Path] | None = None,
) -> str:
    resolved_component = Path(component_path).resolve()
    related = {
        str(role): Path(path).resolve()
        for role, path in (related_paths or {}).items()
        if str(path).strip()
    }
    # Preserve the old deterministic wire value for synthetic/non-local test
    # descriptors. Real loader outputs always point at an existing directory and
    # therefore receive the enriched local snapshot identity below.
    if not resolved_component.exists() and not related:
        payload = (
            f"{release_fingerprint}\0{component_kind}\0{resolved_component}"
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    value = {
        "release_fingerprint": release_fingerprint,
        "component_kind": component_kind,
        "component": _component_artifact_snapshot(resolved_component),
        "related": {
            role: _component_artifact_snapshot(path)
            for role, path in sorted(related.items())
        },
    }
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _computed_release_fingerprint(
    model_root: str,
    partition: str,
    metadata: Mapping[str, Any],
) -> str:
    return compute_release_fingerprint(model_root, partition, metadata)


def _computed_component_fingerprint(
    release_fingerprint: str,
    component_kind: str,
    component_path: str,
    *,
    related_paths: Mapping[str, str | Path] | None = None,
) -> str:
    return compute_component_fingerprint(
        release_fingerprint,
        component_kind,
        component_path,
        related_paths=related_paths,
    )


def validate_component_for_task(
    component: Any,
    *,
    component_kind: str,
    task: str,
) -> dict[str, Any]:
    """Validate one loader wrapper before any large handle is used."""

    normalized_task = normalize_task(task)
    if component_kind not in _H3_COMPONENT_SCHEMA_OPTIONS:
        raise H3ContractError(
            f"未知 component_kind {component_kind!r}；可选值为 model/text_encoder/vae"
        )
    if not isinstance(component, Mapping):
        raise H3ContractError(f"{component_kind} component 必须是对象")
    schema = component.get("schema")
    if schema not in _H3_COMPONENT_SCHEMA_OPTIONS[component_kind]:
        raise H3ContractError(
            f"{component_kind}.schema 不匹配；期望 "
            f"{sorted(_H3_COMPONENT_SCHEMA_OPTIONS[component_kind])!r}"
        )
    is_v2 = schema == H3_COMPONENT_SCHEMAS_V2[component_kind]
    if not is_v2 and normalized_task != H3_TASK_T2VA:
        raise H3ContractError(
            f"{component_kind} 的 v1 wrapper 只允许 T2VA；"
            f"{normalized_task.upper()} 必须重新经过 task-aware v2 Loader"
        )
    partition = validate_task_partition(normalized_task, component.get("partition"))
    if is_v2 and normalize_task(component.get("task")) != normalized_task:
        raise H3ContractError(
            f"{component_kind}.task 必须明确等于当前 task {normalized_task!r}"
        )
    release_raw = component.get("release_metadata")
    release = validate_release_for_task(
        release_raw,
        normalized_task,
        allow_missing=True,
    )
    declared = component.get("tasks")
    if declared is None and release:
        declared = release["tasks"]
    if declared is None and component.get("task") is not None:
        declared = [component.get("task")]
    if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
        raise H3ContractError(
            f"{component_kind} 必须声明 task 或 tasks 列表"
        )
    tasks: list[str] = []
    for value in declared:
        declared_task = normalize_task(value)
        validate_task_partition(declared_task, partition)
        tasks.append(declared_task)
    if normalized_task not in tasks:
        raise H3ContractError(
            f"{component_kind} 未声明 task {normalized_task!r}；tasks={tasks!r}"
        )
    root = component.get("model_root")
    if not isinstance(root, str) or not root.strip():
        raise H3ContractError(f"{component_kind}.model_root 必须是非空字符串")
    resolved_root = str(Path(root).resolve())
    clean = dict(component)
    clean["partition"] = partition
    clean["tasks"] = tasks
    clean["model_root"] = resolved_root
    if release_raw is None:
        # Downstream wrapper fingerprint helpers require a mapping.  Missing
        # official metadata is represented canonically as an empty mapping.
        clean["release_metadata"] = {}
    if is_v2:
        # Loader fingerprints are computed from the exact metadata object that
        # was read from disk.  Validate a normalized copy above, but preserve
        # the raw JSON representation here (for example 12 versus 12.0).
        fingerprint_metadata = (
            dict(release_raw) if isinstance(release_raw, Mapping) else {}
        )
        expected_release_fingerprint = _computed_release_fingerprint(
            resolved_root,
            partition,
            fingerprint_metadata,
        )
        if component.get("release_fingerprint") != expected_release_fingerprint:
            raise H3ContractError(
                f"{component_kind}.release_fingerprint 与 root/partition/metadata 不一致"
            )
        path_field = _H3_COMPONENT_PATH_FIELDS[component_kind]
        component_path = component.get(path_field)
        if not isinstance(component_path, str) or not component_path.strip():
            raise H3ContractError(f"{component_kind}.{path_field} 必须是非空路径")
        resolved_component_path = Path(component_path).resolve()
        try:
            resolved_component_path.relative_to(Path(resolved_root))
        except ValueError as exc:
            raise H3ContractError(
                f"{component_kind}.{path_field} 必须位于 resolved model_root 内"
            ) from exc
        related_paths: dict[str, Path] = {}
        for related_field in _H3_COMPONENT_RELATED_PATH_FIELDS[component_kind]:
            related_value = component.get(related_field)
            if related_value is None:
                continue
            if not isinstance(related_value, str) or not related_value.strip():
                raise H3ContractError(
                    f"{component_kind}.{related_field} 必须是非空路径"
                )
            resolved_related = Path(related_value).resolve()
            try:
                resolved_related.relative_to(Path(resolved_root))
            except ValueError as exc:
                raise H3ContractError(
                    f"{component_kind}.{related_field} 必须位于 resolved model_root 内"
                ) from exc
            related_paths[related_field.removesuffix("_path")] = resolved_related
            clean[related_field] = str(resolved_related)
        expected_component_fingerprint = _computed_component_fingerprint(
            expected_release_fingerprint,
            _H3_COMPONENT_FINGERPRINT_KINDS[component_kind],
            str(resolved_component_path),
            related_paths=related_paths,
        )
        if component.get("component_fingerprint") != expected_component_fingerprint:
            raise H3ContractError(
                f"{component_kind}.component_fingerprint 与组件路径不一致"
            )
        specific_field = _H3_COMPONENT_FINGERPRINT_FIELDS[component_kind]
        if component.get(specific_field) != expected_component_fingerprint:
            raise H3ContractError(
                f"{component_kind}.{specific_field} 与 component_fingerprint 不一致"
            )
        clean["release_fingerprint"] = expected_release_fingerprint
        clean["component_fingerprint"] = expected_component_fingerprint
        clean[path_field] = str(resolved_component_path)
    return clean


def component_compatibility_fingerprint(
    component: Any,
    *,
    component_kind: str,
    task: str,
) -> tuple[str, str, tuple[str, ...], int | None]:
    """Stable data-only identity used to reject mixed loader outputs."""

    clean = validate_component_for_task(
        component,
        component_kind=component_kind,
        task=task,
    )
    release = clean.get("release_metadata")
    schema_version = (
        int(release["schema_version"])
        if isinstance(release, Mapping) and release.get("schema_version") is not None
        else None
    )
    return (
        clean["model_root"],
        clean["partition"],
        tuple(clean["tasks"]),
        schema_version,
    )


def validate_component_compatibility(
    *,
    task: str,
    model: Any = None,
    text_encoder: Any = None,
    vae: Any = None,
    target: Any = None,
    conditioning: Any = None,
    av_latent: Any = None,
    require_same_model_root: bool = True,
) -> dict[str, Any]:
    """Cross-check every connected H3 value at one workflow boundary."""

    normalized_task = normalize_task(task)
    clean: dict[str, Any] = {
        "task": normalized_task,
        "partition": partition_for_task(normalized_task),
    }
    components: dict[str, dict[str, Any]] = {}
    for component_kind, value in (
        ("model", model),
        ("text_encoder", text_encoder),
        ("vae", vae),
    ):
        if value is None:
            continue
        components[component_kind] = validate_component_for_task(
            value,
            component_kind=component_kind,
            task=normalized_task,
        )
    if require_same_model_root and components:
        roots = {value["model_root"] for value in components.values()}
        if len(roots) != 1:
            detail = ", ".join(
                f"{name}={value['model_root']!r}"
                for name, value in components.items()
            )
            raise H3ContractError(
                "MiniMax-H3 组件来自不同 model_root，禁止混用：" + detail
            )
    release_fingerprints = {
        name: value.get("release_fingerprint")
        for name, value in components.items()
        if value.get("release_fingerprint") is not None
    }
    if len(set(release_fingerprints.values())) > 1:
        raise H3ContractError("MiniMax-H3 v2 组件 release_fingerprint 不一致")
    clean["components"] = components
    clean.update(components)

    clean_target: dict[str, Any] | None = None
    if target is not None:
        if isinstance(target, Mapping) and target.get("schema") == H3_TARGET_SCHEMA_V2:
            clean_target = validate_target_v2(
                target,
                expected_task=normalized_task,
            )
            clean["target"] = clean_target
        elif normalized_task == H3_TASK_T2VA:
            clean["target"] = validate_target(target)
        else:
            raise H3ContractError(
                f"{normalized_task.upper()} 必须连接 target v2"
            )
    clean_conditioning: dict[str, Any] | None = None
    if conditioning is not None:
        if (
            isinstance(conditioning, Mapping)
            and conditioning.get("schema") == H3_CONDITIONING_SCHEMA_V2
        ):
            clean_conditioning = validate_conditioning_v2(
                conditioning,
                expected_task=normalized_task,
            )
            clean["conditioning"] = clean_conditioning
        elif normalized_task == H3_TASK_T2VA:
            clean["conditioning"] = validate_conditioning(conditioning)
        else:
            raise H3ContractError(
                f"{normalized_task.upper()} 必须连接 conditioning v2"
            )
    clean_latent: dict[str, Any] | None = None
    if av_latent is not None:
        if (
            isinstance(av_latent, Mapping)
            and av_latent.get("schema") == H3_AV_LATENT_SCHEMA_V2
        ):
            clean_latent = validate_av_latent_v2(
                av_latent,
                expected_task=normalized_task,
            )
            clean["av_latent"] = clean_latent
        elif normalized_task == H3_TASK_T2VA:
            clean["av_latent"] = validate_av_latent(av_latent)
        else:
            raise H3ContractError(
                f"{normalized_task.upper()} 必须连接 av_latent v2"
            )

    target_fingerprints: dict[str, str] = {}
    if clean_target is not None:
        target_fingerprints["target"] = target_compatibility_fingerprint(clean_target)
    if clean_conditioning is not None:
        conditioning_target = clean_conditioning.get("target")
        if isinstance(conditioning_target, Mapping):
            target_fingerprints["conditioning"] = target_compatibility_fingerprint(
                conditioning_target
            )
    if clean_latent is not None:
        target_fingerprints["av_latent"] = target_compatibility_fingerprint(
            clean_latent["target"]
        )
    if len(set(target_fingerprints.values())) > 1:
        raise H3ContractError("target/conditioning/av_latent.target 指纹不一致")

    if clean_conditioning is not None:
        conditioning_release = clean_conditioning.get("release_fingerprint")
        if conditioning_release is not None:
            if not isinstance(conditioning_release, str) or not conditioning_release:
                raise H3ContractError(
                    "conditioning.release_fingerprint 必须是非空字符串"
                )
            for name, fingerprint in release_fingerprints.items():
                if fingerprint != conditioning_release:
                    raise H3ContractError(
                        f"conditioning 与 {name} release_fingerprint 不一致"
                    )
        text_encoder_fingerprint = clean_conditioning.get(
            "text_encoder_fingerprint"
        )
        if "text_encoder" in components and text_encoder_fingerprint is not None:
            actual = components["text_encoder"]["component_fingerprint"]
            if text_encoder_fingerprint != actual:
                raise H3ContractError(
                    "conditioning.text_encoder_fingerprint 与 Text Encoder 不一致"
                )
        vae_fingerprint = clean_conditioning.get("vae_fingerprint")
        if "vae" in components and vae_fingerprint is not None:
            actual = components["vae"]["component_fingerprint"]
            if vae_fingerprint != actual:
                raise H3ContractError(
                    "conditioning.vae_fingerprint 与 VAE 不一致"
                )
    if clean_latent is not None:
        latent_release = clean_latent.get("release_fingerprint")
        if latent_release is not None:
            if not isinstance(latent_release, str) or not latent_release:
                raise H3ContractError("av_latent.release_fingerprint 必须是非空字符串")
            for name, fingerprint in release_fingerprints.items():
                if fingerprint != latent_release:
                    raise H3ContractError(
                        f"av_latent 与 {name} release_fingerprint 不一致"
                    )
        latent_vae = clean_latent.get("vae_fingerprint")
        if latent_vae is not None and "vae" in components:
            if latent_vae != components["vae"]["component_fingerprint"]:
                raise H3ContractError("av_latent.vae_fingerprint 与 VAE 不一致")
    return clean


def validate_seed(seed: Any) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > (1 << 63) - 1
    ):
        raise H3ContractError("seed 必须是 0 到 int64 最大值之间的整数")
    return int(seed)


def validate_sigma_request(
    *,
    sigma_points: Any,
    video_shift: Any,
    audio_shift: Any,
) -> tuple[int, float, float]:
    if (
        isinstance(sigma_points, bool)
        or not isinstance(sigma_points, int)
        or sigma_points < 2
    ):
        raise H3ContractError("sigma_points 必须是不小于 2 的整数")
    return (
        int(sigma_points),
        _finite_float(video_shift, "video_shift", positive=True),
        _finite_float(audio_shift, "audio_shift", positive=True),
    )


__all__ = [
    "H3_ASPECT_RATIOS",
    "H3_AUDIO_CHANNELS",
    "H3_AUDIO_LATENT_CHANNELS",
    "H3_AUDIO_REF_COND_TIMESTEP",
    "H3_AUDIO_ROW_WIDTH",
    "H3_AV_LATENT_SCHEMA",
    "H3_AV_LATENT_SCHEMA_V2",
    "H3_COMPONENT_SCHEMAS_V2",
    "H3_CONDITION_MATERIAL_SCHEMA_V2",
    "H3_CONDITION_ROLE_KEYFRAME",
    "H3_CONDITION_ROLE_REFERENCE",
    "H3_CONDITIONING_SCHEMA",
    "H3_CONDITIONING_SCHEMA_V2",
    "H3ComponentDescriptorV2",
    "H3ConditionMaterialV2",
    "H3ConditioningV2",
    "H3ContractError",
    "H3_DEFAULT_AUDIO_SHIFT",
    "H3_DEFAULT_SIGMA_POINTS",
    "H3_DEFAULT_VIDEO_SHIFT",
    "H3_FINITE_ASPECT_RATIOS",
    "H3_FL2VA_KEYFRAME_SIGNATURES",
    "H3_FPS",
    "H3_IMGVID_COND_TIMESTEP",
    "H3_MODEL_SCHEMA",
    "H3_MODEL_SCHEMA_V2",
    "H3_PARTITIONS",
    "H3_REFERENCE_LIST_SCHEMA_V2",
    "H3_REF2VA_AUDIO_BEARING_TYPES",
    "H3_REF2VA_PARTITION",
    "H3_REF2VA_REFERENCE_TYPES",
    "H3ReferenceListV2",
    "H3_TASK_FL2VA",
    "H3_TASK_PARTITIONS",
    "H3_TASK_REF2VA",
    "H3_TASK_T2VA",
    "H3_TASKS",
    "H3_TARGET_SCHEMA",
    "H3_TARGET_SCHEMA_V2",
    "H3TargetV2",
    "H3TaskNotImplementedError",
    "H3_TEXT_ENCODER_SCHEMA",
    "H3_TEXT_ENCODER_SCHEMA_V2",
    "H3_TEXT_WIDTH",
    "H3_T2VA_PARTITION",
    "H3_VAE_SCHEMA",
    "H3_VAE_SCHEMA_V2",
    "H3_VIDEO_CHANNELS",
    "H3_VIDEO_PATCH_SIZE",
    "H3_VIDEO_ROW_WIDTH",
    "align_frame_count",
    "append_ref2va_reference",
    "audio_latent_t",
    "component_compatibility_fingerprint",
    "compute_component_fingerprint",
    "compute_release_fingerprint",
    "condition_order_fingerprint",
    "frame_count_from_video_latent_t",
    "make_conditioning_v2",
    "make_fl2va_keyframe",
    "make_ref2va_reference",
    "make_ref2va_references",
    "make_t2va_conditioning",
    "material_compatibility_fingerprint",
    "normalize_task",
    "partition_for_task",
    "require_t2va",
    "resolve_deferred_target_v2",
    "resolve_fl2va_target_v2",
    "resolve_ref2va_target_v2",
    "resolve_spatial_shape",
    "resolve_t2va_target",
    "resolve_t2va_target_v2",
    "target_compatibility_fingerprint",
    "validate_av_latent",
    "validate_av_latent_v2",
    "validate_component_compatibility",
    "validate_component_for_task",
    "validate_conditioning",
    "validate_conditioning_v2",
    "validate_fl2va_keyframes",
    "validate_ref2va_references",
    "validate_release_for_task",
    "validate_seed",
    "validate_sigma_request",
    "validate_target_v2",
    "validate_task_partition",
    "validate_target",
    "video_latent_t",
]
