"""Data contracts shared by the direct MiniMax-H3 ComfyUI nodes.

This module intentionally has no torch or ComfyUI imports.  Apart from making
the geometry rules easy to test, that keeps an incomplete/broken torch install
from hiding useful model-package validation errors during ComfyUI start-up.

The v0 direct node path implements only T2VA.  FL2VA and Ref2VA are named here
so callers receive a precise error instead of silently running the wrong model
partition or conditioning recipe.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

H3_TASK_T2VA = "t2va"
H3_TASK_FL2VA = "fl2va"
H3_TASK_REF2VA = "ref2va"
H3_TASKS = (H3_TASK_T2VA, H3_TASK_FL2VA, H3_TASK_REF2VA)

H3_T2VA_PARTITION = "fl2va"
H3_REF2VA_PARTITION = "ref2va"

H3_TARGET_SCHEMA = "minimax_h3_target/v1"
H3_CONDITIONING_SCHEMA = "minimax_h3_conditioning/v1"
H3_AV_LATENT_SCHEMA = "minimax_h3_av_latent/v1"
H3_MODEL_SCHEMA = "minimax_h3_model/v1"
H3_TEXT_ENCODER_SCHEMA = "minimax_h3_text_encoder/v1"
H3_VAE_SCHEMA = "minimax_h3_vae_bundle/v1"

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


class H3ContractError(ValueError):
    """Raised when values connected in a workflow violate the H3 contract."""


class H3TaskNotImplementedError(NotImplementedError):
    """Raised for a known H3 task that the first direct node path cannot run."""


def require_t2va(task: str) -> str:
    """Validate a task and fail explicitly for unfinished conditioning paths."""

    normalized = str(task or "").strip().lower()
    if normalized == H3_TASK_T2VA:
        return normalized
    if normalized in (H3_TASK_FL2VA, H3_TASK_REF2VA):
        raise H3TaskNotImplementedError(
            f"MiniMax-H3 Direct v0 目前只实现 T2VA；{normalized.upper()} "
            "仍缺少原仓库的条件 VAE 编码、presentation 和 packed anchor 移植。"
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
    "H3_CONDITIONING_SCHEMA",
    "H3ContractError",
    "H3_DEFAULT_AUDIO_SHIFT",
    "H3_DEFAULT_SIGMA_POINTS",
    "H3_DEFAULT_VIDEO_SHIFT",
    "H3_FPS",
    "H3_IMGVID_COND_TIMESTEP",
    "H3_MODEL_SCHEMA",
    "H3_REF2VA_PARTITION",
    "H3_TASK_FL2VA",
    "H3_TASK_REF2VA",
    "H3_TASK_T2VA",
    "H3_TASKS",
    "H3_TARGET_SCHEMA",
    "H3TaskNotImplementedError",
    "H3_TEXT_ENCODER_SCHEMA",
    "H3_TEXT_WIDTH",
    "H3_T2VA_PARTITION",
    "H3_VAE_SCHEMA",
    "H3_VIDEO_CHANNELS",
    "H3_VIDEO_PATCH_SIZE",
    "H3_VIDEO_ROW_WIDTH",
    "align_frame_count",
    "audio_latent_t",
    "frame_count_from_video_latent_t",
    "make_t2va_conditioning",
    "require_t2va",
    "resolve_spatial_shape",
    "resolve_t2va_target",
    "validate_av_latent",
    "validate_conditioning",
    "validate_seed",
    "validate_sigma_request",
    "validate_target",
    "video_latent_t",
]
