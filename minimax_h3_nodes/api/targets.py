"""Target 节点。"""
from __future__ import annotations
from ._shared import *  # noqa: F403

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
        from ..contracts import H3_MAX_DURATION_SECONDS, H3_MIN_DURATION_SECONDS

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


