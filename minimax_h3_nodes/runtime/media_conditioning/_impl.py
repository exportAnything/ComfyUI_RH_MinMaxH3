# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 FL2VA/Ref2VA media preparation and condition encoders.

The geometry and media plans in this module are intentionally data-only.  PIL,
PyTorch, torchaudio, and subprocess are imported only by the functions that
need them, which keeps request validation and node discovery lightweight.

The implementation mirrors the released MiniMax H3 contracts:

* the first semantic FL keyframe (including a last-only request) is stretched
  to the target canvas; only the second first+last keyframe is cover-cropped;
* reference images keep their display ratio and use a 2048px short edge;
* reference videos are independently resolved with ``adapt_shape_v1``, then
  materialized as direct-scaled CFR-24 video and truncated without padding;
* a reference video's visual stream uses the prepared file while its audio
  stream always comes from the original, untruncated source;
* visual conditions use the sampled Video VAE recipe and ``[1, 2, 2]``
  patchification; audio conditions use channel-major posterior-mean rows.
"""

from __future__ import annotations

import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..h3_settings import (
    FFMPEG_BIN,
    FFPROBE_BIN,
    H3_REFERENCE_IMAGE_SIZE_MATCH,
    H3_REFERENCE_IMAGE_SIZE_MAX,
    H3_REFERENCE_IMAGE_SIZE_MODES,
)

LOGGER = logging.getLogger(__name__)
H3_CANVAS_MULTIPLE = 32
H3_TARGET_SHORT_EDGE = 768
H3_TARGET_MAX_PIXELS = H3_TARGET_SHORT_EDGE * 1344
H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
H3_REFERENCE_FPS = 24.0
H3_REFERENCE_AUDIO_SAMPLE_RATE = 32_000
H3_VIDEO_SOUNDTRACK_SAMPLE_RATE = 44_100
H3_REFERENCE_AUDIO_CHANNELS = 2
H3_VISUAL_PATCH_SIZE = (1, 2, 2)
H3_CONDITION_ENCODE_SEED = 42
H3_MEDIA_PROCESS_TIMEOUT_SECONDS = 30 * 60.0
_REFERENCE_KINDS = frozenset({"image", "audio", "video", "video_audio"})
_FFMPEG_CACHE: tuple[str, str] | BaseException | None = None


def ensure_ffmpeg_tools(*, required: bool = True) -> tuple[str, str]:
    """Probe ffmpeg/ffprobe on PATH and cache the result once per process."""
    global _FFMPEG_CACHE
    if isinstance(_FFMPEG_CACHE, tuple):
        return _FFMPEG_CACHE
    if isinstance(_FFMPEG_CACHE, BaseException):
        if required: raise _FFMPEG_CACHE
        return ("", "")
    ff, fp = shutil.which(FFMPEG_BIN), shutil.which(FFPROBE_BIN)
    if ff and fp:
        _FFMPEG_CACHE = (ff, fp); return _FFMPEG_CACHE
    err = RuntimeError(
        f"Ref2VA requires {FFMPEG_BIN}/{FFPROBE_BIN} on the system PATH; "
        f"currently {FFMPEG_BIN}={ff!r} {FFPROBE_BIN}={fp!r}"
    )
    _FFMPEG_CACHE = err
    if required: raise err
    LOGGER.warning("%s", err); return ("", "")


def _run_cancellable_process(
    command: Sequence[str],
    *,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = H3_MEDIA_PROCESS_TIMEOUT_SECONDS,
    popen_factory: Callable[..., Any] | None = None,
    poll_interval_seconds: float = 0.1,
    capture_output: bool = False,
    text: bool = False,
) -> Any:
    """Run one media command with cooperative cancellation and a hard bound."""

    import subprocess
    import time

    timeout = _positive_finite(timeout_seconds, "media process timeout_seconds")
    poll_interval = max(0.0, float(poll_interval_seconds))
    argv = [str(argument) for argument in command]
    factory = subprocess.Popen if popen_factory is None else popen_factory
    popen_kwargs: dict[str, Any] = {}
    if capture_output:
        popen_kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if text:
        popen_kwargs["text"] = True
    process = factory(argv, **popen_kwargs)
    started = time.monotonic()

    def terminate() -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                stdout = stderr = None
                if capture_output:
                    stdout, stderr = process.communicate()
                if int(returncode) != 0:
                    raise subprocess.CalledProcessError(
                        int(returncode), argv, output=stdout, stderr=stderr
                    )
                return subprocess.CompletedProcess(
                    argv, int(returncode), stdout=stdout, stderr=stderr
                )
            if interrupt_check is not None:
                interrupt_check()
            if time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired(argv, timeout)
            time.sleep(poll_interval)
    except BaseException:
        terminate()
        raise


def _positive_finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _parse_display_ratio(value: Any) -> float:
    """Parse ffprobe/PyAV ratio spellings, returning zero when unavailable.

    PyAV stringifies ``Fraction(1, 1)`` as ``"1"`` while ffprobe commonly
    reports ``"1:1"``.  Treat both as the same numeric ratio.  Invalid source
    metadata follows the official probe path and falls back at the call site.
    """

    if value is None:
        return 0.0
    raw = str(value).strip()
    if not raw or raw.upper() == "N/A":
        return 0.0
    separator = ":" if ":" in raw else "/" if "/" in raw else None
    try:
        if separator is None:
            ratio = float(raw)
        else:
            numerator, denominator = raw.split(separator, 1)
            ratio = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return ratio if math.isfinite(ratio) and ratio > 0.0 else 0.0


def _canonical_sample_aspect_ratio(value: Any) -> str:
    """Return a stable SAR spelling while preserving non-square ratios."""

    if value is None or not str(value).strip():
        return "1:1"
    raw = str(value).strip()
    ratio = _parse_display_ratio(raw)
    if ratio <= 0.0:
        # Preserve corrupt/non-numeric metadata so the normalized-output
        # validator can fail closed instead of silently blessing it as square.
        return raw
    if abs(ratio - 1.0) <= 1e-12:
        return "1:1"
    if "/" in raw:
        raw = raw.replace("/", ":", 1)
    return raw


def _normalized_rotation_degrees(value: Any) -> float:
    """Mirror ffprobe admission: ignore invalid values and canonicalize turns."""

    try:
        rotation = float(value)
    except (TypeError, ValueError):
        return 0.0
    return rotation % 360.0 if math.isfinite(rotation) else 0.0


def _nearest_multiple(value: float, multiple: int = H3_CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _validate_ratio(width: float, height: float, *, context: str) -> float:
    ratio = width / height
    if ratio < 0.25 or ratio > 4.0:
        raise ValueError(
            f"{context} ratio must be within the inclusive range 1:4 to 4:1, "
            f"got {width:g}x{height:g}"
        )
    return ratio


def cover_crop_plan(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    allow_upscale: bool = False,
) -> dict[str, Any]:
    """Return the release's deterministic LANCZOS cover-crop plan."""

    source_width = _positive_int(source_width, "source_width")
    source_height = _positive_int(source_height, "source_height")
    target_width = _positive_int(target_width, "target_width")
    target_height = _positive_int(target_height, "target_height")
    scale = max(
        target_width / float(source_width), target_height / float(source_height)
    )
    if scale > 1.0 and not allow_upscale:
        raise ValueError(
            "cover crop would upscale the source; "
            f"source={source_width}x{source_height}, "
            f"target={target_width}x{target_height}"
        )
    resized_width = max(target_width, int(round(source_width * scale)))
    resized_height = max(target_height, int(round(source_height * scale)))
    left = max(0, (resized_width - target_width) // 2)
    top = max(0, (resized_height - target_height) // 2)
    return {
        "mode": "cover_crop",
        "resample": "lanczos",
        "allow_upscale": bool(allow_upscale),
        "scale": scale,
        "resized_size": (resized_width, resized_height),
        "crop_box": (left, top, left + target_width, top + target_height),
    }


def prepare_fl_keyframe_canvas(
    image: Any,
    *,
    target_width: int,
    target_height: int,
    keyframe_ordinal: int = 0,
) -> Any:
    """Prepare one FL keyframe using official anchor/follower semantics.

    ``keyframe_ordinal=0`` is the first *semantic* keyframe, so a last-only
    request also stretches its sole image.  Ordinal 1 is legal only for the
    first+last extension and uses cover-crop with upscaling enabled.
    """

    from PIL import Image

    target_width = _positive_int(target_width, "target_width")
    target_height = _positive_int(target_height, "target_height")
    if keyframe_ordinal not in (0, 1):
        raise ValueError("FL keyframe_ordinal must be 0 or 1")
    prepared = image.convert("RGB")
    target_size = (target_width, target_height)
    if prepared.size == target_size:
        return prepared
    if keyframe_ordinal == 0:
        return prepared.resize(target_size, Image.Resampling.LANCZOS)
    plan = cover_crop_plan(
        source_width=prepared.size[0],
        source_height=prepared.size[1],
        target_width=target_width,
        target_height=target_height,
        allow_upscale=True,
    )
    return prepared.resize(
        plan["resized_size"], Image.Resampling.LANCZOS
    ).crop(plan["crop_box"])


def _validate_reference_image_size_mode(size_mode: Any) -> str:
    mode = str(size_mode or "").strip().lower()
    if mode not in H3_REFERENCE_IMAGE_SIZE_MODES:
        raise ValueError(
            "reference image size_mode must be one of "
            f"{H3_REFERENCE_IMAGE_SIZE_MODES}, got {size_mode!r}"
        )
    return mode


def resolve_reference_image_shape(
    *,
    width: int | float,
    height: int | float,
    size_mode: str = H3_REFERENCE_IMAGE_SIZE_MAX,
    canvas_width: int | float | None = None,
    canvas_height: int | float | None = None,
) -> dict[str, Any]:
    """Resolve a Ref2VA image's own canvas.

    ``max`` keeps the reference pipeline's independent 2048px short edge (best
    identity fidelity).  ``match`` scales the reference down—never up—to the
    generation canvas' pixel area, keeping its aspect ratio: reference tokens
    ride through every sampling step, so the token count is what the extra
    resolution actually costs.
    """

    mode = _validate_reference_image_size_mode(size_mode)
    source_width = _positive_finite(width, "reference image width")
    source_height = _positive_finite(height, "reference image height")
    _validate_ratio(source_width, source_height, context="reference image")
    if mode == H3_REFERENCE_IMAGE_SIZE_MAX:
        scale = H3_REFERENCE_IMAGE_SHORT_EDGE / min(source_width, source_height)
        policy, allow_upscale = "reference_image_short_edge_v1", True
        resolved_size_mode = "short_edge"
    else:
        canvas_pixels = _positive_finite(
            canvas_width, "reference image canvas width"
        ) * _positive_finite(canvas_height, "reference image canvas height")
        scale = min(1.0, math.sqrt(canvas_pixels / (source_width * source_height)))
        policy, allow_upscale = "reference_image_match_area_v1", False
        resolved_size_mode = "area"
    target_width = _nearest_multiple(source_width * scale)
    target_height = _nearest_multiple(source_height * scale)
    shape = {
        "geometry": "reference_image_resolved",
        "shape_policy_version": policy,
        "base_short_edge": H3_REFERENCE_IMAGE_SHORT_EDGE,
        "effective_short_edge": min(target_width, target_height),
        "size_mode": resolved_size_mode,
        "reference_size_mode": mode,
        "multiple": H3_CANVAS_MULTIPLE,
        "rounding": "nearest",
        "allow_upscale": allow_upscale,
        "width": target_width,
        "height": target_height,
    }
    if mode == H3_REFERENCE_IMAGE_SIZE_MATCH:
        shape["max_pixels"] = int(canvas_pixels)
    return shape


def prepare_reference_image(
    image: Any,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    size_mode: str = H3_REFERENCE_IMAGE_SIZE_MAX,
    canvas_width: int | float | None = None,
    canvas_height: int | float | None = None,
) -> Any:
    """EXIF-normalize, RGB-convert, and LANCZOS-resize a Ref2VA image."""

    from PIL import Image, ImageOps

    prepared = ImageOps.exif_transpose(image).convert("RGB")
    if target_width is None and target_height is None:
        shape = resolve_reference_image_shape(
            width=prepared.size[0],
            height=prepared.size[1],
            size_mode=size_mode,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        target_width, target_height = shape["width"], shape["height"]
    elif target_width is None or target_height is None:
        raise ValueError("target_width and target_height must be supplied together")
    target_width = _positive_int(target_width, "target_width")
    target_height = _positive_int(target_height, "target_height")
    if target_width % H3_CANVAS_MULTIPLE or target_height % H3_CANVAS_MULTIPLE:
        raise ValueError(
            f"reference image dimensions must be aligned to {H3_CANVAS_MULTIPLE}"
        )
    if prepared.size == (target_width, target_height):
        return prepared
    return prepared.resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )


def resolve_reference_video_shape(
    *, width: int | float, height: int | float
) -> dict[str, Any]:
    """Resolve a reference video's independent ``adapt_shape_v1`` canvas."""

    source_width = _positive_finite(width, "reference video width")
    source_height = _positive_finite(height, "reference video height")
    ratio = _validate_ratio(source_width, source_height, context="reference video")
    if ratio >= 1.0:
        nominal_width = H3_TARGET_SHORT_EDGE * ratio
        nominal_height = float(H3_TARGET_SHORT_EDGE)
    else:
        nominal_width = float(H3_TARGET_SHORT_EDGE)
        nominal_height = H3_TARGET_SHORT_EDGE / ratio
    nominal_area = nominal_width * nominal_height
    size_mode = "short_edge"
    if nominal_area > H3_TARGET_MAX_PIXELS:
        size_mode = "area"
        scale = math.sqrt(H3_TARGET_MAX_PIXELS / nominal_area)
        nominal_width *= scale
        nominal_height *= scale
    resolved_width = _nearest_multiple(nominal_width)
    resolved_height = _nearest_multiple(nominal_height)
    return {
        "geometry": "resolved_v2",
        "shape_policy_version": "adapt_shape_v1",
        "base_short_edge": H3_TARGET_SHORT_EDGE,
        "effective_short_edge": min(resolved_width, resolved_height),
        "size_mode": size_mode,
        "max_pixels": H3_TARGET_MAX_PIXELS,
        "multiple": H3_CANVAS_MULTIPLE,
        "rounding": "nearest",
        "width": resolved_width,
        "height": resolved_height,
    }


def reference_video_display_geometry(
    *,
    coded_width: int | float,
    coded_height: int | float,
    sample_aspect_ratio: Any = "1:1",
    display_aspect_ratio: Any = None,
    rotation_degrees: int | float = 0.0,
) -> tuple[float, float]:
    """Resolve square-pixel display geometry from coded stream metadata.

    This follows the official H3 probe contract: a valid stream DAR has
    priority, otherwise coded width is corrected by SAR, and display rotation
    is applied last.  Keeping the result as floats avoids changing an aspect
    ratio before ``adapt_shape_v1`` rounds to its 32-pixel grid.
    """

    width = _positive_finite(coded_width, "coded video width")
    height = _positive_finite(coded_height, "coded video height")
    sar = _parse_display_ratio(sample_aspect_ratio) or 1.0
    dar = _parse_display_ratio(display_aspect_ratio)
    physical_width = dar * height if dar > 0.0 else width * sar
    rotation = _normalized_rotation_degrees(rotation_degrees)
    quarter_turns = round(rotation / 90.0)
    if abs(rotation - quarter_turns * 90.0) <= 1e-6:
        return (height, physical_width) if quarter_turns % 2 else (physical_width, height)
    radians = math.radians(rotation)
    cosine = abs(math.cos(radians))
    sine = abs(math.sin(radians))
    return (
        physical_width * cosine + height * sine,
        physical_width * sine + height * cosine,
    )


@dataclass(frozen=True)
class ReferenceVideoMetadata:
    """Probe facts needed to prepare a Ref2VA visual reference."""

    width: float
    height: float
    fps: float
    frame_count: int
    has_audio: bool
    sample_aspect_ratio: str = "1:1"
    rotation_degrees: float = 0.0
    display_aspect_ratio: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReferenceVideoMetadata":
        display_width = value.get("display_width")
        display_height = value.get("display_height")
        if (display_width is None) != (display_height is None):
            raise ValueError(
                "display_width and display_height must be supplied together"
            )
        sar = value.get("sample_aspect_ratio") or "1:1"
        rotation = _normalized_rotation_degrees(
            value.get("rotation_degrees") or 0.0
        )
        if display_width is None:
            coded_width = value.get("coded_width", value.get("width"))
            coded_height = value.get("coded_height", value.get("height"))
            display_width, display_height = reference_video_display_geometry(
                coded_width=coded_width,
                coded_height=coded_height,
                sample_aspect_ratio=sar,
                display_aspect_ratio=value.get("display_aspect_ratio"),
                rotation_degrees=rotation,
            )
        else:
            display_width = _positive_finite(display_width, "video display width")
            display_height = _positive_finite(display_height, "video display height")
        return cls(
            width=float(display_width),
            height=float(display_height),
            fps=_positive_finite(value.get("fps"), "video fps"),
            frame_count=_positive_int(value.get("frame_count"), "video frame_count"),
            has_audio=bool(value.get("has_audio", False)),
            sample_aspect_ratio=_canonical_sample_aspect_ratio(sar),
            rotation_degrees=rotation,
            display_aspect_ratio=float(display_width) / float(display_height),
        )

    @classmethod
    def from_coded(
        cls,
        *,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
        has_audio: bool,
        sample_aspect_ratio: Any = "1:1",
        display_aspect_ratio: Any = None,
        rotation_degrees: float = 0.0,
    ) -> "ReferenceVideoMetadata":
        display_width, display_height = reference_video_display_geometry(
            coded_width=width,
            coded_height=height,
            sample_aspect_ratio=sample_aspect_ratio,
            display_aspect_ratio=display_aspect_ratio,
            rotation_degrees=rotation_degrees,
        )
        return cls(
            width=display_width,
            height=display_height,
            fps=_positive_finite(fps, "video fps"),
            frame_count=_positive_int(frame_count, "video frame_count"),
            has_audio=bool(has_audio),
            sample_aspect_ratio=_canonical_sample_aspect_ratio(
                sample_aspect_ratio
            ),
            rotation_degrees=_normalized_rotation_degrees(rotation_degrees),
            display_aspect_ratio=display_width / display_height,
        )

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ReferenceVideoPlan:
    """Data-only normalization/truncation plan for one reference video."""

    kind: str
    source: ReferenceVideoMetadata
    width: int
    height: int
    target_frame_count: int
    fps: float = H3_REFERENCE_FPS
    visual_source: str = "prepared_cfr24"
    soundtrack_source: str = "original_untruncated"
    scale_mode: str = "direct_lanczos"
    crop: bool = False
    pad_if_short: bool = False

    @property
    def input_has_audio(self) -> bool:
        return self.source.has_audio

    @property
    def filtergraph(self) -> str:
        return (
            f"fps={self.fps:g},scale={self.width}:{self.height}:flags=lanczos,setsar=1"
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["input_has_audio"] = self.input_has_audio
        result["filtergraph"] = self.filtergraph
        return result


def build_reference_video_plan(
    metadata: ReferenceVideoMetadata | Mapping[str, Any],
    *,
    target_frame_count: int,
    kind: str = "video",
    target_width: int | None = None,
    target_height: int | None = None,
) -> ReferenceVideoPlan:
    """Build the official CFR-24/no-crop/leading-truncation plan."""

    if isinstance(metadata, Mapping):
        metadata = ReferenceVideoMetadata.from_mapping(metadata)
    if not isinstance(metadata, ReferenceVideoMetadata):
        raise TypeError("metadata must be ReferenceVideoMetadata or a mapping")
    if kind not in {"video", "video_audio"}:
        raise ValueError("reference video kind must be 'video' or 'video_audio'")
    if kind == "video_audio" and not metadata.has_audio:
        raise ValueError("video_audio reference requires an audio stream")
    if target_width is None and target_height is None:
        shape = resolve_reference_video_shape(
            width=metadata.width, height=metadata.height
        )
        target_width, target_height = shape["width"], shape["height"]
    elif target_width is None or target_height is None:
        raise ValueError("target_width and target_height must be supplied together")
    target_width = _positive_int(target_width, "target_width")
    target_height = _positive_int(target_height, "target_height")
    if target_width % H3_CANVAS_MULTIPLE or target_height % H3_CANVAS_MULTIPLE:
        raise ValueError(
            f"reference video dimensions must be aligned to {H3_CANVAS_MULTIPLE}"
        )
    return ReferenceVideoPlan(
        kind=kind,
        source=metadata,
        width=target_width,
        height=target_height,
        target_frame_count=_positive_int(target_frame_count, "target_frame_count"),
    )


def reference_video_normalize_command(
    source_path: str | Path,
    output_path: str | Path,
    plan: ReferenceVideoPlan,
) -> list[str]:
    """Build the ffmpeg command that materializes the shared visual source."""

    return [
        FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(source_path),
        "-map", "0:v:0", "-an", "-vf", plan.filtergraph,
        "-metadata:s:v:0", "rotate=0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output_path),
    ]


def reference_video_truncate_command(
    prepared_path: str | Path,
    output_path: str | Path,
    plan: ReferenceVideoPlan,
) -> list[str]:
    """Build the separate leading-frame truncation command (never pads)."""

    return [
        FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(prepared_path),
        "-map", "0:v:0", "-an", "-frames:v", str(plan.target_frame_count),
        "-metadata:s:v:0",
        "rotate=0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


def validate_prepared_reference_video(
    metadata: ReferenceVideoMetadata | Mapping[str, Any],
    plan: ReferenceVideoPlan,
    *,
    expected_frame_count: int | None = None,
) -> ReferenceVideoMetadata:
    """Fail closed if a materialized stream misses the visual contract."""

    raw_rotation = (
        metadata.get("rotation_degrees", 0.0)
        if isinstance(metadata, Mapping)
        else metadata.rotation_degrees
    )
    if raw_rotation in (None, ""):
        raw_rotation = 0.0
    try:
        parsed_rotation = float(raw_rotation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "prepared reference video has invalid rotation metadata"
        ) from exc
    if not math.isfinite(parsed_rotation):
        raise ValueError("prepared reference video has invalid rotation metadata")
    if isinstance(metadata, Mapping):
        metadata = ReferenceVideoMetadata.from_mapping(metadata)
    if (metadata.width, metadata.height) != (plan.width, plan.height):
        raise ValueError(
            "prepared reference video has unexpected geometry: "
            f"expected={plan.width}x{plan.height}, "
            f"actual={metadata.width}x{metadata.height}"
        )
    if abs(metadata.fps - plan.fps) >= 1e-6:
        raise ValueError("prepared reference video is not CFR-24")
    if abs(_parse_display_ratio(metadata.sample_aspect_ratio) - 1.0) > 1e-12:
        raise ValueError("prepared reference video did not normalize SAR to 1:1")
    normalized_rotation = parsed_rotation % 360.0
    if min(normalized_rotation, 360.0 - normalized_rotation) > 1e-6:
        raise ValueError("prepared reference video retained rotation metadata")
    if expected_frame_count is not None and metadata.frame_count != expected_frame_count:
        raise ValueError(
            "prepared reference video has unexpected frame count: "
            f"expected={expected_frame_count}, actual={metadata.frame_count}"
        )
    return metadata


def execute_reference_video_plan(
    source_path: str | Path,
    plan: ReferenceVideoPlan,
    *,
    workdir: str | Path,
    probe: Callable[[str], ReferenceVideoMetadata | Mapping[str, Any]],
    runner: Callable[..., Any] | None = None,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = H3_MEDIA_PROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Materialize a plan with injectable process/probe boundaries.

    The returned ``original_path`` is deliberately retained for soundtrack
    extraction.  Qwen and the visual VAE must use ``prepared_path``.
    """
    if runner is None:
        ensure_ffmpeg_tools(required=True)

        def default_runner(command: Sequence[str], **_kwargs: Any) -> None:
            _run_cancellable_process(
                command,
                interrupt_check=interrupt_check,
                timeout_seconds=timeout_seconds,
            )

        runner = default_runner
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    normalized = workdir / f"refvid_{plan.width}x{plan.height}.mp4"
    runner(reference_video_normalize_command(source_path, normalized, plan), check=True)
    current = normalized
    current_meta = validate_prepared_reference_video(probe(str(current)), plan)
    if current_meta.frame_count > plan.target_frame_count:
        truncated = workdir / f"refvid_frames{plan.target_frame_count}.mp4"
        runner(reference_video_truncate_command(current, truncated, plan), check=True)
        current = truncated
        current_meta = validate_prepared_reference_video(
            probe(str(current)), plan, expected_frame_count=plan.target_frame_count
        )
    return {
        "kind": plan.kind,
        "prepared_path": str(current),
        "original_path": str(source_path),
        "audio_source_path": str(source_path) if plan.input_has_audio else None,
        "input_has_audio": plan.input_has_audio,
        "target_frame_count": plan.target_frame_count,
        "frame_count": current_meta.frame_count,
        "width": plan.width,
        "height": plan.height,
        "fps": plan.fps,
    }


def decode_reference_video_samples(
    prepared_path: str | Path,
    sample_indices: Sequence[int],
    *,
    decoder: Callable[[str, Sequence[int]], Any] | None = None,
    interrupt_check: Callable[[], None] | None = None,
) -> Any:
    """Decode only Qwen-selected frames from the prepared CFR stream.

    Timestamp/index planning deliberately lives in ``runtime.presentation``;
    callers pass its indices here.  An injectable decoder keeps this boundary
    hermetic in tests, while the default PyAV path returns RGB uint8
    ``[sample,H,W,3]``.
    """

    indices = [int(index) for index in sample_indices]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("sample_indices must contain non-negative frame indices")
    if indices != sorted(set(indices)):
        raise ValueError("sample_indices must be strictly increasing and unique")
    if decoder is not None:
        return decoder(str(prepared_path), indices)

    import av
    import numpy as np

    wanted = set(indices)
    selected: dict[int, Any] = {}
    if interrupt_check is not None:
        interrupt_check()
    with av.open(str(prepared_path), mode="r") as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if interrupt_check is not None:
                interrupt_check()
            if frame_index in wanted:
                selected[frame_index] = frame.to_ndarray(format="rgb24")
                if len(selected) == len(indices):
                    break
    missing = [index for index in indices if index not in selected]
    if missing:
        raise ValueError(
            "prepared reference video ended before Qwen sample indices: "
            f"{missing[:8]}"
        )
    return np.stack([selected[index] for index in indices], axis=0)


@dataclass(frozen=True)
class ReferenceAudioPlan:
    """Canonical audio route for a pure-audio or video reference."""

    kind: str
    enabled: bool
    required: bool
    input_has_audio: bool
    source_channels: int | None
    source_sample_rate: int | None
    decode_sample_rate: int | None
    target_channels: int = H3_REFERENCE_AUDIO_CHANNELS
    target_sample_rate: int = H3_REFERENCE_AUDIO_SAMPLE_RATE
    soundtrack_source: str = "original_untruncated"
    resample_count: int = 0
    channel_policy: str = "stereo_passthrough"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_reference_audio_plan(
    *,
    kind: str,
    input_has_audio: bool = True,
    source_channels: int | None = None,
    source_sample_rate: int | None = None,
) -> ReferenceAudioPlan:
    """Build the release audio route, including its single-resample rule."""

    if kind not in {"audio", "video", "video_audio"}:
        raise ValueError("reference audio kind must be audio, video, or video_audio")
    required = kind in {"audio", "video_audio"}
    if required and not input_has_audio:
        raise ValueError(f"{kind} reference requires an audio stream")
    if not input_has_audio:
        return ReferenceAudioPlan(
            kind=kind,
            enabled=False,
            required=False,
            input_has_audio=False,
            source_channels=source_channels,
            source_sample_rate=source_sample_rate,
            decode_sample_rate=None,
        )
    if source_channels is not None:
        source_channels = _positive_int(source_channels, "source_channels")
    if source_sample_rate is not None:
        source_sample_rate = _positive_int(source_sample_rate, "source_sample_rate")
    decode_rate = (
        H3_VIDEO_SOUNDTRACK_SAMPLE_RATE
        if kind in {"video", "video_audio"}
        else source_sample_rate
    )
    channel_policy = "stereo_passthrough"
    if source_channels == 1:
        channel_policy = "duplicate_mono"
    elif source_channels is not None and source_channels > 2:
        channel_policy = "downmix_stereo"
    resample_count = int(
        decode_rate is not None and decode_rate != H3_REFERENCE_AUDIO_SAMPLE_RATE
    )
    return ReferenceAudioPlan(
        kind=kind,
        enabled=True,
        required=required,
        input_has_audio=True,
        source_channels=source_channels,
        source_sample_rate=source_sample_rate,
        decode_sample_rate=decode_rate,
        resample_count=resample_count,
        channel_policy=channel_policy,
    )


def reference_audio_extract_command(
    source_path: str | Path,
    output_path: str | Path,
    plan: ReferenceAudioPlan,
) -> list[str]:
    """Build the lossless stereo-normalization/extraction ffmpeg command."""

    if not plan.enabled:
        raise ValueError("silent video references have no audio extraction command")
    command = [FFMPEG_BIN, "-y", "-loglevel", "error", "-i", str(source_path)]
    if plan.kind == "audio":
        command += ["-map", "0:a:0"]
        if plan.channel_policy == "duplicate_mono":
            command += ["-af", "pan=stereo|c0=c0|c1=c0", "-c:a", "flac"]
        elif plan.channel_policy == "downmix_stereo":
            command += ["-c:a", "flac", "-ac", "2"]
        else:
            command += ["-c:a", "flac"]
    else:
        # This always points at the ORIGINAL video.  The prepared/truncated
        # visual file must never be substituted here.
        command += [
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(H3_VIDEO_SOUNDTRACK_SAMPLE_RATE),
            "-f",
            "wav",
        ]
    command.append(str(output_path))
    return command


def execute_reference_audio_plan(
    source_path: str | Path,
    plan: ReferenceAudioPlan,
    *,
    workdir: str | Path,
    runner: Callable[..., Any] | None = None,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = H3_MEDIA_PROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Materialize canonical stereo audio, or a zero-row silent-video marker."""

    if not plan.enabled:
        return {
            "kind": plan.kind,
            "audio_path": None,
            "source_path": str(source_path),
            "sample_rate": None,
            "input_has_audio": False,
            "audio_rows": None,
            "ref_audio_t": 0,
        }
    if runner is None:
        ensure_ffmpeg_tools(required=True)

        def default_runner(command: Sequence[str], **_kwargs: Any) -> None:
            _run_cancellable_process(
                command,
                interrupt_check=interrupt_check,
                timeout_seconds=timeout_seconds,
            )

        runner = default_runner
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    extension = ".flac" if plan.kind == "audio" else ".wav"
    output = workdir / f"reference_audio{extension}"
    runner(reference_audio_extract_command(source_path, output, plan), check=True)
    return {
        "kind": plan.kind,
        "audio_path": str(output),
        "source_path": str(source_path),
        "sample_rate": plan.decode_sample_rate,
        "input_has_audio": True,
    }


def patchify_video_condition_rows(
    latent: Any,
    *,
    patch_size: Sequence[int] = H3_VISUAL_PATCH_SIZE,
) -> Any:
    """Patchify ``[B,C,T,H,W]`` to H3 visual condition rows."""

    import torch

    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
        raise ValueError("visual latent must be [B,C,T,H,W]")
    if len(patch_size) != 3:
        raise ValueError("patch_size must contain three values")
    pt, ph, pw = (_positive_int(value, "patch_size") for value in patch_size)
    batch, channels, full_t, full_h, full_w = map(int, latent.shape)
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            f"visual latent shape {tuple(latent.shape)} is not divisible by "
            f"patch size {(pt, ph, pw)}"
        )
    time, height, width = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(
        batch, channels, time, pt, height, ph, width, pw
    )
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(
        batch * time * height * width, channels * pt * ph * pw
    ).contiguous()


def pack_audio_condition_rows(latent: Any) -> Any:
    """Pack normalized ``[channels, latent_dim, T]`` channel-major rows."""

    import torch

    if not isinstance(latent, torch.Tensor) or latent.ndim != 3:
        raise ValueError("audio latent must be [channels, latent_dim, T]")
    channels, latent_dim, steps = map(int, latent.shape)
    return latent.permute(0, 2, 1).reshape(
        channels * steps, latent_dim
    ).contiguous()


def _condition_block_base(
    *,
    condition_index: int,
    kind: str,
    prepared_media: Any,
    material_fingerprint: str | None,
    semantic_frame_index: int | None,
    resolved_frame_index: int | None,
) -> dict[str, Any]:
    if isinstance(condition_index, bool) or int(condition_index) != condition_index:
        raise ValueError("condition_index must be an integer")
    if int(condition_index) < 0:
        raise ValueError("condition_index must be non-negative")
    kind = str(kind).strip().lower()
    if kind not in _REFERENCE_KINDS:
        raise ValueError(f"unsupported condition kind {kind!r}")
    is_reference = kind in {"audio", "video", "video_audio"} or (
        kind == "image" and semantic_frame_index is None
    )
    if is_reference and material_fingerprint is None:
        raise ValueError("Ref2VA condition block requires material_fingerprint")
    if material_fingerprint is not None and (
        not isinstance(material_fingerprint, str)
        or len(material_fingerprint) != 64
        or any(
            character not in "0123456789abcdef"
            for character in material_fingerprint
        )
    ):
        raise ValueError("material_fingerprint must be 64 lowercase hex characters")
    block = {
        "condition_index": int(condition_index),
        "kind": kind,
        "visual_rows": None,
        "audio_rows": None,
        "latent_t": None,
        "latent_h": None,
        "latent_w": None,
        "ref_audio_t": 0,
        "semantic_frame_index": semantic_frame_index,
        "resolved_frame_index": resolved_frame_index,
        "prepared_media": prepared_media,
    }
    if material_fingerprint is not None:
        block["material_fingerprint"] = material_fingerprint
    return block


def encode_visual_condition_rows(
    video_vae: Any,
    pixels: Any,
    *,
    condition_index: int,
    kind: str,
    process_image: bool | None = None,
    prepared_media: Any = None,
    material_fingerprint: str | None = None,
    semantic_frame_index: int | None = None,
    resolved_frame_index: int | None = None,
    seed: int = H3_CONDITION_ENCODE_SEED,
    target_fingerprint: str = "",
    vae_fingerprint: str = "",
    shape_policy: str = "",
) -> dict[str, Any]:
    """Encode one image/video condition and return a unified block entry.

    ``shape_policy`` describes which sizing policy prepared the incoming pixels
    (for example, match/max for reference images). It must be part of the cache key:
    switching policies for the same asset and target produces different pixels and
    must not reuse rows from the previous policy.
    """

    import torch
    from ..encode_cache import get_encode_cache, vae_rows_cache_key
    from ..h3_settings import OPT_ENCODE_CACHE

    kind = str(kind).strip().lower()
    if kind not in {"image", "video", "video_audio"}:
        raise ValueError("visual condition kind must be image, video, or video_audio")
    if process_image is None:
        process_image = kind == "image"
    cache_fp = material_fingerprint or f"fl|{target_fingerprint}|{resolved_frame_index}|{semantic_frame_index}"
    policy_tag = f"|{str(shape_policy)}" if shape_policy else ""
    cache_key = vae_rows_cache_key(
        material_fp=str(cache_fp), target_fp=str(target_fingerprint or ""),
        vae_fp=str(vae_fingerprint or ""),
        kind=f"visual|{kind}|{int(bool(process_image))}|{int(seed)}{policy_tag}",
    )
    if OPT_ENCODE_CACHE:
        hit = get_encode_cache().get(cache_key)
        if isinstance(hit, dict) and "visual_rows" in hit:
            block = dict(hit)
            block["condition_index"] = int(condition_index)
            if material_fingerprint is not None:
                block["material_fingerprint"] = material_fingerprint
            if prepared_media is not None:
                block["prepared_media"] = prepared_media
            return block
    latent = video_vae.encode(
        pixels,
        process_image=bool(process_image),
        seed=int(seed),
        use_fp16_latent=True,
        parallel_tiling=False,
    )
    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
        raise ValueError("Video VAE condition encode must return [B,C,T,H,W]")
    if int(latent.shape[0]) != 1:
        raise ValueError("MiniMax H3 condition encoding supports batch=1")
    rows = patchify_video_condition_rows(latent).to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    if int(rows.shape[1]) != 96:
        raise ValueError(
            f"MiniMax H3 visual condition rows must be 96-wide, got {rows.shape[1]}"
        )
    block = _condition_block_base(
        condition_index=condition_index,
        kind=kind,
        prepared_media=prepared_media,
        material_fingerprint=material_fingerprint,
        semantic_frame_index=semantic_frame_index,
        resolved_frame_index=resolved_frame_index,
    )
    block.update(
        {
            "visual_rows": rows,
            "latent_t": int(latent.shape[2]),
            "latent_h": int(latent.shape[3]),
            "latent_w": int(latent.shape[4]),
        }
    )
    if OPT_ENCODE_CACHE:
        get_encode_cache().put(cache_key, {k: v for k, v in block.items() if k != "prepared_media"})
    return block


def encode_audio_condition_rows(
    audio_vae: Any,
    waveform: Any,
    *,
    condition_index: int,
    kind: str,
    sample_rate: int | None = None,
    prepared_media: Any = None,
    material_fingerprint: str | None = None,
    target_fingerprint: str = "",
    vae_fingerprint: str = "",
) -> dict[str, Any]:
    """Encode one audio reference as posterior-mean channel-major rows."""

    import torch
    from ..encode_cache import get_encode_cache, vae_rows_cache_key
    from ..h3_settings import OPT_ENCODE_CACHE

    kind = str(kind).strip().lower()
    if kind not in {"audio", "video", "video_audio"}:
        raise ValueError("audio condition kind must be audio, video, or video_audio")
    cache_fp = material_fingerprint or f"audio|{target_fingerprint}|{condition_index}"
    cache_key = vae_rows_cache_key(
        material_fp=str(cache_fp), target_fp=str(target_fingerprint or ""),
        vae_fp=str(vae_fingerprint or ""), kind=f"audio|{kind}|{int(sample_rate or 0)}",
    )
    if OPT_ENCODE_CACHE:
        hit = get_encode_cache().get(cache_key)
        if isinstance(hit, dict) and "audio_rows" in hit:
            block = dict(hit)
            block["condition_index"] = int(condition_index)
            if material_fingerprint is not None:
                block["material_fingerprint"] = material_fingerprint
            if prepared_media is not None:
                block["prepared_media"] = prepared_media
            return block
    latent = audio_vae.encode(waveform, sample_rate=sample_rate)
    if not isinstance(latent, torch.Tensor) or latent.ndim != 3:
        raise ValueError("Audio VAE condition encode must return [2,32,T]")
    if tuple(map(int, latent.shape[:2])) != (2, 32):
        raise ValueError(
            "MiniMax H3 audio condition latent must start with [2,32], got "
            f"{tuple(latent.shape)}"
        )
    rows = pack_audio_condition_rows(latent).to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    block = _condition_block_base(
        condition_index=condition_index,
        kind=kind,
        prepared_media=prepared_media,
        material_fingerprint=material_fingerprint,
        semantic_frame_index=None,
        resolved_frame_index=None,
    )
    block.update({"audio_rows": rows, "ref_audio_t": int(latent.shape[2])})
    if OPT_ENCODE_CACHE:
        get_encode_cache().put(cache_key, {k: v for k, v in block.items() if k != "prepared_media"})
    return block


def merge_condition_blocks(
    visual: Mapping[str, Any] | None,
    audio: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge visual/audio entries for the same ordered reference material."""

    if visual is None and audio is None:
        raise ValueError("at least one condition block is required")
    first = visual if visual is not None else audio
    assert first is not None
    result = dict(first)
    if visual is not None and audio is not None:
        for field in ("condition_index", "kind", "material_fingerprint"):
            if visual.get(field) != audio.get(field):
                raise ValueError(f"cannot merge condition blocks with different {field}")
        result.update(
            {
                "audio_rows": audio.get("audio_rows"),
                "ref_audio_t": int(audio.get("ref_audio_t") or 0),
            }
        )
        if result.get("prepared_media") is None:
            result["prepared_media"] = audio.get("prepared_media")
    return result


def order_condition_blocks(
    blocks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate uniqueness and restore canonical request/condition order."""

    ordered = sorted(
        (dict(block) for block in blocks),
        key=lambda item: int(item["condition_index"]),
    )
    indices = [int(block["condition_index"]) for block in ordered]
    if any(index < 0 for index in indices):
        raise ValueError("condition_index must be non-negative")
    if len(indices) != len(set(indices)):
        raise ValueError("condition_index values must be unique")
    return ordered


__all__ = [
    "H3_CANVAS_MULTIPLE",
    "H3_CONDITION_ENCODE_SEED",
    "H3_REFERENCE_AUDIO_CHANNELS",
    "H3_REFERENCE_AUDIO_SAMPLE_RATE",
    "H3_REFERENCE_FPS",
    "H3_REFERENCE_IMAGE_SHORT_EDGE",
    "H3_REFERENCE_IMAGE_SIZE_MATCH",
    "H3_REFERENCE_IMAGE_SIZE_MAX",
    "H3_REFERENCE_IMAGE_SIZE_MODES",
    "H3_TARGET_MAX_PIXELS",
    "H3_TARGET_SHORT_EDGE",
    "H3_VIDEO_SOUNDTRACK_SAMPLE_RATE",
    "H3_VISUAL_PATCH_SIZE",
    "ReferenceAudioPlan",
    "ReferenceVideoMetadata",
    "ReferenceVideoPlan",
    "build_reference_audio_plan",
    "build_reference_video_plan",
    "cover_crop_plan",
    "decode_reference_video_samples",
    "encode_audio_condition_rows",
    "encode_visual_condition_rows",
    "ensure_ffmpeg_tools",
    "execute_reference_audio_plan",
    "execute_reference_video_plan",
    "merge_condition_blocks",
    "order_condition_blocks",
    "pack_audio_condition_rows",
    "patchify_video_condition_rows",
    "prepare_fl_keyframe_canvas",
    "prepare_reference_image",
    "reference_audio_extract_command",
    "reference_video_normalize_command",
    "reference_video_display_geometry",
    "reference_video_truncate_command",
    "resolve_reference_image_shape",
    "resolve_reference_video_shape",
    "validate_prepared_reference_video",
]
