"""MiniMax-H3 packed-token layouts and latent transforms.

This is a direct, single-device adaptation of the model's native T2VA, FL2VA,
and Ref2VA layouts.  It keeps the model's fp64 three-axis position grid and
64-row sequence alignment; neither field is interchangeable with a regular
Comfy latent mask.

The module deliberately imports torch lazily.  ComfyUI can therefore report
model-package errors even when its torch environment is incomplete, and the
lightweight descriptor objects can safely cross node boundaries on CPU.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PACKED_ALIGNMENT = 64
TEXT_ID = -5
IMGVID_COND_ID = -11
AUDIO_REF_COND_ID = -17
AUDIO_FIRST_ID = -15
AUDIO_ID = -14
VIDEO_FIRST_ID = -3
VIDEO_ID = -2
VIDEO_LAST_ID = -4
PAD_ID = -1

# Source-compatible aliases make comparison tests and staged ports explicit.
MINIMAX_H3_TEXT_ID = TEXT_ID
MINIMAX_H3_IMGVID_COND_ID = IMGVID_COND_ID
MINIMAX_H3_AUDIO_REF_COND_ID = AUDIO_REF_COND_ID
MINIMAX_H3_AUDIO_FIRST_ID = AUDIO_FIRST_ID
MINIMAX_H3_AUDIO_ID = AUDIO_ID
MINIMAX_H3_VIDEO_FIRST_ID = VIDEO_FIRST_ID
MINIMAX_H3_VIDEO_ID = VIDEO_ID
MINIMAX_H3_VIDEO_LAST_ID = VIDEO_LAST_ID
MINIMAX_H3_PAD_ID = PAD_ID

VIDEO_CHANNELS = 24
AUDIO_LATENT_DIM = 32
PATCH_SIZE = (1, 2, 2)
INTERP = 32
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
FL2VA_KEYFRAME_SIGNATURES = ((0,), (-1,), (0, -1))

@dataclass(frozen=True, slots=True)
class H3ConditionBlockDescriptor:
    """One ordered FL2VA/Ref2VA condition and its encoded latent rows.

    ``visual_rows`` and ``audio_rows`` are intentionally typed as ``Any`` so
    importing this module does not import torch.  High-level builders validate
    them as rank-2 tensors with widths 96 and 32 respectively.

    FL2VA uses ``kind="keyframe"`` with ``semantic_frame_index`` equal to 0 or
    -1.  Ref2VA uses ``image``, ``audio``, ``video``, or ``video_audio`` and
    consumes the descriptors in list order.
    """

    kind: str
    condition_index: int
    visual_rows: Any | None = None
    audio_rows: Any | None = None
    latent_t: int | None = None
    latent_h: int | None = None
    latent_w: int | None = None
    ref_audio_t: int | None = None
    semantic_frame_index: int | None = None
    resolved_frame_index: int | None = None

def _positive(value: int, name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        converted = 0
    if isinstance(value, bool) or converted != value or converted <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return converted

def _non_negative(value: int, name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        converted = -1
    if isinstance(value, bool) or converted != value or converted < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return converted

def _stereo_audio_channel(value: int, name: str = "audio_channel") -> int:
    """Validate H3's fixed two-channel latent-audio contract."""

    converted = _positive(value, name)
    if converted != 2:
        raise ValueError(
            f"{name} must be exactly 2 for MiniMax-H3 stereo audio, got {value!r}"
        )
    return converted

def _validate_video_geometry(
    latent_t: int,
    latent_h: int,
    latent_w: int,
    *,
    prefix: str = "H3 video latent",
) -> tuple[int, int, int]:
    latent_t = _positive(latent_t, f"{prefix} T")
    latent_h = _positive(latent_h, f"{prefix} H")
    latent_w = _positive(latent_w, f"{prefix} W")
    patch_t, patch_h, patch_w = PATCH_SIZE
    if latent_t % patch_t or latent_h % patch_h or latent_w % patch_w:
        raise ValueError(
            f"{prefix} dimensions must be divisible by patch_size "
            f"{PATCH_SIZE}, got {(latent_t, latent_h, latent_w)}"
        )
    return latent_t, latent_h, latent_w

def _range_for_slice(value: slice):
    import torch

    return torch.arange(value.start, value.stop, dtype=torch.long)

def _cat_ranges(parts: Sequence[Any]):
    import torch

    if parts:
        return torch.cat(list(parts))
    return torch.empty(0, dtype=torch.long)

def _audio_width_positions(
    *,
    steps: int,
    channels: int,
    left: float,
    right: float,
):
    """Return the official channel-major left/right W coordinates."""

    import torch

    channels = _stereo_audio_channel(channels, "channels")
    if steps == 0:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(
        (
            torch.full((steps,), left, dtype=torch.float64),
            torch.full((steps,), right, dtype=torch.float64),
        )
    )

def _temporal_position_span_fl(temporal_length: int) -> float:
    """FL2VA endpoint span with the source's NumPy pairwise fp64 sum."""

    import numpy as np

    temporal_length = _positive(temporal_length, "temporal_length")
    spans = np.ones(temporal_length, dtype=np.float64) * FRAME_RESCALE
    for token_index in range(len(FRAME_PER_TOKEN)):
        spans[token_index :: len(FRAME_PER_TOKEN)] *= FRAME_PER_TOKEN[token_index]
    return float(spans.sum())

def _video_t_span_ref(temporal_length: int) -> float:
    """Ref2VA cursor span with the source's sequential Python fp64 sum."""

    temporal_length = _positive(temporal_length, "temporal_length")
    return sum(
        FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
        for index in range(temporal_length)
    )

def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float):
    import numpy as np
    import torch

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * INTERP
    return torch.from_numpy(grid).to(torch.float64)

def _sqrt_area(height: int, width: int):
    """Keep official NumPy scalar math without making module import eager."""

    import numpy as np

    return np.sqrt(height * width)

def _video_t_grid(length: int, origin: float):
    import torch

    spans = torch.tensor(
        [FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(length)],
        dtype=torch.float64,
    )
    return origin + torch.cat(
        [torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)]
    )

def patchify_video_latent(latent, *, patch_size: Sequence[int] = PATCH_SIZE):
    """Pack ``[B,C,T,H,W]`` video latents into H3's 96-wide token rows."""

    import torch

    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
        raise ValueError("video latent must be a rank-5 torch.Tensor [B,C,T,H,W]")
    patch_t, patch_h, patch_w = (int(item) for item in patch_size)
    batch, channels, full_t, full_h, full_w = (int(item) for item in latent.shape)
    if full_t % patch_t or full_h % patch_h or full_w % patch_w:
        raise ValueError(
            f"video latent shape {tuple(latent.shape)} is not divisible by "
            f"patch_size {tuple(patch_size)}"
        )
    time, height, width = (
        full_t // patch_t,
        full_h // patch_h,
        full_w // patch_w,
    )
    packed = latent.reshape(
        batch,
        channels,
        time,
        patch_t,
        height,
        patch_h,
        width,
        patch_w,
    )
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(
        batch * time * height * width,
        channels * patch_t * patch_h * patch_w,
    ).contiguous()

def unpatchify_video_tokens(
    rows,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int] = PATCH_SIZE,
):
    """Unpack H3 video rows to ``[B,C,T,H,W]``."""

    import torch

    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("video rows must be a rank-2 torch.Tensor")
    time, height, width, channels = (int(item) for item in latent_shape)
    patch_t, patch_h, patch_w = (int(item) for item in patch_size)
    expected_width = channels * patch_t * patch_h * patch_w
    if int(rows.shape[1]) != expected_width:
        raise ValueError(
            f"video row width {int(rows.shape[1])} != expected {expected_width}"
        )
    rows_per_sample = time * height * width
    if int(rows.shape[0]) % rows_per_sample:
        raise ValueError(
            f"video row count {int(rows.shape[0])} is not divisible by {rows_per_sample}"
        )
    packed = rows.reshape(
        -1,
        time,
        height,
        width,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(
        -1,
        channels,
        time * patch_t,
        height * patch_h,
        width * patch_w,
    ).contiguous()

def pack_audio_latent(latent):
    """Pack ``[channels, latent_dim, T]`` into channel-major rows."""

    import torch

    if not isinstance(latent, torch.Tensor) or latent.ndim != 3:
        raise ValueError("audio latent must be a rank-3 torch.Tensor [C,D,T]")
    channels, latent_dim, steps = (int(item) for item in latent.shape)
    _stereo_audio_channel(channels, "audio latent channels")
    return latent.permute(0, 2, 1).reshape(channels * steps, latent_dim).contiguous()

def unpack_audio_tokens(rows, *, audio_t: int, audio_channel: int = 2):
    """Unpack channel-major rows into ``[channels, latent_dim, T]``."""

    audio_channel = _stereo_audio_channel(audio_channel)

    import torch

    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("audio rows must be a rank-2 torch.Tensor")
    audio_t = _positive(audio_t, "audio_t")
    expected_rows = audio_t * audio_channel
    if int(rows.shape[0]) != expected_rows:
        raise ValueError(
            f"audio row count {int(rows.shape[0])} != {expected_rows} "
            f"({audio_channel} channels x {audio_t} steps)"
        )
    native = rows.reshape(audio_channel, audio_t, int(rows.shape[1]))
    return native.permute(0, 2, 1).contiguous()

def aligned_frame_count(duration_seconds: float, fps: int = 24) -> int:
    """Round a duration up to H3's ``17*n + 5`` frame lattice."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    fps = _positive(fps, "fps")
    requested = max(1, round(float(duration_seconds) * fps))
    if requested <= 5:
        return 5
    return math.ceil((requested - 5) / 17) * 17 + 5

def video_latent_time(frame_count: int) -> int:
    frame_count = _positive(frame_count, "frame_count")
    if frame_count <= 5:
        return 2
    if (frame_count - 5) % 17:
        raise ValueError("frame_count must lie on H3's 17*n+5 lattice")
    return ((frame_count - 5) // 17) * 5 + 2

def audio_latent_time(frame_count: int, fps: int = 24) -> int:
    frame_count = _positive(frame_count, "frame_count")
    fps = _positive(fps, "fps")
    return round((frame_count / fps) * 40)

__all__ = [n for n in list(globals()) if not n.startswith("__")]
