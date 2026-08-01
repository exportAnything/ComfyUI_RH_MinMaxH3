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


def minimax_h3_packed_sequence_t2va(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
):
    """Build H3's ``[text | audio | target-video | pad]`` structural fields."""

    audio_channel = _stereo_audio_channel(audio_channel)

    import torch

    text_len = _positive(text_len, "text_len")
    latent_t = _positive(latent_t, "latent_t")
    latent_h = _positive(latent_h, "latent_h")
    latent_w = _positive(latent_w, "latent_w")
    audio_t = _positive(audio_t, "audio_t")
    patch_t, patch_h, patch_w = PATCH_SIZE
    if latent_t % patch_t or latent_h % patch_h or latent_w % patch_w:
        raise ValueError(
            "H3 video latent dimensions must be divisible by patch_size "
            f"{PATCH_SIZE}, got {(latent_t, latent_h, latent_w)}"
        )

    patched_h = latent_h // patch_h
    patched_w = latent_w // patch_w
    frame_rows = patched_h * patched_w
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + audio_rows + video_rows
    seq_len = math.ceil(used / PACKED_ALIGNMENT) * PACKED_ALIGNMENT

    text_slice = slice(0, text_len)
    audio_slice = slice(text_slice.stop, text_slice.stop + audio_rows)
    video_slice = slice(audio_slice.stop, audio_slice.stop + video_rows)
    pad_slice = slice(video_slice.stop, seq_len)

    input_ids = torch.full((seq_len,), PAD_ID, dtype=torch.int64)
    input_ids[text_slice] = TEXT_ID
    input_ids[audio_slice] = AUDIO_ID
    input_ids[audio_slice.start] = AUDIO_FIRST_ID
    input_ids[video_slice] = VIDEO_ID
    input_ids[video_slice.start] = VIDEO_FIRST_ID
    input_ids[video_slice.stop - 1] = VIDEO_LAST_ID

    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    image_mask[video_slice] = True
    audio_mask = torch.zeros(seq_len, dtype=torch.bool)
    audio_mask[audio_slice] = True
    img_pos = torch.arange(video_slice.start, video_slice.stop)
    audio_pos = torch.arange(audio_slice.start, audio_slice.stop)
    text_pos = torch.arange(text_slice.start, text_slice.stop)
    update_mask = torch.ones(video_rows, dtype=torch.bool)
    audio_update_mask = torch.ones(audio_rows, dtype=torch.bool)

    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[text_slice, 0] = torch.arange(text_len, dtype=torch.float64)
    sqrt_area = _sqrt_area(latent_h, latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, patch_h, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, patch_w, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    spatial_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    video_grid = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
    video_grid[:, :, 0] = _video_t_grid(latent_t, float(text_len))[:, None]
    video_grid[:, :, 1:] = spatial_frame[None]
    positions[video_slice] = video_grid.reshape(-1, 3)

    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    positions[audio_slice, 0] = audio_t_grid.repeat(audio_channel)
    positions[audio_slice, 2] = _audio_width_positions(
        steps=audio_t,
        channels=audio_channel,
        left=float(w_grid[0]),
        right=float(w_grid[-1]),
    )

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_slice] = 1
    token_tags[audio_slice] = 2
    token_tags[video_slice] = 0

    cu_seqlens = torch.tensor([0, used, seq_len], dtype=torch.int32)
    document_id = torch.zeros(seq_len, dtype=torch.int32)
    document_id[pad_slice] = 1
    return {
        "seq_len": torch.tensor(seq_len),
        "used_len": torch.tensor(used),
        "input_ids": input_ids,
        "image_mask": image_mask,
        "audio_mask": audio_mask,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": positions,
        "token_tags": token_tags,
        "cu_seqlens": cu_seqlens,
        "document_id": document_id,
    }


def _normalise_text_token_tags(
    text_token_tags: Any | None,
    *,
    text_len: int,
):
    import torch

    if text_token_tags is None:
        return torch.ones(text_len, dtype=torch.long)
    if not isinstance(text_token_tags, torch.Tensor):
        text_token_tags = torch.as_tensor(text_token_tags)
    tags = text_token_tags.detach().to(device="cpu", dtype=torch.long).view(-1)
    if int(tags.numel()) != text_len:
        raise ValueError(
            f"text_token_tags has {int(tags.numel())} rows, expected {text_len}"
        )
    invalid = (tags < 0) | (tags > 1)
    if bool(invalid.any()):
        raise ValueError("presentation text_token_tags values must be 0 or 1")
    return tags.contiguous()


def _apply_text_token_tags(
    packed: dict[str, Any],
    text_token_tags: Any | None,
) -> dict[str, Any]:
    text_pos = packed["text_pos"].view(-1)
    tags = _normalise_text_token_tags(
        text_token_tags,
        text_len=int(text_pos.numel()),
    )
    packed["token_tags"][text_pos] = tags
    packed["text_token_tags"] = tags
    return packed


def _resolve_fl_keyframes(
    keyframe_frame_indices: Sequence[int] | None,
    *,
    frame_count: int | None,
) -> tuple[list[int], list[int], int]:
    if keyframe_frame_indices is None:
        raise ValueError("strict fl2va packed layout requires keyframe_frame_indices")
    if isinstance(keyframe_frame_indices, (str, bytes)) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in keyframe_frame_indices
    ):
        raise ValueError(
            "strict fl2va packed layout requires integer keyframe_frame_indices"
        )
    semantic = list(keyframe_frame_indices)
    if tuple(semantic) not in FL2VA_KEYFRAME_SIGNATURES:
        raise ValueError(
            "strict fl2va packed layout requires keyframe_frame_indices in "
            f"{FL2VA_KEYFRAME_SIGNATURES!r}, got {semantic!r}"
        )
    if frame_count is None:
        raise ValueError("frame_count is required when keyframe_frame_indices are provided")
    frame_count = _positive(frame_count, "frame_count")
    resolved = [0 if value == 0 else frame_count - 1 for value in semantic]
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "keyframe frame indices resolve to the same pixel frame; "
            f"semantic={semantic!r}, resolved={resolved!r}"
        )
    return semantic, resolved, frame_count


def minimax_h3_packed_sequence(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    include_keyframe_cond: bool = False,
    keyframe_frame_indices: Sequence[int] | None = None,
    frame_count: int | None = None,
    text_token_tags: Any | None = None,
):
    """Build the official T2VA or strict endpoint-only FL2VA layout.

    The no-condition branch calls the original v1 T2VA function directly so
    existing workflows retain byte-for-byte structural tensors.  FL2VA adds
    one or two frozen visual blocks immediately after text.
    """

    audio_channel = _stereo_audio_channel(audio_channel)

    import torch

    if not isinstance(include_keyframe_cond, bool):
        raise ValueError("include_keyframe_cond must be a bool")
    if not include_keyframe_cond:
        if keyframe_frame_indices is not None:
            raise ValueError(
                "keyframe_frame_indices must be omitted when keyframe cond is not included"
            )
        packed = minimax_h3_packed_sequence_t2va(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            audio_channel=audio_channel,
        )
        packed["target_img_pos"] = packed["img_pos"]
        packed["condition_img_pos"] = torch.empty(0, dtype=torch.long)
        packed["target_audio_pos"] = packed["audio_pos"]
        packed["reference_audio_pos"] = torch.empty(0, dtype=torch.long)
        return _apply_text_token_tags(packed, text_token_tags)

    text_len = _positive(text_len, "text_len")
    latent_t, latent_h, latent_w = _validate_video_geometry(
        latent_t, latent_h, latent_w
    )
    audio_t = _positive(audio_t, "audio_t")
    semantic_indices, resolved_indices, frame_count = _resolve_fl_keyframes(
        keyframe_frame_indices,
        frame_count=frame_count,
    )

    _, patch_h, patch_w = PATCH_SIZE
    patched_h = latent_h // patch_h
    patched_w = latent_w // patch_w
    frame_rows = patched_h * patched_w
    condition_rows = len(semantic_indices) * frame_rows
    video_rows = latent_t * frame_rows
    audio_rows = audio_t * audio_channel
    used = text_len + condition_rows + audio_rows + video_rows
    seq_len = math.ceil(used / PACKED_ALIGNMENT) * PACKED_ALIGNMENT

    text_slice = slice(0, text_len)
    condition_slice = slice(text_slice.stop, text_slice.stop + condition_rows)
    audio_slice = slice(condition_slice.stop, condition_slice.stop + audio_rows)
    video_slice = slice(audio_slice.stop, audio_slice.stop + video_rows)
    pad_slice = slice(video_slice.stop, seq_len)

    input_ids = torch.full((seq_len,), PAD_ID, dtype=torch.int64)
    input_ids[text_slice] = TEXT_ID
    input_ids[condition_slice] = IMGVID_COND_ID
    input_ids[audio_slice] = AUDIO_ID
    input_ids[audio_slice.start] = AUDIO_FIRST_ID
    input_ids[video_slice] = VIDEO_ID
    input_ids[video_slice.start] = VIDEO_FIRST_ID
    input_ids[video_slice.stop - 1] = VIDEO_LAST_ID

    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    image_mask[condition_slice] = True
    image_mask[video_slice] = True
    audio_mask = torch.zeros(seq_len, dtype=torch.bool)
    audio_mask[audio_slice] = True

    condition_img_pos = _range_for_slice(condition_slice)
    target_img_pos = _range_for_slice(video_slice)
    img_pos = torch.cat((condition_img_pos, target_img_pos))
    target_audio_pos = _range_for_slice(audio_slice)
    text_pos = _range_for_slice(text_slice)
    update_mask = torch.zeros(condition_rows + video_rows, dtype=torch.bool)
    update_mask[condition_rows:] = True
    audio_update_mask = torch.ones(audio_rows, dtype=torch.bool)

    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[text_slice, 0] = torch.arange(text_len, dtype=torch.float64)
    sqrt_area = _sqrt_area(latent_h, latent_w)
    h_grid = _axis_from_sqrt_area(latent_h, patch_h, sqrt_area)
    w_grid = _axis_from_sqrt_area(latent_w, patch_w, sqrt_area)
    hh, ww = torch.meshgrid(h_grid, w_grid, indexing="ij")
    spatial_frame = torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1)

    endpoint_span = _temporal_position_span_fl(latent_t)
    condition_positions: list[float] = []
    for block_index, pixel_index in enumerate(resolved_indices):
        block_slice = slice(
            condition_slice.start + block_index * frame_rows,
            condition_slice.start + (block_index + 1) * frame_rows,
        )
        if pixel_index == 0:
            condition_t = float(text_len)
        elif pixel_index == frame_count - 1:
            condition_t = float(text_len) + endpoint_span - FRAME_RESCALE
        else:  # Defensive: signatures above should make this unreachable.
            raise ValueError(
                "fl2va packed layout only supports first/last keyframe anchors"
            )
        condition_positions.append(condition_t)
        positions[block_slice, 0] = condition_t
        positions[block_slice, 1:] = spatial_frame

    video_grid = torch.empty(latent_t, frame_rows, 3, dtype=torch.float64)
    video_grid[:, :, 0] = _video_t_grid(latent_t, float(text_len))[:, None]
    video_grid[:, :, 1:] = spatial_frame[None]
    positions[video_slice] = video_grid.reshape(-1, 3)

    audio_t_grid = float(text_len) + torch.arange(audio_t, dtype=torch.float64)
    positions[audio_slice, 0] = audio_t_grid.repeat(audio_channel)
    positions[audio_slice, 2] = _audio_width_positions(
        steps=audio_t,
        channels=audio_channel,
        left=float(w_grid[0]),
        right=float(w_grid[-1]),
    )

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_slice] = 1
    token_tags[audio_slice] = 2
    token_tags[img_pos] = 0
    cu_seqlens = torch.tensor([0, used, seq_len], dtype=torch.int32)
    document_id = torch.zeros(seq_len, dtype=torch.int32)
    document_id[pad_slice] = 1
    packed = {
        "seq_len": torch.tensor(seq_len),
        "used_len": torch.tensor(used),
        "input_ids": input_ids,
        "image_mask": image_mask,
        "audio_mask": audio_mask,
        "img_pos": img_pos,
        "audio_pos": target_audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": positions,
        "token_tags": token_tags,
        "cu_seqlens": cu_seqlens,
        "document_id": document_id,
        "condition_img_pos": condition_img_pos,
        "target_img_pos": target_img_pos,
        "reference_audio_pos": torch.empty(0, dtype=torch.long),
        "target_audio_pos": target_audio_pos,
        "keyframe_frame_indices": tuple(semantic_indices),
        "resolved_keyframe_frame_indices": tuple(resolved_indices),
        "keyframe_temporal_positions": tuple(condition_positions),
    }
    return _apply_text_token_tags(packed, text_token_tags)


def minimax_h3_packed_sequence_fl2va(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    keyframe_frame_indices: Sequence[int],
    frame_count: int,
    audio_channel: int = 2,
    text_token_tags: Any | None = None,
):
    """Named FL2VA wrapper for integrations that dispatch by task."""

    return minimax_h3_packed_sequence(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
        include_keyframe_cond=True,
        keyframe_frame_indices=keyframe_frame_indices,
        frame_count=frame_count,
        text_token_tags=text_token_tags,
    )


def _block_value(raw: Mapping[str, Any] | H3ConditionBlockDescriptor, key: str):
    if isinstance(raw, H3ConditionBlockDescriptor):
        return getattr(raw, key, None)
    if key == "kind":
        return raw.get("kind", raw.get("type"))
    return raw.get(key)


def _block_int(
    raw: Mapping[str, Any] | H3ConditionBlockDescriptor,
    key: str,
    path: str,
    *,
    allow_zero: bool = False,
) -> int:
    value = _block_value(raw, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        predicate = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path}.{key} must be {predicate}")
    return int(value)


def _parse_ref_blocks(
    ref_blocks: Sequence[Mapping[str, Any] | H3ConditionBlockDescriptor],
    *,
    audio_channel: int,
) -> tuple[list[dict[str, Any]], int, int]:
    if not isinstance(ref_blocks, Sequence) or isinstance(ref_blocks, (str, bytes)):
        raise ValueError("ref_blocks must be a sequence")
    parsed: list[dict[str, Any]] = []
    ref_visual_rows = 0
    ref_audio_rows = 0
    condition_indices: set[int] = set()
    for index, raw in enumerate(ref_blocks):
        path = f"ref_blocks[{index}]"
        if not isinstance(raw, (Mapping, H3ConditionBlockDescriptor)):
            raise ValueError(f"{path} must be an object or H3ConditionBlockDescriptor")
        kind = _block_value(raw, "kind")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"{path}.kind must be a non-empty string")
        kind = kind.strip().lower()
        condition_index_value = _block_value(raw, "condition_index")
        if condition_index_value is None:
            condition_index = index
        else:
            if (
                isinstance(condition_index_value, bool)
                or not isinstance(condition_index_value, int)
                or condition_index_value < 0
            ):
                raise ValueError(f"{path}.condition_index must be a non-negative integer")
            condition_index = int(condition_index_value)
        if condition_index in condition_indices:
            raise ValueError(f"duplicate condition_index {condition_index}")
        condition_indices.add(condition_index)

        if kind == "image":
            latent_h = _block_int(raw, "latent_h", path)
            latent_w = _block_int(raw, "latent_w", path)
            _validate_video_geometry(
                1, latent_h, latent_w, prefix=f"{path} image latent"
            )
            rows = (latent_h // PATCH_SIZE[1]) * (latent_w // PATCH_SIZE[2])
            item = {
                "kind": kind,
                "condition_index": condition_index,
                "latent_t": 1,
                "latent_h": latent_h,
                "latent_w": latent_w,
                "visual_rows_count": rows,
                "audio_rows_count": 0,
            }
            ref_visual_rows += rows
        elif kind == "audio":
            ref_audio_t = _block_int(raw, "ref_audio_t", path, allow_zero=True)
            audio_rows_count = ref_audio_t * audio_channel
            item = {
                "kind": kind,
                "condition_index": condition_index,
                "ref_audio_t": ref_audio_t,
                "visual_rows_count": 0,
                "audio_rows_count": audio_rows_count,
            }
            ref_audio_rows += audio_rows_count
        elif kind in ("video", "video_audio"):
            ref_audio_t = _block_int(raw, "ref_audio_t", path, allow_zero=True)
            latent_t = _block_int(raw, "latent_t", path)
            latent_h = _block_int(raw, "latent_h", path)
            latent_w = _block_int(raw, "latent_w", path)
            _validate_video_geometry(
                latent_t,
                latent_h,
                latent_w,
                prefix=f"{path} video latent",
            )
            frame_rows = (latent_h // PATCH_SIZE[1]) * (
                latent_w // PATCH_SIZE[2]
            )
            audio_rows_count = ref_audio_t * audio_channel
            visual_rows_count = latent_t * frame_rows
            item = {
                "kind": kind,
                "condition_index": condition_index,
                "ref_audio_t": ref_audio_t,
                "latent_t": latent_t,
                "latent_h": latent_h,
                "latent_w": latent_w,
                "frame_rows": frame_rows,
                "visual_rows_count": visual_rows_count,
                "audio_rows_count": audio_rows_count,
            }
            ref_visual_rows += visual_rows_count
            ref_audio_rows += audio_rows_count
        else:
            raise ValueError(f"{path}.kind unsupported for ref2va: {kind!r}")
        parsed.append(item)
    return parsed, ref_visual_rows, ref_audio_rows


def minimax_h3_packed_sequence_ref2va_blocks(
    *,
    text_len: int,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    ref_blocks: Sequence[Mapping[str, Any] | H3ConditionBlockDescriptor],
    audio_channel: int = 2,
    seq_len: int | None = None,
    text_token_tags: Any | None = None,
):
    """Build the official ordered Ref2VA packed layout.

    A video-bearing reference places its audio rows before its visual rows in
    packed sequence space.  Both start at the same temporal origin, then the
    cursor advances by the longer of the audio length and video RoPE span.
    Each visual reference owns an independent spatial grid.
    """

    audio_channel = _stereo_audio_channel(audio_channel)

    import torch

    text_len = _positive(text_len, "text_len")
    latent_t, latent_h, latent_w = _validate_video_geometry(
        latent_t, latent_h, latent_w
    )
    audio_t = _positive(audio_t, "audio_t")
    parsed, ref_visual_rows, ref_audio_rows = _parse_ref_blocks(
        ref_blocks,
        audio_channel=audio_channel,
    )

    patched_h = latent_h // PATCH_SIZE[1]
    patched_w = latent_w // PATCH_SIZE[2]
    target_frame_rows = patched_h * patched_w
    target_video_rows = latent_t * target_frame_rows
    target_audio_rows = audio_t * audio_channel
    used = (
        text_len
        + ref_visual_rows
        + ref_audio_rows
        + target_audio_rows
        + target_video_rows
    )
    if seq_len is None:
        seq_len = math.ceil(used / PACKED_ALIGNMENT) * PACKED_ALIGNMENT
    else:
        seq_len = _positive(seq_len, "seq_len")
        if seq_len < used:
            raise ValueError(f"seq_len {seq_len} < used rows {used}")

    text_slice = slice(0, text_len)
    cursor = text_slice.stop
    block_slices: list[dict[str, Any]] = []
    for item in parsed:
        kind = item["kind"]
        if kind == "image":
            visual_slice = slice(cursor, cursor + item["visual_rows_count"])
            cursor = visual_slice.stop
            block_slices.append({**item, "visual_slice": visual_slice})
        elif kind == "audio":
            audio_ref_slice = slice(cursor, cursor + item["audio_rows_count"])
            cursor = audio_ref_slice.stop
            block_slices.append({**item, "audio_slice": audio_ref_slice})
        else:
            # Official ordering is audio immediately before visual.
            audio_ref_slice = slice(cursor, cursor + item["audio_rows_count"])
            visual_slice = slice(
                audio_ref_slice.stop,
                audio_ref_slice.stop + item["visual_rows_count"],
            )
            cursor = visual_slice.stop
            block_slices.append(
                {
                    **item,
                    "audio_slice": audio_ref_slice,
                    "visual_slice": visual_slice,
                }
            )
    target_audio_slice = slice(cursor, cursor + target_audio_rows)
    target_video_slice = slice(
        target_audio_slice.stop,
        target_audio_slice.stop + target_video_rows,
    )
    pad_slice = slice(target_video_slice.stop, seq_len)

    input_ids = torch.full((seq_len,), PAD_ID, dtype=torch.int64)
    input_ids[text_slice] = TEXT_ID
    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    audio_mask = torch.zeros(seq_len, dtype=torch.bool)
    positions = torch.zeros(seq_len, 3, dtype=torch.float64)
    positions[text_slice, 0] = torch.arange(text_len, dtype=torch.float64)

    target_sqrt_area = _sqrt_area(latent_h, latent_w)
    target_h_grid = _axis_from_sqrt_area(
        latent_h, PATCH_SIZE[1], target_sqrt_area
    )
    target_w_grid = _axis_from_sqrt_area(
        latent_w, PATCH_SIZE[2], target_sqrt_area
    )
    target_hh, target_ww = torch.meshgrid(
        target_h_grid, target_w_grid, indexing="ij"
    )
    target_spatial_frame = torch.stack(
        [target_hh.reshape(-1), target_ww.reshape(-1)], dim=-1
    )

    ref_img_pos_parts: list[Any] = []
    ref_audio_pos_parts: list[Any] = []
    condition_metadata: list[dict[str, Any]] = []
    temporal_cursor = float(text_len)
    for item in block_slices:
        kind = item["kind"]
        metadata: dict[str, Any] = {
            key: value
            for key, value in item.items()
            if key not in {"visual_slice", "audio_slice"}
        }
        metadata["temporal_origin"] = temporal_cursor
        if kind == "image":
            visual_slice = item["visual_slice"]
            input_ids[visual_slice] = IMGVID_COND_ID
            image_mask[visual_slice] = True
            visual_pos = _range_for_slice(visual_slice)
            ref_img_pos_parts.append(visual_pos)
            ref_h = item["latent_h"]
            ref_w = item["latent_w"]
            ref_area = _sqrt_area(ref_h, ref_w)
            ref_hh, ref_ww = torch.meshgrid(
                _axis_from_sqrt_area(ref_h, PATCH_SIZE[1], ref_area),
                _axis_from_sqrt_area(ref_w, PATCH_SIZE[2], ref_area),
                indexing="ij",
            )
            positions[visual_slice, 0] = temporal_cursor
            positions[visual_slice, 1] = ref_hh.reshape(-1)
            positions[visual_slice, 2] = ref_ww.reshape(-1)
            temporal_cursor += 1.0
            metadata["visual_sequence_slice"] = (
                visual_slice.start,
                visual_slice.stop,
            )
        elif kind == "audio":
            audio_ref_slice = item["audio_slice"]
            input_ids[audio_ref_slice] = AUDIO_REF_COND_ID
            audio_mask[audio_ref_slice] = True
            audio_pos = _range_for_slice(audio_ref_slice)
            ref_audio_pos_parts.append(audio_pos)
            ref_audio_t = item["ref_audio_t"]
            positions[audio_ref_slice, 0] = (
                temporal_cursor
                + torch.arange(ref_audio_t, dtype=torch.float64)
            ).repeat(audio_channel)
            positions[audio_ref_slice, 2] = _audio_width_positions(
                steps=ref_audio_t,
                channels=audio_channel,
                left=float(target_w_grid[0]),
                right=float(target_w_grid[-1]),
            )
            temporal_cursor += float(ref_audio_t)
            metadata["audio_sequence_slice"] = (
                audio_ref_slice.start,
                audio_ref_slice.stop,
            )
        else:
            audio_ref_slice = item["audio_slice"]
            visual_slice = item["visual_slice"]
            input_ids[audio_ref_slice] = AUDIO_REF_COND_ID
            input_ids[visual_slice] = IMGVID_COND_ID
            audio_mask[audio_ref_slice] = True
            image_mask[visual_slice] = True
            audio_pos = _range_for_slice(audio_ref_slice)
            visual_pos = _range_for_slice(visual_slice)
            ref_audio_pos_parts.append(audio_pos)
            ref_img_pos_parts.append(visual_pos)

            ref_t = item["ref_audio_t"]
            ref_latent_t = item["latent_t"]
            ref_h = item["latent_h"]
            ref_w = item["latent_w"]
            ref_area = _sqrt_area(ref_h, ref_w)
            ref_h_grid = _axis_from_sqrt_area(
                ref_h, PATCH_SIZE[1], ref_area
            )
            ref_w_grid = _axis_from_sqrt_area(
                ref_w, PATCH_SIZE[2], ref_area
            )
            ref_hh, ref_ww = torch.meshgrid(
                ref_h_grid, ref_w_grid, indexing="ij"
            )
            positions[audio_ref_slice, 0] = (
                temporal_cursor + torch.arange(ref_t, dtype=torch.float64)
            ).repeat(audio_channel)
            positions[audio_ref_slice, 2] = _audio_width_positions(
                steps=ref_t,
                channels=audio_channel,
                left=float(ref_w_grid[0]),
                right=float(ref_w_grid[-1]),
            )
            ref_spatial_frame = torch.stack(
                [ref_hh.reshape(-1), ref_ww.reshape(-1)], dim=-1
            )
            ref_grid = torch.empty(
                ref_latent_t,
                item["frame_rows"],
                3,
                dtype=torch.float64,
            )
            ref_grid[:, :, 0] = _video_t_grid(
                ref_latent_t, temporal_cursor
            )[:, None]
            ref_grid[:, :, 1:] = ref_spatial_frame[None]
            positions[visual_slice] = ref_grid.reshape(-1, 3)
            temporal_cursor += max(
                float(ref_t), _video_t_span_ref(ref_latent_t)
            )
            metadata["audio_sequence_slice"] = (
                audio_ref_slice.start,
                audio_ref_slice.stop,
            )
            metadata["visual_sequence_slice"] = (
                visual_slice.start,
                visual_slice.stop,
            )
        metadata["temporal_end"] = temporal_cursor
        condition_metadata.append(metadata)

    input_ids[target_audio_slice] = AUDIO_ID
    input_ids[target_audio_slice.start] = AUDIO_FIRST_ID
    input_ids[target_video_slice] = VIDEO_ID
    input_ids[target_video_slice.start] = VIDEO_FIRST_ID
    input_ids[target_video_slice.stop - 1] = VIDEO_LAST_ID
    audio_mask[target_audio_slice] = True
    image_mask[target_video_slice] = True

    positions[target_audio_slice, 0] = (
        temporal_cursor + torch.arange(audio_t, dtype=torch.float64)
    ).repeat(audio_channel)
    positions[target_audio_slice, 2] = _audio_width_positions(
        steps=audio_t,
        channels=audio_channel,
        left=float(target_w_grid[0]),
        right=float(target_w_grid[-1]),
    )
    target_video_grid = torch.empty(
        latent_t, target_frame_rows, 3, dtype=torch.float64
    )
    target_video_grid[:, :, 0] = _video_t_grid(
        latent_t, temporal_cursor
    )[:, None]
    target_video_grid[:, :, 1:] = target_spatial_frame[None]
    positions[target_video_slice] = target_video_grid.reshape(-1, 3)

    condition_img_pos = _cat_ranges(ref_img_pos_parts)
    reference_audio_pos = _cat_ranges(ref_audio_pos_parts)
    target_img_pos = _range_for_slice(target_video_slice)
    target_audio_pos = _range_for_slice(target_audio_slice)
    img_pos = torch.cat((condition_img_pos, target_img_pos))
    audio_pos = torch.cat((reference_audio_pos, target_audio_pos))
    update_mask = torch.zeros(ref_visual_rows + target_video_rows, dtype=torch.bool)
    update_mask[ref_visual_rows:] = True
    audio_update_mask = torch.zeros(
        ref_audio_rows + target_audio_rows, dtype=torch.bool
    )
    audio_update_mask[ref_audio_rows:] = True
    text_pos = _range_for_slice(text_slice)

    token_tags = torch.full((seq_len,), -1, dtype=torch.long)
    token_tags[text_slice] = 1
    token_tags[reference_audio_pos] = 2
    token_tags[target_audio_slice] = 2
    token_tags[img_pos] = 0
    cu_seqlens = torch.tensor([0, used, seq_len], dtype=torch.int32)
    document_id = torch.zeros(seq_len, dtype=torch.int32)
    document_id[pad_slice] = 1
    packed = {
        "seq_len": torch.tensor(seq_len),
        "used_len": torch.tensor(used),
        "input_ids": input_ids,
        "image_mask": image_mask,
        "audio_mask": audio_mask,
        "img_pos": img_pos,
        "audio_pos": audio_pos,
        "text_pos": text_pos,
        "update_mask": update_mask,
        "audio_update_mask": audio_update_mask,
        "img_position_ids": positions,
        "token_tags": token_tags,
        "cu_seqlens": cu_seqlens,
        "document_id": document_id,
        "condition_img_pos": condition_img_pos,
        "target_img_pos": target_img_pos,
        "reference_audio_pos": reference_audio_pos,
        "target_audio_pos": target_audio_pos,
        "condition_blocks": tuple(condition_metadata),
        "target_temporal_origin": temporal_cursor,
    }
    return _apply_text_token_tags(packed, text_token_tags)


def _normalise_prompt_embeddings(prompt_embeds: Any):
    import torch

    if not isinstance(prompt_embeds, torch.Tensor):
        raise TypeError("prompt_embeds must be a torch.Tensor")
    if prompt_embeds.ndim == 3 and int(prompt_embeds.shape[0]) == 1:
        prompt_embeds = prompt_embeds[0]
    if prompt_embeds.ndim != 2 or int(prompt_embeds.shape[1]) != 5120:
        raise ValueError(
            "H3 prompt embeddings must have shape [text_len, 5120], got "
            f"{tuple(prompt_embeds.shape)}"
        )
    return prompt_embeds.contiguous()


def _resolve_presentation_tags(
    *,
    text_token_tags: Any | None,
    presentation_token_tags: Any | None,
) -> Any | None:
    if text_token_tags is not None and presentation_token_tags is not None:
        raise ValueError(
            "pass only one of text_token_tags or presentation_token_tags"
        )
    return (
        text_token_tags
        if text_token_tags is not None
        else presentation_token_tags
    )


def _coerce_condition_descriptor(
    raw: Mapping[str, Any] | H3ConditionBlockDescriptor,
    *,
    default_index: int,
) -> H3ConditionBlockDescriptor:
    if isinstance(raw, H3ConditionBlockDescriptor):
        descriptor = raw
    elif isinstance(raw, Mapping):
        kind = raw.get("kind", raw.get("type"))
        if not isinstance(kind, str):
            raise ValueError(
                f"condition_blocks[{default_index}].kind must be a string"
            )
        normalised_kind = kind.strip().lower()
        rows_alias = raw.get("rows")
        visual_rows = raw.get("visual_rows")
        audio_rows = raw.get("audio_rows")
        if visual_rows is None and normalised_kind in {
            "keyframe",
            "image",
            "video",
            "video_audio",
        }:
            visual_rows = rows_alias
        if audio_rows is None and normalised_kind == "audio":
            audio_rows = rows_alias
        descriptor = H3ConditionBlockDescriptor(
            kind=normalised_kind,
            condition_index=raw.get("condition_index", default_index),
            visual_rows=visual_rows,
            audio_rows=audio_rows,
            latent_t=raw.get("latent_t"),
            latent_h=raw.get("latent_h"),
            latent_w=raw.get("latent_w"),
            ref_audio_t=raw.get("ref_audio_t"),
            semantic_frame_index=raw.get(
                "semantic_frame_index", raw.get("frame_index")
            ),
            resolved_frame_index=raw.get("resolved_frame_index"),
        )
    else:
        raise ValueError(
            f"condition_blocks[{default_index}] must be a mapping or descriptor"
        )
    if (
        isinstance(descriptor.condition_index, bool)
        or not isinstance(descriptor.condition_index, int)
        or descriptor.condition_index < 0
    ):
        raise ValueError(
            f"condition_blocks[{default_index}].condition_index must be a "
            "non-negative integer"
        )
    kind = str(descriptor.kind).strip().lower()
    if kind != descriptor.kind:
        descriptor = H3ConditionBlockDescriptor(
            **{
                field: getattr(descriptor, field)
                for field in descriptor.__dataclass_fields__
                if field != "kind"
            },
            kind=kind,
        )
    return descriptor


def _condition_rows(
    value: Any,
    *,
    expected_rows: int,
    width: int,
    name: str,
):
    import torch

    if value is None:
        if expected_rows == 0:
            return torch.empty(0, width, dtype=torch.float32)
        raise ValueError(f"{name} is required ({expected_rows} rows expected)")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 torch.Tensor")
    if tuple(int(part) for part in value.shape) != (expected_rows, width):
        raise ValueError(
            f"{name} shape {tuple(value.shape)} != {(expected_rows, width)}"
        )
    # Condition encoders are transient GPU residents; keep their outputs on
    # CPU until the sampler materialises the denoise branch.
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _attach_common_conditioning_fields(
    packed: dict[str, Any],
    *,
    prompt_embeds: Any,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int,
    task: str,
    partition: str,
) -> dict[str, Any]:
    packed["text_embeddings"] = prompt_embeds
    packed["prompt_embeds"] = prompt_embeds
    packed["latent_shape"] = (
        int(latent_t),
        int(latent_h),
        int(latent_w),
        VIDEO_CHANNELS,
    )
    packed["audio_shape"] = (
        int(audio_channel),
        AUDIO_LATENT_DIM,
        int(audio_t),
    )
    packed["task"] = task
    packed["partition"] = partition
    packed["cfg_distilled"] = True
    packed["packed_schema"] = "minimax_h3_packed/v2"
    return packed


def build_t2va_packed_conditioning(
    prompt_embeds,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
    text_token_tags: Any | None = None,
    presentation_token_tags: Any | None = None,
):
    """Attach validated positive Qwen features to the native T2VA layout."""

    audio_channel = _stereo_audio_channel(audio_channel)
    prompt_embeds = _normalise_prompt_embeddings(prompt_embeds)
    tags = _resolve_presentation_tags(
        text_token_tags=text_token_tags,
        presentation_token_tags=presentation_token_tags,
    )
    packed = minimax_h3_packed_sequence(
        text_len=int(prompt_embeds.shape[0]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
        include_keyframe_cond=False,
        text_token_tags=tags,
    )
    packed["visual_cond_rows"] = None
    packed["audio_ref_rows"] = None
    packed["visual_condition_shapes"] = []
    packed["audio_reference_t"] = []
    packed["condition_blocks"] = ()
    return _attach_common_conditioning_fields(
        packed,
        prompt_embeds=prompt_embeds,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
        task="t2va",
        partition="fl2va",
    )


def build_fl2va_packed_conditioning(
    prompt_embeds,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    frame_count: int,
    condition_blocks: Sequence[
        Mapping[str, Any] | H3ConditionBlockDescriptor
    ]
    | None = None,
    keyframe_cond_rows: Any | None = None,
    keyframe_frame_indices: Sequence[int] | None = None,
    audio_channel: int = 2,
    text_token_tags: Any | None = None,
    presentation_token_tags: Any | None = None,
):
    """Build FL2VA conditioning with exact first/last frozen anchors."""

    audio_channel = _stereo_audio_channel(audio_channel)

    import torch

    prompt_embeds = _normalise_prompt_embeddings(prompt_embeds)
    tags = _resolve_presentation_tags(
        text_token_tags=text_token_tags,
        presentation_token_tags=presentation_token_tags,
    )
    latent_t, latent_h, latent_w = _validate_video_geometry(
        latent_t, latent_h, latent_w
    )
    audio_t = _positive(audio_t, "audio_t")
    frame_count = _positive(frame_count, "frame_count")
    descriptors: list[H3ConditionBlockDescriptor] = []
    visual_parts: list[Any] = []
    if condition_blocks is not None:
        if keyframe_cond_rows is not None or keyframe_frame_indices is not None:
            raise ValueError(
                "condition_blocks cannot be combined with keyframe_cond_rows or "
                "keyframe_frame_indices"
            )
        seen_indices: set[int] = set()
        for index, raw in enumerate(condition_blocks):
            descriptor = _coerce_condition_descriptor(raw, default_index=index)
            kind = descriptor.kind
            if kind not in {"keyframe", "image"}:
                raise ValueError(
                    f"FL2VA condition_blocks[{index}].kind must be keyframe"
                )
            if descriptor.condition_index in seen_indices:
                raise ValueError(
                    f"duplicate condition_index {descriptor.condition_index}"
                )
            seen_indices.add(descriptor.condition_index)
            semantic = descriptor.semantic_frame_index
            if isinstance(semantic, bool) or semantic not in (0, -1):
                raise ValueError(
                    f"FL2VA condition_blocks[{index}].semantic_frame_index "
                    "must be 0 or -1"
                )
            block_h = latent_h if descriptor.latent_h is None else descriptor.latent_h
            block_w = latent_w if descriptor.latent_w is None else descriptor.latent_w
            if block_h != latent_h or block_w != latent_w:
                raise ValueError(
                    "FL2VA keyframe rows must use the resolved target canvas "
                    f"{latent_h}x{latent_w}, got {block_h}x{block_w}"
                )
            frame_rows = (latent_h // PATCH_SIZE[1]) * (
                latent_w // PATCH_SIZE[2]
            )
            visual_parts.append(
                _condition_rows(
                    descriptor.visual_rows,
                    expected_rows=frame_rows,
                    width=VIDEO_CHANNELS
                    * PATCH_SIZE[0]
                    * PATCH_SIZE[1]
                    * PATCH_SIZE[2],
                    name=f"condition_blocks[{index}].visual_rows",
                )
            )
            resolved = 0 if semantic == 0 else int(frame_count) - 1
            if (
                descriptor.resolved_frame_index is not None
                and descriptor.resolved_frame_index != resolved
            ):
                raise ValueError(
                    f"condition_blocks[{index}].resolved_frame_index does not "
                    f"match semantic index {semantic}: expected {resolved}"
                )
            descriptors.append(
                H3ConditionBlockDescriptor(
                    kind="keyframe",
                    condition_index=descriptor.condition_index,
                    visual_rows=visual_parts[-1],
                    latent_t=1,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    semantic_frame_index=semantic,
                    resolved_frame_index=resolved,
                )
            )
        semantic_indices = [
            int(descriptor.semantic_frame_index) for descriptor in descriptors
        ]
    else:
        semantic_indices, resolved_indices, _ = _resolve_fl_keyframes(
            keyframe_frame_indices,
            frame_count=frame_count,
        )
        frame_rows = (latent_h // PATCH_SIZE[1]) * (
            latent_w // PATCH_SIZE[2]
        )
        all_rows = _condition_rows(
            keyframe_cond_rows,
            expected_rows=len(semantic_indices) * frame_rows,
            width=VIDEO_CHANNELS
            * PATCH_SIZE[0]
            * PATCH_SIZE[1]
            * PATCH_SIZE[2],
            name="keyframe_cond_rows",
        )
        for index, (semantic, resolved) in enumerate(
            zip(semantic_indices, resolved_indices, strict=True)
        ):
            part = all_rows[index * frame_rows : (index + 1) * frame_rows]
            visual_parts.append(part)
            descriptors.append(
                H3ConditionBlockDescriptor(
                    kind="keyframe",
                    condition_index=index,
                    visual_rows=part,
                    latent_t=1,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    semantic_frame_index=semantic,
                    resolved_frame_index=resolved,
                )
            )

    # Structural validation also enforces the only three legal signatures.
    packed = minimax_h3_packed_sequence_fl2va(
        text_len=int(prompt_embeds.shape[0]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        keyframe_frame_indices=semantic_indices,
        frame_count=frame_count,
        audio_channel=audio_channel,
        text_token_tags=tags,
    )
    visual_cond_rows = torch.cat(visual_parts, dim=0).contiguous()
    packed["visual_cond_rows"] = visual_cond_rows
    packed["keyframe_cond_rows"] = visual_cond_rows
    packed["audio_ref_rows"] = None
    packed["visual_condition_shapes"] = [
        (1, latent_h, latent_w) for _ in descriptors
    ]
    packed["audio_reference_t"] = []
    packed["condition_blocks"] = tuple(
        {
            "kind": "keyframe",
            "condition_index": descriptor.condition_index,
            "latent_t": 1,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "semantic_frame_index": descriptor.semantic_frame_index,
            "resolved_frame_index": descriptor.resolved_frame_index,
        }
        for descriptor in descriptors
    )
    return _attach_common_conditioning_fields(
        packed,
        prompt_embeds=prompt_embeds,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
        task="fl2va",
        partition="fl2va",
    )


def build_ref2va_packed_conditioning(
    prompt_embeds,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    condition_blocks: Sequence[
        Mapping[str, Any] | H3ConditionBlockDescriptor
    ],
    audio_channel: int = 2,
    text_token_tags: Any | None = None,
    presentation_token_tags: Any | None = None,
):
    """Build ordered Ref2VA conditioning and attach encoded anchor rows."""

    audio_channel = _stereo_audio_channel(audio_channel)

    import torch

    prompt_embeds = _normalise_prompt_embeddings(prompt_embeds)
    tags = _resolve_presentation_tags(
        text_token_tags=text_token_tags,
        presentation_token_tags=presentation_token_tags,
    )
    if not isinstance(condition_blocks, Sequence) or isinstance(
        condition_blocks, (str, bytes)
    ):
        raise ValueError("condition_blocks must be a sequence")
    if not condition_blocks:
        raise ValueError("Ref2VA requires at least one ordered condition block")
    latent_t, latent_h, latent_w = _validate_video_geometry(
        latent_t, latent_h, latent_w
    )
    audio_t = _positive(audio_t, "audio_t")

    descriptors: list[H3ConditionBlockDescriptor] = []
    layout_blocks: list[dict[str, Any]] = []
    visual_parts: list[Any] = []
    audio_parts: list[Any] = []
    visual_shapes: list[tuple[int, int, int]] = []
    audio_reference_t: list[int] = []
    seen_indices: set[int] = set()
    visual_width = VIDEO_CHANNELS * math.prod(PATCH_SIZE)
    for index, raw in enumerate(condition_blocks):
        descriptor = _coerce_condition_descriptor(raw, default_index=index)
        kind = descriptor.kind
        if kind not in {"image", "audio", "video", "video_audio"}:
            raise ValueError(
                f"Ref2VA condition_blocks[{index}].kind unsupported: {kind!r}"
            )
        if descriptor.condition_index in seen_indices:
            raise ValueError(f"duplicate condition_index {descriptor.condition_index}")
        seen_indices.add(descriptor.condition_index)
        if kind == "image":
            ref_h = _positive(descriptor.latent_h, f"condition_blocks[{index}].latent_h")
            ref_w = _positive(descriptor.latent_w, f"condition_blocks[{index}].latent_w")
            _validate_video_geometry(
                1, ref_h, ref_w, prefix=f"condition_blocks[{index}] image latent"
            )
            expected_visual = (ref_h // PATCH_SIZE[1]) * (
                ref_w // PATCH_SIZE[2]
            )
            visual = _condition_rows(
                descriptor.visual_rows,
                expected_rows=expected_visual,
                width=visual_width,
                name=f"condition_blocks[{index}].visual_rows",
            )
            descriptor = H3ConditionBlockDescriptor(
                kind=kind,
                condition_index=descriptor.condition_index,
                visual_rows=visual,
                latent_t=1,
                latent_h=ref_h,
                latent_w=ref_w,
            )
            layout_blocks.append(
                {
                    "kind": kind,
                    "condition_index": descriptor.condition_index,
                    "latent_h": ref_h,
                    "latent_w": ref_w,
                }
            )
            visual_parts.append(visual)
            visual_shapes.append((1, ref_h, ref_w))
        elif kind == "audio":
            ref_t = _positive(
                descriptor.ref_audio_t,
                f"condition_blocks[{index}].ref_audio_t",
            )
            audio = _condition_rows(
                descriptor.audio_rows,
                expected_rows=ref_t * audio_channel,
                width=AUDIO_LATENT_DIM,
                name=f"condition_blocks[{index}].audio_rows",
            )
            descriptor = H3ConditionBlockDescriptor(
                kind=kind,
                condition_index=descriptor.condition_index,
                audio_rows=audio,
                ref_audio_t=ref_t,
            )
            layout_blocks.append(
                {
                    "kind": kind,
                    "condition_index": descriptor.condition_index,
                    "ref_audio_t": ref_t,
                }
            )
            if ref_t:
                audio_parts.append(audio)
                audio_reference_t.append(ref_t)
        else:
            # A plain ``video`` reference is allowed to be silent.  The node
            # path then has only a visual block, so ``ref_audio_t`` is absent
            # rather than explicitly zero.  Canonicalise that one case here;
            # ``video_audio`` remains an explicit promise that an audio track
            # was encoded and must therefore carry a positive latent length.
            raw_ref_t = descriptor.ref_audio_t
            if kind == "video" and raw_ref_t is None:
                raw_ref_t = 0
            ref_t = (
                _non_negative(
                    raw_ref_t,
                    f"condition_blocks[{index}].ref_audio_t",
                )
                if kind == "video"
                else _positive(
                    raw_ref_t,
                    f"condition_blocks[{index}].ref_audio_t",
                )
            )
            ref_latent_t = _positive(
                descriptor.latent_t,
                f"condition_blocks[{index}].latent_t",
            )
            ref_h = _positive(
                descriptor.latent_h,
                f"condition_blocks[{index}].latent_h",
            )
            ref_w = _positive(
                descriptor.latent_w,
                f"condition_blocks[{index}].latent_w",
            )
            _validate_video_geometry(
                ref_latent_t,
                ref_h,
                ref_w,
                prefix=f"condition_blocks[{index}] video latent",
            )
            expected_visual = (
                ref_latent_t
                * (ref_h // PATCH_SIZE[1])
                * (ref_w // PATCH_SIZE[2])
            )
            visual = _condition_rows(
                descriptor.visual_rows,
                expected_rows=expected_visual,
                width=visual_width,
                name=f"condition_blocks[{index}].visual_rows",
            )
            audio = _condition_rows(
                descriptor.audio_rows,
                expected_rows=ref_t * audio_channel,
                width=AUDIO_LATENT_DIM,
                name=f"condition_blocks[{index}].audio_rows",
            )
            descriptor = H3ConditionBlockDescriptor(
                kind=kind,
                condition_index=descriptor.condition_index,
                visual_rows=visual,
                audio_rows=audio,
                latent_t=ref_latent_t,
                latent_h=ref_h,
                latent_w=ref_w,
                ref_audio_t=ref_t,
            )
            layout_blocks.append(
                {
                    "kind": kind,
                    "condition_index": descriptor.condition_index,
                    "ref_audio_t": ref_t,
                    "latent_t": ref_latent_t,
                    "latent_h": ref_h,
                    "latent_w": ref_w,
                }
            )
            if ref_t:
                audio_parts.append(audio)
                audio_reference_t.append(ref_t)
            visual_parts.append(visual)
            visual_shapes.append((ref_latent_t, ref_h, ref_w))
        descriptors.append(descriptor)

    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=int(prompt_embeds.shape[0]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        ref_blocks=layout_blocks,
        audio_channel=audio_channel,
        text_token_tags=tags,
    )
    packed["condition_layout"] = packed["condition_blocks"]
    packed["visual_cond_rows"] = (
        torch.cat(visual_parts, dim=0).contiguous() if visual_parts else None
    )
    packed["audio_ref_rows"] = (
        torch.cat(audio_parts, dim=0).contiguous() if audio_parts else None
    )
    packed["visual_condition_shapes"] = visual_shapes
    packed["audio_reference_t"] = audio_reference_t
    packed["condition_blocks"] = tuple(layout_blocks)
    return _attach_common_conditioning_fields(
        packed,
        prompt_embeds=prompt_embeds,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
        task="ref2va",
        partition="ref2va",
    )


def build_h3_packed_conditioning(
    prompt_embeds,
    *,
    task: str,
    **kwargs: Any,
):
    """Task-dispatching convenience API used by task-aware ComfyUI nodes."""

    normalised = str(task or "").strip().lower()
    if normalised == "t2va":
        return build_t2va_packed_conditioning(prompt_embeds, **kwargs)
    if normalised == "fl2va":
        return build_fl2va_packed_conditioning(prompt_embeds, **kwargs)
    if normalised == "ref2va":
        return build_ref2va_packed_conditioning(prompt_embeds, **kwargs)
    raise ValueError(f"unsupported MiniMax-H3 task {task!r}")


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


__all__ = [
    "AUDIO_FIRST_ID",
    "AUDIO_ID",
    "AUDIO_LATENT_DIM",
    "AUDIO_REF_COND_ID",
    "FL2VA_KEYFRAME_SIGNATURES",
    "H3ConditionBlockDescriptor",
    "IMGVID_COND_ID",
    "MINIMAX_H3_AUDIO_FIRST_ID",
    "MINIMAX_H3_AUDIO_ID",
    "MINIMAX_H3_AUDIO_REF_COND_ID",
    "MINIMAX_H3_IMGVID_COND_ID",
    "MINIMAX_H3_PAD_ID",
    "MINIMAX_H3_TEXT_ID",
    "MINIMAX_H3_VIDEO_FIRST_ID",
    "MINIMAX_H3_VIDEO_ID",
    "MINIMAX_H3_VIDEO_LAST_ID",
    "PACKED_ALIGNMENT",
    "PAD_ID",
    "PATCH_SIZE",
    "TEXT_ID",
    "VIDEO_CHANNELS",
    "VIDEO_FIRST_ID",
    "VIDEO_ID",
    "VIDEO_LAST_ID",
    "aligned_frame_count",
    "audio_latent_time",
    "build_fl2va_packed_conditioning",
    "build_h3_packed_conditioning",
    "build_ref2va_packed_conditioning",
    "build_t2va_packed_conditioning",
    "minimax_h3_packed_sequence",
    "minimax_h3_packed_sequence_fl2va",
    "minimax_h3_packed_sequence_ref2va_blocks",
    "minimax_h3_packed_sequence_t2va",
    "pack_audio_latent",
    "patchify_video_latent",
    "unpack_audio_tokens",
    "unpatchify_video_tokens",
    "video_latent_time",
]
