"""MiniMax-H3 T2VA packed-token layout and latent transforms.

This is a direct, single-device adaptation of the model's native layout.  It
keeps the model's fp64 three-axis position grid and 64-row sequence alignment;
neither field is interchangeable with a regular Comfy latent mask.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

PACKED_ALIGNMENT = 64
TEXT_ID = -5
AUDIO_FIRST_ID = -15
AUDIO_ID = -14
VIDEO_FIRST_ID = -3
VIDEO_ID = -2
VIDEO_LAST_ID = -4
PAD_ID = -1

VIDEO_CHANNELS = 24
AUDIO_LATENT_DIM = 32
PATCH_SIZE = (1, 2, 2)
INTERP = 32
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float):
    import torch

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    right = left + ratio
    grid = np.linspace(left, right, dim // patch, endpoint=False) * INTERP
    return torch.from_numpy(grid).to(torch.float64)


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

    import torch

    text_len = _positive(text_len, "text_len")
    latent_t = _positive(latent_t, "latent_t")
    latent_h = _positive(latent_h, "latent_h")
    latent_w = _positive(latent_w, "latent_w")
    audio_t = _positive(audio_t, "audio_t")
    audio_channel = _positive(audio_channel, "audio_channel")
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
    sqrt_area = np.sqrt(latent_h * latent_w)
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
    channel_width_positions = [
        torch.full((audio_t,), float(w_grid[0]), dtype=torch.float64),
        torch.full((audio_t,), float(w_grid[-1]), dtype=torch.float64),
    ]
    if audio_channel > 2:
        for channel in range(1, audio_channel - 1):
            fraction = channel / (audio_channel - 1)
            value = float(w_grid[0]) * (1.0 - fraction) + float(w_grid[-1]) * fraction
            channel_width_positions.insert(
                channel,
                torch.full((audio_t,), value, dtype=torch.float64),
            )
    positions[audio_slice, 2] = torch.cat(channel_width_positions)

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


def build_t2va_packed_conditioning(
    prompt_embeds,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    audio_channel: int = 2,
):
    """Attach validated positive Qwen features to the native T2VA layout."""

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
    packed = minimax_h3_packed_sequence_t2va(
        text_len=int(prompt_embeds.shape[0]),
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        audio_channel=audio_channel,
    )
    packed["text_embeddings"] = prompt_embeds
    packed["latent_shape"] = (int(latent_t), int(latent_h), int(latent_w), VIDEO_CHANNELS)
    packed["audio_shape"] = (int(audio_channel), AUDIO_LATENT_DIM, int(audio_t))
    packed["task"] = "t2va"
    packed["cfg_distilled"] = True
    return packed


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
    return latent.permute(0, 2, 1).reshape(channels * steps, latent_dim).contiguous()


def unpack_audio_tokens(rows, *, audio_t: int, audio_channel: int = 2):
    """Unpack channel-major rows into ``[channels, latent_dim, T]``."""

    import torch

    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise ValueError("audio rows must be a rank-2 torch.Tensor")
    audio_t = _positive(audio_t, "audio_t")
    audio_channel = _positive(audio_channel, "audio_channel")
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
    "AUDIO_LATENT_DIM",
    "PACKED_ALIGNMENT",
    "PATCH_SIZE",
    "VIDEO_CHANNELS",
    "aligned_frame_count",
    "audio_latent_time",
    "build_t2va_packed_conditioning",
    "minimax_h3_packed_sequence_t2va",
    "pack_audio_latent",
    "patchify_video_latent",
    "unpack_audio_tokens",
    "unpatchify_video_tokens",
    "video_latent_time",
]
