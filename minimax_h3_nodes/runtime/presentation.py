"""Official MiniMax-H3 Qwen3-VL presentation builders.

The H3 DiT consumes both the layer-50 Qwen features and a parallel AdaLN
token-tag stream.  Building both in one accumulator is intentional: a label,
timestamp or vision delimiter must never be added to one stream without the
other.

This module mirrors the public H3 source presentation contract:

* T2VA is the prompt verbatim, with no automatically-added special tokens.
* FL2VA prepends one ``<Picture n>: `` + image vision block per keyframe.
* Ref2VA emits ordered Picture/Audio/Video labels.  Audio has no Qwen payload;
  video is split into timestamped temporal blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

TEXT_TOKEN_TAG = 1
VISION_TOKEN_TAG = 0

QWEN_VIDEO_SAMPLE_FPS = 2.0
QWEN_VIDEO_TEMPORAL_PATCH = 2


def _torch():
    import torch

    return torch


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer(text, add_special_tokens=False)
    if not isinstance(value, Mapping) or "input_ids" not in value:
        raise TypeError("Qwen tokenizer must return a mapping containing input_ids")
    ids = value["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("H3 presentation only supports tokenizer batch=1")
        ids = ids[0]
    return [int(item) for item in ids]


def _special_token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None:
        raise ValueError(f"Qwen tokenizer does not define required token {token!r}")
    try:
        token_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Qwen tokenizer returned invalid id {value!r} for {token!r}"
        ) from exc
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == int(unk_id):
        raise ValueError(f"Qwen tokenizer mapped required token {token!r} to UNK")
    return token_id


def _vision_block_ids(tokenizer: Any, pad_token: str, count: int) -> list[int]:
    count = int(count)
    if count <= 0:
        raise ValueError("vision block token count must be positive")
    return (
        [_special_token_id(tokenizer, VISION_START)]
        + [_special_token_id(tokenizer, pad_token)] * count
        + [_special_token_id(tokenizer, VISION_END)]
    )


class _Presentation:
    """Accumulate aligned token ids and AdaLN modality tags."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: Sequence[int]) -> None:
        values = [int(item) for item in token_ids]
        self.ids.extend(values)
        self.tags.extend([TEXT_TOKEN_TAG] * len(values))

    def vision(self, token_ids: Sequence[int]) -> None:
        values = [int(item) for item in token_ids]
        self.ids.extend(values)
        self.tags.extend([VISION_TOKEN_TAG] * len(values))

    def build(self):
        torch = _torch()
        return (
            torch.tensor(self.ids, dtype=torch.long),
            torch.tensor(self.tags, dtype=torch.long),
        )


def _timestamped_video_blocks(
    presentation: _Presentation,
    tokenizer: Any,
    *,
    counts: Sequence[int],
    timestamps: Sequence[float],
) -> None:
    counts = [int(item) for item in counts]
    timestamps = [float(item) for item in timestamps]
    if not counts or len(counts) != len(timestamps):
        raise ValueError("video block token counts and timestamps must align")
    for count, timestamp in zip(counts, timestamps):
        if count <= 0:
            raise ValueError("video block token count must be positive")
        # Python's normal .1f formatting (including bankers rounding) is part
        # of the released H3 prompt contract.
        presentation.text(_text_ids(tokenizer, f"<{timestamp:.1f} seconds>"))
        presentation.vision(_vision_block_ids(tokenizer, VIDEO_PAD, count))


def minimax_h3_text_only_ids(tokenizer: Any, prompt: str):
    """T2VA presentation: verbatim prompt and no special tokens."""

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    return _torch().tensor(_text_ids(tokenizer, prompt), dtype=_torch().long)


def _multi_image_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: Sequence[int],
):
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    if not image_token_counts:
        raise ValueError("image_token_counts must be non-empty")
    presentation = _Presentation()
    for index, count in enumerate(image_token_counts, start=1):
        presentation.text(_text_ids(tokenizer, f"<Picture {index}>: "))
        presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, int(count)))
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def minimax_h3_multi_image_presentation_ids(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: Sequence[int],
):
    """FL2VA positive ids for one or two ordered keyframe images."""

    ids, _ = _multi_image_presentation(
        tokenizer, prompt=prompt, image_token_counts=image_token_counts
    )
    return ids


def minimax_h3_multi_image_presentation_token_tags(
    tokenizer: Any,
    *,
    prompt: str,
    image_token_counts: Sequence[int],
):
    """AdaLN token tags aligned with the FL2VA presentation ids."""

    _, tags = _multi_image_presentation(
        tokenizer, prompt=prompt, image_token_counts=image_token_counts
    )
    return tags


def _as_int_list(value: int | Sequence[int] | None, *, name: str) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an int or a sequence of ints")
    return [int(item) for item in value]


def _as_nested_list(
    value: Sequence[Any] | None,
    *,
    name: str,
    cast,
) -> list[list[Any]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if not value:
        return []
    first = value[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        output: list[list[Any]] = []
        for group in value:
            if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
                raise ValueError(f"{name} must not mix nested and flat entries")
            output.append([cast(item) for item in group])
        return output
    return [[cast(item) for item in value]]


def minimax_h3_ref2va_video_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: Sequence[tuple[str, int]],
    image_token_count: int | Sequence[int] | None,
    video_block_token_counts: Sequence[int] | Sequence[Sequence[int]] | None,
    video_block_timestamps: Sequence[float] | Sequence[Sequence[float]] | None,
):
    """Build the ordered Ref2VA Picture/Audio/Video presentation.

    ``condition_labels`` is already in request order and contains per-type,
    one-based ordinals.  A video carrying audio is represented by an Audio
    label immediately followed by its Video label; this keeps soundtrack
    probing outside this pure presentation function.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty")
    image_counts = _as_int_list(image_token_count, name="image_token_count")
    video_counts = _as_nested_list(
        video_block_token_counts,
        name="video_block_token_counts",
        cast=int,
    )
    video_timestamps = _as_nested_list(
        video_block_timestamps,
        name="video_block_timestamps",
        cast=float,
    )
    if len(video_counts) != len(video_timestamps):
        raise ValueError("video block token counts and timestamps must align")

    presentation = _Presentation()
    image_seen = 0
    video_seen = 0
    for raw_type, raw_ordinal in condition_labels:
        condition_type = str(raw_type)
        ordinal = int(raw_ordinal)
        if ordinal <= 0:
            raise ValueError("Ref2VA condition ordinals must be positive")
        if condition_type == "image":
            if image_seen >= len(image_counts):
                raise ValueError("image_token_count required for an image reference")
            count = image_counts[image_seen]
            image_seen += 1
            presentation.text(_text_ids(tokenizer, f"<Picture {ordinal}>: "))
            presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))
        elif condition_type == "audio":
            presentation.text(_text_ids(tokenizer, f"<Audio {ordinal}>: "))
        elif condition_type == "video":
            if video_seen >= len(video_counts):
                raise ValueError(
                    "video reference requires block token counts and timestamps"
                )
            counts = video_counts[video_seen]
            timestamps = video_timestamps[video_seen]
            video_seen += 1
            presentation.text(_text_ids(tokenizer, f"<Video {ordinal}>: "))
            _timestamped_video_blocks(
                presentation,
                tokenizer,
                counts=counts,
                timestamps=timestamps,
            )
        else:
            raise ValueError(f"unsupported ref2va condition type {condition_type!r}")

    if image_seen != len(image_counts):
        raise ValueError("unused image_token_count entries")
    if video_seen != len(video_counts):
        raise ValueError("unused video block token count entries")
    presentation.text(_text_ids(tokenizer, prompt))
    return presentation.build()


def minimax_h3_ref2va_presentation(
    tokenizer: Any,
    *,
    prompt: str,
    condition_labels: Sequence[tuple[str, int]],
    image_token_count: int | Sequence[int] | None,
):
    """Ref2VA Picture/Audio-only presentation."""

    return minimax_h3_ref2va_video_presentation(
        tokenizer,
        prompt=prompt,
        condition_labels=condition_labels,
        image_token_count=image_token_count,
        video_block_token_counts=None,
        video_block_timestamps=None,
    )


def minimax_h3_qwen_video_sample_plan(
    frame_count: int,
    *,
    source_fps: float = 24.0,
    sample_fps: float = QWEN_VIDEO_SAMPLE_FPS,
    temporal_patch: int = QWEN_VIDEO_TEMPORAL_PATCH,
) -> tuple[list[int], list[float]]:
    """Return official Qwen frame indices and temporal-block timestamps.

    Frames are selected with ``cursor += source_fps/sample_fps`` followed by
    Python ``round`` and de-duplication.  Timestamp positions are based on the
    resulting 2-fps stream (not source-frame PTS).  An odd final sample repeats
    its timestamp for Qwen's temporal patch of two.
    """

    frame_count = int(frame_count)
    source_fps = float(source_fps)
    sample_fps = float(sample_fps)
    temporal_patch = int(temporal_patch)
    if frame_count <= 0:
        raise ValueError("reference video frame_count must be positive")
    if source_fps <= 0 or sample_fps <= 0 or sample_fps > source_fps:
        raise ValueError("source_fps/sample_fps must satisfy 0 < sample <= source")
    if temporal_patch <= 0:
        raise ValueError("temporal_patch must be positive")

    ratio = source_fps / sample_fps
    indices: list[int] = []
    cursor = 0.0
    while True:
        index = int(round(cursor))
        if index >= frame_count:
            break
        if not indices or index > indices[-1]:
            indices.append(index)
        cursor += ratio
    if not indices:
        raise ValueError("reference video produced no Qwen samples")

    timestamps = [index / sample_fps for index in range(len(indices))]
    padding = (-len(timestamps)) % temporal_patch
    timestamps.extend([timestamps[-1]] * padding)
    block_timestamps = [
        (
            timestamps[start]
            + timestamps[start + temporal_patch - 1]
        )
        / 2.0
        for start in range(0, len(timestamps), temporal_patch)
    ]
    return indices, block_timestamps


__all__ = [
    "IMAGE_PAD",
    "QWEN_VIDEO_SAMPLE_FPS",
    "QWEN_VIDEO_TEMPORAL_PATCH",
    "TEXT_TOKEN_TAG",
    "VIDEO_PAD",
    "VISION_END",
    "VISION_START",
    "VISION_TOKEN_TAG",
    "minimax_h3_multi_image_presentation_ids",
    "minimax_h3_multi_image_presentation_token_tags",
    "minimax_h3_qwen_video_sample_plan",
    "minimax_h3_ref2va_presentation",
    "minimax_h3_ref2va_video_presentation",
    "minimax_h3_text_only_ids",
]
