from __future__ import annotations
from ._core import *  # noqa: F403
from .sequences import *  # noqa: F403

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

__all__ = [n for n in list(globals()) if not n.startswith("__")]
