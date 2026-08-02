from __future__ import annotations
from ._core import *  # noqa: F403

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

__all__ = [n for n in list(globals()) if not n.startswith("__")]
