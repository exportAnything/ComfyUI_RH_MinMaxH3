"""Conditioning 节点。"""
from __future__ import annotations
from ._shared import *  # noqa: F403

class MiniMaxH3FL2VAFirstFrameCondition:
    """Create the only valid first-only or first+last FL2VA signatures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"first_frame": ("IMAGE",)},
            "optional": {"last_frame": ("IMAGE",)},
        }

    RETURN_TYPES = (FL_KEYFRAMES_TYPE,)
    RETURN_NAMES = ("keyframes",)
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/fl2va"

    def build(self, first_frame: Any, last_frame: Any | None = None):
        first_width, first_height = _single_image_dimensions(
            first_frame, "first_frame"
        )
        items = [
            make_fl2va_keyframe(
                first_frame,
                0,
                display_width=first_width,
                display_height=first_height,
            )
        ]
        if last_frame is not None:
            last_width, last_height = _single_image_dimensions(
                last_frame, "last_frame"
            )
            items.append(
                make_fl2va_keyframe(
                    last_frame,
                    -1,
                    display_width=last_width,
                    display_height=last_height,
                )
            )
        return (validate_fl2va_keyframes(items),)


class MiniMaxH3FL2VALastFrameCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"last_frame": ("IMAGE",)}}

    RETURN_TYPES = (FL_KEYFRAMES_TYPE,)
    RETURN_NAMES = ("keyframes",)
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/fl2va"

    def build(self, last_frame: Any):
        width, height = _single_image_dimensions(last_frame, "last_frame")
        item = make_fl2va_keyframe(
            last_frame,
            -1,
            display_width=width,
            display_height=height,
        )
        return (validate_fl2va_keyframes([item]),)


class MiniMaxH3Ref2VAImageReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"references": (REFERENCES_TYPE,)},
        }

    RETURN_TYPES = (REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/ref2va"

    def append(self, image: Any, references: Mapping[str, Any] | None = None):
        width, height = _single_image_dimensions(image, "image")
        reference = make_ref2va_reference(
            "image",
            image,
            display_width=width,
            display_height=height,
        )
        return (append_ref2va_reference(references, reference),)


class MiniMaxH3Ref2VAAudioReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {"references": (REFERENCES_TYPE,)},
        }

    RETURN_TYPES = (REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/ref2va"

    def append(self, audio: Any, references: Mapping[str, Any] | None = None):
        audio = _downmix_comfy_audio(audio, "audio")
        _, _, duration = _audio_metadata(audio, "audio")
        reference = make_ref2va_reference(
            "audio",
            audio,
            has_audio=True,
            audio_duration_seconds=duration,
        )
        return (append_ref2va_reference(references, reference),)


class MiniMaxH3Ref2VAVideoReference:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            _runtime_module("media_conditioning").ensure_ffmpeg_tools(required=False)
        except Exception as exc:
            logging.getLogger(__name__).warning("Ref2VA ffmpeg 预检异常: %s", exc)
        return {
            "required": {
                "video": ("VIDEO",),
                "reference_type": (
                    ["video", "video_audio"],
                    {"default": "video"},
                ),
            },
            "optional": {"references": (REFERENCES_TYPE,)},
        }

    RETURN_TYPES = (REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "append"
    CATEGORY = f"{CATEGORY}/ref2va"

    def append(
        self,
        video: Any,
        reference_type: str,
        references: Mapping[str, Any] | None = None,
    ):
        width, height = _video_dimensions(video, "video")
        has_audio, audio_duration = _probe_video_audio(video, "video")
        if reference_type == "video_audio" and not has_audio:
            raise ValueError("video_audio reference 必须包含音轨")
        reference = make_ref2va_reference(
            str(reference_type),
            video,
            display_width=width,
            display_height=height,
            has_audio=has_audio,
            audio_duration_seconds=audio_duration,
        )
        return (append_ref2va_reference(references, reference),)


class MiniMaxH3FL2VAEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "target": (TARGET_TYPE,),
                "keyframes": (FL_KEYFRAMES_TYPE,),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/fl2va"

    def encode(
        self,
        h3_text_encoder: Mapping[str, Any],
        h3_vae_bundle: Mapping[str, Any],
        target: Mapping[str, Any],
        keyframes: Any,
        prompt: str,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        clean_target = validate_target_v2(
            target, expected_task=H3_TASK_FL2VA, require_resolved=True
        )
        clean_keyframes = validate_fl2va_keyframes(
            keyframes, frame_count=int(clean_target["frame_count"])
        )
        progress_total = 3 + 2 * len(clean_keyframes)
        progress_current = 0
        progress = _new_progress(progress_total)
        _preflight_target_conditions(
            task=H3_TASK_FL2VA,
            target=clean_target,
            conditions=clean_keyframes,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        compatibility = validate_component_compatibility(
            task=H3_TASK_FL2VA,
            text_encoder=h3_text_encoder,
            vae=h3_vae_bundle,
            target=clean_target,
        )
        text_wrapper = compatibility["text_encoder"]
        vae_wrapper = compatibility["vae"]
        release_fingerprint = _require_same_release(
            ("text_encoder", text_wrapper), ("vae", vae_wrapper)
        )
        prepared_images = []
        media = _runtime_module("media_conditioning")
        for ordinal, item in enumerate(clean_keyframes):
            _check_interrupted()
            source = _image_tensor_to_pil(item["media"], f"keyframes[{ordinal}]")
            prepared_images.append(
                media.prepare_fl_keyframe_canvas(
                    source,
                    target_width=int(clean_target["width"]),
                    target_height=int(clean_target["height"]),
                    keyframe_ordinal=ordinal,
                )
            )
            progress_current += 1
            _advance_progress(progress, progress_current, progress_total)

        _check_interrupted()
        text_handle = _unwrap_runtime_handle(
            text_wrapper, H3_TEXT_ENCODER_SCHEMA_V2, "h3_text_encoder"
        )
        target_fp = _target_fingerprint(clean_target)
        te_fp = _wrapper_component_fingerprint(
            text_wrapper, component_kind="text_encoder",
            path_key="text_encoder_path", fingerprint_key="text_encoder_fingerprint",
        )
        vae_fp = _wrapper_component_fingerprint(
            vae_wrapper, component_kind="vae", path_key="vae_path", fingerprint_key="vae_fingerprint",
        )
        encoded = _multimodal_qwen_encode(
            text_handle,
            task=H3_TASK_FL2VA,
            prompt=prompt,
            images=prepared_images,
            cache_parts=(target_fp, *(int(k["resolved_frame_index"]) for k in clean_keyframes)),
            te_fingerprint=te_fp,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)

        video_vae, _ = _vae_bundle_components(vae_wrapper)
        _require_comfy()
        model_management.unload_all_models()
        load_device = model_management.get_torch_device()
        condition_blocks = []
        with _vae_device_session(video_vae, load_device):
            for index, (item, prepared) in enumerate(
                zip(clean_keyframes, prepared_images, strict=True)
            ):
                _check_interrupted()
                pixels = _pil_to_image_tensor(prepared)
                condition_blocks.append(
                    media.encode_visual_condition_rows(
                        video_vae,
                        pixels,
                        condition_index=index,
                        kind="image",
                        process_image=True,
                        semantic_frame_index=int(item["frame_index"]),
                        resolved_frame_index=int(item["resolved_frame_index"]),
                        target_fingerprint=target_fp,
                        vae_fingerprint=vae_fp,
                    )
                )
                progress_current += 1
                _advance_progress(progress, progress_current, progress_total)

        conditioning = make_conditioning_v2(
            H3_TASK_FL2VA,
            prompt,
            encoded["prompt_embeds"],
            conditions=clean_keyframes,
            text_token_tags=encoded["text_token_tags"],
        )
        conditioning.update(
            {
                "condition_blocks": condition_blocks,
                "target": clean_target,
                "target_fingerprint": _target_fingerprint(clean_target),
                "release_fingerprint": release_fingerprint,
                "text_encoder_fingerprint": _wrapper_component_fingerprint(
                    text_wrapper,
                    component_kind="text_encoder",
                    path_key="text_encoder_path",
                    fingerprint_key="text_encoder_fingerprint",
                ),
                "vae_fingerprint": _wrapper_component_fingerprint(
                    vae_wrapper,
                    component_kind="vae",
                    path_key="vae_path",
                    fingerprint_key="vae_fingerprint",
                ),
            }
        )
        clean_conditioning = validate_conditioning_v2(
            conditioning, expected_task=H3_TASK_FL2VA
        )
        # Repeat the complete cross-value gate after the expensive encoders:
        # validators added by newer wire-contract versions must see the final
        # condition rows and fingerprints before anything reaches sampling.
        validate_component_compatibility(
            task=H3_TASK_FL2VA,
            text_encoder=text_wrapper,
            vae=vae_wrapper,
            target=clean_target,
            conditioning=clean_conditioning,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        return (clean_conditioning,)


class MiniMaxH3Ref2VAEncode:
    @classmethod
    def INPUT_TYPES(cls):
        try:  # object_info 阶段软探测，缺失只 warning 不阻断清单生成
            _runtime_module("media_conditioning").ensure_ffmpeg_tools(required=False)
        except Exception as exc:
            logging.getLogger(__name__).warning("Ref2VA ffmpeg 预检异常: %s", exc)
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "target": (TARGET_TYPE,),
                "references": (REFERENCES_TYPE,),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": True},
                ),
            },
            "optional": {
                "ref_image_size": (
                    list(H3_REFERENCE_IMAGE_SIZE_MODES),
                    {
                        "default": H3_REFERENCE_IMAGE_SIZE_MATCH,
                        "tooltip": (
                            "参考图尺寸策略。match：按生成画布的像素面积等比只缩不放；"
                            "max：参考管线独立的 2048 短边，identity 保真最好。"
                            "参考 token 每个采样步都参与注意力，max 可能慢数倍。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/ref2va"

    def encode(
        self,
        h3_text_encoder: Mapping[str, Any],
        h3_vae_bundle: Mapping[str, Any],
        target: Mapping[str, Any],
        references: Mapping[str, Any],
        prompt: str,
        ref_image_size: str = H3_REFERENCE_IMAGE_SIZE_MATCH,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        image_size_mode = str(ref_image_size or "").strip().lower()
        if image_size_mode not in H3_REFERENCE_IMAGE_SIZE_MODES:
            raise ValueError(
                "ref_image_size 必须是 "
                f"{list(H3_REFERENCE_IMAGE_SIZE_MODES)} 之一，实际 {ref_image_size!r}"
            )
        clean_target = validate_target_v2(
            target, expected_task=H3_TASK_REF2VA, require_resolved=True
        )
        clean_references = validate_ref2va_references(references)
        progress_total = 3 + 3 * len(clean_references)
        progress_current = 0
        progress = _new_progress(progress_total)
        _preflight_target_conditions(
            task=H3_TASK_REF2VA,
            target=clean_target,
            conditions=clean_references,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        compatibility = validate_component_compatibility(
            task=H3_TASK_REF2VA,
            text_encoder=h3_text_encoder,
            vae=h3_vae_bundle,
            target=clean_target,
        )
        text_wrapper = compatibility["text_encoder"]
        vae_wrapper = compatibility["vae"]
        release_fingerprint = _require_same_release(
            ("text_encoder", text_wrapper), ("vae", vae_wrapper)
        )
        media = _runtime_module("media_conditioning")
        prepared: list[dict[str, Any]] = []
        qwen_images: list[Any] = []
        qwen_videos: list[Any] = []
        condition_labels: list[tuple[str, int]] = []
        ordinals = {"image": 0, "audio": 0, "video": 0}

        with _h3_temporary_directory("h3_ref2va_") as workdir_value:
            workdir = Path(workdir_value)
            for index, item in enumerate(clean_references):
                _check_interrupted()
                kind = str(item["type"])
                record: dict[str, Any] = {
                    "condition_index": index,
                    "kind": kind,
                    "material_fingerprint": str(item["material_fingerprint"]),
                }
                if kind == "image":
                    source = _image_tensor_to_pil(item["media"], f"references[{index}]")
                    image = media.prepare_reference_image(
                        source,
                        size_mode=image_size_mode,
                        canvas_width=int(clean_target["width"]),
                        canvas_height=int(clean_target["height"]),
                    )
                    record["visual_pixels"] = _pil_to_image_tensor(image)
                    record["shape_policy"] = f"refimg={image_size_mode}"
                    qwen_images.append(image)
                    ordinals["image"] += 1
                    condition_labels.append(("image", ordinals["image"]))
                elif kind == "audio":
                    waveform, sample_rate, _ = _audio_metadata(
                        item["media"], f"references[{index}]"
                    )
                    media.build_reference_audio_plan(
                        kind="audio",
                        input_has_audio=True,
                        source_channels=int(waveform.shape[1]),
                        source_sample_rate=sample_rate,
                    )
                    record.update(
                        {"audio_waveform": waveform, "audio_sample_rate": sample_rate}
                    )
                    ordinals["audio"] += 1
                    condition_labels.append(("audio", ordinals["audio"]))
                else:
                    video = item["media"]
                    item_dir = workdir / f"condition_{index}"
                    item_dir.mkdir(parents=True, exist_ok=True)
                    source_path = _video_source_path(video, item_dir)
                    source_meta = _probe_video_path(source_path)
                    plan = media.build_reference_video_plan(
                        source_meta,
                        target_frame_count=int(clean_target["frame_count"]),
                        kind=kind,
                    )
                    materialized = media.execute_reference_video_plan(
                        source_path,
                        plan,
                        workdir=item_dir,
                        probe=_probe_video_path,
                        interrupt_check=_check_interrupted,
                    )
                    prepared_meta = _probe_video_path(materialized["prepared_path"])
                    presentation = _runtime_module("presentation")
                    sample_indices, _timestamps = (
                        presentation.minimax_h3_qwen_video_sample_plan(
                            int(prepared_meta.frame_count),
                            source_fps=float(prepared_meta.fps),
                        )
                    )
                    qwen_videos.append(
                        media.decode_reference_video_samples(
                            materialized["prepared_path"],
                            sample_indices,
                            interrupt_check=_check_interrupted,
                        )
                    )
                    frames = _decode_video_file(materialized["prepared_path"])
                    record["visual_pixels"] = frames.unsqueeze(0)
                    has_audio = bool(materialized["input_has_audio"])
                    if has_audio:
                        audio_plan = media.build_reference_audio_plan(
                            kind=kind,
                            input_has_audio=True,
                        )
                        extracted = media.execute_reference_audio_plan(
                            materialized["original_path"],
                            audio_plan,
                            workdir=item_dir,
                            interrupt_check=_check_interrupted,
                        )
                        audio = _load_pcm_wav(extracted["audio_path"])
                        waveform, sample_rate, _ = _audio_metadata(
                            audio, f"references[{index}].soundtrack"
                        )
                        record.update(
                            {
                                "audio_waveform": waveform,
                                "audio_sample_rate": sample_rate,
                            }
                        )
                        ordinals["audio"] += 1
                        condition_labels.append(("audio", ordinals["audio"]))
                    elif kind == "video_audio":
                        raise ValueError("video_audio reference 必须包含音轨")
                    ordinals["video"] += 1
                    condition_labels.append(("video", ordinals["video"]))
                prepared.append(record)
                progress_current += 1
                _advance_progress(progress, progress_current, progress_total)

            _check_interrupted()
            text_handle = _unwrap_runtime_handle(
                text_wrapper, H3_TEXT_ENCODER_SCHEMA_V2, "h3_text_encoder"
            )
            target_fp = _target_fingerprint(clean_target)
            te_fp = _wrapper_component_fingerprint(
                text_wrapper, component_kind="text_encoder",
                path_key="text_encoder_path", fingerprint_key="text_encoder_fingerprint",
            )
            vae_fp = _wrapper_component_fingerprint(
                vae_wrapper, component_kind="vae", path_key="vae_path", fingerprint_key="vae_fingerprint",
            )
            encoded = _multimodal_qwen_encode(
                text_handle,
                task=H3_TASK_REF2VA,
                prompt=prompt,
                images=qwen_images,
                condition_labels=condition_labels,
                videos=qwen_videos,
                # 尺寸策略进缓存键：同素材换 match/max 会送给 Qwen 不同的像素
                cache_parts=tuple(
                    str(r["material_fingerprint"]) for r in prepared
                ) + (f"refimg={image_size_mode}",),
                te_fingerprint=te_fp,
            )
            progress_current += 1
            _advance_progress(progress, progress_current, progress_total)

            video_vae, audio_vae = _vae_bundle_components(vae_wrapper)
            _require_comfy()
            model_management.unload_all_models()
            load_device = model_management.get_torch_device()
            visual_blocks: dict[int, Mapping[str, Any]] = {}
            audio_blocks: dict[int, Mapping[str, Any]] = {}
            with _vae_device_session(video_vae, load_device):
                for record in prepared:
                    _check_interrupted()
                    pixels = record.get("visual_pixels")
                    if pixels is not None:
                        index = int(record["condition_index"])
                        kind = str(record["kind"])
                        visual_blocks[index] = media.encode_visual_condition_rows(
                            video_vae,
                            pixels,
                            condition_index=index,
                            kind=kind,
                            process_image=(kind == "image"),
                            material_fingerprint=str(
                                record["material_fingerprint"]
                            ),
                            target_fingerprint=target_fp,
                            vae_fingerprint=vae_fp,
                            shape_policy=str(record.get("shape_policy", "")),
                        )
                    progress_current += 1
                    _advance_progress(progress, progress_current, progress_total)

            with _vae_device_session(audio_vae, load_device):
                for record in prepared:
                    _check_interrupted()
                    waveform = record.get("audio_waveform")
                    if waveform is not None:
                        index = int(record["condition_index"])
                        audio_blocks[index] = media.encode_audio_condition_rows(
                            audio_vae,
                            waveform,
                            condition_index=index,
                            kind=str(record["kind"]),
                            sample_rate=int(record["audio_sample_rate"]),
                            material_fingerprint=str(
                                record["material_fingerprint"]
                            ),
                            target_fingerprint=target_fp,
                            vae_fingerprint=vae_fp,
                        )
                    progress_current += 1
                    _advance_progress(progress, progress_current, progress_total)

        condition_blocks = [
            media.merge_condition_blocks(
                visual_blocks.get(index), audio_blocks.get(index)
            )
            for index in range(len(clean_references))
        ]
        conditioning = make_conditioning_v2(
            H3_TASK_REF2VA,
            prompt,
            encoded["prompt_embeds"],
            conditions=clean_references,
            text_token_tags=encoded["text_token_tags"],
        )
        conditioning.update(
            {
                "condition_blocks": condition_blocks,
                "condition_labels": condition_labels,
                "target": clean_target,
                "target_fingerprint": _target_fingerprint(clean_target),
                "release_fingerprint": release_fingerprint,
                "text_encoder_fingerprint": _wrapper_component_fingerprint(
                    text_wrapper,
                    component_kind="text_encoder",
                    path_key="text_encoder_path",
                    fingerprint_key="text_encoder_fingerprint",
                ),
                "vae_fingerprint": _wrapper_component_fingerprint(
                    vae_wrapper,
                    component_kind="vae",
                    path_key="vae_path",
                    fingerprint_key="vae_fingerprint",
                ),
            }
        )
        clean_conditioning = validate_conditioning_v2(
            conditioning, expected_task=H3_TASK_REF2VA
        )
        validate_component_compatibility(
            task=H3_TASK_REF2VA,
            text_encoder=text_wrapper,
            vae=vae_wrapper,
            target=clean_target,
            conditioning=clean_conditioning,
        )
        progress_current += 1
        _advance_progress(progress, progress_current, progress_total)
        return (clean_conditioning,)


class MiniMaxH3T2VATextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/conditioning"

    def encode(self, h3_text_encoder: Mapping[str, Any], prompt: str):
        handle = _unwrap_runtime_handle(
            h3_text_encoder, H3_TEXT_ENCODER_SCHEMA, "h3_text_encoder"
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        prompt_embeds = _encode_prompt(handle, prompt)
        return (make_t2va_conditioning(prompt, prompt_embeds),)


class MiniMaxH3UnsupportedConditioning:
    """Legacy fail-closed placeholder retained for old serialized workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (
                    [H3_TASK_FL2VA, H3_TASK_REF2VA],
                    {"default": H3_TASK_FL2VA},
                )
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "fail"
    CATEGORY = f"{CATEGORY}/conditioning"

    def fail(self, task: str):
        normalized = normalize_task(task)
        destination = (
            "MiniMaxH3FL2VAEncode"
            if normalized == H3_TASK_FL2VA
            else "MiniMaxH3Ref2VAEncode"
        )
        raise H3TaskNotImplementedError(
            "MiniMaxH3UnsupportedConditioning 是旧工作流的迁移错误占位节点；"
            f"请删除它并连接新的 {destination} 节点。"
        )


