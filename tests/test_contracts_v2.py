import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from minimax_h3_nodes.contracts import (
    H3_AV_LATENT_SCHEMA_V2,
    H3_FL2VA_KEYFRAME_SIGNATURES,
    H3_MODEL_SCHEMA_V2,
    H3_REF2VA_PARTITION,
    H3_TASK_FL2VA,
    H3_TASK_PARTITIONS,
    H3_TASK_REF2VA,
    H3_TASK_T2VA,
    H3_TEXT_ENCODER_SCHEMA_V2,
    H3_T2VA_PARTITION,
    H3_VAE_SCHEMA_V2,
    H3ContractError,
    append_ref2va_reference,
    component_compatibility_fingerprint,
    compute_component_fingerprint,
    compute_release_fingerprint,
    condition_order_fingerprint,
    make_conditioning_v2,
    make_fl2va_keyframe,
    make_ref2va_reference,
    make_ref2va_references,
    material_compatibility_fingerprint,
    partition_for_task,
    resolve_deferred_target_v2,
    resolve_fl2va_target_v2,
    resolve_ref2va_target_v2,
    resolve_t2va_target,
    resolve_t2va_target_v2,
    target_compatibility_fingerprint,
    validate_av_latent_v2,
    validate_component_compatibility,
    validate_conditioning_v2,
    validate_fl2va_keyframes,
    validate_ref2va_references,
    validate_release_for_task,
    validate_target_v2,
    validate_task_partition,
)


def _media(height=720, width=1280):
    return SimpleNamespace(shape=(1, height, width, 3))


def _release(partition, tasks):
    return {
        "schema_version": 1,
        "partition": partition,
        "tasks": list(tasks),
        "task_aliases": {},
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
    }


def _component(
    kind,
    task,
    *,
    root="/models/MiniMax-H3/FL2VA",
    metadata_present=True,
    integer_scales=False,
):
    partition = partition_for_task(task)
    tasks = (
        [H3_TASK_T2VA, H3_TASK_FL2VA]
        if partition == H3_T2VA_PARTITION
        else [H3_TASK_REF2VA]
    )
    schemas = {
        "model": H3_MODEL_SCHEMA_V2,
        "text_encoder": H3_TEXT_ENCODER_SCHEMA_V2,
        "vae": H3_VAE_SCHEMA_V2,
    }
    root = str(Path(root).resolve())
    release = _release(partition, tasks) if metadata_present else {}
    if integer_scales:
        release["sigma_shift_scales"] = {"video": 12, "audio": 3}
    release_payload = json.dumps(
        {"root": root, "partition": partition, "metadata": release},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    release_fingerprint = hashlib.sha256(release_payload).hexdigest()
    component_names = {
        "model": "transformer",
        "text_encoder": "text_encoder",
        "vae": "vae",
    }
    path_fields = {
        "model": "transformer_path",
        "text_encoder": "text_encoder_path",
        "vae": "vae_path",
    }
    fingerprint_fields = {
        "model": "transformer_fingerprint",
        "text_encoder": "text_encoder_fingerprint",
        "vae": "vae_fingerprint",
    }
    component_path = str(Path(root, component_names[kind]).resolve())
    component_payload = (
        f"{release_fingerprint}\0{component_names[kind]}\0{component_path}"
    ).encode("utf-8")
    component_fingerprint = hashlib.sha256(component_payload).hexdigest()
    result = {
        "schema": schemas[kind],
        "task": task,
        "tasks": tasks,
        "partition": partition,
        "model_root": root,
        "release_metadata": release,
        "release_fingerprint": release_fingerprint,
        "component_fingerprint": component_fingerprint,
        fingerprint_fields[kind]: component_fingerprint,
        path_fields[kind]: component_path,
    }
    return result


class TaskPartitionV2Tests(unittest.TestCase):
    def test_official_task_partition_mapping(self):
        self.assertEqual(
            dict(H3_TASK_PARTITIONS),
            {
                H3_TASK_T2VA: H3_T2VA_PARTITION,
                H3_TASK_FL2VA: H3_T2VA_PARTITION,
                H3_TASK_REF2VA: H3_REF2VA_PARTITION,
            },
        )
        self.assertEqual(partition_for_task(" FL2VA "), H3_T2VA_PARTITION)
        self.assertEqual(partition_for_task("REF2VA"), H3_REF2VA_PARTITION)

    def test_partition_mismatch_fails_before_loader_use(self):
        with self.assertRaisesRegex(H3ContractError, "必须使用 'ref2va'"):
            validate_task_partition(H3_TASK_REF2VA, H3_T2VA_PARTITION)


class FL2VAContractV2Tests(unittest.TestCase):
    def test_only_three_public_signatures_are_accepted(self):
        self.assertEqual(
            H3_FL2VA_KEYFRAME_SIGNATURES,
            ((0,), (-1,), (0, -1)),
        )
        for signature in H3_FL2VA_KEYFRAME_SIGNATURES:
            with self.subTest(signature=signature):
                keyframes = [make_fl2va_keyframe(_media(), index) for index in signature]
                clean = validate_fl2va_keyframes(keyframes, frame_count=124)
                self.assertEqual(
                    [item["frame_index"] for item in clean],
                    list(signature),
                )
                expected_pixels = [123 if index == -1 else 0 for index in signature]
                self.assertEqual(
                    [item["resolved_frame_index"] for item in clean],
                    expected_pixels,
                )

    def test_middle_reverse_duplicate_and_too_many_signatures_fail(self):
        valid_first = make_fl2va_keyframe(_media(), 0)
        valid_last = make_fl2va_keyframe(_media(), -1)
        invalid_middle = dict(valid_first, frame_index=52)
        for keyframes in (
            [valid_last, valid_first],
            [valid_first, valid_first],
            [invalid_middle],
            [valid_first, invalid_middle, valid_last],
        ):
            with self.subTest(keyframes=keyframes), self.assertRaises(H3ContractError):
                validate_fl2va_keyframes(keyframes)

    def test_auto_geometry_uses_first_semantic_anchor(self):
        first = make_fl2va_keyframe(_media(900, 2100), 0)
        last = make_fl2va_keyframe(_media(2100, 900), -1)
        target = resolve_fl2va_target_v2(
            aspect_ratio="auto",
            duration_seconds=5.0,
            keyframes=[first, last],
        )
        self.assertEqual((target["width"], target["height"]), (1536, 672))
        self.assertEqual(target["geometry_source"], "first_keyframe")
        self.assertEqual(target["semantic_frame_indices"], [0, -1])
        self.assertEqual(target["pixel_frame_indices"], [0, 123])
        validate_target_v2(target, expected_task=H3_TASK_FL2VA)

    def test_auto_geometry_last_only_and_deferred_forms(self):
        last = make_fl2va_keyframe(
            object(),
            -1,
            display_width=1,
            display_height=2,
        )
        target = resolve_fl2va_target_v2(
            aspect_ratio="auto",
            duration_seconds=5.0,
            keyframes=[last],
        )
        self.assertEqual((target["width"], target["height"]), (704, 1440))
        self.assertEqual(target["geometry_source"], "last_keyframe")

        deferred = resolve_fl2va_target_v2(
            aspect_ratio="auto",
            duration_seconds=5.0,
        )
        self.assertEqual(deferred["geometry"], "deferred")
        validate_target_v2(deferred, require_resolved=False)
        with self.assertRaisesRegex(H3ContractError, "仍未"):
            validate_target_v2(deferred)

    def test_fl_accepts_flexible_ratio_but_keeps_shared_ratio_bounds(self):
        target = resolve_fl2va_target_v2(
            aspect_ratio="7:4",
            duration_seconds=5.0,
            keyframes=[make_fl2va_keyframe(_media(), 0)],
        )
        self.assertEqual((target["width"], target["height"]), (1344, 768))
        with self.assertRaisesRegex(H3ContractError, "1:4"):
            resolve_fl2va_target_v2(
                aspect_ratio="5:1",
                duration_seconds=5.0,
                keyframes=[make_fl2va_keyframe(_media(), 0)],
            )


class Ref2VAContractV2Tests(unittest.TestCase):
    def test_ordered_reference_types_are_preserved_without_artificial_max(self):
        ordered = [
            make_ref2va_reference("audio", object()),
            make_ref2va_reference("video", object(), has_audio=False),
            make_ref2va_reference("video_audio", object(), has_audio=True),
            make_ref2va_reference("image", object()),
            make_ref2va_reference("image", object()),
        ]
        wrapper = make_ref2va_references(ordered)
        self.assertEqual(
            [item["type"] for item in validate_ref2va_references(wrapper)],
            ["audio", "video", "video_audio", "image", "image"],
        )
        appended = append_ref2va_reference(
            wrapper,
            make_ref2va_reference("audio", object()),
        )
        self.assertEqual(len(appended["materials"]), 6)
        self.assertEqual(len(wrapper["materials"]), 5)

    def test_ref_target_buckets_and_auto_policy_default(self):
        reference = make_ref2va_references(
            [make_ref2va_reference("image", object())]
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio="auto",
            duration_seconds=5.0,
            references=reference,
        )
        self.assertEqual((target["width"], target["height"]), (1344, 768))
        self.assertEqual(target["geometry_source"], "policy_default")
        with self.assertRaisesRegex(H3ContractError, "官方六个比例桶"):
            resolve_ref2va_target_v2(
                aspect_ratio="7:4",
                duration_seconds=5.0,
                references=reference,
            )

    def test_ref_target_accepts_explicit_resolution_pair(self):
        reference = make_ref2va_references(
            [make_ref2va_reference("image", object())]
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            references=reference,
            width=832,
            height=480,
        )
        self.assertEqual(target["geometry"], "explicit_v1")
        self.assertEqual(target["geometry_source"], "explicit_target")
        self.assertEqual((target["width"], target["height"]), (832, 480))
        self.assertEqual(
            (target["requested_width"], target["requested_height"]), (832, 480)
        )
        self.assertEqual(
            (target["video_latent_h"], target["video_latent_w"]), (30, 52)
        )
        validate_target_v2(target, expected_task=H3_TASK_REF2VA)

        with self.assertRaisesRegex(H3ContractError, "同时填写"):
            resolve_ref2va_target_v2(
                aspect_ratio="16:9",
                duration_seconds=5.0,
                references=reference,
                width=832,
                height=0,
            )

    def test_single_audio_reference_may_resolve_or_defer_duration(self):
        resolved_refs = make_ref2va_references(
            [
                make_ref2va_reference("image", object()),
                make_ref2va_reference(
                    "audio",
                    object(),
                    audio_duration_seconds=5.0,
                ),
            ]
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=None,
            references=resolved_refs,
        )
        self.assertEqual(target["temporal"], "resolved_from_audio_reference")
        self.assertEqual(target["frame_count"], 124)
        self.assertEqual(target["audio_latent_t"], 207)
        self.assertIsNone(target["requested_duration_seconds"])

        unresolved_refs = make_ref2va_references(
            [make_ref2va_reference("audio", object())]
        )
        deferred = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=None,
            references=unresolved_refs,
        )
        self.assertEqual(deferred["temporal"], "deferred_from_audio_reference")
        validate_target_v2(deferred, require_resolved=False)
        resolved = resolve_deferred_target_v2(deferred, resolved_refs)
        self.assertEqual(resolved["frame_count"], 124)

    def test_audio_deferred_ref_target_preserves_explicit_resolution(self):
        unresolved_refs = make_ref2va_references(
            [make_ref2va_reference("audio", object())]
        )
        deferred = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=None,
            references=unresolved_refs,
            width=832,
            height=480,
        )
        validate_target_v2(deferred, require_resolved=False)

        resolved_refs = make_ref2va_references(
            [
                make_ref2va_reference(
                    "audio",
                    object(),
                    audio_duration_seconds=5.0,
                )
            ]
        )
        resolved = resolve_deferred_target_v2(deferred, resolved_refs)
        self.assertEqual((resolved["width"], resolved["height"]), (832, 480))
        self.assertEqual(resolved["geometry"], "explicit_v1")

    def test_omitted_duration_rejects_zero_multiple_or_silent_sources(self):
        image_only = make_ref2va_references(
            [make_ref2va_reference("image", object())]
        )
        multiple = make_ref2va_references(
            [
                make_ref2va_reference("audio", object()),
                make_ref2va_reference("video", object(), has_audio=True),
            ]
        )
        silent = make_ref2va_references(
            [make_ref2va_reference("video", object(), has_audio=False)]
        )
        for references in (image_only, multiple, silent):
            with self.subTest(references=references), self.assertRaises(H3ContractError):
                resolve_ref2va_target_v2(
                    aspect_ratio="16:9",
                    duration_seconds=None,
                    references=references,
                )

    def test_omitted_duration_ignores_silent_video_next_to_audio(self):
        references = make_ref2va_references(
            [
                make_ref2va_reference("video", object(), has_audio=False),
                make_ref2va_reference(
                    "audio",
                    object(),
                    audio_duration_seconds=5.0,
                ),
            ]
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=None,
            references=references,
        )
        self.assertEqual(target["duration_source_condition_index"], 1)
        self.assertEqual(target["temporal"], "resolved_from_audio_reference")
        self.assertEqual(target["frame_count"], 124)

    def test_omitted_duration_uses_only_sounded_video_among_multiple_videos(self):
        references = make_ref2va_references(
            [
                make_ref2va_reference("video", object(), has_audio=False),
                make_ref2va_reference(
                    "video",
                    object(),
                    has_audio=True,
                    audio_duration_seconds=6.0,
                ),
                make_ref2va_reference("video", object(), has_audio=False),
            ]
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=None,
            references=references,
        )
        self.assertEqual(target["duration_source_condition_index"], 1)
        self.assertEqual(target["temporal"], "resolved_from_audio_reference")
        self.assertEqual(target["audio_reference_duration_seconds"], 6.0)

    def test_omitted_duration_cannot_use_only_silent_videos(self):
        references = make_ref2va_references(
            [
                make_ref2va_reference("video", object(), has_audio=False),
                make_ref2va_reference("video", object(), has_audio=False),
            ]
        )
        with self.assertRaisesRegex(H3ContractError, "需要且只允许一个"):
            resolve_ref2va_target_v2(
                aspect_ratio="16:9",
                duration_seconds=None,
                references=references,
            )

    def test_video_audio_explicitly_rejects_missing_soundtrack(self):
        with self.assertRaisesRegex(H3ContractError, "必须包含音轨"):
            make_ref2va_reference("video_audio", object(), has_audio=False)


class TargetForgeryV2Tests(unittest.TestCase):
    def test_timing_is_recomputed_from_requested_duration(self):
        target = resolve_t2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
        )
        forged = dict(
            target,
            frame_count=141,
            duration_seconds=141 / 24,
            video_latent_t=42,
            audio_latent_t=235,
        )
        with self.assertRaisesRegex(H3ContractError, r"17n\+5 对齐结果"):
            validate_target_v2(forged)

    def test_duration_over_max_cannot_be_forged_with_consistent_latents(self):
        target = resolve_t2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=15.0,
        )
        forged = dict(
            target,
            requested_duration_seconds=15.5,
            frame_count=379,
            duration_seconds=379 / 24,
            video_latent_t=112,
            audio_latent_t=632,
        )
        with self.assertRaisesRegex(H3ContractError, "5 到 15"):
            validate_target_v2(forged)
        validate_target_v2(target)

    def test_resolved_geometry_must_match_declared_ratio_and_pixel_cap(self):
        target = resolve_t2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
        )
        forged = dict(
            target,
            width=4096,
            height=4096,
            video_latent_h=256,
            video_latent_w=256,
        )
        with self.assertRaisesRegex(H3ContractError, "声明比例"):
            validate_target_v2(forged)

        explicit = dict(
            target,
            geometry="explicit_v1",
            width=4096,
            height=4096,
            requested_width=4096,
            requested_height=4096,
            video_latent_h=256,
            video_latent_w=256,
        )
        with self.assertRaisesRegex(H3ContractError, "最多允许"):
            validate_target_v2(explicit)

    def test_deferred_fields_require_pre_media_stage(self):
        deferred = resolve_fl2va_target_v2(
            aspect_ratio="auto",
            duration_seconds=5.0,
        )
        with self.assertRaisesRegex(H3ContractError, "resolution_stage"):
            validate_target_v2(
                dict(deferred, resolution_stage="resolved"),
                require_resolved=False,
            )


class ConditioningSafetyV2Tests(unittest.TestCase):
    def _fl_values(self):
        keyframes = [make_fl2va_keyframe(_media(), 0)]
        target = resolve_fl2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            keyframes=keyframes,
        )
        keyframes = validate_fl2va_keyframes(keyframes, frame_count=124)
        return target, keyframes

    def test_multimodal_tags_are_required_and_length_checked(self):
        _target, keyframes = self._fl_values()
        embeds = SimpleNamespace(shape=(4, 5120))
        with self.assertRaisesRegex(H3ContractError, "必须包含"):
            make_conditioning_v2(
                H3_TASK_FL2VA,
                "p",
                embeds,
                conditions=keyframes,
            )
        with self.assertRaisesRegex(H3ContractError, "长度"):
            make_conditioning_v2(
                H3_TASK_FL2VA,
                "p",
                embeds,
                conditions=keyframes,
                text_token_tags=[0, 1, 1],
            )

    def test_legacy_token_tags_are_explicitly_normalized(self):
        _target, keyframes = self._fl_values()
        conditioning = make_conditioning_v2(
            H3_TASK_FL2VA,
            "p",
            SimpleNamespace(shape=(4, 5120)),
            conditions=keyframes,
            token_tags=[0, 1, 1, 1],
        )
        self.assertEqual(conditioning["text_token_tags"], [0, 1, 1, 1])
        self.assertNotIn("token_tags", conditioning)

    def test_raw_conditions_and_encoded_blocks_must_match(self):
        references = make_ref2va_references(
            [
                make_ref2va_reference("image", object()),
                make_ref2va_reference("audio", object()),
            ]
        )
        clean_references = validate_ref2va_references(references)
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            references=references,
        )
        conditioning = make_conditioning_v2(
            H3_TASK_REF2VA,
            "p",
            SimpleNamespace(shape=(4, 5120)),
            conditions=clean_references,
            text_token_tags=[0, 1, 1, 1],
        )
        conditioning.update(
            {
                "condition_blocks": [
                    {
                        "condition_index": 0,
                        "kind": "image",
                        "material_fingerprint": clean_references[0][
                            "material_fingerprint"
                        ],
                    },
                    {
                        "condition_index": 1,
                        "kind": "audio",
                        "material_fingerprint": clean_references[1][
                            "material_fingerprint"
                        ],
                    },
                ],
                "target": target,
                "target_fingerprint": target_compatibility_fingerprint(target),
                "release_fingerprint": "release",
                "text_encoder_fingerprint": "te",
                "vae_fingerprint": "vae",
            }
        )
        validate_conditioning_v2(conditioning)
        wrong = dict(
            conditioning,
            condition_blocks=[
                {"condition_index": 0, "kind": "audio"},
                {"condition_index": 1, "kind": "image"},
            ],
        )
        with self.assertRaisesRegex(H3ContractError, r"references\[0\]"):
            validate_conditioning_v2(wrong)
        wrong_fingerprint = dict(
            conditioning,
            condition_order_fingerprint=condition_order_fingerprint(
                H3_TASK_REF2VA,
                list(reversed(clean_references)),
            ),
        )
        with self.assertRaisesRegex(H3ContractError, "order_fingerprint"):
            validate_conditioning_v2(wrong_fingerprint)

    def test_same_kind_reference_swap_and_cross_branch_blocks_are_rejected(self):
        references = make_ref2va_references(
            [
                make_ref2va_reference("image", object()),
                make_ref2va_reference("image", object()),
            ]
        )
        clean_references = validate_ref2va_references(references)
        self.assertNotEqual(
            clean_references[0]["material_fingerprint"],
            clean_references[1]["material_fingerprint"],
        )
        self.assertEqual(
            material_compatibility_fingerprint(clean_references[0]),
            clean_references[0]["material_fingerprint"],
        )
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            references=references,
        )
        conditioning = make_conditioning_v2(
            H3_TASK_REF2VA,
            "p",
            SimpleNamespace(shape=(4, 5120)),
            conditions=clean_references,
            text_token_tags=[0, 1, 1, 1],
        )
        conditioning.update(
            {
                "target": target,
                "target_fingerprint": target_compatibility_fingerprint(target),
                "release_fingerprint": "release",
                "text_encoder_fingerprint": "te",
                "vae_fingerprint": "vae",
            }
        )
        swapped_blocks = [
            {
                "condition_index": index,
                "kind": "image",
                "material_fingerprint": clean_references[1 - index][
                    "material_fingerprint"
                ],
            }
            for index in range(2)
        ]
        with self.assertRaisesRegex(H3ContractError, "material_fingerprint"):
            validate_conditioning_v2(
                dict(conditioning, condition_blocks=swapped_blocks)
            )

        other = make_ref2va_reference("image", object())
        cross_branch_blocks = [
            {
                "condition_index": 0,
                "kind": "image",
                "material_fingerprint": other["material_fingerprint"],
            },
            {
                "condition_index": 1,
                "kind": "image",
                "material_fingerprint": clean_references[1][
                    "material_fingerprint"
                ],
            },
        ]
        with self.assertRaisesRegex(H3ContractError, "material_fingerprint"):
            validate_conditioning_v2(
                dict(conditioning, condition_blocks=cross_branch_blocks)
            )


class V2CompatibilityTests(unittest.TestCase):
    def test_local_component_fingerprint_tracks_lightweight_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Ref2VA"
            component = root / "text_encoder"
            tokenizer = root / "tokenizer"
            component.mkdir(parents=True)
            tokenizer.mkdir()
            model_index = root / "model_index.json"
            config = component / "config.json"
            index = component / "model.safetensors.index.json"
            shard = component / "model-00001-of-00001.safetensors"
            tokenizer_config = tokenizer / "tokenizer_config.json"
            model_index.write_text(json.dumps({"text_encoder": {}}), encoding="utf-8")
            config.write_text(json.dumps({"hidden_size": 5120}), encoding="utf-8")
            index.write_text(
                json.dumps({"weight_map": {"x": shard.name}}),
                encoding="utf-8",
            )
            shard.write_bytes(b"tiny-test-shard")
            tokenizer_config.write_text(
                json.dumps({"model_max_length": 1024}),
                encoding="utf-8",
            )

            release = compute_release_fingerprint(root, "ref2va", {})
            fingerprint = compute_component_fingerprint(
                release,
                "text_encoder",
                component,
                related_paths={"tokenizer": tokenizer},
            )

            # JSON formatting alone is normalized and must not change identity.
            config.write_text('{\n  "hidden_size": 5120\n}\n', encoding="utf-8")
            self.assertEqual(
                compute_component_fingerprint(
                    release,
                    "text_encoder",
                    component,
                    related_paths={"tokenizer": tokenizer},
                ),
                fingerprint,
            )

            stat = shard.stat()
            os.utime(
                shard,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
            )
            after_shard_change = compute_component_fingerprint(
                release,
                "text_encoder",
                component,
                related_paths={"tokenizer": tokenizer},
            )
            self.assertNotEqual(after_shard_change, fingerprint)

            tokenizer_config.write_text(
                json.dumps({"model_max_length": 2048}),
                encoding="utf-8",
            )
            self.assertNotEqual(
                compute_component_fingerprint(
                    release,
                    "text_encoder",
                    component,
                    related_paths={"tokenizer": tokenizer},
                ),
                after_shard_change,
            )

            model_index.write_text(
                json.dumps({"text_encoder": {}, "tokenizer": {}}),
                encoding="utf-8",
            )
            self.assertNotEqual(
                compute_release_fingerprint(root, "ref2va", {}),
                release,
            )

    def test_release_and_components_admit_fl2va_together(self):
        release = validate_release_for_task(
            _release(H3_T2VA_PARTITION, [H3_TASK_T2VA, H3_TASK_FL2VA]),
            H3_TASK_FL2VA,
        )
        self.assertEqual(release["partition"], H3_T2VA_PARTITION)
        components = {
            kind: _component(kind, H3_TASK_FL2VA)
            for kind in ("model", "text_encoder", "vae")
        }
        result = validate_component_compatibility(
            task=H3_TASK_FL2VA,
            model=components["model"],
            text_encoder=components["text_encoder"],
            vae=components["vae"],
        )
        self.assertEqual(result["partition"], H3_T2VA_PARTITION)
        fingerprint = component_compatibility_fingerprint(
            components["model"],
            component_kind="model",
            task=H3_TASK_FL2VA,
        )
        self.assertEqual(
            fingerprint,
            (
                "/models/MiniMax-H3/FL2VA",
                H3_T2VA_PARTITION,
                (H3_TASK_T2VA, H3_TASK_FL2VA),
                1,
            ),
        )

    def test_official_metadata_free_wrapper_uses_computed_fingerprints(self):
        components = {
            kind: _component(
                kind,
                H3_TASK_REF2VA,
                root="/models/MiniMax-H3/Ref2VA",
                metadata_present=False,
            )
            for kind in ("model", "text_encoder", "vae")
        }
        components["model"].pop("release_metadata")
        result = validate_component_compatibility(
            task=H3_TASK_REF2VA,
            model=components["model"],
            text_encoder=components["text_encoder"],
            vae=components["vae"],
        )
        self.assertEqual(result["model"]["release_metadata"], {})
        self.assertEqual(
            result["model"]["release_fingerprint"],
            result["vae"]["release_fingerprint"],
        )

    def test_release_fingerprint_preserves_raw_valid_metadata_json(self):
        model = _component(
            "model",
            H3_TASK_FL2VA,
            integer_scales=True,
        )
        result = validate_component_compatibility(
            task=H3_TASK_FL2VA,
            model=model,
        )
        self.assertEqual(
            result["model"]["release_fingerprint"],
            model["release_fingerprint"],
        )
        self.assertEqual(
            result["model"]["release_metadata"]["sigma_shift_scales"],
            {"video": 12, "audio": 3},
        )

    def test_v2_wrapper_rejects_tampered_path_task_and_fingerprint(self):
        model = _component("model", H3_TASK_FL2VA)
        cases = (
            dict(model, task=H3_TASK_T2VA),
            dict(model, release_fingerprint="forged"),
            dict(model, transformer_path="/outside/transformer"),
        )
        for wrapper in cases:
            with self.subTest(wrapper=wrapper), self.assertRaises(H3ContractError):
                validate_component_compatibility(
                    task=H3_TASK_FL2VA,
                    model=wrapper,
                )

    def test_partition_and_root_mixes_fail(self):
        model = _component("model", H3_TASK_FL2VA)
        wrong_partition = dict(model, partition=H3_REF2VA_PARTITION)
        with self.assertRaisesRegex(H3ContractError, "必须使用 'fl2va'"):
            validate_component_compatibility(
                task=H3_TASK_FL2VA,
                model=wrong_partition,
            )
        with self.assertRaisesRegex(H3ContractError, "不同 model_root"):
            validate_component_compatibility(
                task=H3_TASK_FL2VA,
                model=model,
                vae=_component(
                    "vae",
                    H3_TASK_FL2VA,
                    root="/models/other/FL2VA",
                ),
            )

    def test_conditioning_target_and_latent_share_task(self):
        keyframes = [make_fl2va_keyframe(_media(), 0)]
        target = resolve_fl2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            keyframes=keyframes,
        )
        keyframes = validate_fl2va_keyframes(keyframes, frame_count=124)
        conditioning = make_conditioning_v2(
            H3_TASK_FL2VA,
            "a test",
            SimpleNamespace(shape=(12, 5120)),
            conditions=keyframes,
            text_token_tags=[0, *([1] * 11)],
        )
        text_encoder = _component("text_encoder", H3_TASK_FL2VA)
        vae = _component("vae", H3_TASK_FL2VA)
        conditioning.update(
            {
                "condition_blocks": [
                    {
                        "condition_index": 0,
                        "kind": "image",
                        "semantic_frame_index": 0,
                        "resolved_frame_index": 0,
                    }
                ],
                "target": target,
                "target_fingerprint": target_compatibility_fingerprint(target),
                "release_fingerprint": text_encoder["release_fingerprint"],
                "text_encoder_fingerprint": text_encoder[
                    "component_fingerprint"
                ],
                "vae_fingerprint": vae["component_fingerprint"],
            }
        )
        validate_conditioning_v2(conditioning, expected_task=H3_TASK_FL2VA)
        latent = {
            "schema": H3_AV_LATENT_SCHEMA_V2,
            "task": H3_TASK_FL2VA,
            "partition": H3_T2VA_PARTITION,
            "target": target,
            "video": SimpleNamespace(shape=(1, 24, 37, 48, 84)),
            "audio": SimpleNamespace(shape=(2, 32, 207)),
            "sampled": False,
        }
        validate_av_latent_v2(latent, expected_task=H3_TASK_FL2VA)
        with self.assertRaisesRegex(H3ContractError, "不一致"):
            validate_av_latent_v2(latent, expected_task=H3_TASK_REF2VA)

        result = validate_component_compatibility(
            task=H3_TASK_FL2VA,
            text_encoder=text_encoder,
            vae=vae,
            target=target,
            conditioning=conditioning,
            av_latent=latent,
        )
        self.assertEqual(result["text_encoder"], result["components"]["text_encoder"])
        mismatched_latent = dict(
            latent,
            target=resolve_fl2va_target_v2(
                aspect_ratio="4:3",
                duration_seconds=5.0,
                keyframes=[make_fl2va_keyframe(_media(), 0)],
            ),
            video=SimpleNamespace(shape=(1, 24, 37, 48, 64)),
        )
        with self.assertRaisesRegex(H3ContractError, "指纹不一致"):
            validate_component_compatibility(
                task=H3_TASK_FL2VA,
                text_encoder=text_encoder,
                vae=vae,
                target=target,
                conditioning=conditioning,
                av_latent=mismatched_latent,
            )

    def test_v1_t2va_target_wire_format_is_unchanged(self):
        old = resolve_t2va_target(aspect_ratio="16:9", duration_seconds=5.0)
        new = resolve_t2va_target_v2(aspect_ratio="16:9", duration_seconds=5.0)
        self.assertEqual(old["schema"], "minimax_h3_target/v1")
        self.assertEqual(new["schema"], "minimax_h3_target/v2")
        for field in (
            "width",
            "height",
            "frame_count",
            "video_latent_t",
            "audio_latent_t",
        ):
            self.assertEqual(old[field], new[field])


if __name__ == "__main__":
    unittest.main()
