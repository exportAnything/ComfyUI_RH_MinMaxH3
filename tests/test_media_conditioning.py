# SPDX-License-Identifier: Apache-2.0
"""Hermetic goldens for official FL2VA/Ref2VA media semantics."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

from minimax_h3_nodes.runtime.media_conditioning import (
    ReferenceVideoMetadata,
    _run_cancellable_process,
    build_reference_audio_plan,
    build_reference_video_plan,
    cover_crop_plan,
    decode_reference_video_samples,
    execute_reference_audio_plan,
    execute_reference_video_plan,
    merge_condition_blocks,
    order_condition_blocks,
    prepare_fl_keyframe_canvas,
    prepare_reference_image,
    reference_audio_extract_command,
    reference_video_display_geometry,
    reference_video_normalize_command,
    resolve_reference_image_shape,
    resolve_reference_video_shape,
    validate_prepared_reference_video,
)


HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class GeometryTests(unittest.TestCase):
    def test_cover_crop_matches_official_math(self):
        plan = cover_crop_plan(
            source_width=1280,
            source_height=808,
            target_width=1216,
            target_height=768,
        )
        self.assertAlmostEqual(plan["scale"], 768 / 808, places=9)
        self.assertEqual(plan["resized_size"], (1217, 768))
        self.assertEqual(plan["crop_box"], (0, 0, 1216, 768))

    @unittest.skipUnless(HAS_PIL, "requires Pillow")
    def test_first_anchor_stretches_and_second_cover_crops(self):
        from PIL import Image

        image = Image.new("RGB", (4, 2))
        image.putdata(
            [(255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)] * 2
        )
        first = prepare_fl_keyframe_canvas(
            image, target_width=4, target_height=4, keyframe_ordinal=0
        )
        follower = prepare_fl_keyframe_canvas(
            image, target_width=4, target_height=4, keyframe_ordinal=1
        )
        self.assertEqual(first.size, (4, 4))
        self.assertEqual(follower.size, (4, 4))
        self.assertNotEqual(first.tobytes(), follower.tobytes())

    def test_reference_image_shape_is_independent_2048_short_edge(self):
        shape = resolve_reference_image_shape(width=320, height=240)
        self.assertEqual((shape["width"], shape["height"]), (2720, 2048))
        self.assertTrue(shape["allow_upscale"])
        self.assertEqual(shape["multiple"], 32)

    @unittest.skipUnless(HAS_PIL, "requires Pillow")
    def test_reference_image_resize_consumes_resolved_shape(self):
        from PIL import Image

        image = Image.new("RGB", (320, 240), "red")
        prepared = prepare_reference_image(image)
        self.assertEqual(prepared.size, (2720, 2048))

    def test_reference_video_adapt_shape_goldens(self):
        cases = {
            (21, 9): (1536, 672),
            (16, 9): (1344, 768),
            (4, 3): (1024, 768),
            (9, 21): (672, 1536),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                shape = resolve_reference_video_shape(
                    width=source[0], height=source[1]
                )
                self.assertEqual((shape["width"], shape["height"]), expected)

    def test_reference_video_geometry_prefers_dar_then_applies_rotation(self):
        self.assertEqual(
            reference_video_display_geometry(
                coded_width=720,
                coded_height=576,
                sample_aspect_ratio="1:1",
                display_aspect_ratio="4:3",
            ),
            (768.0, 576.0),
        )
        self.assertEqual(
            reference_video_display_geometry(
                coded_width=1920,
                coded_height=1080,
                rotation_degrees=90,
            ),
            (1080.0, 1920.0),
        )

        metadata = ReferenceVideoMetadata.from_mapping(
            {
                "coded_width": 720,
                "coded_height": 576,
                "display_aspect_ratio": "4:3",
                "sample_aspect_ratio": "1:1",
                "rotation_degrees": 90,
                "fps": 24.0,
                "frame_count": 48,
                "has_audio": False,
            }
        )
        self.assertEqual((metadata.width, metadata.height), (576.0, 768.0))
        self.assertAlmostEqual(metadata.display_aspect_ratio, 3 / 4)
        plan = build_reference_video_plan(metadata, target_frame_count=48)
        self.assertEqual((plan.width, plan.height), (768, 1024))

        sar_fallback = ReferenceVideoMetadata.from_coded(
            width=720,
            height=576,
            fps=24.0,
            frame_count=48,
            has_audio=False,
            sample_aspect_ratio="16:15",
        )
        fallback_plan = build_reference_video_plan(
            sar_fallback, target_frame_count=48
        )
        self.assertEqual((fallback_plan.width, fallback_plan.height), (1024, 768))

    def test_fractional_display_geometry_is_not_rounded_before_adapt_shape(self):
        metadata = ReferenceVideoMetadata.from_coded(
            width=320,
            height=480,
            fps=24.0,
            frame_count=48,
            has_audio=False,
            sample_aspect_ratio="8:9",
        )
        self.assertAlmostEqual(metadata.width, 320 * 8 / 9)
        plan = build_reference_video_plan(metadata, target_frame_count=48)
        self.assertEqual((plan.width, plan.height), (768, 1280))

    def test_reference_ratios_fail_closed(self):
        for resolver in (resolve_reference_image_shape, resolve_reference_video_shape):
            with self.assertRaisesRegex(ValueError, "inclusive range"):
                resolver(width=401, height=100)


class ReferenceVideoPlanTests(unittest.TestCase):
    def _metadata(self, *, has_audio=True, frames=100):
        return ReferenceVideoMetadata.from_coded(
            width=1920,
            height=1080,
            fps=30.0,
            frame_count=frames,
            has_audio=has_audio,
            sample_aspect_ratio="16:15",
            rotation_degrees=90.0,
        )

    def test_square_sar_accepts_pyav_fraction_string(self):
        plan = build_reference_video_plan(
            self._metadata(),
            kind="video",
            target_frame_count=48,
            target_width=1536,
            target_height=672,
        )
        pyav_sar = str(Fraction(1, 1))
        self.assertEqual(pyav_sar, "1")
        prepared = ReferenceVideoMetadata(
            width=1536,
            height=672,
            fps=24.0,
            frame_count=48,
            has_audio=False,
            sample_aspect_ratio=pyav_sar,
            rotation_degrees=360.0,
        )
        self.assertIs(validate_prepared_reference_video(prepared, plan), prepared)

        for invalid_sar in ("N/A", "0:1", "not-a-ratio"):
            with self.subTest(invalid_sar=invalid_sar):
                invalid = ReferenceVideoMetadata.from_mapping(
                    {
                        "width": 1536,
                        "height": 672,
                        "fps": 24.0,
                        "frame_count": 48,
                        "has_audio": False,
                        "sample_aspect_ratio": invalid_sar,
                    }
                )
                with self.assertRaisesRegex(ValueError, "normalize SAR"):
                    validate_prepared_reference_video(invalid, plan)

        for invalid_rotation in ("not-a-number", float("nan"), float("inf")):
            with self.subTest(invalid_rotation=invalid_rotation):
                with self.assertRaisesRegex(ValueError, "invalid rotation"):
                    validate_prepared_reference_video(
                        {
                            "width": 1536,
                            "height": 672,
                            "fps": 24.0,
                            "frame_count": 48,
                            "has_audio": False,
                            "sample_aspect_ratio": "1:1",
                            "rotation_degrees": invalid_rotation,
                        },
                        plan,
                    )

    def test_node_probe_passes_dar_and_prefers_display_matrix_rotation(self):
        from minimax_h3_nodes import nodes

        stream = types.SimpleNamespace(
            average_rate=Fraction(24, 1),
            base_rate=None,
            frames=48,
            duration=None,
            time_base=None,
            width=720,
            height=576,
            sample_aspect_ratio=Fraction(1, 1),
            display_aspect_ratio=Fraction(16, 9),
            codec_context=types.SimpleNamespace(
                display_aspect_ratio=Fraction(16, 9)
            ),
            metadata={"rotate": "180"},
        )

        class Container:
            streams = types.SimpleNamespace(video=[stream], audio=[])

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        fake_av = types.SimpleNamespace(open=lambda *_args, **_kwargs: Container())
        probe_payload = {
            "streams": [
                {
                    "display_aspect_ratio": "4:3",
                    "tags": {"rotate": "180"},
                    "side_data_list": [{"rotation": 90}],
                }
            ]
        }
        with mock.patch.dict(sys.modules, {"av": fake_av}), mock.patch(
            "minimax_h3_nodes.runtime.media_conditioning._run_cancellable_process",
            return_value=types.SimpleNamespace(stdout=json.dumps(probe_payload)),
        ) as run:
            metadata = nodes._probe_video_path("rotated.mp4")
        self.assertEqual((metadata.width, metadata.height), (576.0, 768.0))
        self.assertEqual(metadata.sample_aspect_ratio, "1:1")
        self.assertEqual(metadata.rotation_degrees, 90.0)
        self.assertAlmostEqual(metadata.display_aspect_ratio, 3 / 4)
        plan = build_reference_video_plan(metadata, target_frame_count=48)
        self.assertEqual((plan.width, plan.height), (768, 1024))
        command = run.call_args.args[0]
        self.assertTrue(
            any("stream_side_data=rotation" in argument for argument in command)
        )
        self.assertIs(run.call_args.kwargs["interrupt_check"], nodes._check_interrupted)
        self.assertEqual(run.call_args.kwargs["timeout_seconds"], 60.0)
        self.assertTrue(run.call_args.kwargs["capture_output"])
        self.assertTrue(run.call_args.kwargs["text"])

    def test_component_video_without_stream_source_falls_back_to_save_to(self):
        from minimax_h3_nodes import nodes

        class ComponentVideo:
            @staticmethod
            def get_dimensions():
                return (832, 480)

            @staticmethod
            def get_components():
                return types.SimpleNamespace(audio=None)

            @staticmethod
            def save_to(path):
                Path(path).write_bytes(b"materialized-video")

        video = ComponentVideo()
        self.assertEqual(nodes._video_dimensions(video, "video"), (832, 480))
        with tempfile.TemporaryDirectory() as workdir:
            path = nodes._video_source_path(video, Path(workdir))
            self.assertEqual(path.name, "reference_input.mp4")
            self.assertEqual(path.read_bytes(), b"materialized-video")

    def test_plan_is_cfr24_direct_scale_no_crop_and_original_audio(self):
        plan = build_reference_video_plan(
            self._metadata(),
            kind="video",
            target_frame_count=48,
            target_width=1536,
            target_height=672,
        )
        self.assertEqual(
            plan.filtergraph,
            "fps=24,scale=1536:672:flags=lanczos,setsar=1",
        )
        self.assertFalse(plan.crop)
        self.assertFalse(plan.pad_if_short)
        self.assertEqual(plan.soundtrack_source, "original_untruncated")
        command = reference_video_normalize_command("/in.mp4", "/out.mp4", plan)
        self.assertIn("-an", command)
        self.assertNotIn("crop", command[command.index("-vf") + 1])

    def test_silent_video_keeps_visual_but_video_audio_fails(self):
        silent = self._metadata(has_audio=False)
        plan = build_reference_video_plan(
            silent, kind="video", target_frame_count=48
        )
        self.assertFalse(plan.input_has_audio)
        with self.assertRaisesRegex(ValueError, "requires an audio stream"):
            build_reference_video_plan(
                silent, kind="video_audio", target_frame_count=48
            )

    def test_executor_normalizes_then_truncates_and_retains_original(self):
        plan = build_reference_video_plan(
            self._metadata(),
            kind="video_audio",
            target_frame_count=48,
            target_width=1536,
            target_height=672,
        )
        commands = []
        probes = iter(
            [
                {
                    "width": 1536,
                    "height": 672,
                    "fps": 24.0,
                    "frame_count": 100,
                    "has_audio": False,
                    "sample_aspect_ratio": "1:1",
                    "rotation_degrees": 0.0,
                },
                {
                    "width": 1536,
                    "height": 672,
                    "fps": 24.0,
                    "frame_count": 48,
                    "has_audio": False,
                    "sample_aspect_ratio": "1:1",
                    "rotation_degrees": 0.0,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_reference_video_plan(
                "/input/original.mp4",
                plan,
                workdir=workdir,
                probe=lambda _path: next(probes),
                runner=lambda command, **_kwargs: commands.append(command),
            )
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            commands[1][commands[1].index("-frames:v") + 1], "48"
        )
        self.assertEqual(result["original_path"], "/input/original.mp4")
        self.assertEqual(result["audio_source_path"], "/input/original.mp4")
        self.assertTrue(result["prepared_path"].endswith("refvid_frames48.mp4"))

    def test_executor_keeps_short_reference_short_without_padding(self):
        plan = build_reference_video_plan(
            self._metadata(frames=12),
            kind="video",
            target_frame_count=48,
            target_width=1536,
            target_height=672,
        )
        commands = []
        with tempfile.TemporaryDirectory() as workdir:
            result = execute_reference_video_plan(
                "/input/short.mp4",
                plan,
                workdir=workdir,
                probe=lambda _path: {
                    "width": 1536,
                    "height": 672,
                    "fps": 24.0,
                    "frame_count": 12,
                    "has_audio": False,
                    "sample_aspect_ratio": str(Fraction(1, 1)),
                    "rotation_degrees": 0.0,
                },
                runner=lambda command, **_kwargs: commands.append(command),
            )
        self.assertEqual(len(commands), 1)
        self.assertEqual(result["frame_count"], 12)
        self.assertTrue(result["prepared_path"].endswith("refvid_1536x672.mp4"))

    def test_qwen_sample_decoder_consumes_presentation_indices_verbatim(self):
        seen = []
        marker = object()
        output = decode_reference_video_samples(
            "/prepared.mp4",
            [0, 12, 24],
            decoder=lambda path, indices: seen.append((path, indices)) or marker,
        )
        self.assertIs(output, marker)
        self.assertEqual(seen, [("/prepared.mp4", [0, 12, 24])])

    def test_media_process_interrupt_terminates_child(self):
        events = []

        class Process:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                events.append("terminate")
                self.returncode = -15

            def wait(self, timeout=None):
                events.append(("wait", timeout))
                return self.returncode

            def kill(self):
                events.append("kill")
                self.returncode = -9

        def interrupted():
            events.append("interrupt_check")
            raise RuntimeError("synthetic Comfy interrupt")

        with self.assertRaisesRegex(RuntimeError, "synthetic Comfy interrupt"):
            _run_cancellable_process(
                ["ffmpeg", "-version"],
                interrupt_check=interrupted,
                popen_factory=lambda _argv: Process(),
                poll_interval_seconds=0.0,
            )
        self.assertEqual(
            events, ["interrupt_check", "terminate", ("wait", 2.0)]
        )

    def test_media_process_timeout_escalates_to_kill(self):
        events = []

        class Process:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                events.append("terminate")

            def wait(self, timeout=None):
                events.append(("wait", timeout))
                if self.returncode is None and timeout is not None:
                    raise subprocess.TimeoutExpired("ffmpeg", timeout)
                return self.returncode

            def kill(self):
                events.append("kill")
                self.returncode = -9

        with self.assertRaises(subprocess.TimeoutExpired):
            _run_cancellable_process(
                ["ffmpeg", "-version"],
                timeout_seconds=1e-12,
                popen_factory=lambda _argv: Process(),
                poll_interval_seconds=0.0,
            )
        self.assertEqual(
            events,
            ["terminate", ("wait", 2.0), "kill", ("wait", None)],
        )

    def test_media_process_capture_returns_bounded_probe_stdout(self):
        factory_calls = []

        class Process:
            returncode = 0

            def poll(self):
                return self.returncode

            @staticmethod
            def communicate():
                return ('{"streams": []}', "")

        def factory(argv, **kwargs):
            factory_calls.append((argv, kwargs))
            return Process()

        result = _run_cancellable_process(
            ["ffprobe", "input.mp4"],
            capture_output=True,
            text=True,
            popen_factory=factory,
        )
        self.assertEqual(result.stdout, '{"streams": []}')
        self.assertEqual(factory_calls[0][0], ["ffprobe", "input.mp4"])
        self.assertIn("stdout", factory_calls[0][1])
        self.assertIn("stderr", factory_calls[0][1])
        self.assertTrue(factory_calls[0][1]["text"])


class ReferenceAudioPlanTests(unittest.TestCase):
    def test_pure_audio_preserves_rate_then_resamples_once(self):
        plan = build_reference_audio_plan(
            kind="audio",
            input_has_audio=True,
            source_channels=6,
            source_sample_rate=48_000,
        )
        self.assertEqual(plan.decode_sample_rate, 48_000)
        self.assertEqual(plan.target_sample_rate, 32_000)
        self.assertEqual(plan.resample_count, 1)
        self.assertEqual(plan.channel_policy, "downmix_stereo")
        command = reference_audio_extract_command("in.wav", "out.flac", plan)
        self.assertIn("-ac", command)
        self.assertNotIn("-ar", command)

    def test_video_audio_extracts_44100_from_original_then_resamples_once(self):
        plan = build_reference_audio_plan(
            kind="video_audio",
            input_has_audio=True,
            source_channels=2,
            source_sample_rate=48_000,
        )
        self.assertEqual(plan.decode_sample_rate, 44_100)
        self.assertEqual(plan.resample_count, 1)
        command = reference_audio_extract_command("original.mp4", "out.wav", plan)
        self.assertEqual(command[command.index("-ar") + 1], "44100")
        self.assertEqual(command[command.index("-ac") + 1], "2")

    def test_silent_plain_video_returns_zero_audio_marker_without_execution(self):
        plan = build_reference_audio_plan(kind="video", input_has_audio=False)
        calls = []
        result = execute_reference_audio_plan(
            "silent.mp4", plan, workdir="/unused", runner=lambda *a, **k: calls.append(a)
        )
        self.assertEqual(calls, [])
        self.assertIsNone(result["audio_rows"])
        self.assertEqual(result["ref_audio_t"], 0)


class ConditionBlockIdentityTests(unittest.TestCase):
    def test_merge_preserves_matching_material_fingerprint(self):
        fingerprint = "a" * 64
        visual = {
            "condition_index": 2,
            "kind": "video_audio",
            "material_fingerprint": fingerprint,
            "visual_rows": object(),
            "audio_rows": None,
            "ref_audio_t": 0,
        }
        audio = {
            "condition_index": 2,
            "kind": "video_audio",
            "material_fingerprint": fingerprint,
            "visual_rows": None,
            "audio_rows": object(),
            "ref_audio_t": 3,
        }
        merged = merge_condition_blocks(visual, audio)
        self.assertEqual(merged["material_fingerprint"], fingerprint)
        self.assertEqual(merged["ref_audio_t"], 3)

    def test_merge_rejects_crossed_same_kind_materials(self):
        visual = {
            "condition_index": 0,
            "kind": "video",
            "material_fingerprint": "a" * 64,
        }
        audio = {
            "condition_index": 0,
            "kind": "video",
            "material_fingerprint": "b" * 64,
        }
        with self.assertRaisesRegex(ValueError, "material_fingerprint"):
            merge_condition_blocks(visual, audio)


@unittest.skipUnless(HAS_TORCH, "requires torch")
class ConditionTensorTests(unittest.TestCase):
    def test_patchify_and_unified_condition_entries(self):
        import torch

        from minimax_h3_nodes.runtime.media_conditioning import (
            encode_audio_condition_rows,
            encode_visual_condition_rows,
            patchify_video_condition_rows,
        )

        latent = torch.arange(1 * 24 * 1 * 4 * 4, dtype=torch.float32).reshape(
            1, 24, 1, 4, 4
        )
        rows = patchify_video_condition_rows(latent)
        self.assertEqual(tuple(rows.shape), (4, 96))

        class Visual:
            def __init__(self):
                self.kwargs = None

            def encode(self, _pixels, **kwargs):
                self.kwargs = kwargs
                return latent

        visual_vae = Visual()
        visual = encode_visual_condition_rows(
            visual_vae,
            torch.zeros(1),
            condition_index=3,
            kind="video_audio",
            process_image=False,
            prepared_media={"prepared_path": "/prepared.mp4"},
            material_fingerprint="a" * 64,
        )
        self.assertEqual(visual_vae.kwargs["seed"], 42)
        self.assertTrue(visual_vae.kwargs["use_fp16_latent"])
        self.assertFalse(visual_vae.kwargs["parallel_tiling"])
        self.assertEqual((visual["latent_t"], visual["latent_h"], visual["latent_w"]), (1, 4, 4))
        self.assertEqual(visual["condition_index"], 3)
        self.assertEqual(visual["visual_rows"].dtype, torch.float32)
        self.assertEqual(visual["visual_rows"].device.type, "cpu")

        class Audio:
            def encode(self, _waveform, sample_rate=None):
                self.sample_rate = sample_rate
                return torch.arange(2 * 32 * 3, dtype=torch.float32).reshape(2, 32, 3)

        audio = encode_audio_condition_rows(
            Audio(),
            torch.zeros(1),
            condition_index=3,
            kind="video_audio",
            sample_rate=32_000,
            material_fingerprint="a" * 64,
        )
        self.assertEqual(tuple(audio["audio_rows"].shape), (6, 32))
        self.assertEqual(audio["ref_audio_t"], 3)
        merged = merge_condition_blocks(visual, audio)
        self.assertIs(merged["visual_rows"], visual["visual_rows"])
        self.assertIs(merged["audio_rows"], audio["audio_rows"])
        ordered = order_condition_blocks(
            [{"condition_index": 5}, {"condition_index": 1}, merged]
        )
        self.assertEqual([item["condition_index"] for item in ordered], [1, 3, 5])

    def test_video_vae_release_encode_rounds_fp16_and_restores_state(self):
        import torch

        from minimax_h3_nodes.runtime.vae_adapter import (
            H3LatentStats,
            MiniMaxH3VideoVAEAdapter,
        )

        class Processor:
            @staticmethod
            def get_suitable_video_length(frames):
                return frames

            @staticmethod
            def _align_to_total_patch_size(height, width):
                return height, width

            @staticmethod
            def _crop_to_align(value, height, width, is_video=False):
                return value[..., :height, :width]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
                self.processor = Processor()
                self.parallel_tiling = True
                self.tiling_during_encode = None
                self.dtype_during_encode = None
                self.draw = None

            @staticmethod
            def transform(value):
                return value

            def encode_base(self, value, process_image=False):
                self.tiling_during_encode = self.parallel_tiling
                self.dtype_during_encode = self.weight.dtype
                self.draw = torch.randn(4)
                return torch.full(
                    (int(value.shape[0]), 24, 1, 2, 2),
                    1.0003,
                    device=value.device,
                    dtype=torch.float32,
                )

        class RecordingStats:
            def __init__(self):
                self.inputs = []

            def normalize(self, value):
                self.inputs.append((value.device.type, value.dtype))
                return value

        model = Model()
        if torch.cuda.is_available():
            model = model.to("cuda")
        stats = RecordingStats()
        adapter = MiniMaxH3VideoVAEAdapter(
            model,
            stats,
            component_dir=Path("."),
            compute_dtype=torch.float32,
        )
        pixels = torch.zeros(1, 32, 32, 3)
        torch.manual_seed(1234)
        expected_after = torch.randn(4)
        torch.manual_seed(1234)
        first = adapter.encode(pixels)
        first_draw = model.draw.clone()
        second = adapter.encode(pixels)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.device.type, "cpu")
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(torch.equal(first_draw, model.draw))
        self.assertTrue(torch.equal(torch.randn(4), expected_after))
        self.assertTrue(torch.equal(first, first.half().float()))
        self.assertFalse(model.tiling_during_encode)
        self.assertTrue(model.parallel_tiling)
        self.assertEqual(model.dtype_during_encode, torch.float32)
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertEqual(
            stats.inputs,
            [("cpu", torch.float32), ("cpu", torch.float32)],
        )

    def test_audio_vae_keeps_stereo_posterior_mean_and_rejects_unknown_layout(self):
        import torch

        from minimax_h3_nodes.runtime.vae_adapter import (
            H3VAEError,
            H3LatentStats,
            MiniMaxH3AudioVAEAdapter,
        )

        class Model(torch.nn.Module):
            attn_proj = False

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
                self.preprocessed = None

            def preprocess(self, value, _sample_rate):
                self.preprocessed = value.detach().clone()
                return value

            @staticmethod
            def encoder(value):
                return value.expand(-1, 32, -1)

            @staticmethod
            def mean_proj(value):
                return value

        class RecordingStats:
            def __init__(self):
                self.inputs = []

            def normalize(self, value):
                self.inputs.append((value.device.type, value.dtype))
                return value

        model = Model()
        stats = RecordingStats()
        adapter = MiniMaxH3AudioVAEAdapter(
            model,
            stats,
            component_dir=Path("."),
            compute_dtype=torch.float32,
        )
        waveform = torch.stack(
            [torch.zeros(4), torch.full((4,), 3.0)]
        ).unsqueeze(0)
        latent = adapter.encode(waveform, sample_rate=32_000)
        self.assertEqual(tuple(latent.shape), (2, 32, 4))
        self.assertEqual(latent.device.type, "cpu")
        self.assertEqual(latent.dtype, torch.float32)
        self.assertEqual(stats.inputs, [("cpu", torch.float32)])
        self.assertTrue(torch.equal(model.preprocessed[:, 0], waveform[0]))

        multichannel = torch.stack(
            [torch.zeros(4), torch.full((4,), 3.0), torch.full((4,), 9.0)]
        ).unsqueeze(0)
        preprocessed = model.preprocessed
        with self.assertRaisesRegex(H3VAEError, "no channel-layout metadata"):
            adapter.encode(multichannel, sample_rate=32_000)
        self.assertIs(model.preprocessed, preprocessed)

        class TimeMajorModel(Model):
            @staticmethod
            def mean_proj(value):
                return value.transpose(1, 2)

        time_major = MiniMaxH3AudioVAEAdapter(
            TimeMajorModel(),
            RecordingStats(),
            component_dir=Path("."),
            compute_dtype=torch.float32,
        )
        canonical = time_major.encode(waveform, sample_rate=32_000)
        self.assertEqual(tuple(canonical.shape), (2, 32, 4))
        self.assertTrue(torch.equal(canonical, latent))

    def test_multichannel_comfy_audio_fails_at_reference_node_boundary(self):
        import torch

        from minimax_h3_nodes.nodes import MiniMaxH3Ref2VAAudioReference

        audio = {
            "waveform": torch.zeros(1, 6, 32_000),
            "sample_rate": 32_000,
        }
        with self.assertRaisesRegex(ValueError, "不含 channel layout"):
            MiniMaxH3Ref2VAAudioReference().append(audio)

    def test_video_decode_waits_until_encode_restores_model_state(self):
        import threading
        import torch

        from minimax_h3_nodes.runtime.vae_adapter import (
            MiniMaxH3VideoVAEAdapter,
        )

        encode_entered = threading.Event()
        release_encode = threading.Event()
        decode_entered = threading.Event()
        errors = []

        class Processor:
            @staticmethod
            def get_suitable_video_length(frames):
                return frames

            @staticmethod
            def _align_to_total_patch_size(height, width):
                return height, width

            @staticmethod
            def _crop_to_align(value, height, width, is_video=False):
                return value[..., :height, :width]

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
                self.processor = Processor()
                self.parallel_tiling = True
                self.decode_state = None

            @staticmethod
            def transform(value):
                return value

            def encode_base(self, value, process_image=False):
                encode_entered.set()
                if not release_encode.wait(5):
                    raise TimeoutError("test did not release video encode")
                return torch.zeros(
                    int(value.shape[0]), 24, 1, 2, 2, device=value.device
                )

            def decode_base(self, value, frame_num=None):
                self.decode_state = (self.weight.dtype, self.parallel_tiling)
                decode_entered.set()
                return value

        class IdentityStats:
            @staticmethod
            def normalize(value):
                return value

        model = Model()
        adapter = MiniMaxH3VideoVAEAdapter(
            model,
            IdentityStats(),
            component_dir=Path("."),
            compute_dtype=torch.float32,
        )

        def run_encode():
            try:
                adapter.encode(torch.zeros(1, 32, 32, 3))
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        def run_decode():
            try:
                adapter.decode_base(torch.zeros(1, 24, 1, 2, 2))
            except Exception as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        encode_thread = threading.Thread(target=run_encode)
        decode_thread = threading.Thread(target=run_decode)
        encode_thread.start()
        try:
            self.assertTrue(encode_entered.wait(5))
            self.assertEqual(model.weight.dtype, torch.float32)
            self.assertFalse(model.parallel_tiling)
            decode_thread.start()
            self.assertFalse(decode_entered.wait(0.1))
        finally:
            release_encode.set()
            encode_thread.join(5)
            if decode_thread.ident is not None:
                decode_thread.join(5)
        self.assertFalse(encode_thread.is_alive())
        self.assertFalse(decode_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(decode_entered.is_set())
        self.assertEqual(model.decode_state, (torch.float16, True))

    def test_video_and_audio_decode_model_calls_use_shared_runtime_lock(self):
        import torch

        from minimax_h3_nodes.runtime import vae_adapter
        from minimax_h3_nodes.runtime.vae_adapter import (
            H3LatentStats,
            MiniMaxH3AudioVAEAdapter,
            MiniMaxH3VideoVAEAdapter,
        )

        class Guard:
            def __init__(self):
                self.depth = 0

            def __enter__(self):
                self.depth += 1
                return self

            def __exit__(self, *_args):
                self.depth -= 1

            def assert_held(self):
                if self.depth <= 0:
                    raise AssertionError("VAE model action escaped the runtime lock")

        guard = Guard()

        class VideoModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def decode_base(self, value, frame_num=None):
                guard.assert_held()
                return value

        class AudioModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def decode(self, value):
                guard.assert_held()
                return value

        video = MiniMaxH3VideoVAEAdapter(
            VideoModel(),
            H3LatentStats(mean=(0.0,) * 24, std=(1.0,) * 24),
            component_dir=Path("."),
            compute_dtype=torch.float32,
        )
        audio = MiniMaxH3AudioVAEAdapter(
            AudioModel(),
            H3LatentStats(mean=(0.0,) * 32, std=(1.0,) * 32),
            component_dir=Path("."),
            compute_dtype=torch.float32,
        )
        with mock.patch.object(
            vae_adapter, "_VAE_CONDITION_ENCODE_LOCK", guard
        ):
            video.decode_base(torch.zeros(1, 24, 1, 1, 1))
            audio.decode_native(torch.zeros(2, 32, 1))


if __name__ == "__main__":
    unittest.main()
