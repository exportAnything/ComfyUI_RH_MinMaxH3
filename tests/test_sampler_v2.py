import importlib.util
import unittest
from contextlib import contextmanager
from unittest import mock


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "artifact container has no PyTorch")
class ConditionalSamplerTests(unittest.TestCase):
    @contextmanager
    def _tiny_runtime(self, sampling):
        """Bypass production geometry/device gates for tiny CPU sampler tests."""

        passthrough = lambda value, **_kwargs: dict(value)
        with (
            mock.patch.object(
                sampling, "_require_accelerated_sampler_device", return_value=None
            ),
            mock.patch.object(
                sampling, "validate_conditioning_v2", side_effect=passthrough
            ),
            mock.patch.object(
                sampling, "validate_av_latent_v2", side_effect=passthrough
            ),
        ):
            yield

    def _fixture(self):
        import torch

        text_len = 2
        seq_len = 8
        visual_clean = torch.full((1, 96), 0.25, dtype=torch.float32)
        audio_clean = torch.full((2, 32), -0.5, dtype=torch.float32)
        packed = {
            "seq_len": torch.tensor(seq_len),
            "img_pos": torch.tensor([2, 7]),
            "audio_pos": torch.tensor([3, 4, 5, 6]),
            "text_pos": torch.tensor([0, 1]),
            "update_mask": torch.tensor([False, True]),
            "audio_update_mask": torch.tensor([False, False, True, True]),
            "img_position_ids": torch.zeros(seq_len, 3, dtype=torch.float64),
            "token_tags": torch.tensor([1, 1, 0, 2, 2, 2, 2, 0]),
            "cu_seqlens": torch.tensor([0, seq_len, seq_len], dtype=torch.int32),
            "visual_cond_rows": visual_clean,
            "audio_ref_rows": audio_clean,
            "visual_condition_shapes": [(1, 2, 2)],
            "audio_reference_t": [1],
            "latent_shape": (1, 2, 2, 24),
            "audio_shape": (2, 32, 1),
            "task": "ref2va",
            "partition": "ref2va",
            "condition_blocks": (
                {
                    "kind": "video_audio",
                    "condition_index": 0,
                    "latent_t": 1,
                    "latent_h": 2,
                    "latent_w": 2,
                    "ref_audio_t": 1,
                },
            ),
        }
        conditioning = {
            "schema": "minimax_h3_conditioning/v2",
            "task": "ref2va",
            "partition": "ref2va",
            "prompt_embeds": torch.zeros(text_len, 5120),
            "conditions": [{"type": "video_audio", "condition_index": 0}],
        }
        target = {
            "schema": "minimax_h3_target/v2",
            "task": "ref2va",
            "partition": "ref2va",
            "video_latent_t": 1,
            "video_latent_h": 2,
            "video_latent_w": 2,
            "audio_latent_t": 1,
        }
        av_latent = {
            "schema": "minimax_h3_av_latent/v2",
            "task": "ref2va",
            "partition": "ref2va",
            "target": target,
            "video": torch.zeros(1, 24, 1, 2, 2),
            "audio": torch.zeros(2, 32, 1),
            "sampled": False,
        }
        return packed, conditioning, av_latent, visual_clean, audio_clean

    def test_condition_noise_restarts_rng_per_ordered_reference(self):
        import torch

        from minimax_h3_nodes.sampling import (
            noise_audio_reference_rows,
            noise_visual_condition_rows,
        )

        visual_clean = torch.zeros(2, 96)
        visual_before = visual_clean.clone()
        visual = noise_visual_condition_rows(
            visual_clean,
            condition_shapes=[(1, 2, 2), (1, 2, 2)],
            target_latent_t=1,
            seed=19,
            noise_level=0.0,
        )
        self.assertTrue(torch.equal(visual[0], visual[1]))
        self.assertTrue(torch.equal(visual_clean, visual_before))
        self.assertNotEqual(visual.data_ptr(), visual_clean.data_ptr())

        audio_clean = torch.zeros(4, 32)
        audio_before = audio_clean.clone()
        audio = noise_audio_reference_rows(
            audio_clean,
            reference_audio_t=[1, 1],
            seed=19,
            noise_level=0.0,
        )
        self.assertTrue(torch.equal(audio[:2], audio[2:]))
        self.assertTrue(torch.equal(audio_clean, audio_before))
        self.assertNotEqual(audio.data_ptr(), audio_clean.data_ptr())

    def test_full_rows_are_pinned_and_only_targets_are_returned(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, visual_clean, audio_clean = self._fixture()
        visual_before = visual_clean.clone()
        audio_before = audio_clean.clone()

        class RecordingTransformer:
            device = torch.device("cpu")

            def __init__(self):
                self.visual = []
                self.audio = []
                self.visual_floor = []
                self.audio_floor = []

            def __call__(self, **kwargs):
                img_pos = kwargs["img_pos_info"]["position_ids"]
                audio_pos = kwargs["audio_pos_info"]["position_ids"]
                visual_rows = kwargs["x"][0].index_select(0, img_pos)
                audio_rows = kwargs["audio_x"][0].index_select(0, audio_pos)
                self.visual.append(visual_rows[0].clone())
                self.audio.append(audio_rows[:2].clone())

                inverse = kwargs["inverse_indices"]
                unique = kwargs["unique_timesteps"]
                self.visual_floor.append(float(unique[inverse[img_pos[0]]]))
                self.audio_floor.append(float(unique[inverse[audio_pos[0]]]))

                # Deliberately corrupt the temporary model input.  The next
                # step must reconstruct it from sampler state and pinned clones.
                kwargs["x"][0, img_pos[0]].fill_(1234.0)
                kwargs["audio_x"][0, audio_pos[0]].fill_(-1234.0)
                return torch.zeros_like(visual_rows), torch.zeros_like(audio_rows)

        model = RecordingTransformer()
        expected_visual = sampling.noise_visual_condition_rows(
            visual_clean,
            condition_shapes=[(1, 2, 2)],
            target_latent_t=1,
            seed=7,
            noise_level=0.999,
        )
        with self._tiny_runtime(sampling):
            output = sampling.sample_h3(
                transformer=model,
                conditioning=conditioning,
                av_latent=av_latent,
                packed=packed,
                seed=7,
                sigma_points=3,
                video_shift=12.0,
                audio_shift=3.0,
            )

        self.assertEqual(len(model.visual), 2)
        for observed in model.visual:
            self.assertTrue(torch.equal(observed, expected_visual[0]))
        for observed in model.audio:
            self.assertTrue(torch.equal(observed, audio_clean))
        for observed in model.visual_floor:
            self.assertAlmostEqual(observed, 0.999, places=6)
        for observed in model.audio_floor:
            self.assertAlmostEqual(observed, 1.0, places=6)
        self.assertTrue(torch.equal(visual_clean, visual_before))
        self.assertTrue(torch.equal(audio_clean, audio_before))
        self.assertEqual(tuple(output["video"].shape), (1, 24, 1, 2, 2))
        self.assertEqual(tuple(output["audio"].shape), (2, 32, 1))
        self.assertLess(float(output["video"].abs().max()), 1000.0)
        self.assertLess(float(output["audio"].abs().max()), 1000.0)
        self.assertTrue(output["sampled"])

    def test_ref2va_is_exempt_from_generic_slow_step_abort(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, _visual, _audio = self._fixture()

        class SpyTelemetry:
            instances = []

            def __init__(self, **_kwargs):
                self.aborted_reason = None
                self.step_abort_exempt = []
                self.__class__.instances.append(self)

            @contextmanager
            def stage(self, _name):
                yield

            @contextmanager
            def denoise_step(self, step, *, abort_exempt=False):
                self.step_abort_exempt.append((step, abort_exempt))
                yield

            def note(self, **_kwargs):
                pass

            def summary(self):
                return {}

        class ZeroVelocity:
            device = torch.device("cpu")

            def __call__(self, **kwargs):
                video_count = int(kwargs["update_mask"].numel())
                audio_count = int(kwargs["update_audio_mask"].numel())
                return torch.zeros(video_count, 96), torch.zeros(audio_count, 32)

        with (
            self._tiny_runtime(sampling),
            mock.patch("minimax_h3_nodes.runtime.telemetry.H3Telemetry", SpyTelemetry),
        ):
            sampling.sample_h3(
                transformer=ZeroVelocity(),
                conditioning=conditioning,
                av_latent=av_latent,
                packed=packed,
                seed=7,
                sigma_points=3,
                video_shift=12.0,
                audio_shift=3.0,
            )

        self.assertEqual(SpyTelemetry.instances[0].step_abort_exempt, [(0, True), (1, True)])

    def test_noncanonical_anchor_order_fails_before_model_forward(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, _visual, _audio = self._fixture()
        packed["update_mask"] = torch.tensor([True, False])

        class NeverCalled:
            device = torch.device("cpu")

            def __call__(self, **_kwargs):
                raise AssertionError("model must not run for an invalid layout")

        with self._tiny_runtime(sampling):
            with self.assertRaisesRegex(ValueError, "condition rows.*target rows"):
                sampling.sample_h3(
                    transformer=NeverCalled(),
                    conditioning=conditioning,
                    av_latent=av_latent,
                    packed=packed,
                    seed=7,
                    sigma_points=2,
                    video_shift=12.0,
                    audio_shift=3.0,
                )

    def test_condition_shape_metadata_mismatch_fails(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, _visual, _audio = self._fixture()
        packed["visual_condition_shapes"] = [(1, 4, 4)]

        class NeverCalled:
            device = torch.device("cpu")

            def __call__(self, **_kwargs):
                raise AssertionError("model must not run for invalid condition rows")

        with self._tiny_runtime(sampling):
            with self.assertRaisesRegex(ValueError, "visual_condition_shapes.*不一致"):
                sampling.sample_h3(
                    transformer=NeverCalled(),
                    conditioning=conditioning,
                    av_latent=av_latent,
                    packed=packed,
                    seed=7,
                    sigma_points=2,
                    video_shift=12.0,
                    audio_shift=3.0,
                )

    def test_conditional_tasks_reject_unknown_or_v1_schemas(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, _visual, _audio = self._fixture()

        class NeverCalled:
            device = torch.device("cpu")

            def __call__(self, **_kwargs):
                raise AssertionError("schema failure must happen before model use")

        conditioning["schema"] = "unknown-conditioning-schema"
        with self.assertRaisesRegex(ValueError, "schema 必须同时"):
            sampling.sample_h3(
                transformer=NeverCalled(),
                conditioning=conditioning,
                av_latent=av_latent,
                packed=packed,
                seed=7,
                sigma_points=2,
                video_shift=12.0,
                audio_shift=3.0,
            )

        conditioning["schema"] = "minimax_h3_conditioning/v1"
        av_latent["schema"] = "minimax_h3_av_latent/v1"
        with self.assertRaisesRegex(ValueError, "FL2VA/Ref2VA 只接受"):
            sampling.sample_h3(
                transformer=NeverCalled(),
                conditioning=conditioning,
                av_latent=av_latent,
                packed=packed,
                seed=7,
                sigma_points=2,
                video_shift=12.0,
                audio_shift=3.0,
            )

    def test_condition_blocks_are_cross_checked_in_request_order(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, _visual, _audio = self._fixture()
        packed["condition_blocks"][0]["condition_index"] = 1

        class NeverCalled:
            device = torch.device("cpu")

            def __call__(self, **_kwargs):
                raise AssertionError("invalid block metadata must fail before DiT")

        with self._tiny_runtime(sampling):
            with self.assertRaisesRegex(ValueError, "condition_index.*期望 0"):
                sampling.sample_h3(
                    transformer=NeverCalled(),
                    conditioning=conditioning,
                    av_latent=av_latent,
                    packed=packed,
                    seed=7,
                    sigma_points=2,
                    video_shift=12.0,
                    audio_shift=3.0,
                )

    def test_nonfinite_velocity_and_target_stop_before_decode(self):
        import torch

        import minimax_h3_nodes.sampling as sampling

        packed, conditioning, av_latent, _visual, _audio = self._fixture()

        class NaNVelocity:
            device = torch.device("cpu")

            def __call__(self, **kwargs):
                video_count = int(kwargs["update_mask"].numel())
                audio_count = int(kwargs["update_audio_mask"].numel())
                return (
                    torch.full((video_count, 96), torch.nan),
                    torch.zeros(audio_count, 32),
                )

        with self._tiny_runtime(sampling):
            with self.assertRaisesRegex(
                FloatingPointError,
                "task=ref2va step=1 modality=video transformer velocity",
            ):
                sampling.sample_h3(
                    transformer=NaNVelocity(),
                    conditioning=conditioning,
                    av_latent=av_latent,
                    packed=packed,
                    seed=7,
                    sigma_points=2,
                    video_shift=12.0,
                    audio_shift=3.0,
                )

        class FiniteVelocity(NaNVelocity):
            def __call__(self, **kwargs):
                video_count = int(kwargs["update_mask"].numel())
                audio_count = int(kwargs["update_audio_mask"].numel())
                return torch.zeros(video_count, 96), torch.zeros(audio_count, 32)

        real_euler = sampling.euler_eta0_step

        def corrupt_video_target(state, denoised, **kwargs):
            output = real_euler(state, denoised, **kwargs)
            if int(output.shape[1]) == 96:
                output = output.clone()
                output[0, 0] = torch.inf
            return output

        with (
            self._tiny_runtime(sampling),
            mock.patch.object(
                sampling, "euler_eta0_step", side_effect=corrupt_video_target
            ),
        ):
            with self.assertRaisesRegex(
                FloatingPointError,
                "task=ref2va step=1 modality=video target latent",
            ):
                sampling.sample_h3(
                    transformer=FiniteVelocity(),
                    conditioning=conditioning,
                    av_latent=av_latent,
                    packed=packed,
                    seed=7,
                    sigma_points=2,
                    video_shift=12.0,
                    audio_shift=3.0,
                )


if __name__ == "__main__":
    unittest.main()
