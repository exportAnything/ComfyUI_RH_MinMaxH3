from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import save_file
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from minimax_h3_nodes.runtime.vae_adapter import _impl as vae


def _stats(channels: int, *, mean: float = 0.0) -> dict[str, list[float]]:
    return {
        "latents_mean": [mean] * channels,
        "latents_std": [1.0] * channels,
    }


def _video_config() -> dict:
    return {
        **_stats(vae.VIDEO_LATENT_CHANNELS, mean=0.25),
        "vae_clip_length": 17,
        "vae_token_drop": 3,
        "source_config": dict(vae._VIDEO_SOURCE_CONTRACT),
    }


def _audio_config() -> dict:
    return {
        **_stats(vae.AUDIO_LATENT_CHANNELS, mean=-0.5),
        "sample_rate": vae.AUDIO_SAMPLE_RATE,
        "latent_channels": vae.AUDIO_LATENT_CHANNELS,
        "output_channel": vae.AUDIO_OUTPUT_CHANNELS,
    }


def _write_native_file(
    path: Path,
    metadata_key: str,
    config: dict,
    *,
    dtype: torch.dtype,
) -> None:
    channels = len(config["latents_mean"])
    save_file(
        {
            "probe.weight": torch.ones((1,), dtype=dtype),
            "latents_mean": torch.zeros((channels,), dtype=dtype),
            "latents_std": torch.ones((channels,), dtype=dtype),
        },
        str(path),
        metadata={metadata_key: json.dumps(config)},
    )


class SingleFileVAEAdapterTests(unittest.TestCase):
    def test_direct_video_file_uses_embedded_config_and_fp16_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "minimax_h3_video_vae_fp16.safetensors"
            _write_native_file(
                checkpoint,
                "minimax_h3_video_vae",
                _video_config(),
                dtype=torch.float16,
            )
            fake_model = torch.nn.Linear(1, 1)
            with mock.patch.object(
                vae,
                "_construct_and_load",
                return_value=fake_model,
            ) as construct:
                adapter = vae.load_video_vae(checkpoint)

            self.assertEqual(adapter.component_dir, checkpoint.parent)
            self.assertEqual(adapter.stats.mean[0], 0.25)
            self.assertEqual(adapter.compute_dtype, torch.float16)
            self.assertEqual(construct.call_args.args[1], [checkpoint.resolve()])
            self.assertEqual(construct.call_args.kwargs["weight_dtype"], torch.float16)
            self.assertEqual(
                construct.call_args.kwargs["ignored_checkpoint_keys"],
                vae._SINGLE_FILE_CONFIG_TENSOR_KEYS,
            )

    def test_audio_descriptor_uses_metadata_and_folded_weight_norm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            component = root / "audio_vae"
            component.mkdir()
            (component / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "MiniMaxH3AudioVAE",
                        "backend": "comfy_native_safetensors",
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "minimax_h3_audio_vae_fp32.safetensors"
            _write_native_file(
                checkpoint,
                "minimax_h3_audio_vae",
                _audio_config(),
                dtype=torch.float32,
            )
            fake_model = torch.nn.Linear(1, 1)
            with (
                mock.patch.object(
                    vae,
                    "_construct_and_load",
                    return_value=fake_model,
                ) as construct,
                mock.patch.object(
                    vae,
                    "_audio_vae_factory",
                    wraps=vae._audio_vae_factory,
                ) as factory,
            ):
                adapter = vae.load_audio_vae(
                    component,
                    weight_files=[checkpoint],
                )

            factory.assert_called_once_with(folded_weight_norm=True)
            self.assertEqual(adapter.component_dir, component.resolve())
            self.assertEqual(adapter.stats.mean[0], -0.5)
            self.assertEqual(construct.call_args.kwargs["weight_dtype"], torch.float32)

    def test_runninghub_directory_keeps_external_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            component = Path(temp_dir) / "video_vae"
            source = component / "source"
            source.mkdir(parents=True)
            (component / "config.json").write_text(
                json.dumps(
                    {
                        "_class_name": "MiniMaxH3VideoVAE",
                        **_stats(vae.VIDEO_LATENT_CHANNELS, mean=0.75),
                    }
                ),
                encoding="utf-8",
            )
            (source / "config.json").write_text(
                json.dumps(vae._VIDEO_SOURCE_CONTRACT),
                encoding="utf-8",
            )
            checkpoint = source / "model.safetensors"
            save_file({"probe.weight": torch.ones(1)}, str(checkpoint))
            with mock.patch.object(
                vae,
                "_construct_and_load",
                return_value=torch.nn.Linear(1, 1),
            ) as construct:
                adapter = vae.load_video_vae(component)

            self.assertEqual(adapter.stats.mean[0], 0.75)
            self.assertEqual(construct.call_args.kwargs["weight_dtype"], torch.float32)
            self.assertEqual(construct.call_args.kwargs["ignored_checkpoint_keys"], ())

    def test_direct_file_rejects_other_component_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "wrong.safetensors"
            _write_native_file(
                checkpoint,
                "minimax_h3_video_vae",
                _video_config(),
                dtype=torch.float16,
            )
            with self.assertRaisesRegex(
                vae.H3VAEError,
                "minimax_h3_video_vae.*minimax_h3_audio_vae",
            ):
                vae.load_audio_vae(checkpoint)

    def test_strict_loader_ignores_metadata_stat_tensors_only_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "tiny.safetensors"
            expected_weight = torch.tensor([[2.0]])
            expected_bias = torch.tensor([3.0])
            save_file(
                {
                    "weight": expected_weight,
                    "bias": expected_bias,
                    "latents_mean": torch.zeros(1),
                    "latents_std": torch.ones(1),
                },
                str(checkpoint),
            )
            model = vae._construct_and_load(
                lambda: torch.nn.Linear(1, 1),
                [checkpoint],
                component_name="video_vae",
                device=torch.device("cpu"),
                weight_dtype=torch.float32,
                strict=True,
                low_memory=False,
                ignored_checkpoint_keys=vae._SINGLE_FILE_CONFIG_TENSOR_KEYS,
            )

            torch.testing.assert_close(model.weight, expected_weight)
            torch.testing.assert_close(model.bias, expected_bias)

    def test_folded_audio_path_removes_weight_norm_parametrization(self) -> None:
        model = torch.nn.Sequential(weight_norm(torch.nn.Conv1d(1, 1, 1)))
        self.assertTrue(parametrize.is_parametrized(model[0], "weight"))

        removed = vae._remove_weight_norm_parametrizations(model)

        self.assertEqual(removed, 1)
        self.assertFalse(parametrize.is_parametrized(model[0], "weight"))
        self.assertIn("0.weight", model.state_dict())


if __name__ == "__main__":
    unittest.main()
