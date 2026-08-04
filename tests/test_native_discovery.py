from __future__ import annotations

import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from minimax_h3_nodes.api import _shared
from minimax_h3_nodes.api import loaders
from minimax_h3_nodes.runtime import h3_settings
from minimax_h3_nodes.runtime.components import _impl as components


class _FolderPathsStub:
    def __init__(self, root: Path) -> None:
        self.models_dir = str(root / "models")
        self._paths = {
            "diffusion_models": [str(root / "diffusion_models")],
            "unet": [str(root / "diffusion_models")],
            "text_encoders": [str(root / "text_encoders")],
            "clip": [str(root / "text_encoders")],
            "vae": [str(root / "vae")],
        }

    def get_folder_paths(self, kind: str):
        return self._paths[kind]


class NativeFilenameTests(unittest.TestCase):
    def test_standard_comfy_names_are_classified(self):
        expected = {
            "minimax_h3_fl2va_pruned_fp8_scaled.safetensors": (
                "transformer",
                "fl2va",
            ),
            "minimax_h3_ref2va_pruned_fp8_scaled.safetensors": (
                "transformer",
                "ref2va",
            ),
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": (
                "text_encoder",
                None,
            ),
            "minimax_h3_video_vae_fp16.safetensors": ("video_vae", None),
            "minimax_h3_audio_vae_fp32.safetensors": ("audio_vae", None),
        }
        for name, classification in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    h3_settings.classify_weight_filename(name), classification
                )
                self.assertTrue(h3_settings.is_comfy_native_weight_filename(name))

    def test_standard_folders_are_component_specific(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for name in ("diffusion_models", "text_encoders", "vae"):
                (root / name).mkdir()
            stub = _FolderPathsStub(root)
            transformer_roots = components.weights_root_paths(
                stub, kind="transformer"
            )
            text_roots = components.weights_root_paths(stub, kind="text_encoder")
            vae_roots = components.weights_root_paths(stub, kind="video_vae")
            self.assertIn((root / "diffusion_models").resolve(), transformer_roots)
            self.assertIn((root / "text_encoders").resolve(), text_roots)
            self.assertIn((root / "vae").resolve(), vae_roots)
            self.assertNotIn((root / "vae").resolve(), transformer_roots)


class NativeSelectionTests(unittest.TestCase):
    def test_nested_comfy_weight_uses_relative_dropdown_name(self):
        name = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            nested = root / "MiniMax" / "H3"
            nested.mkdir(parents=True)
            checkpoint = nested / name
            checkpoint.write_bytes(b"test")
            with mock.patch.dict(
                os.environ,
                {components.H3_WEIGHTS_ROOTS_ENV: str(root)},
            ):
                discovered = components.list_h3_weight_files(
                    "transformer", "ref2va"
                )
                self.assertIn(checkpoint.resolve(), discovered)
                choice = components.weight_file_choice_name(
                    checkpoint, "transformer"
                )
                self.assertEqual(choice, f"MiniMax/H3/{name}")
                self.assertEqual(
                    components.resolve_weight_file(
                        choice, "transformer", "ref2va"
                    ),
                    checkpoint.resolve(),
                )

    def test_native_components_share_partition_descriptor_root(self):
        names = {
            "transformer": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
            "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "video_vae": "minimax_h3_video_vae_fp16.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        }
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for name in names.values():
                (root / name).write_bytes(b"test")
            with mock.patch.dict(
                os.environ,
                {components.H3_WEIGHTS_ROOTS_ENV: str(root)},
            ):
                selections = {
                    kind: _shared._native_component_selection(
                        name, kind, "ref2va"
                    )
                    for kind, name in names.items()
                }
            descriptor_roots = {selection[0] for selection in selections.values()}
            self.assertEqual(len(descriptor_roots), 1)
            for kind, selection in selections.items():
                self.assertEqual(selection[1].name, kind)
                self.assertEqual(selection[2], (root / names[kind]).resolve())

    def test_ref2va_transformer_is_rejected_for_fl2va(self):
        name = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / name).write_bytes(b"test")
            with mock.patch.dict(
                os.environ,
                {components.H3_WEIGHTS_ROOTS_ENV: str(root)},
            ):
                with self.assertRaises(components.H3ComponentError):
                    _shared._native_component_selection(
                        name, "transformer", "fl2va"
                    )


class NativeWorkflowChoiceTests(unittest.TestCase):
    def test_example_loader_widgets_are_valid_native_choices(self):
        node_classes = {
            "RHMiniMaxH3DirectModelLoader": loaders.MiniMaxH3DirectModelLoader,
            "RHMiniMaxH3DirectTextEncoderLoader": loaders.MiniMaxH3DirectTextEncoderLoader,
            "RHMiniMaxH3DirectVAELoader": loaders.MiniMaxH3DirectVAELoader,
            "RHMiniMaxH3FL2VAModelLoader": loaders.MiniMaxH3FL2VAModelLoader,
            "RHMiniMaxH3FL2VATextEncoderLoader": loaders.MiniMaxH3FL2VATextEncoderLoader,
            "RHMiniMaxH3FL2VAVAELoader": loaders.MiniMaxH3FL2VAVAELoader,
            "RHMiniMaxH3Ref2VAModelLoader": loaders.MiniMaxH3Ref2VAModelLoader,
            "RHMiniMaxH3Ref2VATextEncoderLoader": loaders.MiniMaxH3Ref2VATextEncoderLoader,
            "RHMiniMaxH3Ref2VAVAELoader": loaders.MiniMaxH3Ref2VAVAELoader,
        }
        native_names = (
            h3_settings.NATIVE_FL2VA_DIT_FILENAME,
            h3_settings.NATIVE_REF2VA_DIT_FILENAME,
            h3_settings.NATIVE_TE_FILENAME,
            h3_settings.NATIVE_VIDEO_VAE_FILENAME,
            h3_settings.NATIVE_AUDIO_VAE_FILENAME,
        )
        workflow_root = Path(__file__).parents[1] / "examples" / "workflows"
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            for name in native_names:
                (root / name).write_bytes(b"test")
            with mock.patch.dict(
                os.environ,
                {components.H3_WEIGHTS_ROOTS_ENV: str(root)},
            ):
                for workflow_path in workflow_root.glob("*.json"):
                    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
                    for node in workflow["nodes"]:
                        node_class = node_classes.get(node["type"])
                        if node_class is None:
                            continue
                        required = node_class.INPUT_TYPES()["required"]
                        for value, (key, spec) in zip(
                            node["widgets_values"], required.items()
                        ):
                            choices = spec[0]
                            if isinstance(choices, list):
                                self.assertIn(
                                    value,
                                    choices,
                                    f"{workflow_path.name}: {node['type']}.{key}",
                                )


class NativeWrapperContractTests(unittest.TestCase):
    def test_native_v2_wrappers_validate_as_one_release(self):
        names = {
            "transformer": h3_settings.NATIVE_REF2VA_DIT_FILENAME,
            "text_encoder": h3_settings.NATIVE_TE_FILENAME,
            "video_vae": h3_settings.NATIVE_VIDEO_VAE_FILENAME,
            "audio_vae": h3_settings.NATIVE_AUDIO_VAE_FILENAME,
        }
        with tempfile.TemporaryDirectory() as raw_root:
            weights_root = Path(raw_root)
            for name in names.values():
                (weights_root / name).write_bytes(b"test")
            with mock.patch.dict(
                os.environ,
                {components.H3_WEIGHTS_ROOTS_ENV: str(weights_root)},
            ):
                selected = {
                    kind: _shared._native_component_selection(
                        name, kind, "ref2va"
                    )
                    for kind, name in names.items()
                }

            root = selected["transformer"][0]
            release = _shared._release_fingerprint(root, "ref2va", {})
            transformer_path, transformer_weights = selected["transformer"][1:3]
            transformer_component = _shared._component_fingerprint(
                release,
                "transformer",
                transformer_path,
                related_paths={"transformer_weights": transformer_weights},
            )
            model = {
                "schema": _shared.H3_MODEL_SCHEMA_V2,
                "handle": object(),
                "model_root": str(root),
                "partition": "ref2va",
                "task": "ref2va",
                "tasks": ("ref2va",),
                "transformer_path": str(transformer_path),
                "transformer_weights_path": str(transformer_weights),
                "release_metadata": {},
                "release_fingerprint": release,
                "component_fingerprint": transformer_component,
                "transformer_fingerprint": transformer_component,
            }

            text_path, text_weights = selected["text_encoder"][1:3]
            text_related = {
                "tokenizer": text_path,
                "processor": text_path,
                "text_encoder_weights": text_weights,
            }
            text_component = _shared._component_fingerprint(
                release,
                "text_encoder",
                text_path,
                related_paths=text_related,
            )
            text = {
                "schema": _shared.H3_TEXT_ENCODER_SCHEMA_V2,
                "handle": object(),
                "model_root": str(root),
                "partition": "ref2va",
                "task": "ref2va",
                "tasks": ("ref2va",),
                "text_encoder_path": str(text_path),
                "text_encoder_weights_path": str(text_weights),
                "tokenizer_path": str(text_path),
                "processor_path": str(text_path),
                "release_metadata": {},
                "release_fingerprint": release,
                "component_fingerprint": text_component,
                "text_encoder_fingerprint": text_component,
            }

            video_path, video_weights = selected["video_vae"][1:3]
            audio_path, audio_weights = selected["audio_vae"][1:3]
            vae_related = {
                "audio_vae": audio_path,
                "video_vae_weights": video_weights,
                "audio_vae_weights": audio_weights,
            }
            vae_component = _shared._component_fingerprint(
                release, "vae", video_path, related_paths=vae_related
            )
            vae = {
                "schema": _shared.H3_VAE_SCHEMA_V2,
                "bundle": object(),
                "model_root": str(root),
                "partition": "ref2va",
                "task": "ref2va",
                "tasks": ("ref2va",),
                "vae_path": str(video_path),
                "video_vae_path": str(video_path),
                "audio_vae_path": str(audio_path),
                "video_vae_weights_path": str(video_weights),
                "audio_vae_weights_path": str(audio_weights),
                "release_metadata": {},
                "release_fingerprint": release,
                "component_fingerprint": vae_component,
                "vae_fingerprint": vae_component,
            }

            clean_model = _shared._validate_component_wrapper(
                model,
                task="ref2va",
                label="h3_model",
                schemas=(_shared.H3_MODEL_SCHEMA_V2,),
            )
            clean_text = _shared._validate_component_wrapper(
                text,
                task="ref2va",
                label="h3_text_encoder",
                schemas=(_shared.H3_TEXT_ENCODER_SCHEMA_V2,),
            )
            clean_vae = _shared._validate_component_wrapper(
                vae,
                task="ref2va",
                label="h3_vae_bundle",
                schemas=(_shared.H3_VAE_SCHEMA_V2,),
            )
            self.assertEqual(
                _shared._require_same_release(
                    ("model", clean_model),
                    ("text", clean_text),
                    ("vae", clean_vae),
                ),
                release,
            )

    def test_external_weight_change_changes_component_fingerprint(self):
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            descriptor = root / "component"
            descriptor.mkdir()
            (descriptor / "config.json").write_text("{}", encoding="utf-8")
            weight = root / "weight.safetensors"
            weight.write_bytes(b"first")
            first = _shared._component_fingerprint(
                "release", "transformer", descriptor,
                related_paths={"transformer_weights": weight},
            )
            weight.write_bytes(b"second-and-different")
            second = _shared._component_fingerprint(
                "release", "transformer", descriptor,
                related_paths={"transformer_weights": weight},
            )
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
