from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from minimax_h3_nodes.runtime.model_loader import _impl as loader
from minimax_h3_nodes.runtime.attention import prepare_checkpoint_tensor
from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
from minimax_h3_nodes.runtime.dit._impl import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTConfig,
)


INSTALLED_FP8 = Path(
    r"C:\ComfyUI\app\models\diffusion_models\minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
)


def _official_curve_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "video_patch_proj.weight": (5376, 96),
        "audio_patch_proj.weight": (5376, 32),
        "final_layer.video_out.weight": (96, 5376),
        "final_layer.audio_out.weight": (32, 5376),
        "blocks.0.attn.q_norm.weight": (128,),
        "blocks.0.attn.qkv_proj.weight": (21504, 5376),
        "blocks.0.mlp.fc1.weight": (28672, 5376),
        "condition_proj.weight": (5376, 5120),
        "rope.inv_freq": (16,),
        "adaln_t_table": (1025, 8),
    }
    for index in range(50):
        shapes[f"blocks.{index}.norm1.weight"] = (5376,)
    for index in range(2):
        shapes[f"token_refiner.blocks.{index}.norm1.weight"] = (5376,)
    return shapes


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config).encode("utf-8")),
        dtype=torch.uint8,
    )


class HeaderConfigTests(unittest.TestCase):
    def test_official_curve_config_is_inferred_from_shapes(self):
        config = loader._infer_transformer_config_from_shapes(
            _official_curve_shapes()
        )

        self.assertEqual(config["_class_name"], "MiniMaxH3DiTModel")
        self.assertEqual(config["num_layers"], 50)
        self.assertEqual(config["token_refiner_num_layers"], 2)
        self.assertEqual(config["hidden_size"], 5376)
        self.assertEqual(config["num_attention_heads"], 56)
        self.assertEqual(config["ffn_hidden_size"], 14336)
        self.assertEqual(config["latents_dim"], 24)
        self.assertEqual(config["audio_latents_dim"], 32)
        self.assertEqual(config["text_dim"], 5120)
        self.assertEqual(config["adaln_curve_grid"], 1025)
        self.assertEqual(config["time_embed_dim"], 8)
        loader._validate_transformer_config(config, Path("header.safetensors"))
        loader._validate_adaln_curve_contract(
            config,
            Path("header.safetensors"),
            (1025, 8),
        )

    def test_noncontiguous_blocks_are_rejected(self):
        shapes = _official_curve_shapes()
        for key in tuple(shapes):
            if key.startswith("blocks.49."):
                del shapes[key]
        shapes["blocks.50.norm1.weight"] = (5376,)

        with self.assertRaisesRegex(loader.H3ComponentError, "not contiguous"):
            loader._infer_transformer_config_from_shapes(shapes)

    def test_native_descriptor_retains_partition_gate(self):
        descriptor = {
            "_class_name": "MiniMaxH3Transformer3DModel",
            "backend": "comfy_native_safetensors",
            "partition": "ref2va",
        }

        self.assertTrue(
            loader._validate_native_config_descriptor(
                descriptor,
                Path("config.json"),
                "ref2va",
            )
        )
        with self.assertRaisesRegex(loader.H3ComponentError, "expected 'fl2va'"):
            loader._validate_native_config_descriptor(
                descriptor,
                Path("config.json"),
                "fl2va",
            )


class QuantMarkerTests(unittest.TestCase):
    def test_curve_adaln_fp32_bypasses_bound_mixed_precision_linear(self):
        class FakeOperations:
            class Linear(torch.nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.bound_dtype = torch.bfloat16

        config = MiniMaxH3DiTConfig(
            adaln_curve_grid=1025,
            time_embed_dim=8,
        )
        projection = MiniMaxH3AdalnProj(
            config,
            config.adaln_out_features,
            expand_ratio=6,
            modality_count=3,
            dtype=torch.float32,
            device="meta",
            operations=FakeOperations,
            apply_silu=False,
        )

        self.assertIs(type(projection.linear), torch.nn.Linear)
        self.assertEqual(projection.linear.weight.dtype, torch.float32)
        self.assertEqual(projection.linear.bias.dtype, torch.float32)

    def test_fp8_input_scale_is_an_explicit_fp32_parameter(self):
        class FakeDiT:
            use_adaln_curves = True

            @staticmethod
            def named_parameters():
                return iter(
                    (
                        (
                            "blocks.0.attn.qkv_proj.input_scale",
                            torch.nn.Parameter(torch.tensor(0.5)),
                        ),
                    )
                )

        names = MiniMaxH3DiTModel.fp32_param_names(FakeDiT())

        self.assertIn("blocks.0.attn.qkv_proj.input_scale", names)

    def test_fp8_scalar_qkv_scale_is_not_row_reordered(self):
        scale = torch.tensor(0.125, dtype=torch.float32)
        prepared = prepare_checkpoint_tensor(
            "blocks.0.attn.qkv_proj.weight_scale",
            scale,
            config={},
        )
        self.assertEqual(tuple(prepared.shape), ())
        self.assertEqual(float(prepared), float(scale))

    def test_scaled_fp8_marker_is_accepted(self):
        bag = {
            "weight": torch.empty((2, 3), dtype=torch.float8_e4m3fn),
            "weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "input_scale": torch.tensor(0.5, dtype=torch.float32),
            "comfy_quant": _marker({"format": "float8_e4m3fn"}),
        }

        config = loader._quantized_linear_config("blocks.0.mlp.fc1.", bag)

        self.assertEqual(config, {"format": "float8_e4m3fn"})

    def test_fp8_full_precision_mm_flag_is_preserved(self):
        bag = {
            "weight": torch.empty((2, 3), dtype=torch.float8_e4m3fn),
            "weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "comfy_quant": _marker(
                {
                    "format": "float8_e4m3fn",
                    "full_precision_matrix_mult": True,
                }
            ),
        }

        config = loader._quantized_linear_config("blocks.0.mlp.fc2.", bag)

        self.assertTrue(config["full_precision_matrix_mult"])

    def test_existing_int8_convrot_marker_remains_accepted(self):
        bag = {
            "weight": torch.zeros((2, 3), dtype=torch.int8),
            "weight_scale": torch.ones(2, dtype=torch.float32),
            "comfy_quant": _marker(
                {
                    "format": loader.INT8_FORMAT,
                    "convrot": True,
                }
            ),
        }

        config = loader._quantized_linear_config("blocks.0.mlp.fc1.", bag)

        self.assertEqual(config["format"], loader.INT8_FORMAT)
        self.assertTrue(config["convrot"])

    def test_int8_without_convrot_is_still_rejected(self):
        bag = {
            "weight": torch.zeros((2, 3), dtype=torch.int8),
            "weight_scale": torch.ones(2, dtype=torch.float32),
            "comfy_quant": _marker({"format": loader.INT8_FORMAT}),
        }

        with self.assertRaisesRegex(loader.H3ComponentError, "convrot=true"):
            loader._quantized_linear_config("blocks.0.mlp.fc1.", bag)

    def test_header_quant_format_detection_reads_only_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mini-fp8.safetensors"
            save_file(
                {
                    "blocks.0.mlp.fc1.weight": torch.zeros(
                        (2, 3), dtype=torch.float8_e4m3fn
                    ),
                    "blocks.0.mlp.fc1.weight_scale": torch.tensor(0.25),
                    "blocks.0.mlp.fc1.comfy_quant": _marker(
                        {"format": "float8_e4m3fn"}
                    ),
                },
                str(path),
            )

            formats = loader._checkpoint_quant_formats(None, [path])

        self.assertEqual(formats, frozenset({"float8_e4m3fn"}))

    def test_fp8_weight_shape_is_checked_against_mixed_precision_linear(self):
        class FakeMixedPrecisionLinear:
            _orig_shape = (2, 3)
            weight = None
            bias = None

        bag = {
            "weight": torch.empty((2, 4), dtype=torch.float8_e4m3fn),
            "weight_scale": torch.tensor(0.25, dtype=torch.float32),
            "comfy_quant": _marker({"format": "float8_e4m3fn"}),
        }

        with self.assertRaisesRegex(loader.H3ComponentError, "Shape mismatch"):
            loader._flush_linear(
                FakeMixedPrecisionLinear(),
                "blocks.0.mlp.fc1.",
                bag,
                torch.device("cpu"),
            )

    def test_meta_assign_preserves_nonquantized_linear_dtype(self):
        class FakeMixedPrecisionLinear(torch.nn.Linear):
            pass

        module = FakeMixedPrecisionLinear(
            3,
            2,
            bias=True,
            device="meta",
            dtype=torch.float32,
        )
        bag = {
            "weight": torch.ones((2, 3), dtype=torch.float16),
            "bias": torch.ones(2, dtype=torch.float16),
        }

        quantized = loader._flush_linear(
            module,
            "blocks.0.adaln_proj.linear.",
            bag,
            torch.device("cpu"),
        )

        self.assertFalse(quantized)
        self.assertEqual(module.weight.dtype, torch.float32)
        self.assertEqual(module.bias.dtype, torch.float32)
        self.assertEqual(module.weight.device.type, "cpu")


@unittest.skipUnless(INSTALLED_FP8.is_file(), "local MiniMax-H3 FP8 file absent")
class InstalledCheckpointHeaderTests(unittest.TestCase):
    def test_installed_checkpoint_header_matches_supported_h3_fp8_contract(self):
        config = loader._infer_transformer_config(None, [INSTALLED_FP8])
        formats = loader._checkpoint_quant_formats(None, [INSTALLED_FP8])

        self.assertEqual(config["num_layers"], 50)
        self.assertEqual(config["token_refiner_num_layers"], 2)
        self.assertEqual(config["adaln_curve_grid"], 1025)
        self.assertEqual(config["time_embed_dim"], 8)
        self.assertEqual(formats, frozenset({"float8_e4m3fn"}))


if __name__ == "__main__":
    unittest.main()
