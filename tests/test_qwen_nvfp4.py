from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from minimax_h3_nodes.runtime.components import H3ComponentError
from minimax_h3_nodes.runtime.qwen_encoder.encoder import (
    MiniMaxH3ComfyTextEncoder,
)
from minimax_h3_nodes.runtime.qwen_encoder.loading import (
    _NATIVE_NVFP4_PROJECTIONS,
    _native_nvfp4_checkpoint_profile,
    _validate_native_nvfp4_profile,
)


def _marker(config: dict) -> torch.Tensor:
    return torch.tensor(
        list(json.dumps(config, sort_keys=True).encode("utf-8")), dtype=torch.uint8
    )


def _native_checkpoint_tensors() -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.comfy_quant": _marker(
            {"format": "int8_tensorwise"}
        ),
        "model.embed_tokens.weight": torch.zeros(1, dtype=torch.int8),
        "model.embed_tokens.weight_scale": torch.ones(1, dtype=torch.float32),
        "visual.deepstack_merger_list.0.norm.weight": torch.ones(
            1, dtype=torch.bfloat16
        ),
    }
    for layer in range(50):
        for projection in _NATIVE_NVFP4_PROJECTIONS:
            prefix = f"model.layers.{layer}.{projection}."
            tensors[f"{prefix}comfy_quant"] = _marker(
                {"format": "nvfp4", "full_precision_matrix_mult": True}
            )
            tensors[f"{prefix}weight"] = torch.zeros(1, dtype=torch.uint8)
            tensors[f"{prefix}weight_scale"] = torch.ones(1, dtype=torch.float8_e4m3fn)
            tensors[f"{prefix}weight_scale_2"] = torch.ones((), dtype=torch.float32)
    tensors["model.layers.0.self_attn.o_proj.pre_quant_scale"] = torch.ones(
        1, dtype=torch.bfloat16
    )
    return tensors


class _FakeClip:
    def __init__(self) -> None:
        self.tokenizer = object()

    def tokenize(self, prompt, *, images=None, minimax_ref_items=None):
        entries = []
        if images:
            for image in images:
                entries.extend(
                    [
                        (151652, 1.0),
                        ({"type": "image", "data": image}, 1.0),
                        (151653, 1.0),
                    ]
                )
        if minimax_ref_items:
            for item in minimax_ref_items:
                if item["type"] == "image":
                    entries.extend(
                        [
                            (151652, 1.0),
                            ({"type": "image", "data": item["data"]}, 1.0),
                            (151653, 1.0),
                        ]
                    )
        entries.append((42, 1.0))
        return {"qwen3vl_32b": [entries]}

    def encode_from_tokens(self, tokens, **_kwargs):
        entries = next(iter(tokens.values()))[0]
        tags = []
        for entry in entries:
            value = entry[0]
            if isinstance(value, dict):
                tags.extend([0] * 4)
            elif value in (151652, 151653):
                tags.append(0)
            else:
                tags.append(1)
        return {
            "cond": torch.zeros((1, len(tags), 5120), dtype=torch.bfloat16),
            "minimax_token_tags": torch.tensor(tags, dtype=torch.long),
        }


class NativeCheckpointProfileTests(unittest.TestCase):
    def test_accepts_native_nvfp4_awq_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3vl_nvfp4.safetensors"
            save_file(_native_checkpoint_tensors(), str(path))
            profile = _native_nvfp4_checkpoint_profile([path], checkpoint=path)

        self.assertIsNotNone(profile)
        self.assertEqual(profile["formats"], {"int8_tensorwise": 1, "nvfp4": 350})
        self.assertEqual(profile["pre_quant_scale_count"], 1)
        self.assertEqual(set(profile["layer_marker_counts"]), set(range(50)))

    def test_rejects_wrong_projection_even_when_counts_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3vl_nvfp4.safetensors"
            save_file(_native_checkpoint_tensors(), str(path))
            profile = _native_nvfp4_checkpoint_profile([path], checkpoint=path)

        altered = dict(profile)
        configs = dict(profile["marker_configs"])
        old = "model.layers.49.self_attn.q_proj.comfy_quant"
        new = "model.layers.49.self_attn.not_q_proj.comfy_quant"
        configs[new] = configs.pop(old)
        altered["marker_configs"] = configs
        altered["keys"] = frozenset((set(profile["keys"]) - {old}) | {new})
        with self.assertRaisesRegex(H3ComponentError, "projection set"):
            _validate_native_nvfp4_profile(altered, checkpoint=path)

    def test_does_not_claim_legacy_int8_convrot_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3vl_int8_convrot.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.self_attn.q_proj.comfy_quant":
                        _marker({"format": "int8_tensorwise", "convrot": True}),
                    "model.language_model.layers.0.self_attn.q_proj.weight":
                        torch.zeros(1, dtype=torch.int8),
                },
                str(path),
            )
            profile = _native_nvfp4_checkpoint_profile([path], checkpoint=path)

        self.assertIsNone(profile)

    def test_does_not_claim_bf16_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qwen3vl_bf16.safetensors"
            save_file(
                {
                    "model.language_model.layers.0.self_attn.q_proj.weight":
                        torch.zeros(1, dtype=torch.bfloat16),
                },
                str(path),
            )
            profile = _native_nvfp4_checkpoint_profile([path], checkpoint=path)

        self.assertIsNone(profile)


class NativeAdapterTests(unittest.TestCase):
    def test_text_only_payload_matches_existing_handle_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = MiniMaxH3ComfyTextEncoder(
                clip=_FakeClip(),
                component_path=root,
                weights_path=root / "weights.safetensors",
                quant_format="nvfp4",
            )
            output = adapter.encode_conditioning("a prompt")

        self.assertEqual(tuple(output["prompt_embeds"].shape), (1, 5120))
        self.assertEqual(output["text_token_tags"].tolist(), [1])
        self.assertEqual(output["presentation_input_ids"].tolist(), [42])
        self.assertEqual(output["presentation"], "t2va_text_only_v1")

    def test_fl2va_expands_native_visual_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = MiniMaxH3ComfyTextEncoder(
                clip=_FakeClip(),
                component_path=root,
                weights_path=root / "weights.safetensors",
                quant_format="nvfp4",
            )
            output = adapter.encode_fl2va_conditioning(
                "a prompt", [torch.zeros((1, 32, 32, 3))]
            )

        self.assertEqual(tuple(output["prompt_embeds"].shape), (7, 5120))
        self.assertEqual(output["image_token_counts"], (4,))
        self.assertEqual(output["text_token_tags"].tolist(), [0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(len(output["presentation_input_ids"]), 7)


if __name__ == "__main__":
    unittest.main()
