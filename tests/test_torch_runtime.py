import importlib.util
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_COMFY = importlib.util.find_spec("comfy") is not None


@unittest.skipUnless(HAS_TORCH, "artifact container has no ComfyUI torch")
class TorchRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(HAS_COMFY, "requires ComfyUI quantized ops")
    def test_meta_int8_linear_materializes_complete_quantized_tensor(self):
        import json
        import torch

        import comfy.ops as cops
        from comfy.quant_ops import QuantizedTensor
        from minimax_h3_nodes.runtime.components import H3ComponentError
        from minimax_h3_nodes.runtime.model_loader import _flush_linear
        from minimax_h3_nodes.runtime.qwen_encoder import _flush_linear_bag

        marker = torch.tensor(
            list(
                json.dumps(
                    {
                        "format": "int8_tensorwise",
                        "convrot": True,
                        "convrot_groupsize": 256,
                    },
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            dtype=torch.uint8,
        )
        bag = {
            "weight": torch.randint(-127, 128, (256, 256), dtype=torch.int8),
            "weight_scale": torch.ones(256, 1, dtype=torch.float32),
            "comfy_quant": marker,
            "bias": torch.zeros(256, dtype=torch.bfloat16),
        }
        ops = cops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)

        for flush in (_flush_linear, _flush_linear_bag):
            linear = ops.Linear(
                256, 256, bias=True, device="meta", dtype=torch.bfloat16
            )
            self.assertTrue(flush(linear, "x.", dict(bag), torch.device("cpu")))
            self.assertIsInstance(linear.weight, QuantizedTensor)
            self.assertEqual(linear.weight._qdata.dtype, torch.int8)
            self.assertEqual(linear.weight._qdata.device.type, "cpu")
            self.assertTrue(torch.equal(linear.weight._qdata, bag["weight"]))
            self.assertEqual(tuple(linear.weight._params.scale.shape), (256, 1))
            self.assertTrue(
                torch.equal(linear.weight._params.scale, bag["weight_scale"])
            )
            self.assertTrue(linear.weight._params.convrot)
            self.assertEqual(linear.quant_format, "int8_tensorwise")
            self.assertEqual(
                set(linear.state_dict(prefix="x.")),
                {"x.weight", "x.weight_scale", "x.comfy_quant", "x.bias"},
            )

        plain = torch.nn.Linear(
            4, 3, bias=True, device="meta", dtype=torch.float32
        )
        plain_bag = {
            "weight": torch.randn(3, 4),
            "bias": torch.randn(3),
        }
        self.assertFalse(_flush_linear(plain, "x.", plain_bag, torch.device("cpu")))
        self.assertTrue(torch.equal(plain.weight, plain_bag["weight"]))
        self.assertTrue(torch.equal(plain.bias, plain_bag["bias"]))
        with self.assertRaisesRegex(H3ComponentError, "plain nn.Linear"):
            _flush_linear(plain, "x.", dict(bag), torch.device("cpu"))

    def test_partial_offload_uses_explicit_cuda_compute_device(self):
        import torch

        from minimax_h3_nodes.sampling import _model_device

        model = torch.nn.Linear(2, 2, device="cpu")
        object.__setattr__(model, "_h3_compute_device", torch.device("cuda:0"))
        self.assertEqual(_model_device(model), torch.device("cuda:0"))

    def test_tiny_dit_meta_and_forward_contract(self):
        import torch

        from minimax_h3_nodes.runtime.dit import (
            MiniMaxH3DiTConfig,
            MiniMaxH3DiTModel,
        )

        config = MiniMaxH3DiTConfig(
            num_layers=1,
            token_refiner_num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            attention_head_dim=8,
            ffn_hidden_size=16,
            latents_dim=1,
            audio_latents_dim=2,
            patch_size=(1, 1, 1),
            text_dim=6,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            time_embed_dim=4,
            adaln_out_features=18 * 8,
            final_adaln_out_features=2 * 8,
            rope_inv_freq_len=1,
        )
        meta_model = MiniMaxH3DiTModel(
            config,
            device="meta",
            dtype=torch.float32,
        )
        self.assertTrue(all(value.device.type == "meta" for value in meta_model.state_dict().values()))

        model = MiniMaxH3DiTModel(config, device="cpu", dtype=torch.float32).eval()
        sequence = 6
        kwargs = {
            "x": torch.randn(1, sequence, 1),
            "audio_x": torch.randn(1, sequence, 2),
            "img_position_ids": torch.zeros(1, sequence, 3, dtype=torch.float64),
            "unique_timesteps": torch.tensor([0.25, 0.5]),
            "inverse_indices": torch.tensor([0, 0, 1, 1, 0, 0]),
            "update_mask": torch.ones(2, dtype=torch.bool),
            "update_audio_mask": torch.ones(2, dtype=torch.bool),
            "token_tags": torch.tensor([1, 1, 2, 2, 0, 0]),
            "skip_mask_out_condition": False,
            "prompt_embeds": torch.randn(2, 6),
            "img_pos_info": {"position_ids": torch.tensor([4, 5])},
            "audio_pos_info": {"position_ids": torch.tensor([2, 3])},
            "text_pos_info": {"position_ids": torch.tensor([0, 1])},
            "img_pos_for_infer_output_info": {
                "position_ids": torch.tensor([4, 5])
            },
            "packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, sequence], dtype=torch.int32),
                "max_seqlen_q": sequence,
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": torch.tensor([0, 2, 2], dtype=torch.int32),
                "max_seqlen_q": 2,
            },
        }
        with torch.inference_mode():
            video, audio = model(**kwargs)
        self.assertEqual(tuple(video.shape), (2, 1))
        self.assertEqual(tuple(audio.shape), (2, 2))
        self.assertTrue(torch.isfinite(video).all())
        self.assertTrue(torch.isfinite(audio).all())

    def test_rope_cpu_buffer_follows_cuda_position_device(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA to reproduce partial-offload devices")

        from minimax_h3_nodes.runtime.dit import (
            MiniMaxH3Rope,
            _apply_rope_qk,
            _rope_cos_sin_cache,
        )

        rope = MiniMaxH3Rope(inv_freq_len=1, device="cpu")
        position_ids = torch.zeros(
            1, 4, 3, dtype=torch.long, device="cuda"
        )
        frequencies = rope(position_ids)
        cache = _rope_cos_sin_cache(frequencies, dtype=torch.float32)
        query = torch.randn(4, 1, 8, device="cuda")

        rotated_query, rotated_key = _apply_rope_qk(
            query, query.clone(), cache
        )

        self.assertEqual(frequencies.device.type, "cuda")
        self.assertEqual(cache.device.type, "cuda")
        self.assertEqual(rotated_query.device.type, "cuda")
        self.assertEqual(rotated_key.device.type, "cuda")
        self.assertEqual(rotated_query.shape, query.shape)
        self.assertEqual(rotated_key.shape, query.shape)

    def test_video_vae_rope_buffer_survives_meta_construction(self):
        import torch

        from minimax_h3_nodes.vendor.minimax_h3_video_vae.base_module import (
            RotaryEmbeddingND,
        )

        with torch.device("meta"):
            rope = RotaryEmbeddingND(dim=12, n_dim=3)

        self.assertEqual(rope.inv_freq.device.type, "cpu")
        self.assertNotIn("inv_freq", rope.state_dict())
        position_ids = torch.zeros(1, 4, 3, dtype=torch.float32)
        cos, sin = rope(position_ids)
        self.assertEqual(cos.device.type, "cpu")
        self.assertEqual(sin.device.type, "cpu")
        self.assertTrue(torch.isfinite(cos).all())
        self.assertTrue(torch.isfinite(sin).all())

    def test_audio_vae_legacy_weight_norm_keys_are_remapped(self):
        import torch

        from minimax_h3_nodes.runtime.vae_adapter import (
            H3VAEError,
            _normalize_checkpoint_keys,
        )

        weight_g = torch.randn(4, 1, 1)
        weight_v = torch.randn(4, 3, 5)
        bias = torch.randn(4)
        legacy_state = {
            "decoder.conv.weight_g": weight_g,
            "decoder.conv.weight_v": weight_v,
            "decoder.conv.bias": bias,
        }
        model_keys = {
            "decoder.conv.parametrizations.weight.original0",
            "decoder.conv.parametrizations.weight.original1",
            "decoder.conv.bias",
        }

        normalized = _normalize_checkpoint_keys(
            legacy_state, model_keys, "audio_vae"
        )

        self.assertEqual(set(normalized), model_keys)
        self.assertIs(
            normalized["decoder.conv.parametrizations.weight.original0"],
            weight_g,
        )
        self.assertIs(
            normalized["decoder.conv.parametrizations.weight.original1"],
            weight_v,
        )
        self.assertIs(normalized["decoder.conv.bias"], bias)

        conflicting = {
            **legacy_state,
            "decoder.conv.parametrizations.weight.original0": weight_g,
        }
        with self.assertRaisesRegex(H3VAEError, "duplicate target"):
            _normalize_checkpoint_keys(conflicting, model_keys, "audio_vae")

    def test_t2va_packed_layout_and_latent_roundtrip(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            minimax_h3_packed_sequence_t2va,
            patchify_video_latent,
            unpatchify_video_tokens,
        )

        packed = minimax_h3_packed_sequence_t2va(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=3,
        )
        self.assertEqual(int(packed["used_len"]), 17)
        self.assertEqual(int(packed["seq_len"]), 64)
        self.assertEqual(tuple(packed["img_position_ids"].shape), (64, 3))
        self.assertEqual(int(packed["img_pos"].numel()), 8)
        self.assertEqual(int(packed["audio_pos"].numel()), 6)

        latent = torch.arange(1 * 24 * 2 * 4 * 4).reshape(1, 24, 2, 4, 4)
        rows = patchify_video_latent(latent)
        restored = unpatchify_video_tokens(
            rows,
            latent_shape=(2, 2, 2, 24),
        )
        self.assertTrue(torch.equal(restored, latent))

    def test_qkv_grouped_reorder(self):
        import torch

        from minimax_h3_nodes.runtime.dit import (
            MiniMaxH3DiTConfig,
            is_qkv_scale_key,
            prepare_checkpoint_tensor,
            reorder_grouped_qkv_to_qkv,
        )

        # Two groups, one Q head per group, scalar head dimension:
        # [q0,k0,v0,q1,k1,v1] -> [q0,q1,k0,k1,v0,v1].
        source = torch.arange(6).reshape(6, 1)
        output = reorder_grouped_qkv_to_qkv(
            source,
            num_query_groups=2,
            heads_per_group=1,
            head_dim=1,
        )
        self.assertEqual(output[:, 0].tolist(), [0, 3, 1, 4, 2, 5])
        self.assertTrue(is_qkv_scale_key("blocks.0.attn.qkv_proj.weight_scale"))
        cfg = MiniMaxH3DiTConfig(
            num_layers=1,
            token_refiner_num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            attention_head_dim=8,
            ffn_hidden_size=16,
            latents_dim=1,
            audio_latents_dim=2,
            patch_size=(1, 1, 1),
            text_dim=6,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            time_embed_dim=4,
            adaln_out_features=18 * 8,
            final_adaln_out_features=2 * 8,
            rope_inv_freq_len=1,
        )
        # Use a valid RoPE head dimension for the config-backed path:
        # 2 groups * (1+2)*8 = 48 rows for scale.
        scale = torch.arange(48, dtype=torch.float32).reshape(48, 1)
        got = prepare_checkpoint_tensor(
            "blocks.0.attn.qkv_proj.weight_scale", scale, config=cfg
        )
        expected = (
            list(range(0, 8))
            + list(range(24, 32))
            + list(range(8, 16))
            + list(range(32, 40))
            + list(range(16, 24))
            + list(range(40, 48))
        )
        self.assertEqual(got[:, 0].tolist(), expected)

    def test_operations_linear_and_adaln_dtype(self):
        import torch
        import torch.nn as nn

        from minimax_h3_nodes.runtime.dit import (
            MiniMaxH3AdalnProj,
            MiniMaxH3DiTConfig,
            MiniMaxH3DiTModel,
            _activation_dtype,
        )

        config = MiniMaxH3DiTConfig(
            num_layers=1,
            token_refiner_num_layers=1,
            hidden_size=8,
            num_attention_heads=1,
            attention_head_dim=8,
            ffn_hidden_size=16,
            latents_dim=1,
            audio_latents_dim=2,
            patch_size=(1, 1, 1),
            text_dim=6,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            time_embed_dim=4,
            adaln_out_features=18 * 8,
            final_adaln_out_features=2 * 8,
            rope_inv_freq_len=1,
        )
        ops = type("Ops", (), {"Linear": nn.Linear})()
        model = MiniMaxH3DiTModel(
            config, device="cpu", dtype=torch.float32, operations=ops
        ).eval()
        self.assertIsInstance(model.blocks[0].attn.qkv_proj, nn.Linear)
        self.assertIsInstance(model.video_patch_proj, nn.Linear)
        adaln = MiniMaxH3AdalnProj(
            config,
            config.adaln_out_features,
            expand_ratio=6,
            modality_count=3,
            dtype=torch.bfloat16,
            device="cpu",
        )
        fake = type("L", (), {"factory_kwargs": {"dtype": torch.bfloat16}, "weight": torch.zeros(1, dtype=torch.int8)})()
        self.assertEqual(_activation_dtype(fake, torch.float32), torch.bfloat16)
        with torch.inference_mode():
            out = adaln(torch.randn(2, config.time_embed_dim))
        self.assertEqual(len(out), 6)

    def test_dual_shift_schedules_are_distinct(self):
        from minimax_h3_nodes.sampling import shifted_sigma_schedule

        video = shifted_sigma_schedule(sigma_points=50, shift=12.0)
        audio = shifted_sigma_schedule(sigma_points=50, shift=3.0)
        self.assertEqual(len(video), 50)
        self.assertEqual(len(audio), 50)
        self.assertEqual(video[0], 1.0)
        self.assertEqual(audio[0], 1.0)
        self.assertEqual(video[-1], 0.0)
        self.assertEqual(audio[-1], 0.0)
        self.assertNotEqual(video[1:-1], audio[1:-1])


if __name__ == "__main__":
    unittest.main()
