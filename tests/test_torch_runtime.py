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
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _flush_linear_bag,
            _make_quantized_linear_patcher,
            _physical_module_bytes,
        )

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

        physical_bytes = _physical_module_bytes(linear)
        logical_bf16_bytes = linear.weight.numel() * linear.weight.element_size()
        self.assertEqual(
            physical_bytes,
            sum(value.nbytes for value in linear.state_dict().values()),
        )
        self.assertLess(physical_bytes, logical_bf16_bytes)
        bank, patcher, bank_bytes = _make_quantized_linear_patcher(
            [linear],
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
        )
        self.assertIs(bank.linears[0], linear)
        self.assertEqual(bank_bytes, physical_bytes)
        self.assertEqual(patcher.model_size(), physical_bytes)

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

    @unittest.skipUnless(HAS_COMFY, "requires ComfyUI quantized ops")
    def test_te_int8_low_vram_linear_streams_to_cuda(self):
        import json
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA for CPU-to-GPU Linear streaming")

        import comfy.ops as cops
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _flush_linear_bag,
            _make_quantized_linear_patcher,
        )

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
        ops = cops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
        linear = ops.Linear(
            256, 256, bias=True, device="meta", dtype=torch.bfloat16
        )
        bag = {
            "weight": torch.randint(-127, 128, (256, 256), dtype=torch.int8),
            "weight_scale": torch.ones(256, 1, dtype=torch.float32),
            "comfy_quant": marker,
            "bias": torch.zeros(256, dtype=torch.bfloat16),
        }
        self.assertTrue(
            _flush_linear_bag(linear, "x.", bag, torch.device("cpu"))
        )
        _bank, patcher, _size = _make_quantized_linear_patcher(
            [linear],
            load_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
        )
        try:
            # A near-zero budget deliberately leaves the weight on CPU.  The
            # real Qwen path relies on this same Comfy cast/stream operation.
            patcher.patch_model(
                device_to=torch.device("cuda:0"),
                lowvram_model_memory=0.1,
            )
            self.assertEqual(linear.weight._qdata.device.type, "cpu")
            inputs = torch.randn(
                1, 4, 256, device="cuda:0", dtype=torch.bfloat16
            )
            output = linear(inputs)
            self.assertEqual(output.device.type, "cuda")
            self.assertTrue(torch.isfinite(output).all())
        finally:
            patcher.detach()
        self.assertEqual(linear.weight._qdata.device.type, "cpu")

    @unittest.skipUnless(HAS_COMFY, "requires ComfyUI quantized ops")
    def test_te_int8_wrapper_patcher_lifecycle_keeps_visual_on_cpu(self):
        import json
        from pathlib import Path

        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA for text-encoder load lifecycle")

        import comfy.ops as cops
        from minimax_h3_nodes.runtime.qwen_encoder import (
            MiniMaxH3TextEncoder,
            _flush_linear_bag,
        )

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
        ops = cops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
        linears = []
        for _index in range(50 * 7):
            linear = ops.Linear(
                256, 256, bias=False, device="meta", dtype=torch.bfloat16
            )
            self.assertTrue(
                _flush_linear_bag(
                    linear,
                    "x.",
                    {
                        "weight": torch.zeros(256, 256, dtype=torch.int8),
                        "weight_scale": torch.ones(
                            256, 1, dtype=torch.float32
                        ),
                        "comfy_quant": marker,
                    },
                    torch.device("cpu"),
                )
            )
            linears.append(linear)

        class FakeLanguage(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(
                    512, 256, dtype=torch.bfloat16
                )
                self.norm = torch.nn.LayerNorm(256, dtype=torch.bfloat16)
                self.linears = torch.nn.ModuleList(linears)

        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(8, 8, dtype=torch.bfloat16)
                self.language_model = FakeLanguage()

        model = FakeBackbone().eval()
        encoder = MiniMaxH3TextEncoder(
            model=model,
            tokenizer=None,
            component_path=Path("/tmp/fake-h3-text"),
            load_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            quantized=True,
        )
        encoder.load_for_inference()
        try:
            self.assertEqual(encoder.device, torch.device("cuda:0"))
            self.assertEqual(model.visual.weight.device.type, "cpu")
            self.assertEqual(
                model.language_model.embed_tokens.weight.device.type, "cuda"
            )
            self.assertGreater(encoder._linear_patcher.loaded_size(), 0)
        finally:
            encoder.offload_after_inference()

        self.assertIsNone(encoder._compute_device)
        self.assertEqual(model.visual.weight.device.type, "cpu")
        self.assertEqual(
            model.language_model.embed_tokens.weight.device.type, "cpu"
        )
        self.assertTrue(
            all(linear.weight._qdata.device.type == "cpu" for linear in linears)
        )

    def test_partial_offload_uses_explicit_cuda_compute_device(self):
        import torch

        from minimax_h3_nodes.sampling import _model_device

        model = torch.nn.Linear(2, 2, device="cpu")
        object.__setattr__(model, "_h3_compute_device", torch.device("cuda:0"))
        self.assertEqual(_model_device(model), torch.device("cuda:0"))

    def test_dit_handle_without_model_patcher_lifecycle(self):
        from pathlib import Path

        import torch

        from minimax_h3_nodes.runtime.model_loader import H3ModelHandle

        model = torch.nn.Linear(2, 2, device="cpu")
        handle = H3ModelHandle(
            model=model,
            model_patcher=None,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            dtype=torch.float32,
            metadata={},
            checkpoint_files=(),
        )

        self.assertIs(handle.load_for_inference(), model)
        self.assertEqual(model._h3_compute_device, torch.device("cpu"))
        self.assertTrue(all(value.device.type == "cpu" for value in model.parameters()))

        handle.offload_after_inference()
        self.assertTrue(all(value.device.type == "cpu" for value in model.parameters()))
        self.assertFalse(hasattr(model, "_h3_compute_device"))

        # Comfy owns the patcher's physical residency, but the handle's compute
        # marker is session-scoped and must not survive an offload boundary.
        handle.model_patcher = object()
        object.__setattr__(model, "_h3_compute_device", torch.device("cuda:0"))
        handle.offload_after_inference()
        self.assertFalse(hasattr(model, "_h3_compute_device"))

    def test_plain_linear_casts_bf16_checkpoint_to_float16_target(self):
        import torch

        from minimax_h3_nodes.runtime.model_loader import _flush_linear

        linear = torch.nn.Linear(
            3,
            2,
            bias=True,
            device="meta",
            dtype=torch.float16,
        )
        checkpoint_weight = torch.randn(2, 3, dtype=torch.bfloat16)
        checkpoint_bias = torch.randn(2, dtype=torch.bfloat16)

        self.assertFalse(
            _flush_linear(
                linear,
                "projection.",
                {
                    "weight": checkpoint_weight,
                    "bias": checkpoint_bias,
                },
                torch.device("cpu"),
            )
        )
        self.assertEqual(linear.weight.dtype, torch.float16)
        self.assertEqual(linear.bias.dtype, torch.float16)
        self.assertTrue(
            torch.equal(linear.weight, checkpoint_weight.to(torch.float16))
        )
        self.assertTrue(
            torch.equal(linear.bias, checkpoint_bias.to(torch.float16))
        )

    def test_nonstreamable_dit_tensor_budget_and_placement_are_explicit(self):
        import torch

        from minimax_h3_nodes.runtime.model_loader import (
            _misplaced_nonstreamable_tensors,
            _move_nonstreamable_tensors,
            _nonstreamable_tensor_bytes,
            _nonstreamable_tensor_items,
        )

        class Streamable(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(4, 4))
                self.comfy_cast_weights = True

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.native = torch.nn.Linear(4, 3)
                self.streamed = Streamable()
                self.register_buffer("native_buffer", torch.ones(2))

        model = TinyModel()
        items = dict(_nonstreamable_tensor_items(model))
        self.assertIn("native.weight", items)
        self.assertIn("native.bias", items)
        self.assertIn("native_buffer", items)
        self.assertNotIn("streamed.weight", items)
        self.assertEqual(
            _nonstreamable_tensor_bytes(model),
            sum(tensor.nbytes for tensor in items.values()),
        )
        self.assertEqual(
            _misplaced_nonstreamable_tensors(model, torch.device("cpu")),
            [],
        )
        misplaced = _misplaced_nonstreamable_tensors(
            model,
            torch.device("meta"),
        )
        self.assertTrue(any(item.startswith("native.weight=") for item in misplaced))
        self.assertFalse(any(item.startswith("streamed.weight=") for item in misplaced))

        native_weight = model.native.weight
        model.native_weight_alias = native_weight
        _move_nonstreamable_tensors(model, torch.device("meta"))
        self.assertEqual(model.native.weight.device.type, "meta")
        self.assertEqual(model.native.bias.device.type, "meta")
        self.assertEqual(model.native_buffer.device.type, "meta")
        self.assertEqual(model.streamed.weight.device.type, "cpu")
        self.assertIs(model.native.weight, model.native_weight_alias)
        self.assertIsInstance(model.native.weight, torch.nn.Parameter)
        self.assertEqual(
            _misplaced_nonstreamable_tensors(model, torch.device("meta")),
            [],
        )

    def test_full_load_placement_failure_unloads_model_patcher(self):
        import sys
        import types
        from pathlib import Path
        from unittest import mock

        import torch

        from minimax_h3_nodes.runtime.model_loader import H3ModelHandle

        class FakePatcher:
            model = object()
            pinned = ()

            def __init__(self):
                self.loaded = 0

            def loaded_size(self):
                return self.loaded

            def detach(self):
                self.loaded = 0

        patcher = FakePatcher()
        unload_calls = []
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")

        def load_models_gpu(_patchers, **_kwargs):
            patcher.loaded = 1

        def unload_model_and_clones(value):
            unload_calls.append(value)
            value.loaded = 0

        management.load_models_gpu = load_models_gpu
        management.unload_model_and_clones = unload_model_and_clones
        comfy.model_management = management
        model = torch.nn.Linear(2, 2, device="cpu")
        handle = H3ModelHandle(
            model=model,
            model_patcher=patcher,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device=torch.device("cuda:0"),
            offload_device=torch.device("cpu"),
            dtype=torch.bfloat16,
            metadata={},
            checkpoint_files=(),
            quantized=False,
        )
        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.model_management": management},
        ), self.assertRaisesRegex(RuntimeError, "did not fully load"):
            handle.load_for_inference()

        self.assertEqual(unload_calls, [patcher])
        self.assertEqual(patcher.loaded, 0)
        self.assertFalse(hasattr(model, "_h3_compute_device"))

    def test_indexless_quant_detection_scans_every_shard(self):
        import tempfile
        from pathlib import Path

        import torch
        from safetensors.torch import save_file

        from minimax_h3_nodes.runtime.model_loader import _is_quantized_map

        with tempfile.TemporaryDirectory() as raw:
            first = Path(raw) / "model-00001.safetensors"
            second = Path(raw) / "model-00002.safetensors"
            save_file({"plain.weight": torch.ones(1)}, str(first))
            save_file(
                {"quantized.comfy_quant": torch.ones(1, dtype=torch.uint8)},
                str(second),
            )
            self.assertTrue(_is_quantized_map(None, [first, second]))

    def test_vae_component_and_shards_cannot_escape_selected_root(self):
        import json
        import tempfile
        from pathlib import Path

        from minimax_h3_nodes.runtime.vae_adapter import (
            H3VAEError,
            _component_config,
            _component_weight_files,
            _files_from_safetensors_index,
            _video_source_config,
            resolve_h3_component_dir,
        )

        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "vae"
            root.mkdir()
            outside_component = base / "outside-video-vae"
            outside_component.mkdir()
            (outside_component / "config.json").write_text(
                json.dumps({"latent_channels": 24}),
                encoding="utf-8",
            )

            (root / "model_index.json").write_text(
                json.dumps(
                    {"video_vae": {"path": "../outside-video-vae"}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(H3VAEError, "must not contain"):
                resolve_h3_component_dir(root, "video_vae")

            (root / "model_index.json").unlink()
            (root / "video_vae").symlink_to(outside_component)
            with self.assertRaisesRegex(H3VAEError, "outside selected VAE root"):
                resolve_h3_component_dir(root, "video_vae")

            component = base / "audio_vae"
            component.mkdir()
            outside_shard = base / "outside.safetensors"
            outside_shard.touch()
            index = component / "model.safetensors.index.json"
            for shard_name in (
                "../outside.safetensors",
                str(outside_shard.resolve()),
            ):
                with self.subTest(shard_name=shard_name):
                    index.write_text(
                        json.dumps({"weight_map": {"x": shard_name}}),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(H3VAEError, "must be relative"):
                        _files_from_safetensors_index(
                            index,
                            component_root=component,
                        )

            linked = component / "linked.safetensors"
            linked.symlink_to(outside_shard)
            index.write_text(
                json.dumps({"weight_map": {"x": linked.name}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(H3VAEError, "outside selected VAE root"):
                _files_from_safetensors_index(
                    index,
                    component_root=component,
                )

            index.unlink()
            linked.unlink()
            (component / "model.safetensors").symlink_to(outside_shard)
            with self.assertRaisesRegex(H3VAEError, "outside selected VAE root"):
                _component_weight_files(component, "audio_vae", {})

            outside_config = base / "outside-config.json"
            outside_config.write_text("{}", encoding="utf-8")
            for component_name in ("video_vae", "audio_vae"):
                escaped_component = base / f"escaped-{component_name}"
                escaped_component.mkdir()
                (escaped_component / "config.json").symlink_to(outside_config)
                with self.subTest(component_name=component_name), self.assertRaisesRegex(
                    H3VAEError, "outside selected VAE root"
                ):
                    _component_config(escaped_component, component_name)

            video_component = base / "video-source-config-escape"
            (video_component / "source").mkdir(parents=True)
            (video_component / "source" / "config.json").symlink_to(
                outside_config
            )
            with self.assertRaisesRegex(H3VAEError, "outside selected VAE root"):
                _video_source_config(video_component, {})

    def test_vae_official_source_checkpoint_remains_allowed(self):
        import tempfile
        from pathlib import Path

        from minimax_h3_nodes.runtime.vae_adapter import _component_weight_files

        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw) / "video_vae"
            checkpoint = component / "source" / "model.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            self.assertEqual(
                _component_weight_files(component, "video_vae", {}),
                [checkpoint.resolve()],
            )

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
