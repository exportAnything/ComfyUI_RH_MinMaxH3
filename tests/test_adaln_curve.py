"""adaLN 曲线表 checkpoint（PR#15224）：低秩基 + 采样表插值。"""
from __future__ import annotations
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

CURVE_GRID = 129
CURVE_RANK = 4  # tiny config 的 time_embed_dim，满秩 → 曲线路径应逐点等价


def _tiny_cfg(**overrides):
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
    base = dict(
        num_layers=2, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=1,
        attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
        patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
        rope_inv_freq_len=1,
    )
    base.update(overrides)
    return MiniMaxH3DiTConfig(**base)


def _fwd_kwargs(seq=6, timesteps=(0.25, 0.5)):
    """默认 timestep 落在 129 点网格上（0.25=32/128, 0.5=64/128），插值无残差。"""
    import torch
    tags = torch.tensor([1, 1, 2, 2, 0, 0])
    inverse = torch.tensor([0, 0, 1, 1, 0, 0])
    return {
        "x": torch.randn(1, seq, 1), "audio_x": torch.randn(1, seq, 2),
        "img_position_ids": torch.zeros(1, seq, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor(list(timesteps)), "inverse_indices": inverse,
        "update_mask": torch.ones(2, dtype=torch.bool),
        "update_audio_mask": torch.ones(2, dtype=torch.bool), "token_tags": tags,
        "prompt_embeds": torch.randn(2, 6),
        "img_pos_info": {"position_ids": torch.tensor([4, 5])},
        "audio_pos_info": {"position_ids": torch.tensor([2, 3])},
        "text_pos_info": {"position_ids": torch.tensor([0, 1])},
        "img_pos_for_infer_output_info": {"position_ids": torch.tensor([4, 5])},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, seq], dtype=torch.int32), "max_seqlen_q": seq},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 2, 2], dtype=torch.int32), "max_seqlen_q": 2},
    }


def _build_pair(rank=CURVE_RANK, grid=CURVE_GRID):
    """返回 (原版模型, 由它转换出的曲线表模型, 基, 采样表)。"""
    import torch
    from minimax_h3_nodes.runtime.adaln_curve import (
        fit_curve_basis, project_adaln_weight, sample_time_embedding_curve,
    )
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel

    torch.manual_seed(7)
    plain = MiniMaxH3DiTModel.from_config(_tiny_cfg(), dtype=torch.float32).eval()
    curve = sample_time_embedding_curve(plain.time_embedder, grid)
    basis, table, _report = fit_curve_basis(curve, rank)

    curved = MiniMaxH3DiTModel.from_config(
        _tiny_cfg(time_embed_dim=rank, adaln_curve_grid=grid), dtype=torch.float32
    ).eval()
    state = {
        k: v for k, v in plain.state_dict().items() if not k.startswith("time_embedder.")
    }
    for key in list(state):
        if key.endswith(".adaln_proj.linear.weight"):
            state[key] = project_adaln_weight(state[key], basis).to(torch.float32)
    state["adaln_t_table"] = table.to(torch.float32)
    curved.load_state_dict(state)
    return plain, curved, basis, table


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestCurveTableMath(unittest.TestCase):
    def test_interpolate_hits_grid_points_exactly(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import curve_grid, interpolate_curve_table
        table = torch.randn(17, 3, dtype=torch.float64)
        points = curve_grid(17)
        got = interpolate_curve_table(table, points)
        self.assertTrue(torch.allclose(got, table, atol=1e-12))

    def test_interpolate_clamps_out_of_range(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import interpolate_curve_table
        table = torch.randn(8, 2, dtype=torch.float64)
        got = interpolate_curve_table(table, torch.tensor([-0.5, 0.0, 1.0, 3.0]))
        self.assertTrue(torch.allclose(got[0], table[0]))
        self.assertTrue(torch.allclose(got[1], table[0]))
        # t=1.0 必须停在最后一行而不是越界读下一行
        self.assertTrue(torch.allclose(got[2], table[-1]))
        self.assertTrue(torch.allclose(got[3], table[-1]))

    def test_interpolate_is_linear_between_rows(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import interpolate_curve_table
        table = torch.tensor([[0.0], [2.0], [10.0]], dtype=torch.float64)
        got = interpolate_curve_table(table, torch.tensor([0.25, 0.75]))
        self.assertAlmostEqual(float(got[0, 0]), 1.0, places=12)
        self.assertAlmostEqual(float(got[1, 0]), 6.0, places=12)

    def test_rejects_degenerate_table(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import interpolate_curve_table
        with self.assertRaises(ValueError):
            interpolate_curve_table(torch.zeros(1, 4), torch.tensor([0.5]))
        with self.assertRaises(ValueError):
            interpolate_curve_table(torch.zeros(4), torch.tensor([0.5]))

    def test_full_rank_basis_reconstructs_curve(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import fit_curve_basis
        torch.manual_seed(3)
        curve = torch.randn(40, 5, dtype=torch.float64)
        basis, table, report = fit_curve_basis(curve, 5)
        self.assertLess(report["relative_error"], 1e-12)
        self.assertTrue(torch.allclose(table @ basis, curve, atol=1e-10))

    def test_projected_weight_matches_dense_product(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import (
            fit_curve_basis, project_adaln_weight,
        )
        torch.manual_seed(5)
        curve = torch.randn(32, 6, dtype=torch.float64)
        weight = torch.randn(11, 6, dtype=torch.float64)
        basis, table, _ = fit_curve_basis(curve, 6)
        projected = project_adaln_weight(weight, basis)
        # 满秩下 W·e(t) 与 W'·c(t) 必须逐点相等
        self.assertTrue(
            torch.allclose(table @ projected.T, curve @ weight.T, atol=1e-9)
        )

    def test_rank_and_shape_guards(self):
        import torch
        from minimax_h3_nodes.runtime.adaln_curve import (
            curve_grid, fit_curve_basis, project_adaln_weight,
        )
        with self.assertRaises(ValueError):
            curve_grid(1)
        with self.assertRaises(ValueError):
            fit_curve_basis(torch.zeros(4, 3, dtype=torch.float64), 5)
        with self.assertRaises(ValueError):
            project_adaln_weight(torch.zeros(4, 3), torch.zeros(2, 5))


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestCurveModel(unittest.TestCase):
    def test_config_flags_and_module_layout(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
        curved = MiniMaxH3DiTModel.from_config(
            _tiny_cfg(adaln_curve_grid=CURVE_GRID), dtype=torch.float32
        )
        self.assertTrue(curved.use_adaln_curves)
        self.assertFalse(hasattr(curved, "time_embedder"))
        self.assertEqual(tuple(curved.adaln_t_table.shape), (CURVE_GRID, 4))
        self.assertFalse(curved.blocks[0].adaln_proj.apply_silu)
        self.assertFalse(curved.final_layer.adaln_proj.apply_silu)
        self.assertEqual(curved.blocks[0].adaln_proj.linear.weight.dtype, torch.float32)

        plain = MiniMaxH3DiTModel.from_config(_tiny_cfg(), dtype=torch.float32)
        self.assertFalse(plain.use_adaln_curves)
        self.assertTrue(hasattr(plain, "time_embedder"))
        self.assertTrue(plain.blocks[0].adaln_proj.apply_silu)

    def test_config_rejects_rank_above_grid(self):
        with self.assertRaises(ValueError):
            _tiny_cfg(adaln_curve_grid=3, time_embed_dim=8)
        with self.assertRaises(ValueError):
            _tiny_cfg(adaln_curve_grid=1)

    def test_forward_matches_monolithic_on_grid_timesteps(self):
        import torch
        plain, curved, _basis, _table = _build_pair()
        kwargs = _fwd_kwargs()  # timestep 落在网格点上
        with torch.no_grad():
            want_v, want_a = plain(**kwargs)
            got_v, got_a = curved(**kwargs)
        # 满秩基 + 网格点：只剩浮点重排误差
        self.assertTrue(torch.allclose(got_v, want_v, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(got_a, want_a, atol=1e-5, rtol=1e-4))

    def test_forward_stays_close_off_grid(self):
        import torch
        plain, curved, _basis, _table = _build_pair()
        kwargs = _fwd_kwargs(timesteps=(0.2137, 0.6042))  # 刻意落在网格点之间
        with torch.no_grad():
            want_v, _want_a = plain(**kwargs)
            got_v, _got_a = curved(**kwargs)
        cosine = torch.nn.functional.cosine_similarity(
            got_v.reshape(-1), want_v.reshape(-1), dim=0
        )
        self.assertGreater(float(cosine), 0.9999)

    def test_timestep_embedding_reproduces_adaln_modulation(self):
        import torch
        plain, curved, _basis, _table = _build_pair()
        t = torch.tensor([0.0, 0.25, 0.5, 1.0])  # 全部落在网格点上
        with torch.no_grad():
            coords = curved.timestep_embedding(t)
            want_mod = plain.blocks[0].adaln_proj(plain.time_embedder(t))
            got_mod = curved.blocks[0].adaln_proj(coords)
        self.assertEqual(tuple(coords.shape), (4, CURVE_RANK))
        for got, want in zip(got_mod, want_mod):
            cosine = torch.nn.functional.cosine_similarity(
                got.reshape(-1), want.reshape(-1), dim=0
            )
            self.assertGreater(float(cosine), 0.99999)

    def test_fp32_policy_covers_curve_tensors(self):
        _plain, curved, _basis, _table = _build_pair()
        # adaLN 低秩权重与采样表都必须留在 fp32；time embedder 已不存在
        names = curved.fp32_param_names()
        self.assertIn("blocks.0.adaln_proj.linear.weight", names)
        self.assertIn("final_layer.adaln_proj.linear.bias", names)
        self.assertNotIn("time_embedder.proj_in.weight", names)
        self.assertIn("adaln_t_table", curved.fp32_buffer_names())

    def test_precompute_is_unsupported_in_curve_mode(self):
        from minimax_h3_nodes.runtime.modulation_cache import (
            H3PrecomputeUnsupported, precompute_dit_modulation,
        )
        _plain, curved, _basis, _table = _build_pair()
        with self.assertRaises(H3PrecomputeUnsupported):
            curved.precompute_modulation([0.25, 0.5])
        with self.assertRaises(H3PrecomputeUnsupported):
            precompute_dit_modulation(curved, [0.25, 0.5])

    def test_frame_rate_adaln_rejected_in_curve_mode(self):
        import torch
        _plain, curved, _basis, _table = _build_pair()
        with self.assertRaises(RuntimeError) as ctx:
            curved.timestep_embedding(torch.tensor([0.5]), frame_rate=30.0)
        self.assertIn("Frame Rate", str(ctx.exception))


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestCurveLoaderDetection(unittest.TestCase):
    def _write_checkpoint(self, directory, tensors, config):
        import json
        from safetensors.torch import save_file
        directory.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(directory / "model.safetensors"))
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def _official_config(self, **overrides):
        config = {
            "_class_name": "MiniMaxH3DiTModel",
            "num_layers": 50, "token_refiner_num_layers": 2, "hidden_size": 5376,
            "num_attention_heads": 56, "attention_head_dim": 128, "ffn_hidden_size": 14336,
            "latents_dim": 24, "audio_latents_dim": 32, "patch_size": [1, 2, 2],
            "text_dim": 5120, "timestep_input_dim": 256, "time_embed_hidden_size": 5376,
            "time_embed_dim": 2688, "adaln_out_features": 18 * 5376,
            "final_adaln_out_features": 2 * 5376, "rope_inv_freq_len": 16,
            "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
        }
        config.update(overrides)
        return config

    def test_detects_curve_shape_and_validates_config(self):
        import tempfile
        from pathlib import Path
        import torch
        from minimax_h3_nodes.runtime.model_loader._impl import (
            _detect_adaln_curve, _validate_adaln_curve_contract, _validate_transformer_config,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "transformer_adaln_curve"
            config = self._official_config(time_embed_dim=48, adaln_curve_grid=256)
            self._write_checkpoint(
                root,
                {"adaln_t_table": torch.zeros(256, 48), "rope.inv_freq": torch.zeros(16)},
                config,
            )
            shards = [root / "model.safetensors"]
            self.assertEqual(_detect_adaln_curve(None, shards), (256, 48))
            _validate_transformer_config(config, root / "config.json")
            _validate_adaln_curve_contract(config, root / "config.json", (256, 48))

    def test_curve_config_may_omit_time_embedder_fields(self):
        from pathlib import Path
        from minimax_h3_nodes.runtime.model_loader._impl import (
            _validate_adaln_curve_contract, _validate_transformer_config,
        )
        config = self._official_config(time_embed_dim=48, adaln_curve_grid=256)
        config.pop("timestep_input_dim")
        config.pop("time_embed_hidden_size")
        _validate_transformer_config(config, Path("config.json"))
        _validate_adaln_curve_contract(config, Path("config.json"), (256, 48))

    def test_plain_config_still_pins_official_time_embed_dim(self):
        from pathlib import Path
        from minimax_h3_nodes.runtime.components import H3ComponentError
        from minimax_h3_nodes.runtime.model_loader._impl import _validate_adaln_curve_contract
        _validate_adaln_curve_contract(self._official_config(), Path("config.json"), None)
        with self.assertRaisesRegex(H3ComponentError, "time_embed_dim"):
            _validate_adaln_curve_contract(
                self._official_config(time_embed_dim=48), Path("config.json"), None
            )

    def test_mismatched_grid_and_missing_table_are_rejected(self):
        from pathlib import Path
        from minimax_h3_nodes.runtime.components import H3ComponentError
        from minimax_h3_nodes.runtime.model_loader._impl import _validate_adaln_curve_contract
        with self.assertRaisesRegex(H3ComponentError, "adaln_curve_grid"):
            _validate_adaln_curve_contract(
                self._official_config(time_embed_dim=48, adaln_curve_grid=99),
                Path("config.json"), (256, 48),
            )
        with self.assertRaisesRegex(H3ComponentError, "adaln_curve_grid"):
            # config 声明曲线表但 checkpoint 里没有
            _validate_adaln_curve_contract(
                self._official_config(adaln_curve_grid=256), Path("config.json"), None,
            )
        with self.assertRaisesRegex(H3ComponentError, "time_embed_dim"):
            # 曲线表 checkpoint 的 time_embed_dim 必须等于表的秩
            _validate_adaln_curve_contract(
                self._official_config(time_embed_dim=2688, adaln_curve_grid=256),
                Path("config.json"), (256, 48),
            )

    def test_plain_checkpoint_detects_none(self):
        import tempfile
        from pathlib import Path
        import torch
        from minimax_h3_nodes.runtime.model_loader._impl import _detect_adaln_curve
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "transformer"
            self._write_checkpoint(
                root, {"time_embedder.proj_in.bias": torch.zeros(8)}, self._official_config()
            )
            self.assertIsNone(_detect_adaln_curve(None, [root / "model.safetensors"]))


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestCurveConversionTool(unittest.TestCase):
    def _tool(self):
        import importlib.util
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / "tools" / "convert_adaln_curve.py"
        spec = importlib.util.spec_from_file_location("convert_adaln_curve", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_round_trip_conversion_reloads_and_matches(self):
        import json
        import tempfile
        from pathlib import Path
        import torch
        from safetensors.torch import save_file
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
        from minimax_h3_nodes.runtime.model_loader._impl import _detect_adaln_curve

        tool = self._tool()
        torch.manual_seed(11)
        plain = MiniMaxH3DiTModel.from_config(_tiny_cfg(), dtype=torch.float32).eval()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "transformer"
            src.mkdir()
            save_file(
                {k: v.contiguous() for k, v in plain.state_dict().items()},
                str(src / "model.safetensors"),
            )
            config = _tiny_cfg().as_dict()
            config["patch_size"] = list(config["patch_size"])
            config.pop("adaln_curve_grid", None)
            (src / "config.json").write_text(json.dumps(config), encoding="utf-8")

            dst = Path(tmp) / "transformer_adaln_curve"
            meta = tool.run_convert(
                src, dst, rank=CURVE_RANK, grid=CURVE_GRID,
                verify=True, verify_layers=None, verify_points=16,
            )
            self.assertEqual(meta["rank"], CURVE_RANK)
            self.assertEqual(meta["adaln_layers"], 3)  # 2 blocks + final layer
            self.assertGreaterEqual(meta["cosine_min"], 0.9999)
            # 单文件 checkpoint 必须改名，否则与原版在节点下拉框里同名
            shard = dst / "model-adaln_curve.safetensors"
            self.assertTrue(shard.is_file())
            self.assertFalse((dst / "model.safetensors").exists())
            self.assertEqual(_detect_adaln_curve(None, [shard]), (CURVE_GRID, CURVE_RANK))
            index = json.loads(
                (dst / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(set(index["weight_map"].values()), {shard.name})

            out_config = json.loads((dst / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(out_config["adaln_curve_grid"], CURVE_GRID)
            self.assertEqual(out_config["time_embed_dim"], CURVE_RANK)
            self.assertNotIn("timestep_input_dim", out_config)
            self.assertTrue(out_config["adaln_curve"]["silu_folded"])

            from safetensors import safe_open
            with safe_open(str(shard), framework="pt") as reader:
                keys = set(reader.keys())
            self.assertIn("adaln_t_table", keys)
            self.assertFalse([k for k in keys if k.startswith("time_embedder.")])

            # 转换产物装回模型后与原版逐点一致（满秩）
            from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
            curved = MiniMaxH3DiTModel.from_config(
                MiniMaxH3DiTConfig.from_dict(out_config), dtype=torch.float32
            ).eval()
            state = {}
            with safe_open(str(shard), framework="pt") as reader:
                for key in reader.keys():
                    state[key] = reader.get_tensor(key)
            curved.load_state_dict(state)
            kwargs = _fwd_kwargs()
            with torch.no_grad():
                want_v, want_a = plain(**kwargs)
                got_v, got_a = curved(**kwargs)
            self.assertTrue(torch.allclose(got_v, want_v, atol=2e-4, rtol=2e-4))
            self.assertTrue(torch.allclose(got_a, want_a, atol=2e-4, rtol=2e-4))

    def test_rejects_already_converted_source(self):
        import json
        import tempfile
        from pathlib import Path
        import torch
        from safetensors.torch import save_file
        tool = self._tool()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "transformer"
            src.mkdir()
            save_file({"adaln_t_table": torch.zeros(8, 4)}, str(src / "model.safetensors"))
            config = _tiny_cfg(adaln_curve_grid=8).as_dict()
            config["patch_size"] = list(config["patch_size"])
            (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(SystemExit):
                tool.run_convert(src, Path(tmp) / "out", rank=4, grid=8, dry_run=True)


if __name__ == "__main__":
    unittest.main()
