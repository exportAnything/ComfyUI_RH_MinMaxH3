"""AdaLN 预计算缓存与即时路径数值等价。"""
from __future__ import annotations
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

def _tiny_cfg():
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
    return MiniMaxH3DiTConfig(
        num_layers=2, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=1,
        attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
        patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
        rope_inv_freq_len=1,
    )

def _fwd_kwargs(seq=6):
    import torch
    tags = torch.tensor([1, 1, 2, 2, 0, 0])
    inverse = torch.tensor([0, 0, 1, 1, 0, 0])
    return {
        "x": torch.randn(1, seq, 1), "audio_x": torch.randn(1, seq, 2),
        "img_position_ids": torch.zeros(1, seq, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.25, 0.5]), "inverse_indices": inverse,
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

@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestAdalnPrecompute(unittest.TestCase):
    def test_enumerate_includes_floors_and_shifts(self):
        from minimax_h3_nodes.runtime.modulation_cache import enumerate_modulation_timesteps
        from minimax_h3_nodes.sampling import shifted_sigma_schedule
        v = shifted_sigma_schedule(sigma_points=8, shift=12.0)
        a = shifted_sigma_schedule(sigma_points=8, shift=3.0)
        ts = enumerate_modulation_timesteps(v, a)
        self.assertIn(0.999, ts)
        self.assertIn(1.0, ts)
        self.assertGreaterEqual(len(ts), 4)
        self.assertEqual(ts, sorted(ts))

    def test_precompute_matches_eager_and_release(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
        model = MiniMaxH3DiTModel(_tiny_cfg(), device="cpu", dtype=torch.float32).eval()
        kwargs = _fwd_kwargs()
        with torch.inference_mode():
            v0, a0 = model(**kwargs)
            cache = model.precompute_modulation(
                [0.25, 0.5, 0.999, 1.0], compute_device="cpu",
                cache_device="cpu", release_weights=True,
            )
            self.assertGreater(cache.bytes(), 0)
            self.assertTrue(model._adaln_weights_released)
            # 权重已清空
            self.assertEqual(int(model.blocks[0].adaln_proj.linear.weight.numel()), 0)
            v1, a1 = model(**kwargs)
        self.assertTrue(torch.allclose(v0, v1, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(a0, a1, atol=1e-5, rtol=1e-5))

    def test_released_rejects_new_timesteps(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
        model = MiniMaxH3DiTModel(_tiny_cfg(), device="cpu", dtype=torch.float32).eval()
        model.precompute_modulation([0.25, 0.5], release_weights=True)
        with self.assertRaises(RuntimeError):
            model.precompute_modulation([0.25, 0.5, 0.75], release_weights=True)

    def test_skips_when_comfy_managed(self):
        """INT8/lowvram 的 comfy ops Linear 由 ModelPatcher 记账，必须跳过预计算。"""
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
        from minimax_h3_nodes.runtime.modulation_cache import (
            H3PrecomputeUnsupported, _comfy_managed,
        )
        self.assertFalse(_comfy_managed(torch.nn.Linear(2, 2)))

        class _ComfyLinear(torch.nn.Linear):  # 模拟 comfy.ops 的 cast-weights Linear
            comfy_cast_weights = True

        self.assertTrue(_comfy_managed(_ComfyLinear(2, 2)))
        model = MiniMaxH3DiTModel(_tiny_cfg(), device="cpu", dtype=torch.float32).eval()
        old = model.blocks[0].adaln_proj.linear
        managed = _ComfyLinear(old.in_features, old.out_features)
        model.blocks[0].adaln_proj.linear = managed
        with self.assertRaises(H3PrecomputeUnsupported):
            model.precompute_modulation([0.25, 0.5], release_weights=True)
        # 未释放任何权重
        self.assertGreater(int(model.blocks[1].adaln_proj.linear.weight.numel()), 0)
        self.assertFalse(getattr(model, "_adaln_weights_released", False))

if __name__ == "__main__":
    unittest.main()
