"""实验性 Frame Rate：TimeEmbedder fps 项 + RoPE 时序缩放。"""
from __future__ import annotations
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestFrameRate(unittest.TestCase):
    def test_validate_and_scale(self):
        from minimax_h3_nodes.runtime.frame_rate import (
            validate_frame_rate_options, rope_temporal_scale, adaln_frame_rate,
        )
        opts = validate_frame_rate_options(frame_rate=12.0, adaln=True, temporal_rope=True)
        self.assertEqual(adaln_frame_rate(opts), 12.0)
        self.assertAlmostEqual(rope_temporal_scale(opts, video_timestep=0.5, video_sigma=0.5), 2.0)
        self.assertEqual(rope_temporal_scale(opts, video_timestep=1.1, video_sigma=0.0), 1.0)
        native = validate_frame_rate_options(frame_rate=24.0, temporal_rope=True)
        self.assertEqual(rope_temporal_scale(native, video_timestep=0.5, video_sigma=0.5), 1.0)
        with self.assertRaises(ValueError):
            validate_frame_rate_options(adaln=False, temporal_rope=False)

    def test_time_embedder_fps_changes_output(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig, MiniMaxH3TimeEmbedder
        cfg = MiniMaxH3DiTConfig(
            timestep_input_dim=8, time_embed_hidden_size=16, time_embed_dim=8,
        )
        emb = MiniMaxH3TimeEmbedder(cfg, device="cpu").eval()
        t = torch.tensor([0.25, 0.75])
        with torch.inference_mode():
            a = emb(t)
            b = emb(t, frame_rate=24.0)
            c = emb(t, frame_rate=12.0)
        self.assertFalse(torch.allclose(a, b))  # 启用 fps 即使 24 也非 no-op
        self.assertFalse(torch.allclose(b, c))

    def test_rope_temporal_scale_affects_video_rows(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3Rope
        rope = MiniMaxH3Rope(4, device="cpu")
        pos = torch.zeros(1, 4, 3, dtype=torch.float32)
        pos[0, :, 0] = torch.arange(4)
        mask = torch.tensor([True, True, False, False])
        with torch.inference_mode():
            base = rope(pos)
            scaled = rope(
                pos, video_mask=mask, temporal_scale=2.0,
                low_frequency_count=2, frequency_profile="hard",
            )
        self.assertFalse(torch.allclose(base, scaled))
        # 非视频行应不变
        self.assertTrue(torch.allclose(base[2:], scaled[2:]))

    def test_precompute_frame_rate_key(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig, MiniMaxH3DiTModel
        cfg = MiniMaxH3DiTConfig(
            num_layers=1, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=1,
            attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
            patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
            time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
            rope_inv_freq_len=1,
        )
        model = MiniMaxH3DiTModel(cfg, device="cpu", dtype=torch.float32).eval()
        cache = model.precompute_modulation(
            [0.25, 0.5], release_weights=True, frame_rate=12.0,
        )
        self.assertEqual(cache.frame_rate, 12.0)
        with self.assertRaises(RuntimeError):
            model.precompute_modulation([0.25, 0.5], release_weights=True, frame_rate=24.0)

if __name__ == "__main__":
    unittest.main()
