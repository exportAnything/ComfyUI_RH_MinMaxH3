"""adaLN 分段广播与 index_select 路径数值等价。"""
from __future__ import annotations
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestAdalnSegmentBroadcast(unittest.TestCase):
    def test_index_runs_merges_contiguous(self):
        import torch
        from minimax_h3_nodes.runtime.attention import _index_runs
        idx = torch.tensor([1, 1, 1, 0, 0, 2, 2, 2, 2])
        self.assertEqual(_index_runs(idx), ((0, 3, 1), (3, 5, 0), (5, 9, 2)))
        self.assertEqual(_index_runs(torch.tensor([7])), ((0, 1, 7),))
        self.assertEqual(_index_runs(torch.empty(0, dtype=torch.long)), ())

    def test_modulate_matches_index_select(self):
        import torch
        from minimax_h3_nodes.runtime.attention import (
            _index_runs, _modulate_scale_shift, _modulate_gate, _silu_mul,
        )
        seq, hidden, rows = 12, 8, 4
        x = torch.randn(seq, hidden)
        shift = torch.randn(rows, hidden)
        scale = torch.randn(rows, hidden)
        gate = torch.randn(rows, hidden)
        indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        # legacy index_select
        legacy_ss = (x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)).to(x.dtype)
        other = torch.randn(seq, hidden)
        legacy_gate = (x + gate.index_select(0, indices) * other).to(x.dtype)
        # segment in-place（需要 fresh clone）
        seg = _index_runs(indices)
        got_ss = _modulate_scale_shift(x.clone(), shift, scale, seg)
        got_gate = _modulate_gate(x, gate, other.clone(), seg)
        self.assertTrue(torch.allclose(got_ss, legacy_ss, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(got_gate, legacy_gate, atol=1e-6, rtol=1e-6))
        # Tensor 入参仍兼容
        got_ss2 = _modulate_scale_shift(x.clone(), shift, scale, indices)
        self.assertTrue(torch.allclose(got_ss2, legacy_ss, atol=1e-6, rtol=1e-6))

    def test_silu_mul_inplace_equiv(self):
        import torch
        from minimax_h3_nodes.runtime.attention import _silu_mul
        h = torch.randn(5, 16)
        gate, up = h.chunk(2, dim=-1)
        expect = torch.nn.functional.silu(gate) * up
        self.assertTrue(torch.allclose(_silu_mul(h.clone()), expect, atol=1e-6, rtol=1e-6))

    def test_tiny_block_forward_with_segments(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig, MiniMaxH3DiTModel
        from minimax_h3_nodes.runtime.attention import _index_runs
        from minimax_h3_nodes.runtime import h3_settings
        cfg = MiniMaxH3DiTConfig(
            num_layers=1, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=1,
            attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
            patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
            time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
            rope_inv_freq_len=1,
        )
        model = MiniMaxH3DiTModel(cfg, device="cpu", dtype=torch.float32).eval()
        seq = 6
        tags = torch.tensor([1, 1, 2, 2, 0, 0])
        inverse = torch.tensor([0, 0, 1, 1, 0, 0])
        self.assertEqual(len(_index_runs(inverse * 3 + tags.clamp(min=0))), 3)
        kwargs = {
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
        with torch.inference_mode():
            self.assertTrue(h3_settings.OPT_ADALN_SEGMENT_BROADCAST)
            v_seg, a_seg = model(**kwargs)
            prev = h3_settings.OPT_ADALN_SEGMENT_BROADCAST
            try:
                h3_settings.OPT_ADALN_SEGMENT_BROADCAST = False
                v_old, a_old = model(**kwargs)
            finally:
                h3_settings.OPT_ADALN_SEGMENT_BROADCAST = prev
        self.assertTrue(torch.allclose(v_seg, v_old, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(a_seg, a_old, atol=1e-5, rtol=1e-5))

if __name__ == "__main__":
    unittest.main()
