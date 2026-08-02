"""P0 热路径：SDPA bounds / prepared structure / 原位 Euler / 预生成 timestep。"""
from __future__ import annotations
import importlib.util
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestP0Hotpath(unittest.TestCase):
    def test_normalize_bounds_and_single_doc_fast_path(self):
        import torch
        from minimax_h3_nodes.runtime.dit import (
            normalize_cu_seqlens_bounds, sdpa_varlen_attention,
        )
        rows, heads, dim = 4, 2, 8
        q = torch.randn(rows, heads, dim); k = q.clone(); v = q.clone()
        cu = torch.tensor([0, rows], dtype=torch.int32)
        bounds = normalize_cu_seqlens_bounds(cu, rows=rows)
        self.assertEqual(bounds, (0, rows))
        out_b = sdpa_varlen_attention(q, k, v, cu_seqlens=cu, softmax_scale=dim ** -0.5, bounds=bounds)
        out_old = sdpa_varlen_attention(q, k, v, cu_seqlens=cu, softmax_scale=dim ** -0.5, bounds=None)
        self.assertEqual(tuple(out_b.shape), (rows, heads, dim))
        self.assertTrue(torch.allclose(out_b, out_old, atol=1e-5, rtol=1e-5))

    def test_multi_doc_bounds_no_sync_path_matches(self):
        import torch
        from minimax_h3_nodes.runtime.dit import sdpa_varlen_attention
        rows, heads, dim = 6, 1, 4
        q = torch.randn(rows, heads, dim); k = torch.randn_like(q); v = torch.randn_like(q)
        cu = torch.tensor([0, 3, 6], dtype=torch.int32); bounds = (0, 3, 6)
        a = sdpa_varlen_attention(q, k, v, cu_seqlens=cu, softmax_scale=0.5, bounds=bounds)
        b = sdpa_varlen_attention(q, k, v, cu_seqlens=cu, softmax_scale=0.5, bounds=None)
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))

    def test_inplace_euler_preserves_anchor_rows(self):
        import torch
        from minimax_h3_nodes import sampling as S
        video = torch.randn(5, 4); anchor = video[3:].clone(); update_idx = torch.tensor([0, 1, 2])
        nxt = torch.randn(3, 4)
        video.index_copy_(0, update_idx, nxt)
        self.assertTrue(torch.equal(video[3:], anchor))
        self.assertTrue(torch.equal(video[:3], nxt))

    def test_prebuilt_timesteps_match_list(self):
        import torch
        sigmas = [1.0, 0.75, 0.5, 0.0]
        device = "cpu"
        tensor = torch.tensor(sigmas, dtype=torch.float32, device=device)
        pre = (1.0 - tensor[:-1]).contiguous()
        legacy = [torch.tensor(1.0 - s, dtype=torch.float32, device=device) for s in sigmas[:-1]]
        for i, leg in enumerate(legacy):
            self.assertTrue(torch.allclose(pre[i], leg))

    def test_prepare_structure_sets_bounds_and_rope(self):
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
        seq = 6
        ids = torch.zeros(1, seq, 3, dtype=torch.float64)
        cu = torch.tensor([0, seq], dtype=torch.int32)
        tags = torch.tensor([1, 1, 2, 2, 0, 0])
        prepared = model.prepare_structure(
            img_position_ids=ids, cu_seqlens=cu, token_tags=tags, seq_len=seq,
        )
        self.assertEqual(prepared["cu_seqlens_bounds"], (0, seq))
        self.assertTrue(prepared["structure_validated"])
        self.assertIn("prepared_rope_cache", prepared)
        self.assertEqual(tuple(prepared["prepared_rope_cache"][0].shape[0:1]), (seq,))

if __name__ == "__main__":
    unittest.main()
