"""V2A packed 冻结与 layout 放宽。"""
from __future__ import annotations
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestV2APrepare(unittest.TestCase):
    def test_prepare_freezes_all_video_rows(self):
        import torch
        from minimax_h3_nodes.runtime.sampler_core import (
            _prepare_v2a_packed, patchify_video_latent,
        )
        t, h, w = 2, 4, 4  # latent T/H/W，patch 后 rows=2*(4/2)*(4/2)=8
        video = torch.randn(1, 24, t, h, w)
        n = int(patchify_video_latent(video).shape[0])
        packed = {
            "update_mask": torch.ones(n, dtype=torch.bool),
            "audio_update_mask": torch.ones(4, dtype=torch.bool),
            "img_pos": torch.arange(n),
            "audio_pos": torch.arange(n, n + 4),
            "seq_len": n + 4,
        }
        out = _prepare_v2a_packed(packed, video)
        self.assertFalse(bool(out["update_mask"].any()))
        self.assertEqual(int(out["visual_cond_rows"].shape[0]), n)
        self.assertEqual(out["visual_condition_shapes"], [(t, h, w)])
        with self.assertRaises(ValueError):
            _prepare_v2a_packed(packed, torch.zeros_like(video))
        bad = dict(packed)
        bad["update_mask"] = torch.tensor([False] + [True] * (n - 1))
        with self.assertRaises(ValueError):
            _prepare_v2a_packed(bad, video)

    def test_layout_allows_frozen_video(self):
        import torch
        from minimax_h3_nodes.runtime.sampler_core import (
            _prepare_v2a_packed, _validate_conditional_layout, patchify_video_latent,
        )
        t, h, w = 2, 4, 4
        video = torch.randn(1, 24, t, h, w)
        n = int(patchify_video_latent(video).shape[0])
        packed = {
            "update_mask": torch.ones(n, dtype=torch.bool),
            "audio_update_mask": torch.ones(4, dtype=torch.bool),
            "img_pos": torch.arange(n),
            "audio_pos": torch.arange(n, n + 4),
            "seq_len": n + 4,
        }
        out = _prepare_v2a_packed(packed, video)
        class _B: pass
        b = _B()
        b.update_mask = out["update_mask"]
        b.audio_update_mask = out["audio_update_mask"]
        b.img_pos = out["img_pos"]
        b.audio_pos = out["audio_pos"]
        b.seq_len = out["seq_len"]
        with self.assertRaises(ValueError):
            _validate_conditional_layout(b, allow_frozen_video=False)
        _validate_conditional_layout(b, allow_frozen_video=True)

if __name__ == "__main__":
    unittest.main()
