"""Cache-DiT quality profile / BlockAdapter 合同（无需真实 GPU / cache-dit）。"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class QualityProfileTests(unittest.TestCase):
    def test_off_returns_none(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_cache_dit_request

        self.assertIsNone(
            resolve_cache_dit_request(
                "off",
                task="t2va",
                target={"width": 1344, "height": 768, "fps": 24, "frame_count": 124},
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
            )
        )

    def test_auto_miss_returns_none(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_cache_dit_request

        self.assertIsNone(
            resolve_cache_dit_request(
                "auto",
                task="t2va",
                target={"width": 960, "height": 544, "fps": 24, "frame_count": 124},
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
            )
        )

    def test_profile_hit_requires_package(self):
        from minimax_h3_nodes.runtime import quality_profiles as qp

        target = {"width": 1344, "height": 768, "fps": 24, "frame_count": 124}
        with mock.patch.object(qp, "_pkg_ok", return_value=None):
            cfg = qp.resolve_cache_dit_request(
                "minimax-h3-cache-v1",
                task="t2va",
                target=target,
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
            )
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.profile_id, "minimax-h3-cache-v1")
        self.assertAlmostEqual(cfg.residual_diff_threshold, 0.08)
        self.assertEqual(cfg.max_continuous_cached_steps, 2)
        self.assertEqual(cfg.max_warmup_steps, 4)

    def test_auto_hit_selects_profile(self):
        from minimax_h3_nodes.runtime import quality_profiles as qp

        target = {"width": 1344, "height": 768, "fps": 24, "frame_count": 124}
        with mock.patch.object(qp, "_pkg_ok", return_value=None):
            cfg = qp.resolve_cache_dit_request(
                "auto",
                task="t2va",
                target=target,
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
            )
        self.assertEqual(cfg.profile_id, "minimax-h3-cache-v1")

    def test_profile_mismatch_raises(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_cache_dit_request

        with self.assertRaisesRegex(ValueError, "不匹配"):
            resolve_cache_dit_request(
                "minimax-h3-cache-v1",
                task="t2va",
                target={"width": 1344, "height": 768, "fps": 24, "frame_count": 100},
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
            )

    def test_manual_uses_cookbook_and_overrides(self):
        from minimax_h3_nodes.runtime import quality_profiles as qp
        from minimax_h3_nodes.runtime.h3_settings import CACHE_DIT_RDT_COOKBOOK

        with mock.patch.object(qp, "_pkg_ok", return_value=None):
            cfg = qp.resolve_cache_dit_request(
                "manual",
                task="t2va",
                target={},
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
                rdt=0.2,
            )
            cfg2 = qp.resolve_cache_dit_request(
                "manual",
                task="t2va",
                target={},
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
            )
        self.assertIsNone(cfg.profile_id)
        self.assertAlmostEqual(cfg.residual_diff_threshold, 0.2)
        self.assertAlmostEqual(cfg2.residual_diff_threshold, CACHE_DIT_RDT_COOKBOOK)


class BlockAdapterContractTests(unittest.TestCase):
    def test_minimax_h3_pattern_3_adapter(self):
        from minimax_h3_nodes.runtime import cache_dit_integration as mod
        from minimax_h3_nodes.runtime.h3_settings import CACHE_DIT_FORWARD_PATTERN

        class MiniMaxH3DiTModel:
            def __init__(self):
                self.blocks = ["b0", "b1"]

        class ForwardPattern:
            Pattern_3 = "Pattern_3"

        captured = {}

        class BlockAdapter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        mod._build_adapter(MiniMaxH3DiTModel(), BlockAdapter, ForwardPattern)
        self.assertEqual(captured["blocks"], ["b0", "b1"])
        self.assertEqual(captured["forward_pattern"], CACHE_DIT_FORWARD_PATTERN)
        self.assertFalse(captured["has_separate_cfg"])

    def test_prepare_noop_when_disabled(self):
        from minimax_h3_nodes.runtime.cache_dit_integration import (
            prepare_transformer_cache_dit,
        )
        from minimax_h3_nodes.runtime.quality_profiles import CacheDitResolved

        model = object()
        cfg = CacheDitResolved(
            False, None, 1, 0, 4, 0.12, 2, False, 1, "none", "dynamic"
        )
        self.assertIs(
            prepare_transformer_cache_dit(model, cfg, num_denoise_steps=49), model
        )


class SamplerCacheHookTests(unittest.TestCase):
    def test_prepare_hook_passes_denoise_step_count(self):
        from minimax_h3_nodes import sampling
        from minimax_h3_nodes.runtime import quality_profiles as qp
        from minimax_h3_nodes.runtime.quality_profiles import CacheDitResolved

        transformer = object()
        seen = {}

        def fake_prepare(model, cfg, *, num_denoise_steps):
            seen["model"] = model
            seen["cfg"] = cfg
            seen["steps"] = num_denoise_steps
            return model

        cfg = CacheDitResolved(
            True, None, 1, 0, 4, 0.12, 2, False, 1, "none", "dynamic"
        )
        with (
            mock.patch.object(qp, "resolve_cache_dit_request", return_value=cfg),
            mock.patch(
                "minimax_h3_nodes.runtime.cache_dit_integration.prepare_transformer_cache_dit",
                side_effect=fake_prepare,
            ),
        ):
            sampling._prepare_cache_dit_for_sample(
                transformer,
                cache_dit="manual",
                task="t2va",
                target={"width": 1344},
                sigma_points=50,
                video_shift=12.0,
                audio_shift=3.0,
                num_denoise_steps=49,
            )
        self.assertIs(seen["model"], transformer)
        self.assertEqual(seen["steps"], 49)
        self.assertTrue(seen["cfg"].enabled)


if __name__ == "__main__":
    unittest.main()
