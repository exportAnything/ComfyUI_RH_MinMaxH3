"""单卡加速：Cache-DiT profile / velocity-cache 调度合同（无需 GPU）。"""
from __future__ import annotations
import sys, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_T2VA = {"width": 1344, "height": 768, "fps": 24, "frame_count": 124}


class QualityProfileTests(unittest.TestCase):
    def test_off_returns_none(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_accel_request

        self.assertIsNone(
            resolve_accel_request(
                "off", task="t2va", target=_T2VA, sigma_points=50,
                video_shift=12.0, audio_shift=3.0,
            )
        )

    def test_auto_prefers_velocity_on_single_gpu_workload(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_accel_request

        cfg = resolve_accel_request(
            "auto", task="t2va", target=_T2VA, sigma_points=50,
            video_shift=12.0, audio_shift=3.0,
        )
        self.assertEqual(cfg.kind, "velocity-cache")
        self.assertEqual(cfg.velocity.profile_id, "minimax-h3-velocity-cache-v1")
        self.assertEqual(cfg.velocity.stride, 4)
        self.assertTrue(cfg.velocity.taylorseer)
        self.assertEqual(cfg.velocity.tail_dense_steps, 2)
        self.assertTrue(cfg.velocity.tail_rebalance)

    def test_auto_miss_returns_none(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_accel_request

        self.assertIsNone(
            resolve_accel_request(
                "auto", task="t2va",
                target={"width": 960, "height": 544, "fps": 24, "frame_count": 124},
                sigma_points=50, video_shift=12.0, audio_shift=3.0,
            )
        )

    def test_cache_dit_profile_still_resolves(self):
        from minimax_h3_nodes.runtime import quality_profiles as qp

        with mock.patch.object(qp, "_pkg_ok", return_value=None):
            cfg = qp.resolve_accel_request(
                "minimax-h3-cache-v1", task="t2va", target=_T2VA,
                sigma_points=50, video_shift=12.0, audio_shift=3.0,
            )
        self.assertEqual(cfg.kind, "cache-dit")
        self.assertAlmostEqual(cfg.cache_dit.residual_diff_threshold, 0.08)

    def test_manual_velocity_no_package(self):
        from minimax_h3_nodes.runtime.quality_profiles import resolve_accel_request

        cfg = resolve_accel_request(
            "manual-velocity", task="t2va", target={}, sigma_points=50,
            video_shift=12.0, audio_shift=3.0, velocity_stride=3,
        )
        self.assertEqual(cfg.kind, "velocity-cache")
        self.assertEqual(cfg.velocity.stride, 3)


class VelocityScheduleTests(unittest.TestCase):
    def _run(self, **kw):
        from minimax_h3_nodes.runtime.velocity_cache import (
            VelocityCacheConfig, VelocityCacheRuntime, should_refresh_velocity,
        )

        cfg = VelocityCacheConfig(**kw)
        n = 10
        rt = VelocityCacheRuntime(cfg).bind(n)
        routed = [i for i in range(n) if should_refresh_velocity(i, n, cfg)]
        # simulate stats bookkeeping
        for step in range(n):
            if rt.refresh(step):
                rt.on_dit(step, step + 1.0, step + 2.0)
            else:
                rt.on_hit(step)
        return routed, rt.stats()

    def test_stride_edges(self):
        routed, stats = self._run(stride=4)
        self.assertEqual(routed, [0, 1, 5, 9])
        self.assertEqual(stats["dit_calls"], 4)
        self.assertEqual(stats["cache_hits"], 6)

    def test_dense_tail(self):
        routed, stats = self._run(stride=4, taylorseer=True, tail_dense_steps=2)
        self.assertEqual(routed, [0, 1, 5, 8, 9])
        self.assertEqual(stats["dit_calls"], 5)

    def test_rebalance_matches_official(self):
        routed, stats = self._run(
            stride=4, taylorseer=True, tail_dense_steps=2, tail_rebalance=True
        )
        self.assertEqual(routed, [0, 1, 8, 9])
        self.assertEqual(stats["dit_calls"], 4)
        self.assertEqual(stats["tail_rebalance"], 1)


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
        self.assertEqual(captured["forward_pattern"], CACHE_DIT_FORWARD_PATTERN)
        self.assertFalse(captured["has_separate_cfg"])


class SamplerAccelHookTests(unittest.TestCase):
    def test_prepare_returns_velocity_runtime(self):
        from minimax_h3_nodes import sampling
        from minimax_h3_nodes.runtime.velocity_cache import VelocityCacheRuntime

        rt = sampling._prepare_accel_for_sample(
            object(),
            accel="manual-velocity",
            task="t2va",
            target={},
            sigma_points=50,
            video_shift=12.0,
            audio_shift=3.0,
            num_denoise_steps=49,
            velocity_stride=4,
        )
        self.assertIsInstance(rt, VelocityCacheRuntime)
        self.assertEqual(rt.cfg.stride, 4)
        self.assertEqual(rt.stats()["global_steps"], 49)


if __name__ == "__main__":
    unittest.main()
