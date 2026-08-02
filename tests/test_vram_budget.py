"""P0-6 激活预算与驻留分档。"""
from __future__ import annotations
import unittest
from minimax_h3_nodes.runtime.vram_budget import (
    estimate_dit_activation_bytes, resolve_activation_reserve, residency_tier,
)
from minimax_h3_nodes.runtime.h3_settings import (
    DIT_INFERENCE_RESERVE, DIT_ACTIVATION_FLOOR, DIT_SAFETY_MARGIN,
)

class TestVramBudget(unittest.TestCase):
    def test_empty_shape_falls_back_to_default_reserve(self):
        self.assertEqual(estimate_dit_activation_bytes(), DIT_INFERENCE_RESERVE)

    def test_seq_len_scales_and_clamps(self):
        small = estimate_dit_activation_bytes(seq_len=10)
        big = estimate_dit_activation_bytes(seq_len=100000)
        self.assertGreaterEqual(small, DIT_ACTIVATION_FLOOR)
        self.assertGreater(big, small)

    def test_resolve_adds_safety_margin(self):
        r = resolve_activation_reserve(seq_len=100)
        self.assertGreaterEqual(r, DIT_ACTIVATION_FLOOR)
        self.assertGreaterEqual(r, DIT_SAFETY_MARGIN)

    def test_residency_tiers(self):
        self.assertEqual(residency_tier(free_bytes=100, weight_bytes=40, activation_bytes=10), "full")
        self.assertEqual(residency_tier(free_bytes=30, weight_bytes=40, activation_bytes=10), "layerwise")
        self.assertEqual(residency_tier(free_bytes=1, weight_bytes=40, activation_bytes=10), "reject")
        self.assertEqual(residency_tier(free_bytes=None, weight_bytes=40, activation_bytes=10), "layerwise")

    def test_ewma_observed_peak_updates_reserve(self):
        from minimax_h3_nodes.runtime import vram_budget
        vram_budget._EWMA.clear()
        hint = {"seq_len": 1000, "task": "t2va", "dtype": "bf16"}
        first = vram_budget.resolve_activation_reserve(seq_len=1000, task="t2va")
        second = vram_budget.note_observed_activation(hint, observed_peak=first * 2)
        self.assertNotEqual(first, second)
        third = vram_budget.resolve_activation_reserve(seq_len=1000, task="t2va")
        self.assertEqual(second, third)

if __name__ == "__main__":
    unittest.main()
