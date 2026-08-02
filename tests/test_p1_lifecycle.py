"""驻留租约 / 降档链 / 编码缓存 / sidecar。"""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest import mock

class TestDownscale(unittest.TestCase):
    def test_16_9_official_chain(self):
        from minimax_h3_nodes.runtime.downscale import downscale_chain, next_downscale, format_downscale_hint
        chain = downscale_chain(1344, 768)
        self.assertEqual(chain[0], (1344, 768))
        self.assertIn((1024, 576), chain)
        self.assertIn((832, 480), chain)
        self.assertIn((640, 352), chain)
        self.assertEqual(next_downscale(1344, 768), (1024, 576))
        self.assertIn("1344x768", format_downscale_hint(1344, 768))

    def test_square_preserves_ratio_direction(self):
        from minimax_h3_nodes.runtime.downscale import downscale_chain
        chain = downscale_chain(768, 768)
        self.assertEqual(chain[0], (768, 768))
        for w, h in chain[1:]:
            self.assertEqual(w, h)
            self.assertLess(w * h, 768 * 768)

class TestEncodeCache(unittest.TestCase):
    def test_lru_put_get_and_evict(self):
        from minimax_h3_nodes.runtime.encode_cache import EncodeCache, prompt_cache_key
        c = EncodeCache(max_bytes=64)
        k1 = prompt_cache_key(prompt="a", te_fingerprint="t1")
        k2 = prompt_cache_key(prompt="b", te_fingerprint="t1")
        with mock.patch("minimax_h3_nodes.runtime.encode_cache.OPT_ENCODE_CACHE", True):
            c.put(k1, b"x" * 40, nbytes=40)
            self.assertEqual(c.get(k1), b"x" * 40)
            c.put(k2, b"y" * 40, nbytes=40)  # 淘汰 k1
            self.assertIsNone(c.get(k1))
            self.assertEqual(c.get(k2), b"y" * 40)

class TestResidency(unittest.TestCase):
    def test_safe_policy_always_cold(self):
        from minimax_h3_nodes.runtime.residency import H3ResidencyManager
        class H:
            quantized = False; load_device = "cuda:0"; metadata = {"release_fingerprint": "fp1", "partition": "FL2VA"}
            def __init__(self): self.calls = []
            def park_after_inference(self): self.calls.append("park"); return "gpu-resident"
            def offload_after_inference(self): self.calls.append("cold")
        mgr = H3ResidencyManager(); h = H()
        with mock.patch("minimax_h3_nodes.runtime.residency.OPT_RESIDENCY_LEASE", True), \
             mock.patch("minimax_h3_nodes.runtime.residency.RESIDENCY_POLICY", "safe"):
            self.assertEqual(mgr.release(h, kind="dit"), "cold")
            self.assertEqual(h.calls, ["cold"])

    def test_balanced_parks_when_vram_ok(self):
        from minimax_h3_nodes.runtime.residency import H3ResidencyManager
        class H:
            quantized = False; load_device = "cuda:0"; metadata = {"release_fingerprint": "fp2", "partition": "FL2VA"}
            def park_after_inference(self): return "layerwise-warm"
            def offload_after_inference(self): raise AssertionError("should park")
        mgr = H3ResidencyManager()
        with mock.patch("minimax_h3_nodes.runtime.residency.OPT_RESIDENCY_LEASE", True), \
             mock.patch("minimax_h3_nodes.runtime.residency.RESIDENCY_POLICY", "balanced"), \
             mock.patch.object(H3ResidencyManager, "_vram_tight", return_value=False):
            self.assertEqual(mgr.release(H(), kind="dit"), "layerwise-warm")

class TestSidecar(unittest.TestCase):
    def test_sidecar_includes_runtime_env(self):
        from minimax_h3_nodes.runtime import sidecar
        meta = sidecar.build_sidecar_meta({"task": "t2va", "seed": 1})
        self.assertIn("env", meta)
        self.assertIsInstance(meta["env"], dict)

    def test_write_sidecar_json(self):
        from minimax_h3_nodes.runtime import sidecar
        meta = sidecar.build_sidecar_meta(
            {"task": "t2va", "seed": 1, "residency_mode": "full",
             "target": {"width": 832, "height": 480, "fps": 24, "frame_count": 124}}
        )
        self.assertEqual(meta["task"], "t2va")
        self.assertEqual(meta["width"], 832)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            sidecar, "OPT_WRITE_SIDECAR", True
        ), mock.patch.dict("sys.modules", {"folder_paths": mock.Mock(get_output_directory=lambda: tmp)}):
            path = sidecar.write_h3_sidecar(meta, stem="unit_test")
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).is_file())

if __name__ == "__main__":
    unittest.main()
