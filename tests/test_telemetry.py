"""telemetry / 矩阵汇总单测（无需 GPU）。"""
from __future__ import annotations
import json, tempfile, unittest
from pathlib import Path
from unittest import mock

class TestTelemetry(unittest.TestCase):
    def test_stage_and_summary_cpu(self):
        from minimax_h3_nodes.runtime.telemetry import H3Telemetry
        with mock.patch("minimax_h3_nodes.runtime.telemetry.OPT_TELEMETRY", True), \
             mock.patch("minimax_h3_nodes.runtime.telemetry.TELEMETRY_CUDA_EVENTS", False):
            tel = H3Telemetry(device="cpu", enabled=True)
            with tel.stage("packed_branch"):
                pass
            with tel.denoise_step(0):
                pass
            with tel.denoise_step(1):
                pass
            s = tel.summary()
            self.assertIn("packed_branch", s["stages_s"])
            self.assertEqual(s["step_count"], 2)
            self.assertIsNotNone(s["step_host_s"]["p50"])

    def test_abort_after_two_slow_steps(self):
        from minimax_h3_nodes.runtime.telemetry import H3Telemetry
        with mock.patch("minimax_h3_nodes.runtime.telemetry.OPT_TELEMETRY", True), \
             mock.patch("minimax_h3_nodes.runtime.telemetry.TELEMETRY_CUDA_EVENTS", False), \
             mock.patch("minimax_h3_nodes.runtime.telemetry.TELEMETRY_STEP_ABORT_SECONDS", 0.0):
            tel = H3Telemetry(enabled=True)
            with tel.denoise_step(0): pass
            with tel.denoise_step(1): pass
            self.assertIsNotNone(tel.aborted_reason)

    def test_abort_exempt(self):
        from minimax_h3_nodes.runtime.telemetry import H3Telemetry
        with mock.patch("minimax_h3_nodes.runtime.telemetry.OPT_TELEMETRY", True), \
             mock.patch("minimax_h3_nodes.runtime.telemetry.TELEMETRY_CUDA_EVENTS", False), \
             mock.patch("minimax_h3_nodes.runtime.telemetry.TELEMETRY_STEP_ABORT_SECONDS", 0.0):
            tel = H3Telemetry(enabled=True)
            with tel.denoise_step(0, abort_exempt=True): pass
            with tel.denoise_step(1, abort_exempt=True): pass
            self.assertIsNone(tel.aborted_reason)

class TestMatrixRunner(unittest.TestCase):
    def test_summarize_sidecars(self):
        from benchmarks import run_matrix
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = {
                "schema": "minimax_h3_sidecar/v1",
                "task": "t2va", "width": 832, "height": 480, "frame_count": 124,
                "seed": 42, "residency_mode": "partial",
                "telemetry": {
                    "step_host_s": {"p50": 1.0, "p95": 2.0},
                    "peak_vram": {"allocated": 1000},
                    "stages_s": {"denoise_loop": 10.0},
                    "accel": "off",
                },
            }
            (root / "h3_a.json").write_text(json.dumps(meta), encoding="utf-8")
            meta2 = json.loads(json.dumps(meta)); meta2["telemetry"]["step_host_s"]["p50"] = 3.0
            (root / "h3_b.json").write_text(json.dumps(meta2), encoding="utf-8")
            report = run_matrix.summarize(list(run_matrix.iter_sidecars(root)))
            self.assertEqual(len(report), 1)
            row = next(iter(report.values()))
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["step_p50_median"], 2.0)

    def test_matrix_lists_24gb_primary(self):
        matrix = json.loads((Path(__file__).resolve().parents[1] / "benchmarks" / "matrix.json").read_text(encoding="utf-8"))
        self.assertEqual(matrix["baseline_vram_gb"], 24)
        ids = [c["id"] for c in matrix["cases"]]
        self.assertIn("t2va_832x480_5s_int8_cold", ids)
        self.assertTrue(any(c.get("vram_tier") == "24gb" for c in matrix["cases"]))

if __name__ == "__main__":
    unittest.main()
