"""5.1：官方 H3 processor 字段契约。"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

class TestProcessorContract(unittest.TestCase):
    def _write(self, root: Path, name: str, data: dict) -> None:
        (root / name).write_text(json.dumps(data), encoding="utf-8")

    def test_official_image_and_video_pass(self):
        from minimax_h3_nodes.runtime.qwen_encoder.helpers import (
            validate_h3_processor_contract,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "preprocessor_config.json", {
                "size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            })
            self._write(root, "video_preprocessor_config.json", {
                "size": {"longest_edge": 25165824, "shortest_edge": 4096},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            })
            validate_h3_processor_contract(root)

    def test_pr15210_hardcoded_pixels_rejected(self):
        from minimax_h3_nodes.runtime.components import H3ComponentError
        from minimax_h3_nodes.runtime.qwen_encoder.helpers import (
            validate_h3_processor_contract,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # PR #15210 硬编码 min_pixels=3136 / max_pixels=12845056
            self._write(root, "preprocessor_config.json", {
                "size": {"min_pixels": 3136, "max_pixels": 12845056},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            })
            with self.assertRaises(H3ComponentError) as ctx:
                validate_h3_processor_contract(root, require_video=False)
            self.assertIn("shortest/longest", str(ctx.exception))

    def test_wrong_mean_rejected(self):
        from minimax_h3_nodes.runtime.components import H3ComponentError
        from minimax_h3_nodes.runtime.qwen_encoder.helpers import (
            validate_h3_processor_contract,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write(root, "preprocessor_config.json", {
                "size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                "image_mean": [0.481, 0.457, 0.408], "image_std": [0.5, 0.5, 0.5],
            })
            with self.assertRaises(H3ComponentError):
                validate_h3_processor_contract(root, require_video=False)

if __name__ == "__main__":
    unittest.main()
