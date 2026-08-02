"""BF16 DiT layerwise offload 合同测试。"""
from __future__ import annotations
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None

def _bf16_handle(**kw):
    from minimax_h3_nodes.runtime.model_loader import H3ModelHandle
    base = dict(
        model=object(), model_patcher=None, component_path=Path("/tmp/x"),
        load_device="cuda:0", offload_device="cpu", dtype="bfloat16",
        metadata={}, checkpoint_files=(), quantized=False,
    ); base.update(kw); return H3ModelHandle(**base)

class TestLayerwiseOffload(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "torch 未安装")
    def test_cpu_toy_blocks_prefetch_and_release(self):
        import torch
        from minimax_h3_nodes.runtime.layerwise_offload import (
            attach_layerwise_offload, get_layerwise_offload,
        )
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(4, 4)
                self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
                self.layer_names = ["blocks"]
            def forward(self, x):
                x = self.head(x)
                for block in self.blocks: x = block(x)
                return x
        model = Toy(); mgr = attach_layerwise_offload(model, device="cpu", layers_attr="blocks")
        self.assertIs(get_layerwise_offload(model), mgr)
        mgr.pin_memory = False; mgr.enable()
        self.assertTrue(mgr._enabled)
        self.assertTrue(all(p.device.type == "cpu" for p in model.head.parameters()))
        y = model(torch.randn(2, 4)); self.assertEqual(tuple(y.shape), (2, 4))
        self.assertEqual(mgr._gpu, set())  # post-hook 释放后不应残留
        mgr.disable(); self.assertFalse(mgr._enabled)

    def test_layerwise_auto_off_when_full_reside_fits(self):
        handle = _bf16_handle()
        with mock.patch(
            "minimax_h3_nodes.runtime.model_loader._impl.ENABLE_DIT_LAYERWISE_OFFLOAD", True
        ), mock.patch.object(handle, "_decide_residency", return_value="full"):
            self.assertFalse(handle._use_layerwise())  # 大显存：整模放得下 → 关

    def test_layerwise_auto_on_when_full_reside_does_not_fit(self):
        handle = _bf16_handle()
        with mock.patch(
            "minimax_h3_nodes.runtime.model_loader._impl.ENABLE_DIT_LAYERWISE_OFFLOAD", True
        ), mock.patch.object(handle, "_decide_residency", return_value="layerwise"):
            self.assertTrue(handle._use_layerwise())  # 24GB：放不下 → 开

    def test_layerwise_reject_raises(self):
        handle = _bf16_handle()
        with mock.patch(
            "minimax_h3_nodes.runtime.model_loader._impl.ENABLE_DIT_LAYERWISE_OFFLOAD", True
        ), mock.patch.object(handle, "_decide_residency", return_value="reject"):
            with self.assertRaisesRegex(RuntimeError, "降低画布"):
                handle._use_layerwise()

    def test_layerwise_disabled_flag_forces_full(self):
        handle = _bf16_handle()
        with mock.patch(
            "minimax_h3_nodes.runtime.model_loader._impl.ENABLE_DIT_LAYERWISE_OFFLOAD", False
        ), mock.patch.object(handle, "_decide_residency", return_value="layerwise"):
            self.assertFalse(handle._use_layerwise())

    def test_quantized_never_uses_layerwise(self):
        handle = _bf16_handle(quantized=True)
        with mock.patch(
            "minimax_h3_nodes.runtime.model_loader._impl.ENABLE_DIT_LAYERWISE_OFFLOAD", True
        ), mock.patch.object(handle, "_decide_residency", return_value="layerwise"):
            self.assertFalse(handle._use_layerwise())

    def test_vram_probe_compares_free_to_weights_plus_reserve(self):
        from minimax_h3_nodes.runtime.model_loader import _impl as model_loader
        handle = _bf16_handle()
        patches = dict(
            activation=mock.patch.object(handle, "_activation_reserve_bytes", return_value=100),
            nbytes=mock.patch.object(model_loader, "_model_nbytes", return_value=1000),
        )
        with patches["activation"], patches["nbytes"], mock.patch.object(
            model_loader, "_device_free_bytes", return_value=2000
        ):
            self.assertEqual(handle._decide_residency(), "full")
        with patches["activation"], patches["nbytes"], mock.patch.object(
            model_loader, "_device_free_bytes", return_value=500
        ):
            self.assertEqual(handle._decide_residency(), "layerwise")
        with patches["activation"], patches["nbytes"], mock.patch.object(
            model_loader, "_device_free_bytes", return_value=None
        ):
            self.assertEqual(handle._decide_residency(), "layerwise")  # 探测失败保守

if __name__ == "__main__":
    unittest.main()
