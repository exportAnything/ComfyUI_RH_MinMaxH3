"""INT8 MLP 的 swiglu 折进激活量化 kernel（PR#15224）：探测 + 回退。"""
from __future__ import annotations
import importlib.util
import sys
import types
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _tiny_cfg():
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
    return MiniMaxH3DiTConfig(
        num_layers=1, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=1,
        attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
        patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
        rope_inv_freq_len=1,
    )


class _NativeOps:
    """operations 注入点：用原生 Linear，只为触发"走 Comfy ops"这条分支。"""

    def __init__(self):
        import torch.nn as nn
        self.Linear = nn.Linear


def _fake_comfy_ops(recorder):
    module = types.ModuleType("comfy.ops")

    def linear_input_act(linear, x, input_act):
        import torch.nn.functional as F
        recorder.append(input_act)
        gate, up = x.chunk(2, dim=-1)
        return linear(F.silu(gate) * up)

    module.linear_input_act = linear_input_act
    parent = types.ModuleType("comfy")
    parent.ops = module
    return {"comfy": parent, "comfy.ops": module}


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestFusedSwiglu(unittest.TestCase):
    def setUp(self):
        from minimax_h3_nodes.runtime.dit._impl import _reset_fused_swiglu_probe
        _reset_fused_swiglu_probe()
        self.addCleanup(_reset_fused_swiglu_probe)

    def _mlp(self, operations):
        from minimax_h3_nodes.runtime.dit._impl import MiniMaxH3MLP
        import torch
        return MiniMaxH3MLP(
            _tiny_cfg(), dtype=torch.float32, device="cpu", operations=operations
        )

    def test_native_linear_never_uses_the_fused_entry(self):
        # 没有 operations 就没有可折叠的量化 kernel，必须留在 eager 路径
        recorder: list[str] = []
        with mock.patch.dict(sys.modules, _fake_comfy_ops(recorder)):
            mlp = self._mlp(None)
        self.assertIsNone(mlp.fused_swiglu)
        import torch
        mlp(torch.randn(3, 8))
        self.assertEqual(recorder, [])

    def test_comfy_ops_path_uses_fused_entry_when_available(self):
        import torch
        recorder: list[str] = []
        with mock.patch.dict(sys.modules, _fake_comfy_ops(recorder)):
            mlp = self._mlp(_NativeOps())
            self.assertIsNotNone(mlp.fused_swiglu)
            got = mlp(torch.randn(3, 8))
        self.assertEqual(recorder, ["swiglu"])
        self.assertEqual(tuple(got.shape), (3, 8))

    def test_fused_and_eager_paths_agree_numerically(self):
        import torch
        from minimax_h3_nodes.runtime.attention import _silu_mul
        recorder: list[str] = []
        with mock.patch.dict(sys.modules, _fake_comfy_ops(recorder)):
            mlp = self._mlp(_NativeOps())
            x = torch.randn(5, 8)
            fused = mlp(x)
            mlp.fused_swiglu = None  # 同一组权重走 eager
            eager = mlp(x)
        self.assertTrue(torch.allclose(fused, eager, atol=1e-6, rtol=1e-6))
        # eager 实现本身与教科书写法一致
        gate_up = mlp.fc1(x)
        gate, up = gate_up.chunk(2, dim=-1)
        self.assertTrue(
            torch.allclose(
                mlp.fc2(_silu_mul(gate_up.clone())),
                mlp.fc2(torch.nn.functional.silu(gate) * up),
                atol=1e-6, rtol=1e-6,
            )
        )

    def test_missing_comfy_falls_back_to_eager(self):
        import torch
        empty = types.ModuleType("comfy.ops")  # 旧版 comfy：没有 linear_input_act
        parent = types.ModuleType("comfy")
        parent.ops = empty
        with mock.patch.dict(sys.modules, {"comfy": parent, "comfy.ops": empty}):
            mlp = self._mlp(_NativeOps())
        self.assertIsNone(mlp.fused_swiglu)
        self.assertEqual(tuple(mlp(torch.randn(2, 8)).shape), (2, 8))

    def test_probe_respects_the_feature_flag(self):
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl
        recorder: list[str] = []
        with mock.patch.dict(sys.modules, _fake_comfy_ops(recorder)), \
                mock.patch.object(dit_impl, "OPT_INT8_FUSED_SWIGLU", False):
            dit_impl._reset_fused_swiglu_probe()
            self.assertIsNone(dit_impl._fused_swiglu_fn())
            mlp = self._mlp(_NativeOps())
        self.assertIsNone(mlp.fused_swiglu)

    def test_probe_result_is_cached(self):
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl
        recorder: list[str] = []
        with mock.patch.dict(sys.modules, _fake_comfy_ops(recorder)):
            first = dit_impl._fused_swiglu_fn()
        # 探测缓存后即使 comfy 不再可导入也保持同一结果，避免每层重复 import
        second = dit_impl._fused_swiglu_fn()
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
