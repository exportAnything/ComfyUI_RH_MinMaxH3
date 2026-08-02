"""融合 per-head RMSNorm + split-half RoPE kernel（PR#15224）：探测 + 回退。

真实 kernel 只有 comfy-kitchen 的 CUDA 实现，本地不可用。这里注入一个按同一契约
写的参考实现（[1,S,1,rot/2,2,2] 旋转表 + rot_dim 局部旋转），从而验证：接线的
形状/权重/rot_dim 是否正确、结果是否与现有 torch 路径逐点一致、以及各种不可用
情形是否老实回退。
"""
from __future__ import annotations
import importlib.util
import sys
import types
import unittest
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None

ROPE_LEN = 1  # tiny config：rope_inv_freq_len=1 → 旋转 6 维，head_dim 8


def _tiny_cfg():
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
    return MiniMaxH3DiTConfig(
        num_layers=1, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=2,
        attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
        patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
        time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
        rope_inv_freq_len=ROPE_LEN,
    )


def _reference_kernel(recorder, *, in_place):
    """按 kernel 契约实现的参考版本：per-head RMSNorm + 前 rot_dim 维 split-half 旋转。"""

    def run(q, k, table, q_weight, k_weight, *, epsilon, rot_dim):
        import torch

        recorder.append(
            {"shape": tuple(q.shape), "rot_dim": int(rot_dim), "eps": float(epsilon),
             "table": tuple(table.shape)}
        )
        half = rot_dim // 2
        # 表是 [1, S, 1, half, 2, 2] 的 [[c,-s],[s,c]]
        cos = table[0, :, 0, :, 0, 0]
        sin = table[0, :, 0, :, 1, 0]
        out = []
        for tensor, weight in ((q, q_weight), (k, k_weight)):
            x = tensor[0].to(torch.float32)  # [S, heads, head_dim]
            normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
            normed = normed * weight.to(torch.float32)
            rotated, passthrough = normed[..., :rot_dim], normed[..., rot_dim:]
            x1, x2 = rotated[..., :half], rotated[..., half:]
            c = cos.unsqueeze(1).to(torch.float32)
            s = sin.unsqueeze(1).to(torch.float32)
            spun = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
            out.append(torch.cat((spun, passthrough), dim=-1).to(tensor.dtype))
        if in_place:
            q[0].copy_(out[0])
            k[0].copy_(out[1])
            return None
        return out[0].unsqueeze(0), out[1].unsqueeze(0)

    return run


def _fake_comfy(recorder, *, in_place=True, present=True):
    quant_ops = types.ModuleType("comfy.quant_ops")
    kitchen = types.SimpleNamespace()
    if present:
        name = "rms_rope_split_half_" if in_place else "rms_rope_split_half"
        setattr(kitchen, name, _reference_kernel(recorder, in_place=in_place))
    quant_ops.ck = kitchen
    parent = types.ModuleType("comfy")
    parent.quant_ops = quant_ops
    return {"comfy": parent, "comfy.quant_ops": quant_ops}


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestFusedQKRope(unittest.TestCase):
    def setUp(self):
        from minimax_h3_nodes.runtime.dit._impl import _reset_fused_qk_rope_probe
        _reset_fused_qk_rope_probe()
        self.addCleanup(_reset_fused_qk_rope_probe)

    def _attention(self):
        import torch
        from minimax_h3_nodes.runtime.dit._impl import MiniMaxH3Attention
        torch.manual_seed(4)
        # 与 loader 一致：推理模型 requires_grad_(False)，就地 kernel 才允许改写视图
        attn = MiniMaxH3Attention(
            _tiny_cfg(), dtype=torch.float32, device="cpu",
            attention_backend="sdpa", attention_function=None,
        ).eval().requires_grad_(False)
        # 非单位 norm 权重，才能验出融合路径确实用了它们
        with torch.no_grad():
            attn.q_norm.weight.copy_(torch.rand(8) + 0.5)
            attn.k_norm.weight.copy_(torch.rand(8) + 0.5)
        return attn

    def _rope_cache(self, seq, *, dtype=None):
        import torch
        from minimax_h3_nodes.runtime.attention import (
            _rope_cos_sin_cache, _rope_rotation_table,
        )
        dtype = dtype or torch.float32
        torch.manual_seed(9)
        # rope 频率是 [S, 6*inv_freq_len]，两半重复
        angles = torch.rand(seq, 3 * ROPE_LEN) * 3.0
        frequencies = torch.cat((angles, angles), dim=-1)
        cos_sin = _rope_cos_sin_cache(frequencies, dtype=dtype)
        return cos_sin, _rope_rotation_table(cos_sin, dtype=dtype)

    def test_rotation_table_shape_and_entries(self):
        import torch
        cos_sin, table = self._rope_cache(5)
        half = cos_sin.shape[-1] // 2
        self.assertEqual(tuple(table.shape), (1, 5, 1, half, 2, 2))
        cos, sin = cos_sin[..., :half], cos_sin[..., half:]
        self.assertTrue(torch.allclose(table[0, :, 0, :, 0, 0], cos))
        self.assertTrue(torch.allclose(table[0, :, 0, :, 0, 1], -sin))
        self.assertTrue(torch.allclose(table[0, :, 0, :, 1, 0], sin))
        self.assertTrue(torch.allclose(table[0, :, 0, :, 1, 1], cos))

    def test_fused_matches_eager_in_place(self):
        self._assert_fused_matches_eager(in_place=True)

    def test_fused_matches_eager_out_of_place(self):
        self._assert_fused_matches_eager(in_place=False)

    def _assert_fused_matches_eager(self, *, in_place):
        import torch
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl

        seq = 6
        attn = self._attention()
        hidden = torch.randn(seq, 8)
        cu_seqlens = torch.tensor([0, seq], dtype=torch.int32)
        cos_sin, table = self._rope_cache(seq)

        eager = attn(
            hidden, rope_cache=(cos_sin, None), cu_seqlens=cu_seqlens, max_seqlen=seq
        )
        recorder: list[dict] = []
        with mock.patch.dict(sys.modules, _fake_comfy(recorder, in_place=in_place)), \
                mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE_CUDA_ONLY", False):
            dit_impl._reset_fused_qk_rope_probe()
            fused = attn(
                hidden, rope_cache=(cos_sin, table), cu_seqlens=cu_seqlens, max_seqlen=seq
            )
        self.assertEqual(len(recorder), 1)
        self.assertEqual(recorder[0]["shape"], (1, seq, 2, 8))
        self.assertEqual(recorder[0]["rot_dim"], 6 * ROPE_LEN)  # 局部旋转，非整个 head_dim
        self.assertEqual(recorder[0]["eps"], attn.q_norm.eps)
        self.assertTrue(torch.allclose(fused, eager, atol=1e-5, rtol=1e-5))

    def test_fused_leaves_the_value_stream_untouched(self):
        """就地 kernel 写的是同一块 qkv 缓冲上的 q/k 视图，v 段不能被殃及。"""
        import torch
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl

        seq = 4
        attn = self._attention()
        hidden = torch.randn(seq, 8)
        cos_sin, table = self._rope_cache(seq)
        with torch.no_grad():
            expected_value = attn.qkv_proj(hidden).chunk(3, dim=-1)[2].clone()

        captured = {}
        original = dit_impl.sdpa_varlen_attention

        def spy(query, key, value, **kwargs):
            captured["value"] = value.clone()
            return original(query, key, value, **kwargs)

        recorder: list[dict] = []
        with mock.patch.dict(sys.modules, _fake_comfy(recorder, in_place=True)), \
                mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE_CUDA_ONLY", False), \
                mock.patch.object(dit_impl, "sdpa_varlen_attention", spy):
            dit_impl._reset_fused_qk_rope_probe()
            attn(
                hidden, rope_cache=(cos_sin, table),
                cu_seqlens=torch.tensor([0, seq], dtype=torch.int32), max_seqlen=seq,
            )
        self.assertTrue(
            torch.allclose(captured["value"].reshape(seq, 16), expected_value)
        )

    def test_missing_kernel_falls_back_to_eager(self):
        import torch
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl

        seq = 4
        attn = self._attention()
        hidden = torch.randn(seq, 8)
        cos_sin, table = self._rope_cache(seq)
        cu_seqlens = torch.tensor([0, seq], dtype=torch.int32)
        eager = attn(hidden, rope_cache=(cos_sin, None), cu_seqlens=cu_seqlens, max_seqlen=seq)
        recorder: list[dict] = []
        # 旧版 comfy：quant_ops.ck 里没有这两个函数
        with mock.patch.dict(sys.modules, _fake_comfy(recorder, present=False)), \
                mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE_CUDA_ONLY", False):
            dit_impl._reset_fused_qk_rope_probe()
            self.assertIsNone(dit_impl._fused_qk_rope_fn())
            # 即使误传了旋转表也必须安全回退
            got = attn(
                hidden, rope_cache=(cos_sin, table), cu_seqlens=cu_seqlens, max_seqlen=seq
            )
        self.assertTrue(torch.allclose(got, eager, atol=1e-6, rtol=1e-6))

    def test_grad_tracked_input_falls_back_from_the_in_place_kernel(self):
        import torch
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl

        seq = 4
        attn = self._attention().requires_grad_(True)  # 带梯度：不能改写 chunk 视图
        hidden = torch.randn(seq, 8)
        cos_sin, table = self._rope_cache(seq)
        cu_seqlens = torch.tensor([0, seq], dtype=torch.int32)
        recorder: list[dict] = []
        with mock.patch.dict(sys.modules, _fake_comfy(recorder, in_place=True)), \
                mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE_CUDA_ONLY", False):
            dit_impl._reset_fused_qk_rope_probe()
            got = attn(
                hidden, rope_cache=(cos_sin, table), cu_seqlens=cu_seqlens, max_seqlen=seq
            )
        self.assertEqual(recorder, [])  # 走了 eager，没调 kernel
        eager = attn(
            hidden, rope_cache=(cos_sin, None), cu_seqlens=cu_seqlens, max_seqlen=seq
        )
        self.assertTrue(torch.allclose(got, eager, atol=1e-6, rtol=1e-6))

    def test_probe_prefers_in_place_variant(self):
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl
        recorder: list[dict] = []
        modules = _fake_comfy(recorder, in_place=True)
        # 两个变体都在时优先就地版
        modules["comfy.quant_ops"].ck.rms_rope_split_half = _reference_kernel(
            recorder, in_place=False
        )
        with mock.patch.dict(sys.modules, modules):
            dit_impl._reset_fused_qk_rope_probe()
            resolved = dit_impl._fused_qk_rope_fn()
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved[1])

    def test_cuda_only_gate_and_feature_flag(self):
        import torch
        from minimax_h3_nodes.runtime.dit import _impl as dit_impl
        recorder: list[dict] = []
        cpu = torch.zeros(2, 4)
        with mock.patch.dict(sys.modules, _fake_comfy(recorder)):
            dit_impl._reset_fused_qk_rope_probe()
            # 默认 CUDA-only：CPU 张量不启用
            self.assertFalse(dit_impl.fused_qk_rope_available(cpu))
            with mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE_CUDA_ONLY", False):
                self.assertTrue(dit_impl.fused_qk_rope_available(cpu))
            with mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE", False), \
                    mock.patch.object(dit_impl, "OPT_FUSED_QK_ROPE_CUDA_ONLY", False):
                dit_impl._reset_fused_qk_rope_probe()
                self.assertFalse(dit_impl.fused_qk_rope_available(cpu))

    def test_rope_cache_omits_the_table_when_kernel_is_unavailable(self):
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel
        model = MiniMaxH3DiTModel.from_config(_tiny_cfg(), dtype=torch.float32)
        cos_sin, table = model._rope_cache(torch.rand(3, 6 * ROPE_LEN))
        self.assertEqual(tuple(cos_sin.shape), (3, 6 * ROPE_LEN))
        self.assertIsNone(table)  # 本地没有 comfy-kitchen，不该白白构造大表


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class TestKernelSignatureGate(unittest.TestCase):
    """只判断"函数存在"会放行不支持部分旋转的老 kernel。

    H3 的 ``rotary_dim = 6 * rope_inv_freq_len``（96）小于 head_dim（128）。早期
    comfy-kitchen 的 ``rms_rope_split_half`` 没有 ``rot_dim``，会把整个 head_dim
    拆成对去套旋转表——形状不匹配时抛 RuntimeError，匹配时结果是错的。
    """

    def _probe_with(self, kitchen_module):
        """必须连父包 comfy 一起打桩，否则会解析到真实环境里的 comfy.quant_ops。"""

        from minimax_h3_nodes.runtime.dit import _impl

        _impl._FUSED_QK_ROPE_PROBE.clear()
        quant_ops = types.ModuleType("comfy.quant_ops")
        quant_ops.ck = kitchen_module
        parent = types.ModuleType("comfy")
        parent.quant_ops = quant_ops
        modules = {"comfy": parent, "comfy.quant_ops": quant_ops}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(_impl, "OPT_FUSED_QK_ROPE", True):
            try:
                return _impl._fused_qk_rope_fn()
            finally:
                _impl._FUSED_QK_ROPE_PROBE.clear()

    def test_kernel_without_rot_dim_is_rejected(self):
        kitchen = types.ModuleType("ck")

        def rms_rope_split_half(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6):
            raise AssertionError("不该被调用")

        kitchen.rms_rope_split_half = rms_rope_split_half
        self.assertIsNone(self._probe_with(kitchen))

    def test_kernel_with_rot_dim_is_accepted(self):
        kitchen = types.ModuleType("ck")

        def rms_rope_split_half(
            q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=None
        ):
            raise AssertionError("不该被调用")

        kitchen.rms_rope_split_half = rms_rope_split_half
        resolved = self._probe_with(kitchen)
        self.assertIsNotNone(resolved)
        self.assertFalse(resolved[1])  # out-of-place

    def test_var_keyword_kernel_is_accepted(self):
        kitchen = types.ModuleType("ck")

        def rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, **kwargs):
            raise AssertionError("不该被调用")

        kitchen.rms_rope_split_half_ = rms_rope_split_half_
        resolved = self._probe_with(kitchen)
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved[1])  # in-place

    def test_call_time_type_error_disables_the_fused_path(self):
        from minimax_h3_nodes.runtime.dit import _impl

        _impl._FUSED_QK_ROPE_PROBE.clear()
        _impl._FUSED_QK_ROPE_PROBE.append(("sentinel", False))
        _impl._disable_fused_qk_rope(TypeError("unexpected keyword argument 'rot_dim'"))
        self.assertIsNone(_impl._fused_qk_rope_fn())
        _impl._FUSED_QK_ROPE_PROBE.clear()
