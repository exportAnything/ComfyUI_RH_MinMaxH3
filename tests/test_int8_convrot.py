"""int8_convrot 量化工具：结构排除 + 小权重量化往返。"""
from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_KITCHEN = importlib.util.find_spec("comfy_kitchen") is not None


def _load_quant_tool():
    """按文件加载，避免 tools/ 包名遮蔽 ComfyUI.tools。"""
    path = ROOT / "tools" / "quantize_int8_convrot.py"
    spec = importlib.util.spec_from_file_location("h3_quantize_int8_convrot", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(HAS_TORCH, "no torch")
class Int8ConvrotToolTests(unittest.TestCase):
    def test_best_gs_and_structure_excludes(self):
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
        qt = _load_quant_tool()
        self.assertEqual(qt.best_gs(5376), 256)
        self.assertEqual(qt.best_gs(14336), 256)
        self.assertIsNone(qt.best_gs(100))

        cfg = MiniMaxH3DiTConfig(
            num_layers=2, token_refiner_num_layers=1, hidden_size=256,
            num_attention_heads=2, attention_head_dim=64, ffn_hidden_size=512,
            latents_dim=4, audio_latents_dim=8, patch_size=(1, 1, 1), text_dim=256,
            timestep_input_dim=32, time_embed_hidden_size=256, time_embed_dim=128,
            adaln_out_features=18 * 256, final_adaln_out_features=2 * 256, rope_inv_freq_len=4,
        )
        qmap = qt.quantizable_linear_names(cfg)
        names = set(qmap)
        self.assertTrue(any(n.startswith("blocks.0.attn.") for n in names))
        self.assertTrue(any(n.startswith("blocks.0.ffn.") or ".fc1" in n or ".fc2" in n for n in names))
        self.assertFalse(any("token_refiner" in n for n in names))
        self.assertFalse(any("adaln_proj" in n for n in names))
        self.assertFalse(any(n.startswith("video_patch_proj") for n in names))
        self.assertFalse(any(n.startswith("final_layer.video_out") for n in names))
        self.assertIn("condition_proj", names)

    @unittest.skipUnless(HAS_KITCHEN, "no comfy_kitchen")
    def test_tiny_quantize_roundtrip(self):
        import torch
        from safetensors.torch import save_file
        qt = _load_quant_tool()
        INT8_FORMAT, cq_tensor = qt.INT8_FORMAT, qt.cq_tensor
        quantize_convrot, recon_metrics, run_quantize = qt.quantize_convrot, qt.recon_metrics, qt.run_quantize

        w = torch.randn(256, 256, dtype=torch.bfloat16)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        qd, scale = quantize_convrot(w, 256, mseclip=False, device=device)
        self.assertEqual(qd.dtype, torch.int8)
        self.assertEqual(tuple(scale.shape), (256, 1))
        cos, rel = recon_metrics(qd, scale, w, 256, device=device)
        self.assertGreater(cos, 0.99)
        self.assertLess(rel, 5.0)
        meta = json.loads(bytes(cq_tensor(256).tolist()).decode("utf-8"))
        self.assertEqual(meta["format"], INT8_FORMAT)
        self.assertTrue(meta["convrot"])
        self.assertEqual(meta["convrot_groupsize"], 256)

        # 迷你 checkpoint 目录：单层 DiT 形状对齐的伪造权重
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig, MiniMaxH3DiTModel

        cfg = MiniMaxH3DiTConfig(
            num_layers=1, token_refiner_num_layers=1, hidden_size=256,
            num_attention_heads=2, attention_head_dim=64, ffn_hidden_size=512,
            latents_dim=4, audio_latents_dim=8, patch_size=(1, 1, 1), text_dim=256,
            timestep_input_dim=32, time_embed_hidden_size=256, time_embed_dim=128,
            adaln_out_features=18 * 256, final_adaln_out_features=2 * 256, rope_inv_freq_len=4,
        )
        model = MiniMaxH3DiTModel.from_config(cfg, device="cpu", dtype=torch.bfloat16)
        sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            from minimax_h3_nodes.runtime.h3_settings import INT8_DIT_DIRNAME, int8_dit_filename
            src, dst = Path(tmp) / "FL2VA" / "transformer", Path(tmp) / "FL2VA" / INT8_DIT_DIRNAME
            src.mkdir(parents=True)
            (src / "config.json").write_text(json.dumps(cfg.to_dict() if hasattr(cfg, "to_dict") else {
                "_class_name": "MiniMaxH3DiTModel",
                **{f.name: (list(getattr(cfg, f.name)) if f.name == "patch_size" else getattr(cfg, f.name))
                   for f in cfg.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            }), encoding="utf-8")
            save_file(sd, str(src / "model.safetensors"))
            meta = run_quantize(src, dst, device=device, verify=True, mseclip=False)
            self.assertGreater(meta["quantized_linears"], 0)
            out = dst / int8_dit_filename("FL2VA")
            self.assertTrue(out.is_file())
            from safetensors import safe_open
            with safe_open(str(out), framework="pt") as r:
                keys = set(r.keys())
            # 至少一个 blocks 权重被量化，且带 scale/comfy_quant；token_refiner 保持浮点
            q_w = next(k for k in keys if k.startswith("blocks.") and k.endswith(".weight") and k.replace(".weight", ".comfy_quant") in keys)
            with safe_open(str(out), framework="pt") as r:
                self.assertEqual(r.get_tensor(q_w).dtype, torch.int8)
                tr = next(k for k in keys if "token_refiner" in k and k.endswith(".weight"))
                self.assertTrue(r.get_tensor(tr).dtype.is_floating_point)


if __name__ == "__main__":
    unittest.main()
