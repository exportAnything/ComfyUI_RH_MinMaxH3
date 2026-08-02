import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


class StaticTests(unittest.TestCase):
    def test_runtime_has_no_sglang_or_diffusers_import(self):
        forbidden = ("sglang", "sgl_kernel", "diffusers", "vllm")
        offenders = []
        for path in ROOT.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for imported in _imports(path):
                if imported in forbidden or imported.startswith(
                    ("sglang.", "sgl_kernel.", "diffusers.", "vllm.")
                ):
                    offenders.append((path.relative_to(ROOT), imported))
        self.assertEqual(offenders, [])

    def test_no_http_client_or_socket_backend(self):
        forbidden_roots = {"requests", "httpx", "aiohttp", "urllib3", "socket"}
        offenders = []
        for path in ROOT.rglob("*.py"):
            for imported in _imports(path):
                if imported.split(".", 1)[0] in forbidden_roots:
                    offenders.append((path.relative_to(ROOT), imported))
        self.assertEqual(offenders, [])

    def test_node_package_exports_direct_nodes(self):
        root_init = (ROOT / "__init__.py").read_text(encoding="utf-8")
        nodes = (ROOT / "minimax_h3_nodes" / "nodes.py").read_text(encoding="utf-8")
        self.assertIn("NODE_CLASS_MAPPINGS", root_init)
        self.assertIn("MiniMaxH3DualSigmaSampler", nodes)
        self.assertIn("MiniMaxH3DecodeAV", nodes)

    def test_dit_device_property_accepts_model_patcher_updates(self):
        dit = (
            ROOT / "minimax_h3_nodes" / "runtime" / "dit" / "_impl.py"
        ).read_text(encoding="utf-8")
        self.assertIn("@device.setter", dit)
        self.assertIn("operations", dit)
        self.assertIn("_activation_dtype", dit)

    def test_quant_checkpoint_markers(self):
        from minimax_h3_nodes.runtime.h3_settings import (
            INT8_FORMAT,
            QUANT_EXCLUDE_HINT,
            QUANT_KEY_SUFFIXES,
        )
        from minimax_h3_nodes.runtime.model_loader import _is_quantized_map

        self.assertEqual(INT8_FORMAT, "int8_tensorwise")
        self.assertIn("adaln_proj", QUANT_EXCLUDE_HINT)
        self.assertTrue(
            _is_quantized_map({"blocks.0.attn.qkv_proj.comfy_quant": "a.safetensors"}, [])
        )
        self.assertFalse(_is_quantized_map({"blocks.0.attn.qkv_proj.weight": "a.safetensors"}, []))
        self.assertTrue(any(s.endswith("comfy_quant") for s in QUANT_KEY_SUFFIXES))

    def test_qwen_cut_key_whitelist_is_narrow(self):
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _intentional_qwen_cut_key,
        )

        self.assertTrue(_intentional_qwen_cut_key("lm_head.weight"))
        self.assertTrue(
            _intentional_qwen_cut_key(
                "language_model.layers.50.self_attn.q_proj.weight"
            )
        )
        self.assertTrue(
            _intentional_qwen_cut_key("language_model.rotary_emb.inv_freq")
        )
        self.assertTrue(
            _intentional_qwen_cut_key("visual.rotary_emb.inv_freq")
        )
        self.assertFalse(
            _intentional_qwen_cut_key(
                "language_model.layers.49.self_attn.q_proj.weight"
            )
        )
        self.assertFalse(_intentional_qwen_cut_key("lm_head.bias"))
        self.assertFalse(_intentional_qwen_cut_key("evil.rotary_emb.inv_freq"))

    def test_loader_model_choices_are_explicit_and_required(self):
        from minimax_h3_nodes.nodes import (
            MiniMaxH3DirectModelLoader,
            MiniMaxH3DirectTextEncoderLoader,
            MiniMaxH3DirectVAELoader,
            _selector_to_component_dirname,
        )
        from minimax_h3_nodes.runtime.h3_settings import (
            AUDIO_VAE_DIRNAME,
            INT8_DIT_DIRNAME,
            INT8_TE_DIRNAME,
            INT8_TE_FILENAME,
            VAE_MERGED_DIRNAME,
            VAE_MERGED_MODEL_NAME,
            VIDEO_VAE_DIRNAME,
            bf16_dit_model_name,
            int8_dit_filename,
        )

        # 每个 loader 只暴露自己那一类的 COMBO；双 VAE 拆成 video/audio 两个。
        cases = (
            (MiniMaxH3DirectModelLoader, "transformer_path"),
            (MiniMaxH3DirectTextEncoderLoader, "text_encoder_path"),
            (MiniMaxH3DirectVAELoader, "video_vae_path"),
            (MiniMaxH3DirectVAELoader, "audio_vae_path"),
        )
        for node, input_name in cases:
            schema = node.INPUT_TYPES()
            self.assertIn(input_name, schema["required"])
            choices, options = schema["required"][input_name]
            self.assertNotIn("auto", choices)
            self.assertIn(options["default"], choices)
            # COMBO 默认应是模型名，而不是裸组件目录名
            self.assertFalse(
                options["default"]
                in {
                    INT8_DIT_DIRNAME,
                    INT8_TE_DIRNAME,
                    VAE_MERGED_DIRNAME,
                    VIDEO_VAE_DIRNAME,
                    AUDIO_VAE_DIRNAME,
                    "transformer",
                    "text_encoder",
                }
            )
        vae_schema = MiniMaxH3DirectVAELoader.INPUT_TYPES()["required"]
        self.assertNotIn("vae_path", vae_schema)

        self.assertNotEqual(
            MiniMaxH3DirectModelLoader.VALIDATE_INPUTS(
                model_root="MiniMax-H3", transformer_path="auto"
            ),
            True,
        )
        self.assertEqual(
            _selector_to_component_dirname(int8_dit_filename("FL2VA"), "transformer"),
            INT8_DIT_DIRNAME,
        )
        self.assertEqual(
            _selector_to_component_dirname(bf16_dit_model_name("FL2VA"), "transformer"),
            "transformer",
        )
        self.assertEqual(
            _selector_to_component_dirname(INT8_TE_FILENAME, "text_encoder"),
            INT8_TE_DIRNAME,
        )
        self.assertEqual(
            _selector_to_component_dirname(VAE_MERGED_MODEL_NAME, "vae"),
            VAE_MERGED_DIRNAME,
        )
        # 旧目录名仍可解析
        self.assertEqual(
            _selector_to_component_dirname(INT8_DIT_DIRNAME, "transformer"),
            INT8_DIT_DIRNAME,
        )

    def test_ref2va_target_exposes_manual_resolution_pair(self):
        from minimax_h3_nodes.nodes import MiniMaxH3Ref2VATarget

        schema = MiniMaxH3Ref2VATarget.INPUT_TYPES()
        self.assertIn("optional", schema)
        for name in ("width", "height"):
            with self.subTest(name=name):
                self.assertIn(name, schema["optional"])
                _type, options = schema["optional"][name]
                self.assertEqual(_type, "INT")
                self.assertEqual(options["default"], 0)
                self.assertEqual(options["step"], 32)


if __name__ == "__main__":
    unittest.main()
