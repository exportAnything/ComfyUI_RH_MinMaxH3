"""扁平权重根 models/MiniMax-H3 + diffusers 分片 release 的双来源选择。

真实安装形态：官方分片 release 留在 ``models/diffusers/MiniMax-H3/<分区>/``
并提供 config/tokenizer/processor；量化与合并产物是 ``models/MiniMax-H3/`` 下
的扁平单文件，没有任何 sidecar，类型与分区一律由文件名判定。
"""
import json
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

DIT_CONFIG = {
    "_class_name": "MiniMaxH3DiTModel",
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "adaln_out_features": 2688,
}
TE_CONFIG = {
    "model_type": "qwen3_vl",
    "architectures": ["Qwen3VLForConditionalGeneration"],
}
VIDEO_CONFIG = {"_class_name": "MiniMaxH3VideoVAE", "latent_channels": 24}
AUDIO_CONFIG = {"_class_name": "MiniMaxH3AudioVAE", "latent_channels": 32}

FLAT_WEIGHTS = (
    "MiniMax-H3-FL2VA-int8_convrot.safetensors",
    "MiniMax-H3-Ref2VA-int8_convrot.safetensors",
    "qwen3-vl-32b-int8_convrot.safetensors",
    "MiniMax-H3-video_vae.safetensors",
    "MiniMax-H3-audio_vae.safetensors",
)


def _component(path: Path, config: dict, weights: tuple[str, ...] = ()) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in weights:
        (path / name).write_bytes(b"")
    return path


@contextmanager
def _installation():
    """Build the real two-location install under a throwaway models directory."""

    with tempfile.TemporaryDirectory() as raw:
        models = Path(raw).resolve() / "models"

        weights_root = models / "MiniMax-H3"
        weights_root.mkdir(parents=True)
        for name in FLAT_WEIGHTS:
            (weights_root / name).write_bytes(b"")
        # 不符合命名规范的文件必须被忽略，而不是猜类型
        (weights_root / "unrelated.safetensors").write_bytes(b"")

        release = models / "diffusers" / "MiniMax-H3"
        for partition in ("FL2VA", "Ref2VA"):
            root = release / partition
            _component(
                root / "transformer",
                DIT_CONFIG,
                ("model-00001-of-00002.safetensors",),
            )
            _component(
                root / "text_encoder",
                TE_CONFIG,
                ("model-00001-of-00003.safetensors",),
            )
            _component(root / "video_vae", VIDEO_CONFIG)
            _component(
                root / "video_vae" / "source", VIDEO_CONFIG, ("model.safetensors",)
            )
            _component(root / "audio_vae", AUDIO_CONFIG, ("model.safetensors",))

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(models)
        folder_paths.get_folder_paths = lambda name: [str(models / name)]
        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}), \
                mock.patch.dict(
                    "os.environ",
                    {"MINIMAX_H3_GLOBAL_MODELS_DIR": str(models / "absent")},
                ):
            yield models, weights_root, release


class WeightFilenameTests(unittest.TestCase):
    def test_filename_carries_component_type_and_partition(self):
        from minimax_h3_nodes.runtime.h3_settings import classify_weight_filename

        cases = {
            "MiniMax-H3-FL2VA-int8_convrot.safetensors": ("transformer", "fl2va"),
            "MiniMax-H3-Ref2VA-int8_convrot.safetensors": ("transformer", "ref2va"),
            "qwen3-vl-32b-int8_convrot.safetensors": ("text_encoder", None),
            "MiniMax-H3-video_vae.safetensors": ("video_vae", None),
            "MiniMax-H3-audio_vae.safetensors": ("audio_vae", None),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(classify_weight_filename(name), expected)

    def test_unconvention_names_are_ignored_not_guessed(self):
        from minimax_h3_nodes.runtime.h3_settings import classify_weight_filename

        for name in (
            "model-00001-of-00002.safetensors",
            "unrelated.safetensors",
            "MiniMax-H3.bin",
            "MiniMax-H3-unknown.safetensors",
        ):
            with self.subTest(name=name):
                self.assertIsNone(classify_weight_filename(name))


class WeightsRootTests(unittest.TestCase):
    def test_flat_root_is_listed_and_filtered_by_kind(self):
        from minimax_h3_nodes.runtime import components

        with _installation() as (_models, weights_root, _release):
            self.assertEqual(components.weights_root_paths(), [weights_root])
            self.assertEqual(
                [path.name for path in components.list_h3_weight_files("transformer")],
                [
                    "MiniMax-H3-FL2VA-int8_convrot.safetensors",
                    "MiniMax-H3-Ref2VA-int8_convrot.safetensors",
                ],
            )
            self.assertEqual(
                [
                    path.name
                    for path in components.list_h3_weight_files(
                        "transformer", "fl2va"
                    )
                ],
                ["MiniMax-H3-FL2VA-int8_convrot.safetensors"],
            )
            # TE / VAE 不带分区标记，两个分区共用同一份文件
            for kind in ("text_encoder", "video_vae", "audio_vae"):
                with self.subTest(kind=kind):
                    self.assertEqual(
                        components.list_h3_weight_files(kind, "fl2va"),
                        components.list_h3_weight_files(kind, "ref2va"),
                    )

    def test_partition_gate_uses_the_filename(self):
        from minimax_h3_nodes.runtime import components

        with _installation() as (_models, weights_root, _release):
            fl2va = weights_root / "MiniMax-H3-FL2VA-int8_convrot.safetensors"
            self.assertEqual(
                components.validate_weight_partition(
                    fl2va, "fl2va", kind="transformer"
                ),
                fl2va,
            )
            with self.assertRaises(components.H3ComponentError):
                components.validate_weight_partition(
                    fl2va, "ref2va", kind="transformer"
                )
            # 类型也必须对得上，避免把 VAE 权重塞进 DiT loader
            with self.assertRaises(components.H3ComponentError):
                components.validate_weight_partition(
                    weights_root / "MiniMax-H3-video_vae.safetensors",
                    "fl2va",
                    kind="transformer",
                )
            with self.assertRaises(components.H3ComponentError):
                components.validate_weight_partition(
                    weights_root / "unrelated.safetensors",
                    "fl2va",
                    kind="transformer",
                )


class ComboTests(unittest.TestCase):
    def test_each_combo_lists_only_its_own_component_type(self):
        from minimax_h3_nodes.api import _shared

        with _installation():
            choices = {
                kind: set(_shared._component_model_choices(kind, "fl2va"))
                for kind in _shared.H3_COMPONENT_KINDS
            }
        self.assertEqual(
            choices["transformer"],
            {"MiniMax-H3-FL2VA-int8_convrot.safetensors", "MiniMax-H3-FL2VA"},
        )
        self.assertEqual(
            choices["text_encoder"],
            {"qwen3-vl-32b-int8_convrot.safetensors", "qwen3-vl-32b"},
        )
        self.assertEqual(
            choices["video_vae"],
            {"MiniMax-H3-video_vae.safetensors", "MiniMax-H3-video_vae"},
        )
        self.assertEqual(
            choices["audio_vae"],
            {"MiniMax-H3-audio_vae.safetensors", "MiniMax-H3-audio_vae"},
        )
        for left in choices:
            for right in choices:
                if left != right:
                    with self.subTest(left=left, right=right):
                        self.assertFalse(choices[left] & choices[right])

    def test_dit_combo_is_partition_scoped(self):
        from minimax_h3_nodes.api import _shared

        with _installation():
            self.assertNotIn(
                "MiniMax-H3-Ref2VA-int8_convrot.safetensors",
                _shared._component_model_choices("transformer", "fl2va"),
            )
            self.assertIn(
                "MiniMax-H3-Ref2VA-int8_convrot.safetensors",
                _shared._component_model_choices("transformer", "ref2va"),
            )

    def test_flat_weight_maps_to_the_release_config_directory(self):
        from minimax_h3_nodes.api import _shared

        with _installation() as (_models, _weights_root, release):
            root = str(release)
            cases = (
                ("MiniMax-H3-FL2VA-int8_convrot.safetensors", "transformer"),
                ("qwen3-vl-32b-int8_convrot.safetensors", "text_encoder"),
                ("MiniMax-H3-video_vae.safetensors", "video_vae"),
                ("MiniMax-H3-audio_vae.safetensors", "audio_vae"),
            )
            for selector, kind in cases:
                with self.subTest(selector=selector):
                    # 单文件没有 config，落到 release 里同类型的组件目录
                    self.assertEqual(
                        _shared._selector_to_component_dirname(
                            selector, kind, "fl2va", model_root=root
                        ),
                        kind,
                    )
                    self.assertIsNotNone(
                        _shared._selector_weight_file(selector, kind, "fl2va")
                    )
            # release 里的分片组件仍按目录解析，且不带外部权重
            for selector, kind in (
                ("MiniMax-H3-FL2VA", "transformer"),
                ("qwen3-vl-32b", "text_encoder"),
                ("MiniMax-H3-video_vae", "video_vae"),
                ("MiniMax-H3-audio_vae", "audio_vae"),
            ):
                with self.subTest(selector=selector):
                    self.assertEqual(
                        _shared._selector_to_component_dirname(
                            selector, kind, "fl2va", model_root=root
                        ),
                        kind if kind != "transformer" else "transformer",
                    )
                    self.assertIsNone(
                        _shared._selector_weight_file(selector, kind, "fl2va")
                    )


class LoaderTests(unittest.TestCase):
    """节点层：选中的单文件必须原样送进 runtime，并折进 component fingerprint。"""

    @contextmanager
    def _stub_runtime(self, **modules):
        from minimax_h3_nodes.api import _shared, loaders

        original = _shared._runtime_module

        def _patched(name):
            return modules.get(name) or original(name)

        with mock.patch.object(_shared, "_runtime_module", _patched), \
                mock.patch.object(loaders, "_runtime_module", _patched):
            yield loaders

    def _stub(self, name: str, function_name: str, calls: dict):
        module = types.ModuleType(f"{name}_stub")

        def _entry(**kwargs):
            calls.update(kwargs)
            return {"stub": True}

        setattr(module, function_name, _entry)
        return module

    def test_dual_vae_forwards_both_components_and_both_weight_files(self):
        from minimax_h3_nodes.contracts import validate_component_for_task

        calls: dict = {}
        with _installation() as (_models, weights_root, release), \
                self._stub_runtime(
                    vae_adapter=self._stub(
                        "vae_adapter", "load_h3_vae_bundle", calls
                    )
                ) as loaders:
            (wrapper,) = loaders.MiniMaxH3FL2VAVAELoader().load(
                str(release),
                "MiniMax-H3-video_vae.safetensors",
                "MiniMax-H3-audio_vae.safetensors",
            )
            partition_root = release / "FL2VA"
            self.assertEqual(
                calls["video_vae_path"], str(partition_root / "video_vae")
            )
            self.assertEqual(
                calls["audio_vae_path"], str(partition_root / "audio_vae")
            )
            self.assertEqual(
                calls["video_vae_weights"],
                str(weights_root / "MiniMax-H3-video_vae.safetensors"),
            )
            self.assertEqual(
                calls["audio_vae_weights"],
                str(weights_root / "MiniMax-H3-audio_vae.safetensors"),
            )
            # fingerprint 覆盖真实文件快照，必须在 fixture still alive 时校验
            clean = validate_component_for_task(
                wrapper, component_kind="vae", task="fl2va"
            )
            self.assertEqual(
                clean["component_fingerprint"], wrapper["vae_fingerprint"]
            )

    def test_swapping_a_weight_file_breaks_the_component_fingerprint(self):
        from minimax_h3_nodes.contracts import H3ContractError
        from minimax_h3_nodes.contracts import validate_component_for_task

        calls: dict = {}
        with _installation() as (_models, weights_root, release), \
                self._stub_runtime(
                    vae_adapter=self._stub(
                        "vae_adapter", "load_h3_vae_bundle", calls
                    )
                ) as loaders:
            (wrapper,) = loaders.MiniMaxH3FL2VAVAELoader().load(
                str(release),
                "MiniMax-H3-video_vae.safetensors",
                "MiniMax-H3-audio_vae.safetensors",
            )
            tampered = dict(wrapper)
            tampered["audio_vae_weights_path"] = str(
                weights_root / "MiniMax-H3-video_vae.safetensors"
            )
            with self.assertRaises(H3ContractError):
                validate_component_for_task(
                    tampered, component_kind="vae", task="fl2va"
                )

    def test_dit_loader_forwards_the_flat_weight_file(self):
        from minimax_h3_nodes.contracts import validate_component_for_task

        calls: dict = {}
        with _installation() as (_models, weights_root, release), \
                self._stub_runtime(
                    model_loader=self._stub("model_loader", "load_h3_model", calls)
                ) as loaders:
            (wrapper,) = loaders.MiniMaxH3FL2VAModelLoader().load(
                str(release), "auto", "MiniMax-H3-FL2VA-int8_convrot.safetensors"
            )
            self.assertEqual(
                calls["transformer_path"], str(release / "FL2VA" / "transformer")
            )
            self.assertEqual(
                calls["transformer_weights"],
                str(weights_root / "MiniMax-H3-FL2VA-int8_convrot.safetensors"),
            )
            self.assertEqual(
                wrapper["transformer_weights_path"], calls["transformer_weights"]
            )
            validate_component_for_task(
                wrapper, component_kind="model", task="fl2va"
            )

    def test_ref2va_loader_rejects_the_fl2va_weight_file(self):
        calls: dict = {}
        with _installation() as (_models, _weights_root, release), \
                self._stub_runtime(
                    model_loader=self._stub("model_loader", "load_h3_model", calls)
                ) as loaders:
            with self.assertRaises(Exception):
                loaders.MiniMaxH3Ref2VAModelLoader().load(
                    str(release),
                    "auto",
                    "MiniMax-H3-FL2VA-int8_convrot.safetensors",
                )

    def test_sharded_release_component_still_loads_without_weight_override(self):
        calls: dict = {}
        with _installation() as (_models, _weights_root, release), \
                self._stub_runtime(
                    model_loader=self._stub("model_loader", "load_h3_model", calls)
                ) as loaders:
            (wrapper,) = loaders.MiniMaxH3FL2VAModelLoader().load(
                str(release), "auto", "MiniMax-H3-FL2VA"
            )
            self.assertEqual(
                calls["transformer_path"], str(release / "FL2VA" / "transformer")
            )
            self.assertIsNone(calls["transformer_weights"])
            self.assertNotIn("transformer_weights_path", wrapper)


if __name__ == "__main__":
    unittest.main()
