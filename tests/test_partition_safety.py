import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from minimax_h3_nodes.runtime.components import (
    H3ComponentError,
    partition_for_task,
    resolve_component,
    validate_release_metadata,
    validate_task_partition,
)
from minimax_h3_nodes.runtime.model_loader import (
    H3ModelHandle,
    _checkpoint_index,
    _validate_quant_meta_partition,
    _validate_transformer_config,
    load_h3_model,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _release(partition: str) -> dict:
    tasks = ["t2va", "fl2va"] if partition == "fl2va" else ["ref2va"]
    return {
        "schema_version": 1,
        "partition": partition,
        "tasks": tasks,
        "task_aliases": {},
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
    }


def _write_release(root: Path, partition: str) -> None:
    _write_json(root / "model_index.json", {"_minimax_h3": _release(partition)})


class PartitionSafetyTests(unittest.TestCase):
    def test_transformer_config_requires_official_architecture_before_allocation(self):
        official = {
            "num_layers": 50,
            "token_refiner_num_layers": 2,
            "hidden_size": 5376,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_hidden_size": 14336,
            "latents_dim": 24,
            "audio_latents_dim": 32,
            "patch_size": [1, 2, 2],
            "text_dim": 5120,
            "timestep_input_dim": 256,
            "time_embed_hidden_size": 5376,
            "time_embed_dim": 2688,
            "adaln_out_features": 18 * 5376,
            "final_adaln_out_features": 2 * 5376,
            "rope_inv_freq_len": 16,
            "norm_eps": 1e-5,
            "qk_norm_eps": 1e-5,
            "final_norm_eps": 1e-5,
        }
        config_path = Path("/untrusted/transformer/config.json")
        _validate_transformer_config(
            {"_class_name": "MiniMaxH3DiTModel", **official},
            config_path,
        )
        # The wrapper form is accepted by MiniMaxH3DiTConfig.from_dict and
        # therefore must receive the same loader-level gate.
        _validate_transformer_config({"arch_config": official}, config_path)

        oversized = dict(official, num_layers=1_000_000)
        with self.assertRaisesRegex(
            H3ComponentError,
            r"official MiniMax-H3 DiT architecture: num_layers=1000000",
        ):
            _validate_transformer_config(oversized, config_path)

        wrong_patch = dict(official, patch_size=[1, 1, 1])
        with self.assertRaisesRegex(H3ComponentError, "patch_size"):
            _validate_transformer_config(wrong_patch, config_path)

    def test_official_task_partition_map(self):
        self.assertEqual(partition_for_task("t2va"), "fl2va")
        self.assertEqual(partition_for_task("FL2VA"), "fl2va")
        self.assertEqual(partition_for_task("ref2va"), "ref2va")
        with self.assertRaisesRegex(H3ComponentError, "Unsupported"):
            partition_for_task("i2v")

    def test_fl_and_ref_release_admission(self):
        fl = _release("fl2va")
        ref = _release("ref2va")
        self.assertEqual(validate_task_partition(fl, "t2va", "fl2va"), "t2va")
        self.assertEqual(validate_task_partition(fl, "fl2va", "fl2va"), "fl2va")
        self.assertEqual(validate_task_partition(ref, "ref2va", "ref2va"), "ref2va")
        # Metadata-free development roots retain the old T2VA path.
        self.assertEqual(validate_task_partition({}, "t2va", "fl2va"), "t2va")

    def test_task_partition_crossing_is_rejected(self):
        with self.assertRaisesRegex(H3ComponentError, "requires partition 'fl2va'"):
            validate_task_partition(_release("ref2va"), "t2va", "ref2va")
        with self.assertRaisesRegex(H3ComponentError, "requires partition 'ref2va'"):
            validate_task_partition(_release("fl2va"), "ref2va", "fl2va")

    def test_release_contract_validates_tasks_and_shifts(self):
        wrong_task = _release("ref2va")
        wrong_task["tasks"] = ["t2va"]
        with self.assertRaisesRegex(H3ComponentError, "does not belong"):
            validate_release_metadata(wrong_task)

        duplicate = _release("fl2va")
        duplicate["tasks"] = ["t2va", "t2va"]
        with self.assertRaisesRegex(H3ComponentError, "duplicates"):
            validate_release_metadata(duplicate)

        missing_shifts = _release("fl2va")
        del missing_shifts["sigma_shift_scales"]
        with self.assertRaisesRegex(H3ComponentError, "must be an object"):
            validate_release_metadata(missing_shifts)

        nonfinite = _release("fl2va")
        nonfinite["sigma_shift_scales"]["video"] = float("inf")
        with self.assertRaisesRegex(H3ComponentError, "positive finite"):
            validate_release_metadata(nonfinite)

    def test_explicit_component_cannot_escape_partition_root(self):
        with tempfile.TemporaryDirectory() as raw:
            release = Path(raw)
            fl = release / "FL2VA"
            ref_transformer = release / "Ref2VA" / "transformer"
            fl.mkdir()
            ref_transformer.mkdir(parents=True)
            _write_json(ref_transformer / "config.json", {})
            with self.assertRaisesRegex(H3ComponentError, "outside the selected"):
                resolve_component(
                    fl,
                    ("transformer",),
                    explicit=ref_transformer,
                    required_files=("config.json",),
                )
            with self.assertRaisesRegex(H3ComponentError, "cross-partition"):
                resolve_component(
                    fl,
                    ("transformer",),
                    explicit="../Ref2VA/transformer",
                    required_files=("config.json",),
                )

    def test_int8_quant_partition_fails_before_config_or_weights(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Ref2VA"
            component = root / "transformer_int8_convrot"
            component.mkdir(parents=True)
            _write_release(root, "ref2va")
            # An invalid config proves quant_meta admission happens first; no
            # safetensors index or model allocation is needed to reject it.
            _write_json(component / "config.json", {})
            _write_json(
                component / "quant_meta.json",
                {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "partition": "FL2VA",
                },
            )
            with self.assertRaisesRegex(H3ComponentError, "partition mismatch"):
                load_h3_model(
                    str(root),
                    partition="ref2va",
                    task="ref2va",
                    transformer_path=str(component),
                )

    def test_int8_quant_meta_is_required_and_validates_converter_contract(self):
        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw) / "transformer_int8_convrot"
            component.mkdir()
            with self.assertRaisesRegex(H3ComponentError, "requires .*quant_meta"):
                _validate_quant_meta_partition(
                    component,
                    "fl2va",
                    required=True,
                )

            valid = {
                "format": "int8_tensorwise",
                "convrot": True,
                "quantized_linears": 201,
                "partition": "FL2VA",
                "exclude": "token_refiner",
                "mseclip": True,
            }
            _write_json(component / "quant_meta.json", valid)
            self.assertEqual(
                _validate_quant_meta_partition(
                    component,
                    "fl2va",
                    required=True,
                ),
                valid,
            )

            for field, value, message in (
                ("format", "other", "format must be"),
                ("convrot", False, "convrot must be true"),
            ):
                with self.subTest(field=field):
                    invalid = dict(valid)
                    invalid[field] = value
                    _write_json(component / "quant_meta.json", invalid)
                    with self.assertRaisesRegex(H3ComponentError, message):
                        _validate_quant_meta_partition(
                            component,
                            "fl2va",
                            required=True,
                        )

    def test_indexless_quant_detection_scans_all_shards_without_torch(self):
        from minimax_h3_nodes.runtime.model_loader import _is_quantized_map

        opened = []
        contents = {
            "part-00001.safetensors": ("plain.weight",),
            "part-00002.safetensors": ("late.comfy_quant",),
        }

        class FakeReader:
            def __init__(self, path):
                self.path = Path(path).name

            def __enter__(self):
                opened.append(self.path)
                return self

            def __exit__(self, *_args):
                return False

            def keys(self):
                return contents[self.path]

        safetensors = types.ModuleType("safetensors")
        safetensors.safe_open = lambda path, **_kwargs: FakeReader(path)
        shards = [Path(name) for name in contents]
        with mock.patch.dict(sys.modules, {"safetensors": safetensors}):
            self.assertTrue(_is_quantized_map(None, shards))
        self.assertEqual(opened, [path.name for path in shards])

    def test_checkpoint_shards_cannot_escape_component_root(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            component = base / "transformer"
            component.mkdir()
            outside = base / "outside.safetensors"
            outside.touch()
            index = component / "model.safetensors.index.json"

            for shard_name in ("../outside.safetensors", str(outside.resolve())):
                with self.subTest(shard_name=shard_name):
                    _write_json(index, {"weight_map": {"x": shard_name}})
                    with self.assertRaisesRegex(H3ComponentError, "stay below"):
                        _checkpoint_index(component)

            linked = component / "linked.safetensors"
            linked.symlink_to(outside)
            _write_json(index, {"weight_map": {"x": linked.name}})
            with self.assertRaisesRegex(H3ComponentError, "resolves outside"):
                _checkpoint_index(component)

            index.unlink()
            linked.unlink()
            (component / "model.safetensors").symlink_to(outside)
            with self.assertRaisesRegex(H3ComponentError, "resolves outside"):
                _checkpoint_index(component)

    def test_checkpoint_index_accepts_contained_nested_shards(self):
        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw) / "transformer"
            shard = component / "shards" / "part-00001.safetensors"
            shard.parent.mkdir(parents=True)
            shard.touch()
            _write_json(
                component / "model.safetensors.index.json",
                {"weight_map": {"x": "shards/part-00001.safetensors"}},
            )
            files, weight_map = _checkpoint_index(component)
            self.assertEqual(files, [shard.resolve()])
            self.assertEqual(
                weight_map,
                {"x": "shards/part-00001.safetensors"},
            )

    def test_model_patcher_offload_uses_manager_and_clears_compute_marker(self):
        class FakeModel:
            def __init__(self):
                self._h3_compute_device = "cuda:0"

            def to(self, _device):
                raise AssertionError("patcher-owned model must not move directly")

        class FakePatcher:
            model = object()
            pinned = ()

            def __init__(self):
                self.loaded = 1
                self.detach_calls = 0

            def loaded_size(self):
                return self.loaded

            def detach(self):
                self.detach_calls += 1
                self.loaded = 0

        patcher = FakePatcher()
        unloaded = []
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")

        def unload_model_and_clones(value):
            unloaded.append(value)
            value.loaded = 0

        management.unload_model_and_clones = unload_model_and_clones
        comfy.model_management = management
        handle = H3ModelHandle(
            model=FakeModel(),
            model_patcher=patcher,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device="cuda:0",
            offload_device="cpu",
            dtype="bfloat16",
            metadata={},
            checkpoint_files=(),
        )
        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.model_management": management},
        ):
            handle.offload_after_inference()

        self.assertEqual(unloaded, [patcher])
        self.assertEqual(patcher.detach_calls, 0)
        self.assertFalse(hasattr(handle.model, "_h3_compute_device"))

        # Context-manager cleanup may be invoked again after an acquisition
        # error; marker removal and patcher unload must remain idempotent.
        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.model_management": management},
        ):
            handle.offload_after_inference()
        self.assertEqual(unloaded, [patcher, patcher])

    def test_int8_partial_load_moves_native_tensors_and_offloads_them(self):
        from minimax_h3_nodes.runtime import model_loader

        class FakeModel:
            pass

        class FakePatcher:
            pass

        class FakeOOM(Exception):
            pass

        torch = types.ModuleType("torch")
        torch.OutOfMemoryError = FakeOOM
        torch.device = lambda value: value
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        model = FakeModel()
        patcher = FakePatcher()
        events = []

        def load_models_gpu(values, **kwargs):
            events.append(("manager-load", values, kwargs))

        management.load_models_gpu = load_models_gpu
        comfy.model_management = management
        handle = H3ModelHandle(
            model=model,
            model_patcher=patcher,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device="cuda:0",
            offload_device="cpu",
            dtype="bfloat16",
            metadata={},
            checkpoint_files=(),
            quantized=True,
        )

        def move(_model, device):
            self.assertIs(_model, model)
            events.append(("move-native", device))

        with mock.patch.dict(
            sys.modules,
            {
                "torch": torch,
                "comfy": comfy,
                "comfy.model_management": management,
            },
        ), mock.patch.object(
            model_loader, "_nonstreamable_tensor_bytes", return_value=123
        ), mock.patch.object(
            model_loader, "_move_nonstreamable_tensors", side_effect=move
        ), mock.patch.object(
            model_loader, "_misplaced_nonstreamable_tensors", return_value=[]
        ), mock.patch.object(
            model_loader,
            "_unload_model_patcher",
            side_effect=lambda value: events.append(("manager-unload", value)),
        ):
            self.assertIs(handle.load_for_inference(), model)
            self.assertEqual(model._h3_compute_device, "cuda:0")
            handle.offload_after_inference()

        self.assertEqual(events[0][0], "manager-load")
        self.assertEqual(events[0][1], [patcher])
        self.assertEqual(
            events[0][2]["memory_required"],
            model_loader.DIT_INFERENCE_RESERVE + 123,
        )
        self.assertEqual(
            events[1:],
            [
                ("move-native", "cuda:0"),
                ("manager-unload", patcher),
                ("move-native", "cpu"),
            ],
        )
        self.assertFalse(hasattr(model, "_h3_compute_device"))

    def test_int8_partial_native_move_failure_rolls_back_both_banks(self):
        from minimax_h3_nodes.runtime import model_loader

        class FakeModel:
            pass

        class FakePatcher:
            pass

        class FakeOOM(Exception):
            pass

        torch = types.ModuleType("torch")
        torch.OutOfMemoryError = FakeOOM
        torch.device = lambda value: value
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        management.load_models_gpu = lambda *_args, **_kwargs: None
        comfy.model_management = management
        model = FakeModel()
        patcher = FakePatcher()
        primary = RuntimeError("native CUDA move failed")
        events = []
        handle = H3ModelHandle(
            model=model,
            model_patcher=patcher,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device="cuda:0",
            offload_device="cpu",
            dtype="bfloat16",
            metadata={},
            checkpoint_files=(),
            quantized=True,
        )

        def move(_model, device):
            events.append(("move-native", device))
            if device == "cuda:0":
                raise primary

        with mock.patch.dict(
            sys.modules,
            {
                "torch": torch,
                "comfy": comfy,
                "comfy.model_management": management,
            },
        ), mock.patch.object(
            model_loader, "_nonstreamable_tensor_bytes", return_value=0
        ), mock.patch.object(
            model_loader, "_move_nonstreamable_tensors", side_effect=move
        ), mock.patch.object(
            model_loader,
            "_unload_model_patcher",
            side_effect=lambda value: events.append(("manager-unload", value)),
        ):
            with self.assertRaises(RuntimeError) as caught:
                handle.load_for_inference()

        self.assertIs(caught.exception, primary)
        self.assertEqual(
            events,
            [
                ("move-native", "cuda:0"),
                ("manager-unload", patcher),
                ("move-native", "cpu"),
            ],
        )
        self.assertFalse(hasattr(model, "_h3_compute_device"))

    def test_full_load_placement_failure_cleans_patcher_and_marker_without_torch(self):
        class FakeTensor:
            device = "cpu"

        class FakeModel:
            def named_parameters(self):
                return (("weight", FakeTensor()),)

            def named_buffers(self):
                return ()

            def to(self, _device):
                raise AssertionError("patcher-owned model must not move directly")

        class FakePatcher:
            model = object()
            pinned = ()

            def __init__(self):
                self.loaded = 0

            def loaded_size(self):
                return self.loaded

            def detach(self):
                self.loaded = 0

        class FakeOOM(Exception):
            pass

        patcher = FakePatcher()
        model = FakeModel()
        unload_calls = []
        torch = types.ModuleType("torch")
        torch.OutOfMemoryError = FakeOOM
        torch.device = lambda value: value
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")

        def load_models_gpu(values, **kwargs):
            self.assertEqual(values, [patcher])
            self.assertTrue(kwargs["force_full_load"])
            patcher.loaded = 1

        def unload_model_and_clones(value):
            unload_calls.append(value)
            value.loaded = 0

        management.load_models_gpu = load_models_gpu
        management.unload_model_and_clones = unload_model_and_clones
        comfy.model_management = management
        handle = H3ModelHandle(
            model=model,
            model_patcher=patcher,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device="cuda:0",
            offload_device="cpu",
            dtype="bfloat16",
            metadata={},
            checkpoint_files=(),
            quantized=False,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "torch": torch,
                "comfy": comfy,
                "comfy.model_management": management,
            },
        ), self.assertRaisesRegex(RuntimeError, "did not fully load"):
            handle.load_for_inference()

        self.assertEqual(unload_calls, [patcher])
        self.assertEqual(patcher.loaded, 0)
        self.assertFalse(hasattr(model, "_h3_compute_device"))

    def test_model_patcher_offload_detaches_when_manager_fails(self):
        class FakeModel:
            def __init__(self):
                self._h3_compute_device = "cuda:0"

        class FakePatcher:
            model = object()
            pinned = ()

            def __init__(self):
                self.detach_calls = 0

            def loaded_size(self):
                return 1

            def detach(self):
                self.detach_calls += 1

        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")

        def fail_unload(_value):
            raise RuntimeError("manager failure")

        management.unload_model_and_clones = fail_unload
        comfy.model_management = management
        patcher = FakePatcher()
        handle = H3ModelHandle(
            model=FakeModel(),
            model_patcher=patcher,
            component_path=Path("/tmp/fake-h3-transformer"),
            load_device="cuda:0",
            offload_device="cpu",
            dtype="bfloat16",
            metadata={},
            checkpoint_files=(),
        )
        with mock.patch.dict(
            sys.modules,
            {"comfy": comfy, "comfy.model_management": management},
        ):
            handle.offload_after_inference()

        self.assertEqual(patcher.detach_calls, 1)
        self.assertFalse(hasattr(handle.model, "_h3_compute_device"))

    def test_ref_loader_passes_partition_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Ref2VA"
            component = root / "transformer"
            component.mkdir(parents=True)
            _write_release(root, "ref2va")
            _write_json(component / "config.json", {})
            # The loader reaches architecture validation instead of the old
            # unconditional "T2VA only" rejection.
            with self.assertRaisesRegex(H3ComponentError, "missing H3 transformer"):
                load_h3_model(
                    str(root),
                    partition="ref2va",
                    task="ref2va",
                    transformer_path=str(component),
                )


if __name__ == "__main__":
    unittest.main()
