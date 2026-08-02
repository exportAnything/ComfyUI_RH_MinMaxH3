import ast
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


_PKG = Path(__file__).resolve().parents[1] / "minimax_h3_nodes"
_API = _PKG / "api"
_NODES = _PKG / "nodes.py"
# 类/helpers 在 api/*；注册表字面量在 nodes.py facade
SOURCE = "\n".join(
    (_API / n).read_text(encoding="utf-8")
    for n in ("_shared.py", "loaders.py", "targets.py", "conditioning.py", "sampling_nodes.py", "decode.py")
)
TREE = ast.parse(SOURCE, filename=str(_API / "_shared.py"))
_NODES_SOURCE = _NODES.read_text(encoding="utf-8")
_NODES_TREE = ast.parse(_NODES_SOURCE, filename=str(_NODES))


def _class(name: str) -> ast.ClassDef:
    for node in TREE.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"missing class {name}")


def _function(name: str) -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _mapping_keys(name: str) -> set[str]:
    for node in _NODES_TREE.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise AssertionError(f"{name} must remain a literal dict")
        return {
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    raise AssertionError(f"missing {name}")


class NodeV2StaticTests(unittest.TestCase):
    LEGACY_NODE_IDS = {
        "RHMiniMaxH3DirectModelLoader",
        "RHMiniMaxH3DirectTextEncoderLoader",
        "RHMiniMaxH3DirectVAELoader",
        "RHMiniMaxH3T2VATarget",
        "RHMiniMaxH3T2VATextEncode",
        "RHMiniMaxH3UnsupportedConditioning",
        "RHMiniMaxH3EmptyAVLatent",
        "RHMiniMaxH3DualSigmaSampler",
        "RHMiniMaxH3DecodeAV",
    }
    V2_NODE_IDS = {
        "RHMiniMaxH3FL2VAModelLoader",
        "RHMiniMaxH3FL2VATextEncoderLoader",
        "RHMiniMaxH3FL2VAVAELoader",
        "RHMiniMaxH3Ref2VAModelLoader",
        "RHMiniMaxH3Ref2VATextEncoderLoader",
        "RHMiniMaxH3Ref2VAVAELoader",
        "RHMiniMaxH3FL2VAFirstFrameCondition",
        "RHMiniMaxH3FL2VALastFrameCondition",
        "RHMiniMaxH3FL2VATarget",
        "RHMiniMaxH3FL2VAEncode",
        "RHMiniMaxH3Ref2VAImageReference",
        "RHMiniMaxH3Ref2VAAudioReference",
        "RHMiniMaxH3Ref2VAVideoReference",
        "RHMiniMaxH3Ref2VATarget",
        "RHMiniMaxH3Ref2VAEncode",
    }

    def test_legacy_and_v2_node_ids_are_registered_and_named(self):
        registered = _mapping_keys("NODE_CLASS_MAPPINGS")
        displayed = _mapping_keys("NODE_DISPLAY_NAME_MAPPINGS")
        expected = self.LEGACY_NODE_IDS | self.V2_NODE_IDS
        self.assertTrue(expected <= registered)
        self.assertTrue(expected <= displayed)

    def test_reference_boundaries_use_standard_comfy_media_types(self):
        requirements = {
            "MiniMaxH3FL2VAFirstFrameCondition": {"IMAGE"},
            "MiniMaxH3FL2VALastFrameCondition": {"IMAGE"},
            "MiniMaxH3Ref2VAImageReference": {"IMAGE"},
            "MiniMaxH3Ref2VAAudioReference": {"AUDIO"},
            "MiniMaxH3Ref2VAVideoReference": {"VIDEO"},
            "MiniMaxH3DecodeAV": {"IMAGE", "AUDIO"},
        }
        for class_name, expected_types in requirements.items():
            constants = {
                node.value
                for node in ast.walk(_class(class_name))
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            with self.subTest(class_name=class_name):
                self.assertTrue(expected_types <= constants)

    def test_heavy_optional_numpy_is_not_imported_at_module_import_time(self):
        for node in TREE.body:
            if isinstance(node, ast.Import):
                self.assertNotIn("numpy", {alias.name for alias in node.names})
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "numpy")

    def test_v2_critical_paths_do_not_silently_drop_arguments(self):
        scopes = {
            "_MiniMaxH3ExplicitModelLoader": _class,
            "_MiniMaxH3ExplicitTextEncoderLoader": _class,
            "_MiniMaxH3ExplicitVAELoader": _class,
            "MiniMaxH3FL2VAEncode": _class,
            "MiniMaxH3Ref2VAEncode": _class,
            # 双 VAE 选择逻辑被两个 loader 家族共用，保证移出类体后仍受同一约束
            "_load_dual_vae": _function,
        }
        for name, lookup in scopes.items():
            calls = {
                node.func.id
                for node in ast.walk(lookup(name))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            with self.subTest(name=name):
                self.assertNotIn("_call_supported", calls)

    def test_all_loader_families_forward_partition_component_file_hints(self):
        # 两个 VAE loader 家族都把选择/解析委托给 _load_dual_vae，v1/v2 各一条分支。
        expected_helpers = (
            ("MiniMaxH3DirectModelLoader", _class, "_resolve_t2va_release"),
            ("MiniMaxH3DirectTextEncoderLoader", _class, "_resolve_t2va_release"),
            ("_MiniMaxH3ExplicitModelLoader", _class, "_resolve_release"),
            ("_MiniMaxH3ExplicitTextEncoderLoader", _class, "_resolve_release"),
            ("_load_dual_vae", _function, "_resolve_t2va_release"),
            ("_load_dual_vae", _function, "_resolve_release"),
        )
        for class_name, lookup, helper_name in expected_helpers:
            calls = [
                node
                for node in ast.walk(lookup(class_name))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == helper_name
            ]
            with self.subTest(class_name=class_name, helper=helper_name):
                self.assertEqual(len(calls), 1)
                keyword_names = {item.arg for item in calls[0].keywords}
                self.assertIn("required_component", keyword_names)
                self.assertIn("required_files", keyword_names)

        for function_name, callee_name in (
            ("_existing_directory", "resolver"),
            ("_resolve_t2va_release", "_existing_directory"),
            ("_resolve_release", "_existing_directory"),
        ):
            calls = [
                node
                for node in ast.walk(_function(function_name))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == callee_name
            ]
            with self.subTest(function_name=function_name):
                self.assertEqual(len(calls), 1)
                keyword_names = {item.arg for item in calls[0].keywords}
                self.assertTrue(
                    {"partition", "required_component", "required_files"}
                    <= keyword_names
                )

        for function_name in ("_resolve_t2va_release", "_resolve_release"):
            source = ast.get_source_segment(SOURCE, _function(function_name))
            self.assertIn("required_component=required_component", source)
            self.assertIn("required_files=required_files", source)

    def test_ref_video_uses_official_sampling_and_prepared_stream(self):
        encode_source = ast.get_source_segment(
            SOURCE, _class("MiniMaxH3Ref2VAEncode")
        )
        self.assertIn("minimax_h3_qwen_video_sample_plan", encode_source)
        self.assertIn("decode_reference_video_samples", encode_source)
        self.assertIn('materialized["prepared_path"]', encode_source)
        self.assertIn('materialized["original_path"]', encode_source)
        self.assertNotIn("_qwen_sample_prepared_video", encode_source)

    def test_ref_image_size_defaults_to_match_and_keys_both_caches(self):
        encode_source = ast.get_source_segment(
            SOURCE, _class("MiniMaxH3Ref2VAEncode")
        )
        self.assertIn("ref_image_size", encode_source)
        self.assertIn("H3_REFERENCE_IMAGE_SIZE_MATCH", encode_source)
        self.assertIn("canvas_width=int(clean_target[\"width\"])", encode_source)
        # 换尺寸策略会换像素：VAE 行缓存与 Qwen 编码缓存都必须带上它
        self.assertIn("shape_policy=str(record.get(\"shape_policy\", \"\"))", encode_source)
        self.assertIn('f"refimg={image_size_mode}"', encode_source)

    def test_crossed_material_branches_fail_before_qwen(self):
        for class_name in ("MiniMaxH3FL2VAEncode", "MiniMaxH3Ref2VAEncode"):
            encode_source = ast.get_source_segment(SOURCE, _class(class_name))
            with self.subTest(class_name=class_name):
                self.assertLess(
                    encode_source.index("_preflight_target_conditions"),
                    encode_source.index("_multimodal_qwen_encode"),
                )

    def test_v2_nodes_use_canonical_text_token_tags(self):
        build_source = ast.get_source_segment(SOURCE, _class("MiniMaxH3FL2VAEncode"))
        ref_source = ast.get_source_segment(SOURCE, _class("MiniMaxH3Ref2VAEncode"))
        packing = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_v2_packed"
        )
        packing_source = ast.get_source_segment(SOURCE, packing)
        for source in (build_source, ref_source, packing_source):
            self.assertIn("text_token_tags", source)
        self.assertNotIn('clean.get("token_tags")', packing_source)

    def test_video_probe_uses_coded_geometry_factory(self):
        functions = [
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_probe_video_path"
        ]
        self.assertEqual(len(functions), 1)
        probe_source = ast.get_source_segment(SOURCE, functions[0])
        self.assertIn("ReferenceVideoMetadata.from_coded", probe_source)
        self.assertIn("display_aspect_ratio=display_aspect_ratio", probe_source)

        helper = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_probe_video_display_metadata"
        )
        helper_source = ast.get_source_segment(SOURCE, helper)
        self.assertIn("display_aspect_ratio", helper_source)
        self.assertIn("stream_side_data=rotation", helper_source)

        source_helper = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_video_source_path"
        )
        source_helper_text = ast.get_source_segment(SOURCE, source_helper)
        self.assertIn("save_to", source_helper_text)

    def test_ref_scratch_space_uses_comfy_managed_temp_root(self):
        helper = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_h3_temporary_directory"
        )
        helper_source = ast.get_source_segment(SOURCE, helper)
        ref_source = ast.get_source_segment(SOURCE, _class("MiniMaxH3Ref2VAEncode"))
        self.assertIn("folder_paths.get_temp_directory", helper_source)
        self.assertIn("_h3_temporary_directory", ref_source)
        self.assertNotIn("tempfile.TemporaryDirectory", ref_source)
        self.assertEqual(SOURCE.count("tempfile.TemporaryDirectory"), 1)
        self.assertNotIn("tempfile.gettempdir", SOURCE)
        self.assertNotIn('"/tmp', SOURCE)

    def test_long_multimodal_encode_nodes_publish_progress(self):
        for class_name in ("MiniMaxH3FL2VAEncode", "MiniMaxH3Ref2VAEncode"):
            encode_source = ast.get_source_segment(SOURCE, _class(class_name))
            with self.subTest(class_name=class_name):
                self.assertIn("_new_progress", encode_source)
                self.assertIn("_advance_progress", encode_source)
                self.assertIn("_check_interrupted", encode_source)

    def test_ref_media_execution_forwards_comfy_cancellation(self):
        encode_source = ast.get_source_segment(
            SOURCE, _class("MiniMaxH3Ref2VAEncode")
        )
        self.assertEqual(
            encode_source.count("interrupt_check=_check_interrupted"), 3
        )
        decode_helper = ast.get_source_segment(
            SOURCE, _function("_decode_video_file")
        )
        probe_helper = ast.get_source_segment(SOURCE, _function("_probe_video_path"))
        self.assertIn("_check_interrupted()", decode_helper)
        self.assertIn("_check_interrupted()", probe_helper)

    def test_multimodal_progress_totals_close_exactly(self):
        fl_source = ast.get_source_segment(SOURCE, _class("MiniMaxH3FL2VAEncode"))
        ref_source = ast.get_source_segment(SOURCE, _class("MiniMaxH3Ref2VAEncode"))
        self.assertIn(
            "progress_total = 3 + 2 * len(clean_keyframes)", fl_source
        )
        self.assertEqual(fl_source.count("_advance_progress"), 5)
        self.assertIn(
            "progress_total = 3 + 3 * len(clean_references)", ref_source
        )
        self.assertEqual(ref_source.count("_advance_progress"), 6)

    def test_same_kind_ref_material_swap_is_rejected_before_encode(self):
        from minimax_h3_nodes.contracts import (
            H3_TASK_REF2VA,
            make_ref2va_reference,
            make_ref2va_references,
            resolve_ref2va_target_v2,
            validate_ref2va_references,
        )
        from minimax_h3_nodes.nodes import _preflight_target_conditions

        first = make_ref2va_reference(
            "image", object(), display_width=640, display_height=480
        )
        second = make_ref2va_reference(
            "image", object(), display_width=640, display_height=480
        )
        original = make_ref2va_references([first, second])
        target = resolve_ref2va_target_v2(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            references=original,
        )
        swapped = validate_ref2va_references(
            make_ref2va_references([second, first])
        )
        self.assertNotEqual(
            first["material_fingerprint"], second["material_fingerprint"]
        )
        with self.assertRaisesRegex(ValueError, "ordered references"):
            _preflight_target_conditions(
                task=H3_TASK_REF2VA,
                target=target,
                conditions=swapped,
            )

    def test_ref_encode_forwards_material_fingerprint_to_both_vae_routes(self):
        ref_source = ast.get_source_segment(SOURCE, _class("MiniMaxH3Ref2VAEncode"))
        self.assertIn(
            '"material_fingerprint": str(item["material_fingerprint"])',
            ref_source,
        )
        self.assertEqual(ref_source.count("material_fingerprint=str("), 2)

    def test_definition_time_partition_choices_never_call_release_resolver(self):
        from minimax_h3_nodes import nodes
        from minimax_h3_nodes.api import _shared

        partition_function = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_partition_roots"
        )
        partition_source = ast.get_source_segment(SOURCE, partition_function)
        self.assertNotIn("resolve_partition_root", partition_source)
        self.assertNotIn("release_metadata", partition_source)

        with tempfile.TemporaryDirectory() as raw:
            release = Path(raw) / "MiniMax-H3"
            fl = release / "FL2VA"
            ref = release / "Ref2VA"
            fl.mkdir(parents=True)
            ref.mkdir()
            # Deliberately expose no resolver/metadata methods.  Any
            # definition-stage call to one fails this test immediately.
            fake_components = types.SimpleNamespace(
                list_h3_model_root_paths=lambda: [release]
            )
            with mock.patch.object(
                _shared, "_runtime_module", return_value=fake_components
            ):
                self.assertEqual(nodes._partition_roots("fl2va"), [fl.resolve()])
                self.assertEqual(nodes._partition_roots("ref2va"), [ref.resolve()])

    def test_transformer_acquire_failure_still_offloads_handle(self):
        from minimax_h3_nodes import nodes

        class FailingHandle:
            def __init__(self):
                self.cleanup_calls = 0

            @staticmethod
            def load_for_inference():
                raise RuntimeError("synthetic acquire failure")

            def offload_after_inference(self):
                self.cleanup_calls += 1

        handle = FailingHandle()
        wrapper = {
            "schema": nodes.H3_MODEL_SCHEMA,
            "handle": handle,
        }
        with self.assertRaisesRegex(RuntimeError, "synthetic acquire failure"):
            with nodes._transformer_session(wrapper):
                self.fail("acquire failure must happen before yield")
        self.assertEqual(handle.cleanup_calls, 1)

    def test_transformer_session_body_failure_still_offloads_handle(self):
        from minimax_h3_nodes import nodes

        class LoadedHandle:
            def __init__(self):
                self.cleanup_calls = 0

            @staticmethod
            def load_for_inference():
                return lambda **_kwargs: None

            def offload_after_inference(self):
                self.cleanup_calls += 1

        handle = LoadedHandle()
        wrapper = {"schema": nodes.H3_MODEL_SCHEMA, "handle": handle}
        with self.assertRaisesRegex(ValueError, "synthetic sampler failure"):
            with nodes._transformer_session(wrapper):
                raise ValueError("synthetic sampler failure")
        self.assertEqual(handle.cleanup_calls, 1)

    def test_all_vae_residency_paths_use_cleanup_session(self):
        expected_session_counts = {
            "MiniMaxH3FL2VAEncode": 1,
            "MiniMaxH3Ref2VAEncode": 2,
            "MiniMaxH3DecodeAV": 2,
        }
        for class_name, expected_count in expected_session_counts.items():
            source = ast.get_source_segment(SOURCE, _class(class_name))
            with self.subTest(class_name=class_name):
                self.assertEqual(
                    source.count("_vae_device_session("), expected_count
                )
                self.assertNotIn(".to(load_device)", source)

    def test_vae_device_session_cleans_up_when_device_move_fails(self):
        from minimax_h3_nodes import nodes
        from minimax_h3_nodes.api import _shared

        events = []

        class FailingAdapter:
            @staticmethod
            def to(_device):
                events.append("to")
                raise RuntimeError("synthetic VAE move failure")

            @staticmethod
            def offload():
                events.append("offload")

        management = types.SimpleNamespace(
            soft_empty_cache=lambda: events.append("soft_empty_cache")
        )
        # 实现在 api._shared；需 patch 该模块的 model_management
        with mock.patch.object(_shared, "model_management", management), mock.patch(
            "minimax_h3_nodes.runtime.h3_settings.OPT_VAE_RESIDENCY", False
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic VAE move failure"):
                with nodes._vae_device_session(FailingAdapter(), "cuda:0"):
                    self.fail("a failed move must not enter the session body")
        self.assertEqual(events, ["to", "offload", "soft_empty_cache"])

    def test_vae_device_session_flushes_cache_when_offload_fails(self):
        from minimax_h3_nodes import nodes
        from minimax_h3_nodes.api import _shared

        events = []

        class FailingOffloadAdapter:
            @staticmethod
            def to(_device):
                events.append("to")

            @staticmethod
            def offload():
                events.append("offload")
                raise RuntimeError("synthetic VAE offload failure")

        management = types.SimpleNamespace(
            soft_empty_cache=lambda: events.append("soft_empty_cache")
        )
        with mock.patch.object(_shared, "model_management", management), mock.patch(
            "minimax_h3_nodes.runtime.h3_settings.OPT_VAE_RESIDENCY", False
        ):
            with self.assertRaisesRegex(
                RuntimeError, "synthetic VAE offload failure"
            ):
                with nodes._vae_device_session(
                    FailingOffloadAdapter(), "cuda:0"
                ):
                    events.append("body")
        self.assertEqual(
            events, ["to", "body", "offload", "soft_empty_cache"]
        )


if __name__ == "__main__":
    unittest.main()
