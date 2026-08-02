import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from minimax_h3_nodes.runtime.components import _impl as components_module
from minimax_h3_nodes.runtime.components import (
    H3ComponentError,
    list_h3_model_roots,
    model_root_path,
    release_metadata,
    release_sigma_shift_scales,
    resolve_component,
    resolve_partition_root,
    validate_t2va_partition,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_partition(
    bundle: Path,
    partition: str,
    components: dict[str, tuple[str, ...]],
    *,
    metadata: bool = True,
) -> Path:
    directory_name = "FL2VA" if partition == "fl2va" else "Ref2VA"
    root = bundle / directory_name
    root.mkdir(parents=True)
    if metadata:
        tasks = ["t2va", "fl2va"] if partition == "fl2va" else ["ref2va"]
        _write_json(
            root / "model_index.json",
            {
                "_minimax_h3": {
                    "schema_version": 1,
                    "partition": partition,
                    "tasks": tasks,
                    "task_aliases": {},
                    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                }
            },
        )
    for component, files in components.items():
        component_root = root / component
        component_root.mkdir(parents=True)
        for relative in files:
            target = component_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_json(target, {})
    return root


@contextmanager
def _resolver_candidates(models_dir: Path, roots: list[Path]):
    fake_fp = mock.Mock()
    fake_fp.models_dir = str(models_dir)
    fake_fp.get_folder_paths.return_value = []
    with mock.patch.dict("sys.modules", {"folder_paths": fake_fp}), mock.patch.object(
        components_module,
        "_model_bucket_paths",
        return_value=[],
    ), mock.patch.object(
        components_module,
        "list_h3_model_root_paths",
        return_value=roots,
    ):
        yield


class ComponentTests(unittest.TestCase):
    def test_model_root_resolves_under_diffusers_alias(self):
        from unittest import mock

        from minimax_h3_nodes.runtime.components import (
            list_h3_model_roots,
            model_root_path,
        )

        with tempfile.TemporaryDirectory() as raw:
            models_dir = Path(raw)
            target = models_dir / "diffusers" / "MiniMax-H3"
            target.mkdir(parents=True)
            (target / "FL2VA").mkdir()
            noise = models_dir / "diffusers" / "unrelated-model"
            noise.mkdir(parents=True)
            fake_fp = mock.Mock()
            fake_fp.models_dir = str(models_dir)
            with mock.patch.dict("sys.modules", {"folder_paths": fake_fp}):
                self.assertEqual(model_root_path("MiniMax-H3"), target.resolve())
                self.assertEqual(
                    model_root_path("diffusers/MiniMax-H3"), target.resolve()
                )
                roots = list_h3_model_roots()
                self.assertTrue(
                    roots == ["MiniMax-H3"]
                    or (len(roots) == 1 and str(roots[0]).endswith("MiniMax-H3")),
                    roots,
                )

    def test_complete_persistent_release_wins_over_stale_local_cache(self):
        from unittest import mock

        from minimax_h3_nodes.runtime.components import model_root_path

        with tempfile.TemporaryDirectory() as raw:
            models_dir = Path(raw) / "models"
            local_base = models_dir / "diffusers"
            persistent_base = Path(raw) / "persistent" / "diffusers"
            local = local_base / "MiniMax-H3"
            persistent = persistent_base / "MiniMax-H3"
            for root in (local, persistent):
                (root / "FL2VA" / "transformer").mkdir(parents=True)
                (root / "FL2VA" / "text_encoder").mkdir()
                _write_json(root / "FL2VA" / "transformer" / "config.json", {})
                _write_json(root / "FL2VA" / "text_encoder" / "config.json", {})
            for component in (
                "transformer_int8_convrot",
                "text_encoder_int8_convrot",
            ):
                (persistent / "FL2VA" / component).mkdir()
                _write_json(
                    persistent / "FL2VA" / component / "config.json", {}
                )
            for component in ("video_vae", "audio_vae"):
                (persistent / "FL2VA" / "vae" / component).mkdir(parents=True)
                _write_json(
                    persistent
                    / "FL2VA"
                    / "vae"
                    / component
                    / "config.json",
                    {},
                )

            fake_fp = mock.Mock()
            fake_fp.models_dir = str(models_dir)
            fake_fp.get_folder_paths.return_value = [
                str(local_base),
                str(persistent_base),
            ]
            with mock.patch.dict("sys.modules", {"folder_paths": fake_fp}):
                self.assertEqual(
                    model_root_path("MiniMax-H3"), persistent.resolve()
                )

    def test_duplicate_basename_choices_are_absolute_but_unique_stays_short(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            first = base / "local" / "MiniMax-H3"
            second = base / "nfs" / "MiniMax-H3"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            with mock.patch.object(
                components_module,
                "list_h3_model_root_paths",
                return_value=[first, second],
            ):
                self.assertEqual(
                    list_h3_model_roots(),
                    [str(first.resolve()), str(second.resolve())],
                )
            with mock.patch.object(
                components_module,
                "list_h3_model_root_paths",
                return_value=[first],
            ):
                roots = list_h3_model_roots()
                self.assertTrue(
                    roots == ["MiniMax-H3"]
                    or (len(roots) == 1 and str(roots[0]).endswith("MiniMax-H3")),
                    roots,
                )

            alias = base / "alias" / "MiniMax-H3"
            alias.parent.mkdir()
            alias.symlink_to(first, target_is_directory=True)
            with mock.patch.object(
                components_module,
                "list_h3_model_root_paths",
                return_value=[first, alias],
            ):
                roots = list_h3_model_roots()
                self.assertTrue(
                    roots == ["MiniMax-H3"]
                    or (len(roots) == 1 and str(roots[0]).endswith("MiniMax-H3")),
                    roots,
                )

    def test_hinted_root_selects_ref_partition_instead_of_local_fl_only(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            local = base / "local" / "MiniMax-H3"
            nfs = base / "nfs" / "MiniMax-H3"
            _write_partition(
                local,
                "fl2va",
                {"transformer": ("config.json",)},
            )
            _write_partition(
                nfs,
                "ref2va",
                {"transformer": ("config.json",)},
            )
            with _resolver_candidates(base / "models", [local, nfs]):
                selected = model_root_path(
                    "MiniMax-H3",
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )
            self.assertEqual(selected, nfs.resolve())

    def test_selected_bf16_component_is_not_stolen_by_int8_root(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            bf16 = base / "bf16" / "MiniMax-H3"
            int8 = base / "int8" / "MiniMax-H3"
            _write_partition(
                bf16,
                "ref2va",
                {"transformer": ("config.json",)},
            )
            _write_partition(
                int8,
                "ref2va",
                {
                    "transformer_int8_convrot": ("config.json",),
                    "text_encoder_int8_convrot": ("config.json",),
                    "vae": (
                        "video_vae/config.json",
                        "audio_vae/config.json",
                    ),
                },
            )
            with _resolver_candidates(base / "models", [bf16, int8]):
                selected = model_root_path(
                    "MiniMax-H3",
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )
            self.assertEqual(selected, bf16.resolve())

    def test_hinted_root_uses_unique_partition_completeness_high_score(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            sparse = base / "sparse" / "MiniMax-H3"
            complete = base / "complete" / "MiniMax-H3"
            _write_partition(
                sparse,
                "ref2va",
                {"transformer": ("config.json",)},
            )
            _write_partition(
                complete,
                "ref2va",
                {
                    "transformer": ("config.json",),
                    "text_encoder": ("config.json",),
                    "vae": (
                        "video_vae/config.json",
                        "audio_vae/config.json",
                    ),
                },
            )
            with _resolver_candidates(base / "models", [sparse, complete]):
                selected = model_root_path(
                    "MiniMax-H3",
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )
            self.assertEqual(selected, complete.resolve())

    def test_hinted_equal_score_follows_search_path_order(self):
        """同名同分时按 ComfyUI 搜索路径顺序取首个，不再判歧义。

        同一份 release 常被挂多次（本机 models 目录 + 全局 NFS），两条路径都命中
        时报错会让用户无从下手。语义对齐 ``folder_paths.get_full_path``：靠前的
        搜索路径优先。完整度仍然先于顺序——见
        ``test_complete_persistent_release_wins_over_stale_local_cache``。
        """

        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            first = base / "one" / "MiniMax-H3"
            second = base / "two" / "MiniMax-H3"
            for root in (first, second):
                _write_partition(
                    root,
                    "ref2va",
                    {"transformer": ("config.json",)},
                )
            with _resolver_candidates(base / "models", [first, second]):
                selected = model_root_path(
                    "MiniMax-H3",
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )
            self.assertEqual(selected, first.resolve())
            # 顺序反过来，选中的也跟着换——证明是顺序决定而不是路径字典序
            with _resolver_candidates(base / "models", [second, first]):
                selected = model_root_path(
                    "MiniMax-H3",
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )
            self.assertEqual(selected, second.resolve())

    def test_absolute_root_is_authoritative_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            explicit = base / "explicit" / "MiniMax-H3"
            other = base / "other" / "MiniMax-H3"
            _write_partition(
                explicit,
                "ref2va",
                {"transformer": ("config.json",)},
            )
            _write_partition(
                other,
                "ref2va",
                {
                    "transformer": ("config.json",),
                    "text_encoder": ("config.json",),
                },
            )
            with _resolver_candidates(base / "models", [explicit, other]):
                self.assertEqual(
                    model_root_path(
                        explicit,
                        partition="ref2va",
                        required_component="transformer",
                        required_files=("config.json",),
                    ),
                    explicit.resolve(),
                )
                with self.assertRaisesRegex(H3ComponentError, "No MiniMax-H3 root"):
                    model_root_path(
                        explicit,
                        partition="ref2va",
                        required_component="missing-transformer",
                        required_files=("config.json",),
                    )

    def test_single_root_keeps_old_no_hint_and_new_hint_behavior(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            release = base / "only" / "MiniMax-H3"
            _write_partition(
                release,
                "fl2va",
                {"transformer": ("config.json",)},
            )
            with _resolver_candidates(base / "models", [release]):
                self.assertEqual(
                    model_root_path("MiniMax-H3"),
                    release.resolve(),
                )
                self.assertEqual(
                    model_root_path(
                        "MiniMax-H3",
                        partition="fl2va",
                        required_component="transformer",
                        required_files=("config.json",),
                    ),
                    release.resolve(),
                )

    def test_metadata_free_named_partition_child_is_supported(self):
        with tempfile.TemporaryDirectory() as raw:
            bundle = Path(raw) / "MiniMax-H3"
            ref = bundle / "Ref2VA"
            (ref / "transformer").mkdir(parents=True)
            _write_json(ref / "transformer" / "config.json", {})
            self.assertEqual(
                resolve_partition_root(bundle, "ref2va"),
                ref.resolve(),
            )

    def test_hinted_required_file_must_be_contained_regular_file(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            bundle = base / "MiniMax-H3"
            partition = bundle / "Ref2VA"
            component = partition / "transformer"
            component.mkdir(parents=True)
            _write_json(
                partition / "model_index.json",
                {"_minimax_h3": {"partition": "ref2va"}},
            )

            (component / "config.json").mkdir()
            with self.assertRaisesRegex(H3ComponentError, "missing files"):
                model_root_path(
                    bundle,
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )
            (component / "config.json").rmdir()

            outside = base / "outside-config.json"
            _write_json(outside, {})
            (component / "config.json").symlink_to(outside)
            with self.assertRaisesRegex(H3ComponentError, "resolves outside"):
                model_root_path(
                    bundle,
                    partition="ref2va",
                    required_component="transformer",
                    required_files=("config.json",),
                )

            for unsafe in ("", ".", "../config.json", str(outside.resolve())):
                with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                    H3ComponentError,
                    "relative path",
                ):
                    model_root_path(
                        bundle,
                        partition="ref2va",
                        required_component="transformer",
                        required_files=(unsafe,),
                    )

    def test_hinted_selector_uses_non_recursive_partition_admission(self):
        import inspect

        selector_source = inspect.getsource(
            components_module._select_hinted_model_root
        )
        admission_source = inspect.getsource(
            components_module._admit_partition_root
        )
        self.assertNotIn("resolve_partition_root(", selector_source)
        self.assertNotIn("model_root_path(", admission_source)
        self.assertNotIn("release_metadata(", admission_source)

    def test_diffusers_pair_resolves_to_key_subfolder(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "transformer").mkdir()
            _write_json(root / "transformer" / "config.json", {"num_layers": 50})
            _write_json(
                root / "model_index.json",
                {"transformer": ["diffusers", "MiniMaxH3Transformer"]},
            )
            self.assertEqual(
                resolve_component(
                    root,
                    ("transformer",),
                    required_files=("config.json",),
                ),
                (root / "transformer").resolve(),
            )

    def test_model_index_component_rejects_parent_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "release"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            _write_json(outside / "config.json", {})
            for value in ("../outside", str(outside.resolve())):
                with self.subTest(value=value):
                    _write_json(
                        root / "model_index.json",
                        {"transformer": {"path": value}},
                    )
                    with self.assertRaisesRegex(
                        H3ComponentError,
                        "relative path without",
                    ):
                        resolve_component(
                            root,
                            ("transformer",),
                            required_files=("config.json",),
                        )

    def test_model_index_component_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "release"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            _write_json(outside / "config.json", {})
            (root / "linked-transformer").symlink_to(
                outside,
                target_is_directory=True,
            )
            _write_json(
                root / "model_index.json",
                {"transformer": {"path": "linked-transformer"}},
            )
            with self.assertRaisesRegex(H3ComponentError, "resolves outside"):
                resolve_component(
                    root,
                    ("transformer",),
                    required_files=("config.json",),
                )

    def test_component_key_fallback_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "release"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()
            _write_json(outside / "config.json", {})
            (root / "transformer").symlink_to(
                outside,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(H3ComponentError, "resolves outside"):
                resolve_component(
                    root,
                    ("transformer",),
                    required_files=("config.json",),
                )
            with self.assertRaisesRegex(
                H3ComponentError,
                "relative path without",
            ):
                resolve_component(
                    root,
                    ("../outside",),
                    required_files=("config.json",),
                )

    def test_explicit_component_checks_required_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "transformer_int8_convrot").mkdir()
            with self.assertRaisesRegex(H3ComponentError, "missing required"):
                resolve_component(
                    root,
                    ("transformer",),
                    explicit="transformer_int8_convrot",
                    required_files=("config.json",),
                )

    def test_explicit_subfolder_and_release_metadata(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "weights" / "dit").mkdir(parents=True)
            _write_json(root / "weights" / "dit" / "config.json", {})
            _write_json(
                root / "model_index.json",
                {
                    "transformer": {"path": "weights/dit"},
                    "_minimax_h3": {
                        "schema_version": 1,
                        "partition": "fl2va",
                        "tasks": ["t2va", "fl2va"],
                        "task_aliases": {},
                        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                    },
                },
            )
            self.assertEqual(
                resolve_component(
                    root,
                    ("transformer",),
                    required_files=("config.json",),
                ),
                (root / "weights" / "dit").resolve(),
            )
            self.assertEqual(release_metadata(root)["partition"], "fl2va")
            validate_t2va_partition(release_metadata(root))
            self.assertEqual(
                release_sigma_shift_scales(release_metadata(root)),
                {"video": 12.0, "audio": 3.0},
            )

    def test_parent_directory_resolves_real_fl2va_partition_layout(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            fl_root = root / "FL2VA"
            ref_root = root / "Ref2VA"
            fl_root.mkdir()
            ref_root.mkdir()
            _write_json(
                fl_root / "model_index.json",
                {
                    "transformer": ["diffusers", "MiniMaxH3DiTModel"],
                    "_minimax_h3": {
                        "schema_version": 1,
                        "partition": "fl2va",
                        "tasks": ["t2va", "fl2va"],
                        "task_aliases": {},
                        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                    },
                },
            )
            _write_json(
                ref_root / "model_index.json",
                {
                    "transformer": ["diffusers", "MiniMaxH3DiTModel"],
                    "_minimax_h3": {
                        "schema_version": 1,
                        "partition": "ref2va",
                        "tasks": ["ref2va"],
                        "task_aliases": {},
                        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                    },
                },
            )
            self.assertEqual(resolve_partition_root(root, "fl2va"), fl_root)
            self.assertEqual(resolve_partition_root(root, "ref2va"), ref_root)

    def test_partition_child_symlink_cannot_escape_selected_root(self):
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "MiniMax-H3"
            outside = base / "outside-ref2va"
            root.mkdir()
            outside.mkdir()
            _write_json(
                outside / "model_index.json",
                {
                    "_minimax_h3": {
                        "schema_version": 1,
                        "partition": "ref2va",
                        "tasks": ["ref2va"],
                        "task_aliases": {},
                        "sigma_shift_scales": {
                            "video": 12.0,
                            "audio": 3.0,
                        },
                    }
                },
            )
            (root / "Ref2VA").symlink_to(
                outside,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(H3ComponentError, "resolves outside"):
                resolve_partition_root(root, "ref2va")

    def test_direct_ref2va_root_is_not_redirected_to_sibling(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            _write_json(
                root / "model_index.json",
                {
                    "_minimax_h3": {
                        "partition": "ref2va",
                        "tasks": ["ref2va"],
                    }
                },
            )
            with self.assertRaisesRegex(H3ComponentError, "declares partition"):
                resolve_partition_root(root, "fl2va")

    def test_metadata_free_direct_root_rejects_opposite_named_partition(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            for directory_name, requested in (
                ("Ref2VA", "fl2va"),
                ("FL2VA", "ref2va"),
            ):
                with self.subTest(
                    directory_name=directory_name,
                    requested=requested,
                ):
                    root = parent / directory_name
                    (root / "transformer").mkdir(parents=True)
                    with self.assertRaisesRegex(
                        H3ComponentError,
                        "explicitly named for partition",
                    ):
                        resolve_partition_root(root, requested)

    def test_metadata_free_model_index_rejects_opposite_named_partition(self):
        with tempfile.TemporaryDirectory() as raw:
            parent = Path(raw)
            cases = (
                ("Ref2VA", "fl2va", {}),
                ("FL2VA", "ref2va", {"transformer": ["diffusers", "H3"]}),
            )
            for directory_name, requested, model_index in cases:
                with self.subTest(
                    directory_name=directory_name,
                    requested=requested,
                ):
                    root = parent / directory_name
                    root.mkdir()
                    _write_json(root / "model_index.json", model_index)
                    with self.assertRaisesRegex(
                        H3ComponentError,
                        "explicitly named for partition",
                    ):
                        resolve_partition_root(root, requested)

    def test_metadata_free_parent_partition_signal_is_enforced(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "Ref2VA" / "development-root"
            (root / "transformer").mkdir(parents=True)
            with self.assertRaisesRegex(
                H3ComponentError,
                "explicitly named for partition 'ref2va'",
            ):
                resolve_partition_root(root, "fl2va")

    def test_generic_metadata_free_direct_root_remains_compatible(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "development-root"
            (root / "transformer").mkdir(parents=True)
            self.assertEqual(resolve_partition_root(root, "fl2va"), root)

    def test_ref_only_partition_is_rejected(self):
        with self.assertRaisesRegex(H3ComponentError, "requires the FL2VA"):
            validate_t2va_partition(
                {"partition": "ref2va", "tasks": ["ref2va"]}
            )

    def test_malformed_tasks_and_sigma_metadata_are_rejected(self):
        with self.assertRaisesRegex(H3ComponentError, "tasks must be a list"):
            validate_t2va_partition(
                {"partition": "fl2va", "tasks": "t2va"}
            )
        with self.assertRaisesRegex(H3ComponentError, "missing audio"):
            validate_t2va_partition(
                {
                    "partition": "fl2va",
                    "tasks": ["t2va"],
                    "sigma_shift_scales": {"video": 12.0},
                }
            )


if __name__ == "__main__":
    unittest.main()
