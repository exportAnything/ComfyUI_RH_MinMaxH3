import json
import tempfile
import unittest
from pathlib import Path

from minimax_h3_nodes.runtime.components import (
    H3ComponentError,
    release_metadata,
    release_sigma_shift_scales,
    resolve_component,
    resolve_partition_root,
    validate_t2va_partition,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


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
                self.assertEqual(list_h3_model_roots(), ["MiniMax-H3"])

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
