#!/usr/bin/env python3
"""Validate a local MiniMax-H3 release without importing Torch or ComfyUI.

The script only reads JSON files, safetensors indexes, and safetensors headers.
It never materialises tensor data, so it is safe to run before the first
ComfyUI load of the roughly 100+ GiB model bundle.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024
QWEN_LAYER_KEY = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


class PreflightError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreflightError(f"missing required file: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"expected a JSON object: {path}")
    return value


def resolve_release_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise PreflightError(f"model root is not a directory: {root}")
    if (root / "model_index.json").is_file():
        return root

    matches: list[Path] = []
    for child in root.iterdir():
        index_path = child / "model_index.json"
        if not child.is_dir() or not index_path.is_file():
            continue
        metadata = read_json(index_path).get("_minimax_h3")
        if isinstance(metadata, dict) and str(metadata.get("partition", "")).lower() == "fl2va":
            matches.append(child.resolve())
    if len(matches) != 1:
        rendered = ", ".join(str(item) for item in matches) or "none"
        raise PreflightError(
            "expected exactly one FL2VA child below the supplied parent; "
            f"found: {rendered}"
        )
    return matches[0]


def component_dir(root: Path, index: dict[str, Any], name: str) -> Path:
    entry = index.get(name)
    relative: str = name
    if isinstance(entry, str) and entry:
        relative = entry
    elif isinstance(entry, dict):
        candidate = entry.get("path") or entry.get("subfolder")
        if isinstance(candidate, str) and candidate:
            relative = candidate
    elif not (isinstance(entry, list) and len(entry) == 2):
        raise PreflightError(f"model_index.json has no valid {name!r} entry")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"unsafe component path for {name}: {relative!r}") from exc
    if not path.is_dir():
        raise PreflightError(f"missing component directory for {name}: {path}")
    return path


def validate_release_metadata(index: dict[str, Any]) -> dict[str, Any]:
    metadata = index.get("_minimax_h3")
    if not isinstance(metadata, dict):
        raise PreflightError("model_index.json._minimax_h3 must be an object")
    if str(metadata.get("partition", "")).lower() != "fl2va":
        raise PreflightError(
            "T2VA needs partition='fl2va'; got "
            f"{metadata.get('partition')!r}"
        )
    tasks = metadata.get("tasks")
    if not isinstance(tasks, list) or "t2va" not in {
        str(item).strip().lower() for item in tasks
    }:
        raise PreflightError(f"FL2VA metadata does not advertise t2va: {tasks!r}")
    scales = metadata.get("sigma_shift_scales")
    if not isinstance(scales, dict) or scales.get("video") != 12.0 or scales.get("audio") != 3.0:
        raise PreflightError(
            "expected sigma_shift_scales={'video': 12.0, 'audio': 3.0}; "
            f"got {scales!r}"
        )
    return metadata


def validate_configs(paths: dict[str, Path]) -> list[str]:
    transformer = read_json(paths["transformer"] / "config.json")
    expected_transformer = {
        "_class_name": "MiniMaxH3DiTModel",
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
    }
    text_encoder = read_json(paths["text_encoder"] / "config.json")
    text_config = text_encoder.get("text_config")
    if not isinstance(text_config, dict):
        raise PreflightError("text_encoder/config.json has no text_config object")
    expected_text = {
        "hidden_size": 5120,
        "intermediate_size": 25600,
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
    }
    video = read_json(paths["video_vae"] / "config.json")
    video_source = read_json(paths["video_vae"] / "source" / "config.json")
    audio = read_json(paths["audio_vae"] / "config.json")

    mismatches: list[str] = []
    for name, actual, expected in (
        ("transformer", transformer, expected_transformer),
        ("text_encoder.text_config", text_config, expected_text),
        (
            "video_vae",
            video,
            {
                "_class_name": "MiniMaxH3VideoVAE",
                "latent_channels": 24,
                "source_path": "source",
                "source_class_name": "AutoencoderKLLegacy",
                "source_safetensors_path": "model.safetensors",
            },
        ),
        (
            "video_vae/source",
            video_source,
            {
                "_class_name": "AutoencoderKLLegacy",
                "z_channels": 24,
                "embed_dim": 24,
                "vae_ratio": 16,
                "vae_ratio_t": 4,
                "use_vit_decoder": True,
            },
        ),
        (
            "audio_vae",
            audio,
            {
                "_class_name": "MiniMaxH3AudioVAE",
                "latent_channels": 32,
                "sample_rate": 32000,
                "output_channel": 2,
                "source_safetensors_path": "model.safetensors",
            },
        ),
    ):
        for key, wanted in expected.items():
            if actual.get(key) != wanted:
                mismatches.append(
                    f"{name}.{key}={actual.get(key)!r} (expected {wanted!r})"
                )
    architectures = text_encoder.get("architectures")
    if architectures != ["Qwen3VLForConditionalGeneration"]:
        mismatches.append(
            "text_encoder.architectures="
            f"{architectures!r} (expected ['Qwen3VLForConditionalGeneration'])"
        )
    for name, config, count in (("video_vae", video, 24), ("audio_vae", audio, 32)):
        for field in ("latents_mean", "latents_std"):
            values = config.get(field)
            if not isinstance(values, list) or len(values) != count:
                mismatches.append(
                    f"{name}.{field} must contain {count} values; got "
                    f"{None if not isinstance(values, list) else len(values)}"
                )
    return mismatches


def indexed_files(index_path: Path) -> list[Path]:
    value = read_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PreflightError(f"invalid safetensors weight_map: {index_path}")
    files = sorted({index_path.parent / str(name) for name in weight_map.values()})
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise PreflightError(
            f"{index_path} references missing shards: {', '.join(missing[:12])}"
        )
    return files


def component_weight_files(path: Path, *, video: bool = False) -> list[Path]:
    search_root = path / "source" if video else path
    indexes = sorted(search_root.glob("*.safetensors.index.json"))
    if indexes:
        return indexed_files(indexes[0])
    files = sorted(search_root.glob("*.safetensors"))
    if not files:
        raise PreflightError(f"no safetensors weights found under {search_root}")
    return files


def safetensors_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise PreflightError(f"truncated safetensors file: {path}")
            (header_size,) = struct.unpack("<Q", prefix)
            if not 2 <= header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise PreflightError(
                    f"invalid safetensors header size {header_size}: {path}"
                )
            raw = handle.read(header_size)
            if len(raw) != header_size:
                raise PreflightError(f"truncated safetensors header: {path}")
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot parse safetensors header {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"invalid safetensors header object: {path}")
    payload_size = path.stat().st_size - 8 - header_size
    data_ends: list[int] = []
    for key, descriptor in value.items():
        if key == "__metadata__":
            continue
        if not isinstance(descriptor, dict):
            raise PreflightError(f"invalid tensor descriptor {key!r}: {path}")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in offsets
            )
            or not 0 <= offsets[0] <= offsets[1] <= payload_size
        ):
            raise PreflightError(
                f"invalid/truncated data_offsets for tensor {key!r}: {path}"
            )
        data_ends.append(offsets[1])
    if data_ends and max(data_ends) != payload_size:
        raise PreflightError(
            f"safetensors payload length does not match its header: {path}"
        )
    return value


def inspect_weights(files: Iterable[Path]) -> tuple[int, Counter[str], set[int], int]:
    tensor_count = 0
    dtypes: Counter[str] = Counter()
    qwen_layers: set[int] = set()
    total_bytes = 0
    seen: set[str] = set()
    for path in files:
        total_bytes += path.stat().st_size
        for key, descriptor in safetensors_header(path).items():
            if key == "__metadata__":
                continue
            if key in seen:
                raise PreflightError(f"duplicate tensor key across shards: {key}")
            seen.add(key)
            tensor_count += 1
            if isinstance(descriptor, dict):
                dtypes[str(descriptor.get("dtype", "?"))] += 1
            match = QWEN_LAYER_KEY.match(key)
            if match:
                qwen_layers.add(int(match.group(1)))
    return tensor_count, dtypes, qwen_layers, total_bytes


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_root", type=Path)
    args = parser.parse_args(argv)
    try:
        root = resolve_release_root(args.model_root)
        index = read_json(root / "model_index.json")
        metadata = validate_release_metadata(index)
        names = ("transformer", "text_encoder", "tokenizer", "video_vae", "audio_vae")
        paths = {name: component_dir(root, index, name) for name in names}
        mismatches = validate_configs(paths)
        if mismatches:
            raise PreflightError("config mismatch:\n  - " + "\n  - ".join(mismatches))

        weights = {
            "transformer": component_weight_files(paths["transformer"]),
            "text_encoder": component_weight_files(paths["text_encoder"]),
            "video_vae": component_weight_files(paths["video_vae"], video=True),
            "audio_vae": component_weight_files(paths["audio_vae"]),
        }
        print(f"[OK] release root: {root}")
        print(f"[OK] partition/tasks: {metadata['partition']} / {metadata['tasks']}")
        print(f"[OK] sigma shifts: {metadata['sigma_shift_scales']}")
        for name, files in weights.items():
            tensor_count, dtypes, qwen_layers, total_bytes = inspect_weights(files)
            suffix = ""
            if name == "text_encoder":
                if qwen_layers != set(range(64)):
                    missing = sorted(set(range(64)) - qwen_layers)
                    raise PreflightError(
                        "Qwen checkpoint does not expose language layers 0..63; "
                        f"missing={missing[:12]}"
                    )
                suffix = "; Qwen layers=0..63 (runtime retains 0..49)"
            dtype_text = ", ".join(f"{key}:{value}" for key, value in sorted(dtypes.items()))
            print(
                f"[OK] {name}: {len(files)} file(s), {tensor_count} tensors, "
                f"{format_gib(total_bytes)}, dtypes={{{dtype_text}}}{suffix}"
            )
        print("[PASS] release structure and safetensors headers are compatible")
        return 0
    except PreflightError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
