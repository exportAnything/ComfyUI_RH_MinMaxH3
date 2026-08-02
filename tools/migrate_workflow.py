#!/usr/bin/env python3
"""把旧的 MiniMax-H3 工作流迁移到当前节点定义。

处理三类失效原因：

1. 节点 ID 加了 ``RH`` 前缀（``MiniMaxH3DecodeAV`` → ``RHMiniMaxH3DecodeAV``），
   旧工作流打开时报 ``Node type not found``；
2. 双 VAE loader 的单个 ``vae_path`` 拆成了 ``video_vae_path`` + ``audio_vae_path``；
3. 节点后来增删过 widget，旧工作流的 ``widgets_values`` 长度对不上。

COMBO 选项若在当前下拉里不存在（例如换了权重布局后的旧模型名），会替换成当前
默认值并逐条报告——不静默留一个选不中的值。

前端格式（``nodes`` + ``links``）与 API 格式（``{id: {class_type, inputs}}``，
可含 ``prompt`` 包装）都支持。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
LEGACY_PREFIX = "MiniMaxH3"
NEW_PREFIX = "RHMiniMaxH3"


def _mappings():
    from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

    return NODE_CLASS_MAPPINGS


def _new_type(node_type: str) -> str | None:
    """Return the current node ID for a possibly-legacy type, or None."""

    mappings = _mappings()
    if node_type in mappings:
        return node_type
    if node_type.startswith(LEGACY_PREFIX):
        candidate = NEW_PREFIX + node_type[len(LEGACY_PREFIX):]
        if candidate in mappings:
            return candidate
    return None


def _widget_fields(node_type: str) -> list[tuple[str, object, object]]:
    """[(field, choices-or-type, default)] in ComfyUI widget order."""

    schema = _mappings()[node_type].INPUT_TYPES()
    entries = list(schema.get("required", {}).items())
    entries += list(schema.get("optional", {}).items())
    out: list[tuple[str, object, object]] = []
    for name, spec in entries:
        kind = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if not (isinstance(kind, list) or kind in WIDGET_TYPES):
            continue
        if "default" in options:
            default = options["default"]
        elif isinstance(kind, list):
            default = kind[0] if kind else ""
        else:
            default = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}[kind]
        out.append((name, kind, default))
        if name == "seed":
            out.append(("control_after_generate", ["fixed"], "fixed"))
    return out


def _vae_defaults(node_type: str) -> tuple[object, object]:
    fields = {name: default for name, _kind, default in _widget_fields(node_type)}
    return fields.get("video_vae_path"), fields.get("audio_vae_path")


def _migrate_widgets(node_type, values, report, label):
    """Rebuild widgets_values against the current signature, keeping what fits."""

    fields = _widget_fields(node_type)
    values = list(values or [])
    is_vae = any(name == "video_vae_path" for name, _k, _d in fields)
    if is_vae and len(values) == 2:
        # 旧签名 [model_root, vae_path] → [model_root, video, audio]
        video, audio = _vae_defaults(node_type)
        report.append(
            f"{label}: vae_path={values[1]!r} 拆分为 "
            f"video_vae_path={video!r} + audio_vae_path={audio!r}"
        )
        values = [values[0], video, audio]

    out = []
    for index, (name, kind, default) in enumerate(fields):
        if index < len(values):
            value = values[index]
            if isinstance(kind, list) and value not in kind:
                report.append(
                    f"{label}.{name}: {value!r} 不在当前选项中，改为默认 {default!r}"
                )
                value = default
        else:
            value = default
            report.append(f"{label}.{name}: 缺失，补默认 {default!r}")
        out.append(value)
    if len(values) > len(fields):
        report.append(
            f"{label}: 丢弃 {len(values) - len(fields)} 个多余 widget 值"
        )
    return out


def migrate(document: dict, report: list[str]) -> tuple[dict, int]:
    """Migrate a UI graph or an API prompt in place; return (doc, touched)."""

    touched = 0
    nodes = document.get("nodes")
    if isinstance(nodes, list):  # 前端格式
        for node in nodes:
            old_type = node.get("type")
            new_type = _new_type(old_type) if isinstance(old_type, str) else None
            if new_type is None:
                continue
            if new_type != old_type:
                report.append(f"节点 {node.get('id')}: {old_type} → {new_type}")
            node["type"] = new_type
            properties = node.setdefault("properties", {})
            if "Node name for S&R" in properties:
                properties["Node name for S&R"] = new_type
            node["widgets_values"] = _migrate_widgets(
                new_type,
                node.get("widgets_values"),
                report,
                f"节点 {node.get('id')} {new_type}",
            )
            for slot in node.get("inputs", []) or []:
                widget = slot.get("widget")
                if isinstance(widget, dict) and widget.get("name") == "vae_path":
                    widget["name"] = "video_vae_path"
                    slot["name"] = "video_vae_path"
            touched += 1
        return document, touched

    prompt = document.get("prompt") if isinstance(document.get("prompt"), dict) else document
    for node in prompt.values():  # API 格式
        if not isinstance(node, dict):
            continue
        old_type = node.get("class_type")
        new_type = _new_type(old_type) if isinstance(old_type, str) else None
        if new_type is None:
            continue
        if new_type != old_type:
            report.append(f"{old_type} → {new_type}")
        node["class_type"] = new_type
        inputs = node.get("inputs")
        if isinstance(inputs, dict) and "vae_path" in inputs:
            video, audio = _vae_defaults(new_type)
            report.append(
                f"{new_type}: vae_path={inputs['vae_path']!r} 拆分为 "
                f"video_vae_path={video!r} + audio_vae_path={audio!r}"
            )
            inputs.pop("vae_path")
            inputs["video_vae_path"], inputs["audio_vae_path"] = video, audio
        touched += 1
    return document, touched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="工作流 JSON（前端或 API 格式）")
    parser.add_argument(
        "--in-place", action="store_true", help="就地覆盖，并留 .bak 备份"
    )
    parser.add_argument("--out-dir", help="输出目录（默认写到 <name>.migrated.json）")
    args = parser.parse_args()

    failed = 0
    for raw in args.paths:
        path = Path(raw).expanduser()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"SKIP {path}: 无法解析（{exc}）")
            failed += 1
            continue
        report: list[str] = []
        document, touched = migrate(document, report)
        if not touched:
            print(f"SKIP {path}: 没有 MiniMax-H3 节点")
            continue
        if args.in_place:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
            target = path
        elif args.out_dir:
            target = Path(args.out_dir).expanduser() / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = path.with_suffix(".migrated.json")
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"OK   {path} → {target}  （{touched} 个 H3 节点）")
        for line in report:
            print(f"       - {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
