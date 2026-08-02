#!/usr/bin/env python3
"""按节点 INPUT_TYPES 生成全部任务类型的 ComfyUI 前端工作流（非 API）。

widget 顺序、默认值与输入插槽全部从节点定义推导，不手写：改了 INPUT_TYPES
重跑本脚本即可，工作流不会和代码漂移。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
WIDGET_FALLBACK = {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}

# 核心 ComfyUI 节点无法在插件进程内自省，按已验证工作流里的形状写死。
CORE_NODES = {
    "LoadImage": {
        "inputs": [],
        "outputs": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        "widgets": ["example.png", "image"],
        "size": [320, 314],
    },
    "LoadAudio": {
        "inputs": [],
        "outputs": [("AUDIO", "AUDIO")],
        "widgets": ["example.mp3"],
        "size": [320, 100],
    },
    "LoadVideo": {
        "inputs": [],
        "outputs": [("VIDEO", "VIDEO")],
        "widgets": ["example.mp4"],
        "size": [320, 100],
    },
    "CreateVideo": {
        "inputs": [("images", "IMAGE"), ("audio", "AUDIO")],
        "widget_inputs": [("fps", "FLOAT"), ("bit_depth", "INT")],
        "outputs": [("VIDEO", "VIDEO")],
        "widgets": [24, 8],
        "size": [300, 100],
    },
    "SaveVideo": {
        "inputs": [("video", "VIDEO")],
        "outputs": [],
        "widgets": ["minimax_h3/output", "mp4", "h264"],
        "size": [400, 130],
    },
}

PROMPTS = {
    "t2va": (
        "A five-second cinematic 16:9 shot of a small paper boat gliding through "
        "a rain-lit city gutter at night. Reflections ripple across the water, a "
        "distant train passes, and the ambient rain and wheel noise stay "
        "synchronized with the movement. Natural camera motion, no subtitles, no "
        "logos."
    ),
    "fl2va": (
        "Continue naturally from the provided keyframe: a slow handheld push-in "
        "with soft ambient room tone and synchronized footsteps. Keep the subject "
        "identity, lighting and colour grade stable, no subtitles, no logos."
    ),
    "ref2va": (
        "Follow the supplied references in order: keep the referenced subject's "
        "identity and wardrobe, match the referenced audio's rhythm, and keep "
        "lip movement synchronized. Natural camera motion, no subtitles, no logos."
    ),
}


def _node_schema(node_type: str):
    from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

    cls = NODE_CLASS_MAPPINGS[node_type]
    schema = cls.INPUT_TYPES()
    entries = list(schema.get("required", {}).items())
    entries += list(schema.get("optional", {}).items())
    slots, widgets = [], []
    for name, spec in entries:
        kind = spec[0]
        options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        if isinstance(kind, list) or kind in WIDGET_TYPES:
            if "default" in options:
                value = options["default"]
            elif isinstance(kind, list):
                value = kind[0] if kind else ""
            else:
                value = WIDGET_FALLBACK[kind]
            widgets.append((name, value))
            if name == "seed":
                # ComfyUI 给 seed 追加一个 control_after_generate widget
                widgets.append(("control_after_generate", "fixed"))
        else:
            slots.append((name, kind))
    names = getattr(cls, "RETURN_NAMES", None) or getattr(cls, "RETURN_TYPES", ())
    outputs = list(zip(names, getattr(cls, "RETURN_TYPES", ())))
    return slots, widgets, outputs


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self.links: list[list] = []
        self._next_node = 1
        self._next_link = 1

    def add(self, key, node_type, *, pos, title=None, overrides=None, size=None):
        core = CORE_NODES.get(node_type)
        if core is not None:
            slots = list(core["inputs"])
            widget_slots = list(core.get("widget_inputs", []))
            widgets = [(f"w{i}", v) for i, v in enumerate(core["widgets"])]
            outputs = list(core["outputs"])
            size = size or core["size"]
        else:
            slots, widgets, outputs = _node_schema(node_type)
            widget_slots = []
            size = size or [380, 100 + 26 * len(widgets)]
        values = dict(overrides or {})
        rendered = []
        for name, value in widgets:
            rendered.append(values.pop(name, value))
        if values:
            raise KeyError(f"{node_type} 没有这些 widget：{sorted(values)}")
        node = {
            "id": self._next_node,
            "type": node_type,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": self._next_node - 1,
            "mode": 0,
            "inputs": [
                {"name": name, "type": kind, "link": None} for name, kind in slots
            ]
            + [
                {
                    "name": name,
                    "type": kind,
                    "link": None,
                    "widget": {"name": name},
                }
                for name, kind in widget_slots
            ],
            "outputs": [
                {"name": name, "type": kind, "links": [], "slot_index": index}
                for index, (name, kind) in enumerate(outputs)
            ],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": rendered,
        }
        if title:
            node["title"] = title
        self.nodes[key] = node
        self._next_node += 1
        return node

    def link(self, from_key, out_name, to_key, in_name):
        src, dst = self.nodes[from_key], self.nodes[to_key]
        out_index = next(
            i for i, o in enumerate(src["outputs"]) if o["name"] == out_name
        )
        in_index = next(
            i for i, s in enumerate(dst["inputs"]) if s["name"] == in_name
        )
        link_id = self._next_link
        self._next_link += 1
        kind = src["outputs"][out_index]["type"]
        src["outputs"][out_index]["links"].append(link_id)
        dst["inputs"][in_index]["link"] = link_id
        self.links.append(
            [link_id, src["id"], out_index, dst["id"], in_index, kind]
        )

    def _assign_execution_order(self) -> None:
        """按依赖拓扑排序写 order；节点添加顺序不必等于执行顺序。"""

        incoming = {node["id"]: set() for node in self.nodes.values()}
        outgoing: dict[int, set[int]] = {node["id"]: set() for node in self.nodes.values()}
        for _link_id, src, _src_slot, dst, _dst_slot, _kind in self.links:
            incoming[dst].add(src)
            outgoing[src].add(dst)
        ready = sorted(node for node, deps in incoming.items() if not deps)
        order, placed = [], set()
        while ready:
            current = ready.pop(0)
            order.append(current)
            placed.add(current)
            for target in sorted(outgoing[current]):
                if incoming[target] <= placed and target not in placed:
                    if target not in ready:
                        ready.append(target)
            ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("工作流存在环或孤立依赖，无法确定执行顺序")
        for index, node_id in enumerate(order):
            node = next(n for n in self.nodes.values() if n["id"] == node_id)
            node["order"] = index

    def dump(self, note: str) -> dict:
        self._assign_execution_order()
        return {
            "last_node_id": self._next_node - 1,
            "last_link_id": self._next_link - 1,
            "nodes": list(self.nodes.values()),
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"ds": {"scale": 0.7, "offset": [20, 30]}, "workflow_note": note},
            "version": 0.4,
        }


def _tail(graph: Graph, *, prefix: str, column: int) -> None:
    """采样 → 解码 → 出片；三条任务链共用。"""

    x = column
    graph.add("latent", "RHMiniMaxH3EmptyAVLatent", pos=[x, 40], title="Empty AV Latent")
    graph.add(
        "sampler",
        "RHMiniMaxH3DualSigmaSampler",
        pos=[x, 180],
        title="Dual Sigma Sampler",
    )
    graph.add("decode", "RHMiniMaxH3DecodeAV", pos=[x + 420, 40], title="Decode Video + Audio")
    graph.add("create", "CreateVideo", pos=[x + 420, 200])
    graph.add(
        "save",
        "SaveVideo",
        pos=[x + 420, 360],
        overrides={"w0": f"minimax_h3/{prefix}"},
    )
    graph.link("latent", "av_latent", "sampler", "av_latent")
    graph.link("sampler", "sampled_av_latent", "decode", "sampled_av_latent")
    graph.link("decode", "frames", "create", "images")
    graph.link("decode", "audio", "create", "audio")
    graph.link("create", "VIDEO", "save", "video")


def _loaders(graph: Graph, family: str, *, column: int = 40) -> None:
    graph.add(
        "te",
        f"RHMiniMaxH3{family}TextEncoderLoader",
        pos=[column, 40],
        title=f"{family} Qwen3-VL Loader",
    )
    graph.add(
        "model",
        f"RHMiniMaxH3{family}ModelLoader",
        pos=[column, 240],
        title=f"{family} DiT Loader",
    )
    graph.add(
        "vae",
        f"RHMiniMaxH3{family}VAELoader",
        pos=[column, 440],
        title=f"{family} Dual VAE Loader",
    )


TARGET = {"aspect_ratio": "16:9", "duration_seconds": 5.0, "width": 832, "height": 480}


def build_t2va() -> dict:
    graph = Graph()
    _loaders(graph, "Direct")
    graph.add("target", "RHMiniMaxH3T2VATarget", pos=[460, 40], overrides=dict(TARGET))
    graph.add(
        "encode",
        "RHMiniMaxH3T2VATextEncode",
        pos=[460, 240],
        overrides={"prompt": PROMPTS["t2va"]},
        size=[420, 220],
    )
    _tail(graph, prefix="t2va", column=920)
    graph.link("te", "h3_text_encoder", "encode", "h3_text_encoder")
    graph.link("target", "target", "latent", "target")
    graph.link("model", "h3_model", "sampler", "h3_model")
    graph.link("encode", "conditioning", "sampler", "conditioning")
    graph.link("vae", "h3_vae_bundle", "decode", "h3_vae_bundle")
    return graph.dump(
        "T2VA：纯文本生成视频+音频。三个 Direct Loader 的模型名请按本地实际选择；"
        "Target 已显式 832×480/5s，留空 width/height 会按比例解析到 1344×768。"
    )


def build_fl2va(variant: str) -> dict:
    graph = Graph()
    _loaders(graph, "FL2VA")
    if variant == "last_frame":
        graph.add("last_image", "LoadImage", pos=[460, 40], title="Last frame")
        graph.add(
            "keyframes",
            "RHMiniMaxH3FL2VALastFrameCondition",
            pos=[820, 40],
            title="Last Only",
        )
        graph.link("last_image", "IMAGE", "keyframes", "last_frame")
    else:
        graph.add("first_image", "LoadImage", pos=[460, 40], title="First frame")
        graph.add(
            "keyframes",
            "RHMiniMaxH3FL2VAFirstFrameCondition",
            pos=[820, 40],
            title="First / First+Last",
        )
        graph.link("first_image", "IMAGE", "keyframes", "first_frame")
        if variant == "first_last_frame":
            graph.add("last_image", "LoadImage", pos=[460, 400], title="Last frame")
            graph.link("last_image", "IMAGE", "keyframes", "last_frame")

    graph.add("target", "RHMiniMaxH3FL2VATarget", pos=[1180, 40], overrides=dict(TARGET))
    graph.add(
        "encode",
        "RHMiniMaxH3FL2VAEncode",
        pos=[1180, 260],
        overrides={"prompt": PROMPTS["fl2va"]},
        size=[420, 260],
    )
    _tail(graph, prefix=f"fl2va_{variant}", column=1640)
    graph.link("keyframes", "keyframes", "target", "keyframes")
    graph.link("keyframes", "keyframes", "encode", "keyframes")
    graph.link("te", "h3_text_encoder", "encode", "h3_text_encoder")
    graph.link("vae", "h3_vae_bundle", "encode", "h3_vae_bundle")
    graph.link("target", "target", "encode", "target")
    graph.link("target", "target", "latent", "target")
    graph.link("model", "h3_model", "sampler", "h3_model")
    graph.link("encode", "conditioning", "sampler", "conditioning")
    graph.link("vae", "h3_vae_bundle", "decode", "h3_vae_bundle")
    labels = {
        "first_frame": "首帧",
        "last_frame": "尾帧",
        "first_last_frame": "首帧+尾帧",
    }
    return graph.dump(
        f"FL2VA（{labels[variant]}）：三种合法条件签名之一。运行前把 LoadImage 换成"
        "已上传到 ComfyUI input/ 的真实图片；Target 与 Encode 必须接同一个 keyframes。"
    )


def build_ref2va(variant: str) -> dict:
    graph = Graph()
    _loaders(graph, "Ref2VA")
    chain: list[str] = []
    if variant == "video_audio":
        graph.add("video", "LoadVideo", pos=[460, 40], title="Reference video")
        graph.add(
            "ref_video",
            "RHMiniMaxH3Ref2VAVideoReference",
            pos=[820, 40],
            overrides={"reference_type": "video_audio"},
            title="Video Reference (with audio)",
        )
        graph.link("video", "VIDEO", "ref_video", "video")
        chain.append("ref_video")
    else:
        graph.add("image", "LoadImage", pos=[460, 40], title="Reference image")
        graph.add(
            "ref_image", "RHMiniMaxH3Ref2VAImageReference", pos=[820, 40],
            title="Image Reference",
        )
        graph.link("image", "IMAGE", "ref_image", "image")
        chain.append("ref_image")
        if variant == "image_audio":
            graph.add("audio", "LoadAudio", pos=[460, 400], title="Reference audio")
            graph.add(
                "ref_audio", "RHMiniMaxH3Ref2VAAudioReference", pos=[820, 400],
                title="Audio Reference",
            )
            graph.link("audio", "AUDIO", "ref_audio", "audio")
            chain.append("ref_audio")

    # 参考链有序：上一个节点的 references 必须接进下一个参考节点。
    for previous, current in zip(chain, chain[1:]):
        graph.link(previous, "references", current, "references")
    last = chain[-1]

    graph.add("target", "RHMiniMaxH3Ref2VATarget", pos=[1180, 40], overrides=dict(TARGET))
    graph.add(
        "encode",
        "RHMiniMaxH3Ref2VAEncode",
        pos=[1180, 280],
        overrides={"prompt": PROMPTS["ref2va"]},
        size=[420, 280],
    )
    _tail(graph, prefix=f"ref2va_{variant}", column=1640)
    graph.link(last, "references", "target", "references")
    graph.link(last, "references", "encode", "references")
    graph.link("te", "h3_text_encoder", "encode", "h3_text_encoder")
    graph.link("vae", "h3_vae_bundle", "encode", "h3_vae_bundle")
    graph.link("target", "target", "encode", "target")
    graph.link("target", "target", "latent", "target")
    graph.link("model", "h3_model", "sampler", "h3_model")
    graph.link("encode", "conditioning", "sampler", "conditioning")
    graph.link("vae", "h3_vae_bundle", "decode", "h3_vae_bundle")
    labels = {
        "image": "单图参考",
        "image_audio": "图片 + 音频参考",
        "video_audio": "带音轨视频参考",
    }
    return graph.dump(
        f"Ref2VA（{labels[variant]}）：参考素材严格有序，改动链路顺序会改变多模态"
        "提示与条件行顺序。运行前把 Load* 换成已上传到 ComfyUI input/ 的真实素材。"
    )


WORKFLOWS = {
    "t2va": build_t2va,
    "fl2va_first_frame": lambda: build_fl2va("first_frame"),
    "fl2va_last_frame": lambda: build_fl2va("last_frame"),
    "fl2va_first_last_frame": lambda: build_fl2va("first_last_frame"),
    "ref2va_image": lambda: build_ref2va("image"),
    "ref2va_image_audio": lambda: build_ref2va("image_audio"),
    "ref2va_video_audio": lambda: build_ref2va("video_audio"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=str(ROOT / "examples" / "workflows"),
        help="输出目录，默认 examples/workflows",
    )
    args = parser.parse_args()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name, builder in WORKFLOWS.items():
        path = out / f"{name}.json"
        path.write_text(
            json.dumps(builder(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {path}")


if __name__ == "__main__":
    main()
