"""examples/workflows/ 下的前端工作流必须与节点 INPUT_TYPES 保持一致。

工作流由 tools/gen_example_workflows.py 从节点定义生成；本测试反向校验，
确保改了节点签名却忘了重跑生成器时能立刻发现。
"""
import json
import unittest
from pathlib import Path

WORKFLOW_DIR = Path(__file__).resolve().parents[1] / "examples" / "workflows"
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}
# 核心 ComfyUI 节点不在本插件里，只校验它们参与的连线类型。
CORE_TYPES = {
    "LoadImage",
    "LoadAudio",
    "LoadVideo",
    "CreateVideo",
    "SaveVideo",
}
EXPECTED = {
    "t2va",
    "fl2va_first_frame",
    "fl2va_last_frame",
    "fl2va_first_last_frame",
    "ref2va_image",
    "ref2va_image_audio",
    "ref2va_video_audio",
}


def _derive(node_type: str):
    """Return (connection slots, required slot names, widget count)."""

    from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

    schema = NODE_CLASS_MAPPINGS[node_type].INPUT_TYPES()
    required = schema.get("required", {})
    slots, required_slots, widgets = [], set(), 0
    for name, spec in list(required.items()) + list(schema.get("optional", {}).items()):
        kind = spec[0]
        if isinstance(kind, list) or kind in WIDGET_TYPES:
            widgets += 2 if name == "seed" else 1
        else:
            slots.append((name, kind))
            if name in required:
                required_slots.add(name)
    return slots, required_slots, widgets


def _workflows():
    for path in sorted(WORKFLOW_DIR.glob("*.json")):
        yield path.stem, json.loads(path.read_text(encoding="utf-8"))


class ExampleWorkflowTests(unittest.TestCase):
    def test_every_task_type_and_variant_is_covered(self):
        self.assertEqual({name for name, _ in _workflows()}, EXPECTED)

    def test_h3_nodes_match_their_input_types(self):
        from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

        for name, workflow in _workflows():
            for node in workflow["nodes"]:
                node_type = node["type"]
                if node_type in CORE_TYPES:
                    continue
                with self.subTest(workflow=name, node=node_type):
                    self.assertIn(node_type, NODE_CLASS_MAPPINGS)
                    slots, _required, widgets = _derive(node_type)
                    self.assertEqual(
                        [(s["name"], s["type"]) for s in node["inputs"]], slots
                    )
                    self.assertEqual(len(node["widgets_values"]), widgets)
                    self.assertEqual(
                        node["properties"]["Node name for S&R"], node_type
                    )

    def test_widget_values_are_admissible_for_combo_inputs(self):
        from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

        for name, workflow in _workflows():
            for node in workflow["nodes"]:
                if node["type"] in CORE_TYPES:
                    continue
                schema = NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES()
                entries = list(schema.get("required", {}).items())
                entries += list(schema.get("optional", {}).items())
                index = 0
                for field, spec in entries:
                    kind = spec[0]
                    if not (isinstance(kind, list) or kind in WIDGET_TYPES):
                        continue
                    value = node["widgets_values"][index]
                    index += 1 + (1 if field == "seed" else 0)
                    if isinstance(kind, list):
                        with self.subTest(workflow=name, node=node["type"], field=field):
                            self.assertIn(value, kind)

    def test_every_required_connection_is_wired(self):
        for name, workflow in _workflows():
            for node in workflow["nodes"]:
                if node["type"] in CORE_TYPES:
                    continue
                _slots, required_slots, _widgets = _derive(node["type"])
                for slot in node["inputs"]:
                    if slot["name"] not in required_slots:
                        continue
                    with self.subTest(
                        workflow=name, node=node["type"], slot=slot["name"]
                    ):
                        self.assertIsNotNone(slot["link"])

    def test_links_are_consistent_and_type_matched(self):
        for name, workflow in _workflows():
            by_id = {node["id"]: node for node in workflow["nodes"]}
            seen = set()
            for link in workflow["links"]:
                link_id, src_id, src_slot, dst_id, dst_slot, kind = link
                with self.subTest(workflow=name, link=link_id):
                    self.assertNotIn(link_id, seen)
                    seen.add(link_id)
                    source = by_id[src_id]["outputs"][src_slot]
                    target = by_id[dst_id]["inputs"][dst_slot]
                    self.assertEqual(source["type"], kind)
                    self.assertEqual(target["type"], kind)
                    self.assertIn(link_id, source["links"])
                    self.assertEqual(target["link"], link_id)
            self.assertEqual(workflow["last_link_id"], len(workflow["links"]))
            self.assertEqual(workflow["last_node_id"], len(workflow["nodes"]))

    def test_每个工作流都以保存节点收尾(self):
        for name, workflow in _workflows():
            types = {node["type"] for node in workflow["nodes"]}
            with self.subTest(workflow=name):
                self.assertIn("SaveVideo", types)
                self.assertIn("RHMiniMaxH3DecodeAV", types)
                self.assertIn("RHMiniMaxH3DualSigmaSampler", types)

    def test_loader_family_matches_the_task(self):
        for name, workflow in _workflows():
            types = {node["type"] for node in workflow["nodes"]}
            family = {
                "t2va": "RHMiniMaxH3Direct",
                "fl2va": "RHMiniMaxH3FL2VA",
                "ref2va": "RHMiniMaxH3Ref2VA",
            }[name.split("_")[0]]
            with self.subTest(workflow=name):
                for component in ("ModelLoader", "TextEncoderLoader", "VAELoader"):
                    self.assertIn(f"{family}{component}", types)

    def _source_of(self, workflow, node, slot_name):
        """Return the id of the node feeding ``slot_name``."""

        link_id = next(
            slot["link"] for slot in node["inputs"] if slot["name"] == slot_name
        )
        return next(
            link[1] for link in workflow["links"] if link[0] == link_id
        )

    def test_target_and_conditioning_share_one_source(self):
        """Target/Encode/EmptyAVLatent 必须接同一个 target 与同一份条件素材。

        接成两个来源在 contracts 层才会报错，属于典型的图连错；这里在示例
        工作流上先钉死。
        """

        shared = {
            "RHMiniMaxH3FL2VAEncode": "keyframes",
            "RHMiniMaxH3Ref2VAEncode": "references",
        }
        for name, workflow in _workflows():
            by_type = {node["type"]: node for node in workflow["nodes"]}
            latent = by_type["RHMiniMaxH3EmptyAVLatent"]
            target_node = next(
                node for node in workflow["nodes"] if node["type"].endswith("Target")
            )
            with self.subTest(workflow=name):
                self.assertEqual(
                    self._source_of(workflow, latent, "target"), target_node["id"]
                )
                for encode_type, field in shared.items():
                    encode = by_type.get(encode_type)
                    if encode is None:
                        continue
                    self.assertEqual(
                        self._source_of(workflow, encode, "target"),
                        target_node["id"],
                    )
                    self.assertEqual(
                        self._source_of(workflow, encode, field),
                        self._source_of(workflow, target_node, field),
                    )

    def test_vae_bundle_is_shared_between_encode_and_decode(self):
        for name, workflow in _workflows():
            by_type = {node["type"]: node for node in workflow["nodes"]}
            decode = by_type["RHMiniMaxH3DecodeAV"]
            encode = next(
                (
                    node
                    for key, node in by_type.items()
                    if key.endswith("Encode")
                    and any(s["name"] == "h3_vae_bundle" for s in node["inputs"])
                ),
                None,
            )
            if encode is None:  # T2VA 的文本编码不吃 VAE
                continue
            with self.subTest(workflow=name):
                self.assertEqual(
                    self._source_of(workflow, decode, "h3_vae_bundle"),
                    self._source_of(workflow, encode, "h3_vae_bundle"),
                )

    def test_execution_order_is_topological(self):
        for name, workflow in _workflows():
            order = {node["id"]: node["order"] for node in workflow["nodes"]}
            for _link_id, src, _src_slot, dst, _dst_slot, _kind in workflow["links"]:
                with self.subTest(workflow=name, edge=(src, dst)):
                    self.assertLess(order[src], order[dst])

    def test_ref2va_reference_chain_is_ordered(self):
        """参考链必须串联：除链首外每个参考节点都要接上一个的 references。"""

        for name, workflow in _workflows():
            if not name.startswith("ref2va"):
                continue
            refs = [
                node
                for node in workflow["nodes"]
                if node["type"].startswith("RHMiniMaxH3Ref2VA")
                and node["type"].endswith("Reference")
            ]
            with self.subTest(workflow=name):
                self.assertTrue(refs)
                unchained = [
                    node
                    for node in refs
                    if all(
                        slot["link"] is None
                        for slot in node["inputs"]
                        if slot["name"] == "references"
                    )
                ]
                self.assertEqual(len(unchained), 1)


if __name__ == "__main__":
    unittest.main()


class WorkflowMigrationTests(unittest.TestCase):
    """tools/migrate_workflow.py 必须把旧工作流修到能通过上面同一套校验。"""

    LEGACY_UI = {
        "nodes": [
            {
                "id": 1,
                "type": "MiniMaxH3DirectVAELoader",
                "properties": {"Node name for S&R": "MiniMaxH3DirectVAELoader"},
                "widgets_values": ["MiniMax-H3", "vae"],
                "inputs": [],
                "outputs": [],
            },
            {
                "id": 2,
                "type": "MiniMaxH3DualSigmaSampler",
                "properties": {"Node name for S&R": "MiniMaxH3DualSigmaSampler"},
                # 老签名：只到 audio_shift，accel 之后的 widget 都还没有
                "widgets_values": [42, "fixed", 2, 12.0, 3.0],
                "inputs": [],
                "outputs": [],
            },
        ],
        "links": [],
    }
    LEGACY_API = {
        "prompt": {
            "7": {
                "class_type": "MiniMaxH3DirectVAELoader",
                "inputs": {"model_root": "MiniMax-H3", "vae_path": "MiniMax-H3-vae"},
            }
        }
    }

    def _migrate(self, document):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "tools" / "migrate_workflow.py"
        spec = importlib.util.spec_from_file_location("migrate_workflow", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report: list[str] = []
        return module.migrate(json.loads(json.dumps(document)), report)[0], report

    def test_ui_graph_gains_new_ids_and_split_vae_widgets(self):
        from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

        migrated, _report = self._migrate(self.LEGACY_UI)
        for node in migrated["nodes"]:
            with self.subTest(node=node["type"]):
                self.assertIn(node["type"], NODE_CLASS_MAPPINGS)
                self.assertTrue(node["type"].startswith("RHMiniMaxH3"))
                self.assertEqual(
                    node["properties"]["Node name for S&R"], node["type"]
                )
                _slots, _required, widgets = _derive(node["type"])
                self.assertEqual(len(node["widgets_values"]), widgets)
        vae = migrated["nodes"][0]
        # 单个 vae_path 变成 video + audio 两个值
        self.assertEqual(len(vae["widgets_values"]), 3)
        self.assertNotIn("vae", vae["widgets_values"])

    def test_migrated_combo_values_are_selectable(self):
        from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS

        migrated, _report = self._migrate(self.LEGACY_UI)
        for node in migrated["nodes"]:
            schema = NODE_CLASS_MAPPINGS[node["type"]].INPUT_TYPES()
            entries = list(schema.get("required", {}).items())
            entries += list(schema.get("optional", {}).items())
            index = 0
            for field, spec in entries:
                kind = spec[0]
                if not (isinstance(kind, list) or kind in WIDGET_TYPES):
                    continue
                value = node["widgets_values"][index]
                index += 1 + (1 if field == "seed" else 0)
                if isinstance(kind, list):
                    with self.subTest(node=node["type"], field=field):
                        self.assertIn(value, kind)

    def test_api_prompt_splits_vae_path(self):
        migrated, _report = self._migrate(self.LEGACY_API)
        node = migrated["prompt"]["7"]
        self.assertEqual(node["class_type"], "RHMiniMaxH3DirectVAELoader")
        self.assertNotIn("vae_path", node["inputs"])
        self.assertIn("video_vae_path", node["inputs"])
        self.assertIn("audio_vae_path", node["inputs"])

    def test_every_substitution_is_reported(self):
        _migrated, report = self._migrate(self.LEGACY_UI)
        joined = "\n".join(report)
        self.assertIn("vae_path", joined)
        self.assertIn("accel", joined)
