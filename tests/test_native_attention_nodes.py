from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock

import torch

from minimax_h3_nodes.api.attention import (
    MiniMaxH3SageAttentionPatch,
    MiniMaxH3SolAttentionPatch,
)
from minimax_h3_nodes.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from minimax_h3_nodes.runtime import attention_backends as backends


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
C0_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f]")


class _FakeSampling:
    @staticmethod
    def percent_to_sigma(percent):
        return 1.0 - float(percent)


class _FakeModel:
    def __init__(self, model_options=None):
        self.model_options = copy.deepcopy(
            model_options or {"transformer_options": {}}
        )

    def clone(self):
        return _FakeModel(self.model_options)

    @staticmethod
    def get_model_object(name):
        if name == "model_sampling":
            return _FakeSampling()
        raise KeyError(name)


def _flatten_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_strings(key)
            yield from _flatten_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_strings(item)


class AttentionNodeContractTests(unittest.TestCase):
    def test_registry_uses_unique_fork_owned_node_ids(self):
        self.assertIs(
            NODE_CLASS_MAPPINGS["RHMiniMaxH3SageAttentionPatch"],
            MiniMaxH3SageAttentionPatch,
        )
        self.assertIs(
            NODE_CLASS_MAPPINGS["RHMiniMaxH3SolAttentionPatch"],
            MiniMaxH3SolAttentionPatch,
        )
        self.assertNotIn("StandaloneSageAttentionPatch", NODE_CLASS_MAPPINGS)
        self.assertNotIn("SolAttnMiniMaxH3Patcher", NODE_CLASS_MAPPINGS)

    def test_all_attention_node_ui_text_is_english(self):
        values = [
            MiniMaxH3SageAttentionPatch.INPUT_TYPES(),
            MiniMaxH3SageAttentionPatch.DESCRIPTION,
            MiniMaxH3SolAttentionPatch.INPUT_TYPES(),
            MiniMaxH3SolAttentionPatch.DESCRIPTION,
            NODE_DISPLAY_NAME_MAPPINGS["RHMiniMaxH3SageAttentionPatch"],
            NODE_DISPLAY_NAME_MAPPINGS["RHMiniMaxH3SolAttentionPatch"],
        ]
        text = "\n".join(
            item for value in values for item in _flatten_strings(value)
        )
        self.assertIsNone(CJK.search(text))

    def test_disabled_nodes_are_passthroughs(self):
        model = _FakeModel()
        self.assertIs(
            MiniMaxH3SageAttentionPatch().patch(model, enabled=False)[0], model
        )
        self.assertIs(
            MiniMaxH3SolAttentionPatch().patch(model, enabled=False)[0], model
        )

    def test_clone_does_not_mutate_original_nested_options(self):
        original_override = object()
        model = _FakeModel(
            {
                "transformer_options": {
                    "optimized_attention_override": original_override,
                    "kept": 7,
                }
            }
        )
        stored_original_override = model.model_options["transformer_options"][
            "optimized_attention_override"
        ]
        replacement = lambda *args, **kwargs: None

        patched = backends.clone_model_with_attention_override(model, replacement)

        self.assertIsNot(patched, model)
        self.assertIs(
            model.model_options["transformer_options"][
                "optimized_attention_override"
            ],
            stored_original_override,
        )
        self.assertIs(
            patched.model_options["transformer_options"][
                "optimized_attention_override"
            ],
            replacement,
        )
        self.assertEqual(patched.model_options["transformer_options"]["kept"], 7)

    def test_missing_sage_backend_leaves_model_runnable(self):
        model = _FakeModel()
        with mock.patch.object(
            backends,
            "_get_registered_sage_attention",
            return_value=None,
        ):
            output = MiniMaxH3SageAttentionPatch().patch(model, enabled=True)[0]
        self.assertIs(output, model)
        self.assertNotIn(
            "optimized_attention_override",
            model.model_options["transformer_options"],
        )

    def test_sage_override_delegates_to_comfy_registered_backend(self):
        calls = []

        def registered(q, k, v, heads, **kwargs):
            calls.append((q, k, v, heads, kwargs))
            return v

        with mock.patch.object(
            backends,
            "_get_registered_sage_attention",
            return_value=registered,
        ):
            override = backends.make_sage_attention_override()

        q = torch.zeros((1, 2, 3, 4), dtype=torch.bfloat16)
        k = q.clone()
        v = torch.arange(24, dtype=torch.float32).to(torch.bfloat16).reshape_as(q)
        mask = torch.ones((3, 3), dtype=torch.bool)

        def should_not_run(*_args, **_kwargs):
            raise AssertionError("Sage path unexpectedly delegated")

        output = override(
            should_not_run,
            q,
            k,
            v,
            2,
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=True,
            scale=0.25,
        )
        self.assertIs(output, v)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], q)
        self.assertIs(calls[0][1], k)
        self.assertIs(calls[0][2], v)
        self.assertEqual(calls[0][3], 2)
        self.assertIs(calls[0][4]["mask"], mask)
        self.assertTrue(calls[0][4]["skip_reshape"])
        self.assertTrue(calls[0][4]["skip_output_reshape"])
        self.assertEqual(calls[0][4]["scale"], 0.25)

    def test_sage_runtime_failure_uses_untouched_original_tensors(self):
        def failing_registered(*_args, **_kwargs):
            raise RuntimeError("kernel failed")

        received = []

        def original(q, k, v, heads, **kwargs):
            received.append((q, k, v, heads, kwargs))
            return q

        with mock.patch.object(
            backends,
            "_get_registered_sage_attention",
            return_value=failing_registered,
        ):
            override = backends.make_sage_attention_override()

        q = torch.zeros((1, 8, 256), dtype=torch.float32)
        k = q.clone()
        v = q.clone()
        output = override(original, q, k, v, 2, skip_reshape=False)

        self.assertIs(output, q)
        self.assertIs(received[0][0], q)
        self.assertIs(received[0][1], k)
        self.assertIs(received[0][2], v)
        self.assertEqual(received[0][0].dtype, torch.float32)
        self.assertEqual(tuple(received[0][0].shape), (1, 8, 256))
        self.assertFalse(received[0][4]["skip_reshape"])

    def test_sage_resolver_uses_unwrapped_comfy_function(self):
        calls = []

        def raw(*args, **kwargs):
            calls.append((args, kwargs))

        def wrapped(*_args, **_kwargs):
            raise AssertionError("The ComfyUI wrapper would recurse")

        wrapped.__wrapped__ = raw
        comfy_module = types.ModuleType("comfy")
        ldm_module = types.ModuleType("comfy.ldm")
        modules_module = types.ModuleType("comfy.ldm.modules")
        attention_module = types.ModuleType("comfy.ldm.modules.attention")
        attention_module.get_attention_function = mock.Mock(return_value=wrapped)
        fake_modules = {
            "comfy": comfy_module,
            "comfy.ldm": ldm_module,
            "comfy.ldm.modules": modules_module,
            "comfy.ldm.modules.attention": attention_module,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            resolved = backends._get_registered_sage_attention(
                allow_compile=True
            )
        resolved("q", "k", "v", 2)
        self.assertEqual(len(calls), 1)

    def test_sol_node_preserves_previous_override(self):
        calls = []

        def previous(function, *args, **kwargs):
            calls.append("previous")
            return function(*args, **kwargs)

        model = _FakeModel(
            {"transformer_options": {"optimized_attention_override": previous}}
        )
        patched = MiniMaxH3SolAttentionPatch().patch(
            model,
            enabled=True,
            tau=1.2,
            start_percent=0.2,
            end_percent=0.9,
            min_tokens=4096,
        )[0]
        installed = patched.model_options["transformer_options"][
            "optimized_attention_override"
        ]

        def original(q, *_args, **_kwargs):
            return q

        q = torch.zeros((1, 2, 32, 128), dtype=torch.bfloat16)
        output = installed(
            original,
            q,
            q,
            q,
            2,
            skip_reshape=True,
            transformer_options={"sigmas": torch.tensor([0.95])},
        )
        self.assertIs(output, q)
        self.assertEqual(calls, ["previous"])

    def test_sol_unsupported_device_delegates_to_previous_override(self):
        marker = torch.tensor([3.0])
        calls = []

        def previous(_function, *_args, **_kwargs):
            calls.append("previous")
            return marker

        override = backends.make_sol_attention_override(
            tau=1.2,
            min_tokens=0,
            previous=previous,
        )
        q = torch.zeros((1, 2, 8, 128), dtype=torch.bfloat16)
        output = override(
            lambda *_args, **_kwargs: None,
            q,
            q,
            q,
            2,
            skip_reshape=True,
        )
        self.assertIs(output, marker)
        self.assertEqual(calls, ["previous"])

    def test_invalid_sol_schedule_is_rejected_in_english(self):
        with self.assertRaisesRegex(ValueError, "start_percent"):
            MiniMaxH3SolAttentionPatch().patch(
                _FakeModel(),
                enabled=True,
                start_percent=0.9,
                end_percent=0.2,
            )

    def test_sol_backend_rejects_cpu_with_clear_message(self):
        q = torch.zeros((1, 128, 1, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            backends.sol_attention_flex(q, q, q)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sol_override_formats_bundled_backend_output(self):
        q = torch.zeros((1, 2, 8, 128), device="cuda", dtype=torch.bfloat16)

        def fake_backend(q_bthd, _k, _v, **_kwargs):
            return torch.ones_like(q_bthd)

        override = backends.make_sol_attention_override(tau=1.2, min_tokens=0)
        with mock.patch.object(backends, "sol_attention_flex", side_effect=fake_backend):
            flat = override(
                lambda *_args, **_kwargs: None,
                q,
                q,
                q,
                2,
                skip_reshape=True,
                skip_output_reshape=False,
            )
            headed = override(
                lambda *_args, **_kwargs: None,
                q,
                q,
                q,
                2,
                skip_reshape=True,
                skip_output_reshape=True,
            )
        self.assertEqual(tuple(flat.shape), (1, 8, 256))
        self.assertEqual(tuple(headed.shape), (1, 2, 8, 128))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sol_runtime_failure_delegates_to_previous_override(self):
        marker = torch.tensor([7.0])
        calls = []

        def previous(_function, *_args, **_kwargs):
            calls.append("previous")
            return marker

        override = backends.make_sol_attention_override(
            tau=1.2,
            min_tokens=0,
            previous=previous,
        )
        q = torch.zeros((1, 2, 8, 128), device="cuda", dtype=torch.bfloat16)
        with mock.patch.object(
            backends,
            "sol_attention_flex",
            side_effect=RuntimeError("compile failed"),
        ):
            output = override(
                lambda *_args, **_kwargs: None,
                q,
                q,
                q,
                2,
                skip_reshape=True,
            )
        self.assertIs(output, marker)
        self.assertEqual(calls, ["previous"])


class AttentionWorkflowTests(unittest.TestCase):
    def _load(self, name):
        return json.loads((WORKFLOWS / name).read_text(encoding="utf-8"))

    @staticmethod
    def _all_nodes(workflow):
        nodes = list(workflow["nodes"])
        for subgraph in workflow.get("definitions", {}).get("subgraphs", []):
            nodes.extend(subgraph["nodes"])
        return nodes

    def test_sage_workflow_uses_only_the_bundled_patch(self):
        workflow = self._load("t2va_native_sage_attention.json")
        subgraph = workflow["definitions"]["subgraphs"][0]
        node = next(node for node in subgraph["nodes"] if node["id"] == 119)
        self.assertEqual(node["type"], "RHMiniMaxH3SageAttentionPatch")
        self.assertEqual(node["widgets_values"], [True])
        self.assertNotIn("aux_id", node["properties"])
        links = {link["id"]: link for link in subgraph["links"]}
        self.assertEqual((links[228]["origin_id"], links[228]["target_id"]), (6, 119))
        self.assertEqual((links[229]["origin_id"], links[229]["target_id"]), (119, 9))
        self.assertEqual((links[230]["origin_id"], links[230]["target_id"]), (119, 16))

    def test_sol_workflow_chains_bundled_sage_then_bundled_sol(self):
        workflow = self._load("t2va_native_sol_attn.json")
        subgraph = workflow["definitions"]["subgraphs"][0]
        nodes = {node["id"]: node for node in subgraph["nodes"]}
        self.assertEqual(nodes[120]["type"], "RHMiniMaxH3SageAttentionPatch")
        self.assertEqual(nodes[119]["type"], "RHMiniMaxH3SolAttentionPatch")
        self.assertEqual(nodes[120]["widgets_values"], [True])
        self.assertEqual(nodes[119]["widgets_values"], [True, 1.2, 0.2, 0.9, 4096])
        self.assertNotIn("aux_id", nodes[119]["properties"])
        links = {link["id"]: link for link in subgraph["links"]}
        self.assertEqual((links[228]["origin_id"], links[228]["target_id"]), (6, 120))
        self.assertEqual((links[231]["origin_id"], links[231]["target_id"]), (120, 119))
        self.assertEqual((links[229]["origin_id"], links[229]["target_id"]), (119, 9))
        self.assertEqual((links[230]["origin_id"], links[230]["target_id"]), (119, 16))

    def test_native_workflows_have_no_external_attention_node_ids_or_cjk(self):
        forbidden = {"StandaloneSageAttentionPatch", "SolAttnMiniMaxH3Patcher"}
        for name in (
            "t2va_native_sage_attention.json",
            "t2va_native_sol_attn.json",
        ):
            path = WORKFLOWS / name
            raw = path.read_text(encoding="utf-8")
            workflow = json.loads(raw)
            subgraph = workflow["definitions"]["subgraphs"][0]
            self.assertTrue(forbidden.isdisjoint(node["type"] for node in subgraph["nodes"]))
            self.assertIsNone(CJK.search(raw))

    def test_every_published_workflow_uses_native_sampling_and_english_notes(self):
        forbidden_exact = {
            "RHMiniMaxH3DualSigmaSampler",
            "RHMiniMaxH3EmptyAVLatent",
            "RHMiniMaxH3DecodeAV",
            "StandaloneSageAttentionPatch",
            "SolAttnMiniMaxH3Patcher",
            "LTXReframeCustomAudioLoader",
        }
        forbidden_prefixes = (
            "RHMiniMaxH3Direct",
            "RHMiniMaxH3Ref2VA",
            "RHMiniMaxH3T2VA",
        )

        for path in sorted(WORKFLOWS.glob("*.json")):
            with self.subTest(workflow=path.name):
                raw = path.read_text(encoding="utf-8")
                workflow = json.loads(raw)
                types = {node["type"] for node in self._all_nodes(workflow)}
                decoded_text = "\n".join(_flatten_strings(workflow))

                self.assertIsNone(CJK.search(raw))
                self.assertIsNone(C0_CONTROL.search(decoded_text))
                self.assertTrue(forbidden_exact.isdisjoint(types))
                self.assertFalse(
                    any(
                        node_type.startswith(forbidden_prefixes)
                        for node_type in types
                    )
                )
                self.assertIn("BasicScheduler", types)
                self.assertIn("SamplerCustomAdvanced", types)
                self.assertIn("RHMiniMaxH3SageAttentionPatch", types)

    def test_default_t2va_workflow_uses_the_validated_native_subgraph(self):
        workflow = self._load("t2va.json")
        subgraph = workflow["definitions"]["subgraphs"][0]
        types = {node["type"] for node in subgraph["nodes"]}
        group = next(
            node for node in workflow["nodes"] if node["type"] == subgraph["id"]
        )
        save = next(node for node in workflow["nodes"] if node["type"] == "SaveVideo")

        self.assertIn("MiniMaxH3ImageToVideo", types)
        self.assertIn("RHMiniMaxH3SageAttentionPatch", types)
        self.assertIsNone(group["inputs"][0]["link"])
        self.assertIsNone(group["inputs"][1]["link"])
        self.assertEqual(save["widgets_values"], ["minimax_h3/t2va", "mp4", "h264"])

    def test_ref2va_workflows_use_native_typed_reference_inputs(self):
        expected = {
            "ref2va_image.json": {
                "ref_images.ref_image_0": ("LoadImage", 0, "IMAGE"),
            },
            "ref2va_image_audio.json": {
                "ref_images.ref_image_0": ("LoadImage", 0, "IMAGE"),
                "ref_audios.ref_audio_0": ("LoadAudio", 0, "AUDIO"),
            },
            "ref2va_video_audio.json": {
                "ref_videos.ref_video_0": ("GetVideoComponents", 0, "IMAGE"),
                "ref_video_audios.ref_video_audio_0": (
                    "GetVideoComponents",
                    1,
                    "AUDIO",
                ),
            },
            "ref2va_video_custom_audio.json": {
                "ref_videos.ref_video_0": ("GetVideoComponents", 0, "IMAGE"),
                "ref_audios.ref_audio_0": ("TrimAudioDuration", 0, "AUDIO"),
            },
        }

        for name, expected_inputs in expected.items():
            with self.subTest(workflow=name):
                workflow = self._load(name)
                nodes = {node["id"]: node for node in workflow["nodes"]}
                links = {link[0]: link for link in workflow["links"]}
                for link in workflow["links"]:
                    self.assertEqual(len(link), 6)
                    source = nodes[link[1]]
                    target = nodes[link[3]]
                    self.assertIn(link[0], source["outputs"][link[2]]["links"])
                    self.assertEqual(target["inputs"][link[4]]["link"], link[0])
                reference = next(
                    node
                    for node in workflow["nodes"]
                    if node["type"] == "MiniMaxH3ReferenceToVideo"
                )
                input_slots = {
                    input_spec["name"]: (slot, input_spec)
                    for slot, input_spec in enumerate(reference["inputs"])
                }

                for input_name, (origin_type, origin_slot, link_type) in expected_inputs.items():
                    slot, input_spec = input_slots[input_name]
                    link = links[input_spec["link"]]
                    self.assertEqual(nodes[link[1]]["type"], origin_type)
                    self.assertEqual(link[2], origin_slot)
                    self.assertEqual((link[3], link[4], link[5]), (136, slot, link_type))

                model = next(node for node in workflow["nodes"] if node["type"] == "UNETLoader")
                patch = next(
                    node
                    for node in workflow["nodes"]
                    if node["type"] == "RHMiniMaxH3SageAttentionPatch"
                )
                scheduler = next(
                    node for node in workflow["nodes"] if node["type"] == "BasicScheduler"
                )
                guider = next(node for node in workflow["nodes"] if node["type"] == "BasicGuider")
                model_patch_link = links[patch["inputs"][0]["link"]]
                scheduler_link = links[scheduler["inputs"][0]["link"]]
                guider_link = links[guider["inputs"][0]["link"]]
                self.assertEqual((model_patch_link[1], model_patch_link[3]), (model["id"], patch["id"]))
                self.assertEqual((scheduler_link[1], scheduler_link[3]), (patch["id"], scheduler["id"]))
                self.assertEqual((guider_link[1], guider_link[3]), (patch["id"], guider["id"]))

                if name in {
                    "ref2va_video_audio.json",
                    "ref2va_video_custom_audio.json",
                }:
                    components = next(
                        node
                        for node in workflow["nodes"]
                        if node["type"] == "GetVideoComponents"
                    )
                    video_link = links[components["inputs"][0]["link"]]
                    self.assertEqual(nodes[video_link[1]]["type"], "LoadVideo")
                    self.assertEqual(video_link[5], "VIDEO")

                if name == "ref2va_video_custom_audio.json":
                    trim = next(
                        node
                        for node in workflow["nodes"]
                        if node["type"] == "TrimAudioDuration"
                    )
                    audio_link = links[trim["inputs"][0]["link"]]
                    duration_link = links[trim["inputs"][2]["link"]]
                    prompt = next(
                        node
                        for node in workflow["nodes"]
                        if node["type"] == "PrimitiveStringMultiline"
                    )["widgets_values"][0]

                    self.assertEqual(nodes[audio_link[1]]["type"], "LoadAudio")
                    self.assertEqual(nodes[duration_link[1]]["type"], "PrimitiveFloat")
                    self.assertEqual(trim["widgets_values"][0], 0)
                    self.assertIn("<Video 1>", prompt)
                    self.assertIn("<Audio 1>", prompt)
                    self.assertNotIn("<Video 0>", prompt)
                    self.assertNotIn("<Audio 0>", prompt)

    def test_fl2va_workflows_use_native_keyframe_conditioning(self):
        cases = {
            "fl2va_first_frame.json": (True, False),
            "fl2va_last_frame.json": (False, True),
            "fl2va_first_last_frame.json": (True, True),
        }

        for name, expected_links in cases.items():
            with self.subTest(workflow=name):
                path = WORKFLOWS / name
                raw = path.read_text(encoding="utf-8")
                workflow = json.loads(raw)
                subgraph = workflow["definitions"]["subgraphs"][0]
                subgraph_types = {node["type"] for node in subgraph["nodes"]}

                self.assertIn("MiniMaxH3ImageToVideo", subgraph_types)
                self.assertIn("BasicScheduler", subgraph_types)
                self.assertIn("SamplerCustomAdvanced", subgraph_types)
                self.assertIn("RHMiniMaxH3SageAttentionPatch", subgraph_types)
                self.assertNotIn("RHMiniMaxH3DualSigmaSampler", raw)
                self.assertIsNone(CJK.search(raw))

                group = next(
                    node for node in workflow["nodes"] if node["type"] == subgraph["id"]
                )
                links = {link[0]: link for link in workflow["links"]}
                top_nodes = {node["id"]: node for node in workflow["nodes"]}
                for slot, should_be_linked in enumerate(expected_links):
                    link_id = group["inputs"][slot]["link"]
                    self.assertEqual(link_id is not None, should_be_linked)
                    if link_id is not None:
                        link = links[link_id]
                        self.assertEqual(
                            (link[3], link[4], link[5]),
                            (group["id"], slot, "IMAGE"),
                        )
                        self.assertEqual(top_nodes[link[1]]["type"], "LoadImage")


if __name__ == "__main__":
    unittest.main()
