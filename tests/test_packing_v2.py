"""Golden tests for MiniMax-H3 FL2VA/Ref2VA packed layouts.

The expected values are intentionally small versions of the official SGLang
unit cases.  They exercise ordering and fp64 accumulation without allocating
model-sized tensors.
"""

from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TestMiniMaxH3PackingPublicContracts(unittest.TestCase):
    def test_all_audio_channel_entries_reject_non_stereo_before_tensor_work(self):
        from minimax_h3_nodes.runtime.packing import (
            build_fl2va_packed_conditioning,
            build_h3_packed_conditioning,
            build_ref2va_packed_conditioning,
            build_t2va_packed_conditioning,
            minimax_h3_packed_sequence,
            minimax_h3_packed_sequence_fl2va,
            minimax_h3_packed_sequence_ref2va_blocks,
            minimax_h3_packed_sequence_t2va,
            unpack_audio_tokens,
        )

        layout = dict(text_len=2, latent_t=1, latent_h=2, latent_w=2, audio_t=1)
        high_level = dict(latent_t=1, latent_h=2, latent_w=2, audio_t=1)
        calls = {
            "t2va layout": lambda channel: minimax_h3_packed_sequence_t2va(
                **layout, audio_channel=channel
            ),
            "generic layout": lambda channel: minimax_h3_packed_sequence(
                **layout, audio_channel=channel
            ),
            "fl2va layout": lambda channel: minimax_h3_packed_sequence_fl2va(
                **layout,
                keyframe_frame_indices=[0],
                frame_count=5,
                audio_channel=channel,
            ),
            "ref2va layout": lambda channel: minimax_h3_packed_sequence_ref2va_blocks(
                **layout,
                ref_blocks=[],
                audio_channel=channel,
            ),
            "t2va builder": lambda channel: build_t2va_packed_conditioning(
                object(), **high_level, audio_channel=channel
            ),
            "fl2va builder": lambda channel: build_fl2va_packed_conditioning(
                object(),
                **high_level,
                frame_count=5,
                keyframe_frame_indices=[0],
                keyframe_cond_rows=object(),
                audio_channel=channel,
            ),
            "ref2va builder": lambda channel: build_ref2va_packed_conditioning(
                object(),
                **high_level,
                condition_blocks=[],
                audio_channel=channel,
            ),
            "task dispatcher": lambda channel: build_h3_packed_conditioning(
                object(), task="t2va", **high_level, audio_channel=channel
            ),
            "audio unpacker": lambda channel: unpack_audio_tokens(
                object(), audio_t=1, audio_channel=channel
            ),
        }
        for name, call in calls.items():
            for channel in (1, 3):
                with self.subTest(entry=name, audio_channel=channel):
                    with self.assertRaisesRegex(ValueError, "must be exactly 2"):
                        call(channel)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for packed layout tests")
class TestMiniMaxH3PackingV2(unittest.TestCase):
    def test_t2va_v1_structural_tensors_are_unchanged(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            minimax_h3_packed_sequence,
            minimax_h3_packed_sequence_t2va,
        )

        kwargs = dict(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=3,
            audio_channel=2,
        )
        previous = minimax_h3_packed_sequence_t2va(**kwargs)
        generic = minimax_h3_packed_sequence(
            **kwargs,
            include_keyframe_cond=False,
        )
        for key, expected in previous.items():
            with self.subTest(key=key):
                self.assertTrue(torch.equal(generic[key], expected))

    def test_fl2va_first_last_has_exact_endpoint_anchors_and_tags(self):
        import numpy as np
        import torch

        from minimax_h3_nodes.runtime.packing import (
            FRAME_PER_TOKEN,
            FRAME_RESCALE,
            IMGVID_COND_ID,
            VIDEO_FIRST_ID,
            VIDEO_LAST_ID,
            minimax_h3_packed_sequence_fl2va,
        )

        text_len = 11
        latent_t = 37
        text_tags = torch.tensor([0, 0, 0] + [1] * 8)
        packed = minimax_h3_packed_sequence_fl2va(
            text_len=text_len,
            latent_t=latent_t,
            latent_h=48,
            latent_w=76,
            audio_t=203,
            keyframe_frame_indices=[0, -1],
            frame_count=124,
            text_token_tags=text_tags,
        )
        frame_rows = 24 * 38
        condition_rows = frame_rows * 2
        self.assertEqual(int((~packed["update_mask"]).sum()), condition_rows)
        self.assertTrue(
            (packed["input_ids"][text_len : text_len + condition_rows]
             == IMGVID_COND_ID).all()
        )
        self.assertTrue(torch.equal(packed["token_tags"][:text_len], text_tags))
        condition_pos = packed["condition_img_pos"].reshape(2, frame_rows)
        first_t = float(
            packed["img_position_ids"][condition_pos[0], 0].unique().item()
        )
        last_t = float(
            packed["img_position_ids"][condition_pos[1], 0].unique().item()
        )
        spans = np.ones(latent_t, dtype=np.float64) * FRAME_RESCALE
        for index, multiplier in enumerate(FRAME_PER_TOKEN):
            spans[index::len(FRAME_PER_TOKEN)] *= multiplier
        self.assertEqual(first_t, float(text_len))
        self.assertEqual(last_t, float(text_len) + float(spans.sum()) - FRAME_RESCALE)
        target_pos = packed["target_img_pos"]
        self.assertEqual(int(packed["input_ids"][target_pos[0]]), VIDEO_FIRST_ID)
        self.assertEqual(int(packed["input_ids"][target_pos[-1]]), VIDEO_LAST_ID)

    def test_fl_pairwise_and_ref_sequential_fp64_spans_stay_distinct(self):
        import numpy as np

        from minimax_h3_nodes.runtime.packing import (
            FRAME_PER_TOKEN,
            FRAME_RESCALE,
            minimax_h3_packed_sequence_fl2va,
            minimax_h3_packed_sequence_ref2va_blocks,
        )

        text_len = 5
        ref_t = 16
        fl = minimax_h3_packed_sequence_fl2va(
            text_len=text_len,
            latent_t=ref_t,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            keyframe_frame_indices=[-1],
            frame_count=56,
        )
        ref = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=text_len,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            ref_blocks=[
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": ref_t,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            ],
        )
        values = np.asarray(
            [
                FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
                for index in range(ref_t)
            ],
            dtype=np.float64,
        )
        fl_span = (
            fl["keyframe_temporal_positions"][0]
            + FRAME_RESCALE
            - float(text_len)
        )
        ref_span = float(ref["target_temporal_origin"]) - float(text_len)
        self.assertEqual(fl_span, float(values.sum()))
        self.assertEqual(
            ref_span,
            sum(
                FRAME_RESCALE
                * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
                for index in range(ref_t)
            ),
        )
        # These differ by one ulp from n=16 onward in the official source.
        self.assertNotEqual(fl_span, ref_span)

    def test_ref2va_video_audio_is_audio_then_visual_in_sequence(self):
        from minimax_h3_nodes.runtime.packing import (
            AUDIO_REF_COND_ID,
            IMGVID_COND_ID,
            minimax_h3_packed_sequence_ref2va_blocks,
        )

        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=7,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=5,
            ref_blocks=[
                {
                    "kind": "video_audio",
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                }
            ],
        )
        self.assertTrue((packed["input_ids"][7:13] == AUDIO_REF_COND_ID).all())
        self.assertTrue((packed["input_ids"][13:21] == IMGVID_COND_ID).all())
        self.assertEqual(int((~packed["audio_update_mask"]).sum()), 6)
        self.assertEqual(int((~packed["update_mask"]).sum()), 8)
        self.assertAlmostEqual(
            float(packed["target_temporal_origin"]),
            7.0 + 25.0 / 3.0,
        )

    def test_ref2va_mixed_blocks_have_independent_spatial_grids(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            minimax_h3_packed_sequence_ref2va_blocks,
        )

        packed = minimax_h3_packed_sequence_ref2va_blocks(
            text_len=3,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            ref_blocks=[
                {"kind": "image", "latent_h": 4, "latent_w": 8},
                {"kind": "audio", "ref_audio_t": 2},
                {
                    "kind": "video",
                    "ref_audio_t": 0,
                    "latent_t": 2,
                    "latent_h": 8,
                    "latent_w": 4,
                },
            ],
            text_token_tags=torch.tensor([0, 1, 1]),
        )
        image_rows = (4 // 2) * (8 // 2)
        video_rows = 2 * (8 // 2) * (4 // 2)
        self.assertEqual(int((~packed["update_mask"]).sum()), image_rows + video_rows)
        self.assertEqual(int((~packed["audio_update_mask"]).sum()), 4)
        self.assertEqual(packed["condition_blocks"][0]["temporal_origin"], 3.0)
        self.assertEqual(packed["condition_blocks"][1]["temporal_origin"], 4.0)
        self.assertEqual(packed["condition_blocks"][2]["temporal_origin"], 6.0)
        image_positions = packed["condition_img_pos"][:image_rows]
        video_positions = packed["condition_img_pos"][image_rows:]
        image_h = packed["img_position_ids"][image_positions, 1].unique()
        video_h = packed["img_position_ids"][video_positions, 1].unique()
        self.assertEqual(int(image_h.numel()), 2)
        self.assertEqual(int(video_h.numel()), 4)
        self.assertTrue(torch.equal(packed["token_tags"][:3], torch.tensor([0, 1, 1])))

    def test_ref2va_high_level_silent_video_keeps_visual_block_and_span(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            build_ref2va_packed_conditioning,
        )

        text_len = 3
        ref_latent_t = 2
        video_rows = torch.zeros(8, 96)
        packed = build_ref2va_packed_conditioning(
            torch.zeros(text_len, 5120),
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            condition_blocks=[
                {
                    "kind": "video",
                    "condition_index": 0,
                    "visual_rows": video_rows,
                    "latent_t": ref_latent_t,
                    "latent_h": 4,
                    "latent_w": 4,
                    # Node-produced silent-video blocks intentionally omit
                    # audio_rows/ref_audio_t instead of inventing an audio row.
                }
            ],
        )

        self.assertEqual(packed["condition_blocks"][0]["ref_audio_t"], 0)
        self.assertIsNone(packed["audio_ref_rows"])
        self.assertEqual(packed["audio_reference_t"], [])
        self.assertTrue(torch.equal(packed["visual_cond_rows"], video_rows))
        self.assertEqual(int((~packed["update_mask"]).sum()), 8)
        self.assertEqual(int((~packed["audio_update_mask"]).sum()), 0)

        expected_origin = text_len + (5.0 / 3.0) * (1 + 4)
        self.assertAlmostEqual(packed["target_temporal_origin"], expected_origin)
        target_audio_t0 = packed["img_position_ids"][
            packed["target_audio_pos"][0], 0
        ]
        target_video_t0 = packed["img_position_ids"][
            packed["target_img_pos"][0], 0
        ]
        self.assertEqual(float(target_audio_t0), float(target_video_t0))
        self.assertAlmostEqual(float(target_video_t0), expected_origin)

    def test_ref2va_high_level_video_audio_requires_positive_audio_length(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            build_ref2va_packed_conditioning,
        )

        common = dict(
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
        )
        block = {
            "kind": "video_audio",
            "condition_index": 0,
            "visual_rows": torch.zeros(8, 96),
            "latent_t": 2,
            "latent_h": 4,
            "latent_w": 4,
        }
        with self.assertRaisesRegex(ValueError, "ref_audio_t must be a positive"):
            build_ref2va_packed_conditioning(
                torch.zeros(2, 5120),
                condition_blocks=[block],
                **common,
            )
        with self.assertRaisesRegex(ValueError, "ref_audio_t must be a positive"):
            build_ref2va_packed_conditioning(
                torch.zeros(2, 5120),
                condition_blocks=[{**block, "ref_audio_t": 0}],
                **common,
            )

    def test_ref2va_high_level_audio_requires_positive_audio_length(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            build_ref2va_packed_conditioning,
        )

        common = dict(
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
        )
        block = {
            "kind": "audio",
            "condition_index": 0,
        }
        with self.assertRaisesRegex(ValueError, "ref_audio_t must be a positive"):
            build_ref2va_packed_conditioning(
                torch.zeros(2, 5120),
                condition_blocks=[block],
                **common,
            )
        with self.assertRaisesRegex(ValueError, "ref_audio_t must be a positive"):
            build_ref2va_packed_conditioning(
                torch.zeros(2, 5120),
                condition_blocks=[
                    {
                        **block,
                        "ref_audio_t": 0,
                        "audio_rows": torch.empty(0, 32),
                    }
                ],
                **common,
            )

    def test_audio_latent_pack_and_unpack_require_stereo(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            pack_audio_latent,
            unpack_audio_tokens,
        )

        latent = torch.arange(2 * 32 * 3).reshape(2, 32, 3)
        rows = pack_audio_latent(latent)
        self.assertTrue(torch.equal(unpack_audio_tokens(rows, audio_t=3), latent))
        for channels in (1, 3):
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, "must be exactly 2"):
                    pack_audio_latent(torch.zeros(channels, 32, 3))

    def test_high_level_builders_attach_rows_and_target_metadata(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            H3ConditionBlockDescriptor,
            build_fl2va_packed_conditioning,
            build_ref2va_packed_conditioning,
        )

        prompt = torch.zeros(3, 5120, dtype=torch.bfloat16)
        fl_rows = torch.arange(4 * 96, dtype=torch.float32).reshape(4, 96)
        fl = build_fl2va_packed_conditioning(
            prompt,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            frame_count=5,
            condition_blocks=[
                H3ConditionBlockDescriptor(
                    kind="keyframe",
                    condition_index=4,
                    visual_rows=fl_rows,
                    semantic_frame_index=0,
                )
            ],
            text_token_tags=torch.tensor([0, 1, 1]),
        )
        self.assertEqual(fl["task"], "fl2va")
        self.assertEqual(fl["partition"], "fl2va")
        self.assertTrue(torch.equal(fl["visual_cond_rows"], fl_rows))
        self.assertEqual(fl["visual_condition_shapes"], [(1, 4, 4)])
        self.assertEqual(fl["latent_shape"], (2, 4, 4, 24))

        image_rows = torch.full((4, 96), 1.0)
        audio_rows = torch.full((6, 32), 2.0)
        video_rows = torch.full((8, 96), 3.0)
        ref = build_ref2va_packed_conditioning(
            prompt,
            latent_t=2,
            latent_h=4,
            latent_w=4,
            audio_t=2,
            condition_blocks=[
                {
                    "kind": "image",
                    "condition_index": 0,
                    "visual_rows": image_rows,
                    "latent_h": 4,
                    "latent_w": 4,
                },
                {
                    "kind": "video_audio",
                    "condition_index": 1,
                    "visual_rows": video_rows,
                    "audio_rows": audio_rows,
                    "ref_audio_t": 3,
                    "latent_t": 2,
                    "latent_h": 4,
                    "latent_w": 4,
                },
            ],
        )
        self.assertEqual(ref["task"], "ref2va")
        self.assertEqual(ref["partition"], "ref2va")
        self.assertEqual(ref["visual_condition_shapes"], [(1, 4, 4), (2, 4, 4)])
        self.assertEqual(ref["audio_reference_t"], [3])
        self.assertTrue(
            torch.equal(
                ref["visual_cond_rows"],
                torch.cat((image_rows, video_rows)),
            )
        )
        self.assertTrue(torch.equal(ref["audio_ref_rows"], audio_rows))

    def test_invalid_fl_signature_and_condition_shape_fail_early(self):
        import torch

        from minimax_h3_nodes.runtime.packing import (
            build_fl2va_packed_conditioning,
            minimax_h3_packed_sequence_fl2va,
        )

        with self.assertRaisesRegex(ValueError, "requires keyframe_frame_indices"):
            minimax_h3_packed_sequence_fl2va(
                text_len=3,
                latent_t=2,
                latent_h=4,
                latent_w=4,
                audio_t=2,
                keyframe_frame_indices=[-1, 0],
                frame_count=5,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_fl2va_packed_conditioning(
                torch.zeros(2, 5120),
                latent_t=2,
                latent_h=4,
                latent_w=4,
                audio_t=2,
                frame_count=5,
                keyframe_frame_indices=[0],
                keyframe_cond_rows=torch.zeros(3, 96),
            )


if __name__ == "__main__":
    unittest.main()
