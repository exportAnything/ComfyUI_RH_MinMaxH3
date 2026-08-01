import unittest

from minimax_h3_nodes.contracts import (
    H3ContractError,
    H3TaskNotImplementedError,
    align_frame_count,
    require_t2va,
    resolve_t2va_target,
)


class ContractTests(unittest.TestCase):
    def test_reference_5_second_16_9_geometry(self):
        target = resolve_t2va_target(aspect_ratio="16:9", duration_seconds=5.0)
        self.assertEqual(target["width"], 1344)
        self.assertEqual(target["height"], 768)
        self.assertEqual(target["frame_count"], 124)
        self.assertEqual(target["video_latent_t"], 37)
        self.assertEqual(target["video_latent_h"], 48)
        self.assertEqual(target["video_latent_w"], 84)
        self.assertEqual(target["audio_latent_t"], 207)
        self.assertEqual(target["fps"], 24)

    def test_frame_lattice_is_upward_only(self):
        self.assertEqual(align_frame_count(5), 5)
        self.assertEqual(align_frame_count(6), 22)
        self.assertEqual(align_frame_count(120), 124)
        self.assertEqual(align_frame_count(124), 124)

    def test_explicit_832_by_480_geometry(self):
        target = resolve_t2va_target(
            aspect_ratio="16:9",
            duration_seconds=5.0,
            width=832,
            height=480,
        )
        self.assertEqual(target["width"], 832)
        self.assertEqual(target["height"], 480)
        self.assertEqual(target["geometry"], "explicit_v1")
        self.assertEqual(target["frame_count"], 124)
        self.assertEqual(target["video_latent_h"], 30)
        self.assertEqual(target["video_latent_w"], 52)

    def test_explicit_geometry_requires_a_complete_aligned_pair(self):
        with self.assertRaisesRegex(H3ContractError, "必须同时填写"):
            resolve_t2va_target(
                aspect_ratio="16:9", duration_seconds=5.0, width=832, height=0
            )
        with self.assertRaisesRegex(H3ContractError, "必须按 32 对齐"):
            resolve_t2va_target(
                aspect_ratio="16:9", duration_seconds=5.0, width=830, height=480
            )

    def test_unimplemented_tasks_fail_explicitly(self):
        self.assertEqual(require_t2va("T2VA"), "t2va")
        for task in ("fl2va", "ref2va"):
            with self.assertRaises(H3TaskNotImplementedError):
                require_t2va(task)


if __name__ == "__main__":
    unittest.main()
