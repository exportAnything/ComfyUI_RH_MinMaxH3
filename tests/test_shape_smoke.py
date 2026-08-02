"""packing → sample_h3 形状贯通冒烟（tiny mock DiT，无需真实权重）。"""
from __future__ import annotations
import importlib.util
import unittest
from contextlib import contextmanager
from unittest import mock

HAS_TORCH = importlib.util.find_spec("torch") is not None

@unittest.skipUnless(HAS_TORCH, "torch 未安装")
class PackingToSampleShapeSmoke(unittest.TestCase):
    @contextmanager
    def _cpu_sampler(self, sampling):
        passthrough = lambda value, **_kwargs: dict(value)
        with (
            mock.patch.object(sampling, "_require_accelerated_sampler_device", return_value=None),
            mock.patch.object(sampling, "validate_conditioning", side_effect=passthrough),
            mock.patch.object(sampling, "validate_av_latent", side_effect=passthrough),
            mock.patch.object(sampling, "validate_conditioning_v2", side_effect=passthrough),
            mock.patch.object(sampling, "validate_av_latent_v2", side_effect=passthrough),
        ):
            yield

    def test_t2va_packing_sample_h3_shapes(self):
        import torch
        from minimax_h3_nodes.runtime.packing import build_t2va_packed_conditioning
        import minimax_h3_nodes.sampling as sampling

        latent_t, latent_h, latent_w, audio_t = 1, 2, 2, 2
        prompt = torch.randn(3, 8)
        packed = build_t2va_packed_conditioning(
            prompt, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t,
        )
        target = {
            "schema": "minimax_h3_target/v1", "task": "t2va", "partition": "fl2va",
            "width": 32, "height": 32, "fps": 24, "frame_count": 5,
            "video_latent_t": latent_t, "video_latent_h": latent_h, "video_latent_w": latent_w,
            "audio_latent_t": audio_t,
        }
        conditioning = {
            "schema": "minimax_h3_conditioning/v1", "task": "t2va", "partition": "fl2va",
            "prompt_embeds": prompt,
        }
        av_latent = {
            "schema": "minimax_h3_av_latent/v1", "task": "t2va", "partition": "fl2va",
            "target": target,
            "video": torch.zeros(1, 24, latent_t, latent_h, latent_w),
            "audio": torch.zeros(2, 32, audio_t), "sampled": False,
        }

        class EchoDiT:
            device = torch.device("cpu")
            def __call__(self, **kwargs):
                img = kwargs["img_pos_info"]["position_ids"]
                aud = kwargs["audio_pos_info"]["position_ids"]
                v = kwargs["x"][0].index_select(0, img)
                a = kwargs["audio_x"][0].index_select(0, aud)
                return torch.zeros_like(v), torch.zeros_like(a)

        with self._cpu_sampler(sampling):
            out = sampling.sample_h3(
                transformer=EchoDiT(), conditioning=conditioning, av_latent=av_latent,
                packed=packed, seed=0, sigma_points=3, video_shift=12.0, audio_shift=3.0,
            )
            alias = sampling.sample_t2va(  # 委托入口
                transformer=EchoDiT(), conditioning=conditioning, av_latent=av_latent,
                packed=packed, seed=0, sigma_points=3, video_shift=12.0, audio_shift=3.0,
            )
        self.assertEqual(tuple(out["video"].shape), (1, 24, latent_t, latent_h, latent_w))
        self.assertEqual(tuple(out["audio"].shape), (2, 32, audio_t))
        self.assertTrue(out["sampled"])
        self.assertEqual(out["dit_calls"], 2)  # sigma_points=3 → 2 DiT
        self.assertEqual(out["dit_steps_total"], 2)
        self.assertEqual(tuple(alias["video"].shape), tuple(out["video"].shape))

    def test_tiny_dit_forward_matches_packed_positions(self):
        """meta 构架 + CPU tiny DiT：验证 forward 输出行宽与 patch 合同。"""
        import torch
        from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig, MiniMaxH3DiTModel

        cfg = MiniMaxH3DiTConfig(
            num_layers=1, token_refiner_num_layers=1, hidden_size=8, num_attention_heads=1,
            attention_head_dim=8, ffn_hidden_size=16, latents_dim=1, audio_latents_dim=2,
            patch_size=(1, 1, 1), text_dim=6, timestep_input_dim=4, time_embed_hidden_size=8,
            time_embed_dim=4, adaln_out_features=18 * 8, final_adaln_out_features=2 * 8,
            rope_inv_freq_len=1,
        )
        model = MiniMaxH3DiTModel(cfg, device="cpu", dtype=torch.float32).eval()
        seq = 6
        with torch.inference_mode():
            video, audio = model(
                x=torch.randn(1, seq, 1), audio_x=torch.randn(1, seq, 2),
                img_position_ids=torch.zeros(1, seq, 3, dtype=torch.float64),
                unique_timesteps=torch.tensor([0.25, 0.5]),
                inverse_indices=torch.tensor([0, 0, 1, 1, 0, 0]),
                update_mask=torch.ones(2, dtype=torch.bool),
                update_audio_mask=torch.ones(2, dtype=torch.bool),
                token_tags=torch.tensor([1, 1, 2, 2, 0, 0]),
                prompt_embeds=torch.randn(2, 6),
                img_pos_info={"position_ids": torch.tensor([4, 5])},
                audio_pos_info={"position_ids": torch.tensor([2, 3])},
                text_pos_info={"position_ids": torch.tensor([0, 1])},
                img_pos_for_infer_output_info={"position_ids": torch.tensor([4, 5])},
                packed_seq_params={"cu_seqlens_q": torch.tensor([0, seq], dtype=torch.int32), "max_seqlen_q": seq},
                refiner_packed_seq_params={
                    "cu_seqlens_q": torch.tensor([0, 2, 2], dtype=torch.int32), "max_seqlen_q": 2,
                },
            )
        self.assertEqual(tuple(video.shape), (2, 1))
        self.assertEqual(tuple(audio.shape), (2, 2))
        self.assertTrue(torch.isfinite(video).all() and torch.isfinite(audio).all())

if __name__ == "__main__":
    unittest.main()
