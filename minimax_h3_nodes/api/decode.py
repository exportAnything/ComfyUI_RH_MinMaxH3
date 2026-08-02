"""Decode 节点。"""
from __future__ import annotations
from ._shared import *  # noqa: F403

class MiniMaxH3DecodeAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "sampled_av_latent": (AV_LATENT_TYPE,),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "decode"
    CATEGORY = f"{CATEGORY}/decode"

    def decode(
        self,
        h3_vae_bundle: Mapping[str, Any],
        sampled_av_latent: Mapping[str, Any],
    ):
        is_v2 = (
            isinstance(sampled_av_latent, Mapping)
            and sampled_av_latent.get("schema") == H3_AV_LATENT_SCHEMA_V2
        )
        if is_v2:
            latent = validate_av_latent_v2(sampled_av_latent)
            task = str(latent["task"])
            compatibility = validate_component_compatibility(
                task=task,
                vae=h3_vae_bundle,
                target=latent["target"],
                av_latent=latent,
            )
            vae_wrapper = compatibility["vae"]
            expected_release = latent.get("release_fingerprint")
            actual_release = _wrapper_release_fingerprint(vae_wrapper)
            if expected_release != actual_release:
                raise ValueError("sampled latent 与 Decode VAE 不属于同一 release")
            expected_vae = latent.get("vae_fingerprint")
            actual_vae = _wrapper_component_fingerprint(
                vae_wrapper,
                component_kind="vae",
                path_key="vae_path",
                fingerprint_key="vae_fingerprint",
            )
            if not expected_vae or expected_vae != actual_vae:
                raise ValueError("sampled latent 与 Decode VAE 的 component fingerprint 不一致")
        else:
            if (
                not isinstance(h3_vae_bundle, Mapping)
                or h3_vae_bundle.get("schema") != H3_VAE_SCHEMA
            ):
                raise TypeError(
                    "h3_vae_bundle 端口不是 RunningHub MiniMax H3 Dual VAE Loader 的输出"
                )
            vae_wrapper = h3_vae_bundle
            latent = validate_av_latent(sampled_av_latent)
        if not latent.get("sampled", False):
            raise ValueError(
                "输入仍是空 latent；请先连接 MiniMax H3 Dual Sigma Sampler"
            )
        bundle = vae_wrapper.get("bundle")
        video_vae = _wrapper_value(bundle, "video_vae")
        audio_vae = _wrapper_value(bundle, "audio_vae")
        if video_vae is None or audio_vae is None:
            raise RuntimeError(
                "VAE bundle 必须同时包含 video_vae 和 audio_vae"
            )
        _require_comfy()
        # Decode components sequentially.  Keeping either the 62 GB DiT or
        # both VAEs resident defeats ComfyUI's memory manager on real H3
        # hardware.
        model_management.unload_all_models()
        load_device = model_management.get_torch_device()
        from ..runtime.telemetry import H3Telemetry
        tel = H3Telemetry(device=load_device)
        with tel.stage("video_vae_decode"), _vae_device_session(video_vae, load_device):
            frames = _decode_video(
                video_vae, latent["video"], latent["target"]
            )

        with tel.stage("audio_vae_decode"), _vae_device_session(audio_vae, load_device):
            audio = _decode_audio(
                audio_vae,
                latent["audio"],
                sample_count=int(round(float(latent["target"]["duration_seconds"]) * 32000))
                if latent["target"].get("duration_seconds") is not None
                else None,
            )
        try:
            from ..runtime.sidecar import build_sidecar_meta, write_h3_sidecar
            sample_tel = latent.get("telemetry") if isinstance(latent.get("telemetry"), dict) else {}
            meta = build_sidecar_meta(
                dict(latent),
                extra={
                    "model_root": (
                        vae_wrapper.get("model_root") if isinstance(vae_wrapper, Mapping) else None
                    ),
                    "decode": "RHMiniMaxH3DecodeAV",
                    "telemetry": {**sample_tel, "decode_stages_s": dict(tel.stages), "decode_peak_vram": dict(tel.peak)},
                },
            )
            write_h3_sidecar(meta)
        except Exception as exc:
            logging.getLogger(__name__).warning("H3 sidecar 写入失败: %s", exc)
        return (frames, audio)


