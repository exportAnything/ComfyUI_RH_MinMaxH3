"""采样节点。"""
from __future__ import annotations
from ._shared import *  # noqa: F403

class MiniMaxH3FrameRate:  # 实验性：帧率进 adaLN / 可选时序 RoPE（实验性；非上游契约）
    @classmethod
    def INPUT_TYPES(cls):
        from ..runtime.h3_settings import (
            FRAME_RATE_ROPE_FREQ_PROFILES,
            FRAME_RATE_ROPE_SIGMA_PROFILES,
            H3_NATIVE_FPS,
        )
        return {
            "required": {
                "h3_model": (MODEL_TYPE,),
                "frame_rate": (
                    "FLOAT",
                    {
                        "default": H3_NATIVE_FPS,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 0.01,
                        "tooltip": "实验性。不改 target 24fps 时序格；仅调制嵌入/RoPE。",
                    },
                ),
                "adaln": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "把 fps sinusoidal 加到 TimeEmbedder（即使 24 也非 no-op）。",
                    },
                ),
                "temporal_rope": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "按 24/fps 缩放视频行时序 RoPE 低频；24fps 时无效。",
                    },
                ),
            },
            "optional": {
                "rope_end_timestep": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "rope_low_frequency_count": (
                    "INT",
                    {"default": 16, "min": 0, "max": 16},
                ),
                "rope_frequency_profile": (
                    list(FRAME_RATE_ROPE_FREQ_PROFILES),
                    {"default": "hard"},
                ),
                "rope_sigma_profile": (
                    list(FRAME_RATE_ROPE_SIGMA_PROFILES),
                    {"default": "constant"},
                ),
                "rope_sigma_end": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "apply"
    CATEGORY = f"{CATEGORY}/sampling"

    def apply(
        self,
        h3_model: Mapping[str, Any],
        frame_rate: float = 24.0,
        adaln: bool = True,
        temporal_rope: bool = False,
        rope_end_timestep: float = 1.0,
        rope_low_frequency_count: int = 16,
        rope_frequency_profile: str = "hard",
        rope_sigma_profile: str = "constant",
        rope_sigma_end: float = 0.0,
    ):
        from ..runtime.frame_rate import validate_frame_rate_options
        if not isinstance(h3_model, Mapping):
            raise TypeError("h3_model 必须是 MiniMax H3 Model Loader 输出")
        opts = validate_frame_rate_options(
            frame_rate=frame_rate,
            adaln=adaln,
            temporal_rope=temporal_rope,
            rope_end_timestep=rope_end_timestep,
            rope_low_frequency_count=rope_low_frequency_count,
            rope_frequency_profile=rope_frequency_profile,
            rope_sigma_profile=rope_sigma_profile,
            rope_sigma_end=rope_sigma_end,
        )
        out = dict(h3_model)
        out["frame_rate_options"] = opts
        return (out,)


class MiniMaxH3EmptyAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"target": (TARGET_TYPE,)}}

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY}/latent"

    def generate(self, target: Mapping[str, Any]):
        t = _require_torch()
        is_v2 = (
            isinstance(target, Mapping)
            and target.get("schema") == H3_TARGET_SCHEMA_V2
        )
        clean_target = (
            validate_target_v2(target, require_resolved=True)
            if is_v2
            else validate_target(target)
        )
        # Shape carrier only.  The sampler replaces these zeros with the two
        # independently-seeded source-compatible noise streams.
        video = t.zeros(
            1,
            24,
            int(clean_target["video_latent_t"]),
            int(clean_target["video_latent_h"]),
            int(clean_target["video_latent_w"]),
            dtype=t.float32,
            device="cpu",
        )
        audio = t.zeros(
            2,
            32,
            int(clean_target["audio_latent_t"]),
            dtype=t.float32,
            device="cpu",
        )
        return (
            {
                "schema": H3_AV_LATENT_SCHEMA_V2 if is_v2 else H3_AV_LATENT_SCHEMA,
                "task": str(clean_target["task"]),
                **(
                    {
                        "partition": str(clean_target["partition"]),
                        "target_fingerprint": _target_fingerprint(clean_target),
                    }
                    if is_v2
                    else {}
                ),
                "target": clean_target,
                "video": video,
                "audio": audio,
                "sampled": False,
            },
        )


class MiniMaxH3DualSigmaSampler:
    """Task-aware dual-sigma sampler; standard KSampler cannot express H3 state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_model": (MODEL_TYPE,),
                "conditioning": (CONDITIONING_TYPE,),
                "av_latent": (AV_LATENT_TYPE,),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": (1 << 63) - 1,
                    },
                ),
                "sigma_points": (
                    "INT",
                    {
                        "default": H3_DEFAULT_SIGMA_POINTS,
                        "min": 2,
                        "max": 1000,
                        "tooltip": (
                            "遵循原仓库语义：50 个 sigma 点产生 49 次 DiT forward。"
                        ),
                    },
                ),
                "video_shift": (
                    "FLOAT",
                    {
                        "default": H3_DEFAULT_VIDEO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "audio_shift": (
                    "FLOAT",
                    {
                        "default": H3_DEFAULT_AUDIO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "accel": (
                    list(ACCEL_MODE_CHOICES),
                    {
                        "default": ACCEL_OFF,
                        "tooltip": (
                            "单卡近似加速。off=关闭；auto=优先 velocity-cache profile，"
                            "否则 Cache-DiT；minimax-h3-velocity-cache-v1≈少 DiT 次数（推荐单卡）；"
                            "minimax-h3-cache-v1=Cache-DiT（需 cache-dit）；manual-* 手调。"
                            "仅验证 1344×768/124f/50steps/shift12·3。不可作 GT。"
                        ),
                    },
                ),
                "denoise_video": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "False=V2A：av_latent.video 作干净视觉条件（timestep floor），"
                            "只去噪音频。要求 T2VA 布局且 video 非 Empty 全零；"
                            "可用 Encode Video→AV Latent 或 Combine AV Latent 填入视频。"
                        ),
                    },
                ),
            },
            "optional": {
                "cache_dit_rdt": (
                    "FLOAT",
                    {
                        "default": CACHE_DIT_RDT_COOKBOOK,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "仅 accel=manual-cache-dit：残差阈值，越大越快越糙。",
                    },
                ),
                "cache_dit_mc": (
                    "INT",
                    {
                        "default": CACHE_DIT_MC,
                        "min": 1,
                        "max": 32,
                        "tooltip": "仅 accel=manual-cache-dit：最大连续缓存步数。",
                    },
                ),
                "cache_dit_warmup": (
                    "INT",
                    {
                        "default": CACHE_DIT_WARMUP,
                        "min": 0,
                        "max": 64,
                        "tooltip": "仅 accel=manual-cache-dit：warmup 步数。",
                    },
                ),
                "velocity_stride": (
                    "INT",
                    {
                        "default": VELOCITY_STRIDE,
                        "min": 1,
                        "max": 32,
                        "tooltip": "仅 accel=manual-velocity：DiT 刷新步距；1=精确无缓存。",
                    },
                ),
                # 追加在 optional 末尾：旧工作流的 widgets_values 是本列表前缀，
                # 位置序列化不错位
                "sampler_mode": (
                    list(SAMPLER_MODE_CHOICES),
                    {
                        "default": SAMPLER_MODE_EULER,
                        "tooltip": (
                            "euler=上游一阶（sigma_points=50）。"
                            "res_multistep=二阶多步指数积分器（ComfyUI 上游 H3 "
                            "模板同款），建议 sigma_points=21（20 次 DiT）"
                            "≈ euler-50 质量、快约 2.5×；该模式暂强制 accel=off"
                            "（profile 按 euler-50 标定）。"
                        ),
                    },
                ),
                "allow_accel_with_res_multistep": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "允许 res_multistep 叠加 accel（默认关闭）。该组合未标定："
                            "sigma_points=21 时叠加会明显劣化；需同时调高步数才可用，"
                            "且同等画质下并不比 euler+velocity 省。"
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("sampled_av_latent",)
    FUNCTION = "sample"
    CATEGORY = f"{CATEGORY}/sampling"

    def sample(
        self,
        h3_model: Mapping[str, Any],
        conditioning: Mapping[str, Any],
        av_latent: Mapping[str, Any],
        seed: int,
        sigma_points: int,
        video_shift: float,
        audio_shift: float,
        accel: str = ACCEL_OFF,
        denoise_video: bool = True,
        cache_dit_rdt: float = CACHE_DIT_RDT_COOKBOOK,
        cache_dit_mc: int = CACHE_DIT_MC,
        cache_dit_warmup: int = CACHE_DIT_WARMUP,
        velocity_stride: int = VELOCITY_STRIDE,
        sampler_mode: str = SAMPLER_MODE_EULER,
        allow_accel_with_res_multistep: bool = False,
    ):
        is_v2 = (
            isinstance(conditioning, Mapping)
            and conditioning.get("schema") == H3_CONDITIONING_SCHEMA_V2
        )
        progress_offset = 1 if is_v2 else 0
        progress_total = max(1, int(sigma_points) - 1) + progress_offset
        progress_bar = (
            ProgressBar(progress_total) if ProgressBar is not None else None
        )
        if is_v2:
            clean_conditioning = validate_conditioning_v2(conditioning)
            task = str(clean_conditioning["task"])
            clean_latent = validate_av_latent_v2(av_latent, expected_task=task)
            compatibility = validate_component_compatibility(
                task=task,
                model=h3_model,
                target=clean_latent["target"],
                conditioning=clean_conditioning,
                av_latent=clean_latent,
            )
            model_wrapper = compatibility["model"]
            release_fingerprint = _wrapper_release_fingerprint(model_wrapper)
            _wrapper_component_fingerprint(
                model_wrapper,
                component_kind="transformer",
                path_key="transformer_path",
                fingerprint_key="transformer_fingerprint",
            )
            encoded_release = clean_conditioning.get("release_fingerprint")
            if encoded_release != release_fingerprint:
                raise ValueError(
                    "conditioning 的 TE/VAE release 与 sampler DiT release 不一致"
                )
            _check_interrupted()
            packed = _build_v2_packed(
                clean_conditioning,
                clean_latent["target"],
            )
            _advance_progress(progress_bar, 1, progress_total)
        else:
            clean_conditioning = conditioning
            clean_latent = validate_av_latent(av_latent)
            model_wrapper = h3_model
            packed = _build_t2va_packed(
                conditioning["prompt_embeds"],
                clean_latent["target"],
            )

        def update_progress(current: int, total: int):
            if progress_bar is not None:
                progress_bar.update_absolute(
                    int(current) + progress_offset,
                    progress_total,
                )

        def check_cancelled():
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()

        hint = _activation_hint_from_packed(packed, clean_latent.get("target"))
        from contextlib import ExitStack
        from ..runtime.telemetry import H3Telemetry
        from ..runtime.vram_budget import note_observed_activation
        tel = H3Telemetry()
        with ExitStack() as stack:
            with tel.stage("dit_load"):  # 仅覆盖上卡/驻留决策，不含 denoise
                transformer, handle = stack.enter_context(
                    _transformer_session(model_wrapper, activation_hint=hint)
                )
            tel.device = getattr(transformer, "_h3_compute_device", None) or getattr(
                handle, "load_device", None
            )
            fr_opts = None
            if isinstance(model_wrapper, Mapping):
                raw_fr = model_wrapper.get("frame_rate_options")
                if isinstance(raw_fr, Mapping):
                    fr_opts = dict(raw_fr)
            output = sample_h3(  # v1/v2 统一入口
                transformer=transformer,
                conditioning=clean_conditioning,
                av_latent=clean_latent,
                packed=packed,
                seed=int(seed),
                sigma_points=int(sigma_points),
                video_shift=float(video_shift),
                audio_shift=float(audio_shift),
                denoise_video=bool(denoise_video),
                frame_rate_options=fr_opts,
                progress=update_progress,
                check_cancelled=check_cancelled,
                sampler_mode=str(sampler_mode),
                allow_accel_with_res_multistep=bool(allow_accel_with_res_multistep),
                accel=str(accel),
                cache_dit_rdt=float(cache_dit_rdt),
                cache_dit_mc=int(cache_dit_mc),
                cache_dit_warmup=int(cache_dit_warmup),
                velocity_stride=int(velocity_stride),
                telemetry=tel,
            )
        residency = getattr(handle, "residency_mode", None) or getattr(
            transformer, "_h3_residency_mode", "unknown"
        )
        output["residency_mode"] = residency
        sample_tel = output.get("telemetry") if isinstance(output.get("telemetry"), dict) else tel.summary()
        sample_tel["residency_mode"] = residency
        sample_tel["accel"] = str(accel)
        output["telemetry"] = sample_tel
        try:  # 峰值 − 权重 ≈ 激活，写回 EWMA
            peak = int((sample_tel.get("peak_vram") or {}).get("allocated") or 0)
            weights = int(getattr(handle, "_last_weight_bytes", 0) or 0)
            if not weights:
                from ..runtime.model_loader import _model_nbytes
                weights = int(_model_nbytes(getattr(handle, "model", transformer)))
            if peak > 0:
                note_observed_activation(hint, observed_peak=max(0, peak - weights))
        except Exception:
            pass
        if is_v2:
            output.update(
                {
                    "schema": H3_AV_LATENT_SCHEMA_V2,
                    "partition": partition_for_task(str(clean_conditioning["task"])),
                    "release_fingerprint": release_fingerprint,
                    "target_fingerprint": _target_fingerprint(clean_latent["target"]),
                    "vae_fingerprint": clean_conditioning.get("vae_fingerprint"),
                }
            )
            validate_av_latent_v2(output, expected_task=str(clean_conditioning["task"]))
        return (output,)


def _av_shell(base: Mapping[str, Any], *, video, audio, sampled: bool = False) -> dict[str, Any]:
    """从已有 av_latent 复制元数据并替换模态张量。"""
    out = {
        "schema": base["schema"],
        "task": str(base["task"]),
        "target": base["target"],
        "video": video,
        "audio": audio,
        "sampled": bool(sampled),
    }
    for key in ("partition", "target_fingerprint", "release_fingerprint", "vae_fingerprint"):
        if key in base:
            out[key] = base[key]
    return out


class MiniMaxH3SeparateAVLatent:  # 拆成「仅视频 / 仅音频」壳，便于 V2A 重组合
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"av_latent": (AV_LATENT_TYPE,)}}

    RETURN_TYPES = (AV_LATENT_TYPE, AV_LATENT_TYPE)
    RETURN_NAMES = ("video_av_latent", "audio_av_latent")
    FUNCTION = "separate"
    CATEGORY = f"{CATEGORY}/latent"

    def separate(self, av_latent: Mapping[str, Any]):
        t = _require_torch()
        is_v2 = isinstance(av_latent, Mapping) and av_latent.get("schema") == H3_AV_LATENT_SCHEMA_V2
        clean = validate_av_latent_v2(av_latent) if is_v2 else validate_av_latent(av_latent)
        z_a = t.zeros_like(clean["audio"])
        z_v = t.zeros_like(clean["video"])
        return (
            _av_shell(clean, video=clean["video"], audio=z_a, sampled=False),
            _av_shell(clean, video=z_v, audio=clean["audio"], sampled=False),
        )


class MiniMaxH3CombineAVLatent:  # 合并视频/音频壳；形状必须同 target
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_av_latent": (AV_LATENT_TYPE,),
                "audio_av_latent": (AV_LATENT_TYPE,),
            }
        }

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "combine"
    CATEGORY = f"{CATEGORY}/latent"

    def combine(self, video_av_latent: Mapping[str, Any], audio_av_latent: Mapping[str, Any]):
        is_v2 = (
            isinstance(video_av_latent, Mapping)
            and video_av_latent.get("schema") == H3_AV_LATENT_SCHEMA_V2
        )
        video_clean = (
            validate_av_latent_v2(video_av_latent)
            if is_v2
            else validate_av_latent(video_av_latent)
        )
        audio_clean = (
            validate_av_latent_v2(audio_av_latent, expected_task=str(video_clean["task"]))
            if is_v2
            else validate_av_latent(audio_av_latent)
        )
        if str(audio_clean["task"]) != str(video_clean["task"]):
            raise ValueError("Combine AV：video/audio 壳 task 不一致")
        vt, at = video_clean["target"], audio_clean["target"]
        for key in (
            "video_latent_t", "video_latent_h", "video_latent_w", "audio_latent_t",
        ):
            if int(vt[key]) != int(at[key]):
                raise ValueError(f"Combine AV：target.{key} 不一致")
        if float(video_clean["video"].detach().abs().max()) <= 0.0:
            raise ValueError("Combine AV：video 壳不能是全零（请先 Encode Video→AV）")
        return (
            _av_shell(
                video_clean,
                video=video_clean["video"],
                audio=audio_clean["audio"],
                sampled=False,
            ),
        )


class MiniMaxH3EncodeVideoAVLatent:  # IMAGE→video latent 写入 Empty AV 壳（V2A 入口）
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "av_latent": (AV_LATENT_TYPE,),
                "frames": ("IMAGE",),
                "seed": ("INT", {"default": 42, "min": 0, "max": (1 << 63) - 1}),
            }
        }

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/latent"

    def encode(
        self,
        h3_vae_bundle: Mapping[str, Any],
        av_latent: Mapping[str, Any],
        frames,
        seed: int = 42,
    ):
        t = _require_torch()
        is_v2 = isinstance(av_latent, Mapping) and av_latent.get("schema") == H3_AV_LATENT_SCHEMA_V2
        if is_v2:
            clean = validate_av_latent_v2(av_latent)
            compatibility = validate_component_compatibility(
                task=str(clean["task"]),
                vae=h3_vae_bundle,
                target=clean["target"],
                av_latent=clean,
            )
            vae_wrapper = compatibility["vae"]
        else:
            if not isinstance(h3_vae_bundle, Mapping) or h3_vae_bundle.get("schema") != H3_VAE_SCHEMA:
                raise TypeError(
                    "h3_vae_bundle 端口不是 RunningHub MiniMax H3 Dual VAE Loader 的输出"
                )
            vae_wrapper, clean = h3_vae_bundle, validate_av_latent(av_latent)
        video_vae = _wrapper_value(vae_wrapper.get("bundle"), "video_vae")
        if video_vae is None:
            raise RuntimeError("VAE bundle 缺少 video_vae")
        _require_comfy()
        model_management.unload_all_models()
        load_device = model_management.get_torch_device()
        pixels = frames if isinstance(frames, t.Tensor) else t.as_tensor(frames)
        if pixels.ndim == 4:  # [T,H,W,C] Comfy IMAGE
            pixels = pixels.unsqueeze(0)
        target = clean["target"]
        need_t = int(target["frame_count"])
        if int(pixels.shape[1]) < need_t:
            raise ValueError(f"frames T={int(pixels.shape[1])} < target.frame_count={need_t}")
        pixels = pixels[:, :need_t, : int(target["height"]), : int(target["width"]), :3]
        with _vae_device_session(video_vae, load_device):
            encoded = video_vae.encode(pixels, process_image=False, seed=int(seed))
        if not isinstance(encoded, t.Tensor):
            raise TypeError("video VAE encode 必须返回 tensor")
        expect = (
            1, 24,
            int(target["video_latent_t"]),
            int(target["video_latent_h"]),
            int(target["video_latent_w"]),
        )
        if tuple(int(x) for x in encoded.shape) != expect:
            raise ValueError(f"编码 video latent shape={tuple(encoded.shape)}，期望 {expect}")
        out = _av_shell(
            clean,
            video=encoded.detach().to(device="cpu", dtype=t.float32).contiguous(),
            audio=t.zeros_like(clean["audio"]),
            sampled=False,
        )
        if is_v2:
            out["vae_fingerprint"] = _wrapper_component_fingerprint(
                vae_wrapper,
                component_kind="vae",
                path_key="vae_path",
                fingerprint_key="vae_fingerprint",
            )
            out["release_fingerprint"] = _wrapper_release_fingerprint(vae_wrapper)
        return (out,)


