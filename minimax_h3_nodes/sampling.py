"""MiniMax-H3 T2VA sampler used directly inside the ComfyUI process.

This is deliberately not a ``comfy.samplers.KSAMPLER`` adapter.  H3 advances
video and audio with different sigma schedules in one DiT forward, while the
standard KSampler contract supplies one sigma for one state tensor.  Treating
either schedule as the other changes the model's timestep conditioning and the
Euler update.

The equations and packed-row behaviour below are ports of the supplied H3
runtime:

* independent CPU generators re-seeded with the same seed per modality;
* ``x0 = xt + sigma * velocity`` for rectified flow;
* deterministic Euler eta=0 update;
* video shift 12 and audio shift 3 by default;
* one positive forward per sigma interval (the checkpoint is CFG-distilled).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any

try:
    import torch
except ImportError:  # Allows static contract tests without a full Comfy install.
    torch = None  # type: ignore[assignment]

from .contracts import (
    H3_AUDIO_REF_COND_TIMESTEP,
    H3_AUDIO_ROW_WIDTH,
    H3_IMGVID_COND_TIMESTEP,
    H3_VIDEO_CHANNELS,
    H3_VIDEO_PATCH_SIZE,
    H3_VIDEO_ROW_WIDTH,
    validate_av_latent,
    validate_conditioning,
    validate_seed,
    validate_sigma_request,
)


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "MiniMax-H3 Direct 需要 ComfyUI 自带的 PyTorch；当前 Python 环境未安装 torch"
        )
    return torch


def shifted_sigma_schedule(
    *,
    sigma_points: int,
    shift: float,
) -> list[float]:
    """Return H3's float32 time-shift schedule.

    The original ``num_inference_steps=50`` means 50 sigma points and therefore
    49 DiT calls.  The node calls this value ``sigma_points`` to avoid silently
    changing the supplied model's serving contract.
    """

    sigma_points, shift, _ = validate_sigma_request(
        sigma_points=sigma_points,
        video_shift=shift,
        audio_shift=1.0,
    )
    t = _require_torch()
    base = t.linspace(1.0, 0.0, sigma_points, device="cpu", dtype=t.float32)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    shifted = t.unique_consecutive(shifted)
    if sigma_points > 1 and float(shifted[-1]) > 0.0:
        shifted = t.cat((shifted, t.zeros(1, dtype=t.float32)))
    return [float(value) for value in shifted.tolist()]


def patchify_video_latent(latent: Any) -> Any:
    """Pack ``[B,24,T,H,W]`` to H3's ``[rows,96]`` order."""

    t = _require_torch()
    if not isinstance(latent, t.Tensor) or latent.ndim != 5:
        raise ValueError("video latent 必须是 rank-5 tensor [B,24,T,H,W]")
    batch, channels, full_t, full_h, full_w = (int(x) for x in latent.shape)
    if channels != H3_VIDEO_CHANNELS:
        raise ValueError(
            f"video latent channels 必须为 {H3_VIDEO_CHANNELS}，实际为 {channels}"
        )
    pt, ph, pw = H3_VIDEO_PATCH_SIZE
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            "video latent T/H/W 必须可被 patch_size "
            f"{H3_VIDEO_PATCH_SIZE} 整除，实际 shape={list(latent.shape)}"
        )
    out_t, out_h, out_w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(
        batch, channels, out_t, pt, out_h, ph, out_w, pw
    )
    packed = t.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(
        batch * out_t * out_h * out_w,
        channels * pt * ph * pw,
    ).contiguous()


def unpatchify_video_rows(
    rows: Any,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
) -> Any:
    """Unpack H3's ``[rows,96]`` output to ``[B,24,T,H,W]``."""

    t = _require_torch()
    if not isinstance(rows, t.Tensor) or rows.ndim != 2:
        raise ValueError("video rows 必须是 rank-2 tensor")
    pt, ph, pw = H3_VIDEO_PATCH_SIZE
    token_t = int(latent_t) // pt
    token_h = int(latent_h) // ph
    token_w = int(latent_w) // pw
    if int(rows.shape[1]) != H3_VIDEO_ROW_WIDTH:
        raise ValueError(
            f"video row width 必须为 {H3_VIDEO_ROW_WIDTH}，实际为 {int(rows.shape[1])}"
        )
    rows_per_sample = token_t * token_h * token_w
    if rows_per_sample <= 0 or int(rows.shape[0]) % rows_per_sample:
        raise ValueError(
            f"video rows={int(rows.shape[0])} 与 latent geometry 不匹配"
        )
    packed = rows.reshape(
        -1,
        token_t,
        token_h,
        token_w,
        H3_VIDEO_CHANNELS,
        pt,
        ph,
        pw,
    )
    latent = t.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(
        -1,
        H3_VIDEO_CHANNELS,
        int(latent_t),
        int(latent_h),
        int(latent_w),
    ).contiguous()


def unpack_audio_rows(
    rows: Any,
    *,
    audio_t: int,
    audio_channels: int = 2,
) -> Any:
    """Unpack ``[audio_channels*audio_t,32]`` to ``[C,32,T]``."""

    t = _require_torch()
    if not isinstance(rows, t.Tensor) or rows.ndim != 2:
        raise ValueError("audio rows 必须是 rank-2 tensor")
    expected_rows = int(audio_channels) * int(audio_t)
    if tuple(int(x) for x in rows.shape) != (expected_rows, H3_AUDIO_ROW_WIDTH):
        raise ValueError(
            "audio rows shape 必须为 "
            f"({expected_rows},{H3_AUDIO_ROW_WIDTH})，实际为 {tuple(rows.shape)}"
        )
    native = rows.reshape(
        int(audio_channels), int(audio_t), H3_AUDIO_ROW_WIDTH
    )
    return native.permute(0, 2, 1).contiguous()


def rf_velocity_to_x0(state: Any, velocity: Any, timestep: Any) -> Any:
    t = _require_torch()
    if state.shape != velocity.shape:
        raise ValueError(
            f"state/velocity shape 不一致：{tuple(state.shape)} vs {tuple(velocity.shape)}"
        )
    cond_t = t.as_tensor(timestep, device=state.device, dtype=state.dtype)
    while cond_t.ndim < state.ndim:
        cond_t = cond_t.unsqueeze(-1)
    return state + (1.0 - cond_t) * velocity


def euler_eta0_step(
    state: Any,
    denoised: Any,
    *,
    sigma_curr: float,
    sigma_next: float,
    sigma_ratio: Any | None = None,
) -> Any:
    if state.shape != denoised.shape:
        raise ValueError("Euler state/denoised shape 不一致")
    if sigma_curr < 0.0 or sigma_next < 0.0:
        raise ValueError("sigma 不能为负数")
    if sigma_curr == 0.0:
        if sigma_next != 0.0:
            raise ValueError("sigma_curr=0 时 sigma_next 也必须为 0")
        return state
    t = _require_torch()
    compute_dtype = (
        t.float32 if state.dtype in (t.float16, t.bfloat16) else state.dtype
    )
    if sigma_ratio is None:
        ratio = state.new_tensor(
            sigma_next / sigma_curr, dtype=compute_dtype
        )
    else:
        ratio = sigma_ratio.to(device=state.device, dtype=compute_dtype)
    out = (
        ratio * state.to(dtype=compute_dtype)
        + (1.0 - ratio) * denoised.to(dtype=compute_dtype)
    )
    return out.to(dtype=state.dtype)


class H3DenoiseBranch:
    """Step-static packed layout for a single positive T2VA branch."""

    REQUIRED_PACKED_KEYS = (
        "seq_len",
        "img_pos",
        "audio_pos",
        "text_pos",
        "update_mask",
        "img_position_ids",
        "token_tags",
        "cu_seqlens",
    )

    def __init__(
        self,
        *,
        packed: Mapping[str, Any],
        prompt_embeds: Any,
        device: Any,
        transformer: Any | None = None,
    ) -> None:
        t = _require_torch()
        missing = [key for key in self.REQUIRED_PACKED_KEYS if key not in packed]
        if missing:
            raise ValueError(
                "runtime.packing 返回值缺少字段：" + ", ".join(missing)
            )
        seq_len = int(packed["seq_len"])
        self.seq_len = seq_len
        self.img_pos = packed["img_pos"].view(-1).to(t.long)
        self.audio_pos = packed["audio_pos"].view(-1).to(t.long)
        self.update_mask = packed["update_mask"].view(-1).to(t.bool)
        self.audio_update_mask = packed.get("audio_update_mask")
        if self.audio_update_mask is None:
            self.audio_update_mask = t.ones(
                self.audio_pos.shape[0], dtype=t.bool
            )
        else:
            self.audio_update_mask = self.audio_update_mask.view(-1).to(t.bool)

        if prompt_embeds.ndim == 3:
            if int(prompt_embeds.shape[0]) != 1:
                raise ValueError("MiniMax-H3 Direct v0 只支持 batch=1")
            prompt_embeds = prompt_embeds[0]
        text_len = int(packed["text_pos"].view(-1).shape[0])
        if int(prompt_embeds.shape[0]) != text_len:
            raise ValueError(
                f"prompt rows={int(prompt_embeds.shape[0])} 与 packed "
                f"text_len={text_len} 不一致"
            )
        token_tags = packed["token_tags"].view(-1).to(t.long)
        if int(token_tags.shape[0]) != seq_len:
            raise ValueError(
                f"token_tags length={int(token_tags.shape[0])}，应为 {seq_len}"
            )

        self.img_pos_dev = self.img_pos.to(device)
        self.audio_pos_dev = self.audio_pos.to(device)
        self.update_mask_dev = self.update_mask.to(device)
        self.audio_update_mask_dev = self.audio_update_mask.to(device)
        self.img_cond_seq_idx = self.img_pos_dev[~self.update_mask_dev]
        self.audio_target_seq_idx = self.audio_pos_dev[
            self.audio_update_mask_dev
        ]
        self.audio_ref_seq_idx = self.audio_pos_dev[
            ~self.audio_update_mask_dev
        ]
        self.update_row_idx = t.nonzero(self.update_mask_dev).view(-1)
        self.cond_row_idx = t.nonzero(~self.update_mask_dev).view(-1)
        self.audio_update_row_idx = t.nonzero(
            self.audio_update_mask_dev
        ).view(-1)
        self.audio_ref_row_idx = t.nonzero(
            ~self.audio_update_mask_dev
        ).view(-1)
        self.n_video_timestep_rows = (
            seq_len
            - int(self.img_cond_seq_idx.numel())
            - int(self.audio_target_seq_idx.numel())
            - int(self.audio_ref_seq_idx.numel())
        )
        self.x_buffer = t.zeros(
            1,
            seq_len,
            H3_VIDEO_ROW_WIDTH,
            dtype=t.float32,
            device=device,
        )
        self.audio_x_buffer = t.zeros(
            1,
            seq_len,
            H3_AUDIO_ROW_WIDTH,
            dtype=t.float32,
            device=device,
        )
        cu = packed["cu_seqlens"].to(t.int32)
        text_pos_dev = packed["text_pos"].view(-1).to(t.long).to(device)
        refiner_cu = t.tensor(
            [0, text_len, text_len], dtype=t.int32, device=device
        )

        selected_embeds = prompt_embeds
        refined_length: int | None = None
        refine = getattr(transformer, "refine_prompt_embeds", None)
        if callable(refine):
            # This is request-static.  Running the two refiner blocks once
            # avoids repeating them for every one of the 49 default DiT calls.
            with t.inference_mode():
                selected_embeds = refine(
                    prompt_embeds,
                    refiner_cu,
                    device=device,
                )
            refined_length = text_len

        self.static_kwargs: dict[str, Any] = {
            "img_position_ids": packed["img_position_ids"][None].to(device),
            "update_mask": self.update_mask_dev,
            "update_audio_mask": self.audio_update_mask_dev,
            "token_tags": token_tags.to(device),
            "skip_mask_out_condition": False,
            "prompt_embeds": selected_embeds.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {
                "position_ids": self.img_pos_dev
            },
            "packed_seq_params": {
                "cu_seqlens_q": cu.to(device),
                "max_seqlen_q": int(cu[1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": refiner_cu,
                "max_seqlen_q": text_len,
            },
        }
        if refined_length is not None:
            self.static_kwargs["refined_prompt_embeds_length"] = refined_length

    def _step_timesteps(
        self,
        *,
        video_timestep: float,
        audio_timestep: float,
        imgvid_cond_timestep: float,
        audio_ref_cond_timestep: float,
    ) -> tuple[Any, Any]:
        t = _require_torch()
        candidates: list[float] = []
        fill_groups: list[tuple[Any, int]] = []
        base_slot = -1
        if self.n_video_timestep_rows > 0:
            base_slot = len(candidates)
            candidates.append(float(video_timestep))
        for sequence_indices, value in (
            (self.img_cond_seq_idx, imgvid_cond_timestep),
            (self.audio_target_seq_idx, audio_timestep),
            (self.audio_ref_seq_idx, audio_ref_cond_timestep),
        ):
            if sequence_indices.numel() > 0:
                fill_groups.append((sequence_indices, len(candidates)))
                candidates.append(float(value))
        unique_cpu, slot_to_unique = t.unique(
            t.tensor(candidates, dtype=t.float32),
            sorted=True,
            return_inverse=True,
        )
        device = self.img_pos_dev.device
        base_index = int(slot_to_unique[base_slot]) if base_slot >= 0 else 0
        inverse = t.full(
            (self.seq_len,), base_index, dtype=t.long, device=device
        )
        for sequence_indices, slot in fill_groups:
            inverse.index_fill_(
                0, sequence_indices, int(slot_to_unique[slot])
            )
        return unique_cpu.to(device), inverse

    def forward_kwargs(
        self,
        *,
        video_rows: Any,
        audio_rows: Any,
        video_timestep: float,
        audio_timestep: float,
    ) -> dict[str, Any]:
        self.x_buffer[0].index_copy_(0, self.img_pos_dev, video_rows)
        self.audio_x_buffer[0].index_copy_(
            0, self.audio_pos_dev, audio_rows
        )
        imgvid_cond_t = max(
            float(video_timestep), H3_IMGVID_COND_TIMESTEP
        )
        audio_ref_cond_t = max(
            float(audio_timestep), H3_AUDIO_REF_COND_TIMESTEP
        )
        unique, inverse = self._step_timesteps(
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
            imgvid_cond_timestep=imgvid_cond_t,
            audio_ref_cond_timestep=audio_ref_cond_t,
        )
        return {
            **self.static_kwargs,
            "x": self.x_buffer,
            "audio_x": self.audio_x_buffer,
            "unique_timesteps": unique,
            "inverse_indices": inverse,
        }


def _model_device(model: Any) -> Any:
    t = _require_torch()
    # A partially-offloaded INT8 model is expected to have CPU parameters.
    # The loader records the activation/compute device explicitly after
    # ComfyUI has loaded the ModelPatcher; prefer that contract over parameter
    # placement.  Older patchers may only set the compatibility marker.
    for attribute in ("_h3_compute_device", "_comfy_device_marker"):
        explicit = getattr(model, attribute, None)
        if explicit is not None:
            try:
                device = t.device(explicit)
            except (TypeError, RuntimeError):
                continue
            if device.type != "cpu":
                return device
    # Parameters are authoritative: ModelPatcher may leave a compatibility
    # ``model.device`` attribute stale while moving the actual tensors.
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            return next(parameters()).device
        except StopIteration:
            pass
    explicit = getattr(model, "device", None)
    if explicit is not None:
        try:
            return t.device(explicit)
        except (TypeError, RuntimeError):
            pass
    raise RuntimeError(
        "无法确定 H3 DiT 所在设备；model loader 必须先把 transformer "
        "交给 ComfyUI model management 加载"
    )


def sample_t2va(
    *,
    transformer: Any,
    conditioning: Mapping[str, Any],
    av_latent: Mapping[str, Any],
    packed: Mapping[str, Any],
    seed: int,
    sigma_points: int,
    video_shift: float,
    audio_shift: float,
    progress: Callable[[int, int], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run a complete in-process T2VA denoise and return native VAE latents."""

    t = _require_torch()
    clean_conditioning = validate_conditioning(conditioning)
    clean_latent = validate_av_latent(av_latent)
    seed = validate_seed(seed)
    sigma_points, video_shift, audio_shift = validate_sigma_request(
        sigma_points=sigma_points,
        video_shift=video_shift,
        audio_shift=audio_shift,
    )
    if not callable(transformer):
        raise TypeError(
            "H3 transformer 不可调用；model loader 必须返回 "
            "MiniMaxH3DiTModel 或可解析到该模型的 handle"
        )
    target = clean_latent["target"]
    device = _model_device(transformer)
    if device.type == "cpu":
        raise RuntimeError(
            "MiniMax-H3 DiT 仍在 CPU。请使用 model loader 的 auto/cuda "
            "模式并确认 ComfyUI 已成功装载模型；该 33B DiT 不支持 CPU 推理。"
        )

    # Exact source semantics: each modality owns an independent CPU generator,
    # and both generators are re-seeded with the same request seed.
    video_generator = t.Generator(device="cpu").manual_seed(seed)
    raw_video_noise = t.randn(
        tuple(int(x) for x in clean_latent["video"].shape),
        generator=video_generator,
        dtype=t.float32,
        device="cpu",
    )
    video_rows = patchify_video_latent(raw_video_noise).to(device=device)
    audio_generator = t.Generator(device="cpu").manual_seed(seed)
    audio_rows = t.randn(
        int(target["audio_latent_t"]) * 2,
        H3_AUDIO_ROW_WIDTH,
        generator=audio_generator,
        dtype=t.float32,
        device="cpu",
    ).to(device=device)

    branch = H3DenoiseBranch(
        packed=packed,
        prompt_embeds=clean_conditioning["prompt_embeds"],
        device=device,
        transformer=transformer,
    )
    expected_video_rows = int(branch.img_pos.shape[0])
    expected_audio_rows = int(branch.audio_pos.shape[0])
    if tuple(video_rows.shape) != (
        expected_video_rows,
        H3_VIDEO_ROW_WIDTH,
    ):
        raise ValueError(
            f"packed video rows 期望 {(expected_video_rows, H3_VIDEO_ROW_WIDTH)}，"
            f"但初始 latent 产生 {tuple(video_rows.shape)}"
        )
    if tuple(audio_rows.shape) != (
        expected_audio_rows,
        H3_AUDIO_ROW_WIDTH,
    ):
        raise ValueError(
            f"packed audio rows 期望 {(expected_audio_rows, H3_AUDIO_ROW_WIDTH)}，"
            f"但初始 latent 产生 {tuple(audio_rows.shape)}"
        )
    if int((~branch.update_mask).sum()) != 0:
        raise ValueError(
            "T2VA packed layout 不应包含 video condition rows；"
            "请检查 runtime.packing 是否误用了 FL2VA 布局"
        )
    if int((~branch.audio_update_mask).sum()) != 0:
        raise ValueError(
            "T2VA packed layout 不应包含 audio reference rows"
        )

    video_sigmas = shifted_sigma_schedule(
        sigma_points=sigma_points, shift=video_shift
    )
    audio_sigmas = shifted_sigma_schedule(
        sigma_points=sigma_points, shift=audio_shift
    )
    if len(video_sigmas) != len(audio_sigmas):
        raise RuntimeError("video/audio sigma schedule 长度不一致")
    total = len(video_sigmas) - 1
    video_sigma_tensor = t.tensor(
        video_sigmas, dtype=t.float32, device=device
    )
    audio_sigma_tensor = t.tensor(
        audio_sigmas, dtype=t.float32, device=device
    )
    video_ratios = video_sigma_tensor[1:] / video_sigma_tensor[:-1]
    audio_ratios = audio_sigma_tensor[1:] / audio_sigma_tensor[:-1]
    video_timesteps = [
        t.tensor(1.0 - sigma, dtype=t.float32, device=device)
        for sigma in video_sigmas[:-1]
    ]
    audio_timesteps = [
        t.tensor(1.0 - sigma, dtype=t.float32, device=device)
        for sigma in audio_sigmas[:-1]
    ]

    for step in range(total):
        if check_cancelled is not None:
            check_cancelled()
        sigma_video = video_sigmas[step]
        sigma_audio = audio_sigmas[step]
        kwargs = branch.forward_kwargs(
            video_rows=video_rows,
            audio_rows=audio_rows,
            video_timestep=1.0 - sigma_video,
            audio_timestep=1.0 - sigma_audio,
        )
        with t.inference_mode():
            velocity_video, velocity_audio = transformer(**kwargs)
        if tuple(velocity_video.shape) != tuple(video_rows.shape):
            raise ValueError(
                "DiT video velocity shape 不匹配："
                f"{tuple(velocity_video.shape)} vs {tuple(video_rows.shape)}"
            )
        if tuple(velocity_audio.shape) != tuple(audio_rows.shape):
            raise ValueError(
                "DiT audio velocity shape 不匹配："
                f"{tuple(velocity_audio.shape)} vs {tuple(audio_rows.shape)}"
            )
        denoised_video = rf_velocity_to_x0(
            video_rows, velocity_video.float(), video_timesteps[step]
        )
        denoised_audio = rf_velocity_to_x0(
            audio_rows, velocity_audio.float(), audio_timesteps[step]
        )
        video_rows = euler_eta0_step(
            video_rows,
            denoised_video,
            sigma_curr=sigma_video,
            sigma_next=video_sigmas[step + 1],
            sigma_ratio=video_ratios[step],
        )
        audio_rows = euler_eta0_step(
            audio_rows,
            denoised_audio,
            sigma_curr=sigma_audio,
            sigma_next=audio_sigmas[step + 1],
            sigma_ratio=audio_ratios[step],
        )
        if progress is not None:
            progress(step + 1, total)

    video = unpatchify_video_rows(
        video_rows.to(device="cpu"),
        latent_t=int(target["video_latent_t"]),
        latent_h=int(target["video_latent_h"]),
        latent_w=int(target["video_latent_w"]),
    )
    audio = unpack_audio_rows(
        audio_rows.to(device="cpu"),
        audio_t=int(target["audio_latent_t"]),
    )
    return {
        **clean_latent,
        "video": video,
        "audio": audio,
        "seed": seed,
        "sigma_points": sigma_points,
        "video_shift": video_shift,
        "audio_shift": audio_shift,
        "video_sigmas": video_sigmas,
        "audio_sigmas": audio_sigmas,
        "sampled": True,
    }


__all__ = [
    "H3DenoiseBranch",
    "euler_eta0_step",
    "patchify_video_latent",
    "rf_velocity_to_x0",
    "sample_t2va",
    "shifted_sigma_schedule",
    "unpack_audio_rows",
    "unpatchify_video_rows",
]
