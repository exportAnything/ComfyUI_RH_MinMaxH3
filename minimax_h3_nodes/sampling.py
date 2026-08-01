"""MiniMax-H3 target and conditional sampler used inside the ComfyUI process.

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
    H3_AV_LATENT_SCHEMA,
    H3_AV_LATENT_SCHEMA_V2,
    H3_CONDITIONING_SCHEMA,
    H3_CONDITIONING_SCHEMA_V2,
    H3_IMGVID_COND_TIMESTEP,
    H3_VIDEO_CHANNELS,
    H3_VIDEO_PATCH_SIZE,
    H3_VIDEO_ROW_WIDTH,
    H3_TASK_T2VA,
    validate_av_latent,
    validate_av_latent_v2,
    validate_conditioning,
    validate_conditioning_v2,
    validate_seed,
    validate_sigma_request,
    validate_task_partition,
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


def _condition_noise_level(value: Any, name: str) -> float:
    """Validate a rectified-flow condition timestep/noise mixture."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是 0 到 1 之间的数字")
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} 必须在 [0,1]，实际为 {out!r}")
    return out


def noise_visual_condition_rows(
    clean_rows: Any,
    *,
    condition_shapes: Sequence[Sequence[int]],
    target_latent_t: int,
    seed: int,
    noise_level: float = H3_IMGVID_COND_TIMESTEP,
) -> Any:
    """Materialize official H3 visual-condition noise without mutating cache.

    Each ordered visual condition restarts an independent CPU generator at the
    request seed.  The draw includes target temporal noise before slicing the
    condition prefix; this deliberately mirrors the supplied H3 runtime rather
    than drawing one concatenated row tensor.
    """

    t = _require_torch()
    level = _condition_noise_level(noise_level, "visual condition noise")
    if not isinstance(clean_rows, t.Tensor):
        raise TypeError("visual condition rows 必须是 torch.Tensor")
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != H3_VIDEO_ROW_WIDTH:
        raise ValueError(
            "visual condition rows 必须是 [N,96]，实际为 "
            f"{tuple(int(x) for x in clean_rows.shape)}"
        )
    if isinstance(target_latent_t, bool) or int(target_latent_t) <= 0:
        raise ValueError("target_latent_t 必须是正整数")

    parsed: list[tuple[int, int, int]] = []
    expected_rows = 0
    for index, raw_shape in enumerate(condition_shapes):
        if len(raw_shape) != 3:
            raise ValueError(
                f"visual_condition_shapes[{index}] 必须是 (T,H,W)"
            )
        latent_t, latent_h, latent_w = (int(value) for value in raw_shape)
        if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
            raise ValueError(
                f"visual_condition_shapes[{index}] 必须全部为正数"
            )
        if latent_h % H3_VIDEO_PATCH_SIZE[1] or latent_w % H3_VIDEO_PATCH_SIZE[2]:
            raise ValueError(
                f"visual_condition_shapes[{index}] 的 H/W 必须可被 2 整除"
            )
        parsed.append((latent_t, latent_h, latent_w))
        expected_rows += (
            latent_t
            * (latent_h // H3_VIDEO_PATCH_SIZE[1])
            * (latent_w // H3_VIDEO_PATCH_SIZE[2])
        )
    if not parsed:
        if int(clean_rows.shape[0]) != 0:
            raise ValueError("有 visual condition rows 时必须提供 shape 元数据")
        return clean_rows.detach().clone().to(dtype=t.float32).contiguous()
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"visual condition rows={int(clean_rows.shape[0])}，"
            f"但 shape 元数据推导为 {expected_rows}"
        )

    # The 1.0 path still clones: callers may pin/reset the returned anchors in
    # place and must never corrupt cached clean VAE rows.
    if level == 1.0:
        return clean_rows.detach().clone().to(dtype=t.float32).contiguous()

    output: list[Any] = []
    offset = 0
    full_t = int(target_latent_t) + len(parsed)
    mix = t.tensor(level, dtype=t.float32, device=clean_rows.device)
    for index, (latent_t, latent_h, latent_w) in enumerate(parsed):
        if full_t < latent_t:
            raise ValueError(
                f"visual condition {index} latent_t={latent_t} 超过 "
                f"noise draw temporal length={full_t}"
            )
        generator = t.Generator(device="cpu").manual_seed(int(seed))
        noise = t.randn(
            1,
            H3_VIDEO_CHANNELS,
            full_t,
            latent_h,
            latent_w,
            generator=generator,
            dtype=t.float32,
            device="cpu",
        )[:, :, :latent_t]
        noise_rows = patchify_video_latent(noise).to(
            device=clean_rows.device, dtype=t.float32
        )
        row_count = int(noise_rows.shape[0])
        clean_part = clean_rows[offset : offset + row_count].detach().to(
            dtype=t.float32
        )
        output.append(mix * clean_part + (1.0 - mix) * noise_rows)
        offset += row_count
    return t.cat(output, dim=0).contiguous()


def noise_audio_reference_rows(
    clean_rows: Any,
    *,
    reference_audio_t: Sequence[int],
    seed: int,
    noise_level: float = H3_AUDIO_REF_COND_TIMESTEP,
) -> Any:
    """Materialize ordered stereo reference-audio anchors on CPU in fp32."""

    t = _require_torch()
    level = _condition_noise_level(noise_level, "audio reference noise")
    if not isinstance(clean_rows, t.Tensor):
        raise TypeError("audio reference rows 必须是 torch.Tensor")
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != H3_AUDIO_ROW_WIDTH:
        raise ValueError(
            "audio reference rows 必须是 [N,32]，实际为 "
            f"{tuple(int(x) for x in clean_rows.shape)}"
        )
    parsed = [int(value) for value in reference_audio_t]
    if any(value <= 0 for value in parsed):
        raise ValueError("audio_reference_t 中每个值都必须是正整数")
    expected_rows = 2 * sum(parsed)
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"audio reference rows={int(clean_rows.shape[0])}，"
            f"但 audio_reference_t 推导为 {expected_rows}"
        )
    if not parsed:
        if int(clean_rows.shape[0]) != 0:
            raise ValueError("有 audio reference rows 时必须提供长度元数据")
        return clean_rows.detach().clone().to(dtype=t.float32).contiguous()
    if level == 1.0:
        return clean_rows.detach().clone().to(dtype=t.float32).contiguous()

    output: list[Any] = []
    offset = 0
    mix = t.tensor(level, dtype=t.float32, device="cpu")
    for audio_t in parsed:
        row_count = 2 * audio_t
        clean_part = clean_rows[offset : offset + row_count].detach().to(
            device="cpu", dtype=t.float32
        )
        # Official Ref2VA restarts seed+1 for every ordered audio condition.
        generator = t.Generator(device="cpu").manual_seed(int(seed) + 1)
        noise = t.randn(
            clean_part.shape,
            generator=generator,
            dtype=t.float32,
            device="cpu",
        )
        output.append(mix * clean_part + (1.0 - mix) * noise)
        offset += row_count
    return (
        t.cat(output, dim=0)
        .to(device=clean_rows.device, dtype=t.float32)
        .contiguous()
    )


def _require_finite_step_tensor(
    value: Any,
    *,
    task: str,
    step: int,
    modality: str,
    phase: str,
) -> None:
    """Stop non-finite model output before it can become a snow-noise video."""

    t = _require_torch()
    if not isinstance(value, t.Tensor):
        raise TypeError(
            f"MiniMax-H3 task={task} step={step} modality={modality} "
            f"{phase} 不是 tensor"
        )
    if not bool(t.isfinite(value).all()):
        raise FloatingPointError(
            f"MiniMax-H3 task={task} step={step} modality={modality} "
            f"{phase} 出现 NaN/Inf，已停止解码"
        )


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
        if int(self.update_mask.numel()) != int(self.img_pos.numel()):
            raise ValueError(
                "update_mask length 必须等于 img_pos rows："
                f"{int(self.update_mask.numel())} != {int(self.img_pos.numel())}"
            )
        if int(self.audio_update_mask.numel()) != int(self.audio_pos.numel()):
            raise ValueError(
                "audio_update_mask length 必须等于 audio_pos rows："
                f"{int(self.audio_update_mask.numel())} != "
                f"{int(self.audio_pos.numel())}"
            )

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
        imgvid_cond_timestep_floor: float = H3_IMGVID_COND_TIMESTEP,
        audio_ref_cond_timestep_floor: float = H3_AUDIO_REF_COND_TIMESTEP,
    ) -> dict[str, Any]:
        self.x_buffer[0].index_copy_(0, self.img_pos_dev, video_rows)
        self.audio_x_buffer[0].index_copy_(
            0, self.audio_pos_dev, audio_rows
        )
        imgvid_cond_t = max(
            float(video_timestep), float(imgvid_cond_timestep_floor)
        )
        audio_ref_cond_t = max(
            float(audio_timestep), float(audio_ref_cond_timestep_floor)
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


def _validate_generic_sampler_inputs(
    *,
    conditioning: Mapping[str, Any],
    av_latent: Mapping[str, Any],
    packed: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any, Any]:
    """Validate task-neutral sampler tensors without a T2VA-only contract."""

    t = _require_torch()
    if not isinstance(conditioning, Mapping):
        raise TypeError("conditioning 必须是 mapping")
    if not isinstance(av_latent, Mapping):
        raise TypeError("av_latent 必须是 mapping")
    if not isinstance(packed, Mapping):
        raise TypeError("packed conditioning 必须是 mapping")

    raw_target = av_latent.get("target")
    target_task = raw_target.get("task") if isinstance(raw_target, Mapping) else None
    task_values = {
        str(value).strip().lower()
        for value in (
            conditioning.get("task"),
            av_latent.get("task"),
            target_task,
            packed.get("task"),
        )
        if value is not None and str(value).strip()
    }
    if len(task_values) > 1:
        raise ValueError(
            "conditioning/target/latent/packed task 不一致："
            + ", ".join(sorted(task_values))
        )
    task = next(iter(task_values), None)

    target_partition = (
        raw_target.get("partition") if isinstance(raw_target, Mapping) else None
    )
    partition_values = {
        str(value).strip().lower()
        for value in (
            conditioning.get("partition"),
            av_latent.get("partition"),
            target_partition,
            packed.get("partition"),
        )
        if value is not None and str(value).strip()
    }
    if len(partition_values) > 1:
        raise ValueError(
            "conditioning/target/latent/packed partition 不一致："
            + ", ".join(sorted(partition_values))
        )
    if task is not None and partition_values:
        validate_task_partition(task, next(iter(partition_values)))

    conditioning_schema = conditioning.get("schema")
    latent_schema = av_latent.get("schema")
    if (
        conditioning_schema == H3_CONDITIONING_SCHEMA_V2
        and latent_schema == H3_AV_LATENT_SCHEMA_V2
    ):
        if task is None:
            raise ValueError("V2 conditioning 缺少 task")
        clean_conditioning = validate_conditioning_v2(
            conditioning, expected_task=task
        )
        clean_latent = validate_av_latent_v2(av_latent, expected_task=task)
    elif (
        conditioning_schema == H3_CONDITIONING_SCHEMA
        and latent_schema == H3_AV_LATENT_SCHEMA
    ):
        # V1 is retained solely for the already shipped T2VA workflow.  A
        # conditional task must never fall back to structural duck typing.
        if task != H3_TASK_T2VA:
            raise ValueError(
                "FL2VA/Ref2VA 只接受 conditioning/av_latent v2 schema；"
                f"实际 task={task!r}"
            )
        clean_conditioning = validate_conditioning(conditioning)
        clean_latent = validate_av_latent(av_latent)
    else:
        raise ValueError(
            "conditioning/av_latent schema 必须同时为 MiniMax-H3 v2，"
            "或同时为原 T2VA v1；实际为 "
            f"{conditioning_schema!r} / {latent_schema!r}"
        )

    prompt_embeds = clean_conditioning.get("prompt_embeds")
    if not isinstance(prompt_embeds, t.Tensor):
        raise TypeError("conditioning 缺少 prompt_embeds tensor")
    if prompt_embeds.ndim == 3:
        if int(prompt_embeds.shape[0]) != 1:
            raise ValueError("MiniMax-H3 sampler 只支持 batch=1")
    elif prompt_embeds.ndim != 2:
        raise ValueError("prompt_embeds 必须是 [L,D] 或 [1,L,D]")

    video = clean_latent.get("video")
    audio = clean_latent.get("audio")
    if not isinstance(video, t.Tensor) or video.ndim != 5:
        raise ValueError("av_latent.video 必须是 [1,24,T,H,W] tensor")
    if tuple(int(x) for x in video.shape[:2]) != (1, H3_VIDEO_CHANNELS):
        raise ValueError(
            "av_latent.video 前两维必须是 [1,24]，实际为 "
            f"{tuple(int(x) for x in video.shape[:2])}"
        )
    if not isinstance(audio, t.Tensor) or audio.ndim != 3:
        raise ValueError("av_latent.audio 必须是 [2,32,T] tensor")
    if tuple(int(x) for x in audio.shape[:2]) != (2, H3_AUDIO_ROW_WIDTH):
        raise ValueError(
            "av_latent.audio 前两维必须是 [2,32]，实际为 "
            f"{tuple(int(x) for x in audio.shape[:2])}"
        )

    latent_t, latent_h, latent_w = (int(x) for x in video.shape[2:])
    if (
        latent_t % H3_VIDEO_PATCH_SIZE[0]
        or latent_h % H3_VIDEO_PATCH_SIZE[1]
        or latent_w % H3_VIDEO_PATCH_SIZE[2]
    ):
        raise ValueError(
            "target video latent T/H/W 必须可被 patch size "
            f"{H3_VIDEO_PATCH_SIZE} 整除"
        )
    audio_t = int(audio.shape[2])
    if audio_t <= 0:
        raise ValueError("target audio latent T 必须是正整数")

    target = clean_latent.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("av_latent 缺少 target mapping")
    for key, actual in (
        ("video_latent_t", latent_t),
        ("video_latent_h", latent_h),
        ("video_latent_w", latent_w),
        ("audio_latent_t", audio_t),
    ):
        declared = target.get(key)
        if declared is not None and int(declared) != actual:
            raise ValueError(
                f"target.{key}={declared} 与 latent tensor={actual} 不一致"
            )

    packed_latent_shape = packed.get("latent_shape")
    if packed_latent_shape is not None:
        expected = (latent_t, latent_h, latent_w, H3_VIDEO_CHANNELS)
        if tuple(int(x) for x in packed_latent_shape) != expected:
            raise ValueError(
                f"packed latent_shape 必须为 {expected}，"
                f"实际为 {tuple(packed_latent_shape)}"
            )
    packed_audio_shape = packed.get("audio_shape")
    if packed_audio_shape is not None:
        expected_audio = (2, H3_AUDIO_ROW_WIDTH, audio_t)
        if tuple(int(x) for x in packed_audio_shape) != expected_audio:
            raise ValueError(
                f"packed audio_shape 必须为 {expected_audio}，"
                f"实际为 {tuple(packed_audio_shape)}"
            )

    return clean_conditioning, clean_latent, video, audio


def _validate_conditional_layout(branch: H3DenoiseBranch) -> None:
    """Fail before allocating target noise when packed row order is invalid."""

    t = _require_torch()
    if int(branch.update_mask.numel()) != int(branch.img_pos.numel()):
        raise ValueError(
            "update_mask length 必须等于 img_pos rows："
            f"{int(branch.update_mask.numel())} != {int(branch.img_pos.numel())}"
        )
    if int(branch.audio_update_mask.numel()) != int(branch.audio_pos.numel()):
        raise ValueError(
            "audio_update_mask length 必须等于 audio_pos rows："
            f"{int(branch.audio_update_mask.numel())} != "
            f"{int(branch.audio_pos.numel())}"
        )
    if not bool(branch.update_mask.any()):
        raise ValueError("packed layout 至少需要一个 target video row")
    if not bool(branch.audio_update_mask.any()):
        raise ValueError("packed layout 至少需要一个 target audio row")

    for name, positions in (
        ("img_pos", branch.img_pos),
        ("audio_pos", branch.audio_pos),
    ):
        if positions.numel() == 0:
            raise ValueError(f"{name} 不能为空")
        if bool((positions < 0).any()) or bool((positions >= branch.seq_len).any()):
            raise ValueError(f"{name} 包含 seq_len 范围外的位置")
        if positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError(f"{name} 必须严格递增且不能重复")
    if bool(t.isin(branch.img_pos, branch.audio_pos).any()):
        raise ValueError("img_pos 与 audio_pos 不能重叠")

    # Official FL2VA/Ref2VA layouts concatenate condition rows before target
    # rows inside each modality.  Enforcing this catches anchor-order drift
    # before clean rows are scattered into the wrong positions.
    for name, mask in (
        ("update_mask", branch.update_mask),
        ("audio_update_mask", branch.audio_update_mask),
    ):
        target_indices = t.nonzero(mask).view(-1)
        first_target = int(target_indices[0])
        if bool(mask[:first_target].any()) or not bool(mask[first_target:].all()):
            raise ValueError(f"{name} 必须按 [condition rows | target rows] 排列")


def _validate_condition_block_metadata(
    *,
    packed: Mapping[str, Any],
    branch: H3DenoiseBranch,
    conditioning: Mapping[str, Any],
    task: str,
    visual_condition_rows: Any | None,
    audio_reference_rows: Any | None,
    visual_condition_shapes: Sequence[Sequence[int]] | None,
    audio_reference_t: Sequence[int] | None,
) -> None:
    """Cross-check ordered block metadata against every frozen row stream."""

    t = _require_torch()
    raw_blocks = packed.get("condition_blocks")
    if raw_blocks is None and task == H3_TASK_T2VA:
        raw_blocks = ()
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)):
        raise ValueError("packed.condition_blocks 必须是有序列表")
    blocks = list(raw_blocks)

    raw_conditions = conditioning.get("conditions", [])
    if raw_conditions is None:
        raw_conditions = []
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ):
        raise ValueError("conditioning.conditions 必须是有序列表")
    conditions = list(raw_conditions)

    if task == H3_TASK_T2VA:
        if blocks or conditions:
            raise ValueError("T2VA 不允许 condition_blocks")
    else:
        if not blocks:
            raise ValueError(f"{task.upper()} packed.condition_blocks 不能为空")
        if len(blocks) != len(conditions):
            raise ValueError(
                "packed.condition_blocks 与 conditioning.conditions 数量不一致："
                f"{len(blocks)} != {len(conditions)}"
            )

    def metadata_int(
        value: Any,
        path: str,
        *,
        allow_zero: bool = False,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} 必须是整数")
        invalid = value < 0 if allow_zero else value <= 0
        if invalid:
            qualifier = "非负" if allow_zero else "正"
            raise ValueError(f"{path} 必须是{qualifier}整数")
        return int(value)

    derived_visual_shapes: list[tuple[int, int, int]] = []
    derived_audio_t: list[int] = []
    visual_rows_per_block: list[int] = []
    audio_rows_per_block: list[int] = []
    for index, raw_block in enumerate(blocks):
        path = f"packed.condition_blocks[{index}]"
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"{path} 必须是对象")
        condition_index = raw_block.get("condition_index")
        if (
            isinstance(condition_index, bool)
            or not isinstance(condition_index, int)
            or condition_index != index
        ):
            raise ValueError(
                f"{path}.condition_index 必须按请求顺序连续排列，"
                f"期望 {index}，实际为 {condition_index!r}"
            )
        kind = raw_block.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path}.kind 必须是字符串")
        if index >= len(conditions) or not isinstance(conditions[index], Mapping):
            raise ValueError(f"conditioning.conditions[{index}] 必须是对象")
        if task == "fl2va":
            expected_kind = "keyframe"
        elif task == "ref2va":
            expected_kind = conditions[index].get("type")
        else:
            expected_kind = None
        if kind != expected_kind:
            raise ValueError(
                f"{path}.kind 顺序不匹配：期望 {expected_kind!r}，实际为 {kind!r}"
            )
        expected_condition_index = conditions[index].get("condition_index")
        if expected_condition_index != condition_index:
            raise ValueError(
                f"{path}.condition_index 与 conditioning.conditions[{index}] "
                f"不一致：{condition_index!r} != {expected_condition_index!r}"
            )

        visual_count = 0
        if kind in {"keyframe", "image", "video", "video_audio"}:
            if kind in {"keyframe", "image"}:
                declared_latent_t = raw_block.get("latent_t")
                if declared_latent_t is not None and metadata_int(
                    declared_latent_t, f"{path}.latent_t"
                ) != 1:
                    raise ValueError(f"{path}.latent_t 必须为 1")
                latent_t = 1
            else:
                latent_t = metadata_int(
                    raw_block.get("latent_t"), f"{path}.latent_t"
                )
            latent_h = metadata_int(raw_block.get("latent_h"), f"{path}.latent_h")
            latent_w = metadata_int(raw_block.get("latent_w"), f"{path}.latent_w")
            if latent_h % H3_VIDEO_PATCH_SIZE[1] or latent_w % H3_VIDEO_PATCH_SIZE[2]:
                raise ValueError(f"{path} visual latent H/W 必须可被 2 整除")
            visual_count = (
                latent_t
                * (latent_h // H3_VIDEO_PATCH_SIZE[1])
                * (latent_w // H3_VIDEO_PATCH_SIZE[2])
            )
            derived_visual_shapes.append((latent_t, latent_h, latent_w))
        declared_visual_count = raw_block.get("visual_rows_count")
        if declared_visual_count is not None and metadata_int(
            declared_visual_count,
            f"{path}.visual_rows_count",
            allow_zero=True,
        ) != visual_count:
            raise ValueError(f"{path}.visual_rows_count 与 latent shape 不一致")
        visual_rows_per_block.append(visual_count)

        audio_count = 0
        if kind in {"audio", "video", "video_audio"}:
            ref_t = metadata_int(
                raw_block.get("ref_audio_t"),
                f"{path}.ref_audio_t",
                allow_zero=kind == "video",
            )
            if ref_t:
                derived_audio_t.append(ref_t)
                audio_count = 2 * ref_t
        elif raw_block.get("ref_audio_t") not in (None, 0):
            raise ValueError(f"{path} 不允许 ref_audio_t")
        declared_audio_count = raw_block.get("audio_rows_count")
        if declared_audio_count is not None and metadata_int(
            declared_audio_count,
            f"{path}.audio_rows_count",
            allow_zero=True,
        ) != audio_count:
            raise ValueError(f"{path}.audio_rows_count 与 ref_audio_t 不一致")
        audio_rows_per_block.append(audio_count)

    def strict_shapes(raw: Any) -> list[tuple[int, int, int]]:
        if raw is None:
            return []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("visual_condition_shapes 必须是有序列表")
        output: list[tuple[int, int, int]] = []
        for index, shape in enumerate(raw):
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
                raise ValueError(f"visual_condition_shapes[{index}] 必须是 (T,H,W)")
            if len(shape) != 3:
                raise ValueError(f"visual_condition_shapes[{index}] 必须是 (T,H,W)")
            output.append(
                tuple(
                    metadata_int(value, f"visual_condition_shapes[{index}][{axis}]")
                    for axis, value in enumerate(shape)
                )
            )
        return output

    supplied_shapes = strict_shapes(visual_condition_shapes)
    if supplied_shapes != derived_visual_shapes:
        raise ValueError(
            "visual_condition_shapes 与 condition_blocks 的 visual 顺序/shape 不一致："
            f"{supplied_shapes!r} != {derived_visual_shapes!r}"
        )
    if audio_reference_t is None:
        supplied_audio_t: list[int] = []
    elif isinstance(audio_reference_t, Sequence) and not isinstance(
        audio_reference_t, (str, bytes)
    ):
        supplied_audio_t = [
            metadata_int(value, f"audio_reference_t[{index}]")
            for index, value in enumerate(audio_reference_t)
        ]
    else:
        raise ValueError("audio_reference_t 必须是有序列表")
    if supplied_audio_t != derived_audio_t:
        raise ValueError(
            "audio_reference_t 与 condition_blocks 的 audio 顺序/长度不一致："
            f"{supplied_audio_t!r} != {derived_audio_t!r}"
        )

    expected_visual_rows = sum(visual_rows_per_block)
    expected_audio_rows = sum(audio_rows_per_block)
    for name, rows, expected, width in (
        ("visual_cond_rows", visual_condition_rows, expected_visual_rows, 96),
        ("audio_ref_rows", audio_reference_rows, expected_audio_rows, 32),
    ):
        if expected == 0 and rows is None:
            continue
        if not isinstance(rows, t.Tensor) or rows.ndim != 2:
            raise ValueError(f"{name} 必须是 rank-2 tensor")
        if tuple(int(value) for value in rows.shape) != (expected, width):
            raise ValueError(
                f"{name} shape={tuple(rows.shape)} 与 condition_blocks 推导的 "
                f"{(expected, width)} 不一致"
            )
    if expected_visual_rows != int((~branch.update_mask).sum()):
        raise ValueError("condition_blocks visual rows 与 update_mask false rows 不一致")
    if expected_audio_rows != int((~branch.audio_update_mask).sum()):
        raise ValueError(
            "condition_blocks audio rows 与 audio_update_mask false rows 不一致"
        )


def _require_accelerated_sampler_device(device: Any) -> None:
    """Keep production behaviour explicit while allowing the loop to be unit-tested."""

    if device.type == "cpu":
        raise RuntimeError(
            "MiniMax-H3 DiT 仍在 CPU。请使用 model loader 的 auto/cuda "
            "模式并确认 ComfyUI 已成功装载模型；该 33B DiT 不支持 CPU 推理。"
        )


def _prepare_accel_for_sample(
    transformer: Any,
    *,
    accel: str | None,
    task: str,
    target: Mapping[str, Any],
    sigma_points: int,
    video_shift: float,
    audio_shift: float,
    num_denoise_steps: int,
    cache_dit_rdt: float | None = None,
    cache_dit_mc: int | None = None,
    cache_dit_warmup: int | None = None,
    velocity_stride: int | None = None,
):
    """单卡加速：velocity-cache→runtime；cache-dit→挂接；off→None。无多卡门禁。"""
    from .runtime.h3_settings import ACCEL_OFF
    from .runtime.quality_profiles import resolve_accel_request
    from .runtime.cache_dit_integration import prepare_transformer_cache_dit
    from .runtime.velocity_cache import VelocityCacheRuntime

    mode = ACCEL_OFF if accel is None else str(accel)
    resolved = resolve_accel_request(
        mode, task=task, target=target, sigma_points=sigma_points,
        video_shift=video_shift, audio_shift=audio_shift, rdt=cache_dit_rdt,
        mc=cache_dit_mc, warmup=cache_dit_warmup, velocity_stride=velocity_stride,
    )
    if resolved is None:
        return None
    if resolved.kind == "velocity-cache" and resolved.velocity is not None:
        return VelocityCacheRuntime(resolved.velocity).bind(int(num_denoise_steps))
    if resolved.kind == "cache-dit" and resolved.cache_dit is not None:
        prepare_transformer_cache_dit(
            transformer, resolved.cache_dit, num_denoise_steps=int(num_denoise_steps)
        )
    return None


def _prepare_cache_dit_for_sample(transformer: Any, *, cache_dit: str | None, **kw):
    return _prepare_accel_for_sample(transformer, accel=cache_dit, **kw)




def _dit_velocity_for_step(
    *,
    transformer: Any,
    kwargs: Mapping[str, Any],
    video_rows: Any,
    audio_rows: Any,
    video_update: Any | None,
    audio_update: Any | None,
    velocity_rt: Any | None,
    task: str,
    step: int,
):
    """单步 velocity：可选 whole-step cache（只缓存 update rows，对齐官方）。"""
    t = _require_torch()
    if velocity_rt is not None and not velocity_rt.refresh(step):
        mv, ma = velocity_rt.on_hit(step)
        _require_finite_step_tensor(mv, task=task, step=step + 1, modality="video", phase="cached velocity")
        _require_finite_step_tensor(ma, task=task, step=step + 1, modality="audio", phase="cached velocity")
        return mv, ma
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
    _require_finite_step_tensor(
        velocity_video, task=task, step=step + 1, modality="video", phase="transformer velocity"
    )
    _require_finite_step_tensor(
        velocity_audio, task=task, step=step + 1, modality="audio", phase="transformer velocity"
    )
    if video_update is None:
        mv, ma = velocity_video.float(), velocity_audio.float()
    else:
        mv = velocity_video.float()[video_update]
        ma = velocity_audio.float()[audio_update]
    if velocity_rt is not None:
        mv, ma = velocity_rt.on_dit(step, mv, ma)
    return mv, ma


def sample_h3(
    *,
    transformer: Any,
    conditioning: Mapping[str, Any],
    av_latent: Mapping[str, Any],
    packed: Mapping[str, Any],
    seed: int,
    sigma_points: int,
    video_shift: float,
    audio_shift: float,
    visual_condition_rows: Any | None = None,
    audio_reference_rows: Any | None = None,
    visual_condition_shapes: Sequence[Sequence[int]] | None = None,
    audio_reference_t: Sequence[int] | None = None,
    visual_condition_noise: float = H3_IMGVID_COND_TIMESTEP,
    audio_reference_noise: float = H3_AUDIO_REF_COND_TIMESTEP,
    progress: Callable[[int, int], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    accel: str | None = None,
    cache_dit: str | None = None,
    cache_dit_rdt: float | None = None,
    cache_dit_mc: int | None = None,
    cache_dit_warmup: int | None = None,
    velocity_stride: int | None = None,
) -> dict[str, Any]:
    """Run task-neutral H3 denoising with frozen visual/audio anchors.

    ``av_latent`` contains only the native target shapes.  The packed mapping
    carries full image/audio positions and update masks; its clean condition
    rows and shape metadata are used by default, but may be supplied explicitly
    for callers that keep VAE condition caches outside the packed structure.
    """

    t = _require_torch()
    clean_conditioning, clean_latent, target_video, target_audio = (
        _validate_generic_sampler_inputs(
            conditioning=conditioning,
            av_latent=av_latent,
            packed=packed,
        )
    )
    seed = validate_seed(seed)
    sigma_points, video_shift, audio_shift = validate_sigma_request(
        sigma_points=sigma_points,
        video_shift=video_shift,
        audio_shift=audio_shift,
    )
    visual_condition_noise = _condition_noise_level(
        visual_condition_noise, "visual_condition_noise"
    )
    audio_reference_noise = _condition_noise_level(
        audio_reference_noise, "audio_reference_noise"
    )
    if not callable(transformer):
        raise TypeError(
            "H3 transformer 不可调用；model loader 必须返回 "
            "MiniMaxH3DiTModel 或可解析到该模型的 handle"
        )
    device = _model_device(transformer)
    _require_accelerated_sampler_device(device)

    branch = H3DenoiseBranch(
        packed=packed,
        prompt_embeds=clean_conditioning["prompt_embeds"],
        device=device,
        transformer=transformer,
    )
    _validate_conditional_layout(branch)

    target_video_generator = t.Generator(device="cpu").manual_seed(seed)
    raw_target_video_noise = t.randn(
        tuple(int(x) for x in target_video.shape),
        generator=target_video_generator,
        dtype=t.float32,
        device="cpu",
    )
    target_video_rows = patchify_video_latent(raw_target_video_noise).to(
        device=device, dtype=t.float32
    )
    target_audio_generator = t.Generator(device="cpu").manual_seed(seed)
    target_audio_rows = t.randn(
        int(target_audio.shape[2]) * 2,
        H3_AUDIO_ROW_WIDTH,
        generator=target_audio_generator,
        dtype=t.float32,
        device="cpu",
    ).to(device=device)

    expected_target_video = int(branch.update_mask.sum())
    expected_target_audio = int(branch.audio_update_mask.sum())
    if tuple(int(x) for x in target_video_rows.shape) != (
        expected_target_video,
        H3_VIDEO_ROW_WIDTH,
    ):
        raise ValueError(
            "target video rows 与 packed update_mask 不一致："
            f"noise={tuple(target_video_rows.shape)}，mask={expected_target_video}"
        )
    if tuple(int(x) for x in target_audio_rows.shape) != (
        expected_target_audio,
        H3_AUDIO_ROW_WIDTH,
    ):
        raise ValueError(
            "target audio rows 与 packed audio_update_mask 不一致："
            f"noise={tuple(target_audio_rows.shape)}，mask={expected_target_audio}"
        )

    if visual_condition_rows is None:
        visual_condition_rows = packed.get("visual_cond_rows")
    if audio_reference_rows is None:
        audio_reference_rows = packed.get("audio_ref_rows")
    if visual_condition_shapes is None:
        visual_condition_shapes = packed.get("visual_condition_shapes")
    if audio_reference_t is None:
        audio_reference_t = packed.get("audio_reference_t")

    task = str(clean_latent.get("task", packed.get("task", ""))).strip().lower()
    _validate_condition_block_metadata(
        packed=packed,
        branch=branch,
        conditioning=clean_conditioning,
        task=task,
        visual_condition_rows=visual_condition_rows,
        audio_reference_rows=audio_reference_rows,
        visual_condition_shapes=visual_condition_shapes,
        audio_reference_t=audio_reference_t,
    )

    n_visual_condition = int((~branch.update_mask).sum())
    if n_visual_condition:
        if visual_condition_rows is None:
            raise ValueError(
                f"packed layout 有 {n_visual_condition} 个 visual condition rows，"
                "但未提供 visual_cond_rows"
            )
        if visual_condition_shapes is None:
            raise ValueError("visual condition rows 缺少 visual_condition_shapes")
        visual_anchor = noise_visual_condition_rows(
            visual_condition_rows,
            condition_shapes=visual_condition_shapes,
            target_latent_t=int(target_video.shape[2]),
            seed=seed,
            noise_level=visual_condition_noise,
        )
        if tuple(int(x) for x in visual_anchor.shape) != (
            n_visual_condition,
            H3_VIDEO_ROW_WIDTH,
        ):
            raise ValueError(
                "visual condition rows 与 packed update_mask 的 false rows 不一致"
            )
        visual_anchor = visual_anchor.to(
            device=device, dtype=t.float32
        ).clone()
    else:
        if visual_condition_rows is not None and int(visual_condition_rows.shape[0]):
            raise ValueError("layout 没有 visual condition，但传入了非空 condition rows")
        visual_anchor = None

    n_audio_reference = int((~branch.audio_update_mask).sum())
    if n_audio_reference:
        if audio_reference_rows is None:
            raise ValueError(
                f"packed layout 有 {n_audio_reference} 个 audio reference rows，"
                "但未提供 audio_ref_rows"
            )
        if audio_reference_t is None:
            raise ValueError("audio reference rows 缺少 audio_reference_t")
        audio_anchor = noise_audio_reference_rows(
            audio_reference_rows,
            reference_audio_t=audio_reference_t,
            seed=seed,
            noise_level=audio_reference_noise,
        )
        if tuple(int(x) for x in audio_anchor.shape) != (
            n_audio_reference,
            H3_AUDIO_ROW_WIDTH,
        ):
            raise ValueError(
                "audio reference rows 与 audio_update_mask 的 false rows 不一致"
            )
        audio_anchor = audio_anchor.to(device=device, dtype=t.float32).clone()
    else:
        if audio_reference_rows is not None and int(audio_reference_rows.shape[0]):
            raise ValueError("layout 没有 audio reference，但传入了非空 ref rows")
        audio_anchor = None

    # Expand target noise into full modality-row order.  Clean cache tensors
    # are never used as mutable state; only cloned/noised anchors enter here.
    video_rows = t.zeros(
        int(branch.img_pos.numel()),
        H3_VIDEO_ROW_WIDTH,
        dtype=t.float32,
        device=device,
    )
    video_rows[branch.update_mask_dev] = target_video_rows
    if visual_anchor is not None:
        video_rows[~branch.update_mask_dev] = visual_anchor
    audio_rows = t.zeros(
        int(branch.audio_pos.numel()),
        H3_AUDIO_ROW_WIDTH,
        dtype=t.float32,
        device=device,
    )
    audio_rows[branch.audio_update_mask_dev] = target_audio_rows
    if audio_anchor is not None:
        audio_rows[~branch.audio_update_mask_dev] = audio_anchor

    video_sigmas = shifted_sigma_schedule(
        sigma_points=sigma_points, shift=video_shift
    )
    audio_sigmas = shifted_sigma_schedule(
        sigma_points=sigma_points, shift=audio_shift
    )
    if len(video_sigmas) != len(audio_sigmas):
        raise RuntimeError("video/audio sigma schedule 长度不一致")
    total = len(video_sigmas) - 1
    video_sigma_tensor = t.tensor(video_sigmas, dtype=t.float32, device=device)
    audio_sigma_tensor = t.tensor(audio_sigmas, dtype=t.float32, device=device)
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

    velocity_rt = _prepare_accel_for_sample(
        transformer,
        accel=accel if accel is not None else cache_dit,
        task=task,
        target=clean_latent["target"],
        sigma_points=sigma_points,
        video_shift=video_shift,
        audio_shift=audio_shift,
        num_denoise_steps=total,
        cache_dit_rdt=cache_dit_rdt,
        cache_dit_mc=cache_dit_mc,
        cache_dit_warmup=cache_dit_warmup,
        velocity_stride=velocity_stride,
    )

    video_update = branch.update_mask_dev
    audio_update = branch.audio_update_mask_dev
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
            imgvid_cond_timestep_floor=visual_condition_noise,
            audio_ref_cond_timestep_floor=audio_reference_noise,
        )
        mv_video, mv_audio = _dit_velocity_for_step(
            transformer=transformer,
            kwargs=kwargs,
            video_rows=video_rows,
            audio_rows=audio_rows,
            video_update=video_update,
            audio_update=audio_update,
            velocity_rt=velocity_rt,
            task=task,
            step=step,
        )

        denoised_video = rf_velocity_to_x0(
            video_rows[video_update],
            mv_video,
            video_timesteps[step],
        )
        next_video_target = euler_eta0_step(
            video_rows[video_update],
            denoised_video,
            sigma_curr=sigma_video,
            sigma_next=video_sigmas[step + 1],
            sigma_ratio=video_ratios[step],
        )
        _require_finite_step_tensor(
            next_video_target,
            task=task,
            step=step + 1,
            modality="video",
            phase="target latent",
        )
        next_video_rows = video_rows.clone()
        next_video_rows[video_update] = next_video_target
        if visual_anchor is not None:
            next_video_rows[~video_update] = visual_anchor
        video_rows = next_video_rows

        denoised_audio = rf_velocity_to_x0(
            audio_rows[audio_update],
            mv_audio,
            audio_timesteps[step],
        )
        next_audio_target = euler_eta0_step(
            audio_rows[audio_update],
            denoised_audio,
            sigma_curr=sigma_audio,
            sigma_next=audio_sigmas[step + 1],
            sigma_ratio=audio_ratios[step],
        )
        _require_finite_step_tensor(
            next_audio_target,
            task=task,
            step=step + 1,
            modality="audio",
            phase="target latent",
        )
        next_audio_rows = audio_rows.clone()
        next_audio_rows[audio_update] = next_audio_target
        if audio_anchor is not None:
            next_audio_rows[~audio_update] = audio_anchor
        audio_rows = next_audio_rows
        if progress is not None:
            progress(step + 1, total)

    # Never leak condition rows into the native target latents sent to VAE.
    target_video_rows = video_rows[video_update].to(device="cpu")
    target_audio_rows = audio_rows[audio_update].to(device="cpu")
    video = unpatchify_video_rows(
        target_video_rows,
        latent_t=int(target_video.shape[2]),
        latent_h=int(target_video.shape[3]),
        latent_w=int(target_video.shape[4]),
    )
    audio = unpack_audio_rows(
        target_audio_rows,
        audio_t=int(target_audio.shape[2]),
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
        "visual_condition_noise": visual_condition_noise,
        "audio_reference_noise": audio_reference_noise,
        "sampled": True,
    }


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
    accel: str | None = None,
    cache_dit: str | None = None,
    cache_dit_rdt: float | None = None,
    cache_dit_mc: int | None = None,
    cache_dit_warmup: int | None = None,
    velocity_stride: int | None = None,
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

    velocity_rt = _prepare_accel_for_sample(
        transformer,
        accel=accel if accel is not None else cache_dit,
        task=H3_TASK_T2VA,
        target=target,
        sigma_points=sigma_points,
        video_shift=video_shift,
        audio_shift=audio_shift,
        num_denoise_steps=total,
        cache_dit_rdt=cache_dit_rdt,
        cache_dit_mc=cache_dit_mc,
        cache_dit_warmup=cache_dit_warmup,
        velocity_stride=velocity_stride,
    )

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
        mv_video, mv_audio = _dit_velocity_for_step(
            transformer=transformer,
            kwargs=kwargs,
            video_rows=video_rows,
            audio_rows=audio_rows,
            video_update=None,
            audio_update=None,
            velocity_rt=velocity_rt,
            task=H3_TASK_T2VA,
            step=step,
        )
        denoised_video = rf_velocity_to_x0(
            video_rows, mv_video, video_timesteps[step]
        )
        denoised_audio = rf_velocity_to_x0(
            audio_rows, mv_audio, audio_timesteps[step]
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
        _require_finite_step_tensor(
            video_rows,
            task=H3_TASK_T2VA,
            step=step + 1,
            modality="video",
            phase="target latent",
        )
        _require_finite_step_tensor(
            audio_rows,
            task=H3_TASK_T2VA,
            step=step + 1,
            modality="audio",
            phase="target latent",
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
    "noise_audio_reference_rows",
    "noise_visual_condition_rows",
    "patchify_video_latent",
    "rf_velocity_to_x0",
    "sample_h3",
    "sample_t2va",
    "shifted_sigma_schedule",
    "unpack_audio_rows",
    "unpatchify_video_rows",
]
