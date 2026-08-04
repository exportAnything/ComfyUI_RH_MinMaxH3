"""Sampling core (P2 moved from sampling.py)."""
from __future__ import annotations
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    import torch
except ImportError:  # Allows static contract tests without a full Comfy install.
    torch = None  # type: ignore[assignment]

from ..contracts import (
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

LOGGER = logging.getLogger("MiniMaxH3.sampling")


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "MiniMax-H3 Direct requires the PyTorch bundled with ComfyUI; torch is not installed in this Python environment"
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
        raise ValueError("video latent must be a rank-5 tensor [B,24,T,H,W]")
    batch, channels, full_t, full_h, full_w = (int(x) for x in latent.shape)
    if channels != H3_VIDEO_CHANNELS:
        raise ValueError(
            f"video latent channels must be {H3_VIDEO_CHANNELS}; got {channels}"
        )
    pt, ph, pw = H3_VIDEO_PATCH_SIZE
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            "video latent T/H/W must be divisible by patch_size "
            f"{H3_VIDEO_PATCH_SIZE}; got shape={list(latent.shape)}"
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
        raise ValueError("video rows must be a rank-2 tensor")
    pt, ph, pw = H3_VIDEO_PATCH_SIZE
    token_t = int(latent_t) // pt
    token_h = int(latent_h) // ph
    token_w = int(latent_w) // pw
    if int(rows.shape[1]) != H3_VIDEO_ROW_WIDTH:
        raise ValueError(
            f"video row width must be {H3_VIDEO_ROW_WIDTH}; got {int(rows.shape[1])}"
        )
    rows_per_sample = token_t * token_h * token_w
    if rows_per_sample <= 0 or int(rows.shape[0]) % rows_per_sample:
        raise ValueError(
            f"video rows={int(rows.shape[0])} does not match latent geometry"
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
        raise ValueError("audio rows must be a rank-2 tensor")
    expected_rows = int(audio_channels) * int(audio_t)
    if tuple(int(x) for x in rows.shape) != (expected_rows, H3_AUDIO_ROW_WIDTH):
        raise ValueError(
            "audio rows shape must be "
            f"({expected_rows},{H3_AUDIO_ROW_WIDTH}); got {tuple(rows.shape)}"
        )
    native = rows.reshape(
        int(audio_channels), int(audio_t), H3_AUDIO_ROW_WIDTH
    )
    return native.permute(0, 2, 1).contiguous()


def rf_velocity_to_x0(state: Any, velocity: Any, timestep: Any) -> Any:
    t = _require_torch()
    if state.shape != velocity.shape:
        raise ValueError(
            f"state/velocity shape mismatch: {tuple(state.shape)} vs {tuple(velocity.shape)}"
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
        raise ValueError("Euler state/denoised shape mismatch")
    if sigma_curr < 0.0 or sigma_next < 0.0:
        raise ValueError("sigma cannot be negative")
    if sigma_curr == 0.0:
        if sigma_next != 0.0:
            raise ValueError("When sigma_curr=0, sigma_next must also be 0")
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


def res_multistep_coeffs(
    sigmas: Sequence[float],
) -> list[tuple[float, float] | None]:
    """Per-step second-order coefficients for the eta=0 ``res_multistep`` sampler.

    Return ``(h*b1, h*b2)`` for each denoising step. ``None`` means the step can use
    first-order Euler only (the first step has no history, or the final step has
    sigma_next=0). Coefficients are precomputed on the host in float64, with no
    device synchronization inside the loop.

    Mathematical form (exponential integrator,
    https://arxiv.org/pdf/2308.02157): in ``t = -log(sigma)`` space,
    ``x' = ratio*x + h*(b1*denoised + b2*old_denoised)``, where
    ``ratio = sigma_next/sigma_curr = exp(-h)``. It always satisfies
    ``h*b1 + h*b2 == 1 - ratio``, so when ``old_denoised == denoised`` it reduces
    exactly to :func:`euler_eta0_step`.
    """

    values = [float(s) for s in sigmas]
    if any(v < 0.0 for v in values):
        raise ValueError("sigma cannot be negative")
    coeffs: list[tuple[float, float] | None] = []
    for i in range(len(values) - 1):
        s_curr, s_next = values[i], values[i + 1]
        if i == 0 or s_curr == 0.0 or s_next == 0.0:
            coeffs.append(None)
            continue
        s_prev = values[i - 1]
        h = math.log(s_curr / s_next)
        c2 = math.log(s_curr / s_prev) / h  # (t_prev - t_curr)/h, < 0 when strictly decreasing.
        if h <= 0.0 or c2 == 0.0 or not math.isfinite(c2):
            coeffs.append(None)
            continue
        phi1 = math.expm1(-h) / (-h)
        phi2 = (phi1 - 1.0) / (-h)
        b1 = phi1 - phi2 / c2
        b2 = phi2 / c2
        coeffs.append((h * b1, h * b2))
    return coeffs


def res_multistep_eta0_step(
    state: Any,
    denoised: Any,
    old_denoised: Any,
    *,
    sigma_curr: float,
    sigma_next: float,
    hb1: float,
    hb2: float,
    sigma_ratio: Any | None = None,
) -> Any:
    """Second-order multistep update on one stream's own sigma schedule.

    Isomorphic to :func:`euler_eta0_step`: beyond ``ratio*x``, split the
    ``(1-ratio)`` denoised weight between the current and previous steps according
    to the exponential integrator. The video and audio streams integrate independently
    on their own shift schedules without ComfyUI slope folding.
    """

    if state.shape != denoised.shape or state.shape != old_denoised.shape:
        raise ValueError("res_multistep state/denoised shape mismatch")
    if sigma_curr <= 0.0 or sigma_next <= 0.0:
        raise ValueError("res_multistep requires positive sigma_curr and sigma_next")
    t = _require_torch()
    compute_dtype = (
        t.float32 if state.dtype in (t.float16, t.bfloat16) else state.dtype
    )
    if sigma_ratio is None:
        ratio = state.new_tensor(sigma_next / sigma_curr, dtype=compute_dtype)
    else:
        ratio = sigma_ratio.to(device=state.device, dtype=compute_dtype)
    out = (
        ratio * state.to(dtype=compute_dtype)
        + hb1 * denoised.to(dtype=compute_dtype)
        + hb2 * old_denoised.to(dtype=compute_dtype)
    )
    return out.to(dtype=state.dtype)


def _condition_noise_level(value: Any, name: str) -> float:
    """Validate a rectified-flow condition timestep/noise mixture."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric between 0 and 1")
    out = float(value)
    if not 0.0 <= out <= 1.0:
        raise ValueError(f"{name} must be in [0,1]; got {out!r}")
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
        raise TypeError("visual condition rows must be a torch.Tensor")
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != H3_VIDEO_ROW_WIDTH:
        raise ValueError(
            "visual condition rows must be [N,96]; got "
            f"{tuple(int(x) for x in clean_rows.shape)}"
        )
    if isinstance(target_latent_t, bool) or int(target_latent_t) <= 0:
        raise ValueError("target_latent_t must be a positive integer")

    parsed: list[tuple[int, int, int]] = []
    expected_rows = 0
    for index, raw_shape in enumerate(condition_shapes):
        if len(raw_shape) != 3:
            raise ValueError(
                f"visual_condition_shapes[{index}] must be (T,H,W)"
            )
        latent_t, latent_h, latent_w = (int(value) for value in raw_shape)
        if latent_t <= 0 or latent_h <= 0 or latent_w <= 0:
            raise ValueError(
                f"visual_condition_shapes[{index}] values must all be positive"
            )
        if latent_h % H3_VIDEO_PATCH_SIZE[1] or latent_w % H3_VIDEO_PATCH_SIZE[2]:
            raise ValueError(
                f"visual_condition_shapes[{index}] H/W must be divisible by 2"
            )
        parsed.append((latent_t, latent_h, latent_w))
        expected_rows += (
            latent_t
            * (latent_h // H3_VIDEO_PATCH_SIZE[1])
            * (latent_w // H3_VIDEO_PATCH_SIZE[2])
        )
    if not parsed:
        if int(clean_rows.shape[0]) != 0:
            raise ValueError("shape metadata is required when visual condition rows are present")
        return clean_rows.detach().clone().to(dtype=t.float32).contiguous()
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"visual condition rows={int(clean_rows.shape[0])}, "
            f"but shape metadata implies {expected_rows}"
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
                f"visual condition {index} latent_t={latent_t} exceeds "
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
        raise TypeError("audio reference rows must be a torch.Tensor")
    if clean_rows.ndim != 2 or int(clean_rows.shape[1]) != H3_AUDIO_ROW_WIDTH:
        raise ValueError(
            "audio reference rows must be [N,32]; got "
            f"{tuple(int(x) for x in clean_rows.shape)}"
        )
    parsed = [int(value) for value in reference_audio_t]
    if any(value <= 0 for value in parsed):
        raise ValueError("Every value in audio_reference_t must be a positive integer")
    expected_rows = 2 * sum(parsed)
    if int(clean_rows.shape[0]) != expected_rows:
        raise ValueError(
            f"audio reference rows={int(clean_rows.shape[0])}, "
            f"but audio_reference_t implies {expected_rows}"
        )
    if not parsed:
        if int(clean_rows.shape[0]) != 0:
            raise ValueError("Length metadata is required when audio reference rows are present")
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
            f"{phase} is not a tensor"
        )
    if not bool(t.isfinite(value).all()):
        raise FloatingPointError(
            f"MiniMax-H3 task={task} step={step} modality={modality} "
            f"{phase} contains NaN/Inf; decoding was stopped"
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
                "runtime.packing output is missing fields: " + ", ".join(missing)
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
                "update_mask length must equal img_pos rows: "
                f"{int(self.update_mask.numel())} != {int(self.img_pos.numel())}"
            )
        if int(self.audio_update_mask.numel()) != int(self.audio_pos.numel()):
            raise ValueError(
                "audio_update_mask length must equal audio_pos rows: "
                f"{int(self.audio_update_mask.numel())} != "
                f"{int(self.audio_pos.numel())}"
            )

        if prompt_embeds.ndim == 3:
            if int(prompt_embeds.shape[0]) != 1:
                raise ValueError("MiniMax-H3 Direct v0 supports batch=1 only")
            prompt_embeds = prompt_embeds[0]
        text_len = int(packed["text_pos"].view(-1).shape[0])
        if int(prompt_embeds.shape[0]) != text_len:
            raise ValueError(
                f"prompt rows={int(prompt_embeds.shape[0])} does not match packed "
                f"text_len={text_len}"
            )
        token_tags = packed["token_tags"].view(-1).to(t.long)
        if int(token_tags.shape[0]) != seq_len:
            raise ValueError(
                f"token_tags length={int(token_tags.shape[0])}; expected {seq_len}"
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
                try:
                    selected_embeds = refine(
                        prompt_embeds, refiner_cu, device=device, text_length=text_len,
                    )
                except TypeError:  # Compatibility with mocks/legacy signatures that lack text_length.
                    selected_embeds = refine(prompt_embeds, refiner_cu, device=device)
            refined_length = text_len

        cu_dev = cu.to(device)
        token_tags_dev = token_tags.to(device)
        img_position_ids = packed["img_position_ids"][None].to(device)
        self.static_kwargs: dict[str, Any] = {
            "img_position_ids": img_position_ids,
            "update_mask": self.update_mask_dev,
            "update_audio_mask": self.audio_update_mask_dev,
            "token_tags": token_tags_dev,
            "skip_mask_out_condition": False,
            "prompt_embeds": selected_embeds.to(device),
            "img_pos_info": {"position_ids": self.img_pos_dev},
            "audio_pos_info": {"position_ids": self.audio_pos_dev},
            "text_pos_info": {"position_ids": text_pos_dev},
            "img_pos_for_infer_output_info": {
                "position_ids": self.img_pos_dev
            },
            "packed_seq_params": {
                "cu_seqlens_q": cu_dev,
                "max_seqlen_q": int(cu[1].item()) if hasattr(cu[1], "item") else int(cu[1]),
            },
            "refiner_packed_seq_params": {
                "cu_seqlens_q": refiner_cu,
                "max_seqlen_q": text_len,
            },
            "structure_validated": True,  # Validate token_tags and related structure once during prepare.
        }
        if refined_length is not None:
            self.static_kwargs["refined_prompt_embeds_length"] = refined_length
        prepare = getattr(transformer, "prepare_structure", None)
        if callable(prepare):
            with t.inference_mode():
                self.static_kwargs.update(
                    prepare(
                        img_position_ids=img_position_ids,
                        cu_seqlens=cu_dev,
                        token_tags=token_tags_dev,
                        seq_len=seq_len,
                    )
                )

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
        frame_rate_options: Mapping[str, Any] | None = None,
        video_sigma: float | None = None,
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
        out = {
            **self.static_kwargs,
            "x": self.x_buffer,
            "audio_x": self.audio_x_buffer,
            "unique_timesteps": unique,
            "inverse_indices": inverse,
            "video_timestep": float(video_timestep),
            "video_sigma": float(
                1.0 - float(video_timestep) if video_sigma is None else video_sigma
            ),
        }
        if frame_rate_options is not None:
            out["frame_rate_options"] = frame_rate_options
        return out


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
        "Could not determine the H3 DiT device; the model loader must first hand the "
        "transformer to ComfyUI model management for loading"
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
        raise TypeError("conditioning must be a mapping")
    if not isinstance(av_latent, Mapping):
        raise TypeError("av_latent must be a mapping")
    if not isinstance(packed, Mapping):
        raise TypeError("packed conditioning must be a mapping")

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
            "conditioning/target/latent/packed task mismatch: "
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
            "conditioning/target/latent/packed partition mismatch: "
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
            raise ValueError("V2 conditioning is missing task")
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
                "FL2VA/Ref2VA accepts conditioning/av_latent v2 schema only; "
                f"got task={task!r}"
            )
        clean_conditioning = validate_conditioning(conditioning)
        clean_latent = validate_av_latent(av_latent)
    else:
        raise ValueError(
            "conditioning/av_latent schemas must both be MiniMax-H3 v2 or both be "
            "the original T2VA v1; got "
            f"{conditioning_schema!r} / {latent_schema!r}"
        )

    prompt_embeds = clean_conditioning.get("prompt_embeds")
    if not isinstance(prompt_embeds, t.Tensor):
        raise TypeError("conditioning is missing the prompt_embeds tensor")
    if prompt_embeds.ndim == 3:
        if int(prompt_embeds.shape[0]) != 1:
            raise ValueError("MiniMax-H3 sampler supports batch=1 only")
    elif prompt_embeds.ndim != 2:
        raise ValueError("prompt_embeds must be [L,D] or [1,L,D]")

    video = clean_latent.get("video")
    audio = clean_latent.get("audio")
    if not isinstance(video, t.Tensor) or video.ndim != 5:
        raise ValueError("av_latent.video must be a [1,24,T,H,W] tensor")
    if tuple(int(x) for x in video.shape[:2]) != (1, H3_VIDEO_CHANNELS):
        raise ValueError(
            "The first two dimensions of av_latent.video must be [1,24]; got "
            f"{tuple(int(x) for x in video.shape[:2])}"
        )
    if not isinstance(audio, t.Tensor) or audio.ndim != 3:
        raise ValueError("av_latent.audio must be a [2,32,T] tensor")
    if tuple(int(x) for x in audio.shape[:2]) != (2, H3_AUDIO_ROW_WIDTH):
        raise ValueError(
            "The first two dimensions of av_latent.audio must be [2,32]; got "
            f"{tuple(int(x) for x in audio.shape[:2])}"
        )

    latent_t, latent_h, latent_w = (int(x) for x in video.shape[2:])
    if (
        latent_t % H3_VIDEO_PATCH_SIZE[0]
        or latent_h % H3_VIDEO_PATCH_SIZE[1]
        or latent_w % H3_VIDEO_PATCH_SIZE[2]
    ):
        raise ValueError(
            "target video latent T/H/W must be divisible by patch size "
            f"{H3_VIDEO_PATCH_SIZE}"
        )
    audio_t = int(audio.shape[2])
    if audio_t <= 0:
        raise ValueError("target audio latent T must be a positive integer")

    target = clean_latent.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("av_latent is missing the target mapping")
    for key, actual in (
        ("video_latent_t", latent_t),
        ("video_latent_h", latent_h),
        ("video_latent_w", latent_w),
        ("audio_latent_t", audio_t),
    ):
        declared = target.get(key)
        if declared is not None and int(declared) != actual:
            raise ValueError(
                f"target.{key}={declared} does not match latent tensor={actual}"
            )

    packed_latent_shape = packed.get("latent_shape")
    if packed_latent_shape is not None:
        expected = (latent_t, latent_h, latent_w, H3_VIDEO_CHANNELS)
        if tuple(int(x) for x in packed_latent_shape) != expected:
            raise ValueError(
                f"packed latent_shape must be {expected}; "
                f"got {tuple(packed_latent_shape)}"
            )
    packed_audio_shape = packed.get("audio_shape")
    if packed_audio_shape is not None:
        expected_audio = (2, H3_AUDIO_ROW_WIDTH, audio_t)
        if tuple(int(x) for x in packed_audio_shape) != expected_audio:
            raise ValueError(
                f"packed audio_shape must be {expected_audio}; "
                f"got {tuple(packed_audio_shape)}"
            )

    return clean_conditioning, clean_latent, video, audio


def _validate_conditional_layout(
    branch: H3DenoiseBranch, *, allow_frozen_video: bool = False
) -> None:
    """Fail before allocating target noise when packed row order is invalid."""

    t = _require_torch()
    if int(branch.update_mask.numel()) != int(branch.img_pos.numel()):
        raise ValueError(
            "update_mask length must equal img_pos rows: "
            f"{int(branch.update_mask.numel())} != {int(branch.img_pos.numel())}"
        )
    if int(branch.audio_update_mask.numel()) != int(branch.audio_pos.numel()):
        raise ValueError(
            "audio_update_mask length must equal audio_pos rows: "
            f"{int(branch.audio_update_mask.numel())} != "
            f"{int(branch.audio_pos.numel())}"
        )
    if not bool(branch.update_mask.any()) and not allow_frozen_video:
        raise ValueError("packed layout requires at least one target video row")
    if not bool(branch.audio_update_mask.any()):
        raise ValueError("packed layout requires at least one target audio row")

    for name, positions in (
        ("img_pos", branch.img_pos),
        ("audio_pos", branch.audio_pos),
    ):
        if positions.numel() == 0:
            raise ValueError(f"{name} cannot be empty")
        if bool((positions < 0).any()) or bool((positions >= branch.seq_len).any()):
            raise ValueError(f"{name} contains a position outside seq_len")
        if positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError(f"{name} must be strictly increasing with no duplicates")
    if bool(t.isin(branch.img_pos, branch.audio_pos).any()):
        raise ValueError("img_pos and audio_pos cannot overlap")

    # Official FL2VA/Ref2VA layouts concatenate condition rows before target
    # rows inside each modality.  Enforcing this catches anchor-order drift
    # before clean rows are scattered into the wrong positions.
    for name, mask in (
        ("update_mask", branch.update_mask),
        ("audio_update_mask", branch.audio_update_mask),
    ):
        target_indices = t.nonzero(mask).view(-1)
        if target_indices.numel() == 0:  # V2A: video is fully frozen.
            if name == "update_mask" and allow_frozen_video:
                continue
            raise ValueError(f"{name} requires at least one target row")
        first_target = int(target_indices[0])
        if bool(mask[:first_target].any()) or not bool(mask[first_target:].all()):
            raise ValueError(f"{name} must be ordered as [condition rows | target rows]")


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
    v2a: bool = False,
) -> None:
    """Cross-check ordered block metadata against every frozen row stream."""

    t = _require_torch()
    raw_blocks = packed.get("condition_blocks")
    if raw_blocks is None and task == H3_TASK_T2VA:
        raw_blocks = ()
    if not isinstance(raw_blocks, Sequence) or isinstance(raw_blocks, (str, bytes)):
        raise ValueError("packed.condition_blocks must be an ordered list")
    blocks = list(raw_blocks)

    raw_conditions = conditioning.get("conditions", [])
    if raw_conditions is None:
        raw_conditions = []
    if not isinstance(raw_conditions, Sequence) or isinstance(
        raw_conditions, (str, bytes)
    ):
        raise ValueError("conditioning.conditions must be an ordered list")
    conditions = list(raw_conditions)

    if task == H3_TASK_T2VA:
        if blocks or conditions:
            raise ValueError("T2VA does not allow condition_blocks")
    else:
        if not blocks:
            raise ValueError(f"{task.upper()} packed.condition_blocks cannot be empty")
        if len(blocks) != len(conditions):
            raise ValueError(
                "packed.condition_blocks and conditioning.conditions counts do not match: "
                f"{len(blocks)} != {len(conditions)}"
            )

    def metadata_int(
        value: Any,
        path: str,
        *,
        allow_zero: bool = False,
    ) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        invalid = value < 0 if allow_zero else value <= 0
        if invalid:
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{path} must be a {qualifier} integer")
        return int(value)

    derived_visual_shapes: list[tuple[int, int, int]] = []
    derived_audio_t: list[int] = []
    visual_rows_per_block: list[int] = []
    audio_rows_per_block: list[int] = []
    for index, raw_block in enumerate(blocks):
        path = f"packed.condition_blocks[{index}]"
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"{path} must be an object")
        condition_index = raw_block.get("condition_index")
        if (
            isinstance(condition_index, bool)
            or not isinstance(condition_index, int)
            or condition_index != index
        ):
            raise ValueError(
                f"{path}.condition_index must be contiguous in request order; "
                f"expected {index}, got {condition_index!r}"
            )
        kind = raw_block.get("kind")
        if not isinstance(kind, str):
            raise ValueError(f"{path}.kind must be a string")
        if index >= len(conditions) or not isinstance(conditions[index], Mapping):
            raise ValueError(f"conditioning.conditions[{index}] must be an object")
        if task == "fl2va":
            expected_kind = "keyframe"
        elif task == "ref2va":
            expected_kind = conditions[index].get("type")
        else:
            expected_kind = None
        if kind != expected_kind:
            raise ValueError(
                f"{path}.kind order mismatch: expected {expected_kind!r}, got {kind!r}"
            )
        expected_condition_index = conditions[index].get("condition_index")
        if expected_condition_index != condition_index:
            raise ValueError(
                f"{path}.condition_index does not match conditioning.conditions[{index}]: "
                f"{condition_index!r} != {expected_condition_index!r}"
            )

        visual_count = 0
        if kind in {"keyframe", "image", "video", "video_audio"}:
            if kind in {"keyframe", "image"}:
                declared_latent_t = raw_block.get("latent_t")
                if declared_latent_t is not None and metadata_int(
                    declared_latent_t, f"{path}.latent_t"
                ) != 1:
                    raise ValueError(f"{path}.latent_t must be 1")
                latent_t = 1
            else:
                latent_t = metadata_int(
                    raw_block.get("latent_t"), f"{path}.latent_t"
                )
            latent_h = metadata_int(raw_block.get("latent_h"), f"{path}.latent_h")
            latent_w = metadata_int(raw_block.get("latent_w"), f"{path}.latent_w")
            if latent_h % H3_VIDEO_PATCH_SIZE[1] or latent_w % H3_VIDEO_PATCH_SIZE[2]:
                raise ValueError(f"{path} visual latent H/W must be divisible by 2")
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
            raise ValueError(f"{path}.visual_rows_count does not match the latent shape")
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
            raise ValueError(f"{path} does not allow ref_audio_t")
        declared_audio_count = raw_block.get("audio_rows_count")
        if declared_audio_count is not None and metadata_int(
            declared_audio_count,
            f"{path}.audio_rows_count",
            allow_zero=True,
        ) != audio_count:
            raise ValueError(f"{path}.audio_rows_count does not match ref_audio_t")
        audio_rows_per_block.append(audio_count)

    def strict_shapes(raw: Any) -> list[tuple[int, int, int]]:
        if raw is None:
            return []
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("visual_condition_shapes must be an ordered list")
        output: list[tuple[int, int, int]] = []
        for index, shape in enumerate(raw):
            if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
                raise ValueError(f"visual_condition_shapes[{index}] must be (T,H,W)")
            if len(shape) != 3:
                raise ValueError(f"visual_condition_shapes[{index}] must be (T,H,W)")
            output.append(
                tuple(
                    metadata_int(value, f"visual_condition_shapes[{index}][{axis}]")
                    for axis, value in enumerate(shape)
                )
            )
        return output

    supplied_shapes = strict_shapes(visual_condition_shapes)
    if v2a:  # Visual rows come from the fully frozen target segment, not condition_blocks.
        if derived_visual_shapes:
            raise ValueError("V2A requires packed data with no existing visual condition_blocks")
        expected_visual_rows = int((~branch.update_mask).sum())
        shape_rows = sum(
            lt * (lh // H3_VIDEO_PATCH_SIZE[1]) * (lw // H3_VIDEO_PATCH_SIZE[2])
            for lt, lh, lw in supplied_shapes
        )
        if shape_rows != expected_visual_rows:
            raise ValueError(
                f"V2A visual_condition_shapes imply rows={shape_rows}, which does not match "
                f"update_mask false rows={expected_visual_rows}"
            )
    else:
        if supplied_shapes != derived_visual_shapes:
            raise ValueError(
                "visual_condition_shapes do not match the visual order/shape in condition_blocks: "
                f"{supplied_shapes!r} != {derived_visual_shapes!r}"
            )
        expected_visual_rows = sum(visual_rows_per_block)
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
        raise ValueError("audio_reference_t must be an ordered list")
    if supplied_audio_t != derived_audio_t:
        raise ValueError(
            "audio_reference_t does not match the audio order/length in condition_blocks: "
            f"{supplied_audio_t!r} != {derived_audio_t!r}"
        )

    expected_audio_rows = sum(audio_rows_per_block)
    for name, rows, expected, width in (
        ("visual_cond_rows", visual_condition_rows, expected_visual_rows, 96),
        ("audio_ref_rows", audio_reference_rows, expected_audio_rows, 32),
    ):
        if expected == 0 and rows is None:
            continue
        if not isinstance(rows, t.Tensor) or rows.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 tensor")
        if tuple(int(value) for value in rows.shape) != (expected, width):
            raise ValueError(
                f"{name} shape={tuple(rows.shape)} does not match condition_blocks-derived "
                f"{(expected, width)}"
            )
    if expected_visual_rows != int((~branch.update_mask).sum()):
        raise ValueError("condition_blocks visual rows do not match update_mask false rows")
    if expected_audio_rows != int((~branch.audio_update_mask).sum()):
        raise ValueError(
            "condition_blocks audio rows do not match audio_update_mask false rows"
        )


def _require_accelerated_sampler_device(device: Any) -> None:
    """Keep production behaviour explicit while allowing the loop to be unit-tested."""

    if device.type == "cpu":
        raise RuntimeError(
            "MiniMax-H3 DiT is still on CPU. Use the model loader's auto/cuda mode and "
            "confirm that ComfyUI loaded the model successfully; this 33B DiT does not support CPU inference."
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
    """Single-GPU acceleration: velocity-cache→runtime; cache-dit→attach; off→None. No multi-GPU gating."""
    from .h3_settings import ACCEL_OFF
    from .quality_profiles import resolve_accel_request
    from .cache_dit_integration import prepare_transformer_cache_dit
    from .velocity_cache import VelocityCacheRuntime

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




def _log_accel_step_stats(velocity_rt: Any | None, *, total: int, dit_calls: int) -> None:
    """Report actual/theoretical DiT calls after sampling (acceleration quality guard)."""
    if velocity_rt is not None and hasattr(velocity_rt, "stats"):
        st = velocity_rt.stats()
        LOGGER.info(
            "accel velocity-cache complete: DiT %s/%s (cache_hits=%s taylor=%s stride=%s), approximate and not ground truth",
            st.get("dit_calls", dit_calls), total, st.get("cache_hits", 0),
            st.get("taylorseer_steps", 0), st.get("stride", "?"),
        )
        return
    LOGGER.info("Sampling complete: DiT forwards=%s/%s", dit_calls, total)


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
    dit_counter: list[int] | None = None,
):
    """Compute one velocity step with optional whole-step cache (update rows only, aligned with upstream)."""
    t = _require_torch()
    if velocity_rt is not None and not velocity_rt.refresh(step):
        mv, ma = velocity_rt.on_hit(step)
        _require_finite_step_tensor(mv, task=task, step=step + 1, modality="video", phase="cached velocity")
        _require_finite_step_tensor(ma, task=task, step=step + 1, modality="audio", phase="cached velocity")
        return mv, ma
    if dit_counter is not None: dit_counter[0] += 1
    with t.inference_mode():
        velocity_video, velocity_audio = transformer(**kwargs)
    if tuple(velocity_video.shape) != tuple(video_rows.shape):
        raise ValueError(
            "DiT video velocity shape mismatch: "
            f"{tuple(velocity_video.shape)} vs {tuple(video_rows.shape)}"
        )
    if tuple(velocity_audio.shape) != tuple(audio_rows.shape):
        raise ValueError(
            "DiT audio velocity shape mismatch: "
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


def _prepare_v2a_packed(packed: Mapping[str, Any], target_video: Any) -> dict[str, Any]:
    """Freeze the entire target video segment as visual conditioning (layouts with no existing visual conditioning only)."""
    t = _require_torch()
    out = dict(packed)
    mask = out["update_mask"].view(-1).to(t.bool)
    if bool((~mask).any()):
        raise ValueError(
            "V2A (denoise_video=False) requires packed data with no existing visual condition; "
            "use a T2VA layout or separate the conditions first"
        )
    if not bool(mask.any()):
        raise ValueError("V2A requires non-empty video rows")
    rows = patchify_video_latent(target_video.detach().to(dtype=t.float32).cpu())
    if int(rows.shape[0]) != int(mask.numel()):
        raise ValueError(
            f"V2A clean video rows={int(rows.shape[0])} does not match update_mask={int(mask.numel())}"
        )
    # Non-empty video: reject the all-zero Empty AV Latent placeholder.
    if float(target_video.detach().abs().max()) <= 0.0:
        raise ValueError("V2A requires an encoded, non-empty video latent (not an Empty AV Latent zero tensor)")
    out["update_mask"] = t.zeros_like(mask)
    out["visual_cond_rows"] = rows.contiguous()
    out["visual_condition_shapes"] = [(
        int(target_video.shape[2]),
        int(target_video.shape[3]),
        int(target_video.shape[4]),
    )]
    out["v2a"] = True
    return out


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
    denoise_video: bool = True,
    frame_rate_options: Mapping[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
    sampler_mode: str | None = None,
    allow_accel_with_res_multistep: bool = False,
    accel: str | None = None,
    cache_dit: str | None = None,
    cache_dit_rdt: float | None = None,
    cache_dit_mc: int | None = None,
    cache_dit_warmup: int | None = None,
    velocity_stride: int | None = None,
    telemetry: Any | None = None,
) -> dict[str, Any]:
    """Run task-neutral H3 denoising with frozen visual/audio anchors.

    ``av_latent`` contains only the native target shapes.  The packed mapping
    carries full image/audio positions and update masks; its clean condition
    rows and shape metadata are used by default, but may be supplied explicitly
    for callers that keep VAE condition caches outside the packed structure.

    ``denoise_video=False`` (V2A): use the video latent as a clean condition
    (timestep floor) and denoise audio only.
    """

    t = _require_torch()
    from .telemetry import H3Telemetry
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
    from .h3_settings import (
        SAMPLER_MODE_CHOICES,
        SAMPLER_MODE_EULER,
        SAMPLER_MODE_RES_MULTISTEP,
    )
    sampler_mode = (
        SAMPLER_MODE_EULER if sampler_mode is None else str(sampler_mode)
    )
    if sampler_mode not in SAMPLER_MODE_CHOICES:
        raise ValueError(
            f"Unknown sampler_mode: {sampler_mode!r}; choices: {SAMPLER_MODE_CHOICES}"
        )
    res_multistep = sampler_mode == SAMPLER_MODE_RES_MULTISTEP
    visual_condition_noise = _condition_noise_level(
        visual_condition_noise, "visual_condition_noise"
    )
    audio_reference_noise = _condition_noise_level(
        audio_reference_noise, "audio_reference_noise"
    )
    if not callable(transformer):
        raise TypeError(
            "H3 transformer is not callable; the model loader must return "
            "MiniMaxH3DiTModel or a handle that resolves to it"
        )
    device = _model_device(transformer)
    _require_accelerated_sampler_device(device)
    tel = telemetry if isinstance(telemetry, H3Telemetry) else H3Telemetry(device=device)
    target_meta = clean_latent.get("target") if isinstance(clean_latent.get("target"), Mapping) else {}

    v2a = not bool(denoise_video)
    if v2a:
        packed = _prepare_v2a_packed(packed, target_video)
        visual_condition_rows = packed["visual_cond_rows"]
        visual_condition_shapes = packed["visual_condition_shapes"]

    with tel.stage("packed_branch"):
        branch = H3DenoiseBranch(
            packed=packed,
            prompt_embeds=clean_conditioning["prompt_embeds"],
            device=device,
            transformer=transformer,
        )
        _validate_conditional_layout(branch, allow_frozen_video=v2a)

    if v2a:
        # No video target noise; use zero rows for shape validation.
        target_video_rows = t.zeros(0, H3_VIDEO_ROW_WIDTH, dtype=t.float32, device=device)
    else:
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
            "target video rows do not match packed update_mask: "
            f"noise={tuple(target_video_rows.shape)}, mask={expected_target_video}"
        )
    if tuple(int(x) for x in target_audio_rows.shape) != (
        expected_target_audio,
        H3_AUDIO_ROW_WIDTH,
    ):
        raise ValueError(
            "target audio rows do not match packed audio_update_mask: "
            f"noise={tuple(target_audio_rows.shape)}, mask={expected_target_audio}"
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
    # Ref2VA's ordered image/video/audio references add enough context rows
    # that a healthy low-VRAM INT8 forward can exceed the generic 75-second
    # telemetry threshold.  Keep timing telemetry, but do not turn that
    # observation into an inference failure.  The existing large 15-second
    # matrix case is likewise exempt.
    abort_exempt = task == "ref2va" or (
        int(target_meta.get("width") or 0) >= 1344
        and int(target_meta.get("frame_count") or 0) >= 124
    )
    _validate_condition_block_metadata(
        packed=packed,
        branch=branch,
        conditioning=clean_conditioning,
        task=task,
        visual_condition_rows=visual_condition_rows,
        audio_reference_rows=audio_reference_rows,
        visual_condition_shapes=visual_condition_shapes,
        audio_reference_t=audio_reference_t,
        v2a=v2a,
    )

    n_visual_condition = int((~branch.update_mask).sum())
    if n_visual_condition:
        if visual_condition_rows is None:
            raise ValueError(
                f"packed layout has {n_visual_condition} visual condition rows, "
                "but visual_cond_rows was not provided"
            )
        if visual_condition_shapes is None:
            raise ValueError("visual condition rows are missing visual_condition_shapes")
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
                "visual condition rows do not match the false rows in packed update_mask"
            )
        visual_anchor = visual_anchor.to(
            device=device, dtype=t.float32
        ).clone()
    else:
        if visual_condition_rows is not None and int(visual_condition_rows.shape[0]):
            raise ValueError("The layout has no visual condition, but non-empty condition rows were provided")
        visual_anchor = None

    n_audio_reference = int((~branch.audio_update_mask).sum())
    if n_audio_reference:
        if audio_reference_rows is None:
            raise ValueError(
                f"packed layout has {n_audio_reference} audio reference rows, "
                "but audio_ref_rows was not provided"
            )
        if audio_reference_t is None:
            raise ValueError("audio reference rows are missing audio_reference_t")
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
                "audio reference rows do not match the false rows in audio_update_mask"
            )
        audio_anchor = audio_anchor.to(device=device, dtype=t.float32).clone()
    else:
        if audio_reference_rows is not None and int(audio_reference_rows.shape[0]):
            raise ValueError("The layout has no audio reference, but non-empty reference rows were provided")
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
        raise RuntimeError("video/audio sigma schedule lengths do not match")
    total = len(video_sigmas) - 1
    from .h3_settings import (
        OPT_ADALN_PRECOMPUTE,
        OPT_ADALN_RELEASE_WEIGHTS,
        OPT_INPLACE_EULER_UPDATE,
        OPT_PREBUILT_TIMESTEPS,
    )
    video_sigma_tensor = t.tensor(video_sigmas, dtype=t.float32, device=device)
    audio_sigma_tensor = t.tensor(audio_sigmas, dtype=t.float32, device=device)
    video_ratios = video_sigma_tensor[1:] / video_sigma_tensor[:-1]
    audio_ratios = audio_sigma_tensor[1:] / audio_sigma_tensor[:-1]
    if OPT_PREBUILT_TIMESTEPS:  # Contiguous tensors; the loop performs indexing only.
        video_timesteps = (1.0 - video_sigma_tensor[:-1]).contiguous()
        audio_timesteps = (1.0 - audio_sigma_tensor[:-1]).contiguous()
    else:
        video_timesteps = [
            t.tensor(1.0 - sigma, dtype=t.float32, device=device)
            for sigma in video_sigmas[:-1]
        ]
        audio_timesteps = [
            t.tensor(1.0 - sigma, dtype=t.float32, device=device)
            for sigma in audio_sigmas[:-1]
        ]

    from .frame_rate import adaln_frame_rate
    fr_opts = dict(frame_rate_options) if isinstance(frame_rate_options, Mapping) else None
    adaln_fr = adaln_frame_rate(fr_opts)
    if adaln_fr is not None and getattr(transformer, "use_adaln_curves", False):
        # Curve-table checkpoints have no time embedder, so there is nowhere to add
        # frame-rate conditioning. Fail early rather than at the first forward pass
        # (the temporal_rope branch is unaffected).
        raise RuntimeError(
            "Curve-table checkpoints do not support the Frame Rate node's adaln option "
            "(there is no time embedder). Disable adaln and keep temporal_rope only, or use the original DiT weights"
        )
    if OPT_ADALN_PRECOMPUTE:
        from .modulation_cache import (
            H3PrecomputeUnsupported,
            enumerate_modulation_timesteps,
        )
        precompute = getattr(transformer, "precompute_modulation", None)
        if callable(precompute):
            with tel.stage("adaln_precompute"):
                mod_ts = enumerate_modulation_timesteps(
                    video_sigmas, audio_sigmas,
                    visual_floor=visual_condition_noise,
                    audio_floor=audio_reference_noise,
                )
                try:
                    mod_cache = precompute(
                        mod_ts, compute_device=device,
                        release_weights=OPT_ADALN_RELEASE_WEIGHTS,
                        frame_rate=adaln_fr,
                    )
                except H3PrecomputeUnsupported as exc:
                    # frame_rate still applies through the forward pass's immediate TimeEmbedder path.
                    LOGGER.info("%s (output is unaffected; only one VRAM optimization is skipped)", exc)
                else:
                    LOGGER.info(
                        "AdaLN modulation precomputation complete: %d timesteps, %.2f GiB (%s), "
                        "release_weights=%s, frame_rate=%s",
                        len(mod_ts), mod_cache.bytes() / 1024 ** 3,
                        str(mod_cache.blocks.device), OPT_ADALN_RELEASE_WEIGHTS, adaln_fr,
                    )

    requested_accel = accel if accel is not None else cache_dit
    if res_multistep:
        from .h3_settings import ACCEL_OFF
        if requested_accel is not None and str(requested_accel) != ACCEL_OFF:
            # Acceleration tiers are calibrated for euler-50. Second-order sampling
            # uses fewer steps and has less room to skip: measured at 21 points,
            # stride 4 leaves only 7 DiT calls and introduces ringing/banding. At 31
            # points, stride 2 becomes usable again (16 DiT calls / PSNR 24.2dB), but
            # is no cheaper than euler-50 + stride 4 (14 calls / 24.1dB). Both
            # accelerations compete for the same budget; the more steps skipped, the
            # less opportunity second-order per-step accuracy has to pay off. Keep it
            # disabled by default while allowing callers to opt into the risk explicitly.
            if allow_accel_with_res_multistep:
                LOGGER.warning(
                    "res_multistep combined with accel=%s is uncalibrated; increase "
                    "sigma_points at the same time (quality degrades noticeably at 21 points)",
                    requested_accel,
                )
            else:
                LOGGER.info(
                    "res_multistep forces accel=off (requested %s; profiles are calibrated for "
                    "euler-50; enable allow_accel_with_res_multistep to combine them)",
                    requested_accel,
                )
                requested_accel = ACCEL_OFF
        else:
            requested_accel = ACCEL_OFF
        video_res_coeffs = res_multistep_coeffs(video_sigmas)
        audio_res_coeffs = res_multistep_coeffs(audio_sigmas)
    else:
        video_res_coeffs = audio_res_coeffs = None

    velocity_rt = _prepare_accel_for_sample(
        transformer,
        accel=requested_accel,
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
    dit_counter = [0]
    # Previous denoised history for res_multistep (one target-row tensor per stream).
    old_denoised_video: Any = None
    old_denoised_audio: Any = None
    with tel.stage("denoise_loop"):
        for step in range(total):
            if check_cancelled is not None:
                check_cancelled()
            if tel.aborted_reason:
                raise RuntimeError(f"H3 telemetry abort: {tel.aborted_reason}")
            with tel.denoise_step(step, abort_exempt=abort_exempt):
                sigma_video = video_sigmas[step]
                sigma_audio = audio_sigmas[step]
                kwargs = branch.forward_kwargs(
                    video_rows=video_rows,
                    audio_rows=audio_rows,
                    video_timestep=1.0 - sigma_video,
                    audio_timestep=1.0 - sigma_audio,
                    imgvid_cond_timestep_floor=visual_condition_noise,
                    audio_ref_cond_timestep_floor=audio_reference_noise,
                    frame_rate_options=fr_opts,
                    video_sigma=sigma_video,
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
                    dit_counter=dit_counter,
                )

                if expected_target_video:  # V2A has no video target; skip video Euler.
                    cur_video = video_rows[video_update]
                    denoised_video = rf_velocity_to_x0(
                        cur_video, mv_video, video_timesteps[step]
                    )
                    video_coeff = (
                        video_res_coeffs[step]
                        if video_res_coeffs is not None
                        else None
                    )
                    if video_coeff is not None and old_denoised_video is not None:
                        next_video_target = res_multistep_eta0_step(
                            cur_video, denoised_video, old_denoised_video,
                            sigma_curr=sigma_video,
                            sigma_next=video_sigmas[step + 1],
                            hb1=video_coeff[0], hb2=video_coeff[1],
                            sigma_ratio=video_ratios[step],
                        )
                    else:
                        next_video_target = euler_eta0_step(
                            cur_video, denoised_video,
                            sigma_curr=sigma_video, sigma_next=video_sigmas[step + 1],
                            sigma_ratio=video_ratios[step],
                        )
                    if res_multistep:
                        old_denoised_video = denoised_video
                    _require_finite_step_tensor(
                        next_video_target, task=task, step=step + 1,
                        modality="video", phase="target latent",
                    )
                    if OPT_INPLACE_EULER_UPDATE:  # Write target only; velocity cache owns an independent mv.
                        video_rows.index_copy_(0, branch.update_row_idx, next_video_target)
                    else:
                        next_video_rows = video_rows.clone()
                        next_video_rows[video_update] = next_video_target
                        if visual_anchor is not None:
                            next_video_rows[~video_update] = visual_anchor
                        video_rows = next_video_rows

                cur_audio = audio_rows[audio_update]
                denoised_audio = rf_velocity_to_x0(cur_audio, mv_audio, audio_timesteps[step])
                audio_coeff = (
                    audio_res_coeffs[step]
                    if audio_res_coeffs is not None
                    else None
                )
                if audio_coeff is not None and old_denoised_audio is not None:
                    next_audio_target = res_multistep_eta0_step(
                        cur_audio, denoised_audio, old_denoised_audio,
                        sigma_curr=sigma_audio, sigma_next=audio_sigmas[step + 1],
                        hb1=audio_coeff[0], hb2=audio_coeff[1],
                        sigma_ratio=audio_ratios[step],
                    )
                else:
                    next_audio_target = euler_eta0_step(
                        cur_audio, denoised_audio,
                        sigma_curr=sigma_audio, sigma_next=audio_sigmas[step + 1],
                        sigma_ratio=audio_ratios[step],
                    )
                if res_multistep:
                    old_denoised_audio = denoised_audio
                _require_finite_step_tensor(
                    next_audio_target, task=task, step=step + 1, modality="audio", phase="target latent",
                )
                if OPT_INPLACE_EULER_UPDATE:
                    audio_rows.index_copy_(0, branch.audio_update_row_idx, next_audio_target)
                else:
                    next_audio_rows = audio_rows.clone()
                    next_audio_rows[audio_update] = next_audio_target
                    if audio_anchor is not None:
                        next_audio_rows[~audio_update] = audio_anchor
                    audio_rows = next_audio_rows
            if progress is not None:
                progress(step + 1, total)

    _log_accel_step_stats(velocity_rt, total=total, dit_calls=dit_counter[0])
    # Never leak condition rows into the native target latents sent to VAE.
    target_audio_rows = audio_rows[audio_update].to(device="cpu")
    if v2a:  # Return the clean input video instead of unpatchifying an empty target mask.
        video = target_video.detach().to(device="cpu", dtype=t.float32).contiguous()
    else:
        video = unpatchify_video_rows(
            video_rows[video_update].to(device="cpu"),
            latent_t=int(target_video.shape[2]),
            latent_h=int(target_video.shape[3]),
            latent_w=int(target_video.shape[4]),
        )
    audio = unpack_audio_rows(
        target_audio_rows,
        audio_t=int(target_audio.shape[2]),
    )
    tel.note(
        task=task,
        seq_len=int(branch.seq_len),
        width=target_meta.get("width"),
        height=target_meta.get("height"),
        frame_count=target_meta.get("frame_count"),
        sampler_mode=sampler_mode,
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
        "denoise_video": not v2a,
        "frame_rate_options": fr_opts,
        "dit_calls": (
            velocity_rt.stats()["dit_calls"] if velocity_rt is not None and hasattr(velocity_rt, "stats")
            else dit_counter[0]
        ),
        "dit_steps_total": total,
        "telemetry": tel.summary(),
    }


def sample_t2va(**kwargs: Any) -> dict[str, Any]:
    """V1 T2VA entry point; delegate to sample_h3 to avoid duplicate denoising loops."""
    return sample_h3(**kwargs)

__all__ = [n for n in list(globals()) if not n.startswith("__")]
