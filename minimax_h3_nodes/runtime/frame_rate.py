"""实验性帧率条件：adaLN 时间嵌入 + 可选时序 RoPE 缩放。非上游契约。"""
from __future__ import annotations
from typing import Any, Mapping
from .h3_settings import (
    FRAME_RATE_ROPE_FREQ_PROFILES,
    FRAME_RATE_ROPE_SIGMA_PROFILES,
    H3_NATIVE_FPS,
)

def validate_frame_rate_options(
    *,
    frame_rate: float = H3_NATIVE_FPS,
    adaln: bool = True,
    temporal_rope: bool = False,
    rope_end_timestep: float = 1.0,
    rope_low_frequency_count: int = 16,
    rope_frequency_profile: str = "hard",
    rope_sigma_profile: str = "constant",
    rope_sigma_end: float = 0.0,
) -> dict[str, Any]:
    """校验并规范化 Frame Rate 节点选项。"""
    fr = float(frame_rate)
    if not 1.0 <= fr <= 120.0:
        raise ValueError(f"frame_rate 必须在 [1,120]，实际 {fr}")
    end_t = float(rope_end_timestep)
    if not 0.0 <= end_t <= 1.0:
        raise ValueError("rope_end_timestep 必须在 [0,1]")
    low_n = int(rope_low_frequency_count)
    if not 0 <= low_n <= 16:
        raise ValueError("rope_low_frequency_count 必须在 [0,16]")
    freq_p = str(rope_frequency_profile)
    if freq_p not in FRAME_RATE_ROPE_FREQ_PROFILES:
        raise ValueError(f"rope_frequency_profile 必须是 {FRAME_RATE_ROPE_FREQ_PROFILES}")
    sigma_p = str(rope_sigma_profile)
    if sigma_p not in FRAME_RATE_ROPE_SIGMA_PROFILES:
        raise ValueError(f"rope_sigma_profile 必须是 {FRAME_RATE_ROPE_SIGMA_PROFILES}")
    sigma_end = float(rope_sigma_end)
    if not 0.0 <= sigma_end <= 0.99:
        raise ValueError("rope_sigma_end 必须在 [0,0.99]")
    if not adaln and not temporal_rope:
        raise ValueError("Frame Rate：adaln 与 temporal_rope 至少启用一项")
    return {
        "frame_rate": fr,
        "adaln": bool(adaln),
        "temporal_rope": bool(temporal_rope),
        "rope_end_timestep": end_t,
        "rope_low_frequency_count": low_n,
        "rope_frequency_profile": freq_p,
        "rope_sigma_profile": sigma_p,
        "rope_sigma_end": sigma_end,
    }

def adaln_frame_rate(options: Mapping[str, Any] | None) -> float | None:
    """供 TimeEmbedder / 预计算：未启用 adaln 时返回 None。"""
    if not options or not options.get("adaln"):
        return None
    return float(options["frame_rate"])

def rope_temporal_scale(
    options: Mapping[str, Any] | None,
    *,
    video_timestep: float,
    video_sigma: float,
) -> float:
    """PR：scale=24/fps，可按 sigma 剖面淡入；native 24fps 或未启用则为 1。"""
    if not options or not options.get("temporal_rope"):
        return 1.0
    fr = float(options["frame_rate"])
    if fr == H3_NATIVE_FPS or float(video_timestep) > float(options["rope_end_timestep"]):
        return 1.0
    scale = H3_NATIVE_FPS / fr
    profile = str(options.get("rope_sigma_profile") or "constant")
    if profile == "constant":
        return scale
    sigma_end = float(options.get("rope_sigma_end") or 0.0)
    denom = 1.0 - sigma_end
    weight = 0.0 if denom <= 0 else max(0.0, min(1.0, (float(video_sigma) - sigma_end) / denom))
    if profile == "smoothstep":
        weight = weight * weight * (3.0 - 2.0 * weight)
    return 1.0 + (scale - 1.0) * weight

def apply_temporal_freq_scale(
    time_freq: Any,
    *,
    video_mask: Any,
    temporal_scale: float,
    low_frequency_count: int,
    frequency_profile: str,
) -> Any:
    """就地缩放视频行时间轴低频 RoPE；scale≈1 或 low_n=0 时跳过。"""
    if video_mask is None or low_frequency_count <= 0 or abs(float(temporal_scale) - 1.0) < 1e-12:
        return time_freq
    import torch
    n = min(int(low_frequency_count), int(time_freq.shape[-1]))
    if n <= 0:
        return time_freq
    device = time_freq.device
    if frequency_profile == "hard":
        weights = torch.ones(n, dtype=torch.float32, device=device)
    else:
        weights = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float32, device=device)[1:]
        if frequency_profile == "smoothstep":
            weights = weights.square() * (3.0 - 2.0 * weights)
    mask = video_mask.to(device=device, dtype=torch.bool).view(-1)
    time_freq = time_freq.clone()
    time_freq[mask, -n:] *= 1.0 + (float(temporal_scale) - 1.0) * weights
    return time_freq

__all__ = [
    "validate_frame_rate_options",
    "adaln_frame_rate",
    "rope_temporal_scale",
    "apply_temporal_freq_scale",
]
