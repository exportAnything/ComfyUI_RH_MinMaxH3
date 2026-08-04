"""DiT activation VRAM estimation and EWMA correction (P0-6; 24GB baseline)."""
from __future__ import annotations
import logging
from typing import Any
from .h3_settings import (
    DIT_ACTIVATION_BYTES_PER_TOKEN,
    DIT_ACTIVATION_EWMA_ALPHA,
    DIT_ACTIVATION_FLOOR,
    DIT_ACTIVATION_CEIL,
    DIT_INFERENCE_RESERVE,
    DIT_SAFETY_MARGIN,
)

LOGGER = logging.getLogger(__name__)
_EWMA: dict[str, float] = {}  # fingerprint → corrected activation bytes.


def shape_fingerprint(
    *, task: str = "", seq_len: int = 0, video_latent_t: int = 0,
    video_latent_h: int = 0, video_latent_w: int = 0, ref_rows: int = 0,
    dtype: str = "bf16",
) -> str:
    return (
        f"{task}|s{int(seq_len)}|t{int(video_latent_t)}x{int(video_latent_h)}x{int(video_latent_w)}"
        f"|r{int(ref_rows)}|{dtype}"
    )


def estimate_dit_activation_bytes(
    *, seq_len: int = 0, video_latent_t: int = 0, video_latent_h: int = 0,
    video_latent_w: int = 0, ref_rows: int = 0, hidden_size: int = 3072,
    num_layers: int = 50, bytes_per_elem: int = 2,
) -> int:
    """Estimate one-step activations: token-dominated plus reference-row overhead; use a fixed reserve when shape is absent."""
    if seq_len <= 0 and video_latent_t > 0 and video_latent_h > 0 and video_latent_w > 0:
        seq_len = int(video_latent_t) * int(video_latent_h) * int(video_latent_w) + int(ref_rows)
    if seq_len <= 0:
        return int(DIT_INFERENCE_RESERVE)
    # Approximate attention/FFN workspace: tokens × hidden × layer factor;
    # resolve adds the safety margin.
    per_token = int(DIT_ACTIVATION_BYTES_PER_TOKEN) or (
        int(hidden_size) * int(bytes_per_elem) * 8
    )
    raw = int(seq_len) * per_token + int(ref_rows) * per_token // 2
    # Normalize layer count relative to 50; deeper models add a small increment.
    raw = int(raw * max(1.0, float(num_layers) / 50.0))
    return max(int(DIT_ACTIVATION_FLOOR), min(int(DIT_ACTIVATION_CEIL), raw))


def resolve_activation_reserve(
    *, seq_len: int = 0, video_latent_t: int = 0, video_latent_h: int = 0,
    video_latent_w: int = 0, ref_rows: int = 0, task: str = "",
    dtype: str = "bf16", observed_peak: int | None = None,
) -> int:
    """Return activation_reserve including a safety margin; optional EWMA corrects it from measured peaks."""
    fp = shape_fingerprint(
        task=task, seq_len=seq_len, video_latent_t=video_latent_t,
        video_latent_h=video_latent_h, video_latent_w=video_latent_w,
        ref_rows=ref_rows, dtype=dtype,
    )
    est = float(estimate_dit_activation_bytes(
        seq_len=seq_len, video_latent_t=video_latent_t,
        video_latent_h=video_latent_h, video_latent_w=video_latent_w,
        ref_rows=ref_rows,
    ))
    if observed_peak is not None and observed_peak > 0:
        prev = _EWMA.get(fp, est)
        alpha = float(DIT_ACTIVATION_EWMA_ALPHA)
        _EWMA[fp] = (1.0 - alpha) * prev + alpha * float(observed_peak)
        est = _EWMA[fp]
    elif fp in _EWMA:
        est = _EWMA[fp]
    reserve = int(est) + int(DIT_SAFETY_MARGIN)
    reserve = max(int(DIT_ACTIVATION_FLOOR), min(int(DIT_ACTIVATION_CEIL), reserve))
    LOGGER.info("DiT activation_reserve=%s fp=%s", reserve, fp)
    return reserve


def note_observed_activation(hint: dict[str, Any] | None, *, observed_peak: int) -> int:
    """Write the measured activation peak to the EWMA after sampling for the next resolve."""
    h = hint or {}
    return resolve_activation_reserve(
        seq_len=int(h.get("seq_len", 0) or 0),
        video_latent_t=int(h.get("video_latent_t", 0) or 0),
        video_latent_h=int(h.get("video_latent_h", 0) or 0),
        video_latent_w=int(h.get("video_latent_w", 0) or 0),
        ref_rows=int(h.get("ref_rows", 0) or 0),
        task=str(h.get("task", "") or ""),
        dtype=str(h.get("dtype", "bf16") or "bf16"),
        observed_peak=int(observed_peak),
    )


def residency_tier(*, free_bytes: int | None, weight_bytes: int, activation_bytes: int) -> str:
    """full | layerwise | reject — tier by available VRAM."""
    if free_bytes is None:
        return "layerwise"  # Probe failed: use the conservative 24GB baseline.
    need_full = int(weight_bytes) + int(activation_bytes)
    if free_bytes >= need_full:
        return "full"
    # layerwise: must fit at least one estimated layer (weights/layer count) plus activations.
    layer = max(1, int(weight_bytes) // 50) + int(activation_bytes)
    if free_bytes >= layer:
        return "layerwise"
    return "reject"
