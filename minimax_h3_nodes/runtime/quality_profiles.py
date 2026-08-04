"""Resolve H3 acceleration quality profiles (Cache-DiT / velocity-cache)."""
from __future__ import annotations
import json, logging
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping
from .h3_settings import (
    ACCEL_AUTO, ACCEL_CACHE_DIT_PROFILE, ACCEL_MANUAL_CACHE_DIT, ACCEL_MANUAL_VELOCITY,
    ACCEL_OFF, ACCEL_VELOCITY_PROFILE, CACHE_DIT_BN, CACHE_DIT_FN, CACHE_DIT_MC,
    CACHE_DIT_MIN_VERSION, CACHE_DIT_PKG, CACHE_DIT_RDT_COOKBOOK, CACHE_DIT_RDT_PROFILE,
    CACHE_DIT_SCM_POLICY, CACHE_DIT_SCM_PRESET, CACHE_DIT_TAYLORSEER, CACHE_DIT_TS_ORDER,
    CACHE_DIT_WARMUP, VELOCITY_FINAL_REFRESH, VELOCITY_STRIDE, VELOCITY_TAIL_DENSE,
    VELOCITY_TAIL_REBALANCE, VELOCITY_TAYLORSEER, VELOCITY_TS_ORDER,
)
from .velocity_cache import VelocityCacheConfig
LOGGER = logging.getLogger("MiniMaxH3.quality_profiles")
_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

@dataclass(frozen=True)
class CacheDitResolved:
    enabled: bool; profile_id: str | None; Fn_compute_blocks: int; Bn_compute_blocks: int
    max_warmup_steps: int; residual_diff_threshold: float; max_continuous_cached_steps: int
    enable_taylorseer: bool; taylorseer_order: int; scm_preset: str; scm_policy: str
    steps_computation_mask: list[int] | None = None

@dataclass(frozen=True)
class AccelResolved:
    kind: str  # "cache-dit" | "velocity-cache"
    cache_dit: CacheDitResolved | None = None
    velocity: VelocityCacheConfig | None = None

def _load_profile(profile_id: str) -> dict[str, Any]:
    path = _PROFILES_DIR / f"{profile_id}.json"
    if not path.is_file(): raise FileNotFoundError(f"Quality profile not found: {profile_id} ({path})")
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("profile_id", "")) != profile_id:
        raise ValueError(f"profile_id does not match the filename: {data.get('profile_id')!r} vs {profile_id!r}")
    if str(data.get("status", "")) != "validated": raise ValueError(f"profile {profile_id!r} status is not validated")
    return data

def _pkg_ok(required: Mapping[str, str]) -> None:
    for name, want in required.items():
        try: got = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Acceleration requires {name}>={want} (not currently installed)") from exc
        if tuple(int(x) for x in got.split(".")[:3]) < tuple(int(x) for x in want.split(".")[:3]):
            raise RuntimeError(f"{name} version is too old: need >={want}, got {got}")

def _workload_match(wl: Mapping[str, Any], *, task: str, target: Mapping[str, Any],
                    sigma_points: int, video_shift: float, audio_shift: float) -> bool:
    return (str(wl.get("task", "")).lower() == str(task).lower()
            and int(wl.get("width", -1)) == int(target.get("width", -2))
            and int(wl.get("height", -1)) == int(target.get("height", -2))
            and int(wl.get("fps", -1)) == int(target.get("fps", -2))
            and int(wl.get("frame_count", -1)) == int(target.get("frame_count", -2))
            and int(wl.get("num_inference_steps", -1)) == int(sigma_points)
            and abs(float(wl.get("flow_shift", 0)) - float(video_shift)) < 1e-6
            and abs(float(wl.get("audio_flow_shift", 0)) - float(audio_shift)) < 1e-6)

def _match_any(data: Mapping[str, Any], **kw) -> bool:
    return any(_workload_match(wl, **kw) for wl in data.get("workloads", ()))

def _cache_from(raw: Mapping[str, Any], *, profile_id: str | None, rdt_default: float) -> CacheDitResolved:
    return CacheDitResolved(
        enabled=bool(raw.get("enabled", True)), profile_id=profile_id,
        Fn_compute_blocks=int(raw.get("Fn_compute_blocks", CACHE_DIT_FN)),
        Bn_compute_blocks=int(raw.get("Bn_compute_blocks", CACHE_DIT_BN)),
        max_warmup_steps=int(raw.get("max_warmup_steps", CACHE_DIT_WARMUP)),
        residual_diff_threshold=float(raw.get("residual_diff_threshold", rdt_default)),
        max_continuous_cached_steps=int(raw.get("max_continuous_cached_steps", CACHE_DIT_MC)),
        enable_taylorseer=bool(raw.get("enable_taylorseer", CACHE_DIT_TAYLORSEER)),
        taylorseer_order=int(raw.get("taylorseer_order", CACHE_DIT_TS_ORDER)),
        scm_preset=str(raw.get("scm_preset", CACHE_DIT_SCM_PRESET)),
        scm_policy=str(raw.get("scm_policy", CACHE_DIT_SCM_POLICY)),
    )

def _velocity_from(raw: Mapping[str, Any], *, profile_id: str | None) -> VelocityCacheConfig:
    return VelocityCacheConfig(
        stride=int(raw.get("stride", VELOCITY_STRIDE)),
        taylorseer=bool(raw.get("taylorseer", VELOCITY_TAYLORSEER)),
        taylorseer_order=int(raw.get("taylorseer_order", VELOCITY_TS_ORDER)),
        tail_dense_steps=int(raw.get("tail_dense_steps", VELOCITY_TAIL_DENSE)),
        tail_rebalance=bool(raw.get("tail_rebalance", VELOCITY_TAIL_REBALANCE)),
        final_refresh=bool(raw.get("final_refresh", VELOCITY_FINAL_REFRESH)),
        profile_id=profile_id,
    )

def _alias(mode: str) -> str:
    key = str(mode or ACCEL_OFF).strip().lower()
    return {"none": ACCEL_OFF, "false": ACCEL_OFF, "0": ACCEL_OFF, "manual": ACCEL_MANUAL_CACHE_DIT,
            "cache-dit": ACCEL_CACHE_DIT_PROFILE, "velocity": ACCEL_VELOCITY_PROFILE,
            "velocity-cache": ACCEL_VELOCITY_PROFILE}.get(key, key)

def resolve_accel_request(
    mode: str, *, task: str, target: Mapping[str, Any], sigma_points: int,
    video_shift: float, audio_shift: float, rdt: float | None = None,
    mc: int | None = None, warmup: int | None = None, velocity_stride: int | None = None,
) -> AccelResolved | None:
    """Resolve the Dual Sigma Sampler acceleration mode; off or an unmatched auto returns None."""
    key = _alias(mode)
    kw = dict(task=task, target=target, sigma_points=sigma_points, video_shift=video_shift, audio_shift=audio_shift)
    if key in ("", ACCEL_OFF): return None
    wl = (f"task={task} {int(target.get('width', 0))}x{int(target.get('height', 0))} "
          f"f={int(target.get('frame_count', 0))} steps={sigma_points} "
          f"shift={video_shift}/{audio_shift}")
    if key == ACCEL_MANUAL_CACHE_DIT:
        _pkg_ok({CACHE_DIT_PKG: CACHE_DIT_MIN_VERSION})
        LOGGER.warning("accel=manual-cache-dit: approximate acceleration (not ground truth); %s", wl)
        return AccelResolved("cache-dit", cache_dit=_cache_from({
            "enabled": True, "Fn_compute_blocks": CACHE_DIT_FN, "Bn_compute_blocks": CACHE_DIT_BN,
            "max_warmup_steps": warmup if warmup is not None else CACHE_DIT_WARMUP,
            "residual_diff_threshold": rdt if rdt is not None else CACHE_DIT_RDT_COOKBOOK,
            "max_continuous_cached_steps": mc if mc is not None else CACHE_DIT_MC,
            "enable_taylorseer": CACHE_DIT_TAYLORSEER, "taylorseer_order": CACHE_DIT_TS_ORDER,
            "scm_preset": CACHE_DIT_SCM_PRESET, "scm_policy": CACHE_DIT_SCM_POLICY,
        }, profile_id=None, rdt_default=CACHE_DIT_RDT_COOKBOOK))
    if key == ACCEL_MANUAL_VELOCITY:
        stride = int(velocity_stride if velocity_stride is not None else VELOCITY_STRIDE)
        LOGGER.warning(
            "accel=manual-velocity: approximate acceleration stride=%s (not ground truth); theoretical full steps≈%s; %s",
            stride, max(0, int(sigma_points) - 1), wl,
        )
        return AccelResolved("velocity-cache", velocity=VelocityCacheConfig(
            stride=stride, taylorseer=VELOCITY_TAYLORSEER, taylorseer_order=VELOCITY_TS_ORDER,
            tail_dense_steps=VELOCITY_TAIL_DENSE, tail_rebalance=VELOCITY_TAIL_REBALANCE,
            final_refresh=VELOCITY_FINAL_REFRESH, profile_id=None,
        ))
    if key == ACCEL_AUTO:
        for pid in (ACCEL_VELOCITY_PROFILE, ACCEL_CACHE_DIT_PROFILE):  # Prefer ~3× velocity.
            data = _load_profile(pid)
            if _match_any(data, **kw):
                return resolve_accel_request(pid, **kw, rdt=rdt, mc=mc, warmup=warmup,
                                             velocity_stride=velocity_stride)
        LOGGER.warning("accel=auto: no validated profile matched; leaving acceleration off; %s", wl); return None
    data = _load_profile(key)
    if not _match_any(data, **kw):
        raise ValueError(
            f"accel={key!r} is validated only for a specific workload (for example, 1344x768/124f/50 steps/shift 12/3); "
            f"the current workload does not match ({wl}). Use manual-* or return the parameters to the validated contract."
        )
    technique = str(data.get("technique") or ("velocity-cache" if "velocity_cache" in data else "cache-dit"))
    if technique == "velocity-cache":
        LOGGER.info("accel=%s velocity-cache enabled (approximate); %s", key, wl)
        return AccelResolved("velocity-cache", velocity=_velocity_from(data.get("velocity_cache") or {}, profile_id=key))
    _pkg_ok(data.get("required_packages") or {CACHE_DIT_PKG: CACHE_DIT_MIN_VERSION})
    raw = dict(data.get("cache_dit") or {}); raw.setdefault("residual_diff_threshold", CACHE_DIT_RDT_PROFILE)
    LOGGER.info("accel=%s Cache-DiT enabled (block-level approximation); %s", key, wl)
    return AccelResolved("cache-dit", cache_dit=_cache_from(raw, profile_id=key, rdt_default=CACHE_DIT_RDT_PROFILE))

# Legacy-name compatibility.
def resolve_cache_dit_request(mode: str, **kw) -> CacheDitResolved | None:
    resolved = resolve_accel_request(mode, **kw)
    if resolved is None: return None
    if resolved.kind != "cache-dit" or resolved.cache_dit is None:
        raise ValueError(f"resolve_cache_dit_request does not support accel={mode!r} (use resolve_accel_request)")
    return resolved.cache_dit

__all__ = ["AccelResolved", "CacheDitResolved", "resolve_accel_request", "resolve_cache_dit_request"]
