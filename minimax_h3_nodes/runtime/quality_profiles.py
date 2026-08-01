"""H3 Cache-DiT quality profile 解析（官方 --quality 合同的 Comfy 精简版）。"""
from __future__ import annotations
import json, logging
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping
from .h3_settings import (
    CACHE_DIT_BN, CACHE_DIT_FN, CACHE_DIT_MC, CACHE_DIT_MIN_VERSION, CACHE_DIT_MODE_AUTO,
    CACHE_DIT_MODE_MANUAL, CACHE_DIT_MODE_OFF, CACHE_DIT_PKG, CACHE_DIT_PROFILE_ID,
    CACHE_DIT_RDT_COOKBOOK, CACHE_DIT_RDT_PROFILE, CACHE_DIT_SCM_POLICY, CACHE_DIT_SCM_PRESET,
    CACHE_DIT_TAYLORSEER, CACHE_DIT_TS_ORDER, CACHE_DIT_WARMUP,
)
LOGGER = logging.getLogger("MiniMaxH3.quality_profiles")
_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"

@dataclass(frozen=True)
class CacheDitResolved:
    enabled: bool; profile_id: str | None; Fn_compute_blocks: int; Bn_compute_blocks: int
    max_warmup_steps: int; residual_diff_threshold: float; max_continuous_cached_steps: int
    enable_taylorseer: bool; taylorseer_order: int; scm_preset: str; scm_policy: str
    steps_computation_mask: list[int] | None = None

def _load_profile(profile_id: str) -> dict[str, Any]:
    path = _PROFILES_DIR / f"{profile_id}.json"
    if not path.is_file(): raise FileNotFoundError(f"未找到 quality profile：{profile_id} ({path})")
    data = json.loads(path.read_text(encoding="utf-8"))
    if str(data.get("profile_id", "")) != profile_id:
        raise ValueError(f"profile_id 与文件名不一致：{data.get('profile_id')!r} vs {profile_id!r}")
    if str(data.get("status", "")) != "validated": raise ValueError(f"profile {profile_id!r} status 非 validated")
    return data

def _pkg_ok(required: Mapping[str, str]) -> None:
    for name, want in required.items():
        try: got = metadata.version(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"启用 Cache-DiT 需要安装 {name}>={want}（当前未安装）") from exc
        if tuple(int(x) for x in got.split(".")[:3]) < tuple(int(x) for x in want.split(".")[:3]):
            raise RuntimeError(f"{name} 版本过低：需要 >={want}，当前 {got}")

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

def _from_cache_dict(raw: Mapping[str, Any], *, profile_id: str | None, rdt_default: float) -> CacheDitResolved:
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

def _manual(rdt: float | None = None, mc: int | None = None, warmup: int | None = None) -> CacheDitResolved:
    return CacheDitResolved(
        enabled=True, profile_id=None, Fn_compute_blocks=CACHE_DIT_FN, Bn_compute_blocks=CACHE_DIT_BN,
        max_warmup_steps=int(warmup if warmup is not None else CACHE_DIT_WARMUP),
        residual_diff_threshold=float(rdt if rdt is not None else CACHE_DIT_RDT_COOKBOOK),
        max_continuous_cached_steps=int(mc if mc is not None else CACHE_DIT_MC),
        enable_taylorseer=CACHE_DIT_TAYLORSEER, taylorseer_order=CACHE_DIT_TS_ORDER,
        scm_preset=CACHE_DIT_SCM_PRESET, scm_policy=CACHE_DIT_SCM_POLICY,
    )

def resolve_cache_dit_request(
    mode: str, *, task: str, target: Mapping[str, Any], sigma_points: int,
    video_shift: float, audio_shift: float, rdt: float | None = None,
    mc: int | None = None, warmup: int | None = None,
) -> CacheDitResolved | None:
    """解析 Dual Sigma Sampler 的 cache_dit 模式；off/不匹配 → None。"""
    key = str(mode or CACHE_DIT_MODE_OFF).strip().lower()
    if key in ("", CACHE_DIT_MODE_OFF, "none", "false", "0"): return None
    if key == CACHE_DIT_MODE_MANUAL:
        cfg = _manual(rdt=rdt, mc=mc, warmup=warmup); _pkg_ok({CACHE_DIT_PKG: CACHE_DIT_MIN_VERSION}); return cfg
    profile_id = CACHE_DIT_PROFILE_ID if key == CACHE_DIT_MODE_AUTO else key
    data = _load_profile(profile_id)
    matched = any(_workload_match(wl, task=task, target=target, sigma_points=sigma_points,
                                  video_shift=video_shift, audio_shift=audio_shift)
                  for wl in data.get("workloads", ()))
    if key == CACHE_DIT_MODE_AUTO and not matched:
        LOGGER.warning("cache_dit=auto：当前 workload 未命中 %s，保持关闭", profile_id); return None
    if key != CACHE_DIT_MODE_AUTO and not matched:
        raise ValueError(
            f"cache_dit={profile_id!r} 仅验证过特定 workload（如 1344x768/124f/50steps/shift12·3）；"
            "当前请求不匹配。可改用 manual，或把参数调到已验证合同。"
        )
    _pkg_ok(data.get("required_packages") or {CACHE_DIT_PKG: CACHE_DIT_MIN_VERSION})
    raw = dict(data.get("cache_dit") or {})
    raw.setdefault("residual_diff_threshold", CACHE_DIT_RDT_PROFILE)
    return _from_cache_dict(raw, profile_id=profile_id, rdt_default=CACHE_DIT_RDT_PROFILE)

__all__ = ["CacheDitResolved", "resolve_cache_dit_request"]
