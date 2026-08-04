"""Upstream H3 whole-step velocity cache: periodic refresh, Taylor extrapolation, and dense tail."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class VelocityCacheConfig:
    stride: int = 1; taylorseer: bool = False; taylorseer_order: int = 1
    tail_dense_steps: int = 1; tail_rebalance: bool = False; final_refresh: bool = True
    profile_id: str | None = None
    def __post_init__(self):
        if self.stride < 1: raise ValueError("velocity_cache stride must be >= 1")
        if self.tail_dense_steps < 1: raise ValueError("velocity_cache tail_dense_steps must be >= 1")
        if self.taylorseer_order not in (1, 2): raise ValueError("taylorseer_order must be 1 or 2")

def should_refresh_velocity(step: int, num_steps: int, cfg: VelocityCacheConfig) -> bool:
    """Bit-exact alignment with the upstream denoise_loop refresh decision (zero-based step)."""
    if cfg.stride == 1 or step < 2: return True
    final_step = step == num_steps - 1
    tail_start = num_steps - cfg.tail_dense_steps
    periodic = (step - 1) % cfg.stride == 0
    if cfg.tail_rebalance and cfg.tail_dense_steps > 1:
        midpoint = num_steps // 2
        rebalance_step = 1 + round((midpoint - 1) / cfg.stride) * cfg.stride
        if step == rebalance_step: periodic = False
    return (not final_step or cfg.final_refresh) and (step >= tail_start or periodic)

@dataclass
class VelocityCacheRuntime:
    cfg: VelocityCacheConfig
    cached_video: Any = None; cached_audio: Any = None
    video_d1: Any = None; audio_d1: Any = None; video_d2: Any = None; audio_d2: Any = None
    last_refresh_step: int = -1
    dit_calls: int = 0; cache_hits: int = 0; taylorseer_steps: int = 0

    def refresh(self, step: int) -> bool: return should_refresh_velocity(step, self._n, self.cfg)
    def bind(self, num_steps: int) -> "VelocityCacheRuntime": self._n = int(num_steps); return self

    def on_dit(self, step: int, v_video: Any, v_audio: Any) -> tuple[Any, Any]:
        prev_v, prev_a, prev_step = self.cached_video, self.cached_audio, self.last_refresh_step
        self.cached_video, self.cached_audio = v_video, v_audio
        if (self.cfg.taylorseer and prev_v is not None and prev_a is not None and prev_step > 0):
            window = step - prev_step
            nv = (v_video - prev_v) / window; na = (v_audio - prev_a) / window
            if self.cfg.taylorseer_order == 2 and self.video_d1 is not None and self.audio_d1 is not None:
                self.video_d2 = (nv - self.video_d1) / window
                self.audio_d2 = (na - self.audio_d1) / window
            self.video_d1, self.audio_d1 = nv, na
        self.last_refresh_step = step; self.dit_calls += 1
        return v_video, v_audio

    def on_hit(self, step: int) -> tuple[Any, Any]:
        self.cache_hits += 1
        assert self.cached_video is not None and self.cached_audio is not None
        if not (self.cfg.taylorseer and self.video_d1 is not None and self.audio_d1 is not None):
            return self.cached_video, self.cached_audio
        elapsed = step - self.last_refresh_step
        mv, ma = self.cached_video + self.video_d1 * elapsed, self.cached_audio + self.audio_d1 * elapsed
        if self.cfg.taylorseer_order == 2 and self.video_d2 is not None and self.audio_d2 is not None:
            scale = 0.5 * elapsed * elapsed
            mv = mv + self.video_d2 * scale; ma = ma + self.audio_d2 * scale
        self.taylorseer_steps += 1
        return mv, ma

    def stats(self) -> dict[str, int]:
        return {
            "stride": self.cfg.stride, "global_steps": getattr(self, "_n", 0),
            "dit_calls": self.dit_calls, "cache_hits": self.cache_hits,
            "taylorseer_steps": self.taylorseer_steps,
            "taylorseer_order": self.cfg.taylorseer_order if self.cfg.taylorseer else 0,
            "tail_dense_steps": self.cfg.tail_dense_steps,
            "tail_rebalance": int(self.cfg.tail_rebalance),
            "final_refresh": int(self.cfg.final_refresh),
        }

__all__ = ["VelocityCacheConfig", "VelocityCacheRuntime", "should_refresh_velocity"]
