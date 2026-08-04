"""Observability: stage timing, CUDA events, peak VRAM, and per-step percentiles."""
from __future__ import annotations
import logging, statistics, time
from contextlib import contextmanager
from typing import Any, Iterator
from .h3_settings import (
    OPT_TELEMETRY, TELEMETRY_CUDA_EVENTS, TELEMETRY_STEP_ABORT_SECONDS,
)

LOGGER = logging.getLogger(__name__)

def _now() -> float: return time.perf_counter()

def _cuda_mem(device: Any = None) -> dict[str, int | None]:
    out: dict[str, int | None] = {"allocated": None, "reserved": None, "free": None, "total": None}
    try:
        import torch
        if not torch.cuda.is_available(): return out
        d = torch.device(device) if device is not None else torch.device("cuda")
        if d.type != "cuda": return out
        out["allocated"] = int(torch.cuda.memory_allocated(d))
        out["reserved"] = int(torch.cuda.memory_reserved(d))
        free, total = torch.cuda.mem_get_info(d)
        out["free"], out["total"] = int(free), int(total)
    except Exception: pass
    return out

def _percentile(xs: list[float], p: float) -> float | None:
    if not xs: return None
    s = sorted(xs); k = (len(s) - 1) * p; f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)

class H3Telemetry:
    def __init__(self, *, device: Any = None, enabled: bool | None = None) -> None:
        self.enabled = OPT_TELEMETRY if enabled is None else bool(enabled)
        self.device = device
        self.stages: dict[str, float] = {}
        self.step_host_s: list[float] = []
        self.step_cuda_ms: list[float] = []
        self.peak = {"allocated": 0, "reserved": 0}
        self.meta: dict[str, Any] = {}
        self._abort_streak = 0
        self.aborted_reason: str | None = None

    def note(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)

    def snap_peak(self) -> None:
        if not self.enabled: return
        m = _cuda_mem(self.device)
        for k in ("allocated", "reserved"):
            v = m.get(k)
            if v is not None: self.peak[k] = max(int(self.peak[k]), int(v))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield; return
        t0 = _now(); ev0 = ev1 = None
        if TELEMETRY_CUDA_EVENTS:
            try:
                import torch
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
            except Exception: ev0 = ev1 = None
        try:
            yield
        finally:
            host = _now() - t0
            if ev0 is not None and ev1 is not None:
                try:
                    import torch
                    ev1.record(); torch.cuda.synchronize(self.device)
                    self.stages[name] = float(ev0.elapsed_time(ev1)) / 1000.0  # Prefer CUDA seconds.
                except Exception:
                    self.stages[name] = host
            else:
                self.stages[name] = host
            self.snap_peak()
            LOGGER.info("telemetry stage=%s host/cuda_s=%.4f", name, self.stages[name])

    @contextmanager
    def denoise_step(self, step: int, *, abort_exempt: bool = False) -> Iterator[None]:
        if not self.enabled:
            yield; return
        t0 = _now(); ev0 = ev1 = None
        if TELEMETRY_CUDA_EVENTS:
            try:
                import torch
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
                    ev0.record()
            except Exception: ev0 = ev1 = None
        try:
            yield
        finally:
            host = _now() - t0
            self.step_host_s.append(host)
            if ev0 is not None and ev1 is not None:
                try:
                    import torch
                    ev1.record(); torch.cuda.synchronize(self.device)
                    self.step_cuda_ms.append(float(ev0.elapsed_time(ev1)))
                except Exception: pass
            self.snap_peak()
            if not abort_exempt and host > float(TELEMETRY_STEP_ABORT_SECONDS):
                self._abort_streak += 1
                if self._abort_streak >= 2:
                    self.aborted_reason = f"step>{TELEMETRY_STEP_ABORT_SECONDS}s twice (last={host:.1f}s @step={step})"
            else:
                self._abort_streak = 0

    def summary(self) -> dict[str, Any]:
        steps = self.step_host_s
        return {
            "stages_s": dict(self.stages),
            "step_count": len(steps),
            "step_host_s": {
                "p50": _percentile(steps, 0.50),
                "p90": _percentile(steps, 0.90),
                "p95": _percentile(steps, 0.95),
                "mean": statistics.fmean(steps) if steps else None,
                "max": max(steps) if steps else None,
            },
            "step_cuda_ms": {
                "p50": _percentile(self.step_cuda_ms, 0.50),
                "p90": _percentile(self.step_cuda_ms, 0.90),
                "p95": _percentile(self.step_cuda_ms, 0.95),
            } if self.step_cuda_ms else None,
            "peak_vram": dict(self.peak),
            "mem_end": _cuda_mem(self.device),
            "aborted_reason": self.aborted_reason,
            **self.meta,
        }

def attach_telemetry(obj: Any, tel: H3Telemetry | None) -> None:
    if tel is None: return
    try: object.__setattr__(obj, "_h3_telemetry", tel)
    except Exception: pass

def get_telemetry(obj: Any) -> H3Telemetry | None:
    tel = getattr(obj, "_h3_telemetry", None)
    return tel if isinstance(tel, H3Telemetry) else None
