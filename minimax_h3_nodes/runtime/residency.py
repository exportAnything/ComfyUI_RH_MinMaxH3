"""Worker-level model residency leases: gpu-resident / layerwise-warm / cold."""
from __future__ import annotations
import logging, threading, time
from typing import Any
from .h3_settings import (
    OPT_RESIDENCY_LEASE, RESIDENCY_POLICY, RESIDENCY_TTL_SECONDS, RESIDENCY_VAE_TTL_SECONDS,
)

LOGGER = logging.getLogger(__name__)
_LOCK = threading.RLock()
_MGR: "H3ResidencyManager | None" = None

def lease_key(handle: Any, *, kind: str = "dit") -> str:
    meta = getattr(handle, "metadata", None) or {}
    fp = meta.get("release_fingerprint") or getattr(handle, "component_path", "") or id(handle)
    part = meta.get("partition") or ""
    quant = "int8" if getattr(handle, "quantized", False) else "bf16"
    dev = str(getattr(handle, "load_device", "cuda"))
    return f"{kind}|{fp}|{part}|{quant}|{dev}"

class H3ResidencyManager:
    def __init__(self) -> None:
        self._leases: dict[str, dict[str, Any]] = {}

    def acquire(self, handle: Any, *, kind: str = "dit") -> str:
        key = lease_key(handle, kind=kind)
        with _LOCK:
            self.expire()
            ent = self._leases.get(key)
            if ent is not None:
                ent["last_used"] = time.monotonic(); ent["handle"] = handle
                LOGGER.info("residency acquire hit key=%s state=%s", key, ent.get("state"))
                return str(ent.get("state") or "warm")
            self._leases[key] = {"handle": handle, "kind": kind, "state": "acquired", "last_used": time.monotonic()}
            return "acquired"

    def release(self, handle: Any, *, kind: str = "dit", force_cold: bool = False) -> str:
        key = lease_key(handle, kind=kind)
        policy = str(RESIDENCY_POLICY).strip().lower()
        cold = force_cold or (not OPT_RESIDENCY_LEASE) or policy == "safe"
        with _LOCK:
            if cold:
                self._cold(handle); self._leases.pop(key, None)
                LOGGER.info("residency release cold key=%s", key); return "cold"
            park = getattr(handle, "park_after_inference", None)
            state = park() if callable(park) else self._cold(handle) or "cold"
            # balanced: downgrade to cold when VRAM is tight.
            if policy == "balanced" and state == "gpu-resident" and self._vram_tight(handle):
                self._cold(handle); state = "cold"; self._leases.pop(key, None)
                LOGGER.info("residency balanced demote to cold key=%s", key); return state
            self._leases[key] = {"handle": handle, "kind": kind, "state": state, "last_used": time.monotonic()}
            LOGGER.info("residency park key=%s state=%s policy=%s", key, state, policy)
            return state

    def free_memory(self, *_a, **_k) -> None:
        with _LOCK:
            for key in list(self._leases):
                ent = self._leases.pop(key); self._cold(ent.get("handle"))
            LOGGER.info("residency free_memory: cleared all leases")

    def expire(self) -> None:
        now = time.monotonic()
        with _LOCK:
            for key, ent in list(self._leases.items()):
                ttl = RESIDENCY_VAE_TTL_SECONDS if ent.get("kind") == "vae" else RESIDENCY_TTL_SECONDS
                if now - float(ent.get("last_used", now)) >= float(ttl):
                    self._cold(ent.get("handle")); self._leases.pop(key, None)
                    LOGGER.info("residency TTL expire key=%s", key)

    @staticmethod
    def _cold(handle: Any) -> str:
        if handle is None: return "cold"
        off = getattr(handle, "offload_after_inference", None) or getattr(handle, "offload", None)
        if callable(off):
            try: off()
            except Exception as exc: LOGGER.warning("residency cold offload failed: %s", exc)
        return "cold"

    @staticmethod
    def _vram_tight(handle: Any) -> bool:
        try:
            from .model_loader import _device_free_bytes
            free = _device_free_bytes(getattr(handle, "load_device", "cuda"))
            return free is not None and free < (4 << 30)
        except Exception:
            return False

def get_residency_manager() -> H3ResidencyManager:
    global _MGR
    with _LOCK:
        if _MGR is None: _MGR = H3ResidencyManager()
        return _MGR
