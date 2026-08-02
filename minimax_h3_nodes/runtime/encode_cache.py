"""Qwen/条件编码分层 LRU 缓存；值驻 CPU，按字节上限淘汰。"""
from __future__ import annotations
import hashlib, logging, threading
from collections import OrderedDict
from typing import Any
from .h3_settings import ENCODE_CACHE_MAX_BYTES, OPT_ENCODE_CACHE

LOGGER = logging.getLogger(__name__)
_LOCK = threading.RLock()

def _hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="replace")); h.update(b"\0")
    return h.hexdigest()

def prompt_cache_key(*, prompt: str, task: str = "", te_fingerprint: str = "", tok_fingerprint: str = "") -> str:
    return "txt:" + _hash("v1", prompt.strip(), task, te_fingerprint, tok_fingerprint)

def visual_cache_key(*, material_fp: str, plan: str = "", te_fingerprint: str = "") -> str:
    return "vis:" + _hash("v1", material_fp, plan, te_fingerprint)

def vae_rows_cache_key(*, material_fp: str, target_fp: str = "", vae_fp: str = "", kind: str = "") -> str:
    return "vae:" + _hash("v1", material_fp, target_fp, vae_fp, kind)

class EncodeCache:
    def __init__(self, max_bytes: int = ENCODE_CACHE_MAX_BYTES) -> None:
        self.max_bytes = int(max_bytes); self._od: OrderedDict[str, tuple[Any, int]] = OrderedDict(); self._bytes = 0

    def get(self, key: str) -> Any | None:
        if not OPT_ENCODE_CACHE: return None
        with _LOCK:
            item = self._od.get(key)
            if item is None: return None
            self._od.move_to_end(key); LOGGER.info("encode_cache HIT %s", key[:48]); return item[0]

    def put(self, key: str, value: Any, *, nbytes: int | None = None) -> None:
        if not OPT_ENCODE_CACHE: return
        size = int(nbytes) if nbytes is not None else _estimate_nbytes(value)
        if size <= 0 or size > self.max_bytes: return
        with _LOCK:
            old = self._od.pop(key, None)
            if old is not None: self._bytes -= old[1]
            while self._od and self._bytes + size > self.max_bytes:
                _, (_, sz) = self._od.popitem(last=False); self._bytes -= sz
            self._od[key] = (value, size); self._bytes += size
            LOGGER.info("encode_cache PUT %s bytes=%s total=%s", key[:48], size, self._bytes)

    def clear(self) -> None:
        with _LOCK: self._od.clear(); self._bytes = 0

def _estimate_nbytes(value: Any) -> int:
    try:
        if hasattr(value, "numel") and hasattr(value, "element_size"):
            return int(value.numel()) * int(value.element_size())
        if isinstance(value, (bytes, bytearray)): return len(value)
        if isinstance(value, dict):
            return sum(_estimate_nbytes(v) for v in value.values())
    except Exception: pass
    return 0

_CACHE: EncodeCache | None = None

def get_encode_cache() -> EncodeCache:
    global _CACHE
    with _LOCK:
        if _CACHE is None: _CACHE = EncodeCache()
        return _CACHE
