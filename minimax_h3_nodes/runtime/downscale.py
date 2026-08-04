"""Fine-grained downscaling that preserves the source aspect ratio and reuses dynamic activation-reserve rejection hints."""
from __future__ import annotations
from .h3_settings import H3_DOWNSCALE_16_9, H3_DOWNSCALE_SHORT_EDGES
from ..contracts import H3_CANVAS_MULTIPLE, H3_MAX_PIXELS

def _align(v: int, multiple: int = H3_CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(v / multiple) * multiple))

def _is_approx_16_9(width: int, height: int) -> bool:
    return abs((width / height) - (16 / 9)) < 0.05

def downscale_chain(width: int, height: int) -> list[tuple[int, int]]:
    """Generate an aspect-preserving downscale chain. Use upstream buckets for 16:9 and short-edge steps for other ratios."""
    w, h = _align(int(width)), _align(int(height))
    if w <= 0 or h <= 0: raise ValueError("width/height must be positive")
    if _is_approx_16_9(w, h):
        chain = [p for p in H3_DOWNSCALE_16_9 if p[0] * p[1] <= w * h]
        return chain or list(H3_DOWNSCALE_16_9[-1:])
    short, long_ = (w, h) if w <= h else (h, w)
    landscape = w >= h
    out: list[tuple[int, int]] = [(w, h)]
    for edge in H3_DOWNSCALE_SHORT_EDGES:
        if edge >= short: continue
        scale = edge / float(short)
        nw = _align(long_ * scale) if landscape else _align(edge)
        nh = _align(edge) if landscape else _align(long_ * scale)
        if nw * nh > H3_MAX_PIXELS:
            r = (H3_MAX_PIXELS / float(max(1, nw * nh))) ** 0.5
            nw, nh = _align(nw * r), _align(nh * r)
        if nw * nh < out[-1][0] * out[-1][1]:
            out.append((nw, nh))
    return out

def next_downscale(width: int, height: int) -> tuple[int, int] | None:
    chain = downscale_chain(width, height)
    cur = (_align(width), _align(height))
    for i, pair in enumerate(chain):
        if pair == cur and i + 1 < len(chain): return chain[i + 1]
    for pair in chain:
        if pair[0] * pair[1] < cur[0] * cur[1]: return pair
    return None

def format_downscale_hint(width: int = 1344, height: int = 768) -> str:
    return "→".join(f"{a}x{b}" for a, b in downscale_chain(width, height))
