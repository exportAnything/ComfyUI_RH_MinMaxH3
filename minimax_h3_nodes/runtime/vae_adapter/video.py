"""vae_adapter.video facade."""
from __future__ import annotations
from ._impl import (
    MiniMaxH3VideoVAEAdapter,
    load_video_vae,
)
__all__ = [
    "MiniMaxH3VideoVAEAdapter",
    "load_video_vae",
]
