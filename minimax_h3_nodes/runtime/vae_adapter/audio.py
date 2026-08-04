"""vae_adapter.audio facade."""
from __future__ import annotations
from ._impl import (
    MiniMaxH3AudioVAEAdapter,
    load_audio_vae,
)
__all__ = [
    "MiniMaxH3AudioVAEAdapter",
    "load_audio_vae",
]
