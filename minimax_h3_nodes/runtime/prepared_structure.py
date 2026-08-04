"""Prepared-structure entry point; implementation lives in MiniMaxH3DiTModel.prepare_structure."""
from __future__ import annotations
from .attention import normalize_cu_seqlens_bounds
from .h3_settings import OPT_PREPARED_STRUCTURE, DIT_DEBUG_STRUCTURE_CHECKS
__all__ = ["normalize_cu_seqlens_bounds", "OPT_PREPARED_STRUCTURE", "DIT_DEBUG_STRUCTURE_CHECKS"]
