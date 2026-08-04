"""Lifecycle entry point; implementation lives in residency."""
from .residency import H3ResidencyManager, get_residency_manager, lease_key
__all__ = ["H3ResidencyManager", "get_residency_manager", "lease_key"]
