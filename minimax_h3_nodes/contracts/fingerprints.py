"""contracts.fingerprints facade (implementation lives in _impl)."""
from __future__ import annotations
from ._impl import (
    component_compatibility_fingerprint,
    compute_component_fingerprint,
    compute_release_fingerprint,
    condition_order_fingerprint,
    material_compatibility_fingerprint,
    target_compatibility_fingerprint,
)
__all__ = [
    "component_compatibility_fingerprint",
    "compute_component_fingerprint",
    "compute_release_fingerprint",
    "condition_order_fingerprint",
    "material_compatibility_fingerprint",
    "target_compatibility_fingerprint",
]
