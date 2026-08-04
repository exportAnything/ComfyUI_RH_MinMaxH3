"""contracts.target facade (implementation lives in _impl)."""
from __future__ import annotations
from ._impl import (
    append_ref2va_reference,
    make_fl2va_keyframe,
    make_ref2va_reference,
    make_ref2va_references,
    resolve_deferred_target_v2,
    resolve_fl2va_target_v2,
    resolve_ref2va_target_v2,
    resolve_t2va_target,
    resolve_t2va_target_v2,
    validate_fl2va_keyframes,
    validate_ref2va_references,
    validate_target,
    validate_target_v2,
)
__all__ = [
    "append_ref2va_reference",
    "make_fl2va_keyframe",
    "make_ref2va_reference",
    "make_ref2va_references",
    "resolve_deferred_target_v2",
    "resolve_fl2va_target_v2",
    "resolve_ref2va_target_v2",
    "resolve_t2va_target",
    "resolve_t2va_target_v2",
    "validate_fl2va_keyframes",
    "validate_ref2va_references",
    "validate_target",
    "validate_target_v2",
]
