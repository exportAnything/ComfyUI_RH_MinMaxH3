"""contracts.components facade（实现见 _impl）。"""
from __future__ import annotations
from ._impl import (
    validate_component_compatibility,
    validate_component_for_task,
    validate_release_for_task,
)
__all__ = [
    "validate_component_compatibility",
    "validate_component_for_task",
    "validate_release_for_task",
]
