"""media_conditioning.reference_image facade。"""
from __future__ import annotations
from ._impl import (
    H3_REFERENCE_IMAGE_SIZE_MATCH,
    H3_REFERENCE_IMAGE_SIZE_MAX,
    H3_REFERENCE_IMAGE_SIZE_MODES,
    resolve_reference_image_shape,
    prepare_reference_image,
)
__all__ = [
    "H3_REFERENCE_IMAGE_SIZE_MATCH",
    "H3_REFERENCE_IMAGE_SIZE_MAX",
    "H3_REFERENCE_IMAGE_SIZE_MODES",
    "resolve_reference_image_shape",
    "prepare_reference_image",
]
