"""media_conditioning.reference_video facade."""
from __future__ import annotations
from ._impl import (
    resolve_reference_video_shape,
    reference_video_display_geometry,
    ReferenceVideoMetadata,
    ReferenceVideoPlan,
    build_reference_video_plan,
    reference_video_normalize_command,
    reference_video_truncate_command,
    validate_prepared_reference_video,
    execute_reference_video_plan,
    decode_reference_video_samples,
)
__all__ = [
    "resolve_reference_video_shape",
    "reference_video_display_geometry",
    "ReferenceVideoMetadata",
    "ReferenceVideoPlan",
    "build_reference_video_plan",
    "reference_video_normalize_command",
    "reference_video_truncate_command",
    "validate_prepared_reference_video",
    "execute_reference_video_plan",
    "decode_reference_video_samples",
]
