"""Sampling facade; implementation lives in runtime.sampler_core."""
from __future__ import annotations
from .runtime.sampler_core import *  # noqa: F403
from .runtime import sampler_core as _sampler_core
globals().update({k: getattr(_sampler_core, k) for k in dir(_sampler_core) if not k.startswith("__")})
from .runtime.sampler_core import __all__  # noqa: F401
