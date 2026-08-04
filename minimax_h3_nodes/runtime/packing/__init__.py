"""packing package facade."""
from __future__ import annotations
from ._core import *  # noqa: F403
from .sequences import *  # noqa: F403
from .builders import *  # noqa: F403
from . import _core, sequences, builders
for _m in (_core, sequences, builders):
    globals().update({k: getattr(_m, k) for k in dir(_m) if not k.startswith("__")})
__all__ = [n for n in list(globals()) if not n.startswith("__")]
