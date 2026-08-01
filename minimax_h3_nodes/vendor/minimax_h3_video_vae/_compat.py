# SPDX-License-Identifier: Apache-2.0
"""Small Diffusers compatibility surface used by the released visual VAE.

Only module/config registration is needed for direct inference.  Keeping it
local avoids making a Diffusers installation or a Diffusers pipeline part of
the ComfyUI node runtime.
"""

from __future__ import annotations

import inspect
import logging as python_logging
from functools import wraps

import torch.nn as nn


class ConfigDict(dict):
    """Dictionary with the attribute access used by Diffusers ConfigMixin."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class ConfigMixin:
    config: ConfigDict

    def register_to_config(self, **kwargs):
        if not hasattr(self, "config"):
            self.config = ConfigDict()
        self.config.update(kwargs)


class ModelMixin(nn.Module):
    pass


class FromOriginalModelMixin:
    pass


def register_to_config(init):
    """Capture constructor values in ``self.config`` before model creation."""

    signature = inspect.signature(init)

    @wraps(init)
    def wrapped(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("self", None)
        extra = values.pop("kwargs", {})
        self.config = ConfigDict(values)
        if isinstance(extra, dict):
            self.config.update(extra)
        return init(self, *args, **kwargs)

    return wrapped


def maybe_allow_in_graph(value):
    return value


class _LoggingFacade:
    @staticmethod
    def get_logger(name):
        return python_logging.getLogger(name)


logging = _LoggingFacade()


__all__ = [
    "ConfigDict",
    "ConfigMixin",
    "FromOriginalModelMixin",
    "ModelMixin",
    "logging",
    "maybe_allow_in_graph",
    "register_to_config",
]
