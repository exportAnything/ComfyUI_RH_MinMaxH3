"""Fork-owned attention patch nodes for native ComfyUI MiniMax H3 models."""
from __future__ import annotations

import logging
from typing import Any

from ..runtime.attention_backends import (
    clone_model_with_attention_override,
    make_sage_attention_override,
    make_sol_attention_override,
)


LOGGER = logging.getLogger(__name__)
CATEGORY = "RunningHub/MiniMax H3/attention"


def _current_attention_override(model: Any):
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        return None
    transformer_options = model_options.get("transformer_options")
    if not isinstance(transformer_options, dict):
        return None
    return transformer_options.get("optimized_attention_override")


class MiniMaxH3SageAttentionPatch:
    """Use SageAttention when available without depending on another node pack."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect a native ComfyUI MiniMax H3 MODEL. This node is "
                            "already included in the RunningHub MiniMax H3 pack."
                        )
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Use SageAttention's automatic backend selection. If a compatible "
                            "SageAttention Python/CUDA build is unavailable, the model passes "
                            "through and ComfyUI's normal attention remains active."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Patches native ComfyUI attention with SageAttention auto-selection. The node itself "
        "is bundled with this repository; only the optional hardware-specific SageAttention "
        "Python/CUDA library is external. Missing or incompatible kernels fall back safely."
    )

    def patch(self, model: Any, enabled: bool = True):
        if not enabled:
            return (model,)
        override = make_sage_attention_override(allow_compile=False)
        if override is None:
            return (model,)
        return (clone_model_with_attention_override(model, override),)


class MiniMaxH3SolAttentionPatch:
    """Use the bundled experimental Sol-style FlexAttention middle-step path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect the bundled SageAttention patch before this node. Sol-style "
                            "sparse attention handles eligible middle steps; Sage handles the "
                            "early, late, short, masked, or unsupported calls."
                        )
                    },
                ),
                "enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Enable the bundled experimental Sol-style sparse FlexAttention path."
                        ),
                    },
                ),
                "tau": (
                    "FLOAT",
                    {
                        "default": 1.2,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Sparse-routing threshold. Higher values keep fewer exact blocks and "
                            "may run faster, but can change output quality. Start with 1.2."
                        ),
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Keep the beginning of sampling on the previous backend. Sol-style "
                            "attention starts at this fraction of the schedule."
                        ),
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Return to the previous backend after this fraction of the sampling "
                            "schedule."
                        ),
                    },
                ),
                "min_tokens": (
                    "INT",
                    {
                        "default": 4096,
                        "min": 0,
                        "max": 1048576,
                        "step": 512,
                        "tooltip": (
                            "Sequences shorter than this stay on the previous attention backend. "
                            "Sparse routing normally helps only on long H3 sequences."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY
    EXPERIMENTAL = True
    DESCRIPTION = (
        "Experimental MiniMax H3 Sol-style block-sparse FlexAttention patch bundled with this "
        "repository. It is not the official NVIDIA Sol-Attn kernel and can change quality. "
        "Eligible middle sampling steps use sparse attention; every other call delegates to the "
        "previous patch (normally SageAttention) or to ComfyUI's standard attention."
    )

    def patch(
        self,
        model: Any,
        enabled: bool = True,
        tau: float = 1.2,
        start_percent: float = 0.2,
        end_percent: float = 0.9,
        min_tokens: int = 4096,
    ):
        if not enabled:
            return (model,)
        start_percent = float(start_percent)
        end_percent = float(end_percent)
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise ValueError(
                "Sol-style attention requires 0 <= start_percent <= end_percent <= 1."
            )

        get_model_object = getattr(model, "get_model_object", None)
        if not callable(get_model_object):
            raise TypeError(
                "The Sol-style patch requires a standard ComfyUI MODEL with model_sampling."
            )
        model_sampling = get_model_object("model_sampling")
        percent_to_sigma = getattr(model_sampling, "percent_to_sigma", None)
        if not callable(percent_to_sigma):
            raise TypeError(
                "The Sol-style patch could not read the ComfyUI model sampling schedule."
            )

        sigma_start = float(percent_to_sigma(start_percent))
        sigma_end = float(percent_to_sigma(end_percent))
        previous = _current_attention_override(model)
        if previous is not None:
            LOGGER.info(
                "The bundled Sol-style patch will delegate ineligible calls to the existing "
                "attention override."
            )
        override = make_sol_attention_override(
            tau=float(tau),
            routing="diag",
            min_tokens=int(min_tokens),
            sigma_start=sigma_start,
            sigma_end=sigma_end,
            previous=previous,
        )
        return (clone_model_with_attention_override(model, override),)


__all__ = [
    "MiniMaxH3SageAttentionPatch",
    "MiniMaxH3SolAttentionPatch",
]
