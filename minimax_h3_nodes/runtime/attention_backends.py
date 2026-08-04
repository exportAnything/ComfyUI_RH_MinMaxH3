"""ComfyUI MODEL attention patches bundled with the MiniMax H3 node pack.

The native ComfyUI MiniMax H3 implementation calls ``optimized_attention``
with HND tensors (``[batch, heads, sequence, head_dim]``).  The helpers in
this module install a standard ``optimized_attention_override`` on a cloned
ComfyUI ``ModelPatcher`` without importing another custom-node package.

SageAttention remains an optional Python/CUDA library.  If it is unavailable,
the Sage patch node leaves the model unchanged so a workflow remains runnable.
The Sol-Attn backend is implemented here with PyTorch ``flex_attention`` and
falls back to the original ComfyUI attention implementation when its optimized
path is not applicable. Its routing code is a substantially modified Apache-2.0
adaptation of KingGore/ComfyUI_sol-attn_Blackwell commit ``de7ffe3``; see
``NOTICE.md`` for provenance and a summary of the changes.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)

SOL_ROUTING_MODES = ("diag",)
SOL_HEAD_DIM = 128
SOL_FLEX_BLOCK = 128

_compiled_flex_attention: Callable[..., torch.Tensor] | None = None
_warned_messages: set[str] = set()
_info_messages: set[str] = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key in _warned_messages:
        return
    _warned_messages.add(key)
    LOGGER.warning(message, *args)


def _info_once(key: str, message: str, *args: Any) -> None:
    if key in _info_messages:
        return
    _info_messages.add(key)
    LOGGER.info(message, *args)


def _call_original_attention(
    function: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask: torch.Tensor | None,
    attn_precision: torch.dtype | None,
    skip_reshape: bool,
    skip_output_reshape: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    return function(
        q,
        k,
        v,
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs,
    )


def _call_chained_attention(
    previous: Callable[..., Any] | None,
    function: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    *,
    mask: torch.Tensor | None,
    attn_precision: torch.dtype | None,
    skip_reshape: bool,
    skip_output_reshape: bool,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Delegate to an earlier override (for example Sage), then ComfyUI."""

    if previous is None:
        return _call_original_attention(
            function,
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            kwargs=kwargs,
        )
    return previous(
        function,
        q,
        k,
        v,
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs,
    )


def clone_model_with_attention_override(model: Any, override: Callable[..., Any]) -> Any:
    """Clone a ComfyUI ModelPatcher and install one attention override safely."""

    clone = getattr(model, "clone", None)
    if not callable(clone):
        raise TypeError(
            "The attention patch requires a standard ComfyUI MODEL input "
            "with a clone() method."
        )
    patched = clone()
    options = getattr(patched, "model_options", None)
    if not isinstance(options, dict):
        raise TypeError(
            "The attention patch requires a standard ComfyUI MODEL input "
            "with model_options."
        )

    # Clone both mapping levels so parallel branches cannot overwrite one
    # another through a shared nested transformer_options dictionary.
    model_options = dict(options)
    transformer_options = dict(model_options.get("transformer_options") or {})
    transformer_options["optimized_attention_override"] = override
    model_options["transformer_options"] = transformer_options
    patched.model_options = model_options
    return patched


def _get_registered_sage_attention(
    *, allow_compile: bool,
) -> Callable[..., torch.Tensor] | None:
    """Return ComfyUI's unwrapped Sage implementation when it is registered."""

    try:
        from comfy.ldm.modules.attention import get_attention_function
    except ImportError:
        return None

    registered = get_attention_function("sage", None)
    if not callable(registered):
        return None

    # Registered attention functions are wrapped by ComfyUI's override hook.
    # Calling the underlying function avoids entering this same override again.
    implementation = getattr(registered, "__wrapped__", registered)
    compiler = getattr(torch, "compiler", None)
    if not allow_compile and compiler is not None and hasattr(compiler, "disable"):
        implementation = compiler.disable(implementation)
    return implementation


def make_sage_attention_override(
    *,
    allow_compile: bool = False,
) -> Callable[..., Any] | None:
    """Build an override around ComfyUI's registered SageAttention backend."""

    sage_attention = _get_registered_sage_attention(allow_compile=allow_compile)
    if sage_attention is None:
        LOGGER.warning(
            "ComfyUI has no compatible SageAttention backend registered. The model will "
            "use ComfyUI's normal attention. No extra custom-node pack is required; "
            "install a SageAttention build matching this ComfyUI Python, Torch, CUDA, "
            "and GPU only if acceleration is wanted."
        )
        return None

    _info_once(
        "sage-registered",
        "The bundled MiniMax H3 Sage patch is using ComfyUI's registered "
        "SageAttention backend.",
    )

    def override(
        function,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        try:
            return sage_attention(
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        except Exception as exc:
            _warn_once(
                f"sage-runtime-{type(exc).__name__}",
                "The registered SageAttention backend failed (%s: %s); falling back to "
                "ComfyUI attention.",
                type(exc).__name__,
                exc,
            )
            return _call_original_attention(
                function,
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                kwargs=dict(kwargs),
            )

    return override


def _get_compiled_flex_attention() -> Callable[..., torch.Tensor]:
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def _sol_block_means(
    tensor: torch.Tensor,
    blocks: int,
    *,
    real_tokens: int,
) -> torch.Tensor:
    batch, heads, _tokens, dim = tensor.shape
    block_sums = tensor.view(
        batch, heads, blocks, SOL_FLEX_BLOCK, dim
    ).sum(dim=3)
    starts = torch.arange(blocks, device=tensor.device) * SOL_FLEX_BLOCK
    lengths = (int(real_tokens) - starts).clamp(min=1, max=SOL_FLEX_BLOCK)
    return block_sums / lengths.view(1, 1, blocks, 1)


def _build_sol_routing_mask(
    q_h: torch.Tensor,
    k_h: torch.Tensor,
    *,
    real_tokens: int,
    tau: float,
):
    """Build the 128-token sparse routing mask used by the bundled Sol backend."""

    from torch.nn.attention.flex_attention import BlockMask

    batch, heads, padded_tokens, _dim = q_h.shape
    blocks = padded_tokens // SOL_FLEX_BLOCK
    q_mean = _sol_block_means(q_h, blocks, real_tokens=real_tokens)
    k_mean = _sol_block_means(k_h, blocks, real_tokens=real_tokens)

    key_center = k_mean.mean(dim=2, keepdim=True)
    key_variance = k_mean.var(dim=2, unbiased=False, keepdim=True)
    threshold_mean = (q_mean @ key_center.transpose(-1, -2)).squeeze(-1)
    threshold_variance = (
        (q_mean * q_mean) @ key_variance.transpose(-1, -2)
    ).squeeze(-1)
    threshold = threshold_mean + float(tau) * torch.sqrt(
        torch.clamp(threshold_variance, min=0.0) + 1e-6
    )

    pilot_scores = torch.einsum("bhnd,bhmd->bhnm", q_mean, k_mean)
    selected = pilot_scores > threshold.unsqueeze(-1)

    block_index = torch.arange(blocks, device=q_h.device)
    query_block = block_index.view(1, 1, blocks, 1)
    key_block = block_index.view(1, 1, 1, blocks)
    selected |= (query_block - key_block).abs() <= 1

    kv_num_blocks = selected.sum(dim=-1).to(torch.int32)
    expanded_index = block_index.view(1, 1, 1, blocks).expand_as(selected)
    # Selected entries must occupy the prefix consumed by kv_num_blocks.  The
    # secondary key keeps the result deterministic and retains block zero.
    sort_key = selected.to(torch.int64) * (blocks + 1) - expanded_index
    order = torch.argsort(sort_key, dim=-1, descending=True)
    kv_indices = torch.gather(expanded_index, dim=-1, index=order).to(torch.int32)

    def boundary_mask(_batch, _head, query_index, key_index):
        return (query_index < real_tokens) & (key_index < real_tokens)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        BLOCK_SIZE=SOL_FLEX_BLOCK,
        mask_mod=boundary_mask,
        seq_lengths=(padded_tokens, padded_tokens),
    )


def sol_attention_flex(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Run self-contained Sol-style sparse attention on BF16 BTHD tensors."""

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [batch, tokens, heads, 128]")
    if q.shape[-1] != SOL_HEAD_DIM:
        raise ValueError("The MiniMax H3 Sol-Attn backend requires head dimension 128")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("The MiniMax H3 Sol-Attn backend requires bfloat16 tensors")
    if q.device.type != "cuda":
        raise ValueError("The MiniMax H3 Sol-Attn backend requires a CUDA device")

    _batch, tokens, _heads, dim = q.shape
    if tokens == 0:
        return q.clone()
    attention_scale = dim**-0.5 if scale is None else float(scale)
    padded_tokens = (
        (tokens + SOL_FLEX_BLOCK - 1) // SOL_FLEX_BLOCK
    ) * SOL_FLEX_BLOCK

    if padded_tokens != tokens:
        padding = padded_tokens - tokens
        q_padded = F.pad(q, (0, 0, 0, 0, 0, padding))
        k_padded = F.pad(k, (0, 0, 0, 0, 0, padding))
        v_padded = F.pad(v, (0, 0, 0, 0, 0, padding))
    else:
        q_padded, k_padded, v_padded = q, k, v

    q_h = q_padded.permute(0, 2, 1, 3).contiguous()
    k_h = k_padded.permute(0, 2, 1, 3).contiguous()
    v_h = v_padded.permute(0, 2, 1, 3).contiguous()

    block_mask = _build_sol_routing_mask(
        q_h,
        k_h,
        real_tokens=tokens,
        tau=float(tau),
    )
    output_h = _get_compiled_flex_attention()(
        q_h,
        k_h,
        v_h,
        block_mask=block_mask,
        scale=attention_scale,
    )
    _info_once(
        "sol-flex-active",
        "The bundled experimental MiniMax H3 Sol-style sparse "
        "FlexAttention backend is active.",
    )
    output = output_h.permute(0, 2, 1, 3).contiguous()
    return output[:, :tokens]


def make_sol_attention_override(
    *,
    tau: float = 1.0,
    routing: str = "diag",
    min_tokens: int = 4096,
    sigma_start: float | None = None,
    sigma_end: float | None = None,
    previous: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    if routing not in SOL_ROUTING_MODES:
        raise ValueError(f"Unsupported Sol-Attn routing mode: {routing!r}")
    if isinstance(min_tokens, bool) or int(min_tokens) < 0:
        raise ValueError("min_tokens must be a non-negative integer")
    min_tokens = int(min_tokens)

    def override(
        function,
        q,
        k,
        v,
        heads,
        mask=None,
        attn_precision=None,
        skip_reshape=False,
        skip_output_reshape=False,
        **kwargs,
    ):
        fallback_kwargs = dict(kwargs)
        transformer_options = fallback_kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict):
            transformer_options = {}

        # The recommended composition keeps early/late diffusion steps on the
        # previous backend (normally SageAttention) and uses Sol only in the
        # configured middle window.  MiniMax H3 publishes the current sigma in
        # transformer_options for every optimized_attention call.
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None and (sigma_start is not None or sigma_end is not None):
            try:
                sigma = float(sigmas[0])
            except (IndexError, TypeError, ValueError):
                sigma = None
            if sigma is not None and (
                (sigma_start is not None and sigma > float(sigma_start))
                or (sigma_end is not None and sigma < float(sigma_end))
            ):
                return _call_chained_attention(
                    previous,
                    function,
                    q,
                    k,
                    v,
                    heads,
                    mask=mask,
                    attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape,
                    kwargs=fallback_kwargs,
                )

        tokens = q.shape[2] if skip_reshape and q.ndim == 4 else (
            q.shape[1] if q.ndim >= 2 else 0
        )
        compatible = (
            skip_reshape
            and mask is None
            and q.ndim == 4
            and q.shape == k.shape == v.shape
            and q.shape[-1] == SOL_HEAD_DIM
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.device.type == "cuda"
            and not fallback_kwargs.get("enable_gqa", False)
            and fallback_kwargs.get("low_precision_attention", True) is not False
            and attn_precision != torch.float32
            and int(tokens) >= min_tokens
        )
        if not compatible:
            return _call_chained_attention(
                previous,
                function,
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                kwargs=fallback_kwargs,
            )

        try:
            batch, attention_heads, tokens, dim = q.shape
            q_bthd = q.permute(0, 2, 1, 3).contiguous()
            k_bthd = k.permute(0, 2, 1, 3).contiguous()
            v_bthd = v.permute(0, 2, 1, 3).contiguous()
            output = sol_attention_flex(
                q_bthd,
                k_bthd,
                v_bthd,
                scale=fallback_kwargs.get("scale"),
                tau=float(tau),
            )
            if skip_output_reshape:
                return output.permute(0, 2, 1, 3).contiguous()
            return output.reshape(batch, tokens, attention_heads * dim)
        except Exception as exc:
            _warn_once(
                f"sol-runtime-{type(exc).__name__}",
                "The bundled Sol-Attn backend failed (%s: %s); delegating to the "
                "preceding attention backend.",
                type(exc).__name__,
                exc,
            )
            return _call_chained_attention(
                previous,
                function,
                q,
                k,
                v,
                heads,
                mask=mask,
                attn_precision=attn_precision,
                skip_reshape=skip_reshape,
                skip_output_reshape=skip_output_reshape,
                kwargs=fallback_kwargs,
            )

    return override


__all__ = [
    "SOL_ROUTING_MODES",
    "clone_model_with_attention_override",
    "make_sage_attention_override",
    "make_sol_attention_override",
    "sol_attention_flex",
]
