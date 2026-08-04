# SPDX-License-Identifier: Apache-2.0
"""Pure PyTorch MiniMax-H3 packed-token DiT.

This module is a direct, single-device port of the H3 DiT found in the
MiniMax SGLang reference implementation.  It intentionally has no dependency
on SGLang, vLLM, Triton, FlashAttention, or ``torch.distributed``.  PyTorch
scaled-dot-product attention supplies the CUDA flash/memory-efficient kernels
when the installed PyTorch/GPU combination supports them and remains a
correctness fallback on CPU.

The model consumes the *packed* tensors produced by the H3 packing code.  It
does not turn ordinary BCHW/BCTHW latents into that representation itself.
Keeping this boundary explicit is important: H3 combines video, text and audio
tokens in one self-attention sequence and assigns per-token timesteps.

Checkpoint contract
-------------------

After stripping an optional outer prefix such as ``transformer.``, checkpoint
keys use the same names as :meth:`torch.nn.Module.state_dict`, for example::

    video_patch_proj.weight
    token_refiner.blocks.0.attn.qkv_proj.weight
    blocks.0.attn.qkv_proj.weight
    blocks.49.adaln_proj.linear.bias
    final_layer.audio_out.weight

The checkpoint QKV matrices are grouped per attention head as
``[head0_q, head0_k, head0_v, head1_q, ...]``.  ``nn.Linear`` expects
``[all_q, all_k, all_v]``.  A streaming loader must call
:func:`prepare_checkpoint_tensor` for every tensor before assigning it.

Optional ``operations`` (Comfy ``mixed_precision_ops`` / ``manual_cast``) injects
quantizable Linear layers; FP32 layers (patch/time/final) always use native ``nn.Linear``.
"""

from __future__ import annotations
from ..attention import (  # noqa: F401
    normalize_cu_seqlens_bounds, sdpa_varlen_attention,
    _rms_norm, _rotate_half, _modulate_scale_shift, _modulate_gate, _silu_mul,
    _index_runs, _rope_cos_sin, _rope_cos_sin_cache, _rope_rotation_table,
    _apply_rope_cos_sin, _apply_rope,
    _apply_qk_norm, _apply_rope_qk, reorder_grouped_qkv_to_qkv,
    is_qkv_weight_key, is_qkv_scale_key, prepare_checkpoint_tensor, prepare_state_dict_qkv_,
)

import logging
import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..h3_settings import (
    DIT_DEBUG_STRUCTURE_CHECKS,
    OPT_ADALN_SEGMENT_BROADCAST,
    OPT_FUSED_QK_ROPE,
    OPT_FUSED_QK_ROPE_CUDA_ONLY,
    OPT_INT8_FUSED_SWIGLU,
    OPT_PREPARED_STRUCTURE,
    OPT_SDPA_PRECOMPUTED_BOUNDS,
    QKV_SCALE_SUFFIX,
    QKV_WEIGHT_SUFFIX,
)
from ..adaln_curve import (
    ADALN_CURVE_DTYPE,
    ADALN_CURVE_TABLE_KEY,
    interpolate_curve_table,
)
from ..modulation_cache import (
    H3ModulationCache,
    H3PrecomputeUnsupported,
    precompute_dit_modulation,
)
from ..frame_rate import (
    adaln_frame_rate,
    apply_temporal_freq_scale,
    rope_temporal_scale,
)


LOGGER = logging.getLogger(__name__)


def _linear_cls(operations: Any | None):
    return operations.Linear if operations is not None else nn.Linear  # Comfy ops or native.


_FUSED_SWIGLU_PROBE: list[Any] = []  # Empty=not probed; [None]=unavailable; [fn]=available.


def _reset_fused_swiglu_probe() -> None:
    """Clear the probe cache after changing a flag or injecting a test implementation."""
    _FUSED_SWIGLU_PROBE.clear()


def _fused_swiglu_fn() -> Any | None:
    """Probe ``comfy.ops.linear_input_act``.

    INT8 Linear already quantizes its input. Folding swiglu into that kernel avoids
    one write and read of the ``[seq, ffn]`` intermediate (about 1GB/layer/step at
    37k rows). Older Comfy versions lack this function; a failed probe falls back to
    the existing eager path.
    """

    if not _FUSED_SWIGLU_PROBE:
        resolved = None
        if OPT_INT8_FUSED_SWIGLU:
            try:
                import comfy.ops as comfy_ops

                candidate = getattr(comfy_ops, "linear_input_act", None)
                resolved = candidate if callable(candidate) else None
            except Exception:  # Treat missing Comfy or import-time failures as unavailable.
                resolved = None
            LOGGER.info(
                "INT8 fused swiglu (comfy.ops.linear_input_act): %s",
                "enabled" if resolved is not None else "unavailable; falling back to eager",
            )
        _FUSED_SWIGLU_PROBE.append(resolved)
    return _FUSED_SWIGLU_PROBE[0]


_FUSED_QK_ROPE_PROBE: list[Any] = []  # Empty=not probed; [None]=unavailable; [(fn, in_place)]=available.


def _reset_fused_qk_rope_probe() -> None:
    """Clear the probe cache after changing a flag or injecting a test implementation."""
    _FUSED_QK_ROPE_PROBE.clear()


def _disable_fused_qk_rope(reason: BaseException) -> None:
    """Permanently disable an incompatible fused kernel discovered at call time and fall back to torch."""

    if _FUSED_QK_ROPE_PROBE:
        _FUSED_QK_ROPE_PROBE[0] = None
    else:
        _FUSED_QK_ROPE_PROBE.append(None)
    LOGGER.warning(
        "Fused RMSNorm+RoPE kernel call failed (%s); this process will use the torch implementation from now on",
        reason,
    )


def _fused_kernel_accepts_rot_dim(candidate: Any, name: str) -> bool:
    """The fused kernel must support ``rot_dim`` or it will rotate the entire head.

    H3 uses partial rotation (``rotary_dim = 6 * rope_inv_freq_len`` = 96,
    head_dim = 128, leaving 32 dimensions untouched). Early comfy-kitchen
    ``rms_rope_split_half`` implementations lacked ``rot_dim`` and split the entire
    head_dim into pairs for the rotation table. They raise on a shape mismatch and
    produce incorrect output when shapes happen to match. Checking only whether the
    function exists would accept those versions, so the signature must be validated.

    Treat implementations that cannot be introspected (such as C extensions) as
    available; the call site still has a TypeError fallback.
    """

    import inspect

    try:
        signature = inspect.signature(candidate)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    if "rot_dim" in parameters:
        return True
    LOGGER.info(
        "comfy-kitchen %s has no rot_dim parameter (signature: %s), so it cannot express H3 partial rotation; "
        "using the torch implementation",
        name,
        signature,
    )
    return False


def _fused_qk_rope_fn() -> tuple[Any, bool] | None:
    """Probe comfy-kitchen's fused per-head RMSNorm + split-half RoPE kernel.

    Return ``(fn, in_place)``. The in-place version rewrites q/k directly in the qkv
    buffer (saving one ``[S, inner]`` allocation and two memory passes); the
    out-of-place version returns new q/k tensors. If neither is available, fall back
    to the existing torch implementation.
    """

    if not _FUSED_QK_ROPE_PROBE:
        resolved = None
        if OPT_FUSED_QK_ROPE:
            try:
                import comfy.quant_ops as quant_ops

                kitchen = getattr(quant_ops, "ck", None)
                for name, in_place in (
                    ("rms_rope_split_half_", True),
                    ("rms_rope_split_half", False),
                ):
                    candidate = getattr(kitchen, name, None)
                    if callable(candidate) and _fused_kernel_accepts_rot_dim(
                        candidate, name
                    ):
                        resolved = (candidate, in_place)
                        break
            except Exception:  # Missing Comfy/comfy-kitchen or import-time failure.
                resolved = None
            LOGGER.info(
                "Fused RMSNorm+RoPE kernel (comfy.quant_ops.ck): %s",
                f"enabled {'in-place' if resolved[1] else 'out-of-place'}"
                if resolved is not None
                else "unavailable; falling back to torch",
            )
        _FUSED_QK_ROPE_PROBE.append(resolved)
    return _FUSED_QK_ROPE_PROBE[0]


def fused_qk_rope_available(reference: torch.Tensor) -> bool:
    """Return whether this forward pass should use the fused kernel (comfy-kitchen provides CUDA only)."""
    if OPT_FUSED_QK_ROPE_CUDA_ONLY and reference.device.type != "cuda":
        return False
    return _fused_qk_rope_fn() is not None


def _activation_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Linear input dtype: use compute/factory dtype for QuantizedTensor; never read int8 weight.dtype."""
    fk = getattr(module, "factory_kwargs", None) or {}
    if fk.get("dtype") is not None:
        return fk["dtype"]
    w = getattr(module, "weight", None)
    if w is not None and getattr(w, "is_floating_point", lambda: False)():
        return w.dtype
    return fallback


def _adaln_dtype(config: "MiniMaxH3DiTConfig", fallback: torch.dtype) -> torch.dtype:
    """Store adaLN weights in fp32 for curve-table mode (width is only k, so cost is negligible and matches table precision)."""
    return ADALN_CURVE_DTYPE if config.use_adaln_curves else fallback


FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})
ADALN_MODALITY_COUNT = 3
PACKED_SEQUENCE_ALIGNMENT = 64

# Source-compatible aliases are useful when reusing the reference contract
# tests against this standalone module.
MINIMAX_H3_FP32_PARAM_NAMES = FP32_PARAM_NAMES
MINIMAX_H3_FP32_BUFFER_NAMES = FP32_BUFFER_NAMES
MINIMAX_H3_ADALN_MODALITY_NUM = ADALN_MODALITY_COUNT
MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = PACKED_SEQUENCE_ALIGNMENT


@dataclass
class MiniMaxH3DiTConfig:
    """Architecture values published by the H3 reference implementation.

    Smaller values can be supplied for unit tests.  ``hidden_size`` does not
    have to equal ``num_attention_heads * attention_head_dim``: H3 projects
    5376-wide residuals to a 7168-wide attention inner dimension.
    """

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    # Curve-table checkpoint: when non-None, a [grid, k] sample table replaces the
    # time embedder and ``time_embed_dim`` becomes curve-basis rank k. See
    # runtime/adaln_curve.py.
    adaln_curve_grid: int | None = None

    @property
    def use_adaln_curves(self) -> bool:
        return self.adaln_curve_grid is not None

    def __post_init__(self) -> None:
        self.patch_size = tuple(int(value) for value in self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(
                f"patch_size must contain (t, h, w), got {self.patch_size!r}"
            )
        positive = (
            "num_layers",
            "token_refiner_num_layers",
            "hidden_size",
            "num_attention_heads",
            "attention_head_dim",
            "ffn_hidden_size",
            "latents_dim",
            "audio_latents_dim",
            "text_dim",
            "timestep_input_dim",
            "time_embed_hidden_size",
            "time_embed_dim",
            "rope_inv_freq_len",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        expected_adaln = 6 * ADALN_MODALITY_COUNT * self.hidden_size
        if self.adaln_out_features != expected_adaln:
            raise ValueError(
                "adaln_out_features must equal 6 * 3 * hidden_size: "
                f"{self.adaln_out_features} != {expected_adaln}"
            )
        expected_final = 2 * self.hidden_size
        if self.final_adaln_out_features != expected_final:
            raise ValueError(
                "final_adaln_out_features must equal 2 * hidden_size: "
                f"{self.final_adaln_out_features} != {expected_final}"
            )
        if self.timestep_input_dim % 2:
            raise ValueError("timestep_input_dim must be even")
        if self.adaln_curve_grid is not None:
            self.adaln_curve_grid = int(self.adaln_curve_grid)
            if self.adaln_curve_grid < 2:
                raise ValueError(
                    "adaln_curve_grid must be at least 2 rows, got "
                    f"{self.adaln_curve_grid}"
                )
            if self.time_embed_dim > self.adaln_curve_grid:
                # Rank cannot exceed the sample count; otherwise config and checkpoint do not match.
                raise ValueError(
                    "adaln curve rank (time_embed_dim) must not exceed "
                    f"adaln_curve_grid: {self.time_embed_dim} > "
                    f"{self.adaln_curve_grid}"
                )
        rotary_dim = 6 * self.rope_inv_freq_len
        if rotary_dim > self.attention_head_dim:
            raise ValueError(
                "3D RoPE rotates 6 * rope_inv_freq_len dimensions, which must "
                f"fit in attention_head_dim: {rotary_dim} > "
                f"{self.attention_head_dim}"
            )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MiniMaxH3DiTConfig":
        """Create a config from either an arch dict or an outer model config."""

        data: Mapping[str, Any] = raw
        for wrapper_key in ("arch_config", "transformer_config", "dit_config"):
            nested = data.get(wrapper_key)
            if isinstance(nested, Mapping):
                data = nested
                break

        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)

    @property
    def video_patch_dim(self) -> int:
        return self.latents_dim * math.prod(self.patch_size)

    @property
    def attention_inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


# Compatibility with the SGLang naming used by the source repository.
MiniMaxH3DiTArchConfig = MiniMaxH3DiTConfig


def _coerce_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = str(dtype).lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported model dtype: {dtype!r}")
    return aliases[normalized]


def _required_kwarg(kwargs: Mapping[str, Any], key: str) -> Any:
    if key not in kwargs or kwargs[key] is None:
        raise ValueError(f"MiniMaxH3DiTModel.forward requires kwarg {key!r}")
    return kwargs[key]


FORWARD_SUPPORTED_KWARGS = frozenset(
    {
        "x",
        "audio_x",
        "img_position_ids",
        "unique_timesteps",
        "inverse_indices",
        "update_mask",
        "update_audio_mask",
        "token_tags",
        "skip_mask_out_condition",
        "prompt_embeds",
        "refined_prompt_embeds_length",
        "img_pos_info",
        "audio_pos_info",
        "text_pos_info",
        "img_pos_for_infer_output_info",
        "packed_seq_params",
        "refiner_packed_seq_params",
        "cu_seqlens_bounds",  # P0-1 precomputed bounds.
        "prepared_rope_cache",  # P0-2 session RoPE
        "structure_validated",  # P0-3 one-time validation marker.
        "frame_rate_options",  # Experimental frame-rate conditioning.
        "video_timestep",  # Used by temporal_rope.
        "video_sigma",
    }
)
_FORWARD_SUPPORTED_KWARGS = FORWARD_SUPPORTED_KWARGS




# Keep the private source name importable for parity tests.
_reorder_grouped_qkv_to_qkv = reorder_grouped_qkv_to_qkv




















class MiniMaxH3Rope(nn.Module):
    """Three-dimensional RoPE over ``(time, height, width)`` positions."""

    def __init__(
        self,
        inv_freq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        axis_dim = inv_freq_len * 2
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)
                / axis_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(
        self,
        img_position_ids: torch.Tensor,
        *,
        video_mask: torch.Tensor | None = None,
        temporal_scale: float = 1.0,
        low_frequency_count: int = 16,
        frequency_profile: str = "hard",
    ) -> torch.Tensor:
        if img_position_ids.ndim != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                "img_position_ids must be [1, sequence, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        if img_position_ids.shape[-1] != 3:
            raise ValueError(
                f"img_position_ids last dimension must be 3, got "
                f"{img_position_ids.shape[-1]}"
            )
        # INT8 partial offload can keep this tiny FP32 buffer on CPU while
        # attention activations run on CUDA.  Position IDs arrive on the
        # active compute device, so move the buffer to them instead of moving
        # the positions back to the buffer's offload device.
        positions = img_position_ids[0].to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(
            device=positions.device, dtype=torch.float32
        )
        per_axis = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        time_freq, height_freq, width_freq = per_axis.unbind(dim=1)
        if video_mask is not None and abs(float(temporal_scale) - 1.0) >= 1e-12:
            time_freq = apply_temporal_freq_scale(
                time_freq, video_mask=video_mask, temporal_scale=float(temporal_scale),
                low_frequency_count=int(low_frequency_count),
                frequency_profile=str(frequency_profile),
            )
        half = torch.cat((time_freq, height_freq, width_freq), dim=-1)
        return torch.cat((half, half), dim=-1)


















_sdpa_varlen_attention = sdpa_varlen_attention


AttentionFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor
]


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = config.timestep_input_dim
        self.proj_in = nn.Linear(
            config.timestep_input_dim,
            config.time_embed_hidden_size,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            config.time_embed_hidden_size,
            config.time_embed_dim,
            bias=True,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self, timestep: torch.Tensor, frame_rate: float | torch.Tensor | None = None
    ) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half, dtype=torch.float32, device=timestep.device
            )
            / half
        )
        t = timestep.to(torch.float32).view(-1)
        arguments = t[:, None] * frequencies[None]
        embedding = torch.cat(
            (torch.cos(arguments), torch.sin(arguments)), dim=-1
        )
        if frame_rate is not None:  # Experimental: add the fps sinusoid to the t embedding.
            fr = torch.as_tensor(frame_rate, dtype=torch.float32, device=t.device).flatten()
            if int(fr.numel()) == 1:
                fr = fr.expand(t.shape[0])
            elif int(fr.numel()) != int(t.numel()):
                raise ValueError(
                    f"frame_rate length {int(fr.numel())} does not match timestep length {int(t.numel())}"
                )
            fps_args = fr[:, None] * frequencies[None]
            embedding = embedding + torch.cat(
                (torch.cos(fps_args), torch.sin(fps_args)), dim=-1
            )
        return self.proj_out(F.silu(self.proj_in(embedding)))


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        attention_backend: str,
        attention_function: AttentionFunction | None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        if attention_backend not in {"auto", "sdpa"}:
            raise ValueError(
                "The direct H3 port currently supports attention_backend "
                "'auto' or 'sdpa'"
            )
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.inner_dim = config.attention_inner_dim
        self.softmax_scale = self.head_dim**-0.5
        self.attention_backend = attention_backend
        self.attention_function = attention_function
        Linear = _linear_cls(operations)

        self.qkv_proj = Linear(
            config.hidden_size,
            self.inner_dim * 3,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_norm = _rms_norm(
            self.head_dim, eps=config.qk_norm_eps, dtype=dtype, device=device
        )
        self.k_norm = _rms_norm(
            self.head_dim, eps=config.qk_norm_eps, dtype=dtype, device=device
        )
        self.out_proj = Linear(
            self.inner_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_bounds: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        del max_seqlen  # PyTorch SDPA derives lengths from tensor slices.
        rows = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        value = value.view(rows, self.num_heads, self.head_dim)
        # rope_cache = (cos|sin cache, rotation-matrix table for fused kernel or None).
        rotation_table = (
            rope_cache[1] if rope_cache is not None and len(rope_cache) > 1 else None
        )
        fused = _fused_qk_rope_fn() if rotation_table is not None else None
        if fused is not None and fused[1] and query.requires_grad:
            # The in-place version rewrites an autograd view produced by chunk, which
            # PyTorch rejects when gradients are enabled. Inference paths
            # (requires_grad_(False) / no_grad) are unaffected.
            fused = None
        if fused is not None:
            # The fused kernel performs per-head RMSNorm and split-half RoPE together,
            # directly in the q/k views of the qkv buffer (the v segment is untouched).
            fused_fn, in_place = fused
            query = query.view(1, rows, self.num_heads, self.head_dim)
            key = key.view(1, rows, self.num_heads, self.head_dim)
            rotary_dim = int(rotation_table.shape[-3]) * 2
            args = (query, key, rotation_table, self.q_norm.weight, self.k_norm.weight)
            kwargs = {"epsilon": self.q_norm.eps, "rot_dim": rotary_dim}
            try:
                if in_place:
                    fused_fn(*args, **kwargs)
                else:
                    query, key = fused_fn(*args, **kwargs)
            except TypeError as exc:
                # Non-introspectable implementations (for example C extensions) reveal
                # incompatibility only here. Permanently disable the fused path and fall
                # back instead of failing the entire sampling run.
                _disable_fused_qk_rope(exc)
                fused = None
            else:
                query, key = query[0], key[0]
        if fused is None:
            query = query.view(rows, self.num_heads, self.head_dim)
            key = key.view(rows, self.num_heads, self.head_dim)
            query, key = _apply_qk_norm(
                query, key, self.q_norm, self.k_norm, self.head_dim
            )
            if rope_cache is not None:
                query, key = _apply_rope_qk(query, key, rope_cache[0])

        if self.attention_function is None:
            output = sdpa_varlen_attention(
                query, key, value,
                cu_seqlens=cu_seqlens, softmax_scale=self.softmax_scale,
                bounds=cu_seqlens_bounds,
            )
        else:
            output = self.attention_function(
                query, key, value, cu_seqlens, self.softmax_scale
            )
        return self.out_proj(output.reshape(rows, self.inner_dim))


class MiniMaxH3MLP(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        Linear = _linear_cls(operations)
        self.fc1 = Linear(
            config.hidden_size,
            config.ffn_hidden_size * 2,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.fc2 = Linear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        # Only Comfy ops (quantized/lowvram) provides an activation-quantization kernel
        # that can absorb this operation. Native nn.Linear would route through the
        # fused entry point only to return to the same eager implementation.
        self.fused_swiglu = _fused_swiglu_fn() if operations is not None else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.fc1(hidden_states)
        if self.fused_swiglu is not None:
            return self.fused_swiglu(self.fc2, gate_up, "swiglu")
        return self.fc2(_silu_mul(gate_up))


class MiniMaxH3AdalnProj(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        out_features: int,
        *,
        expand_ratio: int,
        modality_count: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
        operations: Any | None = None,
        apply_silu: bool = True,
    ) -> None:
        super().__init__()
        expected = expand_ratio * config.hidden_size * modality_count
        if out_features != expected:
            raise ValueError(f"AdaLN output width {out_features} != {expected}")
        self.expand_ratio = expand_ratio
        self.modality_count = modality_count
        self.hidden_size = config.hidden_size
        self.compute_dtype = dtype
        # Curve-table checkpoints bake silu into the sample table; do not apply it again here.
        self.apply_silu = bool(apply_silu)
        # Comfy's mixed_precision_ops binds every Linear to the model compute
        # dtype and ignores this constructor's explicit dtype. Curve-table H3
        # checkpoints require their small low-rank adaLN projections to remain
        # FP32, so keep those projections as native Linears.
        Linear = nn.Linear if dtype == torch.float32 else _linear_cls(operations)
        self.linear = Linear(
            config.time_embed_dim,
            out_features,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def project(self, timestep_embedding: torch.Tensor) -> torch.Tensor:
        """[M,t_dim] -> [M, modalities, expand, hidden] for the precomputation cache."""
        act_dtype = _activation_dtype(self.linear, self.compute_dtype)
        activated = (
            F.silu(timestep_embedding) if self.apply_silu else timestep_embedding
        )
        projected = self.linear(activated.to(dtype=act_dtype))
        return projected.view(
            projected.shape[0], self.modality_count, self.expand_ratio, self.hidden_size
        )

    def forward(self, timestep_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        projected = self.project(timestep_embedding)
        flat = projected.reshape(
            projected.shape[0] * self.modality_count,
            self.expand_ratio * self.hidden_size,
        )
        return tuple(flat.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        attention_backend: str,
        attention_function: AttentionFunction | None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = _rms_norm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device
        )
        self.norm2 = _rms_norm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device
        )
        self.attn = MiniMaxH3Attention(
            config,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            attention_function=attention_function,
            operations=operations,
        )
        self.mlp = MiniMaxH3MLP(
            config, dtype=dtype, device=device, operations=operations
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_bounds: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cu_seqlens_bounds=cu_seqlens_bounds,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        attention_backend: str,
        attention_function: AttentionFunction | None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    config,
                    dtype=dtype,
                    device=device,
                    attention_backend=attention_backend,
                    attention_function=attention_function,
                    operations=operations,
                )
                for _ in range(config.token_refiner_num_layers)
            ]
        )
        self.final_norm = _rms_norm(
            config.hidden_size,
            eps=config.final_norm_eps,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_bounds: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_bounds=cu_seqlens_bounds,
            )
        return self.final_norm(hidden_states)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        attention_backend: str,
        attention_function: AttentionFunction | None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = _rms_norm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device
        )
        self.norm2 = _rms_norm(
            config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device
        )
        self.attn = MiniMaxH3Attention(
            config,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            attention_function=attention_function,
            operations=operations,
        )
        self.mlp = MiniMaxH3MLP(
            config, dtype=dtype, device=device, operations=operations
        )
        self.adaln_proj = MiniMaxH3AdalnProj(
            config,
            config.adaln_out_features,
            expand_ratio=6,
            modality_count=ADALN_MODALITY_COUNT,
            dtype=_adaln_dtype(config, dtype),
            device=device,
            operations=operations,
            apply_silu=not config.use_adaln_curves,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor | None,
        combined_indices: Any,  # Tensor or _index_runs table.
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_bounds: tuple[int, ...] | None = None,
        modulation: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            if timestep_embedding is None:
                raise RuntimeError("AdaLN requires timestep_embedding or precomputed modulation")
            mod_values = self.adaln_proj(timestep_embedding)
        else:
            mod_values = modulation
        (
            shift_attention, scale_attention, gate_attention,
            shift_mlp, scale_mlp, gate_mlp,
        ) = mod_values

        residual = hidden_states
        normed = _modulate_scale_shift(
            self.norm1(hidden_states), shift_attention, scale_attention, combined_indices,
        )
        attended = self.attn(
            normed, rope_cache=rope_cache, cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen, cu_seqlens_bounds=cu_seqlens_bounds,
        )
        hidden_states = _modulate_gate(
            residual, gate_attention, attended, combined_indices
        )
        residual = hidden_states
        normed = _modulate_scale_shift(
            self.norm2(hidden_states), shift_mlp, scale_mlp, combined_indices,
        )
        return _modulate_gate(residual, gate_mlp, self.mlp(normed), combined_indices)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3DiTConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        self.norm = _rms_norm(
            config.hidden_size,
            eps=config.final_norm_eps,
            dtype=dtype,
            device=device,
        )
        self.adaln_proj = MiniMaxH3AdalnProj(
            config,
            config.final_adaln_out_features,
            expand_ratio=2,
            modality_count=1,
            dtype=_adaln_dtype(config, dtype),
            device=device,
            operations=operations,
            apply_silu=not config.use_adaln_curves,
        )
        self.video_out = nn.Linear(  # FP32 always uses native Linear.
            config.hidden_size,
            config.video_patch_dim,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.audio_out = nn.Linear(
            config.hidden_size,
            config.audio_latents_dim,
            bias=True,
            device=device,
            dtype=torch.float32,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor | None,
        inverse_indices: Any,  # Tensor or _index_runs table.
        modulation: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if modulation is None:
            if timestep_embedding is None:
                raise RuntimeError("FinalLayer AdaLN requires timestep_embedding or precomputed modulation")
            shift, scale = self.adaln_proj(timestep_embedding)
        else:
            shift, scale = modulation
        hidden_states = _modulate_scale_shift(
            self.norm(hidden_states), shift, scale, inverse_indices
        ).to(torch.float32)
        return self.video_out(hidden_states), self.audio_out(hidden_states)


class MiniMaxH3DiTModel(nn.Module):
    """Single-device H3 transformer with the reference packed forward API."""

    def __init__(
        self,
        config: MiniMaxH3DiTConfig | Mapping[str, Any] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str = torch.bfloat16,
        attention_backend: str = "sdpa",
        attention_function: AttentionFunction | None = None,
        operations: Any | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = MiniMaxH3DiTConfig()
        elif not isinstance(config, MiniMaxH3DiTConfig):
            config = MiniMaxH3DiTConfig.from_dict(config)
        dtype = _coerce_dtype(dtype)

        self.config = config
        self.arch = config
        self.model_dtype = dtype
        self.operations = operations
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.latents_dim
        self.attention_backend = attention_backend
        Linear = _linear_cls(operations)

        self.video_patch_proj = nn.Linear(  # FP32 always uses native Linear.
            config.video_patch_dim,
            config.hidden_size,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.audio_patch_proj = nn.Linear(
            config.audio_latents_dim,
            config.hidden_size,
            bias=True,
            device=device,
            dtype=torch.float32,
        )
        self.condition_proj = Linear(
            config.text_dim,
            config.hidden_size,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.use_adaln_curves = config.use_adaln_curves
        if self.use_adaln_curves:
            # Curve-table checkpoint: the time embedder is baked into the [grid, k] sample table.
            self.register_buffer(
                ADALN_CURVE_TABLE_KEY,
                torch.empty(
                    int(config.adaln_curve_grid),
                    config.time_embed_dim,
                    dtype=ADALN_CURVE_DTYPE,
                    device=device,
                ),
                persistent=True,
            )
        else:
            self.time_embedder = MiniMaxH3TimeEmbedder(config, device=device)
        self.rope = MiniMaxH3Rope(config.rope_inv_freq_len, device=device)
        self.token_refiner = MiniMaxH3TokenRefiner(
            config,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            attention_function=attention_function,
            operations=operations,
        )
        self.blocks = nn.ModuleList(
            [
                MiniMaxH3DiTBlock(
                    config,
                    dtype=dtype,
                    device=device,
                    attention_backend=attention_backend,
                    attention_function=attention_function,
                    operations=operations,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.layer_names = ["blocks"]
        self.final_layer = MiniMaxH3FinalLayer(
            config, dtype=dtype, device=device, operations=operations
        )
        self._modulation_cache: H3ModulationCache | None = None
        self._adaln_weights_released = False

    @classmethod
    def from_config(
        cls,
        config: MiniMaxH3DiTConfig | Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str = torch.bfloat16,
        attention_backend: str = "sdpa",
        attention_function: AttentionFunction | None = None,
        operations: Any | None = None,
    ) -> "MiniMaxH3DiTModel":
        return cls(
            config,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
            attention_function=attention_function,
            operations=operations,
        )

    @property
    def device(self) -> torch.device:
        # The parameter is authoritative even if an older memory manager left
        # its compatibility marker stale.
        return self.video_patch_proj.weight.device

    @device.setter
    def device(self, value: torch.device | str) -> None:
        # ComfyUI's ModelPatcher writes ``model.device`` after both load and
        # unload.  Accept that write without moving tensors behind the
        # patcher's back; ``nn.Module.to``/ModelPatcher already performed the
        # actual move before setting this marker.
        object.__setattr__(self, "_comfy_device_marker", torch.device(value))

    @property
    def dtype(self) -> torch.dtype:
        return self.model_dtype

    @staticmethod
    def _position_ids(position_info: Any, name: str) -> torch.Tensor:
        if torch.is_tensor(position_info):
            ids = position_info
        elif isinstance(position_info, Mapping):
            ids = position_info.get("position_ids")
        else:
            ids = getattr(position_info, "position_ids", None)
        if ids is None:
            raise ValueError(f"{name}.position_ids is required")
        return ids.view(-1).to(torch.long)

    @staticmethod
    def _packed_field(packed: Any, name: str, field: str) -> Any:
        if isinstance(packed, Mapping):
            value = packed.get(field)
        else:
            value = getattr(packed, field, None)
        if value is None:
            raise ValueError(f"{name}.{field} is required")
        return value

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        text_length: int | None = None,
    ) -> torch.Tensor:
        """Project and run the two request-static token-refiner blocks."""

        if device is None:
            device = self.condition_proj.weight.device
        if text_length is None:
            text_length = int(normalize_cu_seqlens_bounds(refiner_cu_seqlens)[1])
        text_length = int(text_length)
        if text_length <= 0 or text_length > int(prompt_embeds.shape[0]):
            raise ValueError(
                "live text length must be in "
                f"[1, {int(prompt_embeds.shape[0])}], got {text_length}"
            )
        text_rows = prompt_embeds[:text_length].to(
            device=device, dtype=self.model_dtype
        )
        # Drop the bucket padding segment.  The refiner operates only on live
        # text, preserving bitwise-equivalent GEMM dimensions.
        refiner_bounds = (0, text_length, text_length)
        true_cu_seqlens = torch.tensor(
            refiner_bounds, device=device, dtype=torch.int32,
        )
        projected = self.condition_proj(text_rows)
        return self.token_refiner(
            projected,
            cu_seqlens=true_cu_seqlens,
            max_seqlen=text_length,
            cu_seqlens_bounds=refiner_bounds,
        )

    def _rope_cache(
        self, frequencies: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(cos|sin cache, fused-kernel rotation-matrix table or None)``.

        Build the rotation-matrix table only when the fused kernel can actually use
        it. The table is twice as large as the cos/sin cache, and fallback paths never
        read it. Both derive from the same angles, so there are not two numerical sources.
        """
        cos_sin = _rope_cos_sin_cache(frequencies, dtype=self.model_dtype)
        table = (
            _rope_rotation_table(cos_sin, dtype=self.model_dtype)
            if fused_qk_rope_available(cos_sin)
            else None
        )
        return (cos_sin, table)

    def prepare_structure(
        self,
        *,
        img_position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        token_tags: torch.Tensor,
        seq_len: int,
    ) -> dict[str, Any]:
        """Build RoPE/bounds once before the sampling loop for H3DenoiseBranch to inject into static_kwargs."""
        bounds = normalize_cu_seqlens_bounds(cu_seqlens, rows=int(seq_len))
        tags = token_tags.view(-1).to(torch.long)
        if int(tags.numel()) != int(seq_len):
            raise ValueError(f"token_tags length={int(tags.numel())}; expected {seq_len}")
        tmin = int(tags.min().item()); tmax = int(tags.max().item())
        if tmin < -1 or tmax > 2:
            raise ValueError("token_tags values must be -1, 0, 1, or 2")
        rope_cache = None
        if OPT_PREPARED_STRUCTURE:
            freqs = self.rope(img_position_ids)
            rope_cache = self._rope_cache(freqs)
        out: dict[str, Any] = {
            "cu_seqlens_bounds": bounds if OPT_SDPA_PRECOMPUTED_BOUNDS else None,
            "structure_validated": True,
        }
        if rope_cache is not None:
            out["prepared_rope_cache"] = rope_cache
        return out

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        prompt_embeds: torch.Tensor,
        unique_timesteps: torch.Tensor,
        image_positions: torch.Tensor,
        audio_positions: torch.Tensor,
        text_positions: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        refined_prompt_embeds_length: int | None,
        frame_rate: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_length = (
            int(refined_prompt_embeds_length)
            if refined_prompt_embeds_length is not None
            else int(normalize_cu_seqlens_bounds(refiner_cu_seqlens)[1])
        )
        if text_length <= 0 or text_length > int(prompt_embeds.shape[0]):
            raise ValueError(
                f"invalid live text length {text_length} for "
                f"{int(prompt_embeds.shape[0])} prompt rows"
            )
        text_positions = text_positions[:text_length]
        if refined_prompt_embeds_length is None:
            text_embeddings = self.refine_prompt_embeds(
                prompt_embeds, refiner_cu_seqlens, device=x.device,
                text_length=text_length,
            )
        else:
            text_embeddings = prompt_embeds[:text_length].to(
                device=x.device, dtype=self.model_dtype
            )
            if text_embeddings.shape[-1] != self.hidden_size:
                raise ValueError(
                    "refined prompt embedding width must be "
                    f"{self.hidden_size}, got {text_embeddings.shape[-1]}"
                )

        sequence_length = x.shape[1]
        embeddings = torch.zeros(
            (sequence_length, self.hidden_size),
            device=x.device,
            dtype=self.model_dtype,
        )
        embeddings.index_add_(
            0,
            text_positions,
            text_embeddings.to(dtype=self.model_dtype),
        )

        flat_video = x.reshape(-1, x.shape[-1])
        video_rows = flat_video.index_select(0, image_positions).to(torch.float32)
        video_embeddings = self.video_patch_proj(video_rows)
        embeddings.index_add_(
            0, image_positions, video_embeddings.to(dtype=self.model_dtype)
        )

        flat_audio = audio_x.reshape(-1, audio_x.shape[-1])
        audio_rows = flat_audio.index_select(0, audio_positions).to(torch.float32)
        audio_embeddings = self.audio_patch_proj(audio_rows)
        embeddings.index_add_(
            0, audio_positions, audio_embeddings.to(dtype=self.model_dtype)
        )

        cache = self._modulation_cache
        if cache is not None:
            return embeddings, None  # type: ignore[return-value]
        if self._adaln_weights_released:
            raise RuntimeError("AdaLN weights were released and no modulation cache is available")
        return embeddings, self.timestep_embedding(
            unique_timesteps, frame_rate=frame_rate
        )

    def timestep_embedding(
        self,
        timesteps: torch.Tensor,
        *,
        frame_rate: float | None = None,
    ) -> torch.Tensor:
        """Map ``[M]`` timestep to adaLN input ``[M, time_embed_dim]``.

        Curve-table checkpoints interpolate the sample table (with silu baked in);
        original checkpoints use the time embedder.
        """
        if not self.use_adaln_curves:
            return self.time_embedder(timesteps, frame_rate=frame_rate)
        if frame_rate is not None:
            raise RuntimeError(
                "Curve-table checkpoints have no time embedder, so experimental frame-rate conditioning "
                "cannot be applied. Disable the Frame Rate node's adaln option (temporal_rope is unaffected), "
                "or use the original DiT weights"
            )
        table = getattr(self, ADALN_CURVE_TABLE_KEY)
        if table.device != timesteps.device:
            # Under INT8/layerwise, this small table may still be on the offload device.
            table = table.to(timesteps.device)
        return interpolate_curve_table(table, timesteps)

    def precompute_modulation(
        self,
        timesteps: Any,
        *,
        compute_device: torch.device | str | None = None,
        cache_device: torch.device | str | None = None,
        release_weights: bool = True,
        frame_rate: float | None = None,
    ) -> H3ModulationCache:
        """Precompute adaLN for every timestep before sampling; see runtime/modulation_cache.py."""
        if self.use_adaln_curves:
            # The curve table already compresses adaLN weights to [out, k], so
            # precomputation offers no benefit and there is no time embedder to run.
            # sampler_core catches this exception and samples normally.
            raise H3PrecomputeUnsupported(
                "Curve-table checkpoint adaLN weights are already low-rank; skipping AdaLN precomputation"
            )
        fr_key = None if frame_rate is None else round(float(frame_rate), 6)
        if self._adaln_weights_released and self._modulation_cache is not None:
            needed = {
                round(float(t), 6)
                for t in torch.as_tensor(timesteps, dtype=torch.float32).tolist()
            }
            cached_fr = getattr(self._modulation_cache, "frame_rate", None)
            if not needed.issubset(self._modulation_cache.timestep_rows) or cached_fr != fr_key:
                raise RuntimeError(
                    "AdaLN weights were released, so modulation cannot be recomputed for new timestep/frame_rate values; reload DiT"
                )
            return self._modulation_cache
        return precompute_dit_modulation(
            self, timesteps, compute_device=compute_device,
            cache_device=cache_device, release_weights=release_weights,
            frame_rate=frame_rate,
        )

    def clear_modulation_cache(self) -> None:
        self._modulation_cache = None

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the reference packed H3 forward contract.

        Returns video rows of width ``latents_dim * prod(patch_size)`` and
        audio rows of width ``audio_latents_dim``.
        """

        unexpected = sorted(set(kwargs) - FORWARD_SUPPORTED_KWARGS)
        if unexpected:
            raise TypeError(
                "MiniMaxH3DiTModel.forward received unexpected kwargs "
                f"{unexpected}; supported kwargs are "
                f"{sorted(FORWARD_SUPPORTED_KWARGS)}"
            )

        x = _required_kwarg(kwargs, "x")
        audio_x = _required_kwarg(kwargs, "audio_x")
        image_position_ids = _required_kwarg(kwargs, "img_position_ids")
        unique_timesteps = _required_kwarg(kwargs, "unique_timesteps")
        inverse_indices = (
            _required_kwarg(kwargs, "inverse_indices").view(-1).to(torch.long)
        )
        update_mask = _required_kwarg(kwargs, "update_mask")
        token_tags = (
            _required_kwarg(kwargs, "token_tags").view(-1).to(torch.long)
        )
        prompt_embeds = _required_kwarg(kwargs, "prompt_embeds")
        structure_validated = bool(kwargs.get("structure_validated"))
        prepared_rope_cache = kwargs.get("prepared_rope_cache")
        cu_seqlens_bounds = kwargs.get("cu_seqlens_bounds")

        image_positions = self._position_ids(
            _required_kwarg(kwargs, "img_pos_info"), "img_pos_info"
        )
        audio_positions = self._position_ids(
            _required_kwarg(kwargs, "audio_pos_info"), "audio_pos_info"
        )
        text_positions = self._position_ids(
            _required_kwarg(kwargs, "text_pos_info"), "text_pos_info"
        )
        output_image_positions = self._position_ids(
            _required_kwarg(kwargs, "img_pos_for_infer_output_info"),
            "img_pos_for_infer_output_info",
        )

        packed = _required_kwarg(kwargs, "packed_seq_params")
        cu_seqlens = self._packed_field(
            packed, "packed_seq_params", "cu_seqlens_q"
        )
        max_seqlen = int(
            self._packed_field(packed, "packed_seq_params", "max_seqlen_q")
        )
        refiner_packed = _required_kwarg(kwargs, "refiner_packed_seq_params")
        refiner_cu_seqlens = self._packed_field(
            refiner_packed, "refiner_packed_seq_params", "cu_seqlens_q"
        )
        # Required by the reference serving contract even though the direct
        # refiner trims its input to the live text length before attention.
        _refiner_max_seqlen = int(
            self._packed_field(
                refiner_packed,
                "refiner_packed_seq_params",
                "max_seqlen_q",
            )
        )

        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"x must be [1, sequence, channels], got {list(x.shape)}")
        if audio_x.ndim != 3 or audio_x.shape[0] != 1:
            raise ValueError(
                "audio_x must be [1, sequence, channels], got "
                f"{list(audio_x.shape)}"
            )
        if x.shape[-1] != self.config.video_patch_dim:
            raise ValueError(
                f"x last dimension must be {self.config.video_patch_dim}, "
                f"got {x.shape[-1]}"
            )
        if audio_x.shape[-1] != self.config.audio_latents_dim:
            raise ValueError(
                f"audio_x last dimension must be "
                f"{self.config.audio_latents_dim}, got {audio_x.shape[-1]}"
            )

        sequence_length = int(x.shape[1])
        if audio_x.shape[1] != sequence_length:
            raise ValueError("x and audio_x must use the same sparse sequence length")
        if token_tags.numel() != sequence_length:
            raise ValueError(
                f"token_tags must have {sequence_length} rows, got "
                f"{token_tags.numel()}"
            )
        if inverse_indices.numel() != sequence_length:
            raise ValueError(
                f"inverse_indices must have {sequence_length} rows, got "
                f"{inverse_indices.numel()}"
            )

        device = x.device
        # Prepared-structure path: the branch already moved structural tensors to the compute device.
        if image_positions.device != device:
            image_positions = image_positions.to(device)
        if audio_positions.device != device:
            audio_positions = audio_positions.to(device)
        if text_positions.device != device:
            text_positions = text_positions.to(device)
        if output_image_positions.device != device:
            output_image_positions = output_image_positions.to(device)
        if inverse_indices.device != device:
            inverse_indices = inverse_indices.to(device)
        if token_tags.device != device:
            token_tags = token_tags.to(device)
        if cu_seqlens.device != device or cu_seqlens.dtype != torch.int32:
            cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        if refiner_cu_seqlens.device != device or refiner_cu_seqlens.dtype != torch.int32:
            refiner_cu_seqlens = refiner_cu_seqlens.to(device=device, dtype=torch.int32)

        if cu_seqlens_bounds is None and OPT_SDPA_PRECOMPUTED_BOUNDS:
            cu_seqlens_bounds = normalize_cu_seqlens_bounds(
                cu_seqlens, rows=sequence_length
            )
        elif cu_seqlens_bounds is not None:
            cu_seqlens_bounds = normalize_cu_seqlens_bounds(
                cu_seqlens_bounds, rows=sequence_length
            )

        fr_opts = kwargs.get("frame_rate_options")
        adaln_fr = adaln_frame_rate(fr_opts if isinstance(fr_opts, Mapping) else None)
        rope_scale = 1.0
        if isinstance(fr_opts, Mapping) and fr_opts.get("temporal_rope"):
            rope_scale = rope_temporal_scale(
                fr_opts,
                video_timestep=float(kwargs.get("video_timestep", 0.0) or 0.0),
                video_sigma=float(kwargs.get("video_sigma", 1.0) or 1.0),
            )
        if (
            prepared_rope_cache is not None
            and OPT_PREPARED_STRUCTURE
            and abs(rope_scale - 1.0) < 1e-12
        ):
            rope_cache = prepared_rope_cache
        else:
            video_mask = (token_tags.to(device) == 0) if abs(rope_scale - 1.0) >= 1e-12 else None
            rope_frequencies = self.rope(
                image_position_ids.to(device),
                video_mask=video_mask,
                temporal_scale=rope_scale,
                low_frequency_count=int((fr_opts or {}).get("rope_low_frequency_count", 16)),
                frequency_profile=str((fr_opts or {}).get("rope_frequency_profile", "hard")),
            )
            rope_cache = self._rope_cache(rope_frequencies)

        unique_timesteps = unique_timesteps.view(-1).to(device)
        hidden_states, timestep_embeddings = self._embed(
            x=x,
            audio_x=audio_x,
            prompt_embeds=prompt_embeds,
            unique_timesteps=unique_timesteps,
            image_positions=image_positions,
            audio_positions=audio_positions,
            text_positions=text_positions,
            refiner_cu_seqlens=refiner_cu_seqlens,
            refined_prompt_embeds_length=kwargs.get(
                "refined_prompt_embeds_length"
            ),
            frame_rate=adaln_fr,
        )

        combined_indices = (
            inverse_indices * ADALN_MODALITY_COUNT + token_tags.clamp(min=0)
        )
        maximum_condition = int(unique_timesteps.shape[0])
        if (not structure_validated) or DIT_DEBUG_STRUCTURE_CHECKS:
            if inverse_indices.min().item() < 0 or inverse_indices.max().item() >= maximum_condition:
                raise ValueError(
                    "inverse_indices contains a value outside unique_timesteps: "
                    f"valid [0, {maximum_condition - 1}]"
                )
            if token_tags.min().item() < -1 or token_tags.max().item() > 2:
                raise ValueError("token_tags values must be -1, 0, 1, or 2")

        # One RLE per step; block/final broadcast by runs to avoid six index_select
        # operations materializing [S,H].
        if OPT_ADALN_SEGMENT_BROADCAST:
            mod_segments = _index_runs(combined_indices)
            inv_segments = _index_runs(inverse_indices)
        else:
            mod_segments, inv_segments = combined_indices, inverse_indices

        mod_cache = self._modulation_cache
        final_modulation = None
        if mod_cache is not None:
            final_modulation = mod_cache.final_layer(unique_timesteps, device=device)

        for i, block in enumerate(self.blocks):
            modulation = (
                mod_cache.block(i, unique_timesteps, device=device)
                if mod_cache is not None else None
            )
            hidden_states = block(
                hidden_states,
                timestep_embedding=timestep_embeddings,
                combined_indices=mod_segments,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_bounds=cu_seqlens_bounds,
                modulation=modulation,
            )

        video_logits, audio_logits = self.final_layer(
            hidden_states,
            timestep_embedding=timestep_embeddings,
            inverse_indices=inv_segments,
            modulation=final_modulation,
        )
        video_logits = video_logits.index_select(0, output_image_positions)
        audio_logits = audio_logits.index_select(0, audio_positions)

        if not bool(kwargs.get("skip_mask_out_condition", False)):
            video_mask = update_mask.view(-1).to(
                device=device, dtype=video_logits.dtype
            )
            if video_mask.numel() != video_logits.shape[0]:
                raise ValueError(
                    "update_mask length does not match selected video rows: "
                    f"{video_mask.numel()} != {video_logits.shape[0]}"
                )
            video_logits = video_logits * video_mask.unsqueeze(-1)

            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_mask = update_audio_mask.view(-1).to(
                    device=device, dtype=audio_logits.dtype
                )
                if audio_mask.numel() != audio_logits.shape[0]:
                    raise ValueError(
                        "update_audio_mask length does not match audio rows: "
                        f"{audio_mask.numel()} != {audio_logits.shape[0]}"
                    )
                audio_logits = audio_logits * audio_mask.unsqueeze(-1)

        return video_logits, audio_logits

    def post_load_weights(self) -> None:
        """Validate the mixed-precision policy required for H3 parity."""

        fp32_names = self.fp32_param_names()
        for name, parameter in self.named_parameters():
            if not parameter.is_floating_point():  # Skip INT8 QuantizedTensor.
                continue
            expected = torch.float32 if name in fp32_names else self.model_dtype
            if parameter.dtype != expected:
                raise ValueError(
                    f"{name} must be {expected} after loading, got "
                    f"{parameter.dtype}"
                )
        fp32_buffers = self.fp32_buffer_names()
        for name, buffer in self.named_buffers():
            if name in fp32_buffers and buffer.dtype != torch.float32:
                raise ValueError(
                    f"{name} must stay float32, got {buffer.dtype}"
                )

    def fp32_param_names(self) -> frozenset[str]:
        """Return parameter names that must remain fp32 for this checkpoint form."""
        if self.use_adaln_curves:
            # Curve-table mode: no time embedder; all low-rank adaLN weights
            # use fp32.
            base = {
                name
                for name in FP32_PARAM_NAMES
                if not name.startswith("time_embedder.")
            }
            base.update(
                name
                for name, _ in self.named_parameters()
                if ".adaln_proj.linear." in name
            )
        else:
            base = set(FP32_PARAM_NAMES)
        # Comfy registers quantized input calibration scales as parameters and
        # intentionally preserves their checkpoint FP32 dtype. They are cast
        # to the active input device/dtype by mixed_precision_ops at runtime.
        base.update(
            name
            for name, _ in self.named_parameters()
            if name.endswith(".input_scale")
        )
        return frozenset(base)

    def fp32_buffer_names(self) -> frozenset[str]:
        if not self.use_adaln_curves:
            return FP32_BUFFER_NAMES
        return frozenset(FP32_BUFFER_NAMES | {ADALN_CURVE_TABLE_KEY})

    def checkpoint_shape_contract(self) -> dict[str, tuple[int, ...]]:
        """Return expected key shapes without exposing parameter values."""

        return {name: tuple(tensor.shape) for name, tensor in self.state_dict().items()}


# Name used by dynamic model loaders.
EntryClass = MiniMaxH3DiTModel


__all__ = [
    "ADALN_MODALITY_COUNT",
    "EntryClass",
    "FORWARD_SUPPORTED_KWARGS",
    "FP32_BUFFER_NAMES",
    "FP32_PARAM_NAMES",
    "MINIMAX_H3_ADALN_MODALITY_NUM",
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT",
    "MiniMaxH3DiTArchConfig",
    "MiniMaxH3DiTConfig",
    "MiniMaxH3DiTModel",
    "MiniMaxH3Rope",
    "PACKED_SEQUENCE_ALIGNMENT",
    "is_qkv_weight_key",
    "is_qkv_scale_key",
    "normalize_cu_seqlens_bounds",
    "prepare_checkpoint_tensor",
    "prepare_state_dict_qkv_",
    "reorder_grouped_qkv_to_qkv",
    "sdpa_varlen_attention",
]
