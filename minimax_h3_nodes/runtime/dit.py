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

可选 ``operations``（Comfy ``mixed_precision_ops`` / ``manual_cast``）注入可量化
Linear；FP32 层（patch/time/final）始终用原生 ``nn.Linear``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .h3_settings import QKV_SCALE_SUFFIX, QKV_WEIGHT_SUFFIX


def _linear_cls(operations: Any | None):
    return operations.Linear if operations is not None else nn.Linear  # Comfy ops 或原生


def _activation_dtype(module: nn.Module, fallback: torch.dtype) -> torch.dtype:
    """Linear 输入 dtype：QuantizedTensor 用 compute/factory dtype，禁止读 int8 weight.dtype。"""
    fk = getattr(module, "factory_kwargs", None) or {}
    if fk.get("dtype") is not None:
        return fk["dtype"]
    w = getattr(module, "weight", None)
    if w is not None and getattr(w, "is_floating_point", lambda: False)():
        return w.dtype
    return fallback


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
    }
)
_FORWARD_SUPPORTED_KWARGS = FORWARD_SUPPORTED_KWARGS


def reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Convert grouped checkpoint rows to the fused ``[Q, K, V]`` layout."""

    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.ndim < 1 or weight.shape[0] != expected_out:
        raise ValueError(
            "QKV tensor has an incompatible first dimension for grouped layout: "
            f"got {tuple(weight.shape)}, expected {expected_out}"
        )
    rest_shape = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest_shape)
    q, k, v = torch.split(
        grouped,
        (heads_per_group * head_dim, head_dim, head_dim),
        dim=1,
    )
    return torch.cat(
        (
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest_shape),
            k.reshape(num_query_groups * head_dim, *rest_shape),
            v.reshape(num_query_groups * head_dim, *rest_shape),
        ),
        dim=0,
    )


# Keep the private source name importable for parity tests.
_reorder_grouped_qkv_to_qkv = reorder_grouped_qkv_to_qkv


def is_qkv_weight_key(key: str) -> bool:
    """Return whether ``key`` is one of H3's fused attention matrices."""

    return key.endswith(QKV_WEIGHT_SUFFIX) or key == "attn.qkv_proj.weight"


def is_qkv_scale_key(key: str) -> bool:
    return key.endswith(QKV_SCALE_SUFFIX) or key == "attn.qkv_proj.weight_scale"


def prepare_checkpoint_tensor(
    key: str,
    tensor: torch.Tensor,
    *,
    config: MiniMaxH3DiTConfig | Mapping[str, Any] | None = None,
    qkv_layout: str = "grouped",
) -> torch.Tensor:
    """Prepare one streamed checkpoint tensor for assignment to the model.

    The operation is intentionally tensor-at-a-time so a safetensors loader
    does not need to materialize a second 60+ GiB state dict.
    INT8 ``weight`` / ``weight_scale`` 同行置换（per-row scale 与 QKV 行布局一致）。
    """

    if not (is_qkv_weight_key(key) or is_qkv_scale_key(key)):
        return tensor
    if qkv_layout == "qkv":
        return tensor
    if qkv_layout != "grouped":
        raise ValueError("qkv_layout must be 'grouped' or 'qkv'")
    arch = (
        config
        if isinstance(config, MiniMaxH3DiTConfig)
        else MiniMaxH3DiTConfig.from_dict(config or {})
    )
    return reorder_grouped_qkv_to_qkv(
        tensor,
        num_query_groups=arch.num_attention_heads,
        heads_per_group=1,
        head_dim=arch.attention_head_dim,
    )


def prepare_state_dict_qkv_(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    config: MiniMaxH3DiTConfig | Mapping[str, Any] | None = None,
    prefix: str = "",
) -> MutableMapping[str, torch.Tensor]:
    """In-place convenience converter for tests and non-streaming loaders."""

    for key in list(state_dict):
        local_key = key[len(prefix) :] if prefix and key.startswith(prefix) else key
        if is_qkv_weight_key(local_key):
            state_dict[key] = prepare_checkpoint_tensor(
                local_key, state_dict[key], config=config
            )
    return state_dict


def _rms_norm(
    size: int,
    *,
    eps: float,
    dtype: torch.dtype,
    device: torch.device | str | None,
) -> nn.RMSNorm:
    return nn.RMSNorm(size, eps=eps, dtype=dtype, device=device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    indices = indices.to(device=x.device, dtype=torch.long)
    return (
        x * (1.0 + scale.index_select(0, indices))
        + shift.index_select(0, indices)
    ).to(x.dtype)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    indices = indices.to(device=x.device, dtype=torch.long)
    return (x + gate.index_select(0, indices) * other).to(x.dtype)


def _silu_mul(hidden: torch.Tensor) -> torch.Tensor:
    gate, up = hidden.chunk(2, dim=-1)
    return F.silu(gate) * up


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

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
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
        half = torch.cat((time_freq, height_freq, width_freq), dim=-1)
        return torch.cat((half, half), dim=-1)


def _rope_cos_sin(
    frequencies: torch.Tensor, *, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        frequencies.cos().to(dtype).unsqueeze(1),
        frequencies.sin().to(dtype).unsqueeze(1),
    )


def _rope_cos_sin_cache(
    frequencies: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    half = frequencies.shape[-1] // 2
    return torch.cat(
        (frequencies[:, :half].cos(), frequencies[:, :half].sin()), dim=-1
    ).to(dtype=dtype, copy=False).contiguous()


def _apply_rope_cos_sin(
    tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    rotated, passthrough = tensor[..., :rotary_dim], tensor[..., rotary_dim:]
    rotated = rotated * cos + _rotate_half(rotated) * sin
    return torch.cat((rotated, passthrough), dim=-1)


def _apply_rope(tensor: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    return _apply_rope_cos_sin(
        tensor, *_rope_cos_sin(frequencies, dtype=tensor.dtype)
    )


def _apply_qk_norm(
    query: torch.Tensor,
    key: torch.Tensor,
    query_norm: nn.RMSNorm,
    key_norm: nn.RMSNorm,
    _head_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query_norm(query), key_norm(key)


def _apply_rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    _positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)
    return (
        _apply_rope_cos_sin(query, cos, sin),
        _apply_rope_cos_sin(key, cos, sin),
    )


def sdpa_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Non-causal variable-length self-attention using PyTorch SDPA.

    The packed documents are processed independently, exactly matching a
    ``flash_attn_varlen_func`` call.  On CUDA, PyTorch selects its best
    available SDPA implementation.  No attention crosses a document boundary.
    """

    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a 1D tensor with at least 2 values")
    bounds = cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    if bounds[0] != 0 or bounds[-1] != query.shape[0]:
        raise ValueError(
            "cu_seqlens must cover the entire packed sequence: "
            f"bounds=({bounds[0]}, {bounds[-1]}), rows={query.shape[0]}"
        )
    if any(stop < start for start, stop in zip(bounds[:-1], bounds[1:])):
        raise ValueError("cu_seqlens must be monotonically non-decreasing")

    output = torch.empty_like(query)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if stop == start:
            continue
        # SDPA layout is [batch, heads, sequence, head_dim].
        q_segment = query[start:stop].transpose(0, 1).unsqueeze(0)
        k_segment = key[start:stop].transpose(0, 1).unsqueeze(0)
        v_segment = value[start:stop].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )
        output[start:stop] = attended.squeeze(0).transpose(0, 1)
    return output


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

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(
                half, dtype=torch.float32, device=timestep.device
            )
            / half
        )
        arguments = timestep.to(torch.float32)[:, None] * frequencies[None]
        embedding = torch.cat(
            (torch.cos(arguments), torch.sin(arguments)), dim=-1
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
    ) -> torch.Tensor:
        del max_seqlen  # PyTorch SDPA derives lengths from tensor slices.
        rows = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(rows, self.num_heads, self.head_dim)
        key = key.view(rows, self.num_heads, self.head_dim)
        value = value.view(rows, self.num_heads, self.head_dim)
        query, key = _apply_qk_norm(
            query, key, self.q_norm, self.k_norm, self.head_dim
        )
        if rope_cache is not None:
            query, key = _apply_rope_qk(
                query, key, rope_cache[0], rope_cache[1]
            )

        if self.attention_function is None:
            output = sdpa_varlen_attention(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                softmax_scale=self.softmax_scale,
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

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(_silu_mul(self.fc1(hidden_states)))


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
    ) -> None:
        super().__init__()
        expected = expand_ratio * config.hidden_size * modality_count
        if out_features != expected:
            raise ValueError(f"AdaLN output width {out_features} != {expected}")
        self.expand_ratio = expand_ratio
        self.modality_count = modality_count
        self.hidden_size = config.hidden_size
        self.compute_dtype = dtype
        Linear = _linear_cls(operations)
        self.linear = Linear(
            config.time_embed_dim,
            out_features,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, timestep_embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        act_dtype = _activation_dtype(self.linear, self.compute_dtype)
        projected = self.linear(F.silu(timestep_embedding).to(dtype=act_dtype))
        conditions = projected.shape[0]
        projected = projected.view(
            conditions * self.modality_count,
            self.expand_ratio * self.hidden_size,
        )
        return tuple(projected.chunk(self.expand_ratio, dim=-1))


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
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
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
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
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
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        timestep_embedding: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaln_proj(timestep_embedding)

        residual = hidden_states
        normed = _modulate_scale_shift(
            self.norm1(hidden_states),
            shift_attention,
            scale_attention,
            combined_indices,
        )
        attended = self.attn(
            normed,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = _modulate_gate(
            residual, gate_attention, attended, combined_indices
        )

        residual = hidden_states
        normed = _modulate_scale_shift(
            self.norm2(hidden_states),
            shift_mlp,
            scale_mlp,
            combined_indices,
        )
        return _modulate_gate(
            residual, gate_mlp, self.mlp(normed), combined_indices
        )


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
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.video_out = nn.Linear(  # FP32 固定原生 Linear
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
        timestep_embedding: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(timestep_embedding)
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

        self.video_patch_proj = nn.Linear(  # FP32 固定原生 Linear
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
    ) -> torch.Tensor:
        """Project and run the two request-static token-refiner blocks."""

        if device is None:
            device = self.condition_proj.weight.device
        text_length = int(refiner_cu_seqlens[1].item())
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
        true_cu_seqlens = torch.tensor(
            [0, text_length, text_length],
            device=device,
            dtype=torch.int32,
        )
        projected = self.condition_proj(text_rows)
        return self.token_refiner(
            projected,
            cu_seqlens=true_cu_seqlens,
            max_seqlen=text_length,
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_length = (
            int(refiner_cu_seqlens[1].item())
            if refined_prompt_embeds_length is None
            else int(refined_prompt_embeds_length)
        )
        if text_length <= 0 or text_length > int(prompt_embeds.shape[0]):
            raise ValueError(
                f"invalid live text length {text_length} for "
                f"{int(prompt_embeds.shape[0])} prompt rows"
            )
        text_positions = text_positions[:text_length]
        if refined_prompt_embeds_length is None:
            text_embeddings = self.refine_prompt_embeds(
                prompt_embeds, refiner_cu_seqlens, device=x.device
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

        timestep_embeddings = self.time_embedder(unique_timesteps)
        return embeddings, timestep_embeddings

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
        image_positions = image_positions.to(device)
        audio_positions = audio_positions.to(device)
        text_positions = text_positions.to(device)
        output_image_positions = output_image_positions.to(device)
        inverse_indices = inverse_indices.to(device)
        token_tags = token_tags.to(device)
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        refiner_cu_seqlens = refiner_cu_seqlens.to(
            device=device, dtype=torch.int32
        )

        rope_frequencies = self.rope(image_position_ids.to(device))
        rope_cache = (
            _rope_cos_sin_cache(
                rope_frequencies, dtype=self.model_dtype
            ),
            torch.arange(sequence_length, device=device, dtype=torch.long),
        )

        hidden_states, timestep_embeddings = self._embed(
            x=x,
            audio_x=audio_x,
            prompt_embeds=prompt_embeds,
            unique_timesteps=unique_timesteps.view(-1).to(device),
            image_positions=image_positions,
            audio_positions=audio_positions,
            text_positions=text_positions,
            refiner_cu_seqlens=refiner_cu_seqlens,
            refined_prompt_embeds_length=kwargs.get(
                "refined_prompt_embeds_length"
            ),
        )

        combined_indices = (
            inverse_indices * ADALN_MODALITY_COUNT + token_tags.clamp(min=0)
        )
        maximum_condition = int(timestep_embeddings.shape[0])
        if inverse_indices.min().item() < 0 or inverse_indices.max().item() >= maximum_condition:
            raise ValueError(
                "inverse_indices contains a value outside unique_timesteps: "
                f"valid [0, {maximum_condition - 1}]"
            )
        if token_tags.min().item() < -1 or token_tags.max().item() > 2:
            raise ValueError("token_tags values must be -1, 0, 1, or 2")

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                timestep_embedding=timestep_embeddings,
                combined_indices=combined_indices,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        video_logits, audio_logits = self.final_layer(
            hidden_states,
            timestep_embedding=timestep_embeddings,
            inverse_indices=inverse_indices,
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

        for name, parameter in self.named_parameters():
            if not parameter.is_floating_point():  # INT8 QuantizedTensor 跳过
                continue
            expected = (
                torch.float32 if name in FP32_PARAM_NAMES else self.model_dtype
            )
            if parameter.dtype != expected:
                raise ValueError(
                    f"{name} must be {expected} after loading, got "
                    f"{parameter.dtype}"
                )
        for name, buffer in self.named_buffers():
            if name in FP32_BUFFER_NAMES and buffer.dtype != torch.float32:
                raise ValueError(
                    f"{name} must stay float32, got {buffer.dtype}"
                )

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
    "prepare_checkpoint_tensor",
    "prepare_state_dict_qkv_",
    "reorder_grouped_qkv_to_qkv",
    "sdpa_varlen_attention",
]
