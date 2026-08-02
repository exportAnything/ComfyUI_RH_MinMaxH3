"""DiT 注意力/RoPE/QKV 工具（P2 自 dit.py 抽出）。"""
from __future__ import annotations
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

import math
from dataclasses import dataclass, fields
from typing import Any, Callable, Mapping, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .h3_settings import (
    DIT_DEBUG_STRUCTURE_CHECKS,
    OPT_PREPARED_STRUCTURE,
    OPT_SDPA_PRECOMPUTED_BOUNDS,
    QKV_SCALE_SUFFIX,
    QKV_WEIGHT_SUFFIX,
)



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
    # ``dit._impl`` imports the attention helpers while defining the model and
    # its config class.  Resolve the config only when checkpoint conversion is
    # actually requested so the split modules do not form an import-time cycle.
    from .dit import MiniMaxH3DiTConfig

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

def _index_runs(indices: torch.Tensor) -> tuple[tuple[int, int, int], ...]:
    """1D 索引张量 → 连续段 (start, stop, row)；段数通常十几个，DtoH 可忽略。"""
    idx = indices.view(-1)
    n = int(idx.numel())
    if n == 0:
        return ()
    if n == 1:
        return ((0, 1, int(idx[0].item())),)
    cuts = torch.nonzero(idx[1:] != idx[:-1], as_tuple=False).view(-1).add_(1)
    n_runs = int(cuts.numel()) + 1
    starts = torch.empty(n_runs, dtype=torch.long, device=idx.device)
    starts[0] = 0
    if n_runs > 1:
        starts[1:] = cuts
    starts_l, rows_l = starts.tolist(), idx.index_select(0, starts).tolist()
    starts_l.append(n)
    return tuple((starts_l[i], starts_l[i + 1], int(rows_l[i])) for i in range(n_runs))

def _as_mod_segments(indices_or_segments) -> tuple[tuple[int, int, int], ...]:
    if isinstance(indices_or_segments, torch.Tensor):
        return _index_runs(indices_or_segments.to(dtype=torch.long))
    return tuple(indices_or_segments)

def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices_or_segments,
) -> torch.Tensor:
    """对 fresh 的 norm 输出按段 in-place 调制；也接受旧的 per-token indices。"""
    # 曲线表 checkpoint 的调制是 fp32，先按 x 的 dtype 取行再 in-place（等价于
    # 原版 checkpoint 的 bf16 调制精度，且避免 in-place 类型提升报错）
    for a, b, row in _as_mod_segments(indices_or_segments):
        x[a:b].mul_(1.0 + scale[row].to(x.dtype)).add_(shift[row].to(x.dtype))
    return x

def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices_or_segments,
) -> torch.Tensor:
    """gated residual：对 fresh 的 attn/mlp 输出 in-place 累加。"""
    for a, b, row in _as_mod_segments(indices_or_segments):
        other[a:b].mul_(gate[row].to(other.dtype)).add_(x[a:b])
    return other

def _silu_mul(hidden: torch.Tensor) -> torch.Tensor:
    gate, up = hidden.chunk(2, dim=-1)
    return F.silu(gate).mul_(up)  # gate 为 chunk 视图，silu 出新张量后再 mul_

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

def _rope_rotation_table(
    cos_sin_cache: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """``[S, rot]`` cos|sin → ``[1, S, 1, rot/2, 2, 2]`` 旋转矩阵表。

    这是 comfy-kitchen 融合 RMSNorm+RoPE kernel 的入参形状（与 PR#15224 一致）。
    直接从既有 cos/sin 缓存构造，保证融合路径与 eager 路径同源、不重算角度。
    """
    half = cos_sin_cache.shape[-1] // 2
    cos, sin = cos_sin_cache[..., :half], cos_sin_cache[..., half:]
    table = torch.stack((cos, -sin, sin, cos), dim=-1)
    return table.reshape(1, cos_sin_cache.shape[0], 1, half, 2, 2).to(dtype).contiguous()


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

def normalize_cu_seqlens_bounds(
    cu_seqlens: Any, *, rows: int | None = None
) -> tuple[int, ...]:
    """一次验证并生成不可变 bounds；热路径禁止再 .tolist()。"""
    if isinstance(cu_seqlens, tuple):
        bounds = tuple(int(x) for x in cu_seqlens)
    else:
        cached = getattr(cu_seqlens, "_h3_bounds", None)
        if isinstance(cached, tuple) and cached:
            bounds = cached
        else:
            if not hasattr(cu_seqlens, "ndim") or cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
                raise ValueError("cu_seqlens must be a 1D tensor with at least 2 values")
            bounds = tuple(
                int(x) for x in cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
            )
    if len(bounds) < 2:
        raise ValueError("cu_seqlens bounds need at least 2 values")
    if bounds[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {bounds[0]}")
    if rows is not None and bounds[-1] != int(rows):
        raise ValueError(
            f"cu_seqlens must cover the entire packed sequence: "
            f"bounds=({bounds[0]}, {bounds[-1]}), rows={rows}"
        )
    if any(stop < start for start, stop in zip(bounds[:-1], bounds[1:])):
        raise ValueError("cu_seqlens must be monotonically non-decreasing")
    return bounds

def sdpa_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    softmax_scale: float,
    bounds: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Non-causal variable-length self-attention using PyTorch SDPA.

    The packed documents are processed independently, exactly matching a
    ``flash_attn_varlen_func`` call.  On CUDA, PyTorch selects its best
    available SDPA implementation.  No attention crosses a document boundary.
    Prefer precomputed ``bounds`` to avoid per-layer CUDA→CPU sync.
    """
    rows = int(query.shape[0])
    if bounds is None:
        bounds = normalize_cu_seqlens_bounds(cu_seqlens, rows=rows)
    else:
        bounds = normalize_cu_seqlens_bounds(bounds, rows=rows)

    segments = [(start, stop) for start, stop in zip(bounds[:-1], bounds[1:]) if stop > start]
    # 单 document（或仅一段非空）快路径：一次 SDPA，无 output 切片回填循环
    if len(segments) == 1 and segments[0] == (0, rows):
        attended = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )
        return attended.squeeze(0).transpose(0, 1)

    output = torch.empty_like(query)
    for start, stop in segments:
        q_segment = query[start:stop].transpose(0, 1).unsqueeze(0)
        k_segment = key[start:stop].transpose(0, 1).unsqueeze(0)
        v_segment = value[start:stop].transpose(0, 1).unsqueeze(0)
        attended = F.scaled_dot_product_attention(
            q_segment, k_segment, v_segment,
            dropout_p=0.0, is_causal=False, scale=softmax_scale,
        )
        output[start:stop] = attended.squeeze(0).transpose(0, 1)
    return output
