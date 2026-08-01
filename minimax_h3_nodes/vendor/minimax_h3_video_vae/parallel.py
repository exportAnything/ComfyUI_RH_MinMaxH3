# SPDX-License-Identifier: Apache-2.0
"""Single-process compatibility layer for the released MiniMax H3 video VAE.

The upstream implementation can shard VAE tensors over an SGLang decode
process group.  A ComfyUI custom node is a normal in-process PyTorch module,
so every collective deliberately collapses to its world-size-one equivalent.
The public function names are retained so the model code stays checkpoint
compatible with the reference implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def get_group_rank(group_size: int) -> int:
    if group_size != 1:
        raise RuntimeError("The ComfyUI MiniMax H3 VAE port only supports one process")
    return 0


def get_parallel_state() -> dict:
    return {
        "group_size": 1,
        "group_rank": 0,
        "local_process_group": None,
        "sp_size": 1,
        "sp_rank": 0,
        "sp_enabled": False,
        "sp_process_group": None,
        "tp_size": 1,
        "tp_rank": 0,
    }


def all_gather(tensor: torch.Tensor, group=None):
    del group
    return (tensor,)


def all_gather_var_shape(tensor: torch.Tensor, group=None):
    del group
    return (tensor,)


def all_reduce(input_: torch.Tensor, op=None, group=None):
    del op, group
    return input_


def all_to_all_single(input_: torch.Tensor, group=None):
    del group
    return input_


def get_subseq(input_: torch.Tensor, sp_size=None):
    if sp_size not in (None, 1):
        raise RuntimeError("Sequence sharding is not available in the ComfyUI port")
    return input_


def gather_subseq(
    input_: torch.Tensor,
    sp_size=None,
    local_process_group=None,
):
    del local_process_group
    if sp_size not in (None, 1):
        raise RuntimeError("Sequence sharding is not available in the ComfyUI port")
    return input_


def all_to_all_4D(
    input_: torch.Tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
):
    del scatter_idx, gather_idx, group
    return input_


def exchange_borders(
    input_: torch.Tensor,
    padding: int,
    pad_mode: str,
    sp_rank: int,
    sp_size: int,
    group,
    dim: int = -1,
    async_op: bool = False,
):
    """World-size-one equivalent of the spatial border exchange.

    This path should normally be unreachable because all vendored layers have
    spatial parallelism disabled.  Keeping the local padding behavior makes a
    mistakenly enabled layer fail safely and deterministically.
    """

    del group, async_op
    if sp_rank != 0 or sp_size != 1:
        raise RuntimeError("Spatial VAE sharding is not available in the ComfyUI port")
    if padding == 0:
        return input_
    if dim < 0:
        pad_dim = -1 - dim
    else:
        pad_dim = input_.ndim - 1 - dim
    pad_size = [0] * ((input_.ndim - 2) * 2)
    pad_size[pad_dim * 2] = padding
    pad_size[pad_dim * 2 + 1] = padding
    return F.pad(input_, pad_size, mode=pad_mode)


def exchange_strides(
    input_: torch.Tensor,
    pad_mode: str,
    sp_rank: int,
    sp_size: int,
    group,
    dim: int = -1,
    async_op: bool = False,
):
    """World-size-one equivalent of stride-boundary exchange."""

    del group, async_op
    if sp_rank != 0 or sp_size != 1:
        raise RuntimeError("Spatial VAE sharding is not available in the ComfyUI port")
    if dim not in (-1, -2):
        raise ValueError("dim must be -1 (W) or -2 (H) for exchange_strides")
    if input_.ndim not in (4, 5):
        raise ValueError(f"Input must have 4 or 5 dimensions, got {input_.ndim}")

    # Matches the non-parallel Downsample3D path: extend the bottom and right
    # boundaries before the strided convolution.
    if input_.ndim == 5:
        return F.pad(input_, (0, 1, 0, 1, 0, 0), mode=pad_mode)
    return F.pad(input_, (0, 1, 0, 1), mode=pad_mode)


__all__ = [
    "all_gather",
    "all_gather_var_shape",
    "all_reduce",
    "all_to_all_4D",
    "all_to_all_single",
    "exchange_borders",
    "exchange_strides",
    "gather_subseq",
    "get_group_rank",
    "get_parallel_state",
    "get_subseq",
]
