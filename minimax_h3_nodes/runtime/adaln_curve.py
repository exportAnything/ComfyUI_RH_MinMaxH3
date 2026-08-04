"""AdaLN curve-table checkpoint: low-rank basis plus grid samples of the time-embedding curve.

In the original checkpoint, each layer's ``adaln_proj.linear`` is
``[6*3*H, time_embed_dim]`` (upstream 96768×2688 ≈ 260M parameters × 50 layers
≈ 26GB, about 39% of BF16 DiT and 55% of INT8). Its input,
``silu(time_embedder(t))``, is only a one-dimensional curve over ``t∈[0,1]``.
After projecting this curve onto a shared rank-``k`` basis:

    e(t) ≈ Bᵀ c(t),        B: [k, D] with orthonormal rows
    W e(t) ≈ (W Bᵀ) c(t)   → weight width D → k

each layer's weights shrink to ``[6*3*H, k]``, reducing the checkpoint by about
40%. At inference, the time embedder disappears entirely and coordinates are
obtained by linearly interpolating ``adaln_t_table [grid, k]`` at ``t``.

This module is the single contract source for the format:
* :func:`interpolate_curve_table` — inference-side lookup (used by ``dit``)
* :func:`curve_grid` / :func:`fit_curve_basis` / :func:`project_adaln_weight`
  — used for offline conversion

This matches the checkpoint format supported by upstream ComfyUI H3: buffer name
``adaln_t_table``, shape ``[grid, k]`` fp32, and ``adaln_proj`` no longer applies
silu because it is baked into the table.
"""
from __future__ import annotations

from typing import Any

import torch

from .h3_settings import ADALN_CURVE_TABLE_KEY

#: Storage precision for adaLN weights and tables in curve mode (k is small, so fp32 cost is negligible).
ADALN_CURVE_DTYPE = torch.float32


def curve_grid(grid: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``[grid]``: uniformly spaced ``t`` samples over ``[0, 1]``, corresponding one-to-one with interpolation coordinates."""
    if int(grid) < 2:
        raise ValueError(f"adaln curve grid must have at least 2 rows; got {grid}")
    return torch.linspace(0.0, 1.0, int(grid), dtype=dtype)


def interpolate_curve_table(
    table: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """Map ``[M]`` timestep to ``[M, k]`` curve coordinates (linear interpolation, clamped to endpoints)."""
    if table.ndim != 2:
        raise ValueError(f"adaln_t_table must be [grid, k]; got {list(table.shape)}")
    grid = int(table.shape[0])
    if grid < 2:
        raise ValueError(f"adaln_t_table must have at least 2 rows; got {grid}")
    t = timesteps.to(dtype=table.dtype, device=table.device).view(-1)
    # t∈[0,1] → fractional grid coordinate; clamp out-of-range values to the curve endpoints.
    pos = t.clamp(0.0, 1.0) * (grid - 1)
    # The max clamp keeps t=1.0 in the final interval instead of reading past the last row.
    lower = pos.floor().long().clamp(max=grid - 2)
    weight = (pos - lower).unsqueeze(1)
    return torch.lerp(table[lower], table[lower + 1], weight)


def sample_time_embedding_curve(
    time_embedder: Any, grid: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """``[grid, D]`` fp64: ``silu(time_embedder(t))``, the actual adaLN input curve.

    silu must be applied here because ``adaln_proj`` in a curve-table checkpoint no longer contains the activation.
    """
    import torch.nn.functional as F

    points = curve_grid(grid).to(device=device)
    # Use no_grad rather than inference_mode because the result participates in
    # later offline computations such as SVD/projection.
    with torch.no_grad():
        embedded = time_embedder(points.to(torch.float32))
        return F.silu(embedded).to(torch.float64)


def fit_curve_basis(
    curve: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Map curve ``[G, D]`` to ``(basis [k, D], table [G, k], error report)``.

    Uncentered SVD: the curve's constant component naturally falls into the first
    singular vector, so no extra mean term is needed.
    """
    if curve.ndim != 2:
        raise ValueError(f"curve must be [grid, dim]; got {list(curve.shape)}")
    grid, dim = int(curve.shape[0]), int(curve.shape[1])
    k = int(rank)
    if not 1 <= k <= min(grid, dim):
        raise ValueError(f"rank must be in [1, {min(grid, dim)}]; got {k}")
    work = curve.to(torch.float64)
    _, singular, vh = torch.linalg.svd(work, full_matrices=False)
    basis = vh[:k].contiguous()          # [k, D], orthonormal rows.
    table = work @ basis.transpose(0, 1)  # [G, k]
    reconstructed = table @ basis
    residual = reconstructed - work
    energy = float(singular.square().sum())
    kept = float(singular[:k].square().sum())
    report = {
        "rank": float(k),
        "grid": float(grid),
        "dim": float(dim),
        # Singular-value energy fraction: 1 minus this value is the relative
        # squared error of the optimal rank-k projection.
        "energy_retained": kept / energy if energy > 0 else 1.0,
        "relative_error": float(residual.norm() / work.norm().clamp(min=1e-30)),
        "max_abs_error": float(residual.abs().max()),
        "spectrum_tail": float(singular[k].item()) if k < singular.numel() else 0.0,
    }
    return basis, table, report


def project_adaln_weight(weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Map ``W [out, D]`` to ``W Bᵀ [out, k]``; bias is unchanged."""
    if weight.ndim != 2:
        raise ValueError(f"adaLN weights must be [out, dim]; got {list(weight.shape)}")
    if weight.shape[1] != basis.shape[1]:
        raise ValueError(
            f"adaLN weight column count {weight.shape[1]} does not match basis dimension {basis.shape[1]}"
        )
    return (weight.to(torch.float64) @ basis.transpose(0, 1).to(torch.float64))


def curve_output_error(
    weight: torch.Tensor,
    projected: torch.Tensor,
    curve_table: torch.Tensor,
    timesteps: torch.Tensor,
    time_embedder: Any,
) -> dict[str, float]:
    """Compare ``W silu(te(t))`` with the curve path at arbitrary ``t`` (including interpolation error).

    ``t`` is meaningful between grid points because both rank error and lerp error
    must be counted. Matrix multiplication uses fp32: one upstream [96768, 2688]
    layer takes tens of seconds in fp64, while fp32 accumulation noise (~1e-6) is
    two orders of magnitude below the approximation error being measured.
    """
    import torch.nn.functional as F

    with torch.no_grad():
        exact_input = F.silu(time_embedder(timesteps.to(torch.float32)))
    exact = exact_input.to(torch.float32) @ weight.to(torch.float32).transpose(0, 1)
    coords = interpolate_curve_table(curve_table.to(torch.float32), timesteps)
    approx = coords @ projected.to(torch.float32).transpose(0, 1)
    diff = approx - exact
    cosine = torch.nn.functional.cosine_similarity(
        approx.reshape(-1), exact.reshape(-1), dim=0
    )
    return {
        "cosine": float(cosine),
        "relative_error": float(diff.norm() / exact.norm().clamp(min=1e-30)),
        "max_abs_error": float(diff.abs().max()),
    }


__all__ = [
    "ADALN_CURVE_DTYPE",
    "ADALN_CURVE_TABLE_KEY",
    "curve_grid",
    "curve_output_error",
    "fit_curve_basis",
    "interpolate_curve_table",
    "project_adaln_weight",
    "sample_time_embedding_curve",
]
