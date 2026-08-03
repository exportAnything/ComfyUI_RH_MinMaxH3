"""AdaLN 曲线表 checkpoint：时间嵌入曲线的低秩基 + 网格采样。

原版 checkpoint 每层 ``adaln_proj.linear`` 是 ``[6*3*H, time_embed_dim]``
（上游 96768×2688 ≈ 260 M 参数 × 50 层 ≈ 26 GB，占 BF16 DiT 的 ~39%、INT8 的
~55%）。而它的输入 ``silu(time_embedder(t))`` 只是 ``t∈[0,1]`` 上的一条一维曲线，
把这条曲线投影到秩 ``k`` 的共享基后：

    e(t) ≈ Bᵀ c(t)，       B: [k, D] 行正交
    W e(t) ≈ (W Bᵀ) c(t)   → 权重宽度 D → k

于是每层权重收缩到 ``[6*3*H, k]``，checkpoint 整体缩小约 40%；推理时 time
embedder 完全消失，改为在 ``adaln_t_table [grid, k]`` 上按 ``t`` 线性插值取坐标。

本模块是该格式的唯一契约来源：
* :func:`interpolate_curve_table` —— 推理侧取值（``dit`` 用）
* :func:`curve_grid` / :func:`fit_curve_basis` / :func:`project_adaln_weight`
  —— 离线转换用

与 ComfyUI 上游 H3 支持的 checkpoint 格式一致：buffer 名 ``adaln_t_table``，形状
``[grid, k]`` fp32，``adaln_proj`` 不再自带 silu（已烤进表里）。
"""
from __future__ import annotations

from typing import Any

import torch

from .h3_settings import ADALN_CURVE_TABLE_KEY

#: 曲线模式下 adaLN 权重与表的存储精度（k 很小，fp32 代价可忽略）。
ADALN_CURVE_DTYPE = torch.float32


def curve_grid(grid: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``[grid]``：``t`` 在 ``[0, 1]`` 上的等距采样点，与插值坐标一一对应。"""
    if int(grid) < 2:
        raise ValueError(f"adaln 曲线网格至少 2 行，实际 {grid}")
    return torch.linspace(0.0, 1.0, int(grid), dtype=dtype)


def interpolate_curve_table(
    table: torch.Tensor, timesteps: torch.Tensor
) -> torch.Tensor:
    """``[M]`` timestep → ``[M, k]`` 曲线坐标（线性插值，越界钳到端点）。"""
    if table.ndim != 2:
        raise ValueError(f"adaln_t_table 必须是 [grid, k]，实际 {list(table.shape)}")
    grid = int(table.shape[0])
    if grid < 2:
        raise ValueError(f"adaln_t_table 至少 2 行，实际 {grid}")
    t = timesteps.to(dtype=table.dtype, device=table.device).view(-1)
    # t∈[0,1] → 分数网格坐标；越界钳到曲线两端
    pos = t.clamp(0.0, 1.0) * (grid - 1)
    # max 钳位让 t=1.0 落在最后一个区间上而不是越界读下一行
    lower = pos.floor().long().clamp(max=grid - 2)
    weight = (pos - lower).unsqueeze(1)
    return torch.lerp(table[lower], table[lower + 1], weight)


def sample_time_embedding_curve(
    time_embedder: Any, grid: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """``[grid, D]`` fp64：``silu(time_embedder(t))``，即 adaLN 的真实输入曲线。

    silu 必须在这里施加——曲线表 checkpoint 的 ``adaln_proj`` 不再自带激活。
    """
    import torch.nn.functional as F

    points = curve_grid(grid).to(device=device)
    # no_grad 而非 inference_mode：返回值还要参与 SVD/投影等后续离线计算
    with torch.no_grad():
        embedded = time_embedder(points.to(torch.float32))
        return F.silu(embedded).to(torch.float64)


def fit_curve_basis(
    curve: torch.Tensor, rank: int
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """曲线 ``[G, D]`` → ``(basis [k, D], table [G, k], 误差报告)``。

    未中心化 SVD：曲线的常数分量自然落在第一个奇异向量里，无需额外均值项。
    """
    if curve.ndim != 2:
        raise ValueError(f"曲线必须是 [grid, dim]，实际 {list(curve.shape)}")
    grid, dim = int(curve.shape[0]), int(curve.shape[1])
    k = int(rank)
    if not 1 <= k <= min(grid, dim):
        raise ValueError(f"rank 必须在 [1, {min(grid, dim)}]，实际 {k}")
    work = curve.to(torch.float64)
    _, singular, vh = torch.linalg.svd(work, full_matrices=False)
    basis = vh[:k].contiguous()          # [k, D]，行正交
    table = work @ basis.transpose(0, 1)  # [G, k]
    reconstructed = table @ basis
    residual = reconstructed - work
    energy = float(singular.square().sum())
    kept = float(singular[:k].square().sum())
    report = {
        "rank": float(k),
        "grid": float(grid),
        "dim": float(dim),
        # 奇异值能量占比：1 - 它就是最优秩-k 投影的相对平方误差
        "energy_retained": kept / energy if energy > 0 else 1.0,
        "relative_error": float(residual.norm() / work.norm().clamp(min=1e-30)),
        "max_abs_error": float(residual.abs().max()),
        "spectrum_tail": float(singular[k].item()) if k < singular.numel() else 0.0,
    }
    return basis, table, report


def project_adaln_weight(weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``W [out, D]`` → ``W Bᵀ [out, k]``；bias 不变。"""
    if weight.ndim != 2:
        raise ValueError(f"adaLN 权重必须是 [out, dim]，实际 {list(weight.shape)}")
    if weight.shape[1] != basis.shape[1]:
        raise ValueError(
            f"adaLN 权重列数 {weight.shape[1]} 与基维度 {basis.shape[1]} 不一致"
        )
    return (weight.to(torch.float64) @ basis.transpose(0, 1).to(torch.float64))


def curve_output_error(
    weight: torch.Tensor,
    projected: torch.Tensor,
    curve_table: torch.Tensor,
    timesteps: torch.Tensor,
    time_embedder: Any,
) -> dict[str, float]:
    """在任意 ``t`` 上比较 ``W silu(te(t))`` 与曲线路径（含插值误差）。

    ``t`` 取网格点之间才有意义：秩误差与 lerp 误差都要计进去。矩阵乘走 fp32：
    上游 [96768, 2688] 在 fp64 下一层就要几十秒，而 fp32 的累加噪声（~1e-6）比
    待测的近似误差小两个量级。
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
