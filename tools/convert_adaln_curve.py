#!/usr/bin/env python3
"""H3 DiT → adaLN 曲线表 checkpoint（PR#15224 格式）：低秩基折入 adaLN 权重。

原理见 ``minimax_h3_nodes/runtime/adaln_curve.py``：adaLN 的输入
``silu(time_embedder(t))`` 只是一条一维曲线，投影到秩 k 的共享基后每层权重从
``[96768, 2688]`` 收缩到 ``[96768, k]``，time embedder 被 ``adaln_t_table
[grid, k]`` 取代。官方 BF16 DiT 66.3 GB → 约 40 GB；INT8 47.0 GB → 约 21 GB
（INT8 下 adaLN 未量化，收益更大）。

用法::

    python3 tools/convert_adaln_curve.py --src .../FL2VA/transformer --verify
    python3 tools/convert_adaln_curve.py --src .../FL2VA/transformer_int8_convrot \\
        --rank 64 --grid 1024 --verify --verify-layers all

输出目录默认 ``<src>_adaln_curve``，保留原分片结构、config.json（追加
``adaln_curve_grid``/``time_embed_dim`` 与 ``adaln_curve`` 元数据块）以及
``quant_meta.json``（若源是 INT8）。
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_nodes.runtime.h3_settings import (  # noqa: E402
    ADALN_CURVE_DEFAULT_GRID,
    ADALN_CURVE_DEFAULT_RANK,
    ADALN_CURVE_DIRNAME_SUFFIX,
    ADALN_CURVE_TABLE_KEY,
)

ADALN_WEIGHT_SUFFIX = ".adaln_proj.linear.weight"
TIME_EMBEDDER_PREFIX = "time_embedder."


def _checkpoint_shards(component: Path) -> tuple[list[Path], dict[str, str], str | None]:
    """返回 (分片列表, weight_map, index 文件名或 None)。"""
    from safetensors import safe_open

    for name in (
        "diffusion_pytorch_model.safetensors.index.json",
        "model.safetensors.index.json",
        "transformer.safetensors.index.json",
    ):
        index = component / name
        if not index.is_file():
            continue
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        if not weight_map:
            raise SystemExit(f"{index} 无 weight_map")
        files = sorted({component / str(v) for v in weight_map.values()})
        missing = [p for p in files if not p.is_file()]
        if missing:
            raise SystemExit(f"分片缺失: {missing[:5]}")
        return files, {str(k): str(v) for k, v in weight_map.items()}, name

    singles = sorted(component.glob("*.safetensors"))
    if not singles:
        raise SystemExit(f"{component} 无 safetensors")
    weight_map = {}
    for path in singles:
        with safe_open(str(path), framework="pt") as reader:
            for key in reader.keys():
                weight_map[key] = path.name
    return singles, weight_map, None


def _load_time_embedder(component: Path, weight_map: dict[str, str]):
    """从 checkpoint 复原 time embedder（仅 4 个小张量）。"""
    import torch
    from safetensors import safe_open

    from minimax_h3_nodes.runtime.dit._impl import (
        MiniMaxH3DiTConfig,
        MiniMaxH3TimeEmbedder,
    )

    wanted = {
        f"{TIME_EMBEDDER_PREFIX}proj_in.weight",
        f"{TIME_EMBEDDER_PREFIX}proj_in.bias",
        f"{TIME_EMBEDDER_PREFIX}proj_out.weight",
        f"{TIME_EMBEDDER_PREFIX}proj_out.bias",
    }
    found: dict[str, Any] = {}
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for key, shard_name in weight_map.items():
        if key in wanted:
            by_shard[shard_name].append(key)
    for shard_name, keys in by_shard.items():
        with safe_open(str(component / shard_name), framework="pt", device="cpu") as reader:
            for key in keys:
                found[key] = reader.get_tensor(key)
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(
            f"checkpoint 缺少 time embedder 张量 {missing!r}；"
            "该 checkpoint 可能已经是曲线表格式"
        )

    raw = json.loads((component / "config.json").read_text(encoding="utf-8"))
    config = MiniMaxH3DiTConfig.from_dict(raw)
    if config.use_adaln_curves:
        raise SystemExit("源 config 已声明 adaln_curve_grid，无需再次转换")
    embedder = MiniMaxH3TimeEmbedder(config, device="cpu")
    embedder.load_state_dict(
        {k[len(TIME_EMBEDDER_PREFIX):]: v.to(torch.float32) for k, v in found.items()}
    )
    return embedder.eval().requires_grad_(False), config, raw


def run_convert(
    src: Path,
    dst: Path,
    *,
    rank: int = ADALN_CURVE_DEFAULT_RANK,
    grid: int = ADALN_CURVE_DEFAULT_GRID,
    verify: bool = False,
    verify_layers: int | None = 4,
    verify_points: int = 64,
    cosine_floor: float = 0.9999,
    dry_run: bool = False,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from minimax_h3_nodes.runtime.adaln_curve import (
        ADALN_CURVE_DTYPE,
        curve_output_error,
        fit_curve_basis,
        project_adaln_weight,
        sample_time_embedding_curve,
    )

    src, dst = Path(src), Path(dst)
    _shards, weight_map, index_name = _checkpoint_shards(src)
    embedder, _config, config_raw = _load_time_embedder(src, weight_map)
    adaln_keys = sorted(k for k in weight_map if k.endswith(ADALN_WEIGHT_SUFFIX))
    if not adaln_keys:
        raise SystemExit(f"{src} 里没有 {ADALN_WEIGHT_SUFFIX} 权重")

    curve = sample_time_embedding_curve(embedder, grid)  # [G, D] fp64
    basis, table, report = fit_curve_basis(curve, rank)
    print(f"SRC {src}")
    print(f"CURVE grid={grid} dim={int(curve.shape[1])} rank={rank}")
    print(
        "BASIS energy_retained={energy_retained:.9f} rel_err={relative_error:.3e} "
        "max_abs={max_abs_error:.3e}".format(**report)
    )
    print(f"ADALN layers {len(adaln_keys)} → 权重宽度 {int(curve.shape[1])} -> {rank}")
    if dry_run:
        print("[dry-run] nothing written.")
        return {"report": report, "adaln_layers": len(adaln_keys)}

    dst.mkdir(parents=True, exist_ok=True)
    shard_keys: dict[str, list[str]] = collections.defaultdict(list)
    for key, shard_name in weight_map.items():
        shard_keys[shard_name].append(key)
    # 单文件 checkpoint 的文件名就是节点 COMBO 里的模型名：加后缀，否则曲线表版
    # 与原版在下拉框里同名、无法区分（多分片按目录名展示，不受影响）。
    renamed = {}
    if len(shard_keys) == 1:
        only = next(iter(shard_keys))
        stem = Path(only).stem
        renamed[only] = f"{stem}-{ADALN_CURVE_DIRNAME_SUFFIX}.safetensors"
    # 采样表挂在含 adaLN 权重最多的分片上，跟着它一起加载
    table_shard = max(
        shard_keys,
        key=lambda name: sum(k.endswith(ADALN_WEIGHT_SUFFIX) for k in shard_keys[name]),
    )

    # 校验点刻意取网格点之间，把 lerp 误差也计进去；固定种子便于复现
    verify_points_t = None
    if verify:
        generator = torch.Generator().manual_seed(42)
        verify_points_t = torch.rand(
            int(verify_points), generator=generator, dtype=torch.float64
        )
    verify_budget = (
        len(adaln_keys)
        if verify_layers is None
        else max(0, int(verify_layers))
    )
    errors: list[tuple[float, float, str]] = []

    new_weight_map: dict[str, str] = {}
    total_in = total_out = 0
    started = time.time()
    for shard_name in sorted(shard_keys):
        out: dict[str, Any] = {}
        with safe_open(str(src / shard_name), framework="pt", device="cpu") as reader:
            for key in shard_keys[shard_name]:
                if key.startswith(TIME_EMBEDDER_PREFIX):
                    continue  # 已烤进采样表
                tensor = reader.get_tensor(key)
                total_in += tensor.numel() * tensor.element_size()
                if key.endswith(ADALN_WEIGHT_SUFFIX):
                    projected = project_adaln_weight(tensor, basis)
                    if verify and len(errors) < verify_budget:
                        metrics = curve_output_error(
                            tensor, projected, table, verify_points_t, embedder
                        )
                        errors.append((metrics["cosine"], metrics["relative_error"], key))
                        if metrics["cosine"] < cosine_floor:
                            raise SystemExit(
                                f"曲线近似不达标 {key}: cosine={metrics['cosine']:.6f} "
                                f"< {cosine_floor}；请提高 --rank 或 --grid"
                            )
                    tensor = projected.to(ADALN_CURVE_DTYPE)
                elif key.endswith(".adaln_proj.linear.bias"):
                    tensor = tensor.to(ADALN_CURVE_DTYPE)  # 与低秩权重同精度
                out[key] = tensor.contiguous()
                total_out += out[key].numel() * out[key].element_size()
        if shard_name == table_shard:
            out[ADALN_CURVE_TABLE_KEY] = table.to(ADALN_CURVE_DTYPE).contiguous()
            total_out += out[ADALN_CURVE_TABLE_KEY].numel() * 4
        out_name = renamed.get(shard_name, shard_name)
        save_file(out, str(dst / out_name))
        new_weight_map.update({k: out_name for k in out})
        print(f"  shard done {shard_name} -> {out_name} ({len(out)} tensors)", flush=True)
        del out

    architecture = dict(config_raw)
    for wrapper in ("arch_config", "transformer_config", "dit_config"):
        if isinstance(config_raw.get(wrapper), dict):
            architecture = dict(config_raw[wrapper])
            break
    architecture["time_embed_dim"] = int(rank)
    architecture["adaln_curve_grid"] = int(grid)
    architecture.pop("timestep_input_dim", None)
    architecture.pop("time_embed_hidden_size", None)
    out_config = dict(config_raw)
    for wrapper in ("arch_config", "transformer_config", "dit_config"):
        if isinstance(config_raw.get(wrapper), dict):
            out_config[wrapper] = architecture
            break
    else:
        out_config = architecture
    out_config["adaln_curve"] = {
        "grid": int(grid),
        "rank": int(rank),
        "source_dim": int(curve.shape[1]),
        "energy_retained": report["energy_retained"],
        "basis_relative_error": report["relative_error"],
        "silu_folded": True,
        "source": str(src),
    }
    (dst / "config.json").write_text(
        json.dumps(out_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    index_payload = {
        "metadata": {"total_size": total_out},
        "weight_map": new_weight_map,
    }
    (dst / (index_name or "model.safetensors.index.json")).write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8"
    )
    if (src / "quant_meta.json").is_file():
        shutil.copy2(src / "quant_meta.json", dst / "quant_meta.json")

    meta = {
        "grid": int(grid),
        "rank": int(rank),
        "adaln_layers": len(adaln_keys),
        "bytes_before": total_in,
        "bytes_after": total_out,
        "shrink_pct": round(100.0 * (1.0 - total_out / max(1, total_in)), 2),
        "basis": report,
        "seconds": round(time.time() - started, 1),
        "verified_layers": len(errors),
        "verify_points": int(verify_points) if verify else 0,
    }
    if errors:
        meta["cosine_min"] = min(c for c, _, _ in errors)
        meta["relative_error_max"] = max(r for _, r, _ in errors)
    (dst / "adaln_curve_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"DONE: {len(adaln_keys)} adaLN layers, "
        f"{total_in / 1024 ** 3:.2f} GiB -> {total_out / 1024 ** 3:.2f} GiB "
        f"(缩减 {meta['shrink_pct']:.2f}%), {meta['seconds']}s -> {dst}"
    )
    if errors:
        worst = min(errors)
        print(
            f"=== verify {len(errors)} layers × {verify_points} t: "
            f"cosine min {worst[0]:.7f} relerr max {meta['relative_error_max']:.3e} "
            f"({worst[2]}) ==="
        )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax-H3 DiT → adaLN 曲线表")
    parser.add_argument("--src", required=True, help="transformer 目录（config.json + 分片）")
    parser.add_argument("--dst", default=None, help=f"输出目录，默认 <src>_{ADALN_CURVE_DIRNAME_SUFFIX}")
    parser.add_argument("--rank", type=int, default=ADALN_CURVE_DEFAULT_RANK, help="曲线基的秩 k")
    parser.add_argument("--grid", type=int, default=ADALN_CURVE_DEFAULT_GRID, help="采样表行数")
    parser.add_argument("--verify", action="store_true", help="逐层比对曲线路径与真实 adaLN 输出")
    parser.add_argument(
        "--verify-layers", default="4",
        help="校验层数，'all' 为全部（每层一次 [96768,2688] 矩阵乘，较慢）",
    )
    parser.add_argument("--verify-points", type=int, default=64, help="每层随机 t 的个数")
    parser.add_argument("--cosine-floor", type=float, default=0.9999, help="低于该 cosine 直接失败")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = (
        Path(args.dst).expanduser().resolve()
        if args.dst
        else src.parent / f"{src.name}_{ADALN_CURVE_DIRNAME_SUFFIX}"
    )
    layers = None if str(args.verify_layers).lower() == "all" else int(args.verify_layers)
    run_convert(
        src, dst,
        rank=args.rank,
        grid=args.grid,
        verify=args.verify,
        verify_layers=layers,
        verify_points=args.verify_points,
        cosine_floor=args.cosine_floor,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
