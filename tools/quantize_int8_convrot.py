#!/usr/bin/env python3
"""H3 DiT → int8_convrot 离线量化：流式分片、结构推导可量化 Linear、Hadamard+per-row int8。"""
from __future__ import annotations
import argparse, collections, json, re, shutil, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_nodes.runtime.h3_settings import INT8_DIT_DIRNAME, INT8_FORMAT, QUANT_EXCLUDE_HINT, int8_dit_filename  # noqa: E402

VALID_GS = (256, 64, 16)  # convrot Hadamard，优先最大可整除
CLIP_GRID_N, CLIP_LO, CLIP_HI = 80, 0.55, 1.0
MIN_GEMM = 256  # 短边过小则 int8 无收益
DEFAULT_EXCLUDE = QUANT_EXCLUDE_HINT


def best_gs(k: int) -> int | None:
    return next((g for g in VALID_GS if k % g == 0), None)


def _hadamard_fns():
    try:
        from comfy_kitchen.tensor.int8 import _build_hadamard, _rotate_weight
        return _build_hadamard, _rotate_weight
    except ImportError:
        from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_weight
        return _build_hadamard, _rotate_weight


def cq_tensor(gs: int):
    import torch
    cfg = {"format": INT8_FORMAT, "convrot": True, "convrot_groupsize": int(gs)}
    return torch.tensor(list(json.dumps(cfg).encode("utf-8")), dtype=torch.uint8)


def quantize_convrot(w, gs: int, *, mseclip=True, device="cuda"):
    """fp32 upcast → 块 Hadamard → per-row absmax(+MSE clip) → int8+scale。"""
    import torch
    build_h, rotate = _hadamard_fns()
    wf = w.to(device, torch.float32)
    h = build_h(gs, device=wf.device, dtype=torch.float32)
    wr = rotate(wf, h, gs)
    absmax = wr.abs().amax(dim=1, keepdim=True).clamp(min=1e-30)
    if not mseclip:
        scale = (absmax / 127.0).clamp(min=1e-30)
        return (wr / scale).round().clamp(-127, 127).to(torch.int8), scale.to(torch.float32)
    best_mse = torch.full_like(absmax, float("inf"))
    best_scale = absmax / 127.0
    best_q = None
    for a in torch.linspace(CLIP_LO, CLIP_HI, CLIP_GRID_N).tolist():
        scale = (absmax * a / 127.0).clamp(min=1e-30)
        q = (wr / scale).round().clamp(-127, 127)
        mse = ((q * scale - wr) ** 2).mean(dim=1, keepdim=True)
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_scale = torch.where(better, scale, best_scale)
        best_q = q.clone() if best_q is None else torch.where(better.expand_as(q), q, best_q)
    return best_q.to(torch.int8), best_scale.to(torch.float32)


def recon_metrics(qd, scale, w_ref, gs: int, *, device="cuda"):
    import torch
    build_h, rotate = _hadamard_fns()
    deq = rotate(qd.to(device).float() * scale.to(device), build_h(gs, device=device, dtype=torch.float32), gs)
    wf = w_ref.to(device).float()
    cos = torch.nn.functional.cosine_similarity(deq.flatten(), wf.flatten(), dim=0).item()
    rel = ((deq - wf).norm() / wf.norm().clamp(min=1e-30)).item() * 100.0
    return cos, rel


def _mark_ops():
    import torch
    import torch.nn as nn

    class _MarkLinear(nn.Module):  # 非 nn.Linear，仅占 operations.Linear 注入点
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, **_):
            super().__init__()
            self.in_features, self.out_features = int(in_features), int(out_features)
            self.register_buffer("weight", torch.empty(out_features, in_features, device="meta", dtype=dtype or torch.bfloat16), persistent=False)
            if bias:
                self.register_buffer("bias", torch.empty(out_features, device="meta", dtype=dtype or torch.bfloat16), persistent=False)
            else:
                self.bias = None

    class _MarkOps:
        Linear = _MarkLinear

    return _MarkOps, _MarkLinear


def quantizable_linear_names(config, exclude: str | None = DEFAULT_EXCLUDE) -> dict[str, int]:
    """从 DiT 结构推导可量化 Linear 前缀 → groupsize（排除原生 nn.Linear 与 exclude）。"""
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTModel

    MarkOps, MarkLinear = _mark_ops()
    model = MiniMaxH3DiTModel.from_config(config, device="meta", dtype="bfloat16", operations=MarkOps())
    exc = re.compile(exclude) if exclude else None
    out: dict[str, int] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, MarkLinear):
            continue
        if exc and exc.search(name):
            continue
        gs = best_gs(mod.in_features)
        if gs is None or min(mod.out_features, mod.in_features) < MIN_GEMM:
            continue
        out[name] = gs
    return out


def _checkpoint_shards(component: Path) -> tuple[list[Path], dict[str, str]]:
    for name in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json", "transformer.safetensors.index.json"):
        idx = component / name
        if not idx.is_file():
            continue
        wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map") or {}
        if not wm:
            raise SystemExit(f"{idx} 无 weight_map")
        files = sorted({component / str(v) for v in wm.values()})
        miss = [p for p in files if not p.is_file()]
        if miss:
            raise SystemExit(f"分片缺失: {miss[:5]}")
        return files, {str(k): str(v) for k, v in wm.items()}
    singles = sorted(component.glob("*.safetensors"))
    if not singles:
        raise SystemExit(f"{component} 无 safetensors")
    from safetensors import safe_open
    wm = {}
    for p in singles:
        with safe_open(str(p), framework="pt") as r:
            for k in r.keys():
                wm[k] = p.name
    return singles, wm


def _load_config(component: Path):
    from minimax_h3_nodes.runtime.dit import MiniMaxH3DiTConfig
    raw = json.loads((component / "config.json").read_text(encoding="utf-8"))
    return MiniMaxH3DiTConfig.from_dict(raw), raw


def _infer_partition(src: Path, partition: str | None) -> str:
    if partition:
        return partition
    parent = Path(src).resolve().parent.name  # .../FL2VA/transformer → FL2VA
    return parent if parent in ("FL2VA", "Ref2VA") else "FL2VA"


def run_quantize(
    src: Path,
    dst: Path,
    *,
    device: str = "cuda",
    dry_run: bool = False,
    verify: bool = False,
    mseclip: bool = True,
    exclude: str | None = DEFAULT_EXCLUDE,
    warn_thresh: float = 2.0,
    verify_report: Path | None = None,
    partition: str | None = None,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    part = _infer_partition(src, partition)
    out_name = int8_dit_filename(part)  # MiniMax-H3-{FL2VA|Ref2VA}-int8_convrot.safetensors
    config, config_raw = _load_config(src)
    qmap = quantizable_linear_names(config, exclude=exclude)  # name -> gs
    shards, weight_map = _checkpoint_shards(src)
    keys = list(weight_map)
    plan = sorted(qmap)  # Linear 前缀
    by_pat: dict[str, list] = collections.defaultdict(lambda: [0, None, None])
    for name in plan:
        pat = re.sub(r"\d+", "N", name)
        by_pat[pat][0] += 1
        by_pat[pat][2] = qmap[name]
    print(f"SRC {src}")
    print(f"PARTITION {part} OUT {out_name}")
    print(f"QUANTIZE {len(plan)} linears (int8+convrot, {'MSE-clip' if mseclip else 'absmax'}):")
    for pat in sorted(by_pat):
        c, _, gs = by_pat[pat]
        print(f"  x{c:<4d} gs{gs:<3d} {pat}")
    print(f"EXCLUDE regex: {exclude!r}")
    if dry_run:
        print("[dry-run] nothing written.")
        return {"plan": plan, "qmap": qmap, "partition": part, "output_name": out_name}

    dst.mkdir(parents=True, exist_ok=True)
    out_file = dst / out_name
    shutil.copy2(src / "config.json", dst / "config.json")

    # 按分片聚合 key，避免反复 open
    shard_keys: dict[str, list[str]] = collections.defaultdict(list)
    for k, sn in weight_map.items():
        shard_keys[sn].append(k)

    out: dict[str, Any] = {}
    errs: list[tuple] = []
    nq = 0
    t0 = time.time()
    for sn in sorted(shard_keys):
        sp = src / sn
        with safe_open(str(sp), framework="pt", device="cpu") as reader:
            for key in shard_keys[sn]:
                t = reader.get_tensor(key)
                if not key.endswith(".weight"):
                    out[key] = t
                    continue
                base = key[: -len(".weight")]
                if base not in qmap:
                    out[key] = t
                    continue
                # 全零/近零权重（损坏分片或未写入）透传，避免 cos(0,0)=0 误杀
                if not bool(torch.isfinite(t).all()) or float(t.float().abs().max()) < 1e-12:
                    print(f" SKIP zero/nonfinite passthrough {base} shape={tuple(t.shape)}", flush=True)
                    out[key] = t
                    continue
                gs = qmap[base]
                qd, scale = quantize_convrot(t, gs, mseclip=mseclip, device=device)
                if verify:
                    cos, rel = recon_metrics(qd, scale, t, gs, device=device)
                    if cos <= 0.99:
                        raise RuntimeError(f"量化损坏 {base} cos={cos:.5f} relerr={rel:.2f}%")
                    if rel > warn_thresh:
                        print(f" WARN {base} gs={gs} relerr={rel:.2f}% cos={cos:.5f}", flush=True)
                    errs.append((rel, cos, gs, base))
                out[key] = qd.cpu()
                out[f"{base}.weight_scale"] = scale.cpu()
                out[f"{base}.comfy_quant"] = cq_tensor(gs)
                nq += 1
                if nq % 50 == 0:
                    print(f"  {nq}/{len(plan)} ... {base} gs={gs}", flush=True)
                del qd, scale, t
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        print(f"  shard done {sn} ({nq}/{len(plan)})", flush=True)

    save_file(out, str(out_file))
    # 写轻量 index，便于与分片布局兼容
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": out_file.stat().st_size}, "weight_map": {k: out_file.name for k in out}}, indent=2),
        encoding="utf-8",
    )
    meta = {
        "format": INT8_FORMAT,
        "convrot": True,
        "quantized_linears": nq,
        "partition": part,
        "exclude": exclude,
        "mseclip": mseclip,
        "seconds": round(time.time() - t0, 1),
        "output": str(out_file),
        "config": {k: config_raw.get(k) for k in ("hidden_size", "num_layers", "ffn_hidden_size")},
    }
    (dst / "quant_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: quantized {nq} linears, {len(out)} tensors, {meta['seconds']}s -> {out_file}")
    if errs:
        errs.sort(reverse=True)
        rvals = [e[0] for e in errs]
        print(f"=== verify mean {sum(rvals)/len(rvals):.3f}% max {max(rvals):.3f}% ===")
        for r, c, gs, b in errs[:8]:
            print(f"  {r:6.3f}% cos {c:.5f} gs{gs:<3d} {b}")
        if verify_report:
            verify_report.write_text(
                "relerr_pct\tcosine\tgroupsize\tlayer\n" + "".join(f"{r:.4f}\t{c:.6f}\t{gs}\t{b}\n" for r, c, gs, b in errs),
                encoding="utf-8",
            )
    return meta


def main():
    ap = argparse.ArgumentParser(description="MiniMax-H3 DiT → int8_convrot")
    ap.add_argument("--src", required=True, help="BF16 transformer 目录（含 config.json + 分片）")
    ap.add_argument("--dst", default=None, help=f"输出目录，默认 <src父>/{INT8_DIT_DIRNAME}")
    ap.add_argument("--partition", default=None, help="模型类型 FL2VA/Ref2VA，默认从 src 父目录推断")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="逐层反量化校验 cosine/relerr")
    ap.add_argument("--mseclip", action="store_true", default=True)
    ap.add_argument("--no-mseclip", action="store_true")
    ap.add_argument("--exclude", default=DEFAULT_EXCLUDE, help=f"额外排除 regex，默认 {DEFAULT_EXCLUDE!r}")
    ap.add_argument("--warn-thresh", type=float, default=2.0)
    ap.add_argument("--verify-report", default=None)
    args = ap.parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve() if args.dst else src.parent / INT8_DIT_DIRNAME
    run_quantize(
        src, dst,
        device=args.device,
        dry_run=args.dry_run,
        verify=args.verify,
        mseclip=not args.no_mseclip,
        exclude=args.exclude,
        warn_thresh=args.warn_thresh,
        verify_report=Path(args.verify_report) if args.verify_report else None,
        partition=args.partition,
    )


if __name__ == "__main__":
    main()
