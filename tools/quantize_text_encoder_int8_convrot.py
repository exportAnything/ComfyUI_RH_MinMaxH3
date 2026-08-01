#!/usr/bin/env python3
"""H3 Qwen3-VL text_encoder → int8_convrot：仅量化 language layers[0,50) 的 attn/mlp 投影。"""
from __future__ import annotations
import argparse, collections, importlib.util, json, re, shutil, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_nodes.runtime.h3_settings import (  # noqa: E402
    INT8_FORMAT,
    INT8_TE_DIRNAME,
    INT8_TE_FILENAME,
    TEXT_ENCODER_DROP_KEY,
    TEXT_ENCODER_QUANT_LINEAR,
    TEXT_ENCODER_SELECTED_LAYERS,
)


def _dit_tool():
    path = ROOT / "tools" / "quantize_int8_convrot.py"
    spec = importlib.util.spec_from_file_location("h3_quantize_int8_convrot", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


QT = _dit_tool()
LINEAR_RE = re.compile(TEXT_ENCODER_QUANT_LINEAR)
DROP_RE = re.compile(TEXT_ENCODER_DROP_KEY)


def normalize_te_partition(value: str) -> str:
    """Normalize TE provenance; Qwen weights are shared across both partitions."""

    normalized = str(value).strip().lower()
    names = {"shared": "shared", "fl2va": "FL2VA", "ref2va": "Ref2VA"}
    if normalized not in names:
        raise ValueError(
            "text-encoder partition must be 'shared', 'FL2VA', or 'Ref2VA'"
        )
    return names[normalized]


def classify_te_weight(base: str, shape: tuple[int, ...]) -> int | None:
    """返回 groupsize；不可量化返回 None。"""
    m = LINEAR_RE.match(base)
    if not m or int(m.group(1)) >= TEXT_ENCODER_SELECTED_LAYERS:
        return None
    if len(shape) != 2:
        return None
    n, k = int(shape[0]), int(shape[1])
    gs = QT.best_gs(k)
    if gs is None or min(n, k) < QT.MIN_GEMM:
        return None
    return gs


def plan_te(weight_map: dict[str, str], src: Path) -> dict[str, int]:
    from safetensors import safe_open
    qmap: dict[str, int] = {}
    # 按分片读 shape，避免全量驻留
    by_shard: dict[str, list[str]] = collections.defaultdict(list)
    for k, sn in weight_map.items():
        if k.endswith(".weight"):
            by_shard[sn].append(k)
    for sn, keys in by_shard.items():
        with safe_open(str(src / sn), framework="pt") as r:
            for key in keys:
                base = key[: -len(".weight")]
                shape = tuple(r.get_slice(key).get_shape())
                gs = classify_te_weight(base, shape)
                if gs is not None:
                    qmap[base] = gs
    return qmap


def run_quantize(
    src: Path,
    dst: Path,
    *,
    device: str = "cuda",
    dry_run: bool = False,
    verify: bool = False,
    mseclip: bool = True,
    warn_thresh: float = 2.0,
    keep_dropped: bool = False,
    partition: str = "shared",
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    partition = normalize_te_partition(partition)
    if not (src / "config.json").is_file():
        raise SystemExit(f"缺少 config.json: {src}")
    shards, weight_map = QT._checkpoint_shards(src)
    qmap = plan_te(weight_map, src)
    expected_linears = TEXT_ENCODER_SELECTED_LAYERS * 7
    if len(qmap) != expected_linears:
        raise RuntimeError(
            "Qwen text-encoder quantization plan is incomplete: "
            f"expected {expected_linears} linears, found {len(qmap)}"
        )
    by_pat: dict[str, list] = collections.defaultdict(lambda: [0, None, None])
    for name, gs in qmap.items():
        pat = re.sub(r"\d+", "N", name)
        by_pat[pat][0] += 1
        by_pat[pat][2] = gs
    print(f"SRC {src}")
    print(f"QUANTIZE {len(qmap)} text-encoder linears (layers<={TEXT_ENCODER_SELECTED_LAYERS-1}, int8+convrot):")
    for pat in sorted(by_pat):
        c, _, gs = by_pat[pat]
        print(f"  x{c:<4d} gs{gs:<3d} {pat}")
    if dry_run:
        print("[dry-run] nothing written.")
        return {"qmap": qmap, "partition": partition}

    dst.mkdir(parents=True, exist_ok=True)
    out_file = dst / INT8_TE_FILENAME  # 模型文件名带模型信息+量化格式
    shutil.copy2(src / "config.json", dst / "config.json")
    # 若有 generation_config 等小文件一并复制
    for name in ("generation_config.json", "preprocessor_config.json"):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)

    shard_keys: dict[str, list[str]] = collections.defaultdict(list)
    for k, sn in weight_map.items():
        shard_keys[sn].append(k)

    out: dict[str, Any] = {}
    errs: list[tuple] = []
    nq = dropped = 0
    t0 = time.time()
    for sn in sorted(shard_keys):
        with safe_open(str(src / sn), framework="pt", device="cpu") as reader:
            for key in shard_keys[sn]:
                if not keep_dropped and DROP_RE.search(key):
                    dropped += 1
                    continue
                t = reader.get_tensor(key)
                if not key.endswith(".weight"):
                    out[key] = t
                    continue
                base = key[: -len(".weight")]
                if base not in qmap:
                    out[key] = t
                    continue
                if not bool(torch.isfinite(t).all()) or float(t.float().abs().max()) < 1e-12:
                    print(f" SKIP zero/nonfinite {base}", flush=True)
                    out[key] = t
                    continue
                gs = qmap[base]
                qd, scale = QT.quantize_convrot(t, gs, mseclip=mseclip, device=device)
                if verify:
                    cos, rel = QT.recon_metrics(qd, scale, t, gs, device=device)
                    if cos <= 0.99:
                        raise RuntimeError(f"量化损坏 {base} cos={cos:.5f} relerr={rel:.2f}%")
                    if rel > warn_thresh:
                        print(f" WARN {base} gs={gs} relerr={rel:.2f}%", flush=True)
                    errs.append((rel, cos, gs, base))
                out[key] = qd.cpu()
                out[f"{base}.weight_scale"] = scale.cpu()
                out[f"{base}.comfy_quant"] = QT.cq_tensor(gs)
                nq += 1
                if nq % 50 == 0:
                    print(f"  {nq}/{len(qmap)} ... {base}", flush=True)
                del qd, scale, t
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
        print(f"  shard done {sn} ({nq}/{len(qmap)})", flush=True)

    if nq != expected_linears:
        raise RuntimeError(
            "Qwen text-encoder quantization did not cover every planned "
            f"Linear: expected {expected_linears}, quantized {nq}"
        )
    save_file(out, str(out_file))
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": out_file.stat().st_size}, "weight_map": {k: out_file.name for k in out}}, indent=2),
        encoding="utf-8",
    )
    # 固化 trim 层数，便于 HF/自定义加载器一致
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    if tc is not None:
        tc["num_hidden_layers"] = TEXT_ENCODER_SELECTED_LAYERS
        (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "format": INT8_FORMAT,
        "convrot": True,
        "partition": partition,
        "arch": "qwen3_vl_text_encoder",
        "quantized_linears": nq,
        "dropped_keys": dropped,
        "selected_layers": TEXT_ENCODER_SELECTED_LAYERS,
        "mseclip": mseclip,
        "seconds": round(time.time() - t0, 1),
        "output": str(out_file),
    }
    (dst / "quant_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: quantized {nq} linears, dropped {dropped} keys, {len(out)} tensors, {meta['seconds']}s -> {out_file}")
    if errs:
        errs.sort(reverse=True)
        rvals = [e[0] for e in errs]
        print(f"=== verify mean {sum(rvals)/len(rvals):.3f}% max {max(rvals):.3f}% ===")
        for r, c, gs, b in errs[:8]:
            print(f"  {r:6.3f}% cos {c:.5f} gs{gs:<3d} {b}")
    return meta


def main():
    ap = argparse.ArgumentParser(description="MiniMax-H3 text_encoder → int8_convrot")
    ap.add_argument("--src", required=True, help="BF16 text_encoder 目录")
    ap.add_argument("--dst", default=None, help=f"默认 <父>/{INT8_TE_DIRNAME}")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mseclip", action="store_true", default=True, help="默认开启 MSE clip")
    ap.add_argument("--no-mseclip", action="store_true")
    ap.add_argument("--keep-dropped", action="store_true", help="保留 lm_head 与 layer>=50（默认丢弃）")
    ap.add_argument("--warn-thresh", type=float, default=2.0)
    ap.add_argument(
        "--partition",
        default="shared",
        choices=("shared", "FL2VA", "Ref2VA"),
        help="TE 权重归属；官方 Qwen 在两个 H3 分区共用，默认 shared",
    )
    args = ap.parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve() if args.dst else src.parent / INT8_TE_DIRNAME
    run_quantize(
        src, dst, device=args.device, dry_run=args.dry_run, verify=args.verify,
        mseclip=not args.no_mseclip, warn_thresh=args.warn_thresh, keep_dropped=args.keep_dropped,
        partition=args.partition,
    )


if __name__ == "__main__":
    main()
