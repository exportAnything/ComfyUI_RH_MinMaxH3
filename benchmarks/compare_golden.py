#!/usr/bin/env python3
"""比较两次采样 latent（accel=off golden）：max_abs / cosine。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

def _as_tensor(path: Path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        if "video" in obj: return obj["video"].float(), obj.get("audio")
        if "latent" in obj: return obj["latent"].float(), None
    return obj.float(), None

def cosine(a, b) -> float:
    import torch
    x, y = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item())

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--max-abs", type=float, default=2e-3)
    ap.add_argument("--cosine", type=float, default=0.9999)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    import torch
    rv, ra = _as_tensor(args.ref); cv, ca = _as_tensor(args.cand)
    mad = float((rv - cv).abs().max().item())
    cos = cosine(rv, cv)
    ok = mad <= args.max_abs and cos >= args.cosine
    report = {"max_abs": mad, "cosine": cos, "pass": ok, "max_abs_limit": args.max_abs, "cosine_limit": args.cosine}
    if ra is not None and ca is not None:
        report["audio_max_abs"] = float((ra - ca).abs().max().item())
        report["audio_cosine"] = cosine(ra, ca)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out: args.out.write_text(text, encoding="utf-8")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
