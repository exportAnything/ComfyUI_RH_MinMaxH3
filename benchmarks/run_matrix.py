#!/usr/bin/env python3
"""H3 基准矩阵 runner：汇总 sidecar telemetry，输出 P50/P95 报告。

用法（在已跑完工作流、sidecar 落在 output/ 之后）::
  python3 benchmarks/run_matrix.py --sidecars /path/to/ComfyUI/output --out benchmarks/results
  python3 benchmarks/run_matrix.py --matrix benchmarks/matrix.json --list
"""
from __future__ import annotations
import argparse, json, statistics, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def _pct(xs: list[float], p: float) -> float | None:
    if not xs: return None
    s = sorted(xs); k = (len(s) - 1) * p; f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)

def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def iter_sidecars(dir_path: Path):
    for p in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception: continue
        if data.get("schema") == "minimax_h3_sidecar/v1" or "telemetry" in data:
            yield p, data

def case_key(meta: dict) -> str:
    t = meta.get("telemetry") or {}
    return "|".join(str(x) for x in (
        meta.get("task"), meta.get("width"), meta.get("height"),
        meta.get("frame_count"), meta.get("residency_mode") or t.get("residency_mode"),
        meta.get("seed"), t.get("accel", meta.get("accel", "off")),
    ))

def summarize(sidecars: list[tuple[Path, dict]]) -> dict:
    by: dict[str, list[dict]] = defaultdict(list)
    for path, meta in sidecars:
        tel = meta.get("telemetry") or {}
        step = (tel.get("step_host_s") or {})
        by[case_key(meta)].append({
            "path": str(path),
            "task": meta.get("task"),
            "width": meta.get("width"), "height": meta.get("height"),
            "residency_mode": meta.get("residency_mode") or tel.get("residency_mode"),
            "dit_calls": meta.get("dit_calls"),
            "step_p50": step.get("p50"), "step_p95": step.get("p95"),
            "peak_alloc": (tel.get("peak_vram") or {}).get("allocated"),
            "stages_s": tel.get("stages_s") or tel.get("decode_stages_s") or {},
            "aborted": tel.get("aborted_reason"),
        })
    report = {}
    for key, rows in by.items():
        p50s = [r["step_p50"] for r in rows if r["step_p50"] is not None]
        p95s = [r["step_p95"] for r in rows if r["step_p95"] is not None]
        peaks = [r["peak_alloc"] for r in rows if r["peak_alloc"] is not None]
        report[key] = {
            "n": len(rows),
            "step_p50_median": _pct(p50s, 0.5),
            "step_p95_median": _pct(p95s, 0.5),
            "peak_alloc_max": max(peaks) if peaks else None,
            "residency_modes": sorted({r["residency_mode"] for r in rows if r["residency_mode"]}),
            "runs": rows,
        }
    return report

def write_markdown(report: dict, out: Path, *, title: str) -> None:
    lines = [f"# {title}", "", "| case | n | step P50 | step P95 | peak alloc | residency |", "|---|---:|---:|---:|---:|---|"]
    for key, row in sorted(report.items()):
        lines.append(
            f"| `{key}` | {row['n']} | {row['step_p50_median']} | {row['step_p95_median']} | "
            f"{row['peak_alloc_max']} | {','.join(row['residency_modes']) or '-'} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="H3 benchmark matrix aggregator")
    ap.add_argument("--matrix", type=Path, default=Path(__file__).with_name("matrix.json"))
    ap.add_argument("--sidecars", type=Path, help="含 h3_*.json sidecar 的目录")
    ap.add_argument("--out", type=Path, default=Path(__file__).with_name("results"))
    ap.add_argument("--list", action="store_true", help="只列出矩阵 case")
    args = ap.parse_args(argv)
    matrix = load_matrix(args.matrix)
    if args.list:
        for c in matrix.get("cases", []):
            print(c["id"], c.get("task"), f"{c.get('width')}x{c.get('height')}", c.get("dtype"), c.get("vram_tier"))
        return 0
    if not args.sidecars:
        ap.error("--sidecars 必填（或使用 --list）")
    sidecars = list(iter_sidecars(args.sidecars))
    report = summarize(sidecars)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.out / "summary.md", title="MiniMax-H3 Benchmark Summary")
    print(f"cases={len(matrix.get('cases', []))} sidecars={len(sidecars)} groups={len(report)} -> {args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
