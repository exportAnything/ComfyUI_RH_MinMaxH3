#!/usr/bin/env python3
"""合并 MiniMax-H3 video_vae + audio_vae，权重文件名带类型：MiniMax-H3-{video|audio}_vae.safetensors。"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_nodes.runtime.h3_settings import (  # noqa: E402
    AUDIO_VAE_FILENAME,
    MODEL_NAME,
    VAE_MERGED_DIRNAME,
    VIDEO_VAE_FILENAME,
)


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _rename_weight(comp: Path, *, is_video: bool, filename: str) -> Path:
    """定位 model.safetensors（video 在 source/），改名为带类型的文件名并回写 config。"""
    candidates = [
        *( [comp / "source" / "model.safetensors"] if is_video else [] ),
        comp / "model.safetensors",
        *sorted(comp.rglob("model.safetensors")),
    ]
    src_w = next((p for p in candidates if p.is_file()), None)
    if src_w is None:
        raise FileNotFoundError(f"未找到 model.safetensors: {comp}")
    dst_w = src_w.with_name(filename)
    if src_w != dst_w:
        src_w.rename(dst_w)
    for cfg_path in (comp / "config.json", comp / "source" / "config.json"):
        if not cfg_path.is_file():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "source_safetensors_path" in cfg or cfg_path.parent == comp:
            cfg["source_safetensors_path"] = filename
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dst_w


def merge_vae(partition_dir: Path, dst: Path | None = None) -> dict:
    src = Path(partition_dir).expanduser().resolve()
    v_src, a_src = src / "video_vae", src / "audio_vae"
    if not v_src.is_dir() or not a_src.is_dir():
        raise FileNotFoundError(f"需要 {v_src} 与 {a_src}")
    out = Path(dst).expanduser().resolve() if dst else src / VAE_MERGED_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    v_dst, a_dst = out / "video_vae", out / "audio_vae"
    print(f"MERGE {v_src} + {a_src} -> {out}", flush=True)
    _copytree(v_src, v_dst)
    _copytree(a_src, a_dst)
    vw = _rename_weight(v_dst, is_video=True, filename=VIDEO_VAE_FILENAME)
    aw = _rename_weight(a_dst, is_video=False, filename=AUDIO_VAE_FILENAME)
    meta = {
        "model": MODEL_NAME,
        "output": str(out),
        "video_vae": str(vw),
        "audio_vae": str(aw),
        "video_bytes": vw.stat().st_size,
        "audio_bytes": aw.stat().st_size,
    }
    (out / "merge_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE video={vw.name} ({vw.stat().st_size//(1<<20)}MB) audio={aw.name} ({aw.stat().st_size//(1<<20)}MB)", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description="合并 H3 video/audio VAE 并规范类型文件名")
    ap.add_argument("--src", required=True, help="分区目录，如 .../MiniMax-H3/FL2VA")
    ap.add_argument("--dst", default=None, help=f"默认 <src>/{VAE_MERGED_DIRNAME}")
    args = ap.parse_args()
    merge_vae(Path(args.src), Path(args.dst) if args.dst else None)


if __name__ == "__main__":
    main()
