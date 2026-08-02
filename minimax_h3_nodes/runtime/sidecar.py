"""采样/解码元数据 JSON sidecar。"""
from __future__ import annotations
import hashlib, json, logging, subprocess, time
from pathlib import Path
from typing import Any
from .h3_settings import OPT_WRITE_SIDECAR

LOGGER = logging.getLogger(__name__)
_PLUGIN_ROOT = Path(__file__).resolve().parents[2]

def _prompt_hash(prompt: Any) -> str:
    return hashlib.sha256(str(prompt or "").encode("utf-8", errors="replace")).hexdigest()[:16]

def _runtime_env() -> dict[str, Any]:
    env: dict[str, Any] = {}
    try:
        env["plugin_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_PLUGIN_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception: pass
    try:
        import torch
        env["torch_version"] = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
            env["gpu_count"] = int(torch.cuda.device_count())
    except Exception: pass
    try:
        import comfy  # type: ignore
        env["comfy_version"] = getattr(comfy, "__version__", None)
    except Exception: pass
    return {k: v for k, v in env.items() if v is not None}

def build_sidecar_meta(latent: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    target = latent.get("target") if isinstance(latent.get("target"), dict) else {}
    meta = {
        "schema": "minimax_h3_sidecar/v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "task": latent.get("task") or target.get("task"),
        "seed": latent.get("seed"),
        "prompt_hash": _prompt_hash(latent.get("prompt") or latent.get("prompt_hash")),
        "sigma_points": latent.get("sigma_points"),
        "video_shift": latent.get("video_shift"),
        "audio_shift": latent.get("audio_shift"),
        "residency_mode": latent.get("residency_mode"),
        "dit_calls": latent.get("dit_calls"),
        "dit_steps_total": latent.get("dit_steps_total"),
        "partition": latent.get("partition"),
        "release_fingerprint": latent.get("release_fingerprint"),
        "target_fingerprint": latent.get("target_fingerprint"),
        "vae_fingerprint": latent.get("vae_fingerprint"),
        "width": target.get("width"),
        "height": target.get("height"),
        "fps": target.get("fps"),
        "frame_count": target.get("frame_count"),
        "duration_seconds": target.get("duration_seconds"),
        "requested_duration_seconds": target.get("requested_duration_seconds"),
        "model_root": latent.get("model_root"),
        "dtype": latent.get("dtype"),
        "quant_format": latent.get("quant_format"),
        "telemetry": latent.get("telemetry"),
        "env": _runtime_env(),
    }
    if extra: meta.update(extra)
    return {k: v for k, v in meta.items() if v is not None}

def write_h3_sidecar(meta: dict[str, Any], *, stem: str | None = None) -> Path | None:
    if not OPT_WRITE_SIDECAR: return None
    try:
        import folder_paths  # Comfy
        out_dir = Path(folder_paths.get_output_directory())
    except Exception:
        out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = stem or f"h3_{meta.get('task','job')}_{meta.get('seed','x')}_{int(time.time())}"
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("H3 sidecar written: %s", path)
    return path
