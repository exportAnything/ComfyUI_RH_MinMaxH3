"""AdaLN modulation precomputation cache: compute every timestep before sampling, then use table lookups during inference."""
from __future__ import annotations
from typing import Any, Sequence
import torch
from ..contracts import H3_AUDIO_REF_COND_TIMESTEP, H3_IMGVID_COND_TIMESTEP
from .h3_settings import OPT_ADALN_CACHE_DEVICE

def enumerate_modulation_timesteps(
    video_sigmas: Sequence[float],
    audio_sigmas: Sequence[float],
    *,
    visual_floor: float = H3_IMGVID_COND_TIMESTEP,
    audio_floor: float = H3_AUDIO_REF_COND_TIMESTEP,
) -> list[float]:
    """Enumerate every DiT timestep reached during sampling, including the conditioning floor."""
    values = {float(visual_floor), float(audio_floor)}
    for sigma_v, sigma_a in zip(video_sigmas[:-1], audio_sigmas[:-1]):
        t_v, t_a = 1.0 - float(sigma_v), 1.0 - float(sigma_a)
        values.update((t_v, t_a, max(t_v, float(visual_floor)), max(t_a, float(audio_floor))))
    return sorted(values)

class H3PrecomputeUnsupported(RuntimeError):
    """Signal that precomputation is unsafe in the current residency mode (caller should skip rather than fail)."""

def _comfy_managed(linear: torch.nn.Module) -> bool:
    """Comfy ops Linear (INT8/lowvram) is tracked by ModelPatcher and must not be moved/released behind its back."""
    return hasattr(linear, "comfy_cast_weights") or hasattr(linear, "prev_comfy_cast_weights")

def _empty_linear(linear: torch.nn.Module) -> None:
    """Release Linear weight storage while retaining the module shell for state_dict/layerwise traversal."""
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(linear, name, None)
            if param is None or not isinstance(param, torch.nn.Parameter):
                continue
            param.data = torch.empty(0, device="cpu", dtype=param.dtype)

class H3ModulationCache:
    """blocks: [L,6,M,3,H]; final: [2,M,1,H]; table lookup by unique_timesteps."""

    def __init__(
        self,
        timesteps: torch.Tensor,
        blocks: torch.Tensor,
        final: torch.Tensor,
        *,
        frame_rate: float | None = None,
    ):
        self.timesteps = timesteps.detach().to(dtype=torch.float32).cpu().contiguous()
        self.blocks = blocks.detach().contiguous()
        self.final = final.detach().contiguous()
        self.frame_rate = None if frame_rate is None else round(float(frame_rate), 6)
        # round prevents key drift from float32 round trips.
        self.timestep_rows = {
            round(float(t), 6): i for i, t in enumerate(self.timesteps.tolist())
        }

    def bytes(self) -> int:
        return int(self.timesteps.nbytes + self.blocks.nbytes + self.final.nbytes)

    def _rows(self, unique_timesteps: torch.Tensor) -> torch.Tensor:
        try:
            rows = [
                self.timestep_rows[round(float(t), 6)]
                for t in unique_timesteps.detach().cpu().tolist()
            ]
        except KeyError as exc:
            raise RuntimeError(f"AdaLN modulation cache is missing timestep {exc.args[0]}") from None
        return torch.tensor(rows, dtype=torch.long, device=self.blocks.device)

    def block(self, layer: int, unique_timesteps: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Return the same six-tuple as AdalnProj.forward, each [U*3, H]."""
        rows = self._rows(unique_timesteps)
        # [6, U, 3, H] -> 6 × [U*3, H]
        gathered = self.blocks[layer].index_select(1, rows).to(device, non_blocking=True)
        return tuple(y.reshape(-1, y.shape[-1]) for y in gathered.unbind(dim=0))

    def final_layer(self, unique_timesteps: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, ...]:
        rows = self._rows(unique_timesteps)
        gathered = self.final.index_select(1, rows).to(device, non_blocking=True)  # [2,U,1,H]
        return tuple(y.reshape(-1, y.shape[-1]) for y in gathered.unbind(dim=0))

def cache_device_from_setting(compute_device: torch.device) -> torch.device:
    loc = str(OPT_ADALN_CACHE_DEVICE or "auto").lower()
    if loc == "vram" and compute_device.type == "cuda":
        return compute_device
    if loc == "ram":
        return torch.device("cpu")
    # auto: use VRAM on cards over 40GB, otherwise RAM (matching the PR threshold).
    if compute_device.type == "cuda":
        try:
            total = torch.cuda.get_device_properties(compute_device).total_memory
            if total > 40 * (1024 ** 3):
                return compute_device
        except Exception:
            pass
    return torch.device("cpu")

def precompute_dit_modulation(
    model: Any,
    timesteps: Sequence[float] | torch.Tensor,
    *,
    compute_device: torch.device | str | None = None,
    cache_device: torch.device | str | None = None,
    release_weights: bool = True,
    frame_rate: float | None = None,
) -> H3ModulationCache:
    """Precompute layer by layer as a stream, optionally releasing adaln/time_embedder weights."""
    if not hasattr(model, "blocks"):
        raise TypeError("precompute_dit_modulation requires MiniMaxH3DiTModel")
    if getattr(model, "use_adaln_curves", False) or not hasattr(model, "time_embedder"):
        # A curve-table checkpoint has no time embedder; its adaLN weights are already low-rank.
        raise H3PrecomputeUnsupported(
            "Curve-table checkpoint adaLN weights are already low-rank; skipping AdaLN precomputation"
        )
    ts = torch.as_tensor(list(timesteps) if not isinstance(timesteps, torch.Tensor) else timesteps,
                         dtype=torch.float32)
    ts = torch.unique(ts, sorted=True)
    if ts.numel() == 0:
        raise ValueError("modulation timesteps are empty")
    # Under INT8/lowvram, adaln_proj.linear is a Comfy cast-weights Linear whose
    # weight lifecycle belongs to ModelPatcher. Calling .to() or clearing param.data
    # behind it would corrupt VRAM accounting and pinned buffers.
    if any(_comfy_managed(b.adaln_proj.linear) for b in model.blocks):
        raise H3PrecomputeUnsupported(
            "DiT is managed by Comfy ModelPatcher (INT8/lowvram); skipping AdaLN precomputation"
        )
    device = torch.device(compute_device or next(model.parameters()).device)
    store = torch.device(cache_device) if cache_device is not None else cache_device_from_setting(device)
    # Time embedding (fp32 island); experimental frame_rate modifies the embedding before projection.
    model.time_embedder.to(device)
    with torch.inference_mode():
        t_emb = model.time_embedder(ts.to(device), frame_rate=frame_rate).to(
            dtype=getattr(model, "model_dtype", torch.bfloat16)
        )
    block_mods: list[torch.Tensor] = []
    for block in model.blocks:
        proj = block.adaln_proj
        proj.to(device)
        with torch.inference_mode():
            # [M, 3, 6, H] -> [6, M, 3, H]
            projected = proj.project(t_emb).permute(2, 0, 1, 3).contiguous()
        block_mods.append(projected.to(store, non_blocking=store.type == "cuda"))
        if release_weights:
            _empty_linear(proj.linear)
            proj.to("cpu")
        elif store.type == "cpu":
            proj.to("cpu")
    final_proj = model.final_layer.adaln_proj
    final_proj.to(device)
    with torch.inference_mode():
        final = final_proj.project(t_emb.to(dtype=getattr(final_proj, "compute_dtype", t_emb.dtype)))
        final = final.permute(2, 0, 1, 3).contiguous().to(store, non_blocking=store.type == "cuda")
    if release_weights:
        _empty_linear(final_proj.linear)
        final_proj.to("cpu")
        # time_embedder can be released too.
        for mod in (model.time_embedder.proj_in, model.time_embedder.proj_out):
            _empty_linear(mod)
        model.time_embedder.to("cpu")
        model._adaln_weights_released = True  # noqa: SLF001
    else:
        final_proj.to("cpu" if store.type == "cpu" else device)
    blocks = torch.stack(block_mods, dim=0)
    cache = H3ModulationCache(ts, blocks, final, frame_rate=frame_rate)
    model._modulation_cache = cache  # noqa: SLF001
    return cache

__all__ = [
    "H3ModulationCache",
    "H3PrecomputeUnsupported",
    "enumerate_modulation_timesteps",
    "precompute_dit_modulation",
    "cache_device_from_setting",
]
