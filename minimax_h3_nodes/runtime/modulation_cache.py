"""AdaLN 调制预计算缓存：采样前算完全部 timestep，推理只查表。"""
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
    """枚举采样会碰到的全部 DiT timestep（含 cond floor）。"""
    values = {float(visual_floor), float(audio_floor)}
    for sigma_v, sigma_a in zip(video_sigmas[:-1], audio_sigmas[:-1]):
        t_v, t_a = 1.0 - float(sigma_v), 1.0 - float(sigma_a)
        values.update((t_v, t_a, max(t_v, float(visual_floor)), max(t_a, float(audio_floor))))
    return sorted(values)

class H3PrecomputeUnsupported(RuntimeError):
    """当前驻留模式下不能安全预计算（调用方应跳过而非失败）。"""

def _comfy_managed(linear: torch.nn.Module) -> bool:
    """Comfy ops Linear（INT8/lowvram）由 ModelPatcher 记账，禁止绕过它搬运/释放。"""
    return hasattr(linear, "comfy_cast_weights") or hasattr(linear, "prev_comfy_cast_weights")

def _empty_linear(linear: torch.nn.Module) -> None:
    """释放 Linear 权重占位，保留模块壳给 state_dict/layerwise 遍历。"""
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(linear, name, None)
            if param is None or not isinstance(param, torch.nn.Parameter):
                continue
            param.data = torch.empty(0, device="cpu", dtype=param.dtype)

class H3ModulationCache:
    """blocks: [L,6,M,3,H]；final: [2,M,1,H]；按 unique_timesteps 查表。"""

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
        # round 防 float32 往返键漂移
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
            raise RuntimeError(f"AdaLN modulation cache 缺少 timestep {exc.args[0]}") from None
        return torch.tensor(rows, dtype=torch.long, device=self.blocks.device)

    def block(self, layer: int, unique_timesteps: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, ...]:
        """返回与 AdalnProj.forward 相同的 6 元组，各 [U*3, H]。"""
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
    # auto：>40GB 卡放 VRAM，否则 RAM（与 PR 阈值一致）
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
    """流式逐层预计算；可选释放 adaln/time_embedder 权重。"""
    if not hasattr(model, "blocks"):
        raise TypeError("precompute_dit_modulation 需要 MiniMaxH3DiTModel")
    if getattr(model, "use_adaln_curves", False) or not hasattr(model, "time_embedder"):
        # 曲线表 checkpoint 没有 time embedder，adaLN 权重本就是低秩形式
        raise H3PrecomputeUnsupported(
            "曲线表 checkpoint 的 adaLN 权重已是低秩形式，跳过 AdaLN 预计算"
        )
    ts = torch.as_tensor(list(timesteps) if not isinstance(timesteps, torch.Tensor) else timesteps,
                         dtype=torch.float32)
    ts = torch.unique(ts, sorted=True)
    if ts.numel() == 0:
        raise ValueError("modulation timesteps 为空")
    # INT8/lowvram 下 adaln_proj.linear 是 Comfy cast-weights Linear，权重生命周期
    # 归 ModelPatcher；绕过它 .to()/清空 param.data 会破坏显存记账与 pinned buffer。
    if any(_comfy_managed(b.adaln_proj.linear) for b in model.blocks):
        raise H3PrecomputeUnsupported(
            "DiT 由 Comfy ModelPatcher 管理（INT8/lowvram），跳过 AdaLN 预计算"
        )
    device = torch.device(compute_device or next(model.parameters()).device)
    store = torch.device(cache_device) if cache_device is not None else cache_device_from_setting(device)
    # time embed（fp32 孤岛）；实验性 frame_rate 写入嵌入后再投影
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
        # time_embedder 也可释放
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
