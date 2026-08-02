"""BF16 DiT 按层 CPU↔GPU offload（对齐官方 LayerwiseOffloadManager 合同，单卡精简版）。"""
from __future__ import annotations
import logging
from typing import Any
from .h3_settings import DIT_LAYERWISE_PIN_MEMORY, DIT_LAYERWISE_PREFETCH
LOGGER = logging.getLogger(__name__)
_ATTR = "_h3_layerwise"

class H3LayerwiseOffload:  # blocks 常驻 pinned CPU，推理时按层预取到 GPU
    def __init__(self, model: Any, *, device: Any, layers_attr: str = "blocks", prefetch: int | None = None, pin_memory: bool | None = None):
        import torch
        layers = getattr(model, layers_attr, None)
        if not isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential)) or not len(layers):
            raise RuntimeError(f"DiT 缺少可 offload 的 {layers_attr}")
        self.model, self.layers_attr, self.layers = model, layers_attr, layers
        self.device = torch.device(device)
        self.num_layers = len(layers)
        self.prefetch = min(self.num_layers, max(1, int(prefetch if prefetch is not None else DIT_LAYERWISE_PREFETCH)))
        self.pin_memory = bool(DIT_LAYERWISE_PIN_MEMORY if pin_memory is None else pin_memory)
        self._hooks: list[Any] = []; self._gpu: set[int] = set(); self._events: dict[int, Any] = {}
        self._stream = None; self._enabled = False

    def _pin_block(self, block: Any) -> None:
        import torch
        for tensor in (*block.parameters(), *block.buffers()):
            if tensor.device.type != "cpu": tensor.data = tensor.data.detach().to("cpu")
            if self.pin_memory and torch.cuda.is_available() and not tensor.data.is_pinned():
                try: tensor.data = tensor.data.pin_memory()
                except Exception: pass

    def _move_block(self, idx: int, device: Any, *, non_blocking: bool = False) -> None:
        self.layers[idx].to(device=device, non_blocking=non_blocking)

    def prefetch_layer(self, idx: int, *, non_blocking: bool = True) -> None:
        if idx < 0 or idx >= self.num_layers or idx in self._gpu: return
        import torch
        if self._stream is not None and non_blocking:
            self._stream.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.cuda.stream(self._stream): self._move_block(idx, self.device, non_blocking=True)
            evt = torch.cuda.Event(); evt.record(self._stream); self._events[idx] = evt
        else: self._move_block(idx, self.device, non_blocking=False)
        self._gpu.add(idx)

    def release_layer(self, idx: int) -> None:
        if idx not in self._gpu: return
        self._events.pop(idx, None); self._move_block(idx, "cpu", non_blocking=False)
        self._pin_block(self.layers[idx]); self._gpu.discard(idx)

    def prepare(self, *, non_blocking: bool = True) -> None:
        for i in range(self.prefetch): self.prefetch_layer(i, non_blocking=non_blocking)
        if not non_blocking and self._stream is not None:
            import torch; torch.cuda.current_stream(device=self.device).wait_stream(self._stream)

    def enable(self) -> None:
        import torch
        if self._enabled: self.prepare(non_blocking=False); return
        for block in self.layers: self._pin_block(block)
        for name, child in self.model.named_children():
            if name != self.layers_attr: child.to(self.device)
        if self.device.type == "cuda": self._stream = torch.cuda.Stream(device=self.device)
        self._register_hooks(); self.prepare(non_blocking=False); self._enabled = True
        LOGGER.info("DiT layerwise offload 已启用：layers=%s prefetch=%s device=%s", self.num_layers, self.prefetch, self.device)

    def disable(self) -> None:
        if not self._enabled: return
        self._remove_hooks()
        for idx in list(self._gpu): self.release_layer(idx)
        for name, child in self.model.named_children():
            if name != self.layers_attr: child.to("cpu")
        self._stream = None; self._enabled = False

    def _register_hooks(self) -> None:
        self._remove_hooks()
        for i, layer in enumerate(self.layers):
            def pre(_m, _inp, i=i):
                import torch
                if i == 0: self.prepare(non_blocking=False)
                if i not in self._gpu: self.prefetch_layer(i, non_blocking=False)
                if i in self._events: torch.cuda.current_stream(device=self.device).wait_event(self._events[i])
                if i % self.prefetch == 0:
                    for j in range(i + self.prefetch, i + 2 * self.prefetch):
                        self.prefetch_layer(j % self.num_layers, non_blocking=True)
            def post(_m, _inp, _out, i=i): self.release_layer(i)
            self._hooks += [layer.register_forward_pre_hook(pre), layer.register_forward_hook(post)]

    def _remove_hooks(self) -> None:
        for h in self._hooks: h.remove()
        self._hooks.clear()

def attach_layerwise_offload(model: Any, *, device: Any, layers_attr: str = "blocks") -> H3LayerwiseOffload:
    existing = getattr(model, _ATTR, None)
    if isinstance(existing, H3LayerwiseOffload): return existing
    names = list(getattr(model, "layer_names", None) or [layers_attr])
    mgr = H3LayerwiseOffload(model, device=device, layers_attr=names[0])
    object.__setattr__(model, _ATTR, mgr); return mgr

def get_layerwise_offload(model: Any) -> H3LayerwiseOffload | None:
    mgr = getattr(model, _ATTR, None); return mgr if isinstance(mgr, H3LayerwiseOffload) else None
