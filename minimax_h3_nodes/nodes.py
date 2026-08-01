"""ComfyUI nodes for in-process MiniMax-H3 inference.

No node in this module opens a socket, submits a job, or imports SGLang.  The
runtime components are ordinary Python/PyTorch modules loaded into the same
process as ComfyUI.

The first direct milestone is intentionally T2VA-only.  H3's FL2VA and Ref2VA
paths need additional visual/audio condition encoders and different packed
layouts; accepting those tasks before the ports exist would produce plausible
but incorrect output.
"""

from __future__ import annotations

import importlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
except ImportError:  # Keep import errors actionable during Comfy start-up.
    torch = None  # type: ignore[assignment]

try:
    import comfy.model_management as model_management
    from comfy.utils import ProgressBar
except ImportError:
    model_management = None
    ProgressBar = None

from .contracts import (
    H3_ASPECT_RATIOS,
    H3_AUDIO_CHANNELS,
    H3_AV_LATENT_SCHEMA,
    H3_CONDITIONING_SCHEMA,
    H3_DEFAULT_AUDIO_SHIFT,
    H3_DEFAULT_SIGMA_POINTS,
    H3_DEFAULT_VIDEO_SHIFT,
    H3_MODEL_SCHEMA,
    H3_TASK_FL2VA,
    H3_TASK_REF2VA,
    H3_TASK_T2VA,
    H3_TARGET_SCHEMA,
    H3_TEXT_ENCODER_SCHEMA,
    H3_T2VA_PARTITION,
    H3_VAE_SCHEMA,
    H3TaskNotImplementedError,
    make_t2va_conditioning,
    require_t2va,
    resolve_t2va_target,
    validate_av_latent,
    validate_target,
)
from .runtime.h3_settings import (
    INT8_DIT_DIRNAME,
    INT8_TE_DIRNAME,
    VAE_MERGED_DIRNAME,
)
from .sampling import sample_t2va

CATEGORY = "MiniMax H3 Direct"
MODEL_TYPE = "MINIMAX_H3_DIRECT_MODEL"
TEXT_ENCODER_TYPE = "MINIMAX_H3_TEXT_ENCODER"
VAE_BUNDLE_TYPE = "MINIMAX_H3_VAE_BUNDLE"
TARGET_TYPE = "MINIMAX_H3_TARGET"
CONDITIONING_TYPE = "MINIMAX_H3_CONDITIONING"
AV_LATENT_TYPE = "MINIMAX_H3_AV_LATENT"


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "MiniMax-H3 Direct 需要 ComfyUI 自带的 PyTorch；当前 Python 环境未安装 torch"
        )
    return torch


def _require_comfy():
    if model_management is None:
        raise RuntimeError(
            "该节点只能在 ComfyUI 进程中执行；未找到 comfy.model_management"
        )


def _runtime_module(name: str):
    """Import a sibling runtime module without any SGLang fallback."""

    try:
        return importlib.import_module(f".runtime.{name}", package=__package__)
    except ImportError as exc:
        raise RuntimeError(
            f"MiniMax-H3 Direct runtime.{name} 未安装完整：{exc}. "
            "请确认 custom node 目录包含原生 runtime 代码，而不是旧版服务端节点包。"
        ) from exc


def _model_root_choices() -> list[str]:
    """Loader COMBO 选项；失败时回退占位，保证 object_info 可生成。"""

    module = _runtime_module("components")
    lister = getattr(module, "list_h3_model_roots", None)
    if callable(lister):
        try:
            choices = list(lister())
            if choices:
                return choices
        except Exception:
            pass
    return ["MiniMax-H3"]


def _partition_roots() -> list[Path]:
    """Return concrete FL2VA roots in preferred release order."""

    module = _runtime_module("components")
    lister = getattr(module, "list_h3_model_root_paths", None)
    roots = list(lister()) if callable(lister) else []
    if not roots:
        roots = [module.model_root_path("MiniMax-H3")]
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            partition = Path(
                module.resolve_partition_root(root, H3_T2VA_PARTITION)
            ).resolve()
        except Exception:
            continue
        key = str(partition)
        if key not in seen:
            seen.add(key)
            out.append(partition)
    return out


def _component_dir_choices(prefix: str) -> list[str]:
    """List explicit FL2VA component directories; never expose an auto mode."""

    preferred = {
        "transformer": INT8_DIT_DIRNAME,
        "text_encoder": INT8_TE_DIRNAME,
    }.get(prefix, prefix)
    names: set[str] = set()
    try:
        for root in _partition_roots():
            names.update(
                d.name
                for d in root.iterdir()
                if d.is_dir()
                and d.name.startswith(prefix)
                and (d / "config.json").is_file()
            )
    except Exception:
        pass
    if not names:
        return [preferred]
    return sorted(names, key=lambda name: (name != preferred, name == prefix, name))


def _vae_dir_choices() -> list[str]:
    names: set[str] = set()
    try:
        for root in _partition_roots():
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                if (
                    (child / "video_vae" / "config.json").is_file()
                    and (child / "audio_vae" / "config.json").is_file()
                ):
                    names.add(child.name)
    except Exception:
        pass
    return sorted(names, key=lambda name: (name != VAE_MERGED_DIRNAME, name)) or [
        VAE_MERGED_DIRNAME
    ]


def _component_dir_input(prefix: str, label: str):
    choices = _component_dir_choices(prefix)
    return (
        choices,
        {
            "default": choices[0],
            "tooltip": (
                f"必须明确选择{label}模型目录；不会再自动切换量化/BF16 权重。"
            ),
        },
    )


def _vae_dir_input():
    choices = _vae_dir_choices()
    return (
        choices,
        {
            "default": choices[0],
            "tooltip": (
                "必须明确选择同时包含 video_vae 与 audio_vae 的合并 VAE 目录；"
                "官方合并产物目录名为 vae。"
            ),
        },
    )


def _model_root_input():
    return (
        _model_root_choices(),
        {
            "tooltip": (
                "从 models/diffusers 或 models/minimax_h3 下选择 MiniMax-H3 "
                "权重根目录（可含 FL2VA 分区）。"
            ),
        },
    )


def _validate_model_root_input(model_root="MiniMax-H3", **kwargs):
    # RH / 工作流可能注入不在本机 COMBO 列表中的真实目录名，放行后由执行期解析。
    if model_root is None:
        return True
    if not str(model_root).strip():
        return "model_root 不能为空"
    return True


def _validate_explicit_component_input(value, label: str):
    # A dynamically connected input cannot be validated until execution.
    if value is None:
        return True
    text = str(value).strip()
    if not text or text.lower() == "auto":
        return f"{label} 必须显式选择模型，不能使用 auto 或空值"
    return True


def _existing_directory(value: str, label: str = "model_root") -> Path:
    path = Path(str(value or "").strip()).expanduser()
    if not str(value or "").strip():
        raise ValueError(f"{label} 不能为空")
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"{label} 不是目录：{path}")
    if path.is_dir():
        return path.resolve()

    # Keep the node UI aligned with runtime.components.model_root_path(): a
    # simple folder name below models/diffusers 或 models/minimax_h3 也有效。
    module = _runtime_module("components")
    resolver = getattr(module, "model_root_path", None)
    if callable(resolver):
        try:
            return Path(resolver(value)).resolve()
        except (ValueError, OSError) as exc:
            raise FileNotFoundError(f"{label} 不存在：{path}") from exc
    raise FileNotFoundError(f"{label} 不存在：{path}")


def _resolve_t2va_release(value: str) -> tuple[Path, dict[str, Any], dict[str, float]]:
    """Resolve/validate the FL2VA release before any large component loads."""

    root = _existing_directory(value)
    module = _runtime_module("components")
    resolve_partition = _find_callable(
        module,
        ("resolve_partition_root",),
        "partition root",
    )
    root = Path(resolve_partition(root, H3_T2VA_PARTITION)).resolve()
    metadata_reader = _find_callable(
        module,
        ("release_metadata",),
        "release metadata",
    )
    validator = _find_callable(
        module,
        ("validate_t2va_partition",),
        "T2VA partition validator",
    )
    sigma_reader = _find_callable(
        module,
        ("release_sigma_shift_scales",),
        "sigma-shift metadata parser",
    )
    metadata = metadata_reader(root)
    validator(metadata)
    declared_scales = sigma_reader(metadata)
    sigma_scales = declared_scales or {
        "video": float(H3_DEFAULT_VIDEO_SHIFT),
        "audio": float(H3_DEFAULT_AUDIO_SHIFT),
    }
    return root, dict(metadata), dict(sigma_scales)


def _resolve_selected_component(
    root: Path,
    value: str,
    *,
    keys: tuple[str, ...],
    label: str,
    required_files: tuple[str, ...] = (),
) -> Path:
    validation = _validate_explicit_component_input(value, label)
    if validation is not True:
        raise ValueError(validation)
    module = _runtime_module("components")
    resolver = _find_callable(module, ("resolve_component",), label)
    return Path(
        resolver(
            root,
            keys,
            explicit=str(value).strip(),
            required_files=required_files,
        )
    ).resolve()


def _resolve_selected_vae(root: Path, value: str) -> Path:
    selected = _resolve_selected_component(
        root,
        value,
        keys=(VAE_MERGED_DIRNAME,),
        label="VAE",
    )
    missing = [
        name
        for name in ("video_vae", "audio_vae")
        if not (selected / name / "config.json").is_file()
    ]
    if missing:
        raise ValueError(
            f"VAE 目录 {selected} 缺少合并组件配置：{', '.join(missing)}"
        )
    return selected


def _runtime_dtype_name(name: str, *, allow_float32: bool = True) -> str:
    """Normalise UI dtype values to the local runtime loader contract."""

    normalized = str(name).strip().lower()
    if normalized == "auto":
        # H3's released base weights are BF16.  Individual FP32 layers are
        # selected by the architecture/loader and must not be inferred here.
        return "bfloat16"
    allowed = {"bfloat16", "float16"}
    if allow_float32:
        allowed.add("float32")
    if normalized not in allowed:
        supported = "、".join(sorted(allowed))
        raise ValueError(f"dtype {name!r} 不受支持；可选：auto、{supported}")
    return normalized


def _call_supported(function, **kwargs):
    """Call a runtime entry point while allowing harmless API growth."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return function(**kwargs)
    accepted = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return function(**accepted)


def _find_callable(module: Any, names: tuple[str, ...], component: str):
    for name in names:
        function = getattr(module, name, None)
        if callable(function):
            return function
    raise RuntimeError(
        f"runtime.{module.__name__.rsplit('.', 1)[-1]} 没有 {component} loader；"
        f"期望其中一个入口：{', '.join(names)}"
    )


def _devices() -> tuple[Any, Any]:
    _require_comfy()
    load_device = model_management.get_torch_device()
    offload_getter = getattr(model_management, "unet_offload_device", None)
    offload_device = (
        offload_getter() if callable(offload_getter) else _require_torch().device("cpu")
    )
    return load_device, offload_device


def _wrapper_value(wrapper: Any, *names: str) -> Any | None:
    if isinstance(wrapper, Mapping):
        for name in names:
            if name in wrapper and wrapper[name] is not None:
                return wrapper[name]
    for name in names:
        value = getattr(wrapper, name, None)
        if value is not None:
            return value
    return None


def _unwrap_runtime_handle(wrapper: Any, expected_schema: str, label: str):
    if not isinstance(wrapper, Mapping) or wrapper.get("schema") != expected_schema:
        raise TypeError(
            f"{label} 端口不是 MiniMax-H3 Direct loader 的输出"
        )
    handle = wrapper.get("handle")
    if handle is None:
        raise RuntimeError(f"{label} wrapper 缺少 runtime handle")
    return handle


def _normalise_prompt_embeddings(value: Any):
    t = _require_torch()
    if isinstance(value, Mapping):
        for key in ("prompt_embeds", "text_embeddings", "last_hidden_state"):
            if key in value:
                value = value[key]
                break
    if isinstance(value, (tuple, list)):
        if not value:
            raise RuntimeError("Qwen encoder 返回了空序列")
        value = value[0]
    if not isinstance(value, t.Tensor):
        raise TypeError(
            "Qwen encoder 必须返回 torch.Tensor 或包含 prompt_embeds 的 mapping"
        )
    if value.ndim == 3:
        if int(value.shape[0]) != 1:
            raise ValueError("MiniMax-H3 Direct v0 只支持 batch=1")
        value = value[0]
    if value.ndim != 2:
        raise ValueError(
            f"Qwen encoder 输出必须为 [L,5120]，实际 shape={tuple(value.shape)}"
        )
    # The 64 GB text encoder must not remain co-resident with the 62 GB DiT.
    return value.detach().to(device="cpu").contiguous()


def _encode_prompt(handle: Any, prompt: str):
    encode = _wrapper_value(handle, "encode_prompt", "encode_conditioning")
    if not callable(encode):
        raise RuntimeError(
            "text encoder handle 缺少 encode_prompt(prompt)；"
            "请更新 minimax_h3_nodes/runtime/qwen_encoder.py"
        )
    try:
        try:
            result = _call_supported(
                encode,
                prompt=prompt,
                text=prompt,
                task=H3_TASK_T2VA,
                conditions=[],
            )
        except TypeError:
            # Keep the minimal, documented handle contract friendly to simple
            # implementations that use one positional-only prompt.
            result = encode(prompt)
        return _normalise_prompt_embeddings(result)
    finally:
        # The source runtime never co-resides the ~64 GB Qwen encoder with the
        # ~62 GB DiT.  Enforce that boundary even if encode_prompt raises.
        offload = _wrapper_value(
            handle,
            "offload_after_inference",
            "offload",
            "release_from_device",
        )
        if callable(offload):
            offload()


def _build_t2va_packed(prompt_embeds: Any, target: Mapping[str, Any]):
    module = _runtime_module("packing")
    build = _find_callable(
        module,
        (
            "build_t2va_packed_conditioning",
            "build_t2va_packed",
            "minimax_h3_packed_sequence",
        ),
        "T2VA packing",
    )
    packed = _call_supported(
        build,
        prompt_embeds=prompt_embeds,
        text_embeddings=prompt_embeds,
        text_len=int(prompt_embeds.shape[0]),
        latent_t=int(target["video_latent_t"]),
        latent_h=int(target["video_latent_h"]),
        latent_w=int(target["video_latent_w"]),
        audio_t=int(target["audio_latent_t"]),
        audio_channel=H3_AUDIO_CHANNELS,
        include_keyframe_cond=False,
        keyframe_frame_indices=None,
        frame_count=int(target["frame_count"]),
    )
    if not isinstance(packed, Mapping):
        raise TypeError("runtime.packing 必须返回 packed conditioning mapping")
    # Some adapters return {"packed": structural_fields}; accept that without
    # weakening the strict structural checks in H3DenoiseBranch.
    nested = packed.get("packed")
    if isinstance(nested, Mapping):
        packed = nested
    return dict(packed)


def _load_model_patcher_if_present(handle: Any) -> Any | None:
    patcher = _wrapper_value(handle, "model_patcher", "patcher")
    if patcher is None:
        return None
    _require_comfy()
    model_management.load_models_gpu([patcher])
    return patcher


def _resolve_transformer(handle: Any) -> Any:
    """Materialise a transformer through a runtime handle/ModelPatcher."""

    prepare = _wrapper_value(
        handle,
        "load_for_inference",
        "prepare_for_inference",
        "get_model",
    )
    prepared = prepare() if callable(prepare) else None
    if prepared is not None:
        candidate = _wrapper_value(prepared, "transformer", "model") or prepared
        if callable(candidate):
            return candidate

    patcher = _load_model_patcher_if_present(handle)
    if patcher is not None:
        candidate = _wrapper_value(handle, "transformer")
        if candidate is None:
            candidate = getattr(patcher, "model", None)
        # A BaseModel-style patcher may wrap the actual diffusion module.
        nested = _wrapper_value(
            candidate,
            "diffusion_model",
            "transformer",
            "model",
        )
        if callable(nested):
            candidate = nested
        if callable(candidate):
            return candidate

    candidate = _wrapper_value(handle, "transformer", "model", "module")
    if candidate is None and callable(handle):
        candidate = handle
    if not callable(candidate):
        raise RuntimeError(
            "无法从 model handle 取得 MiniMaxH3DiTModel；"
            "handle 应暴露 transformer/model，或 load_for_inference()"
        )

    # A simple runtime implementation may return an offloaded raw nn.Module.
    # Move device only; never cast the whole module because H3 deliberately
    # keeps selected projections/heads/RoPE state in fp32.
    try:
        parameter = next(candidate.parameters())
    except (AttributeError, StopIteration):
        parameter = None
    if parameter is not None and parameter.device.type == "cpu":
        load_device, _ = _devices()
        candidate.to(device=load_device)
    eval_method = getattr(candidate, "eval", None)
    if callable(eval_method):
        eval_method()
    return candidate


@contextmanager
def _transformer_session(model_wrapper: Mapping[str, Any]):
    handle = _unwrap_runtime_handle(
        model_wrapper, H3_MODEL_SCHEMA, "h3_model"
    )
    transformer = _resolve_transformer(handle)
    try:
        yield transformer
    finally:
        offload = _wrapper_value(
            handle,
            "offload_after_inference",
            "offload",
            "release_from_device",
        )
        if callable(offload):
            offload()


def _decode_video(adapter: Any, latent: Any, target: Mapping[str, Any]):
    t = _require_torch()
    decode = getattr(adapter, "decode", None)
    if not callable(decode):
        raise RuntimeError("video VAE adapter 缺少 decode(latents)")
    frames = _call_supported(
        decode,
        normalized_latents=latent,
        latents=latent,
        latent=latent,
        frame_count=int(target["frame_count"]),
        target_height=int(target["height"]),
        target_width=int(target["width"]),
    )
    if isinstance(frames, Mapping):
        frames = frames.get("images", frames.get("frames"))
    if isinstance(frames, (tuple, list)):
        frames = frames[0] if frames else None
    if not isinstance(frames, t.Tensor):
        raise TypeError("video VAE decode 必须返回 torch.Tensor")

    if frames.ndim == 5:
        if int(frames.shape[1]) in (3, 4):  # [B,C,T,H,W]
            frames = frames.permute(0, 2, 3, 4, 1)
        elif int(frames.shape[-1]) not in (3, 4):  # not [B,T,H,W,C]
            raise ValueError(
                f"无法识别 video decode shape={tuple(frames.shape)}"
            )
        frames = frames.reshape(-1, *frames.shape[-3:])
    elif frames.ndim == 4:
        if int(frames.shape[-1]) in (3, 4):  # Comfy IMAGE
            pass
        elif int(frames.shape[1]) in (3, 4):  # [T,C,H,W]
            frames = frames.permute(0, 2, 3, 1)
        else:
            raise ValueError(
                f"无法识别 video decode shape={tuple(frames.shape)}"
            )
    else:
        raise ValueError(
            f"video decode 应返回 rank-4/5 tensor，实际 {tuple(frames.shape)}"
        )
    frames = frames[..., :3].to(dtype=t.float32)
    height, width = int(target["height"]), int(target["width"])
    if int(frames.shape[1]) < height or int(frames.shape[2]) < width:
        raise ValueError(
            f"decoded frames={int(frames.shape[2])}x{int(frames.shape[1])} "
            f"小于 target={width}x{height}"
        )
    # The native VAE's tile padding is bottom/right; match the source crop.
    return (
        frames[:, :height, :width, :]
        .to(device="cpu")
        .contiguous()
    )


def _decode_audio(
    adapter: Any,
    latent: Any,
    *,
    sample_count: int | None = None,
) -> dict[str, Any]:
    t = _require_torch()
    decode = getattr(adapter, "decode", None)
    if not callable(decode):
        raise RuntimeError("audio VAE adapter 缺少 decode(latents)")
    output = _call_supported(
        decode,
        normalized_latents=latent,
        latents=latent,
        latent=latent,
        sample_count=(int(sample_count) if sample_count is not None else None),
    )
    if isinstance(output, Mapping):
        waveform = output.get("waveform")
        sample_rate = int(
            output.get(
                "sample_rate",
                getattr(adapter, "sample_rate", 32000),
            )
        )
    else:
        waveform = output[0] if isinstance(output, (tuple, list)) else output
        sample_rate = int(getattr(adapter, "sample_rate", 32000))
    if not isinstance(waveform, t.Tensor):
        raise TypeError("audio VAE decode 必须返回 waveform tensor")
    if waveform.ndim == 2:  # [C,L]
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 3:
        # Native H3 audio VAE returns [C,1,L]; Comfy AUDIO is [B,C,L].
        if (
            int(waveform.shape[0]) in (1, 2)
            and int(waveform.shape[1]) == 1
            and int(waveform.shape[0]) != 1
        ):
            waveform = waveform.permute(1, 0, 2)
    else:
        raise ValueError(
            f"audio waveform 必须为 [C,L]、[C,1,L] 或 [B,C,L]，实际 {tuple(waveform.shape)}"
        )
    if waveform.ndim != 3:
        raise ValueError("无法把 audio waveform 转换为 Comfy AUDIO [B,C,L]")
    return {
        "waveform": waveform.to(device="cpu", dtype=t.float32).contiguous(),
        "sample_rate": sample_rate,
    }


class MiniMaxH3DirectModelLoader:
    """Load the native H3 DiT without a server or Diffusers pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "dtype": (
                    ["auto", "bfloat16", "float16"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto/bfloat16 推荐；runtime 会保留 H3 指定的 fp32 层。"
                        ),
                    },
                ),
                "transformer_path": _component_dir_input("transformer", "DiT"),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        transformer_path=INT8_DIT_DIRNAME,
        **kwargs,
    ):
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(transformer_path, "DiT")

    def load(
        self,
        model_root: str,
        dtype: str,
        transformer_path: str = INT8_DIT_DIRNAME,
    ):
        root, release_info, sigma_scales = _resolve_t2va_release(model_root)
        selected_transformer = _resolve_selected_component(
            root,
            transformer_path,
            keys=("transformer", "dit"),
            label="DiT",
            required_files=("config.json",),
        )
        module = _runtime_module("model_loader")
        loader = _find_callable(
            module,
            (
                "load_h3_model",
                "load_h3_transformer",
                "load_minimax_h3_model",
            ),
            "DiT",
        )
        handle = _call_supported(
            loader,
            model_root=str(root),
            root=str(root),
            partition=H3_T2VA_PARTITION,
            task=H3_TASK_T2VA,
            dtype=_runtime_dtype_name(dtype, allow_float32=False),
            dtype_name=_runtime_dtype_name(dtype, allow_float32=False),
            device="auto",
            offload_device="auto",
            transformer_path=str(selected_transformer),
        )
        if handle is None:
            raise RuntimeError("runtime.model_loader 返回了 None")
        return (
            {
                "schema": H3_MODEL_SCHEMA,
                "handle": handle,
                "model_root": str(root),
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "dtype": dtype,
                "transformer_path": str(selected_transformer),
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


class MiniMaxH3DirectTextEncoderLoader:
    """Load the Qwen3-VL-32B layer-50 conditioning encoder in process."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "text_encoder_path": _component_dir_input("text_encoder", "文本编码器"),
            },
        }

    RETURN_TYPES = (TEXT_ENCODER_TYPE,)
    RETURN_NAMES = ("h3_text_encoder",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        text_encoder_path=INT8_TE_DIRNAME,
        **kwargs,
    ):
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(
            text_encoder_path, "Text Encoder"
        )

    def load(
        self,
        model_root: str,
        dtype: str,
        text_encoder_path: str = INT8_TE_DIRNAME,
    ):
        root, release_info, sigma_scales = _resolve_t2va_release(model_root)
        selected_text_encoder = _resolve_selected_component(
            root,
            text_encoder_path,
            keys=("text_encoder", "qwen3vl", "qwen"),
            label="Text Encoder",
            required_files=("config.json",),
        )
        module = _runtime_module("qwen_encoder")
        loader = _find_callable(
            module,
            (
                "load_h3_text_encoder",
                "load_h3_qwen_encoder",
                "load_qwen_encoder",
            ),
            "Qwen3-VL",
        )
        handle = _call_supported(
            loader,
            model_root=str(root),
            root=str(root),
            partition=H3_T2VA_PARTITION,
            dtype=_runtime_dtype_name(dtype),
            dtype_name=_runtime_dtype_name(dtype),
            device="auto",
            offload_device="cpu",
            text_encoder_path=str(selected_text_encoder),
        )
        if handle is None:
            raise RuntimeError("runtime.qwen_encoder 返回了 None")
        return (
            {
                "schema": H3_TEXT_ENCODER_SCHEMA,
                "handle": handle,
                "model_root": str(root),
                "dtype": dtype,
                "text_encoder_path": str(selected_text_encoder),
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


class MiniMaxH3DirectVAELoader:
    """Load the native 24-channel video and 32-channel audio VAEs.

    Both released VAE weight sets stay FP32.  The video adapter independently
    applies FP16 autocast to decode operations where the source runtime does;
    exposing a generic weight dtype here would incorrectly suggest that BF16
    VAE weights are part of the H3 contract.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "vae_path": _vae_dir_input(),
            }
        }

    RETURN_TYPES = (VAE_BUNDLE_TYPE,)
    RETURN_NAMES = ("h3_vae_bundle",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        vae_path=VAE_MERGED_DIRNAME,
        **kwargs,
    ):
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(vae_path, "VAE")

    def load(self, model_root: str, vae_path: str = VAE_MERGED_DIRNAME):
        root, release_info, sigma_scales = _resolve_t2va_release(model_root)
        selected_vae = _resolve_selected_vae(root, vae_path)
        module = _runtime_module("vae_adapter")
        loader = _find_callable(
            module,
            ("load_h3_vae_bundle", "load_minimax_h3_vae_bundle"),
            "dual VAE",
        )
        bundle = _call_supported(
            loader,
            model_root=str(root),
            root=str(root),
            vae_path=str(selected_vae),
            device="cpu",
            offload_device="cpu",
            video_compute_dtype="float16",
            audio_compute_dtype="float32",
        )
        if bundle is None:
            raise RuntimeError("runtime.vae_adapter 返回了 None")
        return (
            {
                "schema": H3_VAE_SCHEMA,
                "bundle": bundle,
                "model_root": str(root),
                "vae_path": str(selected_vae),
                "weight_dtype": "float32",
                "video_decode_compute": "float16_autocast",
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


class MiniMaxH3T2VATarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (
                    list(H3_ASPECT_RATIOS),
                    {"default": "16:9"},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 5.0,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": "时长秒数，必须在 5.0–15.0（执行期再次校验）。",
                    },
                ),
            },
            "optional": {
                "width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "显式输出宽度；0 表示按 aspect_ratio 自动计算。需与 height 同时填写并按 32 对齐。",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 32,
                        "tooltip": "显式输出高度；0 表示按 aspect_ratio 自动计算。需与 width 同时填写并按 32 对齐。",
                    },
                ),
            },
        }

    RETURN_TYPES = (TARGET_TYPE, "STRING")
    RETURN_NAMES = ("target", "shape_info")
    FUNCTION = "build"
    CATEGORY = f"{CATEGORY}/conditioning"

    @classmethod
    def VALIDATE_INPUTS(
        cls, duration_seconds=5.0, width=0, height=0, **kwargs
    ):
        # 连接外部节点时可能绕过 UI min/max，这里先拦一层
        if duration_seconds is None:
            return True
        try:
            value = float(duration_seconds)
        except (TypeError, ValueError):
            return f"Invalid duration_seconds: {duration_seconds!r}"
        from .contracts import H3_MAX_DURATION_SECONDS, H3_MIN_DURATION_SECONDS

        if not (H3_MIN_DURATION_SECONDS <= value <= H3_MAX_DURATION_SECONDS):
            return (
                f"Invalid duration_seconds: {value}. "
                f"Allowed: [{H3_MIN_DURATION_SECONDS}, {H3_MAX_DURATION_SECONDS}]"
            )
        # 显式尺寸同样在提交期拦截；动态连接无法静态取值时留给执行期合同校验。
        if width is None or height is None:
            return True
        try:
            resolve_t2va_target(
                aspect_ratio=str(kwargs.get("aspect_ratio", "16:9")),
                duration_seconds=value,
                width=width,
                height=height,
            )
        except (TypeError, ValueError) as exc:
            return str(exc)
        return True

    def build(
        self,
        aspect_ratio: str,
        duration_seconds: float,
        width: int = 0,
        height: int = 0,
    ):
        target = resolve_t2va_target(
            aspect_ratio=aspect_ratio,
            duration_seconds=float(duration_seconds),
            width=width,
            height=height,
        )
        info = {
            key: target[key]
            for key in (
                "width",
                "height",
                "frame_count",
                "duration_seconds",
                "fps",
                "video_latent_t",
                "video_latent_h",
                "video_latent_w",
                "audio_latent_t",
            )
        }
        return (target, json.dumps(info, ensure_ascii=False, indent=2))


class MiniMaxH3T2VATextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_text_encoder": (TEXT_ENCODER_TYPE,),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                    },
                ),
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = f"{CATEGORY}/conditioning"

    def encode(self, h3_text_encoder: Mapping[str, Any], prompt: str):
        handle = _unwrap_runtime_handle(
            h3_text_encoder, H3_TEXT_ENCODER_SCHEMA, "h3_text_encoder"
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt 不能为空")
        prompt_embeds = _encode_prompt(handle, prompt)
        return (make_t2va_conditioning(prompt, prompt_embeds),)


class MiniMaxH3UnsupportedConditioning:
    """Visible marker for the next two ports; never falls back to T2VA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task": (
                    [H3_TASK_FL2VA, H3_TASK_REF2VA],
                    {"default": H3_TASK_FL2VA},
                )
            }
        }

    RETURN_TYPES = (CONDITIONING_TYPE,)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "fail"
    CATEGORY = f"{CATEGORY}/conditioning"

    def fail(self, task: str):
        require_t2va(task)
        raise AssertionError("unreachable")


class MiniMaxH3EmptyAVLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"target": (TARGET_TYPE,)}}

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "generate"
    CATEGORY = f"{CATEGORY}/latent"

    def generate(self, target: Mapping[str, Any]):
        t = _require_torch()
        clean_target = validate_target(target)
        # Shape carrier only.  The sampler replaces these zeros with the two
        # independently-seeded source-compatible noise streams.
        video = t.zeros(
            1,
            24,
            int(clean_target["video_latent_t"]),
            int(clean_target["video_latent_h"]),
            int(clean_target["video_latent_w"]),
            dtype=t.float32,
            device="cpu",
        )
        audio = t.zeros(
            2,
            32,
            int(clean_target["audio_latent_t"]),
            dtype=t.float32,
            device="cpu",
        )
        return (
            {
                "schema": H3_AV_LATENT_SCHEMA,
                "task": H3_TASK_T2VA,
                "target": clean_target,
                "video": video,
                "audio": audio,
                "sampled": False,
            },
        )


class MiniMaxH3DualSigmaSampler:
    """Dedicated T2VA sampler; standard KSampler cannot express this state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_model": (MODEL_TYPE,),
                "conditioning": (CONDITIONING_TYPE,),
                "av_latent": (AV_LATENT_TYPE,),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": (1 << 63) - 1,
                    },
                ),
                "sigma_points": (
                    "INT",
                    {
                        "default": H3_DEFAULT_SIGMA_POINTS,
                        "min": 2,
                        "max": 1000,
                        "tooltip": (
                            "遵循原仓库语义：50 个 sigma 点产生 49 次 DiT forward。"
                        ),
                    },
                ),
                "video_shift": (
                    "FLOAT",
                    {
                        "default": H3_DEFAULT_VIDEO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "audio_shift": (
                    "FLOAT",
                    {
                        "default": H3_DEFAULT_AUDIO_SHIFT,
                        "min": 0.01,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = (AV_LATENT_TYPE,)
    RETURN_NAMES = ("sampled_av_latent",)
    FUNCTION = "sample"
    CATEGORY = f"{CATEGORY}/sampling"

    def sample(
        self,
        h3_model: Mapping[str, Any],
        conditioning: Mapping[str, Any],
        av_latent: Mapping[str, Any],
        seed: int,
        sigma_points: int,
        video_shift: float,
        audio_shift: float,
    ):
        clean_latent = validate_av_latent(av_latent)
        packed = _build_t2va_packed(
            conditioning["prompt_embeds"],
            clean_latent["target"],
        )
        progress_bar = (
            ProgressBar(max(1, int(sigma_points) - 1))
            if ProgressBar is not None
            else None
        )

        def update_progress(current: int, total: int):
            if progress_bar is not None:
                progress_bar.update_absolute(current, total)

        def check_cancelled():
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()

        with _transformer_session(h3_model) as transformer:
            output = sample_t2va(
                transformer=transformer,
                conditioning=conditioning,
                av_latent=clean_latent,
                packed=packed,
                seed=int(seed),
                sigma_points=int(sigma_points),
                video_shift=float(video_shift),
                audio_shift=float(audio_shift),
                progress=update_progress,
                check_cancelled=check_cancelled,
            )
        return (output,)


class MiniMaxH3DecodeAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "h3_vae_bundle": (VAE_BUNDLE_TYPE,),
                "sampled_av_latent": (AV_LATENT_TYPE,),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "decode"
    CATEGORY = f"{CATEGORY}/decode"

    def decode(
        self,
        h3_vae_bundle: Mapping[str, Any],
        sampled_av_latent: Mapping[str, Any],
    ):
        if (
            not isinstance(h3_vae_bundle, Mapping)
            or h3_vae_bundle.get("schema") != H3_VAE_SCHEMA
        ):
            raise TypeError(
                "h3_vae_bundle 端口不是 MiniMax H3 Direct VAE Loader 的输出"
            )
        latent = validate_av_latent(sampled_av_latent)
        if not latent.get("sampled", False):
            raise ValueError(
                "输入仍是空 latent；请先连接 MiniMax H3 Dual Sigma Sampler"
            )
        bundle = h3_vae_bundle.get("bundle")
        video_vae = _wrapper_value(bundle, "video_vae")
        audio_vae = _wrapper_value(bundle, "audio_vae")
        if video_vae is None or audio_vae is None:
            raise RuntimeError(
                "VAE bundle 必须同时包含 video_vae 和 audio_vae"
            )
        _require_comfy()
        # Decode components sequentially.  Keeping either the 62 GB DiT or
        # both VAEs resident defeats ComfyUI's memory manager on real H3
        # hardware.
        model_management.unload_all_models()
        load_device = model_management.get_torch_device()
        video_vae.to(load_device)
        try:
            frames = _decode_video(
                video_vae, latent["video"], latent["target"]
            )
        finally:
            video_vae.offload()
            model_management.soft_empty_cache()

        audio_vae.to(load_device)
        try:
            audio = _decode_audio(
                audio_vae,
                latent["audio"],
            )
        finally:
            audio_vae.offload()
            model_management.soft_empty_cache()
        return (frames, audio)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DirectModelLoader": MiniMaxH3DirectModelLoader,
    "MiniMaxH3DirectTextEncoderLoader": MiniMaxH3DirectTextEncoderLoader,
    "MiniMaxH3DirectVAELoader": MiniMaxH3DirectVAELoader,
    "MiniMaxH3T2VATarget": MiniMaxH3T2VATarget,
    "MiniMaxH3T2VATextEncode": MiniMaxH3T2VATextEncode,
    "MiniMaxH3UnsupportedConditioning": MiniMaxH3UnsupportedConditioning,
    "MiniMaxH3EmptyAVLatent": MiniMaxH3EmptyAVLatent,
    "MiniMaxH3DualSigmaSampler": MiniMaxH3DualSigmaSampler,
    "MiniMaxH3DecodeAV": MiniMaxH3DecodeAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectModelLoader": "MiniMax H3 Model Loader (Direct)",
    "MiniMaxH3DirectTextEncoderLoader": "MiniMax H3 Qwen3-VL Loader (Direct)",
    "MiniMaxH3DirectVAELoader": "MiniMax H3 Dual VAE Loader (Direct)",
    "MiniMaxH3T2VATarget": "MiniMax H3 T2VA Target",
    "MiniMaxH3T2VATextEncode": "MiniMax H3 T2VA Text Encode",
    "MiniMaxH3UnsupportedConditioning": "MiniMax H3 FL/Ref (Not Implemented)",
    "MiniMaxH3EmptyAVLatent": "MiniMax H3 Empty AV Latent",
    "MiniMaxH3DualSigmaSampler": "MiniMax H3 Dual Sigma Sampler",
    "MiniMaxH3DecodeAV": "MiniMax H3 Decode Video + Audio",
}


__all__ = [
    "AV_LATENT_TYPE",
    "CONDITIONING_TYPE",
    "MODEL_TYPE",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "TARGET_TYPE",
    "TEXT_ENCODER_TYPE",
    "VAE_BUNDLE_TYPE",
]
