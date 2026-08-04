"""Loader nodes."""
from __future__ import annotations
from ._shared import *  # noqa: F403


def _validate_dual_vae_inputs(video_vae_path, audio_vae_path):
    for value, label in (
        (video_vae_path, _COMPONENT_KIND_LABELS[VIDEO_VAE_DIRNAME]),
        (audio_vae_path, _COMPONENT_KIND_LABELS[AUDIO_VAE_DIRNAME]),
    ):
        if value is None:
            continue
        check = _validate_explicit_component_input(value, label)
        if check is not True:
            return check
    return True


def _load_dual_vae(
    model_root: str,
    video_vae_path: str,
    audio_vae_path: str,
    *,
    task: str,
    schema: str,
) -> dict:
    """Select and load the 24-channel video and 32-channel audio VAEs together.

    Legacy release weights stay FP32. Native Comfy single files preserve their
    declared storage dtype (currently FP16 video and FP32 audio), while the
    video adapter independently applies FP16 autocast to decode operations.
    """

    is_v2 = schema == H3_VAE_SCHEMA_V2
    normalized_task = normalize_task(task)
    video_selector = video_vae_path or _default_video_vae_model_name()
    audio_selector = audio_vae_path or _default_audio_vae_model_name()
    partition_hint = partition_for_task(normalized_task)
    native_video = _native_component_selection(
        video_selector, VIDEO_VAE_DIRNAME, partition_hint
    )
    native_audio = _native_component_selection(
        audio_selector, AUDIO_VAE_DIRNAME, partition_hint
    )
    if (native_video is None) != (native_audio is None):
        raise ValueError(
            "Select both MiniMax-H3 VAEs from the standard Comfy VAE folder, "
            "or select both from the same RunningHub release; mixed VAE layouts "
            "cannot prove a shared release identity."
        )
    if native_video is not None and native_audio is not None:
        root, selected_video, video_weights, release_info, sigma_scales = native_video
        audio_root, selected_audio, audio_weights, _audio_release, _audio_scales = native_audio
        if audio_root != root:
            raise ValueError("Native MiniMax-H3 video/audio VAE descriptors do not share a root")
        partition = partition_hint
    else:
        # Partition admission needs only one component as a hint; validate the
        # audio side separately after resolving the root.
        video_component = _selector_to_component_dirname(
            video_selector,
            VIDEO_VAE_DIRNAME,
            partition_hint,
            model_root=model_root,
        )
        if is_v2:
            root, partition, release_info, sigma_scales = _resolve_release(
                model_root,
                task=normalized_task,
                required_component=video_component,
                required_files=("config.json",),
            )
        else:
            root, release_info, sigma_scales = _resolve_t2va_release(
                model_root,
                required_component=video_component,
                required_files=("config.json",),
            )
            partition = H3_T2VA_PARTITION
        selected_video, video_weights = _resolve_selected_vae_component(
            root, video_selector, kind=VIDEO_VAE_DIRNAME, partition=partition
        )
        selected_audio, audio_weights = _resolve_selected_vae_component(
            root, audio_selector, kind=AUDIO_VAE_DIRNAME, partition=partition
        )
    loader = _find_callable(
        _runtime_module("vae_adapter"),
        ("load_h3_vae_bundle",),
        "dual VAE",
    )
    # Both component paths are correctness-critical and cannot use the optional
    # _call_supported path: silently dropping one would fall back to the other
    # weight under model_root.
    bundle = loader(
        model_root=str(root),
        video_vae_path=str(selected_video),
        audio_vae_path=str(selected_audio),
        video_vae_weights=_optional_path(video_weights),
        audio_vae_weights=_optional_path(audio_weights),
        device="cpu",
        video_compute_dtype="float16",
        audio_compute_dtype="float32",
    )
    if bundle is None:
        raise RuntimeError("runtime.vae_adapter returned None")
    wrapper = {
        "schema": schema,
        "bundle": bundle,
        "model_root": str(root),
        # vae_path remains the primary path recognized by contracts (video side);
        # audio and both single-file weights are folded into the same component
        # fingerprint as related paths.
        "vae_path": str(selected_video),
        "video_vae_path": str(selected_video),
        "audio_vae_path": str(selected_audio),
        **_weights_field("video_vae_weights_path", video_weights),
        **_weights_field("audio_vae_weights_path", audio_weights),
        "weight_dtype": (
            "video=float16,audio=float32"
            if native_video is not None
            else "float32"
        ),
        "video_decode_compute": "float16_autocast",
        "partition": partition,
        "task": normalized_task,
        "release_metadata": release_info,
        "sigma_shift_scales": sigma_scales,
    }
    if is_v2:
        related_paths = {"audio_vae": selected_audio}
        if video_weights is not None:
            related_paths["video_vae_weights"] = video_weights
        if audio_weights is not None:
            related_paths["audio_vae_weights"] = audio_weights
        release_fingerprint = _release_fingerprint(root, partition, release_info)
        component_fingerprint = _component_fingerprint(
            release_fingerprint,
            "vae",
            selected_video,
            related_paths=related_paths,
        )
        wrapper.update(
            {
                "tasks": _supported_tasks_for_release(release_info, partition),
                "release_fingerprint": release_fingerprint,
                "component_fingerprint": component_fingerprint,
                "vae_fingerprint": component_fingerprint,
            }
        )
    return wrapper


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
                            "auto/bfloat16 is recommended; the runtime preserves H3's designated fp32 layers."
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
        transformer_path=None,
        **kwargs,
    ):
        if transformer_path is None:
            transformer_path = _default_dit_model_name(H3_T2VA_PARTITION)
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(transformer_path, "DiT")

    def load(
        self,
        model_root: str,
        dtype: str,
        transformer_path: str = "",
    ):
        selector = transformer_path or _default_dit_model_name(H3_T2VA_PARTITION)
        native = _native_component_selection(
            selector, "transformer", H3_T2VA_PARTITION
        )
        if native is not None:
            root, selected_transformer, transformer_weights, release_info, sigma_scales = native
        else:
            component_dir = _selector_to_component_dirname(
                selector, "transformer", H3_T2VA_PARTITION, model_root=model_root
            )
            root, release_info, sigma_scales = _resolve_t2va_release(
                model_root,
                required_component=component_dir,
                required_files=("config.json",),
            )
            selected_transformer, transformer_weights = _resolve_selected_component(
                root,
                selector,
                keys=("transformer", "dit"),
                label="DiT",
                partition=H3_T2VA_PARTITION,
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
        if transformer_weights is not None:
            _require_parameter(loader, "transformer_weights", "DiT")
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
            transformer_weights=_optional_path(transformer_weights),
        )
        if handle is None:
            raise RuntimeError("runtime.model_loader returned None")
        return (
            {
                "schema": H3_MODEL_SCHEMA,
                "handle": handle,
                "model_root": str(root),
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "dtype": dtype,
                "transformer_path": str(selected_transformer),
                **_weights_field("transformer_weights_path", transformer_weights),
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
                "text_encoder_path": _component_dir_input("text_encoder", "text encoder"),
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
        text_encoder_path=None,
        **kwargs,
    ):
        if text_encoder_path is None:
            text_encoder_path = _default_te_model_name()
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
        text_encoder_path: str = "",
    ):
        selector = text_encoder_path or _default_te_model_name()
        native = _native_component_selection(
            selector, "text_encoder", H3_T2VA_PARTITION
        )
        if native is not None:
            root, selected_text_encoder, text_encoder_weights, release_info, sigma_scales = native
        else:
            component_dir = _selector_to_component_dirname(
                selector, "text_encoder", H3_T2VA_PARTITION, model_root=model_root
            )
            root, release_info, sigma_scales = _resolve_t2va_release(
                model_root,
                required_component=component_dir,
                required_files=("config.json",),
            )
            selected_text_encoder, text_encoder_weights = _resolve_selected_component(
                root,
                selector,
                keys=("text_encoder", "qwen3vl", "qwen"),
                label="Text Encoder",
                partition=H3_T2VA_PARTITION,
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
        if text_encoder_weights is not None:
            _require_parameter(loader, "text_encoder_weights", "Qwen3-VL")
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
            text_encoder_weights=_optional_path(text_encoder_weights),
        )
        if handle is None:
            raise RuntimeError("runtime.qwen_encoder returned None")
        return (
            {
                "schema": H3_TEXT_ENCODER_SCHEMA,
                "handle": handle,
                "model_root": str(root),
                "dtype": dtype,
                "text_encoder_path": str(selected_text_encoder),
                **_weights_field("text_encoder_weights_path", text_encoder_weights),
                "partition": H3_T2VA_PARTITION,
                "task": H3_TASK_T2VA,
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
            },
        )


class MiniMaxH3DirectVAELoader:
    """Load the native 24-channel video and 32-channel audio VAEs.

    Native single files preserve their declared storage dtype. Legacy release
    weights remain FP32; the video adapter independently applies FP16 autocast
    to decode operations where the source runtime does.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": _model_root_input(),
                "video_vae_path": _video_vae_input(),
                "audio_vae_path": _audio_vae_input(),
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
        video_vae_path=None,
        audio_vae_path=None,
        **kwargs,
    ):
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_dual_vae_inputs(video_vae_path, audio_vae_path)

    def load(
        self,
        model_root: str,
        video_vae_path: str = "",
        audio_vae_path: str = "",
    ):
        return (
            _load_dual_vae(
                model_root,
                video_vae_path,
                audio_vae_path,
                task=H3_TASK_T2VA,
                schema=H3_VAE_SCHEMA,
            ),
        )


def _supported_tasks_for_release(
    metadata: Mapping[str, Any],
    partition: str,
) -> tuple[str, ...]:
    declared = metadata.get("tasks")
    if isinstance(declared, list) and declared and all(
        isinstance(item, str) and item for item in declared
    ):
        return tuple(str(item).strip().lower() for item in declared)
    return (
        (H3_TASK_T2VA, H3_TASK_FL2VA)
        if partition == H3_T2VA_PARTITION
        else (H3_TASK_REF2VA,)
    )


class _MiniMaxH3ExplicitModelLoader:
    """Partition-explicit v2 DiT loader shared by FL2VA and Ref2VA."""

    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        partition = partition_for_task(cls.TASK)
        return {
            "required": {
                "model_root": _model_root_input(partition),
                "dtype": (
                    ["auto", "bfloat16", "float16"],
                    {
                        "default": "auto",
                        "tooltip": "auto/bfloat16 is recommended; the runtime preserves H3's designated fp32 layers.",
                    },
                ),
                "transformer_path": _component_dir_input(
                    "transformer", "DiT", partition
                ),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("h3_model",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        transformer_path=None,
        **kwargs,
    ):
        if transformer_path is None:
            transformer_path = _default_dit_model_name(partition_for_task(cls.TASK))
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_explicit_component_input(transformer_path, "DiT")

    def load(
        self,
        model_root: str,
        dtype: str,
        transformer_path: str = "",
    ):
        task = normalize_task(self.TASK)
        partition_hint = partition_for_task(task)
        selector = transformer_path or _default_dit_model_name(partition_hint)
        native = _native_component_selection(selector, "transformer", partition_hint)
        if native is not None:
            root, selected_transformer, transformer_weights, release_info, sigma_scales = native
            partition = partition_hint
        else:
            component_dir = _selector_to_component_dirname(
                selector, "transformer", partition_hint, model_root=model_root
            )
            root, partition, release_info, sigma_scales = _resolve_release(
                model_root,
                task=task,
                required_component=component_dir,
                required_files=("config.json",),
            )
            selected_transformer, transformer_weights = _resolve_selected_component(
                root,
                selector,
                keys=("transformer", "dit"),
                label="DiT",
                partition=partition,
                required_files=("config.json",),
            )
        loader = _find_callable(
            _runtime_module("model_loader"),
            ("load_h3_model",),
            "DiT",
        )
        # task/partition/explicit path are critical.  A runtime without this
        # exact contract must fail instead of silently loading FL weights.
        handle = loader(
            model_root=str(root),
            partition=partition,
            task=task,
            transformer_path=str(selected_transformer),
            transformer_weights=_optional_path(transformer_weights),
            dtype=_runtime_dtype_name(dtype, allow_float32=False),
            device="auto",
            offload_device="auto",
        )
        if handle is None:
            raise RuntimeError("runtime.model_loader returned None")
        release_fingerprint = _release_fingerprint(root, partition, release_info)
        component_fingerprint = _component_fingerprint(
            release_fingerprint,
            "transformer",
            selected_transformer,
            related_paths=(
                {"transformer_weights": transformer_weights}
                if transformer_weights is not None
                else None
            ),
        )
        return (
            {
                "schema": H3_MODEL_SCHEMA_V2,
                "handle": handle,
                "model_root": str(root),
                "partition": partition,
                "task": task,
                "tasks": _supported_tasks_for_release(release_info, partition),
                "dtype": dtype,
                "transformer_path": str(selected_transformer),
                **_weights_field("transformer_weights_path", transformer_weights),
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
                "release_fingerprint": release_fingerprint,
                "component_fingerprint": component_fingerprint,
                "transformer_fingerprint": component_fingerprint,
            },
        )


class MiniMaxH3FL2VAModelLoader(_MiniMaxH3ExplicitModelLoader):
    TASK = H3_TASK_FL2VA


class MiniMaxH3Ref2VAModelLoader(_MiniMaxH3ExplicitModelLoader):
    TASK = H3_TASK_REF2VA


class _MiniMaxH3ExplicitTextEncoderLoader:
    """Partition-explicit v2 Qwen3-VL loader."""

    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        partition = partition_for_task(cls.TASK)
        return {
            "required": {
                "model_root": _model_root_input(partition),
                "dtype": (
                    ["auto", "bfloat16", "float16", "float32"],
                    {"default": "auto"},
                ),
                "text_encoder_path": _component_dir_input(
                    "text_encoder", "text encoder", partition
                ),
            }
        }

    RETURN_TYPES = (TEXT_ENCODER_TYPE,)
    RETURN_NAMES = ("h3_text_encoder",)
    FUNCTION = "load"
    CATEGORY = f"{CATEGORY}/loaders"

    @classmethod
    def VALIDATE_INPUTS(
        cls,
        model_root="MiniMax-H3",
        text_encoder_path=None,
        **kwargs,
    ):
        if text_encoder_path is None:
            text_encoder_path = _default_te_model_name()
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
        text_encoder_path: str = "",
    ):
        task = normalize_task(self.TASK)
        partition_hint = partition_for_task(task)
        selector = text_encoder_path or _default_te_model_name()
        native = _native_component_selection(
            selector, "text_encoder", partition_hint
        )
        if native is not None:
            root, selected_text_encoder, text_encoder_weights, release_info, sigma_scales = native
            partition = partition_hint
        else:
            component_dir = _selector_to_component_dirname(
                selector, "text_encoder", partition_hint, model_root=model_root
            )
            root, partition, release_info, sigma_scales = _resolve_release(
                model_root,
                task=task,
                required_component=component_dir,
                required_files=("config.json",),
            )
            selected_text_encoder, text_encoder_weights = _resolve_selected_component(
                root,
                selector,
                keys=("text_encoder", "qwen3vl", "qwen"),
                label="Text Encoder",
                partition=partition,
                required_files=("config.json",),
            )
        loader = _find_callable(
            _runtime_module("qwen_encoder"),
            ("load_h3_text_encoder",),
            "Qwen3-VL",
        )
        handle = loader(
            model_root=str(root),
            partition=partition,
            require_multimodal_processor=True,
            text_encoder_path=str(selected_text_encoder),
            text_encoder_weights=_optional_path(text_encoder_weights),
            dtype=_runtime_dtype_name(dtype),
            device="auto",
            offload_device="cpu",
        )
        if handle is None:
            raise RuntimeError("runtime.qwen_encoder returned None")
        if native is not None:
            # Native Comfy bundles tokenizer/processor assets with its own H3
            # text encoder implementation.  Keep the wrapper's logical paths
            # inside the native descriptor root; the handle owns the real
            # packaged assets.
            tokenizer_component = selected_text_encoder.resolve()
            processor_component = selected_text_encoder.resolve()
        else:
            tokenizer_component = Path(
                getattr(handle, "tokenizer_component_path", selected_text_encoder)
            ).resolve()
            processor_value = getattr(handle, "processor_component_path", None)
            processor_component = (
                Path(processor_value).resolve()
                if processor_value is not None
                else None
            )
        related_paths = {"tokenizer": tokenizer_component}
        if processor_component is not None:
            related_paths["processor"] = processor_component
        if text_encoder_weights is not None:
            related_paths["text_encoder_weights"] = text_encoder_weights
        release_fingerprint = _release_fingerprint(root, partition, release_info)
        component_fingerprint = _component_fingerprint(
            release_fingerprint,
            "text_encoder",
            selected_text_encoder,
            related_paths=related_paths,
        )
        return (
            {
                "schema": H3_TEXT_ENCODER_SCHEMA_V2,
                "handle": handle,
                "model_root": str(root),
                "partition": partition,
                "task": task,
                "tasks": _supported_tasks_for_release(release_info, partition),
                "dtype": dtype,
                "text_encoder_path": str(selected_text_encoder),
                **_weights_field("text_encoder_weights_path", text_encoder_weights),
                "tokenizer_path": str(tokenizer_component),
                **(
                    {"processor_path": str(processor_component)}
                    if processor_component is not None
                    else {}
                ),
                "release_metadata": release_info,
                "sigma_shift_scales": sigma_scales,
                "release_fingerprint": release_fingerprint,
                "component_fingerprint": component_fingerprint,
                "text_encoder_fingerprint": component_fingerprint,
            },
        )


class MiniMaxH3FL2VATextEncoderLoader(_MiniMaxH3ExplicitTextEncoderLoader):
    TASK = H3_TASK_FL2VA


class MiniMaxH3Ref2VATextEncoderLoader(_MiniMaxH3ExplicitTextEncoderLoader):
    TASK = H3_TASK_REF2VA


class _MiniMaxH3ExplicitVAELoader:
    """Partition-explicit v2 dual VAE loader."""

    TASK = ""

    @classmethod
    def INPUT_TYPES(cls):
        partition = partition_for_task(cls.TASK)
        return {
            "required": {
                "model_root": _model_root_input(partition),
                "video_vae_path": _video_vae_input(partition),
                "audio_vae_path": _audio_vae_input(partition),
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
        video_vae_path=None,
        audio_vae_path=None,
        **kwargs,
    ):
        root_check = _validate_model_root_input(model_root=model_root, **kwargs)
        if root_check is not True:
            return root_check
        return _validate_dual_vae_inputs(video_vae_path, audio_vae_path)

    def load(
        self,
        model_root: str,
        video_vae_path: str = "",
        audio_vae_path: str = "",
    ):
        return (
            _load_dual_vae(
                model_root,
                video_vae_path,
                audio_vae_path,
                task=normalize_task(self.TASK),
                schema=H3_VAE_SCHEMA_V2,
            ),
        )


class MiniMaxH3FL2VAVAELoader(_MiniMaxH3ExplicitVAELoader):
    TASK = H3_TASK_FL2VA


class MiniMaxH3Ref2VAVAELoader(_MiniMaxH3ExplicitVAELoader):
    TASK = H3_TASK_REF2VA


