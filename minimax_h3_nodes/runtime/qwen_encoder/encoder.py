from __future__ import annotations
from .helpers import *  # noqa: F403
from .loading import (  # noqa: F401
    _component_is_quantized, _load_quantized_text_encoder, _load_qwen_bf16_checkpoint,
    _qwen_causal_lm_class, _validate_qwen_config, _validate_loading_info,
    _validate_text_encoder_quant_marker, _validate_text_encoder_quant_metadata,
)

class MiniMaxH3TextEncoder:
    """Resident/offload-aware wrapper around the trimmed Qwen3-VL backbone."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        processor: Any | None = None,
        component_path: Path,
        load_device: Any,
        offload_device: Any,
        quantized: bool = False,
        tokenizer_component_path: Path | None = None,
        processor_component_path: Path | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.component_path = component_path
        self.load_device = load_device
        self.offload_device = offload_device
        self.quantized = bool(quantized)
        self.tokenizer_component_path = (
            tokenizer_component_path.resolve()
            if tokenizer_component_path is not None
            else component_path.resolve()
        )
        self.processor_component_path = (
            processor_component_path.resolve()
            if processor_component_path is not None
            else None
        )
        self._lock = threading.RLock()
        self._compute_device = None
        self._inference_active = False
        self._linear_bank = None
        self._linear_patcher = None
        self._streaming_linears: tuple[Any, ...] = ()
        self._linear_storage_bytes = 0
        self._static_storage_bytes = 0

    def _movable(self):  # Direct submodules used by text encoding (optionally skip the visual tower).
        return [m for n, m in self.model.named_children() if not (TE_VISUAL_ON_CPU and n == "visual")]

    def _actual_device(self):
        for m in self._movable():
            for p in m.parameters():
                return p.device
            for b in m.buffers():
                return b.device
        return next(self.model.parameters()).device

    @property
    def device(self):
        # Partial load intentionally leaves many Linear weights on CPU.  The
        # execution device must follow activations, never the first parameter.
        return self._compute_device or self._actual_device()

    def _visual_module(self):
        visual = getattr(self.model, "visual", None)
        if visual is None:
            raise H3ComponentError(
                "Loaded Qwen3-VL backbone has no visual tower; multimodal "
                "FL2VA/Ref2VA encoding is unavailable"
            )
        return visual

    def _offload_visual_after_staged(
        self, visual: Any, *, preserve_primary_error: bool
    ) -> bool:
        """Return the visual tower to CPU without masking a primary failure."""

        try:
            visual.to(self.offload_device)
        except BaseException:
            if not preserve_primary_error:
                raise
            LOGGER.exception(
                "Failed to offload Qwen visual tower while preserving the "
                "original staged-encode error"
            )
            return False
        return True

    def _require_processor(self, *, video: bool = False):
        processor = self.processor
        if processor is None or not callable(
            getattr(processor, "image_processor", None)
        ):
            # HF image processors are callable objects rather than functions.
            image_processor = getattr(processor, "image_processor", None)
            if image_processor is None or not callable(image_processor):
                raise H3ComponentError(
                    "MiniMax-H3 multimodal Qwen encoding requires the local "
                    "AutoProcessor component with an image_processor"
                )
        if video:
            video_processor = getattr(processor, "video_processor", None)
            if video_processor is None or not callable(video_processor):
                raise H3ComponentError(
                    "MiniMax-H3 Ref2VA video encoding requires the local "
                    "AutoProcessor component with a video_processor"
                )
        return processor

    def _token_id(self, name: str) -> int:
        config = getattr(self.model, "config", None)
        value = getattr(config, name, None)
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise H3ComponentError(
                f"Qwen3-VL config is missing required {name}"
            ) from exc
        if token_id < 0:
            raise H3ComponentError(f"Qwen3-VL config has invalid {name}={token_id}")
        return token_id

    @staticmethod
    def _output_hidden(output: Any):
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, dict):
            hidden = output.get("last_hidden_state")
        if hidden is None:
            raise H3ComponentError(
                "Qwen3-VL language_model did not return last_hidden_state"
            )
        return hidden

    @staticmethod
    def _call_vision_feature_getter(getter, pixel_values, grid_thw):
        """Call get_*_features across Transformers APIs with/without return_dict."""

        import inspect

        kwargs = {}
        try:
            if "return_dict" in inspect.signature(getter).parameters:
                kwargs["return_dict"] = True
        except (TypeError, ValueError):
            pass
        return getter(pixel_values, grid_thw, **kwargs)

    @staticmethod
    def _call_rope_index(
        rope,
        input_ids,
        *,
        mm_token_type_ids,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
    ):
        """Call Qwen3-VL M-RoPE across legacy and current Transformers APIs."""

        import inspect

        kwargs = {
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask,
        }
        try:
            parameters = inspect.signature(rope).parameters
        except (TypeError, ValueError):
            parameters = None

        accepts_mm_types = parameters is None or "mm_token_type_ids" in parameters
        if parameters is not None and not accepts_mm_types:
            accepts_mm_types = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
        if accepts_mm_types:
            kwargs["mm_token_type_ids"] = mm_token_type_ids

        try:
            return rope(input_ids, **kwargs)
        except TypeError as exc:
            # Some extension callables do not expose a usable signature.  In
            # that case, retry only the argument-binding failure from the
            # pre-mm_token_type_ids API; do not mask TypeErrors from the body.
            if parameters is not None or not re.search(
                r"unexpected keyword argument ['\"]mm_token_type_ids['\"]",
                str(exc),
            ):
                raise
            kwargs.pop("mm_token_type_ids", None)
            return rope(input_ids, **kwargs)

    @staticmethod
    def _vision_output_to_cpu(output: Any, *, context: str) -> dict[str, Any]:
        """Retain every DeepStack tensor needed by the later LM phase."""

        import torch

        pooled = getattr(output, "pooler_output", None)
        deepstack = getattr(output, "deepstack_features", None)
        # transformers>=4.57 get_image_features returns (embeds, deepstack_list)
        if (pooled is None or deepstack is None) and isinstance(
            output, (tuple, list)
        ) and len(output) == 2:
            pooled, deepstack = output
        if pooled is None or deepstack is None:
            raise H3ComponentError(
                f"Qwen3-VL {context} visual output lacks pooled embeds or "
                "DeepStack features; refusing an incorrect pooled-only encode"
            )
        if isinstance(pooled, torch.Tensor):
            pooled_items = [pooled]
        elif isinstance(pooled, (tuple, list)) and pooled:
            pooled_items = list(pooled)
        else:
            raise H3ComponentError(
                f"Qwen3-VL {context} pooler_output has unsupported type "
                f"{type(pooled).__name__}"
            )
        if not isinstance(deepstack, (tuple, list)) or not deepstack:
            raise H3ComponentError(
                f"Qwen3-VL {context} returned no DeepStack features"
            )
        if not all(isinstance(item, torch.Tensor) for item in pooled_items):
            raise H3ComponentError(
                f"Qwen3-VL {context} returned non-tensor pooled features"
            )
        if not all(isinstance(item, torch.Tensor) for item in deepstack):
            raise H3ComponentError(
                f"Qwen3-VL {context} returned non-tensor DeepStack features"
            )
        pooled_tensor = torch.cat(pooled_items, dim=0).detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        deepstack_tensors = tuple(
            item.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            for item in deepstack
        )
        if pooled_tensor.ndim != 2:
            raise H3ComponentError(
                f"Qwen3-VL {context} pooled feature shape must be [N,D], got "
                f"{tuple(pooled_tensor.shape)}"
            )
        if any(
            item.ndim != 2
            or int(item.shape[0]) != int(pooled_tensor.shape[0])
            or int(item.shape[1]) != int(pooled_tensor.shape[1])
            for item in deepstack_tensors
        ):
            raise H3ComponentError(
                f"Qwen3-VL {context} DeepStack tensors do not align with "
                f"pooled shape {tuple(pooled_tensor.shape)}"
            )
        return {"pooled": pooled_tensor, "deepstack": deepstack_tensors}

    def _set_compute_device(self, device: Any | None) -> None:
        import torch

        self._compute_device = None if device is None else torch.device(device)
        if self._compute_device is None:
            if hasattr(self.model, "_h3_compute_device"):
                delattr(self.model, "_h3_compute_device")
        else:
            object.__setattr__(
                self.model, "_h3_compute_device", self._compute_device
            )

    def _ensure_linear_patcher(self):
        if self._linear_patcher is not None:
            return self._linear_patcher
        roots = self._movable()
        linears = _quantized_language_linears(roots)
        expected = SELECTED_LAYERS * 7
        if len(linears) != expected:
            raise H3ComponentError(
                "H3 INT8 text streaming requires exactly "
                f"{expected} complete quantized Linear modules, got {len(linears)}"
            )
        bank, patcher, linear_bytes = _make_quantized_linear_patcher(
            linears,
            load_device=self.load_device,
            offload_device=self.offload_device,
        )
        self._linear_bank = bank
        self._linear_patcher = patcher
        self._streaming_linears = tuple(linears)
        excluded = {id(module) for module in linears}
        self._linear_storage_bytes = linear_bytes
        self._static_storage_bytes = _direct_tensor_bytes(roots, excluded)
        LOGGER.info(
            "H3 text_encoder physical storage %.2fGB "
            "(INT8 Linear %.2fGB + static %.2fGB); visual stays on CPU",
            (linear_bytes + self._static_storage_bytes) / 2**30,
            linear_bytes / 2**30,
            self._static_storage_bytes / 2**30,
        )
        return patcher

    def _move_static_tensors(self, device: Any) -> None:
        excluded = {id(module) for module in self._streaming_linears}
        _move_direct_tensors(self._movable(), excluded, device)

    def _unload_linear_patcher(self) -> None:
        patcher = self._linear_patcher
        if patcher is None:
            return
        try:
            import comfy.model_management as mm

            unload = getattr(mm, "unload_model_and_clones", None)
            if callable(unload):
                unload(patcher)
            else:
                patcher.detach()
        finally:
            # If load_models_gpu failed before registering the LoadedModel, the
            # global unload helper cannot see it.  Detach any remaining load.
            loaded_size = getattr(patcher, "loaded_size", lambda: 0)()
            bank = getattr(patcher, "model", None)
            if (
                loaded_size
                or bool(getattr(bank, "model_lowvram", False))
                or bool(getattr(patcher, "pinned", ()))
            ):
                patcher.detach()

    def _rollback_failed_quantized_load(self) -> bool:
        """Best-effort, repeatable rollback for a partial INT8 acquisition."""

        clean = True
        try:
            self._unload_linear_patcher()
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to detach Qwen INT8 Linear patcher during load rollback"
            )
        try:
            self._move_static_tensors(self.offload_device)
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to return Qwen static tensors to the offload device "
                "during load rollback"
            )
        try:
            self._set_compute_device(None)
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to clear Qwen compute-device marker during load rollback"
            )
            # Keep the wrapper state conservative even if a foreign model
            # object rejects marker removal.  Repeating the rollback remains
            # safe and will retry all physical cleanup actions.
            self._compute_device = None
        self._inference_active = False
        return clean

    def _rollback_failed_full_load(self, modules: Sequence[Any]) -> bool:
        """Best-effort rollback after a BF16/FP16 module move fails."""

        clean = True
        for module in modules:
            try:
                module.to(self.offload_device)
            except BaseException:
                clean = False
                LOGGER.exception(
                    "Failed to return a Qwen module to the offload device "
                    "during full-load rollback"
                )
        try:
            self._set_compute_device(None)
        except BaseException:
            clean = False
            LOGGER.exception(
                "Failed to clear Qwen compute-device marker during "
                "full-load rollback"
            )
            self._compute_device = None
        self._inference_active = False
        return clean

    def load_for_inference(
        self, *, extra_headroom: int = 0
    ) -> "MiniMaxH3TextEncoder":
        import torch

        extra_headroom = max(0, int(extra_headroom))

        if self.load_device.type != "cuda":
            return self
        if self._inference_active and self.device == self.load_device:
            return self

        if self.quantized and ALLOW_PARTIAL_OFFLOAD_INT8:
            patcher = self._ensure_linear_patcher()
            reserve = (
                TE_GPU_HEADROOM
                + self._static_storage_bytes
                + extra_headroom
            )
            try:
                import comfy.model_management as mm

                # Reserve static embedding/norm/RoPE bytes plus encode
                # workspace.  ModelPatcher uses the remainder for resident
                # INT8 Linears and streams all others on Comfy's CUDA streams.
                mm.load_models_gpu([patcher], memory_required=reserve)
                self._move_static_tensors(self.load_device)
            except BaseException as exc:
                rollback_clean = self._rollback_failed_quantized_load()
                if isinstance(exc, torch.OutOfMemoryError) and rollback_clean:
                    LOGGER.warning(
                        "text_encoder INT8 partial load OOM; cleaned up and "
                        "falling back to CPU encode"
                    )
                    try:
                        mm.soft_empty_cache()
                    except BaseException:
                        try:
                            torch.cuda.empty_cache()
                        except BaseException:
                            LOGGER.exception(
                                "Failed to empty CUDA cache after Qwen INT8 OOM"
                            )
                    return self
                # Non-OOM errors must retain their original type, instance and
                # traceback.  An OOM whose rollback was incomplete is also
                # re-raised rather than running a mixed-device CPU fallback.
                raise
            self._set_compute_device(self.load_device)
            self._inference_active = True
            loaded = int(getattr(patcher, "loaded_size", lambda: 0)())
            LOGGER.info(
                "H3 text_encoder CUDA streaming ready: static %.2fGB, "
                "Linear resident %.2f/%.2fGB, workspace reserve %.2fGB, "
                "multimodal reserve %.2fGB",
                self._static_storage_bytes / 2**30,
                loaded / 2**30,
                self._linear_storage_bytes / 2**30,
                TE_GPU_HEADROOM / 2**30,
                extra_headroom / 2**30,
            )
            return self

        if self._actual_device() == self.load_device:
            self._set_compute_device(self.load_device)
            self._inference_active = True
            return self
        mods = self._movable()
        need = sum(_physical_module_bytes(module) for module in mods)
        free = torch.cuda.mem_get_info(self.load_device)[0] if torch.cuda.is_available() else 0
        if free < need + TE_GPU_HEADROOM + extra_headroom:
            LOGGER.warning("text_encoder needs %.1fGB + %.1fGB workspace > %.1fGB free; falling back to CPU encode",
                           need / 2**30, TE_GPU_HEADROOM / 2**30, free / 2**30)
            return self
        try:
            for m in mods:
                m.to(self.load_device)
        except BaseException as exc:
            rollback_clean = self._rollback_failed_full_load(mods)
            if isinstance(exc, torch.OutOfMemoryError) and rollback_clean:
                LOGGER.warning("text_encoder ran out of memory on GPU; falling back to CPU encode")
                try:
                    torch.cuda.empty_cache()
                except BaseException:
                    LOGGER.exception(
                        "Failed to empty CUDA cache after Qwen full-load OOM"
                    )
                return self
            raise
        self._set_compute_device(self.load_device)
        self._inference_active = True
        return self

    def offload_after_inference(self) -> None:
        import torch

        with self._lock:
            cleanup_error: BaseException | None = None

            def cleanup(action, message: str) -> None:
                nonlocal cleanup_error
                try:
                    action()
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                    else:
                        LOGGER.exception(message)

            if self._linear_patcher is not None:
                cleanup(
                    self._unload_linear_patcher,
                    "Additional Qwen patcher-offload failure",
                )
                cleanup(
                    lambda: self._move_static_tensors(self.offload_device),
                    "Additional Qwen static-tensor offload failure",
                )
            else:
                try:
                    needs_offload = self._actual_device() != self.offload_device
                except BaseException as exc:
                    cleanup_error = exc
                    needs_offload = True
                if needs_offload:
                    for module in self._movable():
                        cleanup(
                            lambda module=module: module.to(self.offload_device),
                            "Additional Qwen module offload failure",
                        )
            cleanup(
                lambda: self._set_compute_device(None),
                "Additional Qwen compute-marker cleanup failure",
            )
            if hasattr(self.model, "_h3_compute_device"):
                cleanup(
                    lambda: delattr(self.model, "_h3_compute_device"),
                    "Additional Qwen model-marker cleanup failure",
                )
            # Logical residency state is session-scoped and must never survive
            # a failed physical cleanup.  A later idempotent offload call can
            # retry any tensor move which failed above.
            self._compute_device = None
            self._inference_active = False
            try:
                import comfy.model_management as mm

                mm.soft_empty_cache()
            except BaseException:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except BaseException:
                    LOGGER.exception("Failed to empty cache after Qwen offload")
            if cleanup_error is not None:
                raise cleanup_error

    def _encode_visual_features_staged(
        self,
        *,
        pixel_values: Any | None,
        image_grid_thw: Any | None,
        pixel_values_videos: Any | None,
        video_grid_thw: Any | None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Run only the visual tower, retain pooled + DeepStack on CPU.

        The 7GB visual tower and the streamed 32B language model must not be
        resident on a 24GB card at the same time.  Calling the full Qwen model
        cannot provide that guarantee, so this phase explicitly uses
        ``get_image_features``/``get_video_features`` and offloads visual
        before the language phase starts.
        """

        import torch

        if not TE_VISUAL_ON_CPU:
            raise H3ComponentError(
                "Staged multimodal Qwen encoding requires TE_VISUAL_ON_CPU=True; "
                "otherwise load_for_inference could co-reside the visual tower "
                "with the language model"
            )

        # This is deliberately unconditional: a previous direct caller may
        # have left the language patcher resident after encode_prompt().
        self.offload_after_inference()
        visual = self._visual_module()
        visual_device = self.load_device
        if visual_device.type == "cuda":
            try:
                import comfy.model_management as mm

                mm.free_memory(
                    _physical_module_bytes(visual) + TE_GPU_HEADROOM,
                    visual_device,
                )
            except ImportError:
                # Plain-PyTorch environments have no other Comfy-managed
                # models to evict.  The explicit OOM path below remains safe.
                pass
        try:
            visual.to(visual_device)
        except BaseException as exc:
            self._offload_visual_after_staged(
                visual, preserve_primary_error=True
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if isinstance(exc, torch.OutOfMemoryError):
                raise H3ComponentError(
                    "Qwen3-VL visual tower does not fit after language-model "
                    "offload; staged multimodal encoding cannot continue safely"
                ) from exc
            raise

        image_features = None
        video_features = None
        try:
            parameter = next(visual.parameters(), None)
            visual_dtype = (
                parameter.dtype
                if parameter is not None and parameter.is_floating_point()
                else torch.bfloat16
            )
            if pixel_values is not None:
                getter = getattr(self.model, "get_image_features", None)
                if not callable(getter):
                    raise H3ComponentError(
                        "Installed Transformers Qwen3-VL lacks "
                        "get_image_features; refusing an unverified staged path"
                    )
                output = self._call_vision_feature_getter(
                    getter,
                    pixel_values.to(device=visual_device, dtype=visual_dtype),
                    image_grid_thw.to(device=visual_device, dtype=torch.long),
                )
                image_features = self._vision_output_to_cpu(
                    output, context="image"
                )
                del output
            if pixel_values_videos is not None:
                getter = getattr(self.model, "get_video_features", None)
                if not callable(getter):
                    raise H3ComponentError(
                        "Installed Transformers Qwen3-VL lacks "
                        "get_video_features; refusing an unverified staged path"
                    )
                output = self._call_vision_feature_getter(
                    getter,
                    pixel_values_videos.to(
                        device=visual_device, dtype=visual_dtype
                    ),
                    video_grid_thw.to(device=visual_device, dtype=torch.long),
                )
                video_features = self._vision_output_to_cpu(
                    output, context="video"
                )
                del output
        except torch.OutOfMemoryError as exc:
            raise H3ComponentError(
                "Qwen3-VL visual forward exhausted device memory after "
                "staged eviction; reduce reference count/resolution"
            ) from exc
        finally:
            self._offload_visual_after_staged(
                visual, preserve_primary_error=sys.exc_info()[0] is not None
            )
            if torch.cuda.is_available():
                try:
                    import comfy.model_management as mm

                    mm.soft_empty_cache()
                except ImportError:
                    torch.cuda.empty_cache()
        return image_features, video_features

    def _encode_staged_language(
        self,
        input_ids: Any,
        *,
        image_grid_thw: Any | None,
        video_grid_thw: Any | None,
        image_features: dict[str, Any] | None,
        video_features: dict[str, Any] | None,
    ):
        """Inject cached visual features and run the trimmed language model."""

        import torch

        feature_bytes = 0
        for feature_set in (image_features, video_features):
            if feature_set is None:
                continue
            feature_bytes += int(feature_set["pooled"].nbytes)
            feature_bytes += sum(
                int(item.nbytes) for item in feature_set["deepstack"]
            )
        # When image and video coexist, HF builds a modality-ordered joint
        # DeepStack list before the LM forward.  Reserve that second copy too.
        if image_features is not None and video_features is not None:
            feature_bytes *= 2
        self.load_for_inference(extra_headroom=feature_bytes)
        device = self.device
        visual = self._visual_module()
        if device.type == "cuda" and any(
            parameter.device.type == "cuda" for parameter in visual.parameters()
        ):
            raise H3ComponentError(
                "Qwen staged-residency invariant violated: visual tower is "
                "still on CUDA while loading the language model"
            )

        ids = input_ids.to(device=device, dtype=torch.long)[None]
        attention_mask = torch.ones_like(ids, dtype=torch.long)
        image_token_id = self._token_id("image_token_id")
        video_token_id = self._token_id("video_token_id")
        mm_types = minimax_h3_mm_token_type_ids(
            ids,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )

        embedding = getattr(self.model, "get_input_embeddings", None)
        if not callable(embedding):
            raise H3ComponentError(
                "Qwen3-VL backbone lacks get_input_embeddings()"
            )
        inputs_embeds = embedding()(ids)

        image_positions = ids == image_token_id
        video_positions = ids == video_token_id
        image_mask = None
        video_mask = None
        image_deepstack = None
        video_deepstack = None

        if image_features is not None:
            pooled = image_features["pooled"].to(
                device=device, dtype=inputs_embeds.dtype
            )
            if int(image_positions.sum().item()) != int(pooled.shape[0]):
                raise H3ComponentError(
                    "Qwen image placeholder/features mismatch: "
                    f"tokens={int(image_positions.sum().item())}, "
                    f"features={int(pooled.shape[0])}"
                )
            image_mask = image_positions.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, pooled)
            image_deepstack = tuple(
                item.to(device=device, dtype=inputs_embeds.dtype)
                for item in image_features["deepstack"]
            )
        elif bool(image_positions.any().item()):
            raise H3ComponentError(
                "Qwen presentation contains image pads but no image features"
            )

        if video_features is not None:
            pooled = video_features["pooled"].to(
                device=device, dtype=inputs_embeds.dtype
            )
            if int(video_positions.sum().item()) != int(pooled.shape[0]):
                raise H3ComponentError(
                    "Qwen video placeholder/features mismatch: "
                    f"tokens={int(video_positions.sum().item())}, "
                    f"features={int(pooled.shape[0])}"
                )
            video_mask = video_positions.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, pooled)
            video_deepstack = tuple(
                item.to(device=device, dtype=inputs_embeds.dtype)
                for item in video_features["deepstack"]
            )
        elif bool(video_positions.any().item()):
            raise H3ComponentError(
                "Qwen presentation contains video pads but no video features"
            )

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_positions = image_mask[..., 0]
            video_positions = video_mask[..., 0]
            visual_pos_masks = image_positions | video_positions
            if image_deepstack is None or video_deepstack is None or len(
                image_deepstack
            ) != len(video_deepstack):
                raise H3ComponentError(
                    "Qwen image/video DeepStack layer counts do not match"
                )
            image_joint = image_positions[visual_pos_masks]
            video_joint = video_positions[visual_pos_masks]
            combined = []
            for image_item, video_item in zip(
                image_deepstack, video_deepstack
            ):
                joint = image_item.new_zeros(
                    int(visual_pos_masks.sum().item()), image_item.shape[-1]
                )
                joint[image_joint] = image_item
                joint[video_joint] = video_item
                combined.append(joint)
            deepstack_visual_embeds = tuple(combined)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = image_deepstack
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = video_deepstack

        if visual_pos_masks is None or not deepstack_visual_embeds:
            raise H3ComponentError(
                "Multimodal Qwen encode produced no DeepStack injection payload"
            )
        language_model = getattr(self.model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if language_model is None or layers is None:
            raise H3ComponentError(
                "Qwen3-VL backbone lacks language_model.layers"
            )
        if len(deepstack_visual_embeds) > len(layers):
            raise H3ComponentError(
                "Qwen visual DeepStack has more injection layers than the "
                "trimmed language model"
            )

        rope = getattr(self.model, "get_rope_index", None)
        if not callable(rope):
            raise H3ComponentError(
                "Installed Transformers Qwen3-VL lacks get_rope_index(); "
                "multimodal M-RoPE cannot be reproduced safely"
            )
        position_ids, _rope_delta = self._call_rope_index(
            rope,
            ids,
            mm_token_type_ids=mm_types,
            image_grid_thw=(
                None
                if image_grid_thw is None
                else image_grid_thw.to(device=device, dtype=torch.long)
            ),
            video_grid_thw=(
                None
                if video_grid_thw is None
                else video_grid_thw.to(device=device, dtype=torch.long)
            ),
            attention_mask=attention_mask,
        )
        try:
            outputs = language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=list(deepstack_visual_embeds),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )
        except TypeError as exc:
            raise H3ComponentError(
                "Installed Transformers Qwen3-VL language model does not "
                "support verified DeepStack injection; refusing fallback"
            ) from exc
        hidden = self._output_hidden(outputs)[0].to(
            device="cpu", dtype=torch.bfloat16
        )
        expected = (int(ids.shape[1]), HIDDEN_SIZE)
        if tuple(hidden.shape) != expected:
            raise H3ComponentError(
                f"Unexpected Qwen3-VL feature shape {tuple(hidden.shape)}; "
                f"expected {expected}"
            )
        return hidden.contiguous()

    def encode_ids(
        self,
        input_ids: Any,
        *,
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_grid_thw: Any | None = None,
    ):
        """Encode official H3 presentation ids into CPU BF16 layer-50 states.

        Multimodal calls use a strict two-phase DeepStack path.  Text-only
        callers are routed through the existing T2VA implementation.
        """

        import torch

        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1:
            shape = getattr(input_ids, "shape", None)
            raise H3ComponentError(
                f"Qwen input_ids must be a 1-D torch.Tensor, got {shape!r}"
            )
        if int(input_ids.numel()) <= 0:
            raise H3ComponentError("Qwen input_ids must be non-empty")
        if (pixel_values is None) != (image_grid_thw is None):
            raise H3ComponentError(
                "pixel_values and image_grid_thw must be supplied together"
            )
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise H3ComponentError(
                "pixel_values_videos and video_grid_thw must be supplied together"
            )
        has_multimodal = (
            pixel_values is not None or pixel_values_videos is not None
        )
        if not has_multimodal:
            # Keep text-only semantics centralized in encode_prompt.  Decode
            # ids back is not lossless, so run the same backbone directly.
            with self._lock, torch.inference_mode():
                self.load_for_inference()
                with _scoped_cudnn_sdp(self.device.type == "cuda"):
                    ids = input_ids.to(self.device, dtype=torch.long)[None]
                    outputs = self.model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                        use_cache=False,
                    )
                    hidden = self._output_hidden(outputs)[0].to(
                        device="cpu", dtype=torch.bfloat16
                    )
                    expected = (int(ids.shape[1]), HIDDEN_SIZE)
                    if tuple(hidden.shape) != expected:
                        raise H3ComponentError(
                            f"Unexpected Qwen3-VL feature shape "
                            f"{tuple(hidden.shape)}; expected {expected}"
                        )
                    return hidden.contiguous()

        with self._lock, torch.inference_mode():
            with _scoped_cudnn_sdp(self.load_device.type == "cuda"):
                image_features, video_features = (
                    self._encode_visual_features_staged(
                        pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw,
                        pixel_values_videos=pixel_values_videos,
                        video_grid_thw=video_grid_thw,
                    )
                )
                return self._encode_staged_language(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_features=image_features,
                    video_features=video_features,
                )

    def encode_prompt(self, prompt: str):
        """Return H3's positive layer-50 features as CPU ``[L, 5120]`` BF16."""

        import torch
        from ..encode_cache import get_encode_cache, prompt_cache_key
        from ..h3_settings import OPT_ENCODE_CACHE

        if not isinstance(prompt, str) or not prompt:
            raise H3ComponentError("MiniMax-H3 prompt must be a non-empty string")
        te_fp = str(getattr(self, "fingerprint", "") or getattr(self, "component_path", "") or "")
        tok_fp = str(getattr(getattr(self, "tokenizer", None), "name_or_path", "") or "")
        cache_key = prompt_cache_key(prompt=prompt, task="t2va", te_fingerprint=te_fp, tok_fingerprint=tok_fp)
        if OPT_ENCODE_CACHE:
            hit = get_encode_cache().get(cache_key)
            if hit is not None:
                return hit
        with self._lock, torch.inference_mode():
            self.load_for_inference()
            # Match the native H3 encoder recipe.  This is independent of the
            # DiT attention backend and materially reduces Qwen's long-sequence
            # attention workspace on supported CUDA builds.
            with _scoped_cudnn_sdp(self.device.type == "cuda"):
                encoded = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device, dtype=torch.long)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device, dtype=torch.long)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = outputs.last_hidden_state[0].to(
                    dtype=torch.bfloat16, device="cpu"
                )
                expected = (int(input_ids.shape[1]), HIDDEN_SIZE)
                if tuple(hidden.shape) != expected:
                    raise H3ComponentError(
                        f"Unexpected Qwen3-VL feature shape {tuple(hidden.shape)}; "
                        f"expected {expected}"
                    )
                hidden = hidden.contiguous()
                if OPT_ENCODE_CACHE:
                    get_encode_cache().put(cache_key, hidden)
                return hidden

    @staticmethod
    def _conditioning_payload(
        *,
        prompt: str,
        hidden: Any,
        token_tags: Any,
        input_ids: Any,
        **metadata: Any,
    ) -> dict[str, Any]:
        import torch

        hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        token_tags = token_tags.detach().to(device="cpu", dtype=torch.long).contiguous()
        input_ids = input_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        if hidden.ndim != 2 or int(hidden.shape[1]) != HIDDEN_SIZE:
            raise H3ComponentError(
                f"Qwen conditioning must be [L,{HIDDEN_SIZE}], got "
                f"{tuple(hidden.shape)}"
            )
        if token_tags.ndim != 1 or int(token_tags.shape[0]) != int(
            hidden.shape[0]
        ):
            raise H3ComponentError(
                "Qwen text_token_tags must be [L] and align with prompt_embeds"
            )
        if input_ids.ndim != 1 or int(input_ids.shape[0]) != int(hidden.shape[0]):
            raise H3ComponentError(
                "Qwen presentation input_ids must align with prompt_embeds"
            )
        output = {
            "prompt": prompt,
            "prompt_embeds": hidden,
            # Compatibility alias used by the original T2VA runtime.
            "hidden_states": hidden,
            "text_len": int(hidden.shape[0]),
            "text_token_tags": token_tags,
            "presentation_input_ids": input_ids,
            "cfg_distilled": True,
        }
        output.update(metadata)
        return output

    def encode_fl2va_conditioning(
        self,
        prompt: str,
        images: Sequence[Any],
    ) -> dict[str, Any]:
        """Encode one/two already-prepared FL target-canvas keyframes.

        The caller owns FL canvas semantics.  The exact same prepared images
        must also be passed to the Video-VAE keyframe encoder.
        """

        from ..presentation import (
            minimax_h3_multi_image_presentation_ids,
            minimax_h3_multi_image_presentation_token_tags,
        )

        if not isinstance(prompt, str) or not prompt:
            raise H3ComponentError("MiniMax-H3 prompt must be a non-empty string")
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes)):
            raise H3ComponentError("FL2VA images must be an ordered sequence")
        images = list(images)
        if len(images) not in (1, 2):
            raise H3ComponentError(
                "FL2VA Qwen presentation requires exactly one or two keyframes"
            )
        processor = self._require_processor()
        vision = processor.image_processor(
            images=images,
            return_tensors="pt",
        )
        try:
            pixel_values = vision["pixel_values"]
            image_grid_thw = vision["image_grid_thw"]
        except (KeyError, TypeError) as exc:
            raise H3ComponentError(
                "Qwen image_processor must return pixel_values and image_grid_thw"
            ) from exc
        if image_grid_thw.ndim != 2 or tuple(image_grid_thw.shape[1:]) != (3,):
            raise H3ComponentError(
                f"Qwen image_grid_thw must be [N,3], got "
                f"{tuple(image_grid_thw.shape)}"
            )
        if int(image_grid_thw.shape[0]) != len(images):
            raise H3ComponentError(
                f"Qwen processor returned {int(image_grid_thw.shape[0])} grids "
                f"for {len(images)} FL2VA images"
            )
        merge_size = int(getattr(processor.image_processor, "merge_size", 0))
        if merge_size <= 0:
            raise H3ComponentError(
                "Qwen image_processor has no positive spatial merge_size"
            )
        merge_area = merge_size**2
        image_token_counts = [
            int(image_grid_thw[index].prod().item()) // merge_area
            for index in range(len(images))
        ]
        if any(count <= 0 for count in image_token_counts):
            raise H3ComponentError("Qwen processor produced an empty image grid")
        input_ids = minimax_h3_multi_image_presentation_ids(
            self.tokenizer,
            prompt=prompt,
            image_token_counts=image_token_counts,
        )
        token_tags = minimax_h3_multi_image_presentation_token_tags(
            self.tokenizer,
            prompt=prompt,
            image_token_counts=image_token_counts,
        )
        hidden = self.encode_ids(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        return self._conditioning_payload(
            prompt=prompt,
            hidden=hidden,
            token_tags=token_tags,
            input_ids=input_ids,
            presentation="fl2va_multi_image_v1",
            image_token_counts=tuple(image_token_counts),
        )

    def encode_ref2va_conditioning(
        self,
        prompt: str,
        condition_labels: Sequence[tuple[str, int]],
        *,
        images: Sequence[Any] = (),
        videos: Sequence[Any] = (),
    ) -> dict[str, Any]:
        """Encode ordered Ref2VA references.

        ``videos`` contains one array/list of frames per Video label.  Frames
        must already be sampled from the prepared reference video at exactly
        2fps; this method disables processor-side sampling and derives the
        official temporal-patch timestamps from that sampled stream.
        """

        import numpy as np
        import torch

        from ..presentation import (
            QWEN_VIDEO_SAMPLE_FPS,
            minimax_h3_qwen_video_sample_plan,
            minimax_h3_ref2va_video_presentation,
        )

        if not isinstance(prompt, str) or not prompt:
            raise H3ComponentError("MiniMax-H3 prompt must be a non-empty string")
        labels = [(str(kind), int(ordinal)) for kind, ordinal in condition_labels]
        if not labels:
            raise H3ComponentError("Ref2VA requires at least one condition label")
        counters = {"image": 0, "audio": 0, "video": 0}
        for kind, ordinal in labels:
            if kind not in counters:
                raise H3ComponentError(
                    f"Unsupported Ref2VA Qwen condition label {kind!r}"
                )
            counters[kind] += 1
            if ordinal != counters[kind]:
                raise H3ComponentError(
                    f"Ref2VA {kind} ordinals must be consecutive and 1-based; "
                    f"expected {counters[kind]}, got {ordinal}"
                )
        images = list(images)
        videos = list(videos)
        if len(images) != counters["image"]:
            raise H3ComponentError(
                f"Ref2VA labels require {counters['image']} images, got "
                f"{len(images)}"
            )
        if len(videos) != counters["video"]:
            raise H3ComponentError(
                f"Ref2VA labels require {counters['video']} sampled videos, got "
                f"{len(videos)}"
            )
        processor = self._require_processor(video=bool(videos))
        merge_size = int(getattr(processor.image_processor, "merge_size", 0))
        if merge_size <= 0:
            raise H3ComponentError(
                "Qwen image_processor has no positive spatial merge_size"
            )
        merge_area = merge_size**2

        pixel_values = None
        image_grid_thw = None
        image_token_counts: list[int] = []
        if images:
            image_output = processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            try:
                pixel_values = image_output["pixel_values"]
                image_grid_thw = image_output["image_grid_thw"]
            except (KeyError, TypeError) as exc:
                raise H3ComponentError(
                    "Qwen image_processor must return pixel_values and "
                    "image_grid_thw"
                ) from exc
            if int(image_grid_thw.shape[0]) != len(images):
                raise H3ComponentError(
                    "Qwen image grid count does not match Ref2VA image count"
                )
            image_token_counts = [
                int(image_grid_thw[index].prod().item()) // merge_area
                for index in range(len(images))
            ]

        pixel_values_videos = None
        video_grid_thw = None
        video_block_token_counts: list[list[int]] = []
        video_block_timestamps: list[list[float]] = []
        if videos:
            stacked_videos = []
            sampled_counts = []
            for index, video in enumerate(videos):
                if isinstance(video, torch.Tensor):
                    array = video.detach().cpu().numpy()
                elif isinstance(video, np.ndarray):
                    array = video
                else:
                    try:
                        array = np.stack(list(video))
                    except Exception as exc:
                        raise H3ComponentError(
                            f"Ref2VA sampled video {index} cannot be stacked"
                        ) from exc
                if array.ndim != 4 or int(array.shape[0]) <= 0 or int(
                    array.shape[-1]
                ) != 3:
                    raise H3ComponentError(
                        "Ref2VA sampled videos must have shape [T,H,W,3], got "
                        f"{tuple(array.shape)} for video {index}"
                    )
                stacked_videos.append(array)
                sampled_counts.append(int(array.shape[0]))
            video_output = processor.video_processor(
                videos=stacked_videos,
                do_sample_frames=False,
                return_tensors="pt",
            )
            try:
                pixel_values_videos = video_output["pixel_values_videos"]
                video_grid_thw = video_output["video_grid_thw"]
            except (KeyError, TypeError) as exc:
                raise H3ComponentError(
                    "Qwen video_processor must return pixel_values_videos and "
                    "video_grid_thw"
                ) from exc
            if int(video_grid_thw.shape[0]) != len(videos):
                raise H3ComponentError(
                    "Qwen video grid count does not match Ref2VA video count"
                )
            for index, sample_count in enumerate(sampled_counts):
                _indices, timestamps = minimax_h3_qwen_video_sample_plan(
                    sample_count,
                    source_fps=QWEN_VIDEO_SAMPLE_FPS,
                    sample_fps=QWEN_VIDEO_SAMPLE_FPS,
                )
                temporal_blocks = int(video_grid_thw[index, 0].item())
                if temporal_blocks != len(timestamps):
                    raise H3ComponentError(
                        "Qwen temporal-patch mismatch for Ref2VA video "
                        f"{index}: processor={temporal_blocks}, "
                        f"timestamps={len(timestamps)}"
                    )
                per_block = (
                    int(video_grid_thw[index, 1].item())
                    * int(video_grid_thw[index, 2].item())
                    // merge_area
                )
                if per_block <= 0:
                    raise H3ComponentError(
                        f"Qwen processor produced empty video grid {index}"
                    )
                video_block_token_counts.append(
                    [per_block] * temporal_blocks
                )
                video_block_timestamps.append(timestamps)

        input_ids, token_tags = minimax_h3_ref2va_video_presentation(
            self.tokenizer,
            prompt=prompt,
            condition_labels=labels,
            image_token_count=image_token_counts or None,
            video_block_token_counts=video_block_token_counts or None,
            video_block_timestamps=video_block_timestamps or None,
        )
        hidden = self.encode_ids(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        return self._conditioning_payload(
            prompt=prompt,
            hidden=hidden,
            token_tags=token_tags,
            input_ids=input_ids,
            presentation="ref2va_ordered_v1",
            condition_labels=tuple(labels),
            image_token_counts=tuple(image_token_counts),
            video_block_token_counts=tuple(
                tuple(item) for item in video_block_token_counts
            ),
            video_block_timestamps=tuple(
                tuple(item) for item in video_block_timestamps
            ),
            video_sample_fps=QWEN_VIDEO_SAMPLE_FPS,
        )

    def encode_conditioning(self, prompt: str) -> dict[str, Any]:
        import torch

        hidden = self.encode_prompt(prompt)
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        return self._conditioning_payload(
            prompt=prompt,
            hidden=hidden,
            token_tags=torch.ones(int(hidden.shape[0]), dtype=torch.long),
            input_ids=input_ids,
            presentation="t2va_text_only_v1",
        )

__all__ = [n for n in list(globals()) if not n.startswith("__")]
