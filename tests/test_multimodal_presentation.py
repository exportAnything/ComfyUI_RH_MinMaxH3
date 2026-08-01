import importlib.util
import inspect
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock


HAS_TORCH = importlib.util.find_spec("torch") is not None


class ScopedCudnnSdpTests(unittest.TestCase):
    @staticmethod
    def _scope():
        from minimax_h3_nodes.runtime.qwen_encoder import _scoped_cudnn_sdp

        return _scoped_cudnn_sdp

    @staticmethod
    def _fake_torch(cuda_backend):
        return types.SimpleNamespace(
            backends=types.SimpleNamespace(cuda=cuda_backend)
        )

    def test_restores_backend_after_success_and_exception(self):
        class Backend:
            def __init__(self):
                self.state = False
                self.events = []

            def cudnn_sdp_enabled(self):
                return self.state

            def enable_cudnn_sdp(self, value):
                self.state = bool(value)
                self.events.append(self.state)

        backend = Backend()
        with mock.patch.dict(sys.modules, {"torch": self._fake_torch(backend)}):
            with self._scope()(True):
                self.assertTrue(backend.state)
            self.assertFalse(backend.state)
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with self._scope()(True):
                    self.assertTrue(backend.state)
                    raise RuntimeError("boom")

        self.assertFalse(backend.state)
        self.assertEqual(backend.events, [True, False, True, False])

    def test_does_not_mutate_when_getter_is_unavailable(self):
        class SetterOnlyBackend:
            def __init__(self):
                self.calls = []

            def enable_cudnn_sdp(self, value):
                self.calls.append(bool(value))

        backend = SetterOnlyBackend()
        with mock.patch.dict(sys.modules, {"torch": self._fake_torch(backend)}):
            with self._scope()(True):
                pass
        self.assertEqual(backend.calls, [])

    def test_global_lock_serializes_independent_scopes(self):
        class Backend:
            def __init__(self):
                self.state = False
                self.events = []

            def cudnn_sdp_enabled(self):
                return self.state

            def enable_cudnn_sdp(self, value):
                self.state = bool(value)
                self.events.append(self.state)

        backend = Backend()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        scope = self._scope()

        def first():
            with scope(True):
                first_entered.set()
                release_first.wait(timeout=2)

        def second():
            with scope(True):
                second_entered.set()

        with mock.patch.dict(sys.modules, {"torch": self._fake_torch(backend)}):
            first_thread = threading.Thread(target=first)
            second_thread = threading.Thread(target=second)
            first_thread.start()
            self.assertTrue(first_entered.wait(timeout=2))
            second_thread.start()
            self.assertFalse(second_entered.wait(timeout=0.05))
            release_first.set()
            first_thread.join(timeout=2)
            second_thread.join(timeout=2)

        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertTrue(second_entered.is_set())
        self.assertFalse(backend.state)
        self.assertEqual(backend.events, [True, False, True, False])

    def test_all_encoder_entry_points_use_the_shared_scope(self):
        from minimax_h3_nodes.runtime.backend_state import (
            TORCH_BACKEND_STATE_LOCK,
        )
        from minimax_h3_nodes.runtime import qwen_encoder
        from minimax_h3_nodes.runtime.qwen_encoder import MiniMaxH3TextEncoder

        encode_ids = inspect.getsource(MiniMaxH3TextEncoder.encode_ids)
        encode_prompt = inspect.getsource(MiniMaxH3TextEncoder.encode_prompt)
        self.assertEqual(encode_ids.count("_scoped_cudnn_sdp"), 2)
        self.assertEqual(encode_prompt.count("_scoped_cudnn_sdp"), 1)
        self.assertNotIn("enable_cudnn_sdp", encode_ids)
        self.assertNotIn("enable_cudnn_sdp", encode_prompt)
        self.assertIs(
            qwen_encoder.TORCH_BACKEND_STATE_LOCK,
            TORCH_BACKEND_STATE_LOCK,
        )
        self.assertIn(
            "with TORCH_BACKEND_STATE_LOCK",
            inspect.getsource(qwen_encoder._scoped_cudnn_sdp),
        )


class QwenConfigValidationTests(unittest.TestCase):
    @staticmethod
    def _config():
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _TEXT_CONFIG_CONTRACT,
            _VISION_CONFIG_CONTRACT,
            _VISION_DEEPSTACK_INDEXES,
        )

        return types.SimpleNamespace(
            model_type="qwen3_vl",
            architectures=["Qwen3VLForConditionalGeneration"],
            text_config=types.SimpleNamespace(
                **_TEXT_CONFIG_CONTRACT,
                num_hidden_layers=64,
            ),
            vision_config=types.SimpleNamespace(
                **_VISION_CONFIG_CONTRACT,
                deepstack_visual_indexes=list(_VISION_DEEPSTACK_INDEXES),
            ),
        )

    def test_accepts_the_released_qwen_32b_vision_contract(self):
        from minimax_h3_nodes.runtime.qwen_encoder import _validate_qwen_config

        config = self._config()
        self.assertIs(
            _validate_qwen_config(config, Path("/model/text_encoder")),
            config.text_config,
        )

    def test_rejects_each_vision_dimension_before_model_construction(self):
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _VISION_CONFIG_CONTRACT,
            _validate_qwen_config,
        )

        for name, expected in _VISION_CONFIG_CONTRACT.items():
            with self.subTest(name=name):
                config = self._config()
                setattr(config.vision_config, name, expected + 1)
                with self.assertRaisesRegex(
                    Exception, "released Qwen3-VL-32B vision architecture"
                ):
                    _validate_qwen_config(config, Path("/model/text_encoder"))

        config = self._config()
        config.vision_config.deepstack_visual_indexes = [8, 16, 25]
        with self.assertRaisesRegex(Exception, "deepstack_visual_indexes"):
            _validate_qwen_config(config, Path("/model/text_encoder"))


class QwenLoaderSafetyTests(unittest.TestCase):
    class FakeOutOfMemoryError(Exception):
        pass

    @staticmethod
    def _fake_modules(load_models_gpu, soft_empty_cache=lambda: None):
        fake_torch = types.ModuleType("torch")
        fake_torch.OutOfMemoryError = QwenLoaderSafetyTests.FakeOutOfMemoryError
        fake_torch.cuda = types.SimpleNamespace(
            empty_cache=lambda: None,
            is_available=lambda: False,
        )
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        management = types.ModuleType("comfy.model_management")
        management.load_models_gpu = load_models_gpu
        management.soft_empty_cache = soft_empty_cache
        comfy.model_management = management
        return {
            "torch": fake_torch,
            "comfy": comfy,
            "comfy.model_management": management,
        }

    @staticmethod
    def _encoder(*, static_load_error=None, unload_error=None):
        from minimax_h3_nodes.runtime.qwen_encoder import MiniMaxH3TextEncoder

        encoder = object.__new__(MiniMaxH3TextEncoder)
        encoder.load_device = types.SimpleNamespace(type="cuda", name="load")
        encoder.offload_device = types.SimpleNamespace(type="cpu", name="offload")
        encoder.quantized = True
        encoder._inference_active = False
        encoder._compute_device = "stale"
        encoder._static_storage_bytes = 17
        encoder._linear_storage_bytes = 23
        encoder.model = types.SimpleNamespace(_h3_compute_device="stale")
        patcher = types.SimpleNamespace()
        events = []
        encoder._ensure_linear_patcher = lambda: patcher

        def unload():
            events.append("unload")
            if unload_error is not None:
                raise unload_error

        def move(device):
            events.append(f"move:{device.name}")
            if device is encoder.load_device and static_load_error is not None:
                raise static_load_error

        def set_compute(device):
            events.append(f"compute:{device}")
            encoder._compute_device = device
            if device is None and hasattr(encoder.model, "_h3_compute_device"):
                delattr(encoder.model, "_h3_compute_device")

        encoder._unload_linear_patcher = unload
        encoder._move_static_tensors = move
        encoder._set_compute_device = set_compute
        return encoder, patcher, events

    def test_non_oom_acquisition_failures_cleanup_and_reraise_original(self):
        for stage in ("manager", "static"):
            with self.subTest(stage=stage):
                primary = RuntimeError(f"{stage} failed")
                static_error = primary if stage == "static" else None
                encoder, _patcher, events = self._encoder(
                    static_load_error=static_error
                )

                def load_models_gpu(_models, *, memory_required):
                    self.assertGreater(memory_required, 0)
                    events.append("manager-load")
                    if stage == "manager":
                        raise primary

                modules = self._fake_modules(load_models_gpu)
                with mock.patch.dict(sys.modules, modules):
                    for _attempt in range(2):
                        with self.assertRaises(RuntimeError) as caught:
                            encoder.load_for_inference()
                        self.assertIs(caught.exception, primary)
                        self.assertIsNone(encoder._compute_device)
                        self.assertFalse(encoder._inference_active)
                        self.assertFalse(
                            hasattr(encoder.model, "_h3_compute_device")
                        )

                self.assertEqual(events.count("unload"), 2)
                self.assertEqual(events.count("move:offload"), 2)

    def test_cleanup_failure_does_not_mask_non_oom_primary_error(self):
        primary = RuntimeError("manager failed")
        encoder, _patcher, events = self._encoder(
            unload_error=RuntimeError("cleanup failed")
        )

        def load_models_gpu(_models, *, memory_required):
            raise primary

        with mock.patch.dict(sys.modules, self._fake_modules(load_models_gpu)):
            with mock.patch(
                "minimax_h3_nodes.runtime.qwen_encoder.LOGGER.exception"
            ):
                with self.assertRaises(RuntimeError) as caught:
                    encoder.load_for_inference()
        self.assertIs(caught.exception, primary)
        self.assertIn("move:offload", events)
        self.assertIsNone(encoder._compute_device)
        self.assertFalse(encoder._inference_active)

    def test_oom_keeps_cpu_fallback_only_after_clean_rollback(self):
        primary = self.FakeOutOfMemoryError("oom")
        encoder, _patcher, events = self._encoder()
        cache_calls = []

        def load_models_gpu(_models, *, memory_required):
            raise primary

        modules = self._fake_modules(
            load_models_gpu, soft_empty_cache=lambda: cache_calls.append(True)
        )
        with mock.patch.dict(sys.modules, modules):
            self.assertIs(encoder.load_for_inference(), encoder)
        self.assertEqual(cache_calls, [True])
        self.assertIn("unload", events)
        self.assertIn("move:offload", events)
        self.assertIsNone(encoder._compute_device)
        self.assertFalse(encoder._inference_active)

    def test_quantized_offload_clears_state_even_when_cleanup_fails(self):
        cleanup_error = RuntimeError("manager unload failed")
        encoder, patcher, events = self._encoder(unload_error=cleanup_error)
        encoder._lock = threading.RLock()
        encoder._linear_patcher = patcher
        encoder._inference_active = True
        modules = self._fake_modules(lambda *_args, **_kwargs: None)
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError) as caught:
                encoder.offload_after_inference()
        self.assertIs(caught.exception, cleanup_error)
        self.assertIn("move:offload", events)
        self.assertIsNone(encoder._compute_device)
        self.assertFalse(encoder._inference_active)
        self.assertFalse(hasattr(encoder.model, "_h3_compute_device"))

        encoder._unload_linear_patcher = lambda: events.append("unload-retry")
        with mock.patch.dict(sys.modules, modules):
            encoder.offload_after_inference()
            encoder.offload_after_inference()
        self.assertEqual(events.count("unload-retry"), 2)

    def test_quantized_offload_retries_static_move_after_first_failure(self):
        encoder, patcher, events = self._encoder()
        encoder._lock = threading.RLock()
        encoder._linear_patcher = patcher
        encoder._inference_active = True
        static_error = RuntimeError("static offload failed")
        attempts = []

        def move(_device):
            attempts.append(True)
            if len(attempts) == 1:
                raise static_error

        encoder._move_static_tensors = move
        modules = self._fake_modules(lambda *_args, **_kwargs: None)
        with mock.patch.dict(sys.modules, modules):
            with self.assertRaises(RuntimeError) as caught:
                encoder.offload_after_inference()
            self.assertIs(caught.exception, static_error)
            self.assertIsNone(encoder._compute_device)
            self.assertFalse(encoder._inference_active)
            encoder.offload_after_inference()
        self.assertEqual(len(attempts), 2)

    def test_full_precision_load_is_transactional_for_all_exceptions(self):
        from minimax_h3_nodes.runtime.qwen_encoder import MiniMaxH3TextEncoder

        for error in (
            RuntimeError("ordinary move failure"),
            self.FakeOutOfMemoryError("oom move failure"),
        ):
            with self.subTest(error=type(error).__name__):
                encoder = object.__new__(MiniMaxH3TextEncoder)
                encoder.load_device = types.SimpleNamespace(
                    type="cuda", name="load"
                )
                encoder.offload_device = types.SimpleNamespace(
                    type="cpu", name="offload"
                )
                encoder.quantized = False
                encoder._inference_active = False
                encoder._compute_device = "stale"
                encoder.model = types.SimpleNamespace(
                    _h3_compute_device="stale"
                )
                events = []

                class Module:
                    def state_dict(self):
                        return {}

                    def to(self, device):
                        events.append(device.name)
                        if device is encoder.load_device:
                            raise error

                modules_to_move = [Module(), Module()]
                encoder._movable = lambda: modules_to_move
                encoder._actual_device = lambda: encoder.offload_device

                def set_compute(device):
                    encoder._compute_device = device
                    if device is None and hasattr(
                        encoder.model, "_h3_compute_device"
                    ):
                        delattr(encoder.model, "_h3_compute_device")

                encoder._set_compute_device = set_compute
                fake_modules = self._fake_modules(
                    lambda *_args, **_kwargs: None
                )
                fake_cuda = fake_modules["torch"].cuda
                fake_cuda.is_available = lambda: True
                fake_cuda.mem_get_info = lambda _device: (1 << 60, 1 << 60)
                with mock.patch.dict(sys.modules, fake_modules):
                    if isinstance(error, self.FakeOutOfMemoryError):
                        self.assertIs(encoder.load_for_inference(), encoder)
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            encoder.load_for_inference()
                        self.assertIs(caught.exception, error)
                self.assertEqual(events.count("load"), 1)
                self.assertEqual(events.count("offload"), 2)
                self.assertIsNone(encoder._compute_device)
                self.assertFalse(encoder._inference_active)
                self.assertFalse(
                    hasattr(encoder.model, "_h3_compute_device")
                )

    def test_visual_offload_failure_preserves_primary_error(self):
        from minimax_h3_nodes.runtime.qwen_encoder import MiniMaxH3TextEncoder

        encoder = object.__new__(MiniMaxH3TextEncoder)
        encoder.offload_device = "cpu"
        cleanup_error = RuntimeError("visual offload failed")

        class Visual:
            def to(self, _device):
                raise cleanup_error

        primary = ValueError("visual forward failed")
        with mock.patch(
            "minimax_h3_nodes.runtime.qwen_encoder.LOGGER.exception"
        ):
            try:
                try:
                    raise primary
                finally:
                    encoder._offload_visual_after_staged(
                        Visual(),
                        preserve_primary_error=sys.exc_info()[0] is not None,
                    )
            except ValueError as caught:
                self.assertIs(caught, primary)
            else:
                self.fail("primary visual error was swallowed")

        with self.assertRaises(RuntimeError) as caught:
            encoder._offload_visual_after_staged(
                Visual(), preserve_primary_error=False
            )
        self.assertIs(caught.exception, cleanup_error)

    def test_bf16_loader_forces_local_safetensors(self):
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _load_qwen_bf16_checkpoint,
            load_h3_text_encoder,
        )

        source = inspect.getsource(load_h3_text_encoder)
        self.assertIn('"local_files_only": True', source)
        self.assertIn('"trust_remote_code": False', source)
        self.assertIn('"use_safetensors": True', source)

        calls = []

        class FakeModelClass:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls.append((path, dict(kwargs)))
                if "dtype" in kwargs:
                    raise TypeError("unexpected keyword argument 'dtype'")
                return "loaded"

        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        self.assertEqual(
            _load_qwen_bf16_checkpoint(
                FakeModelClass,
                Path("/model/text_encoder"),
                model_dtype="bf16-sentinel",
                load_kwargs=load_kwargs,
            ),
            "loaded",
        )
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call[1]["use_safetensors"] for call in calls))
        self.assertEqual(calls[0][1]["dtype"], "bf16-sentinel")
        self.assertEqual(calls[1][1]["torch_dtype"], "bf16-sentinel")

    def test_v2_multimodal_loader_requires_processor_before_model_load(self):
        from minimax_h3_nodes import nodes
        from minimax_h3_nodes.runtime.qwen_encoder import load_h3_text_encoder

        node_source = inspect.getsource(
            nodes._MiniMaxH3ExplicitTextEncoderLoader.load
        )
        self.assertIn("require_multimodal_processor=True", node_source)
        loader_source = inspect.getsource(load_h3_text_encoder)
        self.assertLess(
            loader_source.index("AutoProcessor.from_pretrained"),
            loader_source.index("AutoConfig.from_pretrained"),
        )
        self.assertIn(
            "if processor_path or require_multimodal_processor",
            loader_source,
        )


class QwenQuantMetadataTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, value: dict):
        import json

        (path / "quant_meta.json").write_text(
            json.dumps(value), encoding="utf-8"
        )

    @staticmethod
    def _valid(**updates):
        value = {
            "format": "int8_tensorwise",
            "convrot": True,
            "arch": "qwen3_vl_text_encoder",
            "selected_layers": 50,
            "quantized_linears": 350,
        }
        value.update(updates)
        return value

    def test_manifest_is_required_and_format_convrot_are_strict(self):
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _validate_text_encoder_quant_metadata,
        )

        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw)
            with self.assertRaisesRegex(Exception, "requires .*quant_meta"):
                _validate_text_encoder_quant_metadata(
                    component, partition="fl2va"
                )
            for name, value in (
                ("format", "wrong"),
                ("convrot", False),
                ("quantized_linears", 349),
            ):
                with self.subTest(name=name):
                    self._write(component, self._valid(**{name: value}))
                    with self.assertRaises(Exception):
                        _validate_text_encoder_quant_metadata(
                            component, partition="fl2va"
                        )

    def test_legacy_shared_and_partition_provenance_policy(self):
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _validate_text_encoder_quant_metadata,
        )

        with tempfile.TemporaryDirectory() as raw:
            component = Path(raw)
            self._write(component, self._valid())
            with mock.patch(
                "minimax_h3_nodes.runtime.qwen_encoder.LOGGER.warning"
            ) as warning:
                _validate_text_encoder_quant_metadata(
                    component, partition="ref2va"
                )
            warning.assert_called_once()

            self._write(component, self._valid(partition="shared"))
            _validate_text_encoder_quant_metadata(
                component, partition="fl2va"
            )
            _validate_text_encoder_quant_metadata(
                component, partition="ref2va"
            )

            self._write(component, self._valid(partition="FL2VA"))
            with self.assertRaisesRegex(Exception, "partition mismatch"):
                _validate_text_encoder_quant_metadata(
                    component, partition="ref2va"
                )

    def test_each_linear_marker_must_match_manifest(self):
        from minimax_h3_nodes.runtime.qwen_encoder import (
            _validate_text_encoder_quant_marker,
        )

        manifest = self._valid(partition="shared")
        _validate_text_encoder_quant_marker(
            "layer.0.",
            {"format": "int8_tensorwise", "convrot": True},
            manifest,
        )
        with self.assertRaisesRegex(Exception, "metadata mismatch"):
            _validate_text_encoder_quant_marker(
                "layer.0.",
                {"format": "int8_tensorwise", "convrot": False},
                manifest,
            )


class _Tokenizer:
    special = {
        "<|vision_start|>": 9001,
        "<|vision_end|>": 9002,
        "<|image_pad|>": 9003,
        "<|video_pad|>": 9004,
    }
    unk_token_id = 9999

    def __call__(self, text, *, add_special_tokens=False, **_kwargs):
        if add_special_tokens:
            raise AssertionError("H3 presentations must disable special tokens")
        return {"input_ids": [100 + ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return self.special.get(token, self.unk_token_id)


class VideoSamplePlanTests(unittest.TestCase):
    def test_matches_official_24_to_2fps_recipe(self):
        from minimax_h3_nodes.runtime.presentation import (
            minimax_h3_qwen_video_sample_plan,
        )

        indices, timestamps = minimax_h3_qwen_video_sample_plan(124)
        self.assertEqual(indices, list(range(0, 121, 12)))
        self.assertEqual(timestamps, [0.25, 1.25, 2.25, 3.25, 4.25, 5.0])


@unittest.skipUnless(HAS_TORCH, "requires torch")
class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _Tokenizer()

    def text_ids(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def test_t2va_is_verbatim_without_special_tokens(self):
        from minimax_h3_nodes.runtime.presentation import (
            minimax_h3_text_only_ids,
        )

        ids = minimax_h3_text_only_ids(self.tokenizer, "  prompt\n")
        self.assertEqual(ids.tolist(), self.text_ids("  prompt\n"))

    def test_fl2va_picture_blocks_and_tags_are_exactly_aligned(self):
        from minimax_h3_nodes.runtime.presentation import (
            minimax_h3_multi_image_presentation_ids,
            minimax_h3_multi_image_presentation_token_tags,
        )

        ids = minimax_h3_multi_image_presentation_ids(
            self.tokenizer,
            prompt="keep prompt verbatim",
            image_token_counts=[2, 1],
        )
        tags = minimax_h3_multi_image_presentation_token_tags(
            self.tokenizer,
            prompt="keep prompt verbatim",
            image_token_counts=[2, 1],
        )
        first_label = self.text_ids("<Picture 1>: ")
        second_label = self.text_ids("<Picture 2>: ")
        prompt = self.text_ids("keep prompt verbatim")
        first_vision = [9001, 9003, 9003, 9002]
        second_vision = [9001, 9003, 9002]
        expected = (
            first_label
            + first_vision
            + second_label
            + second_vision
            + prompt
        )
        expected_tags = (
            [1] * len(first_label)
            + [0] * len(first_vision)
            + [1] * len(second_label)
            + [0] * len(second_vision)
            + [1] * len(prompt)
        )
        self.assertEqual(ids.tolist(), expected)
        self.assertEqual(tags.tolist(), expected_tags)
        self.assertEqual(ids.shape, tags.shape)

    def test_ref2va_preserves_order_and_video_timestamp_blocks(self):
        from minimax_h3_nodes.runtime.presentation import (
            minimax_h3_ref2va_video_presentation,
        )

        ids, tags = minimax_h3_ref2va_video_presentation(
            self.tokenizer,
            prompt="final prompt",
            condition_labels=[
                ("audio", 1),
                ("image", 1),
                ("audio", 2),
                ("video", 1),
            ],
            image_token_count=3,
            video_block_token_counts=[[2, 1]],
            video_block_timestamps=[[0.25, 0.75]],
        )
        segments = [
            (self.text_ids("<Audio 1>: "), 1),
            (self.text_ids("<Picture 1>: "), 1),
            ([9001, 9003, 9003, 9003, 9002], 0),
            (self.text_ids("<Audio 2>: "), 1),
            (self.text_ids("<Video 1>: "), 1),
            # Python .1f bankers-rounding is the official prompt contract.
            (self.text_ids("<0.2 seconds>"), 1),
            ([9001, 9004, 9004, 9002], 0),
            (self.text_ids("<0.8 seconds>"), 1),
            ([9001, 9004, 9002], 0),
            (self.text_ids("final prompt"), 1),
        ]
        expected_ids = [item for values, _ in segments for item in values]
        expected_tags = [
            tag for values, tag in segments for _item in values
        ]
        self.assertEqual(ids.tolist(), expected_ids)
        self.assertEqual(tags.tolist(), expected_tags)

    def test_ref2va_rejects_unused_or_missing_media_counts(self):
        from minimax_h3_nodes.runtime.presentation import (
            minimax_h3_ref2va_video_presentation,
        )

        with self.assertRaisesRegex(ValueError, "unused image"):
            minimax_h3_ref2va_video_presentation(
                self.tokenizer,
                prompt="p",
                condition_labels=[("audio", 1)],
                image_token_count=[4],
                video_block_token_counts=None,
                video_block_timestamps=None,
            )
        with self.assertRaisesRegex(ValueError, "video reference requires"):
            minimax_h3_ref2va_video_presentation(
                self.tokenizer,
                prompt="p",
                condition_labels=[("video", 1)],
                image_token_count=None,
                video_block_token_counts=None,
                video_block_timestamps=None,
            )

    def test_mm_token_types_only_mark_pad_tokens(self):
        import torch

        from minimax_h3_nodes.runtime.qwen_encoder import (
            minimax_h3_mm_token_type_ids,
        )

        ids = torch.tensor([[10, 9001, 9003, 9002, 9004, 11]])
        types = minimax_h3_mm_token_type_ids(
            ids,
            image_token_id=9003,
            video_token_id=9004,
        )
        self.assertEqual(types.dtype, torch.int32)
        self.assertEqual(types.tolist(), [[0, 0, 1, 0, 2, 0]])


if __name__ == "__main__":
    unittest.main()
