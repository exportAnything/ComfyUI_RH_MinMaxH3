from __future__ import annotations

import importlib.util
import threading
import unittest


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class BackendStateLockTests(unittest.TestCase):
    def test_default_dtype_scopes_serialize_and_restore_process_state(self):
        import torch

        from minimax_h3_nodes.runtime.vae_adapter import _default_dtype

        original = torch.get_default_dtype()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        errors: list[BaseException] = []

        def first_scope() -> None:
            try:
                with _default_dtype(torch.float64):
                    if torch.get_default_dtype() is not torch.float64:
                        raise AssertionError("first dtype scope was not installed")
                    first_entered.set()
                    if not release_first.wait(5):
                        raise TimeoutError("test did not release first dtype scope")
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        def second_scope() -> None:
            try:
                with _default_dtype(torch.float16):
                    if torch.get_default_dtype() is not torch.float16:
                        raise AssertionError("second dtype scope was not installed")
                    second_entered.set()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        first = threading.Thread(target=first_scope)
        second = threading.Thread(target=second_scope)
        restored = None
        first.start()
        try:
            self.assertTrue(first_entered.wait(5))
            second.start()
            self.assertFalse(second_entered.wait(0.1))
        finally:
            release_first.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
            restored = torch.get_default_dtype()
            # Keep the test process safe even if an assertion above fails.
            torch.set_default_dtype(original)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(second_entered.is_set())
        self.assertIs(restored, original)

    def test_audio_backend_scope_uses_process_wide_backend_lock(self):
        from minimax_h3_nodes.runtime import backend_state, vae_adapter

        self.assertIs(
            vae_adapter.TORCH_BACKEND_STATE_LOCK,
            backend_state.TORCH_BACKEND_STATE_LOCK,
        )

        entered = threading.Event()
        errors: list[BaseException] = []

        def audio_scope() -> None:
            try:
                with vae_adapter._audio_vae_determinism():
                    entered.set()
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)

        worker = threading.Thread(target=audio_scope)
        with backend_state.TORCH_BACKEND_STATE_LOCK:
            worker.start()
            self.assertFalse(entered.wait(0.1))
        worker.join(5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(entered.is_set())


if __name__ == "__main__":
    unittest.main()
