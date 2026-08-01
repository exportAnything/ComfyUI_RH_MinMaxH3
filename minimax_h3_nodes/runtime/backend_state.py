# SPDX-License-Identifier: Apache-2.0
"""Process-wide guards for temporary PyTorch backend state changes."""

from __future__ import annotations

import threading


# PyTorch default dtype and ``torch.backends`` flags are process-global.  Every
# runtime scope that temporarily mutates either must share this lock so two
# concurrent ComfyUI workflows cannot interleave their save/restore sequences.
TORCH_BACKEND_STATE_LOCK = threading.RLock()
