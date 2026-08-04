# Notices and provenance

This custom-node package contains adaptations of the MiniMax-H3 model runtime
released by MiniMax-AI. The reference snapshot used for the native H3
implementation is:

- upstream project: MiniMax-AI (https://github.com/MiniMax-AI)
- upstream project license: Apache License 2.0

Adapted portions include the H3 DiT architecture and weight-layout handling,
the packed audio/video sequence contract, the rectified-flow scheduler, and
the MiniMax-H3 video/audio VAE modules.

The adapted files were changed to:

- remove SGLang server, manager, distributed, and compiled-kernel dependencies;
- replace tensor-parallel linear layers with single-device PyTorch modules;
- use PyTorch SDPA and eager PyTorch rotary/math fallbacks;
- expose ComfyUI node and model-lifecycle contracts;
- support streamed safetensors materialisation from a meta-device model.

No MiniMax-H3 model weights are included. Users remain responsible for the
licence and confidentiality terms that apply to their checkpoint files.

ComfyUI and Transformers are runtime dependencies/interfaces; their source code
is not bundled in this archive.

## Bundled Sol-style sparse FlexAttention

`minimax_h3_nodes/runtime/attention_backends.py` contains an adapted and
substantially revised block-routing/FlexAttention path based on the Apache-2.0
project:

- KingGore/ComfyUI_sol-attn_Blackwell
- source snapshot: `de7ffe310fbfedb2920489fc4690a98410a189bb`
- upstream license: Apache License 2.0

The adaptation removes startup warm-up and Torch-install mutation, fixes
attention-scale/output-layout/fallback handling, preserves previously installed
ComfyUI attention overrides, adds sampling-window and minimum-sequence gating,
uses deterministic routed indices, and handles partial block means correctly.

This bundled path is described as **Sol-style sparse FlexAttention**. It is not
the official NVIDIA Sol-Attn kernel: unselected blocks are omitted instead of
receiving the proxy correction used by the official algorithm. No code from the
Kijai `ComfyUI-SolAttn_triton` repository is bundled. The algorithm reference
is the NVLabs Sol-Attn project: <https://nvlabs.github.io/Sana/Sol-Attn/>.

SageAttention is an optional runtime library/interface. Its CUDA/Triton kernels
are not copied into this repository.
