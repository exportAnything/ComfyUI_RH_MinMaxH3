# Notices and provenance

This custom-node package contains adaptations of the MiniMax-H3 model runtime
from the official `Internal-0727-private-3` source package. The reference
snapshot used for the native H3 implementation is:

- source package: `MiniMax-AI-Dev/Internal-0727-private-3`
- baseline commit: `5d8a20b1717f`
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
