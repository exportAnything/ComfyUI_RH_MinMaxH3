"""ComfyUI nodes facade: node registry and compatibility imports; implementations live in api/."""
from __future__ import annotations

from .api._shared import *  # noqa: F403
from .api.loaders import (
    _MiniMaxH3ExplicitModelLoader,
    _MiniMaxH3ExplicitTextEncoderLoader,
    _MiniMaxH3ExplicitVAELoader,
    MiniMaxH3DirectModelLoader,
    MiniMaxH3DirectTextEncoderLoader,
    MiniMaxH3DirectVAELoader,
    MiniMaxH3FL2VAModelLoader,
    MiniMaxH3FL2VATextEncoderLoader,
    MiniMaxH3FL2VAVAELoader,
    MiniMaxH3Ref2VAModelLoader,
    MiniMaxH3Ref2VATextEncoderLoader,
    MiniMaxH3Ref2VAVAELoader,
)
from .api.targets import (
    MiniMaxH3FL2VATarget,
    MiniMaxH3Ref2VATarget,
    MiniMaxH3T2VATarget,
)
from .api.conditioning import (
    MiniMaxH3FL2VAEncode,
    MiniMaxH3FL2VAFirstFrameCondition,
    MiniMaxH3FL2VALastFrameCondition,
    MiniMaxH3Ref2VAAudioReference,
    MiniMaxH3Ref2VAEncode,
    MiniMaxH3Ref2VAImageReference,
    MiniMaxH3Ref2VAVideoReference,
    MiniMaxH3T2VATextEncode,
    MiniMaxH3UnsupportedConditioning,
)
from .api.sampling_nodes import (
    MiniMaxH3CombineAVLatent,
    MiniMaxH3DualSigmaSampler,
    MiniMaxH3EmptyAVLatent,
    MiniMaxH3EncodeVideoAVLatent,
    MiniMaxH3FrameRate,
    MiniMaxH3SeparateAVLatent,
)
from .api.decode import MiniMaxH3DecodeAV
from .api.attention import (
    MiniMaxH3SageAttentionPatch,
    MiniMaxH3SolAttentionPatch,
)

NODE_CLASS_MAPPINGS = {
    "RHMiniMaxH3DirectModelLoader": MiniMaxH3DirectModelLoader,
    "RHMiniMaxH3DirectTextEncoderLoader": MiniMaxH3DirectTextEncoderLoader,
    "RHMiniMaxH3DirectVAELoader": MiniMaxH3DirectVAELoader,
    "RHMiniMaxH3FL2VAModelLoader": MiniMaxH3FL2VAModelLoader,
    "RHMiniMaxH3FL2VATextEncoderLoader": MiniMaxH3FL2VATextEncoderLoader,
    "RHMiniMaxH3FL2VAVAELoader": MiniMaxH3FL2VAVAELoader,
    "RHMiniMaxH3Ref2VAModelLoader": MiniMaxH3Ref2VAModelLoader,
    "RHMiniMaxH3Ref2VATextEncoderLoader": MiniMaxH3Ref2VATextEncoderLoader,
    "RHMiniMaxH3Ref2VAVAELoader": MiniMaxH3Ref2VAVAELoader,
    "RHMiniMaxH3T2VATarget": MiniMaxH3T2VATarget,
    "RHMiniMaxH3T2VATextEncode": MiniMaxH3T2VATextEncode,
    "RHMiniMaxH3FL2VAFirstFrameCondition": MiniMaxH3FL2VAFirstFrameCondition,
    "RHMiniMaxH3FL2VALastFrameCondition": MiniMaxH3FL2VALastFrameCondition,
    "RHMiniMaxH3FL2VATarget": MiniMaxH3FL2VATarget,
    "RHMiniMaxH3FL2VAEncode": MiniMaxH3FL2VAEncode,
    "RHMiniMaxH3Ref2VAImageReference": MiniMaxH3Ref2VAImageReference,
    "RHMiniMaxH3Ref2VAAudioReference": MiniMaxH3Ref2VAAudioReference,
    "RHMiniMaxH3Ref2VAVideoReference": MiniMaxH3Ref2VAVideoReference,
    "RHMiniMaxH3Ref2VATarget": MiniMaxH3Ref2VATarget,
    "RHMiniMaxH3Ref2VAEncode": MiniMaxH3Ref2VAEncode,
    "RHMiniMaxH3UnsupportedConditioning": MiniMaxH3UnsupportedConditioning,
    "RHMiniMaxH3EmptyAVLatent": MiniMaxH3EmptyAVLatent,
    "RHMiniMaxH3SeparateAVLatent": MiniMaxH3SeparateAVLatent,
    "RHMiniMaxH3CombineAVLatent": MiniMaxH3CombineAVLatent,
    "RHMiniMaxH3EncodeVideoAVLatent": MiniMaxH3EncodeVideoAVLatent,
    "RHMiniMaxH3FrameRate": MiniMaxH3FrameRate,
    "RHMiniMaxH3DualSigmaSampler": MiniMaxH3DualSigmaSampler,
    "RHMiniMaxH3DecodeAV": MiniMaxH3DecodeAV,
    "RHMiniMaxH3SageAttentionPatch": MiniMaxH3SageAttentionPatch,
    "RHMiniMaxH3SolAttentionPatch": MiniMaxH3SolAttentionPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RHMiniMaxH3DirectModelLoader": "RunningHub MiniMax H3 Model Loader (Direct)",
    "RHMiniMaxH3DirectTextEncoderLoader": "RunningHub MiniMax H3 Qwen3-VL Loader (Direct)",
    "RHMiniMaxH3DirectVAELoader": "RunningHub MiniMax H3 Dual VAE Loader (Direct)",
    "RHMiniMaxH3FL2VAModelLoader": "RunningHub MiniMax H3 FL2VA Model Loader (Direct)",
    "RHMiniMaxH3FL2VATextEncoderLoader": "RunningHub MiniMax H3 FL2VA Qwen3-VL Loader (Direct)",
    "RHMiniMaxH3FL2VAVAELoader": "RunningHub MiniMax H3 FL2VA Dual VAE Loader (Direct)",
    "RHMiniMaxH3Ref2VAModelLoader": "RunningHub MiniMax H3 Ref2VA Model Loader (Direct)",
    "RHMiniMaxH3Ref2VATextEncoderLoader": "RunningHub MiniMax H3 Ref2VA Qwen3-VL Loader (Direct)",
    "RHMiniMaxH3Ref2VAVAELoader": "RunningHub MiniMax H3 Ref2VA Dual VAE Loader (Direct)",
    "RHMiniMaxH3T2VATarget": "RunningHub MiniMax H3 T2VA Target",
    "RHMiniMaxH3T2VATextEncode": "RunningHub MiniMax H3 T2VA Text Encode",
    "RHMiniMaxH3FL2VAFirstFrameCondition": "RunningHub MiniMax H3 FL2VA First / First+Last",
    "RHMiniMaxH3FL2VALastFrameCondition": "RunningHub MiniMax H3 FL2VA Last Only",
    "RHMiniMaxH3FL2VATarget": "RunningHub MiniMax H3 FL2VA Target",
    "RHMiniMaxH3FL2VAEncode": "RunningHub MiniMax H3 FL2VA Encode",
    "RHMiniMaxH3Ref2VAImageReference": "RunningHub MiniMax H3 Ref2VA Image Reference",
    "RHMiniMaxH3Ref2VAAudioReference": "RunningHub MiniMax H3 Ref2VA Audio Reference",
    "RHMiniMaxH3Ref2VAVideoReference": "RunningHub MiniMax H3 Ref2VA Video Reference",
    "RHMiniMaxH3Ref2VATarget": "RunningHub MiniMax H3 Ref2VA Target",
    "RHMiniMaxH3Ref2VAEncode": "RunningHub MiniMax H3 Ref2VA Encode",
    "RHMiniMaxH3UnsupportedConditioning": (
        "RunningHub MiniMax H3 Legacy Unsupported Conditioning (Migration Error)"
    ),
    "RHMiniMaxH3EmptyAVLatent": "RunningHub MiniMax H3 Empty AV Latent",
    "RHMiniMaxH3SeparateAVLatent": "RunningHub MiniMax H3 Separate AV Latent",
    "RHMiniMaxH3CombineAVLatent": "RunningHub MiniMax H3 Combine AV Latent",
    "RHMiniMaxH3EncodeVideoAVLatent": "RunningHub MiniMax H3 Encode Video → AV Latent",
    "RHMiniMaxH3FrameRate": "RunningHub MiniMax H3 Frame Rate (Experimental)",
    "RHMiniMaxH3DualSigmaSampler": "RunningHub MiniMax H3 Dual Sigma Sampler",
    "RHMiniMaxH3DecodeAV": "RunningHub MiniMax H3 Decode Video + Audio",
    "RHMiniMaxH3SageAttentionPatch": (
        "RunningHub MiniMax H3 SageAttention Patch (Bundled)"
    ),
    "RHMiniMaxH3SolAttentionPatch": (
        "RunningHub MiniMax H3 Sol-Style Sparse Attention (Experimental)"
    ),
}


__all__ = [
    "AV_LATENT_TYPE",
    "CONDITIONING_TYPE",
    "FL_KEYFRAMES_TYPE",
    "MODEL_TYPE",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "REFERENCES_TYPE",
    "TARGET_TYPE",
    "TEXT_ENCODER_TYPE",
    "VAE_BUNDLE_TYPE",
]
