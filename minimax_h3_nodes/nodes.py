"""ComfyUI nodes facade：节点注册表与兼容导入；实现见 api/。"""
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

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3DirectModelLoader": MiniMaxH3DirectModelLoader,
    "MiniMaxH3DirectTextEncoderLoader": MiniMaxH3DirectTextEncoderLoader,
    "MiniMaxH3DirectVAELoader": MiniMaxH3DirectVAELoader,
    "MiniMaxH3FL2VAModelLoader": MiniMaxH3FL2VAModelLoader,
    "MiniMaxH3FL2VATextEncoderLoader": MiniMaxH3FL2VATextEncoderLoader,
    "MiniMaxH3FL2VAVAELoader": MiniMaxH3FL2VAVAELoader,
    "MiniMaxH3Ref2VAModelLoader": MiniMaxH3Ref2VAModelLoader,
    "MiniMaxH3Ref2VATextEncoderLoader": MiniMaxH3Ref2VATextEncoderLoader,
    "MiniMaxH3Ref2VAVAELoader": MiniMaxH3Ref2VAVAELoader,
    "MiniMaxH3T2VATarget": MiniMaxH3T2VATarget,
    "MiniMaxH3T2VATextEncode": MiniMaxH3T2VATextEncode,
    "MiniMaxH3FL2VAFirstFrameCondition": MiniMaxH3FL2VAFirstFrameCondition,
    "MiniMaxH3FL2VALastFrameCondition": MiniMaxH3FL2VALastFrameCondition,
    "MiniMaxH3FL2VATarget": MiniMaxH3FL2VATarget,
    "MiniMaxH3FL2VAEncode": MiniMaxH3FL2VAEncode,
    "MiniMaxH3Ref2VAImageReference": MiniMaxH3Ref2VAImageReference,
    "MiniMaxH3Ref2VAAudioReference": MiniMaxH3Ref2VAAudioReference,
    "MiniMaxH3Ref2VAVideoReference": MiniMaxH3Ref2VAVideoReference,
    "MiniMaxH3Ref2VATarget": MiniMaxH3Ref2VATarget,
    "MiniMaxH3Ref2VAEncode": MiniMaxH3Ref2VAEncode,
    "MiniMaxH3UnsupportedConditioning": MiniMaxH3UnsupportedConditioning,
    "MiniMaxH3EmptyAVLatent": MiniMaxH3EmptyAVLatent,
    "MiniMaxH3SeparateAVLatent": MiniMaxH3SeparateAVLatent,
    "MiniMaxH3CombineAVLatent": MiniMaxH3CombineAVLatent,
    "MiniMaxH3EncodeVideoAVLatent": MiniMaxH3EncodeVideoAVLatent,
    "MiniMaxH3FrameRate": MiniMaxH3FrameRate,
    "MiniMaxH3DualSigmaSampler": MiniMaxH3DualSigmaSampler,
    "MiniMaxH3DecodeAV": MiniMaxH3DecodeAV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3DirectModelLoader": "MiniMax H3 Model Loader (Direct)",
    "MiniMaxH3DirectTextEncoderLoader": "MiniMax H3 Qwen3-VL Loader (Direct)",
    "MiniMaxH3DirectVAELoader": "MiniMax H3 Dual VAE Loader (Direct)",
    "MiniMaxH3FL2VAModelLoader": "MiniMax H3 FL2VA Model Loader (Direct)",
    "MiniMaxH3FL2VATextEncoderLoader": "MiniMax H3 FL2VA Qwen3-VL Loader (Direct)",
    "MiniMaxH3FL2VAVAELoader": "MiniMax H3 FL2VA Dual VAE Loader (Direct)",
    "MiniMaxH3Ref2VAModelLoader": "MiniMax H3 Ref2VA Model Loader (Direct)",
    "MiniMaxH3Ref2VATextEncoderLoader": "MiniMax H3 Ref2VA Qwen3-VL Loader (Direct)",
    "MiniMaxH3Ref2VAVAELoader": "MiniMax H3 Ref2VA Dual VAE Loader (Direct)",
    "MiniMaxH3T2VATarget": "MiniMax H3 T2VA Target",
    "MiniMaxH3T2VATextEncode": "MiniMax H3 T2VA Text Encode",
    "MiniMaxH3FL2VAFirstFrameCondition": "MiniMax H3 FL2VA First / First+Last",
    "MiniMaxH3FL2VALastFrameCondition": "MiniMax H3 FL2VA Last Only",
    "MiniMaxH3FL2VATarget": "MiniMax H3 FL2VA Target",
    "MiniMaxH3FL2VAEncode": "MiniMax H3 FL2VA Encode",
    "MiniMaxH3Ref2VAImageReference": "MiniMax H3 Ref2VA Image Reference",
    "MiniMaxH3Ref2VAAudioReference": "MiniMax H3 Ref2VA Audio Reference",
    "MiniMaxH3Ref2VAVideoReference": "MiniMax H3 Ref2VA Video Reference",
    "MiniMaxH3Ref2VATarget": "MiniMax H3 Ref2VA Target",
    "MiniMaxH3Ref2VAEncode": "MiniMax H3 Ref2VA Encode",
    "MiniMaxH3UnsupportedConditioning": (
        "MiniMax H3 Legacy Unsupported Conditioning (Migration Error)"
    ),
    "MiniMaxH3EmptyAVLatent": "MiniMax H3 Empty AV Latent",
    "MiniMaxH3SeparateAVLatent": "MiniMax H3 Separate AV Latent",
    "MiniMaxH3CombineAVLatent": "MiniMax H3 Combine AV Latent",
    "MiniMaxH3EncodeVideoAVLatent": "MiniMax H3 Encode Video → AV Latent",
    "MiniMaxH3FrameRate": "MiniMax H3 Frame Rate (Experimental)",
    "MiniMaxH3DualSigmaSampler": "MiniMax H3 Dual Sigma Sampler",
    "MiniMaxH3DecodeAV": "MiniMax H3 Decode Video + Audio",
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
