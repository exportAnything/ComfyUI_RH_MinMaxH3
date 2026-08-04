"""Canonical H3 runtime constants (quantization markers, loading policy, acceleration); do not redefine elsewhere."""

MODEL_NAME = "MiniMax-H3"
DEFAULT_PARTITION = "FL2VA"
QUANT_KEY_SUFFIXES = (".comfy_quant", ".weight_scale")  # Checkpoint quantization markers.
QKV_WEIGHT_SUFFIX = ".attn.qkv_proj.weight"
QKV_SCALE_SUFFIX = ".attn.qkv_proj.weight_scale"
INT8_FORMAT = "int8_tensorwise"  # Comfy QUANT_ALGOS / comfy_quant.format
QUANT_NAME_TAG = "int8_convrot"  # Quantization-format tag used in artifact filenames.
TEXT_ENCODER_MODEL_SLUG = "qwen3-vl-32b"  # TE weight filename prefix.
# Directories: component+format; DiT file: MiniMax-H3-{FL2VA|Ref2VA}-int8_convrot;
# TE: qwen3-vl-32b-int8_convrot.
INT8_DIT_DIRNAME = f"transformer_{QUANT_NAME_TAG}"
INT8_TE_DIRNAME = f"text_encoder_{QUANT_NAME_TAG}"
VAE_MERGED_DIRNAME = "vae"  # Combined video_vae+audio_vae package.
VIDEO_VAE_FILENAME = f"{MODEL_NAME}-video_vae.safetensors"
AUDIO_VAE_FILENAME = f"{MODEL_NAME}-audio_vae.safetensors"
INT8_TE_FILENAME = f"{TEXT_ENCODER_MODEL_SLUG}-{QUANT_NAME_TAG}.safetensors"
NATIVE_FL2VA_DIT_FILENAME = "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
NATIVE_REF2VA_DIT_FILENAME = "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
NATIVE_TE_FILENAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
NATIVE_VIDEO_VAE_FILENAME = "minimax_h3_video_vae_fp16.safetensors"
NATIVE_AUDIO_VAE_FILENAME = "minimax_h3_audio_vae_fp32.safetensors"
BF16_TE_MODEL_NAME = TEXT_ENCODER_MODEL_SLUG  # Sharded BF16 logical name → text_encoder/.
VAE_MERGED_MODEL_NAME = f"{MODEL_NAME}-vae"  # Combined dual-VAE logical name → vae/.
VIDEO_VAE_DIRNAME, AUDIO_VAE_DIRNAME = "video_vae", "audio_vae"
VIDEO_VAE_MODEL_NAME = f"{MODEL_NAME}-video_vae"  # Sharded/source video VAE logical name.
AUDIO_VAE_MODEL_NAME = f"{MODEL_NAME}-audio_vae"  # Sharded/source audio VAE logical name.
# ---- Dedicated weight root ComfyUI/models/MiniMax-H3: flat converted single-file artifacts ----
# The upstream sharded release remains under models/diffusers/MiniMax-H3/<partition>/
# and supplies config.json / source/config.json / tokenizer / preprocessor. The
# dedicated root contains weights only; component type and partition are determined
# exclusively from filenames (see classify_weight_filename).
WEIGHTS_ROOT_DIRNAME = MODEL_NAME
H3_COMPONENT_KINDS = (
    "transformer",
    "text_encoder",
    VIDEO_VAE_DIRNAME,
    AUDIO_VAE_DIRNAME,
)
QUANT_EXCLUDE_HINT = "adaln_proj|token_refiner"  # Suggested --exclude for the DiT quantization script.
# ---- adaLN curve-table checkpoint (see runtime/adaln_curve.py) ----
ADALN_CURVE_TABLE_KEY = "adaln_t_table"  # Checkpoint buffer name, [grid, rank] fp32.
ADALN_CURVE_DEFAULT_GRID = 1024  # Default number of conversion samples.
ADALN_CURVE_DEFAULT_RANK = 64  # Default basis rank k for conversion (adaLN weight width 2688 → k).
ADALN_CURVE_DIRNAME_SUFFIX = "adaln_curve"  # Output-directory suffix: transformer_adaln_curve.
TEXT_ENCODER_SELECTED_LAYERS = 50  # Must match qwen_encoder.SELECTED_LAYERS.
TEXT_ENCODER_QUANT_LINEAR = (  # Quantize only language_model.layers.<N>.{self_attn.*_proj,mlp.*_proj}, N<50.
    r"^model\.language_model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)
TEXT_ENCODER_DROP_KEY = r"(?:^lm_head\.|language_model\.layers\.(?:[5-9]\d|\d{3,})\.)"  # Drop lm_head and layers >=50.
FORCE_FULL_LOAD_BF16 = True  # When layerwise is off, still load the full BF16 model onto GPU.
ALLOW_PARTIAL_OFFLOAD_INT8 = True  # MixedPrecisionOps supports partial offload.
ENABLE_DIT_LAYERWISE_OFFLOAD = True  # BF16: True=auto (disable when full model+reserve fits); False=force full model.
DIT_LAYERWISE_PREFETCH = 1  # Number of blocks to prefetch (matches upstream prefetch_size default).
DIT_LAYERWISE_PIN_MEMORY = True  # Pin block weights in host memory for nonblocking H2D.
TE_GPU_HEADROOM = 3 << 30  # Additional encode-workspace VRAM reserved before moving TE to GPU (bytes).
TE_VISUAL_ON_CPU = True  # Do not move the visual tower for text-only encoding; saves ~7GB VRAM.
DIT_INFERENCE_RESERVE = 6 << 30  # Default activation reserve when no shape hint is available.
DIT_ACTIVATION_FLOOR = 2 << 30  # Activation-budget floor.
DIT_ACTIVATION_CEIL = 24 << 30  # Activation-budget ceiling to prevent runaway estimates.
DIT_SAFETY_MARGIN = 2 << 30  # Additional safety margin during resolution.
DIT_ACTIVATION_BYTES_PER_TOKEN = 0  # 0=estimate as hidden×8×element size; set explicitly to override.
DIT_ACTIVATION_EWMA_ALPHA = 0.3  # EWMA coefficient for measured peaks.
# ---- Hot-path feature flags (rollback by changing a flag, no reinstall required) ----
OPT_SDPA_PRECOMPUTED_BOUNDS = True  # Precompute cu_seqlens bounds to remove per-layer DtoH.
OPT_PREPARED_STRUCTURE = True  # Cache RoPE/structural tensors for the session.
OPT_INPLACE_EULER_UPDATE = True  # Update target rows in place to avoid cloning the full sequence.
OPT_ADALN_SEGMENT_BROADCAST = True  # Broadcast adaLN by contiguous runs in place, avoiding materialized index_select.
OPT_ADALN_PRECOMPUTE = True  # Precompute adaLN for every timestep before sampling; inference uses table lookups.
OPT_ADALN_RELEASE_WEIGHTS = True  # Release adaln/time_embedder weights after precompute (~40% of DiT).
OPT_ADALN_CACHE_DEVICE = "auto"  # auto|ram|vram; auto uses VRAM above 40GB total, otherwise RAM.
OPT_INT8_FUSED_SWIGLU = True  # INT8 MLP: fold swiglu into the activation-quantization kernel (needs newer comfy.ops).
OPT_FUSED_QK_ROPE = True  # Attention: fused per-head RMSNorm + split-half RoPE (needs comfy-kitchen).
OPT_FUSED_QK_ROPE_CUDA_ONLY = True  # comfy-kitchen implements CUDA only; always fall back elsewhere.
OPT_PREBUILT_TIMESTEPS = True  # Prebuild contiguous sigma/timestep tensors.
OPT_DYNAMIC_ACTIVATION_RESERVE = True  # Estimate activation reserve from the canvas.
DIT_DEBUG_STRUCTURE_CHECKS = False  # True restores per-step .item() checks in forward.
# ---- Lifecycle / cache / downscaling ----
OPT_RESIDENCY_LEASE = True  # Keep a residency lease after inference instead of cold-unloading every run.
RESIDENCY_POLICY = "balanced"  # safe|balanced|resident
RESIDENCY_TTL_SECONDS = 120  # Cold-unload after the idle TTL.
RESIDENCY_VAE_TTL_SECONDS = 60  # Short VAE TTL.
OPT_ENCODE_CACHE = True  # Qwen/conditioning encoding LRU.
ENCODE_CACHE_MAX_BYTES = 2 << 30  # Text+visual feature cache limit.
OPT_VAE_RESIDENCY = True  # Reuse VAE sessions with a short TTL.
OPT_WRITE_SIDECAR = True  # Write JSON metadata alongside Decode output.
FORCE_ABSOLUTE_MODEL_ROOTS = False  # True=always use absolute COMBO paths; False=fall back only on name collisions.
# ---- Observability / benchmarks ----
OPT_TELEMETRY = True  # Stage timings + peak VRAM + per-step statistics.
TELEMETRY_CUDA_EVENTS = True  # Dual timing with CUDA events (automatically degrades without CUDA).
TELEMETRY_STEP_ABORT_SECONDS = 75.0  # Flag anomaly after two consecutive slow steps (matrix slow path may be exempt).
TELEMETRY_STALL_SECONDS = 900.0  # Sampling-without-progress threshold.
BENCHMARK_DEFAULT_SEED = 42
BENCHMARK_DEFAULT_REPEATS = 3
BENCHMARK_ACCEL_GOLDEN = "off"  # Golden runs must use accel=off.
# Upstream 16:9 downscale chain; other ratios preserve aspect using short-edge steps aligned to 32.
H3_DOWNSCALE_16_9 = ((1344, 768), (1024, 576), (832, 480), (640, 352))
H3_DOWNSCALE_SHORT_EDGES = (768, 576, 480, 352)
FFMPEG_BIN, FFPROBE_BIN = "ffmpeg", "ffprobe"  # Ref2VA media tools.
TRANSFORMERS_MIN_VERSION = "4.57.0"  # Lowest validated Qwen3-VL version.
TRANSFORMERS_MAX_VERSION = "5.8.1"  # Current inclusive upper pin.
# ---- Sampler (euler=upstream 50-step first order; res_multistep=second-order multistep, ~21 sigma points at equal quality) ----
SAMPLER_MODE_EULER = "euler"
SAMPLER_MODE_RES_MULTISTEP = "res_multistep"
SAMPLER_MODE_CHOICES = (SAMPLER_MODE_EULER, SAMPLER_MODE_RES_MULTISTEP)

# ---- Sampling acceleration (approximate; off by default; Cache-DiT and velocity-cache are mutually exclusive) ----
ACCEL_OFF, ACCEL_AUTO = "off", "auto"
ACCEL_CACHE_DIT_PROFILE = "minimax-h3-cache-v1"
ACCEL_VELOCITY_PROFILE = "minimax-h3-velocity-cache-v1"
ACCEL_MANUAL_CACHE_DIT, ACCEL_MANUAL_VELOCITY = "manual-cache-dit", "manual-velocity"
ACCEL_MODE_CHOICES = (
    ACCEL_OFF, ACCEL_AUTO, ACCEL_VELOCITY_PROFILE, ACCEL_CACHE_DIT_PROFILE,
    ACCEL_MANUAL_VELOCITY, ACCEL_MANUAL_CACHE_DIT,
)
# Legacy workflow field-name aliases.
CACHE_DIT_MODE_OFF, CACHE_DIT_MODE_AUTO, CACHE_DIT_MODE_MANUAL = ACCEL_OFF, ACCEL_AUTO, ACCEL_MANUAL_CACHE_DIT
CACHE_DIT_PROFILE_ID = ACCEL_CACHE_DIT_PROFILE
CACHE_DIT_MODE_CHOICES = ACCEL_MODE_CHOICES
CACHE_DIT_PKG, CACHE_DIT_MIN_VERSION = "cache-dit", "1.3.0"
CACHE_DIT_BLOCKS_ATTR, CACHE_DIT_FORWARD_PATTERN = "blocks", "Pattern_3"
CACHE_DIT_FN, CACHE_DIT_BN, CACHE_DIT_WARMUP = 1, 0, 4
CACHE_DIT_RDT_COOKBOOK, CACHE_DIT_RDT_PROFILE = 0.12, 0.08
CACHE_DIT_MC = 2
CACHE_DIT_TAYLORSEER, CACHE_DIT_TS_ORDER = False, 1
CACHE_DIT_SCM_PRESET, CACHE_DIT_SCM_POLICY = "none", "dynamic"
CACHE_DIT_MARK = "_h3_cache_dit_enabled"
# Upstream-validated velocity-cache controls (upstream validation manifest).
VELOCITY_STRIDE, VELOCITY_TAYLORSEER, VELOCITY_TS_ORDER = 4, True, 1
VELOCITY_TAIL_DENSE, VELOCITY_TAIL_REBALANCE, VELOCITY_FINAL_REFRESH = 2, True, True
# ---- Reference-image sizing policy ----
# match: downscale only to the generated canvas area while preserving aspect ratio;
# max: independent 2048-pixel short edge for the reference pipeline. Reference tokens
# participate in attention at every sampling step, so resolution directly affects per-step cost.
H3_REFERENCE_IMAGE_SIZE_MATCH = "match"
H3_REFERENCE_IMAGE_SIZE_MAX = "max"
H3_REFERENCE_IMAGE_SIZE_MODES = (
    H3_REFERENCE_IMAGE_SIZE_MATCH,
    H3_REFERENCE_IMAGE_SIZE_MAX,
)
# ---- Experimental frame-rate conditioning (not an upstream contract; does not change the 24 fps time grid) ----
H3_NATIVE_FPS = 24.0
FRAME_RATE_ROPE_FREQ_PROFILES = ("hard", "linear", "smoothstep")
FRAME_RATE_ROPE_SIGMA_PROFILES = ("constant", "linear", "smoothstep")


def classify_weight_filename(name: str) -> tuple[str, str | None] | None:
    """Map a flat weight filename to ``(component kind, partition | None)``.

    The converters deliberately encode model, component type and quantization
    format in the filename, so the flat weights root needs no sidecar to be
    classified.  ``None`` partition means the weights serve both partitions.
    """

    stem = str(name).strip().replace("\\", "/").rsplit("/", 1)[-1]
    if not stem.lower().endswith(".safetensors") or "-of-" in stem.lower():
        return None
    stem = stem[: -len(".safetensors")]
    # Comfy's native H3 conversions use underscores while the original
    # RunningHub converter uses hyphens.  Normalize both spellings for type and
    # partition classification; the original spelling remains available to the
    # format-specific loader gate below.
    lowered = stem.lower()
    normalized = lowered.replace("_", "-")
    model_prefix = f"{MODEL_NAME.lower()}-"
    if normalized.startswith(model_prefix):
        suffix = normalized[len(model_prefix) :]
        for partition in ("fl2va", "ref2va"):
            if suffix == partition or suffix.startswith(f"{partition}-"):
                return ("transformer", partition)
        video_name = VIDEO_VAE_DIRNAME.replace("_", "-")
        audio_name = AUDIO_VAE_DIRNAME.replace("_", "-")
        if suffix == video_name or suffix.startswith(f"{video_name}-"):
            return (VIDEO_VAE_DIRNAME, None)
        if suffix == audio_name or suffix.startswith(f"{audio_name}-"):
            return (AUDIO_VAE_DIRNAME, None)
        return None
    converted_text_prefix = TEXT_ENCODER_MODEL_SLUG.replace("_", "-")
    comfy_text_prefix = "qwen3vl-32b-minimax-h3"
    if (
        normalized == converted_text_prefix
        or normalized.startswith(f"{converted_text_prefix}-")
        or normalized == comfy_text_prefix
        or normalized.startswith(f"{comfy_text_prefix}-")
    ):
        return ("text_encoder", None)
    return None


def is_comfy_native_weight_filename(
    name: str,
    kind: str | None = None,
) -> bool:
    """Return whether ``name`` follows Comfy's native single-file H3 naming.

    This deliberately distinguishes native files from RunningHub's
    ``*-int8_convrot.safetensors`` artifacts.  Both are classified above, but
    they require different configuration and loading paths.
    """

    text = str(name).strip().lower().replace("\\", "/").rsplit("/", 1)[-1]
    if not text.endswith(".safetensors") or "-of-" in text:
        return False
    classified = classify_weight_filename(text)
    if classified is None:
        return False
    file_kind, _partition = classified
    if kind is not None and file_kind != str(kind).strip().lower():
        return False
    stem = text[: -len(".safetensors")]
    if file_kind == "transformer":
        return stem.startswith("minimax_h3_")
    if file_kind == "text_encoder":
        return stem.startswith("qwen3vl_32b_minimax_h3_")
    if file_kind == VIDEO_VAE_DIRNAME:
        return stem.startswith("minimax_h3_video_vae_")
    if file_kind == AUDIO_VAE_DIRNAME:
        return stem.startswith("minimax_h3_audio_vae_")
    return False


def int8_dit_filename(partition: str | None = None) -> str:  # MiniMax-H3 + model type + quantization format.
    return f"{MODEL_NAME}-{partition or DEFAULT_PARTITION}-{QUANT_NAME_TAG}.safetensors"


def bf16_dit_model_name(partition: str | None = None) -> str:  # Sharded BF16 DiT logical name → transformer/.
    return f"{MODEL_NAME}-{partition or DEFAULT_PARTITION}"


INT8_DIT_FILENAME = int8_dit_filename(DEFAULT_PARTITION)  # Defaults to FL2VA; scripts can override by partition.
BF16_DIT_MODEL_NAME = bf16_dit_model_name(DEFAULT_PARTITION)
