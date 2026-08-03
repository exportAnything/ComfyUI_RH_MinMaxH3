"""H3 runtime 统一常量（量化标记 / 加载策略 / 加速），禁止在别处重复定义。"""

MODEL_NAME = "MiniMax-H3"
DEFAULT_PARTITION = "FL2VA"
QUANT_KEY_SUFFIXES = (".comfy_quant", ".weight_scale")  # checkpoint 量化标记
QKV_WEIGHT_SUFFIX = ".attn.qkv_proj.weight"
QKV_SCALE_SUFFIX = ".attn.qkv_proj.weight_scale"
INT8_FORMAT = "int8_tensorwise"  # Comfy QUANT_ALGOS / comfy_quant.format
QUANT_NAME_TAG = "int8_convrot"  # 产物文件名用的量化格式标签
TEXT_ENCODER_MODEL_SLUG = "qwen3-vl-32b"  # TE 权重文件名前缀
# 目录：组件+格式；DiT 文件：MiniMax-H3-{FL2VA|Ref2VA}-int8_convrot；TE：qwen3-vl-32b-int8_convrot
INT8_DIT_DIRNAME = f"transformer_{QUANT_NAME_TAG}"
INT8_TE_DIRNAME = f"text_encoder_{QUANT_NAME_TAG}"
VAE_MERGED_DIRNAME = "vae"  # video_vae+audio_vae 合并包
VIDEO_VAE_FILENAME = f"{MODEL_NAME}-video_vae.safetensors"
AUDIO_VAE_FILENAME = f"{MODEL_NAME}-audio_vae.safetensors"
INT8_TE_FILENAME = f"{TEXT_ENCODER_MODEL_SLUG}-{QUANT_NAME_TAG}.safetensors"
BF16_TE_MODEL_NAME = TEXT_ENCODER_MODEL_SLUG  # 分片 BF16 逻辑名 → text_encoder/
VAE_MERGED_MODEL_NAME = f"{MODEL_NAME}-vae"  # 合并双 VAE 逻辑名 → vae/
VIDEO_VAE_DIRNAME, AUDIO_VAE_DIRNAME = "video_vae", "audio_vae"
VIDEO_VAE_MODEL_NAME = f"{MODEL_NAME}-video_vae"  # 分片/原始 video VAE 逻辑名
AUDIO_VAE_MODEL_NAME = f"{MODEL_NAME}-audio_vae"  # 分片/原始 audio VAE 逻辑名
# ---- 专属权重根 ComfyUI/models/MiniMax-H3：扁平存放单文件转换产物 ----
# 官方分片 release 留在 models/diffusers/MiniMax-H3/<分区>/ 下，继续提供
# config.json / source/config.json / tokenizer / preprocessor；专属根只放权重，
# 组件类型与分区一律由文件名判定（见 classify_weight_filename）。
WEIGHTS_ROOT_DIRNAME = MODEL_NAME
H3_COMPONENT_KINDS = (
    "transformer",
    "text_encoder",
    VIDEO_VAE_DIRNAME,
    AUDIO_VAE_DIRNAME,
)
QUANT_EXCLUDE_HINT = "adaln_proj|token_refiner"  # DiT 量化脚本建议 --exclude
# ---- adaLN 曲线表 checkpoint（PR#15224；见 runtime/adaln_curve.py）----
ADALN_CURVE_TABLE_KEY = "adaln_t_table"  # checkpoint buffer 名，[grid, rank] fp32
ADALN_CURVE_DEFAULT_GRID = 1024  # 转换默认采样点数
ADALN_CURVE_DEFAULT_RANK = 64  # 转换默认基秩 k（adaLN 权重宽度 2688 → k）
ADALN_CURVE_DIRNAME_SUFFIX = "adaln_curve"  # 输出目录后缀：transformer_adaln_curve
TEXT_ENCODER_SELECTED_LAYERS = 50  # 与 qwen_encoder.SELECTED_LAYERS 一致
TEXT_ENCODER_QUANT_LINEAR = (  # 仅量化 language_model.layers.<N>.{self_attn.*_proj,mlp.*_proj}，N<50
    r"^model\.language_model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)
TEXT_ENCODER_DROP_KEY = r"(?:^lm_head\.|language_model\.layers\.(?:[5-9]\d|\d{3,})\.)"  # 丢弃 lm_head 与 layer>=50
FORCE_FULL_LOAD_BF16 = True  # layerwise 关闭时 BF16 仍整模上卡
ALLOW_PARTIAL_OFFLOAD_INT8 = True  # MixedPrecisionOps 可部分 offload
ENABLE_DIT_LAYERWISE_OFFLOAD = True  # BF16：True=auto（整模+reserve 放得下则关）；False=强制整模
DIT_LAYERWISE_PREFETCH = 1  # 预取 block 数（与官方 prefetch_size 默认一致）
DIT_LAYERWISE_PIN_MEMORY = True  # block 权重 pin 到主机内存，便于非阻塞 H2D
TE_GPU_HEADROOM = 3 << 30  # TE 上卡前额外预留的 encode 工作区显存（字节）
TE_VISUAL_ON_CPU = True  # 纯文本编码不搬 visual 塔，常驻 CPU 省 ~7GB 显存
DIT_INFERENCE_RESERVE = 6 << 30  # 无 shape 提示时的默认激活预留
DIT_ACTIVATION_FLOOR = 2 << 30  # 激活预算下限
DIT_ACTIVATION_CEIL = 24 << 30  # 激活预算上限（防估炸）
DIT_SAFETY_MARGIN = 2 << 30  # resolve 时额外安全余量
DIT_ACTIVATION_BYTES_PER_TOKEN = 0  # 0=按 hidden×8×elem 估算；可钉死覆盖
DIT_ACTIVATION_EWMA_ALPHA = 0.3  # 实测峰值 EWMA 系数
# ---- 热路径 feature flags（回滚只切 flag，不重装环境）----
OPT_SDPA_PRECOMPUTED_BOUNDS = True  # 预计算 cu_seqlens bounds，消除每层 DtoH
OPT_PREPARED_STRUCTURE = True  # session 缓存 RoPE/结构张量
OPT_INPLACE_EULER_UPDATE = True  # 原位更新 target rows，避免整序列 clone
OPT_ADALN_SEGMENT_BROADCAST = True  # adaLN 按连续段广播 in-place，避免 index_select 物化
OPT_ADALN_PRECOMPUTE = True  # 采样前预计算全部 timestep 的 adaLN，推理只查表
OPT_ADALN_RELEASE_WEIGHTS = True  # 预计算后释放 adaln/time_embedder 权重（约 40% DiT）
OPT_ADALN_CACHE_DEVICE = "auto"  # auto|ram|vram；auto：总显存>40GB 放 VRAM 否则 RAM
OPT_INT8_FUSED_SWIGLU = True  # INT8 MLP：swiglu 折进激活量化 kernel（需新版 comfy.ops）
OPT_FUSED_QK_ROPE = True  # 注意力：融合 per-head RMSNorm + split-half RoPE（需 comfy-kitchen）
OPT_FUSED_QK_ROPE_CUDA_ONLY = True  # comfy-kitchen 只有 CUDA 实现，非 CUDA 一律回退
OPT_PREBUILT_TIMESTEPS = True  # 预生成连续 sigma/timestep 张量
OPT_DYNAMIC_ACTIVATION_RESERVE = True  # 按画布估算激活预留
DIT_DEBUG_STRUCTURE_CHECKS = False  # True 时 forward 每步恢复 .item() 校验
# ---- 生命周期 / 缓存 / 降档 ----
OPT_RESIDENCY_LEASE = True  # 推理后租约驻留，非每次冷卸载
RESIDENCY_POLICY = "balanced"  # safe|balanced|resident
RESIDENCY_TTL_SECONDS = 120  # 空闲 TTL 后冷卸载
RESIDENCY_VAE_TTL_SECONDS = 60  # VAE 短 TTL
OPT_ENCODE_CACHE = True  # Qwen/条件编码 LRU
ENCODE_CACHE_MAX_BYTES = 2 << 30  # 文本+视觉特征缓存上限
OPT_VAE_RESIDENCY = True  # VAE session 短 TTL 复用
OPT_WRITE_SIDECAR = True  # Decode 旁写 JSON 元数据
FORCE_ABSOLUTE_MODEL_ROOTS = False  # True=COMBO 一律写绝对路径；False 时同名冲突才回退绝对路径
# ---- 可观测性 / 基准 ----
OPT_TELEMETRY = True  # 阶段计时 + 峰值显存 + 每步统计
TELEMETRY_CUDA_EVENTS = True  # CUDA Event 双计时（无 CUDA 时自动降级）
TELEMETRY_STEP_ABORT_SECONDS = 75.0  # 连续两步超过则标记异常（矩阵慢路径可豁免）
TELEMETRY_STALL_SECONDS = 900.0  # 采样无步进判定
BENCHMARK_DEFAULT_SEED = 42
BENCHMARK_DEFAULT_REPEATS = 3
BENCHMARK_ACCEL_GOLDEN = "off"  # golden 必须 accel=off
# 16:9 官方降档链；其它比例按短边序列保比例 32 对齐
H3_DOWNSCALE_16_9 = ((1344, 768), (1024, 576), (832, 480), (640, 352))
H3_DOWNSCALE_SHORT_EDGES = (768, 576, 480, 352)
FFMPEG_BIN, FFPROBE_BIN = "ffmpeg", "ffprobe"  # Ref2VA 媒体工具
TRANSFORMERS_MIN_VERSION = "4.57.0"  # Qwen3-VL 最低验证
TRANSFORMERS_MAX_VERSION = "5.8.1"  # 当前钉死上限（含）
# ---- 采样器（euler=官方 50 步一阶；res_multistep=二阶多步，~21 sigma 点等质）----
SAMPLER_MODE_EULER = "euler"
SAMPLER_MODE_RES_MULTISTEP = "res_multistep"
SAMPLER_MODE_CHOICES = (SAMPLER_MODE_EULER, SAMPLER_MODE_RES_MULTISTEP)

# ---- 采样加速（近似；默认 off；Cache-DiT / velocity-cache 互斥）----
ACCEL_OFF, ACCEL_AUTO = "off", "auto"
ACCEL_CACHE_DIT_PROFILE = "minimax-h3-cache-v1"
ACCEL_VELOCITY_PROFILE = "minimax-h3-velocity-cache-v1"
ACCEL_MANUAL_CACHE_DIT, ACCEL_MANUAL_VELOCITY = "manual-cache-dit", "manual-velocity"
ACCEL_MODE_CHOICES = (
    ACCEL_OFF, ACCEL_AUTO, ACCEL_VELOCITY_PROFILE, ACCEL_CACHE_DIT_PROFILE,
    ACCEL_MANUAL_VELOCITY, ACCEL_MANUAL_CACHE_DIT,
)
# 旧工作流字段名别名
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
# velocity-cache 官方验证旋钮（PR#20 manifest）
VELOCITY_STRIDE, VELOCITY_TAYLORSEER, VELOCITY_TS_ORDER = 4, True, 1
VELOCITY_TAIL_DENSE, VELOCITY_TAIL_REBALANCE, VELOCITY_FINAL_REFRESH = 2, True, True
# ---- 参考图尺寸策略（PR#15224）----
# match：按生成画布像素面积等比只缩不放；max：参考管线独立的 2048 短边。
# 参考 token 每个采样步都参与注意力，分辨率直接换算成每步开销。
H3_REFERENCE_IMAGE_SIZE_MATCH = "match"
H3_REFERENCE_IMAGE_SIZE_MAX = "max"
H3_REFERENCE_IMAGE_SIZE_MODES = (
    H3_REFERENCE_IMAGE_SIZE_MATCH,
    H3_REFERENCE_IMAGE_SIZE_MAX,
)
# ---- 实验性帧率条件（PR#15210；非官方，不改 24fps 时序格）----
H3_NATIVE_FPS = 24.0
FRAME_RATE_ROPE_FREQ_PROFILES = ("hard", "linear", "smoothstep")
FRAME_RATE_ROPE_SIGMA_PROFILES = ("constant", "linear", "smoothstep")


def classify_weight_filename(name: str) -> tuple[str, str | None] | None:
    """Map a flat weight filename to ``(component kind, partition | None)``.

    The converters deliberately encode model, component type and quantization
    format in the filename, so the flat weights root needs no sidecar to be
    classified.  ``None`` partition means the weights serve both partitions.
    """

    stem = str(name).strip()
    if not stem.endswith(".safetensors") or "-of-" in stem:
        return None
    stem = stem[: -len(".safetensors")]
    lowered = stem.lower()
    if lowered.startswith(f"{MODEL_NAME.lower()}-"):
        suffix = lowered[len(MODEL_NAME) + 1 :]
        for partition in ("fl2va", "ref2va"):
            if suffix == partition or suffix.startswith(f"{partition}-"):
                return ("transformer", partition)
        if suffix == VIDEO_VAE_DIRNAME or suffix.startswith(f"{VIDEO_VAE_DIRNAME}-"):
            return (VIDEO_VAE_DIRNAME, None)
        if suffix == AUDIO_VAE_DIRNAME or suffix.startswith(f"{AUDIO_VAE_DIRNAME}-"):
            return (AUDIO_VAE_DIRNAME, None)
        return None
    if lowered == TEXT_ENCODER_MODEL_SLUG or lowered.startswith(
        f"{TEXT_ENCODER_MODEL_SLUG}-"
    ):
        return ("text_encoder", None)
    return None


def int8_dit_filename(partition: str | None = None) -> str:  # MiniMax-H3+模型类型+量化格式
    return f"{MODEL_NAME}-{partition or DEFAULT_PARTITION}-{QUANT_NAME_TAG}.safetensors"


def bf16_dit_model_name(partition: str | None = None) -> str:  # 分片 BF16 DiT 逻辑名 → transformer/
    return f"{MODEL_NAME}-{partition or DEFAULT_PARTITION}"


INT8_DIT_FILENAME = int8_dit_filename(DEFAULT_PARTITION)  # 默认 FL2VA，脚本可按分区覆盖
BF16_DIT_MODEL_NAME = bf16_dit_model_name(DEFAULT_PARTITION)
