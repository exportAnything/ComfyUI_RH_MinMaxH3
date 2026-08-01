"""H3 runtime 统一常量（量化标记 / 加载策略），禁止在别处重复定义。"""

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
QUANT_EXCLUDE_HINT = "adaln_proj|token_refiner"  # DiT 量化脚本建议 --exclude
TEXT_ENCODER_SELECTED_LAYERS = 50  # 与 qwen_encoder.SELECTED_LAYERS 一致
TEXT_ENCODER_QUANT_LINEAR = (  # 仅量化 language_model.layers.<N>.{self_attn.*_proj,mlp.*_proj}，N<50
    r"^model\.language_model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$"
)
TEXT_ENCODER_DROP_KEY = r"(?:^lm_head\.|language_model\.layers\.(?:[5-9]\d|\d{3,})\.)"  # 丢弃 lm_head 与 layer>=50
FORCE_FULL_LOAD_BF16 = True  # BF16 普通 Linear 必须整模上卡
ALLOW_PARTIAL_OFFLOAD_INT8 = True  # MixedPrecisionOps 可部分 offload
TE_GPU_HEADROOM = 3 << 30  # TE 上卡前额外预留的 encode 工作区显存（字节）
TE_VISUAL_ON_CPU = True  # 纯文本编码不搬 visual 塔，常驻 CPU 省 ~7GB 显存
DIT_INFERENCE_RESERVE = 6 << 30  # DiT partial load 时给采样激活预留的显存（字节）
# Cache-DiT（官方 MiniMax-H3 BlockAdapter / quality profile；近似加速，默认关闭）
CACHE_DIT_PKG = "cache-dit"
CACHE_DIT_MIN_VERSION = "1.3.0"
CACHE_DIT_MODE_OFF, CACHE_DIT_MODE_AUTO, CACHE_DIT_MODE_MANUAL = "off", "auto", "manual"
CACHE_DIT_PROFILE_ID = "minimax-h3-cache-v1"  # 本地 profile；源自官方 h200x4-cache-v1 旋钮
CACHE_DIT_MODE_CHOICES = (CACHE_DIT_MODE_OFF, CACHE_DIT_MODE_AUTO, CACHE_DIT_PROFILE_ID, CACHE_DIT_MODE_MANUAL)
CACHE_DIT_BLOCKS_ATTR = "blocks"  # MiniMaxH3DiTModel 主栈
CACHE_DIT_FORWARD_PATTERN = "Pattern_3"  # hidden-state in/out，与官方测试一致
CACHE_DIT_FN, CACHE_DIT_BN, CACHE_DIT_WARMUP = 1, 0, 4  # cookbook / profile 共用
CACHE_DIT_RDT_COOKBOOK, CACHE_DIT_RDT_PROFILE = 0.12, 0.08  # cookbook 保守 / 官方已验证
CACHE_DIT_MC = 2
CACHE_DIT_TAYLORSEER, CACHE_DIT_TS_ORDER = False, 1
CACHE_DIT_SCM_PRESET, CACHE_DIT_SCM_POLICY = "none", "dynamic"
CACHE_DIT_MARK = "_h3_cache_dit_enabled"  # 挂在 transformer 上，避免重复 enable


def int8_dit_filename(partition: str | None = None) -> str:  # MiniMax-H3+模型类型+量化格式
    return f"{MODEL_NAME}-{partition or DEFAULT_PARTITION}-{QUANT_NAME_TAG}.safetensors"


def bf16_dit_model_name(partition: str | None = None) -> str:  # 分片 BF16 DiT 逻辑名 → transformer/
    return f"{MODEL_NAME}-{partition or DEFAULT_PARTITION}"


INT8_DIT_FILENAME = int8_dit_filename(DEFAULT_PARTITION)  # 默认 FL2VA，脚本可按分区覆盖
BF16_DIT_MODEL_NAME = bf16_dit_model_name(DEFAULT_PARTITION)
