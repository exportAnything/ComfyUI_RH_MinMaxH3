# ComfyUI-MiniMax-H3

ComfyUI 进程内直接推理 MiniMax-H3（T2VA）的自定义节点，不依赖 SGLang / Diffusers Pipeline。

## 要求

- ComfyUI ≥ 0.27（原生 `int8_tensorwise` + convrot；推荐 0.28+）
- `comfy-kitchen`、PyTorch CUDA、Triton
- 权重目录：`models/diffusers/MiniMax-H3/{FL2VA|Ref2VA}/`（含 `transformer` / `text_encoder` / `video_vae` / `audio_vae`）
- 注意：`tools/` 不是 Python 包（无 `__init__.py`），避免遮蔽 ComfyUI 自带的 `tools` 模块

## 节点用法（T2VA）

1. `MiniMax H3 Direct Model Loader`：在 `transformer_path` 明确选择 DiT
2. `MiniMax H3 Direct Text Encoder Loader`：在 `text_encoder_path` 明确选择 Qwen3-VL 条件编码器
3. `MiniMax H3 Direct VAE Loader`：在 `vae_path` 明确选择合并后的 video/audio VAE 包
4. `T2VA Target` → `T2VA Text Encode` → `Empty AV Latent` → `Dual Sigma Sampler` → `Decode AV`

### 量化产物命名

| 组件 | 目录 | 权重文件 |
|------|------|----------|
| DiT | `transformer_int8_convrot` | `MiniMax-H3-{FL2VA\|Ref2VA}-int8_convrot.safetensors` |
| Text Encoder | `text_encoder_int8_convrot` | `qwen3-vl-32b-int8_convrot.safetensors` |
| VAE（合并包） | `vae/` | `MiniMax-H3-video_vae.safetensors` / `MiniMax-H3-audio_vae.safetensors` |

三个 Loader 的 `transformer_path` / `text_encoder_path` / `vae_path` 都是**必填下拉直选**，不存在 `auto` 模型选择。推荐分别选择 `transformer_int8_convrot`、`text_encoder_int8_convrot`、`vae`。加载器识别 `.comfy_quant` / `.weight_scale`，Comfy 内部 format 仍为 `int8_tensorwise`。

Text Encoder 上卡策略（`h3_settings.TE_GPU_HEADROOM` / `TE_VISUAL_ON_CPU`）：INT8 路径按真实 `qdata + scale` 字节数统计权重，embedding / norm / RoPE 常驻 GPU，350 个量化 Linear 由 Comfy ModelPatcher 按剩余显存部分常驻，其余逐层流式上卡；visual 塔默认常驻 CPU（纯文本编码用不到，省 ~7GB）。只有 CUDA OOM 时才清理并回退 CPU encode。BF16 路径仍需整个文本编码器容量。

DiT 上卡策略（`h3_settings.DIT_INFERENCE_RESERVE`，默认 6GB）：int8 partial load 时把该额度留给采样激活，权重按剩余显存部分上卡（其余留 CPU 由 MixedPrecisionOps 兜底）；上卡 OOM 自动清卡并以 2 倍预留重试一次。

## 生成 int8_convrot 权重 / 合并 VAE

```bash
cd custom_nodes/ComfyUI-MiniMax-H3
BASE=/path/to/models/diffusers/MiniMax-H3

# DiT（按分区分别量化；文件名自动带 FL2VA/Ref2VA）
python3 tools/quantize_int8_convrot.py --src $BASE/FL2VA/transformer --device cuda --verify
python3 tools/quantize_int8_convrot.py --src $BASE/Ref2VA/transformer --device cuda --verify

# Text Encoder（FL2VA/Ref2VA 权重相同可只转一次，再复制目录）
python3 tools/quantize_text_encoder_int8_convrot.py --src $BASE/FL2VA/text_encoder --device cuda --verify

# 合并 video_vae + audio_vae，权重名带类型
python3 tools/merge_vae.py --src $BASE/FL2VA
python3 tools/merge_vae.py --src $BASE/Ref2VA
```

常用参数：`--dry-run` / `--verify` / `--exclude` / `--no-mseclip` / `--partition`。

行为摘要：

- DiT：结构推导可量化 Linear；`adaln_proj|token_refiner` 默认排除；Hadamard+per-row int8
- Text Encoder：量化 `language_model.layers[0..49]` 的 attn/mlp；visual/embed/norm 透传
- VAE：不量化，仅合并打包并规范 `MiniMax-H3-{video|audio}_vae.safetensors`

### 显存 / 磁盘

| 阶段 | 需求 |
|------|------|
| 量化 | 逐层上 GPU，4090 24GB 足够；系统内存建议 ≥ 48GB |
| 推理 | int8 DiT ~44GB；int8 TE ~25.3GiB；4090D 均使用 partial/streaming offload |
| 磁盘 | BF16 transformer ~59–62GB；int8 DiT ~44GB；int8 TE ~26GB；VAE 合并包 ~10.4GB |

240.2 使用官方校验通过的干净权重实测：

| 产物 | 规模 | 耗时 | mean relerr |
|------|------|------|-------------|
| `MiniMax-H3-FL2VA-int8_convrot.safetensors` | 201 Linear / 47.0GB | 475s | 0.915% |
| `MiniMax-H3-Ref2VA-int8_convrot.safetensors` | 201 Linear / 47.0GB | 500s | 0.915% |
| `qwen3-vl-32b-int8_convrot.safetensors` | 350 Linear / 27.1GB | 306s | 0.889% |
| `vae/` 合并包 | video 9.8GB + audio 578MB | 复制改名 | — |

单测：`python3 -m unittest tests.test_int8_convrot -v`

## 限制

- 当前采样路径仅 T2VA / FL2VA；Ref2VA 条件分支未接完
- VAE 未做 int8 量化
- 源 checkpoint 全零 Linear 会透传并打印 `SKIP zero/nonfinite`
- 量化前应先用 `hf cache verify` 校验官方 checkpoint；不要跨 FL2VA/Ref2VA 修补权重
