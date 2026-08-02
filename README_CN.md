# ComfyUI-RH-MiniMax-H3

[English documentation](README.md)

RunningHub 的 MiniMax-H3 ComfyUI 音视频扩散节点。DiT、Qwen3-VL 文本/多模态
编码器以及视频/音频 VAE 全部在 ComfyUI 进程内运行，不依赖 SGLang 服务，也不
调用 Diffusers Pipeline。

当前任务感知路径已覆盖 T2VA、FL2VA（首/尾帧生成视频与音频）和 Ref2VA（有序
图片/音频/视频参考）。三条路径均已完成本地合同、打包、采样器、媒体预处理、
节点静态检查和单元测试。Ref2VA 已用完整发布权重完成真实 CUDA 端到端验证。
FL2VA 与 T2VA 共用 FL2VA 分区及同一套编码/采样合同；上线前建议再做一次本地
CUDA smoke。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> ComfyUI-RH-MiniMax-H3
pip install -r ComfyUI-RH-MiniMax-H3/requirements.txt
```

装好后需重启 ComfyUI：节点定义只在进程启动时读取一次。

## 节点清单

全部节点注册 ID 带 `RHMiniMaxH3` 前缀，归入 `RunningHub/MiniMax H3` 分类。
**节点 ID** 是工作流里保存的 `class_type` / `type`，**显示名**是画布上看到的标题。

**`RunningHub/MiniMax H3/loaders`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3DirectModelLoader` | RunningHub MiniMax H3 Model Loader (Direct) |
| `RHMiniMaxH3DirectTextEncoderLoader` | RunningHub MiniMax H3 Qwen3-VL Loader (Direct) |
| `RHMiniMaxH3DirectVAELoader` | RunningHub MiniMax H3 Dual VAE Loader (Direct) |
| `RHMiniMaxH3FL2VAModelLoader` | RunningHub MiniMax H3 FL2VA Model Loader (Direct) |
| `RHMiniMaxH3FL2VATextEncoderLoader` | RunningHub MiniMax H3 FL2VA Qwen3-VL Loader (Direct) |
| `RHMiniMaxH3FL2VAVAELoader` | RunningHub MiniMax H3 FL2VA Dual VAE Loader (Direct) |
| `RHMiniMaxH3Ref2VAModelLoader` | RunningHub MiniMax H3 Ref2VA Model Loader (Direct) |
| `RHMiniMaxH3Ref2VATextEncoderLoader` | RunningHub MiniMax H3 Ref2VA Qwen3-VL Loader (Direct) |
| `RHMiniMaxH3Ref2VAVAELoader` | RunningHub MiniMax H3 Ref2VA Dual VAE Loader (Direct) |

**`RunningHub/MiniMax H3/conditioning`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3T2VATarget` | RunningHub MiniMax H3 T2VA Target |
| `RHMiniMaxH3T2VATextEncode` | RunningHub MiniMax H3 T2VA Text Encode |
| `RHMiniMaxH3UnsupportedConditioning` | RunningHub MiniMax H3 Legacy Unsupported Conditioning (Migration Error) |

**`RunningHub/MiniMax H3/fl2va`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3FL2VAFirstFrameCondition` | RunningHub MiniMax H3 FL2VA First / First+Last |
| `RHMiniMaxH3FL2VALastFrameCondition` | RunningHub MiniMax H3 FL2VA Last Only |
| `RHMiniMaxH3FL2VATarget` | RunningHub MiniMax H3 FL2VA Target |
| `RHMiniMaxH3FL2VAEncode` | RunningHub MiniMax H3 FL2VA Encode |

**`RunningHub/MiniMax H3/ref2va`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3Ref2VAImageReference` | RunningHub MiniMax H3 Ref2VA Image Reference |
| `RHMiniMaxH3Ref2VAAudioReference` | RunningHub MiniMax H3 Ref2VA Audio Reference |
| `RHMiniMaxH3Ref2VAVideoReference` | RunningHub MiniMax H3 Ref2VA Video Reference |
| `RHMiniMaxH3Ref2VATarget` | RunningHub MiniMax H3 Ref2VA Target |
| `RHMiniMaxH3Ref2VAEncode` | RunningHub MiniMax H3 Ref2VA Encode |

**`RunningHub/MiniMax H3/latent`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3EmptyAVLatent` | RunningHub MiniMax H3 Empty AV Latent |
| `RHMiniMaxH3SeparateAVLatent` | RunningHub MiniMax H3 Separate AV Latent |
| `RHMiniMaxH3CombineAVLatent` | RunningHub MiniMax H3 Combine AV Latent |
| `RHMiniMaxH3EncodeVideoAVLatent` | RunningHub MiniMax H3 Encode Video → AV Latent |

**`RunningHub/MiniMax H3/sampling`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3FrameRate` | RunningHub MiniMax H3 Frame Rate (Experimental) |
| `RHMiniMaxH3DualSigmaSampler` | RunningHub MiniMax H3 Dual Sigma Sampler |

**`RunningHub/MiniMax H3/decode`**

| Node ID | Display name |
|---|---|
| `RHMiniMaxH3DecodeAV` | RunningHub MiniMax H3 Decode Video + Audio |

### 迁移旧工作流

节点 ID 加了 `RH` 前缀，双 VAE loader 的 `vae_path` 也拆成了两个输入，因此更早
保存的工作流会报 `Node type not found`。用迁移工具转换，不必手工重建：

```bash
python3 tools/migrate_workflow.py 旧工作流.json --in-place
```

前端格式与 API 格式都支持。工具会改节点 ID、拆分 VAE 输入、按当前签名补齐缺失的
widget；下拉里已不存在的旧模型名会替换成当前默认值，并逐条打印替换记录供你复核。
`--in-place` 会留一份 `.bak` 备份。

## 环境要求

- ComfyUI 0.27 或更高版本（建议 0.28+）
- 与 ComfyUI 匹配的 PyTorch CUDA、Triton 和 `comfy-kitchen`
- Ref2VA 视频/音频参考需要系统 `PATH` 中存在 `ffmpeg`、`ffprobe`
  （Encode / Video Reference 在节点加载期软探测；真正跑媒体计划时缺失会 fail-closed）
- 另行下载 MiniMax-H3 权重；本仓库不包含模型权重
- 安装 `requirements.txt` 中的 Python 依赖（`transformers>=4.57.0,<=5.8.1`）

该模型规模很大。INT8 主要降低权重磁盘占用和搬运成本，并不会把 H3 变成小模型。
BF16 DiT 的 layerwise offload 为 **auto**（对齐官方 `auto_dit_layerwise_offload`，
优化基准为单卡 24GB）：当可用显存 ≥ 整模权重 + `DIT_INFERENCE_RESERVE` 时自动关闭、
走整模驻留；放不下时非 block 模块常驻 GPU、transformer block 按层预取。开关见
`h3_settings.py` 的 `ENABLE_DIT_LAYERWISE_OFFLOAD`（`False`=强制整模）/
`DIT_LAYERWISE_PREFETCH`。INT8 仍可走 Comfy MixedPrecisionOps 的
partial/streaming offload。两种路径都需要较大的主机内存和高速磁盘。

采样热路径优化默认启用（均可在 `h3_settings.py` 单独关闭以便回滚）：
`OPT_SDPA_PRECOMPUTED_BOUNDS`（预计算 attention bounds，消除每层 CUDA→CPU 同步）、
`OPT_PREPARED_STRUCTURE`（RoPE/结构张量 session 缓存）、
`OPT_INPLACE_EULER_UPDATE`（原位更新 target rows，避免每步整序列 clone）、
`OPT_ADALN_SEGMENT_BROADCAST`（adaLN 按连续段广播 in-place，避免每层
`index_select` 物化整序列调制张量）、
`OPT_ADALN_PRECOMPUTE` / `OPT_ADALN_RELEASE_WEIGHTS`（采样前预计算全部
timestep 的 adaLN 并释放约 40% DiT 权重；缓存设备由 `OPT_ADALN_CACHE_DEVICE`
控制：`auto`/`ram`/`vram`）、
`OPT_PREBUILT_TIMESTEPS`（预生成连续 sigma/timestep 张量）、
`OPT_DYNAMIC_ACTIVATION_RESERVE`（按画布/序列估算激活预留并分档
`full`/`layerwise`/`partial`/`reject`；采样输出带 `residency_mode`）。

生命周期与缓存（均可在 `h3_settings.py` 关闭）：
- `OPT_RESIDENCY_LEASE` + `RESIDENCY_POLICY`（`safe`/`balanced`/`resident`）：
  推理后 DiT 租约驻留（`gpu-resident` / `layerwise-warm`），TTL 后冷卸载；
- `OPT_ENCODE_CACHE`：文本 / 多模态 Qwen / VAE 条件 rows 共用 LRU（CPU，按字节上限）；
- `OPT_VAE_RESIDENCY`：VAE offload 后跳过 `soft_empty_cache`，便于连续任务回载；
- `FORCE_ABSOLUTE_MODEL_ROOTS`：`True` 时 loader COMBO 一律写绝对路径；默认 `False`，
  按 ComfyUI 目录型模型的约定显示相对搜索路径的短名（如 `MiniMax-H3`），解析时按
  `folder_paths` 搜索路径顺序取首个命中；
- `OPT_WRITE_SIDECAR`：Decode 在 Comfy `output/` 写 JSON（任务/几何/驻留/telemetry + `env`：plugin commit / GPU / torch / Comfy）；
- 降档链：16:9 为 `1344x768→1024x576→832x480→640x352`（`runtime/downscale.py`）。

结构治理（公开节点 class 名不变）：
- `nodes.py` 薄 facade → `api/{loaders,targets,conditioning,sampling_nodes,decode,_shared}.py`
- `contracts/` → `constants` / `target` / `conditioning` / `components` / `fingerprints`（+ `_impl`）
- `sampling.py` → `runtime/sampler_core.py`
- `runtime/` 下 `packing` / `qwen_encoder` / `media_conditioning` / `model_loader` / `vae_adapter` / `components` / `dit` 均已分包
- DiT 工具：`runtime/attention.py`、`runtime/prepared_structure.py`

## 基准与可观测性

- `OPT_TELEMETRY`：采样/解码写入阶段耗时、每步 P50/P95、峰值显存；sidecar 带 `telemetry`
- 24GB 主矩阵：`benchmarks/matrix.json` + 采集说明 `benchmarks/BASELINE_24GB.md`
- 汇总：`python3 benchmarks/run_matrix.py --sidecars <Comfy output> --out benchmarks/results`
- latent golden：`python3 benchmarks/compare_golden.py --ref a.pt --cand b.pt`（默认 accel=off）

## 模型目录

权重分两处存放：**官方分片 release** 留在 `models/diffusers`（或
`models/minimax_h3`）下，**单文件转换产物**放进专属根
`ComfyUI/models/MiniMax-H3`。

```text
models/MiniMax-H3/                              # 扁平单文件权重（转换产物）
├── MiniMax-H3-FL2VA-int8_convrot.safetensors
├── MiniMax-H3-Ref2VA-int8_convrot.safetensors
├── qwen3-vl-32b-int8_convrot.safetensors
├── MiniMax-H3-video_vae.safetensors
└── MiniMax-H3-audio_vae.safetensors

models/diffusers/MiniMax-H3/                    # 官方分片 release
├── FL2VA/
│   ├── transformer/                  # 官方 BF16 DiT（分片）
│   ├── text_encoder/                 # 官方 Qwen3-VL（分片 + tokenizer/processor）
│   ├── video_vae/
│   └── audio_vae/
└── Ref2VA/
    └── ...                           # 组件结构相同
```

专属根**只放权重、不带任何 sidecar**：组件类型与分区一律由文件名判定
（`MiniMax-H3-<分区>-<格式>` / `qwen3-vl-32b-*` / `MiniMax-H3-{video,audio}_vae`），
不符合命名规范的文件会被忽略而不是猜类型。`config.json`、`source/config.json`、
tokenizer、`preprocessor_config.json` 仍然从 `model_root` 指向的分片 release 读取，
因此**两处都要有**：release 提供结构，专属根提供张量。

`model_root` 选官方 release 根。FL2VA 节点只解析 `FL2VA` 分区，Ref2VA 节点只解析
`Ref2VA` 分区。每个任务有三个显式组件加载节点，**每个下拉只列出对应类型的模型**：

- `... Model Loader (Direct)`：`transformer_path` 只列 DiT，且按分区过滤；
- `... Qwen3-VL Loader (Direct)`：`text_encoder_path` 只列文本/多模态编码器；
- `... Dual VAE Loader (Direct)`：拆成 `video_vae_path` 与 `audio_vae_path`
  两个下拉，一次性选择并加载 24 通道视频 VAE 和 32 通道音频 VAE。

节点不会在 BF16 和 INT8 之间静默切换。请按模型名选择，例如：

- DiT INT8（单文件）：`MiniMax-H3-FL2VA-int8_convrot.safetensors` /
  `MiniMax-H3-Ref2VA-int8_convrot.safetensors`
- DiT BF16（分片）：逻辑名 `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA`
- TE INT8（单文件）：`qwen3-vl-32b-int8_convrot.safetensors`
- TE BF16（分片）：逻辑名 `qwen3-vl-32b`
- VAE 单文件：`MiniMax-H3-video_vae.safetensors` /
  `MiniMax-H3-audio_vae.safetensors`
- VAE 分片/原始包：逻辑名 `MiniMax-H3-video_vae` / `MiniMax-H3-audio_vae`

扁平单文件没有 `quant_meta.json`，**分区凭证就是文件名**：把 Ref2VA 的 DiT 选进
FL2VA 节点会直接报错。选中的单文件路径会折进组件 fingerprint，换掉权重会立刻
被下游校验发现。

旧工作流里的目录名（如 `transformer_int8_convrot` / `vae`）以及 release 内的合并
双 VAE 包仍可解析；但旧的单 `vae_path` 输入已被 `video_vae_path` +
`audio_vae_path` 取代，含 VAE Loader 的旧工作流需要重连该节点。

加载 Qwen processor 时会校验官方 `preprocessor_config.json` /
`video_preprocessor_config.json`（短边/长边像素、patch/merge、mean/std）。
通用 Qwen3-VL 或错误硬编码像素阈值会直接报错，避免条件 embedding 静默偏移。

## FL2VA 工作流

支持首帧、尾帧、首帧+尾帧三种合法条件签名。条件图像与语义帧位置会作为整体
传递，并在采样前再次校验。

1. 使用 ComfyUI `LoadImage` 加载图片；
2. 使用 `RunningHub MiniMax H3 FL2VA First / First+Last`（或 `Last Only`）构造关键帧；
3. 分别使用三个 FL2VA Loader 加载 DiT、Qwen3-VL 与 VAE；
4. 创建 `FL2VA Target`，再执行 `FL2VA Encode`；
5. 将同一个 target 接到 `Empty AV Latent`；
6. 依次连接 `Dual Sigma Sampler`、`Decode Video + Audio`、`CreateVideo`、
   `SaveVideo`。

三种签名各一个工作流：
[`fl2va_first_frame.json`](examples/workflows/fl2va_first_frame.json) ·
[`fl2va_last_frame.json`](examples/workflows/fl2va_last_frame.json) ·
[`fl2va_first_last_frame.json`](examples/workflows/fl2va_first_last_frame.json)。
运行前请替换其中的占位输入图片名。

## Ref2VA 工作流

Ref2VA 的参考素材有严格顺序。添加每一个图片、音频、视频或带音轨视频时，应将
上一个节点的 `references` 输出接到下一个参考节点；改变链路顺序会改变多模态提示
和条件行的顺序。

1. 使用标准 `LoadImage`、`LoadAudio` 或 `LoadVideo` 加载素材；
2. 使用对应的 `RunningHub MiniMax H3 Ref2VA ... Reference` 节点按顺序追加；
3. 分别使用三个 Ref2VA Loader 加载 DiT、Qwen3-VL 与 VAE；
4. 将最终 reference 链同时接到 `Ref2VA Target` 和 `Ref2VA Encode`；
5. 依次连接 `Empty AV Latent`、`Dual Sigma Sampler`、
   `Decode Video + Audio`、`CreateVideo`、`SaveVideo`。

三种参考形态各一个工作流：
[`ref2va_image.json`](examples/workflows/ref2va_image.json) ·
[`ref2va_image_audio.json`](examples/workflows/ref2va_image_audio.json) ·
[`ref2va_video_audio.json`](examples/workflows/ref2va_video_audio.json)。
Ref2VA 的 Target 建议显式填 width/height；留空时按 aspect_ratio 默认解析到
1344×768，序列和耗时会大幅上升。
运行前请替换图片和音频占位文件名。

`Ref2VA Encode` 新增 `ref_image_size`，决定每张参考图解析到多大：

- `match`（默认）：按生成画布的像素面积等比只缩不放；
- `max`：参考管线独立的 2048 短边，identity 保真最好。

参考 token 每个采样步都参与注意力，同画布下 `max` 可能比 `match` 慢数倍。
此选项出现之前保存的工作流现在会按 `match` 运行；要精确复现旧结果请显式选
`max`。切换策略会重新编码，不会复用另一策略的缓存。

Ref2VA 视频参考按官方路径规范为 24 fps，Qwen 展示序列再从该序列按 2 fps 采样。
`video_audio` 类型必须包含实际音轨；参考音频会进入立体声/32 kHz 的音频 VAE
预处理路径。Comfy `AUDIO` 超过双声道且无 layout 时，会在 Reference 节点 / VAE
边界均值下混为 stereo；需要 ffmpeg 布局感知 `-ac 2` 时优先走文件/视频参考。

## Target 与采样语义

- 对外时长限制为 5–15 秒。运行时会把请求帧数向上对齐到 H3 的 `17n+5` 时间
  边界，例如 24 fps 下请求 5.0 秒会解析为 124 帧；
- FL2VA 的 `auto` 尺寸跟随关键帧素材；指定比例时使用官方 `adapt_shape_v1`
  画布规则。Ref2VA 使用六个官方比例桶：`21:9`、`16:9`、`4:3`、`1:1`、
  `3:4`、`9:16`，`auto` 默认解析为 16:9；
- `Ref2VA Target` 也支持可选的 `width`、`height`：两者都为 `0` 时保持上述比例桶
  策略；两者同时填写时以手动画布为准。尺寸必须为 32 的倍数、宽高比在 1:4–4:1
  内，且不超过 H3 的像素上限；
- Ref2VA 时长填 `0` 表示从唯一一个真实带音频 reference 推导。没有带音频参考或
  存在多个带音频参考时，必须显式填写 5–15 秒；
- 采样器分别创建视频、音频噪声。视觉条件行在每一步固定为 sigma `0.999`，音频
  参考行固定为 sigma `1.0`。50 个 sigma 点对应 49 次 DiT forward；
- 编码、采样、解码之间会校验 target、条件顺序、任务分区、release 和组件指纹。
  FL2VA/Ref2VA 组件交叉连接会直接报错，不会继续生成未定义结果。

## V2A（视频→音频，可选）

`Dual Sigma Sampler` 的 `denoise_video=False`：把 `av_latent.video` 整段冻结为
视觉条件（timestep floor `0.999`），只去噪音频。当前要求 **T2VA 布局**（packed
无既有 visual condition）。

典型接线：

1. `T2VA Target` + `Empty AV Latent`；
2. `Encode Video → AV Latent`（VAE + 与 target 对齐的 `IMAGE` 帧序列）写入视频；
   或用 `Separate` / `Combine AV Latent` 从已有 AV 壳拼出非空视频；
3. `T2VA Text Encode` + `Dual Sigma Sampler`（`denoise_video=false`）；
4. `Decode Video + Audio`（视频回传输入 latent，音频为新采样结果）。

`Empty AV Latent` 全零视频会直接报错；勿在 FL2VA/Ref2VA 既有 visual cond 布局上开 V2A。

## 帧率条件（实验性，可选）

`RunningHub MiniMax H3 Frame Rate (Experimental)` 对应 PR#15210 的实验能力，**不是**官方
训练契约，也不改 `target.fps=24` 时序格：

- `adaln=True`：把 fps 的 sinusoidal 加到 `TimeEmbedder`（即使填 24 也非 no-op）；
  与 adaLN 预计算兼容，会写入 modulation cache 键；
- `temporal_rope=True`：按 `24/fps` 缩放视频行时序 RoPE 低频（可选 hard/linear/
  smoothstep 频率与 sigma 剖面）；24fps 时缩放为 1。

接线：`Model Loader` → `Frame Rate` → `Dual Sigma Sampler`。换 fps 后若 DiT 已
释放 adaLN 权重，需重新加载模型。

## 单卡采样加速（可选）

开发/部署按**单卡**设计：不做多卡或 Ulysses 门禁。官方 4×H200 数字只作旋钮与
质量参考；单卡收益来自减少 DiT 次数（velocity-cache）或跳过 block 计算
（Cache-DiT）。默认 `accel=off`。

`Dual Sigma Sampler` 的 `accel`：

| 值 | 行为 | 单卡建议 |
| --- | --- | --- |
| `off` | 关闭（可作 GT） | 默认 |
| `auto` | 命中 1344×768/124f/50steps/shift12·3 时优先 velocity-cache | 推荐试 |
| `minimax-h3-velocity-cache-v1` | 整步 velocity 复用 + Taylor（无额外依赖） | **首选** |
| `minimax-h3-cache-v1` | Cache-DiT DBCache（需 `pip install cache-dit>=1.3.0`） | 备选 |
| `manual-velocity` / `manual-cache-dit` | 手调 stride 或 RDT/MC/warmup | 调试 |

官方参考：velocity-cache 约 **3.2×**、Cache-DiT 约 **2×**（均为 4×H200 证据）。
近似加速，**不要**当 consistency GT。profile 在
`minimax_h3_nodes/runtime/profiles/`。采样结束会打实际/理论 DiT 次数日志；
`auto` 未命中与 `manual-*` 也会记录当前 workload 并标明非 GT。

在 [`examples/workflows/`](examples/workflows) 任一工作流的采样器上
设置 `accel` 即可；导入前确认模型名与本地 INT8/VAE 选择一致。

与 `accel` 无关的两条融合 kernel 路径（均来自上游 PR#15224）：安装的 Comfy 暴露
了对应入口就自动启用，每进程探测一次并打日志；入口缺失（旧版 Comfy、没有
comfy-kitchen、非 CUDA 设备）则原样走现有 PyTorch 路径。

| 开关 | Kernel | 收益 |
| --- | --- | --- |
| `OPT_INT8_FUSED_SWIGLU` | `comfy.ops.linear_input_act` | INT8 MLP：swiglu 折进激活量化 kernel，省掉每层每步一次全尺寸中间张量 |
| `OPT_FUSED_QK_ROPE` | `comfy.quant_ops.ck.rms_rope_split_half_` | 注意力：per-head RMSNorm 与 split-half RoPE 一趟做完，就地写在 qkv 缓冲上 |

两者都在 `minimax_h3_nodes/runtime/h3_settings.py`；
`OPT_FUSED_QK_ROPE_CUDA_ONLY` 让 RoPE kernel 在非 CUDA 设备上一律回退
（comfy-kitchen 没有对应实现）。带梯度时融合 RoPE 也会自动让路——它要就地改写
autograd 视图。

## INT8 转换与 VAE 合并

必须按任务分区分别转换：

```bash
cd custom_nodes/ComfyUI-RH-MiniMax-H3
BASE=/path/to/ComfyUI/models/diffusers/MiniMax-H3

python3 tools/quantize_int8_convrot.py \
  --src "$BASE/FL2VA/transformer" --device cuda --verify
python3 tools/quantize_int8_convrot.py \
  --src "$BASE/Ref2VA/transformer" --device cuda --verify

python3 tools/quantize_text_encoder_int8_convrot.py \
  --src "$BASE/FL2VA/text_encoder" --device cuda --verify
python3 tools/quantize_text_encoder_int8_convrot.py \
  --src "$BASE/Ref2VA/text_encoder" --device cuda --verify

python3 tools/merge_vae.py --src "$BASE/FL2VA"
python3 tools/merge_vae.py --src "$BASE/Ref2VA"
```

VAE 只做合并打包，不做 INT8 量化。即使文件名相同，也不要用一个分区的文件修补
另一个分区；转换前应先校验完整的官方下载权重。

工具输出的是**组件目录**（`config.json` + 单文件权重 + `quant_meta.json`）。要用
专属根的扁平布局，把其中的 `.safetensors` 挪到 `models/MiniMax-H3/` 即可——文件名
已经带了模型/类型/量化格式，节点靠它判定类型与分区，配置继续从 `$BASE` 的分片
release 读：

```bash
FLAT=/path/to/ComfyUI/models/MiniMax-H3
mkdir -p "$FLAT"
mv "$BASE/FL2VA/transformer_int8_convrot/MiniMax-H3-FL2VA-int8_convrot.safetensors" "$FLAT/"
mv "$BASE/Ref2VA/transformer_int8_convrot/MiniMax-H3-Ref2VA-int8_convrot.safetensors" "$FLAT/"
mv "$BASE/FL2VA/text_encoder_int8_convrot/qwen3-vl-32b-int8_convrot.safetensors" "$FLAT/"
mv "$BASE/FL2VA/vae/video_vae/MiniMax-H3-video_vae.safetensors" "$FLAT/"
mv "$BASE/FL2VA/vae/audio_vae/MiniMax-H3-audio_vae.safetensors" "$FLAT/"
```

留在 release 里的组件目录（含 `quant_meta.json`）同样可用，两种形态都会出现在
对应类型的下拉里。

## AdaLN 曲线表 DiT（可选，checkpoint 缩小约 40%）

DiT 每层的 adaLN 投影是 `[96768, 2688]`，50 层合计 26 GB——占 BF16 DiT 的 39%、
INT8 的 55%（adaLN 不参与量化）。而它的输入只是 `silu(time_embedder(t))` 这条
一维曲线：把曲线投影到秩 `k` 的共享基后，基被折进每层权重（`[96768, k]`），
time embedder 由 `adaln_t_table [grid, k]` 采样表 + 线性插值取代。这就是上游
PR #15224 引入的 checkpoint 格式；加载器按 `adaln_t_table` 张量自动识别，两种
形态走同一套节点。

```bash
python3 tools/convert_adaln_curve.py \
  --src "$BASE/FL2VA/transformer" --verify           # BF16: 66.3 GiB -> 约 40 GiB
python3 tools/convert_adaln_curve.py \
  --src "$BASE/FL2VA/transformer_int8_convrot" --verify   # INT8: 47.0 GiB -> 约 21 GiB
```

产物写到 `<src>_adaln_curve/`，在 DiT 选择器里作为独立模型名出现。`--verify`
会在随机的非网格 timestep 上比对曲线路径与真实 adaLN 输出，低于
`--cosine-floor`（0.9999）直接失败——此时提高 `--rank` / `--grid`。默认
rank 64 / grid 1024。

与运行时 adaLN 预计算（原版 checkpoint 仍默认走它）的取舍：

- 磁盘更小，无需预计算，不占调制缓存，任意 timestep 都可用。
- adaLN 输入是秩 `k` 近似，而非精确值。
- 实验性 Frame Rate 节点的 `adaln` 模式依赖 time embedder，曲线表 checkpoint 会
  直接报错；`temporal_rope` 模式不受影响。

## 本地验证

```bash
python3 -m compileall -q minimax_h3_nodes tools tests
python3 -m unittest discover -s tests -v
```

这些命令验证本地结构以及 CPU 可执行合同，不能代替使用完整发布权重的真实 CUDA
端到端运行。

## 许可证与上游

插件代码按本仓库 Apache-2.0 许可证发布。模型权重不包含在仓库中，仍受上游模型
许可证及条款约束。实现参考官方
[MiniMax-H3 源码包](https://github.com/MiniMax-AI-Dev/Internal-0727-private-3)。
