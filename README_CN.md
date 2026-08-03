# ComfyUI-RH-MiniMax-H3

<p align="center">
  <img src="assets/runninghub-minimax-banner.png" alt="RunningHub × MiniMax 联合出品" width="760">
</p>

<p align="center">
  <a href="https://www.runninghub.cn/?inviteCode=rh-v1367"><img alt="RunningHub 中国站" src="https://img.shields.io/badge/RunningHub-%E4%B8%AD%E5%9B%BD%E7%AB%99%20Online%20Platform-2f80ed?labelColor=333333"></a>
  <a href="https://www.runninghub.ai/?inviteCode=rh-v1367"><img alt="RunningHub 国际站" src="https://img.shields.io/badge/RunningHub-%E5%9B%BD%E9%99%85%E7%AB%99%20Online%20Platform-2f80ed?labelColor=333333"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green"></a>
</p>

[English documentation](README.md)

**MiniMax-H3** 在同一次扩散过程中生成视频和音频——不是先出画面再配音，而是两条流
在同一组采样步里共同去噪。音画同步来自生成过程本身，不依赖后期对齐。

本插件把完整的 H3 运行时搬进 ComfyUI 进程内，覆盖上游的三条任务链路：文本生成
视频+音频（T2VA）、关键帧驱动（FL2VA，只给首帧即图生视频）、有序多模态参考
（Ref2VA）。配合 INT8 权重与按层 offload，24GB 单卡即可运行。

[RunningHub](https://www.runninghub.cn/?inviteCode=rh-v1367) 是 MiniMax 的合作伙伴，
本插件由 RunningHub 开发并维护。

## ✨ 功能特点

- **三种任务一个插件**——纯文本生成视频+音频（T2VA）、关键帧驱动（FL2VA：首帧、
  尾帧或首尾帧）、有序图片/音频/视频参考（Ref2VA）。
- **图生视频就是"FL2VA 只给首帧"**——给的图会成为输出视频真正的第 0 帧。
- **音视频联合采样**——双 sigma 整流流采样器分别调度视频与音频的 shift，使音画同步。
- **`res_multistep` 去噪计算量降到 1/2.5**——二阶指数积分器，约 21 个 sigma 点
  （20 次 DiT，而非 49 次）即可达到 50 步 Euler 的质量。同一份权重，无需重新
  量化；默认仍是 `euler`。端到端提升取决于负载——文本编码、DiT 加载与 VAE
  解码不受影响。
- **INT8 或 BF16**——单文件 INT8 权重或上游分片 BF16 release，节点不会在两者之间
  静默切换。
- **面向 24GB 单卡**——DiT 自动按层 offload、adaLN 预计算后释放约 40% DiT 权重、
  按画布估算激活预留、任务间租约驻留。
- **带类型的组件契约**——每个 loader 输出都带 release 与组件 fingerprint，
  FL2VA 与 Ref2VA 组件混接、或中途换掉权重，都会直接报错而不是产出错误结果。
- **可选的近似加速**——velocity-cache 与 Cache-DiT，默认都关闭。

## 🛠️ 安装指南

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3.git
pip install -r ComfyUI_RH_MinMaxH3/requirements.txt
```

装好后需重启 ComfyUI——节点定义只在进程启动时读取一次。

**环境要求**

- ComfyUI 0.27 或更高（建议 0.28+）
- 与 ComfyUI 匹配的 PyTorch CUDA 构建，以及 Triton、`comfy-kitchen`
- Ref2VA 的视频/音频参考需要系统 `PATH` 中有 `ffmpeg`、`ffprobe`
- `transformers>=4.57.0,<=5.8.1`（Qwen3-VL 支持）

MiniMax-H3 体量很大。INT8 只降低磁盘占用与搬运成本，并不会把它变成小模型——
除显存外还需要较大的主机内存和高速磁盘。

## 📦 模型下载与安装

权重分两处存放，**两处都必须有**：

| 位置 | 存放内容 | 为什么需要 |
|------|---------|-----------|
| `ComfyUI/models/MiniMax-H3/` | 扁平单文件转换权重 | 提供张量 |
| `ComfyUI/models/diffusers/MiniMax-H3/` | 上游分片 release | 提供 `config.json`、`source/config.json`、tokenizer、`preprocessor_config.json` |

专属根**只放权重、不带任何 sidecar 文件**：组件类型与分区完全由文件名判定，
模型结构则从 `model_root` 指向的分片 release 读取。

### 模型目录结构

```
ComfyUI/
└── models/
    ├── MiniMax-H3/                                    # 转换后的单文件权重
    │   ├── MiniMax-H3-FL2VA-int8_convrot.safetensors  # DiT，FL2VA 分区（T2VA + FL2VA）
    │   ├── MiniMax-H3-Ref2VA-int8_convrot.safetensors # DiT，Ref2VA 分区
    │   ├── qwen3-vl-32b-int8_convrot.safetensors      # Qwen3-VL 文本/多模态编码器
    │   ├── MiniMax-H3-video_vae.safetensors           # 24 通道视频 VAE
    │   └── MiniMax-H3-audio_vae.safetensors           # 32 通道音频 VAE
    │
    └── diffusers/
        └── MiniMax-H3/                                # 上游分片 release
            ├── FL2VA/
            │   ├── transformer/                       # BF16 DiT + config.json
            │   ├── text_encoder/                      # Qwen3-VL + tokenizer/processor
            │   ├── video_vae/
            │   └── audio_vae/
            └── Ref2VA/
                └── ...                                # 组件结构相同
```

### 下载方式

#### 方式一：HuggingFace

```bash
hf download Gluttony10/MiniMax-H3-INT8-CONVROT --local-dir ComfyUI/models/MiniMax-H3
```

#### 方式二：ModelScope（国内推荐）

```bash
pip install modelscope
modelscope download --model Gluttony10/MiniMax-H3-INT8-CONVROT --local_dir ComfyUI/models/MiniMax-H3
```

#### 方式三：手动下载

| 模型 | 链接 | 说明 |
|------|------|------|
| INT8-CONVROT 权重 | [HuggingFace](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT) · [ModelScope](https://modelscope.cn/models/Gluttony10/MiniMax-H3-INT8-CONVROT) | 转换后的单文件 DiT / 文本编码器 / VAE 权重 → `models/MiniMax-H3/` |
| MiniMax-H3 release | 需向 MiniMax 获取 | 分片组件及其配置 → `models/diffusers/MiniMax-H3/` |

> 上游 release 提供 `config.json`、`source/config.json`、tokenizer 和
> `preprocessor_config.json`。**只有转换权重是跑不起来的。**

### 模型选择指南

| 选择 | 下拉框中的值 | 说明 |
|------|-------------|------|
| DiT INT8 | `MiniMax-H3-FL2VA-int8_convrot.safetensors` / `MiniMax-H3-Ref2VA-int8_convrot.safetensors` | 占用最小；分区由文件名证明 |
| DiT BF16（分片） | `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA` | 保真度最高；按层 offload 使其在 24GB 上仍可用 |
| 文本编码器 INT8 | `qwen3-vl-32b-int8_convrot.safetensors` | 约 26GB，BF16 约 62GB |
| 文本编码器 BF16 | `qwen3-vl-32b` | 分片组件目录 |
| 视频 / 音频 VAE | `MiniMax-H3-video_vae.safetensors` / `MiniMax-H3-audio_vae.safetensors` | 权重恒为 FP32，两者独立选择 |

每个 loader 的下拉框**只列出自己那一类组件**，并按任务分区过滤——FL2VA 节点绝不会
列出 Ref2VA 的权重。

## 🚀 使用方法

每种任务都是：三个 loader（DiT、Qwen3-VL、双 VAE）→ target → 条件/编码 →
空 AV latent → 双 sigma 采样器 → AV 解码。

### 示例工作流

可直接加载的工作流在 [`examples/workflows/`](examples/workflows)：

| 任务 | 工作流 | 条件素材 |
|------|--------|---------|
| T2VA | [`t2va.json`](examples/workflows/t2va.json) | 无 |
| FL2VA | [`fl2va_first_frame.json`](examples/workflows/fl2va_first_frame.json) | 首帧——**这就是图生视频** |
| FL2VA | [`fl2va_last_frame.json`](examples/workflows/fl2va_last_frame.json) | 尾帧 |
| FL2VA | [`fl2va_first_last_frame.json`](examples/workflows/fl2va_first_last_frame.json) | 首帧 + 尾帧 |
| Ref2VA | [`ref2va_image.json`](examples/workflows/ref2va_image.json) | 单张参考图 |
| Ref2VA | [`ref2va_image_audio.json`](examples/workflows/ref2va_image_audio.json) | 图片 → 音频，有序链 |
| Ref2VA | [`ref2va_video_audio.json`](examples/workflows/ref2va_video_audio.json) | 带音轨的参考视频 |

共同默认值：显式 `832×480`、5 秒、50 个 sigma 点、shift `12/3`、`accel=off`、
`denoise_video=true`。运行前把占位素材文件名换成已上传到 ComfyUI `input/` 的真实文件。

**关键帧和参考是两种不同机制。** FL2VA 关键帧占据输出时间轴上的真实帧位
（第 `0` 帧或最后一帧），且必须与目标画布尺寸一致；Ref2VA 参考没有帧位，可以用
自己的分辨率，作用是引导身份特征而不是成为某一帧。

**Ref2VA 的参考顺序有意义。** 每个参考节点的 `references` 输出要接到下一个参考
节点；改变链路顺序会改变多模态提示与条件行的顺序。

## 📝 节点参考

全部节点归入 `RunningHub/MiniMax H3/*` 分类。

### 加载器

| 节点 | 用途 |
|------|------|
| `RHMiniMaxH3DirectModelLoader` | T2VA DiT |
| `RHMiniMaxH3DirectTextEncoderLoader` | T2VA Qwen3-VL |
| `RHMiniMaxH3DirectVAELoader` | T2VA 双 VAE（`video_vae_path` + `audio_vae_path`） |
| `RHMiniMaxH3FL2VAModelLoader` / `…TextEncoderLoader` / `…VAELoader` | FL2VA 分区 |
| `RHMiniMaxH3Ref2VAModelLoader` / `…TextEncoderLoader` / `…VAELoader` | Ref2VA 分区 |

### 条件

| 节点 | 用途 |
|------|------|
| `RHMiniMaxH3T2VATarget` / `RHMiniMaxH3T2VATextEncode` | 纯文本 target 与提示词编码 |
| `RHMiniMaxH3FL2VAFirstFrameCondition` | 首帧，可选尾帧 |
| `RHMiniMaxH3FL2VALastFrameCondition` | 仅尾帧 |
| `RHMiniMaxH3FL2VATarget` / `RHMiniMaxH3FL2VAEncode` | 关键帧 target 与编码 |
| `RHMiniMaxH3Ref2VAImageReference` / `…AudioReference` / `…VideoReference` | 有序参考链 |
| `RHMiniMaxH3Ref2VATarget` / `RHMiniMaxH3Ref2VAEncode` | 参考 target 与编码 |

### Latent、采样与解码

| 节点 | 用途 |
|------|------|
| `RHMiniMaxH3EmptyAVLatent` | 由 target 分配联合 AV latent |
| `RHMiniMaxH3SeparateAVLatent` / `RHMiniMaxH3CombineAVLatent` | 拆分或重新合并视频与音频流 |
| `RHMiniMaxH3EncodeVideoAVLatent` | 把已有帧编码进 AV latent（视频转音频） |
| `RHMiniMaxH3FrameRate` | 实验性帧率条件 |
| `RHMiniMaxH3DualSigmaSampler` | 视频 + 音频联合采样。`sampler_mode` 选 `euler`（默认，50 个 sigma 点）或 `res_multistep`（二阶，约 21 点）。完整参数说明见 [docs/sampling_CN.md](docs/sampling_CN.md) |
| `RHMiniMaxH3DecodeAV` | 解码为 `IMAGE` 帧与 `AUDIO` |

## ⚙️ 进阶

功能开关都在 `minimax_h3_nodes/runtime/h3_settings.py`，每一项都可单独关闭以便回滚。

- **显存** —— `ENABLE_DIT_LAYERWISE_OFFLOAD`（auto）、`DIT_LAYERWISE_PREFETCH`、
  `OPT_DYNAMIC_ACTIVATION_RESERVE`、`OPT_RESIDENCY_LEASE` + `RESIDENCY_POLICY`。
- **热路径** —— `OPT_ADALN_PRECOMPUTE` / `OPT_ADALN_RELEASE_WEIGHTS`、
  `OPT_SDPA_PRECOMPUTED_BOUNDS`、`OPT_PREPARED_STRUCTURE`、
  `OPT_INPLACE_EULER_UPDATE`、`OPT_FUSED_QK_ROPE`。
- **加速（近似，默认关闭）** —— 在采样器上把 `accel` 设为
  `minimax-h3-velocity-cache-v1` 或 `minimax-h3-cache-v1`。这些不是 GT 路径，
  sidecar 会记录本次实际走了哪条。
- **可观测性** —— `OPT_TELEMETRY` 记录阶段耗时、每步 P50/P95 与峰值显存；
  `OPT_WRITE_SIDECAR` 在每次产出旁写一份 JSON。

## 📋 更新日志

### 0.4.0

- **`res_multistep` 采样模式。** 视频与音频跑在不同的 shift 调度上，现在各自在
  自身调度上做二阶指数积分。约 21 个 sigma 点（20 次 DiT）即可达到 50 步 Euler
  的质量，同 seed 下无可见差异。单卡 832×480/125 帧实测，两次均为热态：

  | | denoise 循环 | 端到端 |
  |---|---|---|
  | `euler`，50 点（49 次 DiT） | 248.7 秒 | 549 秒 |
  | `res_multistep`，21 点（20 次 DiT） | 101.2 秒 | 406 秒 |
  | | **2.46×** | **1.35×** |

  该尺寸下 denoise 循环约占墙钟时间的 45%，其余是文本编码、DiT 加载与 VAE
  解码——本改动不涉及这些阶段。画布越大，denoise 占比越高，端到端收益也越大。
  在采样器的 `sampler_mode` 选择该模式并把 `sigma_points` 设为 21 即可；默认
  仍是 `euler`，存量工作流不受影响。该模式会强制 `accel=off`——velocity-cache
  与 Cache-DiT 的档位按 50 步标定，20 步上会过度跳步。
- **视频 VAE 分配消除。** 无仿射参数时 Q/K norm 跳过多余的 fp32 往返（CUDA 本就
  在 fp32 累加，半精度结果逐位相同）；门控 FFN、缩放残差与 `norm_silu` 改为就地；
  因果时间维 padding 用单次 `F.pad` 替代 `zeros_like` + `cat`。输出逐位不变——
  同 seed 端到端比对 PSNR = inf。
- **最短输出时长由 5 秒放宽到 4 秒。** 控件默认值仍为 5.0，存量工作流不受影响。

### 0.3.0

- 各加载器的 COMBO 只显示自己那一类；双 VAE 加载器拆成 `video_vae_path` 与
  `audio_vae_path` 两个选择。
- 新增专属权重根 `ComfyUI/models/MiniMax-H3/`，扁平存放单文件权重，
  由文件名判定组件类型，无需 sidecar。
- 补齐全部任务类型的示例工作流，见 [`examples/workflows/`](examples/workflows/)。

## 📄 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

本插件包含对 MiniMax-H3 运行时的改编；出处与被改编组件清单记录在
[NOTICE.md](NOTICE.md)。**不包含任何模型权重。** 权重文件仍受其自身的许可与保密
条款约束——再分发或商用前请先确认这些条款。

## 🔗 相关链接

- [RunningHub 中国站](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub 国际站](https://www.runninghub.ai/?inviteCode=rh-v1367)
- [MiniMax · GitHub](https://github.com/MiniMax-AI)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [MiniMax-H3-INT8-CONVROT · HuggingFace](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT)
- [MiniMax-H3-INT8-CONVROT · ModelScope](https://modelscope.cn/models/Gluttony10/MiniMax-H3-INT8-CONVROT)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## 🙏 致谢

基于 [MiniMax](https://www.minimax.io/)（[GitHub](https://github.com/MiniMax-AI)）的
MiniMax-H3 音视频联合扩散模型构建。
本插件的原生运行时改编自 MiniMax 发布的 H3 源码包，遵循 Apache License 2.0；基线快照与改动
清单见 [NOTICE.md](NOTICE.md)。

本插件运行于 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 之上，部分实现
借鉴了 ComfyUI 上游代码与约定。ComfyUI 以 GPL-3.0 授权，此处作为运行时依赖，
不包含其源码。

由 [RunningHub](https://www.runninghub.cn/?inviteCode=rh-v1367) 封装为 ComfyUI 插件。
