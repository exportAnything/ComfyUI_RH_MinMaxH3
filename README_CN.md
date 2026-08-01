# ComfyUI-MiniMax-H3

[English documentation](README.md)

这是 MiniMax-H3 的 ComfyUI 本地音视频扩散节点。DiT、Qwen3-VL 文本/多模态
编码器以及视频/音频 VAE 都在 ComfyUI 进程内运行，不依赖 SGLang 服务，也不调用
Diffusers Pipeline。

当前任务感知路径已覆盖 T2VA、FL2VA（首/尾帧生成视频与音频）和 Ref2VA（有序
图片/音频/视频参考）。新加入的 FL2VA、Ref2VA 已完成本地合同、打包、采样器、
媒体预处理、节点静态检查和单元测试；尚未使用完整发布权重完成真实 CUDA 端到端
验证，因此在该验证完成前应视为实现预览版。

## 环境要求

- ComfyUI 0.27 或更高版本（建议 0.28+）
- 与 ComfyUI 匹配的 PyTorch CUDA、Triton 和 `comfy-kitchen`
- Ref2VA 视频/音频参考需要系统 `PATH` 中存在 `ffmpeg`、`ffprobe`
- 另行下载 MiniMax-H3 权重；本仓库不包含模型权重
- 安装 `requirements.txt` 中的 Python 依赖

该模型规模很大。INT8 主要降低权重磁盘占用和搬运成本，并不会把 H3 变成小模型；
partial/streaming offload 仍需要较大的内存和高速磁盘。

## 模型目录

模型根目录可放在 `ComfyUI/models/diffusers` 或
`ComfyUI/models/minimax_h3` 下，并同时包含两个官方任务分区：

```text
models/diffusers/MiniMax-H3/
├── FL2VA/
│   ├── transformer/                  # 官方 BF16 DiT
│   ├── transformer_int8_convrot/     # 可选转换产物
│   ├── text_encoder/                 # 官方 Qwen3-VL
│   ├── text_encoder_int8_convrot/    # 可选转换产物
│   ├── video_vae/
│   ├── audio_vae/
│   └── vae/                          # 合并后的双 VAE 包
│       ├── video_vae/
│       └── audio_vae/
└── Ref2VA/
    └── ...                           # 组件结构相同
```

FL2VA 节点只解析 `FL2VA` 分区，Ref2VA 节点只解析 `Ref2VA` 分区。每个任务都有
三个显式组件加载节点：

- `... Model Loader (Direct)`：选择 DiT **模型名**；
- `... Qwen3-VL Loader (Direct)`：选择文本/多模态编码器 **模型名**；
- `... Dual VAE Loader (Direct)`：选择合并双 VAE **模型名**。

节点不会在 BF16 和 INT8 之间静默切换。请按模型名选择，例如：

- DiT INT8：`MiniMax-H3-FL2VA-int8_convrot.safetensors` /
  `MiniMax-H3-Ref2VA-int8_convrot.safetensors`
- DiT BF16（分片）：逻辑名 `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA`
- TE INT8：`qwen3-vl-32b-int8_convrot.safetensors`
- TE BF16（分片）：逻辑名 `qwen3-vl-32b`
- 合并 VAE：逻辑名 `MiniMax-H3-vae`

旧工作流里的目录名（如 `transformer_int8_convrot` / `vae`）仍可解析。

## FL2VA 工作流

支持首帧、尾帧、首帧+尾帧三种合法条件签名。条件图像与语义帧位置会作为整体
传递，并在采样前再次校验。

1. 使用 ComfyUI `LoadImage` 加载图片；
2. 使用 `MiniMax H3 FL2VA First / First+Last`（或 `Last Only`）构造关键帧；
3. 分别使用三个 FL2VA Loader 加载 DiT、Qwen3-VL 与 VAE；
4. 创建 `FL2VA Target`，再执行 `FL2VA Encode`；
5. 将同一个 target 接到 `Empty AV Latent`；
6. 依次连接 `Dual Sigma Sampler`、`Decode Video + Audio`、`CreateVideo`、
   `SaveVideo`。

前端工作流示例：[`examples/fl2va_first_frame_5s.json`](examples/fl2va_first_frame_5s.json)。
API 示例：[`examples/fl2va_first_frame_5s_api.json`](examples/fl2va_first_frame_5s_api.json)。
运行前请替换其中的占位输入图片名。

## Ref2VA 工作流

Ref2VA 的参考素材有严格顺序。添加每一个图片、音频、视频或带音轨视频时，应将
上一个节点的 `references` 输出接到下一个参考节点；改变链路顺序会改变多模态提示
和条件行的顺序。

1. 使用标准 `LoadImage`、`LoadAudio` 或 `LoadVideo` 加载素材；
2. 使用对应的 `MiniMax H3 Ref2VA ... Reference` 节点按顺序追加；
3. 分别使用三个 Ref2VA Loader 加载 DiT、Qwen3-VL 与 VAE；
4. 将最终 reference 链同时接到 `Ref2VA Target` 和 `Ref2VA Encode`；
5. 依次连接 `Empty AV Latent`、`Dual Sigma Sampler`、
   `Decode Video + Audio`、`CreateVideo`、`SaveVideo`。

前端工作流示例：[`examples/ref2va_image_audio_5s.json`](examples/ref2va_image_audio_5s.json)。
API 示例：[`examples/ref2va_image_audio_5s_api.json`](examples/ref2va_image_audio_5s_api.json)。
加速示例：[`examples/ref2va_video_audio_832x480_velocity_5s.json`](examples/ref2va_video_audio_832x480_velocity_5s.json)
（Target 显式 832×480 + `accel=manual-velocity`；Ref2VA 的 Target 不填 width/height 时按 aspect_ratio 默认解析到 1344×768，序列和耗时会大幅上升）。
运行前请替换图片和音频占位文件名。

Ref2VA 视频参考按官方路径规范为 24 fps，Qwen 展示序列再从该序列按 2 fps 采样。
`video_audio` 类型必须包含实际音轨；参考音频会进入立体声/32 kHz 的音频 VAE
预处理路径。

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
`minimax_h3_nodes/runtime/profiles/`。

前端示例（非 API）：[`examples/t2va_velocity_cache_5s.json`](examples/t2va_velocity_cache_5s.json)  
T2VA · 1344×768 · 5s · `accel=minimax-h3-velocity-cache-v1`。导入前确认模型名与本地 INT8/VAE 选择一致。

## INT8 转换与 VAE 合并

必须按任务分区分别转换：

```bash
cd custom_nodes/ComfyUI-MiniMax-H3
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
