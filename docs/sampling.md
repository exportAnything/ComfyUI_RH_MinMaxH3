# Dual Sigma Sampler 参数与选择指南

`RHMiniMaxH3DualSigmaSampler` 是 H3 唯一的采样入口。它同时推进视频和音频两条
流，因此标准 KSampler 无法替代。

---

## 一、参数速查

### 必填

| 参数 | 默认 | 说明 |
|---|---|---|
| `seed` | 42 | 随机种子。视频与音频各自独立加噪，同 seed 可完全复现 |
| `sigma_points` | 50 | **sigma 点数，不是步数**。50 个点产生 49 次 DiT forward |
| `video_shift` | 12.0 | 视频流的 flow shift |
| `audio_shift` | 3.0 | 音频流的 flow shift |
| `accel` | `off` | 近似加速档位，见第四节 |
| `denoise_video` | `true` | `false` = V2A：视频作干净条件，只去噪音频 |

### 可选

| 参数 | 默认 | 生效条件 |
|---|---|---|
| `sampler_mode` | `euler` | 见第二节 |
| `cache_dit_rdt` | 0.12 | 仅 `accel=manual-cache-dit` |
| `cache_dit_mc` | 2 | 仅 `accel=manual-cache-dit` |
| `cache_dit_warmup` | 4 | 仅 `accel=manual-cache-dit` |
| `velocity_stride` | 4 | 仅 `accel=manual-velocity` |
| `allow_accel_with_res_multistep` | `false` | 仅 `sampler_mode=res_multistep`，见第二节 |

---

## 二、`sampler_mode`：先调这个

H3 的视频和音频跑在**不同的 shift 调度**上。两种模式都让每条流在自己的调度上
积分，区别只在积分器阶数。

| | `euler` | `res_multistep` |
|---|---|---|
| 积分器 | 一阶 | 二阶指数积分器 |
| 建议 `sigma_points` | **50** | **21** |
| DiT 次数 | 49 | 20 |
| denoise 循环 | 248.7s | **101.2s**（2.46×） |
| 端到端 | 549s | **406s**（1.35×） |
| `accel` | 可用 | 默认 off，可显式打开 |

> 实测条件：单卡，t2va，832×480，125 帧，两次均为热态。同 seed 逐帧比对无
> 可见质量退化（PSNR 26.3 dB —— 不同采样器走不同轨迹到达不同的有效解，该值
> 衡量的是差异而非优劣）。

**两个口径为什么差这么多**：该负载下 denoise 循环只占墙钟时间约 45%，其余是
文本编码（Qwen3-VL 32B）、`dit_load`、VAE 解码 —— 换采样器完全不影响这些阶段。
画布越大，denoise 占比越高，端到端收益越接近 2.46×。

每次 DiT 调用的成本恒定在 5.05s，所以 **denoise 层面的加速比就等于 DiT 次数比**。

**`res_multistep` 为什么能减半步数**：二阶方法单步的局部误差是一阶的平方阶，
所以 20 步的累积误差大致相当于 Euler 的 40–50 步。同一份权重，**不需要重新
量化**。

**为什么默认关掉 `accel`**：velocity-cache 与 Cache-DiT 的档位都按 50 步标定，
二阶采样步数少，跳步的余量也小。实测叠加 velocity-cache（同 seed，对比 euler-50
基准）：

| 配置 | DiT 次数 | denoise | 端到端 | PSNR |
|---|---|---|---|---|
| `euler` 50 | 49/49 | 248.7s | 549s | 基准 |
| `res_multistep` 21 | 20/20 | 101.2s | 406s | 26.3 dB |
| `euler` 50 + velocity stride4 | 14/49 | 70.9s | 501s | 24.1 dB |
| `res_multistep` 21 + velocity stride2 | 11/20 | 55.8s | 304s | **21.0 dB** |
| `res_multistep` 21 + velocity stride4 | 7/20 | 35.5s | 176s | **15.3 dB** |
| `res_multistep` 31 + velocity stride2 | 16/30 | 81.1s | — | 24.2 dB |
| `res_multistep` 41 + velocity stride4 | 12/40 | 60.6s | — | 20.6 dB |

（端到端一栏留空处为热态差异过大、不可横比；请以 denoise 与 DiT 次数为准。）

**21 点下叠加会明显劣化**：stride2 已经换了场景（不是噪声差异），stride4 出现
振铃伪影与色带。20 步本身余量就小，再跳掉 45%–65% 的 DiT 调用没有回旋空间。

**但加步数能救回来**：31 点叠 stride 2 用 16 次 DiT 拿到 24.2 dB，画面完全可用。
所以限制不是硬性的——节点上有 `allow_accel_with_res_multistep`（默认关闭）可以
显式打开。

只是**它并不比 euler + velocity 划算**：同为 24 dB 档，`euler`-50 + stride 4 只
要 14 次 DiT / 70.9s，而 `res_multistep`-31 + stride 2 要 16 次 / 81.1s。原因是
二阶积分器的优势在于每步走得更准，而 velocity-cache 是整步跳过——跳掉的步越多，
二阶的精度优势越没机会兑现，两种加速在争同一份预算。

所以：要最好的质量成本比，用 `res_multistep` 单独（20 次 / 26.3 dB）；要再快，
走 `euler` + velocity。叠加只在你已经熟悉两边旋钮、并愿意自己调步数时才有意义。

### 怎么选

- **默认用 `res_multistep` + `sigma_points=21`** —— 快一倍多，质量不掉
- 需要与旧结果逐位复现，或要用 accel 档位 → 留 `euler` + `50`

---

## 三、`video_shift` / `audio_shift`

Flow shift 控制 sigma 调度的疏密：值越大，采样步越集中在高噪声段（构图、
运动等大结构），低噪声段（细节）分配得越少。

**默认 12.0 / 3.0 是模型发布时的配置，不建议随意改。** 两个值必须配套 ——
音频调度是从视频调度推导出来的，单独改一个会破坏音画对齐。

真要调：

| 现象 | 方向 |
|---|---|
| 大结构/运动不稳 | 调高 `video_shift`（如 14–16） |
| 细节糊、纹理少 | 调低 `video_shift`（如 8–10） |

改完必须重新目视验证音画同步。已验证的加速档位只在 `12.0/3.0` 下标定过,
改了 shift 就不能再用 profile 档位（会被 workload 匹配拒绝）。

---

## 四、`accel`：近似加速

**全部是近似路径，输出与 `off` 不同，不能作为基准。** 默认 `off`。

| 档位 | 机制 | 标定加速比 | LPIPS |
|---|---|---|---|
| `off` | 不加速 | 1× | — |
| `auto` | 优先 velocity，否则 cache-dit；不匹配则保持关闭 | — | — |
| `minimax-h3-velocity-cache-v1` | 整步 velocity 复用 + Taylor 外推 | **3.20×** | 0.426 |
| `minimax-h3-cache-v1` | Cache-DiT 块级缓存（需 `pip install cache-dit`） | **1.99×** | 0.233 |
| `manual-velocity` | 手填 `velocity_stride` | — | — |
| `manual-cache-dit` | 手填 rdt / mc / warmup | — | — |

> 上表加速比与 LPIPS 来自上游在 4×H200 上的标定，工作负载为
> **t2va 1344×768 / 124 帧 / 50 步 / shift 12·3**。单卡只复用同一组旋钮，
> 不做多卡门禁；实际加速比会低于该数字。

### 两个 profile 档位有 workload 门禁

参数不落在上面那个标定合同上会**直接报错拒绝执行**，而不是静默降级。想在
其它分辨率/步数下用，只能走 `manual-*`。

### `manual-cache-dit` 的三个旋钮

| 旋钮 | 含义 | 越大 |
|---|---|---|
| `cache_dit_rdt` | 残差差异阈值，低于它就复用缓存 | 越快越糙 |
| `cache_dit_mc` | 最多连续跳几步 | 越快越糙 |
| `cache_dit_warmup` | 前几步不跳 | 越大越稳 |

对应上游的质量档位：

| 档位 | rdt | mc | warmup |
|---|---|---|---|
| 上游 high | 0.04 | 1 | 4 |
| **本插件已验证 profile** | **0.08** | **2** | **4** |
| manual 默认 | 0.12 | 2 | 4 |
| 上游 medium | 0.12 | 3 | 4 |
| 上游 low | 0.24 | 3 | 4 |

**注意 `manual-cache-dit` 的默认 rdt 是 0.12，比已验证 profile 的 0.08 激进。**
想要等同已验证质量，手动改成 `rdt=0.08 / mc=2 / warmup=4`。

上游三档只有旋钮值、没有本地实测数据，所以没做成预设档位（`_load_profile`
只接受实测验证过的 profile）。要用就按上表手填。

### `manual-velocity` 的 `velocity_stride`

DiT 刷新步距。`1` = 每步都算（等于不加速），`4` = 每 4 步算一次、中间用
Taylor 外推。越大越快越糙。

---

## 五、`denoise_video`

| 值 | 行为 |
|---|---|
| `true`（默认） | 正常生成，视频和音频都去噪 |
| `false` | **V2A 模式**：把 `av_latent.video` 当作干净视觉条件，只生成音频 |

`false` 要求 T2VA 布局且 video 不是全零 —— 用 `Encode Video → AV Latent` 或
`Combine AV Latent` 把真实视频填进去。

---

## 六、推荐组合

**日常出片**
```
sampler_mode = res_multistep
sigma_points = 21
accel        = off
```
最快且无近似损失。

**要复现旧结果 / 做质量基准**
```
sampler_mode = euler
sigma_points = 50
accel        = off
```

**极限压时间（可接受明显近似）**
```
sampler_mode = euler
sigma_points = 50
accel        = minimax-h3-velocity-cache-v1
```
仅在 1344×768 / 124 帧 / shift 12·3 下可用，其它参数会被拒绝。
LPIPS 0.426 意味着与基准差异已相当可见。

**叠加 `res_multistep` + accel**：打开 `allow_accel_with_res_multistep` 即可，
但务必同时调高 `sigma_points`（21 点下会崩；31 点叠 stride 2 才可用）。同等画质
下并不比 `euler` + velocity 省，见第二节表格。

**给视频配音（V2A）**
```
denoise_video = false
```
配合 `Encode Video → AV Latent` 使用。

---

## 七、常见问题

**`sigma_points` 填 20 还是 21？**
填 21。这个参数是 sigma **点**数，21 个点产生 20 次 DiT forward。

**切了 `res_multistep` 但没变快？**
`sigma_points` 忘了从 50 改成 21。模式本身不改变步数。

**选了 profile 档位却报 workload 不匹配？**
两个 profile 只在 1344×768 / 124 帧 / 50 步 / shift 12·3 下标定过。改用
`manual-*`，或把参数调回该合同。

**输出与上次不一致？**
检查 `seed`、`sampler_mode`、`sigma_points`、两个 shift、`accel` 是否全部相同。
任一不同都会改变结果。
