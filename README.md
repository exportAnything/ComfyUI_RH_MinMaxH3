# ComfyUI-RH-MiniMax-H3

<p align="center">
  <img src="assets/runninghub-minimax-banner.png" alt="Created jointly by RunningHub and MiniMax" width="760">
</p>

<p align="center">
  <a href="https://www.runninghub.cn/?inviteCode=rh-v1367"><img alt="RunningHub China" src="https://img.shields.io/badge/RunningHub-China%20Online%20Platform-2f80ed?labelColor=333333"></a>
  <a href="https://www.runninghub.ai/?inviteCode=rh-v1367"><img alt="RunningHub International" src="https://img.shields.io/badge/RunningHub-International%20Online%20Platform-2f80ed?labelColor=333333"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green"></a>
</p>

**MiniMax-H3** generates video and audio in the same diffusion process — not a
render followed by a dubbing pass, but two streams denoised across one shared
set of sampling steps. Synchronization comes out of generation itself rather
than from aligning tracks afterwards.

This plugin brings the full H3 runtime inside the ComfyUI process and covers all
three task paths: text to video+audio (T2VA), keyframe driven
generation (FL2VA — supply only a first frame and it is image-to-video), and
ordered multimodal references (Ref2VA). With INT8 weights and layerwise offload
it runs on a single 24GB GPU.

[RunningHub](https://www.runninghub.cn/?inviteCode=rh-v1367) is a MiniMax
partner; this plugin is developed and maintained by RunningHub.

## ✨ Features

- **Three tasks in one plugin** — text to video+audio (T2VA), keyframe driven
  generation (FL2VA: first frame, last frame, or both), and ordered
  image/audio/video references (Ref2VA).
- **Image-to-video is FL2VA with a first frame only** — the supplied image
  becomes the literal frame 0 of the output.
- **Joint AV sampling** — a dual-sigma rectified-flow sampler drives video and
  audio with independent shift schedules, keeping audio locked to motion.
- **2.5× less denoising work with `res_multistep`** — a second-order exponential
  integrator reaches the quality of 50-step Euler in about 21 sigma points
  (20 DiT calls instead of 49). Same weights, no requantisation; `euler` stays
  the default. End-to-end gain depends on the workload — text encoding, DiT
  load and VAE decode are not affected.
- **FP8, NVFP4, INT8, or BF16** — native Comfy single-file FP8 DiT/NVFP4 Qwen
  checkpoints, RunningHub INT8-CONVROT files, or the upstream sharded BF16
  release. The loaders never switch between them silently.
- **24GB-class single GPU** — automatic layerwise DiT offload, adaLN precompute
  with weight release (~40% of DiT weights dropped afterwards), shape-aware
  activation reserve, and residency leasing between runs.
- **Typed component contracts** — every loader output carries release and
  component fingerprints, so mixing FL2VA and Ref2VA parts, or swapping a
  checkpoint mid-graph, fails closed instead of rendering something wrong.
- **Optional approximate acceleration** — velocity-cache or Cache-DiT, both off
  by default.
- **Bundled native attention patches** — fork-owned SageAttention and
  experimental Sol-style sparse-attention nodes eliminate the need for separate
  Sage/Sol custom-node packs. The Sol workflow composes them automatically:
  Sage handles fallback calls and Sol-style routing takes eligible middle steps.

## 🛠️ Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/exportAnything/ComfyUI_RH_MinMaxH3.git
pip install -r ComfyUI_RH_MinMaxH3/requirements.txt
```

Restart ComfyUI afterwards — node definitions are read once at start-up.

**Requirements**

- ComfyUI 0.30.0+ (0.30.1 recommended for native MiniMax-H3 support)
- A CUDA build of PyTorch matching your ComfyUI, plus Triton and `comfy-kitchen`
- `ffmpeg` and `ffprobe` on `PATH` for Ref2VA video and audio references
- `transformers>=4.57.0`; the legacy Transformers-based Qwen loader validates
  `<=5.8.1`, while native NVFP4 uses ComfyUI's built-in MiniMax encoder
- SageAttention is optional and hardware-specific. The bundled Sage patch node
  uses a compatible installed `sageattention` Python/CUDA build when available
  and otherwise passes through safely to ComfyUI attention. It is intentionally
  not pinned in `requirements.txt` because its wheel must match Python, PyTorch,
  CUDA, and the GPU architecture.

MiniMax-H3 is a large model. INT8 reduces storage and transfer cost but does
not make it small — expect substantial host RAM and fast storage on top of VRAM.

## 📦 Model Download & Installation

The fork supports two model layouts. The native Comfy layout is recommended
for FP8/NVFP4 and does **not** require a copied Diffusers release tree:

| Location | Holds | Why it is needed |
|----------|-------|------------------|
| `ComfyUI/models/diffusion_models/` | FP8 MiniMax-H3 DiT | Standard Comfy diffusion-model discovery |
| `ComfyUI/models/text_encoders/` | NVFP4 MiniMax-H3 Qwen3-VL | Standard Comfy text-encoder discovery |
| `ComfyUI/models/vae/` | Video and audio VAEs | Standard Comfy VAE discovery |

Folders added through `extra_model_paths.yaml` work too. Native checkpoint type,
partition, quantization, and architecture are validated from the filename and
safetensors header. The `model_root` value is retained for old workflows but is
not used when all selected components are native single files.

The older RunningHub layout remains supported: converted INT8 files go in
`ComfyUI/models/MiniMax-H3/`, while their matching configuration/tokenizer tree
goes in `ComfyUI/models/diffusers/MiniMax-H3/`. Both roots are required only for
that legacy INT8-CONVROT layout.

Use one complete layout per workflow. Native FP8/NVFP4/VAE files cannot be
mixed with components from a RunningHub release tree because the loaders cannot
prove that those independently packaged files belong to the same release; the
cross-node fingerprint check deliberately rejects that combination.

### Model Directory Structure

```
ComfyUI/
└── models/
    ├── diffusion_models/
    │   ├── minimax_h3_fl2va_pruned_fp8_scaled.safetensors  # T2VA + FL2VA
    │   └── minimax_h3_ref2va_pruned_fp8_scaled.safetensors # Ref2VA
    ├── text_encoders/
    │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
    └── vae/
        ├── minimax_h3_video_vae_fp16.safetensors
        └── minimax_h3_audio_vae_fp32.safetensors
```

T2VA and keyframe workflows use the **FL2VA** DiT. A Ref2VA-only transformer
cannot run those graphs; install the matching FL2VA FP8 file as well.

### Legacy INT8-CONVROT Download Methods

#### Method 1: HuggingFace

```bash
hf download Gluttony10/MiniMax-H3-INT8-CONVROT --local-dir ComfyUI/models/MiniMax-H3
```

#### Method 2: ModelScope (for users in China)

```bash
pip install modelscope
modelscope download --model Gluttony10/MiniMax-H3-INT8-CONVROT --local_dir ComfyUI/models/MiniMax-H3
```

#### Method 3: Manual Download

| Model | Link | Description |
|-------|------|-------------|
| INT8-CONVROT weights | [HuggingFace](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT) \| [ModelScope](https://modelscope.cn/models/Gluttony10/MiniMax-H3-INT8-CONVROT) | Converted single-file DiT / text-encoder / VAE weights → `models/MiniMax-H3/` |
| MiniMax-H3 release | obtain from MiniMax | Sharded components and their configs → `models/diffusers/MiniMax-H3/` |

> For the legacy INT8-CONVROT path, the upstream release supplies
> `config.json`, `source/config.json`, the tokenizer and
> `preprocessor_config.json`; those converted weights alone are not enough.
> Native FP8/NVFP4 checkpoints do not need that release tree.

### Model Selection Guide

| Selection | Loader dropdown value | Notes |
|-----------|----------------------|-------|
| DiT FP8 (native) | `minimax_h3_fl2va_pruned_fp8_scaled.safetensors` / `minimax_h3_ref2va_pruned_fp8_scaled.safetensors` | Normal `diffusion_models` folder; partition is enforced |
| DiT INT8 | `MiniMax-H3-FL2VA-int8_convrot.safetensors` / `MiniMax-H3-Ref2VA-int8_convrot.safetensors` | Smallest footprint; the partition is proven by the filename |
| DiT BF16 (sharded) | `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA` | Highest fidelity; layerwise offload keeps it viable on 24GB |
| Text encoder NVFP4 (native) | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | Normal `text_encoders` folder; loaded through Comfy's native mixed-precision ops |
| Text encoder INT8 | `qwen3-vl-32b-int8_convrot.safetensors` | Roughly 26GB versus 62GB for BF16 |
| Text encoder BF16 | `qwen3-vl-32b` | Sharded component directory |
| Video / Audio VAE (native) | `minimax_h3_video_vae_fp16.safetensors` / `minimax_h3_audio_vae_fp32.safetensors` | Normal `vae` folder; configuration and latent statistics come from embedded metadata |
| Video / Audio VAE (legacy) | `MiniMax-H3-video_vae.safetensors` / `MiniMax-H3-audio_vae.safetensors` | Selected independently from the RunningHub release layout |

Each loader dropdown lists **only its own component type**, filtered by task
partition — an FL2VA node never offers Ref2VA weights.

## 🚀 Usage

Every task wires three loaders (DiT, Qwen3-VL, dual VAE) into a target, a
conditioning/encode step, an empty AV latent, the dual-sigma sampler, and the
AV decode.

### Example Workflows

Ready-to-load graphs live in [`examples/workflows/`](examples/workflows):

| Task | Workflow | Condition material |
|------|----------|--------------------|
| T2VA | [`t2va.json`](examples/workflows/t2va.json) | none |
| T2VA | [`t2va_native_sage_attention.json`](examples/workflows/t2va_native_sage_attention.json) | native Comfy graph + bundled Sage patch |
| T2VA | [`t2va_native_sol_attn.json`](examples/workflows/t2va_native_sol_attn.json) | native Comfy graph + bundled Sage fallback and Sol-style sparse patch |
| FL2VA | [`fl2va_first_frame.json`](examples/workflows/fl2va_first_frame.json) | first frame — **this is image-to-video** |
| FL2VA | [`fl2va_last_frame.json`](examples/workflows/fl2va_last_frame.json) | last frame |
| FL2VA | [`fl2va_first_last_frame.json`](examples/workflows/fl2va_first_last_frame.json) | first + last frame |
| Ref2VA | [`ref2va_image.json`](examples/workflows/ref2va_image.json) | one reference image |
| Ref2VA | [`ref2va_image_audio.json`](examples/workflows/ref2va_image_audio.json) | image → audio, ordered chain |
| Ref2VA | [`ref2va_video_audio.json`](examples/workflows/ref2va_video_audio.json) | video carrying an audio track |

Shared defaults: explicit `832×480`, five seconds, 50 sigma points, shifts
`12/3`, `accel=off`, `denoise_video=true`. Replace the placeholder media
filenames with files already uploaded to ComfyUI `input/`.

**Keyframes and references are different mechanisms.** An FL2VA keyframe
occupies a real frame position in the output (index `0` or the final frame) and
must match the target canvas. A Ref2VA reference has no frame position, may use
its own resolution, and steers identity rather than becoming a frame.

**Ref2VA reference order is significant.** Chain each reference node's
`references` output into the next one; reordering the chain changes the
multimodal prompt and the condition rows.

## 📝 Node Reference

All nodes register under the `RunningHub/MiniMax H3/*` category.

### Loaders

| Node | Purpose |
|------|---------|
| `RHMiniMaxH3DirectModelLoader` | T2VA DiT |
| `RHMiniMaxH3DirectTextEncoderLoader` | T2VA Qwen3-VL |
| `RHMiniMaxH3DirectVAELoader` | T2VA dual VAE (`video_vae_path` + `audio_vae_path`) |
| `RHMiniMaxH3FL2VAModelLoader` / `…TextEncoderLoader` / `…VAELoader` | FL2VA partition |
| `RHMiniMaxH3Ref2VAModelLoader` / `…TextEncoderLoader` / `…VAELoader` | Ref2VA partition |

### Conditioning

| Node | Purpose |
|------|---------|
| `RHMiniMaxH3T2VATarget` / `RHMiniMaxH3T2VATextEncode` | Text-only target and prompt encode |
| `RHMiniMaxH3FL2VAFirstFrameCondition` | First frame, optional last frame |
| `RHMiniMaxH3FL2VALastFrameCondition` | Last frame only |
| `RHMiniMaxH3FL2VATarget` / `RHMiniMaxH3FL2VAEncode` | Keyframe target and encode |
| `RHMiniMaxH3Ref2VAImageReference` / `…AudioReference` / `…VideoReference` | Ordered reference chain |
| `RHMiniMaxH3Ref2VATarget` / `RHMiniMaxH3Ref2VAEncode` | Reference target and encode |

### Latent, sampling and decode

| Node | Purpose |
|------|---------|
| `RHMiniMaxH3EmptyAVLatent` | Allocate the joint AV latent from a target |
| `RHMiniMaxH3SeparateAVLatent` / `RHMiniMaxH3CombineAVLatent` | Split or rejoin the video and audio streams |
| `RHMiniMaxH3EncodeVideoAVLatent` | Encode existing frames into an AV latent (video-to-audio) |
| `RHMiniMaxH3FrameRate` | Experimental frame-rate conditioning |
| `RHMiniMaxH3DualSigmaSampler` | Joint video + audio sampling. `sampler_mode` selects `euler` (default, 50 sigma points) or `res_multistep` (second order, ~21 points). Full parameter guide: [docs/sampling.md](docs/sampling.md) |
| `RHMiniMaxH3DecodeAV` | Decode to `IMAGE` frames and `AUDIO` |

### Native MODEL attention patches

These nodes use the standard ComfyUI `MODEL` type and are intended for the two
`t2va_native_*` workflows. They do not attach to the legacy direct-runtime
`MINIMAX_H3_DIRECT_MODEL` socket.

| Node | Purpose |
|------|---------|
| `RHMiniMaxH3SageAttentionPatch` | Bundled English SageAttention patch node. Uses Sage's automatic backend selection when its optional ABI-compatible Python/CUDA library is present; otherwise leaves normal ComfyUI attention active. |
| `RHMiniMaxH3SolAttentionPatch` | Bundled English experimental Sol-style sparse FlexAttention node. Chains the preceding override, so Sage handles early/late, short, masked, unsupported, or failed calls. No separate Sol custom-node pack is needed. |

Recommended order for the Sol workflow:

```text
Load Diffusion Model → RHMiniMaxH3SageAttentionPatch
                     → RHMiniMaxH3SolAttentionPatch
                     → Scheduler / Guider
```

The Sol-style path is quality-changing and is not the official NVIDIA Sol-Attn
kernel. Its conservative defaults use `tau=1.2`, the middle 20%–90% of the
sampling schedule, and sequences of at least 4096 tokens.

## ⚙️ Advanced

Feature flags live in `minimax_h3_nodes/runtime/h3_settings.py` and can each be
turned off independently for rollback.

- **Memory** — `ENABLE_DIT_LAYERWISE_OFFLOAD` (auto), `DIT_LAYERWISE_PREFETCH`,
  `OPT_DYNAMIC_ACTIVATION_RESERVE`, `OPT_RESIDENCY_LEASE` + `RESIDENCY_POLICY`.
- **Hot path** — `OPT_ADALN_PRECOMPUTE` / `OPT_ADALN_RELEASE_WEIGHTS`,
  `OPT_SDPA_PRECOMPUTED_BOUNDS`, `OPT_PREPARED_STRUCTURE`,
  `OPT_INPLACE_EULER_UPDATE`, `OPT_FUSED_QK_ROPE`.
- **Acceleration (approximate, off by default)** — set `accel` on the sampler to
  `minimax-h3-velocity-cache-v1` or `minimax-h3-cache-v1`. These are not
  ground-truth paths; the sidecar records which one ran.
- **Observability** — `OPT_TELEMETRY` records stage timings, per-step P50/P95
  and peak VRAM; `OPT_WRITE_SIDECAR` writes a JSON sidecar beside each render.

## 📋 Changelog

### 0.6.0

- Added fork-owned English SageAttention and Sol-style sparse-attention patch
  nodes, removing external custom-node dependencies from the native workflows.
- The Sol-style patch preserves and delegates to a preceding Sage override for
  early/late schedule steps and all ineligible or failed calls.
- Updated the native Sage and Sol workflow files to use only nodes registered by
  this repository.

### 0.5.0

- Native Comfy FP8 MiniMax-H3 transformers and NVFP4/AWQ Qwen3-VL text
  encoders are discovered in the standard model folders and validated from
  their safetensors headers.
- Native FP16 video and FP32 audio VAE single files read their architecture and
  latent statistics from embedded metadata.
- RunningHub INT8-CONVROT and sharded BF16 release layouts remain supported.
- Example workflows now use the native single-file names and explicitly keep
  FL2VA and Ref2VA transformer partitions separate.

### 0.4.0

- **`res_multistep` sampler mode.** The video and audio streams run on
  different shift schedules, so each is now integrated with a second-order
  exponential integrator on its own schedule. About 21 sigma points (20 DiT
  calls) match the quality of 50-step Euler, with no visible difference on the
  same seed. Measured on a single GPU at 832×480/125f, both runs warm:

  | | denoise loop | end to end |
  |---|---|---|
  | `euler`, 50 points (49 DiT calls) | 248.7s | 549s |
  | `res_multistep`, 21 points (20 DiT calls) | 101.2s | 406s |
  | | **2.46×** | **1.35×** |

  The denoise loop is about 45% of wall-clock time at this size; the rest is
  text encoding, DiT load and VAE decode, which this change does not touch.
  Larger canvases spend proportionally more time denoising, so the end-to-end
  gain there is higher. Select it with `sampler_mode` on the sampler and set
  `sigma_points` to 21; `euler` remains the default so existing workflows are
  untouched. The mode forces `accel=off` — the velocity-cache and Cache-DiT
  profiles are calibrated for 50 steps and would over-skip at 20.
- **Video VAE allocation elisions.** Q/K norm skips a redundant fp32 round trip
  when the norm has no affine parameters (CUDA already accumulates in fp32, so
  the half-precision result is bit-identical); gated FFN, scaled residuals and
  `norm_silu` became in-place; causal temporal padding is now a single `F.pad`
  instead of `zeros_like` + `cat`. Output is bit-identical — verified
  end-to-end at PSNR = inf on the same seed.
- **Minimum output duration lowered from 5s to 4s.** The widget default stays
  at 5.0, so existing workflows are unaffected.

### 0.3.0

- Per-type loader COMBOs; the dual VAE loader takes `video_vae_path` and
  `audio_vae_path` separately.
- Flat single-file weights root at `ComfyUI/models/MiniMax-H3/`, classified by
  filename with no sidecar required.
- Example workflows for all task types under [`examples/workflows/`](examples/workflows/).

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE).

This plugin contains adaptations of the MiniMax-H3 runtime; provenance and the
list of adapted components are recorded in [NOTICE.md](NOTICE.md). **No model
weights are bundled.** Checkpoint files remain subject to the licence and
confidentiality terms that apply to them — review those terms before
redistributing the weights or using them commercially.

## 🔗 Links

- [RunningHub China](https://www.runninghub.cn/?inviteCode=rh-v1367)
- [RunningHub International](https://www.runninghub.ai/?inviteCode=rh-v1367)
- [MiniMax on GitHub](https://github.com/MiniMax-AI)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [MiniMax-H3-INT8-CONVROT on HuggingFace](https://huggingface.co/Gluttony10/MiniMax-H3-INT8-CONVROT)
- [MiniMax-H3-INT8-CONVROT on ModelScope](https://modelscope.cn/models/Gluttony10/MiniMax-H3-INT8-CONVROT)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## 🙏 Acknowledgements

Built on the MiniMax-H3 joint audio-video diffusion model by
[MiniMax](https://www.minimax.io/) ([GitHub](https://github.com/MiniMax-AI)). The native runtime here is adapted from the
H3 source package released by MiniMax under Apache License 2.0; see [NOTICE.md](NOTICE.md)
for the baseline snapshot and the list of changes.

Built to run inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI), whose
upstream code and conventions parts of this plugin draw on. ComfyUI is licensed
GPL-3.0 and is a runtime dependency here; its source is not bundled.

Packaged for ComfyUI by [RunningHub](https://www.runninghub.cn/?inviteCode=rh-v1367).
