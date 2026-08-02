# ComfyUI-RH-MiniMax-H3

[Chinese documentation](README_CN.md)

RunningHub MiniMax-H3 audio-video diffusion nodes for ComfyUI. The plugin runs
every model component inside the ComfyUI process; it does not call an SGLang
server or a Diffusers pipeline.

The task-aware path exposes T2VA, FL2VA (first/last-frame-to-video+audio), and
Ref2VA (ordered image/audio/video references). All three paths have passed
local contract, packing, sampler, media-preprocessing, static-node, and unit
validation. Ref2VA has also completed a real CUDA end-to-end run with released
weights. FL2VA shares the same FL2VA partition and encoding/sampling contract;
treat a first local CUDA smoke as recommended before production use.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> ComfyUI-RH-MiniMax-H3
pip install -r ComfyUI-RH-MiniMax-H3/requirements.txt
```

Restart ComfyUI afterwards: node definitions are read once at start-up.

## Nodes

Every node is registered under an `RHMiniMaxH3` prefix and grouped below the
`RunningHub/MiniMax H3` category. The **node ID** is what a saved workflow
stores as `class_type` / `type`; the **display name** is what the canvas shows.

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

## Requirements

- ComfyUI 0.27 or newer (0.28+ recommended)
- A CUDA build of PyTorch compatible with ComfyUI, plus Triton and
  `comfy-kitchen`
- `ffmpeg` and `ffprobe` on `PATH` for Ref2VA video/audio references
  (Ref2VA Encode / Video Reference probe at node-load time; missing tools
  warn early and fail closed when a media plan actually runs)
- MiniMax-H3 weights downloaded separately; weights are not bundled here
- Python dependencies from `requirements.txt`
  (`transformers>=4.57.0,<=5.8.1`)

The runtime is large. INT8 reduces checkpoint storage and transfer cost, but
does not make MiniMax-H3 a small model. BF16 DiT layerwise offload is **auto**
(official `auto_dit_layerwise_offload`, baseline single-GPU 24GB): when free
VRAM ≥ full weights + `DIT_INFERENCE_RESERVE`, layerwise turns off and the DiT
fully resides; otherwise non-block modules stay on GPU and transformer blocks
are prefetched one layer at a time (`ENABLE_DIT_LAYERWISE_OFFLOAD` — `False`
forces full load — / `DIT_LAYERWISE_PREFETCH` in
`minimax_h3_nodes/runtime/h3_settings.py`). INT8 can still use Comfy
MixedPrecisionOps partial/streaming offload. Both paths need substantial host
RAM and fast storage.

Sampler hot-path opts are on by default (toggle independently in
`h3_settings.py` for rollback): `OPT_SDPA_PRECOMPUTED_BOUNDS` (precomputed
attention bounds, no per-layer CUDA→CPU sync), `OPT_PREPARED_STRUCTURE`
(session-cached RoPE/structure tensors), `OPT_INPLACE_EULER_UPDATE` (in-place
target-row updates, no full-row clone), `OPT_ADALN_SEGMENT_BROADCAST`
(segment-wise in-place adaLN broadcast instead of per-layer full-sequence
`index_select`), `OPT_ADALN_PRECOMPUTE` / `OPT_ADALN_RELEASE_WEIGHTS`
(precompute all schedule AdaLN rows once, then drop ~40% of DiT weights;
cache placement via `OPT_ADALN_CACHE_DEVICE`: `auto`/`ram`/`vram`),
`OPT_PREBUILT_TIMESTEPS` (contiguous sigma/timestep
tensors), `OPT_DYNAMIC_ACTIVATION_RESERVE` (shape-aware
activation reserve with `full`/`layerwise`/`partial`/`reject` tiers; sampler
output includes `residency_mode`).

Lifecycle/cache flags (all in `h3_settings.py`):
- `OPT_RESIDENCY_LEASE` + `RESIDENCY_POLICY` (`safe`/`balanced`/`resident`):
  keep DiT warm after inference (`gpu-resident` / `layerwise-warm`) with TTL;
- `OPT_ENCODE_CACHE`: LRU for text prompt, multimodal Qwen, and VAE condition rows (CPU, byte-capped);
- `OPT_VAE_RESIDENCY`: skip `soft_empty_cache` after VAE offload for faster reload;
- `FORCE_ABSOLUTE_MODEL_ROOTS`: loaders/COMBO emit absolute roots (no name clash);
- `OPT_WRITE_SIDECAR`: Decode writes JSON under Comfy `output/` (task/geometry/residency/telemetry + `env`: plugin commit / GPU / torch / Comfy);
- Downscale chain for 16:9: `1344x768→1024x576→832x480→640x352`
  (`runtime/downscale.py`).

Packaging (public node class names unchanged):
- `nodes.py` → thin facade; impl in `api/{loaders,targets,conditioning,sampling_nodes,decode,_shared}.py`
- `contracts/` → `constants` / `target` / `conditioning` / `components` / `fingerprints` (+ `_impl`)
- `sampling.py` → `runtime/sampler_core.py`
- `runtime/packing/` · `qwen_encoder/` · `media_conditioning/` · `model_loader/` · `vae_adapter/` · `components/` · `dit/`
- DiT helpers: `runtime/attention.py`, `runtime/prepared_structure.py`

## Telemetry & baseline

- `OPT_TELEMETRY`: stage timers, per-step P50/P95, peak VRAM; sidecar includes `telemetry`
- 24GB primary matrix: `benchmarks/matrix.json` + `benchmarks/BASELINE_24GB.md`
- Aggregate: `python3 benchmarks/run_matrix.py --sidecars <Comfy output> --out benchmarks/results`
- Latent golden: `python3 benchmarks/compare_golden.py --ref a.pt --cand b.pt` (`accel=off`)

## Model layout

Weights live in two places: the **official sharded release** stays under
`ComfyUI/models/diffusers` (or `models/minimax_h3`), while **single-file
conversion artifacts** go into the dedicated root `ComfyUI/models/MiniMax-H3`.

```text
models/MiniMax-H3/                              # flat single-file weights
├── MiniMax-H3-FL2VA-int8_convrot.safetensors
├── MiniMax-H3-Ref2VA-int8_convrot.safetensors
├── qwen3-vl-32b-int8_convrot.safetensors
├── MiniMax-H3-video_vae.safetensors
└── MiniMax-H3-audio_vae.safetensors

models/diffusers/MiniMax-H3/                    # official sharded release
├── FL2VA/
│   ├── transformer/                  # official BF16 DiT (sharded)
│   ├── text_encoder/                 # official Qwen3-VL + tokenizer/processor
│   ├── video_vae/
│   └── audio_vae/
└── Ref2VA/
    └── ...                           # the same component layout
```

The dedicated root holds **weights only, with no sidecar**: component type and
partition are decided entirely by the filename (`MiniMax-H3-<partition>-<format>`,
`qwen3-vl-32b-*`, `MiniMax-H3-{video,audio}_vae`), and a file that does not
follow the convention is ignored rather than guessed at. `config.json`,
`source/config.json`, the tokenizer and `preprocessor_config.json` are still read
from the sharded release that `model_root` points at, so **both locations are
required**: the release supplies the architecture, the dedicated root the tensors.

`model_root` selects the official release. FL2VA nodes only resolve the `FL2VA`
partition; Ref2VA nodes only resolve the `Ref2VA` partition. Each task has three
explicit component loaders and **every dropdown lists only its own component
type**:

- `... Model Loader (Direct)`: `transformer_path` lists DiT weights only,
  filtered by partition.
- `... Qwen3-VL Loader (Direct)`: `text_encoder_path` lists text/multimodal
  encoders only.
- `... Dual VAE Loader (Direct)`: split into `video_vae_path` and
  `audio_vae_path`, selecting and loading the 24-channel video VAE and the
  32-channel audio VAE together.

The selectors never silently switch between BF16 and INT8. Prefer weight
filenames / logical names, for example:

- DiT INT8 (single file): `MiniMax-H3-FL2VA-int8_convrot.safetensors` /
  `MiniMax-H3-Ref2VA-int8_convrot.safetensors`
- DiT BF16 (sharded): logical name `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA`
- TE INT8 (single file): `qwen3-vl-32b-int8_convrot.safetensors`
- TE BF16 (sharded): logical name `qwen3-vl-32b`
- VAE single file: `MiniMax-H3-video_vae.safetensors` /
  `MiniMax-H3-audio_vae.safetensors`
- VAE sharded/original: logical name `MiniMax-H3-video_vae` /
  `MiniMax-H3-audio_vae`

A flat single file carries no `quant_meta.json`, so **the filename is the
partition proof**: feeding a Ref2VA DiT into an FL2VA node fails closed. The
selected weight path is folded into the component fingerprint, so swapping a
checkpoint is detected downstream.

Legacy directory names such as `transformer_int8_convrot` / `vae`, and merged
dual-VAE bundles inside a release, still resolve for older workflows. The old
single `vae_path` input has been replaced by `video_vae_path` +
`audio_vae_path`, so existing workflows containing a VAE loader must reconnect
that node.

Loading the Qwen processor validates the official
`preprocessor_config.json` / `video_preprocessor_config.json` (shortest/longest
edge, patch/merge, mean/std). Generic Qwen3-VL processors or wrong hardcoded
pixel caps fail closed so conditioning embeddings cannot silently drift.

## FL2VA workflow

The supported keyframe signatures are first frame, last frame, or first+last
frame. Conditions and their semantic frame positions are carried together and
validated again before sampling.

1. Load an image with ComfyUI `LoadImage`.
2. Build `RunningHub MiniMax H3 FL2VA First / First+Last` (or `Last Only`).
3. Load FL2VA DiT, Qwen3-VL, and VAE with the three FL2VA loaders.
4. Build `RunningHub MiniMax H3 FL2VA Target`, then run `FL2VA Encode`.
5. Connect the same target to `Empty AV Latent`.
6. Run `Dual Sigma Sampler`, `Decode Video + Audio`, `CreateVideo`, and
   `SaveVideo`.

Workflows, one per legal signature:
[`fl2va_first_frame.json`](examples/workflows/fl2va_first_frame.json) ·
[`fl2va_last_frame.json`](examples/workflows/fl2va_last_frame.json) ·
[`fl2va_first_last_frame.json`](examples/workflows/fl2va_first_last_frame.json).
Replace the placeholder input image name before queueing either workflow.

## Ref2VA workflow

Ref2VA references are ordered. Chain the optional `references` input when
adding each image, audio, video, or video+audio item; changing the chain order
changes the multimodal presentation and conditioning rows.

1. Load source media with the standard ComfyUI `LoadImage`, `LoadAudio`, or
   `LoadVideo` nodes.
2. Append each item with the matching `RunningHub MiniMax H3 Ref2VA ... Reference` node.
3. Load Ref2VA DiT, Qwen3-VL, and VAE with the three Ref2VA loaders.
4. Feed the final ordered reference chain to both `Ref2VA Target` and
   `Ref2VA Encode`.
5. Finish with `Empty AV Latent`, `Dual Sigma Sampler`,
   `Decode Video + Audio`, `CreateVideo`, and `SaveVideo`.

Workflows, one per reference shape:
[`ref2va_image.json`](examples/workflows/ref2va_image.json) ·
[`ref2va_image_audio.json`](examples/workflows/ref2va_image_audio.json) ·
[`ref2va_video_audio.json`](examples/workflows/ref2va_video_audio.json).
Replace both placeholder media names before queueing either workflow.

`Ref2VA Encode` exposes `ref_image_size`, which decides how large each
reference image is resolved:

- `match` (default) scales the reference down — never up — to the generation
  canvas' pixel area, keeping its aspect ratio.
- `max` keeps the reference pipeline's independent 2048px short edge, the best
  identity fidelity.

Reference tokens ride through every sampling step, so `max` can be several
times slower than `match` on the same canvas. Workflows saved before this
option existed now run `match`; set it to `max` to reproduce their earlier
output exactly. Switching modes re-encodes rather than reusing a cached one.

Ref2VA video references are normalized to the official 24 fps preparation
path; the Qwen presentation samples that prepared sequence at 2 fps. A
`video_audio` reference must contain a soundtrack. Reference audio is prepared
for the model's stereo/32 kHz VAE path. Comfy `AUDIO` with more than two
channels (no layout metadata) is mean-downmixed to stereo at the reference
node / VAE boundary; prefer file/video references when you need ffmpeg's
layout-aware `-ac 2`.

## Target and sampler semantics

- Public target duration is 5–15 seconds. The runtime aligns the requested
  frame count upward to MiniMax-H3's `17n+5` temporal boundary. For example,
  a 5.0-second request at 24 fps resolves to 124 frames.
- `auto` FL2VA geometry follows the keyframe media. A finite aspect ratio uses
  the official `adapt_shape_v1` canvas policy. Ref2VA uses the official aspect
  buckets (`21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`); its `auto` default is
  16:9.
- `Ref2VA Target` also accepts optional `width` and `height`. Leaving both at
  `0` preserves the bucket policy above; setting both makes that explicit
  canvas authoritative. Values must be multiples of 32, stay within a 1:4–4:1
  ratio, and respect the H3 pixel cap.
- Ref2VA duration `0` means infer the duration from exactly one real
  audio-bearing reference. Use an explicit 5–15 second value when there are
  zero or multiple audio-bearing references.
- The sampler uses separate video and audio noise streams. Visual condition
  rows are pinned at sigma `0.999`; audio-reference rows are pinned at sigma
  `1.0` at every step. With 50 sigma points the model performs 49 DiT forwards.
- The target, ordered conditions, partition, and release/component
  fingerprints are checked across encoder, sampler, and decoder. Cross-wiring
  FL2VA/Ref2VA components fails closed instead of producing undefined output.

## V2A (video → audio, optional)

Set `denoise_video=False` on `Dual Sigma Sampler` to freeze `av_latent.video`
as a clean visual condition (timestep floor `0.999`) and denoise audio only.
Requires a **T2VA** packed layout (no prior visual condition rows).

Typical graph:

1. `T2VA Target` + `Empty AV Latent`
2. `Encode Video → AV Latent` (VAE + `IMAGE` frames aligned to target), or
   `Separate` / `Combine AV Latent` to assemble a non-zero video shell
3. `T2VA Text Encode` + `Dual Sigma Sampler` with `denoise_video=false`
4. `Decode Video + Audio` (video is the input latent; audio is newly sampled)

All-zero `Empty AV Latent` video is rejected. Do not enable V2A on
FL2VA/Ref2VA layouts that already carry visual condition rows.

## Frame-rate conditioning (experimental, optional)

`RunningHub MiniMax H3 Frame Rate (Experimental)` mirrors PR#15210. It is **not** part of
the official training contract and does **not** change the `target.fps=24` grid:

- `adaln=True`: add an fps sinusoid into `TimeEmbedder` (even `24` is not a
  no-op); compatible with AdaLN precompute (stored in the cache key)
- `temporal_rope=True`: scale video-row temporal RoPE low frequencies by
  `24/fps` (optional hard/linear/smoothstep frequency and sigma profiles);
  no-op at 24 fps

Wire `Model Loader` → `Frame Rate` → `Dual Sigma Sampler`. After changing fps,
reload the DiT if AdaLN weights were already released.

## Optional single-GPU acceleration

This plugin targets **single-GPU** Comfy. There is no multi-GPU / Ulysses gate.
Upstream 4×H200 numbers are knobs/quality references only; single-GPU gains come
from fewer DiT calls (velocity-cache) or skipped blocks (Cache-DiT). Default
`accel=off`.

| Value | Behavior | Single-GPU note |
| --- | --- | --- |
| `off` | Disabled (GT-safe) | Default |
| `auto` | On validated 1344×768/124f/50steps/shifts 12·3, prefer velocity-cache | Good first try |
| `minimax-h3-velocity-cache-v1` | Whole-step velocity reuse + Taylor (no extra package) | **Preferred** |
| `minimax-h3-cache-v1` | Cache-DiT DBCache (`pip install cache-dit>=1.3.0`) | Alternative |
| `manual-velocity` / `manual-cache-dit` | Tune stride or RDT/MC/warmup | Debug |

Upstream references: ~**3.2×** velocity-cache, ~**2×** Cache-DiT on 4×H200.
Approximate—do **not** use as consistency GT. Profiles live under
`minimax_h3_nodes/runtime/profiles/`. Sampler logs actual vs theoretical DiT
call counts when velocity-cache runs; `auto` miss / `manual-*` also log the
workload and that the path is non-GT.

Set `accel` on the sampler in any of the bundled workflows under
[`examples/workflows/`](examples/workflows). Give Ref2VA Target an
explicit width/height: leaving them empty resolves to 1344×768 by aspect ratio
and costs far more.

Independently of `accel`, two fused-kernel paths engage automatically when the
installed Comfy exposes them (both from upstream PR #15224). Each is probed once
per process and logged; when the entry point is missing — older Comfy, no
comfy-kitchen, non-CUDA device — the existing PyTorch path runs unchanged.

| Setting | Kernel | What it saves |
| --- | --- | --- |
| `OPT_INT8_FUSED_SWIGLU` | `comfy.ops.linear_input_act` | INT8 MLP: swiglu folds into the activation quantizer, dropping one full-size intermediate per layer per step |
| `OPT_FUSED_QK_ROPE` | `comfy.quant_ops.ck.rms_rope_split_half_` | Attention: per-head RMSNorm + split-half RoPE in one pass, written in place on the qkv buffer |

Both live in `minimax_h3_nodes/runtime/h3_settings.py`;
`OPT_FUSED_QK_ROPE_CUDA_ONLY` keeps the RoPE kernel off non-CUDA devices, where
comfy-kitchen has no implementation. The fused RoPE path also steps aside when
gradients are live, since it rewrites autograd views in place.

## INT8 conversion and VAE merge

Run conversion from this repository and keep each partition separate:

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

The VAE is merged, not INT8-quantized. Do not repair one partition with files
from the other partition, even when filenames look identical. Verify the
downloaded checkpoint before conversion.

Each tool emits a **component directory** (`config.json` + a single-file weight +
`quant_meta.json`). To use the flat dedicated root, move the `.safetensors` out
of it into `models/MiniMax-H3/` — the filename already encodes model, component
type and quantization format, which is what the nodes classify on, and the
config keeps coming from the sharded release under `$BASE`:

```bash
FLAT=/path/to/ComfyUI/models/MiniMax-H3
mkdir -p "$FLAT"
mv "$BASE/FL2VA/transformer_int8_convrot/MiniMax-H3-FL2VA-int8_convrot.safetensors" "$FLAT/"
mv "$BASE/Ref2VA/transformer_int8_convrot/MiniMax-H3-Ref2VA-int8_convrot.safetensors" "$FLAT/"
mv "$BASE/FL2VA/text_encoder_int8_convrot/qwen3-vl-32b-int8_convrot.safetensors" "$FLAT/"
mv "$BASE/FL2VA/vae/video_vae/MiniMax-H3-video_vae.safetensors" "$FLAT/"
mv "$BASE/FL2VA/vae/audio_vae/MiniMax-H3-audio_vae.safetensors" "$FLAT/"
```

Component directories left inside the release keep working; both shapes show up
in the matching per-type dropdown.

## AdaLN curve-table DiT (optional, ~40% smaller checkpoint)

Every DiT layer carries a `[96768, 2688]` adaLN projection — 26 GB in total,
39% of the BF16 DiT and 55% of the INT8 one (adaLN is never quantized). Its
input is only the one-dimensional curve `silu(time_embedder(t))`, so projecting
that curve onto a shared rank-`k` basis folds the basis into each layer's
weight (`[96768, k]`) and replaces the time embedder with an `adaln_t_table`
`[grid, k]` sampled table read by linear interpolation. This is the checkpoint
format introduced by upstream PR #15224; the loader detects it from the
`adaln_t_table` tensor, so both variants load through the same nodes.

```bash
python3 tools/convert_adaln_curve.py \
  --src "$BASE/FL2VA/transformer" --verify           # BF16: 66.3 GiB -> ~40 GiB
python3 tools/convert_adaln_curve.py \
  --src "$BASE/FL2VA/transformer_int8_convrot" --verify   # INT8: 47.0 GiB -> ~21 GiB
```

The output lands in `<src>_adaln_curve/` and appears in the DiT selector as its
own model name. `--verify` compares the curve path against the real adaLN
output at random off-grid timesteps and aborts below `--cosine-floor` (0.9999);
raise `--rank` / `--grid` if it does. Defaults are rank 64 / grid 1024.

Trade-offs versus the runtime adaLN precompute (which stays the default for
stock checkpoints):

- Smaller on disk, no precompute pass, no modulation cache, any timestep works.
- The adaLN input is a rank-`k` approximation instead of exact.
- The experimental Frame Rate node's `adaln` mode needs the time embedder and is
  therefore rejected on curve checkpoints; its `temporal_rope` mode still works.

## Local validation

```bash
python3 -m compileall -q minimax_h3_nodes tools tests
python3 -m unittest discover -s tests -v
```

These checks cover local structure and CPU-testable contracts. Passing them is
not a substitute for a real CUDA run with the complete released weights.

## License and upstream

Plugin code is distributed under the repository's Apache-2.0 license. Model
weights are not included and remain subject to their upstream license and
terms. The implementation is based on the official
[MiniMax-H3 source package](https://github.com/MiniMax-AI-Dev/Internal-0727-private-3).
