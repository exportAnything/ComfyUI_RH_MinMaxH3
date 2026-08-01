# ComfyUI-MiniMax-H3

[Chinese documentation](README_CN.md)

Local, in-process MiniMax-H3 audio-video diffusion nodes for ComfyUI. The
plugin runs the model components inside the ComfyUI process; it does not call
an SGLang server or a Diffusers pipeline.

The task-aware path now exposes T2VA, FL2VA (first/last-frame-to-video+audio),
and Ref2VA (ordered image/audio/video references). FL2VA and Ref2VA have passed
local contract, packing, sampler, media-preprocessing, static-node, and unit
validation. A full end-to-end run with the released CUDA checkpoints has not
yet been completed for these two new paths, so treat them as an implementation
preview until that validation is recorded.

## Requirements

- ComfyUI 0.27 or newer (0.28+ recommended)
- A CUDA build of PyTorch compatible with ComfyUI, plus Triton and
  `comfy-kitchen`
- `ffmpeg` and `ffprobe` on `PATH` for Ref2VA video/audio references
- MiniMax-H3 weights downloaded separately; weights are not bundled here
- Python dependencies from `requirements.txt`

The runtime is large. INT8 reduces checkpoint storage and transfer cost, but
does not make MiniMax-H3 a small model. Partial/streaming offload still needs
substantial host RAM and fast storage.

## Model layout

Place the release under either `ComfyUI/models/diffusers` or
`ComfyUI/models/minimax_h3`. A single model root may contain both official task
partitions:

```text
models/diffusers/MiniMax-H3/
├── FL2VA/
│   ├── transformer/                  # official BF16 DiT
│   ├── transformer_int8_convrot/     # optional converted DiT
│   ├── text_encoder/                 # official Qwen3-VL encoder
│   ├── text_encoder_int8_convrot/    # optional converted encoder
│   ├── video_vae/
│   ├── audio_vae/
│   └── vae/                          # generated merged VAE bundle
│       ├── video_vae/
│       └── audio_vae/
└── Ref2VA/
    └── ...                           # the same component layout
```

FL2VA nodes only resolve the `FL2VA` partition; Ref2VA nodes only resolve the
`Ref2VA` partition. Each task has three explicit component loaders:

- `... Model Loader (Direct)` selects the DiT directory.
- `... Qwen3-VL Loader (Direct)` selects the text-encoder directory.
- `... Dual VAE Loader (Direct)` selects the merged video/audio VAE directory.

The selectors never silently switch between BF16 and INT8. Select
`transformer`/`text_encoder` for official BF16 weights, or the corresponding
`*_int8_convrot` directories for converted weights. Select `vae` for the
merged VAE bundle.

## FL2VA workflow

The supported keyframe signatures are first frame, last frame, or first+last
frame. Conditions and their semantic frame positions are carried together and
validated again before sampling.

1. Load an image with ComfyUI `LoadImage`.
2. Build `MiniMax H3 FL2VA First / First+Last` (or `Last Only`).
3. Load FL2VA DiT, Qwen3-VL, and VAE with the three FL2VA loaders.
4. Build `MiniMax H3 FL2VA Target`, then run `FL2VA Encode`.
5. Connect the same target to `Empty AV Latent`.
6. Run `Dual Sigma Sampler`, `Decode Video + Audio`, `CreateVideo`, and
   `SaveVideo`.

Frontend workflow: [`examples/fl2va_first_frame_5s.json`](examples/fl2va_first_frame_5s.json).
API workflow: [`examples/fl2va_first_frame_5s_api.json`](examples/fl2va_first_frame_5s_api.json).
Replace the placeholder input image name before queueing either workflow.

## Ref2VA workflow

Ref2VA references are ordered. Chain the optional `references` input when
adding each image, audio, video, or video+audio item; changing the chain order
changes the multimodal presentation and conditioning rows.

1. Load source media with the standard ComfyUI `LoadImage`, `LoadAudio`, or
   `LoadVideo` nodes.
2. Append each item with the matching `MiniMax H3 Ref2VA ... Reference` node.
3. Load Ref2VA DiT, Qwen3-VL, and VAE with the three Ref2VA loaders.
4. Feed the final ordered reference chain to both `Ref2VA Target` and
   `Ref2VA Encode`.
5. Finish with `Empty AV Latent`, `Dual Sigma Sampler`,
   `Decode Video + Audio`, `CreateVideo`, and `SaveVideo`.

Frontend workflow: [`examples/ref2va_image_audio_5s.json`](examples/ref2va_image_audio_5s.json).
API workflow: [`examples/ref2va_image_audio_5s_api.json`](examples/ref2va_image_audio_5s_api.json).
Replace both placeholder media names before queueing either workflow.

Ref2VA video references are normalized to the official 24 fps preparation
path; the Qwen presentation samples that prepared sequence at 2 fps. A
`video_audio` reference must contain a soundtrack. Reference audio is prepared
for the model's stereo/32 kHz VAE path.

## Target and sampler semantics

- Public target duration is 5–15 seconds. The runtime aligns the requested
  frame count upward to MiniMax-H3's `17n+5` temporal boundary. For example,
  a 5.0-second request at 24 fps resolves to 124 frames.
- `auto` FL2VA geometry follows the keyframe media. A finite aspect ratio uses
  the official `adapt_shape_v1` canvas policy. Ref2VA uses the official aspect
  buckets (`21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`); its `auto` default is
  16:9.
- Ref2VA duration `0` means infer the duration from exactly one real
  audio-bearing reference. Use an explicit 5–15 second value when there are
  zero or multiple audio-bearing references.
- The sampler uses separate video and audio noise streams. Visual condition
  rows are pinned at sigma `0.999`; audio-reference rows are pinned at sigma
  `1.0` at every step. With 50 sigma points the model performs 49 DiT forwards.
- The target, ordered conditions, partition, and release/component
  fingerprints are checked across encoder, sampler, and decoder. Cross-wiring
  FL2VA/Ref2VA components fails closed instead of producing undefined output.

## INT8 conversion and VAE merge

Run conversion from this repository and keep each partition separate:

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

The VAE is merged, not INT8-quantized. Do not repair one partition with files
from the other partition, even when filenames look identical. Verify the
downloaded checkpoint before conversion.

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
