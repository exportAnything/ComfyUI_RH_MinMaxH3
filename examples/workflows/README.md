# MiniMax-H3 Frontend Workflows

Covers every task type in `H3_TASKS` and all valid conditioning/reference configurations.
Each file uses the ComfyUI frontend format (`nodes` + `links`) and can be dragged
directly onto the canvas or opened with **Load**.

| Task | Workflow | Conditioning input |
|------|--------|----------|
| T2VA | `t2va.json` | Native graph; text only; bundled SageAttention |
| T2VA | `t2va_native_sage_attention.json` | None; native ComfyUI graph with SageAttention |
| T2VA | `t2va_native_sol_attn.json` | None; native ComfyUI graph with Sol-Attn |
| FL2VA | `fl2va_first_frame.json` | Native graph; first-frame image |
| FL2VA | `fl2va_last_frame.json` | Native graph; last-frame image |
| FL2VA | `fl2va_first_last_frame.json` | Native graph; first-frame + last-frame images |
| Ref2VA | `ref2va_image.json` | Native graph; reference image |
| Ref2VA | `ref2va_image_audio.json` | Native graph; reference image + standalone audio |
| Ref2VA | `ref2va_video_audio.json` | Native graph; 24-fps reference video + paired soundtrack |

All default workflows now use native ComfyUI MiniMax H3 conditioning and stock
sampling. They default to 16:9 / 0.4 megapixels (`864×480`), 5 seconds, 20
steps, the `simple` scheduler, and `res_multistep`. Replace placeholder media
filenames before queueing.

## Native Attention Variants

Every workflow in this directory uses ComfyUI's native MiniMax H3
implementation and requires ComfyUI 0.30.1 or newer. They are preconfigured for
the official INT8-CONVROT transformer, NVFP4-AWQ text encoder, and native
video/audio VAEs. Their model dropdowns can also select compatible native FP8
files discovered by ComfyUI.

Choose an attention workflow:

- `t2va_native_sage_attention.json` uses the fork-owned SageAttention patch.
- `t2va.json`, the three `fl2va_*` workflows, and the three `ref2va_*`
  workflows use that same bundled SageAttention patch as their safe default.
- `t2va_native_sol_attn.json` chains the fork-owned SageAttention patch into the
  experimental Sol-style sparse-attention patch. Sol handles eligible middle
  steps; Sage is retained as the fallback for every other attention call.

No additional Sage or Sol custom-node pack is required. A compatible
hardware-specific `sageattention` Python/CUDA library remains optional; when it
is missing, the included Sage node safely leaves normal ComfyUI attention active.
The included Sol-style node has no additional dependency beyond ComfyUI's PyTorch
runtime. Both graphs are already wired between Load Diffusion Model and the
scheduler/guider.

## Native Keyframe Variants

The three `fl2va_*` files are rebuilt from the native SageAttention graph. Their
Load Image nodes are already connected to `MiniMaxH3ImageToVideo` inside the
group as follows:

- `fl2va_first_frame.json`: `first_frame`
- `fl2va_last_frame.json`: `last_frame`
- `fl2va_first_last_frame.json`: both inputs

The images are encoded conditioning anchors, not pixel-locked copies. Do not
insert the legacy Dual Sigma Sampler into these graphs; native H3 derives the
audio schedule internally and is designed to use ComfyUI's stock sampler.

## Native Reference Variants

The three `ref2va_*` files use `MiniMaxH3ReferenceToVideo` and one-based prompt
tags such as `<Picture 1>`, `<Video 1>`, and `<Audio 1>`:

- `ref2va_image.json` connects one image to `ref_images.ref_image_0`.
- `ref2va_image_audio.json` connects an image and standalone audio to their
  native typed reference inputs.
- `ref2va_video_audio.json` uses `LoadVideo` and `GetVideoComponents`, pairing
  the frames and soundtrack through matching `ref_video_0` and
  `ref_video_audio_0` suffixes.

The native reference-video node expects 24-fps frames and does not resample the
input. Use a 24-fps, 2–15 second MP4 with an audio track for the video+audio
workflow.

## Before Running

1. **Model names**: The diffusion-model, text-encoder, and VAE dropdowns must
   match your local installation. See the "Model Directory" section of the
   repository's root README for weight placement rules. T2VA/FL2VA workflows
   require an FL2VA transformer; Ref2VA workflows require a Ref2VA transformer.
2. **Input assets**: The filenames in `LoadImage`, `LoadAudio`, and `LoadVideo` are
   placeholders. Replace them with real files uploaded to ComfyUI's `input/` folder.
3. **Ref2VA prompt tags**: Reference tags are one-based and grouped by media
   type. Keep `<Picture N>`, `<Video N>`, and `<Audio N>` synchronized with the
   corresponding native autogrow socket order.
4. **Target dimensions**: Width and height come from Resolution Selector and
   are rounded to multiples of 32.
