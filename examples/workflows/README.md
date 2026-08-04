# MiniMax-H3 Frontend Workflows

Covers every task type in `H3_TASKS` and all valid conditioning/reference configurations.
Each file uses the ComfyUI frontend format (`nodes` + `links`) and can be dragged
directly onto the canvas or opened with **Load**.

| Task | Workflow | Conditioning input |
|------|--------|----------|
| T2VA | `t2va.json` | None |
| T2VA | `t2va_native_sage_attention.json` | None; native ComfyUI graph with SageAttention |
| T2VA | `t2va_native_sol_attn.json` | None; native ComfyUI graph with Sol-Attn |
| FL2VA | `fl2va_first_frame.json` | First-frame image |
| FL2VA | `fl2va_last_frame.json` | Last-frame image |
| FL2VA | `fl2va_first_last_frame.json` | First-frame + last-frame images |
| Ref2VA | `ref2va_image.json` | Reference image |
| Ref2VA | `ref2va_image_audio.json` | Reference image → reference audio (ordered chain) |
| Ref2VA | `ref2va_video_audio.json` | Reference video with an audio track |

Shared defaults: explicit `832×480` resolution, 5 seconds, 50 sigma points,
shift `12/3`, `accel=off`, and `denoise_video=true`. The Loaders use native FP8
DiT, NVFP4 Qwen, and single-file VAE names from the documentation. If your local
installation uses RunningHub INT8-CONVROT or BF16 shards, select the matching
entries from the dropdowns.

## Native Attention Variants

The two `t2va_native_*` workflows use ComfyUI's native MiniMax H3 implementation
and require ComfyUI 0.30.1 or newer. They are preconfigured for the official
INT8-CONVROT transformer, NVFP4-AWQ text encoder, and native video/audio VAEs.

Choose an attention workflow:

- `t2va_native_sage_attention.json` uses the fork-owned SageAttention patch.
- `t2va_native_sol_attn.json` chains the fork-owned SageAttention patch into the
  experimental Sol-style sparse-attention patch. Sol handles eligible middle
  steps; Sage is retained as the fallback for every other attention call.

No additional Sage or Sol custom-node pack is required. A compatible
hardware-specific `sageattention` Python/CUDA library remains optional; when it
is missing, the included Sage node safely leaves normal ComfyUI attention active.
The included Sol-style node has no additional dependency beyond ComfyUI's PyTorch
runtime. Both graphs are already wired between Load Diffusion Model and the
scheduler/guider.

## Before Running

1. **Model names**: The three Loaders' `model_root` values and component dropdowns
   must match your local installation. See the "Model Directory" section of the
   repository's root README for weight placement rules. T2VA/FL2VA workflows
   require an FL2VA transformer; the Ref2VA FP8 file cannot replace it.
2. **Input assets**: The filenames in `LoadImage`, `LoadAudio`, and `LoadVideo` are
   placeholders. Replace them with real files uploaded to ComfyUI's `input/` folder.
3. **Ref2VA reference order**: The reference chain is strictly ordered. Connect each
   reference node's `references` output to the next reference node. Changing the chain
   order changes the order of the multimodal prompt and conditioning rows.
4. **Target dimensions**: `width` and `height` are explicitly set. When left blank,
   they are resolved from `aspect_ratio`; Ref2VA will resolve to 1344×768, greatly
   increasing the sequence length and processing time.
