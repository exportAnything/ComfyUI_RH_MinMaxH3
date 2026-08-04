# Dual Sigma Sampler — parameters and how to choose

`RHMiniMaxH3DualSigmaSampler` is the only sampling entry point for H3. It
advances the video and audio streams together, which is why a standard
KSampler cannot stand in for it.

---

## 1. Parameter reference

### Required

| Parameter | Default | Notes |
|---|---|---|
| `seed` | 42 | Video and audio are noised independently; the same seed reproduces a run exactly |
| `sigma_points` | 50 | **Sigma points, not steps.** 50 points produce 49 DiT forwards |
| `video_shift` | 12.0 | Flow shift for the video stream |
| `audio_shift` | 3.0 | Flow shift for the audio stream |
| `accel` | `off` | Approximate acceleration tier, see section 4 |
| `denoise_video` | `true` | `false` = V2A: video is treated as a clean condition, only audio is denoised |

### Optional

| Parameter | Default | Applies when |
|---|---|---|
| `sampler_mode` | `euler` | See section 2 |
| `cache_dit_rdt` | 0.12 | `accel=manual-cache-dit` only |
| `cache_dit_mc` | 2 | `accel=manual-cache-dit` only |
| `cache_dit_warmup` | 4 | `accel=manual-cache-dit` only |
| `velocity_stride` | 4 | `accel=manual-velocity` only |
| `allow_accel_with_res_multistep` | `false` | `sampler_mode=res_multistep` only, see section 2 |

---

## 2. `sampler_mode`: set this first

H3's video and audio run on **different shift schedules**. Both modes integrate
each stream on its own schedule; they differ only in the order of the
integrator.

| | `euler` | `res_multistep` |
|---|---|---|
| Integrator | first order | second-order exponential |
| Suggested `sigma_points` | **50** | **21** |
| DiT calls | 49 | 20 |
| Denoise loop | 248.7s | **101.2s** (2.46×) |
| End to end | 549s | **406s** (1.35×) |
| `accel` | available | off by default, can be enabled explicitly |

> Measured on a single GPU: t2va, 832×480, 125 frames, both runs warm. Same
> seed, no visible quality difference frame by frame (PSNR 26.3 dB — different
> samplers follow different trajectories to different valid solutions, so this
> number measures divergence, not quality).

**Why the two figures differ so much**: the denoise loop is only about 45% of
wall-clock time at this size. The rest is text encoding (Qwen3-VL 32B),
`dit_load` and VAE decode — none of which a sampler change touches. Larger
canvases spend proportionally more time denoising, so the end-to-end figure
moves closer to 2.46×.

Each DiT call costs a constant 5.05s, so **the denoise speedup is exactly the
ratio of DiT calls**.

**Why `res_multistep` halves the step count**: a second-order method's local
error per step is the square of a first-order one's, so 20 steps accumulate
roughly what Euler accumulates in 40–50. Same weights, **no requantisation
needed**.

**Why `accel` is off by default here**: the velocity-cache and Cache-DiT tiers
are calibrated for 50 steps. A second-order run has fewer steps, and therefore
much less headroom to skip any. Measured, stacking velocity-cache on top (same
seed, compared against the euler-50 baseline):

| Configuration | DiT calls | Denoise | End to end | PSNR |
|---|---|---|---|---|
| `euler` 50 | 49/49 | 248.7s | 549s | baseline |
| `res_multistep` 21 | 20/20 | 101.2s | 406s | 26.3 dB |
| `euler` 50 + velocity stride 4 | 14/49 | 70.9s | 501s | 24.1 dB |
| `res_multistep` 21 + velocity stride 2 | 11/20 | 55.8s | 304s | **21.0 dB** |
| `res_multistep` 21 + velocity stride 4 | 7/20 | 35.5s | 176s | **15.3 dB** |
| `res_multistep` 31 + velocity stride 2 | 16/30 | 81.1s | — | 24.2 dB |
| `res_multistep` 41 + velocity stride 4 | 12/40 | 60.6s | — | 20.6 dB |

(Blank end-to-end cells were measured from too different a warm state to
compare across rows; use the denoise and DiT-call columns instead.)

**At 21 points the stack degrades badly**: stride 2 already renders a different
scene (not just noise), stride 4 shows ringing and banding. 20 steps leave no
room to skip another 45–65% of the DiT calls.

**More steps recover it**: 31 points with stride 2 reaches 24.2 dB in 16 DiT
calls, which is perfectly usable. So the restriction is not absolute — the node
exposes `allow_accel_with_res_multistep` (off by default) to enable it.

It is **still not a win**, though. At the same 24 dB tier, `euler`-50 with
stride 4 needs 14 DiT calls / 70.9s, while `res_multistep`-31 with stride 2
needs 16 / 81.1s. A second-order integrator pays off by stepping more
accurately, while velocity-cache skips whole steps — the more steps are
skipped, the less that accuracy can be cashed in. The two accelerations compete
for the same budget.

So: for the best quality-per-cost, use `res_multistep` alone (20 calls,
26.3 dB). To go faster, use `euler` + velocity. Stacking only makes sense if
you already know both sets of knobs and are willing to tune the step count
yourself.

### How to choose

- **Default to `res_multistep` with `sigma_points=21`** — twice as fast, no
  quality loss
- Need to reproduce older results bit for bit, or want an `accel` tier → stay
  on `euler` + `50`

---

## 3. `video_shift` / `audio_shift`

Flow shift controls how the sigma schedule is distributed: a higher value
concentrates sampling steps in the high-noise region (composition, motion,
large structure) and leaves fewer for the low-noise region (detail).

**The defaults of 12.0 / 3.0 are the released configuration; changing them is
not recommended.** The two values must move together — the audio schedule is
derived from the video one, so changing only one breaks audio/visual alignment.

If you must:

| Symptom | Direction |
|---|---|
| Unstable large structure or motion | Raise `video_shift` (say 14–16) |
| Mushy detail, little texture | Lower `video_shift` (say 8–10) |

Re-verify audio/visual sync visually afterwards. The validated acceleration
tiers were only calibrated at `12.0/3.0`, so changing shift disqualifies the
profile tiers (the workload match will reject them).

---

## 4. `accel`: approximate acceleration

**Every tier is an approximation. Output differs from `off` and must not be
used as a reference.** Default is `off`.

| Tier | Mechanism | Calibrated speedup | LPIPS |
|---|---|---|---|
| `off` | no acceleration | 1× | — |
| `auto` | prefers velocity, else cache-dit; stays off if nothing matches | — | — |
| `minimax-h3-velocity-cache-v1` | whole-step velocity reuse + Taylor extrapolation | **3.20×** | 0.426 |
| `minimax-h3-cache-v1` | Cache-DiT block caching (needs `pip install cache-dit`) | **1.99×** | 0.233 |
| `manual-velocity` | set `velocity_stride` yourself | — | — |
| `manual-cache-dit` | set rdt / mc / warmup yourself | — | — |

> The speedups and LPIPS above come from upstream calibration on 4×H200 with a
> workload of **t2va 1344×768 / 124 frames / 50 steps / shift 12/3**. A single
> GPU reuses the same knobs without the multi-GPU gating, so real speedups are
> lower than these numbers.

### The two profile tiers are workload-gated

Parameters outside that calibrated contract **fail with an error** rather than
silently degrading. To use acceleration at other resolutions or step counts,
go through `manual-*`.

### The three `manual-cache-dit` knobs

| Knob | Meaning | Higher means |
|---|---|---|
| `cache_dit_rdt` | residual-difference threshold below which the cache is reused | faster, rougher |
| `cache_dit_mc` | maximum consecutive cached steps | faster, rougher |
| `cache_dit_warmup` | steps at the start that never skip | more stable |

Mapped to the upstream quality tiers:

| Tier | rdt | mc | warmup |
|---|---|---|---|
| upstream high | 0.04 | 1 | 4 |
| **this plugin's validated profile** | **0.08** | **2** | **4** |
| manual default | 0.12 | 2 | 4 |
| upstream medium | 0.12 | 3 | 4 |
| upstream low | 0.24 | 3 | 4 |

**Note that `manual-cache-dit` defaults to rdt 0.12, which is more aggressive
than the validated profile's 0.08.** To match the validated quality, set
`rdt=0.08 / mc=2 / warmup=4` by hand.

The three upstream tiers ship only as knob values with no local measurements,
so they are not offered as presets — `_load_profile` only accepts profiles that
have been measured here. Use the table above with `manual-cache-dit` instead.

### `manual-velocity`'s `velocity_stride`

The DiT refresh stride. `1` computes every step (no acceleration); `4` computes
every fourth step and extrapolates the rest with Taylor. Higher is faster and
rougher.

---

## 5. `denoise_video`

| Value | Behaviour |
|---|---|
| `true` (default) | Normal generation; both video and audio are denoised |
| `false` | **V2A mode**: `av_latent.video` is used as a clean visual condition and only audio is generated |

`false` requires a T2VA layout with a non-zero video latent — fill it via
`Encode Video → AV Latent` or `Combine AV Latent`.

---

## 6. Recommended combinations

**Everyday generation**
```
sampler_mode = res_multistep
sigma_points = 21
accel        = off
```
Fastest without any approximation.

**Reproducing older results / quality baseline**
```
sampler_mode = euler
sigma_points = 50
accel        = off
```

**Squeezing time, accepting visible approximation**
```
sampler_mode = euler
sigma_points = 50
accel        = minimax-h3-velocity-cache-v1
```
Only available at 1344×768 / 124 frames / shift 12/3; other parameters are
rejected. LPIPS 0.426 means the difference from the baseline is quite visible.

**Stacking `res_multistep` + accel**: enable
`allow_accel_with_res_multistep`, but raise `sigma_points` at the same time —
it collapses at 21 points and only becomes usable around 31 with stride 2. It
is not cheaper than `euler` + velocity at equal quality; see the table in
section 2.

**Dubbing an existing video (V2A)**
```
denoise_video = false
```
Use with `Encode Video → AV Latent`.

---

## 7. FAQ

**Should `sigma_points` be 20 or 21?**
21. The parameter counts sigma *points*; 21 points produce 20 DiT forwards.

**Switched to `res_multistep` but nothing got faster?**
`sigma_points` is still 50. The mode itself does not change the step count.

**Selected a profile tier and got a workload mismatch error?**
Both profiles were only calibrated at 1344×768 / 124 frames / 50 steps /
shift 12/3. Use `manual-*`, or move the parameters back onto that contract.

**Output differs from last time?**
Check that `seed`, `sampler_mode`, `sigma_points`, both shifts and `accel` are
all identical. Any one of them changes the result.
