# Benchmark results — RTX 3060 12 GiB

torch 2.11.0+cu128 · diffusers 0.40.0 · Python 3.11 · fp16 · TAESD tiny VAE
No TensorRT, no xformers, no `torch.compile` (see note below). Windows 11.

Timings are steady-state: the stream-batch pipeline is primed and CUDA kernels are
warmed before the clock starts. `img/s` counts finished images, not UNet calls.

## SD-Turbo, single denoising step

| config | img/s | ms/img | peak VRAM |
|---|---:|---:|---:|
| 512×512, frame buffer 1 | 11.37 | 88.0 | 2.59 GiB |
| 512×512, frame buffer 4 | 14.44 | 69.2 | 3.06 GiB |
| 512×512, frame buffer 8 | 15.43 | 64.8 | 3.69 GiB |
| 512×512, frame buffer 12 | 15.51 | 64.5 | 4.31 GiB |
| 512×512, frame buffer 8, tensor output (no PIL) | 17.12 | 58.4 | 3.69 GiB |
| **384×384, frame buffer 8** | **26.17** | **38.2** | 3.14 GiB |

Reading these:

- **Batching is the main lever available here.** Going from frame buffer 1 → 8 buys
  36% throughput; the GPU is latency-bound at batch 1 and compute-bound by batch 8.
  Past 8 it is flat (15.43 → 15.51), so 8 is the knee on this card.
- **PIL conversion costs ~10%** (17.12 → 15.43 img/s). That is CPU-side
  tensor→uint8→PIL work, not generation. Keep `output_type="pt"` if you are feeding
  a display surface or encoder rather than saving files.
- **Resolution dominates.** 384×384 is 1.7× the throughput of 512×512 for
  (512/384)² = 1.78× fewer pixels — almost exactly linear, confirming the UNet is
  compute-bound at these batch sizes.
- VRAM never exceeds 4.4 GiB, so there is headroom on a 12 GiB card; the ceiling here
  is compute, not memory.

### Caveat on multi-step SD-Turbo

`StreamDiffusionWrapper` routes SD-Turbo models to `txt2img_sd_turbo()`, which issues
**one** UNet call over the whole batch and ignores the stream-batch buffer. So passing
a longer `t_index_list` with an SD-Turbo model does not add denoising steps — it only
multiplies the UNet batch size. Measured: `2step fbs4` gave 15.52 img/s, matching
`1step fbs8` (15.43) because both are simply batch-8 single-step. Multi-step
denoising requires a non-turbo model, below.

## Note on further acceleration

Upstream's fastest paths are unavailable or impractical on this box:

- **TensorRT** — upstream's `acceleration="tensorrt"` needs `polygraphy`, `tensorrt`,
  `onnx`, and `cuda-python`, pinned to versions from 2023. This is where upstream's
  headline numbers come from and would likely be the largest remaining win.
- **`torch.compile`** — needs Triton and MSVC ≥ 2022 on Windows; neither is present
  (`cl.exe` not on PATH, only VS 2017/2019 installed, no `triton`).
- **xformers** — skipped deliberately. Current diffusers already defaults to torch
  SDPA attention, which is competitive with xformers on Ampere.

## SD 1.5 + LCM-LoRA — the actual stream-batch algorithm

SD-Turbo bypasses StreamDiffusion's stream batch entirely (see caveat above), so the
algorithm's contribution only shows up on a multi-step model. All rows 512×512,
frame buffer 1, so each call returns exactly one finished image.

| config | img/s | ms/img | peak VRAM |
|---|---:|---:|---:|
| 4 steps, **sequential** (`use_denoising_batch=False`) | 3.32 | 300.9 | 2.31 GiB |
| 4 steps, **stream batch** | 4.40 | 227.2 | 2.41 GiB |
| 2 steps, stream batch | 7.23 | 138.3 | 2.31 GiB |
| 1 step, stream batch | 10.79 | 92.7 | 2.31 GiB |

**Stream batch is worth 1.33× at 4 steps** (300.9 → 227.2 ms/img) for +0.1 GiB.

This is the whole idea of the paper: instead of running four sequential UNet calls at
batch 1 to finish one image, keep four *different* images in flight — each at a
different point in the denoising schedule — and denoise all four in a single batch-4
UNet call. Every call retires one finished image and admits one new noise sample. Same
arithmetic, but issued as one large kernel launch instead of four small ones, which is
exactly what an underutilised GPU wants.

The gain is bounded by how latency-bound the batch-1 case was — 1.33× here, and it
would be larger on a card with more idle SMs at batch 1.

## Which model to use on this card

| | SD-Turbo, 1 step | SD 1.5 + LCM-LoRA, 4 steps |
|---|---|---|
| throughput | 11.4 img/s | 4.4 img/s |
| quality at that setting | sharp, usable | sharp, usable |
| quality at 1 step | — | blurry, unusable |

LCM-LoRA needs ~4 steps to resolve; at 1 step (`outputs/lcm1_stream_0.png`) it is a
smear, even though it runs at 10.8 img/s. SD-Turbo is distilled for single-step
sampling and is sharp there. **For maximum speed on this device, SD-Turbo at 1 step
with frame buffer 8 is the configuration to use.**

## Headline

- **26 img/s** at 384×384 (38 ms/img)
- **15.4 img/s** at 512×512 (65 ms/img)
- **12 images with 12 different prompts in 1.40 s** (`outputs/contact_turbo.png`)
- under 4.4 GiB VRAM in every configuration
