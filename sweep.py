"""Sweep StreamDiffusion configs on this GPU and print a throughput table."""
import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))

import torch

import compat

compat.apply()

from utils.wrapper import StreamDiffusionWrapper

PROMPT = "a red fox in a snowy forest, golden hour, highly detailed photo"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

CONFIGS = [
    # name, model, t_index, fbs, w, h, lcm, output_type
    ("sd-turbo 512 1step fbs1", "stabilityai/sd-turbo", [0], 1, 512, 512, False, "pil"),
    ("sd-turbo 512 1step fbs4", "stabilityai/sd-turbo", [0], 4, 512, 512, False, "pil"),
    ("sd-turbo 512 1step fbs8", "stabilityai/sd-turbo", [0], 8, 512, 512, False, "pil"),
    ("sd-turbo 512 1step fbs12", "stabilityai/sd-turbo", [0], 12, 512, 512, False, "pil"),
    ("sd-turbo 512 1step fbs8 (pt)", "stabilityai/sd-turbo", [0], 8, 512, 512, False, "pt"),
    ("sd-turbo 384 1step fbs8", "stabilityai/sd-turbo", [0], 8, 384, 384, False, "pil"),
    ("sd-turbo 512 2step fbs4", "stabilityai/sd-turbo", [0, 16], 4, 512, 512, False, "pil"),
]


def bench(name, model, t_index, fbs, w, h, lcm, output_type, iters=30, save=False):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    stream = StreamDiffusionWrapper(
        model_id_or_path=model, t_index_list=t_index, frame_buffer_size=fbs,
        width=w, height=h, warmup=10, acceleration="none", mode="txt2img",
        use_denoising_batch=True, cfg_type="none", use_lcm_lora=lcm,
        use_tiny_vae=True, output_type=output_type, seed=2,
    )
    stream.prepare(prompt=PROMPT, num_inference_steps=50)
    for _ in range(stream.batch_size + 6):
        stream()
    torch.cuda.synchronize()

    last = None
    t0 = time.perf_counter()
    for _ in range(iters):
        last = stream()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    n_per_call = stream.batch_size if stream.sd_turbo else stream.frame_buffer_size
    n = iters * n_per_call
    vram = torch.cuda.max_memory_allocated() / 2**30

    if save and output_type == "pil":
        os.makedirs(OUT, exist_ok=True)
        imgs = last if isinstance(last, list) else [last]
        imgs[0].save(os.path.join(OUT, f"sweep_{name.replace(' ', '_').replace('(','').replace(')','')}.png"))

    del stream
    gc.collect()
    torch.cuda.empty_cache()
    return n / dt, 1000 * dt / n, vram, n


def main():
    rows = []
    for cfg in CONFIGS:
        name = cfg[0]
        print(f"--- {name}", flush=True)
        try:
            fps, ms, vram, n = bench(*cfg, save=True)
            rows.append((name, fps, ms, vram))
            print(f"    {fps:.2f} img/s  {ms:.1f} ms/img  {vram:.2f} GiB", flush=True)
        except Exception as e:
            rows.append((name, None, None, None))
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)

    print("\n" + "=" * 62)
    print(f"{'config':<32}{'img/s':>10}{'ms/img':>10}{'VRAM':>10}")
    print("=" * 62)
    for name, fps, ms, vram in rows:
        if fps is None:
            print(f"{name:<32}{'FAILED':>10}")
        else:
            print(f"{name:<32}{fps:>10.2f}{ms:>10.1f}{vram:>9.2f}G")
    print("=" * 62)


if __name__ == "__main__":
    main()
