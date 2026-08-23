"""StreamDiffusion throughput benchmark + image dump for this machine."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))

import torch
import compat

compat.apply()

from utils.wrapper import StreamDiffusionWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def build(model, t_index, fbs, w, h, use_lcm, acceleration, denoising_batch=True):
    return StreamDiffusionWrapper(
        model_id_or_path=model,
        t_index_list=t_index,
        frame_buffer_size=fbs,
        width=w,
        height=h,
        warmup=10,
        acceleration=acceleration,
        mode="txt2img",
        use_denoising_batch=denoising_batch,
        cfg_type="none",
        use_lcm_lora=use_lcm,
        use_tiny_vae=True,
        output_type="pil",
        seed=2,
    )


def run(stream, prompt, iters, tag, save=2):
    stream.prepare(prompt=prompt, num_inference_steps=50)

    # fill the stream-batch pipeline, then warm the kernels
    for _ in range(stream.batch_size - 1):
        stream()
    for _ in range(6):
        stream()
    torch.cuda.synchronize()

    per_call = []
    imgs_last = None
    t0 = time.perf_counter()
    for _ in range(iters):
        c0 = time.perf_counter()
        imgs_last = stream()
        torch.cuda.synchronize()
        per_call.append(time.perf_counter() - c0)
    total = time.perf_counter() - t0

    n_per_call = stream.batch_size if stream.sd_turbo else stream.frame_buffer_size
    n_imgs = iters * n_per_call
    fps = n_imgs / total
    ms = 1000.0 * total / n_imgs

    os.makedirs(OUT, exist_ok=True)
    imgs = imgs_last if isinstance(imgs_last, list) else [imgs_last]
    for i, im in enumerate(imgs[:save]):
        im.save(os.path.join(OUT, f"{tag}_{i}.png"))

    peak = torch.cuda.max_memory_allocated() / 2**30
    return dict(tag=tag, fps=fps, ms=ms, imgs=n_imgs, total=total, vram=peak,
                batch=stream.batch_size, n_per_call=n_per_call)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/sd-turbo")
    p.add_argument("--prompt", default="a photo of a red fox in a snowy forest, golden hour, highly detailed")
    p.add_argument("--t-index", default="0")
    p.add_argument("--fbs", type=int, default=1)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--lcm-lora", action="store_true")
    p.add_argument("--acceleration", default="none", choices=["none", "xformers", "tensorrt"])
    p.add_argument("--no-denoising-batch", action="store_true",
                   help="disable StreamDiffusion stream batch (sequential denoising)")
    p.add_argument("--tag", default="run")
    a = p.parse_args()

    t_index = [int(x) for x in a.t_index.split(",")]
    stream = build(a.model, t_index, a.fbs, a.width, a.height, a.lcm_lora,
                   a.acceleration, denoising_batch=not a.no_denoising_batch)
    r = run(stream, a.prompt, a.iters, a.tag)
    print(
        f"\n[{r['tag']}] {a.width}x{a.height} steps={len(t_index)} fbs={a.fbs} "
        f"unet_batch={r['batch']}\n"
        f"  {r['fps']:.2f} img/s   {r['ms']:.1f} ms/img   "
        f"({r['imgs']} imgs in {r['total']:.2f}s)   peak VRAM {r['vram']:.2f} GiB"
    )


if __name__ == "__main__":
    main()
