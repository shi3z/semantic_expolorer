"""Fast image generation demo: streams N images and tiles them into a contact sheet."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))

import torch
from PIL import Image

import compat

compat.apply()

from utils.wrapper import StreamDiffusionWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

PROMPTS = [
    "a red fox in a snowy forest, golden hour, highly detailed photo",
    "a lighthouse on a cliff during a storm, dramatic sky, cinematic",
    "a bowl of ramen on a wooden table, steam, shallow depth of field",
    "a neon-lit tokyo alley at night, rain, reflections, 35mm photo",
    "a hot air balloon over lavender fields at sunrise",
    "an astronaut sitting on a rock on mars, wide shot, film still",
    "a mossy stone bridge in a misty pine forest",
    "a vintage motorcycle in a garage, warm tungsten light",
    "a snow leopard on a rocky ledge, national geographic photo",
    "a glass of iced coffee on a marble counter, morning light",
    "a wooden sailboat on a calm turquoise sea, aerial view",
    "a cozy library with tall shelves and a reading lamp",
]


def tile(images, cols, path, cell=None):
    if cell is None:
        cell = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell[0], rows * cell[1]), "black")
    for i, im in enumerate(images):
        if im.size != cell:
            im = im.resize(cell)
        sheet.paste(im, ((i % cols) * cell[0], (i // cols) * cell[1]))
    sheet.save(path)
    return sheet.size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/sd-turbo")
    p.add_argument("--t-index", default="0")
    p.add_argument("--fbs", type=int, default=1)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--lcm-lora", action="store_true")
    p.add_argument("--out", default="contact_sheet.png")
    a = p.parse_args()

    t_index = [int(x) for x in a.t_index.split(",")]
    stream = StreamDiffusionWrapper(
        model_id_or_path=a.model,
        t_index_list=t_index,
        frame_buffer_size=a.fbs,
        width=a.width,
        height=a.height,
        warmup=10,
        acceleration="none",
        mode="txt2img",
        use_denoising_batch=True,
        cfg_type="none",
        use_lcm_lora=a.lcm_lora,
        use_tiny_vae=True,
        output_type="pil",
        seed=2,
    )

    images = []
    stream.prepare(prompt=PROMPTS[0], num_inference_steps=50)
    # prime the stream-batch pipeline and warm CUDA kernels before timing
    for _ in range(stream.batch_size + 5):
        stream()
    torch.cuda.synchronize()

    print(f"\ngenerating {a.count} images at {a.width}x{a.height} ...")
    t0 = time.perf_counter()
    i = 0
    while len(images) < a.count:
        prompt = PROMPTS[i % len(PROMPTS)]
        stream.stream.update_prompt(prompt)
        # a prompt change needs the in-flight pipeline stages refreshed
        for _ in range(len(t_index) - 1):
            stream()
        out = stream()
        out = out if isinstance(out, list) else [out]
        images.extend(out)
        i += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    images = images[: a.count]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.out)
    size = tile(images, a.cols, path)
    print(
        f"{len(images)} images in {dt:.2f}s  ->  {len(images)/dt:.2f} img/s, "
        f"{1000*dt/len(images):.1f} ms/img"
    )
    print(f"contact sheet: {path}  ({size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
