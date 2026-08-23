"""Regenerate the pad-mutation calibration sheet.

Top row: one hit on the same pad at rising velocity, each from the same starting
latent - this is the "harder you hit, the further you go" curve made visible.
Bottom row: repeated full-velocity hits, walking away from where you started.

Use it after changing MUT_STEP, MUT_CURVE or MUT_JITTER. If the top row barely
moves the step is too small; if even a soft tap lands somewhere unrelated it is
too big.

    python pad_sheet.py
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

import midi_latent as M  # inserts upstream/ on sys.path and applies the compat shims
from utils.wrapper import StreamDiffusionWrapper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
VELOCITIES = [0.0, 0.25, 0.5, 0.75, 1.0]
ACCUM = [0, 1, 2, 4, 8]
CELL = 210


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/sd-turbo")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--scene", type=int, default=0, help="index into midi_latent.SCENES")
    p.add_argument("--pad", type=int, default=0, help="which pad to strike")
    p.add_argument("--out", default="pad_mutation.png")
    a = p.parse_args()

    stream = StreamDiffusionWrapper(
        model_id_or_path=a.model, t_index_list=[0], frame_buffer_size=1,
        width=512, height=512, warmup=10, acceleration="none", mode="txt2img",
        use_denoising_batch=True, cfg_type="none", use_lcm_lora=False,
        use_tiny_vae=True, output_type="pt", seed=a.seed,
    )
    scene = M.SCENES[a.scene]
    stream.prepare(prompt=scene, num_inference_steps=50)
    sd = stream.stream
    batch = sd.batch_size
    shape = (4, sd.latent_height, sd.latent_width)
    device, dtype = sd.device, sd.dtype

    emb = M.encode(sd, scene, batch).to(dtype)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    base0 = torch.randn((batch, *shape), generator=gen, dtype=torch.float32).to(device, dtype)
    base_norm = base0.float().norm()
    dirs = M.orthonormal_basis(M.N_PADS, shape, gen, device, dtype)
    dir_norm = dirs[0].float().norm()

    def renorm(z):
        return (z * (base_norm / z.float().norm().clamp(min=1e-6))).to(dtype)

    def hit(base, slot, vel):
        """Exactly the mutation midi_latent applies on a pad strike."""
        d = dirs[slot] + M.MUT_JITTER * torch.randn(
            shape, generator=gen, dtype=torch.float32).to(device, dtype)
        d = d * (dir_norm / d.float().norm().clamp(min=1e-6))
        return renorm(base + (vel ** M.MUT_CURVE) * M.MUT_STEP * d)

    def angle(x, y):
        c = (x.flatten().float() @ y.flatten().float()) / (x.float().norm() * y.float().norm())
        return math.degrees(math.acos(float(c.clamp(-1, 1))))

    def cell(latent, label):
        img = cv2.resize(M.to_bgr(M.generate(sd, latent, emb)), (CELL, CELL))
        cv2.rectangle(img, (0, CELL - 24), (CELL, CELL), (14, 14, 14), -1)
        cv2.putText(img, label, (5, CELL - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (236, 236, 236), 1, cv2.LINE_AA)
        return img

    rows = []

    cells = []
    for v in VELOCITIES:
        b = base0 if v == 0 else hit(base0, a.pad, v)
        cells.append(cell(b, "start" if v == 0 else f"vel {v:.2f}  {angle(base0, b):.0f} deg"))
        print(f"  vel {v:.2f} -> {angle(base0, b):5.1f} deg", flush=True)
    rows.append(np.concatenate(cells, axis=1))

    cells, b, n = [], base0.clone(), 0
    for target in ACCUM:
        while n < target:
            b = hit(b, n % M.N_PADS, 1.0)
            n += 1
        cells.append(cell(b, "start" if n == 0 else f"{n} hits  {angle(base0, b):.0f} deg"))
        print(f"  {n} hits -> {angle(base0, b):5.1f} deg", flush=True)
    rows.append(np.concatenate(cells, axis=1))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.out)
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(path, sheet)
    print(f"wrote {path}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
