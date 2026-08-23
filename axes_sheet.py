"""Regenerate the semantic-axis calibration sheet.

Sweeps every axis in midi_latent.AXES from -1 to +1 with the noise latent held fixed,
so the only thing changing across a row is the concept. Use it after editing an axis
prompt pair or its strength - if a row's extremes overshoot into abstraction, the
strength is too high; if nothing moves, it is too low.

    python axes_sheet.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

import midi_latent as M  # inserts upstream/ on sys.path and applies the compat shims
from utils.wrapper import StreamDiffusionWrapper

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
STEPS = [-1.0, -0.5, 0.0, 0.5, 1.0]
CELL = 210


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/sd-turbo")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--scene", type=int, default=0, help="index into midi_latent.SCENES")
    p.add_argument("--out", default="semantic_axes.png")
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

    axes = M.axis_vectors(sd, batch)
    base = M.encode(sd, scene, batch)

    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    latent = torch.randn((batch, 4, sd.latent_height, sd.latent_width),
                         generator=gen, dtype=torch.float32).to(sd.device, sd.dtype)

    rows = []
    for i, (left, right, *_rest) in enumerate(M.AXES):
        cells = []
        for k in STEPS:
            emb = base + (k * M.SEM_DEPTH * M.SEM_GAIN) * axes[i]
            img = cv2.resize(M.to_bgr(M.generate(sd, latent, emb.to(sd.dtype))),
                             (CELL, CELL))
            label = f"{left if k < 0 else right if k > 0 else 'neutral'} {k:+.1f}"
            cv2.rectangle(img, (0, CELL - 24), (CELL, CELL), (14, 14, 14), -1)
            cv2.putText(img, label, (5, CELL - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (236, 236, 236), 1, cv2.LINE_AA)
            cells.append(img)
        rows.append(np.concatenate(cells, axis=1))
        print(f"  swept {left} <-> {right}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.out)
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(path, sheet)
    print(f"wrote {path}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()
