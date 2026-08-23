"""Play SD-Turbo with a MIDI controller, in real time.

All eight knobs are *semantic* axes: each is a direction in CLIP text-embedding
space, built as the difference between two contrasting prompts and rescaled so every
knob lands with comparable force. Turning a knob slides the image along that concept.
Pads mutate the base noise latent for good. Each pad owns the axis of the same
number, every hit jumps a random distance along it, and the harder you hit the
further you go - so the image wanders somewhere new and stays there. Press r to
start over from a fresh latent.

    .venv/Scripts/python.exe midi_latent.py
    .venv/Scripts/python.exe midi_latent.py --width 384 --height 384   # ~26 fps

Knobs (left to right, after you have moved each one once):
    1 man <-> woman          4 busy <-> silent      7 japanese <-> western
    2 fantasy <-> cyberpunk  5 new <-> old          8 organic <-> mechanic
    3 life <-> artificial    6 photo <-> anime

Semantic depth is fixed at SEM_DEPTH (1.5) so every knob can be an axis.

Pads: velocity-scaled permanent mutations of the base latent, one axis each.

Keys in the window:
    r  reset the latent         d  toggle drift        s  save frame
    n  new pad directions       h  toggle HUD          [ / ]  prev / next scene
    c  recentre knobs           - / =  mutation size   q / esc  quit
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream"))

import cv2
import mido
import numpy as np
import torch

import compat

compat.apply()

from utils.wrapper import StreamDiffusionWrapper

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# Scenes the axes act on. [ and ] cycle these.
SCENES = [
    "a portrait photograph, natural light, highly detailed, 35mm",
    "a full body photograph, studio light, highly detailed",
    "a wide landscape photograph, golden hour, highly detailed",
    "a street photograph at night, neon, 35mm",
]

# (left label, right label, prompt at knob -1, prompt at knob +1, strength)
#
# Both prompts share a template so the difference isolates the concept, and minimal
# pairs beat verbose ones - extra adjectives spread the difference over filler tokens
# and dilute it. `strength` is the length each direction is rescaled to; it is
# calibrated per axis because concepts do not all need the same push to land against
# a portrait base (a robot has to overcome a very confident human face; a wolf does
# not). Raw lengths vary 76..200, so without this some knobs would do nothing while
# others blew the image out.
AXES = [
    ("man", "woman",
     "a portrait photograph of a man, male, masculine face",
     "a portrait photograph of a woman, female, feminine face", 100.0),
    # Two opposite departures from the real, so centre is the unmodified scene:
    # magic and the pre-industrial past one way, neon and the high-tech future the
    # other. About genre and setting, not material - that is knob 8's job.
    ("fantasy", "cyberpunk",
     "a photograph of a fantasy world, magic, medieval, dragons, enchanted",
     "a photograph of a cyberpunk world, neon, chrome, cybernetic, futuristic", 95.0),
    ("life", "artificial",
     "a portrait photograph of a living organic creature, flesh, alive",
     "a photograph of a machine, robot, chrome, circuitry, mechanical", 93.0),
    ("busy", "silent",
     "a crowded busy chaotic scene, many people, traffic, clutter, noise",
     "an empty silent still scene, nobody, bare, minimal, quiet", 105.0),
    ("new", "old",
     "a portrait photograph of a young child, youthful, smooth skin",
     "a portrait photograph of a very old elderly person, deeply wrinkled, ancient", 105.0),
    ("photo", "anime",
     "a photorealistic photograph of a person, real life, 35mm",
     "an anime manga illustration of a person, cel shaded drawing", 100.0),
    ("japanese", "western",
     "a photograph of a japanese person in japan, japanese architecture",
     "a photograph of a western european person in europe, western architecture", 85.0),
    # Deliberately about material and form, not aliveness - that is knob 3's job.
    ("organic", "mechanic",
     "a photograph of soft organic natural forms, flowing curves, plants, flesh",
     "a photograph of hard mechanical engineered forms, gears, metal parts", 88.0),
]

N_PADS = 8

SEM_DEPTH = 1.5      # semantic depth, fixed - all eight knobs are axes now
SEM_GAIN = 1.0       # multiplier at knob = 1.0 and depth = 1.0
MUT_STEP = 0.6       # how far a full-velocity hit moves the base latent
MUT_CURVE = 1.5      # velocity exponent - soft taps nudge, hard hits leap
MUT_JITTER = 0.7     # fresh randomness blended into the pad's own direction, so
                     # no two hits on the same pad ever land in the same place
PAD_TAU = 0.35       # HUD flash decay, seconds - the mutation itself is permanent
DRIFT = 0.014        # random-walk step per frame when drift is on


# --------------------------------------------------------------------------- midi

class Midi:
    """Reads a controller on a background thread and keeps a snapshot of its state.

    Slots are assigned by sorting the CC / note numbers seen so far, so once you
    have touched every control the mapping matches the physical layout regardless
    of which preset the controller is in.
    """

    def __init__(self, port_hint=None):
        names = mido.get_input_names()
        if not names:
            raise RuntimeError("no MIDI input ports found - is the controller plugged in?")
        if port_hint:
            matches = [n for n in names if port_hint.lower() in n.lower()]
            if not matches:
                raise RuntimeError(f"no port matching {port_hint!r}; available: {names}")
            self.name = matches[0]
        else:
            self.name = names[0]

        self.raw = {}            # cc number -> 0..127
        self.cc_slots = []       # sorted cc numbers
        self.note_slots = []     # sorted note numbers
        self.impulses = deque()  # (slot, amp, t0) - HUD flash only
        self.pending = deque()   # (slot, velocity) - unconsumed mutations
        self.offset = {}         # cc -> value treated as centre
        self.lock = threading.Lock()
        self._stop = False

        self.port = mido.open_input(self.name)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        for msg in self.port:
            if self._stop:
                return
            now = time.perf_counter()
            with self.lock:
                if msg.type == "control_change":
                    self.raw[msg.control] = msg.value
                    if msg.control not in self.cc_slots:
                        self.cc_slots = sorted(self.raw)
                elif msg.type == "note_on" and msg.velocity > 0:
                    if msg.note not in self.note_slots:
                        self.note_slots = sorted(set(self.note_slots) | {msg.note})
                    slot = self.note_slots.index(msg.note)
                    if slot < N_PADS:
                        vel = msg.velocity / 127.0
                        self.impulses.append((slot, vel, now))
                        self.pending.append((slot, vel))

    def knob(self, slot, default=0.0):
        """Knob `slot` as -1..1 around its centre, or `default` until first moved."""
        with self.lock:
            if slot >= len(self.cc_slots):
                return default
            cc = self.cc_slots[slot]
            v = self.raw.get(cc)
            centre = self.offset.get(cc, 64)
        if v is None:
            return default
        span = max(centre, 127 - centre) or 1
        return float(np.clip((v - centre) / span, -1.0, 1.0))

    def knob01(self, slot, default=0.0):
        """Knob `slot` as 0..1 absolute, or `default` until first moved."""
        with self.lock:
            if slot >= len(self.cc_slots):
                return default
            v = self.raw.get(self.cc_slots[slot])
        return default if v is None else v / 127.0

    def recentre(self):
        with self.lock:
            self.offset = dict(self.raw)

    def take_hits(self):
        """Drain the pad hits since the last call, as (slot, velocity) pairs.

        Consuming rather than sampling matters here: a mutation must be applied
        exactly once, however many frames the hit straddles.
        """
        with self.lock:
            hits, self.pending = list(self.pending), deque()
        return hits

    def pad_envelopes(self, now):
        amps = np.zeros(N_PADS, dtype=np.float32)
        with self.lock:
            while self.impulses and now - self.impulses[0][2] > 6 * PAD_TAU:
                self.impulses.popleft()
            for slot, amp, t0 in self.impulses:
                amps[slot] += amp * float(np.exp(-(now - t0) / PAD_TAU))
        return amps

    def close(self):
        self._stop = True
        try:
            self.port.close()
        except Exception:
            pass


# ------------------------------------------------------------------------- tensors

def orthonormal_basis(n, shape, generator, device, dtype):
    """n independent directions in latent space, scaled to unit-sigma noise."""
    flat = torch.randn((n, int(np.prod(shape))), generator=generator,
                       device="cpu", dtype=torch.float32)
    q, _ = torch.linalg.qr(flat.T)
    basis = q.T * float(np.sqrt(np.prod(shape)))
    return basis.reshape((n, *shape)).to(device=device, dtype=dtype)


@torch.no_grad()
def encode(sd, prompt, batch):
    out = sd.pipe.encode_prompt(prompt=prompt, device=sd.device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False)
    return out[0].repeat(batch, 1, 1)


@torch.no_grad()
def axis_vectors(sd, batch):
    """One rescaled CLIP direction per semantic axis, so each knob lands equally."""
    vecs = []
    for left_lbl, right_lbl, lp, rp, strength in AXES:
        d = encode(sd, rp, batch) - encode(sd, lp, batch)
        raw = d.float().norm()
        vecs.append((d * (strength / raw.clamp(min=1e-6))).to(sd.dtype))
        print(f"  {left_lbl:>6} <-> {right_lbl:<11} |d| {raw:6.1f} -> {strength:.0f}")
    return vecs


@torch.no_grad()
def generate(sd, latent, embeds):
    """One SD-Turbo denoising step on a latent we control, then decode."""
    model_pred = sd.unet(latent, sd.sub_timesteps_tensor,
                         encoder_hidden_states=embeds, return_dict=False)[0]
    x0 = (latent - sd.beta_prod_t_sqrt * model_pred) / sd.alpha_prod_t_sqrt
    return sd.decode_image(x0)


def to_bgr(tensor):
    img = (tensor[0].permute(1, 2, 0) / 2 + 0.5).clamp(0, 1)
    return img.mul(255).to(torch.uint8).cpu().numpy()[:, :, ::-1].copy()


# ------------------------------------------------------------------------- display

FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL = 204
INK = (196, 206, 216)
DIM = (132, 140, 150)


def draw_hud(frame, knobs, mut_size, amps, fps, scene, drift, saved):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - PANEL), (w, h), (16, 13, 11), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)

    top = h - PANEL
    cv2.putText(frame, f"{fps:5.1f} fps", (14, top + 20), FONT, 0.48,
                (210, 235, 255), 1, cv2.LINE_AA)
    tag = "SAVED" if saved else ("DRIFT" if drift else "")
    if tag:
        cv2.putText(frame, tag, (108, top + 20), FONT, 0.46,
                    (120, 235, 190), 1, cv2.LINE_AA)
    text = scene if len(scene) <= 60 else scene[:57] + "..."
    cv2.putText(frame, text, (14, top + 38), FONT, 0.38, DIM, 1, cv2.LINE_AA)

    # five bipolar semantic axes
    bar_x, bar_w, row_h = 92, 148, 17
    y = top + 58
    for i, (lname, rname, *_rest) in enumerate(AXES):
        cy = y + i * row_h
        # pad i mutates the latent along axis i - flash the labels so you can see it
        lab = tuple(int(c + (250 - c) * float(np.clip(amps[i], 0, 1))) for c in DIM)
        (tw, _), _ = cv2.getTextSize(lname, FONT, 0.36, 1)
        cv2.putText(frame, lname, (bar_x - 8 - tw, cy + 5), FONT, 0.36, lab, 1, cv2.LINE_AA)
        cv2.putText(frame, rname, (bar_x + bar_w + 8, cy + 5), FONT, 0.36, lab, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, cy - 3), (bar_x + bar_w, cy + 4), (54, 60, 68), -1)
        mid = bar_x + bar_w // 2
        v = int(knobs[i] * (bar_w // 2))
        col = (60, 150, 235) if v >= 0 else (225, 150, 60)
        if v >= 0:
            cv2.rectangle(frame, (mid, cy - 3), (mid + v, cy + 4), col, -1)
        else:
            cv2.rectangle(frame, (mid + v, cy - 3), (mid, cy + 4), col, -1)
        cv2.line(frame, (mid, cy - 6), (mid, cy + 7), (140, 148, 158), 1)

    # right column: depths + pads
    rx = w - 208
    cv2.putText(frame, f"sem {SEM_DEPTH:.1f} fixed", (rx, top + 20), FONT, 0.36,
                (140, 200, 90), 1, cv2.LINE_AA)
    yy = top + 28
    cv2.putText(frame, "mut", (rx, yy + 8), FONT, 0.36, DIM, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (rx + 56, yy), (rx + 176, yy + 7), (54, 60, 68), -1)
    cv2.rectangle(frame, (rx + 56, yy),
                  (rx + 56 + int(120 * np.clip(mut_size / 2.0, 0, 1)), yy + 7),
                  (90, 170, 235), -1)

    py = top + 60
    cv2.putText(frame, "pads", (rx, py + 12), FONT, 0.36, DIM, 1, cv2.LINE_AA)
    for i in range(N_PADS):
        a = float(np.clip(amps[i], 0, 1))
        col = (int(46 + 150 * a), int(42 + 90 * a), int(52 + 210 * a))
        x = rx + 56 + (i % 4) * 31
        yy = py + (i // 4) * 22
        cv2.rectangle(frame, (x, yy), (x + 25, yy + 17), col, -1)
    return frame


# ---------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="stabilityai/sd-turbo")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--scale", type=float, default=1.5, help="window magnification")
    p.add_argument("--port", default=None, help="substring of the MIDI port name")
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()

    midi = Midi(a.port)
    print(f"MIDI in: {midi.name}")

    stream = StreamDiffusionWrapper(
        model_id_or_path=a.model, t_index_list=[0], frame_buffer_size=1,
        width=a.width, height=a.height, warmup=10, acceleration="none",
        mode="txt2img", use_denoising_batch=True, cfg_type="none",
        use_lcm_lora=False, use_tiny_vae=True, output_type="pt", seed=a.seed,
    )
    stream.prepare(prompt=SCENES[0], num_inference_steps=50)
    sd = stream.stream
    batch = sd.batch_size
    shape = (4, sd.latent_height, sd.latent_width)
    device, dtype = sd.device, sd.dtype

    print("semantic axes:")
    axes = axis_vectors(sd, batch)
    scene_embs = [encode(sd, s, batch) for s in SCENES]

    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    def new_base():
        z = torch.randn((batch, *shape), generator=gen, dtype=torch.float32)
        return z.to(device=device, dtype=dtype)

    base = new_base()
    base_norm = base.float().norm()
    latent_dirs = orthonormal_basis(N_PADS, shape, gen, device, dtype)
    dir_norm = latent_dirs[0].float().norm()

    def renorm(z):
        """Back onto the noise hypersphere. Mutations accumulate, and off the
        sphere the UNet gets something it never saw in training and washes out."""
        return (z * (base_norm / z.float().norm().clamp(min=1e-6))).to(dtype)

    win = "StreamDiffusion - MIDI"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    scene_i, show_hud, drift, saved_until = 0, True, False, 0.0
    mut_size = 1.0          # - and = adjust; all eight knobs are spoken for
    fps, frames, t_fps = 0.0, 0, time.perf_counter()
    print("\nrunning - focus the image window for keys, ctrl-c here to stop\n")

    try:
        while True:
            now = time.perf_counter()

            knobs = np.array([midi.knob(i) for i in range(len(AXES))], dtype=np.float32)

            amps = midi.pad_envelopes(now)

            # --- semantics live in the text embedding
            emb = scene_embs[scene_i]
            for i, k in enumerate(knobs):
                if abs(k) > 1e-3:
                    emb = emb + (float(k) * SEM_DEPTH * SEM_GAIN) * axes[i]

            # --- pads mutate the base latent for good, one axis each
            for slot, vel in midi.take_hits():
                d = latent_dirs[slot] + MUT_JITTER * torch.randn(
                    shape, generator=gen, dtype=torch.float32).to(device, dtype)
                d = d * (dir_norm / d.float().norm().clamp(min=1e-6))
                base = renorm(base + (vel ** MUT_CURVE) * MUT_STEP * mut_size * d)

            if drift:
                base = renorm(base + DRIFT * torch.randn(base.shape, generator=gen,
                                                         dtype=torch.float32).to(device, dtype))
            latent = base

            frame = to_bgr(generate(sd, latent, emb.to(dtype)))

            frames += 1
            if now - t_fps >= 0.5:
                fps, frames, t_fps = frames / (now - t_fps), 0, now

            if a.scale != 1.0:
                frame = cv2.resize(frame, None, fx=a.scale, fy=a.scale,
                                   interpolation=cv2.INTER_LINEAR)
            if show_hud:
                frame = draw_hud(frame, knobs, mut_size, amps, fps,
                                 SCENES[scene_i], drift, now < saved_until)

            cv2.imshow(win, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("r"):
                base = new_base()
                base_norm = base.float().norm()
            elif k == ord("n"):
                latent_dirs = orthonormal_basis(N_PADS, shape, gen, device, dtype)
            elif k == ord("c"):
                midi.recentre()
            elif k == ord("-"):
                mut_size = max(0.0, mut_size - 0.1)
            elif k in (ord("="), ord("+")):
                mut_size = min(2.0, mut_size + 0.1)
            elif k == ord("d"):
                drift = not drift
            elif k == ord("h"):
                show_hud = not show_hud
            elif k == ord("]"):
                scene_i = (scene_i + 1) % len(SCENES)
            elif k == ord("["):
                scene_i = (scene_i - 1) % len(SCENES)
            elif k == ord("s"):
                os.makedirs(OUT, exist_ok=True)
                path = os.path.join(OUT, f"midi_{int(time.time())}.png")
                cv2.imwrite(path, to_bgr(generate(sd, latent, emb.to(dtype))))
                print(f"saved {path}")
                saved_until = now + 1.0

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        midi.close()
        cv2.destroyAllWindows()
        print("stopped")


if __name__ == "__main__":
    main()
