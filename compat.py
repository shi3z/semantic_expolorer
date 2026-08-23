"""Compatibility shims so upstream StreamDiffusion (pinned to diffusers 0.24)
runs on a modern diffusers / transformers / torch stack.

Import this BEFORE utils.wrapper.
"""
import torch
import diffusers
from diffusers import StableDiffusionPipeline

_patches = []


# --- 1. fuse_lora(): diffusers dropped fuse_unet=/fuse_text_encoder= in favour
#        of components=[...] around 0.28. Re-express the old call in the new API.
def _patch_fuse_lora():
    from streamdiffusion.pipeline import StreamDiffusion

    def fuse_lora(self, fuse_unet=True, fuse_text_encoder=True,
                  lora_scale=1.0, safe_fusing=False):
        components = []
        if fuse_unet:
            components.append("unet")
        if fuse_text_encoder:
            components.append("text_encoder")
        self.pipe.fuse_lora(components=components, lora_scale=lora_scale,
                            safe_fusing=safe_fusing)

    StreamDiffusion.fuse_lora = fuse_lora
    _patches.append("StreamDiffusion.fuse_lora -> components= API")


# --- 2. Never pull the 1.2 GB safety checker; upstream calls from_pretrained()
#        with no kwargs at all.
def _patch_from_pretrained():
    orig = StableDiffusionPipeline.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("safety_checker", None)
        kwargs.setdefault("requires_safety_checker", False)
        kwargs.setdefault("torch_dtype", torch.float16)
        if "variant" not in kwargs:
            # halves the download when the repo ships fp16 weights
            try:
                return orig(cls, *args, variant="fp16", **kwargs)
            except Exception:
                pass
        return orig(cls, *args, **kwargs)

    StableDiffusionPipeline.from_pretrained = classmethod(from_pretrained)
    _patches.append("StableDiffusionPipeline.from_pretrained -> no safety checker, fp16")


# --- 3. retrieve_latents moved out of the img2img pipeline module in newer
#        diffusers; put it back where upstream expects it.
def _patch_retrieve_latents():
    mod = diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img
    if hasattr(mod, "retrieve_latents"):
        return

    def retrieve_latents(encoder_output, generator=None, sample_mode="sample"):
        if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
            return encoder_output.latent_dist.sample(generator)
        if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
            return encoder_output.latent_dist.mode()
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        raise AttributeError("Could not access latents of provided encoder_output")

    mod.retrieve_latents = retrieve_latents
    _patches.append("retrieve_latents restored")


def apply():
    _patch_retrieve_latents()
    _patch_from_pretrained()
    _patch_fuse_lora()
    print(f"[compat] diffusers {diffusers.__version__}, torch {torch.__version__}")
    for p in _patches:
        print(f"[compat]   {p}")
    return _patches
