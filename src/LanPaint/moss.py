"""MOSS image/mask preparation and measured native LanPaint sampling."""

import time

import torch


class MossLanPaintPrepareImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",), "mask": ("MASK",)},
            "optional": {"source_fill": (["neutral", "source"], {"default": "neutral"})},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("neutral_image", "edit_mask")
    FUNCTION = "prepare"
    CATEGORY = "MOSS/LanPaint"

    def prepare(self, image, mask, source_fill="neutral"):
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("Source must be an RGB image [B,H,W,3].")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or tuple(mask.shape) != tuple(image.shape[:3]):
            raise ValueError("Source and grayscale mask must have identical dimensions and batch size.")
        if min(image.shape[1:3]) <= 0 or any(n % 16 for n in image.shape[1:3]):
            raise ValueError("Image width and height must be positive multiples of 16.")
        if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
            raise ValueError("Source and mask must contain finite values.")
        edit = (mask > 0.5).to(image)
        if source_fill not in ("neutral", "source"):
            raise ValueError("Source fill must be neutral or source.")
        neutral = torch.where(edit.unsqueeze(-1).bool(), 128 / 255, image) if source_fill == "neutral" else image
        return neutral, edit


class MossLanPaintPrepareMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "dilation": ("INT", {"default": 0, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "prepare"
    CATEGORY = "MOSS/LanPaint"

    def prepare(self, latent, mask, dilation=0):
        import torch.nn.functional as F

        samples = latent["samples"]
        if samples.ndim != 4 or samples.shape[1] != 128:
            raise ValueError("MOSS requires Flux2 latents [B,128,H/16,W/16].")
        if mask.ndim != 3 or tuple(mask.shape) != (samples.shape[0], samples.shape[2] * 16, samples.shape[3] * 16):
            raise ValueError("Mask dimensions must match the source encoded by the MOSS VAE.")
        if not torch.isfinite(mask).all():
            raise ValueError("Mask must contain finite values.")
        edit = (mask > 0.5).float().unsqueeze(1)
        if dilation:
            edit = F.max_pool2d(edit, 2 * dilation + 1, stride=1, padding=dilation)
        result = latent.copy()
        result["noise_mask"] = F.max_pool2d(edit, 16, stride=16)
        return (result,)


class MossLanPaintSampler:
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0, "max": 100, "step": 0.1}),
                "inner_steps": ("INT", {"default": 3, "min": 0, "max": 100}),
                "step_size": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 1, "step": 0.01}),
                "lamb": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 50, "step": 0.1}),
                "prompt_mode": (["Image First", "Prompt First"],),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "MOSS/LanPaint"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, inner_steps, step_size, lamb, prompt_mode):
        from .nodes import LanPaint_KSamplerAdvanced

        measured = model.clone()
        trace = {"denoiser_evaluations": 0, "transformer_forwards": 0, "branch_evaluations": 0, "sigmas": []}
        previous_wrapper = measured.model_options.get("model_function_wrapper")

        def forward(apply_model, args):
            trace["transformer_forwards"] += 1
            trace["branch_evaluations"] += len(args["cond_or_uncond"])
            if previous_wrapper is not None:
                return previous_wrapper(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        def record(args):
            trace["denoiser_evaluations"] += 1
            trace["sigmas"].append(float(args["sigma"][0]))
            return args["conds_out"]

        measured.set_model_unet_function_wrapper(forward)
        measured.set_model_sampler_pre_cfg_function(record)
        device = measured.load_device
        cuda = device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        # Fork the RNG because upstream's inner dynamics use the global generator.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            (result,) = LanPaint_KSamplerAdvanced().sample(
                measured,
                "enable",
                seed,
                steps,
                cfg,
                "euler",
                "simple",
                positive,
                negative,
                latent_image,
                0,
                steps,
                "disable",
                LanPaint_NumSteps=inner_steps,
                LanPaint_Lambda=lamb,
                LanPaint_StepSize=step_size,
                LanPaint_PromptMode=prompt_mode,
            )
        if cuda:
            torch.cuda.synchronize(device)
            trace.update(
                gpu=torch.cuda.get_device_name(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        trace["sampling_seconds"] = time.perf_counter() - started
        if not torch.isfinite(result["samples"]).all():
            raise RuntimeError("MOSS sampler returned nonfinite latents.")
        result["moss_lanpaint"] = trace
        return (result,)


class MossLanPaintStatistics:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    RETURN_TYPES = ()
    FUNCTION = "report"
    OUTPUT_NODE = True
    CATEGORY = "MOSS/LanPaint"

    def report(self, latent):
        import json

        return {"ui": {"text": [json.dumps(latent["moss_lanpaint"])]}}


class MossLanPaintComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "generated": ("IMAGE",),
                "mask": ("MASK",),
                "feather": ("INT", {"default": 0, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "composite"
    CATEGORY = "MOSS/LanPaint"

    def composite(self, source, generated, mask, feather=0):
        import numpy as np
        from .videomask import _edt_2d

        if source.shape != generated.shape or tuple(mask.shape) != tuple(source.shape[:3]):
            raise ValueError("Composite source, generated image and mask dimensions must match.")
        edit = mask > 0.5
        alpha = edit.to(source)
        if feather:
            for batch in range(len(mask)):
                # All-white masks retain ordinary full-image delivery, without source blending.
                if edit[batch].all():
                    continue
                pixels = edit[batch].cpu().numpy()
                distance = _edt_2d(~np.pad(pixels, 1))[1:-1, 1:-1]
                alpha[batch] = torch.as_tensor(np.clip((distance - 1) / feather, 0, 1), device=source.device, dtype=source.dtype)
        output = torch.lerp(source, generated.to(source), alpha.unsqueeze(-1))
        return (output,)


NODE_CLASS_MAPPINGS = {
    cls.__name__: cls
    for cls in (
        MossLanPaintPrepareImage,
        MossLanPaintPrepareMask,
        MossLanPaintSampler,
        MossLanPaintStatistics,
        MossLanPaintComposite,
    )
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MossLanPaintPrepareImage": "Prepare MOSS Source",
    "MossLanPaintPrepareMask": "Prepare MOSS Latent Mask",
    "MossLanPaintSampler": "MOSS LanPaint Sampler",
    "MossLanPaintComposite": "MOSS Composite",
    "MossLanPaintStatistics": "MOSS Sampling Statistics",
}
