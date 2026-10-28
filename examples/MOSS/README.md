# MOSS Image inpainting

This example uses ComfyUI's native MOSS Image model, Flux2 VAE, and negative-branch layer-22 skip guidance with the existing LanPaint sampler. No model training or changes to the LanPaint numerical kernel are required.

## Requirements and model files

Use a ComfyUI version with native `moss_image` CLIP loading and `SkipLayerGuidanceDiTSimple`. Development validation used ComfyUI v0.42.0, commit `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`, Python 3.12, and PyTorch 2.11.0. Older ComfyUI releases without native MOSS support will not load this workflow.

Download the following from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) into your ComfyUI model directories (the validated model revision is `3284a1dbb351e3396d70f39f2a71cbf6ed2ef1bb`):

| ComfyUI directory | File |
| --- | --- |
| `models/diffusion_models/` | `moss_image_sft_bf16.safetensors` |
| `models/text_encoders/` | `qwen3.5_9b_bf16.safetensors` |
| `models/vae/` | `flux2-vae_small.safetensors` |

Use BF16 for the diffusion model and text encoder, and FP32 for the VAE. The reference launch options were `--bf16-unet --bf16-text-enc --fp32-vae`. Models and test outputs are not bundled in this repository.

## Run the workflow

1. Load `moss_lanpaint.json` in ComfyUI. `moss_lanpaint_api.json` is the corresponding API prompt, for clients posting to `/prompt`.
2. Select an RGB source in the source loader and a matching grayscale mask in the mask loader. Width and height must be multiples of 16. White means edit; black means preserve. A red mask overlay in the editor is only a preview. For transparent example PNGs, select alpha and invert it to obtain the white-edit mask used here.
3. Set the positive prompt to describe the desired scene. The negative text is empty; `SkipLayerGuidanceDiTSimple` skips single block 22 only on that negative branch.
4. Select `source_fill`: `neutral` replaces masked source pixels with gray before encoding, while `source` encodes the original image for source-conditioned editing. The supplied workflow defaults to `neutral`; `source` helped clothing and recoloring tasks in development tests.
5. Run with the supplied starting settings: seed 42, Euler/simple, 50 outer steps, CFG 3, Image First, at most 3 inner steps, step size 0.1, lambda 5. These are practical starting settings rather than a quality optimum for every image.

The helper pools the binary pixel mask conservatively over each 16-by-16 VAE cell, so a narrow edit is not lost during downsampling. Optional dilation is measured in source pixels. The latent must be native MOSS `[B,128,H/16,W/16]`. Keep ComfyUI's native MOSS FLOW sampling defaults; do not add an SD3 model-sampling override.

Outputs include the raw decode, composite, and original-source VAE roundtrip. The composite copies all outside-mask source pixels exactly; this is a delivery operation, not a claim about the raw model's preservation. Optional feathering blends inward only. A full-white mask uses the generated image without source blending. Empty masks should be bypassed by API clients if they want to avoid unnecessary sampling.

The statistics node reports denoiser evaluations, transformer forwards, branch evaluations, sampling time, and PyTorch peak allocation. The sampler clones the model and forks the sampling RNG for repeatable local inner dynamics. PyTorch allocation counters exclude some ComfyUI dynamic model allocations and are not total device memory measurements.

## Compatibility fixes

- Retain negative conditioning whenever either guidance scale needs it. In Prompt First mode the second scale is -0.5, even when ordinary CFG is 1.
- Dispatch through `sampler_calc_cond_batch_function` when present. This preserves native skip-layer guidance and other ComfyUI conditioning hooks. Existing pre-CFG hooks still run afterward.

## Validation and limitations

The original integration was exercised on one A100 80 GB with 13 real-image functional cases: empty/full/thin masks, full-mask source independence, two photographic object replacements, non-square inputs, Prompt First at CFG 1, repeated seed, 1024-square outpainting, and source-conditioned feathering. Repeated-seed raw images matched, full-mask results were source-independent, and delivered composites preserved outside-mask pixels. These are historical development results, not a fresh GPU execution of this packaged branch.

Representative 50-step sampling times were 33.74 seconds for 768-by-512 and 91.07 seconds for 1024-square outpainting. These exclude text encoding, VAE work, and model loading. A sampled device-memory peak was approximately 28 GB; sampling can miss brief peaks. Different runtime versions and hardware can change results.

Hard masks can leave visible seams, and partially masked objects or complex clothing edits can retain artifacts. Use a generous edit region and try source conditioning or inward feathering as appropriate. This example establishes functional MOSS support, not a comparative image-quality benchmark.

CPU regression tests cover thin-mask retention, malformed input rejection, source conditioning, exact outside-mask composition, full-mask behavior, conditioning callback dispatch, pre-CFG hooks, and both guidance scales. Run `pytest tests/test_moss.py` from the repository root.

The packaged contribution passed the full CPU suite (139 passed, 1 skipped) and `ruff check .`. A separate check against the real ComfyUI runtime confirmed that skip-layer guidance patches single block 22 on the negative branch only, including Prompt First at CFG 1. No new GPU allocation was used to package this branch.
