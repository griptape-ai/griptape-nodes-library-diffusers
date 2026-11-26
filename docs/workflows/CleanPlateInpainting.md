---
description: Paint out part of a reference frame to make a clean plate for compositing, such as match move cleanup in Nuke.
when_to_use: Use it when you need a clean patch to track or project over live action footage. Use Inpainting for general object replacement.
---

This workflow makes a clean plate. A clean plate is a frame with an unwanted element removed, which compositors use to cover that element in a shot. The workflow inpaints a masked area of one reference frame. Inpainting repaints only the masked area. The template uses FLUX.1 Fill dev, a model built for filling masked areas. The sample frame is a mural on a wall at Venice Beach, and the mask marks the area to clean up.

The template notes describe the use. You pick the sharpest, least blocked frame of your target in a compositing app such as Nuke and mask the area to clean. You bring the result back into Nuke, where it can be match moved to the plate with 2D tracking or projected in 3D. The example in the template is a 3D projection cleanup.

## What it connects

1. Reference Frame is a Load Image node that supplies the frame.
1. Reference Frame Mask is a Load Image node that supplies the mask.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) encodes the frame and the mask into a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) repaints the masked area. The prompt is empty in the template. It runs 20 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.
1. Compare Images shows your original frame and the result together.

## Steps

1. Open the **CleanPlateInpainting** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If Model Manager says you need access, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Load your frame in Reference Frame and your mask in Reference Frame Mask.
1. Optional: type a prompt in Generate Latent that describes what the cleaned patch should look like. The notes say FLUX.1 Fill may not need one, but other pipelines do. A description makes the desired result more likely.
1. Run the workflow and check the result in Compare Images.

## The mask

White areas are repainted and black areas are kept. Paint the mask at the same aspect ratio as the frame. See [Inpainting](Inpainting.md#the-mask) for how gray and a different mask size are handled.

## Defaults

- Model: FLUX.1 Fill dev (`black-forest-labs/FLUX.1-Fill-dev`), pipeline type `FluxFillPipeline`.
- Strength: 0.95. This is set on Encode Masked Media. See [strength](../nodes/VaeMaskEncodeNode.md#strength).
- Steps: 20.
- Guidance: 30.
- Prompt: empty.
- Seed: 42.

## Tips

- Other pipelines work too. The template notes list the ones Encode Masked Media supports: `FluxFillPipeline`, `StableDiffusionXLPipeline`, `FluxPipeline`, FluxKontext, `Flux2KleinPipeline`, `QwenImagePipeline`, `QwenImageEditPipeline`, `StableDiffusion3Pipeline` and `ZImagePipeline`. To change the pipeline, set `provider`, then `pipeline_type`, then `model` on Build Pipeline. Changing `provider` resets `pipeline_type` and the model settings. Memory settings and LoRAs keep their values.
- Encode Masked Media prepares the noise itself, so Generate Latent hides its own Add Noise switch. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Make the mask a little larger than the object you are removing so the edges blend in.
