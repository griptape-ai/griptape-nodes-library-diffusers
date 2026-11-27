---
description: Replace a masked area of an image while a ControlNet guide image controls the shape of what appears there.
when_to_use: Use it when inpainting needs to follow a specific outline, such as an edge map of the object you want. Use Inpainting when a prompt is enough.
---

This workflow repaints only the masked area of an image, and a guide image shapes what appears there. A ControlNet is an add-on model that reads a guide image and steers the result to match it. The template uses an edge map, called a canny image, and the prompt "A rusty vintage car". It uses FLUX.1 dev.

## What it connects

1. Two Load Image nodes supply the picture and the mask. A third, named control_image, supplies the guide image.
1. [Load ControlNet](../nodes/ControlNetNode.md) sets up the ControlNet with its model, guide image and strength. On the canvas it is named Configure ControlNet.
1. Two [Load Component](../nodes/LoadComponent.md) nodes supply the text encoders, which read your prompt. One is `google/t5-v1_1-xxl` and one is `openai/clip-vit-large-patch14`. See [What are a text encoder and a tokenizer?](../guides/what-is-a-text-encoder.md).
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model with those text encoders.
1. [Apply ControlNet](../nodes/ControlNetDiffusionPipelineBuilderNode.md) attaches the ControlNet to the pipeline. On the canvas it is named ControlNet Pipeline. Its `control_parameters` output also goes to Generate Latent.
1. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) encodes the picture and the mask into a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and repaints the masked area. It runs 40 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **ControlnetInpainting** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. FLUX.1 dev is a gated model, so you must accept its terms on Hugging Face first. See [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Download the ControlNet model shown in `controlnet_model` on Load ControlNet. Load ControlNet only lists models you have downloaded.
1. Check that the two text encoder repos are in your Hugging Face cache. Download them in Model Manager if they are not.
1. Load your picture and your mask in the first two Load Image nodes.
1. Load the guide image in control_image. It must match the `control_mode` on Load ControlNet. This template uses `canny`, so it needs an edge map.
1. Describe what should appear in the masked area in the Text Input node.
1. Run the workflow.

## The mask

White areas are repainted and black areas are kept. Paint the mask at the same aspect ratio as the picture. See [Inpainting](Inpainting.md#the-mask) for how gray and a different mask size are handled.

## Defaults

- Model: FLUX.1 dev (`black-forest-labs/FLUX.1-dev`), pipeline type `FluxPipeline`.
- ControlNet: `control_mode` canny, strength 0.5, active from 0 to 1 of the steps.
- Strength: 1.0 in the template. The Encode Masked Media default is 0.95. This is set on Encode Masked Media. See [strength](../nodes/VaeMaskEncodeNode.md#strength).
- Steps: 40.
- Guidance: 3.5.
- `true_cfg_scale`: 1.0.
- Seed: 42.

## Tips

- Raise `controlnet_conditioning_scale` on Load ControlNet to follow the guide image more closely. See [controlnet_conditioning_scale](../nodes/ControlNetNode.md#controlnet_conditioning_scale).
- Lower `control_guidance_end` to let the model fill in detail freely in the last steps. See [control_guidance_end](../nodes/ControlNetNode.md#control_guidance_end).
- The mask decides where the image changes. The guide image decides the structure inside that area.
- Encode Masked Media prepares the noise itself, so Generate Latent hides its own Add Noise switch. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
