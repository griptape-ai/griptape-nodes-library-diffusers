---
description: Replace a masked area of an image with something new, with an optional reference image to guide the result, using FLUX.1 Kontext.
when_to_use: Use it to swap an object in one area while keeping the rest, and when a reference picture helps show what you want. Use Inpainting for a prompt-only version.
---

This workflow repaints only the masked area of an image, guided by a prompt and a reference image. This is called inpainting. The model is FLUX.1 Kontext dev. The template swaps a book for a plate with a slice of cake. The prompt is "Replace the book with a plate containing a slice of cake." and the reference picture shows the plate of cake.

## What it connects

1. Two Load Image nodes supply the picture and the mask.
1. Two [Load Component](../nodes/LoadComponent.md) nodes supply the text encoders, which read your prompt. One is `openai/clip-vit-large-patch14` and one is `google/t5-v1_1-xxl`. See [What are a text encoder and a tokenizer?](../guides/what-is-a-text-encoder.md).
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model with those text encoders.
1. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) encodes the picture and the mask into a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. A third Load Image node supplies the reference image. It goes to the `image` input of Generate Latent.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the instruction from a Text Input node named Prompt and repaints the masked area. It runs 20 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **FluxKontextInpainting** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If Model Manager says you need access, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Check that the two text encoder repos are in your Hugging Face cache. Download them in Model Manager if they are not.
1. Load your picture in the first Load Image node and your mask in the second.
1. Optional: load a reference image in the third Load Image node. It is not required for inpainting.
1. Describe the change in the Prompt node.
1. Run the workflow.

## The mask

White areas are repainted and black areas are kept. Paint the mask at the same aspect ratio as the picture. See [Inpainting](Inpainting.md#the-mask) for how gray and a different mask size are handled.

## Defaults

- Model: FLUX.1 Kontext dev (`black-forest-labs/FLUX.1-Kontext-dev`), pipeline type `FluxKontextPipeline`.
- Strength: 0.95. This is set on Encode Masked Media. A lower value keeps more of the original. See [strength](../nodes/VaeMaskEncodeNode.md#strength).
- Steps: 20.
- Guidance: 3.5.
- `true_cfg_scale`: 1.0.
- Seed: 42.

## Tips

- Encode Masked Media prepares the noise itself, so Generate Latent hides its own Add Noise switch. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- The reference image influences the style and content of the new area. Disconnect it to inpaint from the prompt alone.
- Make the mask a little larger than the object you are replacing so the edges blend in.
