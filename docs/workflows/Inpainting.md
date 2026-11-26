---
description: Paint a mask over part of an image and describe what should appear there. The rest is kept as close to the original as the model allows.
when_to_use: Use it to replace or fix one area of an image, such as removing an object or adding one. Use Image2Image to rework the whole picture.
---

This workflow repaints only the masked area of an image to match a text description, called a prompt. This is called inpainting. The mask works like a matte: it marks where the model may change pixels. The rest of the picture is kept as close to the original as the model allows. Pixels can shift very slightly because the whole image passes through the [VAE](../guides/what-is-a-vae.md), the part of the model that converts between pixels and latents. It uses Flux2 Klein 4B. The template comes with a sample image, a sample mask and the prompt "Yellow cat sitting on the grass".

## What it connects

1. Two Load Image nodes supply the picture and the mask.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) encodes the picture and the mask into a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and repaints the masked area.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image.

## Steps

1. Open the template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your picture in the first Load Image node and your mask in the second.
1. Describe what should appear in the masked area in the Text Input node.
1. Run the workflow.

## The mask

White areas are repainted and black areas are kept as close to the original as the model allows. Gray is treated as white from 50% up, and as black below that. The mask is resized to the picture's size, so it can be a different size, but it is stretched, not cropped. Paint it at the same aspect ratio as the picture so it lines up with the area you meant.

## Defaults

- Model: Flux2 Klein 4B (`black-forest-labs/FLUX.2-klein-4B`).
- Strength: 1.0. This is set on Encode Masked Media. At 1.0 the masked area is fully replaced. A lower value keeps more of the original. See [strength](../nodes/VaeMaskEncodeNode.md#strength).
- Guidance: 4.0. A higher value follows the prompt more closely.
- Seed: 42.

## Tips

- Describe what belongs in the masked area, not the whole picture.
- Encode Masked Media prepares the noise itself, so Generate Latent hides its own Add Noise switch. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Masking behaves differently when you use Set Latent Mask instead. See [Masked denoising](../guides/generate-latent-by-model.md#masked-denoising).
- Make the mask a little larger than the object you are replacing so the edges blend in.
