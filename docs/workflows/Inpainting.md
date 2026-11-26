---
description: Paint a mask over part of an image and describe what should appear there. Everything outside the mask stays the same.
when_to_use: Use it to replace or fix one area of an image, such as removing an object or adding one. Use Image2Image to rework the whole picture.
---

This workflow repaints only the masked area of an image from a prompt. It uses Flux2 Klein 4B. The template comes with a sample image, a sample mask and the prompt "Yellow cat sitting on the grass".

## What it connects

1. Two Load Image nodes supply the picture and the mask.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) combines the picture and the mask.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and repaints the masked area.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image.

## Steps

1. Open the template. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your picture in the first Load Image node and your mask in the second.
1. Describe what should appear in the masked area in the Text Input node.
1. Run the workflow.

## The mask

White areas are repainted and black areas are kept. Gray is treated as white from 50% up, and as black below that. The mask is resized to match the picture.

## Defaults

- Model: Flux2 Klein 4B (`black-forest-labs/FLUX.2-klein-4B`).
- Strength: 1.0, which fully replaces the masked area. The Encode Masked Media node starts at 0.95 when you add it yourself. Lower the strength to blend with the original.
- Guidance: 4.0.
- Seed: 42.

## Tips

- Describe what belongs in the masked area, not the whole picture.
- Generate Latent hides Add Noise while Encode Masked Media is connected. The inpainting step handles the noise itself, so there is nothing to set.
- Make the mask a little larger than the object you are replacing so the edges blend in.
