---
description: Replace a masked area of an image and guide the new content with a reference image, using Flux2 Klein 4B.
when_to_use: Use it when you want to swap one object for another and have a picture that shows what the new one should look like. Use Inpainting for a prompt-only version.
---

This workflow repaints only the masked area of an image, guided by a prompt and a reference image. This is called inpainting. It uses Flux2 Klein 4B. The template swaps a ball using the prompt "Replace the ball" and a reference picture of what to put there.

## What it connects

1. Two Load Image nodes, named input_image and input_mask, supply the picture and the mask.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) encodes the picture and the mask into a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. A third Load Image node supplies the reference image.
1. [Reference Media](../nodes/MediaGenConditioningNode.md) packages the reference image as guidance. It is set to `image` mode with one image.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and receives the reference on `reference_images`. It repaints the masked area.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **Flux2KleinGuidedInpainting** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If Model Manager says you need access, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Load your picture in input_image and your mask in input_mask.
1. Load the reference image in the third Load Image node.
1. Describe the change in the Text Input node.
1. Run the workflow.

## The mask

White areas are repainted and black areas are kept. Paint the mask at the same aspect ratio as the picture. See [Inpainting](Inpainting.md#the-mask) for how gray and a different mask size are handled.

## Defaults

- Model: Flux2 Klein 4B (`black-forest-labs/FLUX.2-klein-4B`), pipeline type `Flux2KleinPipeline`.
- Memory strategy: Manual.
- Strength: 1.0 in the template. The Encode Masked Media default is 0.95. This is set on Encode Masked Media. See [strength](../nodes/VaeMaskEncodeNode.md#strength).
- Guidance: 4.0.
- Seed: 42.

## Tips

- To use more than one reference image, raise `num_images` on Reference Media. Flux2 Klein takes 1 to 8. Set it before you connect images, because changing it removes the connections. See [num_images](../nodes/MediaGenConditioningNode.md#num_images).
- The template notes say the 4B model is faster and the 9B model gives higher quality. Both work for inpainting and reference images. To try 9B, change `model` on Build Pipeline.
- Encode Masked Media prepares the noise itself, so Generate Latent hides its own Add Noise switch. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Make the mask a little larger than the object you are replacing so the edges blend in.
