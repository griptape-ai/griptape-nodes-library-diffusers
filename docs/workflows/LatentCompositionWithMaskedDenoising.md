---
description: Paste one picture into another in latent form, then let the model blend the seam. It shows the result with and without masked denoising.
when_to_use: Use it when you want to place a subject into a scene and have the model make them look like they belong. Copy the masked branch to protect part of the picture while the rest is redone.
---

This workflow places a hamster into a garden. It does this in latent form, not in pixels. A [latent](../guides/what-is-a-latent.md) is the compressed form of an image that the model works in. The two latents are pasted together, and then the model cleans up the result.

It runs the cleanup two ways from the same pasted latent, so you can compare. The masked branch uses Set Latent Mask. The standard branch does not. The template is laid out in three groups named Latent Composition, Masked Denoising and Standard Denoising.

The sample files are a garden scene, a hamster picture and a depth image of the hamster that is used as the mask. The prompt describes a heroic hamster with an acorn-cap helmet. The model is FLUX.1 [schnell].

## What it connects

Latent Composition group:

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. Two [Encode Media](../nodes/VaeEncodeNode.md) nodes turn the garden image and the hamster image into latents.
1. [Composite Latents with Mask](../nodes/LatentCompositeMaskNode.md) pastes the hamster latent onto the garden latent. In the template this node is named Latents Composite Mask. The mask image is the hamster depth image, read from its red channel. White shows the source, black keeps the destination, and gray blends the two.

Masked Denoising group:

1. [Set Latent Mask](../nodes/SetLatentMaskNode.md) attaches the same mask to the composite.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and denoises the masked areas.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image. Compare Images shows it beside the original garden.

Standard Denoising group:

1. A second Generate Latent denoises the composite without Set Latent Mask. The whole composite can change.
1. A second Decode Media turns it into an image. A second Compare Images shows the masked result beside the standard result.

## Steps

1. Open the **LatentCompositionWithMaskedDenoising** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your background in the Load Image node that feeds the first Encode Media. Load your subject in the one that feeds the second.
1. Load a mask image in the Load Image node that feeds Composite Latents with Mask.
1. Describe the final picture in the Text Input node.
1. Run the workflow and compare the two results.

## Defaults

- Model: FLUX.1 [schnell] (`black-forest-labs/FLUX.1-schnell`).
- Composite: both latents are 512 x 512 in the placement editor, with offsets 0 and 0 and Resize Source off.
- Mask: red channel, not inverted, grow or shrink -3.0, blur 0.0. A negative grow or shrink value grows the mask.
- Add Noise: on, for both Generate Latent nodes.
- Start step: 5, for both.
- Steps: 20.
- Guidance: 3.5.
- Seed: 42.

## Tips

- Masked denoising needs Add Noise on. Without it, denoising starts from the clean source image. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added) and [Masked denoising](../guides/generate-latent-by-model.md#masked-denoising).
- A start step of 0 redoes the white areas fully. A higher start step keeps more of the original. This template uses 5.
- Masked denoising works on image models only. See the guide for the list.
- To place the subject somewhere else, change `x_offset` and `y_offset` on Composite Latents with Mask, or drag it in the placement editor. The editor shows the sizes after the node has run once.
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
