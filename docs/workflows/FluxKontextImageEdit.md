---
description: Change an image you already have by describing the edit in words, using FLUX.1 Kontext.
when_to_use: Use it to change clothing, objects or lighting in a picture while keeping the person and composition. Use Inpainting when you want to limit the change to a painted area.
---

This workflow edits a picture from a written instruction. You load an image and describe the change. The model is FLUX.1 Kontext dev, which is built for edits like this. The template loads a photo of a woman and asks for a dark green leather jacket instead of the beige trench coat, an iced coffee in a clear glass, and a rainy evening, while keeping her identity, pose and composition the same.

## What it connects

1. A Load Image node supplies your picture.
1. Two [Load Component](../nodes/LoadComponent.md) nodes supply the text encoders, which read your prompt. One is `openai/clip-vit-large-patch14` and one is `google/t5-v1_1-xxl`. See [What are a text encoder and a tokenizer?](../guides/what-is-a-text-encoder.md).
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model with those text encoders.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) receives your picture on its `image` input, reads the instruction from a Text Input node named Prompt and runs 20 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **FluxKontextImageEdit** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If Model Manager says you need access, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Check that the two text encoder repos are in your Hugging Face cache. Download them in Model Manager if they are not.
1. Load your picture in the Load Image node.
1. Describe the edit in the Prompt node. Say what to change, and say what must stay the same.
1. Run the workflow.

## Defaults

- Model: FLUX.1 Kontext dev (`black-forest-labs/FLUX.1-Kontext-dev`), pipeline type `FluxKontextPipeline`.
- Size: 1024 x 1024.
- Steps: 20.
- Guidance: 3.5.
- `true_cfg_scale`: 1.0.
- Seed: 42.

## Tips

- The image input is optional. Disconnect it and the same workflow makes an image from the prompt alone.
- Add Noise is off on Generate Latent, so the seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- To limit the edit to part of the picture, use [FluxKontextInpainting](FluxKontextInpainting.md).
- To learn how the pieces fit, read [How pipelines work](../guides/how-pipelines-work.md).
