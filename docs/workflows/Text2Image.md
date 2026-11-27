---
description: Type a description and get an image back. This is the simplest workflow in the library.
when_to_use: Start here to make your first image, or to see how the diffusion nodes connect before you build your own.
---

This workflow turns a text description into an image. The description is called a prompt. It uses Z-Image Turbo, a fast image model. The workflow opens with the prompt, the size and the seed already filled in, so you can run it as soon as the model is downloaded.

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in. The seed decides which noise you get. The same seed and prompt give the same image.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and runs 20 steps. The model removes the noise over several steps, 20 here.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result back into an image you can see.

## Steps

1. Open the **Text2Image** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Type your prompt in the Text Input node.
1. Run the workflow.

## Defaults

- Model: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- Size: 1024 x 1024.
- Steps: 20.
- Seed: 42.

## Tips

- Change the seed on Create Noise Latent to get a different image from the same prompt. Add Noise is off on Generate Latent in this template, because the starting noise already exists. The seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Change only the prompt and run again. The loaded model is reused, so the second run is much faster.
- To try another model, set these on Build Pipeline in this order. Changing `provider` resets `pipeline_type` and the model settings. Memory settings and LoRAs keep their values.
  1. `provider` is the model family. This template uses `Z-Image`.
  1. `pipeline_type` is the variant within the family. This template uses `ZImagePipeline`.
  1. `model` is the model itself.
