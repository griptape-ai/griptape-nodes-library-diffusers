---
description: Type a description and get an image back. This is the simplest workflow in the library.
when_to_use: Start here to make your first image, or to see how the diffusion nodes connect before you build your own.
---

This workflow turns a text description into an image. The description is called a prompt. It uses Z-Image Turbo, a fast image model. The workflow opens with the prompt, the size and the seed already filled in, so you can run it as soon as the model is downloaded. The seed is a number that fixes the random starting point, so the same seed and prompt give the same image.

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes the random noise the model draws from, as a latent, which is the compressed form the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and runs the 20 diffusion steps. Each step cleans up the noise a little more.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result back into an image you can see.

## Steps

1. Open the template. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Type your prompt in the Text Input node.
1. Run the workflow.

## Defaults

- Model: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- Size: 1024 x 1024.
- Steps: 20 (rounds of clean-up).
- Seed: 42.

## Tips

- Change the seed on Create Noise Latent to get a different image from the same prompt. Add Noise is off on Generate Latent, so the seed on that node has no effect here. The starting noise comes from Create Noise Latent.
- Change only the prompt and run again. The loaded model is reused, so the second run is much faster.
- To try another model, set the provider on Build Pipeline first, because changing it resets the other settings. Then pick the pipeline type, which is the model family, and the model.
