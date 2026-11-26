---
description: Start from your own picture and rework it to match a prompt. Good for restyling and for variations.
when_to_use: Use it when you have an image you want to change while keeping its overall layout. Use Text2Image when you are starting from nothing.
---

This workflow takes an image you already have and repaints it to match a text description, called a prompt. It keeps the overall layout. The model adds noise to your picture, then cleans it up toward your prompt. Noise is the random grain that diffusion removes to make a picture. It uses Z-Image Turbo. A Compare Images node shows the original and the result side by side.

## What it connects

1. A Load Image node supplies your picture.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Encode Media](../nodes/VaeEncodeNode.md) converts the picture into a latent, the compressed form the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and repaints the image, with `start_step` set to 5 of 20. A step is one round of clean-up. The first steps of a run do the big, coarse work of deciding the layout. Skipping the first 5 means the picture is only partly noised, so much of your original layout and color survives.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result back into an image.

## Steps

1. Open the template. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your image in Load Image.
1. Describe the result you want in the Text Input node.
1. Run the workflow and check the result in Compare Images.

## Defaults

- Model: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- Steps: 20, with `start_step` at 5.
- Seed: 42, the number that fixes the random noise.

## Tips

- `start_step` controls how much of your picture stays. A higher value keeps more of the original. A lower one gives the prompt more room to change it. It has to be lower than the number of steps.
- Leave Add Noise on in Generate Latent, as the template sets it. Encode Media does not add noise, so Generate Latent adds it. With Add Noise off, your picture is not noised, so the model has little to clean up and the seed has no effect. The template's own note says the output will be poor. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
