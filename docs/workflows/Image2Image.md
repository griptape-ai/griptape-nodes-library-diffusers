---
description: Start from your own picture and rework it to match a prompt. Good for restyling and for variations.
when_to_use: Use it when you have an image you want to change while keeping its overall layout. Use Text2Image when you are starting from nothing.
---

This workflow loads an image, adds noise to it, and lets the model repaint it to match your prompt. It uses Z-Image Turbo. A Compare Images node shows the original and the result side by side.

## What it connects

1. A Load Image node supplies your picture.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Encode Media](../nodes/VaeEncodeNode.md) converts the picture into the form the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and repaints the image, starting from step 5 of 20.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result back into an image.

## Steps

1. Open the template. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your image in Load Image.
1. Describe the result you want in the Text Input node.
1. Run the workflow and check the result in Compare Images.

## Defaults

- Model: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- Steps: 20, starting from step 5.
- Seed: 42.

## Tips

- The start step controls how much of your picture stays. A higher start step keeps more of the original. A lower one gives the prompt more room to change it. The start step has to be lower than the number of steps.
- Leave Add Noise on in Generate Latent, as the template sets it. With it off, no noise is added to your picture and the seed has no effect.
- Encode Media does not add noise. Generate Latent adds it when Add Noise is on.
