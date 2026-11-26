---
description: Give it a starting image and an ending image, and it creates the video that moves between them.
when_to_use: Use it when you know how a shot should begin and end and want the model to fill in the motion. Use Text2Image or Image2Image for stills.
---

This workflow makes a video that starts on your first image and ends on your last image, guided by a prompt. It uses Wan 2.2 Image-to-Video. The template comes with two sample images and the prompt "A cup of green tea on a saucer, the camera pans over the top of it."

## What it connects

1. Two Load Image nodes supply the first and last frames. Two Resize Image nodes fit them to the video size.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Reference Media](../nodes/MediaGenConditioningNode.md) holds the two frames, set to First + Last.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the video size and length. Two number nodes, Width and Height, feed it and the Resize Image nodes, so the frames always match the video.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and makes the video.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video.

## Steps

1. Open the template. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your first image in the first Load Image node and your last image in the second.
1. Describe the motion between them in the Text Input node.
1. Run the workflow.

## Defaults

- Model: Wan 2.2 Image-to-Video A14B (`Wan-AI/Wan2.2-I2V-A14B-Diffusers`).
- Size: 1280 x 720, set with the Width and Height nodes.
- Length: 81 frames, played at 25 frames per second.
- Steps: 20.
- Guidance: 5.0.
- Seed: 42.

## Tips

- Pick two images that could belong to the same shot. The model fills in the motion between them.
- To change the video size, change the Width and Height nodes. They drive both Create Noise Latent and the Resize Image nodes, so the frames keep matching the video.
- To check how much memory a run needs before you start, use [Estimate Pipeline Memory](../nodes/PipelineMemoryEstimateNode.md).
