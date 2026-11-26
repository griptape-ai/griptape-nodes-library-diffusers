---
description: Give it a character image, a pose video and a face video, and it makes a video of that character moving and acting like the two videos.
when_to_use: Use it when you have separate videos for the body movement and the face. WanAnimate2 needs only one driving video. WanReplace puts the character into an existing scene.
---

This workflow animates the character in a picture. The pose video supplies the body movement. The face video supplies the facial expression. The model makes a new video of your character with that movement and expression. It uses Wan 2.2 Animate 14B in its `animate` mode.

The template comes with a sample character image, a sample pose video and a sample face video. The prompt is "A dancing lady".

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model. The `pipeline_type` is `WanAnimatePipeline`.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the video size, the length and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. A Load Image node supplies the character image.
1. Two Load Video nodes supply the pose video and the face video.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) takes the noise, the image and the two videos. Its `mode` is set to `animate`. The prompt is typed on this node.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video. Set the playback speed in its `fps` parameter.

## Steps

1. Open the **WanAnimate** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If the download asks for access on Hugging Face, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Load your character image in the Load Image node.
1. Load your pose video in the first Load Video node and your face video in the second.
1. Write a short prompt on Generate Latent.
1. Run the workflow.

## Defaults

- Model: Wan 2.2 Animate 14B (`Wan-AI/Wan2.2-Animate-14B-Diffusers`).
- Mode: `animate`.
- Size: 576 x 1024, set on Create Noise Latent.
- Length: 121 frames, set as `num_frames` on Create Noise Latent. About 4.8 seconds at 25 frames per second.
- Frame rate: 25 frames per second, set as `fps` on Decode Media.
- Steps: 20.
- Guidance: 1.0.
- Seed: 42.

## Tips

- The template has a note that says the character in the image is animated using the pose and face reference videos.
- Add Noise is off on Generate Latent, because the starting latent is already noise. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- To place the character into an existing video instead, use [WanReplace](WanReplace.md). To use one driving video instead of two, use [WanAnimate2](WanAnimate2.md).
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
