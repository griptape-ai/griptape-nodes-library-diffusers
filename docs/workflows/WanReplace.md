---
description: Give it a character image, a background video, a mask video, a pose video and a face video, and it puts your character into the background video.
when_to_use: Use it when you want to swap a person in an existing shot for a different character. WanAnimate makes a new video from scratch with no background video.
---

This workflow replaces a person in a video with a character from a picture. The model puts the character into your background video. The mask video marks where the character appears in each frame. The pose video sets how the body moves and the face video sets the expression. It uses Wan 2.2 Animate 14B in its `replace` mode.

The template comes with sample files for all five inputs. The prompt is "A dancing lady".

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model. The `pipeline_type` is `WanAnimatePipeline`.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the video size, the length and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. A Load Image node supplies the character image.
1. Four Load Video nodes supply the pose video, the face video, the background video and the mask video.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) takes the noise, the image and the four videos. Its `mode` is set to `replace`. The prompt is typed on this node.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video. Set the playback speed in its `fps` parameter.

## Steps

1. Open the **WanReplace** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If the download asks for access on Hugging Face, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Load your character image in the Load Image node.
1. Load the four videos. Connect the pose video to `pose_video`, the face video to `face_video`, the background video to `background_video` and the mask video to `mask_video`.
1. Write a short prompt on Generate Latent.
1. Run the workflow.

## Defaults

- Model: Wan 2.2 Animate 14B (`Wan-AI/Wan2.2-Animate-14B-Diffusers`).
- Mode: `replace`.
- Size: 576 x 1024, set on Create Noise Latent.
- Length: 121 frames, set as `num_frames` on Create Noise Latent. About 4.8 seconds at 25 frames per second.
- Frame rate: 25 frames per second, set as `fps` on Decode Media.
- Steps: 20.
- Guidance: 1.0.
- Seed: 42.

## Tips

- The template has a note that explains the mode. In `replace` mode the model composites the character from the reference image into the background video. The mask video controls where the character appears in each frame. The pose video controls what the character does.
- Add Noise is off on Generate Latent, because the starting latent is already noise. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- If you do not have a background video, use [WanAnimate](WanAnimate.md).
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
