---
description: Give it a character image and a driving video, and it makes a video of that character moving the way the driving video moves.
when_to_use: Use it when you want one character to copy the motion of a clip, such as a dance or a gesture. You only need a driving video. WanAnimate needs a pose video and a face video instead.
---

This workflow animates a character. You provide a picture of the character and a video that shows the movement you want. The model makes a new video of your character performing that movement. The video is called the driving video. It uses Wan 2.2 Animate 2 14B Distilled. See [WAN Animate 2](../guides/build-pipeline-by-model.md#wan-animate-2) for how this model handles memory.

The template comes with a sample reference image and a sample driving video. The prompt describes the character's clothes and hair, and the background.

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model. The `pipeline_type` is `WanAnimate2Pipeline`.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the video size, the length and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. A Load Image node supplies the character image.
1. A Load Video node supplies the driving video.
1. A Text Input node holds the prompt.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) takes the noise, the image, the driving video and the prompt, and makes the video. See [WAN Animate 2](../guides/generate-latent-by-model.md#wan-animate-2) for the settings that are specific to this model.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video. Set the playback speed in its `fps` parameter.

## Steps

1. Open the **WanAnimate2** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If the download asks for access on Hugging Face, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Load your character image in the Load Image node.
1. Load your driving video in the Load Video node.
1. Describe the character and the background in the Text Input node. The template has a note that explains how: the reference image sets how the character looks, the driving video sets how it moves, and the prompt helps the model keep the character and the background steady.
1. Run the workflow.

## Defaults

- Model: Wan 2.2 Animate 2 14B Distilled (`Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`).
- Size: 432 x 768, set on Create Noise Latent.
- Length: 81 frames, set as `num_frames` on Create Noise Latent. About 3.4 seconds at 24 frames per second.
- Frame rate: 24 frames per second, set as `fps` on Decode Media.
- Steps: 10.
- Guidance: 1.0.
- Segment length: 81 frames, with 1 overlapping frame.
- Seed: 42.

## Tips

- The image and every frame of the driving video are stretched to the size on Create Noise Latent. Pick a size with the same shape as your image.
- The driving video needs at least `num_frames` frames at 24 frames per second. Extra frames at the end are ignored. If your video was recorded at another rate, set `driving_video_fps` on Generate Latent to that rate.
- Clips longer than `segment_frame_length` are made in overlapping segments. Both `segment_frame_length` and `prev_segment_conditioning_frames` must be a multiple of 4 plus 1: 1, 5, 9, 13 and so on.
- The first run is slower than later runs, because the model sets itself up once.
- Add Noise is off on Generate Latent, because the starting latent is already noise. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
