---
description: Give it a reference video and a prompt, and it makes a new video that follows the reference, using an LTX-2.3 IC-LoRA.
when_to_use: Use it when you have a control video and want a new video guided by it. Copy it to try other IC-LoRAs.
---

This workflow makes a video with LTX-2.3 and an IC-LoRA. A LoRA is a small add-on file that changes a model's behavior. An IC-LoRA uses a reference image or video to steer the new video. Here the LoRA is `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`, and the reference is a sample video called `canny_video.mp4`.

The prompt describes a translucent jellyfish drifting through a blue ocean.

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Load LoRA](../nodes/LoadLora.md) points at the IC-LoRA file.
1. [Apply Runtime LoRA](../nodes/LoraActivationPipelineNode.md) switches the LoRA on for the generation. In the template it is named LoRA Pipeline.
1. A Load Video node supplies the reference video.
1. [Reference Media](../nodes/MediaGenConditioningNode.md) holds the reference video at frame 0 with strength 1.0. In the template it is named Media Generation Conditioning. It goes to `reference_conditions` on Generate Latent.
1. [Create Empty Latent](../nodes/EmptyLatentNode.md) sets the size and length. It makes a blank [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. Two Text Input nodes hold the prompt and the negative prompt, which lists what to steer away from.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) makes the video. See [LTX2](../guides/generate-latent-by-model.md#ltx2) for the settings specific to this model.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video. In the template it is named VAE Decode Latents.

## Steps

1. Open the **LTX23-IC-LoRA** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If the download asks for access on Hugging Face, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Get an IC-LoRA file. The library does not ship one. The template notes suggest two repos: `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` and `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control`. Set the file path on Load LoRA to your file.
1. Load your reference video in the Load Video node.
1. Describe the new video in the prompt Text Input node.
1. Run the workflow.

## Defaults

- Model: LTX 2.3 Distilled (`dg845/LTX-2.3-Distilled-Diffusers`), with the `LTX2Pipeline` type.
- LoRA: `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`, weight 1.0.
- Size: 1536 x 1024, set on Create Empty Latent.
- Length: 121 frames, set as `num_frames` on Create Empty Latent. About 4.8 seconds at 25 frames per second.
- Frame rate: 25 frames per second, set as `fps` on Decode Media.
- Steps: 8.
- Guidance: 1.0.
- Seed: 42.

## Tips

- Read the LoRA's model card before you pick a reference. The template notes say the reference must line up with the output and must have the right resolution for the LoRA. The Union Control LoRA was trained with references at half the size, so the reference video should be half the width and height of the generated video.
- The distilled model follows a fixed schedule. The template note lists 8 steps for the first pass and 4 steps when `use_stage_2` is on. `use_stage_2` is off in this template.
- Add Noise is off on Generate Latent in this template. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added) for how the setting works with other inputs.
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md). For LTX2 pipeline types, see [Build Pipeline by model](../guides/build-pipeline-by-model.md#ltx2).
