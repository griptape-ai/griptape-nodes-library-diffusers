---
title: Build Pipeline by model
description: Notes for the models that behave differently in Build Pipeline.
---

These are notes for models that differ from the general behavior described in [How pipelines work](how-pipelines-work.md). Find your model's heading below.

The pipeline types of MiniMax-H3, Flux3 and WAN Animate 2 manage their own memory and ignore quantization, layerwise casting and CPU offload. The WAN, WAN I2V, WAN Animate, WAN VACE and HunyuanVideo 1.5 pipeline types also ignore CPU offload.

## MiniMax-H3

MiniMax-H3 makes video together with a soundtrack. Pick the pipeline type for the job:

- `MiniMaxH3ModularPipeline` makes a video from a text prompt, or from keyframes (pictures that set what the video should look like at chosen points). The soundtrack is generated with the video.
- `MiniMaxH3Ref2VAModularPipeline` makes a video from reference material: up to 9 images and 3 video clips. It loads its own transformer, which is stored in the same Hugging Face repo as the base model. See [Hugging Face and Diffusers](hugging-face-and-diffusers.md).

This model manages its own memory, so it ignores the [Memory settings](how-pipelines-work.md#memory-settings). The transformer is held in your computer's main memory. It is moved onto the graphics card (GPU) when it is needed, and moved back to main memory when another part needs the room. The text encoder stays on disk and is loaded each time it is needed.

- **Hardware:** the model's authors recommend one graphics card with 80 GB of memory. The transformer and the text encoder are each about 62 GB, so the computer also needs a large amount of main memory.
- **Downloads:** do not delete the downloaded files from the cache after the first run. The text encoder is read from disk each time.
- **LoRAs:** not supported. See [Load LoRA](../nodes/LoadLora.md).

## Flux3

Flux3 makes images. Pick the pipeline type for the speed you want:

- `Flux3Pipeline` is the base model.
- `Flux3DistilledPipeline` is a distilled version, which means it was retrained to need fewer steps (rounds of clean-up). It needs 16.

Like MiniMax-H3, Flux3 manages its own memory and ignores the [Memory settings](how-pipelines-work.md#memory-settings). Each part of the model moves onto the graphics card only while it works.

`prompt_upsampler_model` is for Flux3 only. A prompt upsampler rewrites your prompt in more detail before the image is made, so with one selected the prompt is expanded before generation. The option is `black-forest-labs/flux-3-image-klein-pup`. Choose `Disabled` to use your prompt as written.

The model list first shows the standard Flux3 model. Once the compact versions are downloaded, two more rows appear: `fp8r` and `nvfp4`. Each is a smaller version of the transformer that uses less graphics card memory. Only the transformer changes.

`fp8r` runs on any device. It is faster only on NVIDIA cards with compute capability 8.9 or higher (RTX 40-series and newer, and data-center cards such as L40S and H100). On a Mac or an older card it saves space but not time, because the numbers are converted back to a larger format as they are used. If a run on a new RTX card stops with an fp8 error, choose the standard row or `nvfp4`.

`nvfp4` is faster only on NVIDIA Blackwell cards (compute capability 10 or higher, which includes the RTX 50-series) with the optional FlashInfer package installed. Elsewhere it still runs by converting the weights back to a larger format, which saves memory but not time.

## WAN Animate 2

WAN Animate 2 (`WanAnimate2Pipeline`) animates a character image with the motion from a driving video, the clip that supplies the movement. `model` offers the standard and the distilled checkpoint. With the distilled checkpoint, Generate Latent defaults to 10 steps and `guidance_scale` 1.0, which controls how closely the model follows the prompt.

It manages its own memory and ignores the [Memory settings](how-pipelines-work.md#memory-settings). The model is stored in your computer's main memory, and each part moves onto the graphics card only while it works.

The first Generate Latent run compiles the transformer, a one-time setup that makes it slower than later runs.

## LTX2

LTX2 makes video. The `LTX2Pipeline` type lists the LTX-2.3, LTX-2.3 Distilled and LTX-2 checkpoints in `model`. `LTX-2.5 Distilled` and `LTX-2.5 Full (SFT)` are separate pipeline types in the Build Pipeline list, both built from the one `Lightricks/LTX-2.5-Diffusers` repo. They differ only in the transformer: distilled, or full. The VAE used to decode the video is the same for all of them.

For the settings every model shares, see [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) and [How pipelines work](how-pipelines-work.md).
