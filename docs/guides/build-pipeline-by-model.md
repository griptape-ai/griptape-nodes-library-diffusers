---
title: Build Pipeline by model
description: Notes for the models that behave differently in Build Pipeline.
---

These are notes for models that differ from the general behavior described in [How pipelines work](how-pipelines-work.md). Find your model's heading below.

## MiniMax-H3

MiniMax-H3 makes video together with a soundtrack. Pick the pipeline type for the job:

- `MiniMaxH3ModularPipeline` makes a video from a text prompt, or from keyframes (pictures that set what the video should look like at chosen points). The soundtrack is generated with the video.
- `MiniMaxH3Ref2VAModularPipeline` makes a video from reference material: up to 9 images and 3 video clips. It loads its own transformer, which is stored in the same Hugging Face repo as the base model. A repo is a named folder of files on Hugging Face.

This model manages its own memory, so it ignores the [Memory settings](how-pipelines-work.md#memory-settings). The transformer is held in your computer's main memory and moves onto the graphics card only while it works. The text encoder stays on disk and is loaded each time it is needed.

- **Hardware:** the model card recommends one graphics card with 80 GB of memory, and a large amount of main memory as well. The transformer and the text encoder are each about 62 GB.
- **Downloads:** the text encoder stays on disk, so keep the model files downloaded.
- **LoRAs:** not supported. A LoRA is a small add-on file that gives a model a style or subject.

## Flux3

Flux3 makes images. Pick the pipeline type for the speed you want:

- `Flux3Pipeline` is the base model.
- `Flux3DistilledPipeline` is a distilled version, trained to run in 16 steps.

Like MiniMax-H3, Flux3 manages its own memory and ignores the [Memory settings](how-pipelines-work.md#memory-settings). Each part of the model moves onto the graphics card only while it works.

`prompt_upsampler_model` is for Flux3 only. A prompt upsampler rewrites your prompt in more detail before the image is made. The option is `black-forest-labs/flux-3-image-klein-pup`. Choose `Disabled` to use your prompt as written.

The model list first shows the standard Flux3 model. Once the compact versions are downloaded, two more rows appear: `fp8r` and `nvfp4`. Each uses a smaller transformer that needs less graphics-card memory. Only the transformer changes.

`fp8r` runs on any device. On an NVIDIA card with CUDA compute capability 8.9 or higher (RTX 40-series and newer, including RTX 50-series) it uses fast fp8 math. Elsewhere, including a Mac, the weights are converted to a larger number format as they are used, so it saves disk space but gains no speed. The run stops with a message only on a card of 8.9 or higher whose PyTorch install cannot do the fp8 matrix multiply. Then pick the default row or `nvfp4`.

## WAN Animate 2

WAN Animate 2 (`WanAnimate2Pipeline`) animates a character image with the motion from a driving video, the clip that supplies the movement. `model` offers the standard and the distilled checkpoint. A distilled checkpoint is a version trained to run in fewer steps. With it, Generate Latent defaults to 10 steps and `guidance_scale` 1.0.

It manages its own memory and ignores the [Memory settings](how-pipelines-work.md#memory-settings). The model is stored in your computer's main memory, and each part moves onto the graphics card only while it works.

The first Generate run compiles the transformer, a one-time setup that makes it slower than later runs.

## LTX2

LTX2 makes video. The `LTX2Pipeline` type lists the LTX-2.3, LTX-2.3 Distilled and LTX-2 checkpoints in `model`. The `LTX-2.5 Distilled` and `LTX-2.5 Full (SFT)` types are both built from the `Lightricks/LTX-2.5-Diffusers` repo. They differ only in the transformer: distilled, or full. The VAE used to decode the video is the same for all of them.

For the settings every model shares, see [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) and [How pipelines work](how-pipelines-work.md).
