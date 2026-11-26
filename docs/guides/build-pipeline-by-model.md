---
title: Build Pipeline by model
description: Model-specific and memory notes for the Build Pipeline node.
---

Read this when you need the memory settings in detail, the component override messages, or the notes for MiniMax-H3, Flux3, WAN Animate 2 or LTX2.

## Memory settings

These appear, together with `quantization_mode`, when `memory_optimization_strategy` is `Manual`. The `Manual` setting is the default. `Automatic` hides them and chooses settings from the GPU memory available.

### attention_slicing

Off by default. Works through attention in slices. This lowers peak memory with a small loss of speed.

### vae_slicing

Off by default. Decodes one slice at a time, which lowers peak memory for large images or batches.

### vae_tiling

Off by default. Decodes in tiles, which lowers memory for large images and long videos.

### transformer_layerwise_casting

Off by default. Stores the transformer in fp8 and converts each layer as it runs. This saves memory and can slow runs slightly. It has no effect on models that ship quantized weights.

### cpu_offload_strategy

`None` (default), `Model` or `Sequential`. Moves idle weights to the CPU to free GPU memory. `Model` moves whole parts and `Sequential` moves smaller pieces at a larger cost in speed.

## Component overrides

The Component Overrides group has one port for each part this pipeline type has, such as Transformer or VAE. MiniMax-H3, Flux3 and WAN Animate 2 have none. Connect a [Load Component](../nodes/LoadComponent.md) node to replace that part of the model with your own weights.

If a replacement does not fit, a message appears on the node. A VAE whose channel count does not match the model, or a text encoder whose output size does not match it, stops the run. A VAE with a different scaling, a scheduler with a different prediction type, and a tokenizer that does not pair with its text encoder show as warnings only.

## MiniMax-H3

The quantization, layerwise casting and CPU offload settings are not used. The model is loaded in bfloat16 with automatic CPU offload instead. It needs a single 80 GB accelerator plus plenty of system memory. The text encoder is reloaded from disk each time it is needed, so the model files must stay downloaded. LoRAs are not supported.

`MiniMaxH3ModularPipeline` does text-to-video and keyframe-to-video, with a soundtrack generated together with the video. `MiniMaxH3Ref2VAModularPipeline` does reference-to-video from up to 9 images and 3 video clips, and uses a separate transformer in the same repo.

## Flux3

`prompt_upsampler_model` is for Flux3 only. It is an optional prompt upsampler, `black-forest-labs/flux-3-image-klein-pup`. Choose `Disabled` to pass prompts through unchanged.

`Flux3Pipeline` is the base model. `Flux3DistilledPipeline` is the 16-step distilled model. The quantization, layerwise casting and CPU offload settings are not used. The model is loaded with automatic CPU offload instead.

The model list offers the full-precision repo and, when downloaded, `fp8r` and `nvfp4` rows with quantized transformers. A quantized row replaces only the transformer. `fp8r` stops the run with a message on a GPU or torch build that cannot run per-row scaled matrix multiplication. Pick the default row or `nvfp4` there.

## WAN Animate 2

`WanAnimate2Pipeline` animates a character image with the motion of a driving video. `model` offers the standard and the distilled checkpoint. With the distilled one, Generate Latent defaults to 10 steps and `guidance_scale` 1.0.

The quantization, layerwise casting and CPU offload settings are not used. The model is loaded in bfloat16 with automatic CPU offload instead. The transformer is compiled during the first Generate run, so that run takes noticeably longer.

## LTX2

The `LTX2Pipeline` type lists LTX-2.3, LTX-2.3 Distilled and LTX-2 checkpoints. `LTX-2.5 Distilled` and `LTX-2.5 Full (SFT)` both build an `LTX2Pipeline` from the `Lightricks/LTX-2.5-Diffusers` repo. They differ only in the transformer: distilled or full. Decoding always uses the plain VAE.

For the settings every model shares, see [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md).
