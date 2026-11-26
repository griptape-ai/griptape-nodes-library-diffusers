---
title: Build Pipeline by model
description: Model-specific and memory notes for the Build Pipeline node.
---

Build Pipeline loads a diffusion model and its settings so the other nodes can use it. The loaded model and its settings are called the pipeline. This page covers the memory settings, which help a large model fit on your graphics card, and the component overrides, which swap one part of a model for your own. Both apply to every model. The sections for MiniMax-H3, Flux3, WAN Animate 2 and LTX2 cover what is different for those models.

## Memory settings

These appear, together with `quantization_mode` (which stores weights at lower precision to save memory), when `memory_optimization_strategy` is `Manual`. The `Manual` setting is the default. `Automatic` hides them and chooses settings from the GPU memory available.

### attention_slicing

Off by default. Attention is one of the model's main calculation steps. This setting works through it in slices. This lowers peak memory with a small loss of speed.

### vae_slicing

Off by default. Decodes one slice at a time. The decoder is the part of the [VAE](what-is-a-vae.md) that turns the model's working format, the latent, back into pixels. This lowers peak memory for large images or batches.

### vae_tiling

Off by default. Decodes in tiles, which lowers memory for large images and long videos.

### transformer_layerwise_casting

Off by default. Stores the transformer, the part that makes the picture, in fp8 (a compact number format) and converts each layer as it runs. This saves memory and can slow runs slightly. It has no effect on models that ship quantized weights.

### cpu_offload_strategy

`None` (default), `Model` or `Sequential`. Moves idle weights to the CPU (system memory) to free GPU memory. `Model` moves whole parts and `Sequential` moves smaller pieces at a larger cost in speed.

### Example settings by hardware

These are starting points. Watch the first run and adjust.

**Mac with Apple Silicon.** The GPU shares memory with the rest of the computer, so the model competes with the OS and your other apps.

- Start with `Automatic`. If the whole pipeline fits in the memory macOS recommends for the GPU, it runs there. If it does not fit, `Automatic` turns on VAE slicing and still tries the run. It does not use fp8 casting or CPU offload on a Mac.
- If a run still fails, switch to `Manual` and turn on `vae_slicing` and `vae_tiling`. These lower the memory used when the image or video is decoded.
- Close other heavy apps before a long video run.
- LTX2 decodes its audio on the CPU on a Mac. This is automatic.
- `fp8r` Flux3 rows need an NVIDIA GPU. Use the default row on a Mac.

**NVIDIA Blackwell (RTX 50 series and data centre B-series).**

- Start with `Automatic`. It moves the whole pipeline to the GPU when it fits with 20% spare. If not, it tries fp8 layerwise casting on the transformer, then model CPU offload, then sequential CPU offload. It also turns on VAE slicing and tiling.
- On a card with plenty of memory, `Manual` with every option off is the fastest setting, because nothing is converted or moved.
- On a smaller card, turn on `transformer_layerwise_casting` first, then `cpu_offload_strategy` `Model`. Use `Sequential` only as a last resort because it is the slowest.
- Flux3 `fp8r` rows run on cards with CUDA compute capability 8.9 or higher, which includes Blackwell. `nvfp4` is the other quantized row.

## Component overrides

The Component Overrides group has one port for each part this pipeline type has, such as Transformer or VAE. MiniMax-H3, Flux3 and WAN Animate 2 have none. Connect a [Load Component](../nodes/LoadComponent.md) node to replace that part of the model with your own weights, the files that hold what that part learned. To learn what the VAE does and when to replace it, see [What is a VAE?](what-is-a-vae.md). The other parts have guides too: [What are a transformer and a UNet?](what-is-a-transformer.md), [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md) and [What is a scheduler?](what-is-a-scheduler.md).

If a replacement does not fit, a message appears on the node. A VAE whose channel count does not match the model, or a text encoder whose output size does not match it, stops the run. A VAE with a different scaling, a scheduler with a different prediction type, and a tokenizer that does not pair with its text encoder show as warnings only.

## MiniMax-H3

The quantization, layerwise casting and CPU offload settings are not used. The model is loaded in bfloat16 (a 16-bit number format) with automatic CPU offload instead. It needs a single 80 GB accelerator plus plenty of system memory. The text encoder is reloaded from disk each time it is needed, so the model files must stay downloaded. LoRAs, small add-on files for style or subject, are not supported.

`MiniMaxH3ModularPipeline` does text-to-video and keyframe-to-video, with a soundtrack generated together with the video. `MiniMaxH3Ref2VAModularPipeline` does reference-to-video from up to 9 images and 3 video clips, and uses a separate transformer in the same repo.

## Flux3

`prompt_upsampler_model` is for Flux3 only. It is an optional prompt upsampler, which expands your prompt before generation, `black-forest-labs/flux-3-image-klein-pup`. Choose `Disabled` to pass prompts through unchanged.

`Flux3Pipeline` is the base model. `Flux3DistilledPipeline` is the 16-step distilled model, a version made to run in fewer steps. The quantization, layerwise casting and CPU offload settings are not used. The model is loaded with automatic CPU offload instead.

The model list offers the full-precision repo and, when downloaded, `fp8r` and `nvfp4` rows with quantized transformers. A quantized row replaces only the transformer. `fp8r` stops the run with a message on a GPU or torch build that cannot run per-row scaled matrix multiplication. Pick the default row or `nvfp4` there.

## WAN Animate 2

`WanAnimate2Pipeline` animates a character image with the motion of a driving video, the clip that supplies the movement. `model` offers the standard and the distilled checkpoint. With the distilled one, Generate Latent defaults to 10 steps and `guidance_scale` 1.0.

The quantization, layerwise casting and CPU offload settings are not used. The model is loaded in bfloat16 with automatic CPU offload instead. The transformer is compiled during the first Generate run, so that run takes noticeably longer.

## LTX2

The `LTX2Pipeline` type lists LTX-2.3, LTX-2.3 Distilled and LTX-2 checkpoints. `LTX-2.5 Distilled` and `LTX-2.5 Full (SFT)` both build an `LTX2Pipeline` from the `Lightricks/LTX-2.5-Diffusers` repo. They differ only in the transformer: distilled or full. Decoding always uses the plain VAE.

For the settings every model shares, see [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md).
