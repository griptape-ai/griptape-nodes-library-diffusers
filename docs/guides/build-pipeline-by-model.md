---
title: Build Pipeline by model
description: Model-specific and memory notes for the Build Pipeline node.
---

Build Pipeline loads a diffusion model and its settings so the other nodes can use it. The loaded model and its settings are called the pipeline. The sections follow the order of the node. First come the notes for MiniMax-H3, Flux3, WAN Animate 2 and LTX2, which cover what is different for those models. Then come the memory settings, which help a large model fit on your graphics card, and the component overrides, which swap one part of a model for your own. The last two apply to every model.

## MiniMax-H3

MiniMax-H3 makes video together with a soundtrack. Pick the pipeline type for the job:

- `MiniMaxH3ModularPipeline` makes a video from a text prompt, or from keyframes (pictures that set what the video should look like at chosen points). The soundtrack is generated with the video.
- `MiniMaxH3Ref2VAModularPipeline` makes a video from reference material: up to 9 images and 3 video clips. It uses a separate transformer from the same repo.

This model manages its own memory, so the node's memory settings (`quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy`) do nothing for it. The model is stored in your computer's main memory, and each part moves onto the graphics card only while it works. It stores its numbers in 16 bits (bfloat16).

- **Hardware:** it needs one accelerator, meaning a graphics card, with 80 GB of memory, and plenty of main memory as well.
- **Downloads:** the text encoder is loaded from disk each time it is needed, so keep the model files downloaded.
- **LoRAs:** not supported. A LoRA is a small add-on file that gives a model a style or subject.

## Flux3

Flux3 makes images. Pick the pipeline type for the speed you want:

- `Flux3Pipeline` is the base model.
- `Flux3DistilledPipeline` is a distilled version, trained to run in 16 steps.

Like MiniMax-H3, Flux3 manages its own memory. The node's memory settings (`quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy`) do nothing for it. Each part of the model moves onto the graphics card only while it works.

`prompt_upsampler_model` is for Flux3 only. A prompt upsampler rewrites your prompt in more detail before the image is made. The option is `black-forest-labs/flux-3-image-klein-pup`. Choose `Disabled` to use your prompt as written.

The model list shows the full-precision model. After you download them, it also shows `fp8r` and `nvfp4` rows. These swap in a transformer that is stored at lower precision to use less memory. Only the transformer changes. The `fp8r` row stops the run with a message if your GPU or your torch install cannot run it. In that case pick the default row or `nvfp4`.

## WAN Animate 2

WAN Animate 2 (`WanAnimate2Pipeline`) animates a character image with the motion from a driving video, the clip that supplies the movement. `model` offers the standard and the distilled checkpoint. A distilled checkpoint is a version trained to run in fewer steps. With it, Generate Latent defaults to 10 steps and `guidance_scale` 1.0.

The node's memory settings (`quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy`) do nothing for this model. It manages its own memory: the model is stored in your computer's main memory, and each part moves onto the graphics card only while it works. It stores its numbers in 16 bits (bfloat16).

The first Generate run is slower than later ones, because the transformer is optimised on that first run.

## LTX2

LTX2 makes video. The `LTX2Pipeline` type lists the LTX-2.3, LTX-2.3 Distilled and LTX-2 checkpoints. `LTX-2.5 Distilled` and `LTX-2.5 Full (SFT)` are both built from the `Lightricks/LTX-2.5-Diffusers` repo. They differ only in the transformer: distilled, or full. The VAE used to decode the video is the same for all of them.

For the settings every model shares, see [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md).

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
