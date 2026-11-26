---
title: Build Pipeline by model
description: Model-specific and memory notes for the Build Pipeline node.
---

Build Pipeline loads a diffusion model and its settings so the other nodes can use it. The loaded model and its settings are called the pipeline. The sections follow the order of the node.

First come notes for MiniMax-H3, Flux3, WAN Animate 2 and LTX2, which cover what is different for those models. Then come the memory settings, which help a large model fit on your graphics card, and the component overrides, which swap one part of a model for your own.

`quantization_mode` and `transformer_layerwise_casting` do nothing for MiniMax-H3, Flux3 and WAN Animate 2. Those three manage their own memory. `cpu_offload_strategy` is ignored for those three too. The component overrides also skip those three. They do not expose any parts for replacement.

## MiniMax-H3

MiniMax-H3 makes video together with a soundtrack. Pick the pipeline type for the job:

- `MiniMaxH3ModularPipeline` makes a video from a text prompt, or from keyframes (pictures that set what the video should look like at chosen points). The soundtrack is generated with the video.
- `MiniMaxH3Ref2VAModularPipeline` makes a video from reference material: up to 9 images and 3 video clips. It loads its own transformer, which is stored in the same Hugging Face repo as the base model. A repo is a named folder of files on Hugging Face.

This model manages its own memory, so the node's memory settings (`quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy`) do nothing for it. The transformer is held in your computer's main memory and moves onto the graphics card only while it works. The text encoder stays on disk and is loaded each time it is needed.

- **Hardware:** the model card recommends one graphics card with 80 GB of memory, and a large amount of main memory as well. The transformer and the text encoder are each about 62 GB.
- **Downloads:** the text encoder stays on disk, so keep the model files downloaded.
- **LoRAs:** not supported. A LoRA is a small add-on file that gives a model a style or subject.

## Flux3

Flux3 makes images. Pick the pipeline type for the speed you want:

- `Flux3Pipeline` is the base model.
- `Flux3DistilledPipeline` is a distilled version, trained to run in 16 steps.

Like MiniMax-H3, Flux3 manages its own memory. The node's memory settings (`quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy`) do nothing for it. Each part of the model moves onto the graphics card only while it works.

`prompt_upsampler_model` is for Flux3 only. A prompt upsampler rewrites your prompt in more detail before the image is made. The option is `black-forest-labs/flux-3-image-klein-pup`. Choose `Disabled` to use your prompt as written.

The model list first shows the standard Flux3 model. Once the compact versions are downloaded, two more rows appear: `fp8r` and `nvfp4`. Each uses a smaller transformer that needs less graphics-card memory. Only the transformer changes.

`fp8r` runs on any device. On an NVIDIA card with CUDA compute capability 8.9 or higher (RTX 40-series and newer, including RTX 50-series) it uses fast fp8 math. Elsewhere, including a Mac, the weights are converted to a larger number format as they are used, so it saves disk space but gains no speed. The run stops with a message only on a card of 8.9 or higher whose PyTorch install cannot do the fp8 matrix multiply. Then pick the default row or `nvfp4`.

## WAN Animate 2

WAN Animate 2 (`WanAnimate2Pipeline`) animates a character image with the motion from a driving video, the clip that supplies the movement. `model` offers the standard and the distilled checkpoint. A distilled checkpoint is a version trained to run in fewer steps. With it, Generate Latent defaults to 10 steps and `guidance_scale` 1.0.

The node's memory settings (`quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy`) do nothing for this model. It manages its own memory: the model is stored in your computer's main memory, and each part moves onto the graphics card only while it works.

The first Generate run compiles the transformer, a one-time setup that makes it slower than later runs.

## LTX2

LTX2 makes video. The `LTX2Pipeline` type lists the LTX-2.3, LTX-2.3 Distilled and LTX-2 checkpoints. `LTX-2.5 Distilled` and `LTX-2.5 Full (SFT)` are both built from the `Lightricks/LTX-2.5-Diffusers` repo. They differ only in the transformer: distilled, or full. The VAE used to decode the video is the same for all of them.

For the settings every model shares, see [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md).

## Memory settings

These settings help a large model fit on your graphics card. They appear when `memory_optimization_strategy` is `Manual`, which is the default. `Automatic` hides them and chooses settings from the GPU memory available.

A model's weights are the numbers it learned in training. Some settings below store them in a more compact form to save memory.

### attention_slicing

Off by default. Attention is one of the heavy calculations the model repeats at every step. This setting does it in pieces, which lowers peak memory and runs slightly slower.

### vae_slicing

Off by default. The decoder is the part of the [VAE](what-is-a-vae.md) that turns the model's working format, the latent, back into pixels. This setting makes it decode one slice at a time, which lowers peak memory for large images or batches.

### vae_tiling

Off by default. This setting makes the decoder work on one tile of the picture at a time, which lowers memory for large images and long videos.

### transformer_layerwise_casting

Off by default. This setting stores the transformer, the part that makes the picture, in fp8, a compact number format, and converts each layer as it runs. It saves memory and can slow runs slightly. It has no effect on models that already ship in a compact format. It is also ignored for Z-Image and whenever a quantized component override is connected.

### cpu_offload_strategy

`None` (default), `Model` or `Sequential`. This setting moves idle weights to the CPU (system memory) to free GPU memory. `Model` moves whole parts. `Sequential` moves smaller pieces and costs more speed. It is ignored for MiniMax-H3, Flux3, WAN Animate 2 and the WAN, WAN I2V, WAN Animate, WAN VACE and HunyuanVideo 1.5 pipelines, which place their own parts on the card. Attention slicing and VAE slicing and tiling still run before that.

### quantization_mode

`None` (default), `fp8`, `int8` or `int4`. This setting converts the weights of every part of the model to a more compact format before the run, using the optimum-quanto library. `fp8` and `int8` both store weights in 8 bits and `int4` stores them in 4 bits, so `int4` saves the most memory. More savings cost more quality and speed.

### Example settings by hardware

These are starting points. Watch the first run and adjust.

**Mac with Apple Silicon.** The GPU shares memory with the rest of the computer, so the model competes with the OS and your other apps.

- Start with `Automatic`. If the whole pipeline fits in the memory macOS recommends for the GPU, it runs there. If it does not fit, `Automatic` turns on VAE slicing and still tries the run. It does not use fp8 casting or CPU offload on a Mac.
- If a run still fails, switch to `Manual` and turn on `vae_slicing` and `vae_tiling`. These lower the memory used when the image or video is decoded.
- Close other heavy apps before a long video run.
- LTX2 decodes its audio on the CPU on a Mac. This is automatic.
- Flux3 `fp8r` gives no speed benefit on a Mac.

**NVIDIA RTX 50-series (Blackwell), such as an RTX 5090.**

- Start with `Automatic`. It moves the whole pipeline to the GPU when it fits with 20% spare. If not, it tries fp8 layerwise casting on the transformer, then model CPU offload, then sequential CPU offload. It also turns on VAE slicing and tiling.
- If the model fits in your card's memory, `Manual` with every option off is the fastest setting, because nothing is converted or moved.
- On a smaller card, turn on `transformer_layerwise_casting` first, then `cpu_offload_strategy` `Model`. Use `Sequential` only as a last resort because it is the slowest.
- Flux3 `fp8r` uses fast fp8 math on RTX 40-series and newer cards, which includes the RTX 50-series. `nvfp4` is the other compact row.

## Component overrides

The Component Overrides group has one port for each part this pipeline type has, such as Transformer or VAE. MiniMax-H3, Flux3 and WAN Animate 2 have none. Connect a [Load Component](../nodes/LoadComponent.md) node to replace that part of the model with your own weights, the files that hold what that part learned. To learn what the VAE does and when to replace it, see [What is a VAE?](what-is-a-vae.md). The other parts have guides too: [What are a transformer and a UNet?](what-is-a-transformer.md), [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md) and [What is a scheduler?](what-is-a-scheduler.md).

If a replacement does not fit, a message appears on the node. A VAE whose channel count does not match the model, or a text encoder whose output size does not match it, stops the run. A VAE with a different scaling, a scheduler with a different prediction type, and a tokenizer that does not pair with its text encoder show as warnings only.
