---
title: How pipelines work
description: What a pipeline is, how Build Pipeline makes one, how it is reused, and the settings that decide how it uses memory and which parts it is made of.
---

A pipeline is a model that has been loaded, together with its settings, so the other nodes can use it. It holds the parts the model is made of: the [transformer or UNet](what-is-a-transformer.md) that makes the picture, the [VAE](what-is-a-vae.md) that converts between pictures and latents, the [text encoder and tokenizer](what-is-a-text-encoder.md) that turn your prompt into numbers, and the [scheduler](what-is-a-scheduler.md) that sets how the clean-up proceeds step by step. This differs from a studio's production pipeline, which moves work between departments. A pipeline here is one loaded model.

## The journey through the nodes

Every workflow follows the same path:

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model and outputs the pipeline.
2. [Create Noise Latent](../nodes/NoiseLatentNode.md) makes a random starting point, or [Encode Media](../nodes/VaeEncodeNode.md) turns a picture or clip of yours into one.
3. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) cleans the noise out of that starting point, step by step, to match your prompt.
4. [Decode Media](../nodes/VaeDecodeNode.md) turns the result back into a picture or clip.

The starting point and the result are latents, the compressed form of a picture that the model works on. See [What is a latent?](what-is-a-latent.md).

## What Build Pipeline decides

These are the choices on [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md), in the order they appear on the node.

- **`provider`** is the model family, such as Flux, WAN or LTX2. Set it first, because changing it resets the settings below it.
- **`pipeline_type`** is the variant within the family. It decides what the pipeline can do and which checkpoints are offered. Some variants have their own notes in [Build Pipeline by model](build-pipeline-by-model.md).
- **`model`** is the checkpoint, chosen from the models you have downloaded. See [Hugging Face and Diffusers](hugging-face-and-diffusers.md).
- **Memory settings** decide how the model fits on your graphics card. See [Memory settings](#memory-settings) below.
- **`loras`** fuse one or more [LoRAs](../nodes/LoadLora.md) into the model. A LoRA is a small add-on file that gives a model a style or subject. Changing one rebuilds the pipeline.
- **`component_overrides`** replace single parts of the model with your own files. See [Component overrides](#component-overrides) below.

## Built once and reused

Build Pipeline builds the pipeline when it runs and keeps it in memory. The pipeline is identified by the pipeline type, the model, the LoRAs and the memory settings. If you run the node again with the same values, it reuses the pipeline it already holds. If you change any of them, it builds a new one. Prompts, step counts and other settings on later nodes are not part of that identity, so changing them never rebuilds.

Only one pipeline is held at a time. Before a new one is built, the engine releases the one it holds. [Clear Pipeline Cache](../nodes/ClearPipelineCacheNode.md) releases every held pipeline on demand, for example after an out-of-memory error. The cache lives in memory only, so after a restart the next run builds the pipeline again.

Building reads the model files from your disk. It does not download anything. Download models first, as described in [Hugging Face and Diffusers](hugging-face-and-diffusers.md).

## Nodes that use the pipeline

- [Load LoRA](../nodes/LoadLora.md) and [Apply Runtime LoRA](../nodes/LoraActivationPipelineNode.md) add LoRAs, either fused at build time or switched on for a single generation.
- [Apply ControlNet](../nodes/ControlNetDiffusionPipelineBuilderNode.md) attaches guide images, such as edges or depth, to the pipeline.
- [Load Scheduler](../nodes/LoadSchedulerComponent.md) supplies a different scheduler through a component override.
- [Estimate Pipeline Memory](../nodes/PipelineMemoryEstimateNode.md) shows how much graphics card memory a run needs.
- [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) runs the model.
- [Decode Media](../nodes/VaeDecodeNode.md) uses the pipeline's VAE to produce the picture or clip.

## Memory settings

These settings help a large model fit on your graphics card. They appear when `memory_optimization_strategy` is `Manual`, which is the default. `Automatic` hides them and chooses settings from the GPU memory available.

MiniMax-H3, Flux3 and WAN Animate 2 manage their own memory. `quantization_mode`, `transformer_layerwise_casting` and `cpu_offload_strategy` do nothing for them. Their notes are in [Build Pipeline by model](build-pipeline-by-model.md).

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

`None` (default), `Model` or `Sequential`. This setting moves idle weights to the CPU (system memory) to free GPU memory. `Model` moves whole parts. `Sequential` moves smaller pieces and costs more speed. It is also ignored for the WAN, WAN I2V, WAN Animate, WAN VACE and HunyuanVideo 1.5 pipelines, which place their own parts on the card. Attention slicing and VAE slicing and tiling still run before that.

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

The Component Overrides group has one port for each part this pipeline type has, such as Transformer or VAE. The three models that manage their own memory (see [Memory settings](#memory-settings)) have none. Connect a [Load Component](../nodes/LoadComponent.md) node to replace that part of the model with your own weights, the files that hold what that part learned. To learn what the VAE does and when to replace it, see [What is a VAE?](what-is-a-vae.md). The other parts have guides too: [What are a transformer and a UNet?](what-is-a-transformer.md), [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md) and [What is a scheduler?](what-is-a-scheduler.md).

If a replacement does not fit, a message appears on the node. A VAE whose channel count does not match the model, or a text encoder whose output size does not match it, stops the run. A VAE with a different scaling, a scheduler with a different prediction type, and a tokenizer that does not pair with its text encoder show as warnings only.
