---
title: What are a transformer and a UNet?
description: The network that does the actual image and video making, and when to swap it in the nodes.
---

The transformer is the part of a diffusion model that actually makes the picture or clip. It is the engine, and the other parts feed it or translate its output. It works on a [latent](what-is-a-latent.md), the compressed block of numbers that stands for an image or video. Starting from noise, it cleans the latent up a little at a time (denoising), guided by your prompt.

## The idea

The transformer is the main neural network inside a diffusion model. A neural network is a fixed arrangement of calculations. Its behavior comes from its weights, the numbers it learned in training. The transformer takes a noisy latent and your encoded prompt and produces a slightly cleaner latent. It does this once per step, for every step of a run, so it is where most of a run's time goes. The [VAE](what-is-a-vae.md) and the [text encoder](what-is-a-text-encoder.md) only translate at either end.

Stable Diffusion XL, and the Stable Diffusion 1.x and 2.x models before it, use a different design for this network, called a **UNet**. It does the same job. In the nodes, a pipeline type has either a Transformer port or a UNet port, never both. In this library only the Stable Diffusion XL pipeline type has a UNet port. The other pipeline types that have override ports use a Transformer port.

```text
noisy latent + prompt numbers ──transformer (or UNet)──▶ cleaner latent   (repeated each step)
```

The transformer works on the latent, never on pixels.

## Where it shows up in the nodes

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the transformer or UNet with the rest of the model. Its Component Overrides group shows a Transformer port or a UNet port, depending on the pipeline type.
- [Load Component](../nodes/LoadComponent.md) with `component` set to `Transformer` or `UNet` swaps in your own. Some pipeline types also have a Transformer 2 port, which a `Transformer` loader can feed.
- [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) is where the denoising steps run.
- The `quantization_mode`, `transformer_layerwise_casting` and CPU offload settings in [Memory settings](how-pipelines-work.md#memory-settings) reduce the memory the transformer needs without replacing it. Layerwise casting keeps the transformer in fp8, a compact number format, and expands one layer at a time as it runs. CPU offload keeps idle parts in the computer's main memory instead of on the graphics card.

## Why swap it

Most workflows should keep the transformer that comes with the model. People swap it for a few reasons:

- **A fine-tune or checkpoint.** A checkpoint is a saved set of model weights. A fine-tune is a base model retrained to favor a look or subject. A community checkpoint changes the look or the subject matter the model is good at, without changing the rest of the pipeline.
- **A GGUF or other quantized file.** Quantizing stores the weights at lower precision, which makes the file smaller. GGUF is one such file format. A quantized transformer is smaller, so a model that would not fit in memory can run. The quality loss depends on how hard it was quantized.
- **A single file instead of the Diffusers format.** The Diffusers format stores a model as a folder of separate parts. See [Hugging Face and Diffusers](hugging-face-and-diffusers.md). Many checkpoints are shared as one file. Build Pipeline's model list expects the Diffusers format, so a single file goes in through Load Component.

To swap it, connect a [Load Component](../nodes/LoadComponent.md) node with `component` set to `Transformer` or `UNet` to the matching port in Build Pipeline's Component Overrides. For a GGUF file, set `source_type` to `Single File`. `config_source` is optional. Leave it blank and it is found automatically from the Hugging Face cache. If it cannot be found automatically, set `config_source` to the folder of the matching original transformer, to its `config.json`, or to its Hugging Face repo name. The config describes the transformer's layout, which a GGUF file does not carry.

## Things to know

- The replacement has to belong to the same model family as the rest of the pipeline. A transformer from a different model does not fit.
- The transformer has to agree with the VAE. If the VAE's latent channel count does not match what the transformer expects, the run stops with a message on the Build Pipeline node.
- A quantized override switches off `transformer_layerwise_casting`, because the weights are already stored in a smaller form.
- Load Component describes where the weights are. Build Pipeline loads them when it builds the pipeline, and it rebuilds the pipeline when you change the loader.
- The weights have to be on your machine already. Nothing is downloaded.

Related: [What is a text encoder?](what-is-a-text-encoder.md), [What is a scheduler?](what-is-a-scheduler.md).
