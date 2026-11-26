---
title: What are a transformer and a UNet?
description: The network that does the actual image and video making, and when to swap it in the nodes.
---

The transformer is the part of a diffusion model that actually makes the picture or clip. It is the engine, and the other parts feed it or translate its output. It works on a [latent](what-is-a-latent.md), the compressed block of numbers that stands for an image or video. Starting from noise, it cleans the latent up a little at a time (denoising), guided by your prompt.

## The idea

The transformer is the main neural network inside a diffusion model. A neural network is a fixed arrangement of calculations. Its behavior comes from the numbers it learned in training. The transformer takes a noisy latent and your encoded prompt and produces a slightly cleaner latent. It does this once per step, for every step of a run, so it is where most of a run's time goes. The [VAE](what-is-a-vae.md) and the [text encoder](what-is-a-text-encoder.md) only translate at either end.

Stable Diffusion XL, and the Stable Diffusion 1.x and 2.x models before it, use a different design for this network, called a **UNet**. It does the same job. In the nodes, each [pipeline type](how-pipelines-work.md#what-build-pipeline-decides) has either a Transformer port or a UNet port, never both. In this library only the Stable Diffusion XL pipeline type has a UNet port. The other pipeline types that have override ports use a Transformer port.

```text
noisy latent + prompt numbers ──transformer (or UNet)──▶ cleaner latent   (repeated each step)
```

## Where it shows up in the nodes

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the transformer or UNet with the rest of the model. Its Component Overrides group shows a Transformer port or a UNet port, depending on the pipeline type.
- [Load Component](../nodes/LoadComponent.md) with `component` set to `Transformer` or `UNet` swaps in your own. Some pipeline types also have a Transformer 2 port, which a `Transformer` loader can feed.
- [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) is where the denoising steps run.
- The memory settings in Build Pipeline (see [How pipelines work](how-pipelines-work.md#memory-settings)) can shrink the model, including the transformer, without replacing it.

## Why swap it

Most workflows should keep the transformer that comes with the model. People swap it for a few reasons:

- **A fine-tune.** A fine-tune is a base model retrained to favor a look or subject. A community fine-tune changes the look or the subject matter the model is good at, without changing the rest of the pipeline.
- **A GGUF or other quantized file.** Quantizing stores the model's numbers with less accuracy, like a lower bit depth. The file is smaller and quality drops a little. GGUF is one such file format. A quantized transformer is smaller, so a model that would not fit in memory can run.
- **A single file instead of the Diffusers format.** A Diffusers-format model is a folder of separate parts. See [Hugging Face and Diffusers](hugging-face-and-diffusers.md). Many community models are shared as one file. Build Pipeline's model list expects the Diffusers format, so a single file goes in through Load Component.

To swap it, connect a [Load Component](../nodes/LoadComponent.md) node with `component` set to `Transformer` or `UNet` to the matching port in Build Pipeline's Component Overrides. A GGUF file holds only the learned numbers, not the description of the network's layout, which is called the config. For a GGUF file, set `source_type` to `Single File`. Griptape Nodes looks for the config in your Hugging Face cache automatically. If it cannot find one, point `config_source` at the original model's folder or repo name.

## Things to know

- The replacement has to belong to the same model family as the rest of the pipeline. A transformer from a different model does not fit.
- The transformer has to agree with the VAE. If the VAE's latent channel count does not match what the transformer expects, the run stops with a message on the Build Pipeline node.
- A quantized override switches off `transformer_layerwise_casting`, because the model's numbers are already stored in a smaller form.
- Load Component describes where the files are. Build Pipeline loads them when it builds the pipeline, and it rebuilds the pipeline when you change the loader.
- The files have to be on your machine already. Nothing is downloaded.

Related: [What is a text encoder?](what-is-a-text-encoder.md), [What is a scheduler?](what-is-a-scheduler.md).
