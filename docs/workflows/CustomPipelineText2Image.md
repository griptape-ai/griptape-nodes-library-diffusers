---
description: Build a model from parts you choose, such as your own transformer, VAE, text encoder, tokenizer and scheduler, and make an image from a prompt.
when_to_use: Use it when you want to swap one or more parts of a model, for example a quantized transformer file or a different VAE. Use Text2Image if the default model parts are fine.
---

This workflow makes an image from a text description, called a prompt, with a model assembled from separate parts. Build Pipeline loads a base model. Each Load Component node then replaces one part of it. You do not have to replace every part. Any part you leave unwired comes from the base model. Replacing a part is called a component override.

The template uses the FLUX.2 Klein family. It replaces all five parts. The prompt is a cartoon hedgehog in a blue cape standing on a mossy pipeline.

## What it connects

1. Four [Load Component](../nodes/LoadComponent.md) nodes supply the Transformer, the VAE, the Text Encoder and the Tokenizer. See the [transformer](../guides/what-is-a-transformer.md), [VAE](../guides/what-is-a-vae.md) and [text encoder and tokenizer](../guides/what-is-a-text-encoder.md) guides.
1. [Load Scheduler](../nodes/LoadSchedulerComponent.md) supplies the scheduler, which sets how the model removes noise step by step. See [What is a scheduler?](../guides/what-is-a-scheduler.md).
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the base model and swaps in the five parts.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and runs 20 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **CustomPipelineText2Image** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the base model is downloaded.
1. Set up each part. The template notes describe where they come from:
   - Transformer: download `flux-2-klein-4b-Q8_0.gguf`. The template also points at a `transformer_config.json`, which is a copy of the one in the `transformer` folder of `black-forest-labs/FLUX.2-klein-4B`. It tells the loader the shape of the model.
   - Tokenizer: the whole `tokenizer` folder comes from `black-forest-labs/FLUX.2-klein-4B`. The template reads it from a local folder, `workflows/assets/models/flux2-klein/tokenizer`. The notes say it can also come from `Qwen/Qwen3-4B`.
   - The library does not ship these files. Create the folder `workflows/assets/models/flux2-klein/` in the library and put the gguf, `transformer_config.json` and the `tokenizer` folder in it, or repoint the nodes.
   - Text Encoder: `Qwen/Qwen3-4B`, read from a Hugging Face repo that must already be in your cache. The notes say `Qwen/Qwen3-4B-FP8` is a smaller version with some loss in quality.
   - VAE: `Felldude/FLUX.2-HDR-VAE`, read from a Hugging Face repo that must already be in your cache. The notes say it gives better range and colour.
   - Scheduler: `FlowMatchEulerDiscreteScheduler`, set from the text configuration on the node. You can edit that text to change the schedule.
1. Type your prompt in the Text Input node.
1. Run the workflow.

Download repos in Model Manager so they are in the cache. See [Hugging Face and Diffusers](../guides/hugging-face-and-diffusers.md). The same page covers loading files from your own disk.

## Defaults

- Base model: Flux2 Klein 9B (`black-forest-labs/FLUX.2-klein-9B`), pipeline type `Flux2KleinPipeline`.
- Size: 1024 x 1024.
- Steps: 20.
- Guidance: 4.0.
- Seed: 42.

## Tips

- To override only some parts, delete the Load Component nodes you do not want. Parts without an override come from the base model.
- Build Pipeline is set to the 9B model, while the part files in the template notes come from the 4B model. Parts you override replace the ones from the base repo.
- To use a different VAE, change `repo_id` on that Load Component, or change `source_type` to load from a local folder. If the repo holds a full pipeline, set `subfolder` to the part's folder.
- A VAE with the wrong channel count, or a text encoder with the wrong output size, stops the run with a message. See [Component overrides](../guides/how-pipelines-work.md#component-overrides).
- Add Noise is off on Generate Latent, so the seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
