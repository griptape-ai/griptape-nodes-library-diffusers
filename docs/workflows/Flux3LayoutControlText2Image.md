---
description: Make an image from a prompt and place each item in the frame with a list of boxes added after the prompt.
when_to_use: Use it for layouts where position matters, such as an advertisement with a product in the middle. Use Text2Image when you do not need to place things.
---

This workflow makes an image with Flux3 and controls where things appear. You write a normal prompt, then add a list of regions after it. Each region has a box and a short description. The prompt points at a region by its name. The template makes an advertisement style image of a pub scene with a beer bottle, a burger and fries, a table and a blurred background.

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model, with a prompt upsampler selected.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt and the region list from a Text Input node and runs 16 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **Flux3LayoutControlText2Image** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If Model Manager says you need access, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Edit the prompt in the Text Input node. Name each region in the prompt as `<id>`, for example `<beer_1>`.
1. Edit the region list after the prompt. Each region has an `id`, a `bbox` and a `desc`.
1. Run the workflow.

## The region list

A `bbox` is a box written as `[y_min, x_min, y_max, x_max]` on a scale from 0 to 1000. Y is first. The scale is relative, so the same box works at any image size. For example, the `background_1` region uses `[0, 0, 1000, 1000]` and covers the whole frame.

The layout takes effect only when a prompt upsampler is selected on Build Pipeline. With it set to `Disabled`, the list is read as plain prompt text. See [Flux3 in Build Pipeline by model](../guides/build-pipeline-by-model.md#flux3) and [Flux3 in Generate Latent by model](../guides/generate-latent-by-model.md#flux3).

## Defaults

- Model: Flux3 (`black-forest-labs/flux-3-image-klein`), pipeline type `Flux3DistilledPipeline`.
- Prompt upsampler: `black-forest-labs/flux-3-image-klein-pup`.
- Memory strategy: Manual. Flux3 manages its own memory, so the memory settings do not apply. See [How pipelines work](../guides/how-pipelines-work.md#memory-settings).
- Regions: `background_1`, `table_1`, `beer_1` and `burger_1`.
- Size: 1024 x 1024.
- Steps: 16.
- Seed: 42 on Generate Latent and 0 on Create Noise Latent.

## Tips

- Add Noise is off on Generate Latent, so the seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- The distilled type shows no negative prompt and no guidance setting.
- Give each region its own description. The description of the whole scene belongs in the prompt.
- To edit an existing picture instead, see [Flux3LayoutControlImageBasedEditing](Flux3LayoutControlImageBasedEditing.md).
