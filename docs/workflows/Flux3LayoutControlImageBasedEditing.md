---
description: Edit a picture by moving an item from a box in your image to a new box, described with a short list entry.
when_to_use: Use it when you want to move or resize something in an existing image and keep the rest. Use Flux3LayoutControlText2Image to make a new image from scratch.
---

This workflow edits a picture you load with Flux3. Instead of a written prompt, you give the model a list entry that says where an item is in your picture, where it should go, and a short description. The template loads a photo of a base jumper and makes the same person slightly larger in a new position.

## What it connects

1. A Load Image node supplies your picture.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model, with a prompt upsampler selected.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) receives your picture on `reference_images`, reads the list entry from a Text Input node and runs 16 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **Flux3LayoutControlImageBasedEditing** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. If Model Manager says you need access, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Load your picture in the Load Image node.
1. Edit the list entry in the Text Input node.
1. Run the workflow.

## The list entry

Each entry has these parts:

- `id` is a name for the box.
- `from` says which input picture the item comes from. The first picture is `ref_image_0`.
- `src_bbox` is the box around the item in that picture.
- `tgt_bbox` is the box where the item should appear in the result.
- `desc` is extra guidance for the model.

A box is written as `[y_min, x_min, y_max, x_max]` on a scale from 0 to 1000. Y is first. The scale is relative, so the same box works at any image size.

The template uses `basejumper_1` with `ref_image_0`, a `src_bbox` of `[130, 723, 455, 1000]`, a `tgt_bbox` of `[193, 355, 656, 701]` and the description "The same base jumper slightly larger".

The layout takes effect only when a prompt upsampler is selected on Build Pipeline. See [Flux3 in Build Pipeline by model](../guides/build-pipeline-by-model.md#flux3) and [Flux3 in Generate Latent by model](../guides/generate-latent-by-model.md#flux3).

## Defaults

- Model: Flux3 (`black-forest-labs/flux-3-image-klein`), pipeline type `Flux3DistilledPipeline`.
- Prompt upsampler: `black-forest-labs/flux-3-image-klein-pup`.
- Memory strategy: Automatic.
- Size: 1024 x 1024.
- Steps: 16.
- Seed: 42.

## Tips

- Add Noise is off on Generate Latent, so the seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Generate Latent takes up to 10 picture inputs on `reference_images`. Video is not supported.
- To place new items without a source picture, see [Flux3LayoutControlText2Image](Flux3LayoutControlText2Image.md).
