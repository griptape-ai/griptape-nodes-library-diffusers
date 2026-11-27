---
description: Make an image in two stages. The first stage starts it with one prompt and the second stage finishes it with a different prompt.
when_to_use: Use it when you want to change the prompt, or any other setting, part way through a generation. Use Text2Image for a single-stage run.
---

This workflow makes one image with two Generate Latent nodes in a row. The first one runs the early steps and stops. The second one picks up where it stopped and runs the rest. Each stage has its own prompt. The template uses Z-Image Turbo, a fast image model. Stage 1 says "A calico cat on a cobble stone wall, minecraft" and stage 2 says "A calico cat on a cobble stone wall, watercolor".

The early steps decide layout and the later steps decide detail and style. Stage 1 sets the layout and stage 2 changes the look.

## What it connects

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) for stage 1 reads the first prompt from a Text Input node. It runs from step 0 to step 7 of 20 and stops with the image still partly noisy.
1. A second [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) for stage 2 reads the second prompt from another Text Input node. It continues from step 7 to the end.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **MultistageText2Image** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Type the stage 1 prompt in the first Text Input node and the stage 2 prompt in the second.
1. Run the workflow.

## Defaults

- Model: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- Size: 1024 x 1024.
- Steps: 20 on both stages.
- Stage 1: `start_step` 0 and `end_step` 7.
- Stage 2: `start_step` 7 and `end_step` -1, which means it runs to the end.
- Add Noise is off on both stages.
- Seed: 42.

## Tips

- The two stages must agree on the split. Stage 2 `start_step` has to equal stage 1 `end_step`, and both must use the same number of steps. See [end_step](../nodes/DiffusionPipelineGenerateLatentNode.md#end_step) and [start_step](../nodes/DiffusionPipelineGenerateLatentNode.md#start_step).
- Move the split to change how much stage 1 decides. A later split locks in more of the layout and leaves less for the second prompt to change.
- Add Noise is off, so the seed on Generate Latent has no effect. Change the seed on Create Noise Latent instead. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Use the same pipeline type for the whole chain. Mixing model pipelines gives poor results.
