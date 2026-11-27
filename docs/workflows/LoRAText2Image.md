---
description: Make an image from a prompt with two LoRAs applied to the model. A LoRA is a small add-on file that gives a model a style, character or effect.
when_to_use: Use it when you have LoRA files and want their style or subject in a text-to-image run. Use Text2Image if you do not need a LoRA.
---

This workflow turns a text description, called a prompt, into an image, with two LoRAs applied. A LoRA is a small file trained separately to give a model a style, a character or an effect. It uses Z-Image Turbo, a fast image model. The prompt in the template is "A man".

The LoRA files are not included with the library. You download them first.

## What it connects

1. Two [Load LoRA](../nodes/LoadLora.md) nodes each load one LoRA file and set its weight. The weight is how strongly the LoRA applies.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model and receives both LoRAs. They are fused into the model when the pipeline is built.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and runs 20 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **LoRAText2Image** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Download the two LoRA files named in the template notes: `z_image_arcane_v1.safetensors` from the Hugging Face repo `DeverStyle/Z-Image-loras` and `zimage_adamdriver_v2_onetrainer.safetensors` from `malcolmrey/zimage`.
1. On each Load LoRA node, set `file_path` to the file on your disk. The template looks for them in the `workflows/assets/lora` folder of the library. A Hugging Face repo name does not work in this field.
1. Type your prompt in the Text Input node.
1. Run the workflow.

To use a different LoRA, choose one made for the same model family as the pipeline. A LoRA made for another family does not work.

## Defaults

- Model: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- LoRAs: two, each with a weight of 0.75.
- Size: 1024 x 1024.
- Steps: 20.
- Seed: 42.

## Tips

- The weight is fixed when the pipeline is built. Changing it rebuilds the pipeline, so the next run is slower. See [weight](../nodes/LoadLora.md#weight).
- Lower a weight to soften that LoRA. Set it to 0 to turn it off.
- If a LoRA has a trigger phrase, type the phrase into your prompt yourself. Load LoRA does not add it for you. See [trigger_phrase](../nodes/LoadLora.md#trigger_phrase).
- To apply a LoRA on only one branch of a workflow, use Apply Runtime LoRA instead. See [Load LoRA](../nodes/LoadLora.md).
- Add Noise is off on Generate Latent, so the seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
