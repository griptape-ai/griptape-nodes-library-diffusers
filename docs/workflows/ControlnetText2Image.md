---
description: Make an image from a prompt and steer its pose and layout with two guide images, a pose image and a depth map.
when_to_use: Use it when the composition matters and a prompt alone does not place things where you want them. Use Text2Image when you do not have guide images.
---

This workflow makes an image from a text description, called a prompt, and follows guide images while it does. A ControlNet is an add-on model that reads a guide image and steers the result to match it. The template uses two. One reads a pose image and one reads a depth map of a city street. It uses FLUX.1 dev. The prompt describes a woman with cybernetic enhancements standing in a neon street in cyberpunk Tokyo.

## What it connects

1. Two Load Image nodes supply the guide images. They sit in a group named Controlnet image setup.
1. Two [Load ControlNet](../nodes/ControlNetNode.md) nodes each set up one ControlNet with its model, guide image and strength. On the canvas they are named Configure ControlNet.
1. Two [Load Component](../nodes/LoadComponent.md) nodes supply the text encoders, which read your prompt. One is `openai/clip-vit-large-patch14` and one is `google/t5-v1_1-xxl`. See [What are a text encoder and a tokenizer?](../guides/what-is-a-text-encoder.md).
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model with those text encoders.
1. [Apply ControlNet](../nodes/ControlNetDiffusionPipelineBuilderNode.md) attaches both ControlNets to the pipeline. On the canvas it is named ControlNet Pipeline. Its `control_parameters` output also goes to Generate Latent.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the image size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt and a negative prompt from two Text Input nodes and runs 20 steps. The negative prompt says what to steer away from.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image you can see.

## Steps

1. Open the **ControlnetText2Image** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Check that the ControlNet model `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` is downloaded. Load ControlNet only lists models you have downloaded.
1. Check that the two text encoder repos are in your Hugging Face cache. Download them in Model Manager if they are not.
1. FLUX.1 dev is a gated model, so you must accept its terms on Hugging Face first. See [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. To use your own guide images, load them in the two Load Image nodes. Each image must match its mode. The pose ControlNet needs a pose image and the depth one needs a depth map.
1. Edit the prompt and the negative prompt in the Text Input nodes.
1. Run the workflow.

## Defaults

- Model: FLUX.1 dev (`black-forest-labs/FLUX.1-dev`), pipeline type `FluxPipeline`.
- ControlNet model: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` on both.
- Pose ControlNet: `control_mode` pose, strength 0.9, active from 0 to 0.5 of the steps.
- Depth ControlNet: `control_mode` depth, strength 0.5, active from 0 to 0.5 of the steps.
- Negative prompt: "3d, cartoon, anime, painting, blurry, bright specs, folds in the image".
- Guidance: 3.5.
- `true_cfg_scale`: 2.
- Size: 1024 x 1024.
- Steps: 20.
- Seed: 42.

## Tips

- `controlnet_conditioning_scale` is how strongly a guide image steers. Higher follows it more closely. See [controlnet_conditioning_scale](../nodes/ControlNetNode.md#controlnet_conditioning_scale).
- `control_guidance_start` and `control_guidance_end` set which part of the run the guide image applies to. Both ControlNets here stop at 0.5, so the second half of the steps fills in detail freely. See [control_guidance_start](../nodes/ControlNetNode.md#control_guidance_start) and [control_guidance_end](../nodes/ControlNetNode.md#control_guidance_end).
- Set a strength to 0 to turn that ControlNet off and see what the other one does alone.
- Apply ControlNet does not work with every pipeline type. See [Apply ControlNet](../nodes/ControlNetDiffusionPipelineBuilderNode.md) for the list.
- Add Noise is off on Generate Latent, so the seed that matters is the one on Create Noise Latent. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- To learn how the pieces fit, read [How pipelines work](../guides/how-pipelines-work.md).
