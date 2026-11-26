---
title: What is a latent?
description: The pattern behind every Diffusion Library workflow, and what a latent is.
---

Read this first if the words "latent", "noise" and "decode" in the node names are new to you.

## The idea

A diffusion model does not paint pixels directly. It works on a compact, compressed form of the picture or clip called a **latent**. A latent is not something you can look at. It is the model's own working format, a block of numbers that stands for an image or a video.

Making an image takes three moves:

1. **Start with a latent.** For a new image this is random noise. For an existing picture or clip, it is that picture encoded into a latent.
1. **Denoise it.** The model changes the latent step by step, guided by your prompt, until the noise has turned into something that matches.
1. **Decode it.** The result is still a latent. Decoding turns it back into pixels you can view and save.

Working in latent space is why the model can run on a normal GPU, and it is why the library splits the job into separate nodes.

## What it looks like

This run used Z-Image Turbo at 768 x 768 with 8 steps and seed 42. The prompt was "a lighthouse on a rocky coast at sunset, cinematic, detailed". After each step, the latent was decoded so you can see what the model has so far.

![The noise latent turns into a lighthouse over eight denoising steps](../assets/latent-denoising.webp)

The top row of the filmstrip shows the latent itself, with three of its channels drawn as colour. The bottom row shows the same latent decoded to pixels.

![Latent and decoded image at steps 0, 2, 4, 6 and 8](../assets/latent-filmstrip.webp)

Step 0 is pure noise. The large shapes appear first, the lighthouse and the horizon by step 4, and the fine detail and clean colour arrive in the last steps. In a workflow, Decode Media decodes only the final latent. The in-between pictures here are for illustration.

## The pattern in nodes

```text
Build Pipeline → Create Noise Latent → Generate Latent → Decode Media
   (the model)     (start)              (denoise)         (pixels)
```

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
- [Create Noise Latent](../nodes/NoiseLatentNode.md) makes the starting latent. The seed sets which noise you get, so the same seed, size and model give the same start.
- [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) runs the denoising steps with your prompt.
- [Decode Media](../nodes/VaeDecodeNode.md) turns the final latent into an image or video.

To start from your own picture or clip instead of noise, put [Encode Media](../nodes/VaeEncodeNode.md) in place of Create Noise Latent. Turn `add_noise` on in Generate Latent so the model has something to rework.

## Why this is useful

Because the latent is a separate step you can see and change, you can:

- Run Generate Latent more than once, feeding one result into the next, to refine an image.
- Paste one image into another before decoding with [Composite Latents with Mask](../nodes/LatentCompositeMaskNode.md), or combine latents with the add, subtract and multiply nodes.
- Enlarge a video latent to twice the width and height with [Upsample Latent](../nodes/LatentUpsamplerNode.md) before you decode it.
- Save a latent with [Save Latent Tensor](../nodes/SaveLatentTensorNode.md) to inspect it or reuse it later.

## Things to know

- A latent is shaped for one model. Create it from the pipeline you will run it through, and do not move a latent between different models.
- A latent has no preview. To see what one holds, decode it.
- Encoding and decoding go through the model's VAE. See [What is a VAE?](what-is-a-vae.md). An image that is encoded and decoded again can come back slightly different from the original.
