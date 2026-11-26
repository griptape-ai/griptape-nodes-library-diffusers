---
title: What is a latent?
description: The pattern behind every Diffusion Library workflow, and what a latent is.
---

A latent is the compressed form of a picture or clip that a diffusion model works on. It is a block of numbers that stands for the image. It is not a picture, so it has to be turned back into pixels before you can view it.

## The idea

A diffusion model does not paint pixels directly. It works on the latent, which is its own working format. The latent is far smaller than a full frame, so the model needs far less graphics card (GPU) memory than it would to work on pixels.

The library also gives each stage its own node: make the starting latent, denoise it, decode it. That lets you step in between the stages.

Making an image takes three moves:

1. **Start with a latent.** For a new image this is random noise, like static on a screen. The noise is random numbers, and a number called the seed fixes which ones you get. The same seed gives the same noise. For an existing picture or clip, the picture is encoded into a latent first. Noise is then mixed into it, because the model cleans up noise and needs some to rework. See [When noise is added](#when-noise-is-added).
1. **Clean it up step by step.** The model changes the latent in a series of steps, guided by your prompt, removing noise a little at a time until what is left matches the prompt. This is called denoising.
1. **Decode it.** The result is still a latent. Decoding turns it back into pixels you can view and save.

## What it looks like

This run used Z-Image Turbo at 768 x 768 with 8 steps and seed 42. A step is one round of clean-up. Z-Image Turbo is a fast model, so 8 steps are enough for this demo. The Text2Image and Image2Image workflows use 20. The prompt was "a lighthouse on a rocky coast at sunset, cinematic, detailed". After each step, the latent was decoded so you can see what the model has so far.

![The noise latent turns into a lighthouse over eight denoising steps](../assets/latent-denoising.webp)

A latent has many layers, called channels. These are not the red, green, blue and alpha channels of an image. The top row of the filmstrip shows only three of the latent's channels, mapped to red, green and blue. That gives a rough view and is not a real picture. The bottom row shows the same latent decoded to pixels.

![Latent and decoded image at steps 0, 2, 4, 6 and 8](../assets/latent-filmstrip.webp)

Step 0 is pure noise. The large shapes appear first, the lighthouse and the horizon by step 4, and the fine detail and clean colour arrive in the last steps. In a workflow, Decode Media decodes only the final latent. The in-between pictures here are for illustration.

## The pattern in nodes

```text
Build Pipeline → Create Noise Latent → Generate Latent → Decode Media
   (the model)     (start)              (denoise)         (pixels)
```

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
- [Create Noise Latent](../nodes/NoiseLatentNode.md) makes the starting latent. The seed sets which noise you get, so the same seed, size and model give the same start.
- [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) runs the denoising steps with your prompt, the text description of what you want.
- [Decode Media](../nodes/VaeDecodeNode.md) turns the final latent into an image or video.

To start from your own picture or clip instead of noise, put [Encode Media](../nodes/VaeEncodeNode.md) in place of Create Noise Latent. Turn `add_noise` on in Generate Latent. `add_noise` mixes noise into your picture, and how much depends on `start_step`. See [When noise is added](#when-noise-is-added).

## When noise is added

The `add_noise` setting on [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) is off by default. When it is on, noise is mixed into the input latent before denoising. A higher `start_step` mixes in less noise and keeps more of the input. A `start_step` of 0 mixes in the most. When it is off, denoising starts from the input exactly as it is. With `add_noise` on, an `end_step` of 0 skips denoising and gives only noise.

Set it by what you connect to Generate Latent:

- [Create Noise Latent](../nodes/NoiseLatentNode.md): leave it off. The input is already noise.
- [Create Empty Latent](../nodes/EmptyLatentNode.md): turn it on. A blank latent has nothing to denoise from.
- [Encode Media](../nodes/VaeEncodeNode.md), or the output of an earlier Generate Latent: turn it on. This is image-to-image and refinement.
- [Set Latent Mask](../nodes/SetLatentMaskNode.md): turn it on. With it off, denoising starts from the clean source image.
- [Encode Masked Media](../nodes/VaeMaskEncodeNode.md), for inpainting: the setting is hidden. Generate Latent handles the noise itself.

## Why this is useful

Because the latent is a separate stage you can step into, you can:

- Run Generate Latent more than once, feeding one result into the next, to refine an image.
- Paste one image into another before decoding with [Composite Latents with Mask](../nodes/LatentCompositeMaskNode.md), or combine latents with the add, subtract and multiply nodes.
- Enlarge a video latent to twice the width and height with [Upsample Latent](../nodes/LatentUpsamplerNode.md) before you decode it.
- Save a latent with [Save Latent](../nodes/SaveLatentTensorNode.md) to inspect it or reuse it later.

## Things to know

- A latent is shaped for one model. Create it from the pipeline you will run it through, and do not move a latent between different models.
- A latent has no real preview. To see what one holds, decode it.
- Encoding and decoding go through the model's VAE, a small companion model that translates between pixels and latents. See [What is a VAE?](what-is-a-vae.md). An image that is encoded and decoded again can come back slightly different from the original.
