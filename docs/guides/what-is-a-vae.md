---
title: What is a VAE?
description: What the VAE does in a diffusion workflow, and where it shows up in the nodes.
---

Read [What is a latent?](what-is-a-latent.md) first. The VAE is the part that moves pictures into latents and back.

## The idea

VAE stands for variational autoencoder. It is a small model that comes with each diffusion model, and it has two halves:

- The **encoder** shrinks an image or a video into a latent.
- The **decoder** expands a latent back into pixels.

The VAE does not make anything up from your prompt. The diffusion model does that work in latent space. The VAE only translates between pixels and latents, like a codec that the diffusion model speaks through.

```text
image or video ──encoder──▶ latent ──(diffusion)──▶ latent ──decoder──▶ image or video
```

A text-to-image run uses only the decoder. It starts from noise, so there is nothing to encode. A run that starts from your own picture or clip uses the encoder as well.

## Where it shows up in the nodes

- [Decode Media](../nodes/VaeDecodeNode.md) runs the decoder on a finished latent. [Convert Latent to Media](../nodes/LatentToMediaNode.md) does the same without a pipeline, and [Decode HDR Media](../nodes/DecodeHdrNode.md) gives float frames for EXR export.
- [Encode Media](../nodes/VaeEncodeNode.md) runs the encoder on an image or a video. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) does the same for an image and a mask, for inpainting.
- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the VAE with the rest of the model. Use [Load Component](../nodes/LoadComponent.md) with the slot set to `VAE` to swap in a different one.
- `vae_slicing` and `vae_tiling` in the [Memory settings](build-pipeline-by-model.md#memory-settings) make the decoder use less memory.

## Things to know

- The VAE is matched to its model. A latent made for one model is decoded by that model's VAE. A replacement VAE has to produce the same channel count, or the run stops. A different scaling shows as a warning only, and the colours or contrast of the result may be off.
- Decoding can be where a run runs out of memory, because the decoder expands a small latent into full-size frames. Turn on `vae_slicing` or `vae_tiling` if it happens, most often with large images or long videos.
- The VAE sets the limit on fine detail. Very small text, thin lines and fine texture can soften, even when the diffusion step got them right.
- Encoding then decoding an image gives a very close copy, not an exact one.
