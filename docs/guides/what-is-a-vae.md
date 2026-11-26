---
title: What is a VAE?
description: What the VAE does in a diffusion workflow, and where it shows up in the nodes.
---

The VAE is the part that moves pictures into latents and back. It builds on [What is a latent?](what-is-a-latent.md).

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

## Why swap the VAE

Most workflows should keep the VAE that comes with the model. People swap it for a few reasons:

- **Color and contrast.** Some base VAEs give washed-out color, banding or odd saturation. A fixed VAE can correct that.
- **Fine detail.** A VAE tuned on faces, text or texture can hold small detail better than the stock one.
- **Memory.** A lighter VAE decodes with less memory, which helps with long video or large images.
- **Speed.** A small or distilled VAE decodes faster, at some cost in quality. This suits quick previews.
- **Matching a fine-tune.** A checkpoint that was trained with its own VAE should be decoded with that VAE.

To swap it, connect a [Load Component](../nodes/LoadComponent.md) node with the slot set to `VAE` to the VAE port in Build Pipeline's Component Overrides. The replacement has to match the model's latent format, as described below. If the result looks washed out or too contrasty after a swap, the scaling is the first thing to check.

## Things to know

- The VAE is matched to its model. A latent made for one model is decoded by that model's VAE. A replacement VAE has to produce the same channel count, or the run stops. A different scaling shows as a warning only, and the colours or contrast of the result may be off.
- Decoding can be where a run runs out of memory, because the decoder expands a small latent into full-size frames. Turn on `vae_slicing` or `vae_tiling` if it happens, most often with large images or long videos.
- The VAE sets the limit on fine detail. Very small text, thin lines and fine texture can soften, even when the diffusion step got them right.
- Encoding then decoding an image gives a very close copy, not an exact one.

Related: [What are a transformer and a UNet?](what-is-a-transformer.md), [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md), [What is a scheduler?](what-is-a-scheduler.md).
