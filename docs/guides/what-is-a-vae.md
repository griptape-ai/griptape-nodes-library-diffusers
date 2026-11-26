---
title: What is a VAE?
description: What the VAE does in a diffusion workflow, and where it shows up in the nodes.
---

The VAE is the part that translates between pictures and the compressed form the model works in. It acts like a codec: it encodes pixels into that form and decodes them back. That compressed form is called a [latent](what-is-a-latent.md). A latent is a block of numbers that stands for an image or video. Like a compressed video file, a latent cannot be viewed until it is decoded.

## The idea

VAE stands for variational autoencoder. It is a small model that comes with each diffusion model, and it has two halves:

- The **encoder** shrinks an image or a video into a latent.
- The **decoder** expands a latent back into pixels.

The VAE does not make anything up from your prompt, the text description you give. The diffusion model does that work on the latent. The VAE only translates.

```text
image or video ──encoder──▶ latent ──(diffusion)──▶ latent ──decoder──▶ image or video
```

A text-to-image run uses only the decoder. It starts from random noise, so there is nothing to encode. A run that starts from your own picture or clip uses the encoder as well.

## Where it shows up in the nodes

- [Decode Media](../nodes/VaeDecodeNode.md) runs the decoder on a finished latent. [Convert Latent to Media](../nodes/LatentToMediaNode.md) does the same without a pipeline, and [Decode HDR Media](../nodes/DecodeHdrNode.md) gives float frames for EXR export.
- [Encode Media](../nodes/VaeEncodeNode.md) runs the encoder on an image or a video. [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) does the same for an image and a mask, for inpainting (repainting only the masked area).
- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the VAE with the rest of the model. Use [Load Component](../nodes/LoadComponent.md) with `component` set to `VAE` to swap in a different one.
- `vae_slicing` and `vae_tiling` in the [Memory settings](build-pipeline-by-model.md#memory-settings) make the decoder use less memory.

## Why swap the VAE

Most workflows should keep the VAE that comes with the model. People swap it for a few reasons:

- **Color and contrast.** Some base VAEs give washed-out color, banding or odd saturation. A corrected VAE, usually released by the community, can fix that.
- **Fine detail.** A VAE tuned on faces, text or texture can hold small detail better than the stock one.
- **Memory.** A lighter VAE decodes with less memory, which helps with long video or large images.
- **Speed.** A smaller, simplified VAE decodes faster, at some cost in quality. This suits quick previews.
- **Matching a fine-tune.** A fine-tune is a community retrain of a base model. If it was trained with its own VAE, decode with that VAE. A checkpoint is a saved set of model weights, which are the numbers a model learned in training.

To swap it, connect a [Load Component](../nodes/LoadComponent.md) node with `component` set to `VAE` to the VAE port in Build Pipeline's Component Overrides. The replacement has to match the model's latent format, which has two parts:

- **Channel count** is the number of layers in the latent. A replacement with a different count cannot work, and the run stops.
- **Scaling** is the factor (and optional shift) that rescales the latent's numbers to the size the diffusion model was trained on. A different scaling only shows a warning, but the colors and contrast can come out wrong.

If the result looks washed out or too contrasty after a swap, check the scaling first.

## Things to know

- The VAE is matched to its model. A latent made for one model is decoded by that model's VAE. A replacement VAE needs the same channel count as the original, or the run stops. A different scaling shows a warning, and the run continues.
- Decoding can be where a run runs out of GPU memory, because the decoder expands a small latent into full-size frames. This is most likely with large images or long videos. The two settings listed above help.
- The VAE sets the limit on fine detail. Very small text, thin lines and fine texture can soften, even when the diffusion step got them right.
- Encoding then decoding an image gives a very close copy, not an exact one.

Related: [What are a transformer and a UNet?](what-is-a-transformer.md), [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md), [What is a scheduler?](what-is-a-scheduler.md).
