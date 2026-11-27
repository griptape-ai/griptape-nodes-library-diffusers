---
description: Creates a blank latent, the model's compressed working form of a picture, filled with zeros and sized for the connected pipeline.
when_to_use: Use it when you need a blank latent to composite into or to add controlled noise to, rather than a random starting point.
---

## When to use it

This node makes a blank [latent](../guides/what-is-a-latent.md). A latent is the compressed form of a picture or clip that the model works on. Here every value is zero, and the size fits the connected pipeline.

Use it when you want a blank base. It works well as the base that [Composite Latents with Mask](LatentCompositeMaskNode.md) pastes other latents onto. For ordinary text-to-image or text-to-video, use [Create Noise Latent](NoiseLatentNode.md) instead.

## What it needs

- A pipeline, which sets the latent's shape and whether `num_frames` is shown.
- Noise added before denoising. A latent of zeros has nothing to denoise from, so set `add_noise` to on in [Generate Latent](DiffusionPipelineGenerateLatentNode.md), or add noise with the composite and math nodes first.

## Where it fits

```text
Build Pipeline → [Create Empty Latent] → Composite Latents with Mask → Generate Latent
```

## Preview

![Create Empty Latent node with width, height and frame count settings](../assets/nodes/create-empty-latents.png)

## Parameters

### pipeline

Connect the pipeline you will generate with. The latent only works with that kind of pipeline.

### width

Width in pixels. The default is 1024. It must be at least 1. Some pipelines need a width that is a multiple of a set number. If yours does and the width does not fit, the run stops with a message. Turn on **Auto Resize** under Library Settings > Diffusion Library to round it up automatically.

### height

Height in pixels. The default is 1024. It must be at least 1. Some pipelines need a height that is a multiple of a set number. If yours does and the height does not fit, the run stops with a message. Turn on **Auto Resize** under Library Settings > Diffusion Library to round it up automatically.

### num_frames

Video pipelines only. The number of frames, default 41. It must be at least 1, and some pipelines need specific counts, which Auto Resize also adjusts. Image pipelines hide this setting.

### output_latent

A latent of zeros. There is no seed, so the result is always the same.

## See also

- [Create Noise Latent](NoiseLatentNode.md)
- [Composite Latents with Mask](LatentCompositeMaskNode.md)
