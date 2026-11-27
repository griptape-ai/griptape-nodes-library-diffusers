---
description: Adds two latents, the model's compressed working form of a picture or clip, together value by value and outputs the sum.
when_to_use: Use it when you want to combine two latents of the same size, such as blending two results or adding noise to another latent.
---

## When to use it

This node adds two [latent](../guides/what-is-a-latent.md) together and gives you the sum as a new latent. A latent is the compressed form of a picture or clip that the model works on. You can't view one directly. Each number in one latent is added to the matching number in the other.

Use it to blend two results or to add noise to a latent.

Both latents must be connected and must be the same size. Latents from different model families, such as Flux and SDXL, normally differ in size. If the sizes happen to match, the sum is still meaningless.

## Where it fits

```text
Generate Latent (A) ─┐
                     ├─→ [Add Latents] → Generate Latent / Decode Media
Generate Latent (B) ─┘
```

To view the sum, connect it to [Decode Media](VaeDecodeNode.md) with the pipeline that made the latents. [Convert Latent to Media](LatentToMediaNode.md) only accepts compatible LTX-2 video latents.

## Preview

![Add Latents node with two latent inputs and one latent output](../assets/nodes/add-latents.png)

## Parameters

### left_latent

The first latent.

### right_latent

The second latent. It must have the same shape as `left_latent`. If the shapes differ, the run stops with a message that shows both shapes. It is converted to the precision of `left_latent` before the sum. Resize or upsample one of them first, for example with [Upsample Latent](LatentUpsamplerNode.md).

### output_latent

The sum of `left_latent` and `right_latent`. The output keeps the size information of `left_latent`. Metadata from both inputs is merged, and `left_latent` wins when both have the same key.

## See also

- [Subtract Latents](SubtractLatentsNode.md)
- [Multiply Latents](MultiplyLatentsNode.md)
- [Composite Latents with Mask](LatentCompositeMaskNode.md)
