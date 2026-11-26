---
description: Unloads every model held in memory so the next generation starts clean.
when_to_use: Use it when a run fails with an out-of-memory error, or before you load a different model.
---

## When to use it

This node unloads the models the engine is holding in memory. A loaded model, together with its supporting parts, is called a pipeline.

Use it when a run fails with an out-of-memory error, or when you are about to load a different model and want the old one gone. It releases every pipeline the running engine still holds, not one chosen model. Building a pipeline that is not already in memory also releases the others first, so only one pipeline is normally held at a time.

The node needs no inputs.

## Where it fits

```text
Build Pipeline → [Clear Pipeline Cache] → Build Pipeline
```

## Parameters

### was_successful

True when the cache was cleared without an error.

### result_details

A short message with the number of pipelines released, for example `Cleared 2 pipeline(s) from cache.`. If clearing fails, it holds the error text.

## Tips

- A cleared pipeline is rebuilt the next time something needs it, so the next generation starts with a model load.
- Place it only where a full reset is intended.

## See also

- [Build Pipeline](LatentDiffusionPipelineBuilderNode.md)
- [Generate Latent](DiffusionPipelineGenerateLatentNode.md)
- [Apply Runtime LoRA](LoraActivationPipelineNode.md)
