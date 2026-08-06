# Clear Pipeline Cache

**Evicts all cached diffusion pipelines so the next build starts from a clean cache state.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Clears every pipeline currently stored in the in-memory model cache.
- Useful when troubleshooting out-of-memory issues or forcing a fresh pipeline load.
- This node does not take pipeline input; it operates on the global cache.
- Status outputs report whether clearing succeeded and how many pipelines were removed.

## Typical workflow position
```text
Pipeline Builder(s) → [Clear Pipeline Cache] → Pipeline Builder
```

## Node preview

<!-- TODO: add docs/assets/nodes/clear-pipeline-cache.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| None | N/A | No | This node has no configurable inputs. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `was_successful` | `bool` | `True` when cache clearing completes without exception. |
| `result_details` | `str` | Includes the number of evicted pipelines, for example `Cleared 2 pipeline(s) from cache.` |

## Tips & pitfalls

- **Place it only where a full cache reset is intentional.** It clears every cached pipeline, not a single selected model.
- **Expect the next generation to rebuild.** The next Pipeline Builder execution after clearing has to load/build the model again.

## See also

- [Modular Diffusion Pipeline Builder](pipeline_builder.md)
- [Generate Media Latents](generate_media_latents.md)
- [LoRA Pipeline](lora_pipeline.md)