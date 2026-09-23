# Release Pipeline

**Releases one connected diffusion pipeline from memory, leaving every other cached pipeline loaded.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Frees the memory held by the single pipeline wired into it, not the whole cache.
- Use it to make room for the next pipeline in a workflow that builds more than one.
- The pipeline configuration is untouched, so a later node that needs it rebuilds it.
- Releasing a pipeline that is not loaded succeeds and reports that nothing was released.

## Typical workflow position
```text
Pipeline Builder (SDXL) → Generate Media Latents → [Release Pipeline] → Pipeline Builder (Flux)
```

## Node preview

<!-- TODO: add docs/assets/nodes/release-pipeline.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `pipeline` | `Pipeline Config` | Yes | The pipeline to release, from a Pipeline Builder, LoRA Pipeline, or ControlNet Pipeline node. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `was_successful` | `bool` | `True` whenever the pipeline is no longer held, including when it was already gone. |
| `result_details` | `str` | Either `Released '<PipelineName>' and freed the memory it held.` or `'<PipelineName>' was not being held. Nothing to release.` |

## Tips & pitfalls

- **Releasing is not destructive.** Only the loaded pipeline goes; the configuration travels on the connection. A downstream node that needs this pipeline again rebuilds it, which costs the load time again.
- **Derived pipelines are separate.** A LoRA or ControlNet pipeline built from this one has its own configuration and its own cache entry. Releasing the base does not release the derived pipeline, and a derived pipeline still holds the components it shares with the base — so release the derived one too if you want the memory back.
- **A missing pipeline is not an error.** The node reports success so it can sit in a workflow that runs repeatedly without failing on the second pass.
- **Reach for [Clear Pipeline Cache](clear_pipeline_cache.md) instead** when you want a clean slate rather than targeted eviction.

## See also

- [Clear Pipeline Cache](clear_pipeline_cache.md)
- [Modular Diffusion Pipeline Builder](pipeline_builder.md)
- [LoRA Pipeline](lora_pipeline.md)
- [ControlNet Pipeline](controlnet_pipeline.md)
