# Load Scheduler

**Configure a scheduler override by choosing a scheduler class and pointing at a `scheduler_config.json` source.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Schedulers are config-only (no weights): pick a scheduler class and a config source, and the node emits a `SchedulerComponentArtifact`.
- Two config source modes: a local `scheduler_config.json` file or folder, or a HuggingFace repo id already in your local cache.
- Wire the output into a Pipeline Builder's `scheduler` override port to replace the pipeline's default scheduler at build time.
- A compatibility warning appears when the chosen scheduler family (flow-matching vs classic) does not match the config's recorded class.

## Typical workflow position
```text
[Load Scheduler] → scheduler → [Pipeline Builder] → [Generate Media Latents]
```

## Node preview

<!-- TODO: add docs/assets/nodes/load_scheduler.png screenshot -->

## Inputs

_None. All fields are node properties._

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `component_output` | `SchedulerComponentArtifact` | Scheduler descriptor. Wire into the `scheduler` override slot on the Pipeline Builder. |

## Parameters

### Common

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `scheduler_class` | dropdown (searchable) | `FlowMatchEulerDiscreteScheduler` | Scheduler class to instantiate. Flow-matching family (`FlowMatchEulerDiscreteScheduler`, `FlowMatchHeunDiscreteScheduler`) drives Flux, SD3, Qwen, Wan, and LTX pipelines. Classic family (`EulerDiscreteScheduler`, `DDIMScheduler`, `DPMSolverMultistepScheduler`, `UniPCMultistepScheduler`, etc.) drives SDXL. |
| `config_source_type` | `Local Path \| HuggingFace Repo` | `Local Path` | Selects which parameter group below is active. The other group is hidden. |

### Local Path

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `config_path` | `str` (file or dir) | `""` | Path to a `scheduler_config.json` file, or a folder containing one (e.g. `.../FLUX.1-dev/scheduler/`). |

### HuggingFace Repo

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `repo_id` | `str` | `""` | HuggingFace repo id, e.g. `black-forest-labs/FLUX.1-dev`. Must already be in the local HF cache; no downloads are triggered. A refresh (↺) button next to the field re-checks the local cache. If the repo is not in the local cache, an **Open Model Manager to Download** button appears below the field — clicking it opens Model Management pre-filtered to that repo. Run-time validation fails if the repo is not cached. |
| `revision` | `str` | `"main"` | Repo revision (branch, tag, or commit hash). |
| `subfolder` | `str` | `""` | Subfolder within the repo containing `scheduler_config.json`. Blank defaults to `scheduler`. |

## Tips & pitfalls

- **Match the scheduler family to your pipeline.** Flow-matching schedulers (`FlowMatch*`) are for Flux, SD3, Qwen, Wan, and LTX. Classic schedulers (`Euler`, `DDIM`, `DPMSolver`, `UniPC`, etc.) are for SDXL. A cross-family scheduler will produce wrong output or error at generation time.
- **Point at the scheduler subfolder, not the pipeline root.** For Local Path, `.../FLUX.1-dev/` is wrong; `.../FLUX.1-dev/scheduler/` (or its `scheduler_config.json`) is right.
- **HuggingFace Repo mode is cache-only.** If the repo id you enter is not in your local HF cache, use the **Open Model Manager to Download** button that appears below the field, or click the refresh (↺) button after downloading to re-check.

## See also

[Modular Diffusion Pipeline Builder](pipeline_builder.md) . [Load Pipeline Component](load_component.md)
