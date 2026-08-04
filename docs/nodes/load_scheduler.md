# Load Scheduler

**Configure a scheduler override by choosing a scheduler class and a config source.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Schedulers are config-only (no weights): pick a **Scheduler Class** and a **Source Type**, and the node emits a `SchedulerComponentArtifact`.
- Three source modes: a local `scheduler_config.json` file, a HuggingFace repo id already in your local cache, or raw JSON-formatted text typed or pasted directly into the node (**Text Config** mode).
- Wire the output into a Pipeline Builder's `scheduler` override port to replace the pipeline's default scheduler at build time.
- A compatibility warning badge appears when the chosen scheduler family does not match the config (All modes), or when the config contains keys the chosen class will ignore (Text Config mode).

## Typical workflow position

```text
[Load Scheduler] → scheduler → [Pipeline Builder] → [Generate Media Latents]
```

## Node preview

<!-- TODO: add docs/assets/nodes/load-scheduler.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `Scheduler Config` | `json \| str \| dict` | No | Text Config mode only. Wire from a **JSON Input**, **Create Dictionary**, or **Merge Key Value Pairs** node. Accepts a dict directly or a JSON-formatted string. Leave the field empty to use the scheduler class's built-in defaults. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `component_output` | `SchedulerComponentArtifact` | Scheduler descriptor. Wire into the `scheduler` override slot on a Pipeline Builder. |

## Parameters

### Common

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `Scheduler Class` | dropdown (searchable) | `FlowMatchEulerDiscreteScheduler` | Scheduler class to instantiate. Flow-matching family (`FlowMatchEulerDiscreteScheduler`, `FlowMatchHeunDiscreteScheduler`) drives Flux, SD3, Qwen, Wan, and LTX pipelines. Classic family (`EulerDiscreteScheduler`, `DDIMScheduler`, `DPMSolverMultistepScheduler`, `UniPCMultistepScheduler`, etc.) drives SDXL. |
| `Source Type` | `Local Path \| HuggingFace Repo \| Text Config` | `Local Path` | Selects which parameter group is active. The other groups are hidden. |

### Local Path

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `Config Path` | `str` (file or dir) | `""` | Path to a `scheduler_config.json` file, or a folder containing one (e.g. `.../FLUX.1-dev/scheduler/`). |

### HuggingFace Repo

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `Repo ID` | `str` | `""` | HuggingFace repo id, e.g. `black-forest-labs/FLUX.1-dev`. Must already be in the local HF cache; no downloads are triggered. A refresh (↺) button next to the field re-checks the local cache. If the repo is not cached, an **Open Model Manager to Download** button appears — clicking it opens Model Management pre-filtered to that repo. |
| `Revision` | `str` | `"main"` | Repo revision (branch, tag, or commit hash). |
| `Subfolder (Optional)` | `str` | `""` | Subfolder within the repo containing `scheduler_config.json`. Leave blank to use the default `scheduler` subfolder. |

### Text Config

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `Scheduler Config` | `json \| str \| dict` | `null` | Paste the contents of a `scheduler_config.json` directly, or wire from a **JSON Input**, **Create Dictionary**, or **Merge Key Value Pairs** node. The value is auto-formatted to pretty-printed JSON after each edit. Leave empty to instantiate the chosen scheduler class with its built-in defaults. |

## Tips & pitfalls

- **Match the scheduler family to your pipeline.** Flow-matching schedulers (`FlowMatch*`) are for Flux, SD3, Qwen, Wan, and LTX. Classic schedulers (`Euler`, `DDIM`, `DPMSolver`, `UniPC`, etc.) are for SDXL. A cross-family scheduler will produce wrong output or error at generation time.
- **Point at the scheduler subfolder, not the pipeline root.** For Local Path, `.../FLUX.1-dev/` is wrong; `.../FLUX.1-dev/scheduler/` (or its `scheduler_config.json`) is right.
- **HuggingFace Repo mode is cache-only.** If the repo id you enter is not in your local HF cache, use the **Open Model Manager to Download** button that appears below the field, or click the refresh (↺) button after downloading to re-check.
- **Text Config: unknown keys are warned, not errored.** If your JSON contains keys the chosen scheduler class doesn't recognise, a warning badge lists them. The scheduler still loads — those keys are silently dropped by diffusers. Remove them to suppress the warning.
- **Text Config: empty field uses class defaults.** Leaving **Scheduler Config** empty is valid — the chosen scheduler class is instantiated with its own built-in defaults, with no config file needed.

## See also

[Modular Diffusion Pipeline Builder](pipeline_builder.md) · [Load Pipeline Component](load_component.md)
