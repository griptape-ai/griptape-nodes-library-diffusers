# Estimate Pipeline Memory

**Estimates per-component GPU memory usage for a diffusion pipeline, built or not, without running it.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Breaks down memory into weights and activations for each resident component (text encoders, transformer/UNet, VAE).
- Folds unfused LoRA adapter weights from a connected LoRA Pipeline into the transformer/UNet estimate; the widget does not add a separate LoRA component row.
- Uses the latent's shape to size the denoiser's and VAE's activation-memory estimate, so it must be wired to a real latent.
- Works whether or not the pipeline has been built yet: if it's already loaded (e.g. after a Generate Media Latents node has run) the estimate is exact; otherwise it's derived from the pipeline's config alone. This node never triggers a pipeline build itself.
- Estimates are analytical, biased to overestimate, and target ~20% relative accuracy — not measured, exact byte counts.

## Typical workflow position

```text
Pipeline Builder → Generate Media Latents → [Estimate Pipeline Memory]
```

## Node preview

<img src="../assets/nodes/estimate-pipeline-memory.png" alt="Estimate Pipeline Memory" width="480">

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `pipeline` | `Pipeline Config` | Yes | Connect from Pipeline Builder. Does not need to be built/resident yet — see Provider / model behavior. |
| `latent` | `LatentArtifact` | Yes | Determines the token count / resolution used in the activation-memory estimate. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `memory_breakdown` | `dict` | Read-only visual breakdown of component totals, weights, activations, warnings, and the separate estimated peak value. |
| `was_successful` | `bool` | `True` when the estimate completed without exception. |
| `result_details` | `str` | Summary line with the estimated peak and component count. |

## Provider / model behavior

Denoiser activation memory is estimated using one of three architecture families: UNet-SDPA (SDXL), Image-DiT-SDPA (Flux, Flux2, SD3, Qwen, Z-Image, ...), or Video-DiT-joint-SDPA (LTX, LTX2, WAN, HunyuanVideo 1.5, MiniMax-H3). A pipeline class without a registered family, or a component whose config can't be read (e.g. a heavily fused/custom checkpoint), falls back to a weight-only estimate with a warning in the widget rather than failing.

**Unfused LoRA.** When the connected pipeline comes from a [LoRA Pipeline](lora_pipeline.md) node, each distinct runtime adapter file is counted once and its header-derived weight bytes are folded into the primary transformer/UNet estimate. The transformer hover text reports the total LoRA bytes as both a percentage of the base transformer weights and a percentage of the combined transformer-plus-LoRA weights. LoRA activation workspace is not separately modeled. The source-only CLI keeps the detailed adapter list in a separate `lora_adapters` section.

The `memory_breakdown` output renders the estimate as a segmented bar and a labeled component list inside the node. Segment widths use the sum of component totals; the estimated peak is shown separately because it applies offload topology, non-overlapping activation phases, and safety headroom. The widget remains read-only and reports loading, unavailable, and warning states without changing the estimate.

**ControlNet (Flux, SD3, SDXL, Qwen, Z-Image).** When the connected pipeline is wrapped by a [ControlNet Pipeline](controlnet_pipeline.md) node, each stacked ControlNet appears as its own `controlnet_0`, `controlnet_1`, … component — weight bytes scale ~linearly with stack size, since each is an independent weight set regardless of the multi-ControlNet wrapper class used under the hood. Weight bytes are read from each ControlNet's own config (its own repo, not the base pipeline's), so a ControlNet's shape is never assumed to match the base denoiser's. Activation bytes reuse the base architecture family's per-token formula against the ControlNet's own (usually smaller) layer count — an approximation, always flagged `[ESTIMATED: ...]`. On SDXL specifically, this reuses the full UNet-SDPA level formula against the real ControlNetModel's down-only architecture (no up-blocks), which overestimates on purpose per this node's bias-toward-overestimate stance. Before the pipeline is built, each ControlNet's config must already be in the warm HuggingFace cache — otherwise that ControlNet's entry falls back to weights-only with a warning.

This node estimates from whichever of two states the pipeline is actually in:

| Pipeline state | Basis | Weight memory | Offload topology |
| --- | --- | --- | --- |
| Already built and resident in the model cache | Exact | Read directly off the resident tensors — reflects the pipeline's actual loaded dtype and quantization. | Detected from the pipeline's live CPU-offload hooks. |
| Not built yet | Config-only | Derived from each component's cached config, built on the meta device (no download, no weight data read) at the dtype/quantization the pipeline *would* load with. | Taken from the requested `cpu_offload_strategy`, **unless** `memory_optimization_strategy` is `Automatic` (see below). |

VAE activation memory reflects whether `vae_tiling`/`vae_slicing` are configured, in both states.

**`memory_optimization_strategy` = `Automatic`, pipeline not yet built:** the real Automatic cascade depends on free VRAM at build time, which this node has no way to predict before loading — it does not simulate the cascade. Instead it reports a conservative upper bound (no offload assumed) and adds a warning to the widget recommending `Manual` for an exact pre-load number. Once the pipeline is actually built, re-running this node picks up whatever topology Automatic really landed on, exactly.

## Command-line workflow estimation

The same estimator can be run against a generated workflow `.py` file without starting Griptape Nodes or running any workflow nodes:

```text
uv run estimate-workflow-memory workflows/templates/Text2Image.py
```

Use `--json` when the result will be consumed by another tool:

```text
uv run estimate-workflow-memory workflows/templates/Text2Image.py --json
```

The command parses the workflow source, extracts pipeline configurations and dimensions associated with recognized noise nodes, resolves the public latent shape through the cached Diffusers prepare-latents path, and calls the existing memory estimator. It does not call `build_workflow()`, build the full diffusion pipeline, load model weights, or run inference. The required component configuration files must already be available in the local Hugging Face cache; if they are missing, the command fails instead of falling back to pixel-space dimensions.

### Command-line arguments

| Argument | Purpose |
| --- | --- |
| `workflow` | Path to the generated workflow `.py` file to inspect. The file is parsed as source and is never imported or executed. |
| `--width WIDTH` | Override the raw width used for every extracted builder. Must be supplied together with `--height`. |
| `--height HEIGHT` | Override the raw height used for every extracted builder. Must be supplied together with `--width`. |
| `--num-frames NUM_FRAMES` | Override the raw video frame count used for every extracted builder. Required when an extracted video builder has no static frame count. It is ignored for image latent shapes and is not reported for image builders. |
| `--json` | Print machine-readable JSON instead of the human-readable report. |

For example, provide dimensions when the workflow does not contain static dimensions on a recognized noise node:

```text
uv run estimate-workflow-memory workflow.py --width 1024 --height 1024
```

For a video workflow, provide the frame count as well:

```text
uv run estimate-workflow-memory workflow.py --width 832 --height 480 --num-frames 25 --json
```

The command uses recorded or overridden dimensions as-is. It does not apply model-specific dimension snapping or validation, so the result includes a raw-dimension approximation warning. It still resolves latent-space shape through the cached Diffusers prepare-latents path, preserving the distinction between pixel-space source dimensions and public latent dimensions.

The JSON result contains a top-level `warnings` list and a `builders` list. Each builder contains its pipeline name, reported dimensions, and memory estimate. Image builders omit hidden `num_frames` placeholder values; video builders include `num_frames`. `Builder 1`, `Builder 2`, and so on identify the order of extracted estimates, not necessarily workflow execution order. Overrides apply to every extracted builder.

## API reference

This node is a thin wrapper — the estimate itself is a plain importable function, callable from any other node or script without going through the node at all.

```python
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import (
    estimate_pipeline_memory_from_artifact,
)

estimate = estimate_pipeline_memory_from_artifact(pipeline_artifact, latent_artifact)
```

Note the import path: `estimate_pipeline_memory_from_artifact` is exported from the `pipeline_memory_estimator` submodule, not from `modular_diffusion_nodes_library.memory_estimation`'s top-level `__init__.py` (which only re-exports `estimate_pipeline_memory` and `estimate_component_memory`).

### `estimate_pipeline_memory_from_artifact(artifact, latent) -> PipelineMemoryEstimate`

| Parameter | Type | Notes |
| --- | --- | --- |
| `artifact` | `DiffusionPipelineArtifact` | Same object as this node's `pipeline` input. |
| `latent` | `LatentArtifact` | Same object as this node's `latent` input. |

`PipelineMemoryEstimate` fields:

| Field | Type | Notes |
| --- | --- | --- |
| `pipeline_name` | `str` | |
| `offload_mode` | `str \| None` | `"model"`, `"sequential"`, or `None`. |
| `estimated_peak_bytes` | `int` | Estimated peak memory after the estimator's 20% safety-headroom factor; the node displays the value with the applied optimization settings. |
| `components` | `list[ComponentMemoryEstimate]` | One entry per weight-bearing component. |
| `warnings` | `list[str]` | Pipeline-level caveats, e.g. the `Automatic`-strategy warning. |
| `lora_adapters` | `list[dict]` | Detailed distinct unfused runtime adapters and their header-derived weight bytes; exposed separately for API and CLI consumers. |
| `to_dict()` | `dict` | JSON-shaped, GB-rounded — the intended API output boundary. Fields on the dataclass itself stay byte-precise for further math. |

`ComponentMemoryEstimate` fields (each entry in `components`):

| Field | Type | Notes |
| --- | --- | --- |
| `component_name` | `str` | e.g. `"transformer"`, `"vae"`, `"text_encoder_2"`. |
| `role` | `str` | `"denoiser"`, `"vae"`, `"text_encoder"`, `"controlnet"`, or `"other"`. |
| `weight_bytes` | `int` | |
| `activation_bytes` | `int` | |
| `total_bytes` | `int` | `weight_bytes + activation_bytes`. |
| `is_estimated` | `bool` | `True` when a formula couldn't run and only weights are shown for this component. |
| `warning` | `str \| None` | Reason, present when `is_estimated` is `True`. |
| `tooltip` | `str \| None` | Optional hover text, including the transformer and LoRA percentage details when runtime adapters are present. |

### Lower-level functions

Also exported from `pipeline_memory_estimator`, for callers that don't want to go through an artifact:

| Function | Use when |
| --- | --- |
| `estimate_pipeline_memory(pipe, latent, optimization_kwargs, pipeline_name)` | You already hold a built `DiffusionPipeline` — skips the `model_cache` lookup entirely. |
| `estimate_component_memory(component, component_name, role, pipe, pipeline_name, latent, optimization_kwargs)` | You want just one component's number, not the whole pipeline. |

## Tips & pitfalls

- **A pre-load estimate is a lower bound on accuracy, not a guess.** Before the pipeline is built, the estimate comes from real config-derived shapes and dtypes (via a meta-device build), not a rough guess — but `Automatic`-strategy topology is unresolved until the pipeline is actually loaded.
- **Switching to `Automatic` hides Manual knobs, it doesn't reset them.** If `quantization_mode` or `transformer_layerwise_casting` were set while on `Manual`, they stay set (just hidden) after switching to `Automatic` — and a pre-load estimate will still apply them, even though the real `Automatic` build never quantizes weights at all (only `Manual` does). Re-check those values, or rebuild the pipeline, before trusting a pre-load `Automatic` estimate's weight numbers.
- **Watch for estimated warnings in the widget.** These flag components whose activation memory could not be computed (unrecognized component, unregistered pipeline family, or an unreadable transformer/VAE config) — only weight memory is shown for those.
- **Treat the number as a guide, not a guarantee.** Estimates are analytical (no GPU profiling) and biased to overestimate; expect divergence from true peak usage.

## See also

- [Modular Diffusion Pipeline Builder](pipeline_builder.md)
- [Generate Media Latents](generate_media_latents.md)
- [Clear Pipeline Cache](clear_pipeline_cache.md)
