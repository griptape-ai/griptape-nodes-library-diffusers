# Estimate Workflow Memory (CLI)

**Estimates per-component GPU memory for a generated workflow file from the command line, without starting Griptape Nodes or running any workflow nodes.**

Command: `estimate-workflow-memory`

## TL;DR
- Run `uv run estimate-workflow-memory <workflow.py>` against a generated workflow `.py` file to get the same estimate the Estimate Pipeline Memory node produces.
- The workflow file is parsed as source and is never imported or executed. No model weights are loaded and no pipeline is built.
- The required component configs must already be in the local Hugging Face cache, or the command fails.
- Use `--json` for machine-readable output.

## Usage

```text
uv run estimate-workflow-memory workflows/templates/Text2Image.py
```

Use `--json` when the result will be consumed by another tool:

```text
uv run estimate-workflow-memory workflows/templates/Text2Image.py --json
```

Provide dimensions to override the width and height set on the workflow's noise nodes:

```text
uv run estimate-workflow-memory workflow.py --width 1024 --height 1024
```

For a video workflow, provide the frame count as well:

```text
uv run estimate-workflow-memory workflow.py --width 832 --height 480 --num-frames 25 --json
```

## Arguments

| Argument | Purpose |
| --- | --- |
| `workflow` | Path to the generated workflow `.py` file to inspect. The file is parsed as source and is never imported or executed. |
| `--width WIDTH` | Override the raw width used for every extracted builder. Must be supplied together with `--height`. |
| `--height HEIGHT` | Override the raw height used for every extracted builder. Must be supplied together with `--width`. |
| `--num-frames NUM_FRAMES` | Override the raw video frame count used for every extracted builder. Required when an extracted video builder has no static frame count. It is ignored for image latent shapes and is not reported for image builders. |
| `--json` | Print machine-readable JSON instead of the human-readable report. |

## How it works

The command parses the workflow source, extracts pipeline configurations and dimensions associated with recognized noise nodes, resolves the public latent shape through the cached Diffusers prepare-latents path, and calls the same memory estimator the node uses. It does not call `build_workflow()`, build the full diffusion pipeline, load model weights, or run inference.

Because nothing is loaded, every estimate is config-derived. Weight sizes are computed on the meta device from each component's cached config, at the dtype and quantization the pipeline would load with. Offload topology comes from the workflow's requested `cpu_offload_strategy`.

The `Automatic` memory optimization strategy is not simulated. The real Automatic cascade depends on free VRAM at build time, which this command cannot predict, so the estimate reports a conservative upper bound with no offload assumed and includes a warning. Set the strategy to `Manual` to get an estimate with a fixed offload topology.

See the [Estimate Pipeline Memory](../nodes/estimate_pipeline_memory.md) node page for how the estimate is calculated once the pipeline is loaded.

## Dimensions

The command uses recorded or overridden dimensions as-is. It does not apply model-specific dimension snapping or validation, so the result includes a raw-dimension approximation warning. It still resolves latent-space shape through the cached Diffusers prepare-latents path, preserving the distinction between pixel-space source dimensions and public latent dimensions.

If the required width and height cannot be read from a recognized noise node, the command fails. Supply `--width` and `--height` to provide them. The command never falls back to pixel-space dimensions.

## Output

The human-readable report lists each builder with its pipeline name, reported dimensions, and memory estimate.

The JSON result contains:

- a top-level `warnings` list, and
- a `builders` list. Each builder contains its pipeline name, reported dimensions, and memory estimate. Image builders omit hidden `num_frames` placeholder values; video builders include `num_frames`.

`Builder 1`, `Builder 2`, and so on identify the order of extracted estimates, not necessarily workflow execution order. Overrides apply to every extracted builder.

## Tips & pitfalls

- **Component configs must already be cached.** The command only reads config files from the local Hugging Face cache. If they are missing, it fails rather than estimating with partial data.
- **Overrides apply to every builder.** `--width`, `--height`, and `--num-frames` are not per-builder. Use them only when all extracted builders share the same dimensions.
- **Treat the number as a guide, not a guarantee.** It has the same limits as the node estimate: it is analytical, does not model allocator behavior, and does not include the CUDA context.

## See also

- [Estimate Pipeline Memory](../nodes/estimate_pipeline_memory.md)
- [Modular Diffusion Pipeline Builder](../nodes/pipeline_builder.md)
