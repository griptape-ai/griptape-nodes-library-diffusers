---
title: Estimate Workflow Memory (command line)
description: A command that estimates how much GPU memory a saved workflow needs, without starting Griptape Nodes or running anything.
---

The `estimate-workflow-memory` command reads a saved workflow file and reports the GPU memory each of its pipelines needs. It gives the same kind of estimate as the [Estimate Pipeline Memory](../nodes/PipelineMemoryEstimateNode.md) node. The workflow file is only read. It is never run, no model weights are loaded and no pipeline is built.

The model component configs must already be in your local Hugging Face cache. If one is missing, the command stops with an error. Add `--json` for output another tool can read.

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
| `workflow` | Path to the generated workflow `.py` file to inspect. The file is parsed as source and is never run. |
| `--width WIDTH` | Override the raw width used for every extracted builder. Must be supplied together with `--height`. |
| `--height HEIGHT` | Override the raw height used for every extracted builder. Must be supplied together with `--width`. |
| `--num-frames NUM_FRAMES` | Override the raw video frame count used for every extracted builder. Required when an extracted video builder has no static frame count. It is ignored for image latent shapes and is not reported for image builders. |
| `--json` | Print machine-readable JSON instead of the human-readable report. |

## How it works

The command parses the workflow source, extracts pipeline configurations and dimensions from the Create Noise Latent and Create Empty Latent nodes, resolves the public latent shape through the cached Diffusers prepare-latents path, and calls the same memory estimator the node uses. It does not build the full diffusion pipeline, load model weights, or run inference.

Because nothing is loaded, every estimate comes from the model configs. Weight sizes are computed on the meta device from each component's cached config, at the dtype and quantization the pipeline would load with. Offload topology comes from the workflow's requested `cpu_offload_strategy`.

The `Automatic` memory optimization strategy is not simulated. The real Automatic cascade depends on free VRAM at build time, which this command cannot predict, so the estimate reports a conservative upper bound with no offload assumed and includes a warning. Set the strategy to `Manual` to get an estimate with a fixed offload topology.

The [Estimate Pipeline Memory](../nodes/PipelineMemoryEstimateNode.md) page covers how the estimate is calculated once the pipeline is loaded.

## Dimensions

The command uses recorded or overridden dimensions as-is. It does not apply model-specific dimension snapping or validation, so the result includes a raw-dimension approximation warning. It still resolves latent-space shape through the cached Diffusers prepare-latents path, preserving the distinction between pixel-space source dimensions and public latent dimensions.

If the width and height cannot be read from one of those nodes, the command fails. Supply `--width` and `--height` to provide them.

## Encode nodes

[Encode Media](../nodes/VaeEncodeNode.md) and [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) are not supported estimation targets. Their source dimensions come from runtime image or video inputs, which the CLI does not inspect or decode; it also does not execute nodes to discover those dimensions. `--width`, `--height`, and `--num-frames` only override dimensions for an already recognized estimate target, so they do not make a workflow containing only encode nodes estimable. A workflow can still be estimated when it contains a separately supported pipeline and dimension assignment.

## Output

The human-readable report starts with the workflow path and any warnings. For each builder it lists the pipeline name, the dimensions, the estimated peak in GB, and one line per component with its total, weights and activations in GB. LoRA adapters are listed with their sizes. Warnings for a builder follow it.

The estimated peak includes a 20% safety margin. The per-component figures do not.

The JSON result contains:

- `workflow`, the path of the file that was read.
- A top-level `warnings` list.
- A `builders` list. Each builder contains `pipeline_name`, `dimensions` and `estimate`. Image builders leave out `num_frames`; video builders include it.
- Inside `estimate`: `offload_mode`, `components`, `estimated_peak_gb`, `warnings` and `lora_adapters`.

If the command cannot produce an estimate, it prints `Workflow memory estimation failed:` and the reason to the error output and exits with code 1.

`Builder 1`, `Builder 2`, and so on identify the order of extracted estimates, not necessarily workflow execution order. Overrides apply to every extracted builder.

## Tips

- **Overrides apply to every pipeline.** `--width`, `--height` and `--num-frames` are not per pipeline. Use them only when all pipelines in the workflow share the same size.
- **Treat the number as a guide.** It is calculated from model configs and the latent size. No pipeline is built and nothing is measured on a GPU.
