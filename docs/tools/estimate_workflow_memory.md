---
title: Estimate Workflow Memory (command line)
description: A command that estimates how much graphics card memory a saved workflow needs, without starting Griptape Nodes or running anything.
---

This tool is for technical staff. Ask your pipeline TD to run it. Griptape Nodes saves workflows as `.py` files, and you do not need to read them.

The `estimate-workflow-memory` command tells you how much graphics card (GPU) memory a saved workflow needs before you run it. Use it to check whether a model fits on your card. It reads a saved workflow file and reports the memory for each pipeline in it. A pipeline is a loaded model together with its settings. The command uses the same calculation as the [Estimate Pipeline Memory](../nodes/PipelineMemoryEstimateNode.md) node.

It only reads the workflow file. It does not run the workflow, load the model or build a pipeline.

Workflows that use [Encode Media](../nodes/VaeEncodeNode.md) or [Encode Masked Media](../nodes/VaeMaskEncodeNode.md) are not supported. That includes Image2Image and Inpainting. Their size comes from the image or video you load, and the command never opens those. `--width`, `--height` and `--num-frames` do not fix this. Use the [Estimate Pipeline Memory](../nodes/PipelineMemoryEstimateNode.md) node for those workflows. A workflow that has a supported Create Noise Latent or Create Empty Latent node next to its Build Pipeline node can still be estimated.

The command needs only small files from the Hugging Face cache (each part's `config.json` and the model's `model_index.json`). It does not need the model weights and never downloads anything. LoRA files in the workflow must be on disk so their sizes can be read.

## Usage

Run the command in a terminal, from the folder of this library's source code. The workflow is a Python (`.py`) file. The templates that ship with the library are in `workflows/templates/`.

```text
uv run estimate-workflow-memory workflows/templates/Text2Image.py
```

To override the width and height set on the workflow's noise nodes:

```text
uv run estimate-workflow-memory workflow.py --width 1024 --height 1024
```

For a video workflow, add the frame count:

```text
uv run estimate-workflow-memory workflow.py --width 832 --height 480 --num-frames 25
```

## Arguments

| Argument | Purpose |
| --- | --- |
| `workflow` | Path to the workflow `.py` file to read. |
| `--width WIDTH` | Use this width instead of the one in the workflow. Give it together with `--height`. |
| `--height HEIGHT` | Use this height instead of the one in the workflow. Give it together with `--width`. |
| `--num-frames NUM_FRAMES` | Use this frame count for video. Required when a video pipeline in the workflow has no fixed frame count. Ignored for images. |
| `--json` | Print the result as JSON instead of a readable report. |

## How it works

The command finds each [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) node in the workflow. It reads the image size, and the frame count for video, from your Create Noise Latent or Create Empty Latent nodes. Those nodes make the starting latent, the compressed form the model works in. It works out the size of that latent and passes it, with the pipeline settings, to the same calculation the Estimate Pipeline Memory node uses.

Nothing is loaded. Model sizes come from the model's settings files. The command works out how much space each part takes from the number format (for example 16-bit or 8-bit) and the compression the pipeline would load with. Which parts move to your computer's main memory comes from the `cpu_offload_strategy` setting in the workflow.

The `Automatic` memory strategy picks its settings when the model loads, based on how much graphics card memory is free at that moment. The command cannot know that in advance. For `Automatic` it reports the worst case, with nothing moved off the card, and adds a warning. To get an estimate that matches a real run, set the strategy to `Manual` in the workflow first.

## Sizes

The command uses the width and height it finds, or the ones you pass, exactly as given. It does not round them to the sizes a model accepts, so the report carries a warning that the size is approximate.

If it cannot read the width and height from Create Noise Latent or Create Empty Latent, it fails. Pass `--width` and `--height` to supply them.

## Output

The readable report starts with the workflow path and any warnings. Then, for each pipeline, it lists the pipeline name, the size, the estimated peak in GB, and one line per model part with its total, weights and working memory in GB. Weights are the model's stored numbers, which take the same space whatever the picture size. Working memory is the extra space needed while the model runs, which grows with picture size and video length. Any LoRAs are listed with their sizes. A LoRA is a small add-on file that changes a model's style or subject. Warnings for a pipeline follow it.

The pipelines are named `Builder 1`, `Builder 2` and so on. These are the Build Pipeline nodes found in the workflow, in the order the command found them. That is not always the order the workflow runs them.

The estimated peak includes a 20% safety margin. The per-part figures do not.

If the command cannot produce an estimate, it prints `Workflow memory estimation failed:` and the reason, and exits with code 1.

## Tips

- **Overrides apply to every pipeline.** `--width`, `--height` and `--num-frames` are not set per pipeline. Use them only when all pipelines in the workflow share the same size.
- **Treat the number as a guide.** It is calculated from model settings files and the latent size. Nothing is measured on a graphics card.

## For scripts

Add `--json` to any of the commands above to get output that another program can read. It contains:

- `workflow`, the path of the file that was read.
- A top-level `warnings` list.
- A `builders` list, one entry per pipeline. Each has `pipeline_name`, `dimensions` and `estimate`. Video entries include `num_frames`. Image entries leave it out.
- Inside `estimate`: `offload_mode`, `components`, `estimated_peak_gb`, `warnings` and `lora_adapters`.
