# Save Pipeline

**Writes a Pipeline Builder config to a JSON recipe, or writes the built weights to a folder.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- **Config Only** writes a small JSON that records how to rebuild the pipeline (model, LoRAs, quantisation). Load rebuilds, re-fuses, and re-quantises.
- **Full Pipeline** writes the built weights to a folder, including fused LoRAs and post-hoc quantised weights. Load does not re-fuse or re-quantise.
- **Full Pipeline does not support pre-quantised Hub repos** (for example bitsandbytes 4-bit or NVFP4 Flux.2 weights). Use Config Only for those.
- Connect `pipeline` from Pipeline Builder. The destination picker is a `.json` file for Config Only and a folder for Full Pipeline.

## Typical workflow position
```text
Pipeline Builder → [Save Pipeline]
Load Pipeline → Generate Media Latents → Decode Media Latent
```

## Node preview

<!-- TODO: add docs/assets/nodes/save-pipeline.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `pipeline` | `Pipeline Config` | Yes | From [Pipeline Builder](pipeline_builder.md) (or Load Pipeline). |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `was_successful` | `bool` | `True` when the write finishes without error. |
| `result_details` | `str` | Success or failure text, including the resolved destination path. |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `save_mode` | `Config Only \| Full Pipeline` | `Config Only` | Switches the destination picker between a JSON file and a folder, and changes the default path. |
| `file_path` | `str` | `{project_dir}/pipeline-config.json` | Destination. Shown as **File Path** in Config Only (default `{project_dir}/pipeline-config.json`) and **Output Folder** in Full Pipeline (default `{project_dir}/baked-pipeline`). `{project_dir}` is the project directory. |

## Provider / model behavior

Full Pipeline is refused before a build when the config is unsupported. Use Config Only in these cases:

| Config | Behavior |
| --- | --- |
| Pre-quantised Hub repo | Refused. The Hub weights (for example `diffusers/FLUX.2-dev-bnb-4bit`, `black-forest-labs/FLUX.2-dev-NVFP4`) are not verified to round-trip through a Full Pipeline save. |
| MiniMax-H3 | Refused. The builder marks MiniMax as pre-quantised so Full Pipeline takes the same path. |
| ControlNet pipeline | Refused. Save the base pipeline instead. |
| GGUF denoiser plus LoRAs | Refused. The `.gguf` file is copied as-is and LoRAs usually fuse into that slot. |

## Tips & pitfalls

- **Full Pipeline writes the session as built.** Fused LoRAs and post-hoc quantisation are already in the folder; changing those settings later requires a new save.
- **ControlNet is Config Only only.** There is no Full Pipeline path for a ControlNet wrapper yet.
- **Pick a folder for Full Pipeline, not a `.json` file.** Changing `save_mode` retargets the picker and resets the default path.

## See also

- [Load Pipeline](load_pipeline.md) — reads the JSON or folder this node writes.
- [Modular Diffusion Pipeline Builder](pipeline_builder.md) — typical `pipeline` source.
