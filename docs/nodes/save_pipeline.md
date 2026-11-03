# Save Pipeline

**Writes a Pipeline Builder config as a recipe, or writes the built weights and a manifest to a folder.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- **Config Only** writes a small YAML recipe that records how to rebuild the pipeline (model, fused and runtime LoRAs, quantisation). Load rebuilds, re-fuses, re-quantises, and re-attaches runtime LoRAs.
- **Full Pipeline** writes the built weights to a folder, including fused LoRAs and post-hoc quantised weights. Runtime LoRAs are copied into `loras/` and re-attached on load.
- **Full Pipeline does not support pre-quantised Hub repos** (for example bitsandbytes 4-bit or NVFP4 Flux.2 weights). Use Config Only for those.
- Connect `pipeline` from Pipeline Builder. The destination picker is a `.yaml` file for Config Only and a folder for Full Pipeline.

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
| `save_mode` | `Config Only \| Full Pipeline` | `Config Only` | Switches the destination picker between a YAML file and a folder, and changes the default path. |
| `file_path` | `str` | `{project_dir}/config.yaml` | Destination. Shown as **File Path** in Config Only (default `{project_dir}/config.yaml`) and **Output Folder** in Full Pipeline (default `{project_dir}/baked-pipeline`). Full Pipeline writes `baked-config.yaml` inside that folder. `{project_dir}` is the project directory. |

## Provider / model behavior

Full Pipeline is refused before a build when the config is unsupported. Use Config Only where the table recommends
it; for a ControlNet wrapper, save the base pipeline and reconnect ControlNet after loading.

| Config | Behavior |
| --- | --- |
| Pre-quantised Hub repo | Refused. The Hub weights (for example `diffusers/FLUX.2-dev-bnb-4bit`, `black-forest-labs/FLUX.2-dev-NVFP4`) are not verified to round-trip through a Full Pipeline save. |
| MiniMax-H3 | Refused. The builder marks MiniMax as pre-quantised so Full Pipeline takes the same path. |
| ControlNet pipeline | Refused. Save the base pipeline instead. |
| GGUF denoiser plus fused LoRAs | Refused. The `.gguf` file is copied as-is and fused LoRAs modify that slot. Runtime LoRAs remain supported. |

## Tips & pitfalls

- **Full Pipeline writes the session as built.** Fused LoRAs and post-hoc quantisation are already in the folder; changing those settings later requires a new save.
- **Runtime LoRAs remain runtime adapters.** Full Pipeline copies their files into `loras/`, records their strengths, and never fuses them into the saved weights.
- **Runtime LoRA filenames must be unique.** Two source files with the same filename cannot both occupy the baked folder's `loras/` directory.
- **ControlNet wrappers cannot be saved.** Save the base pipeline, then reconnect ControlNet after loading.
- **Pick a folder for Full Pipeline.** Changing `save_mode` retargets the picker and resets the default path. Load that result by selecting the generated `baked-config.yaml`.

## See also

- [Load Pipeline](load_pipeline.md) — reads the manifest this node writes.
- [Modular Diffusion Pipeline Builder](pipeline_builder.md) — typical `pipeline` source.
