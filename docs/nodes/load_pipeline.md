# Load Pipeline

**Reconstructs a Pipeline Config from a Config Only or Full Pipeline manifest.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Point `file_path` at a `.yaml` manifest.
- A recipe rebuilds from the recorded model, LoRAs, and quantisation — the same work as Pipeline Builder, including re-fuse, re-quantise, and runtime LoRA attachment.
- A Full Pipeline's `baked-config.yaml` loads the weights in its containing folder as they were written. It does not re-fuse or re-quantise, and re-attaches runtime LoRAs copied into the folder.
- Use the optimisation parameters to override the saved slicing, tiling, layerwise casting, and CPU offload settings.
- Output type: `Pipeline Config`. Wire it where you would wire Pipeline Builder.

## Typical workflow position
```text
Save Pipeline → [Load Pipeline] → Generate Media Latents → Decode Media Latent
```

## Node preview

<!-- TODO: add docs/assets/nodes/load-pipeline.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| None | N/A | No | Path is a node property. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `pipeline` | `Pipeline Config` | Reconstructed artifact. Feed into Generate Media Latents and other pipeline consumers. |
| `was_successful` | `bool` | `True` when the load finishes without error. |
| `result_details` | `str` | Summary of the loaded recipe or baked folder (type, model or path, quantisation). |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `file_path` | `str` | `{project_dir}/config.yaml` | Existing Config Only recipe YAML or a Full Pipeline's `baked-config.yaml`. Relative paths resolve against the project directory. The picker allows `.yaml` files only. Choosing a path also tries to load immediately. |
| `memory_optimization_strategy` | choice | `Manual` | `Automatic` hides the individual controls and lets Griptape select suitable settings. |
| `attention_slicing` | bool | `False` | Process attention in slices to reduce peak VRAM at a small speed cost. |
| `vae_slicing` | bool | `False` | Decode the VAE one slice at a time to reduce peak VRAM. |
| `vae_tiling` | bool | `False` | Decode large images or video frames in tiles to reduce peak VRAM. |
| `transformer_layerwise_casting` | bool | `False` | Store eligible transformer layers in a lower-precision dtype and cast the active layer to the layerwise-casting compute dtype. |
| `cpu_offload_strategy` | choice | `None` | `Model` offloads whole submodels; `Sequential` offloads at a finer granularity with a larger speed cost. |

## Tips & pitfalls

- **A Config Only YAML still needs the model and LoRAs on this machine.** If a Hugging Face revision is not in the local cache, or a LoRA file is missing, the run fails and lists what it could not find.
- **A Full Pipeline folder does not fuse LoRAs or quantise again.** Fused LoRAs are already in the weights. Runtime LoRAs are loaded from the baked folder and activated around each generation with their saved strengths.
- **Quantisation is preserved but not exposed as an override.** Load Pipeline keeps the source artifact's quantisation configuration.
- **Select `baked-config.yaml`, not the baked folder or a Diffusers JSON index.** The node recognizes it as baked only when its parent contains `model_index.json` or `modular_model_index.json`.
- **Run the node after an in-place re-save to the same path.** Changing only the folder contents may not refresh downstream nodes until Load Pipeline runs again.

## See also

- [Save Pipeline](save_pipeline.md) — writes the YAML manifests this node reads.
- [Modular Diffusion Pipeline Builder](pipeline_builder.md) — same `pipeline` output, built from the UI instead of a file.
- [Generate Media Latents](generate_media_latents.md) — typical consumer.
