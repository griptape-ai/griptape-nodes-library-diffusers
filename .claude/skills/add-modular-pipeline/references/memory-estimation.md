# Memory-Estimation Family Registration (Rule 9)

Load this when filling in Phase B's "Memory-estimation family" section, and again before Phase C step 5's registration edits.

Adding a new pipeline TYPE always requires a matching entry in [`memory_estimation/family_registry.py`](../../../../modular_diffusion_nodes_library/memory_estimation/family_registry.py) `_FAMILY_BY_PIPELINE_NAME`, keyed by the exact same pipeline-class-name string used in `_DRIVER_REGISTRY` — the module docstring states the two registries' key sets are kept in sync by a unit test. Skipping this entry does not break at import time; it silently degrades the new pipeline's memory estimate to weights-only (`get_memory_family()` returns `None`, and `_estimate_denoiser_activation_bytes()` in [`pipeline_memory_estimator.py`](../../../../modular_diffusion_nodes_library/memory_estimation/pipeline_memory_estimator.py) reports "No memory-estimation family registered" and skips the activation term).

## Picking the `MemoryFamily`

Three values exist on the enum:

- `UNET_SDPA` — `UNet2DConditionModel`-style denoiser (no flat `num_layers`/`hidden_dim`; shape comes from `block_out_channels` + `transformer_layers_per_block` per resolution level). SDXL is the only precedent today (`get_sdxl_unet_levels()`); a new pipeline only qualifies if its denoiser is a genuine `UNet2DConditionModel`, not a transformer.
- `IMAGE_DIT_SDPA` — a single-stream image DiT transformer (Flux, Qwen, SD3, Z-Image, Flux2 precedents).
- `VIDEO_DIT_JOINT_SDPA` — a joint spatio-temporal DiT with a temporal patch dimension (WAN, LTX, HunyuanVideo1.5, MiniMax H3, LTX2 precedents).

Field names for layers/heads/hidden-dim/patch-size are **not uniform** across diffusers model configs — this is stated verbatim in `family_registry.py`'s module docstring, which lists the transformer config classes it was verified against (`FluxTransformer2DModel`, `LTXVideoTransformer3DModel`, `UNet2DConditionModel`, `QwenImageTransformer2DModel`, `WanTransformer3DModel`, `SD3Transformer2DModel`, `Flux2Transformer2DModel`, `ZImageTransformer2DModel`, `HunyuanVideo15Transformer3DModel`, `LTX2VideoTransformer3DModel`, `MiniMaxH3Transformer3DModel`). Use the existing `_adapt_*` functions in `family_registry.py` as the concrete reference for which attribute names each transformer config actually exposes — do not assume a name from a sibling family without reading the new pipeline's own transformer config class under `.venv/Lib/site-packages/diffusers/models/transformers/`.

## Denoiser config adapter

Unless the family is `UNET_SDPA` (SDXL-only today), also add an entry to `_DENOISER_FIELD_ADAPTERS` mapping the pipeline class name to a function returning `DenoiserConfigFields`. Mirror the same reuse-vs-new decision already made for `_DRIVER_REGISTRY`/driver classes:

- **Reuse an existing sibling's adapter** when the new pipeline's transformer class and config field names match a sibling's exactly — e.g. `_adapt_flux_family` is reused for `FluxPipeline`, `FluxFillPipeline`, `FluxKontextPipeline`, `Flux2Pipeline`, and `Flux2KleinPipeline` in `family_registry.py` because they share the same `num_attention_heads`/`attention_head_dim`/`num_layers`/`num_single_layers`/`patch_size` fields. Verify field-for-field against the new pipeline's actual transformer config before reusing — do not reuse just because the model family name looks similar.
- **Write a new adapter** when field names differ (e.g. `_adapt_zimage` reads `cfg.dim`/`cfg.n_layers`/`cfg.n_refiner_layers`/`cfg.all_patch_size[0]` — none of which match the Flux adapter's field names).

## Verify

- `uv run python -c "from modular_diffusion_nodes_library.memory_estimation.family_registry import get_memory_family; assert get_memory_family('<PipelineClassName>') is not None"`
- For non-`UNET_SDPA` families: manually confirm `_DENOISER_FIELD_ADAPTERS['<PipelineClassName>']` reads only attributes confirmed to exist on the new pipeline's transformer config class.
