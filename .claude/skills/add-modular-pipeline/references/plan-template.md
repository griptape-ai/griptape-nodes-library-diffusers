# Phase B Plan Template

Load this when producing the Phase B plan. Fill in every section — do not omit one because it seems inapplicable; write "n/a" with a one-line reason instead.

```
# Plan: Add <Model Name> Support

## Closest precedent
- Driver: <file> (because ...)
- Standard params: <file>
- Runtime params: <file>

## Pipeline class
- diffusers: <module.ClassName> (verified at <path>)
- Modular blocks: <module.AutoBlocks> (verified at <path>)

## Modular vs DiffusionPipeline split
| Method | Approach | Justification | Required inputs → source |
| encode_media | ModularPipeline.sub_blocks["vae_encoder"] | exists at ... | `image` / `video` ← `media.image` or `media.frames`; `generator` ← `generator_state.to_generator()` |
| encode_prompt | ModularPipeline.sub_blocks["text_encoder"] (or override) | exists at ...; declare extra required inputs (e.g. `prompt_2`, image conditioning) if the block needs them | enumerate every `required=True` `InputParam` |
| decode_latent | ModularPipeline.sub_blocks["decode"] | exists at ... | `latents` ← `latent.to_torch(device, dtype)`; `output_type="pil"` |
| create_noise_latent | ModularPipeline PrepareLatents step | ... | `height`/`width` ← `source_shape[-2:]`; `batch_size=1`; `num_images_per_prompt=1`; `generator` ← `generator_state.to_generator()` |
| add_noise_to_latent | ModularPipeline Img2Img / Vid2Vid noise injection steps | ... | enumerate every `required=True` `InputParam` (see Modular-Blocks-First rule); e.g. `latents` ← `self.create_noise_latent(...).to_torch(device, dtype)`, `image_latents` ← `latent.to_torch(device, dtype)`, `batch_size` ← `image_latents.shape[0]`, `dtype` ← `_get_device_and_type()[1]`, `generator` ← `generator_state.to_generator()`. For video pipelines: `image_latents` ← encoded video latent; noise API is `scheduler.scale_noise` (flow-matching) or `scheduler.add_noise` (DDPM) — verify against the scheduler class. |
| denoise_latent | super() → DiffusionPipeline.__call__ | three blockers, see SKILL | n/a (delegated) |

## Provider classification (must match the Rule 1, Axis B approval)
- Shape: [new Provider] | [extend existing Provider]
- If new Provider:
    - Enum entry: `Provider.<NEW_NAME> = "<Display Name>"`
    - New `LatentPipelineTypeParameters` subclass: `<ClassName>`
    - `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` entry: `Provider.<NEW_NAME>` → `<ClassName>`
- If extending: target subclass = `<ExistingClassName>`; one new dict entry only.

## Runtime parameters class reuse
State whether this pipeline class gets a NEW runtime-parameters class or reuses an existing one. Siblings within a generation commonly share a runtime-params class (see existing reuse precedents in [`parameters/pipeline_parameters.py`](../../../../modular_diffusion_nodes_library/parameters/pipeline_parameters.py) — e.g. `Flux2Pipeline` + `Flux2KleinPipeline`). If reusing, justify by showing the upstream `__call__` signatures are runtime-compatible.

## Files to create
- modular_diffusion_nodes_library/standard_parameters/<file>.py
- modular_diffusion_nodes_library/runtime_parameters/<file>.py  (omit if reusing an existing one)
- modular_diffusion_nodes_library/latent_pipeline_drivers/<file>.py

## Files to modify (only the lines required — see Surgical Changes principle)
- modular_diffusion_nodes_library/latent_pipeline_drivers/driver_factory.py
    → one entry in `_DRIVER_REGISTRY`
- modular_diffusion_nodes_library/parameters/pipeline_parameters.py
    → one or more `case "<PipelineClassName>":` in `set_runtime_parameters`
- modular_diffusion_nodes_library/parameters/pipelinetype_parameters.py
    → EITHER a new subclass + `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` entry (new-Provider shape),
      OR one new entry in an existing subclass's `get_pipeline_type_dict()` (extend-existing shape).
      NEVER both. NEVER name img2img/inpaint/controlnet sibling classes here — those are runtime
      variants and belong to `/add-pipeline-variants`.
- modular_diffusion_nodes_library/parameters/providers.py
    → only when adding a new `Provider` enum entry.
- modular_diffusion_nodes_library/memory_estimation/family_registry.py
    → one entry in `_FAMILY_BY_PIPELINE_NAME`; one entry in `_DENOISER_FIELD_ADAPTERS`
      unless the family is UNET_SDPA (see Rule 9 / references/memory-estimation.md).

## Memory-estimation family (Rule 9 — see references/memory-estimation.md; every cell needs a code-reference citation)
- `MemoryFamily`: UNET_SDPA | IMAGE_DIT_SDPA | VIDEO_DIT_JOINT_SDPA — <justification, e.g. "single-stream image DiT, no temporal patch dim">
- Denoiser config adapter: [new function `_adapt_<model>` | reused `_adapt_<sibling>`] — if reused, cite the sibling pipeline class whose transformer config field names (`num_attention_heads`/`attention_head_dim`/`num_layers`/`patch_size`/etc.) were verified to match, with the diffusers transformer config file path; if new, list each `DenoiserConfigFields` field and the exact config attribute it reads, citing `.venv/Lib/site-packages/diffusers/models/transformers/<file>.py`.
- Omit this section's adapter row only if `MemoryFamily` is UNET_SDPA (SDXL-only today; no other pipeline should classify as UNET_SDPA without discussion).

## Latent contract compliance
- VAE whitening needed? [yes/no, with citation to the VAE config — `latents_mean` / `latents_std` for whitening; `scaling_factor` / `shift_factor` for the scalar path]
- Packing needed inside `prepare_input/output_latent`? [yes/no, citation to the pipeline class]
- Video? [yes/no → `produces_video`, `video_fps`]

## Variants (capability checklist)
- ControlNet supported? [yes/no] — if yes, plan to apply Pattern A from `/add-pipeline-variants`
- Inpaint supported? [yes/no] — if yes, plan to set `_inpaint_pipeline_class` (Pattern B)
- Runtime pipe-swap needed (e.g. conditional variant)? [yes/no] — if yes, plan Pattern C
- **Video-to-video (V2V) supported?** [yes/no] — **video drivers only**. Determines whether `encode_media(VideoMedia)` and `add_noise_to_latent` need real implementations instead of `NotImplementedError`. To answer yes: (a) the VAE must accept `[B, C, T, H, W]` multi-frame input, and (b) the scheduler must expose `scale_noise` or `add_noise` for video latents. Cite the diffusers source for both. If no, raise `NotImplementedError` with an inline comment citing the source that confirms V2V is unsupported.

## Runtime parameter defaults and tooltips (Rule 4 — every cell needs an upstream citation)
| Param | Default value | Default source | Tooltip text source |
|---|---|---|---|
| guidance_scale | <value> | `__call__` signature in `pipeline_<name>.py` L<n> | `__call__` docstring entry for `guidance_scale` in same file, L<n> |
| num_inference_steps | <value> | `EXAMPLE_DOC_STRING` in `pipeline_<name>.py` L<n> | `__call__` docstring entry for `num_inference_steps`, L<n> |
| <other> | <value> | <citation> | <citation> |

## Open questions for the user
- <model-specific decisions, e.g. quantisation, default repo>
```

If any variant capability is in scope, load the `/add-pipeline-variants` skill for that portion of the implementation. The base 5-step process still owns Provider/StandardParams/RuntimeParams/Driver-skeleton/Registration; the variant skill owns ControlNet/inpaint/pipe-swap specifics.
