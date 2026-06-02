---
name: add-modular-pipeline
description: 'Add a new diffusion model / pipeline TYPE to the modular_diffusion_nodes_library. Use ONLY when `<NewPipelineClass>.from_pipe(base_pipe)` cannot produce a working pipeline from the components already loaded on an existing base pipe — i.e. the new pipeline needs a component the base does not have, or needs differently-shaped/-trained weights for an existing component. Examples that qualify: Wan 2.2 T2V vs I2V (different UNet + image encoder), Qwen Edit (different transformer + image conditioning), Flux Fill (different transformer weights). Walks through the 5-step process: Provider enum, Standard Parameters, Runtime Parameters, Driver, Registration. DO NOT use when from_pipe(base) can build the new pipeline against the loaded components (ControlNet, inpaint, LTX media_gen_conditioning) — those are runtime variants and belong in add-pipeline-variants. Default to add-pipeline-variants first; only fall back to this skill when the from_pipe test fails.'
---

# Add a Model to the Modular Diffusion Library

This skill guides the agent through adding a new diffusion model (image or video) to `modular_diffusion_nodes_library/` in this repo.

## Working Principles

These principles apply to every rule and phase below. They exist because LLM agents tend to silently pick interpretations, over-engineer, and touch code beyond the brief. Every Rule that follows is a concrete application of one or more of these.

1. **Think before coding.** State assumptions. If the user's request is ambiguous (e.g. "add Wan" — T2V or I2V? "add Flux" — base, Fill, or Kontext?), surface the ambiguity and ask. Do not pick silently. Do not start coding while confused; name what is unclear and stop. Push back if a simpler approach exists (e.g. "this looks like a variant, not a new pipeline type — use `/add-pipeline-variants`?").
2. **Simplicity first.** Mirror the closest existing driver. Do not introduce new abstractions, helper modules, base classes, or "flexibility" knobs that weren't asked for. No speculative error handling for cases that cannot occur. If a method can be 10 lines instead of 50, write 10. If 200 lines could be 50, rewrite it.
3. **Surgical changes.** In shared files (`providers.py`, `driver_factory.py`, `pipeline_parameters.py`, `pipelinetype_parameters.py`), edit only the lines needed for the new pipeline. Do not reformat, reorder, rename, or "tidy" unrelated entries. Do not drive-by refactor sibling drivers because you noticed something. Match existing style even if you'd write it differently. If you spot unrelated issues, mention them in chat — do not silently change them.
4. **Goal-driven execution.** Every step in Phase C has an explicit verification check stated *before* the change is made. The step is not done until verification passes. "Make it work" is not a verification; "`get_driver_class('SD3Pipeline')` returns the new driver class" is.

## ⛔ Mandatory Workflow Rules

Read these rules before doing anything. Violating them invalidates the work.

### Rule 0 — Update affected node docs in the same change

Adding a new pipeline type rarely creates new nodes, but it almost always changes what existing nodes expose for the new provider. Before finishing, update the `Provider / model behavior` section of every node whose dynamic parameters now branch on the new pipeline type — typically Pipeline Builder, Generate Media Latents, and any encode/decode/conditioning node involved. Follow [docs/node-doc-format.md](../../../docs/node-doc-format.md); for any doc work beyond a one-line tweak, use [.github/skills/document-node/SKILL.md](../document-node/SKILL.md). Report doc changes alongside code in your final summary.

### Rule 1 — Classify the change, then get approval before coding

Before writing anything, classify the request along TWO axes and present the classification for approval.

**Axis A — pipeline TYPE vs variant**:

**Default to variant.** The criterion is: can `<NewPipelineClass>.from_pipe(base_pipe)` produce a *working* pipeline using the components already loaded on an existing base pipeline (UNet/transformer, VAE, text encoders, scheduler)?

- If **yes** → it is a **variant**. STOP and use `/add-pipeline-variants` instead. The user does not load it via `from_pretrained` — they get it for free off the base pipe at runtime. Examples: ControlNet (any model), inpaint (any model), LTX's `LTXConditionPipeline`. Note: img2img is **not** a variant and needs no pipeline class swap at all — the base pipeline usually handles image-to-image natively via noise addition on the input latent. See Rule 7.
- If **no** → it is a new **pipeline type**. The variant class needs a component the base pipeline does not have (extra image encoder, different scheduler family) or needs differently-shaped/-trained weights for an existing component (different transformer weights). Examples: WAN T2V → WAN I2V (different UNet weights + image encoder), Flux → Flux Fill (different transformer weights), Qwen → Qwen Edit (different transformer + image conditioning).


**Axis B — new Provider vs new sibling under existing Provider**:

Inspect [`parameters/providers.py`](../../../modular_diffusion_nodes_library/parameters/providers.py) and [`parameters/pipelinetype_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipelinetype_parameters.py) and decide which shape applies:

- **New `Provider`** — the model is a new architecture / generation. Add a `Provider` enum entry, a new `LatentPipelineTypeParameters` subclass, and a `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` entry. (Shape used by `Provider.FLUX` → `Provider.FLUX2`.)
- **Extend an existing Provider's type dict** — the model is a sibling checkpoint of an existing generation (different weights, same architecture). Add ONE entry to that subclass's `get_pipeline_type_dict()`. Do NOT add a `Provider` enum entry. Do NOT add a `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` entry. (Shape used by `FluxPipeline` + `FluxFillPipeline` under `Provider.FLUX`.)

The decision is **not** "is this the same model family?" — it is "is this the same architecture generation?". Flux and Flux2 are separate Providers despite the shared brand because the weights, text encoders, and pipeline class hierarchy differ.

**Generic Provider names**: if the closest existing Provider's display name is a generic family name (e.g. `Provider.STABLE_DIFFUSION = "Stable Diffusion"`) rather than a specific generation, surface the naming collision to the user before classifying. Do not silently extend a generically-named Provider with a new generation, and do not rename the existing enum value (saved workflows depend on the string). The correct shape in this case is almost always a new, generation-specific Provider (e.g. `STABLE_DIFFUSION_3 = "Stable Diffusion 3"`).

State your classification and the consequent file-edit shape, then STOP and wait for explicit user approval before proceeding to Phase A's remaining steps.

**Variant siblings of the new base pipeline (inpaint / ControlNet of the new architecture) are NOT added in this skill**.

### Rule 2 — Modular-blocks gate fires before Phase B

Before producing any plan, confirm `modular_pipelines/<model>/` exists in the pinned diffusers. If it does not, STOP and execute the gate procedure in Phase A.5 — check upstream `huggingface/diffusers`, present bump-vs-hybrid options, and wait for the user's decision. Do NOT draft Phase B against imaginary block APIs.

The gate is not resolved until a **concrete outcome** exists: either (a) a diffusers bump has been executed, `uv sync` passed, new symbols import cleanly, and `uv run pytest tests` passes; or (b) the user has explicitly accepted a hybrid driver. Only then may Phase B (i.e. the Rule 3 plan) be drafted.

### Rule 3 — Plan and wait for confirmation

Before writing any code or creating any files:

1. Produce a **full implementation plan** covering all 5 steps (Provider, Standard Params, Runtime Params, Driver, Registration).
2. The plan MUST list every file to be created or modified, with absolute paths.
3. The plan MUST state explicitly which `ModularPipeline` blocks (from diffusers) the driver will use for each public method, and which methods (if any) will fall back to `DiffusionPipeline.__call__` and **why**.
4. The plan MUST identify the closest existing driver(s) being used as a template, with file references.
5. STOP and wait for explicit user confirmation (e.g., "approved", "go", "yes") before implementing. Do NOT begin partial implementation while awaiting confirmation.

### Rule 4 — Runtime parameter defaults MUST come from upstream

When choosing default values for the runtime-parameter UI (e.g. `guidance_scale`, `num_inference_steps`, `max_sequence_length`, `true_cfg_scale`, `negative_prompt`, etc.), **do not invent values, do not copy values from a sibling driver, and do not "round to a nice number"**. Take them, in this priority order, from:

1. **The pipeline's `__call__` signature defaults.** Open `.venv/Lib/site-packages/diffusers/pipelines/<model>/pipeline_<name>.py` and read the `def __call__(self, ..., guidance_scale: float = X.X, num_inference_steps: int = N, ...)` signature. These are the upstream-recommended defaults.
2. **`EXAMPLE_DOC_STRING` in the same file.** When `__call__` has no default for a parameter (sometimes left required), or when a value in the example differs from the signature default (the example often reflects the *intended* setting for the headline checkpoint), prefer the example. Cite the line.
3. **The model's HuggingFace model card** — only when neither of the above gives a value, and only with an explicit citation in the plan.

Each runtime parameter MUST have a clear, descriptive **tooltip** in the `Parameter(...)` declaration. Source the wording from the upstream pipeline's `__call__` docstring entry for that argument, the model card's parameter description, the `EXAMPLE_DOC_STRING`, or a concise fusion of those. Do not write tooltips from scratch when upstream already documents the parameter. Strip diffusers-internal jargon (e.g. `do_classifier_free_guidance`) and prefer the user-facing intent; keep value ranges and "higher = X / lower = Y" guidance when upstream provides it.

When the pipeline class exposes multiple sibling defaults that disagree (e.g. base vs turbo vs distilled checkpoints with different example `num_inference_steps`), pick the value matching the **first repo** in the standard parameters' repo list, and note the choice in the plan's Phase B output.

Forbidden:
- Copying `guidance_scale=7.0` from SDXL into an SD3 driver because "that's what SDXL uses".
- Setting a default without reading the upstream pipeline file.
- Setting a default to "what looks good in tests" without citing the upstream source as the *first* attempt.

If the upstream pipeline genuinely has no opinion (parameter is required, no example), surface the question in Phase B's "Open questions for the user" block rather than guessing.

If a parameter is **absent from `__call__` entirely** — for example because the pipeline delegates guidance to a component (e.g. a guider) rather than accepting it as a kwarg — search for a default using the same source priority: modular block `expected_components` FrozenDict configs, `EXAMPLE_DOC_STRING`, model card. Cite whichever source provides the value and recommend it as the default. Also flag in Phase B that the parameter must **not** be forwarded directly to `pipe()` as a kwarg, and state the routing mechanism (e.g. setting the component attribute before calling `super().denoise_latent()`). If the routing mechanism is unclear, raise it as an open question rather than guessing.

### Rule 5 — Adhere to the LatentPipelineDriver contract

The latent shape & space contract is documented in the `LatentPipelineDriver` docstring in [`modular_diffusion_nodes_library/latent_pipeline_drivers/base_driver.py`](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/base_driver.py). **Read it before designing the driver — it is the source of truth.** Summary of the invariants the agent MUST honour:

- **Public latents are unpacked**: 4-D `[B, C, H/vae, W/vae]` for image, 5-D `[B, C, T_lat, H/vae, W/vae]` for video. No model-specific sequence packing on the public surface.
- **Public latents are normalised** (~N(0, 1)). If the VAE config publishes per-channel `latents_mean` / `latents_std`, apply whitening `(z - mean) / std` inside `encode_image`/`encode_video` and the inverse inside `decode_latent`. If it publishes only scalar `scaling_factor` (and optional `shift_factor`), the standard `(z - shift) * scaling` transform is sufficient — no whitening. Verify on the actual VAE config of the model you are adding; do not assume from family name. Mirror the closest existing driver.
- **`latents_source_shape` is pixel-space**, not latent-space. Drivers translate to latent-space dims internally.
- **Packing is transient**: any transformer-side sequence packing happens only inside `prepare_input_latent` / `prepare_output_latent`, never on the public surface. Mirror the closest existing driver that packs.
- **Never bypass `super().denoise_latent()` for callback / partial-denoise / cancellation concerns.** Override `denoise_latent()` only to munge kwargs (e.g. set `image=` for img2img, build video conditions, extract first/last frames), then delegate to `super()`.

### Rule 6 — Prove every claim with code references

Every implementation choice MUST be justified with concrete evidence:

- Cite file paths and (where useful) line numbers or symbol names.
- When copying a pattern from an existing driver, name it explicitly: "modelling `encode_image()` after `FluxLatentPipelineDriver.encode_image()` because both pipelines pack latents transformer-side."
- When referencing a diffusers API, the agent MUST verify it exists by reading the source under `.venv/Lib/site-packages/diffusers` before citing it. **Do not invent symbols, block names, or kwargs.**
- When no precedent exists for a new modular block path, cite the diffusers source file that documents/implements the block.
- Assertions without proof are not acceptable.

### Rule 7 — Always store the BASE pipeline class

In `standard_parameters/<model>_parameters.py`, set `_pipeline_cls` (and report `pipeline_name`) to the **base** pipeline class — `StableDiffusion3Pipeline`, `FluxPipeline`, `WanPipeline`, never `*Img2ImgPipeline` or `*InpaintPipeline`. The base pipeline handles **both text-to-image and image-to-image** for image models (and text-to-video / video-to-video for video models); img2img is invoked by adding noise to encoded image (strength depends on specified start_step) and passing it through `denoise_latent()`, not by swapping the pipeline class. Inpaint is enabled by setting `_inpaint_pipeline_class` on the driver. ControlNet is handled by the variants skill. This keeps the registry key consistent across `_DRIVER_REGISTRY`, `set_runtime_parameters` `case` arms, and `get_pipeline_type_dict()` entries — all three use `_pipeline_cls.__name__`.


### Rule 8 — Prefer bumping diffusers over hybrid driver fallbacks

If the modular blocks needed for the new pipeline aren't in the currently pinned diffusers version, **bumping diffusers is strongly preferred** over working around the gap with a hybrid driver that re-implements the missing steps (e.g., manually constructing latents, calling `DiffusionPipeline.__call__` for ops that have modular equivalents upstream, copying step logic into the driver).

Rationale: hybrid fallbacks drift from upstream, duplicate logic that already exists, and have to be ripped out the next time diffusers is bumped anyway. Bumping is a one-line change in two files.

Procedure when blocks are missing:

1. Check `huggingface/diffusers` `main` (or the relevant PR) to confirm the blocks exist upstream. Cite the upstream file path / PR / commit SHA.
2. Present the user with the options: (a) wait for a tagged release, (b) pin to a tagged release if one already includes the blocks, (c) pin to a specific commit SHA on `main`. Recommend (b) or (c) over building a hybrid driver.
3. If the user approves a bump, update **both** of these files to the same version spec — they MUST stay in sync:
   - [`pyproject.toml`](../../../pyproject.toml) — the `diffusers` entry under `[project] dependencies`.
   - [`griptape_nodes_library.json`](../../../griptape_nodes_library.json) — the `diffusers` entry under `dependencies.python_dependencies`.
4. For a commit-SHA pin, use the PEP 508 direct-reference form in both files: `diffusers @ git+https://github.com/huggingface/diffusers.git@<sha>`. `pyproject.toml` already has `tool.setuptools` / `tool.uv` configured to allow direct references; no extra plumbing is required.
5. After editing, run `uv sync --all-groups --all-extras` to refresh the editable install metadata, verify the new symbols import cleanly, then run `uv run pytest tests` to confirm no regressions in existing drivers. Only proceed to Phase C once all three pass.

A hybrid driver is only acceptable when (a) the user explicitly declines to bump diffusers AND (b) the gap is small and well-isolated (one block, easy to swap back later). Document the deviation in the driver with a `# TODO(diffusers-bump):` comment that names the missing block and the upstream PR/commit so it can be removed later.

---

## Modular-Blocks-First Philosophy

The long-term goal is to use diffusers `ModularPipeline` blocks for ALL operations including the denoise loop. **Prefer modular blocks wherever they exist** for the pipeline being integrated.

Three open blockers force every driver to delegate the **denoise loop** to `DiffusionPipeline.__call__()` today:

1. **Partial denoise** — `PartialDenoisePipelineRunner` in [`modular_diffusion_nodes_library/misc/partial_denoise.py`](../../../modular_diffusion_nodes_library/misc/partial_denoise.py) intercepts `pipe.scheduler.set_timesteps()`; no equivalent injection point exists on a `ModularPipeline` denoise block.
2. **Callback / preview** — `callback_on_step_end(pipe, i, _t, callback_kwargs)` is passed via `pipe_kwargs` to `DiffusionPipeline.__call__()` (see `denoise_latent()` in `base_driver.py`). Modular denoise blocks don't expose this hook.
3. **Cancellation** — Implemented by `pipe._interrupt = True` inside the callback; no equivalent on `ModularPipeline`.

**Implication for the agent**: when planning the driver:
- Modular blocks for: `_create_modular_pipe`, `encode_image`, `encode_video`, `decode_latent`, `create_noise_latent`, `add_noise_to_latent`, `encode_prompt`.
- `DiffusionPipeline.__call__` via `super().denoise_latent()` for the denoise loop. Do not attempt to bypass — fixing the three blockers is a framework-level task, not a per-driver one.

### Read every block's `inputs` before calling `_call_block(...)`

Modular block input contracts are NOT consistent across pipelines, and they are NOT consistent across blocks within the same pipeline. The same block name (e.g. `Img2ImgPrepareLatentsStep`, `PrepareLatentsStep`, `VaeEncoderStep`, `SetTimestepsStep`, `DecodeStep`) can have different required inputs, different optional inputs, and different defaults between SDXL, SD3, Flux, Z-Image, WAN, Qwen, etc.

**Mandatory pre-write step for every `_call_block(SomeBlock(), ...)` site:**

1. Open the block class under `.venv/Lib/site-packages/diffusers/modular_pipelines/<model>/*.py`.
2. Read its `inputs` property end-to-end.
3. List every `InputParam` and classify it:
   - `required=True` → MUST be passed in your `_call_block(...)` kwargs.
   - has a non-None `default=` → safe to omit; document the default in a comment if you rely on it.
   - optional with `default=None` → check the block's `__call__` to see whether `None` is handled (often via `block_state.x or ...` or a conditional branch); if absence triggers an internal fallback (e.g. "if `latents` is None, generate noise"), that fallback is what you are relying on — note it explicitly.
4. Read the block's `__call__` to confirm how each input is consumed, and in particular whether a `required=True` input is used directly (no fallback) — that is the case most likely to silently break when copied from a sibling driver.

**Sourcing required inputs (general principles):**

- Latent/noise tensors: prefer `self.create_noise_latent(...)` over inline `torch.randn(...)` — it goes through the scheduler-aware modular path and keeps noise generation consistent across the driver. Inline `torch.randn` is only acceptable when no modular noise path exists for the pipeline.
- Image latents: pass the caller-provided latent tensor (cast to the right `device` / `dtype` via `self._get_device_and_type()`).
- `batch_size`: derive from the input tensor's `.shape[0]`, never hard-code.
- `generator`: build from `seed` with `torch.Generator(device=device).manual_seed(seed)`.
- `dtype`: pass the `dtype` from `_get_device_and_type()`, not the tensor's current dtype after a `.to(...)` cast.

---

## Procedure

### Phase A — Investigation (no code written, version bump possible)

0. List `/memories/repo/` and read any entries whose filenames suggest relevance to the new model family or to the shared subsystems being touched (latent contract, hashing, offload, denoise, the closest precedent driver). Repo memory captures previously-learned gotchas — treat it as advisory (entries may be outdated), but always check before re-deriving.
1. Read [`base_driver.py`](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/base_driver.py) end-to-end. Internalise the latent shape/space contract.
2. Read [`driver_factory.py`](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/driver_factory.py), [`parameters/pipelinetype_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipelinetype_parameters.py), and [`parameters/pipeline_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipeline_parameters.py) to understand the three registries.
3. Identify the closest existing model/driver to the one being added:
   - Image, no packing, no extra encoders → SDXL
   - Image, packed latents, dual text encoders → Flux
   - Image, normalised whitening, single transformer → Z-Image
   - Video → WAN (t2v) or WAN i2v
   - Edit → Qwen Edit
4. Read that driver, its standard_parameters, and its runtime_parameters fully.
5. Verify the diffusers pipeline class for the new model exists by reading under `.venv/Lib/site-packages/diffusers/pipelines/`. Cite the file. Then verify the modular blocks exist under `.venv/Lib/site-packages/diffusers/modular_pipelines/`. Cite the directory listing.

   **Modular-blocks gate** — if `modular_pipelines/<model>/` is absent from the pinned diffusers, do not proceed to Phase B until you have:

   a. Checked `huggingface/diffusers` `main` (and any open PRs) for the blocks by **actively fetching** `https://github.com/huggingface/diffusers/tree/main/src/diffusers/modular_pipelines` with a web-fetch tool. The gate does not pass until the fetch result is cited. Cite the upstream file path / PR / commit SHA, or state "not present upstream either" with proof from the fetch.

   b. Presented the user with choices and waited for an explicit decision:
      - If blocks exist upstream: offer (i) bump diffusers to a tagged release that includes them, (ii) pin to a specific commit SHA on `main`, (iii) decline and accept a hybrid driver per Rule 8.
      - If blocks do not exist upstream: offer (i) wait and defer this work, (ii) accept a hybrid driver per Rule 8. Do not silently proceed with a hybrid.

   c. If the user approves a diffusers bump, execute Rule 8's procedure now (before Phase B) so the plan can cite the real block APIs.
6. Load [`./references/driver-details.md`](./references/driver-details.md) for the full method-by-method driver implementation reference.

### Phase B — Present the plan

Produce a plan in this format:

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
| encode_image | ModularPipeline.sub_blocks["vae_encoder"] | exists at ... | `image` ← caller |
| encode_prompt | ModularPipeline.sub_blocks["text_encoder"] (or override) | exists at ...; declare extra required inputs (e.g. `prompt_2`, image conditioning) if the block needs them | enumerate every `required=True` `InputParam` |
| decode_latent | ModularPipeline.sub_blocks["decode"] | exists at ... | `latents` ← caller; `output_type="pil"` |
| create_noise_latent | ModularPipeline PrepareLatents step | ... | `height`/`width` ← `latents_source_shape[-2:]`; `batch_size=1`; `num_images_per_prompt=1`; `generator` ← seeded |
| add_noise_to_latent | ModularPipeline Img2Img steps | ... | enumerate every `required=True` `InputParam` (see Modular-Blocks-First rule); e.g. `latents` ← `self.create_noise_latent(...)`, `image_latents` ← caller latent, `batch_size` ← `latents.shape[0]`, `dtype` ← `_get_device_and_type()[1]`, `generator` ← seeded |
| denoise_latent | super() → DiffusionPipeline.__call__ | three blockers, see SKILL | n/a (delegated) |

## Provider classification (must match the Rule 1, Axis B approval)
- Shape: [new Provider] | [extend existing Provider]
- If new Provider:
    - Enum entry: `Provider.<NEW_NAME> = "<Display Name>"`
    - New `LatentPipelineTypeParameters` subclass: `<ClassName>`
    - `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` entry: `Provider.<NEW_NAME>` → `<ClassName>`
- If extending: target subclass = `<ExistingClassName>`; one new dict entry only.

## Runtime parameters class reuse
State whether this pipeline class gets a NEW runtime-parameters class or reuses an existing one. Siblings within a generation commonly share a runtime-params class (see existing reuse precedents in [`parameters/pipeline_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipeline_parameters.py) — e.g. `Flux2Pipeline` + `Flux2KleinPipeline`). If reusing, justify by showing the upstream `__call__` signatures are runtime-compatible.

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

## Latent contract compliance
- VAE whitening needed? [yes/no, with citation to the VAE config — `latents_mean` / `latents_std` for whitening; `scaling_factor` / `shift_factor` for the scalar path]
- Packing needed inside `prepare_input/output_latent`? [yes/no, citation to the pipeline class]
- Video? [yes/no → `produces_video`, `video_fps`]

## Variants (capability checklist)
- ControlNet supported? [yes/no] — if yes, plan to apply Pattern A from `/add-pipeline-variants`
- Inpaint supported? [yes/no] — if yes, plan to set `_inpaint_pipeline_class` (Pattern B)
- Runtime pipe-swap needed (e.g. conditional variant)? [yes/no] — if yes, plan Pattern C

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

Then STOP and wait for user confirmation.

### Phase C — Implementation (only after explicit approval)

State the verification check **before** making each change. A step is not complete until the verification passes.

All `uv run ...` commands below must be executed from the repo root. This repo is uv-backed and Windows-friendly; there is no `make` target.

1. **Provider enum entry** (only for the new-Provider shape) added to [`parameters/providers.py`](../../../modular_diffusion_nodes_library/parameters/providers.py) → verify: `uv run python -c "from modular_diffusion_nodes_library.parameters.providers import Provider; Provider.<NEW_NAME>"` exits 0.
2. **Standard parameters** file created → verify: `uv run ruff format --check` + `uv run ruff check .` pass for the new file.
3. **Runtime parameters** file created (or reuse declared) → verify: every parameter default and tooltip cites an upstream source per Rule 4; `uv run ruff format --check` + `uv run ruff check .` pass.
4. **Driver** file created → verify: file imports cleanly; every `_call_block(...)` site passes every `required=True` input enumerated in Phase B's table; latent contract (Rule 5) is honoured (unpacked, normalised, `latents_source_shape` is pixel-space); `uv run ruff format --check` + `uv run ruff check .` pass.
5. **Registration edits**:
   - Always: one entry in [`driver_factory.py`](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/driver_factory.py) `_DRIVER_REGISTRY`; one or more `case` clauses in [`pipeline_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipeline_parameters.py) `set_runtime_parameters`.
   - **New-Provider shape**: a new `Provider` enum entry in [`providers.py`](../../../modular_diffusion_nodes_library/parameters/providers.py), a new `LatentPipelineTypeParameters` subclass AND a `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` entry in [`pipelinetype_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipelinetype_parameters.py).
   - **Extend-existing shape**: ONE new entry inside an existing subclass's `get_pipeline_type_dict()` in [`pipelinetype_parameters.py`](../../../modular_diffusion_nodes_library/parameters/pipelinetype_parameters.py). No `providers.py` change. No `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` change.

   Verify:
   - `uv run python -c "from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class; assert get_driver_class('<PipelineClassName>') is not None"`
   - For the new-Provider shape only: `uv run python -c "from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import MODULAR_PIPELINE_TYPE_PROVIDER_MAP; from modular_diffusion_nodes_library.parameters.providers import Provider; assert MODULAR_PIPELINE_TYPE_PROVIDER_MAP[Provider.<NEW_NAME>] is not None"`
   - `uv run ruff format --check` and `uv run ruff check .` pass.

   **Surgical-edits check**: diff each shared file and confirm only added lines / one inserted block are present — no reformatted neighbours, no reordered entries, no touched sibling cases.
6. **Smoke test in UI** → verify with the user: drop a builder node, select the new provider/type, build, wire to generate + decode, confirm an image/video comes out.
7. **Node docs updated** (Rule 0) → verify: every node whose dynamic parameters now branch on the new pipeline type (typically Pipeline Builder, Generate Media Latents, and affected encode/decode/conditioning nodes) has its `docs/nodes/<name>.md` updated and passes the `/document-node` skill's drift check.

---

## References

- [`./references/driver-details.md`](./references/driver-details.md) — method-by-method driver implementation guide and the verbatim latent shape & space contract.
- [`../add-pipeline-variants/SKILL.md`](../add-pipeline-variants/SKILL.md) — load this for ControlNet / inpaint / runtime pipe-swap on the new driver.
- [`../../../docs/adding-new-model.md`](../../../docs/adding-new-model.md) — the human-readable contributor doc covering the same process.
