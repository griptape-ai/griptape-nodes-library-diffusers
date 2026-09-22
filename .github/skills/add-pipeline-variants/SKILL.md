---
name: add-pipeline-variants
description: 'Add a runtime variant (ControlNet, inpaint, or input-conditional pipe swap) to an EXISTING driver in modular_diffusion_nodes_library. DEFAULT TO THIS SKILL FIRST when adding any new capability to an existing model. The criterion is: can `<NewPipelineClass>.from_pipe(base_pipe)` produce a working pipeline from the components already loaded on the base pipe (UNet/transformer, VAE, text encoders, scheduler)? If yes — use this skill. Examples: ControlNet (any model), inpaint (any model), LTX media_gen_conditioning → LTXConditionPipeline. DO NOT use when from_pipe cannot produce a working pipeline (variant class needs a component the base does not have, or differently-shaped/-trained weights for an existing component) — that is a new pipeline type and belongs in add-modular-pipeline.'
---

# Add Pipeline Variants to an Existing Driver

This skill adds a **runtime variant** to an existing `LatentPipelineDriver` — a capability that shares the base pipeline's loaded weights and is activated dynamically (at build time or per-call) by detecting a specific input.

## Working Principles

These principles apply to every rule and phase below. Each Rule that follows is a concrete application of one or more of these.

1. **Think before coding.** State assumptions. If unsure whether the request is a variant or a new pipeline type, surface it and ask — do not pick silently. If the variant pattern (A / B / C) is not obvious from the trigger mechanism, ask. Stop when confused; name what is unclear.
2. **Simplicity first.** Pattern B (inpaint) is one line; do not turn it into ten by adding speculative `_get_inpaint_kwargs` overrides that aren't needed. Do not introduce helper classmethods, base mixins, or config knobs that weren't requested. Mirror the closest existing variant of the same Pattern.
3. **Surgical changes.** Variants almost always touch one driver file. Edit only the lines needed for the new variant. Do not reformat the rest of the driver, do not reorder existing methods, do not "tidy" sibling drivers — even if you notice issues. Match existing style. Mention unrelated issues in chat; do not silently fix them.
4. **Goal-driven execution.** Phase C states verification checks before changes. A variant is not done until: (Pattern A) the controlnet-wrapped pipe is built when `control_net_model_lists` is supplied; (Pattern B) `inpaint_mask_artifact` in kwargs routes through the inpaint pipeline class; (Pattern C) `self._pipe` is restored in `finally` after the swap.

## ⛔ Mandatory Workflow Rules

### Rule −1 — Update affected node docs in the same change

Variants don't add nodes, but they change runtime branches on existing ones. Update the `Provider / model behavior` section of every node whose dynamic parameters or runtime path changed — usually Generate Media Latents, the affected encode/decode node, and the conditioning node that triggers the variant. Document the trigger plainly (e.g. "supplying `control_net_model_lists` swaps in `<X>ControlNetPipeline`"). Follow [docs/node-doc-format.md](../../../docs/node-doc-format.md); for anything beyond a one-line tweak, use [.github/skills/document-node/SKILL.md](../document-node/SKILL.md). Report doc changes alongside code in your final summary.

### Rule 0 — Variant vs new pipeline type (FIRST CHECK)

**Default to variant.** Try this skill first; only fall back to `/add-modular-pipeline` when the test below fails.

**The test**: can `<VariantPipelineClass>.from_pipe(base_pipe)` produce a *correct* and *working* pipeline using the components already loaded on `base_pipe` (UNet/transformer, VAE, text encoders, scheduler)?

| Answer | Decision |
|---|---|
| **Yes** — the variant class shares the same component architectures and runs against the loaded weights | **Variant** → continue with this skill |
| **No** — the variant class needs a component the base pipeline does not have, or needs differently-shaped/-trained weights for an existing component (different UNet/transformer weights, an extra image encoder, etc.) | **New pipeline type** → STOP and use `/add-modular-pipeline` instead |

**Examples**
- ControlNet for SDXL: variant. `StableDiffusionXLControlNetPipeline.from_pipe(pipe, controlnet=...)` works against the loaded SDXL components (the ControlNet is supplied separately). → this skill.
- SDXL Inpaint: variant. `StableDiffusionXLInpaintPipeline.from_pipe(pipe)` works against the loaded SDXL components. A separate inpaint checkpoint also exists in the wild, but that is orthogonal — users who want it load it as a regular SDXL build. → this skill.
- LTX with `media_gen_conditioning`: variant. `LTXConditionPipeline.from_pipe(pipe)` works against the loaded LTX components. → this skill.
- **WAN I2V**: **NOT** a variant. `WanImageToVideoPipeline` requires different UNet weights and an image-encoder component the T2V pipe does not have; `from_pipe(t2v)` cannot produce it. → `/add-modular-pipeline`.
- **Flux Fill**: **NOT** a variant. Different transformer weights from Flux base; `from_pipe` cannot produce a working Fill pipeline from a Flux base pipe. → `/add-modular-pipeline`.
- **Qwen Edit**: **NOT** a variant. Different transformer weights and extra image conditioning component. → `/add-modular-pipeline`.

If you cannot verify the `from_pipe` test from the diffusers source, ask the user. Do not guess.

### Rule 1 — Plan and wait for confirmation

Produce a full plan covering:
- Variant pattern being implemented (one of: ControlNet, inpaint, runtime pipe-swap)
- Existing driver to modify (absolute path)
- Diffusers variant pipeline class to be used (verified to exist under `.venv/Lib/site-packages/diffusers/`)
- Files to modify with absolute paths
- For runtime pipe-swap: the trigger kwarg and the swap/restore code structure

STOP and wait for explicit confirmation before writing code.

### Rule 2 — Variant runtime defaults follow the upstream rule too

Any new runtime input added by a variant (e.g. `controlnet_conditioning_scale`, `strength`, `control_guidance_start`/`control_guidance_end`, inpaint `strength`) MUST take its default from the **variant pipeline's** `__call__` signature or `EXAMPLE_DOC_STRING` under `.venv/Lib/site-packages/diffusers/pipelines/<model>/`, and MUST have a clear, descriptive tooltip sourced from the variant pipeline's `__call__` docstring, model card, `EXAMPLE_DOC_STRING`, or a fusion of those — same standard as Rule 2 of [`/add-modular-pipeline`](../add-modular-pipeline/SKILL.md). Do not copy defaults or tooltips across drivers, do not invent values, do not round to a "nice number".

### Rule 3 — Adhere to the LatentPipelineDriver contract

Variants must preserve the public latent contract in [`base_driver.py`](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/base_driver.py) docstring:
- Latents passed across the public surface remain **unpacked + normalised**
- Any packing required by the variant pipeline happens inside `_prepare_input_latent`/`prepare_output_latent` (or transiently inside the overridden `denoise_latent`), never on the public surface
- Never bypass `super().denoise_latent()` for callback / partial-denoise / cancellation concerns

### Rule 4 — Prove every claim with code references

Every implementation choice MUST cite a file path. Variant pipeline classes MUST be verified to exist by reading `.venv/Lib/site-packages/diffusers/` before being referenced. Do not invent class names.

### Rule 5 — Variants normally do NOT need their own memory-estimation family entry

Memory estimation looks up `_FAMILY_BY_PIPELINE_NAME` / `_DENOISER_FIELD_ADAPTERS` in [`memory_estimation/family_registry.py`](../../../modular_diffusion_nodes_library/memory_estimation/family_registry.py) by `artifact.pipeline_name`, which flows into `estimate_pipeline_memory_from_artifact()` → `estimate_pipeline_memory()` / `estimate_pipeline_memory_from_build_data()` in [`pipeline_memory_estimator.py`](../../../modular_diffusion_nodes_library/memory_estimation/pipeline_memory_estimator.py). Verified in [`pipeline_artifact.py`](../../../modular_diffusion_nodes_library/artifact_utils/pipeline_artifact.py): `DiffusionPipelineArtifact.pipeline_name` is set once at construction (`latent_diffusion_pipeline_builder_node.py` line 180, from the *base* `pipeline_params.pipeline_name`), and `BaseDiffusionPipelineArtifact.__init__` (line 309) — the base class for `ControlNetDiffusionPipelineArtifact` — always forwards `pipeline_name=base_artifact.pipeline_name`. No code path ever constructs an artifact with the variant pipeline class's own name:

- **Pattern A (ControlNet)**: `ControlNetDiffusionPipelineArtifact` inherits `pipeline_name` unchanged from its `base_artifact` (`pipeline_artifact.py` line 309). The estimator sees the base pipeline's name even though the built `pipe` object is the `<Model>ControlNetPipeline`-wrapped instance — this is safe because `from_pipe()` carries over the same `unet`/`transformer` component reference the base adapter reads.
- **Pattern B (inpaint)**: `_get_inpaint_pipe()` (`base_driver.py` line 522) swaps `self.pipe` transiently inside `denoise_latent()`; it never touches the artifact, so `artifact.pipeline_name` is untouched.
- **Pattern C (runtime pipe-swap)**: same story — `ltx.py` line 385 and `qwen_edit.py` line 136 reassign `self._pipe`, not the artifact's `pipeline_name`.

**Conclusion**: do not add a `_FAMILY_BY_PIPELINE_NAME` / `_DENOISER_FIELD_ADAPTERS` entry for a variant pipeline class as part of this skill's normal flow, because `get_memory_family()` / `get_denoiser_config_fields()` are always called with the base pipeline's registered name, and the variant shares the base's denoiser weights/config by construction (Rule 0's `from_pipe` test). If you are implementing a case where the variant pipeline class name genuinely becomes its own `artifact.pipeline_name` — i.e. you are giving it a distinct `_DRIVER_REGISTRY` entry and a separate builder path rather than reusing the base pipeline's artifact — that is no longer a pure runtime variant in this skill's sense; stop and re-classify against `/add-modular-pipeline` Rule 9 instead of adding an ad-hoc family entry here.

---

## The Three Variant Patterns

### Pattern A — ControlNet (build-time variant)

**Trigger**: `control_net_model_lists` argument supplied during pipeline build.
**Mechanism**: Two classmethods on the driver, called by the pipeline builder.

Override:
- `can_make_control_pipe_from_standard(cls, control_net_model_lists) -> bool` — return True if you support ControlNet for this list
- `control_pipe_from_standard(cls, pipe, control_net_model_lists)` — load `ControlNetModel`(s) and return `<Model>ControlNetPipeline.from_pipe(pipe, controlnet=...)`

Reference: [stable_diffusion_xl.py](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/stable_diffusion_xl.py) lines 68-96 — canonical implementation.

### Pattern B — Inpaint (denoise-time variant, zero code)

**Trigger**: `inpaint_mask_artifact` kwarg present at denoise time.
**Mechanism**: Set a ClassVar; the base class handles everything.

```python
class MyDriver(LatentPipelineDriver):
    _inpaint_pipeline_class = <Model>InpaintPipeline
```

That's it. The base class `denoise_latent` in [base_driver.py](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/base_driver.py) checks for `inpaint_mask_artifact` in kwargs, calls `_get_inpaint_pipe()` (which uses `create_pipe_variant` under the hood), and passes the right kwargs via `_get_inpaint_kwargs()`.

Override `_get_inpaint_kwargs()` only if your inpaint pipeline needs non-standard kwargs beyond `image`, `mask_image`, and `masked_image_latents`.

Reference: [stable_diffusion_xl.py](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/stable_diffusion_xl.py) line 63 — `_inpaint_pipeline_class = StableDiffusionXLInpaintPipeline`.

### Pattern C — Runtime pipe-swap (denoise-time variant)

**Trigger**: A custom input kwarg at denoise time (e.g., `media_gen_conditioning` for LTX).
**Mechanism**: Override `denoise_latent`, detect kwarg, swap `self._pipe`, restore in `finally`, then delegate to `super()`.

Reference: [ltx.py](../../../modular_diffusion_nodes_library/latent_pipeline_drivers/ltx.py) lines 260-294.

Skeleton:
```python
@override
def denoise_latent(
    self,
    latent: LatentArtifact | InpaintMaskArtifact,
    num_inference_steps: int,
    generator_state: GeneratorState,
    callback: Any = None,
    start_step: int = 0,
    end_step: int = -1,
    return_fully_denoised: bool = False,
    **kwargs: Any,
) -> LatentArtifact:
    trigger_value = kwargs.pop("<custom_kwarg>", None)
    original_pipe = self._pipe
    if trigger_value is not None:
        torch_dtype = self._get_torch_type(self._pipe)
        self._pipe = create_pipe_variant(original_pipe, <VariantPipelineClass>, torch_dtype=torch_dtype)
        # convert trigger_value into the variant pipeline's kwargs
        kwargs["<variant_kwarg>"] = self._convert(trigger_value, ...)
    try:
        return super().denoise_latent(
            latent,
            num_inference_steps=num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )
    finally:
        self._pipe = original_pipe
```

The `finally` block is non-negotiable — failing to restore `self._pipe` leaves the driver in a corrupted state across calls.

---

## Procedure

### Phase A — Investigation

1. Confirm variant vs new pipeline type per Rule 0. If unsure, ask the user.
2. Read the existing driver to be modified, end-to-end.
3. Identify the variant pipeline class in diffusers. Verify it exists under `.venv/Lib/site-packages/diffusers/pipelines/<model>/`. Cite the file.
4. Confirm `from_pipe()` is available on the variant class (it is for all standard diffusers pipelines).
5. Load [`./references/variant-patterns.md`](./references/variant-patterns.md) for the detailed pattern reference.

### Phase B — Present the plan

```
# Plan: Add <Variant> to <Driver>

## Variant classification
- Same model weights as base pipeline? YES — proceed with variant skill.
- Diffusers variant class: <ClassName> at <verified path>
- Trigger: <build-time arg | inpaint_mask_artifact kwarg | custom denoise kwarg>
- Pattern: A (ControlNet) | B (Inpaint) | C (Runtime pipe-swap)

## Files to modify
- modular_diffusion_nodes_library/latent_pipeline_drivers/<driver>.py

## Implementation
- <Concrete code changes, citing the reference pattern>

## Verification
- <How to smoke-test, e.g., wire ControlNet input on builder node, etc.>
```

STOP. Wait for confirmation.

### Phase C — Implementation

State the verification check **before** making each change.

1. **Apply the variant pattern verbatim** from the reference (Pattern A / B / C) → verify: the new code matches the reference shape; for Pattern C, the `finally` block restores `self._pipe`; `make check` passes. **Surgical-edits check**: diff the driver and confirm only the variant-related lines changed — no reformatted methods, no reordered code, no "improved" sibling code.
2. **Smoke test in UI** → verify with the user: (Pattern A) wire a controlnet input on the builder node and confirm the wrapped pipe builds; (Pattern B) supply an inpaint mask and confirm the inpaint pipeline class is used at denoise time; (Pattern C) confirm the trigger kwarg activates the variant pipeline and `self._pipe` is restored on success and on error.
3. **Single-file loader metadata** → if the variant pipeline class is a fresh entry in `_DRIVER_REGISTRY` and users may load its transformer from a single `.gguf`/`.safetensors` file, add the relevant `model_type` → variant-pipeline-class-name entries to [`component_loading/pipeline_type_registry.py`](../../../modular_diffusion_nodes_library/component_loading/pipeline_type_registry.py) `MODEL_TYPE_TO_PIPELINE_TYPE`. Skip this step for pure runtime pipe-swaps that share the base pipeline's transformer weights.
4. **Memory-estimation family registry — confirm no entry is needed (Rule 5)** → verify: `artifact.pipeline_name` for the built pipeline is unchanged from the base pipeline's registered name (inspect the artifact constructed by the pipeline builder node for this variant, or trace `ControlNetDiffusionPipelineArtifact.metadata`/`base_artifact.pipeline_name` for Pattern A); confirm no new key was added to `_FAMILY_BY_PIPELINE_NAME` / `_DENOISER_FIELD_ADAPTERS` in [`memory_estimation/family_registry.py`](../../../modular_diffusion_nodes_library/memory_estimation/family_registry.py). If the variant work you implemented turned out to require the variant pipeline class to carry its own distinct `pipeline_name` (i.e. it stopped being a pure runtime variant per Rule 0), stop and re-classify against `/add-modular-pipeline` Rule 9 instead of registering it here.

---

## References

- [`./references/variant-patterns.md`](./references/variant-patterns.md) — detailed templates for each pattern, including kwargs handling and pitfalls.
- [`../add-modular-pipeline/SKILL.md`](../add-modular-pipeline/SKILL.md) — use this instead when adding a new pipeline type with different weights.
- [`../../../docs/adding-new-model.md`](../../../docs/adding-new-model.md) — the broader contributor doc covering full new-model integration.
