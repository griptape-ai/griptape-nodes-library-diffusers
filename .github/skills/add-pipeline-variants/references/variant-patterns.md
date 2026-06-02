# Variant Pattern Templates

Progressive-load reference. Use when implementing one of the three runtime variant patterns.

---

## Pattern A — ControlNet (build-time variant)

**Trigger**: `control_net_model_lists: list[str] | str | None` is supplied to the pipeline builder. The two driver classmethods are called during pipeline construction, not during denoising.

**Reference**: [`stable_diffusion_xl.py`](../../../../diffusers_nodes_library/latent_pipeline_drivers/stable_diffusion_xl.py) lines 68-96.

### Template

```python
from diffusers import ControlNetModel, <Model>ControlNetPipeline  # verify both exist

class MyDriver(LatentPipelineDriver):

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        # Return True only for inputs you can actually handle.
        # If the model supports a single ControlNet but not multi-control, validate here.
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls,
        pipe: ModularPipeline | DiffusionPipeline,
        control_net_model_lists: list[str] | str | None,
    ) -> DiffusionPipeline:
        if not control_net_model_lists:
            return pipe

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        controlnet_torch_dtype = cls._get_torch_type(pipe)
        from_pretrained_kwargs: dict[str, Any] = {}
        if controlnet_torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = controlnet_torch_dtype

        control_net_models = [
            ControlNetModel.from_pretrained(cn, **from_pretrained_kwargs) for cn in control_net_model_lists
        ]
        controlnet = control_net_models[0] if len(control_net_models) == 1 else control_net_models

        return <Model>ControlNetPipeline.from_pipe(pipe, controlnet=controlnet)
```

### Pitfalls
- **Wrong ControlNet model class**: some models use `<Model>ControlNetModel` (e.g., `FluxControlNetModel`) instead of generic `ControlNetModel`. Read the variant pipeline's source to confirm which type its `controlnet` argument expects.
- **`from_pipe` with quantised base**: if the base pipeline is quantised (bnb-4bit), confirm `from_pipe` handles it; some variants require `torch_dtype=None` and inherit from the base.
- **No matching `<Model>ControlNetPipeline`**: if diffusers doesn't ship a controlnet pipeline for this model, ControlNet is not supportable yet. Return False from `can_make_control_pipe_from_standard` and skip Pattern A.

---

## Pattern B — Inpaint (denoise-time variant)

**Trigger**: `inpaint_mask_artifact` kwarg present in `denoise_latent` kwargs.
**Mechanism**: Base class handles everything. You declare one ClassVar.

**Reference**: [`base_driver.py`](../../../../diffusers_nodes_library/latent_pipeline_drivers/base_driver.py) `_get_inpaint_pipe()` and `_get_inpaint_kwargs()` (lines 284-302), and the `denoise_latent` inpaint branch (lines 232-247). SDXL example: [`stable_diffusion_xl.py`](../../../../diffusers_nodes_library/latent_pipeline_drivers/stable_diffusion_xl.py) line 63.

### Template

```python
from diffusers.pipelines.<model> import <Model>InpaintPipeline  # verify it exists

class MyDriver(LatentPipelineDriver):
    _inpaint_pipeline_class = <Model>InpaintPipeline
```

That is the entire implementation when the inpaint pipeline accepts the standard kwargs: `image`, `mask_image`, optional `masked_image_latents`, `strength`.

### When to override `_get_inpaint_kwargs`

Override only if the inpaint pipeline needs non-standard kwargs. Default in base class:

```python
def _get_inpaint_kwargs(self, artifact, latents, device, dtype):
    result = {
        "image": latents,  # base passes the encoded latent as 'image'
        "mask_image": artifact.mask_image,
    }
    if artifact.masked_latent is not None:
        result["masked_image_latents"] = artifact.masked_latent.to(device=device, dtype=dtype)
    return result
```

For packed-latent models (Flux, Qwen), the base passes unpacked latents — the inpaint pipeline does its own packing internally. If your inpaint pipeline doesn't accept latents as `image`, you must override and provide a decoded image. Cite the inpaint pipeline's signature when justifying.

### Pitfalls
- **No `<Model>InpaintPipeline`**: if diffusers doesn't ship one, inpaint isn't supportable yet. Leave `_inpaint_pipeline_class = None` (the base default).
- **Calling `super().denoise_latent` from an overridden `denoise_latent`**: the base class detects `inpaint_mask_artifact` itself. Do not duplicate detection in the override; just call `super()` and let it route.
- **SDXL-style image+strength setdefault**: SDXL's overridden `denoise_latent` skips the `image`/`strength` setdefault when `inpaint_mask_artifact` is present (line 162). Copy this guard in any similar img2img driver.

---

## Pattern C — Runtime pipe-swap (denoise-time variant)

**Trigger**: A custom input kwarg (e.g., `media_gen_conditioning` for LTX) detected at denoise time.
**Mechanism**: Override `denoise_latent`, swap `self._pipe` to a variant class, restore in `finally`, delegate to `super()`.

**Reference**: [`ltx.py`](../../../../diffusers_nodes_library/latent_pipeline_drivers/ltx.py) lines 260-294 — the canonical implementation using `LTXConditionPipeline`.

### Template

```python
from diffusers import <VariantPipelineClass>  # verify it exists

from diffusers_nodes_library.utils.pipeline_utils import create_pipe_variant

@override
def denoise_latent(
    self,
    latents: torch.Tensor,
    latents_source_shape: tuple[int, ...],
    num_inference_steps: int,
    seed: int = 0,
    callback: Any = None,
    start_step: int = 0,
    end_step: int = -1,
    return_fully_denoised: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    trigger = kwargs.pop("<custom_kwarg>", None)
    original_pipe = self._pipe

    if trigger is not None:
        torch_dtype = self._get_torch_type(self._pipe)
        self._pipe = create_pipe_variant(original_pipe, <VariantPipelineClass>, torch_dtype=torch_dtype)
        kwargs["<variant_kwarg>"] = self._convert_trigger(trigger, latents_source_shape)

    try:
        return super().denoise_latent(
            latents=latents,
            latents_source_shape=latents_source_shape,
            num_inference_steps=num_inference_steps,
            seed=seed,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )
    finally:
        self._pipe = original_pipe
```

### Critical invariants
- **`finally` is mandatory.** An exception during denoising must not leave `self._pipe` swapped. Cite this when justifying the implementation.
- **Use `create_pipe_variant`**, not `<VariantPipelineClass>.from_pipe()` directly. `create_pipe_variant` preserves the source pipeline's CPU/sequential offload state. See [`utils/pipeline_utils.py`](../../../../diffusers_nodes_library/utils/pipeline_utils.py) `create_pipe_variant` (around line 40).
- **`super().denoise_latent` does the actual call.** Do not call `self._pipe(...)` directly — you'd lose partial-denoise, callback, cancellation, and inpaint routing.
- **Pack/unpack inside the override** if the variant pipeline expects packed latents and the base pipeline didn't. See the LTX override at line 280: it calls `self._pack_latents(latents)` only when `media_gen_conditioning_list is None` (the variant path lets the LTXConditionPipeline handle its own packing).

### Pitfalls
- **Pop, don't get**: use `kwargs.pop("<custom_kwarg>", None)` so the custom kwarg doesn't end up in the pipeline call.
- **Single-variant-only assumption**: if a driver may need to swap to one of *multiple* variants depending on inputs, structure the swap as a single decision tree at the top of `denoise_latent`, not nested overrides.
- **Cache invalidation**: `create_pipe_variant` calls `<Variant>.from_pipe(source)`, which shares components but creates a new pipeline instance per call. If this is hot-path, consider caching the variant pipe in an instance attribute keyed by variant class.

---

## Combining patterns

A driver can implement A + B + C simultaneously (e.g., SDXL supports all three). They are orthogonal and don't conflict:
- A is classmethod, runs at build time
- B is a ClassVar + base-class auto-routing
- C is a `denoise_latent` override

When combining B and C, ensure the C override **does not pop or modify `inpaint_mask_artifact`** — the base class needs to see it.
