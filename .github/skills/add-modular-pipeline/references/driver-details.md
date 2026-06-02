# Driver Implementation Details

Progressive-load reference for the `LatentPipelineDriver` subclass. Load this when designing or implementing the driver in step 4 of the add-model procedure.

---

## The Latent Shape & Space Contract (verbatim from `base_driver.py`)

> All latents flowing across the public driver surface (`create_noise_latent`, `encode_image`, `encode_video`, `add_noise_to_latent`, the output of `denoise_latent`, and the input of `decode_latent`) share a single canonical shape **and** statistical space so they can be freely composited, masked, added, or otherwise manipulated by downstream nodes.
>
> **Shape.** Public latents are **unpacked** tensors (no model-specific patchifying / sequence packing). Their shape is:
> - 4-D for image pipelines: `[B, C, H // vae_scale_factor, W // vae_scale_factor]`
> - 5-D for video pipelines: `[B, C, T_latent, H // vae_scale_factor, W // vae_scale_factor]`
>
> where `C` (`num_channels_latents`) is inferred from the pipeline and `T_latent = (T_video - 1) // vae_scale_factor_temporal + 1` for video.
>
> The `latents_source_shape` argument is in **pixel space**, **not** latent space:
> - 4-D for image: `[B, C_image, H_pixel, W_pixel]`
> - 5-D for video: `[B, C_image, T_video, H_pixel, W_pixel]`
>
> Drivers translate between pixel-space dims (`latents_source_shape`) and divided latent-space tensor dims when calling underlying pipelines.
>
> **Space.** Latents are **normalised**: each channel ~N(0, 1), matching `torch.randn`. For VAEs whose raw latents are not unit-variance (e.g. WAN, Flux2), drivers must apply per-channel whitening `(z - latents_mean) / latents_std` inside `encode_image`/`encode_video` and the inverse inside `decode_latent`.
>
> Any model-specific *packing* (e.g. Flux, Qwen) is applied transiently inside `prepare_input_latent` / `prepare_output_latent` and never appears on the public surface.

---

## Abstract Methods — What Each Must Do

### `_create_modular_pipe() -> ModularPipeline`

Return the diffusers `ModularPipeline` for this model. Typical pattern:

```python
from diffusers.modular_pipelines.<model> import <Model>AutoBlocks

@override
def _create_modular_pipe(self) -> ModularPipeline:
    return <Model>AutoBlocks().init_pipeline()
```

The base class then injects `pipe.components` into the modular pipe automatically.

For concrete imports, scan existing drivers in [`diffusers_nodes_library/latent_pipeline_drivers/`](../../../../diffusers_nodes_library/latent_pipeline_drivers/).

If `<Model>AutoBlocks` doesn't exist, check the model's `modular_pipeline.py` for an `init_pipeline`-ready blocks class. If no modular pipeline exists yet for this model in diffusers, that is a blocker — surface it to the user.

### `can_make_control_pipe_from_standard(cls, control_net_model_lists) -> bool`

Classmethod. Return `False` unless ControlNet support is being implemented in this PR. If `True`, also implement `control_pipe_from_standard()`.

### `create_noise_latent(latents_source_shape, seed, num_inference_steps=20) -> torch.Tensor`

Return pure noise latents in the **public contract** (unpacked, normalised, correct rank for image/video).

**Preferred — modular blocks**:

When a block depends on state set by a sibling block (e.g. `PrepareLatentsStep` reads `init_noise_sigma`, which is only valid after `SetTimestepsStep` runs), compose them with `SequentialPipelineBlocks` rather than calling them individually via `sub_blocks[...]`. Calling `prepare_latents` standalone will read stale scheduler state and silently produce wrong-scale noise. The pattern below is the canonical shape.

```python
class _MyPrepareNoiseLatentStep(SequentialPipelineBlocks):
    model_name = "..."
    block_classes = [<Model>SetTimestepsStep, <Model>PrepareLatentsStep]
    block_names = ["set_timesteps", "prepare_latents"]

@override
def create_noise_latent(self, latents_source_shape, seed, num_inference_steps=20):
    output_state = self._call_block(
        _MyPrepareNoiseLatentStep(),
        height=latents_source_shape[-2],
        width=latents_source_shape[-1],
        batch_size=1,
        num_images_per_prompt=1,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    )
    return output_state.get("latents")
```

If the modular block returns packed latents, unpack here so the return value satisfies the public contract.

### `encode_image(image) -> torch.Tensor`

Encode a PIL/tensor image to a public-contract latent.

- Use `self.modular_pipe.blocks.sub_blocks["vae_encoder"]` if available.
- After encoding, apply per-channel whitening if the model's VAE is non-unit-variance: `(z - latents_mean) / latents_std`. The mean/std live on `self.pipe.vae.config.latents_mean` / `latents_std` (or `shift_factor`/`scaling_factor` depending on the VAE).
- **Do not double-apply scalar shift+scale.** Some pipelines' modular `vae_encoder` / `decode` sub-blocks already apply `shift_factor` / `scaling_factor` internally. Before adding any whitening in the driver, read the sub-block's `__call__` to confirm whether it's needed. If the sub-block already handles it, the driver should pass latents through untouched.
- Return unpacked.

Cite the VAE class in `.venv/Lib/site-packages/diffusers/models/autoencoders/` to confirm the whitening constants.

### `decode_latent(latents, latents_source_shape) -> Image | list[Image]`

Inverse of `encode_image`. Apply the inverse whitening **first**, then decode.

- Use `self.modular_pipe.blocks.sub_blocks["decode"]` if available.
- Return PIL.Image for image, `list[Image]` (frames) for video.

### `add_noise_to_latent(latents, latents_source_shape, seed, num_inference_steps, strength) -> torch.Tensor`

Forward-noise an existing latent for img2img/v2v. Preferred — modular `Img2ImgSetTimesteps` + `Img2ImgPrepareLatents`. `stable_diffusion_xl.py` `_SDXLAddNoiseStep` is the canonical no-packing shape; scan other existing drivers for packed-latent and video variants.

**Read the new pipeline's `Img2ImgPrepareLatentsStep` `inputs` property before calling `_call_block(...)`.** Block input contracts vary per pipeline — some treat `latents` as optional and generate noise internally when omitted; others mark `latents` as `required=True` and use it directly. Copying a sibling driver's `_call_block(...)` kwargs blindly is the most common source of `ValueError: Required input 'X' is missing` at runtime. See the "Read every block's `inputs` before calling `_call_block(...)`" rule in the main SKILL for the full pre-write checklist.

When the block requires `latents` (noise), source it via `self.create_noise_latent(...)` rather than inline `torch.randn` so the noise goes through the scheduler-aware modular path and stays consistent with the driver's other noise generation.

### Optional: `encode_video(frames) -> torch.Tensor`

Required for video pipelines. Default raises `NotImplementedError`.

### Optional: `encode_prompt(prompt, negative_prompt, **kwargs)`

Default works for any modular pipeline with a `text_encoder` sub-block accepting `prompt` (and optionally `negative_prompt`). **Inspect the `text_encoder` sub-block's `inputs` before deciding to override** — if it declares extra required inputs the default won't supply, you must override and pass them. Commonly seen extras (scan the named driver for shape, don't copy it):

- Secondary prompt (e.g. `prompt_2`) — see `flux.py`, `stable_diffusion_3.py`.
- Image conditioning for edit models — see `qwen_edit.py`, `flux_fill.py`.

The block contract is the only source of truth — the list above is illustrative, not exhaustive.

---

## Optional Overrides

### `prepare_input_latent(latents, latents_source_shape)` / `prepare_output_latent(...)`

Default is identity. Override whenever the pipeline's transformer expects a tensor shape that differs from the unpacked public-contract latent — inspect the pipeline's `prepare_latents` / `_pack_latents` / `__call__` methods (or the transformer's expected input shape) to decide.

- Common case: transformer-side sequence packing to `[B, seq_len, C*p*p]`. When the pipeline ships `_pack_latents` / `_unpack_latents` helpers (typically in `diffusers.modular_pipelines.<model>.decoders` or the pipeline file), reuse them and cite the helper path in your justification. For shape, see `flux.py`, `qwen.py`, `stable_diffusion_3.py`.
- These methods are called inside `denoise_latent()` — any transient reshaping is invisible to public callers.

### `_extract_latents_from_output(pipe_output)`

Default returns `pipe_output.images`. Override whenever the pipeline's output object exposes the latent tensor under a different attribute — inspect the pipeline's output class (e.g. `<Model>PipelineOutput` in `diffusers/pipelines/<model>/`) and return the correct field. Common attributes include `.frames` for video pipelines, but the only source of truth is the output class itself.

### `denoise_latent(...)`

**Only override to munge kwargs.** Always end with `return super().denoise_latent(...)`. Inspect the underlying `DiffusionPipeline.__call__` signature for the new model to determine what (if anything) needs translating between the driver's public kwargs and the pipeline's expected kwargs — only override when there's a concrete mismatch.

Commonly seen reasons to override (scan the named driver for shape, don't copy it):

- img2img-style pipelines that consume `image` + `strength` directly — see `stable_diffusion_xl.py`.
- Video pipelines that need `num_frames` and/or constructed video conditions — see `ltx.py`, `wan.py`.
- Image-to-video pipelines that extract first/last frames from inputs — see `wan_i2v.py`.
- Edit pipelines that adjust latent dims for the edit task — see `qwen_edit.py`.

The base class handles: partial-denoise via `PartialDenoisePipelineRunner`, `callback_on_step_end` wiring, cancellation via `pipe._interrupt`, and inpaint routing via `_inpaint_pipeline_class`. Do not duplicate or bypass any of these.

### ClassVars

- `produces_video: ClassVar[bool] = True` — for video pipelines (default False).
- `video_fps: ClassVar[int] = 24` — output frame rate (default 16).
- `_inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = <YourInpaintPipeline>` — enables inpaint routing in base class.
- `_partial_denoise_proxy_class: ClassVar[type[PartialDenoiseSchedulerProxy]]` — override if the model's scheduler needs a custom proxy. Default is fine for FlowMatch / DDIM / DPM schedulers.

---

## Justification Patterns (use these in the plan)

When proposing an implementation, justify like this:

- **"Modelling `decode_latent` after `FluxLatentPipelineDriver.decode_latent()` because both pipelines use packed transformer latents — see `flux.py` line N."**
- **"Setting `_inpaint_pipeline_class = <InpaintPipelineClass>` because the diffusers class exists at `<verified path>`."**
- **"VAE whitening needed: `<Model>` VAE has `latents_mean=...` / `latents_std=...` per `<verified path>` — applied in `encode_image` and inverted in `decode_latent`."**
- **"VAE whitening NOT needed: `<Model>` uses scalar `shift_factor` / `scaling_factor` and the modular `vae_encoder` / `decode` sub-blocks apply it internally — verified at `<path>`. Driver passes latents through untouched."**
- **"Not overriding `denoise_latent` — no kwarg munging needed; base class handles everything via `super().denoise_latent()`."**

If a justification cannot be backed by a concrete file reference, the agent must not assert it. Surface the uncertainty to the user instead.
