# Driver Implementation Details

Progressive-load reference for the `LatentPipelineDriver` subclass. Load this when designing or implementing the driver in step 4 of the add-model procedure.

---

## The Latent Shape & Space Contract (verbatim from `base_driver.py`)

> All latents flowing across the public driver surface (`create_noise_latent`, `encode_media`, `add_noise_to_latent`, the output of `denoise_latent`, and the input of `decode_latent`) share a single canonical shape **and** statistical space so they can be freely composited, masked, added, or otherwise manipulated by downstream nodes.
>
> **Shape.** Public latents are **unpacked** tensors (no model-specific patchifying / sequence packing). Their shape is:
> - 4-D for image pipelines: `[B, C, H // vae_scale_factor, W // vae_scale_factor]`
> - 5-D for video pipelines: `[B, C, T_latent, H // vae_scale_factor, W // vae_scale_factor]`
>
> where `C` (`num_channels_latents`) is inferred from the pipeline and `T_latent = (T_video - 1) // vae_scale_factor_temporal + 1` for video.
>
> The `source_shape` carried on `LatentArtifact` / on the input `ImageMedia` / `VideoMedia` / `MaskMedia` dataclass is in **pixel space**, **not** latent space:
> - 4-D for image: `[B, C_image, H_pixel, W_pixel]`
> - 5-D for video: `[B, C_image, T_video, H_pixel, W_pixel]`
>
> Drivers translate between pixel-space dims (`source_shape`) and divided latent-space tensor dims when calling underlying pipelines.
>
> **Space.** Latents are **normalised**: each channel ~N(0, 1), matching `torch.randn`. For VAEs whose raw latents are not unit-variance (e.g. WAN, Flux2), drivers must apply per-channel whitening `(z - latents_mean) / latents_std` inside `encode_media` and the inverse inside `decode_latent`.
>
> Any model-specific *packing* (e.g. Flux, Qwen) is applied transiently inside `prepare_input_latent` / `prepare_output_latent` and never appears on the public surface.

---

## Forwardable signature contract

Four methods — `encode_media`, `decode_latent`, `create_noise_latent`, `add_noise_to_latent` — form the cross-driver public surface. Their signatures MUST match [`FORWARDABLE_METHOD_POSITIONAL`](../../../../modular_diffusion_nodes_library/latent_pipeline_drivers/_base_driver_forwardable_signature.py) exactly:

```python
FORWARDABLE_METHOD_POSITIONAL: dict[str, tuple[str, ...]] = {
    "encode_media":        ("media", "generator_state"),
    "decode_latent":       ("latent",),
    "create_noise_latent": ("source_shape", "generator_state"),
    "add_noise_to_latent": ("latent", "generator_state", "num_inference_steps", "strength"),
}
```

Same parameter names, same order, same count. No extra positional or kw-only parameters. No `*args`/`**kwargs`. `LatentPipelineDriver.__init_subclass__` enforces this and raises `TypeError` at import time on any deviation.

**Why.** Callers invoke these uniformly across drivers; any model-specific tunable on the public surface would force every caller to branch on driver type. Route driver-specific tunables through the driver-namespaced sub-bag on `LatentArtifact.meta` instead (see next section).

---

## Media inputs and `LatentArtifact` exchange

Public methods exchange these types (defined in [`driver_types.py`](../../../../modular_diffusion_nodes_library/latent_pipeline_drivers/driver_types.py)):

- `ImageMedia(image, source_shape)` — PIL image or 4-D tensor with its pixel-space source shape.
- `VideoMedia(frames, source_shape)` — list of PIL frames with its pixel-space source shape.
- `MaskMedia(mask, source_shape)` — L-mode PIL mask paired with its source shape.
- `LatentArtifact` — the canonical latent carrier (see [`latent_artifact.py`](../../../../modular_diffusion_nodes_library/artifact_utils/latent_artifact.py)). `latent.source_shape`, `latent.to_torch(device, dtype)`, `latent.metadata` are the access points.

Image-only drivers should raise `NotImplementedError` for `VideoMedia` inputs to `encode_media`; video-only drivers do the reverse. Mirror SDXL's `encode_media` for the image-only guard and WAN's for the video-only guard.

Always build outputs via `self._make_latent_artifact(tensor, source_shape=..., upstream=..., meta=...)`. The helper:
- copies non-namespace meta from `upstream` (preserving user-set provenance),
- merges any same-driver namespaced meta from `upstream` with the new `meta` dict (new values win),
- stamps `META_DRIVER_KEY` → `self.driver_namespace` so cross-driver consumers can detect ownership.

---

## Driver-namespaced meta and `GeneratorState` round-trip

Driver-specific state lives in a sub-bag at `meta[<driver_namespace>]`. The two helpers you need from [`driver_types.py`](../../../../modular_diffusion_nodes_library/latent_pipeline_drivers/driver_types.py):

- `read_driver_meta(artifact, key, required_driver_name, default=None)` — returns the sub-bag value, scoped to this driver's namespace. Returns `default` if the artifact was produced by a different driver.
- `GeneratorState.from_artifact(artifact)` — returns the post-call `GeneratorState` stamped on the artifact, or `None` if absent or produced by a different driver.

**Generator round-trip.** All four forwardable methods (and `denoise_latent`) accept `generator_state: GeneratorState`. Build a generator with `generator_state.to_generator()`, then — for noise-producing and noise-advancing methods — stamp the post-call state onto the output:

```python
generator = generator_state.to_generator()
# ... use generator in block call ...
meta = {**GeneratorState.from_generator(generator).as_meta(), <other driver meta>}
return self._make_latent_artifact(tensor, source_shape=..., upstream=..., meta=meta)
```

Methods that stamp generator state:
- `create_noise_latent`
- `add_noise_to_latent`
- `denoise_latent` (base class already does this for you — don't duplicate)

**Method that does NOT stamp generator state:**
- `encode_media` — VAE encoding consumes the generator (for sampling from the encoder distribution) but does not advance an RNG chain that downstream nodes need to continue. Whether to carry any other driver-namespaced meta on the encoded output is the driver's call.

**Driver-specific tunables.** Anything you'd be tempted to add as a kw-only param on a forwardable method instead goes into the namespaced sub-bag. The canonical precedent is SDXL's `_KIND_META_KEY` (values `"noise"` / `"image_latents"` / `"noised_image_latents"`) in [`stable_diffusion_xl.py`](../../../../modular_diffusion_nodes_library/latent_pipeline_drivers/stable_diffusion_xl.py) — the SDXL `denoise_latent` reads the tag back via `read_driver_meta(latent, _KIND_META_KEY, self.driver_namespace, "")` to decide whether to apply `init_noise_sigma` scaling.

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

For concrete imports, scan existing drivers in [`modular_diffusion_nodes_library/latent_pipeline_drivers/`](../../../../modular_diffusion_nodes_library/latent_pipeline_drivers/).

If `<Model>AutoBlocks` doesn't exist, check the model's `modular_pipeline.py` for an `init_pipeline`-ready blocks class. If no modular pipeline exists yet for this model in diffusers, that is a blocker — surface it to the user.

### `can_make_control_pipe_from_standard(cls, control_net_model_lists) -> bool`

Classmethod. Return `False` unless ControlNet support is being implemented in this PR. If `True`, also implement `control_pipe_from_standard()`.

### `create_noise_latent(source_shape, generator_state) -> LatentArtifact`

Return pure noise latents in the **public contract** (unpacked, normalised, correct rank for image/video), wrapped in a `LatentArtifact` via `_make_latent_artifact`.

**Preferred — modular blocks**:

When a block depends on state set by a sibling block (e.g. `PrepareLatentsStep` reads `init_noise_sigma`, which is only valid after `SetTimestepsStep` runs), compose them with `SequentialPipelineBlocks` rather than calling them individually via `sub_blocks[...]`. Calling `prepare_latents` standalone will read stale scheduler state and silently produce wrong-scale noise. The pattern below is the canonical shape.

```python
class _MyPrepareNoiseLatentStep(SequentialPipelineBlocks):
    model_name = "..."
    block_classes = [<Model>SetTimestepsStep, <Model>PrepareLatentsStep]
    block_names = ["set_timesteps", "prepare_latents"]

@override
def create_noise_latent(
    self,
    source_shape: tuple[int, ...],
    generator_state: GeneratorState,
) -> LatentArtifact:
    generator = generator_state.to_generator()
    output_state = self._call_block(
        _MyPrepareNoiseLatentStep(),
        height=source_shape[-2],
        width=source_shape[-1],
        batch_size=1,
        num_images_per_prompt=1,
        num_inference_steps=self._DEFAULT_NUM_INFERENCE_STEPS,
        generator=generator,
    )
    latents = output_state.get("latents")
    # Apply any model-specific normalisation here (e.g. divide by init_noise_sigma for SDXL).
    meta = GeneratorState.from_generator(generator).as_meta()
    return self._make_latent_artifact(latents, source_shape=source_shape, meta=meta)
```

If the modular block returns packed latents, unpack here so the return value satisfies the public contract. `num_inference_steps` for noise creation is sourced from the `_DEFAULT_NUM_INFERENCE_STEPS` ClassVar on `LatentPipelineDriver` (default 20) — it is NOT a public parameter on `create_noise_latent` (that would violate the forwardable contract).

### `encode_media(media, generator_state) -> LatentArtifact`

Encode an `ImageMedia` or `VideoMedia` input to a public-contract latent. Image-only drivers raise `NotImplementedError` for `VideoMedia` and vice versa:

```python
@override
def encode_media(
    self,
    media: ImageMedia | VideoMedia,
    generator_state: GeneratorState,
) -> LatentArtifact:
    if isinstance(media, VideoMedia):
        raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video.")
    generator = generator_state.to_generator()
    encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
    output_state = self._call_block(encode_block, image=media.image, generator=generator)
    result = output_state.get("image_latents")
    # Apply per-channel whitening here if the VAE is non-unit-variance.
    return self._make_latent_artifact(result, source_shape=media.source_shape)
```

- Use `self.modular_pipe.blocks.sub_blocks["vae_encoder"]` if available.
- After encoding, apply per-channel whitening if the model's VAE is non-unit-variance: `(z - latents_mean) / latents_std`. The mean/std live on `self.pipe.vae.config.latents_mean` / `latents_std` (or `shift_factor`/`scaling_factor` depending on the VAE).
- **Do not double-apply scalar shift+scale.** Some pipelines' modular `vae_encoder` / `decode` sub-blocks already apply `shift_factor` / `scaling_factor` internally. Before adding any whitening in the driver, read the sub-block's `__call__` to confirm whether it's needed. If the sub-block already handles it, the driver should pass latents through untouched.
- Return unpacked.
- **Do NOT stamp `GeneratorState.from_generator(generator).as_meta()`.** Encoded image latents are not the head of an RNG chain that downstream nodes need to continue. Whether to carry any other driver-namespaced meta (e.g. a kind tag) is the driver's call.

Cite the VAE class in `.venv/Lib/site-packages/diffusers/models/autoencoders/` to confirm the whitening constants.

Video pipelines: dispatch on `isinstance(media, VideoMedia)` and route to a video-encode path that operates on `media.frames`. Set `produces_video = True` and `video_fps` as ClassVars on the driver.

### `decode_latent(latent: LatentArtifact) -> Image | list[Image] | np.ndarray`

Inverse of `encode_media`. Apply the inverse whitening **first**, then decode. `source_shape` is on the input artifact (`latent.source_shape`) — derive `height`/`width`/`num_frames` from it as needed.

```python
@override
def decode_latent(self, latent: LatentArtifact) -> Image:
    device, dtype = self._get_device_and_type()
    latents = latent.to_torch(device=device, dtype=dtype)
    decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
    output_state = self._call_block(decode_block, latents=latents, output_type="pil")
    return output_state.get("images")[0]
```

- Use `self.modular_pipe.blocks.sub_blocks["decode"]` if available.
- Return `PIL.Image` for image, `list[Image]` (frames) for video.
- Does NOT return a `LatentArtifact` — it returns the rendered media (this is the only forwardable method whose return type is `DecodeResult`, not `LatentArtifact`).

### `add_noise_to_latent(latent, generator_state, num_inference_steps, strength) -> LatentArtifact`

Forward-noise an existing latent for img2img/v2v. Preferred — modular `Img2ImgSetTimesteps` + `Img2ImgPrepareLatents`. `stable_diffusion_xl.py` `_SDXLAddNoiseStep` is the canonical no-packing shape; scan other existing drivers for packed-latent and video variants.

**Read the new pipeline's `Img2ImgPrepareLatentsStep` `inputs` property before calling `_call_block(...)`.** Block input contracts vary per pipeline — some treat `latents` as optional and generate noise internally when omitted; others mark `latents` as `required=True` and use it directly. Copying a sibling driver's `_call_block(...)` kwargs blindly is the most common source of `ValueError: Required input 'X' is missing` at runtime. See the "Read every block's `inputs` before calling `_call_block(...)`" rule in the main SKILL for the full pre-write checklist.

When the block requires `latents` (noise), source it via `self.create_noise_latent(...).to_torch(device, dtype)` rather than inline `torch.randn` so the noise goes through the scheduler-aware modular path and stays consistent with the driver's other noise generation.

Stamp the post-call `GeneratorState` onto the returned artifact (this method does advance the RNG chain). SDXL's implementation also writes a `_KIND_META_KEY` tag indicating whether the output is `"noise"` (degenerate input case) or `"noised_image_latents"` (normal case) — mirror that pattern when your driver needs to remember how a latent was produced.

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

Signature:

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
    # ... munge kwargs ...
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
```

Commonly seen reasons to override (scan the named driver for shape, don't copy it):

- img2img-style pipelines that consume `image` + `strength` directly — see `stable_diffusion_xl.py`.
- Video pipelines that need `num_frames` and/or constructed video conditions — see `ltx.py`, `wan.py`.
- Image-to-video pipelines that extract first/last frames from inputs — see `wan_i2v.py`.
- Edit pipelines that adjust latent dims for the edit task — see `qwen_edit.py`.

The base class handles: partial-denoise via `PartialDenoisePipelineRunner`, `callback_on_step_end` wiring, cancellation via `pipe._interrupt`, inpaint routing via `_inpaint_pipeline_class`, **and** stamping the post-call `GeneratorState` onto the returned artifact. Do not duplicate or bypass any of these.

### ClassVars

- `produces_video: ClassVar[bool] = True` — for video pipelines (default False).
- `video_fps: ClassVar[int] = 24` — output frame rate (default 16).
- `_inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = <YourInpaintPipeline>` — enables inpaint routing in base class.
- `_partial_denoise_proxy_class: ClassVar[type[PartialDenoiseSchedulerProxy]]` — override if the model's scheduler needs a custom proxy. Default is fine for FlowMatch / DDIM / DPM schedulers.
- `_DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 20` — the step count used inside `create_noise_latent` when the modular `PrepareLatents` block needs one. Override if the model's scheduler needs a different value to produce well-scaled noise.

---

## Justification Patterns (use these in the plan)

When proposing an implementation, justify like this:

- **"Modelling `decode_latent` after `FluxLatentPipelineDriver.decode_latent()` because both pipelines use packed transformer latents — see `flux.py` line N."**
- **"Modelling `encode_media` after `StableDiffusionXLLatentPipelineDriver.encode_media()` — image-only guard for `VideoMedia`, generator NOT stamped onto output meta."**
- **"Setting `_inpaint_pipeline_class = <InpaintPipelineClass>` because the diffusers class exists at `<verified path>`."**
- **"VAE whitening needed: `<Model>` VAE has `latents_mean=...` / `latents_std=...` per `<verified path>` — applied in `encode_media` and inverted in `decode_latent`."**
- **"VAE whitening NOT needed: `<Model>` uses scalar `shift_factor` / `scaling_factor` and the modular `vae_encoder` / `decode` sub-blocks apply it internally — verified at `<path>`. Driver passes latents through untouched."**
- **"Not overriding `denoise_latent` — no kwarg munging needed; base class handles everything via `super().denoise_latent()`."**

If a justification cannot be backed by a concrete file reference, the agent must not assert it. Surface the uncertainty to the user instead.
