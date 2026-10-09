---
paths:
  - "modular_diffusion_nodes_library/latent_pipeline_drivers/**"
  - "modular_diffusion_nodes_library/nodes/**"
  - "modular_diffusion_nodes_library/misc/partial_denoise.py"
---

# Driver contract and modular-blocks-first

**Driver contract — read this before touching a driver**

`LatentPipelineDriver` defines the **public latent surface** that all nodes operate on:

> Public latents are **unpacked** (4-D image `[B, C, H/vae, W/vae]`, 5-D video `[B, C, T_lat, H/vae, W/vae]`) and **normalised** (~N(0,1)). Per-VAE whitening `(z - mean) / std` is applied inside `encode_image/encode_video`; the inverse runs inside `decode_latent`. Model-specific packing (Flux, Qwen) is applied transiently in `_prepare_input_latent` / `prepare_output_latent` and never appears on the public surface.

If you break this invariant, **every downstream node breaks** (latent math, composite, save/load, upscaler). Match the closest existing driver and copy its structure.

**Modular-blocks-first philosophy**

The long-term goal is to use the diffusers `ModularPipeline` block system for ALL pipeline operations. When integrating a new model:

- **Encode / decode / create-noise / add-noise / encode-prompt** → use modular blocks via `self.modular_pipe.blocks.sub_blocks[...]` and `self._call_block(...)`. Every existing driver does this.
- **Denoise loop** → fall back to `DiffusionPipeline.__call__()` via the base class `denoise_latent()`. Three blockers prevent modular denoise today:
    1. **Partial denoise** — `PartialDenoisePipelineRunner` / `PartialDenoiseSchedulerProxy` in `misc/partial_denoise.py` patch `pipe.scheduler.set_timesteps()`. No equivalent on `ModularPipeline`.
    2. **Callback / preview** — step-end previews and progress use `callback_on_step_end` passed via `pipe_kwargs`. Modular denoise blocks don't expose this hook.
    3. **Cancellation** — implemented by setting `pipe._interrupt = True` inside the step callback. `ModularPipeline` has no equivalent.
- **Only override `denoise_latent()` for model-specific kwarg munging** (e.g. LTX building video conditions, WAN i2v extracting first/last frames). Always end with `super().denoise_latent(...)` so partial-denoise, callback, cancellation, and inpaint routing keep working.
