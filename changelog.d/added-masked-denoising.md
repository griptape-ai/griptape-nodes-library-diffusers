- `Generate Media Latents` supports masked denoising using the Differential Diffusion technique
  ([paper](https://arxiv.org/abs/2306.00950)). Connect the `Set Latent Mask` output to
  `input_latent` and the mask is applied at every denoising step instead of switching to an
  inpaint pipeline. White areas are redone, black areas stay close to the original, and grey areas
  are given less time to change, so edits blend smoothly. It works with Flux, Flux Kontext, Flux 2,
  Flux 2 Klein, Flux 3, Qwen Image, Qwen Image Edit, Stable Diffusion 3, SDXL, and Z-Image, and
  honours `add_noise`, `start_step`, and `end_step`. Other pipelines, schedulers that take more than
  one model call per step report an error when a masked latent is
  connected.
