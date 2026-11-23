- The Transformer, Transformer 2, UNet and VAE ports on `Build Pipeline` now explain what a config file is
  and link to its Hugging Face page for the selected model, so a first-time GGUF or single-file user can
  download just that file instead of the full model. `Load Component` explains the same under
  `Config` parameter and says Text Encoder files need none. The **Component Overrides** group on
  `Build Pipeline` has a badge explaining what an override does and pointing to `Load Component` and
  `Load Scheduler`.
  https://github.com/griptape-ai/griptape-nodes-library-diffusers/issues/43
