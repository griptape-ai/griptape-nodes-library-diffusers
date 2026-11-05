- `Generate Media Latents` now picks a new seed on each run when `randomize_seed` is on, even when it only
  feeds the rest of the workflow by data wires. It reused its last result instead.
  [#127](https://github.com/griptape-ai/griptape-nodes-library-diffusers/issues/127)
