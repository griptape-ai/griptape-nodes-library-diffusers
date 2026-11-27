- `Generate Latent` with an LTX-2 IC-LoRA, HDR included, no longer fails when the reference video's
  frame count isn't `1 + 8k` (for example 120 frames). The reference is padded by repeating its last
  frame up to the next valid count, or trimmed to fit the output length.
  [#150](https://github.com/griptape-ai/griptape-nodes-library-diffusers/issues/150)
