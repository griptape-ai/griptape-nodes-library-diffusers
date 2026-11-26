- `Load Component` hides `Config (Optional)` when `Component` is `Text Encoder`, because a GGUF text
  encoder carries its own config. Its file picker lists only `.gguf` files, including when a saved
  workflow is reopened.
  [#43](https://github.com/griptape-ai/griptape-nodes-library-diffusers/issues/43)
