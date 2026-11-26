- A single file (`.gguf`, `.safetensors`, ...) loaded through `Load Component` now takes its config
  from the `Build Pipeline` node's model when `Config (Optional)` is blank, instead of looking for
  the wrong model's config.
