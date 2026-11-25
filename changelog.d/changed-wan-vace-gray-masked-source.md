- **Breaking:** WAN VACE grays out the white (generate) region of `source_media` before generating
  when a `mask` is connected, which improves object removal. Existing masked workflows give
  different results; turn off the new `gray_masked_source` option to keep the original pixels for
  edits like recoloring.
  [#151](https://github.com/griptape-ai/griptape-nodes-library-diffusers/issues/151)
