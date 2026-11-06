- **Breaking:** Decode HDR Media no longer saves EXR files itself. Workflows using its EXR
  settings must connect the new `raw_media` output to Save EXR Sequence; tone-mapped previews remain
  available from the decoder.
