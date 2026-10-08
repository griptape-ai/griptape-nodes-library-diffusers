# Save EXR Sequence

**Writes decoded float RGB media as a numbered OpenEXR frame sequence.**

Category: `ModularDiffusion/IO`

## TL;DR
- Connect `raw_media` (e.g. from Decode HDR Latents) to save one or more RGB frames as OpenEXR files.
- Choose no conversion for already-linear input, Log-Gamma inversion, or ARRI LogC3 inversion; Log-Gamma is selected by default.
- Files are named `<file_stem>.0001.exr`, `<file_stem>.0002.exr`, and so on in the selected folder.

## Typical workflow position
```text
Generate Media Latents → Decode HDR Latents → [Save EXR Sequence]
```

## Node preview

<!-- TODO: add docs/assets/nodes/save-exr-sequence.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `media` | `FloatMediaArtifact` | Yes | Float RGB image or video frames from Decode HDR Latents. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `saved_paths` | `list[str]` | Ordered filesystem paths of the written EXR frames. |
| `progress` | float | Per-frame write progress. |
| `logs` | str | Per-frame paths and file sizes. |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `transfer_function` | `None` \| `Log-Gamma` \| `ARRI LogC3` | `Log-Gamma` | Converts encoded RGB values to linear values before export, or writes already-linear values unchanged. |
| `output_folder` | path | workspace directory | Destination directory; created if it does not exist. |
| `file_stem` | str | `frame` | Filename prefix without a directory or extension. |
| `save_as_half_float` | bool | `True` | Writes float16 channels when enabled and float32 channels otherwise. |

## Tips & pitfalls

- **Match the transfer conversion to the input.** Choose Log-Gamma or ARRI LogC3 only for input encoded with that curve, and choose `None` for media that is already linear.
- **Use a short, path-free file stem.** The node numbers frames and adds the `.exr` extension itself.
- **Use a project macro for portable output paths.** In-project folders selected with the picker are stored as macros and expanded when saving.
- **Keep the decoder and saver in the same execution.** `FloatMediaArtifact` is intentionally in-process-only and is not serialized into saved workflow data.

## See also

- [Decode HDR Latents](decode_hdr_latents.md) · [DiffHDR Conditioning](diffhdr_conditioning.md)
