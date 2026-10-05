# Save EXR Sequence

**Writes decoded float RGB media as a numbered OpenEXR frame sequence.**

Category: `ModularDiffusion/IO`

## TL;DR
- Connect `media` (e.g. from Decode HDR Latents) to save one or more RGB frames as OpenEXR files.
- Writes the input float values unchanged.
- Each run writes the next free version folder, such as `<stem>_v001/<stem>.0001.exr`, `<stem>_v001/<stem>.0002.exr`, and so on, where `<stem>` comes from `output_file`. Frames are written through the project file system.

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
| `output_file` | path | `raw/frame.exr` | Sequence name comes from the file stem, so `shot.exr` writes `shot_v001/shot.0001.exr`. A folder part such as `renders/shot.exr` writes into a subfolder of the project outputs folder. The extension is ignored. |
| `save_as_half_float` | bool | `True` | Writes float16 channels when enabled and float32 channels otherwise. |

## Tips & pitfalls

- **Prepare values before saving if conversion is needed.** The node writes incoming float values as-is and does not infer or convert their color space.
- **Use a plain file stem.** The node numbers frames and adds the `.exr` extension itself.
- **Each run keeps earlier versions.** The next free `_vNNN` folder is used (the engine fills gaps in the numbering), so re-running never overwrites a previous sequence.
- **Use a project macro for portable output paths.** In-project folders selected with the picker are stored as macros and expanded when saving.
- **Keep the decoder and saver in the same execution.** `FloatMediaArtifact` is intentionally in-process-only and is not serialized into saved workflow data.
- **Check the output folder after a failed run.** Frames written before the error stay on disk.

## See also

- [Decode HDR Latents](decode_hdr_latents.md) · [DiffHDR Conditioning](diffhdr_conditioning.md)
