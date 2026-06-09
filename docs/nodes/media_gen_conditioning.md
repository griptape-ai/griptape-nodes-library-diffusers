# Media Generation Conditioning

**Bundles one or more conditioning images (or a conditioning video) with per-frame placement and strength — consumed by video pipelines that support media-driven conditioning (notably WAN first/last-frame Image-to-Video, LTX media-gen, and LTX2 image/video conditioning + IC-LoRA / HDR IC-LoRA reference video).**

Category: `ModularDiffusion/Conditioning`

## TL;DR
- Pick `mode` (`image` or `video`); the parameters dynamically regenerate.
- In `image` mode, **`num_images`** controls how many image slots appear (0–8). Each image gets its own frame index + strength.
- In `video` mode, a single conditioning video + frame index + strength is exposed.
- Connect the `conditioning` output to the Generate Media Latents node (as `additional_parameters` for the pipelines that support it).

## Typical workflow position
```text
Load Image (first) ──┐
Load Image (last) ───┴─→ [Media Generation Conditioning] ──┐
                                                            ├─→ Generate Media Latents → Decode
Pipeline Builder ───────────────────────────────────────────┘
```

## Node preview

<img src="../assets/nodes/media-gen-conditioning.png" alt="Media Generation Conditioning" width="480">

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `conditioning` | `dict` | Bundle of images/video + per-entry frame index + strength. |

## Parameters

### Mode selection

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `mode` | choice | `image` | `image` or `video`. Switching regenerates the parameter set. |

### Image mode *(dynamic — count driven by `num_images`)*

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `num_images` | int (0–8) | `0` | Number of image slots to expose. |
| `image_{i}` | `ImageArtifact` / `ImageUrlArtifact` | — | Conditioning image at slot `i`. |
| `image_{i}_frame_index` | int | `0` | Output-frame index where this image is applied. |
| `image_{i}_strength` | float (0.0–1.0) | `1.0` | Per-image conditioning weight. |

### Video mode

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `video` | `VideoArtifact` / `VideoUrlArtifact` | — | Conditioning video. |
| `frame_index` | int | `0` | Output-frame index where the conditioning starts. |
| `video_strength` | float (0.0–1.0) | `1.0` | Conditioning weight. |

## Tips & pitfalls

- **`frame_index` is into the *output* timeline**, not the conditioning media. `0` = start of output, `num_frames - 1` = end.
- **First/last-frame Image-to-Video** = drop two image slots, set `image_0_frame_index = 0` and `image_1_frame_index = num_frames - 1`.
- **Image slot count is destructive.** Lowering `num_images` removes the trailing slots and any connections to them.
- **Not every video pipeline reads this conditioning.** It's used by WAN Image-to-Video, LTX media-gen, and LTX2 (which auto-swaps to a conditioning, IC-LoRA, or HDR IC-LoRA variant depending on what's connected); other pipelines ignore the input.
- **LTX2 + HDR IC-LoRA requires a reference video.** When an HDR IC-LoRA is loaded, set `mode = video` and connect a reference clip. The HDR pipeline will not run from images alone.

## See also

- [Generate Media Latents](generate_media_latents.md) — consumer (via `additional_parameters`).
- Workflow templates: `Modular_first_n_last_i2v_workflow.py`, `Modular_t2v_workflow.py`.
