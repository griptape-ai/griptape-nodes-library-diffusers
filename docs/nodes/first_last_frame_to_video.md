# First/Last Frame to Video

**A preconfigured, self-contained workflow that generates a video conditioned on a first and a last frame plus a text prompt — dropped in as a single node.**

Category: `ModularDiffusion/Preconfigured`

## TL;DR
- Ships configured with default strengths at `1.0`; `pipeline_type` also offers various supported pipeline types.
- Outputs `output_video` (`VideoUrlArtifact`).

## Typical workflow position
```text
Load Image (first) ───┐
                      ├─→ [First/Last Frame to Video] → Display Video
Load Image (last)  ───┘
```

## Node preview

<img src="../assets/nodes/FirstLastFrame2Video.png" alt="First/Last Frame to Video" width="480">

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `pipeline_type` | Supported Pipeline Types | Yes | Selects which pipeline family the internal Build Pipeline node uses. |
| `image_0` (display name "First") | `ImageUrlArtifact` | Yes | First-frame conditioning image, forwarded to the internal Reference Media node. |
| `image_1` (display name "Last") | `ImageUrlArtifact` | Yes | Last-frame conditioning image, forwarded to the internal Reference Media node. |
| `prompt` | str | Yes | Positive text prompt, forwarded to the internal Generate Latent node. |
| `seed` | int | No | Forwarded to the internal Create Noise Latent node. |
| `width` | int |Yes | Forwarded to the internal Create Noise Latent node. |
| `height` | int | Yes | Forwarded to the internal Create Noise Latent node. |
| `num_frames` | int | No | Forwarded to the internal Create Noise Latent node. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `output_video` | `VideoUrlArtifact` | Decoded video from the latent tensor. |
| `was_successful` | bool | Whether the internal sub-flow completed without errors. |
| `result_details` | str | Details about the operation result. |

## Provider / model behavior

The workflow starts with a default model. To use another supported pipeline type, select it with `pipeline_type` in the internal Build Pipeline node.
The internal Reference Media node is pre-set with `image_preset="First + Last"` and `num_images=2` (hidden), with both `image_0_strength` and `image_1_strength` at `1.0` (full conditioning strength for each frame).

The internal Generate Latent node ships with `guidance_scale=3.0` and `num_inference_steps=40`. Its runtime-parameter tooltips explicitly reference recommended values regardless of the selected `pipeline_type` — the runtime-parameters class backing this node appears to be shared with a driver. The numeric defaults still apply; treat the specific wording in the tooltips as inherited boilerplate.

## Tips & pitfalls

- **This is a complete sub-flow, not a single primitive.** Double-click the node to inspect and edit its Reference Media, Build Pipeline, Create Noise Latent, Generate Latent, and Decode Media nodes.
- **Fixed model selection.** You cannot change the model selection or other internal parameters directly from the top-level node; you must open the sub-flow to make changes.
- **No download status visibility.** There is no visual indication within the node itself whether the required model has been downloaded or if it needs to be installed via the Model Manager.
- **Model Installation**: Use the Model Manager (via the internal builder's `model_download` button) to install the model before running.
- **`add_noise` is disabled on the internal Generate node.** The Create Noise Latent node already supplies a pure-noise starting latent, so the Generate node denoises it directly without re-noising. The seed set here fully controls reproducibility.
- **Frame conditioning strength is fixed at `1.0` for both frames.** Open the sub-flow and adjust `image_0_strength`/`image_1_strength` on the Reference Media node for a softer blend.
- **The internal Generate node's guidance tooltips mention recommended values** regardless of the selected `pipeline_type` — a wording leftover from a shared runtime-parameters class. The numeric defaults (`guidance_scale=3.0`, `num_inference_steps=40`) still apply.
- **Want more granular control?** Open the sub-flow and edit the internal nodes directly — anything not exposed as a top-level input here can be changed there.

## See also

- [Text to Image](text_to_image.md) · [Inpaint Image](inpainting.md) · [Text to Video](text_to_video.md)
- [Reference Media](media_gen_conditioning.md) · [Build Pipeline](pipeline_builder.md) · [Generate Latent](generate_media_latents.md) — the nodes this workflow wraps internally.
