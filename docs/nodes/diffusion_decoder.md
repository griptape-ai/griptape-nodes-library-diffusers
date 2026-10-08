# Diffusion Decoder

**Loads a selected diffusion decoder to turn its compatible latent tensor into an image or video.**

Category: `ModularDiffusion/Encode\Decode`

## TL;DR
- Select a decoder model and connect its compatible `LatentArtifact` input.
- Output type follows the selected model: image decoders expose `output_image`, while video decoders expose `output_video` and `fps`.
- Normal mode exposes PIL-converted output; raw mode also exposes pipeline-postprocessed floating-point RGB through `raw_media`.
- The LTX-2.5 diffusion decoder is the initial supported model and decodes LTX-2 video latents.

## Typical workflow position
```text
Generate Media Latents → [Diffusion Decoder] → Save Video
```

## Node preview

<!-- TODO: add docs/assets/nodes/diffusion-decoder.png screenshot -->

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `latent_tensor` | `LatentArtifact` | Yes | Latent tensor produced by a compatible model. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `output_image` | `ImageUrlArtifact` | Present when the selected decoder supports images. |
| `output_video` | `VideoUrlArtifact` | Present when the selected decoder supports video. |
| `raw_media` | `FloatMediaArtifact` | Present in raw mode; contains pipeline-postprocessed floating-point RGB data in the display range [0, 1]. |

## Parameters

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `decoder_type` | str | `LTX-2.5` | Selects the decoder family and its model-specific settings. |
| `diffusion_decoder_model` | repo | `Lightricks/LTX-2.5-Diffusers` | Selects the decoder weights; download the repo through Model Manager before decoding. |
| `decoder_num_inference_steps` | int | `1` | Number of denoising steps for the LTX-2.5 diffusion decoder. |
| `decoder_use_tiling` | bool | `False` | Enables overlapping tiles to lower peak memory use on large videos. |
| `seed` | int | `0` | Use a non-negative value for repeatable output; `-1` leaves the generator unset for pipeline-default randomness. |
| `enable_raw_output` | bool | `False` | Exposes pipeline floating-point output as `raw_media`; display outputs are generated separately. |
| `fps` | int (1–120) | `24` | Output frame rate; shown only for video decoder models. |

## Provider / model behavior

| Decoder | Behavior |
| --- | --- |
| LTX-2.5 | Decodes compatible 5-D LTX-2 video latents and publishes an MP4. Its model repository contains the decoder weights. |

The node's decoder specification selects the output media type. FPS is shown only for video decoders. In normal mode, the decoder helper returns PIL frames. In raw mode, it returns pipeline-postprocessed floating-point NumPy RGB data through `raw_media`; the regular image or video output is generated separately for display.

## Tips & pitfalls

- **Use a compatible latent.** LTX-2.5 decoding expects the video latent shape and normalization produced by an LTX-2-family pipeline.
- **Download the selected repository first.** The LTX-2.5 Diffusers repository is gated on Hugging Face and requires accepting its terms and authenticating before download.
- **Enable tiling for large videos.** Tiling reduces peak decoder memory by processing overlapping output regions.

## See also

- [Generate Media Latents](generate_media_latents.md) — produces compatible video latents.
- [Decode Media Latent](decode_media_latent.md) — decodes with the VAE included in a loaded pipeline.
