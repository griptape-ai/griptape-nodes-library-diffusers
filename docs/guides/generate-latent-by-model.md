---
title: Generate Latent by model
description: Model-specific behavior of the Generate Latent node.
---

Generate Latent is the node that makes the picture or the clip. It takes a starting point, a [latent](what-is-a-latent.md) that is usually random noise, and cleans it up step by step to match your prompt. This cleaning-up is called denoising. The result is another latent, which Decode Media turns into pixels. Most of its settings work the same for every model. This page covers the exceptions: masked denoising, and the model families that behave differently.

Each model section below lists only what is different for that model. A step is one round of the clean-up described above. Settings that are not mentioned work as they do in [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md).

## Masked denoising

Masked denoising repaints only part of an image, like using a matte to limit a change. With a latent from [Set Latent Mask](../nodes/SetLatentMaskNode.md), the mask is applied at every step. White is repainted, black is kept. Gray is repainted partly: the lighter the gray, the more the area changes. The [Inpainting workflow](../workflows/Inpainting.md) works differently. It goes through Encode Masked Media, which rounds gray at 50% to white or black, so there is no partial change.

Set `add_noise` to on. Without it, denoising starts from the clean source image instead of noise. See [When noise is added](what-is-a-latent.md#when-noise-is-added). `start_step` is the step the run begins from. Set it to 0 to redo the white areas fully, or raise it to keep more of the original.

Masked denoising works on image models only: Flux, Flux Kontext, Flux2, Flux2 Klein, Flux3, Qwen Image, Qwen Image 2.1, Qwen Image Edit, Stable Diffusion XL, Stable Diffusion 3 and Z-Image. Flux Fill and all video models stop the run with a message. Set Latent Mask must be given an encoded latent.

- Use a standard scheduler. Schedulers that take more than one model pass per step (Heun, KDPM2 and similar) or that do not produce one timestep per step (PNDM) are refused, and the error names the scheduler.
- Options that override the schedule or the noise are not supported: custom timesteps and sigmas, partial denoising strength, and a separate random generator. Use `start_step` to choose how much of the original survives.

## LTX2

Full models default to 40 steps. Distilled models were retrained to need fewer steps. They include `LTX-2.5 Distilled`, hide `num_inference_steps` and follow a fixed schedule. They add `use_stage_2`, off by default. Off runs only the first pass, which generates the video. On runs the second pass instead, which refines the result.

The base, video-conditioning and IC-LoRA [pipeline types](how-pipelines-work.md#what-build-pipeline-decides) make video and audio together in one pass. The output latent carries the audio, and Decode Media adds it to the video. The HDR IC-LoRA path makes no audio.

`media_conditions` takes Reference Media nodes, images or videos that set frames at chosen positions, like keyframes. It is hidden on the HDR IC-LoRA path.

`reference_conditions` appears when a [LoRA](../nodes/LoadLora.md) is connected through [Apply Runtime LoRA](../nodes/LoraActivationPipelineNode.md). It accepts Reference Media, image and video inputs. Images go to the pipeline as still references. Some of these LoRAs are called IC-LoRAs. An IC-LoRA is a LoRA that uses a reference image or clip to steer the new video.

With an IC-LoRA that is not the HDR one, each reference video is rounded up to a length the model accepts by repeating its last frame, and cut to the output length if it is longer.

A LoRA whose file name contains `ic-lora-hdr` selects the HDR IC-LoRA path, which also shows `text_embeddings_path`. On that path, a mix of image and video references is rejected. If the reference videos differ in length, the shorter ones are padded by repeating their last frame.

`motion_speed` defaults to 1.0 and must be above 0. Values below 1.0 slow the motion down while playback speed stays the same. For example, 0.2 gives 5x slow motion with the LTX-2.5 slow-motion LoRA. It is hidden on the HDR IC-LoRA path.

## MiniMax-H3

Makes the video and its soundtrack together. The model-specific inputs are `prompt` and `conditioning_images`. There is no `guidance_scale` (how closely the model follows the prompt) or `negative_prompt` (what to steer away from), and the size comes from the input latent. Connect a Reference Media node to `conditioning_images` for a first or last frame, or leave it empty for text-only generation.

Connect this node straight to Decode Media. The latent carries the audio along with it.

`return_fully_denoised` makes the last step aim at a fully clean latent, so stopping early with `end_step`, the step the run stops at, still gives a clean result. MiniMax-H3 does not support it, and turning it on stops the run with a message.

To rework an existing clip, turn `add_noise` on and feed an encoded video latent. Starting part way through the steps (`start_step` above 0) needs a latent that carries audio. That is the output of an earlier MiniMax-H3 Generate Latent run, or an encoded video that has an audio track. Otherwise the run stops with a message. Inpainting (repainting a masked area) and ControlNet (guides that steer the layout) are not available.

## MiniMax-H3 reference-to-video

`MiniMaxH3Ref2VAModularPipeline` replaces `conditioning_images` with `references`, which takes up to 4 Reference Media nodes. Use one image node with up to 9 images, and one video node for each clip, up to 3. At least one reference is required.

Each clip must be at least 2 seconds long. In the prompt, call images `<Picture 1>`, `<Picture 2>` and so on, and videos `<Video 1>` and so on, each numbered from 1. Audio in the reference videos is dropped, and the result still gets a new soundtrack.

If the clips add up to more than 15 seconds, they are trimmed from the end. The last clip is shortened if it can stay at least 2 seconds long, and otherwise it is dropped. What was cut is shown as a warning on the node and written to the log.

`reference_image_short_edge` defaults to 2048 and sets the shorter side of the image references in pixels. It does not change the output size. The memory estimate does not count the extra data the reference images add, so it can show less than the real use.

## Flux3

`num_inference_steps` defaults to 50 for the base type and 16 for the distilled type. The base type shows `negative_prompt` and `guidance_scale` (default 3.5). The distilled type shows neither.

`reference_images` takes up to 10 images from Reference Media, or one image connected directly. Video is not supported. Size comes from the input latent.

For layout control, type a JSON list of regions after your prompt. Mention each region in the prompt as `<id>`. Boxes are `[y_min, x_min, y_max, x_max]` on a 0 to 1000 scale. Layout takes effect only when a prompt upsampler is selected on Build Pipeline (see [Build Pipeline by model](build-pipeline-by-model.md#flux3)). With `Disabled`, the JSON stays plain prompt text.

- Without reference images, each region has `id`, `bbox` (its box) and `desc` (a description).
- With reference images, each region has `id`, `from`, `src_bbox`, `tgt_bbox` and `desc`:
  - `from` says where the item comes from: `ref_image_0`, `ref_image_1` and so on, or `target` for new content.
  - `src_bbox` is the box around the item in its reference image. It is null for `target`.
  - `tgt_bbox` is the box where the item goes in the output.

A worked example without reference images. The prompt is `A red kite over <1> and a small boat on <2>`, followed by:

```json
[
  {"id": 1, "bbox": [0, 100, 300, 600], "desc": "a hill with green grass"},
  {"id": 2, "bbox": [700, 200, 900, 500], "desc": "a calm lake"}
]
```

Each box is `[y_min, x_min, y_max, x_max]`, so the first region covers the top of the picture from 0 to 300 down and 100 to 600 across.

## Qwen Image 2.1

Qwen Image 2.1 and Qwen Image 2.1 Turbo make images. Both support image-to-image, partial denoising with `start_step`, [masked denoising](#masked-denoising) and images with an alpha channel (transparency).

`reference_images` takes up to 10 pictures from a Reference Media node, or one image wired in directly. Only images are accepted, and more than 10 stops the run with a message.

Guidance works through `true_cfg_scale`. It defaults to 1.0, which leaves guidance off. Set it above 1.0 and give a `negative_prompt` to turn it on. `num_inference_steps` defaults to 40.

Turbo follows a fixed 8-step schedule that is saved with the model. It hides `num_inference_steps`, `negative_prompt` and `true_cfg_scale`, so it has no guidance controls.

## WAN Animate 2

Animates the character in `image` with the motion of `driving_video`, an ordinary video with no pose or face video needed. Both are required. The image and every driving frame are stretched to the width and height of the input latent, so match the image's aspect ratio when you create the latent.

The driving video needs at least `num_frames` frames at 24 fps. Extra frames at the end are ignored. `driving_video_fps` defaults to 0, which uses the frames as they are. Set it to the recorded rate to resample to 24 fps first.

Clips longer than `segment_frame_length` (default 81) are made in overlapping segments. `prev_segment_conditioning_frames` (default 1) sets the overlap. Both must be of the form 4N + 1, which means a multiple of 4 plus 1: 1, 5, 9, 13 and so on. The model rejects other values. The overlap must be smaller than the segment length. `prompt_ref` is empty by default, which uses the model's built-in reference prompt.

The defaults are 40 steps and `guidance_scale` 3.0. With the distilled checkpoint they are 10 and 1.0. Starting part way through the steps works only when the clip fits in one segment, and the input latent must come from another WAN Animate 2 run. Inpainting, ControlNet and `add_noise` are not supported. The first run compiles the transformer, a one-time setup that makes it slower than later runs.

`return_fully_denoised` makes the last step aim at a fully clean latent. WAN Animate 2 does not support it, because each segment builds on the previous segment's finished pixels. Turning it on stops the run with a message.

## WAN VACE

WAN VACE is a WAN model for editing or extending video from a source clip, an optional mask and reference images. It adds `source_media`, `mask` and `reference_images` inputs, each from a Reference Media node or a direct image or video connection. `guidance_scale` defaults to 5.0 and `conditioning_scale` to 1.0. A higher `conditioning_scale` makes the result follow the source more closely. `mask` and `reference_images` need `source_media` connected, otherwise the run stops with a message. `reference_images` takes at most 8 images. A video connected there counts as one reference per frame, so it is easy to go over the limit.

With `gray_masked_source` on, the default, source pixels under white mask areas are replaced with gray before generation, so the model does not see what it is replacing. It has no effect without a connected `mask`. Turn it off for edits that should follow the original, such as recoloring. Frames with no source content are gray either way. If you leave out `mask`, it is made from `source_media`: black where the source has content and white elsewhere.

For the settings every model shares, see [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md).
