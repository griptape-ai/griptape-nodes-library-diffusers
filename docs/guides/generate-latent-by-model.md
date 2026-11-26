---
title: Generate Latent by model
description: Model-specific behavior of the Generate Latent node.
---

Generate Latent is the node that makes the picture or the clip. It takes a starting point, a [latent](what-is-a-latent.md) (the compressed block of numbers the model works on, usually random noise), and cleans it up step by step to match your prompt. This cleaning-up is called denoising. The result is another latent, which Decode Media turns into pixels. Most of its settings work the same for every model. This page covers the exceptions: masked denoising, and the model families that behave differently.

## Masked denoising

Masked denoising repaints only part of an image, like using a matte to limit a change. With a latent from Set Latent Mask, the mask is applied at every step without switching pipelines. White areas are redone, black areas stay close to the original, and gray areas get less time to change. Set `add_noise` to on, and use `start_step`, the step the run begins from, to choose how much of the original survives. Set `start_step` to 0 to redo the white areas fully, or raise it to keep more of the original.

It works with `FluxPipeline`, Flux Kontext, `Flux2Pipeline`, Flux2 Klein, Flux3, Qwen Image, Qwen Image Edit, Stable Diffusion XL, Stable Diffusion 3 and Z-Image. Flux Fill and every video pipeline stop the run with a message. It works on image latents only, and Set Latent Mask must be given an encoded latent. Custom timesteps and sigmas (settings that set the noise levels step by step) are not supported, and neither are schedulers (the rule for how much noise each step removes) that are not first order.

## LTX2

A step is one round of clean-up. Full checkpoints (the complete versions of the model) default to 40 steps. Distilled checkpoints, versions made to run in fewer steps, including `LTX-2.5 Distilled`, hide `num_inference_steps` and follow a fixed schedule. They add `use_stage_2`, off by default, which switches from the initial generation schedule to the refinement schedule.

The base, video-conditioning and IC-LoRA pipelines make video and audio together in one pass. The output latent carries the audio, and Decode Media adds it to the video. The HDR IC-LoRA path makes no audio.

`media_conditions` takes Reference Media nodes, images or videos that set frames at chosen positions, like keyframes. It is hidden on the HDR IC-LoRA path.

`reference_conditions` appears when a LoRA (a small add-on file for style or subject) is connected through [Apply Runtime LoRA](../nodes/LoraActivationPipelineNode.md). It accepts Reference Media, image and video inputs. Images go to the pipeline as still references. A LoRA whose file name contains `ic-lora-hdr` selects the HDR IC-LoRA path, which also shows `text_embeddings_path`. On that path, a mix of image and video references is rejected, and if the reference videos differ in length, the shorter ones are padded by repeating their last frame. On other IC-LoRA runs, each reference video is rounded up to a length the model accepts by repeating its last frame, and cut to the output length if it is longer.

`motion_speed` defaults to 1.0 and must be above 0. Values below 1.0 slow the motion down while playback speed stays the same. For example, 0.2 gives 5x slow motion with the LTX-2.5 slow-motion LoRA. It is hidden on the HDR IC-LoRA path.

## MiniMax-H3

Turning on `return_fully_denoised` stops the run with a message.

Makes the video and its soundtrack together. The model-specific inputs are `prompt` and `conditioning_images`. There is no `guidance_scale` (how closely the model follows the prompt) or `negative_prompt` (what to steer away from), and the size comes from the input latent. Connect a Reference Media node to `conditioning_images` for a first or last frame, or leave it empty for text-only generation.

Connect this node straight to Decode Media. The audio travels in the latent's metadata.

With `add_noise` on and a MiniMax-H3 encoded video latent as input, you get video-to-video, which reworks an existing clip. A `start_step` above 0 needs a latent that carries audio, otherwise the run stops with a message. Chain partial denoising (starting part way through the steps) from another MiniMax-H3 Generate Latent run. Inpainting (repainting a masked area) and ControlNet (guides that steer the layout) are not available.

## MiniMax-H3 reference-to-video

`MiniMaxH3Ref2VAModularPipeline` replaces `conditioning_images` with `references`, which takes up to 4 Reference Media nodes. Use one image node with up to 9 images, and one video node for each clip, up to 3. At least one reference is required.

Each clip must be at least 2 seconds long. In the prompt, call images `<Picture 1>`, `<Picture 2>` and so on, and videos `<Video 1>` and so on, each numbered from 1. Audio in the reference videos is dropped, and the result still gets a new soundtrack.

If the clips add up to more than 15 seconds, they are trimmed from the end. The last clip is shortened if it can stay at least 2 seconds long, and otherwise it is dropped. What was cut is shown as a warning on the node and written to the log.

`reference_image_short_edge` defaults to 2048 and sets the shorter side of the image references in pixels. It does not change the output size. The memory estimate does not count reference tokens, so it can show less than the real use.

## Flux3

`num_inference_steps` (the number of steps) defaults to 50 or 16 for Flux3.

With a prompt upsampler selected on Build Pipeline, the prompt is expanded before generation. The base type shows `negative_prompt` and `guidance_scale` (default 3.5) and runs 50 steps. The distilled type shows neither and runs 16 steps.

`reference_images` takes up to 10 images from Reference Media, or one image connected directly. Video is not supported. Size comes from the input latent. Partial denoise, preview and cancel all work.

For layout control, add a JSON list of regions after the caption, with each region cited as `<id>` in the caption. Boxes are `[y_min, x_min, y_max, x_max]` on a 0 to 1000 scale. Layout takes effect only when a prompt upsampler is selected. With `Disabled`, the JSON stays plain prompt text.

- Without reference images, each region has `id`, `bbox` and `desc`.
- With reference images, each region has `id`, `from`, `src_bbox`, `tgt_bbox` and `desc`. `from` is `ref_image_N`, counted from 0, or `target` for new content. `src_bbox` is where the item sits in its reference and is null for `target`. `tgt_bbox` is where it goes in the output and is null for removed items.

## WAN Animate 2

Turning on `return_fully_denoised` stops the run with a message.

Animates the character in `image` with the motion of `driving_video`, an ordinary video with no pose or face video needed. Both are required. The image and every driving frame are stretched to the width and height of the input latent, so match the image's aspect ratio when you create the latent.

The driving video needs at least `num_frames` frames at 24 fps. Extra frames at the end are ignored. `driving_video_fps` defaults to 0, which uses the frames as they are. Set it to the recorded rate to resample to 24 fps first.

Clips longer than `segment_frame_length` (default 81) are made in overlapping segments. `prev_segment_conditioning_frames` (default 1) sets the overlap. Both must be of the form 4N + 1, and the overlap must be smaller than the segment length. `prompt_ref` is empty by default, which uses the model's built-in reference prompt.

The defaults are 40 steps and `guidance_scale` 3.0. With the distilled checkpoint they are 10 and 1.0. Partial denoise works only when the clip fits in one segment, and must chain from another WAN Animate 2 run. Inpainting, ControlNet and `add_noise` are not supported. The first run compiles the transformer and takes longer.

## WAN VACE

Adds `source_media`, `mask` and `reference_images` inputs, each from a Reference Media node or a direct image or video connection. `guidance_scale` defaults to 5.0 and `conditioning_scale` to 1.0. A higher `conditioning_scale` makes the result follow the source more closely. `mask` and `reference_images` need `source_media` connected, otherwise the run stops with a message. `reference_images` takes at most 8 images. A video connected there counts as one reference per frame, so it is easy to go over the limit.

With `gray_masked_source` on, the default, source pixels under white mask areas are replaced with gray before generation, so the model does not see what it is replacing. It has no effect without a connected `mask`. Turn it off for edits that should follow the original, such as recoloring. Frames with no source content are gray either way. If you leave out `mask`, it is made from `source_media`: black where the source has content and white elsewhere.

For the settings every model shares, see [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md).
