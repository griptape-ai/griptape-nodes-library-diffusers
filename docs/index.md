# Modular Diffusion Nodes — Documentation

Per-node reference for the Modular Diffusion Nodes Library. See the [README](../README.md) for an overview of the library, supported models, and workflow templates.

## Node reference

Node groups mirror the categories in [`griptape-nodes-library.json`](../griptape-nodes-library.json):

### Preconfigured
- [Text to Image](nodes/text_to_image.md)
- [Image to Image](nodes/image_to_image.md)
- [Inpaint Image](nodes/inpainting.md)
- [Text to Video](nodes/text_to_video.md)
- [First/Last Frame to Video](nodes/first_last_frame_to_video.md)

### Pipeline
- [Build Pipeline](nodes/pipeline_builder.md)
- [Apply ControlNet](nodes/controlnet_pipeline.md)
- [Load LoRA](nodes/load_lora.md)
- [Load Component](nodes/load_component.md)
- [Load Scheduler](nodes/load_scheduler.md)
- [Apply Runtime LoRA](nodes/lora_pipeline.md)
- [Clear Pipeline Cache](nodes/clear_pipeline_cache.md)
- [Estimate Pipeline Memory](nodes/estimate_pipeline_memory.md)

### Create
- [Create Noise Latent](nodes/create-noise-latents.md)
- [Create Empty Latent](nodes/empty_latents.md)

### Processing
- [Generate Latent](nodes/generate_media_latents.md)
- [Upsample Latent](nodes/latent_upsampler.md)

### Transform
- [Add Latents](nodes/add_latents.md)
- [Subtract Latents](nodes/subtract_latents.md)
- [Multiply Latents](nodes/multiply_latents.md)
- [Composite Latents with Mask](nodes/latents_composite_mask.md)

### Conditioning
- [Load ControlNet](nodes/configure_controlnet.md)
- [Reference Media](nodes/media_gen_conditioning.md)

### Encode / Decode
- [Encode Media Latent](nodes/encode_media_latent.md)
- [Encode Masked Media Latent](nodes/encode_masked_media_latent.md)
- [Decode Media Latent](nodes/decode_media_latent.md)
- [Decode HDR Media](nodes/decode_hdr_latents.md)
- [Convert Latent to Media](nodes/latent_to_media.md)

### IO
- [Save EXR Sequence](nodes/save_exr_sequence.md)
- [Save Latent](nodes/save_latent_tensor.md)

## Command-line tools
- [Estimate Workflow Memory (CLI)](tools/estimate_workflow_memory.md)
