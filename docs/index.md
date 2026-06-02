# Modular Diffusion Nodes — Documentation

Per-node reference for the Modular Diffusion Nodes Library. See the [README](../README.md) for an overview of the library, supported models, and workflow templates.

## Node reference

Node groups mirror the categories in [`griptape_nodes_library.json`](../griptape_nodes_library.json):

### Pipeline
- [Modular Diffusion Pipeline Builder](nodes/pipeline_builder.md)
- [ControlNet Pipeline](nodes/controlnet_pipeline.md)
- [Load LoRA](nodes/load_lora.md)

### Create
- [Create Noise Latents](nodes/create-noise-latents.md)
- [Create Empty Latents](nodes/empty_latents.md)

### Processing
- [Generate Media Latents](nodes/generate_media_latents.md)
- [Latent Upsampler](nodes/latent_upsampler.md)

### Transform
- [Add Latents](nodes/add_latents.md)
- [Subtract Latents](nodes/subtract_latents.md)
- [Multiply Latents](nodes/multiply_latents.md)
- [Latents Composite Mask](nodes/latents_composite_mask.md)

### Conditioning
- [Configure ControlNet](nodes/configure_controlnet.md)
- [Media Generation Conditioning](nodes/media_gen_conditioning.md)

### Encode / Decode
- [Encode Media Latent](nodes/encode_media_latent.md)
- [Encode Masked Media Latent](nodes/encode_masked_media_latent.md)
- [Decode Media Latent](nodes/decode_media_latent.md)

### IO
- [Save Latent Tensor](nodes/save_latent_tensor.md)
