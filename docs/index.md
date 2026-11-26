---
title: Diffusion Library
---

The Diffusion Library lets you build images and video with Diffusers models, one step at a time. Instead of a single "generate" node, each stage is its own node: load a model, make or encode a starting point, run the diffusion, adjust the result, and decode it into a picture or a clip. You can see, reuse and change every stage.

Because the stages are separate, you can do things a single node can't: refine an image over several passes, stack ControlNets, paste one image into another in latent space, condition a video on its first and last frames, or enlarge a result before decoding it. The model is loaded once and reused across runs, so changing only a prompt is fast.

Download each model in Model Manager before you run a workflow. Build Pipeline shows an **Open Model Manager to Download** button when the selected model is missing. Some models, such as FLUX.1 dev, are gated on Hugging Face and need an `HF_TOKEN` access token in your API keys.

![A workflow built from Diffusion Library nodes](assets/hero.png)

## Where to start

- **Make your first image.** Open [Text2Image](workflows/Text2Image.md). It already has a prompt, a size and a model set.
- **Rework a picture you already have.** [Image2Image](workflows/Image2Image.md) repaints your image to match a prompt.
- **Change one part of an image.** [Inpainting](workflows/Inpainting.md) repaints only the area you mask.
- **Make a video between two images.** [FirstAndLastFrameImage2Video](workflows/FirstAndLastFrameImage2Video.md) fills in the motion from a first frame to a last frame.
- **Understand the pieces.** [Build Pipeline](nodes/LatentDiffusionPipelineBuilderNode.md) loads the model. [Generate Latent](nodes/DiffusionPipelineGenerateLatentNode.md) runs the diffusion. [Decode Media](nodes/VaeDecodeNode.md) turns the result into an image or video.
- **Guide the layout.** [Load ControlNet](nodes/ControlNetNode.md) and [Apply ControlNet](nodes/ControlNetDiffusionPipelineBuilderNode.md) steer a result with edges, depth or a pose.
- **Add a style or subject.** [Load LoRA](nodes/LoadLora.md) adds a LoRA to the model.
- **Check memory.** [Estimate Pipeline Memory](nodes/PipelineMemoryEstimateNode.md) shows how much a run needs. The [command-line version](tools/estimate_workflow_memory.md) does the same for a saved workflow.

## Node categories

The editor groups the nodes under Diffusion.

| Category | What it holds |
| --- | --- |
| Pipeline | Loading models, ControlNets, LoRAs, components and schedulers, and clearing or estimating the loaded pipeline. |
| Create | Starting points: noise or an empty latent. |
| Processing | Running the diffusion and upsampling a latent. |
| Transform | Adding, subtracting, multiplying and compositing latents, and setting a latent mask. |
| Conditioning | Reference images and video that guide a generation. |
| Encode & Decode | Moving between images or video and latents. |
| IO | Saving latents and EXR sequences. |

Models that this library supports include Flux, Qwen-Image, Stable Diffusion, Z-Image, LTX, WAN and HunyuanVideo. Models your license does not permit are marked in the model picker.
