---
title: Diffusion Library
---

The Diffusion Library makes images and video from a text description or from a picture you supply. It uses diffusion models, which start from random noise and refine it step by step into a picture or clip that matches your prompt. Instead of one "generate" node, each stage is its own node: load a model, create or encode a starting point, run the diffusion, adjust the result, and decode it into a picture or a clip. You can see, reuse and change every stage.

Because each stage is its own node, you can refine an image over several passes, stack [ControlNets](nodes/ControlNetNode.md) (extra guides that steer the layout), paste one image into another before it is turned back into pixels, start a video from a chosen first and last frame, or enlarge a result before decoding it. The model is loaded once and reused across runs, so changing only a prompt is fast.

Download each model in Model Manager before you run a workflow. Build Pipeline shows an **Open Model Manager to Download** button when the selected model is missing. The models come from Hugging Face, a site that hosts shared models. Some, such as FLUX.1 dev, are gated and need an `HF_TOKEN` access token in your API keys. See [Hugging Face and Diffusers](guides/hugging-face-and-diffusers.md).

![A workflow built from Diffusion Library nodes](assets/hero.png)

## Where to start

Learn the ideas:

- [What is a latent?](guides/what-is-a-latent.md) explains the start, clean-up and decode pattern behind every workflow, and the working format the model uses in between.
- [What is a VAE?](guides/what-is-a-vae.md) explains the encoder and decoder that move pictures into latents and back.
- [What are a text encoder and a tokenizer?](guides/what-is-a-text-encoder.md), [What are a transformer and a UNet?](guides/what-is-a-transformer.md) and [What is a scheduler?](guides/what-is-a-scheduler.md) explain the parts you can swap with Load Component and Load Scheduler.

Try a workflow:

- [Text2Image](workflows/Text2Image.md) is the place to make your first image. It already has a prompt, a size and a model chosen. Download the model before you run it.
- [Image2Image](workflows/Image2Image.md) repaints an image you already have to match a prompt.
- [Inpainting](workflows/Inpainting.md) repaints only the area you mask.
- [FirstAndLastFrameImage2Video](workflows/FirstAndLastFrameImage2Video.md) fills in the motion from a first frame to a last frame.

Learn the main nodes:

- [Build Pipeline](nodes/LatentDiffusionPipelineBuilderNode.md) loads the model. [Generate Latent](nodes/DiffusionPipelineGenerateLatentNode.md) runs the diffusion. [Decode Media](nodes/VaeDecodeNode.md) turns the result into an image or video.
- [Load ControlNet](nodes/ControlNetNode.md) and [Apply ControlNet](nodes/ControlNetDiffusionPipelineBuilderNode.md) steer a result with edges, depth or a pose.
- [Load LoRA](nodes/LoadLora.md) adds a LoRA, a small add-on file that gives the model a particular style or subject.

Check memory:

- [Estimate Pipeline Memory](nodes/PipelineMemoryEstimateNode.md) shows how much memory on your graphics card (GPU) a run needs.
- The [command-line version](tools/estimate_workflow_memory.md) does the same for a saved workflow.

## Node categories

The editor groups the nodes under Diffusion.

| Category | What it holds |
| --- | --- |
| Pipeline | Loading models (a pipeline is a loaded model plus its settings), ControlNets, LoRAs, components and schedulers, and clearing or estimating the loaded pipeline. |
| Create | Starting points: random noise or an empty latent (the model's working format). |
| Processing | Running the diffusion and enlarging a latent. |
| Transform | Adding, subtracting, multiplying and compositing latents, and setting a latent mask. |
| Conditioning | Reference images and video that guide a generation. |
| Encode & Decode | Moving between images or video and latents. |
| IO | Saving latents and EXR sequences. |

Models that this library supports include Flux, Qwen-Image, Stable Diffusion, Z-Image, LTX, WAN and HunyuanVideo. The model picker also lists models that your Griptape Nodes license does not allow. They are marked as not permitted and cannot be run.
