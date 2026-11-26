---
description: Makes a video from a text prompt in a single node. It wraps the whole text-to-video workflow.
when_to_use: Use it when you want a video clip from a prompt without building the graph yourself. Use FirstLastFrameText2VideoNode when you know the first and last frames.
---

## When to use it

This node makes a short video from a text description, called a prompt. It is a ready-made node that wraps a whole workflow, so you get one node instead of the full graph. The library marks it as a Labs node.

Inside, the node runs [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md), [Create Noise Latent](../nodes/NoiseLatentNode.md), [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) and [Decode Media](../nodes/VaeDecodeNode.md). It starts from noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a clip that the model works in, and decodes the result into a video. Open the workflow behind the node to change the model or the steps.

## What it needs

- A prompt.
- The model for the `pipeline_type` you pick, downloaded in advance in Model Manager. See [Hugging Face and Diffusers](../guides/hugging-face-and-diffusers.md). The default type is `LTXPipeline`.

## Where it fits

```text
[Text to Video] → Display Video
```

## Parameters

### pipeline_type

The model family to use. The default is `LTXPipeline`. The choices include `LTXPipeline`, `LTX2Pipeline`, `LTX-2.5 Distilled`, `LTX-2.5 Full (SFT)`, `MiniMaxH3ModularPipeline` and `WanPipeline`. Each choice picks its own model. See [How pipelines work](../guides/how-pipelines-work.md).

### prompt

The description of the video you want.

### seed

Default 42. The seed decides which noise the video starts from. The same seed and prompt give the same video.

### width

Video width in pixels. The default is 1024.

### height

Video height in pixels. The default is 1024.

### num_frames

How many frames the video has. The default is 41. The workflow plays the video at 25 frames per second, so 41 frames last about 1.6 seconds.

### output_video

The finished video.
