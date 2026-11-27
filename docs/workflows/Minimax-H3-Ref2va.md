---
description: Give it reference images and a structured prompt, and it makes a video with a soundtrack that uses those references.
when_to_use: Use it when you want to build a shot from separate pictures, such as a character and a location. It needs a very large graphics card.
---

This workflow makes a video from reference images with MiniMax-H3, using the `MiniMaxH3Ref2VAModularPipeline` type. You give the model pictures of the things that should appear, and a prompt that says what each one is and what happens. The model makes the video together with a soundtrack.

The template comes with two sample images. The first is a futuristic soldier and the second is a destroyed city. The prompt asks for a high-angle wide shot of the soldier from behind, looking out over the city, with a slow camera tilt and push in.

## What it connects

1. Two Load Image nodes supply the reference images.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Reference Media](../nodes/MediaGenConditioningNode.md) holds the two images. It is set to Custom with two images. In the template it is named Media Generation Conditioning. It goes to `references` on Generate Latent.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the video size, the length and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. A Text Input node holds the prompt.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) makes the video. See [MiniMax-H3 reference-to-video](../guides/generate-latent-by-model.md#minimax-h3-reference-to-video) for the settings specific to this pipeline type.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video with its soundtrack. The video always plays at 24 frames per second. The `fps` parameter has no effect here, because Decode Media uses the model's own rate whenever there is a soundtrack.

## Hardware

The model's authors recommend one graphics card with 80 GB of memory. The transformer and the text encoder are each about 62 GB, so the computer also needs a large amount of main memory. Do not delete the downloaded files from the cache after the first run, because the text encoder is read from disk each time. See [MiniMax-H3](../guides/build-pipeline-by-model.md#minimax-h3).

## Steps

1. Open the **Minimax-H3-Ref2va** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded. The pipeline type loads its own transformer, which is stored in the same Hugging Face repo as the base model. See [Hugging Face and Diffusers](../guides/hugging-face-and-diffusers.md).
1. Load your reference images in the two Load Image nodes. The Reference Media node takes up to 9 images.
1. Rewrite the prompt for your shot. In the prompt, call the images `<Picture 1>`, `<Picture 2>` and so on, numbered from 1.
1. Run the workflow.

## Defaults

- Model: MiniMax-H3 (`MiniMaxAI/MiniMax-H3`), with the `MiniMaxH3Ref2VAModularPipeline` type.
- Size: 512 x 512, set on Create Noise Latent.
- Length: 124 frames, set as `num_frames` on Create Noise Latent. About 5.2 seconds at 24 frames per second.
- Frame rate: 24 frames per second, fixed by the model.
- Steps: 20.
- Reference image short edge: 1024 in the template. The node default is 2048. This sets the shorter side of the image references in pixels. It does not change the output size.
- Seed: 42.

## Tips

- The template prompt has four parts. `subject_definitions` gives each picture a name, such as `<Subject 1>` for the soldier from `<Picture 1>`. `summary` is a one-line statement of the task. In the template it says the video is a VFX shot made from the two subjects as visual references. `retention_analysis` says what to keep from each picture and what to drop. `detailed_description` describes the shot, using the subject names. The template note says this structure stops the pictures from bleeding into one another, so backgrounds and lighting from one image do not show up where they should not.
- MiniMax-H3 reference-to-video can also take video clips. Use one Reference Media node for each clip, up to 3. Each clip must be at least 2 seconds long. Clips over 15 seconds in total are trimmed from the end. The audio in a clip is dropped.
- Add Noise is off on Generate Latent, because the starting latent is already noise. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- There is no guidance setting or negative prompt on this pipeline type.
- LoRAs are not supported with MiniMax-H3.
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
