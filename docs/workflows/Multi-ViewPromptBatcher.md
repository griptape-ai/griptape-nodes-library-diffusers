---
description: Give it one image and a list of camera instructions, and it makes one edited image per line, then shows them all in a grid.
when_to_use: Use it when you want many variations of one image, such as the same subject from different camera angles. Copy the loop pattern for any batch of prompts.
---

This workflow edits one image many times, once for each line of a text list. The template uses Qwen-Image-Edit and eight camera instructions, such as "Turn the camera to a close-up" or "Rotate the camera 90 degrees to the left." It shows the eight results together in one grid image.

It also demonstrates a loop. A ForEach node repeats the same steps for every item in a list. The template note says it plainly: each new line in the text input node is a new prompt that the loop goes over.

The sample image is a photo that the template crops to a square and resizes to 1024 x 1024.

## What it connects

1. A Text Input node holds the prompts, one per line.
1. A Split Text node splits that text at each new line into a list of eight items.
1. A Load Image node supplies the photo. A Crop Image node cuts it to a square and a Resize Image node resizes it to 1024 x 1024.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. A ForEach Start node takes the list and hands out one prompt at a time. Everything between ForEach Start and ForEach End runs once per prompt.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) makes the starting noise. A number node named Seed feeds its seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) takes the current prompt and the resized photo as `image_references`.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into an image and sends it to ForEach End.
1. ForEach End collects every image. When the list is done, a Display Image Grid node arranges them in columns. A number node named Gridsize sets the column count. A Display Image node shows the grid.

## Steps

1. Open the **Multi-ViewPromptBatcher** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your photo in the Load Image node. Adjust the Crop Image node so the part you want is selected.
1. Type your prompts in the Text Input node, one per line.
1. Set Gridsize to the number of columns you want in the grid.
1. Run the workflow.

## Defaults

- Model: Qwen-Image-Edit (`Qwen/Qwen-Image-Edit`), with the `QwenImageEditPipeline` type. The text encoder is `Qwen/Qwen2.5-VL-7B-Instruct`.
- Memory: Manual strategy with Sequential CPU offload.
- Image size: 1024 x 1024, set on Create Noise Latent and Resize Image.
- Crop: 4000 x 4000 pixels, starting 1000 pixels from the top of the sample photo.
- Steps: 20.
- Guidance: 4.0.
- Seed: 0, from the Seed node.
- Grid: 3 columns, 10 pixel spacing, 1200 pixels wide.
- Prompts: eight camera instructions, one per line.

## Tips

- Each prompt gives one image, so adding a line adds one image to the grid.
- The Seed node drives the noise for every pass, so the same seed is used for all prompts.
- To copy the loop pattern, put your steps between ForEach Start and ForEach End. Send the item to be repeated into the steps, and the result into ForEach End.
- Add Noise is off on Generate Latent, because the starting latent is already noise. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
