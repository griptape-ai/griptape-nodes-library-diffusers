---
description: Give it the first and last frames of a clip, plus masks, and it fills in the video. It replaces a masked object, or it extends the picture past its edges.
when_to_use: Use it when you want to change one part of a video, or turn a wide shot into a square one, while the rest stays as it is. Use FirstAndLastFrameImage2Video when you only need motion between two frames.
---

This workflow edits a video with Wan 2.1 VACE. VACE takes text, images and masks together. The masks tell it which parts to keep and which parts to make new from the prompt. A mask is a black and white picture, and it works like a matte.

The template shows two examples at once. Inpainting replaces a masked area, here a cup. Outpainting extends a 16:9 picture to a 1:1 square by adding new content above and below. The sample files are the first and last frames of a green tea shot, with a mask for each.

## What it connects

1. Four Load Image nodes supply the first frame, the last frame, and the mask for each.
1. Four Extend Canvas nodes sit after the Load Image nodes, one per image. They are set to a 1:1 square with a white background. Padding added around the source frames becomes new area for the model to fill. The same step on the masks makes the padding white, so it counts as generate.
1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model. The `pipeline_type` is `WanVACEPipeline`.
1. Two [Reference Media](../nodes/MediaGenConditioningNode.md) nodes hold the frames. Both use the First + Last preset with two images. One holds the source frames and the other holds the mask frames.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the video size, the length and the seed. A number node named Width feeds both its width and its height. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node. The source frames go to `source_media` and the mask frames go to `mask`. See [WAN VACE](../guides/generate-latent-by-model.md#wan-vace) for the settings specific to this model.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into a video. Set the playback speed in its `fps` parameter.

## Steps

1. Open the **VideoInpaintingAndOutpaintingWANVace** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Load your first frame and your last frame in the first two Load Image nodes.
1. Load a mask for each frame in the other two Load Image nodes. In a mask, black is kept from the source and white is generated from the prompt. The template note says that if no mask is connected, VACE makes one from the source: covered positions are kept and uncovered positions are generated.
1. Describe the full result in the Text Input node.
1. Run the workflow.

## Defaults

- Model: Wan 2.1 VACE 1.3B (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`).
- Size: 1280 x 1280. The Width node sets both sides.
- Length: 81 frames, set as `num_frames` on Create Noise Latent. About 3.2 seconds at 25 frames per second.
- Frame rate: 25 frames per second, set as `fps` on Decode Media.
- Steps: 20.
- Guidance: 4 in the template. The WAN VACE default is 5.0.
- Conditioning scale: 1.0. A higher value follows the source more closely.
- Seed: 42.

## Tips

- To change the video size, change the Width node. It drives both the width and the height of Create Noise Latent.
- For outpainting, extend the source frames and set the mask padding to white. The template does this with the 1:1 square setting.
- For inpainting, paint the mask over the object to replace. The template note says you can paint it in the workflow with a Paint Mask node or make it with an Image Segmentation node.
- `reference_images` on Generate Latent is empty in this template.
- Add Noise is off on Generate Latent, because the starting latent is already noise. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md).
