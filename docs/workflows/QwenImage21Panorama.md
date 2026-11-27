---
description: Turn one photo into a 360 degree panorama with Qwen Image 2.1, then repaint the strip where the left and right edges meet so the seam disappears.
when_to_use: Use it when you need an equirectangular environment image for a skybox, a VR view or a lighting reference, built from a single reference photo and a prompt.
---

This workflow makes a 2:1 panorama, the flat layout used for a full 360 by 180 degree view, from a reference photo and a text prompt. It uses Qwen Image 2.1, which can edit a picture from a prompt. The template ships with a photo of a person jumping on a cliff above an alpine lake, and a prompt that keeps the person in the centre while the mountains wrap around them.

A panorama made this way has a visible line where the left and right edges meet, which shows as a seam when the image is wrapped around a sphere. The second half of the workflow moves that line to the middle of the picture and repaints a narrow strip over it.

## What it connects

The graph has two groups, **360 Image Generation** and **360 Image Seam Fixing**.

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads Qwen Image 2.1.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the size and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of an image that the model works in.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) reads the prompt from a Text Input node and the photo from a Load Image node, which goes to `reference_images`. It runs the full 40 steps.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into the first panorama.

The seam fix starts from that panorama.

1. A group called **FlipTiles and Make Mask** cuts the panorama into a left half and a right half and joins them in swapped order, which puts the seam in the middle. The same group draws a black picture with a white vertical strip over the seam. The strip is as wide as the **Mask Width** node says.
1. [Encode Media](../nodes/VaeEncodeNode.md) turns the swapped panorama back into a latent.
1. [Set Latent Mask](../nodes/SetLatentMaskNode.md) attaches the strip as a mask. White is repainted and black is kept.
1. A second [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) repaints the masked strip. This is [masked denoising](../guides/generate-latent-by-model.md#masked-denoising).
1. A second [Decode Media](../nodes/VaeDecodeNode.md) gives the final image. A Compare Images node shows the swapped panorama and the fixed one side by side.

## Steps

1. Open the **QwenImage21Panorama** template from the workflow templates. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. The workflow does not run until the model is downloaded.
1. Models needed: `Qwen/Qwen-Image-2.1`. It is released under the Qwen Research License Agreement, so read the license before you use it.
1. Choose your photo in the Load Image node. Edit the prompt in the Text Input node so it describes your own scene and says what in the photo to keep.
1. Run the workflow.
1. View the result. The Decode Media node shows a flat image. Use the 360 Image Viewer node from the Image Tiles library, or any free 360 viewer website, to look around it.

## Defaults

- Model: `Qwen/Qwen-Image-2.1`, with `QwenImage21Pipeline`.
- Size: 3072 x 1536, set on Create Noise Latent. The node default is 1024 x 1024.
- Seed: 42 on Create Noise Latent and on both Generate Latent nodes. Randomize Seed is off.
- First Generate Latent: 40 steps, Add Noise off, `true_cfg_scale` 1.0, no negative prompt. These match the node defaults.
- Second Generate Latent: 40 steps, Add Noise on, `start_step` 10, empty prompt, no reference image. Starting at step 10 of 40 repaints with 75 percent of the work, so the strip is rebuilt but still blends with what is around it.
- Mask: Mask Width 250 pixels. Set Latent Mask uses the red channel, with Blur Mask 10 to soften the edges, no grow or shrink, and Invert Mask off.
- Memory: Build Pipeline is set to **Automatic**. The node default is **Manual**.

## Tips

- 3072 x 1536 is not one of the seven sizes Qwen Image 2.1 was trained on, so Create Noise Latent shows a size warning. The template uses it for the 2:1 shape that a panorama needs. See [Create Noise Latent](../nodes/NoiseLatentNode.md#what-changes-by-pipeline) for the list.
- `true_cfg_scale` is 1.0, which leaves guidance off. Raise it above 1.0 and add a negative prompt if you want guidance. See [Qwen Image 2.1](../guides/generate-latent-by-model.md#qwen-image-21).
- Add Noise is off on the first Generate Latent, so its seed has no effect. Change the seed on Create Noise Latent to get a different panorama. The second Generate Latent has Add Noise on, so its own seed does matter there. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added).
- Raise `start_step` on the second Generate Latent to keep more of the original strip. Lower it to repaint more. See [start_step](../nodes/DiffusionPipelineGenerateLatentNode.md#start_step).
- If you change the image size, update the Crop Image nodes inside the FlipTiles group, because they cut at the halfway point of 3072 x 1536. Widen Mask Width if you can still see the seam.
- The final image is the swapped version. The workflow does not swap the halves back, so the view is turned by half a turn compared with the first panorama.
- The seam fix is not tied to Qwen. Another model that supports masked denoising can repaint the strip in its place.
