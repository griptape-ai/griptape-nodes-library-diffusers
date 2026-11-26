---
description: Type a description, get a standard video, then turn that video into HDR frames and save them as an EXR sequence.
when_to_use: Use it when you need high dynamic range footage for compositing. The template also shows how to upsample a video latent for comparison. Use Text2Image for a plain still.
---

This workflow makes a video in two stages with LTX-2.3. The first stage turns a text description into an ordinary (SDR) video. The second stage uses the LTX-2.3 HDR IC-LoRA to turn that video into HDR frames, which keep much more detail in the brights and darks. An IC-LoRA is a small add-on file that uses a reference video to steer the new video. The result is saved as an OpenEXR sequence, one file per frame, ready for a compositing program.

The template also has a third group that upsamples the first-stage latent and decodes it. That branch is only for comparison. The template note says its output is not fed into the HDR stage.

The prompt describes a slow pan inside a Gothic cathedral at midnight.

## What it connects

The template is laid out in three groups named Text-to-Video, Latent Upsampling and SDR-to-HDR.

Text-to-Video:

1. [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the model.
1. [Create Noise Latent](../nodes/NoiseLatentNode.md) sets the size, the length and the seed. It makes random noise as a [latent](../guides/what-is-a-latent.md), the compressed form of a video that the model works in.
1. Two Text Input nodes hold the prompt and the negative prompt, which lists what to steer away from.
1. [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) makes the first-stage video. See [LTX2](../guides/generate-latent-by-model.md#ltx2) for the settings specific to this model.
1. [Decode Media](../nodes/VaeDecodeNode.md) turns the result into the SDR video.

Latent Upsampling (comparison only):

1. [Upsample Latent](../nodes/LatentUpsamplerNode.md) doubles the width and height of the first-stage latent. In the template it is named Latent Upsampler.
1. [Convert Latent to Media](../nodes/LatentToMediaNode.md) decodes it with its own decoder. In the template it is named Latent to Media (Standalone Decoder).

SDR-to-HDR:

1. [Load LoRA](../nodes/LoadLora.md) points at the HDR IC-LoRA file.
1. [Apply Runtime LoRA](../nodes/LoraActivationPipelineNode.md) switches the LoRA on for the second stage. In the template it is named LoRA Pipeline. It takes its pipeline from the first-stage Decode Media, so both stages use the same loaded model.
1. [Create Empty Latent](../nodes/EmptyLatentNode.md) makes a blank latent at twice the first-stage size.
1. [Reference Media](../nodes/MediaGenConditioningNode.md) holds the SDR video from the first stage as a video at frame 0. In the template it is named Media Generation Conditioning. It goes to `reference_conditions` on the second Generate Latent.
1. The second [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) makes the HDR frames. It has no prompt. The IC-LoRA path reads scene embeddings from a file instead.
1. [Decode HDR Media](../nodes/DecodeHdrNode.md) turns the result into float frames and a tone-mapped preview. In the template it is named Decode HDR Latents.
1. [Save EXR Sequence](../nodes/SaveExrNode.md) writes the frames to disk.

## Steps

1. Open the **LTX23-HDR-Text2Video-Upsample-Two-Stage** template from the workflow templates.
1. In Model Manager, download the model that Build Pipeline shows, the upsampler `dg845/LTX-2.3-Spatial-Upsampler-Diffusers` for Upsample Latent, and `Lightricks/LTX-2.5-Diffusers` for Convert Latent to Media. If Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish. If a download asks for access on Hugging Face, see [Gated models](../guides/hugging-face-and-diffusers.md#gated-models).
1. Get the HDR IC-LoRA files. The template notes name the Hugging Face repo `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`. You need `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors` and `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors`. The library does not ship them. Set the file path on Load LoRA to the first file and `text_embeddings_path` on the second Generate Latent to the second file.
1. Describe your scene in the first Text Input node.
1. Set the file name on Save EXR Sequence.
1. Run the workflow. Save EXR Sequence and Decode HDR Media must run in the same run, because the float frames are not stored in a saved workflow.

## Defaults

- Model: LTX 2.3 Distilled (`dg845/LTX-2.3-Distilled-Diffusers`), with the `LTX2Pipeline` type.
- First-stage size: 768 x 512, set on Create Noise Latent.
- HDR-stage size: 1536 x 1024, set on Create Empty Latent.
- Length: 121 frames on both. About 4.8 seconds at 25 frames per second.
- Frame rate: 25 frames per second on Decode Media and Decode HDR Media.
- Steps: 8 for each stage.
- First-stage guidance: 3.0. HDR-stage guidance: 1.0.
- Seed: 42.
- LoRA weight: 1.0.
- Tone mapping: `aces_filmic`. This changes only the preview, not the saved frames.
- Save EXR Sequence: `videos/church_sequence.exr`, with half float on.
- Upsampler: `dg845/LTX-2.3-Spatial-Upsampler-Diffusers`.
- Comparison decoder: LTX-2.5, `Lightricks/LTX-2.5-Diffusers`, 1 step.

## Tips

- The template notes say the HDR IC-LoRA needs an empty input latent and a video as conditioning. Both are set up here.
- Add Noise is off on both Generate Latent nodes in this template. See [When noise is added](../guides/what-is-a-latent.md#when-noise-is-added) for how the setting works with other inputs.
- To use the upsampled video, the template note suggests a refinement pass without the LoRA. It says the LTX2 diffusion decoder can improve how steady the output is from frame to frame.
- For how pipelines load and reuse a model, see [How pipelines work](../guides/how-pipelines-work.md). For LTX2 pipeline types, see [Build Pipeline by model](../guides/build-pipeline-by-model.md#ltx2).
