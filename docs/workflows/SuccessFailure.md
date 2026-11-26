---
description: A demo of success and failure paths. If the first model fails to load, the workflow falls back to a second one and keeps going.
when_to_use: Copy the pattern when you want a workflow to recover from an error, such as running out of memory, instead of stopping. This template makes images from short text prompts.
---

This workflow shows how to make a workflow handle errors. The nodes here have an `exec_out` output that runs the next step when the node succeeds. They also have a `failure` output that runs a different step when the node fails. The template uses it to build a fallback. The template note says it is meant to show how this makes workflows stronger against problems like memory errors, mismatched tensors and empty inputs.

The template builds the same model twice. The first copy is in a group named Pipe A. The second is in a group named Pipe B. The template notes say Pipe A is expected to fail and Pipe B is expected to work. One note reads "Here z image in this exact configuration is expected to fail." Another reads "Here Select a model that is expected to work". When Pipe A fails, the flow falls back to Pipe B and carries on.

Both copies load Z-Image Turbo. They differ in their memory settings. Pipe A uses the Manual memory strategy with a Sequential CPU offload. Pipe B uses the Automatic strategy with no CPU offload.

## What it connects

The workflow uses a Start Flow node at the top and an End Flow node at the bottom. The nodes in between are joined by their flow connections as well as by their data connections.

1. Start Flow starts [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) for Pipe A.
1. If Pipe A builds, [Create Noise Latent](../nodes/NoiseLatentNode.md) runs next, then two Text Input nodes, then an IfElse node. The IfElse node reads Build Pipeline's `was_successful` output. If it is true, [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md) for Pipe A runs, with the red biplane prompt. If Generate Latent fails, a Cancel Workflow node stops the run.
1. If Pipe A fails to build, its `failure` output starts a second Build Pipeline for Pipe B. It has its own Create Noise Latent and joins the same IfElse node. The IfElse node then takes its Else path to the Generate Latent for Pipe B, with the green Spitfire prompt. If that one fails, Cancel Workflow stops the run.
1. Each Generate Latent goes to its own [Decode Media](../nodes/VaeDecodeNode.md). Both Decode Media nodes go to End Flow. A failure from Decode Media, or from the Pipe B build, goes to the failed input of End Flow.

The two prompts are "A red biplane on a run way in the middle of a desert road" and "A green spitfire plane flying in the air over the sea". Which one is used depends on which pipe built. Cancel Workflow ends with the message "Diffusion Process Failed. Please open the UI to investigate the cause."

## Steps

1. Open the **SuccessFailure** template from the workflow templates. If either Build Pipeline shows an **Open Model Manager to Download** button, click it and wait for the download to finish.
1. Run the workflow.
1. Check the prompt that was used to see which pipe made the image. The red biplane prompt goes with Pipe A. The green Spitfire prompt goes with Pipe B.

## Defaults

- Model on both pipes: Z-Image Turbo (`Tongyi-MAI/Z-Image-Turbo`).
- Pipe A: Manual memory strategy, Sequential CPU offload.
- Pipe B: Automatic memory strategy, no CPU offload.
- Size: 1024 x 1024, set on Create Noise Latent.
- Steps: 20.
- Seed: 42.

## Tips

- To copy the pattern, wire the `failure` output of a node that might fail to the `exec_in` of a backup path. Use `was_successful` with an IfElse node when the later steps depend on which path ran.
- Use Cancel Workflow on the final failure so the run stops with a clear message.
- The template note says only that the Pipe A configuration is expected to fail. It does not say why.
- For how Build Pipeline memory settings work, see [How pipelines work](../guides/how-pipelines-work.md#memory-settings).
