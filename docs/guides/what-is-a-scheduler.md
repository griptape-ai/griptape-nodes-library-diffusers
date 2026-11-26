---
title: What is a scheduler?
description: What the scheduler controls during denoising, and when to change it in the nodes.
---

The scheduler is the rule that sets the pace of a generation. A diffusion model starts with noise and cleans it up over a number of steps. The scheduler decides how much noise to remove at each step. A change of scheduler can change the look or the speed of a result. It works on a [latent](what-is-a-latent.md), the compressed block of numbers that stands for an image or video.

## The idea

The cleaning-up is called denoising. At each step the [transformer](what-is-a-transformer.md), the part that makes the picture, estimates what the noise looks like. The **scheduler** decides how much of it to remove and what the latent should look like before the next step. In other words, it sets the spacing of the steps and the noise schedule that runs from fully noisy to clean. Many tools call this the sampler.

The scheduler holds no learned weights. It is a method plus a handful of settings, so there is nothing large to load.

```text
noisy latent ──(transformer estimates, scheduler steps)──▶ ... ──▶ clean latent
```

## Where it shows up in the nodes

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) uses the scheduler that comes with the model. Its Component Overrides group has a Scheduler port for the pipeline types that have one.
- [Load Scheduler](../nodes/LoadSchedulerComponent.md) chooses a scheduler class and its settings, and outputs a scheduler for that port.
- The number of steps, one round of clean-up each, is set in [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md), not on the scheduler.

## Why change it

Most workflows should keep the model's own scheduler. People change it for a few reasons:

- **A different look.** Different schedulers can give a slightly different character to the same prompt and seed, such as smoother or more varied results.
- **Speed.** Some schedulers reach a good result in fewer steps, which makes a run shorter.
- **Tuning.** A setting such as `shift` changes how the steps are spread across the noise range. Put it in the Load Scheduler `text_config` as JSON, for example `{"shift": 3.0}`.
- **Matching a fine-tune.** A checkpoint may be meant to run with a particular scheduler.

To change it, connect a [Load Scheduler](../nodes/LoadSchedulerComponent.md) node to the Scheduler port in Build Pipeline's Component Overrides. A change of scheduler needs no extra memory.

## Things to know

- Use a scheduler class from the same family as your model. Flow-matching classes and classic classes are not interchangeable. A class from the wrong family gives wrong output or an error during generation, and a warning appears when the config names a class from a different family than the one you chose.
- A scheduler whose prediction type differs from the base model's shows a warning on Build Pipeline. The run is not stopped.
- The class you pick in Load Scheduler is always the one used. The class name stored in a config file is ignored.
- With `text_config` empty, the class's own defaults are used. Keys the class does not use are ignored, and a warning lists them.
- A scheduler changes the path the denoising takes. The same seed, the number that fixes the starting noise, can give a different image after a change.

Related: [What is a text encoder?](what-is-a-text-encoder.md), [What is a VAE?](what-is-a-vae.md).
