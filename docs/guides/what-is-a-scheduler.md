---
title: What is a scheduler?
description: What the scheduler controls during denoising, and when to change it in the nodes.
---

The scheduler decides how big each step is, so the picture goes from fully noisy to clean over the number of steps you choose. It is a bit like the timing curve on an animation. The start and end stay the same, and the curve sets how the steps are spaced in between. A change of scheduler can change the look or the speed of a result. It works on a [latent](what-is-a-latent.md), the compressed block of numbers that stands for an image or video.

## The idea

The cleaning-up is called denoising. At each step the [transformer](what-is-a-transformer.md), the part that makes the picture, estimates what the noise looks like. The **scheduler** decides how much of it to remove and what the latent should look like before the next step. Some tools call a closely related setting the sampler. The two are not the same thing.

The scheduler has no learned numbers of its own. It is a method plus a handful of settings, so there is nothing large to load.

```text
noisy latent ──(transformer estimates, scheduler steps)──▶ ... ──▶ clean latent
```

## Where it shows up in the nodes

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) uses the scheduler that comes with the model. Its Component Overrides group has a Scheduler port for the pipeline types that have one.
- [Load Scheduler](../nodes/LoadSchedulerComponent.md) chooses a scheduler class and its settings, and outputs a scheduler for that port. A class is the name of one scheduling method, such as `FlowMatchEulerDiscreteScheduler` or `DDIMScheduler`.
- The number of steps, one round of clean-up each, is set in [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md), not on the scheduler.

## Why change it

Most workflows should keep the model's own scheduler. People change it for a few reasons:

- **A different look.** Two schedulers can give a different image from the same prompt and seed, because they take different paths from noise to picture.
- **Speed.** Some schedulers reach a good result in fewer steps, which makes a run shorter.
- **Tuning.** A setting such as `shift` changes how the steps are spread between noisy and clean. Put it in the Load Scheduler `text_config` as JSON (visible only when `source_type` is Text Config), for example `{"shift": 3.0}`. Which values suit a model is set by the model's own `scheduler_config.json`, so start from that.
- **Matching a fine-tune.** A checkpoint may be meant to run with a particular scheduler.

To change it, connect a [Load Scheduler](../nodes/LoadSchedulerComponent.md) node to the Scheduler port in Build Pipeline's Component Overrides. A change of scheduler needs no extra memory.

## Things to know

- Schedulers come in two families. Flow-matching schedulers have `FlowMatch` in the class name, except `LTXEulerAncestralRFScheduler`. Flux, Stable Diffusion 3, Qwen, Z-Image and LTX use them. Stable Diffusion XL uses a classic scheduler such as Euler. Wan pipelines use `UniPCMultistepScheduler` set up for flow-style prediction. Check a model's own `scheduler_config.json` for its default class. Use a class from the same family as your model. A class from the wrong family gives wrong output or an error during generation. Load Scheduler shows a warning if a config file comes from a different family than the class you chose.
- Prediction type is what the model was trained to work out at each step. Many classic schedulers have this setting. Flow-matching ones do not. Build Pipeline shows a warning when an override scheduler's prediction type differs from the base model's, or when only one of them has one. The run is not stopped.
- Load Scheduler always uses the class you pick in its dropdown. If you load a config file that names a different class, only the file's settings are used and its class name is ignored.
- If `text_config` is empty, the scheduler uses its built-in default settings. If it holds settings the chosen class does not recognize, they are ignored and a warning lists them.
- A scheduler changes the path the denoising takes. The same seed, the number that fixes the starting noise, can give a different image after a change.

Related: [What is a text encoder?](what-is-a-text-encoder.md), [What is a VAE?](what-is-a-vae.md).
