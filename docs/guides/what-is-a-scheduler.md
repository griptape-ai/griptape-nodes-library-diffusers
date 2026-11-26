---
title: What is a scheduler?
description: What the scheduler controls during denoising, and when to change it in the nodes.
---

The scheduler decides how big each step is, so the picture goes from fully noisy to clean over the number of steps you choose. A step is one round of clean-up. It is a bit like the timing curve on an animation. The start and end stay the same, and the curve sets how the steps are spaced in between. A change of scheduler can change the look or the speed of a result. It works on a [latent](what-is-a-latent.md), the compressed block of numbers that stands for an image or video.

## The idea

The cleaning-up is called denoising. At each step the [transformer](what-is-a-transformer.md) guesses the noise and the **scheduler** decides how much of that guess to remove. Other tools use the word sampler for a similar setting.

The scheduler has no learned numbers of its own. It is a method plus a handful of settings, so there is nothing large to load.

```text
noisy latent ──(transformer estimates, scheduler steps)──▶ ... ──▶ clean latent
```

## Where it shows up in the nodes

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) uses the scheduler that comes with the model. Its Component Overrides group has a Scheduler port for the pipeline types that have one.
- [Load Scheduler](../nodes/LoadSchedulerComponent.md) chooses a scheduling method and its settings, and outputs a scheduler for that port.
- The number of steps is set in [Generate Latent](../nodes/DiffusionPipelineGenerateLatentNode.md), not on the scheduler.

## Why change it

Most workflows should keep the model's own scheduler. People change it for a few reasons:

- **A different look.** Two schedulers can give a different image from the same prompt and seed, because they take different paths from noise to picture.
- **Speed.** Some schedulers reach a good result in fewer steps, which makes a run shorter.
- **Tuning.** A setting such as `shift` changes how the steps are spread between noisy and clean. In Load Scheduler, set `source_type` to Text Config and type the setting into `text_config`, for example `{"shift": 3.0}`. Good values depend on the model. Copy starting values from the model's scheduler settings file (`scheduler_config.json`) in its Hugging Face folder.

To change it, connect a [Load Scheduler](../nodes/LoadSchedulerComponent.md) node to the Scheduler port in Build Pipeline's Component Overrides. A change of scheduler needs no extra memory.

## Things to know

- Schedulers come in two kinds, flow-matching and classic, and a model works with only one. Leave the scheduler at the model's default unless you know which kind your model uses. The wrong kind gives bad output or an error during generation. Load Scheduler shows a warning if a settings file looks like it is for a different kind than the scheduling method you chose. Flux, Stable Diffusion 3, Qwen, Z-Image and LTX use flow-matching. Stable Diffusion XL uses a classic scheduler. WAN uses UniPC set up for flow-style prediction.
- Prediction type is a setting for how the model's output is interpreted at each step. Leave it alone. Many classic schedulers have it and flow-matching ones do not. Build Pipeline shows a warning when two parts disagree, that is when an override scheduler's prediction type differs from the base model's or only one of them has one. The run is not stopped.
- Load Scheduler always uses the scheduling method you pick in its dropdown. If you load a settings file that names a different method, only the file's settings are used and its method name is ignored.
- If `text_config` is empty, the scheduler uses its built-in default settings. If it holds settings the chosen method does not recognize, they are ignored and a warning lists them.
- A scheduler changes the path the denoising takes, so the same seed can give a different image after a change.

Related: [What is a text encoder?](what-is-a-text-encoder.md), [What is a VAE?](what-is-a-vae.md).
