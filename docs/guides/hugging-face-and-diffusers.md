---
title: Hugging Face and Diffusers
description: How the Diffusion Library gets its models from Hugging Face, and how to handle gated models, the cache and offline use.
---

[Hugging Face](https://huggingface.co) is the main website where diffusion models are shared. The Diffusion Library downloads its models from there and runs them with [Diffusers](https://huggingface.co/docs/diffusers), the Hugging Face software library for diffusion models. This page covers downloads, access tokens, where models are stored on your disk, and offline use.

## How the two fit together

- A model is a folder of files on Hugging Face, split into parts. This split is called the Diffusers format. The parts are:
  - the transformer, which makes the picture: [What is a transformer?](what-is-a-transformer.md)
  - the VAE, which translates between pixels and the model's working format, the latent: [What is a VAE?](what-is-a-vae.md)
  - the text encoder and tokenizer, which read your prompt: [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md)
  - the scheduler settings, which set the pace of the steps: [What is a scheduler?](what-is-a-scheduler.md)

  Because each part is a separate file, Build Pipeline can replace one part, such as the VAE, and leave the others as they are.
- A repo is a named folder of files on Hugging Face. A model is named by its repo, such as `black-forest-labs/FLUX.1-dev`. That name is what you see in the node pickers and in the Load Component repo field.
- Model Manager downloads a model from Hugging Face into your local Hugging Face cache. Build Pipeline loads from that cache and never downloads anything itself. If the model is missing, Build Pipeline shows an **Open Model Manager to Download** button.

## Gated models

Some models, such as FLUX.1 dev, need you to accept the model's terms on its Hugging Face page before you can download it. To set that up:

1. Create a Hugging Face account and open the model's page. Accept its terms and wait for access to be granted.
1. Create an access token with **Read** access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
1. In Griptape Nodes, open **Settings → API Keys & Secrets** and add the token as `HF_TOKEN`.
1. Start the download again in Model Manager.

A token also helps with private repos, and with Hugging Face rate limits on downloads. Treat it like a password and keep it out of screenshots and shared workflows.

## The cache

The cache is the folder where downloaded models are kept. By default it is a folder in your user directory (`~/.cache/huggingface/hub`), unless you changed it. Model Manager lists and deletes models there. Models are large, often tens of gigabytes for a single repo, so check free disk space before you download.

Studio IT or a pipeline TD can change how Hugging Face behaves with settings called environment variables. They are set in the environment Griptape Nodes is launched from:

- `HF_HOME` or `HF_HUB_CACHE` sets where models are stored, for example on a shared network drive.
- `HF_HUB_OFFLINE=1` blocks all internet requests to Hugging Face. Models already in the cache keep working. Downloads and searches in Model Manager fail, and models that are not cached cannot run.
- `HF_ENDPOINT` points to an internal mirror of Hugging Face instead of the public site.

Restart Griptape Nodes after changing them. The [Hugging Face integration guide](https://docs.griptapenodes.com/en/stable/guides/integrations/hugging_face/) in the Griptape Nodes docs covers these in more detail.

## Models that are not on Hugging Face

Model files already on your disk work too. [Load Component](../nodes/LoadComponent.md) accepts a single checkpoint file (a saved set of model weights), a local folder in Diffusers format, or a repo that is already in the cache. For a text encoder, the single-file picker offers only `.gguf` files. Other formats load from a folder or a cached repo, as the [text encoder guide](what-is-a-text-encoder.md) explains. [Load Scheduler](../nodes/LoadSchedulerComponent.md) reads scheduler settings from a `scheduler_config.json` file, a cached repo or JSON text.
