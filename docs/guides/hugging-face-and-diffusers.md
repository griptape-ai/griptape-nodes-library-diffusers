---
title: Hugging Face and Diffusers
description: How the Diffusion Library gets its models from Hugging Face, and how to handle gated models, the cache and offline use.
---

The Diffusion Library is built on [Diffusers](https://huggingface.co/docs/diffusers), the Hugging Face library for running diffusion models. Its models are hosted on [Hugging Face](https://huggingface.co), the main place diffusion models are shared.

## How the two fit together

- A model is a folder of files on Hugging Face, usually one for each part: the transformer, the VAE, the text encoder, the tokenizer and the scheduler settings. This is the Diffusers format, and it is why Build Pipeline can swap one part at a time. The guides for each part are [What is a transformer?](what-is-a-transformer.md), [What is a VAE?](what-is-a-vae.md), [What are a text encoder and a tokenizer?](what-is-a-text-encoder.md) and [What is a scheduler?](what-is-a-scheduler.md).
- A model is named by its Hugging Face repo, such as `black-forest-labs/FLUX.1-dev`. That name is what you see in the node pickers and in the Load Component repo field.
- Model Manager downloads a model from Hugging Face into your local Hugging Face cache. Build Pipeline loads from that cache and never downloads anything itself. If the model is missing, Build Pipeline shows an **Open Model Manager to Download** button.

## Gated models

Some models, such as FLUX.1 dev, need you to accept the model's terms on its Hugging Face page before you can download it. To set that up:

1. Create a Hugging Face account and open the model's page. Accept its terms and wait for access to be granted.
1. Create an access token with **Read** access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
1. In Griptape Nodes, open **Settings → API Keys & Secrets** and add the token as `HF_TOKEN`.
1. Start the download again in Model Manager.

A token also helps with private repos, and with Hugging Face rate limits on downloads. Treat it like a password and keep it out of screenshots and shared workflows.

## The cache

Downloaded models live in the Hugging Face cache, `~/.cache/huggingface/hub` unless you changed it. Model Manager lists and deletes models there. Models are large, often tens of gigabytes for a single repo, so check free disk space before you download.

Teams can control Hugging Face behavior with the standard Hugging Face environment variables, set in the environment you launch Griptape Nodes from:

- `HF_HOME` or `HF_HUB_CACHE` points to a different cache, such as a shared network drive.
- `HF_HUB_OFFLINE=1` stops all requests to Hugging Face. Cached models keep working. Downloads and searches in Model Manager fail, and models that are not cached cannot run.
- `HF_ENDPOINT` points to an internal Hugging Face mirror.

Restart Griptape Nodes after changing them. The [Hugging Face integration guide](https://docs.griptapenodes.com/en/stable/guides/integrations/hugging_face/) in the Griptape Nodes docs covers these in more detail.

## Models that are not on Hugging Face

Weights that are already on your disk work too. [Load Component](../nodes/LoadComponent.md) accepts a single checkpoint file, a local folder in Diffusers format, or a repo that is already in the cache. [Load Scheduler Component](../nodes/LoadSchedulerComponent.md) reads scheduler settings from a `scheduler_config.json` file, a cached repo or JSON text.
