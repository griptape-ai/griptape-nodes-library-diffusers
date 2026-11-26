---
title: What are a text encoder and a tokenizer?
description: How your prompt becomes something a diffusion model can use, and where the text encoder and tokenizer show up in the nodes.
---

The text encoder and the tokenizer are the parts that read your prompt, the text description of what you want. The diffusion model never sees your words, only what these two make from them. They work like a translator that turns your sentence into a set of numbers the model can follow.

## The idea

A prompt is plain text, and the diffusion model works on numbers. Two parts do the conversion, one after the other:

- The **tokenizer** splits your prompt into small pieces called tokens. A token is often a whole word, but a long or unusual word can be split into several.
- The **text encoder** turns those tokens into a block of numbers that describes what the prompt means. The diffusion model reads those numbers at every step, and they steer the [latent](what-is-a-latent.md), the compressed block of numbers the model works on, toward your prompt.

```text
prompt ──tokenizer──▶ tokens ──text encoder──▶ numbers ──(guide the diffusion)──▶ latent
```

The tokenizer and the text encoder are a pair. The tokenizer's list of pieces is the one its text encoder was trained on.

## Where it shows up in the nodes

- [Build Pipeline](../nodes/LatentDiffusionPipelineBuilderNode.md) loads the tokenizer and the text encoder with the rest of the model. Its Component Overrides group has a Text Encoder port and a Tokenizer port for the pipeline types that use them.
- Some pipeline types have more than one pair. Their extra ports are named Text Encoder 2 and Tokenizer 2, and some have Text Encoder 3 and Tokenizer 3. You can connect a Load Component node set to `Text Encoder` or `Tokenizer` to these ports too.
- [Load Component](../nodes/LoadComponent.md) with `component` set to `Text Encoder` or `Tokenizer` swaps in your own.
- There is no separate node for encoding text. The pipeline handles the prompt for you.

## Why it matters to you

- **Wording and length.** Each model has its own tokenizer, so the same prompt can be split differently from one model to the next. A model also has its own limit on how much of a prompt it reads. Prompts that behave well in one model can lose their ending or read differently in another.
- **A second encoder.** Some models use more than one text encoder, each with its own tokenizer. They read the same prompt in different ways, and the model combines the results.
- **Memory.** Text encoders can be large. Together with the diffusion model they take part of the memory a run needs. See the [Memory settings](build-pipeline-by-model.md#memory-settings).

## Why swap it

Most workflows should keep the text encoder that comes with the model. People swap it for a few reasons:

- **Memory.** A GGUF file stores the weights in a compressed, lower-precision format. A quantized text encoder, such as a GGUF one, stores its numbers at lower precision. That makes the file smaller and the encoder use less memory, which leaves more room for the rest of the pipeline.
- **A fine-tuned encoder.** A checkpoint is a saved set of model weights, which are the numbers a model learned in training. Some checkpoints come with a text encoder that was tuned to go with them.
- **A different tokenizer.** This is rare. It only makes sense together with a text encoder that was trained for it.

To swap one, connect a [Load Component](../nodes/LoadComponent.md) node with `component` set to `Text Encoder` or `Tokenizer` to the matching port in Build Pipeline's Component Overrides.

## Things to know

- The text encoder has to fit the diffusion model. If its output size does not match what the diffusion model expects, the run stops with a message on Build Pipeline.
- If the tokenizer was made for a different text encoder, you get a warning and the run continues. The prompt may be read wrongly.
- A text encoder loaded from a single file has to be a `.gguf` file. The file picker in Load Component offers only `.gguf` for `Text Encoder`. Other formats load from a folder or a downloaded Hugging Face repo, which is a named folder of files on Hugging Face.
- A tokenizer is a set of vocabulary and config files, so it has no single-file choice. Load it from a folder or a Hugging Face repo.
- Swapping the text encoder changes how a prompt is read, so the same prompt and seed can give a different image. The seed is the number that fixes the starting noise.

Related: [What is a transformer?](what-is-a-transformer.md), [What is a scheduler?](what-is-a-scheduler.md), [What is a VAE?](what-is-a-vae.md).
