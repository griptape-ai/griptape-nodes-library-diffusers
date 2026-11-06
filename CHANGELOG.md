# Changelog

All notable changes to the Griptape Modular Diffusion Nodes Library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/2.0.0/), and each version is
the `library_version` in `griptape-nodes-library.json`.
**Breaking** marks a change that can stop a saved workflow from working without edits.

## [Unreleased]

<!-- Entries go in changelog.d/, one file each. See changelog.d/README.md. -->

## [0.8.0] - 2026-11-03

### Added

- Flux3 is available as a provider in the Pipeline Builder, with base (`Flux3Pipeline`) and
  distilled (`Flux3DistilledPipeline`) pipelines, `fp8r` and `nvfp4` quantized transformer variants,
  and an optional prompt upsampler.
- Two new templates show Flux3 layout control from text and from a reference image.
- Five ready-made nodes under `ModularDiffusion/Preconfigured` (Text to Image, Image to Image,
  Inpainting, Text to Video, First/Last Frame to Video) each run a complete pipeline from a single
  node, so you can generate without learning the individual nodes.
- A new Estimate Pipeline Memory node shows a per-component weight and activation breakdown for a
  pipeline and latent, including stacked ControlNets, LoRA adapters, and tiled VAE decoding.
  Estimates are approximate and lean toward overestimating.
- The Pipeline Builder's `provider`, `pipeline_type`, and model repo parameters accept a connected
  string, so another node can choose the model. A wired repo overrides the picker selection.
- Generate Media Latents accepts connected integers for `start_step` and `end_step`, so another
  node can set the denoising range.
- A new command-line tool, `estimate-workflow-memory`, takes a generated workflow `.py` file and
  prints its estimated GPU memory (or JSON with `--json`), without starting Griptape Nodes or
  loading model weights. Run it with `uv run estimate-workflow-memory <workflow.py>`.

### Changed

- The library requires Griptape Nodes engine 0.105.0 or later (previously 0.100.2).
  Update the engine before updating the library.
- Nodes now run in a separate worker process, with the heavy packages (PyTorch, Diffusers, and
  related libraries) installed into the worker's own environment. The editor loads without
  importing them.
- Diffusers is updated from 0.41.0 to 0.42.0.
- Create Noise Latents, Generate Media Latents, Encode Media Latent, and Decode Media Latent free
  unused PyTorch GPU cache after each run, so VRAM reserved by one stage is available to the next.

### Fixed

- Loading a saved workflow no longer modifies the graph.
- Create Noise Latents now produces correctly shaped latents for Flux Fill pipelines.
- LTX2 no longer fails on Apple Silicon (MPS) during RoPE computation or audio decoding.

[Unreleased]: https://github.com/griptape-ai/griptape-nodes-library-diffusers/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/griptape-ai/griptape-nodes-library-diffusers/compare/v0.7.1...v0.8.0
