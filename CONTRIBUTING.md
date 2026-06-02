# Contributing to Griptape Nodes Modular Diffusion Library

We welcome contributions to the Griptape Nodes Modular Diffusion Library! This library provides modular, latent-level diffusion nodes for [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes), built on Hugging Face 🧨 Diffusers.

## Development Setup

This is a standalone repository — all development happens here.

1. **Clone the repository:**

    ```shell
    git clone https://github.com/griptape-ai/griptape-nodes-library-modular-diffusion.git
    cd griptape-nodes-library-modular-diffusion
    ```

1. **Install `uv`:** Follow the official instructions at [Astral's uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

1. **Install dependencies:**

    ```shell
    uv sync --all-groups --all-extras
    ```

    This creates a `.venv/` and installs runtime and dev dependencies as defined in `pyproject.toml`.

## Project Layout

Top-level layout of this repository:

- `diffusers_nodes_library/` — the main node package, organized into submodules:
    - `nodes/` — node implementations (Pipeline, Create, Processing, Transform, Conditioning, Encode/Decode, IO, etc.)
    - `latent_pipeline_drivers/` — per-model latent pipeline drivers (Flux, Flux Fill, LTX, LTX2, Qwen, Z-Image, etc.)
    - `artifact_utils/` — shared artifact types (`LatentArtifact`, `InpaintMaskArtifact`, …)
    - `parameters/` — reusable parameter components used by nodes
    - `standard_parameters/` — Pipeline builder parameters
    - `runtime_parameters/` — runtime-resolved parameter helpers for each provider
    - `mixins/` — shared node mixins
    - `utils/` — general utilities (pipeline, torch, memory helpers)
    - `misc/` — miscellaneous helpers
- `workflows/` — workflow templates shipped with the library (plus `assets/`)
- `docs/` — documentation sources (`assets/`, `index.md`)
- `tests/` — unit and workflow tests
- `griptape_nodes_library.json` — library manifest (node registry, settings, dependencies, workflows)
- `pyproject.toml` / `pytest.ini` — project and test configuration

## Contributing Code

1. **Make your changes** — follow the existing code structure and style.

1. **Run tests:**

    ```shell
    uv run pytest tests/
    ```

1. **Check code quality:**

    ```shell
    uv run ruff check .
    uv run ruff format . --check
    ```

    To auto-fix:

    ```shell
    uv run ruff check . --fix
    uv run ruff format .
    ```

1. **Submit a pull request** against the `main` branch of this repository. Describe your changes clearly in the PR description.

## Making a Release (Maintainers)

1. Bump the version in `pyproject.toml` and in the `metadata.library_version` field of `griptape_nodes_library.json`.

1. Commit and push:

    ```shell
    git add pyproject.toml griptape_nodes_library.json
    git commit -m "chore: bump griptape-nodes-library-modular-diffusion to vX.Y.Z"
    git push origin main
    ```

1. Go to the [Actions](https://github.com/griptape-ai/griptape-nodes-library-modular-diffusion/actions) tab on GitHub and run the **Publish Version** workflow manually to:

    - Create the version tag (e.g. `vX.Y.Z`)
    - Update the `stable` tag
    - Create a GitHub release with auto-generated notes

## Questions or Issues?

For questions, bugs, or feature requests, please [open an issue](https://github.com/griptape-ai/griptape-nodes-library-modular-diffusion/issues) in this repository.

Thank you for contributing!
