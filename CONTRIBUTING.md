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

- `modular_diffusion_nodes_library/` — the main node package, organized into submodules:
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
    make test
    ```

    Or run a specific test suite:

    ```shell
    make test/unit
    make test/workflows
    ```

### Workflow Test Commands (Nox)

For workflow templates under `tests/workflows`, run these commands from the repository root:

```shell
uv run --group dev nox -s workflow_collect
uv run --group dev nox -s workflow_tests
uv run --group dev nox -s workflow_tests_strict
uv run --group dev nox -s workflow_tests -- --no-cleanup
uv run --group dev nox -s workflow_single -- Text2Image.py
uv run --group dev nox -s workflow_single -- workflows/templates/Text2Image.py
```

Notes:

- `workflow_collect` runs pytest in collect-only mode and still applies preflight deselection.
- `workflow_tests` runs the full workflow test set with automatic per-run folder cleanup.
- `workflow_tests_strict` fails fast when any required model repo is not present in local cache.
- Add `-- --no-cleanup` to preserve per-run folders for debugging.
- `workflow_single` runs only one parametrized workflow test, selected by template filename or path.

Finding generated images/videos:

- Each workflow test runs in its own per-test folder under a fresh OS temp directory (or the path passed via `-- --workflow-runs-dir <path>`), named `<timestamp>_<WorkflowName>`.
- Generated media is written to `outputs/images/` (or `outputs/videos/`) inside that per-test folder.
- Without `--no-cleanup`, the whole per-test folder (including its outputs) is deleted right after the test finishes — pass `-- --no-cleanup` to keep it around for inspection.

Strict vs default behavior:

- Default mode (`workflow_tests`) deselects workflows that miss required cached repos and runs the rest.
- Strict mode (`workflow_tests_strict`) aborts immediately with a missing-repos summary.

Passing extra pytest arguments:

- Use `--` after the Nox session name to forward flags to pytest.
- Example: `uv run --group dev nox -s workflow_tests -- -k Text2Image --maxfail=1`
- Example: `uv run --group dev nox -s workflow_tests_strict -- --collect-only`

1. **Check code quality:**

    ```shell
    make check  # Check linting, formatting, and type errors
    make fix    # Auto-fix issues where possible
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
