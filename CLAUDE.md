# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Repository

This is the **Modular Diffusion Nodes Library** — a Griptape Nodes library that exposes 🧨 Diffusers pipelines as composable nodes (pipeline builder, VAE encode/decode, noise, denoise, latent math, ControlNet, LoRA, etc.). Each diffusion stage is its own node, so flows can branch, chain, and reorder steps (multi-stage refinement, ControlNet stacking, latent composition, first/last-frame video conditioning, latent upscaling).

It is consumed by the [`griptape-nodes`](../griptape-nodes) engine, which loads it via `griptape-nodes-library.json`.

## Development

**Commands**

All development is uv-backed. Run these directly from the repo root:

```bash
uv sync --all-groups --all-extras            # install deps
make check                                   # format, lint, pyright, JSON (what CI runs)
make fix                                     # auto-fix format and lint
make test/unit                               # unit tests (excludes tests/workflows)
```

`make check` runs `ruff format --check`, `ruff check`, `pyright .`, and JSON validation. Underlying ruff commands: `uv run ruff format --check` / `uv run ruff check .`.

**Iteration Loop**

1. **Make the change**: implement the feature or fix.
2. **Run checks**: `make check` to surface lint, format, and type issues.
3. **Fix issues**: resolve everything from the previous step (`make fix` handles autofixable ones).
4. **Run tests**: `make test/unit` once checks pass.
5. **Continue working**: move on to the next change.

## Working Principles

Adapted from the [Karpathy CLAUDE.md guidelines](https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md). The full text there has the rationale; the rules below are the ones we enforce in this repo.

**Think before coding.** State assumptions explicitly. If a request is ambiguous, name the interpretations and ask — do not pick silently. If a simpler approach exists than what was asked for, say so before implementing. If you are confused, stop and name what's unclear.

**Simplicity first.** Write the minimum code that solves the problem. No speculative features, configurability, abstractions for single-use code, or error handling for impossible scenarios. If 200 lines could be 50, rewrite it. (This generalises the Exception Handling rules below.)

**Surgical changes.** Every changed line should trace directly to the user's request.

- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that aren't broken.
- Match the existing style of the file you're editing, even if you'd do it differently elsewhere.
- Remove imports/variables/functions that *your* changes orphaned. Do not delete pre-existing dead code unless asked — mention it instead.

**Goal-driven execution.** For multi-step work, state a brief plan with per-step verification before starting:

```
1. <step> → verify: <check>
2. <step> → verify: <check>
```

Prefer verifiable goals ("write a failing test for X, then make it pass") over imperative ones ("add validation for X").

## Code Style Preferences

**Avoid Tuples For Return Values** — Tuples should be a last resort. When unavoidable, use `NamedTuple` for clarity. Prefer separate variables, dataclasses, or other named structures.

**Simple, Readable Logic Flow** — Prefer simple, easy-to-follow logic over complex nested expressions:

- Use explicit if/else instead of ternaries or nested conditionals
- Break complex nested expressions into clear, separate statements
- Bad: `value = func() if condition else None`
- Good:
    ```python
    if condition:
        value = func()
    else:
        value = None
    ```

**CRITICAL: Evaluate ALL failure cases first, success path ONLY at the end** — MANDATORY unless explicitly asked otherwise:

- All validation checks, error conditions, and failure cases go at the top of the function
- Each failure case exits immediately (return/raise) with a clear error message
- The success path is at the absolute bottom — never return in the middle
- Multiple returns in the success path usually means you're doing it wrong (unless explicitly requested)
- Bad:
    ```python
    def process_data(value):
        if value > 0:
            result = calculate(value)
            return result  # SUCCESS IN MIDDLE
        return "Error: value must be positive"
    ```
- Good:
    ```python
    def process_data(value):
        if value <= 0:
            return "Error: value must be positive"
        if value > 1000:
            return "Error: value too large"

        result = calculate(value)
        return result
    ```

**CRITICAL: Do NOT use lazy imports** — Imports MUST be at the top of the file:

- All imports at the top of the file, standard order
- NEVER use lazy imports (imports inside functions) unless required to resolve an unavoidable circular import
- If you think you need a lazy import, STOP, explain why, and ASK for confirmation
- If you must use one, add a comment naming the exact circular import being resolved
- Bad:
    ```python
    def process_data(value):
        from some_module import helper  # NO! Move to top

        return helper(value)
    ```

## Exception Handling

**CRITICAL: Only wrap code that actually raises exceptions** — Never add try/except speculatively:

- Verify the code actually raises (docs, source, type hints) before adding handling
- Do not add try/except "just in case"
- If unsure, ASK

**Use specific, narrow exception blocks** — Broad handling makes debugging impossible:

- Catch only the specific exception types that can be raised (e.g. `FileNotFoundError`, not `Exception`)
- Keep try blocks as small as possible — wrap only the exact lines that raise
- Each distinct failing operation gets its own try/except with a specific error message
- Never use bare `except:` or catch `Exception` unless explicitly required

**Include context in error messages** — Use the format **"Attempted to do X. Failed with data Y because of Z."**:

- Include `{self.name}` in node error messages when available
- Include relevant parameter names, repo IDs, pipeline class names, provider, etc.
- Bad:
    ```python
    logger.warning("Invalid input received")
    return "Error: Processing failed"
    ```
- Good:
    ```python
    return f"Attempted to build pipeline. Failed with repo='{repo_id}' provider='{provider}' because '{pipeline_cls}' is not registered in _DRIVER_REGISTRY."
    ```

## Architecture Overview

**Top-level layout** (under `modular_diffusion_nodes_library/`):

- `nodes/` — the user-facing Griptape nodes (Pipeline Builder, Generate Latent, VAE Encode/Decode, Noise/Empty/Add Latent, LoRA, ControlNet, Latent Math, Conditioning, Save Latent Tensor, etc.). Each node is registered in `griptape-nodes-library.json`.
- `latent_pipeline_drivers/` — per-model drivers (`flux`, `flux2`, `qwen`, `z_image`, `ltx`, `wan`, `wan_i2v`, `stable_diffusion_xl`, `stable_diffusion_3`, `flux_fill`, …) all extending `LatentPipelineDriver` in `base_driver.py`. `driver_factory.py` maps a diffusers pipeline class name → driver class.
- `standard_parameters/` — per-model **load-time** parameter sets (repo IDs, dtype, quantization, offload). Subclass `ModularDiffusionPipelineTypePipelineParameters`.
- `runtime_parameters/` — per-model **generate-time** parameter sets (prompt, negative_prompt, guidance, true_cfg, etc.). Subclass `DiffusionPipelineRuntimeParameters` — never re-add `num_inference_steps`, `seed`, or `generator`; the base class owns those.
- `parameters/` — shared parameter machinery: `providers.py` (the `Provider` enum), `pipeline_parameters.py` (runtime-parameter dispatch), `pipelinetype_parameters.py` (load-time dispatch + `MODULAR_PIPELINE_TYPE_PROVIDER_MAP`).
- `artifact_utils/` — `LatentArtifact`, `PipelineArtifact`, `InpaintMaskArtifact` and helpers exchanged between nodes.
- `mixins/` — reusable behavior for nodes (live preview, cancellation, etc.).
- `misc/` — orthogonal helpers, notably `partial_denoise.py` (timestep slicing for multi-stage workflows).
- `utils/` — generic utilities (tensor helpers, dtype mapping, hashing).

**Driver contract and modular-blocks-first**: read [`.claude/rules/latent-drivers.md`](.claude/rules/latent-drivers.md) before touching `latent_pipeline_drivers/`, `nodes/`, or `misc/partial_denoise.py`. The public latent contract (unpacked, normalised) must not break.

**Provider / pipeline-class registration — the three places**

Adding a new diffusers pipeline requires touching all three:

1. `latent_pipeline_drivers/driver_factory.py` → `_DRIVER_REGISTRY` (pipeline class name → driver class)
2. `parameters/pipeline_parameters.py` → `set_runtime_parameters()` `match` (pipeline class name → runtime parameters class)
3. `parameters/pipelinetype_parameters.py` → a `LatentPipelineTypeParameters` subclass plus an entry in `MODULAR_PIPELINE_TYPE_PROVIDER_MAP` (`Provider` enum → type-parameters class)

A miss in any of the three surfaces as a confusing UI failure rather than an import error. See [docs/adding-new-model.md](docs/adding-new-model.md) for the full 6-step walkthrough.

**Every offered repo id must also be declared in the `model_catalog`** in `griptape-nodes-library.json`, and its catalog key added to the owning node's `model_usage` list. The HuggingFace dropdowns refuse an undeclared repo rather than allowing it, so a repo id added in Python alone is unusable at runtime. `tests/test_model_catalog_consistency.py` fails on the omission. This requires engine 0.97.0+ — see Step 6 of the walkthrough.

**Pipeline caching**

The Pipeline Builder caches the loaded pipeline in memory and reuses it across runs. Cache invalidation is driven by hashing the dict returned from `get_config_kwargs()` on the standard parameters. If a load-time parameter affects pipeline identity, it must appear there.

## Node Development

**Creating a node:**

1. Add the implementation under `modular_diffusion_nodes_library/nodes/<your_node>.py`, extending `BaseNode` from the engine.
2. Register it in `griptape-nodes-library.json` (category, metadata, file path).
3. Restart the engine — the node appears under the **ModularDiffusion** categories in the node picker.
4. **Document the node.** Add a page under `docs/nodes/<your_node>.md` following [docs/node-doc-format.md](docs/node-doc-format.md) and link it from `docs/index.md`. Use the `document-node` skill.

**Wiring a node to a model:**

- Load-time parameters (what repo, what dtype, what quantization) go in `standard_parameters/`.
- Runtime parameters (prompt, guidance, etc.) go in `runtime_parameters/`.
- All pipeline I/O goes through the driver — never call `DiffusionPipeline.__call__()` from a node directly.

**Modifying a node:** any change to a node's parameters, inputs, outputs, defaults, category, display name, or provider-specific behavior MUST be accompanied by an update to its `docs/nodes/<name>.md` page in the same change. Invoke the `document-node` skill for the workflow.

## Verification

When adding a model or driver:

1. `uv run ruff format --check` + `uv run ruff check .` — clears lint/format.
2. The relevant `docs/nodes/<name>.md` pages exist and match the changed code (per the `document-node` skill).
3. Open Griptape Nodes, drop a `LatentDiffusionPipelineBuilderNode`, select your provider + pipeline type, fill the repo, resolve.
4. Wire to a `DiffusionPipelineGenerateLatentNode` + VAE decoder; confirm an image/video is produced.
5. Test partial denoise by chaining two generate nodes (e.g. 0–10 then 10–20 steps).
6. Test cancellation mid-run.

## When in doubt

- **Existing drivers are the source of truth.** Pick the closest match and copy its structure.
- **Read the diffusers source** under `.venv/Lib/site-packages/diffusers` before assuming a modular block exists.
- **Don't bypass `LatentPipelineDriver.denoise_latent()`** for callback / partial-denoise / cancellation concerns — those are framework-level open problems.
- Check the feature docs in `docs/` (see `docs/adding-new-model.md`) and the path-scoped rules in `.claude/rules/` before re-deriving a subsystem's behaviour. Verify against current source, since docs can drift.
