---
name: document-node
description: 'Author or update the user-facing documentation page for a node in modular_diffusion_nodes_library. Use whenever a new node is added, an existing node gains/loses/renames a parameter, its category or display name changes, its inputs/outputs change shape, or a provider/pipeline-type-specific behavior is added or removed. Also use when a docs page is missing a screenshot or has drifted from the node implementation. Produces or edits a page under docs/nodes/<name>.md following docs/node-doc-format.md, updates docs/index.md if the page is new, and flags missing screenshots. DO NOT use for code changes to the node itself — pair this with add-modular-pipeline or add-pipeline-variants when shipping a feature.'
---

# Document a Modular Diffusion Node

This skill creates or updates a single node reference page under [docs/nodes/](../../docs/nodes/) so it stays consistent with the rest of the library docs and accurate to the node implementation.

The **canonical template and field conventions** live in [docs/node-doc-format.md](../../docs/node-doc-format.md). This skill is the *workflow* that produces a page matching that template; the template itself is not duplicated here.

## When to invoke this skill

| Trigger | Action |
|---|---|
| A new node was added to `modular_diffusion_nodes_library/nodes/` and registered in `griptape_nodes_library.json`. | **Create** `docs/nodes/<name>.md` and add a link to `docs/index.md`. |
| An existing node's parameters, inputs, outputs, category, or display name changed. | **Update** the existing `docs/nodes/<name>.md` and `docs/index.md` if naming/grouping changed. |
| A provider-specific or pipeline-type-specific behavior was added/removed. | **Update** the `Provider / model behavior` section. |
| Default values, bounds, or required-ness of a parameter changed. | **Update** the affected row in the Parameters / Inputs / Outputs tables. |
| Screenshot is missing from `docs/assets/nodes/`. | Leave the documented TODO marker, surface the gap in your final report. |

If the request is purely a code change with no user-observable surface change (refactor, comment, internal helper), this skill does not apply.

## Working Principles

These apply to every step below.

1. **Source of truth is the code, not your assumptions.** Open the node file, the runtime/standard parameters classes it touches, and `griptape_nodes_library.json` before writing. Never invent parameter names, defaults, or types.
2. **Template is fixed.** Match [docs/node-doc-format.md](../../docs/node-doc-format.md) exactly — section order, casing, header style, table columns. Do not "improve" the template per page.
3. **Surgical updates.** When editing an existing page, change only the rows / paragraphs that the code change actually affected. Do not reformat, reorder, or rewrite unrelated sections.
4. **User-attention budget is small.** Lead with the most useful fact. The TL;DR is for *actionable* info, not a feature list. Cap bullets at the template's limits.
5. **Verify cross-references.** A page is not done until its category matches `griptape_nodes_library.json`, its `See also` links all resolve, and `docs/index.md` lists it.

## ⛔ Mandatory Workflow Rules

### Rule 0 — Read the template first

Open [docs/node-doc-format.md](../../docs/node-doc-format.md) and skim every section before writing. The conventions there override anything you remember from other pages — recent edits to siblings may have set new norms.

### Rule 1 — Read the code before writing the page

Before drafting or editing, you MUST have read, in this order:

1. The node implementation under `modular_diffusion_nodes_library/nodes/<file>.py`.
2. The `griptape_nodes_library.json` entry (for `category`, display name, and metadata).
3. The relevant runtime-parameters class under `runtime_parameters/` if the node hosts dynamic parameters.
4. The relevant standard-parameters class under `standard_parameters/` if the node loads a model.
5. (When editing) the existing `docs/nodes/<name>.md` page.

If you skipped a step, your parameter tables will be wrong. Re-read and try again.

### Rule 2 — One page per node, flat hierarchy

- Pages live directly under `docs/nodes/`. No category subfolders.
- File name follows the convention in [docs/node-doc-format.md § File location and naming](../../docs/node-doc-format.md#file-location-and-naming).
- Cross-links between node pages are siblings: `[Other Node](other_node.md)`.

### Rule 3 — Image policy

- Always use HTML `<img src="../assets/nodes/<file>.png" alt="<Display Name>" width="480">`. Never markdown `![]()`.
- If the screenshot file does not exist, do NOT commit a broken link. Insert `<!-- TODO: add docs/assets/nodes/<file>.png screenshot -->` in its place and report the missing file in your final summary.

### Rule 4 — Update `docs/index.md` when creating or renaming

- New page → add it under the matching `###` group heading in `docs/index.md`.
- Renamed page → update the link target and display text.
- Removed node → delete the bullet from `docs/index.md` and delete the page file.

### Rule 5 — Spell out workflow names

Write "Text-to-Image", "Image-to-Image", "Text-to-Video", "Image-to-Video", "Video-to-Video" in full. Do not use the shorthand `t2i`, `i2i`, `t2v`, `i2v`, `v2v`, `img2img`, `vid2vid`, `txt2img`. See [docs/node-doc-format.md § Terminology](../../docs/node-doc-format.md#terminology).

## Workflow

### Phase A — Gather

Per Rule 1, read the code first. Capture the following before you touch any markdown:

| Field | Source |
|---|---|
| Display name | `griptape_nodes_library.json` → `display_name` |
| Class name | `griptape_nodes_library.json` → `class_name` |
| Category | `griptape_nodes_library.json` → `category` |
| Inputs | `BaseNode.add_parameter(...)` calls with `allowed_modes` containing `INPUT` |
| Outputs | `add_parameter(...)` with `allowed_modes` containing `OUTPUT` |
| Parameters | `add_parameter(...)` with `PROPERTY`; also any runtime/standard params hosted dynamically |
| Defaults / bounds / choices | `default_value`, `Options(choices=...)`, `Slider(min_val=..., max_val=...)` |
| Provider-specific behavior | branches in `process()` / dynamic-parameter dispatch keyed off provider or pipeline class |

### Phase B — Plan

State a one-sentence plan covering:

- Which file(s) you will create or edit.
- Whether `docs/index.md` needs updating (creation / rename only).
- Whether a screenshot is missing.

For pure updates touching ≤ 2 sections, the plan can be a single sentence; no need to wait for confirmation.

### Phase C — Write or update

1. Apply the [page template](../../docs/node-doc-format.md#page-template) verbatim for new pages.
2. For updates, edit only the rows or paragraphs whose backing code changed. Do not reflow other sections.
3. Verify the workflow ASCII renders in VS Code's markdown preview before finishing — per-line backticks break alignment; use a fenced ` ```text ` block.

### Phase D — Verify

Run the [quality checklist](../../docs/node-doc-format.md#quality-checklist-run-before-opening-a-pr). The page is not done until every item passes.

## Output to the user

End with a short report covering:

- Files created or edited (with workspace-relative links).
- Whether `docs/index.md` was updated and why.
- Any missing screenshots flagged with their expected paths under `docs/assets/nodes/`.
- Any code↔docs discrepancies you spotted but did **not** fix (those are bugs in the node, not the doc, and need a separate change).
