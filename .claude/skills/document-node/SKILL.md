---
name: document-node
description: 'Author or update the docs page for a node or workflow template in this library. Use whenever a node is added, an existing node gains, loses or renames a parameter, its category or display name changes, its inputs or outputs change, or provider-specific behavior is added or removed. Also use when a page has drifted from the code. Produces or edits docs/nodes/<ClassName>.md (or docs/workflows/<TemplateName>.md) following dev/docs/node-doc-format.md. DO NOT use for code changes to the node itself; pair this with add-modular-diffusion-pipeline or add-pipeline-variants when shipping a feature.'
---

# Document a Diffusion Node

The docs under `docs/` are shown to artists in the editor's Docs panel. The format and file naming are in [dev/docs/node-doc-format.md](../../../dev/docs/node-doc-format.md). Read it before writing.

## When to invoke this skill

| Trigger | Action |
|---|---|
| A node was added to `modular_diffusion_nodes_library/nodes/` and registered in `griptape-nodes-library.json`. | **Create** `docs/nodes/<ClassName>.md`. |
| A node's parameters, inputs, outputs, defaults, category or display name changed. | **Update** its page. |
| Provider-specific or pipeline-type-specific behavior was added or removed. | **Update** the affected `###` entries under `## Parameters`, and the matching section in `docs/guides/build-pipeline-by-model.md` or `docs/guides/generate-latent-by-model.md`. |
| A workflow template was added, changed or removed. | **Create, update or delete** `docs/workflows/<TemplateName>.md` (template file name without `.py`). |
| A node was renamed or removed. | **Rename or delete** its page, and fix links to it. |

A pure refactor with no user-visible change does not need a page update.

## Rules

1. **The code is the source of truth.** Read the node file, the runtime and standard parameters classes it uses, and its `griptape-nodes-library.json` entry before writing. For a workflow page, read the template file under `workflows/templates/`. Never invent parameter names, defaults or model names.
1. **Name the page after the class.** The file name is the class name from the manifest, spelled exactly. Pages sit flat under `docs/nodes/`.
1. **Frontmatter first.** Exactly `description` and `when_to_use` on node and workflow pages. No H1, no `Category:` line.
1. **Do not list every parameter.** The panel builds the table. Add `## Parameters` with one `###` per parameter, named as the parameter, only where the tooltip is not enough.
1. **Markdown images only**, from `docs/assets/`, with a relative path. No raw HTML.
1. **Link other nodes** as `../nodes/<ClassName>.md` and workflows as `../workflows/<TemplateName>.md`. Every link must resolve.
1. **Write for artists.** State limits as plain facts. No GitHub links, issue numbers, test notes or comments about the code. Spell out "Text-to-Image", "Image-to-Image" and "Text-to-Video".
1. **Make surgical edits.** When updating a page, change only what the code change affected.
1. **Keep the overview current.** When a node is added, renamed or removed, check whether `docs/index.md` and the pages under `docs/guides/` mention it.

## Workflow

1. Gather the display name, class name, category, parameters, defaults and provider-specific branches from the code.
1. Create or edit the page.
1. Run the checklist at the end of [dev/docs/node-doc-format.md](../../../dev/docs/node-doc-format.md).

## Output to the user

End with a short report: files created or edited, whether `docs/index.md` changed, and any code and docs mismatches you found but did not fix.
