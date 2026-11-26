# Node documentation format

The editor's Docs panel shows the pages under `docs/`, so write them for artists. The full rules are in the engine guide, `docs/development/custom_nodes/library_documentation.md` in the engine repo. This page lists what this library does with them.

## Where pages go

- One page per node, named after the node's class: `docs/nodes/LatentDiffusionPipelineBuilderNode.md`. The class name is the key in `griptape-nodes-library.json`. Spell it exactly, since a case mismatch fails on case-sensitive disks.
- One page per workflow template, named after the template file without `.py`: `docs/workflows/Text2Image.md` documents `workflows/templates/Text2Image.py`. Write pages only for templates that need explaining.
- `docs/index.md` is the library overview. Any other page outside `docs/nodes/` and `docs/workflows/` is a library page. This library has `docs/guides/` for model-by-model notes and `docs/tools/` for the command-line tool.
- The template's own `description` and `image` in its metadata header give the workflow a thumbnail and a description. The page's `description` replaces the template's.
- Images go under `docs/assets/` and are linked as markdown images with a relative path: `![Build Pipeline](../assets/nodes/pipeline-builder.png)`.
- Developer docs live in `dev/docs/`, not under `docs/`.

## Page shape

```markdown
---
description: One or two sentences on what the node does.
when_to_use: When to pick this node over another.
---

Short text on what the node is for and how it fits in a workflow.

## Parameters

### start_step

How the parameter behaves, in plain words.
```

- Start with frontmatter. Node and workflow pages use exactly `description` and `when_to_use`. Library pages use `title` and `description`.
- No H1 on a page with `title` or `description` frontmatter. A node page is titled with the node's display name.
- Do not list every parameter. The panel builds a table from the node. Add a `## Parameters` section with one `###` heading per parameter, named exactly as the parameter, only where the tooltip is not enough.
- Link to other nodes with `../nodes/<ClassName>.md`. Links to workflows use `../workflows/<TemplateName>.md`. Links to library pages use `../guides/<name>.md`, with `#anchor` for a heading.
- Spell out "Text-to-Image", "Image-to-Image" and "Text-to-Video" in prose. Do not use `t2i`, `i2v` and similar shorthand.

## Style

- Write for artists. State limits as plain facts.
- No raw HTML. The panel removes it.
- No GitHub links, issue numbers or install steps in `docs/`.
- No test notes, first person or comments about the code.

## Checklist

1. The file name matches the class name or template name exactly.
1. The page starts with the right frontmatter and has no H1.
1. Each `###` under `## Parameters` is a real parameter name, used once.
1. Every relative link and image path resolves.
1. Parameters, defaults and model names match the code and the template source.
