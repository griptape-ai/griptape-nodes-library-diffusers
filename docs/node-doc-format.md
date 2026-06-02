# Node Documentation Format

This page is the canonical template for every node reference page under [docs/nodes/](nodes/). Match it exactly when authoring or updating a node doc — the consistency is what lets users scan dozens of pages quickly.

## File location and naming

- One page per node, flat under `docs/nodes/`.
- File name is the snake_case form of the node's display name with the `Node` suffix dropped — e.g. `LatentCompositeMaskNode` → `latents_composite_mask.md`, `VaeDecodeNode` → `decode_media_latent.md`. Match the spelling already used in [griptape_nodes_library.json](../griptape_nodes_library.json).
- Companion screenshot lives at `docs/assets/nodes/<kebab-case-of-display-name>.png`. If the screenshot does not exist yet, leave an inline `<!-- TODO: add docs/assets/nodes/<filename>.png screenshot -->` marker where the `<img>` would go and surface the gap in the PR description.

## Page template

Copy this skeleton verbatim. Section headings, ordering, and casing are fixed; the placeholder values are the only things you change.

```markdown
# <Display Name as shown in the node picker>

**<One-line role, ending in a period. Bolded. No blockquote.>**

Category: `<ModularDiffusion/<Group>>`

## TL;DR
- 2–4 bullets. Each bullet is a complete sentence. Lead with the most actionable fact.
- Cover: what the node *does*, what its key inputs/outputs are, and the single most common gotcha.

## Typical workflow position
​```text
Upstream Node → [<Display Name>] → Downstream Node
​```

## Node preview

<img src="../assets/nodes/<kebab-name>.png" alt="<Display Name>" width="480">

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `<input>` | `<ArtifactType>` | Yes / No | <Short description.> |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `<output>` | `<ArtifactType>` | <Short description.> |

## Parameters
<!-- OPTIONAL. Group rows by the parameter group shown in the UI. Use H3s if more than one group. -->

| Name | Type | Default | Notes |
| --- | --- | --- | --- |
| `<param>` | <type / choices> | `<default>` | <Short description. Mention units and bounds.> |

## Provider / model behavior
<!-- OPTIONAL. Use only when behavior or available parameters differ by provider, pipeline type, or input shape. -->

## Tips & pitfalls

- **<Imperative summary in bold.>** <One-sentence explanation.>
- **<Next pitfall.>** <Explanation.>
```

## Conventions

### Terminology

- **Spell out workflow names.** Use "Text-to-Image", "Image-to-Image", "Text-to-Video", "Image-to-Video", "Video-to-Video" in prose, headings, and tooltips. Do **not** use shorthand like `t2i`, `i2i`, `t2v`, `i2v`, `v2v`, `img2img`, `vid2vid`, `txt2img` — they are unfriendly to first-time users and to translation.
- Provider names (Flux, SDXL, WAN, LTX, Qwen, Z-Image, etc.) keep their canonical casing.

### Header block

- `#` heading is the human-friendly display name (with spaces, title case).
- The bold tagline directly under the `#` heading is the elevator pitch — one sentence, ends with a period, **not** wrapped in a blockquote.
- The `Category:` line is plain text with the category in inline code. Do **not** include the class name; do not use bold labels (`**Category:**`); do not add `Class:` / `Module:` / version metadata.

### TL;DR

- 2 to 4 bullets — never more. If you need more, you are writing the *Tips* section.
- Each bullet starts with the most useful fact, not setup. Prefer “Output is a latent — decode with VAE Decode” over “This node will produce …”.

### Typical workflow position

- A fenced ` ```text ` block, **not** per-line backticks. Per-line backticks render as inline code on separate paragraph lines and break visual alignment of arrows and box-drawing characters.
- Wrap the node being documented in square brackets so the eye can find it: `[Generate Media Latents]`.
- Keep it to 1–3 lines. Use box drawing only when the graph genuinely branches.

### Node preview

- Always use HTML `<img ... width="480">`, not markdown `![]()`. The fixed width keeps screenshots proportionally sized across all pages.
- Path is relative: `../assets/nodes/<file>.png`.
- `alt` attribute is the display name.
- If the screenshot is missing, omit the `<img>` and leave the `<!-- TODO ... -->` marker described above. Do not commit broken image links.

### Inputs / Outputs / Parameters tables

- Columns and casing are fixed; do not invent new columns or rename them.
- `Name` cells use inline code for the actual parameter name as it appears in code/UI.
- `Type` cells use inline code for artifact / scalar types (`LatentArtifact`, `int`, `str`, `bool`, choice lists like `canny | tile | depth`).
- `Required` is `Yes` or `No`, no other values.
- `Default` is inline code for the literal default (`0`, `True`, `"hf://..."`).
- `Notes` is one short sentence. If you need a paragraph, move it to *Tips & pitfalls*.

### Optional sections

- `## Parameters` — omit if the node has no configurable parameters beyond its inputs.
- `## Provider / model behavior` — include only when behavior varies by provider, pipeline type, or runtime input. Use H3 subheadings per provider when relevant.

### Tips & pitfalls

- 2–6 bullets. Each begins with a **bold imperative** summarising the takeaway, followed by one sentence of explanation.
- Cover real failure modes users actually hit (shape mismatches, silent dropping, encoding-only-applies-to-image-pipelines, etc.). Do not fill with platitudes.

### See also

- Link to sibling nodes only (`name.md`), separated by ` · `.
- Aim for 2–4 links. Prefer the nodes most commonly chained with this one.

## Quality checklist (run before opening a PR)

1. Page matches the section ordering above exactly.
2. Category string matches the node entry in [griptape_nodes_library.json](../griptape_nodes_library.json).
3. `<img>` references an existing file in [docs/assets/nodes/](assets/nodes/) **or** carries a TODO marker.
4. Page is registered in the appropriate group of [docs/index.md](index.md).
5. All `See also` links resolve to existing files.
6. The displayed workflow ASCII renders correctly in VS Code's markdown preview (test by opening the file).
