---
paths:
  - "griptape-nodes-library.json"
  - "preconfigured_workflows/**"
  - "modular_diffusion_nodes_library/nodes/**"
---

# Naming nodes

The `display_name` in `griptape-nodes-library.json` follows three rules.

- **Start with a verb:** `Build Pipeline`, `Load LoRA`, `Upsample Latent`. Pairs share a noun: `Load ControlNet` / `Apply ControlNet`.
- **Say "Latent" in the singular** unless the node takes two or more: `Generate Latent`, `Save Latent`, but `Add Latents`.
- **Keep the name short and put extra detail elsewhere.** No brackets, no library name, no implementation detail. How it works goes in the `description`, and alternative search terms go in `tags`, all lowercase. `Convert Latent to Media` carries `standalone decode` as a tag, not in its name.

Preconfigured workflow nodes take the form `<Input> to <Output>` (`Text to Image`). One that has no single input starts with a verb (`Inpaint Image`).

Change only the `display_name`, never the class name or `node_type`: saved workflows refer to the class name, so renaming it breaks them. The same holds for parameters: change `display_name`, not `name`.
