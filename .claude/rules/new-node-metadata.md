---
paths:
  - "griptape-nodes-library.json"
  - "preconfigured_workflows/**"
  - "modular_diffusion_nodes_library/nodes/**"
---

# New node metadata

Every node added to `griptape-nodes-library.json` sets two things in its `metadata`.

- **Its own `lifecycle_stage`.** A node without one inherits the library's `STABLE`. New entries in `nodes` start at `ALPHA`, new entries in `workflow_nodes` at `LABS`. Promote a node later by changing its stage. Add the stage to the node's existing `declarations` array if it has one (e.g. next to `model_usage`); a second `declarations` key silently overrides the first.

  ```json
  "declarations": [{ "type": "lifecycle_stage", "stage": "ALPHA" }]
  ```

- **`tags` that help search,** all lowercase: what the node does, other words people use for it, and the models or media it supports. `Convert Latent to Media` carries `standalone decode`, `decode` and `diffusion decode`. Don't repeat the name.

Name the node by [node-naming.md](node-naming.md).
