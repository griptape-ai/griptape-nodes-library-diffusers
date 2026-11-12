# Apply Runtime LoRA

**Layers non-fused LoRA adapters on top of an existing pipeline so the same cached model can power multiple branches simultaneously.**

Category: `ModularDiffusion/Pipeline`

## TL;DR
- Connect the pipeline output from the Build Pipeline node and one or more `loras` outputs from Load LoRA nodes. Then connect this node's `lora_pipeline` output to Generate Latent.
- Adapters are activated per generation and released afterward — the base pipeline is **never permanently modified**, so a single cached model can drive multiple branches (one with LoRAs, one without) without rebuilding.
- Use this node instead of fusing LoRAs with the Build Pipeline node when swapping adapters between runs or using in-context (IC) LoRAs, distillation/acceleration LoRAs, or slider LoRAs.

## Typical workflow position
```text
Build Pipeline ────┐
                   ├─→ [Apply Runtime LoRA] → Generate Latent → Decode Media
Load LoRA ─────────┘
```

## Node preview

<img src="../assets/nodes/activate-lora-pipeline.png" alt="Apply Runtime LoRA" width="480">

## Inputs

| Name | Type | Required | Notes |
| --- | --- | --- | --- |
| `pipeline` | `Pipeline Config` | Yes | Base diffusion pipeline. Connect the output of the Build Pipeline node. The base pipeline is reused, not modified, so you can connect it to other nodes too. |
| `loras` | `loras` | Yes | One or more LoRA payloads from Load LoRA nodes. Accepts a list; connect multiple Load LoRA outputs to stack adapters. |

## Outputs

| Name | Type | Notes |
| --- | --- | --- |
| `lora_pipeline` | `Pipeline Config` | Pipeline reference that activates the listed LoRAs around each generation call. Shares the cache entry of the input pipeline — wiring this to Generate Latent does not trigger a rebuild. |
| `logs` | str | Build log, including the pipeline configuration hash. |

## Tips & pitfalls

- **Activation LoRAs differ from fused LoRAs.** Build Pipeline permanently merges LoRAs connected to its `loras` input into the weights. This node activates LoRAs only during generation. Changing a fused LoRA evicts the pipeline cache; changing an activation LoRA does not.
- **Connect at least one LoRA.** The node requires at least one entry in `loras` to activate — wire one or more [Load LoRA](load_lora.md) nodes before running.
- **The `lora_pipeline` output shares the cache with the input `pipeline`.** You can wire both to separate Generate Latent nodes (one with LoRAs, one without) without loading the model twice.
- **Memory estimates include runtime adapter weights.** When the output of an Apply Runtime LoRA node is connected to Estimate Pipeline Memory, each distinct connected adapter contributes its header-derived weight bytes to the transformer estimate. Multiple adapters are aggregated in that visible transformer row; the detailed adapter list is shown separately in human-readable units in the widget and source-only CLI.
- **LoRA weights are applied per run.** Unlike LoRAs that the Build Pipeline node fuses into the model, these weights can change between runs without rebuilding the pipeline.

## See also

- [Load LoRA](load_lora.md) — provides the `loras` input.
- [Build Pipeline](pipeline_builder.md) — provides the base `pipeline` input.
- [Generate Latent](generate_media_latents.md) — consumes the `lora_pipeline` output.
