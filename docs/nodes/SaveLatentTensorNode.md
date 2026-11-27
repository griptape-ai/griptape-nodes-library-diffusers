---
description: Saves a latent, the model's compressed working form of a picture or clip, to disk as a torch .pt file.
when_to_use: Use it when you want to keep a raw latent for debugging, or to build a test file for later work outside the editor.
---

## When to use it

This node saves a [latent](../guides/what-is-a-latent.md) to disk as a file. A latent is the compressed form of a picture or clip that the model works on. It is not a picture you can view.

Use it when you want the latent itself, not a decoded picture. It is useful for debugging a pipeline or keeping a test case. To get viewable frames, run the latent through [Decode Media](VaeDecodeNode.md) instead.

## Where it fits

```text
Generate Latent ──┬→ [Save Latent]
                  └→ Decode Media → Save Image
```

## Parameters

### latent_tensor

Shown as `input_latent`. The latent to save. Without one the run stops with a message.

The file holds only the raw tensor. Other information the latent carries, such as its source shape, is not saved. Record it separately if you need it. The tensor is moved to CPU first, so the file loads on any device.

### file_path

Shown as `output_file`. Where to write the `.pt` file. The default is `debug/latent.pt`. A relative path starts from the workspace folder, and missing folders are created. It must not be empty.

### saved_path

The full path of the file that was written. Use it to pass the location to a later node.

## Tips

- Latent files can be large. Video latents are much larger than image latents.
- This library has no node that loads a saved `.pt` latent back into a workflow.
- Saving again to the same path replaces the earlier file.

## See also

[Generate Latent](DiffusionPipelineGenerateLatentNode.md)
