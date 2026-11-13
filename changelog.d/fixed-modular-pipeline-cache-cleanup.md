- Clearing the pipeline cache now unloads ModularPipeline components and removes their offload hooks,
  releasing manager-held model weights before the next build.
