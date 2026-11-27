- Clearing the pipeline cache now unloads ModularPipeline components and removes their offload hooks,
  releasing manager-held model weights before the next build. Models are freed straight from the GPU
  instead of being copied into system RAM first, so clearing a large pipeline no longer spikes RAM.
