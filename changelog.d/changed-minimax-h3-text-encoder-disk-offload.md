- MiniMax-H3 pipelines in the Pipeline Builder no longer park the 62 GB Qwen3-VL text encoder in system
  RAM when it is moved off the GPU. Its weights are dropped and reloaded from the model files on disk
  the next time it runs, which greatly reduces the RAM held between runs and helps a repeated Generate
  Latent run. `Build Pipeline` also now loads MiniMax-H3 entirely from the downloaded revision you select, 
  instead of failing when offline.
