- The MiniMax-H3 provider in the Pipeline Builder has a new reference-to-video pipeline type,
  `MiniMaxH3Ref2VAModularPipeline`. It generates video with a soundtrack from up to 9 images and up
  to 3 video clips of at least 2 s each. Connect them to `references` on Generate Latent through the
  Reference Media nodes: one image node, plus one video node per clip. Audio in the clips is
  discarded. If the clips add up to more than 15 s, they are trimmed to fit and the trimming is
  logged. Latents work with either MiniMax-H3 pipeline type. Encode Media accepts video for
  MiniMax-H3, and Generate Latent with `add_noise` on can run Video-to-Video from an encoded
  MiniMax-H3 video, re-noising its soundtrack too.
