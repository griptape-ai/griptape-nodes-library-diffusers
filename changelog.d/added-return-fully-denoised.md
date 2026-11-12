- `Generate Latent` has a `return_fully_denoised` option. When on, the last step before
  `end_step` is aimed at a fully clean latent instead of the next point on the schedule, so you
  can stop early and still get a denoised result. It is not supported with a masked latent.
