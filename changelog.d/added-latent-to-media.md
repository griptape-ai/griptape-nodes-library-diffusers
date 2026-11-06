- Added `Convert Latent to Media`, which currently supports only the LTX-2.5
  diffusion decoder. LTX-2.5 uses chunked neighborhood attention by default to keep decoder memory
  use bounded. This node is intended to host future operations that transform latents into media.
