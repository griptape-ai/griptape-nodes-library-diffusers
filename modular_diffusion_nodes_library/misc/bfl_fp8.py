from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]

_PROBE_SIZE = 16
_FP8_SCALED_MM_MIN_CAPABILITY = (8, 9)


def require_fp8r_support(device: torch.device) -> None:
    """Raise if `BFLFP8RLinear` would hit a per-row `torch._scaled_mm` that `device` cannot run."""
    import torch  # type: ignore[reportMissingImports]

    # Diffusers only takes the scaled-matmul path on CUDA devices with compute capability 8.9+.
    if device.type != "cuda":
        return
    if torch.cuda.get_device_capability(device) < _FP8_SCALED_MM_MIN_CAPABILITY:
        return

    activations = torch.zeros(_PROBE_SIZE, _PROBE_SIZE, device=device).to(torch.float8_e4m3fn)
    weight = torch.zeros(_PROBE_SIZE, _PROBE_SIZE, device=device).to(torch.float8_e4m3fn)
    row_scale = torch.ones(_PROBE_SIZE, 1, dtype=torch.float32, device=device)
    column_scale = torch.ones(1, _PROBE_SIZE, dtype=torch.float32, device=device)
    try:
        torch._scaled_mm(  # noqa: SLF001  # type: ignore[reportPrivateImportUsage]
            activations, weight.T, row_scale, column_scale, out_dtype=torch.bfloat16, use_fast_accum=True
        )
    except RuntimeError as e:
        msg = (
            f"Attempted to use FLUX.3 fp8r weights on '{device}'. Failed because per-row scaled matmul "
            f"is not supported by this GPU or torch build ({e}). Select the default or nvfp4 variant instead."
        )
        raise RuntimeError(msg) from e
