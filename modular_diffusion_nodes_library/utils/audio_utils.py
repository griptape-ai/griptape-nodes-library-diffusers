"""Waveform helpers for drivers that condition on a soundtrack."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Resample a stereo float32 ``(2, num_samples)`` waveform from ``source_rate`` to ``target_rate`` Hz.

    The sample count follows the rate ratio, give or take a few samples at the end, so callers that need an exact
    length pad or trim the result.
    """
    import av
    import numpy as np
    import torch  # type: ignore[reportMissingImports]

    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(waveform.cpu().numpy(), dtype=np.float32), format="fltp", layout="stereo"
    )
    frame.sample_rate = source_rate
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_rate)
    # Flushing with None drains the samples the resampler still holds back.
    chunks = [*resampler.resample(frame), *resampler.resample(None)]
    resampled = np.concatenate([chunk.to_ndarray() for chunk in chunks], axis=-1)
    return torch.from_numpy(resampled).to(device=waveform.device)
