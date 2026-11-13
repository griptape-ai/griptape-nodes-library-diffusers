from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="Resamples torch waveforms; run `make test/exec`.")
pytest.importorskip("av", reason="Resamples with PyAV; run `make test/exec`.")

import torch  # noqa: E402

from modular_diffusion_nodes_library.utils.audio_utils import resample_waveform  # noqa: E402


def _tone(frequency: float, sample_rate: int, seconds: float) -> torch.Tensor:
    times = torch.arange(int(sample_rate * seconds), dtype=torch.float32) / sample_rate
    return torch.sin(2 * torch.pi * frequency * times).expand(2, -1).contiguous()


def test_resample_waveform_scales_length_and_keeps_channels_and_dtype() -> None:
    waveform = _tone(440.0, 44_100, 5.0)

    resampled = resample_waveform(waveform, 44_100, 32_000)

    assert resampled.shape[0] == 2
    assert resampled.shape[-1] == pytest.approx(160_000, abs=2)
    assert resampled.dtype == torch.float32


def test_resample_waveform_keeps_the_tone_frequency() -> None:
    waveform = _tone(1_000.0, 44_100, 1.0)

    resampled = resample_waveform(waveform, 44_100, 32_000)

    spectrum = torch.fft.rfftfreq(resampled.shape[-1], d=1 / 32_000)
    magnitude = torch.fft.rfft(resampled[0]).abs()
    assert spectrum[magnitude.argmax()].item() == pytest.approx(1_000.0, abs=2.0)
