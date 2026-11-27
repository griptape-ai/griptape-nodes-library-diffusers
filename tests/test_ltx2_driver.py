from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from modular_diffusion_nodes_library.latent_pipeline_drivers.ltx2 import (
    _fit_hdr_references_to_shared_length,
    _fit_reference_frame_count,
)


@dataclass
class _Reference:
    frames: Any
    strength: float


def _video(length: int, strength: float) -> _Reference:
    return _Reference(frames=list(range(length)), strength=strength)


@pytest.mark.parametrize(
    ("reference_frames", "num_frames", "expected_frames"),
    [
        (30, 121, 33),  # rounded up to the next 1 + 8k
        (33, 121, 33),  # already valid, unchanged
        (130, 121, 121),  # longer than the output, trimmed
        (98, 100, 97),  # output length isn't 1 + 8k, so the cap is the largest 1 + 8k within it
    ],
)
def test_reference_fits_valid_vae_frame_count(reference_frames: int, num_frames: int, expected_frames: int) -> None:
    frames = list(range(reference_frames))

    fitted = _fit_reference_frame_count(frames, num_frames, temporal_ratio=8)

    assert len(fitted) == expected_frames
    assert fitted[: min(reference_frames, expected_frames)] == frames[:expected_frames]


def test_padding_repeats_the_last_frame() -> None:
    fitted = _fit_reference_frame_count(list(range(30)), 121, temporal_ratio=8)

    assert fitted[29:] == [29, 29, 29, 29]


def test_hdr_videos_are_padded_to_the_longest_with_their_last_frame() -> None:
    short = _video(33, 0.3)
    long = _video(121, 1.0)

    fitted = _fit_hdr_references_to_shared_length([long, short])

    assert [len(reference.frames) for reference in fitted] == [121, 121]
    assert fitted[1].frames[32:] == [32] * (121 - 32)


def test_hdr_padding_keeps_each_references_strength_and_leaves_the_input_untouched() -> None:
    short = _video(33, 0.3)

    fitted = _fit_hdr_references_to_shared_length([short, _video(41, 1.0)])

    assert [reference.strength for reference in fitted] == [0.3, 1.0]
    assert len(short.frames) == 33


def test_hdr_images_only_are_unchanged() -> None:
    references = [_Reference(frames="image-a", strength=1.0), _Reference(frames="image-b", strength=0.5)]

    assert _fit_hdr_references_to_shared_length(references) == references


def test_hdr_image_with_video_is_rejected() -> None:
    references = [_Reference(frames="image", strength=1.0), _video(33, 0.5)]

    with pytest.raises(ValueError, match="found 1 image"):
        _fit_hdr_references_to_shared_length(references)


def test_hdr_image_with_one_frame_video_is_allowed() -> None:
    references = [_Reference(frames="image", strength=1.0), _video(1, 0.5)]

    assert _fit_hdr_references_to_shared_length(references) == references
