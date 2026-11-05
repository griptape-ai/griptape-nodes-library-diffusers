from __future__ import annotations

import pytest

from modular_diffusion_nodes_library.utils.minimax_h3_ref2va_utils import plan_reference_video_trim


@pytest.mark.parametrize(
    ("durations", "kept_seconds"),
    [
        ([5.0, 5.0], (5.0, 5.0)),
        ([8.0, 6.0, 3.0], (8.0, 6.0, None)),
        ([10.0, 10.0, 5.0], (10.0, 5.0, None)),
        ([20.0], (15.0,)),
    ],
)
def test_reference_video_trim_examples(
    durations: list[float],
    kept_seconds: tuple[float | None, ...],
) -> None:
    plan = plan_reference_video_trim(durations)

    assert plan.kept_seconds == kept_seconds
    if sum(durations) <= 15.0:
        assert plan.warning_message is None
    else:
        assert plan.warning_message is not None
        assert f"Total reference video duration is {sum(durations):.1f} s" in plan.warning_message


def test_reference_video_trim_warning_describes_the_trim() -> None:
    plan = plan_reference_video_trim([8.0, 8.0])

    assert plan.warning_message == (
        "Total reference video duration is 16.0 s, over the 15 s limit. Video 2 trimmed from 8.0 s to 7.0 s."
    )


def test_video_shorter_than_two_seconds_is_rejected_before_trimming() -> None:
    with pytest.raises(ValueError, match="Video 1 is 1.5 s"):
        plan_reference_video_trim([1.5, 20.0])
