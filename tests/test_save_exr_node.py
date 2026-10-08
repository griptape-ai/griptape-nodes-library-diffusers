import numpy as np
import pytest

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.nodes.save_exr_node import (
    DEFAULT_TRANSFER_FUNCTION,
    TRANSFER_FUNCTION_CHOICES,
    prepare_exr_frames,
)


def test_transfer_function_choices_keep_diffhdr_as_default() -> None:
    assert TRANSFER_FUNCTION_CHOICES == ["None", "Log-Gamma", "ARRI LogC3"]
    assert DEFAULT_TRANSFER_FUNCTION == "Log-Gamma"


@pytest.mark.parametrize("transfer_function", ["None", "Log-Gamma", "ARRI LogC3"])
def test_prepare_exr_frames_selects_transfer_conversion(
    monkeypatch: pytest.MonkeyPatch, transfer_function: str
) -> None:
    source = np.full((2, 3, 4, 3), 0.4, dtype=np.float32)
    media = FloatMediaArtifact(source)
    diffhdr_result = np.full_like(source, 3.0)
    logc3_result = np.full_like(source, 4.0)

    def diffhdr(frames: np.ndarray) -> np.ndarray:
        return diffhdr_result

    def logc3(frames: np.ndarray) -> np.ndarray:
        return logc3_result

    monkeypatch.setattr("modular_diffusion_nodes_library.nodes.save_exr_node.inverse_log_gamma", diffhdr)
    monkeypatch.setattr("modular_diffusion_nodes_library.nodes.save_exr_node.inverse_logc3", logc3)

    result = prepare_exr_frames(media, transfer_function)

    if transfer_function == "Log-Gamma":
        np.testing.assert_array_equal(result, diffhdr_result)
    elif transfer_function == "ARRI LogC3":
        np.testing.assert_array_equal(result, logc3_result)
    else:
        np.testing.assert_array_equal(result, source)
        assert result.dtype == np.float32
