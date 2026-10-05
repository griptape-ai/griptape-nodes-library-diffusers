from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from griptape_nodes.exe_types.node_types import SuccessFailureNode

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.nodes.save_exr_node import (
    DEFAULT_TRANSFER_FUNCTION,
    TRANSFER_FUNCTION_CHOICES,
    SaveExrNode,
    exr_output_folder,
    exr_sequence_macro_path,
    prepare_exr_frames,
)


def test_transfer_function_choices_default_to_none() -> None:
    assert TRANSFER_FUNCTION_CHOICES == ["None", "Log-Gamma", "ARRI LogC3"]
    assert DEFAULT_TRANSFER_FUNCTION == "None"


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


def test_add_parameter_after_init_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    base_add = MagicMock()
    monkeypatch.setattr(SuccessFailureNode, "add_parameter", base_add)
    node = SaveExrNode.__new__(SaveExrNode)
    node._initializing = False

    node.add_parameter(MagicMock())

    base_add.assert_not_called()


def test_add_parameter_during_init_is_added(monkeypatch: pytest.MonkeyPatch) -> None:
    base_add = MagicMock()
    monkeypatch.setattr(SuccessFailureNode, "add_parameter", base_add)
    node = SaveExrNode.__new__(SaveExrNode)
    node._initializing = True
    param = MagicMock()

    node.add_parameter(param)

    base_add.assert_called_once_with(param)


@pytest.mark.parametrize(
    ("output_file", "expected"),
    [
        ("frame.exr", "{outputs}"),
        ("renders/shot.exr", "{outputs}/renders"),
    ],
)
def test_exr_output_folder_is_anchored_to_outputs(output_file: str, expected: str) -> None:
    assert exr_output_folder(output_file) == expected


def test_exr_output_folder_keeps_absolute_output_file(tmp_path: Path) -> None:
    assert exr_output_folder(str(tmp_path / "shot.exr")) == tmp_path.as_posix()


def test_exr_sequence_macro_path_puts_version_in_folder() -> None:
    macro = exr_sequence_macro_path("renders/shot.exr", "/abs/outputs/renders")

    assert macro.parsed_macro.template == "{folder}/shot_v{_index:03}/shot.####.exr"
    assert macro.variables == {"folder": "/abs/outputs/renders"}
