from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("numpy", reason="Builds real NumPy frames; run `make test/exec` for the execution environment.")

import numpy as np  # noqa: E402
from griptape_nodes.exe_types.node_types import SuccessFailureNode  # noqa: E402

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact  # noqa: E402
from modular_diffusion_nodes_library.nodes.save_exr_node import (  # noqa: E402
    SaveExrNode,
    exr_output_folder,
    exr_sequence_macro_path,
    prepare_exr_frames,
)


def test_prepare_exr_frames_preserves_input_values_without_conversion() -> None:
    source = np.full((2, 3, 4, 3), 2.5, dtype=np.float32)
    source[0, 0, 0, 0] = -0.25
    media = FloatMediaArtifact(source)

    result = prepare_exr_frames(media)

    np.testing.assert_array_equal(result, source)
    assert result is media.to_frames()
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
