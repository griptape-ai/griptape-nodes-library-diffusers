import numpy as np
import pytest

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact


def test_float_media_artifact_tracks_shape_and_returns_read_only_array() -> None:
    source = np.ones((3, 4, 5, 3), dtype=np.float32)

    artifact = FloatMediaArtifact(source, color_space="log_gamma")

    assert artifact.shape == source.shape
    assert artifact.dtype == np.dtype(np.float32)
    assert artifact.ndim == 4
    assert artifact.num_frames == 3
    assert artifact.color_space == "log_gamma"
    assert artifact.to_numpy().flags.writeable is False
    assert artifact.to_frames().shape == source.shape
    assert artifact.to_dict()["type"] == "FloatMediaArtifact"


def test_float_media_artifact_normalizes_image_to_one_frame_for_export() -> None:
    artifact = FloatMediaArtifact(np.ones((4, 5, 3), dtype=np.float16))

    frames = artifact.to_frames()

    assert artifact.num_frames == 1
    assert frames.shape == (1, 4, 5, 3)
    assert frames.dtype == np.float16


def test_float_media_artifact_rejects_non_float_and_multi_batch_data() -> None:
    with pytest.raises(TypeError, match="float16 or float32"):
        FloatMediaArtifact(np.zeros((4, 5, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="single video batch"):
        FloatMediaArtifact(np.zeros((2, 3, 4, 5, 3), dtype=np.float32))
