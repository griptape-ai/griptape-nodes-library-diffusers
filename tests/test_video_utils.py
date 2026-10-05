import numpy as np

from modular_diffusion_nodes_library.utils.video_utils import resize_video_frames_torch


def test_resize_video_frames_torch_preserves_float_media_and_shape() -> None:
    frames = np.ones((2, 3, 5, 3), dtype=np.float32)

    resized = resize_video_frames_torch(frames, height=6, width=4)

    assert resized.shape == (2, 6, 4, 3)
    assert resized.dtype == np.float32
    assert np.allclose(resized, 1.0)


def test_resize_video_frames_torch_restores_float16_dtype() -> None:
    frames = np.full((1, 2, 4, 3), 0.25, dtype=np.float16)

    resized = resize_video_frames_torch(frames, height=4, width=2)

    assert resized.shape == (1, 4, 2, 3)
    assert resized.dtype == np.float16
    assert np.allclose(resized, 0.25)
