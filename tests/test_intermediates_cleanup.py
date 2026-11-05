"""Directory cleanup trims the folder preview intermediates are written to.

Previews are UI-only files, so they belong under the static files root rather than the project's outputs.
Each preview is kept as a new file, and cleanup bounds the folder they accumulate in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import PIL.Image
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.os_manager import OSManager

from modular_diffusion_nodes_library.utils import directory_utils
from modular_diffusion_nodes_library.utils.pillow_utils import pil_to_image_artifact

if TYPE_CHECKING:
    import pytest


def test_preview_is_saved_under_static_files_and_that_folder_is_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GTN_CONFIG_WORKSPACE_DIRECTORY", str(tmp_path))
    GriptapeNodes()
    settings = {
        "modular_diffusion_library.enable_directory_cleanup": True,
        "modular_diffusion_library.max_directory_size_gb": 1.0,
    }
    monkeypatch.setattr(directory_utils, "get_config_value", lambda key, default=None: settings.get(key, default))
    cleaned: list[Path] = []
    monkeypatch.setattr(
        OSManager,
        "cleanup_directory_if_needed",
        staticmethod(lambda full_directory_path, max_size_gb: cleaned.append(Path(full_directory_path))),
    )

    for _ in range(2):
        pil_to_image_artifact(PIL.Image.new("RGB", (4, 4)), directory_path="intermediates")

    intermediates = tmp_path / "staticfiles" / "intermediates"
    assert sorted(path.name for path in intermediates.iterdir()) == ["image.png", "image_1.png"]
    assert cleaned == [intermediates, intermediates]
