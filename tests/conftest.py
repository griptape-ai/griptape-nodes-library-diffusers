"""Shared fixtures for the library's tests.

Any test that touches `node.local_objects` resolves the ambient engine, and building an engine reads
and writes the user's config file. Isolating that file is not optional: an un-sandboxed engine build
has pinned a real `workspace_directory` to a throwaway temp path before.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def isolate_user_config() -> Generator[Path, None, None]:
    """Point the engine's user config at a temp file for the duration of each test."""
    import griptape_nodes.retained_mode.managers.config_manager as config_manager_module
    from griptape_nodes.retained_mode.engine import reset_root_engine

    # Drop the root engine so managers re-initialize against the patched config below.
    reset_root_engine()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config_path = Path(temp_dir) / "griptape_nodes_config.json"
        temp_config_path.write_text(json.dumps({}, indent=2))

        with patch.object(config_manager_module, "USER_CONFIG_PATH", temp_config_path):
            yield temp_config_path

            # Drop it again so the next test does not inherit this one's object graph.
            reset_root_engine()
