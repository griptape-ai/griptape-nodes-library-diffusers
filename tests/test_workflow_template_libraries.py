"""Keep templates usable with only this library and the optional standard library."""

from __future__ import annotations

from pathlib import Path

import pytest
from griptape_nodes.node_library.workflow_registry import read_workflow_metadata

from tests.workflows.workflow_configs import get_workflow_paths

MODULAR_LIBRARY = "Griptape Modular Diffusion Nodes Library"
STANDARD_LIBRARY = "Griptape Nodes Library"
ALLOWED_LIBRARIES = {MODULAR_LIBRARY, STANDARD_LIBRARY}


@pytest.mark.parametrize(
    "template_path", sorted(Path(path) for path in get_workflow_paths()), ids=lambda path: path.name
)
def test_workflow_templates_only_use_modular_and_standard_libraries(template_path: Path) -> None:
    # Deliberately metadata-only since the engine uses the metadata alone to decide which libraries a workflow needs.
    metadata = read_workflow_metadata(template_path)
    declared_libraries = {library.library_name for library in metadata.node_libraries_referenced}
    assert MODULAR_LIBRARY in declared_libraries, (
        f"{template_path.name}: node_libraries_referenced must include {MODULAR_LIBRARY!r}"
    )
    unexpected = sorted(declared_libraries - ALLOWED_LIBRARIES)
    assert not unexpected, f"{template_path.name}: references disallowed libraries: {unexpected}"
