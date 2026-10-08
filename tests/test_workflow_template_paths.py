"""Keep machine-specific absolute paths out of shipped templates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.workflows.workflow_configs import get_workflow_paths

# Matches home-directory paths as they appear in template source, including inside pickled
# byte literals, where a Windows separator is written as one or more escaped backslashes.
HOME_PATH_PATTERN = re.compile(r"(?i:[A-Z]:\\+Users\\+)|/home/[^/'\"\\]+/|/Users/[^/'\"\\]+/")


@pytest.mark.parametrize(
    "template_path", sorted(Path(path) for path in get_workflow_paths()), ids=lambda path: path.name
)
def test_workflow_templates_contain_no_home_directory_paths(template_path: Path) -> None:
    source = template_path.read_text(encoding="utf-8")
    match_count = len(HOME_PATH_PATTERN.findall(source))
    assert match_count == 0, (
        f"{template_path.name}: contains {match_count} home-directory path(s). "
        "Use a {workspace_dir} token or drop the value."
    )
