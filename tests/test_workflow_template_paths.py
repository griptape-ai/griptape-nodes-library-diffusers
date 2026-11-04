"""Keep workflow templates portable, repo-local, and valid on disk."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.workflows.workflow_configs import get_workflow_paths

LIBRARY_ROOT = Path(__file__).parents[1]

# Matches home-directory paths as they appear in template source, including inside pickled
# byte literals, where Windows separators may be escaped or embedded in binary blobs.
HOME_PATH_PATTERN = re.compile(r"(?i:[A-Z]:\\+Users\\+)|/home/[^/'\"\\]+/|/Users/[^/'\"\\]+/")
RUNTIME_INPUTS_PATH_PATTERN = re.compile(r"\{inputs\}/[A-Za-z0-9_./:-]+")
# `{outputs}` is deliberately excluded: it is where a template writes results, not something it reads.
MACRO_PATH_PATTERN = re.compile(r"\{(?:workspace|project|workflow)_dir\}/[A-Za-z0-9_./:-]+")
# `{workspace_dir}` is the outer GriptapeNodes workspace, so repo assets live under libraries/<repo>.
ASSETS_PREFIXES = (
    f"{{workspace_dir}}/libraries/{LIBRARY_ROOT.name}/workflows/assets/",
    "{workflow_dir}/assets/",
    "{project_dir}/../workflows/assets/",
)


def _is_repo_asset_path(path: str) -> bool:
    for prefix in ASSETS_PREFIXES:
        if path.startswith(prefix):
            return ".." not in path.removeprefix(prefix)
    return False


@pytest.mark.parametrize(
    "template_path", sorted(Path(path) for path in get_workflow_paths()), ids=lambda path: path.name
)
def test_workflow_templates_contain_no_home_directory_paths(template_path: Path) -> None:
    source = template_path.read_text(encoding="utf-8", errors="ignore")
    match_count = len(HOME_PATH_PATTERN.findall(source))
    assert match_count == 0, (
        f"{template_path.name}: contains {match_count} home-directory path(s). "
        "Use a {workspace_dir} token or drop the value."
    )


@pytest.mark.parametrize(
    "template_path", sorted(Path(path) for path in get_workflow_paths()), ids=lambda path: path.name
)
def test_workflow_templates_only_reference_repo_assets(template_path: Path) -> None:
    source = template_path.read_text(encoding="utf-8", errors="ignore")

    bad_inputs_paths = sorted({match.group(0) for match in RUNTIME_INPUTS_PATH_PATTERN.finditer(source)})
    assert not bad_inputs_paths, (
        f"{template_path.name}: refers to runtime inputs paths instead of repo-local assets: {bad_inputs_paths}"
    )

    # Existence is not checked: LoRA and model files under assets/ may be downloaded after checkout.
    macro_paths = sorted({match.group(0) for match in MACRO_PATH_PATTERN.finditer(source)})
    bad_paths = [path for path in macro_paths if not _is_repo_asset_path(path)]
    assert not bad_paths, f"{template_path.name}: paths must be under one of {ASSETS_PREFIXES}: {bad_paths}"
