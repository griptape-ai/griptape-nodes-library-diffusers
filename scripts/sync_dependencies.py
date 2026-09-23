"""Publish this library's two dependency lists into the library manifest.

`pip_dependencies` is the edit-time set: what the orchestrator needs to import every node module
and build the node classes. `pip_dependencies_exec` is the execution set, installed into a worker's
execution venv. Both are derived from `pyproject.toml` here, so `make install` cannot leave the
manifest inconsistent with the environment the repo actually builds.

Run from the library root.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tomllib

MANIFEST = pathlib.Path("griptape-nodes-library.json")
PYPROJECT = pathlib.Path("pyproject.toml")


def main() -> int:
    pyproject = tomllib.loads(PYPROJECT.read_text())

    # The engine is the host, not a dependency the library installs for itself.
    exec_deps = [d for d in pyproject["project"]["dependencies"] if not d.startswith("griptape-nodes")]
    edit_deps = list(pyproject["tool"]["griptape-nodes"]["edit-time-dependencies"])

    # A worker imports the node modules too, so whatever edit time needs must also be installed
    # in the execution environment.
    missing = [d for d in edit_deps if d not in exec_deps]
    if missing:
        print(
            "Attempted to sync dependencies. Failed because these edit-time entries are absent from "
            f"[project] dependencies, so they would never reach a worker: {missing}."
        )
        return 1

    manifest = json.loads(MANIFEST.read_text())
    dependencies = manifest["metadata"].setdefault("dependencies", {})
    dependencies["pip_dependencies"] = edit_deps
    dependencies["pip_dependencies_exec"] = exec_deps
    MANIFEST.write_text(json.dumps(manifest, indent=4) + "\n")

    print(f"Synced {len(edit_deps)} edit-time and {len(exec_deps)} execution dependencies to {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
