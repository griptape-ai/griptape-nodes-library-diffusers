"""Publish this library's two dependency lists into the library manifest.

`pip_dependencies` is the edit-time set: what the orchestrator needs to import every node module and
build the node classes. It is `[project] dependencies`, so `uv sync` gives a developer the same slim
orchestrator a real install gets, and a heavy package reaching the orchestrator fails there too
instead of quietly working.

`pip_dependencies_exec` is the execution set, installed into a worker's execution venv. It is the
`exec` extra plus the edit-time set, because a worker imports the node modules as well as running
them.

Both are derived from `pyproject.toml` here, so `make install` cannot leave the manifest inconsistent
with the environment the repo actually builds.

Run from the library root.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tomllib

MANIFEST = pathlib.Path("griptape-nodes-library.json")
PYPROJECT = pathlib.Path("pyproject.toml")


#: Packages that must never appear in the edit-time set. The engine splices that environment onto the
#: orchestrator's `sys.path`, so anything here being reachable there is the defect this guards against.
HEAVY_PREFIXES = (
    "accelerate",
    "av",
    "controlnet-aux",
    "diffusers",
    "ftfy",
    "gguf",
    "matplotlib",
    "opencv-python",
    "openexr",
    "optimum",
    "peft",
    "safetensors",
    "sam2",
    "scipy",
    "spandrel",
    "supervision",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "ultralytics",
)


def main() -> int:
    pyproject = tomllib.loads(PYPROJECT.read_text())
    project = pyproject["project"]

    # The engine is the host, not a dependency the library installs for itself.
    edit_deps = [d for d in project["dependencies"] if not d.startswith("griptape-nodes")]
    heavy_deps = [d for d in project["optional-dependencies"]["exec"] if not d.startswith("griptape-nodes")]

    # A worker imports the node modules as well as running them, so it needs both sets.
    exec_deps = edit_deps + [d for d in heavy_deps if d not in edit_deps]

    smuggled = [d for d in edit_deps if d.split("[")[0].split(">")[0].split("=")[0].strip() in HEAVY_PREFIXES]
    if smuggled:
        print(
            "Attempted to sync dependencies. Failed because these belong to the execution set but are "
            f"declared as edit-time dependencies, which puts them on the orchestrator: {smuggled}. Move "
            "them to the `exec` extra."
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
