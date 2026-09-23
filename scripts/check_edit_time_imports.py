"""Report the heavy packages this library pulls in at edit time.

The orchestrator imports every node module and constructs every node class to show a node in the
editor, so whatever that reaches has to be installed in the edit-time environment. Heavy packages
reaching it is what forces `torch` and friends into `pip_dependencies`, which costs a second
multi-gigabyte install and puts every library's copy of a shared package on one `sys.path`.

Two things this checks, because each caught a real failure the other could not:

- **Imports.** Measured against `sys.modules` rather than by walking the AST. A static walk reports
  `from package import submodule` as a reference to `package` alone and cannot see an import-time
  call that loads a pipeline class, so it reported a clean closure while three node modules were
  unimportable without diffusers.
- **Construction.** Importing a module runs nothing that a node's `__init__` runs. Building the node
  classes reached diffusers through `get_component_slots()`, which read the pipeline's `__init__`
  signature -- 1730 heavy imports that an import-only check is structurally blind to.

Run from the library root, in the full environment: heavy packages must be installed, or an
accidental import fails loudly instead of being recorded here.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import sys

PACKAGE = "modular_diffusion_nodes_library"
MANIFEST = pathlib.Path("griptape-nodes-library.json")

# Packages that belong in the execution environment only.
HEAVY = frozenset(
    {
        "accelerate",
        "av",
        "controlnet_aux",
        "cv2",
        "diffusers",
        "ftfy",
        "gguf",
        "Imath",
        "matplotlib",
        "OpenEXR",
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
    }
)


def main() -> int:
    root = pathlib.Path(PACKAGE)
    seeds = sorted(path.stem for path in (root / "nodes").glob("*.py") if path.stem != "__init__")
    if not seeds:
        print(f"No node modules found under {root / 'nodes'}; run this from the library root.")
        return 2

    sys.path.insert(0, str(pathlib.Path.cwd()))
    importlib.import_module("griptape_nodes")
    baseline = {name.split(".")[0] for name in sys.modules}

    failed: list[str] = []
    for stem in seeds:
        try:
            importlib.import_module(f"{PACKAGE}.nodes.{stem}")
        except Exception as exc:  # noqa: BLE001 - any failure here is a reportable defect
            failed.append(f"import {stem}: {type(exc).__name__}: {exc}")

    declared = [
        (node["class_name"], pathlib.Path(node["file_path"]).stem) for node in json.loads(MANIFEST.read_text())["nodes"]
    ]
    built = 0
    for class_name, stem in declared:
        module = sys.modules.get(f"{PACKAGE}.nodes.{stem}")
        if module is None:
            continue
        try:
            getattr(module, class_name)(name=f"{class_name} 1", metadata={"library": "check"})
            built += 1
        except Exception as exc:  # noqa: BLE001 - any failure here is a reportable defect
            failed.append(f"construct {class_name}: {type(exc).__name__}: {exc}")

    reached = {name.split(".")[0] for name in sys.modules} - baseline
    reachable_heavy = sorted(reached & HEAVY)
    stdlib = set(sys.stdlib_module_names)
    third_party = sorted(n for n in reached - {PACKAGE} if not n.startswith("_") and n not in stdlib)

    print(f"node modules imported:   {len(seeds) - len([f for f in failed if f.startswith('import ')])}/{len(seeds)}")
    print(f"node classes built:      {built}/{len(declared)}")
    print(f"heavy packages reached:  {reachable_heavy or 'none'}")
    print(f"third-party reached:     {len(third_party)}")

    if failed:
        print("\nThese did not succeed:")
        for entry in failed:
            print(f"  {entry}")
    if reachable_heavy:
        print("\nEach of these must be installed in the edit-time environment. Move the import into")
        print("the function that uses it, or into a `TYPE_CHECKING` block if it is only an annotation.")
        print("If a node's `__init__` reached one by calling into diffusers, declare the answer instead")
        print("-- `_component_slots` is the precedent, with a test pinning it to the pinned diffusers.")
    if not failed and not reachable_heavy:
        print("\nEdit-time third-party packages (all must be provided by the engine):")
        for name in third_party:
            print(f"  {name}")

    if failed or reachable_heavy:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
