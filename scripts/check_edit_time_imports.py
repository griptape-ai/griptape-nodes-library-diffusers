"""Report the heavy packages this library pulls in at edit time.

The orchestrator imports every node module to build the real node classes, so whatever those
imports reach transitively has to be installed in the edit-time environment. Heavy packages
reaching it is what forces `torch` and friends into `pip_dependencies`, which costs a second
multi-gigabyte install and puts every library's copy of a shared package on one `sys.path`.

This imports the node modules and inspects `sys.modules` rather than walking the AST. A static
walk cannot see an import-time function call that loads a pipeline class, and reports a module
imported as `from package import submodule` as a reference to `package` alone -- both produced
false greens while three node modules were in fact unimportable without diffusers.

Run from the library root, in the full environment. Exits non-zero if any heavy package is reached.
"""

from __future__ import annotations

import importlib
import pathlib
import sys

PACKAGE = "modular_diffusion_nodes_library"

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
            failed.append(f"{stem}: {type(exc).__name__}: {exc}")

    reached = {name.split(".")[0] for name in sys.modules} - baseline
    reachable_heavy = sorted(reached & HEAVY)
    stdlib = set(sys.stdlib_module_names)
    third_party = sorted(n for n in reached - {PACKAGE} if not n.startswith("_") and n not in stdlib)

    print(f"node modules imported:   {len(seeds) - len(failed)}/{len(seeds)}")
    print(f"heavy packages reached:  {reachable_heavy or 'none'}")
    print(f"third-party reached:     {len(third_party)}")

    if failed:
        print("\nThese node modules did not import:")
        for entry in failed:
            print(f"  {entry}")
    if reachable_heavy:
        print("\nEach of these must be installed in the edit-time environment. Move the import into")
        print("the function that uses it, or into a `TYPE_CHECKING` block if it is only an annotation.")
    if not failed and not reachable_heavy:
        print("\nEdit-time third-party packages (all must be provided by the engine):")
        for name in third_party:
            print(f"  {name}")

    if failed or reachable_heavy:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
