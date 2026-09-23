"""Report the heavy packages this library pulls in at edit time.

The orchestrator imports every node module to build the real node classes, so whatever those
imports reach transitively has to be installed in the edit-time environment. Heavy packages
reaching it is what forces `torch` and friends into `pip_dependencies`, which costs a second
multi-gigabyte install and puts every library's copy of a shared package on one `sys.path`.

Run from the library root. Exits non-zero if any heavy package is reachable.
"""

from __future__ import annotations

import ast
import pathlib
import sys

PACKAGE = "modular_diffusion_nodes_library"
ROOT = pathlib.Path(PACKAGE)

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
        "huggingface_hub",
        "matplotlib",
        "numpy",
        "openexr",
        "optimum",
        "peft",
        "PIL",
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


def module_to_path(module: str) -> pathlib.Path | None:
    """The file a dotted module name resolves to inside this library, or None."""
    direct = ROOT.parent / (module.replace(".", "/") + ".py")
    if direct.exists():
        return direct
    package_init = ROOT.parent / module.replace(".", "/") / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def imports_executed_at_import_time(path: pathlib.Path) -> set[str]:
    """Modules this file imports when it is imported.

    Only module-scope imports count. An import inside a function body runs when that function
    is called, and a `TYPE_CHECKING` block never runs at all, so neither reaches edit time.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def main() -> int:
    seeds = [path for path in (ROOT / "nodes").glob("*.py") if path.stem != "__init__"]
    if not seeds:
        print(f"No node modules found under {ROOT / 'nodes'}; run this from the library root.")
        return 2

    visited: set[pathlib.Path] = set()
    queue = list(seeds)
    external: set[str] = set()
    while queue:
        path = queue.pop()
        if path in visited:
            continue
        visited.add(path)
        for module in imports_executed_at_import_time(path):
            if module.startswith(PACKAGE):
                resolved = module_to_path(module)
                if resolved is not None and resolved not in visited:
                    queue.append(resolved)
            else:
                external.add(module.split(".")[0])

    reachable_heavy = sorted(external & HEAVY)
    print(f"node modules:            {len(seeds)}")
    print(f"files reached at import:  {len(visited)}")
    print(f"heavy packages reached:   {reachable_heavy or 'none'}")

    if reachable_heavy:
        print("\nEach of these must be installed in the edit-time environment. To find the path that")
        print("reaches one, grep for its module-scope import and follow the importer chain.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
