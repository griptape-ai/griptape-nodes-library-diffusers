"""Report node code that reaches for an engine manager it cannot have in a worker.

`GriptapeNodes.<Manager>()` raises during node execution in a worker: that process holds its own copy
of the state, so an answer served locally would be silently wrong rather than loudly absent. Node code
is meant to go through `GriptapeNodes.handle_request(...)`, which routes to whichever process owns the
state. `StaticFilesManager` is the one exception, because a worker shares the workspace on disk.

The engine enforces this at runtime, one failure at a time, and only for lines a run happens to reach.
That is a poor way to migrate a library: the first attempt here found the offenders by crashing on them
in sequence. This finds all of them at once.

The guarded list is read from the engine rather than restated, so a manager gaining or losing its guard
upstream is picked up rather than silently diverging.

Run from the library root. Exits non-zero if any guarded access is found outside the allowed places.
"""

from __future__ import annotations

import ast
import pathlib
import sys

PACKAGE = "modular_diffusion_nodes_library"
FACADE = "GriptapeNodes"

#: Functions that run while a node is being built rather than executed. A worker constructs its
#: transient node before opening the scope the guard watches, so a manager read here is legitimate.
CONSTRUCTION_FUNCTIONS = frozenset(
    {
        "__init__",
        "_additional_parameters",
    }
)


def guarded_managers() -> set[str]:
    """Every facade accessor the engine forbids during worker node execution."""
    from griptape_nodes.retained_mode import griptape_nodes as facade_module

    source = pathlib.Path(facade_module.__file__).read_text()
    tree = ast.parse(source)
    facade = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GriptapeNodes")
    return {
        node.name
        for node in facade.body
        if isinstance(node, ast.FunctionDef) and "_forbid_manager_during_worker_execution" in ast.dump(node)
    }


def offences(root: pathlib.Path, guarded: set[str]) -> list[tuple[str, int, str, str]]:
    """Every guarded access, as (file, line, manager, enclosing function)."""
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        if FACADE not in source:
            continue
        tree = ast.parse(source)
        functions = [
            (node.lineno, node.end_lineno or node.lineno, node.name)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            target = node.func.value
            if not (isinstance(target, ast.Name) and target.id == FACADE):
                continue
            if node.func.attr not in guarded:
                continue
            enclosing = [(end - start, name) for start, end, name in functions if start <= node.lineno <= end]
            owner = min(enclosing)[1] if enclosing else "<module>"
            if owner in CONSTRUCTION_FUNCTIONS:
                continue
            found.append((str(path.relative_to(root.parent)), node.lineno, node.func.attr, owner))
    return found


def main() -> int:
    root = pathlib.Path(PACKAGE)
    if not root.is_dir():
        print(f"No {PACKAGE} directory here; run this from the library root.")
        return 2

    guarded = guarded_managers()
    found = offences(root, guarded)

    print(f"guarded managers (from the engine): {len(guarded)}")
    print(f"accesses outside construction:      {len(found)}")

    if found:
        print("\nEach of these raises when the node runs in a worker. Use a request instead:")
        for file, line, manager, owner in found:
            print(f"  {file}:{line}  {manager} in {owner}")
        print(
            "\nThe engine names the replacement in its error message. A manager method that is static and "
            "touches only the filesystem may be imported directly instead, since it holds no state a "
            "worker could disagree about."
        )
        return 1

    print("\nEvery manager access is either a request or happens while the node is built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
