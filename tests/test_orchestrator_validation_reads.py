"""A validation hook that runs on the orchestrator must not read a held parameter.

`validate_before_node_run` and `validate_before_workflow_run` both run on the orchestrator. A
`serializable=False` value stays in the process that built it and travels as a reference, so
`get_parameter_value` raises when the orchestrator asks for one -- `resolve_if_held` refuses rather
than guessing. Even `is None` raises, because the read happens before the comparison.

Anything that needs a real latent belongs in `validate_in_execution_environment`, which runs where the
node runs.

Two nodes shipped this bug past a green end-to-end run, because the workflow that was exercised did not
contain them:

    elementwise_latent_math.validate_before_node_run  -> read left_latent / right_latent
    latent_composite_mask_node.validate_before_node_run -> read destination_latent / source_latent

The held-type set is derived from the library rather than hardcoded: a type is held if any parameter
declared with it passes `serializable=False`. Declare a new held artifact and this check covers it
without being told.
"""

from __future__ import annotations

import ast
import pathlib
from collections import defaultdict

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "modular_diffusion_nodes_library"

#: Hooks the orchestrator runs. `validate_in_execution_environment` is deliberately absent.
ORCHESTRATOR_HOOKS = frozenset({"validate_before_node_run", "validate_before_workflow_run"})

PARAMETER_FACTORIES = frozenset({"Parameter", "ParameterList"})

#: Keywords that carry a parameter's declared type.
TYPE_KEYWORDS = ("type", "output_type", "input_types")


def _source_files() -> list[pathlib.Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _declared_types(call: ast.Call) -> set[str]:
    """Every type name a `Parameter(...)` call declares, across all three type keywords."""
    found: set[str] = set()
    for keyword in call.keywords:
        if keyword.arg not in TYPE_KEYWORDS:
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            found.add(keyword.value.value)
        elif isinstance(keyword.value, ast.List):
            found.update(
                element.value
                for element in keyword.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            )
    return found


def _parameter_calls(tree: ast.AST) -> list[ast.Call]:
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        factory = getattr(node.func, "id", getattr(node.func, "attr", ""))
        if factory in PARAMETER_FACTORIES:
            calls.append(node)
    return calls


def _parameter_name(call: ast.Call) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            return value if isinstance(value, str) else None
    return None


def _is_held(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "serializable" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is False
    return False


def _survey() -> tuple[set[str], dict[str, set[str]], dict[pathlib.Path, dict[str, set[str]]]]:
    """Held type names, the library-wide name->types map, and the same map per file."""
    held_types: set[str] = set()
    globally: dict[str, set[str]] = defaultdict(set)
    per_file: dict[pathlib.Path, dict[str, set[str]]] = {}

    for path in _source_files():
        tree = ast.parse(path.read_text())
        local: dict[str, set[str]] = defaultdict(set)
        for call in _parameter_calls(tree):
            name = _parameter_name(call)
            types = _declared_types(call)
            if not types:
                continue
            if _is_held(call):
                held_types.update(types)
            if name is not None:
                local[name].update(types)
                globally[name].update(types)
        per_file[path] = local

    return held_types, globally, per_file


def _loop_bound_names(function: ast.FunctionDef) -> dict[str, set[str]]:
    """Loop variables bound to a literal sequence of parameter names.

    `for name in ("left_latent", "right_latent"): ... get_parameter_value(name)` reads two named
    parameters, and a scan that only understands a string literal at the call site sees neither.
    """
    bound: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(function):
        if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
            continue
        if not isinstance(node.iter, (ast.Tuple, ast.List)):
            continue
        bound[node.target.id].update(
            element.value
            for element in node.iter.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )
    return bound


def _reads(function: ast.FunctionDef) -> list[tuple[str, int]]:
    """Every parameter `function` reads by a name resolvable statically, as (name, line)."""
    bound = _loop_bound_names(function)
    reads = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "get_parameter_value":
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            reads.append((first.value, node.lineno))
        elif isinstance(first, ast.Name):
            reads.extend((name, node.lineno) for name in sorted(bound.get(first.id, ())))
    return reads


def test_no_orchestrator_hook_reads_a_held_parameter() -> None:
    held_types, globally, per_file = _survey()
    assert held_types, "Found no held parameters at all, so this test is no longer checking anything."

    offenders: list[str] = []
    for path, local in per_file.items():
        tree = ast.parse(path.read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef) or function.name not in ORCHESTRATOR_HOOKS:
                continue
            for name, line in _reads(function):
                # Prefer the declaration in this file. Fall back to the library-wide map only when the
                # name means exactly one thing everywhere, so a reused name cannot manufacture a hit.
                types = local.get(name) or (globally[name] if len(globally.get(name, ())) == 1 else set())
                if types & held_types:
                    offenders.append(
                        f"  {path.relative_to(PACKAGE.parent)}:{line} {function.name} reads "
                        f"'{name}' ({', '.join(sorted(types & held_types))})"
                    )

    assert not offenders, (
        "These hooks run on the orchestrator and read a value held in another process, so the read "
        "raises instead of validating:\n"
        + "\n".join(sorted(offenders))
        + "\nMove the check to `validate_in_execution_environment`, which runs where the node runs."
    )
