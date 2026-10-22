"""No node's orchestrator-side validation may read an input held in another process.

The dynamic counterpart to `test_orchestrator_validation_reads.py`. That one reads the source; this one
runs the hook. Both exist because each sees what the other cannot:

* The static scan needs a parameter name it can resolve -- a string literal, or a literal the loop over
  it is built from. It cannot see `get_parameter_value(self._param_name)`, and it cannot see a read one
  call deep in a helper.
* This test sees any read, however it is spelled, but only for a node it can build and a parameter it
  can recognise as holding.

The conditioning bug is what motivated it. `MediaGenConditioningRuntimeParameter` read its held payload
on the orchestrator both directly and once per slot through `FixedSizeParameterList.get_values`, seven
model families delegated to it, and neither the static scan nor an end-to-end run of Text2Image could
see any of it.

No model is loaded. Every bug this class has produced was in the plumbing -- which process reads what --
and plumbing answers to a forged reference exactly as it answers to a real one.

What it still cannot reach: a parameter that does not exist on a freshly built node. The conditioning
parameters are added once a pipeline is chosen, so this test cannot forge into them, and a node whose
held inputs are all dynamic is reported as skipped rather than covered. Read a skip as "nothing to
forge here", never as "this node is clean".
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "griptape-nodes-library.json"
PACKAGE = "modular_diffusion_nodes_library"

#: A worker id this process cannot be. `LocalObjectScope.look_up` compares the envelope's worker against
#: its own engine id and returns ELSEWHERE when they differ, which is the orchestrator's situation.
FOREIGN_WORKER = "a-worker-that-is-not-this-process"

#: Substrings of the two refusals `resolve_if_held` raises. Matching the message rather than the type,
#: because an unconfigured node raises plenty of legitimate RuntimeErrors of its own.
HELD_REFUSALS = (
    "it is held in another process",
    "it is no longer available",
)


def _declared_nodes() -> list[tuple[str, str]]:
    """Every (class_name, module_stem) the manifest registers."""
    nodes = json.loads(MANIFEST.read_text())["nodes"]
    return [(node["class_name"], pathlib.Path(node["file_path"]).stem) for node in nodes]


def _build(class_name: str, stem: str) -> BaseNode:
    import importlib

    module = importlib.import_module(f"{PACKAGE}.nodes.{stem}")
    return getattr(module, class_name)(name=class_name, metadata={"library": "Modular Diffusion"})


def _data_parameters(node: BaseNode) -> list[Any]:
    """The node's data parameters. `node.parameters` also carries control-flow elements, which
    describe execution order rather than a value and do not declare a type.
    """
    from griptape_nodes.exe_types.elements.control_parameters import ControlParameter

    return [
        parameter
        for parameter in node.parameters
        if not isinstance(parameter, ControlParameter) and hasattr(parameter, "allowed_modes")
    ]


def _declared_types(parameter: Any) -> set[str]:
    declared = {getattr(parameter, "type", None), getattr(parameter, "output_type", None)}
    declared.update(getattr(parameter, "input_types", None) or [])
    return {name for name in declared if isinstance(name, str)}


def _held_type_names() -> set[str]:
    """Types this library holds, learned by asking every node what it parks.

    Built from live parameters rather than the source, so a parameter added during `__init__` by a
    helper counts the same as one written out in the node.
    """
    held: set[str] = set()
    for class_name, stem in _declared_nodes():
        node = _build(class_name, stem)
        for parameter in _data_parameters(node):
            if getattr(parameter, "serializable", True) is False:
                held.update(_declared_types(parameter))
    return held


@pytest.mark.parametrize(("class_name", "stem"), _declared_nodes(), ids=lambda value: value)
def test_validation_does_not_read_a_held_input(class_name: str, stem: str) -> None:
    from griptape_nodes.exe_types.local_objects import make_reference

    held_types = _held_type_names()
    assert held_types, "Found no held types at all, so this test is no longer checking anything."

    node = _build(class_name, stem)

    from griptape_nodes.exe_types.core_types import ParameterMode

    forged: list[str] = []
    for parameter in _data_parameters(node):
        if ParameterMode.INPUT not in parameter.allowed_modes:
            continue
        if not (_declared_types(parameter) & held_types):
            continue
        node.parameter_values[parameter.name] = make_reference(
            worker=FOREIGN_WORKER,
            key=f"{FOREIGN_WORKER}:{node.name}.{parameter.name}",
            source=node.name,
        )
        forged.append(parameter.name)

    if not forged:
        pytest.skip(f"{class_name} takes no input that this library holds")

    # Anything else this hook raises is the node objecting to being unconfigured, which is not the
    # question. Only a refusal to hand over a held value is.
    try:
        node.validate_before_node_run()
    except RuntimeError as err:
        if any(refusal in str(err) for refusal in HELD_REFUSALS):
            pytest.fail(
                f"{class_name}.validate_before_node_run reads a held input on the orchestrator, so it "
                f"raises instead of validating. Held inputs on this node: {', '.join(forged)}.\n"
                f"Move the check to `validate_in_execution_environment`.\n{err}"
            )
    except Exception:  # noqa: BLE001, S110 - an unconfigured node objecting is not what this asks about
        pass
