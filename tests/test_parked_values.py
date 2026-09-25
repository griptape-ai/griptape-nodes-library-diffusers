"""A value that can be sent must not be held in the process that produced it.

`serializable=False` tells the engine to keep the value where it was built and send a reference in its
place. That is right for a latent or a loaded pipeline, which cannot leave their process. It is wrong
for anything that can travel, because the reference is what everyone else then receives -- including the
editor, which has no way to resolve it.

That is not a theoretical failure. `output_image` on the VAE decoder was marked `serializable=False`, so
a successful run left the node blank and the browser was handed:

    data:image/jpeg;base64,{"kind":"local_object_reference","worker":"663f...","key":"...output_image#031a4d68"}

Nothing raised, in either process. The only symptom was an empty image, which is why this is a test
rather than something to notice.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path("modular_diffusion_nodes_library")

#: Types whose value is a URL or a plain scalar: cheap to send, and meaningless as a reference.
SENDABLE_SUFFIXES = ("UrlArtifact",)
SENDABLE_TYPES = frozenset({"str", "int", "float", "bool", "list", "dict", "json"})

PARAMETER_FACTORIES = frozenset({"Parameter", "ParameterList"})


def _is_sendable(type_name: str) -> bool:
    return type_name.endswith(SENDABLE_SUFFIXES) or type_name.lower() in SENDABLE_TYPES


def _parked_parameters() -> list[tuple[str, int, str, str]]:
    """Every `serializable=False` parameter, as (file, line, name, declared type)."""
    found: list[tuple[str, int, str, str]] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text()
        if "serializable=False" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            factory = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if factory not in PARAMETER_FACTORIES:
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords}
            serializable = keywords.get("serializable")
            if not (isinstance(serializable, ast.Constant) and serializable.value is False):
                continue
            declared = keywords.get("output_type") or keywords.get("type")
            name = keywords.get("name")
            found.append(
                (
                    str(path.relative_to(PACKAGE.parent)),
                    node.lineno,
                    name.value if isinstance(name, ast.Constant) else "<computed>",
                    declared.value if isinstance(declared, ast.Constant) else "<computed>",
                )
            )
    return found


def test_nothing_sendable_is_parked() -> None:
    parked = _parked_parameters()
    assert parked, "Found no parked parameters at all, so this test is no longer checking anything."

    offenders = [entry for entry in parked if _is_sendable(entry[3])]

    assert not offenders, (
        "These parameters hold a value that could simply be sent, so every reader -- including the "
        "editor -- receives a reference it cannot resolve:\n"
        + "\n".join(f"  {file}:{line} {name} ({declared})" for file, line, name, declared in offenders)
        + "\nDrop `serializable=False`. Keep it only for a value that cannot leave its process."
    )


class TestAComponentDescriptionTravels:
    """A component artifact says where to load a component from; it never holds the loaded component.

    Every field is a string, a `StrEnum`, or a nested dataclass of strings, and `materialize` does the
    loading in whichever process needs it. Held instead, the pipeline builder was handed a reference it
    could not read, and `validate_before_node_run` raised on `component_transformer`.
    """

    def test_the_artifact_arrives_as_itself(self) -> None:
        from griptape_nodes.exe_types.local_objects import cache_outputs_for_egress, is_reference

        from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
            ComponentSourceType,
            HFRepoRef,
            ModelComponentArtifact,
        )
        from modular_diffusion_nodes_library.nodes.load_component_node import LoadComponent

        node = LoadComponent(name="Load Component", metadata={"library": "Modular Diffusion"})
        artifact = ModelComponentArtifact(
            load_id="transformer@black-forest-labs/FLUX.1-dev",
            source_type=ComponentSourceType.HF_REPO,
            component="transformer",
            repo_ref=HFRepoRef(repo_id="black-forest-labs/FLUX.1-dev"),
        )
        node.parameter_output_values["component_output"] = artifact

        egressed = cache_outputs_for_egress(node.parameter_output_values, node=node)["component_output"]

        assert not is_reference(egressed), (
            "The component description was held in the producing process, so the pipeline builder "
            "receives a reference and its validation raises instead of reading the artifact."
        )
        assert egressed == artifact


class TestTheDecoderOutputActuallyTravels:
    """The static rule above, checked through the engine's real egress path.

    `cache_outputs_for_egress` is what a worker runs on its outputs before they leave, so driving it
    directly reproduces what the editor receives without loading a model. With `serializable=False` in
    place this returned the `local_object_reference` envelope that the browser pasted into a data URI.
    """

    def test_an_image_output_arrives_as_an_image(self) -> None:
        from griptape.artifacts import ImageUrlArtifact
        from griptape_nodes.exe_types.local_objects import cache_outputs_for_egress, is_reference

        from modular_diffusion_nodes_library.nodes.vae_decoder import VaeDecodeNode

        node = VaeDecodeNode(name="Decode Media", metadata={"library": "Modular Diffusion"})
        url = "http://localhost:8124/static/decoded.png"
        node.parameter_output_values["output_image"] = ImageUrlArtifact(value=url)

        egressed = cache_outputs_for_egress(node.parameter_output_values, node=node)["output_image"]

        assert not is_reference(egressed), (
            "The decoder's image output was held in the producing process, so the editor receives a "
            "reference it cannot resolve and the node renders blank."
        )
        assert isinstance(egressed, ImageUrlArtifact)
        assert egressed.value == url
