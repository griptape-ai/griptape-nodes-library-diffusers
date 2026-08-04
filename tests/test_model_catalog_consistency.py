"""Keep the `model_catalog` declaration in sync with the repos the library actually serves.

The catalog is the contract this library exposes to the platform: it is what an admin policy
gates on, and the engine rejects the whole library at load time if a node's `model_usage`
references a model the catalog does not declare. The repo ids themselves live in Python, spread
across the `standard_parameters/` pipeline-type classes, the ControlNet parameter types, and the
upsampler parameter types. These tests walk those Python sources and compare them against the
manifest so the two cannot drift.

Provider ids key the model's **licensing authority**, not the HuggingFace org that hosts the
weights. `provider_id` is the only hierarchical handle in the policy language (the app builds
Cedar parent edges as `("ModelProvider", provider_id)`), so a derivative that inherits a base
model's terms must be keyed to the base authority or a provider-scoped rule would miss it.
`test_flux_derivatives_key_to_black_forest_labs` is the guard on that decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.node_library.library_declarations import find_model_catalog, iter_catalog_models
from griptape_nodes.node_library.library_registry import LibrarySchema
from griptape_nodes.node_library.library_validation import (
    detect_retired_node_declarations,
    validate_library_declarations,
)

from modular_diffusion_nodes_library.parameters.controlnet_node_parameter_types import ControlNetNodesParameterType
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import MODULAR_PIPELINE_TYPE_PROVIDER_MAP
from modular_diffusion_nodes_library.parameters.upsampler_parameter_type import BaseUpsamplerParameters

LIBRARY_JSON = Path(__file__).parents[1] / "griptape-nodes-library.json"

# The engine release that carries HuggingFace dropdown gating. Declarations parse from 0.92.0,
# but enforcement needs this, so it is the floor in both the manifest and pyproject.
ENGINE_VERSION_FLOOR = "0.95.1"

# Node class name -> the Python source its repo choices come from. Mirrors the three node types
# that host a HuggingFaceRepoParameter; every other node in the library reaches no model.
BUILDER_NODE = "LatentDiffusionPipelineBuilderNode"
CONTROLNET_NODE = "ControlNetNode"
UPSAMPLER_NODE = "LatentUpsamplerNode"


def _load_library() -> dict[str, Any]:
    return json.loads(LIBRARY_JSON.read_text())


def _schema() -> LibrarySchema:
    return LibrarySchema.model_validate(_load_library())


def _catalog_by_provider_model_id() -> dict[str, tuple[str, str]]:
    """Map each declared provider_model_id to its (provider_id, catalog model_id)."""
    catalog = find_model_catalog(_schema().metadata.declarations)
    assert catalog is not None, "library declares no model_catalog"
    mapping: dict[str, tuple[str, str]] = {}
    for resolved in iter_catalog_models(catalog):
        provider_model_id = resolved.model.provider_model_id
        assert provider_model_id is not None, f"{resolved.model_id} declares no provider_model_id"
        mapping[provider_model_id] = (resolved.provider_id, resolved.model_id)
    return mapping


def _declared_model_ids(class_name: str) -> set[str]:
    for node in _load_library()["nodes"]:
        if node["class_name"] != class_name:
            continue
        for declaration in node["metadata"].get("declarations", []):
            if declaration.get("type") == "model_usage":
                return set(declaration["model_ids"])
        pytest.fail(f"{class_name} carries no model_usage declaration")
    pytest.fail(f"{class_name} is not registered in the manifest")


def _all_subclasses(cls: type) -> set[type]:
    found: set[type] = set()
    for subclass in cls.__subclasses__():
        found.add(subclass)
        found |= _all_subclasses(subclass)
    return found


def _builder_repo_ids() -> set[str]:
    """Every repo id reachable from the pipeline builder, across all providers and pipeline types."""
    repo_ids: set[str] = set()
    for params_cls in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.values():
        for pipeline_params_cls in params_cls.get_pipeline_type_dict().values():
            repo_ids |= _repo_ids_from_source(pipeline_params_cls)
    return repo_ids


def _controlnet_repo_ids() -> set[str]:
    repo_ids: set[str] = set()
    for subclass in _all_subclasses(ControlNetNodesParameterType):
        repo_ids |= _repo_ids_from_source(subclass)
    return repo_ids


def _upsampler_repo_ids() -> set[str]:
    repo_ids: set[str] = set()
    for subclass in _all_subclasses(BaseUpsamplerParameters):
        repo_ids |= _repo_ids_from_source(subclass)
    return repo_ids


def _repo_ids_from_source(cls: type) -> set[str]:
    """Pull the `owner/name` literals out of the module that defines a parameter class.

    The repo lists are built inside `__init__` / `model_repo_ids` / `_model_repo_id` bodies that
    need a live node to call, so this reads the literals rather than instantiating anything. It
    scans the whole defining module, not just the class body, because several providers hold
    their repo lists in module-level constants (e.g. `FLUX_2_REPO_IDS` in flux2_parameters.py)
    that the class then splats in.
    """
    import inspect
    import re

    try:
        source = inspect.getsource(inspect.getmodule(cls))
    except (OSError, TypeError):
        return set()
    # `owner/name`, optionally with the `::subvariant` postfix the LTX-2 params use.
    pattern = r"\"([A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+(?:::[A-Za-z0-9._-]+)?)\""
    found = set()
    for match in re.findall(pattern, source):
        # Skip paths that are clearly not repo ids (file names, URLs already split off).
        if match.endswith((".safetensors", ".bin", ".pt", ".json", ".py")):
            continue
        found.add(match.split("::", 1)[0])
    return found


def test_manifest_passes_engine_validation() -> None:
    """The manifest must survive exactly what the engine runs at library load."""
    library = _load_library()
    assert detect_retired_node_declarations(library) == []
    assert validate_library_declarations(LibrarySchema.model_validate(library)) == []


def test_schema_and_engine_versions_meet_the_declaration_baseline() -> None:
    """Pin the two version floors this library's declarations depend on.

    `library_schema_version` 0.10.0 is what makes the `model_catalog` and `model_usage`
    declarations parseable at all.

    `engine_version` is the stricter of two requirements. Declarations alone only need 0.92.0
    (the epic baseline), but license *enforcement* on the HuggingFace repo dropdowns needs the
    gating in `HuggingFaceModelParameter`, which ships in 0.95.1. Declaring only 0.92.0 would let
    the library load on an engine that validates the catalog but silently offers denied models --
    a fail-open. `IncompatibleEngineVersionCheck` marks the library UNUSABLE on older PyPI
    engines instead, which is the intended fail-closed behavior.
    """
    schema = _schema()
    assert schema.library_schema_version == "0.10.0"
    assert schema.metadata.engine_version == ENGINE_VERSION_FLOOR


def test_manifest_and_pyproject_engine_floors_agree() -> None:
    """The manifest gate and the installed-package floor must not drift apart.

    The manifest's `engine_version` is what the engine checks at load time; the pyproject
    dependency is what actually gets installed. If only one is bumped, the library either loads
    without the code it needs or refuses to load on an engine that would have worked. Note the
    dependency is on `griptape-nodes-engine`, which provides the `griptape_nodes` package --
    not `griptape-nodes`, which provides `griptape_nodes_app`.
    """
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert f'"griptape-nodes-engine>={ENGINE_VERSION_FLOOR}"' in pyproject


def test_every_catalog_model_is_used_by_some_node() -> None:
    """A cataloged model no node references is dead weight the policy UI would still show."""
    catalog_ids = {model_id for _, model_id in _catalog_by_provider_model_id().values()}
    used = (
        _declared_model_ids(BUILDER_NODE) | _declared_model_ids(CONTROLNET_NODE) | _declared_model_ids(UPSAMPLER_NODE)
    )
    assert catalog_ids - used == set()


@pytest.mark.parametrize(
    ("class_name", "repo_ids_fn"),
    [
        (BUILDER_NODE, _builder_repo_ids),
        (CONTROLNET_NODE, _controlnet_repo_ids),
        (UPSAMPLER_NODE, _upsampler_repo_ids),
    ],
)
def test_node_model_usage_matches_its_python_repo_lists(class_name: str, repo_ids_fn: Any) -> None:
    """Each hosting node must declare exactly the repos its Python offers -- no more, no less.

    Missing entries mean an ungateable dropdown row; extra entries mean the catalog advertises a
    model the node cannot actually select.
    """
    catalog = _catalog_by_provider_model_id()
    repo_ids = repo_ids_fn()
    assert repo_ids, f"found no repo ids for {class_name}; the source-scraping heuristic broke"

    uncataloged = sorted(repo for repo in repo_ids if repo not in catalog)
    assert uncataloged == [], f"{class_name} offers repos absent from the catalog: {uncataloged}"

    expected = {catalog[repo][1] for repo in repo_ids}
    assert _declared_model_ids(class_name) == expected


def test_flux_derivatives_key_to_black_forest_labs() -> None:
    """Third-party-hosted FLUX derivatives must key to BFL, not to their host org.

    These four inherit FLUX's non-commercial terms while living under other orgs. If any were
    keyed to its host, `forbid ... resource in ModelProvider::"black_forest_labs"` would silently
    fail to block it -- the exact hole this keying scheme exists to close.
    """
    catalog = _catalog_by_provider_model_id()
    for repo in (
        "InstantX/FLUX.1-dev-Controlnet-Union",
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
        "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "diffusers/FLUX.2-dev-bnb-4bit",
        "fal/FLUX.2-dev-Turbo",
    ):
        provider_id, _ = catalog[repo]
        assert provider_id == "black_forest_labs", f"{repo} is keyed to {provider_id}"


def test_independently_licensed_weights_keep_their_own_authority() -> None:
    """Weights whose license does not derive from a base model must not inherit its authority.

    The counterweight to the test above: over-keying to a base lineage would overstate
    encumbrance on permissively licensed weights.
    """
    catalog = _catalog_by_provider_model_id()
    for repo, expected_provider in (
        ("xinsir/controlnet-canny-sdxl-1.0", "xinsir"),
        ("InstantX/SD3-Controlnet-Canny", "instantx"),
        ("tensorart/SD3.5M-Controlnet-Canny", "tensorart"),
        ("openai/clip-vit-large-patch14", "openai"),
        ("google/t5-v1_1-xxl", "google"),
    ):
        provider_id, _ = catalog[repo]
        assert provider_id == expected_provider, f"{repo} is keyed to {provider_id}"


def test_gated_repos_require_a_customer_key() -> None:
    """Repos behind an HF license gate must declare REQUIRES_CUSTOMER_KEY.

    This is the signal that separates commercially encumbered weights (FLUX.1-dev, SD3) from
    freely usable ones (FLUX.1-schnell, the klein releases), which is why the catalog exists.
    """
    catalog = find_model_catalog(_schema().metadata.declarations)
    assert catalog is not None
    key_support_by_repo = {
        resolved.model.provider_model_id: resolved.model.key_support for resolved in iter_catalog_models(catalog)
    }
    for repo in (
        "black-forest-labs/FLUX.1-dev",
        "black-forest-labs/FLUX.1-Krea-dev",
        "black-forest-labs/FLUX.1-Fill-dev",
        "black-forest-labs/FLUX.1-Kontext-dev",
        "black-forest-labs/FLUX.2-dev",
        "stabilityai/stable-diffusion-3.5-large",
    ):
        assert key_support_by_repo[repo] == "REQUIRES_CUSTOMER_KEY", repo
    for repo in (
        "black-forest-labs/FLUX.1-schnell",
        "black-forest-labs/FLUX.2-klein-9B",
        "Qwen/Qwen-Image",
    ):
        assert key_support_by_repo[repo] == "NO_KEY_REQUIRED", repo
