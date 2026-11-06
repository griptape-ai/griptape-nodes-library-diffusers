from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters import connectable_huggingface_repo_parameter as module
from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoVariantParameter,
)

REPO_ID = "black-forest-labs/flux-3-image-klein"
REVISION = "a" * 40
VARIANT_FOLDERS = {"fp8r": "variants/fp8r", "nvfp4": "elsewhere/nvfp4-export"}


def _make_parameter(
    monkeypatch: pytest.MonkeyPatch,
    cache: Path,
    *,
    cached_variants: list[str],
    list_all_models: bool = False,
    value: str | None = None,
) -> ConnectableHuggingFaceRepoVariantParameter:
    snapshot = cache / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for variant in cached_variants:
        weights = snapshot / VARIANT_FOLDERS[variant] / "model.safetensors"
        weights.parent.mkdir(parents=True)
        weights.touch()

    monkeypatch.setattr(module, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(module, "list_repo_revisions_in_cache", lambda repo_id: [(repo_id, REVISION)])
    monkeypatch.setattr(
        module, "list_all_repo_revisions_in_cache", lambda: [(REPO_ID, REVISION), ("someone/finetune", "b" * 40)]
    )

    parameter = ConnectableHuggingFaceRepoVariantParameter.__new__(ConnectableHuggingFaceRepoVariantParameter)
    parameter._repo_id = REPO_ID
    parameter._variant_folders = dict(VARIANT_FOLDERS)
    parameter._variants = list(VARIANT_FOLDERS)
    parameter._weights_filename = "model.safetensors"
    parameter._list_all_models = list_all_models
    parameter._parameter_name = "model"
    parameter._node = cast(BaseNode, SimpleNamespace(get_parameter_value=lambda name: value))
    parameter._repo_revisions = parameter.fetch_repo_revisions()
    return parameter


def test_default_and_cached_variants_are_offered_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=["fp8r", "nvfp4"])

    assert parameter.fetch_repo_revisions() == [
        (REPO_ID, REVISION),
        (f"{REPO_ID}/fp8r", REVISION),
        (f"{REPO_ID}/nvfp4", REVISION),
    ]
    assert parameter.get_not_downloaded_choices() == []


def test_variant_missing_from_snapshot_is_not_downloaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=["fp8r"])

    assert [key for key, _ in parameter.fetch_repo_revisions()] == [REPO_ID, f"{REPO_ID}/fp8r"]
    assert parameter.get_not_downloaded_choices() == [f"{REPO_ID}/nvfp4"]


def test_variant_folder_without_weights_file_is_not_offered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=[])
    (tmp_path / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION / "variants" / "fp8r").mkdir(
        parents=True
    )

    assert [key for key, _ in parameter.fetch_repo_revisions()] == [REPO_ID]


def test_each_variant_resolves_to_its_own_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=["fp8r", "nvfp4"])
    snapshot = tmp_path / f"models--{REPO_ID.replace('/', '--')}" / "snapshots" / REVISION

    assert parameter.variant_weights_path("fp8r", REVISION) == snapshot / "variants/fp8r/model.safetensors"
    assert parameter.variant_weights_path("nvfp4", REVISION) == snapshot / "elsewhere/nvfp4-export/model.safetensors"


@pytest.mark.parametrize(
    ("value", "expected_variant"),
    [(REPO_ID, ""), (f"{REPO_ID}/fp8r", "fp8r"), (f"{REPO_ID}/nvfp4", "nvfp4")],
    ids=["default", "fp8r", "nvfp4"],
)
def test_selection_resolves_to_repo_variant_and_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str, expected_variant: str
) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=["fp8r", "nvfp4"], value=value)

    assert parameter.get_repo_variant_revision() == (REPO_ID, expected_variant, REVISION)


@pytest.mark.parametrize(
    "choice",
    [REPO_ID, f"{REPO_ID}/fp8r", f"{REPO_ID} ({REVISION})", f"{REPO_ID}/nvfp4 ({REVISION})"],
    ids=["default", "variant", "default-with-revision", "variant-with-revision"],
)
def test_every_choice_reduces_to_the_catalog_repo_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, choice: str
) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=[])

    assert parameter.repo_id_for_choice(choice) == REPO_ID
    assert parameter._get_model_search_term(choice) == REPO_ID


def test_list_all_models_adds_other_repos_as_default_only_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    parameter = _make_parameter(monkeypatch, tmp_path, cached_variants=["fp8r"], list_all_models=True)

    assert [key for key, _ in parameter.fetch_repo_revisions()] == [REPO_ID, f"{REPO_ID}/fp8r", "someone/finetune"]
    assert parameter.offers_only_declared_repos() is False
    assert parameter.repo_id_for_choice("someone/finetune") == "someone/finetune"


def test_dropdown_accepts_a_wired_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_variant_parameter import (
        HuggingFaceRepoVariantParameter,
    )

    dropdown = SimpleNamespace(allowed_modes={ParameterMode.PROPERTY})
    node = SimpleNamespace(get_parameter_by_name=lambda name: dropdown)
    monkeypatch.setattr(HuggingFaceRepoVariantParameter, "add_input_parameters", lambda self: None)
    parameter = ConnectableHuggingFaceRepoVariantParameter.__new__(ConnectableHuggingFaceRepoVariantParameter)
    parameter._node = cast(BaseNode, node)
    parameter._parameter_name = "model"

    parameter.add_input_parameters()

    assert dropdown.allowed_modes == {ParameterMode.PROPERTY, ParameterMode.INPUT}
