"""HuggingFace repo parameters whose dropdowns also accept a wired connection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import override

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_model_parameter import NO_MODELS_PLACEHOLDER
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_variant_parameter import (
    HuggingFaceRepoVariantParameter,
)
from griptape_nodes.exe_types.param_components.huggingface.huggingface_utils import (
    list_all_repo_revisions_in_cache,
    list_repo_revisions_in_cache,
)
from huggingface_hub.constants import HF_HUB_CACHE


class ConnectableHuggingFaceRepoParameter(HuggingFaceRepoParameter):
    """Same dropdown/gating behavior as HuggingFaceRepoParameter, plus ParameterMode.INPUT.

    The engine's HuggingFaceModelParameter.add_input_parameters() hardcodes the dropdown
    to allowed_modes={PROPERTY}. Widening it here, immediately after creation, lets a wired
    string override the picker selection without touching the engine's dropdown/gating logic.
    """

    @override
    def add_input_parameters(self) -> None:
        super().add_input_parameters()
        parameter = self._node.get_parameter_by_name(self._parameter_name)
        if parameter is not None:
            parameter.allowed_modes = parameter.allowed_modes | {ParameterMode.INPUT}


class ConnectableHuggingFaceRepoVariantParameter(HuggingFaceRepoVariantParameter):
    """One dropdown offering a repo's full-precision default plus variants that each live in a folder.

    The engine's HuggingFaceRepoVariantParameter assumes every variant is a directory at the
    snapshot root and offers only variants, so a repo whose root weights are the default has no
    way to select them. This subclass differs in four ways:

    - The bare ``owner/repo`` key is the default (no variant). The engine already parses a
      two-segment key as "repo, empty variant".
    - Variants are bare names (``fp8r``) mapped by ``variant_folders`` to a folder inside the
      snapshot (``variants/fp8r``), so the ``owner/repo/variant`` key stays three segments and
      each variant can sit in a different folder.
    - A variant is offered only for a revision whose snapshot contains
      ``<folder>/<weights_filename>``.
    - The dropdown also accepts a wired string, and ``list_all_models`` surfaces other cached repos
      as default-only rows, matching ConnectableHuggingFaceRepoParameter.
    """

    def __init__(  # noqa: PLR0913
        self,
        node: BaseNode,
        repo_id: str,
        variant_folders: Mapping[str, str],
        parameter_name: str = "model",
        *,
        weights_filename: str,
        list_all_models: bool = False,
        gated: bool | None = None,
    ):
        # Set before super().__init__: it queries license policy, which consults
        # offers_only_declared_repos(), and then builds the dropdown from fetch_repo_revisions().
        self._variant_folders = dict(variant_folders)
        self._weights_filename = weights_filename
        self._list_all_models = list_all_models
        super().__init__(node, repo_id, list(self._variant_folders), parameter_name, gated=gated)

    @override
    def add_input_parameters(self) -> None:
        super().add_input_parameters()
        parameter = self._node.get_parameter_by_name(self._parameter_name)
        if parameter is not None:
            parameter.allowed_modes = parameter.allowed_modes | {ParameterMode.INPUT}

    @override
    def offers_only_declared_repos(self) -> bool:
        return not self._list_all_models

    def _declared_keys(self) -> list[str]:
        variant_keys = [self._repo_variant_to_key(self._repo_id, variant) for variant in self._variants]
        return [self._repo_id, *variant_keys]

    def _snapshot_path(self, revision: str) -> Path:
        repo_folder = f"models--{self._repo_id.replace('/', '--')}"
        return Path(HF_HUB_CACHE) / repo_folder / "snapshots" / revision

    def variant_weights_path(self, variant: str, revision: str) -> Path:
        """Return the weights file for ``variant`` inside the cached snapshot at ``revision``."""
        return self._snapshot_path(revision) / self._variant_folders[variant] / self._weights_filename

    @override
    def fetch_repo_revisions(self) -> list[tuple[str, str]]:
        revisions: list[tuple[str, str]] = []
        for _, revision in list_repo_revisions_in_cache(self._repo_id):
            revisions.append((self._repo_id, revision))
            for variant in self._variant_folders:
                if self.variant_weights_path(variant, revision).is_file():
                    revisions.append((self._repo_variant_to_key(self._repo_id, variant), revision))

        if self._list_all_models:
            for repo_id, revision in list_all_repo_revisions_in_cache():
                if repo_id != self._repo_id:
                    revisions.append((repo_id, revision))
        return revisions

    @override
    def get_not_downloaded_choices(self) -> list[str]:
        downloaded_keys = {key for key, _ in self.list_repo_revisions()}
        return [key for key in self._declared_keys() if key not in downloaded_keys]

    @override
    def repo_id_for_choice(self, choice: str) -> str | None:
        """Reduce a choice to its repo id, dropping any ``(revision)`` suffix and variant."""
        if not choice or choice == NO_MODELS_PLACEHOLDER:
            return None
        key, _ = self._key_to_repo_revision(choice)
        repo_id, _ = self._key_to_repo_variant(key)
        return repo_id

    @override
    def _get_model_search_term(self, choice: str) -> str:
        repo_id = self.repo_id_for_choice(choice)
        if repo_id is None:
            return choice
        return repo_id
