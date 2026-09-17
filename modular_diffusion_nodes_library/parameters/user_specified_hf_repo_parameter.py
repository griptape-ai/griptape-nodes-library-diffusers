from __future__ import annotations

from typing import ClassVar, override

from griptape_nodes.exe_types.core_types import BadgeData, NodeMessageResult, ParameterMode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_model_parameter import HuggingFaceModelParameter
from griptape_nodes.exe_types.param_components.huggingface.huggingface_utils import list_all_repo_revisions_in_cache
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.parameter_events import AlterParameterDetailsRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload, OnClickMessageResultPayload
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.component_loading.config_resolver import HF_REPO_ID_PATTERN

_INFO_BADGE = BadgeData(
    variant="info",
    title="Repo ID",
    message=(
        "The repo must already be in your local HuggingFace cache — no downloads are triggered from here.\n\n"
        "- Click the refresh (↺) button next to the field to re-check the local cache.\n"
        "- If the repo is not cached, an **Open Model Manager to Download** button will appear below the "
        "field — click it to jump to Model Management pre-filtered to this repo.\n"
        "- Run-time validation fails if the repo is not in the local cache when the node is executed."
    ),
    hide_clear_button=True,
)


class UserSpecifiedHuggingFaceRepoParameter(HuggingFaceModelParameter):
    """User-specified repo ID variant of HuggingFaceModelParameter for the LoadComponent node.

    Unlike HuggingFaceRepoParameter (which presents a dropdown of cached repos),
    this creates a plain text input that accepts any repo ID.
    """

    # Shared across every instance: the local HF cache is process-global, so one scan
    # serves all repo fields. The stored list is an immutable snapshot; callers only read it.
    _cached_repo_choices: ClassVar[list[tuple[str, str]] | None] = None

    @property
    def _download_param_name(self) -> str:
        return f"{self._parameter_name}_download"

    @classmethod
    def _get_cached_repo_choices(cls) -> list[tuple[str, str]]:
        if cls._cached_repo_choices is None:
            cls._cached_repo_choices = list_all_repo_revisions_in_cache()
        return cls._cached_repo_choices

    @classmethod
    def _invalidate_cached_repo_choices(cls) -> None:
        cls._cached_repo_choices = None

    @override
    def fetch_repo_revisions(self) -> list[tuple[str, str]]:
        repo_id = self._node.get_parameter_value(self._parameter_name)
        return self._fetch_repo_revisions(repo_id)

    def _fetch_repo_revisions(self, repo_id: str | None) -> list[tuple[str, str]]:
        if not repo_id or not HF_REPO_ID_PATTERN.fullmatch(str(repo_id)):
            return []
        cached_revisions = self._get_cached_repo_choices()
        return [rev for rev in cached_revisions if rev[0] == str(repo_id)]

    @override
    def get_download_models(self) -> list[str]:
        repo_id = self._node.get_parameter_value(self._parameter_name)
        if not repo_id or not HF_REPO_ID_PATTERN.fullmatch(str(repo_id)):
            return []
        return [str(repo_id)]

    @override
    def get_download_commands(self) -> list[str]:
        return []

    def _on_value_changed(self, value: object) -> object:
        repo_id = ""
        if value is not None:
            repo_id = str(value)
        self._refresh_parameters(repo_id)
        return value

    def _push_repo_choices_to_ui(self) -> None:
        """Push the full cached repo list to the frontend as autocomplete hints.

        ``allow_custom=True`` applies filters client-side, so the backend must supply the whole list.
        The update only emits when the new ``ui_options`` differs from the current one, so overlay
        the new list onto a fresh merged dict; updating the ``Options`` trait first would erase that
        diff and the change would never reach the UI.
        """
        param = self._node.get_parameter_by_name(self._parameter_name)
        if param is None:
            return

        repo_ids = sorted({repo_id for repo_id, _ in self._get_cached_repo_choices()})
        ui_options = {**param.ui_options}
        ui_options["simple_dropdown"] = repo_ids
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name=self._parameter_name,
                node_name=self._node.name,
                ui_options=ui_options,
            )
        )

    def _on_refresh_click(self, _button: Button, _details: ButtonDetailsMessagePayload) -> NodeMessageResult | None:
        self.refresh_parameters()
        return None

    def _on_download_click(
        self, _button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:
        repo_id = str(self._node.get_parameter_value(self._parameter_name) or "")
        href = f"#model-management?search={repo_id}"
        return NodeMessageResult(
            success=True,
            details="Opening Model Manager",
            response=OnClickMessageResultPayload(button_details=button_details, href=href),
            altered_workflow_state=False,
        )

    @override
    def add_input_parameters(self) -> None:
        param = ParameterString(
            name=self._parameter_name,
            default_value="",
            display_name="Repo ID",
            tooltip="HuggingFace repo id, e.g. 'black-forest-labs/FLUX.1-dev'.",
            placeholder_text="e.g. black-forest-labs/FLUX.1-dev",
            allowed_modes={ParameterMode.PROPERTY},
            accept_any=False,
            converters=[self._on_value_changed],
            badge=_INFO_BADGE,
            traits={  # type: ignore[reportArgumentType]
                Options(choices=sorted({repo_id for repo_id, _ in self._get_cached_repo_choices()}), allow_custom=True),
                Button(
                    icon="list-restart",
                    size="icon",
                    variant="secondary",
                    on_click=self._on_refresh_click,
                    tooltip="Refresh model status",
                ),
            },
        )
        self._node.add_parameter(param)

        download_button = ParameterButton(
            name=self._download_param_name,
            label="Open Model Manager to Download",
            icon="download",
            variant="secondary",
            full_width=True,
            on_click=self._on_download_click,
            tooltip="Open Model Manager to download this repo",
            hide=True,
        )
        self._node.add_parameter(download_button)

    @override
    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name(self._parameter_name)
        self._node.remove_parameter_element_by_name(self._download_param_name)

    @override
    def refresh_parameters(self, value_being_set: str | None = None) -> None:
        """Re-check the local cache for the typed repo id.

        Deliberately does not delegate to the base implementation. That one rebuilds a dropdown:
        it adds an `Options` trait to the parameter and overwrites the stored value with a choice
        drawn from the cache scan. This parameter is a free-text field that accepts any repo id,
        so there are no choices to offer and the typed value has to survive a refresh.
        """
        repo_id = value_being_set
        if repo_id is None:
            repo_id = self._node.get_parameter_value(self._parameter_name)
        if repo_id is None:
            repo_id = ""

        self._invalidate_cached_repo_choices()
        self._push_repo_choices_to_ui()
        self._refresh_parameters(repo_id)

    def _refresh_parameters(self, repo_id: str) -> None:
        self._repo_revisions = self._fetch_repo_revisions(repo_id)
        cached_ids = {r for r, _ in self._repo_revisions}
        if repo_id and repo_id not in cached_ids:
            self._node.show_parameter_by_name(self._download_param_name)
        else:
            self._node.hide_parameter_by_name(self._download_param_name)

    @override
    def validate_before_node_run(self) -> list[Exception] | None:
        repo_id = self._node.get_parameter_value(self._parameter_name)
        # Rescan at the run boundary so validation reflects repos downloaded since the last refresh.
        self._invalidate_cached_repo_choices()
        downloaded = {r for r, _ in self._fetch_repo_revisions(repo_id)}
        if repo_id and repo_id not in downloaded:
            return [
                RuntimeError(
                    f"Attempted to run LoadComponent. Failed with repo_id='{repo_id}' "
                    "because it is not in the local HF cache. Download it via Model Management."
                )
            ]
        return None
