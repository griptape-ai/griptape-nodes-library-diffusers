from __future__ import annotations

import hashlib
import inspect
import json
import logging
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    HFRepoRef,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import slot_artifact_type_name
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.parameters.user_specified_hf_repo_parameter import (
    UserSpecifiedHuggingFaceRepoParameter,
)
from modular_diffusion_nodes_library.utils.json_utils import parse_json_to_dict

logger = logging.getLogger("modular_diffusers_nodes_library")

_SCHEDULER_SLOT = "scheduler"

# Curated dropdown of scheduler classes. Flow-matching family (Flux / SD3 / Qwen /
# Wan / LTX) first, then the classic (epsilon / v-prediction) family used by SDXL.
# Restricted on purpose — the full diffusers list includes many model-specific /
# inverse / parallel schedulers that can't drive these pipelines.
_FLOW_MATCHING_CHOICES = [
    "FlowMatchEulerDiscreteScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "LTXEulerAncestralRFScheduler",
]
_CLASSIC_CHOICES = [
    "EulerDiscreteScheduler",
    "EulerAncestralDiscreteScheduler",
    "DDIMScheduler",
    "DDPMScheduler",
    "DPMSolverMultistepScheduler",
    "DPMSolverSinglestepScheduler",
    "UniPCMultistepScheduler",
    "DEISMultistepScheduler",
    "HeunDiscreteScheduler",
    "LMSDiscreteScheduler",
    "KDPM2DiscreteScheduler",
    "KDPM2AncestralDiscreteScheduler",
    "PNDMScheduler",
]
_SCHEDULER_CHOICES = _FLOW_MATCHING_CHOICES + _CLASSIC_CHOICES

# Broader flow-matching set used only to classify a class into a family (for the
# compatibility hint) — includes siblings we don't offer in the dropdown.
_FLOW_MATCHING_SCHEDULERS = {
    *_FLOW_MATCHING_CHOICES,
    "FlowMatchLCMScheduler",
    "FlowMapEulerDiscreteScheduler",
}

_FAMILY_GUIDANCE = (
    "Flow-matching schedulers (FlowMatch*) drive Flux, SD3, Qwen, Wan and LTX; "
    "classic schedulers (Euler, DDIM, DPMSolver, UniPC, …) drive SDXL. "
    "Pick a class from the same family as your pipeline — a cross-family scheduler "
    "will produce wrong output or error at generation time."
)

# Config-source dropdown values.
_SOURCE_LOCAL_PATH = "Local Path"
_SOURCE_HF_REPO = "HuggingFace Repo"
_SOURCE_TEXT_CONFIG = "Text Config"
_SOURCE_TYPE_CHOICES = [_SOURCE_LOCAL_PATH, _SOURCE_HF_REPO, _SOURCE_TEXT_CONFIG]

# Parameters owned by each config-source branch (for show/hide toggling).
_SOURCE_TYPE_PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    _SOURCE_LOCAL_PATH: ("config_path",),
    _SOURCE_HF_REPO: ("repo_id", "repo_id_download", "revision", "subfolder"),
    _SOURCE_TEXT_CONFIG: ("text_config",),
}


def _scheduler_family(scheduler_class: str | None) -> str | None:
    """Classify a scheduler class name into ``'flow'`` or ``'classic'`` (or None)."""
    if not scheduler_class:
        return None
    return "flow" if scheduler_class in _FLOW_MATCHING_SCHEDULERS else "classic"


class LoadSchedulerComponent(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Configure a scheduler and emit a SchedulerComponentArtifact override.

    Schedulers are config-only (no weights): pick a scheduler class and point at a
    ``scheduler_config.json`` (local file/folder or a cached HF repo). The output
    wires into a Pipeline Builder's ``scheduler`` override port.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Warning surface — hidden unless a class/config family mismatch is detected.
        self._compat_message = ParameterMessage(
            name="scheduler_compatibility",
            variant="warning",
            title="Scheduler compatibility",
            value="",
            hide=True,
            markdown=True,
        )
        self.add_node_element(self._compat_message)

        scheduler_class_param = Parameter(
            name="scheduler_class",
            type="str",
            default_value=_FLOW_MATCHING_CHOICES[0],
            traits={Options(choices=_SCHEDULER_CHOICES, show_search=True)},
            tooltip="Scheduler class to instantiate. Must match your pipeline's family (see the badge).",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"display_name": "Scheduler Class"},
        )
        scheduler_class_param.set_badge(variant="help", title="Scheduler compatibility", message=_FAMILY_GUIDANCE)
        self.add_parameter(scheduler_class_param)

        source_type_param = Parameter(
            name="source_type",
            type="str",
            default_value=_SOURCE_LOCAL_PATH,
            traits={Options(choices=_SOURCE_TYPE_CHOICES)},
            tooltip="Where the scheduler_config.json comes from.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"display_name": "Source Type"},
        )
        source_type_param.set_badge(
            variant="help",
            title="Config Source",
            message=(
                "Where to read the scheduler's settings from:\n\n"
                "**Local Path** — a `scheduler_config.json` file, or a folder containing one "
                "(e.g. `.../FLUX.1-dev/scheduler/`).\n\n"
                "**HuggingFace Repo** — a repo id already in your local HF cache; its `scheduler` "
                "subfolder is read. No downloads are triggered.\n\n"
                "**Text Config** — paste JSON-formatted scheduler config directly into the text box, "
                "or wire in a dict from a **JSON Input**, **Create Dictionary**, or **Merge Key Value Pairs** node. "
                "Leave the field empty to use the scheduler class's built-in defaults."
            ),
        )
        self.add_parameter(source_type_param)

        # ------------------------------------------------------------------
        # Local Path branch
        # ------------------------------------------------------------------
        self._config_path_param = FilePathParameter(
            self,
            parameter_name="config_path",
            file_types=[".json"],
            tooltip="Path to a scheduler_config.json file. The selected scheduler class is instantiated from these settings.",
            display_name="Config Path",
        )
        self._config_path_param.add_input_parameters()
        self.get_parameter_by_name("config_path").set_badge(
            variant="help",
            title="Config Path",
            message=(
                "Path to a `scheduler_config.json` file.\n\n"
                "The selected scheduler class is instantiated from these settings; "
                "the config's own `_class_name` is ignored."
            ),
        )

        # ------------------------------------------------------------------
        # HuggingFace Repo branch
        # ------------------------------------------------------------------
        self._repo_param = UserSpecifiedHuggingFaceRepoParameter(self, "repo_id")
        self._repo_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="revision",
                type="str",
                default_value="main",
                tooltip="Repo revision (branch, tag, or commit hash). Defaults to 'main'.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Revision"},
            )
        )

        subfolder_param = Parameter(
            name="subfolder",
            type="str",
            default_value="",
            tooltip="Subfolder within the repo containing scheduler_config.json. Leave blank to use 'scheduler'.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={
                "display_name": "Subfolder (Optional)",
                "placeholder_text": "e.g. scheduler",
            },
        )
        subfolder_param.set_badge(
            variant="help",
            title="Repo Subfolder",
            message=(
                "Path inside the repo that contains `scheduler_config.json`.\n\n"
                "Leave blank to use the default `scheduler` subfolder (where diffusers pipelines keep it)."
            ),
        )
        self.add_parameter(subfolder_param)

        # ------------------------------------------------------------------
        # Manual Config branch
        # ------------------------------------------------------------------
        text_config_param = Parameter(
            name="text_config",
            input_types=["json", "str", "dict"],
            type="json",
            default_value=None,
            tooltip=(
                "Paste the contents of a scheduler_config.json here, or wire from a JSON Input / "
                "Create Dictionary node. Leave empty to use the scheduler class defaults."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={
                "display_name": "Scheduler Config",
                "multiline": True,
                "placeholder_text": '{\n  "shift": 3.0\n}',
            },
        )
        text_config_param.set_badge(
            variant="help",
            title="Manual Scheduler Config",
            message=(
                "Paste a `scheduler_config.json` directly, or wire from a **JSON Input**, "
                "**Create Dictionary**, or **Merge Key Value Pairs** node.\n\n"
                "Leave empty to instantiate the chosen scheduler class with its built-in defaults. "
                "The value is auto-formatted to pretty-printed JSON when you change it."
            ),
        )
        self.add_parameter(text_config_param)

        artifact_type = slot_artifact_type_name(_SCHEDULER_SLOT)
        self.add_parameter(
            Parameter(
                name="component_output",
                type=artifact_type,
                output_type=artifact_type,
                default_value=None,
                tooltip="Scheduler artifact. Wire into a Pipeline Builder scheduler override port.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )

        self._create_status_parameters()
        self._apply_source_type_visibility(_SOURCE_LOCAL_PATH)

    # ------------------------------------------------------------------
    # Value change handling
    # ------------------------------------------------------------------
    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._config_path_param.on_after_value_set(parameter, value)
        if parameter.name == "text_config":
            self._auto_format_text_config(value)

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if initial_setup:
            return

        if param_name == "source_type" and isinstance(value, str):
            self._apply_source_type_visibility(value)
        if param_name != "component_output" and emit_change:
            self._rebuild_output()

    def _apply_source_type_visibility(self, source_type: str) -> None:
        """Show only the parameters belonging to ``source_type``; hide the rest."""
        for branch, param_names in _SOURCE_TYPE_PARAM_GROUPS.items():
            if branch == source_type:
                for name in param_names:
                    self.show_parameter_by_name(name)
            else:
                for name in param_names:
                    self.hide_parameter_by_name(name)
        if source_type == _SOURCE_HF_REPO:
            self._repo_param.refresh_parameters()

    # ------------------------------------------------------------------
    # Validation and execution
    # ------------------------------------------------------------------
    def validate_before_node_run(self) -> list[Exception]:
        scheduler_class = self.get_parameter_value("scheduler_class")
        if scheduler_class not in _SCHEDULER_CHOICES:
            return [
                ValueError(
                    f"Attempted to run LoadSchedulerComponent. Failed with scheduler_class='{scheduler_class}' "
                    f"because it is not one of the supported schedulers: {_SCHEDULER_CHOICES}."
                )
            ]

        errors: list[Exception] = []
        source_type = self.get_parameter_value("source_type")
        if source_type == _SOURCE_HF_REPO:
            cache_errors = self._repo_param.validate_before_node_run()
            if cache_errors:
                errors.extend(cache_errors)
                return errors

        try:
            result = self._resolve_artifact_config()
        except Exception as e:  # noqa: BLE001
            errors.append(
                ValueError(
                    f"Attempted to run LoadSchedulerComponent. "
                    f"Failed because the selected source could not be read: {e}"
                )
            )
            return errors
        if result is None:
            errors.append(
                ValueError(
                    "Attempted to run LoadSchedulerComponent. "
                    "Failed because no scheduler config source is configured. "
                    "Set a local path, HuggingFace repo id, or paste a JSON config in Text Config mode."
                )
            )
            return errors

        return errors

    def process(self) -> None:
        self._clear_execution_status()
        self._run_with_status(
            self._rebuild_output,
            success_msg="Scheduler artifact emitted.",
            failure_log="Scheduler artifact build failed",
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------
    def _resolve_artifact_config(self) -> tuple[SchedulerComponentArtifact, dict[str, Any]] | None:
        """Build the artifact and read its config."""
        artifact = self._build_artifact()
        if artifact is None:
            return None

        config = artifact.read_config()
        return artifact, config

    def _rebuild_output(self) -> None:
        try:
            result = self._resolve_artifact_config()
        except Exception as e:  # noqa: BLE001
            self.set_parameter_value("component_output", None)
            self._show_warning(f"The selected source could not be read ({e}).")
            return
        if result is None:
            self.set_parameter_value("component_output", None)
            self._hide_message()
            return

        artifact, config = result
        self.set_parameter_value("component_output", artifact)
        source_type = self.get_parameter_value("source_type")
        if source_type == _SOURCE_TEXT_CONFIG:
            self._check_unknown_config_keys(config, artifact.scheduler_class)
        else:
            self._update_family_message(config.get("_class_name"))

    def _build_artifact(self) -> SchedulerComponentArtifact | None:
        scheduler_class = self.get_parameter_value("scheduler_class")
        if scheduler_class not in _SCHEDULER_CHOICES:
            return None

        source_type = self.get_parameter_value("source_type")
        if source_type == _SOURCE_LOCAL_PATH:
            return self._build_local_scheduler_artifact(scheduler_class)

        if source_type == _SOURCE_HF_REPO:
            return self._build_hf_scheduler_artifact(scheduler_class)

        if source_type == _SOURCE_TEXT_CONFIG:
            return self._build_text_scheduler_artifact(scheduler_class)

        return None

    def _build_local_scheduler_artifact(self, scheduler_class: str) -> SchedulerComponentArtifact | None:
        raw_path = self.get_parameter_value("config_path")
        if not isinstance(raw_path, str) or not raw_path:
            return None

        config_source = str(self._config_path_param.get_file_path())
        load_id = _compute_load_id(
            source_type=_SOURCE_LOCAL_PATH,
            scheduler_class=scheduler_class,
            config_source=config_source,
            repo_id="",
            revision="",
            subfolder="",
        )
        return SchedulerComponentArtifact(
            load_id=load_id,
            source_type=ComponentSourceType.LOCAL_DIR,
            component=_SCHEDULER_SLOT,
            scheduler_class=scheduler_class,
            config_source=config_source,
        )

    def _build_hf_scheduler_artifact(self, scheduler_class: str) -> SchedulerComponentArtifact | None:
        raw_repo_id = self.get_parameter_value("repo_id")
        if not isinstance(raw_repo_id, str) or not raw_repo_id.strip():
            return None

        repo_id = raw_repo_id.strip()
        revision = (self.get_parameter_value("revision") or "main").strip()
        subfolder = (self.get_parameter_value("subfolder") or "scheduler").strip()
        load_id = _compute_load_id(
            source_type=_SOURCE_HF_REPO,
            scheduler_class=scheduler_class,
            config_source="",
            repo_id=repo_id,
            revision=revision,
            subfolder=subfolder,
        )
        return SchedulerComponentArtifact(
            load_id=load_id,
            source_type=ComponentSourceType.HF_REPO,
            component=_SCHEDULER_SLOT,
            scheduler_class=scheduler_class,
            repo_ref=HFRepoRef(repo_id=repo_id, revision=revision, subfolder=subfolder),
        )

    def _build_text_scheduler_artifact(self, scheduler_class: str) -> SchedulerComponentArtifact:
        raw_text_config = self.get_parameter_value("text_config")
        config_dict = parse_json_to_dict(raw_text_config)
        load_id = _compute_load_id(
            source_type=_SOURCE_TEXT_CONFIG,
            scheduler_class=scheduler_class,
            config_source="",
            repo_id="",
            revision="",
            subfolder="",
            text_config=json.dumps(config_dict, sort_keys=True),
        )
        if not config_dict:
            config_dict = None
        return SchedulerComponentArtifact(
            load_id=load_id,
            source_type=ComponentSourceType.RAW_CONFIG,
            component=_SCHEDULER_SLOT,
            scheduler_class=scheduler_class,
            text_config=config_dict,
        )

    def _auto_format_text_config(self, value: Any) -> None:
        if value is None or value == "":
            return
        try:
            config_dict = parse_json_to_dict(value)
        except ValueError:
            return
        formatted = json.dumps(config_dict, indent=2)
        if formatted == value:
            return
        self.set_parameter_value("text_config", formatted, emit_change=False)

    def _update_family_message(self, config_class_name: str | None) -> bool:
        """Warn when the config's scheduler family differs from the chosen class; else hide. Returns True if warning was shown."""
        if not isinstance(config_class_name, str):
            self._hide_message()
            return False
        scheduler_class = self.get_parameter_value("scheduler_class")
        chosen_family = _scheduler_family(scheduler_class)
        config_family = _scheduler_family(config_class_name)

        if chosen_family is not None and config_family is not None and chosen_family != config_family:
            self._show_warning(
                f"Chosen scheduler '{scheduler_class}' is {chosen_family}-family, but the provided config is from "
                f"a {config_family}-family scheduler ('{config_class_name}'). The chosen class will be used, but its "
                "settings may not transfer correctly. Make sure the class matches your pipeline."
            )
            return True
        self._hide_message()
        return False

    def _check_unknown_config_keys(self, config: dict[str, Any], scheduler_class_name: str) -> None:
        config_class_name = config.get("_class_name")
        if isinstance(config_class_name, str) and self._update_family_message(config_class_name):
            return

        scheduler_cls = getattr(diffusers, scheduler_class_name, None)
        if not isinstance(scheduler_cls, type):
            self._hide_message()
            return

        try:
            valid_params = set(inspect.signature(scheduler_cls.__init__).parameters.keys()) - {"self"}
        except (ValueError, TypeError):
            self._hide_message()
            return

        # Skip _class_name / _diffusers_version etc.
        unknown_keys = sorted(k for k in config if not k.startswith("_") and k not in valid_params)
        if not unknown_keys:
            self._hide_message()
            return

        keys_list = "\n".join(f"- `{k}`" for k in unknown_keys)
        if len(unknown_keys) == 1:
            this_or_these = "this config key"
        else:
            this_or_these = "these config keys"
        self._show_warning(
            f"**{scheduler_class_name}** does not recognise {this_or_these}:\n{keys_list}\n\n"
            f"They will be silently ignored by diffusers."
        )

    def _show_warning(self, message: str) -> None:
        self._compat_message.value = message
        self._compat_message.variant = "warning"
        self._compat_message.hide = False

    def _hide_message(self) -> None:
        self._compat_message.value = ""
        self._compat_message.hide = True


def _compute_load_id(
    *,
    source_type: str,
    scheduler_class: str,
    config_source: str,
    repo_id: str,
    revision: str,
    subfolder: str,
    text_config: str = "",
) -> str:
    """Stable hash for cache-invalidation on the pipeline builder side."""
    payload = "|".join(
        [source_type, _SCHEDULER_SLOT, scheduler_class, config_source, repo_id, revision, subfolder, text_config]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
