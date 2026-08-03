from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    HFRepoRef,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import TextEncoderComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.tokenizer_component_artifact import TokenizerComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import (
    SLOT_DISPLAY_NAMES,
    slot_artifact_type_name,
    slot_component_kind,
)
from modular_diffusion_nodes_library.component_loading.config_resolver import HF_REPO_ID_PATTERN
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.parameters.user_specified_hf_repo_parameter import (
    UserSpecifiedHuggingFaceRepoParameter,
)
from modular_diffusion_nodes_library.utils.connection_utils import drop_outgoing_connections
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")

# Slot names this node currently supports loading (subset of ALLOWED_COMPONENT_SLOTS).
# Add entries here as new component types are implemented; order sets the dropdown order.
_LOADABLE_SLOTS = ["transformer", "unet", "vae", "tokenizer", "text_encoder"]
_LOADABLE_COMPONENTS: dict[str, str] = {SLOT_DISPLAY_NAMES[slot]: slot for slot in _LOADABLE_SLOTS}
_COMPONENT_CHOICES = list(_LOADABLE_COMPONENTS.keys())

# Source-type dropdown values.
_SOURCE_SINGLE_FILE = "Single File"
_SOURCE_LOCAL_FOLDER = "Local Folder"
_SOURCE_HF_REPO = "HuggingFace Repo"
_SOURCE_TYPE_CHOICES = [_SOURCE_SINGLE_FILE, _SOURCE_LOCAL_FOLDER, _SOURCE_HF_REPO]

# File types for different source types.
_SINGLE_FILE_TYPES = [".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin"]
_TEXT_ENCODER_FILE_TYPES = [".gguf"]

# Parameters that only apply to a given source-type branch. Used
# to hide/show the right sub-parameters.
_SOURCE_TYPE_PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    _SOURCE_SINGLE_FILE: ("file_path", "config_source"),
    _SOURCE_LOCAL_FOLDER: ("folder_path",),
    _SOURCE_HF_REPO: ("repo_id", "repo_id_download", "revision", "subfolder"),
}


class LoadComponent(SuccessFailureExecutionMixin, SuccessFailureNode):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="component",
                type="str",
                default_value="Transformer",
                traits={Options(choices=_COMPONENT_CHOICES)},
                tooltip="Which pipeline component slot this loader targets.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"display_name": "Component"},
            )
        )

        source_type_param = Parameter(
            name="source_type",
            type="str",
            default_value=_SOURCE_SINGLE_FILE,
            traits={Options(choices=_SOURCE_TYPE_CHOICES)},
            tooltip="Weight source format.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"display_name": "Source Type"},
        )
        source_type_param.set_badge(
            variant="help",
            title="Source Type Options",
            message=(
                "Choose where your model weights are stored:\n\n"
                "**Single File** — Pick this if you downloaded a single model weight file on your computer "
                "(like a file ending in .gguf or .safetensors).\n\n"
                "**Local Folder** — Pick a folder on your computer in diffusers format for a single component "
                "(containing `config.json` + weight files, e.g. `.../FLUX.1-dev/transformer/`).\n\n"
                "**HuggingFace Repo** — Load from a HuggingFace repo id already in your local HF cache "
                "(e.g. `black-forest-labs/FLUX.1-dev`). No downloads are triggered."
            ),
        )
        self.add_parameter(source_type_param)

        # ------------------------------------------------------------------
        # Single File branch
        # ------------------------------------------------------------------
        self._file_path_param = FilePathParameter(
            self,
            file_types=[".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin"],
            tooltip="Absolute path to a single-file component weight (e.g. flux1-dev-Q8_0.gguf).",
        )
        self._file_path_param.add_input_parameters()

        file_path_param = self.get_parameter_by_name("file_path")
        if file_path_param is not None:
            file_path_param.ui_options = {
                "placeholder_text": "e.g. /path/to/models/flux1-dev-Q8_0.gguf",
                "display_name": "File Path",
            }

        config_source_param = Parameter(
            name="config_source",
            type="str",
            default_value="",
            tooltip=(
                "Local path to a config.json (or its parent directory), or a HuggingFace repo_id. "
                "Leave blank to auto-resolve."
            ),
            allowed_modes={ParameterMode.PROPERTY},
            traits={
                FileSystemPicker(
                    allow_files=True,
                    allow_directories=True,
                    multiple=False,
                )
            },
            ui_options={
                "display_name": "Config (Optional)",
                "placeholder_text": "e.g. repo/id  or  /path/to/config.json",
            },
        )
        config_source_param.set_badge(
            variant="help",
            title="Config Source",
            message=(
                "Accepted values:\n"
                "- **Local path** — a `config.json` file, or a directory containing one "
                "(use the picker, or type directly)\n"
                "- **HF repo_id** — e.g. `black-forest-labs/FLUX.1-dev` "
                "(the HuggingFace cache is checked, no download triggered)\n\n"
                "Leave blank to auto-resolve: warm HF cache for the detected model → "
                "bundled fallback shipped with this library."
            ),
        )
        self.add_parameter(config_source_param)

        # ------------------------------------------------------------------
        # Local Folder branch
        # ------------------------------------------------------------------
        workspace_path = str(GriptapeNodes.ConfigManager().workspace_path)
        folder_path_param = Parameter(
            name="folder_path",
            type="str",
            default_value="",
            tooltip="Absolute path to a diffusers-format component folder containing config.json + weights.",
            allowed_modes={ParameterMode.PROPERTY},
            traits={
                FileSystemPicker(
                    allow_files=False,
                    allow_directories=True,
                    multiple=False,
                    initial_path=workspace_path,
                )
            },
            ui_options={
                "display_name": "Component Folder",
                "placeholder_text": "e.g. /path/to/FLUX.1-dev/transformer",
            },
        )
        folder_path_param.set_badge(
            variant="help",
            title="Component Folder",
            message=(
                "Pick a single diffusers-format component folder — the one that directly contains "
                "`config.json` plus its weight file(s) (e.g. `.../FLUX.1-dev/transformer/`).\n\n"
                "**Supported weight files:**\n"
                "- `diffusion_pytorch_model.safetensors`\n"
                "- `diffusion_pytorch_model.bin`\n"
                "- Sharded index (`diffusion_pytorch_model.safetensors.index.json` + shards)\n\n"
                "**Not supported here:** a `.gguf` file next to `config.json`. "
                "For GGUF weights, use **Single File** mode and point **Config Source** at "
                "the folder (or its `config.json`)."
            ),
        )
        self.add_parameter(folder_path_param)

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
            tooltip="Subfolder within the repo. Leave blank to use the selected component slot name.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={
                "display_name": "Subfolder (Optional)",
                "placeholder_text": "e.g. transformer",
            },
        )
        subfolder_param.set_badge(
            variant="help",
            title="Repo Subfolder",
            message=(
                "Path inside the repo that contains the component's `config.json` + weights.\n\n"
                "Leave blank to auto-derive from the selected **Component** "
                "(e.g. `transformer`, `unet`, `vae`, `tokenizer`, `text_encoder`)."
            ),
        )
        self.add_parameter(subfolder_param)

        # Default component is "Transformer", derive its artifact type name
        default_component_slot = _LOADABLE_COMPONENTS["Transformer"]
        default_artifact_type = slot_artifact_type_name(default_component_slot)

        self.add_parameter(
            Parameter(
                name="component_output",
                type=default_artifact_type,
                output_type=default_artifact_type,
                default_value=None,
                tooltip="Artifact describing this component. Wire into a Pipeline Builder override port.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )

        self._create_status_parameters()
        self._apply_source_type_visibility(_SOURCE_SINGLE_FILE)

    # ------------------------------------------------------------------
    # Value change handling
    # ------------------------------------------------------------------
    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._file_path_param.on_after_value_set(parameter, value)

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

        if param_name == "component":
            self._update_output_type_and_drop_connections()
            self._update_source_type_choices(value)
            self._update_file_types_for_component(value)
        if param_name == "source_type" and isinstance(value, str):
            self._apply_source_type_visibility(value)
        if param_name != "component_output":
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

    def _update_source_type_choices(self, component_display_name: str) -> None:
        """Restrict source_type choices based on the selected component.

        Tokenizers cannot be loaded from a single file — remove that option
        and, if it was selected, reset to Local Folder.
        """
        source_type_param = self.get_parameter_by_name("source_type")
        if source_type_param is None:
            return

        component_slot = _LOADABLE_COMPONENTS.get(component_display_name, "")
        is_tokenizer = slot_component_kind(component_slot) == "tokenizer"

        new_choices = [_SOURCE_LOCAL_FOLDER, _SOURCE_HF_REPO] if is_tokenizer else _SOURCE_TYPE_CHOICES

        options_trait = next(iter(source_type_param.find_elements_by_type(Options)), None)
        if options_trait is not None:
            options_trait.choices = new_choices

        current_source = self.get_parameter_value("source_type")
        if is_tokenizer and current_source == _SOURCE_SINGLE_FILE:
            self.set_parameter_value("source_type", _SOURCE_LOCAL_FOLDER)
        else:
            self.set_parameter_value("source_type", current_source)

    def _update_file_types_for_component(self, component_display_name: str) -> None:
        component_slot = _LOADABLE_COMPONENTS.get(component_display_name, "")
        is_text_encoder = slot_component_kind(component_slot) == "text_encoder"
        self._file_path_param.set_file_types(_TEXT_ENCODER_FILE_TYPES if is_text_encoder else _SINGLE_FILE_TYPES)

    def _update_output_type_and_drop_connections(self) -> None:
        """Update component_output type when component selection changes.

        Also drops any existing outgoing connections from component_output,
        since the type change would make them silently invalid.
        """
        component_display_name = self.get_parameter_value("component")
        if component_display_name not in _LOADABLE_COMPONENTS:
            return

        component_slot = _LOADABLE_COMPONENTS[component_display_name]
        new_artifact_type = slot_artifact_type_name(component_slot)

        output_param = self.get_parameter_by_name("component_output")
        if output_param is None:
            return

        # Update both type and output_type for UI consistency
        output_param.type = new_artifact_type
        output_param.output_type = new_artifact_type

        # Drop any existing outgoing connections - they're now type-incompatible
        drop_outgoing_connections(
            self,
            "component_output",
            reason=f"component type changed to {new_artifact_type}",
        )

    # ------------------------------------------------------------------
    # Validation and execution
    # ------------------------------------------------------------------
    def validate_before_node_run(self) -> list[Exception]:
        errors: list[Exception] = []

        component_display_name = self.get_parameter_value("component")
        if component_display_name not in _LOADABLE_COMPONENTS:
            errors.append(
                ValueError(
                    f"Attempted to run LoadComponent. Failed with component='{component_display_name}' "
                    f"because it is not one of the loadable components: {list(_LOADABLE_COMPONENTS)}."
                )
            )
            return errors

        source_type = self.get_parameter_value("source_type")
        if source_type == _SOURCE_SINGLE_FILE:
            errors.extend(self._validate_single_file())
            return errors
        if source_type == _SOURCE_LOCAL_FOLDER:
            errors.extend(self._validate_local_folder())
            return errors
        if source_type == _SOURCE_HF_REPO:
            errors.extend(self._validate_hf_repo())
            if errors:
                return errors
            cache_errors = self._repo_param.validate_before_node_run()
            if cache_errors:
                errors.extend(cache_errors)
            return errors

        errors.append(
            ValueError(
                f"Attempted to run LoadComponent. Failed with source_type='{source_type}' "
                f"because it is not one of {_SOURCE_TYPE_CHOICES}."
            )
        )
        return errors

    def _validate_single_file(self) -> list[Exception]:
        errors: list[Exception] = []
        component_display_name = self.get_parameter_value("component")
        component_slot = _LOADABLE_COMPONENTS.get(component_display_name, "")
        if slot_component_kind(component_slot) == "tokenizer":
            errors.append(
                ValueError(
                    "Attempted to run LoadComponent. Failed because Single File mode is not supported "
                    "for tokenizer components. Use Local Folder or HuggingFace Repo instead."
                )
            )
            return errors

        raw_file_path = self.get_parameter_value("file_path")
        if not isinstance(raw_file_path, str) or not raw_file_path:
            errors.append(ValueError("Attempted to run LoadComponent. Failed because file_path is empty."))
            return errors

        file_path = self._file_path_param.get_file_path()
        if not file_path.exists():
            errors.append(
                FileNotFoundError(f"Attempted to run LoadComponent. Failed because no file exists at '{file_path}'.")
            )
        return errors

    def _validate_local_folder(self) -> list[Exception]:
        errors: list[Exception] = []
        raw_folder_path = self.get_parameter_value("folder_path")
        if not isinstance(raw_folder_path, str) or not raw_folder_path:
            errors.append(ValueError("Attempted to run LoadComponent. Failed because folder_path is empty."))
            return errors

        folder_path = Path(expand_path_macros(raw_folder_path)).absolute()
        if not folder_path.exists():
            errors.append(
                FileNotFoundError(
                    f"Attempted to run LoadComponent. Failed because no folder exists at '{folder_path}'."
                )
            )
            return errors

        if not folder_path.is_dir():
            errors.append(
                ValueError(f"Attempted to run LoadComponent. Failed because '{folder_path}' is not a directory.")
            )
            return errors

        component_display_name = self.get_parameter_value("component")
        component_slot = _LOADABLE_COMPONENTS.get(component_display_name, "")
        if slot_component_kind(component_slot) == "tokenizer":
            config_file = folder_path / "tokenizer_config.json"
            if not config_file.is_file():
                errors.append(
                    FileNotFoundError(
                        f"Attempted to run LoadComponent. Failed with folder_path='{folder_path}' "
                        f"because it does not contain a 'tokenizer_config.json' file. Pick a "
                        f"diffusers-format tokenizer folder (e.g. '.../FLUX.1-dev/tokenizer/')."
                    )
                )
        else:
            config_file = folder_path / "config.json"
            if not config_file.is_file():
                errors.append(
                    FileNotFoundError(
                        f"Attempted to run LoadComponent. Failed with folder_path='{folder_path}' "
                        f"because it does not contain a 'config.json' file. Pick a diffusers-format "
                        f"component folder (e.g. '.../FLUX.1-dev/transformer/')."
                    )
                )
        return errors

    def _validate_hf_repo(self) -> list[Exception]:
        errors: list[Exception] = []

        raw_repo_id = self.get_parameter_value("repo_id")
        if not isinstance(raw_repo_id, str) or not raw_repo_id.strip():
            errors.append(ValueError("Attempted to run LoadComponent. Failed because repo_id is empty."))
            return errors

        repo_id = raw_repo_id.strip()
        if not HF_REPO_ID_PATTERN.match(repo_id):
            errors.append(
                ValueError(
                    f"Attempted to run LoadComponent. Failed with repo_id='{repo_id}' "
                    f"because it is not a valid HuggingFace repo id (expected 'owner/name')."
                )
            )
        return errors

    def process(self) -> None:
        self._clear_execution_status()
        self._run_with_status(
            self._rebuild_output,
            success_msg="Component artifact emitted.",
            failure_log="Component artifact build failed",
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Output construction
    # ------------------------------------------------------------------
    def _rebuild_output(self) -> None:
        component_display_name = self.get_parameter_value("component")
        if component_display_name not in _LOADABLE_COMPONENTS:
            self.set_parameter_value("component_output", None)
            return

        component_slot = _LOADABLE_COMPONENTS[component_display_name]
        source_type = self.get_parameter_value("source_type")

        if source_type == _SOURCE_SINGLE_FILE:
            artifact = self._build_single_file_artifact(component_slot)
        elif source_type == _SOURCE_LOCAL_FOLDER:
            artifact = self._build_local_folder_artifact(component_slot)
        elif source_type == _SOURCE_HF_REPO:
            artifact = self._build_hf_repo_artifact(component_slot)
        else:
            artifact = None

        self.set_parameter_value("component_output", artifact)

    def _get_artifact_class(self, component_slot: str) -> type[ModelComponentArtifact]:
        kind = slot_component_kind(component_slot)
        if kind == "tokenizer":
            return TokenizerComponentArtifact
        if kind == "text_encoder":
            return TextEncoderComponentArtifact
        return ModelComponentArtifact

    def _build_single_file_artifact(self, component_slot: str) -> ModelComponentArtifact | None:
        raw_file_path = self.get_parameter_value("file_path")
        if not isinstance(raw_file_path, str) or not raw_file_path:
            return None

        file_path = self._file_path_param.get_file_path()
        raw_config_source = self.get_parameter_value("config_source")
        if isinstance(raw_config_source, str) and raw_config_source:
            config_source = expand_path_macros(raw_config_source)
        else:
            config_source = None

        load_id = _compute_load_id(
            source_type=_SOURCE_SINGLE_FILE,
            component=component_slot,
            path=str(file_path),
            config_source=config_source or "",
            repo_id="",
            revision="",
            subfolder="",
        )

        artifact_cls = self._get_artifact_class(component_slot)
        return artifact_cls(
            load_id=load_id,
            source_type=ComponentSourceType.SINGLE_FILE,
            file_path=str(file_path),
            component=component_slot,
            config_source=config_source,
        )

    def _build_local_folder_artifact(self, component_slot: str) -> ModelComponentArtifact | None:
        raw_folder_path = self.get_parameter_value("folder_path")
        if not isinstance(raw_folder_path, str) or not raw_folder_path:
            return None

        folder_path = str(Path(expand_path_macros(raw_folder_path)).absolute())

        load_id = _compute_load_id(
            source_type=_SOURCE_LOCAL_FOLDER,
            component=component_slot,
            path=folder_path,
            config_source="",
            repo_id="",
            revision="",
            subfolder="",
        )

        artifact_cls = self._get_artifact_class(component_slot)
        return artifact_cls(
            load_id=load_id,
            source_type=ComponentSourceType.LOCAL_DIR,
            file_path=folder_path,
            component=component_slot,
        )

    def _build_hf_repo_artifact(self, component_slot: str) -> ModelComponentArtifact | None:
        raw_repo_id = self.get_parameter_value("repo_id")
        if not isinstance(raw_repo_id, str) or not raw_repo_id.strip():
            return None

        repo_id = raw_repo_id.strip()
        revision = (self.get_parameter_value("revision") or "main").strip()
        subfolder = (self.get_parameter_value("subfolder") or "").strip()

        load_id = _compute_load_id(
            source_type=_SOURCE_HF_REPO,
            component=component_slot,
            path="",
            config_source="",
            repo_id=repo_id,
            revision=revision,
            subfolder=subfolder,
        )

        artifact_cls = self._get_artifact_class(component_slot)
        return artifact_cls(
            load_id=load_id,
            source_type=ComponentSourceType.HF_REPO,
            component=component_slot,
            repo_ref=HFRepoRef(repo_id=repo_id, revision=revision, subfolder=subfolder),
        )


def _compute_load_id(
    *,
    source_type: str,
    component: str,
    path: str,
    config_source: str,
    repo_id: str,
    revision: str,
    subfolder: str,
) -> str:
    """Stable hash for cache-invalidation on the pipeline builder side."""
    payload = "|".join([source_type, component, path, config_source, repo_id, revision, subfolder])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
