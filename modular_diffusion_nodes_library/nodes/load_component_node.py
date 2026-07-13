from __future__ import annotations

import hashlib
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentArtifact,
    ComponentSourceType,
)
from modular_diffusion_nodes_library.component_loading.component_slots import SLOT_DISPLAY_NAMES
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter

logger = logging.getLogger("modular_diffusers_nodes_library")

# Slot names this node currently supports loading (subset of ALLOWED_COMPONENT_SLOTS).
# Add entries here as new component types are implemented; order sets the dropdown order.
_LOADABLE_SLOTS = ["transformer"]
_LOADABLE_COMPONENTS: dict[str, str] = {SLOT_DISPLAY_NAMES[slot]: slot for slot in _LOADABLE_SLOTS}
_COMPONENT_CHOICES = list(_LOADABLE_COMPONENTS.keys())
_SOURCE_TYPE_CHOICES = ["Single File", "Local Folder (coming soon)", "HuggingFace Repo (coming soon)"]


class LoadComponent(SuccessFailureExecutionMixin, SuccessFailureNode):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="component",
                type="str",
                default_value="Transformer",
                traits={Options(choices=_COMPONENT_CHOICES)},
                tooltip="Which pipeline component slot this loader targets..",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        source_type_param = Parameter(
            name="source_type",
            type="str",
            default_value="Single File",
            traits={Options(choices=_SOURCE_TYPE_CHOICES)},
            tooltip="Weight source format.",
            allowed_modes={ParameterMode.PROPERTY},
        )
        source_type_param.set_badge(
            variant="help",
            title="Source Type Options",
            message=(
                "Choose where your model weights are stored:\n\n"
                "**Single File** — Pick this if you downloaded a single model weight file on your computer "
                "(like a file ending in .gguf or .safetensors).\n\n"
                "**Local Folder** — You have a folder on your computer in diffusers format "
                "(containing config.json + weight files).\n\n"
                "**HuggingFace Repo** — Load from a HuggingFace repo/id (e.g. `black-forest-labs/FLUX.1-dev`) "
                "with revision `main` and a subfolder like `vae` or `transformer`."
            ),
        )
        self.add_parameter(source_type_param)

        self._file_path_param = FilePathParameter(
            self,
            file_types=[".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin"],
            tooltip="Absolute path to a single-file component weight (e.g. flux1-dev-Q8_0.gguf).",
        )
        self._file_path_param.add_input_parameters()

        file_path_param = self.get_parameter_by_name("file_path")
        if file_path_param is not None:
            file_path_param.ui_options = {"placeholder_text": "e.g. /path/to/models/flux1-dev-Q8_0.gguf"}

        config_source_param = Parameter(
            name="config_source",
            type="str",
            default_value="",
            tooltip=("Local path to a config directory, or a HuggingFace repo_id. Leave blank to auto-resolve."),
            allowed_modes={ParameterMode.PROPERTY},
            traits={
                FileSystemPicker(
                    allow_files=False,
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
                "- **Local path** — directory containing a `config.json` "
                "(use the folder picker, or type directly)\n"
                "- **HF repo_id** — e.g. `black-forest-labs/FLUX.1-dev` "
                "(the HuggingFace cache is checked, no download triggered)\n\n"
                "Leave blank to auto-resolve: warm HF cache for the detected model → "
                "bundled fallback shipped with this library."
            ),
        )
        self.add_parameter(config_source_param)

        self.add_parameter(
            Parameter(
                name="component_output",
                output_type="ComponentArtifact",
                default_value=None,
                tooltip="ComponentArtifact describing this component. Wire into a Pipeline Builder override port.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )

        self._create_status_parameters()

    # ------------------------------------------------------------------
    # Value change handling
    # ------------------------------------------------------------------
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

        if param_name in {"file_path", "config_source", "source_type", "component"}:
            self._rebuild_output()

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
        if source_type != "Single File":
            errors.append(
                ValueError(
                    f"Attempted to run LoadComponent. Failed with source_type='{source_type}' "
                    f"because only 'Single File' is supported."
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

        source_type = self.get_parameter_value("source_type")
        if source_type != "Single File":
            self.set_parameter_value("component_output", None)
            return

        raw_file_path = self.get_parameter_value("file_path")
        if not isinstance(raw_file_path, str) or not raw_file_path:
            self.set_parameter_value("component_output", None)
            return

        file_path = self._file_path_param.get_file_path()
        raw_config_source = self.get_parameter_value("config_source")
        config_source = raw_config_source if isinstance(raw_config_source, str) and raw_config_source else None
        component_slot = _LOADABLE_COMPONENTS[component_display_name]

        load_id = _compute_load_id(
            file_path=str(file_path),
            component=component_slot,
            config_source=config_source or "",
        )

        artifact = ComponentArtifact(
            load_id=load_id,
            source_type=ComponentSourceType.SINGLE_FILE,
            file_path=str(file_path),
            component=component_slot,
            config_source=config_source,
            torch_dtype="bfloat16",
        )

        self.set_parameter_value("component_output", artifact)

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------
    def _update_visibility(self) -> None:
        # TODO: hide config_source when source_type != "Single File"
        pass


def _compute_load_id(
    *,
    file_path: str,
    component: str,
    config_source: str,
) -> str:
    """Stable hash for cache-invalidation on the pipeline builder side."""
    payload = "|".join([file_path, component, config_source])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
