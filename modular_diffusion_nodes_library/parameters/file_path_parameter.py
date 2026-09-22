from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros, resolve_path_to_macro


class FilePathParameter:
    def __init__(
        self,
        node: BaseNode,
        parameter_name: str = "file_path",
        file_types: list[str] | None = None,
        # Where the file picker opens.
        initial_path: str | None = None,
        tooltip: str = "Path to a local file",
        display_name: str | None = None,
        *,
        allow_create: bool = False,
        allowed_modes: set[ParameterMode] | None = None,
        # Stored path default for the string parameter; independent of the picker's initial_path.
        default_value: str | None = None,
    ):
        self._node = node
        self._parameter_name = parameter_name
        self._file_types = file_types
        self._initial_path = initial_path or str(GriptapeNodes.ConfigManager().workspace_path)
        self._tooltip = tooltip
        self._display_name = display_name
        self._allow_create = allow_create
        self._allowed_modes = allowed_modes
        self._default_value = default_value

    def add_input_parameters(self) -> None:
        if self._display_name:
            ui_options = {"display_name": self._display_name}
        else:
            ui_options = {}
        extra_kwargs: dict[str, Any] = {}
        if self._allowed_modes is not None:
            extra_kwargs["allowed_modes"] = self._allowed_modes
        if self._default_value is not None:
            extra_kwargs["default_value"] = self._default_value
        self._node.add_parameter(
            Parameter(
                name=self._parameter_name,
                input_types=["str"],
                type="str",
                tooltip=self._tooltip,
                ui_options=ui_options,
                traits={
                    FileSystemPicker(
                        allow_files=True,
                        allow_directories=False,
                        multiple=False,
                        file_types=self._file_types,
                        initial_path=self._initial_path,
                        allow_create=self._allow_create,
                    )
                },
                **extra_kwargs,
            )
        )

    def get_file_path(self) -> Path:
        # Use absolute() rather than resolve() to preserve symlinks.
        raw_value = self._node.get_parameter_value(self._parameter_name)
        expanded_value = expand_path_macros(raw_value) if isinstance(raw_value, str) else raw_value
        return Path(expanded_value).absolute()

    def set_file_types(self, file_types: list[str]) -> None:
        self._file_types = file_types
        param = self._node.get_parameter_by_name(self._parameter_name)
        if param is None:
            return
        picker = next(iter(param.find_elements_by_type(FileSystemPicker)), None)
        if picker is None:
            return
        picker.file_types = file_types
        # _ui_options overrides traits in the ui_options property, so both stores must stay in sync.
        existing = {k: v for k, v in param.ui_options.items() if k != "fileSystemPicker"}
        existing["fileSystemPicker"] = picker.ui_options_for_trait()["fileSystemPicker"]
        param.ui_options = existing

    def on_after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name != self._parameter_name:
            return
        if not isinstance(value, str) or not value:
            return
        resolved = resolve_path_to_macro(value)
        if resolved != value:
            self._node.set_parameter_value(self._parameter_name, resolved, emit_change=False)

    def validate_parameter_values(self) -> None:
        file_path = self.get_file_path()
        if not file_path.exists():
            msg = f"No file at {file_path} exists"
            raise RuntimeError(msg)
