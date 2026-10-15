"""HuggingFaceRepoParameter variant whose dropdown also accepts a wired connection."""

from __future__ import annotations

from typing import override

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter


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
