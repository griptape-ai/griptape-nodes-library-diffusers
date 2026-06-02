import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("modular_diffusers_nodes_library")


class LatentDiffusionPipelineBuilderControlNetParameter:
    def __init__(self, node: BaseNode):
        self._node = node
        self._control_nets_parameter_name = "control_nets"
        self._control_nets_output_parameter_name = "control_parameters"

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            ParameterList(
                name="control_nets",
                input_types=["control_net"],
                default_value=[],
                type="control_net",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                tooltip="Connect one or more ControlNet nodes to apply their conditioning to the pipeline.",
            )
        )

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name=self._control_nets_output_parameter_name,
                default_value={},
                type="control_parameters",
                output_type="control_parameters",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Aggregated ControlNet configuration passed to a Generate Latents node.",
            )
        )

    def validate_before_node_run(self, provider: str) -> list[Exception] | None:
        control_nets = self.get_control_nets()
        if control_nets is None:
            return None

        error_list = []
        for control_net in control_nets:
            if not isinstance(control_net, dict):
                error_list.append(ValueError(f"Control net entry is not a dict: {control_net}"))
                continue

            if "model" not in control_net:
                error_list.append(ValueError(f"Control net entry is missing 'model' key: {control_net}"))

            if "provider" not in control_net:
                error_list.append(ValueError(f"Control net entry is missing 'provider' key: {control_net}"))

            elif control_net.get("provider") != provider:
                error_list.append(
                    ValueError(
                        f"Control net provider '{control_net.get('provider')}' does not match pipeline provider '{provider}' in control net entry: {control_net}"
                    )
                )

        control_net_parameters = self.get_control_net_parameters(control_nets)
        if control_net_parameters:
            expected_keys = control_net_parameters[0].keys()
            for control_parameter in control_net_parameters:
                if not isinstance(control_parameter, dict):
                    return [ValueError(f"Control parameter entry must be a dict: {control_parameter!r}")]
                if control_parameter.keys() != expected_keys:
                    return [
                        ValueError(
                            f"All control parameter entries must have the same keys. Expected keys: {expected_keys}, got: {control_parameter.keys()} in entry: {control_parameter!r}"
                        )
                    ]

        return error_list if error_list else None

    def get_control_nets(self) -> list[dict[str, Any]]:
        control_nets_list = self._node.get_parameter_value(self._control_nets_parameter_name) or []
        return control_nets_list

    @staticmethod
    def get_control_net_parameters(control_nets: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        if control_nets is not None:
            control_net_parameters = [
                control_net["parameters"] for control_net in control_nets if "parameters" in control_net
            ]
            return control_net_parameters
        return None

    def get_control_net_parameters_as_kwargs(self) -> dict[str, Any]:
        control_nets = self.get_control_nets()
        control_net_parameters = self.get_control_net_parameters(control_nets)
        kwargs = {}
        if control_net_parameters:
            keys = control_net_parameters[0].keys()
            for key in keys:
                entry_values = []
                for control_parameter in control_net_parameters:
                    parameter = control_parameter.get(key)
                    entry_values.append(parameter)
                kwargs[key] = entry_values if len(entry_values) > 1 else entry_values[0]
        return kwargs

    def sync_output_parameters(self) -> None:
        control_net_parameters = self.get_control_net_parameters_as_kwargs()
        self._node.publish_update_to_parameter(self._control_nets_output_parameter_name, control_net_parameters)
        self._node.parameter_output_values[self._control_nets_output_parameter_name] = control_net_parameters
