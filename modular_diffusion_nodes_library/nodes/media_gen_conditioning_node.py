import logging
from typing import Any

from griptape_nodes.exe_types.node_types import SuccessFailureNode

from modular_diffusion_nodes_library.parameters.conditioning_parameters import (
    ModularDiffusionConditioningParameters,
)

from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin

logger = logging.getLogger("diffusers_nodes_library")

class MediaGenConditioningNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Conditioning node for media generation pipelines.

    Supports two modes:
    - image: one or more conditioning images, each with a strength value.
    - video: a single conditioning video with a strength value.

    Optionally accepts a `pipeline` input. When a pipeline is connected, the
    conditioning surface swaps to the configuration tailored to that pipeline
    (e.g. first/last frame presets for Wan I2V, LTX, LTX2; multi-image with
    no strength/frame_index for Flux2 Klein). With no pipeline connected the
    default flexible image-or-video surface is shown.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._conditioning_parameter = ModularDiffusionConditioningParameters(self)

        self._conditioning_parameter.add_output_parameters()
        self._conditioning_parameter.add_input_parameters()
        self._create_status_parameters()

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        # Capture pre-write value so the component can compute
        # against `value` once it has been stored.
        old_value = self.get_parameter_value(param_name)
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )
        self._conditioning_parameter.on_parameter_value_change(
            param_name, old_value, value, initial_setup=initial_setup
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        return self._conditioning_parameter.validate_before_node_run()

    def process(self) -> None:
        self._clear_execution_status()

        def build() -> None:
            self.parameter_output_values["conditioning"] = self._conditioning_parameter.build_conditioning_payload()

        self._run_with_status(
            build,
            success_msg="Conditioning built successfully.",
            failure_log="Conditioning build failed",
            logger=logger,
        )

