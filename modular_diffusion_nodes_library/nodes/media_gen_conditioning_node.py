import logging
from typing import Any

from griptape_nodes.exe_types.node_types import ControlNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning_parameter import (
    ImageConditioningConfig,
    ImageOrVideoConfig,
    MediaGenConditioningParameter,
    VideoConditioningConfig,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

logger = logging.getLogger("diffusers_nodes_library")


class MediaGenConditioningNode(ControlNode):
    """Conditioning node for media generation pipelines.

    Supports two modes:
    - image: one or more conditioning images, each with a strength value.
    - video: a single conditioning video with a strength value.

    The mode can be toggled via the 'mode' dropdown. In image mode the number of
    image inputs is controlled by the 'num_images' property.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._conditioning_parameter = MediaGenConditioningParameter(
            self,
            ImageOrVideoConfig(
                image=ImageConditioningConfig(),
                video=VideoConditioningConfig(),
                default_mode=ConditioningMode.IMAGE,
            ),
        )

        self._conditioning_parameter.add_output_parameters()
        self._conditioning_parameter.add_input_parameters()

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
        self.parameter_output_values["conditioning"] = self._conditioning_parameter.build_conditioning_payload()
