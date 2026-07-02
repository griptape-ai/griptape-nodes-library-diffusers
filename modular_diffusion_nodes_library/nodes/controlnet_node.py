import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.mixins.parameter_connection_preservation_mixin import (
    ParameterConnectionPreservationMixin,
)
from modular_diffusion_nodes_library.parameters.controlnet_node_parameter_types import (
    CONTROLNET_MODEL_PARAMETER_NAME,
    ControlNetNodesParameterType,
    FluxControlNetNodesParameterType,
    QwenImageControlNetNodesParameterType,
    StableDiffusion3ControlNetNodesParameterType,
    StableDiffusionControlNetNodesParameterType,
    ZImageControlNetNodesParameterType,
)
from modular_diffusion_nodes_library.parameters.providers import Provider

logger = logging.getLogger("modular_diffusers_nodes_library")


class ControlNetNode(ParameterConnectionPreservationMixin, ControlNode):
    PROVIDER_MAP: ClassVar[dict[str, type[ControlNetNodesParameterType]]] = {
        Provider.FLUX: FluxControlNetNodesParameterType,
        Provider.QWEN: QwenImageControlNetNodesParameterType,
        Provider.STABLE_DIFFUSION: StableDiffusionControlNetNodesParameterType,
        Provider.STABLE_DIFFUSION_3: StableDiffusion3ControlNetNodesParameterType,
        Provider.Z_IMAGE: ZImageControlNetNodesParameterType,
    }
    STATIC_PARAMS: ClassVar = ["provider"]
    START_PARAMS: ClassVar = ["provider"]
    END_PARAMS: ClassVar = ["controlnet_conditioning_scale", "control_image", "control_net"]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.provider_choices = list(self.PROVIDER_MAP.keys())
        self._controlnet_parameter_type: ControlNetNodesParameterType
        self.did_provider_change = False

        self.add_parameter(
            Parameter(
                name="provider",
                type="str",
                traits={Options(choices=self.provider_choices)},
                tooltip="Select the model family this ControlNet targets (must match the pipeline provider).",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "Select Provider"},
            )
        )
        self.add_parameter(
            Parameter(
                name="controlnet_conditioning_scale",
                default_value=1.0,
                input_types=["float"],
                type="float",
                tooltip="Weight of the ControlNet's contribution to denoising. Higher values strengthen adherence to the control image; 0.0 disables it.",
                ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
            )
        )
        control_image_param = Parameter(
            name="control_image",
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageArtifact",
            tooltip="Control image used as conditioning (e.g. canny edge map, depth map, pose skeleton).",
            allowed_modes={ParameterMode.INPUT},
        )
        control_image_param.set_badge(
            variant="help",
            title="Image must match the ControlNet type",
            message=(
                "The image must match the type of ControlNet model selected.\n\n"
                "Plugging in an incompatible image to the selected ControlNet model will produce weak or no guidance.\n\n"
                "e.g. depth maps only work with depth ControlNets, canny edge maps only work with canny ControlNets, etc."
            ),
        )
        self.add_parameter(control_image_param)
        self.add_parameter(
            Parameter(
                name="control_net",
                default_value={},
                type="control_net",
                output_type="control_net",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Configured ControlNet to connect to a ControlNet Pipeline Builder.",
            )
        )
        self.set_controlnet_parameter_type(self.provider_choices[0])
        self._controlnet_parameter_type.add_input_parameters()
        self.reorder_parameters_by_groups()

    def validate_before_node_run(self) -> list[Exception] | None:
        error_list = self._controlnet_parameter_type.validate_before_node_run()
        if error_list is not None:
            return error_list

        if self._controlnet_parameter_type.is_control_image_required():
            control_image = self.get_parameter_value("control_image")
            if control_image is None:
                return [ValueError("Control image parameter cannot be None")]

        return None

    def process(self) -> None:
        control_net_config = self.get_config_kwargs()
        self.set_parameter_value("control_net", control_net_config)
        self.parameter_output_values["control_net"] = control_net_config

    def set_controlnet_parameter_type(self, provider: str) -> None:
        if provider not in self.PROVIDER_MAP:
            msg = f"Unsupported pipeline provider: {provider}"
            logger.error(msg)
            raise ValueError(msg)

        provider_class = self.PROVIDER_MAP[provider]
        self._controlnet_parameter_type = provider_class(self)

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return

        # During initial_setup, before/after_value_set hooks are not called by the base class.
        # Handle provider change here to ensure proper restore when opening an existing script.
        # Also fire for controlnet_model so its UI hooks (e.g. hide/show control_image) apply on load.
        fire_hooks_on_initial_setup = parameter.name in ("provider", CONTROLNET_MODEL_PARAMETER_NAME)
        if initial_setup and fire_hooks_on_initial_setup:
            self.before_value_set(parameter, value)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if initial_setup and fire_hooks_on_initial_setup:
            self.after_value_set(parameter, value)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        if parameter.name == "provider":
            current_provider = self.get_parameter_value("provider")
            self.did_provider_change = current_provider != value
        return super().before_value_set(parameter, value)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "provider" and self.did_provider_change:
            self.regenerate_controlnet_parameter_type_for_provider(value)
        elif parameter.name == CONTROLNET_MODEL_PARAMETER_NAME:
            self._controlnet_parameter_type.on_model_changed(value)
        return super().after_value_set(parameter, value)

    def regenerate_controlnet_parameter_type_for_provider(self, provider: str) -> None:
        # Save parameter properties and connections before removing parameters
        self.save_parameter_properties()
        saved_incoming, saved_outgoing = self._save_connections()

        self.controlnet_parameter_type.remove_input_parameters()
        self.set_controlnet_parameter_type(provider)
        self.controlnet_parameter_type.add_input_parameters()

        # Restore connections after adding parameters
        self._restore_connections(saved_incoming, saved_outgoing)

        # Reorder parameters to maintain consistent layout
        self.reorder_parameters_by_groups()

        self.clear_parameter_cache()

    @property
    def controlnet_parameter_type(self) -> ControlNetNodesParameterType:
        if self._controlnet_parameter_type is None:
            msg = "Pipeline type parameters not initialized. Ensure provider parameter is set."
            logger.error(msg)
            raise ValueError(msg)
        return self._controlnet_parameter_type

    def get_provider(self) -> str:
        return self.get_parameter_value("provider")

    def get_config_kwargs(self) -> dict:
        return {
            "model": self.get_parameter_value(CONTROLNET_MODEL_PARAMETER_NAME),
            "provider": self.get_provider(),
            "parameters": self._controlnet_parameter_type.get_kwargs(),
        }
