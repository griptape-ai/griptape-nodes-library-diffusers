from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    FlexibleImageConfig,
    MediaGenConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.flux2_runtime_parameters import (
    Flux2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode


class Flux2KleinPipelineRuntimeParameters(Flux2PipelineRuntimeParameters):
    """Flux2 Klein variant \u2014 same runtime params, with media-gen conditioning enabled.

    The Klein pipeline accepts up to 8 reference images as additional conditioning.
    Strength and frame-index controls are not exposed because Klein treats the
    reference images uniformly.
    """

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(
            min_count=1,
            max_count=8,
            expose_strength=False,
            expose_frame_index=False,
        ),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="reference_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="Reference images that guide what the inpainted region should look like. Connect a Media Gen Conditioning node.",
            badge_title="Reference images",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply reference images for **Inpainting** only. "
                "This allows conditioning the inpainted region on a specific reference image. "
                "Only **image**-mode payloads are accepted; **video** payloads are not allowed.\n\n"
                "**Tip:** You can also connect an image directly — without a Media Gen Conditioning node — "
                "for single-image conditioning."
            ),
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "input_latent":
            conditioning_images = self._node.get_parameter_by_name("reference_images")
            if conditioning_images is None:
                return
            if value is not None and isinstance(value, InpaintMaskArtifact):
                conditioning_images.allowed_modes = {ParameterMode.INPUT, ParameterMode.OUTPUT}
            else:
                # disable the parameter
                conditioning_images.allowed_modes = {ParameterMode.OUTPUT}

    def _add_input_parameters(self) -> None:
        super()._add_input_parameters()
        self._media_gen_conditioning_param.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        super()._remove_input_parameters()

    def _requires_conditioning(self) -> bool:
        """True when an InpaintMaskArtifact is connected to input_latent."""
        input_latent = self._node.get_parameter_value("input_latent")
        return isinstance(input_latent, InpaintMaskArtifact)

    def _get_pipe_kwargs(self) -> dict:
        if not self._requires_conditioning():
            return super()._get_pipe_kwargs()
        return {
            **super()._get_pipe_kwargs(),
            **self._media_gen_conditioning_param.get_pipe_kwargs(),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        conditioning_errors = self._media_gen_conditioning_param.validate_before_node_run()
        if conditioning_errors:
            errors.extend(conditioning_errors)
        return errors or None
