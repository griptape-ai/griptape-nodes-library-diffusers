import logging

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from PIL.Image import Image

from modular_diffusion_nodes_library.runtime_parameters.flux_runtime_parameters import (
    FluxPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil

logger = logging.getLogger("diffusers_nodes_library")


class FluxKontextPipelineRuntimeParameters(FluxPipelineRuntimeParameters):
    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        super()._add_input_parameters()
        image_param = Parameter(
            name="image",
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageArtifact",
            tooltip="Context image for editing (when using noise input) or reference image for inpainting (when using inpaint mask input). Optional.",
            ui_options={"hide_property": True},
            allowed_modes={ParameterMode.INPUT},
            user_defined=True,
        )
        image_param.set_badge(
            variant="help",
            title="Dual-purpose image input",
            message=(
                "This input serves two roles depending on what else is connected:\n\n"
                "**Image editing** — connect a context image alongside a noise latent. "
                "Flux Kontext will use it as a reference to guide edits described in the prompt.\n\n"
                "**Inpainting** — connect a reference image alongside an inpaint mask. "
                "Flux Kontext will fill the masked region using the surrounding image context."
            ),
        )
        self._node.add_parameter(image_param)

    def _remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("image")
        super()._remove_input_parameters()

    def _get_image_pil(self) -> Image | None:
        input_image_artifact = self._node.get_parameter_value("image")
        if input_image_artifact is None:
            return None
        if isinstance(input_image_artifact, ImageUrlArtifact):
            input_image_artifact = load_image_from_url_artifact(input_image_artifact)
        input_image_pil = image_artifact_to_pil(input_image_artifact)
        return input_image_pil.convert("RGB")

    def _get_pipe_kwargs(self) -> dict:
        kwargs = super()._get_pipe_kwargs()
        image = self._get_image_pil()
        if image is not None:
            kwargs["image"] = image
        return kwargs
