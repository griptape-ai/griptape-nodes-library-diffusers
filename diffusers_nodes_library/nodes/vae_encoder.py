import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL.Image import Image

from diffusers_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from diffusers_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
from diffusers_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from diffusers_nodes_library.parameters.pipeline_parameters import ModularDiffusionPipelineParameters
from diffusers_nodes_library.utils.image_utils import load_image_from_url_artifact
from diffusers_nodes_library.utils.pillow_utils import image_artifact_to_pil
from diffusers_nodes_library.utils.video_utils import load_video_frames_from_url_artifact

logger = logging.getLogger("modular_diffusers_nodes_library")


class VaeEncodeNode(ControlNode):
    def __init__(self, **kwargs) -> None:
        self._initializing = True
        self._current_input_type = "image"
        super().__init__(**kwargs)

        self.pipe_params = ModularDiffusionPipelineParameters(self)
        self.pipe_params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Input image to encode into VAE latent space.",
                allowed_modes={ParameterMode.INPUT},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="latent_tensor",
                output_type="LatentArtifact",
                tooltip="Encoded latent tensor from the pipeline VAE.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )
        self._initializing = False

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        Input parameters can only be added during initialisation.
        Input media parameters (image, input_video) may be swapped dynamically
        based on the pipeline driver's produces_video property.
        """
        if self._initializing:
            return super().add_parameter(param)

        is_input_media_param = param.name in {"image", "input_video"}
        if not is_input_media_param:
            return

        if not self.does_name_exist(param.name):
            param.user_defined = True
            super().add_parameter(param)

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        if self.get_element_by_name_and_type(element_name):
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(parameter_name=element_name, node_name=self.name)
            )

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

        if parameter.name == "pipeline":
            value = normalize_diffusion_pipeline_value(
                value,
                node_name=self.name,
                raise_on_invalid=True,
            )

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        self.pipe_params.after_value_set(parameter, value)
        self.pipe_params.runtime_parameters.after_value_set(parameter, value)

        if param_name == "pipeline":
            self._update_input_parameter()

    def _update_input_parameter(self) -> None:
        driver_cls = get_driver_class(self.pipe_params.get_pipeline_class())
        if driver_cls is None:
            return

        if driver_cls.produces_video:
            new_input_type = "video"
        else:
            new_input_type = "image"

        if new_input_type == self._current_input_type:
            return

        if new_input_type == "video":
            self.remove_parameter_element_by_name("image")
            self.add_parameter(
                Parameter(
                    name="input_video",
                    input_types=["VideoArtifact", "VideoUrlArtifact"],
                    type="VideoUrlArtifact",
                    tooltip="Input video to encode into VAE latent space.",
                    allowed_modes={ParameterMode.INPUT},
                    user_defined=True,
                )
            )
        else:
            self.remove_parameter_element_by_name("input_video")
            self.add_parameter(
                Parameter(
                    name="image",
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    type="ImageArtifact",
                    tooltip="Input image to encode into VAE latent space.",
                    allowed_modes={ParameterMode.INPUT},
                    user_defined=True,
                )
            )

        self._current_input_type = new_input_type

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        pipeline_errors = self.pipe_params.validate_before_node_run()
        if pipeline_errors is not None:
            errors.extend(pipeline_errors)

        if self._current_input_type == "video":
            if self.get_parameter_value("input_video") is None:
                errors.append(ValueError("Missing required 'input_video' input."))
        else:
            if self.get_parameter_value("image") is None:
                errors.append(ValueError("Missing required 'image' input."))

        return errors or None

    def process(self) -> AsyncResult:
        if self._current_input_type == "video":
            yield lambda: self.convert_video_to_latent()
        else:
            yield lambda: self.convert_image_to_latent()

    def convert_image_to_latent(self) -> None:
        pipe = self.pipe_params.get_pipeline()
        image = self.get_input_image()

        latents_pipeline_driver = create_driver(pipe, self.pipe_params.get_pipeline_class())
        latents = latents_pipeline_driver.encode_image(image)
        image_tensor = pipe.image_processor.preprocess(image)
        if isinstance(image_tensor, (list, tuple)):
            image_tensor = image_tensor[0]

        latent_artifact = LatentArtifact.from_torch(latents, source_shape=image_tensor.shape)
        self.set_parameter_value("latent_tensor", latent_artifact)
        self.parameter_output_values["latent_tensor"] = latent_artifact

    def convert_video_to_latent(self) -> None:
        pipe = self.pipe_params.get_pipeline()
        pipeline_class = self.pipe_params.get_pipeline_class()

        latents_pipeline_driver = create_driver(pipe, pipeline_class)

        video_artifact = self.get_parameter_value("input_video")
        if video_artifact is None:
            msg = "Attempted to encode video. Failed with data input_video=None because input video was missing."
            raise ValueError(msg)

        frames = load_video_frames_from_url_artifact(video_artifact)
        if not frames:
            msg = "Attempted to encode video. Failed because no frames could be loaded from the input video."
            raise ValueError(msg)

        frames_rgb = [f.convert("RGB") for f in frames]
        latents = latents_pipeline_driver.encode_video(frames_rgb)

        # Build 5-D source shape [B, C, T, H, W] so downstream nodes recover height/width correctly.
        sample_tensor = pipe.video_processor.preprocess(frames_rgb[0])

        num_frames = len(frames_rgb)
        b, c, h, w = sample_tensor.shape
        source_shape = (b, c, num_frames, h, w)

        latent_artifact = LatentArtifact.from_torch(latents, source_shape=source_shape)
        self.set_parameter_value("latent_tensor", latent_artifact)
        self.parameter_output_values["latent_tensor"] = latent_artifact

    def get_input_image(self) -> Image:
        image_artifact = self.get_parameter_value("image")
        if image_artifact is None:
            msg = "Attempted to encode image. Failed with data image=None because input image was missing."
            raise ValueError(msg)

        if isinstance(image_artifact, ImageUrlArtifact):
            image_artifact = load_image_from_url_artifact(image_artifact)

        return image_artifact_to_pil(image_artifact).convert("RGB")
