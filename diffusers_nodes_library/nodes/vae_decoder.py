import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from diffusers_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from diffusers_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
from diffusers_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from diffusers_nodes_library.parameters.pipeline_parameters import ModularDiffusionPipelineParameters
from diffusers_nodes_library.utils.pillow_utils import pil_to_image_artifact

logger = logging.getLogger("modular_diffusers_nodes_library")


class VaeDecodeNode(ControlNode):
    def __init__(self, **kwargs) -> None:
        self._initializing = True
        self._current_output_type = "image"
        super().__init__(**kwargs)
        self.pipe_params = ModularDiffusionPipelineParameters(self)
        self.pipe_params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="latent_tensor",
                input_types=["LatentArtifact"],
                type="LatentArtifact",
                tooltip="Latent tensor to decode with the pipeline VAE.",
                allowed_modes={ParameterMode.INPUT},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="output_image",
                output_type="ImageArtifact",
                tooltip="Decoded image from the latent tensor.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )
        self._initializing = False

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        Input parameters can only be added during initialisation.
        Output parameters (output_image, output_video) and video-specific parameters (fps)
        may be swapped dynamically based on the pipeline driver's produces_video property.
        """
        is_dynamic_param = param.name in {"output_image", "output_video", "fps"}
        if not self._initializing and not is_dynamic_param:
            return
        elif param.name in {"output_image", "output_video"}:
            param.user_defined = True

        if not self.does_name_exist(param.name):
            super().add_parameter(param)

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        # Use the retained mode request so connections are also removed.
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
            self._update_output_parameter()

    def _update_output_parameter(self) -> None:
        driver_cls = get_driver_class(self.pipe_params.get_pipeline_class())
        if driver_cls is None:
            return

        if driver_cls.produces_video:
            new_output_type = "video"
        else:
            new_output_type = "image"

        if new_output_type == self._current_output_type:
            return

        if new_output_type == "video":
            self.remove_parameter_element_by_name("output_image")
            # Add FPS parameter for video output (before output to appear above it in GUI)
            if not self.get_parameter_by_name("fps"):
                self.add_parameter(
                    Parameter(
                        name="fps",
                        default_value=25,
                        type="int",
                        tooltip="Frames per second for video output.",
                        allowed_modes={ParameterMode.PROPERTY},
                        user_defined=True,
                        ui_options={"min": 1, "max": 120},
                    )
                )
            self.add_parameter(
                Parameter(
                    name="output_video",
                    output_type="VideoUrlArtifact",
                    tooltip="Generated video.",
                    allowed_modes={ParameterMode.OUTPUT},
                    user_defined=True,
                    serializable=False,
                )
            )
            # Reorder to ensure fps appears before output_video
            self._reorder_parameters_for_video()
        else:
            self.remove_parameter_element_by_name("output_video")
            self.remove_parameter_element_by_name("fps")
            self.add_parameter(
                Parameter(
                    name="output_image",
                    output_type="ImageArtifact",
                    tooltip="Decoded image from the latent tensor.",
                    allowed_modes={ParameterMode.OUTPUT},
                    user_defined=True,
                    serializable=False,
                )
            )

        self._current_output_type = new_output_type

    def _reorder_parameters_for_video(self) -> None:
        """Reorder parameters to ensure fps appears before output_video in the GUI."""
        all_params = [element.name for element in self.root_ui_element._children]
        # Move fps before output_video if both exist
        if "fps" in all_params and "output_video" in all_params:
            all_params.remove("fps")
            video_index = all_params.index("output_video")
            all_params.insert(video_index, "fps")
            self.reorder_elements(all_params)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        pipeline_errors = self.pipe_params.validate_before_node_run()
        if pipeline_errors is not None:
            errors.extend(pipeline_errors)

        latent_tensor = self.get_parameter_value("latent_tensor")
        if latent_tensor is None:
            errors.append(ValueError("Missing required 'latent_tensor' input."))
        elif not isinstance(latent_tensor, LatentArtifact):
            errors.append(ValueError("'latent_tensor' must be a LatentArtifact."))
        elif latent_tensor.source_shape is None or len(latent_tensor.source_shape) < 2:
            errors.append(
                ValueError(
                    f"'latent_tensor' has no valid source_shape (got {latent_tensor.source_shape!r}). "
                    "Ensure the latent was created from an image or has image dimensions set."
                )
            )

        # Validate FPS parameter for video output
        if self._current_output_type == "video":
            fps = self.get_parameter_value("fps")
            if fps is not None:
                try:
                    fps_int = int(fps)
                    if fps_int <= 0:
                        errors.append(ValueError(f"FPS must be a positive integer, got {fps_int}."))
                except (ValueError, TypeError):
                    errors.append(ValueError(f"FPS must be a valid integer, got {fps!r}."))

        return errors or None

    def process(self) -> AsyncResult:
        yield lambda: self._process()

    def _process(self) -> None:
        pipe = self.pipe_params.get_pipeline()
        latent_artifact = self.get_parameter_value("latent_tensor")
        latents = latent_artifact.to_torch()

        source_shape = latent_artifact.source_shape
        latents_pipeline_driver = create_driver(pipe, self.pipe_params.get_pipeline_class())
        output = latents_pipeline_driver.decode_latent(latents, source_shape)

        if latents_pipeline_driver.produces_video:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
                temp_path = Path(temp_file_obj.name)
            try:
                fps = int(self.get_parameter_value("fps") or latents_pipeline_driver.video_fps)
                diffusers.utils.export_to_video(output, str(temp_path), fps=fps)  # type: ignore[attr-defined]
                self._publish_output_video(temp_path)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        else:
            if isinstance(output, list):
                raise ValueError("Decoder returned an unexpected list of outputs.")
            image_artifact = pil_to_image_artifact(output)
            self.set_parameter_value("output_image", image_artifact)
            self.parameter_output_values["output_image"] = image_artifact

    def _publish_output_video(self, video_path: Path) -> None:
        filename = f"{uuid.uuid4()}{video_path.suffix}"
        url = GriptapeNodes.StaticFilesManager().save_static_file(video_path.read_bytes(), filename)
        self.parameter_output_values["output_video"] = VideoUrlArtifact(url)
