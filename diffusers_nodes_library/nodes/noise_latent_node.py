import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter

from diffusers_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from diffusers_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
from diffusers_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from diffusers_nodes_library.latent_pipeline_drivers.stable_diffusion_xl import (
    StableDiffusionXLLatentPipelineDriver,
)
from diffusers_nodes_library.mixins.parameter_connection_preservation_mixin import (
    ParameterConnectionPreservationMixin,
)
from diffusers_nodes_library.parameters.generate_latent_parameters import (
    DEFAULT_NUM_INFERENCE_STEPS,
    DiffusionPipelineGenerateLatentParameters,
)
from diffusers_nodes_library.parameters.pipeline_parameters import (
    ModularDiffusionPipelineParameters,
)
from diffusers_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")


class NoiseLatentNode(ParameterConnectionPreservationMixin, ControlNode):
    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)
        self.pipe_params = ModularDiffusionPipelineParameters(self)
        self.pipe_params.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="num_inference_steps",
                default_value=DEFAULT_NUM_INFERENCE_STEPS,
                type="int",
                tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
            )
        )
        self.add_seed_parameter()

        self.add_parameter(
            Parameter(
                name="width",
                default_value=1024,
                type="int",
                tooltip="Width in pixels of the latent (will be divided by the VAE scale factor internally).",
            )
        )
        self.add_parameter(
            Parameter(
                name="height",
                default_value=1024,
                type="int",
                tooltip="Height in pixels of the latent (will be divided by the VAE scale factor internally).",
            )
        )
        self.add_parameter(
            Parameter(
                name="num_frames",
                default_value=41,
                type="int",
                tooltip="Number of video frames to generate for. Ignored for image pipelines.",
            )
        )
        self.latent_parameter = DiffusionPipelineGenerateLatentParameters(self)  # type: ignore[reportOptionalMemberAccess]
        self.latent_parameter.add_output_parameters()
        self.hide_parameter_by_name("num_inference_steps")
        self._reorder_trailing_parameters()
        self._initializing = False

    def add_seed_parameter(self) -> None:
        self._seed_parameter = SeedParameter(self)
        self._seed_parameter.add_input_parameters()

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

        # hide num_frames parameter if the pipeline doesn't produce video
        if param_name == "pipeline":
            latent_pipeline_driver = get_driver_class(self.pipe_params.get_pipeline_class())
            if latent_pipeline_driver and latent_pipeline_driver.produces_video:
                self.show_parameter_by_name("num_frames")
            else:
                self.hide_parameter_by_name("num_frames")
            # num_inference_steps only applies to SDXL noise scaling
            if latent_pipeline_driver is StableDiffusionXLLatentPipelineDriver:
                self.show_parameter_by_name("num_inference_steps")
            else:
                self.hide_parameter_by_name("num_inference_steps")
            self._reorder_trailing_parameters()

    def _reorder_trailing_parameters(self) -> None:
        """Move ``num_inference_steps`` and ``output_latent`` to the end."""
        trailing = ["num_inference_steps", "output_latent"]
        existing = [element.name for element in self.root_ui_element._children]
        head = [name for name in existing if name not in trailing]
        tail = [name for name in trailing if name in existing]
        self.reorder_elements([*head, *tail])

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        This is only allowed during the initialisation stage.
        This prevents changes to the pipeline and runtime parameters
        dynamically adding parameters and modifying connections.
        """
        if not self._initializing:
            return

        super().add_parameter(param)

    def validate_before_node_run(self) -> list[Exception] | None:
        result = self.pipe_params.validate_before_node_run()
        if result is not None:
            return result

        return None

    def preprocess(self) -> None:
        pass

    def process(self) -> AsyncResult:
        self.preprocess()

        def work() -> Any:
            try:
                latents, source_shape = self._process()
                latent_artifact = LatentArtifact.from_torch(latents, source_shape=source_shape)
                self.publish_update_to_parameter("output_latent", latent_artifact)
                self.set_parameter_value("output_latent", latent_artifact)
                self.parameter_output_values["output_latent"] = latent_artifact

            except Exception:
                logger.exception("%s: Diffusion Pipeline execution failed", self.name)
                # Aggressive cleanup on failure
                cleanup_memory_caches()
                raise

        yield work

    def _process(self) -> Any:
        pipe = self.pipe_params.get_pipeline()
        latent_pipeline_driver = create_driver(pipe, self.pipe_params.get_pipeline_class())
        height = self.get_parameter_value("height")
        width = self.get_parameter_value("width")
        seed = self.get_parameter_value("seed") or 0
        if latent_pipeline_driver.produces_video:
            num_frames = self.get_parameter_value("num_frames") or 1
            latents_source_shape = (1, 3, num_frames, height, width)
        else:
            latents_source_shape = (1, 3, height, width)
        if isinstance(latent_pipeline_driver, StableDiffusionXLLatentPipelineDriver):
            num_inference_steps = int(self.get_parameter_value("num_inference_steps") or 50)
            latents = latent_pipeline_driver.create_noise_latent(
                latents_source_shape,
                seed,
                num_inference_steps=num_inference_steps,
            )
        else:
            latents = latent_pipeline_driver.create_noise_latent(latents_source_shape, seed)
        return latents, latents_source_shape
