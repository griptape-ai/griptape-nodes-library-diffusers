import logging
import math
from datetime import UTC, datetime
from typing import Any, ClassVar

import torch  # type: ignore[import]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import (
    InpaintMaskArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import DecodeResult, LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import create_driver, get_driver_class
from modular_diffusion_nodes_library.utils.directory_utils import (
    check_cleanup_intermediates_directory,
    get_intermediates_directory_path,
)
from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil, pil_to_image_artifact

logger = logging.getLogger("modular_diffusers_nodes_library")


DEFAULT_NUM_INFERENCE_STEPS = 20


class DiffusionPipelineGenerateLatentParameters:
    CONTROL_NET_PARAM_NAME: ClassVar = "controlnet_parameters"
    ADDITIONAL_PARAM_NAME: ClassVar = "additional_parameters"

    def __init__(self, node: BaseNode):
        self._node = node

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="input_latent",
                input_types=["LatentArtifact", "InpaintMaskArtifact"],
                type="LatentArtifact",
                tooltip="Input latent tensor. Connect an Encode Inpaint Latent to enable inpainting mode.",
                allowed_modes={ParameterMode.INPUT},
                serializable=False,
            )
        )

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="output_latent",
                type="LatentArtifact",
                tooltip="Denoised output latent. Connect to a VAE Decode node to render the final image/video.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )

    def add_property_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="add_noise",
                default_value=False,
                type="bool",
                tooltip="Add noise to the input latent before denoising. Enable for image-to-image/video-to-video/refinement workflows.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self._node.add_parameter(
            Parameter(
                name="start_step",
                default_value=0,
                type="int",
                tooltip="Denoising step index (0-based) to start from. Use a non-zero value to resume a multi-stage denoise.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self._node.add_parameter(
            Parameter(
                name="end_step",
                default_value=-1,
                type="int",
                tooltip="Denoising step index (0-based) to stop at. Use -1 to run all remaining steps. Set lower than num_inference_steps to leave the latent partially noised for downstream stages.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self._node.add_parameter(
            Parameter(
                name="return_fully_denoised",
                default_value=False,
                type="bool",
                tooltip="When enabled, the terminal sigma is appended to the sliced schedule so the denoiser always reaches a fully clean latent.",
                allowed_modes={ParameterMode.PROPERTY},
                hide=True,  # TO DO: This parameter doesn't work as intended yet
            )
        )
        self._node.add_parameter(
            Parameter(
                name="preview_image",
                output_type="ImageUrlArtifact",
                tooltip="Live preview of intermediate denoising steps. Updated periodically during generation.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("input_latent")

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("output_latent")

    def _get_pipeline_class_name(self, pipeline_value: Any) -> str | None:
        if pipeline_value is None:
            return None
        if hasattr(pipeline_value, "pipeline_name"):
            return pipeline_value.pipeline_name
        return None

    def _resolve_driver_class(self, pipeline_value: Any) -> type[LatentPipelineDriver] | None:
        return get_driver_class(self._get_pipeline_class_name(pipeline_value))

    def _supports_inpainting(self, pipeline_value: Any) -> bool:
        driver_cls = self._resolve_driver_class(pipeline_value)
        if driver_cls is None:
            return False
        return driver_cls._inpaint_pipeline_class is not None

    def _is_control_net_pipeline(self, pipeline_value: Any) -> bool:
        return isinstance(pipeline_value, ControlNetDiffusionPipelineArtifact)

    def update_add_noise_visibility(self, input_latent_artifact: Any) -> None:
        """Hide ``add_noise`` when the input is an InpaintMaskArtifact; Inpaint flows manage noise internally."""
        if isinstance(input_latent_artifact, InpaintMaskArtifact):
            self._node.hide_parameter_by_name("add_noise")
        else:
            self._node.show_parameter_by_name("add_noise")

    def add_or_remove_control_net_parameter(self, current_pipeline: Any, new_pipeline: Any) -> None:
        if self._is_control_net_pipeline(current_pipeline) and not self._is_control_net_pipeline(new_pipeline):
            self._node.remove_parameter_element_by_name(
                DiffusionPipelineGenerateLatentParameters.CONTROL_NET_PARAM_NAME
            )
        elif not self._is_control_net_pipeline(current_pipeline) and self._is_control_net_pipeline(new_pipeline):
            self._node.add_parameter(
                Parameter(
                    name=DiffusionPipelineGenerateLatentParameters.CONTROL_NET_PARAM_NAME,
                    input_types=["control_parameters"],
                    default_value={},
                    type="control_parameters",
                    allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                    tooltip="ControlNet conditioning bundle from a ControlNet node. Required when the pipeline was built with ControlNet support.",
                )
            )

    def add_additional_parameters(self) -> None:
        additional_parameters_param = ParameterList(
            name=DiffusionPipelineGenerateLatentParameters.ADDITIONAL_PARAM_NAME,
            input_types=["additional_parameters", "dict"],
            default_value=[],
            type="additional_parameters",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            tooltip="Extra pipeline-specific kwargs (e.g. media generation conditioning).",
        )
        additional_parameters_param.set_badge(
            variant="help",
            title="Pipeline-specific inputs",
            message=(
                "Use this only when a selected pipeline requires extra structured inputs. "
                "For example, connect a Media Gen Conditioning output for LTX/WAN video workflows. "
                "Or add any key/value pair and it will be forwarded to the selected pipeline as a kwarg. "
                "If the same key appears multiple times, values are merged into a list."
            ),
        )
        self._node.add_parameter(additional_parameters_param)

    @staticmethod
    def update_end_parameters(current_end_parameters: list[str], is_control_net_pipeline: bool) -> list[str]:
        end_parameters = current_end_parameters
        if is_control_net_pipeline:
            end_parameters = [DiffusionPipelineGenerateLatentParameters.CONTROL_NET_PARAM_NAME] + end_parameters
        end_parameters = [DiffusionPipelineGenerateLatentParameters.ADDITIONAL_PARAM_NAME] + end_parameters
        return end_parameters

    def process_pipeline(self, pipe: DiffusionPipeline, pipeline_class: str | None, pipe_kwargs: dict) -> None:
        num_inference_steps = self.get_num_inference_steps()
        # Default to False for better performance - preview intermediates slow down inference
        enable_preview = GriptapeNodes.ConfigManager().get_config_value(
            "advanced_media_library.enable_image_preview_intermediates", default=False
        )

        strength_affected_steps = math.ceil(num_inference_steps * self.get_strength())

        first_iteration_time = None
        latent_pipeline_driver = create_driver(pipe, pipeline_class)
        pipeline_artifact = self._node.pipe_params.get_pipeline_artifact()
        # TODO: Temporary hack — we mutate the driver instance with the pipeline's provenance
        # metadata so model-specific decode/denoise paths can detect generation-time state
        # (e.g. LTX-2 HDR LoRA via ``runtime_adapter_steps``) without re-activating the pipeline.
        # A more robust solution would pass state in and out of driver stages explicitly
        latent_pipeline_driver.provenance_metadata = dict(pipeline_artifact.metadata)
        pipe_kwargs = self.update_kwargs(pipe_kwargs)

        input_latent_artifact = self._node.get_parameter_value("input_latent")
        if input_latent_artifact is None:
            raise RuntimeError("Input latent is missing.")
        source_shape = input_latent_artifact.source_shape

        def callback_on_step_end(
            pipe: DiffusionPipeline,
            i: int,
            _t: int,
            callback_kwargs: dict,
        ) -> dict:
            nonlocal first_iteration_time
            # Check for cancellation request
            if self._node.is_cancellation_requested:
                if hasattr(pipe, "_interrupt"):
                    pipe._interrupt = True
                self._node.log_params.append_to_logs("Cancellation requested, stopping after this step...\n")  # type: ignore[reportAttributeAccessIssue]
                return callback_kwargs

            if first_iteration_time is None:
                first_iteration_time = datetime.now(tz=UTC)
            if enable_preview:
                self.publish_output_image_preview_latents(
                    callback_kwargs["latents"], source_shape, latent_pipeline_driver
                )
            if i == 0:
                self._node.progress_bar_component.increment()  # type: ignore[reportAttributeAccessIssue]
            else:
                self._node.log_params.append_to_logs(  # type: ignore[reportAttributeAccessIssue]
                    f"Completed inference step {i + 1} of {strength_affected_steps}. {f'{(datetime.now(tz=UTC) - first_iteration_time).total_seconds() / i:.2f}'} s/it\n"
                )
                self._node.progress_bar_component.increment()  # type: ignore[reportAttributeAccessIssue]
            return {}

        self._node.progress_bar_component.initialize(num_inference_steps)  # type: ignore[reportAttributeAccessIssue]
        input_latent_for_denoise = self.prepare_input_latent(input_latent_artifact, latent_pipeline_driver)
        if input_latent_for_denoise is None:
            raise ValueError("Failed to prepare input latent for the pipeline.")

        if self.end_step == 0 and self.add_noise():
            # Noise-only shortcut — skip denoising entirely
            output_latent_artifact = input_latent_for_denoise
        else:
            output_latent_artifact = latent_pipeline_driver.denoise_latent(
                input_latent_for_denoise,
                num_inference_steps,
                callback=callback_on_step_end,
                start_step=self.start_step,
                end_step=self.end_step,
                return_fully_denoised=self.return_fully_denoised,
                seed=self.get_seed(),
                **pipe_kwargs,
            )
        self.publish_output_latent(output_latent_artifact)
        self._node.log_params.append_to_logs("Done.\n")  # type: ignore[reportAttributeAccessIssue]

    def prepare_input_latent(
        self, input_latent_artifact: LatentArtifact | InpaintMaskArtifact, latent_pipeline_driver: LatentPipelineDriver
    ) -> LatentArtifact | InpaintMaskArtifact | None:
        if input_latent_artifact is None:
            raise ValueError("Input latent is missing")
        if not self.add_noise():
            return input_latent_artifact
        if isinstance(input_latent_artifact, InpaintMaskArtifact):
            # Inpaint flows manage their own noising inside denoise_latent.
            return input_latent_artifact
        return latent_pipeline_driver.add_noise_to_latent(
            input_latent_artifact,
            self.get_seed(),
            self.get_num_inference_steps(),
            self.get_strength(),
        )

    def publish_output_latent(self, output_latent_artifact: LatentArtifact) -> None:
        pipeline_artifact = self._node.pipe_params.get_pipeline_artifact()
        merged_meta = dict(output_latent_artifact.meta or {})
        # Carry pipeline-level provenance forward without clobbering driver-namespaced values.
        for key, value in pipeline_artifact.metadata.items():
            merged_meta.setdefault(key, value)
        latent_artifact = LatentArtifact.from_torch(
            output_latent_artifact.to_torch(),
            source_shape=output_latent_artifact.source_shape,
            meta=merged_meta,
        )
        self._node.publish_update_to_parameter("output_latent", latent_artifact)
        self._node.set_parameter_value("output_latent", latent_artifact)
        self._node.parameter_output_values["output_latent"] = latent_artifact

    def latents_to_image_pil(
        self, latents: torch.Tensor, source_shape: tuple[int, ...], latent_pipeline_driver: LatentPipelineDriver
    ) -> DecodeResult:
        unpacked = latent_pipeline_driver.prepare_output_latent(latents, source_shape)
        preview_artifact = latent_pipeline_driver._make_latent_artifact(unpacked, source_shape=source_shape)
        return latent_pipeline_driver.decode_latent(preview_artifact)

    def publish_output_image_preview_latents(
        self, latents: torch.Tensor, source_shape: tuple[int, ...], latent_pipeline_driver: LatentPipelineDriver
    ) -> None:
        # Check to ensure there's enough space in the intermediates directory
        # if that setting is enabled.
        check_cleanup_intermediates_directory()

        preview_image_pil = self.latents_to_image_pil(latents, source_shape, latent_pipeline_driver)
        if isinstance(preview_image_pil, list):
            preview_image_pil = preview_image_pil[0]
        if not isinstance(preview_image_pil, Image):
            # Can't preview if the output isn't a PIL image
            return
        preview_image_artifact = pil_to_image_artifact(
            preview_image_pil, directory_path=get_intermediates_directory_path()
        )
        self._node.publish_update_to_parameter("preview_image", preview_image_artifact)

    def get_num_inference_steps(self) -> int:
        return int(self._node.get_parameter_value("num_inference_steps"))

    def add_noise(self) -> bool:
        return bool(self._node.get_parameter_value("add_noise"))

    def get_seed(self) -> int:
        return int(self._node.get_parameter_value("seed"))

    @property
    def start_step(self) -> int:
        return int(self._node.get_parameter_value("start_step"))

    @property
    def end_step(self) -> int:
        return int(self._node.get_parameter_value("end_step"))

    @property
    def return_fully_denoised(self) -> bool:
        return bool(self._node.get_parameter_value("return_fully_denoised"))

    def get_strength(self) -> float:
        number_of_steps = self.get_num_inference_steps()
        if number_of_steps > 0 and self.start_step > 0:
            return 1.0 - (self.start_step / number_of_steps)
        return 1.0

    def get_control_net_parameters(self) -> dict[str, Any] | None:
        control_net_parameters = self._node.get_parameter_value(
            DiffusionPipelineGenerateLatentParameters.CONTROL_NET_PARAM_NAME
        )
        if control_net_parameters is not None:
            return control_net_parameters
        return None

    @staticmethod
    def _get_control_image_pil(control_image_artifact: ImageArtifact | ImageUrlArtifact) -> Image:
        if isinstance(control_image_artifact, ImageUrlArtifact):
            control_image_artifact = load_image_from_url_artifact(control_image_artifact)
        control_image_pil = image_artifact_to_pil(control_image_artifact)
        control_image_pil = control_image_pil.convert("RGB")
        return control_image_pil

    def update_kwargs(self, pipe_kwargs: dict[str, Any]) -> dict[str, Any]:
        additional_params = self._node.get_parameter_value(
            DiffusionPipelineGenerateLatentParameters.ADDITIONAL_PARAM_NAME
        )
        if additional_params:
            for additional_param in additional_params:
                if isinstance(additional_param, dict):
                    self._merge_additional_param(pipe_kwargs, additional_param)

        pipeline_value = self._node.get_parameter_value("pipeline")
        if self._is_control_net_pipeline(pipeline_value):
            control_net_parameters = self.get_control_net_parameters()
            if control_net_parameters:
                for key, value in control_net_parameters.items():
                    if isinstance(value, (ImageArtifact, ImageUrlArtifact)):
                        pipe_kwargs[key] = self._get_control_image_pil(value)
                    elif isinstance(value, list) and value and isinstance(value[0], (ImageArtifact, ImageUrlArtifact)):
                        pipe_kwargs[key] = [self._get_control_image_pil(artifact) for artifact in value]
                    else:
                        pipe_kwargs[key] = value
        return pipe_kwargs

    @staticmethod
    def _merge_additional_param(pipe_kwargs: dict[str, Any], additional_param: dict[str, Any]) -> None:
        """Merge an additional_param dict into pipe_kwargs.
        Keys forward as plain kwargs. Repeated keys are accumulated into a list.
        """
        for key, value in additional_param.items():
            if key not in pipe_kwargs:
                pipe_kwargs[key] = value
                continue

            existing_value = pipe_kwargs[key]
            if isinstance(existing_value, list):
                existing_value.append(value)
                continue

            pipe_kwargs[key] = [existing_value, value]

    def validate_before_node_run(self) -> list[Exception] | None:
        if self._node.get_parameter_by_name("input_latent") is not None:
            input_latent_artifact = self._node.get_parameter_value("input_latent")
            if input_latent_artifact is None:
                return [ValueError("Input latent is required but not connected.")]

            source_shape = input_latent_artifact.source_shape
            if source_shape is None or len(source_shape) < 2:
                return [
                    ValueError(
                        f"Input latent has no valid source_shape (got {source_shape!r}). "
                        "Ensure the latent was created from an image or has image dimensions set."
                    )
                ]
        else:
            input_latent_artifact = None

        return self._validate_driver_run_configuration(input_latent_artifact)

    def _validate_driver_run_configuration(
        self, input_latent_artifact: LatentArtifact | None
    ) -> list[Exception] | None:
        pipeline_value = self._node.get_parameter_value("pipeline")
        if not isinstance(pipeline_value, DiffusionPipelineArtifact):
            return None

        driver_cls = self._resolve_driver_class(pipeline_value)
        if driver_cls is None:
            return None

        errors = driver_cls.validate_run_configuration(pipeline_value, input_latent_artifact)
        if errors:
            return errors

        return None

    @staticmethod
    def validate_pipeline_class(pipeline_class: str | None) -> bool:
        return get_driver_class(pipeline_class) is not None
