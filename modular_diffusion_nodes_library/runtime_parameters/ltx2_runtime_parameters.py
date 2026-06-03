import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep

logger = logging.getLogger("diffusers_nodes_library")


class LTX2PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    _HDR_LORA_ADAPTER_TOKEN = "ic-lora-hdr"

    def __init__(self, node: BaseNode, pipeline_metadata: dict[str, Any]):
        super().__init__(node)
        self._pipeline_metadata = pipeline_metadata
        self._text_embeddings_path_param = FilePathParameter(
            node=node,
            parameter_name="text_embeddings_path",
            file_types=[".safetensors"],
            tooltip="Optional safetensors file with LTX2 HDR text embeddings. Used only when HDR IC-LoRA is active and overrides prompt text.",
        )

    def add_input_parameters(self) -> None:
        self._add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="num_inference_steps",
                default_value=8 if self._is_distilled else 40,
                type="int",
                tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.",
                hide=self._is_distilled,
            )
        )
        self._seed_parameter.add_input_parameters()

    def _add_input_parameters(self) -> None:
        if self._is_distilled:
            self._node.add_parameter(
                Parameter(
                    name="use_stage_2",
                    default_value=False,
                    type="bool",
                    tooltip="If True, run stage 2 (refinement). If False, run stage 1 (initial generation).",
                )
            )
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide video generation. If not defined, a prompt_embeds tensor must be passed instead.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide video generation. Ignored when not using guidance (i.e. when guidance_scale is less than 1).",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=1.0 if self._is_distilled else 3.0,
                type="float",
                tooltip="Classifier-Free Guidance scale for the video modality. Higher values steer generation more strongly toward the prompt, usually at the expense of visual quality. A separate audio_guidance_scale controls the audio modality. For LTX-2.3 the authors suggest 3.0 for video.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="stg_scale",
                default_value=0.0 if self._is_distilled else 1.0,
                type="float",
                tooltip="Spatio-Temporal Guidance (STG) scale for the video modality. STG moves the sample away from a perturbed version of the denoising model output, requiring one additional forward pass. 0.0 disables STG. For LTX-2.3 a value of 1.0 is suggested.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="modality_scale",
                default_value=1.0 if self._is_distilled else 3.0,
                type="float",
                tooltip="Modality isolation guidance scale for the video modality. Moves the sample away from a version generated without audio-to-video and video-to-audio cross attention, using a CFG-like estimate. Requires one additional forward pass. 1.0 disables modality guidance. For LTX-2.3 a value of 3.0 is suggested.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_rescale",
                default_value=0.0 if self._is_distilled else 0.7,
                type="float",
                tooltip="Guidance rescale factor for the video modality. Proposed by 'Common Diffusion Noise Schedules and Sample Steps are Flawed' to fix overexposure when using zero terminal SNR. 0.0 disables rescaling.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="audio_guidance_scale",
                default_value=1.0 if self._is_distilled else 7.0,
                type="float",
                tooltip="CFG guidance scale for the audio modality. The same CFG update rule applies as for video, but video and audio can use different values. For LTX-2.3 the authors suggest 7.0 for audio alongside 3.0 for video.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="audio_stg_scale",
                default_value=0.0 if self._is_distilled else 1.0,
                type="float",
                tooltip="Spatio-Temporal Guidance scale for the audio modality. Uses the same STG update rule as the video modality. For LTX-2.3 a value of 1.0 is suggested for both video and audio.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="audio_modality_scale",
                default_value=1.0 if self._is_distilled else 3.0,
                type="float",
                tooltip="Modality isolation guidance scale for the audio modality. Uses the same guidance rule as the video modality scale. For LTX-2.3 a value of 3.0 is suggested for both video and audio.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="audio_guidance_rescale",
                default_value=0.0 if self._is_distilled else 0.7,
                type="float",
                tooltip="Guidance rescale factor for the audio modality. Applies the same overexposure-correction logic as guidance_rescale. Defaults to the video guidance_rescale value when not set explicitly.",
            )
        )
        self._text_embeddings_path_param.add_input_parameters()
        if self._is_hdr_lora_active:
            self._node.show_parameter_by_name("text_embeddings_path")
        else:
            self._node.hide_parameter_by_name("text_embeddings_path")

    @property
    def _is_distilled(self) -> bool:
        build_data = self._pipeline_metadata.get("build_data", {})
        return build_data.get("is_distilled", False)

    @property
    def _is_hdr_lora_active(self) -> bool:
        token = self._HDR_LORA_ADAPTER_TOKEN
        for step_metadata in self._pipeline_metadata.get("runtime_adapter_steps", []):
            if step_metadata.get("kind") != LoraPipelineRuntimeAdapterStep.KIND:
                continue
            step_loras = step_metadata.get("loras", {})
            if any(token in str(entry.get("path", "")).lower() for entry in step_loras.values()):
                return True

        return False

    def add_output_parameters(self) -> None:
        pass

    def _remove_input_parameters(self) -> None:
        if self._is_distilled:
            self._node.remove_parameter_element_by_name("use_stage_2")
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._node.remove_parameter_element_by_name("stg_scale")
        self._node.remove_parameter_element_by_name("modality_scale")
        self._node.remove_parameter_element_by_name("guidance_rescale")
        self._node.remove_parameter_element_by_name("audio_guidance_scale")
        self._node.remove_parameter_element_by_name("audio_stg_scale")
        self._node.remove_parameter_element_by_name("audio_modality_scale")
        self._node.remove_parameter_element_by_name("audio_guidance_rescale")
        self._node.remove_parameter_element_by_name("text_embeddings_path")

    def remove_output_parameters(self) -> None:
        pass

    def publish_output_image_preview_placeholder(self) -> None:
        # Video pipelines don't use image placeholders
        pass

    def validate_before_node_run(self) -> list[Exception] | None:
        text_embeddings_path = self._node.get_parameter_value("text_embeddings_path")
        if text_embeddings_path and self._is_hdr_lora_active:
            try:
                self._text_embeddings_path_param.validate_parameter_values()
            except RuntimeError as err:
                return [err]
        return None

    def _get_pipe_kwargs(self) -> dict:
        pipe_kwargs = {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "audio_guidance_scale": self._node.get_parameter_value("audio_guidance_scale"),
            "stg_scale": self._node.get_parameter_value("stg_scale"),
            "audio_stg_scale": self._node.get_parameter_value("audio_stg_scale"),
            "modality_scale": self._node.get_parameter_value("modality_scale"),
            "audio_modality_scale": self._node.get_parameter_value("audio_modality_scale"),
            "guidance_rescale": self._node.get_parameter_value("guidance_rescale"),
            "audio_guidance_rescale": self._node.get_parameter_value("audio_guidance_rescale"),
        }
        text_embeddings_path = self._node.get_parameter_value("text_embeddings_path")
        if text_embeddings_path and self._is_hdr_lora_active:
            pipe_kwargs["text_embeddings_path"] = text_embeddings_path
        if self._is_distilled:
            pipe_kwargs["use_stage_2"] = bool(self._node.get_parameter_value("use_stage_2"))
        return pipe_kwargs
