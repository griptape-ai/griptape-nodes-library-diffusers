from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch  # type: ignore[reportMissingImports]
from diffusers.models.autoencoders import LTX2VideoDiffusionDecoderModel  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import LTX2VideoDiffusionDecodePipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from PIL import Image

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

LTX25_DIFFUSION_DECODER_REPO = "Lightricks/LTX-2.5-Diffusers"
DIFFUSION_DECODER_SUBFOLDER = "diffusion_decoder"


@dataclass(frozen=True)
class DiffusionDecoderSpec:
    media_type: Literal["image", "video"]
    fps: int | None = None


DecoderOutput = list[Image.Image] | list[list[Image.Image]] | np.ndarray


class BaseDiffusionDecoderParameters:
    decoder_type = ""
    spec: DiffusionDecoderSpec
    _DYNAMIC_PARAMETERS = {
        "output_image",
        "output_video",
        "raw_media",
        "fps",
        "enable_raw_output",
    }

    def __init__(self, node: BaseNode) -> None:
        self._node = node

    def add_parameters(self) -> None:
        raise NotImplementedError

    def remove_parameters(self) -> None:
        raise NotImplementedError

    def validate_before_node_run(self) -> list[Exception] | None:
        raise NotImplementedError

    def decode(self, latent: LatentArtifact, seed: int, output_type: Literal["pil", "np"]) -> DecoderOutput:
        """Decode to PIL frames or pipeline-postprocessed floating-point RGB media."""
        raise NotImplementedError


class LTX25DiffusionDecoderParameters(BaseDiffusionDecoderParameters):
    decoder_type = "LTX-2.5"
    spec = DiffusionDecoderSpec(media_type="video", fps=24)
    _DYNAMIC_PARAMETERS = BaseDiffusionDecoderParameters._DYNAMIC_PARAMETERS | {
        "diffusion_decoder_model",
        "diffusion_decoder_model_download",
        "decoder_num_inference_steps",
        "decoder_use_tiling",
    }

    def __init__(self, node: BaseNode) -> None:
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[LTX25_DIFFUSION_DECODER_REPO],
            parameter_name="diffusion_decoder_model",
        )

    def add_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="decoder_num_inference_steps",
                default_value=1,
                type="int",
                tooltip="Number of denoising steps used by the diffusion decoder. The LTX-2.5 decoder defaults to one step.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self._node.add_parameter(
            Parameter(
                name="decoder_use_tiling",
                default_value=False,
                type="bool",
                tooltip="Decode large videos in overlapping tiles to reduce peak memory use.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )

    def remove_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._node.remove_parameter_element_by_name("decoder_num_inference_steps")
        self._node.remove_parameter_element_by_name("decoder_use_tiling")

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = self._model_repo_parameter.validate_before_node_run() or []
        steps = self._node.get_parameter_value("decoder_num_inference_steps")
        try:
            steps_int = int(steps)
            if steps_int <= 0:
                errors.append(ValueError(f"Decoder inference steps must be positive, got {steps_int}."))
        except (TypeError, ValueError):
            errors.append(ValueError(f"Decoder inference steps must be an integer, got {steps!r}."))
        return errors or None

    def decode(self, latent: LatentArtifact, seed: int, output_type: Literal["pil", "np"]) -> DecoderOutput:
        """Return PIL frames or pipeline-postprocessed floating-point RGB media."""
        errors = self.validate_before_node_run()
        if errors:
            raise errors[0]
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        device = get_best_device()
        decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
            repo_id,
            subfolder=DIFFUSION_DECODER_SUBFOLDER,
            revision=revision,
            torch_dtype=torch.bfloat16,
        )
        if self._node.get_parameter_value("decoder_use_tiling"):
            decoder.enable_tiling()

        try:
            decoder.decoder.default_num_inference_steps = int(
                self._node.get_parameter_value("decoder_num_inference_steps")
            )
            pipeline = LTX2VideoDiffusionDecodePipeline(
                diffusion_decoder=decoder,
                scheduler=FlowMatchEulerDiscreteScheduler(),
            )
            pipeline.to(device=device)
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=device).manual_seed(int(seed))
            latents = latent.to_torch(device=device, dtype=torch.bfloat16)
            pipeline_output = pipeline(
                latents=latents,
                generator=generator,
                output_type=output_type,
                return_dict=False,
                denormalize=True,
            )[0]
            if output_type == "np":
                if isinstance(pipeline_output, torch.Tensor):
                    result = pipeline_output.float().cpu().numpy()
                else:
                    result = np.asarray(pipeline_output, dtype=np.float32)
            else:
                result = pipeline_output
        finally:
            decoder.to("cpu")
            cleanup_memory_caches()
        return result


DIFFUSION_DECODER_TYPE_MAP: dict[str, type[BaseDiffusionDecoderParameters]] = {
    LTX25DiffusionDecoderParameters.decoder_type: LTX25DiffusionDecoderParameters,
}


def create_diffusion_decoder_parameters(decoder_type: str, node: BaseNode) -> BaseDiffusionDecoderParameters:
    decoder_cls = DIFFUSION_DECODER_TYPE_MAP.get(decoder_type)
    if decoder_cls is None:
        msg = (
            f"Attempted to create diffusion decoder parameters. Failed with decoder_type='{decoder_type}' "
            f"because it is not a known decoder type. Known types: {list(DIFFUSION_DECODER_TYPE_MAP.keys())}"
        )
        raise ValueError(msg)
    return decoder_cls(node)
