from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.parameters.huggingface_pipeline_parameter import HuggingFacePipelineParameter
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.parameters.providers import Provider
from modular_diffusion_nodes_library.standard_parameters.flux2_klein_parameters import (
    Flux2KleinPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux2_parameters import (
    Flux2PipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux_fill_parameters import (
    FluxFillPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.flux_parameters import (
    FluxPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.ltx_parameters import (
    LTXPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.qwen_edit_parameters import (
    QwenEditPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.qwen_parameters import (
    QwenPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.stable_diffusion_sdxl_parameters import (
    StableDiffusionXLPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.wan_i2v_parameters import (
    WanImageToVideoPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.wan_parameters import (
    WanPipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.z_image_parameters import (
    ZImagePipelineParameters,
)

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )
    from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
        ModularDiffusionPipelineTypePipelineParameters,
    )

logger = logging.getLogger("modular_diffusers_nodes_library")

# This code was copied from diffusers_nodes_library/common/parameters/diffusion/diffusion_pipeline_type_parameters.py.


class LatentPipelineTypeParameters(ABC):
    START_PARAMS: ClassVar = ["pipeline", "provider", "pipeline_type"]
    END_PARAMS: ClassVar = ["loras", "logs"]

    def __init__(self, node: LatentDiffusionPipelineBuilderNode):
        self._node = node
        self.did_pipeline_type_change = False
        self._pipeline_type_pipeline_params: ModularDiffusionPipelineTypePipelineParameters
        self.set_pipeline_type_pipeline_params(self.pipeline_types[0])

    @classmethod
    @abstractmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        raise NotImplementedError

    @property
    def pipeline_type_dict(self) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return self.__class__.get_pipeline_type_dict()

    @property
    def pipeline_types(self) -> list[str]:
        return list(self.pipeline_type_dict.keys())

    def set_pipeline_type_pipeline_params(self, pipeline_type: str) -> None:
        try:
            self._pipeline_type_pipeline_params = self.pipeline_type_dict[pipeline_type](self._node)
        except KeyError as e:
            msg = f"Unsupported pipeline type: {pipeline_type}"
            logger.error(msg)
            raise ValueError(msg) from e

    @property
    def pipeline_type_pipeline_params(self) -> ModularDiffusionPipelineTypePipelineParameters:
        if self._pipeline_type_pipeline_params is None:
            msg = "Pipeline type builder parameters not initialized. Ensure provider parameter is set."
            logger.error(msg)
            raise ValueError(msg)
        return self._pipeline_type_pipeline_params

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="pipeline_type",
                type="str",
                traits={Options(choices=self.pipeline_types)},
                tooltip="Specific pipeline variant within the selected provider (e.g. base, Fill, Edit). Determines which checkpoints and runtime parameters are exposed.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("pipeline_type")
        self.pipeline_type_pipeline_params.remove_input_parameters()

    def before_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "pipeline_type":
            current_pipeline_type = self._node.get_parameter_value("pipeline_type")
            self.did_pipeline_type_change = current_pipeline_type != value

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "pipeline_type" and self.did_pipeline_type_change:
            self.regenerate_elements_for_pipeline_type(value)

    def regenerate_elements_for_pipeline_type(self, pipeline_type: str) -> None:
        self._node.save_parameter_properties()

        self.pipeline_type_pipeline_params.remove_input_parameters()
        self.set_pipeline_type_pipeline_params(pipeline_type)
        self.pipeline_type_pipeline_params.add_input_parameters()

        # Get all current element names
        all_element_names = [element.name for element in self._node.root_ui_element.children]

        # Build parameter groupings
        hf_param_names = HuggingFacePipelineParameter.get_hf_pipeline_parameter_names()
        start_params = LatentPipelineTypeParameters.START_PARAMS
        end_params = [*hf_param_names, *LatentPipelineTypeParameters.END_PARAMS]
        excluded_params = {*start_params, *end_params}

        # Assemble final order: start -> middle -> end
        middle_params = [name for name in all_element_names if name not in excluded_params]
        sorted_parameters = [*start_params, *middle_params, *end_params]

        self._node.reorder_elements(sorted_parameters)

        self._node.clear_parameter_cache()

    def get_config_kwargs(self) -> dict:
        return self.pipeline_type_pipeline_params.get_config_kwargs()


class LatentFluxPipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "FluxPipeline": FluxPipelineParameters,
            "FluxFillPipeline": FluxFillPipelineParameters,
        }


class LatentFlux2PipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "Flux2Pipeline": Flux2PipelineParameters,
            "Flux2KleinPipeline": Flux2KleinPipelineParameters,
        }


class LatentQwenPipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "QwenImagePipeline": QwenPipelineParameters,
            "QwenImageEditPipeline": QwenEditPipelineParameters,
        }


class LatentStableDiffusionPipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "StableDiffusionXLPipeline": StableDiffusionXLPipelineParameters,
        }


class LatentLTXPipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "LTXPipeline": LTXPipelineParameters,
        }


class LatentWanPipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "WanPipeline": WanPipelineParameters,
            "WanImageToVideoPipeline": WanImageToVideoPipelineParameters,
        }


class LatentZImagePipelineTypeParameters(LatentPipelineTypeParameters):
    @classmethod
    def get_pipeline_type_dict(cls) -> dict[str, type[ModularDiffusionPipelineTypePipelineParameters]]:
        return {
            "ZImagePipeline": ZImagePipelineParameters,
        }


MODULAR_PIPELINE_TYPE_PROVIDER_MAP: dict[Provider, type[LatentPipelineTypeParameters]] = {
    Provider.FLUX: LatentFluxPipelineTypeParameters,
    Provider.FLUX2: LatentFlux2PipelineTypeParameters,
    Provider.LTX: LatentLTXPipelineTypeParameters,
    Provider.QWEN: LatentQwenPipelineTypeParameters,
    Provider.STABLE_DIFFUSION: LatentStableDiffusionPipelineTypeParameters,
    Provider.WAN: LatentWanPipelineTypeParameters,
    Provider.Z_IMAGE: LatentZImagePipelineTypeParameters,
}


def find_provider_for_pipeline_type(pipeline_type: str) -> str | None:
    for provider, params_cls in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.items():
        if pipeline_type in params_cls.get_pipeline_type_dict():
            return provider
    return None
