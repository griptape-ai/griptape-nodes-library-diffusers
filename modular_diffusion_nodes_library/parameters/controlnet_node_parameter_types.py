import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.traits.options import Options
from PIL.Image import Image

logger = logging.getLogger("modular_diffusers_nodes_library")
CONTROLNET_MODEL_PARAMETER_NAME = "controlnet_model"


# Base class for all ControlNet node parameters.
class ControlNetNodesParameterType:
    def __init__(self, node: BaseNode):
        self._node = node
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=self.model_repo_ids,
            parameter_name=CONTROLNET_MODEL_PARAMETER_NAME,
            list_all_models=False,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    @property
    def model_repo_ids(self) -> list[str]:
        raise NotImplementedError(
            "Subclasses must implement model_repo_ids property to specify valid model repositories for the control net."
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        return self._model_repo_parameter.validate_before_node_run()

    def get_control_image(self) -> Image | None:
        return self._node.get_parameter_value("control_image")

    def get_kwargs(self) -> dict[str, Any]:
        kwargs = {}
        kwargs["control_image"] = self.get_control_image()
        kwargs["controlnet_conditioning_scale"] = float(self._node.get_parameter_value("controlnet_conditioning_scale"))
        return kwargs


class FluxControlNetNodesParameterType(ControlNetNodesParameterType):
    CONTROL_MODES: ClassVar = {
        "canny": 0,
        "tile": 1,
        "depth": 2,
        "blur": 3,
        "pose": 4,
        "gray": 5,
        "low_quality": 6,
    }

    @property
    def model_repo_ids(self) -> list[str]:
        repo_ids = [
            "InstantX/FLUX.1-dev-Controlnet-Union",
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
            "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        ]
        return repo_ids

    def add_input_parameters(self) -> None:
        super().add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="control_mode",
                default_value=next(iter(self.CONTROL_MODES.keys())),
                input_types=["str"],
                type="str",
                traits={
                    Options(
                        choices=list(self.CONTROL_MODES.keys()),
                    )
                },
                tooltip="Conditioning type expected by the FLUX Union ControlNet (must match what the control image represents).",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="control_guidance_start",
                default_value=0.0,
                input_types=["float"],
                type="float",
                tooltip="Fraction of denoising steps (0.0–1.0) at which ControlNet conditioning begins to apply.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="control_guidance_end",
                default_value=1.0,
                input_types=["float"],
                type="float",
                tooltip="Fraction of denoising steps (0.0–1.0) at which ControlNet conditioning stops applying.",
            )
        )

    def remove_input_parameters(self) -> None:
        super().remove_input_parameters()
        self._node.remove_parameter_element_by_name("control_mode")
        self._node.remove_parameter_element_by_name("control_guidance_start")
        self._node.remove_parameter_element_by_name("control_guidance_end")

    def get_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_kwargs()
        control_mode = self._node.get_parameter_value("control_mode")
        if control_mode in self.CONTROL_MODES:
            kwargs["control_mode"] = int(self.CONTROL_MODES[control_mode])
        kwargs["control_guidance_start"] = float(self._node.get_parameter_value("control_guidance_start"))
        kwargs["control_guidance_end"] = float(self._node.get_parameter_value("control_guidance_end"))
        return kwargs


class QwenImageControlNetNodesParameterType(ControlNetNodesParameterType):
    @property
    def model_repo_ids(self) -> list[str]:
        repo_ids = [
            "InstantX/Qwen-Image-ControlNet-Union",
        ]
        return repo_ids

    def add_input_parameters(self) -> None:
        super().add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="control_guidance_start",
                default_value=0.0,
                input_types=["float"],
                type="float",
                tooltip="Fraction of denoising steps (0.0–1.0) at which ControlNet conditioning begins to apply.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="control_guidance_end",
                default_value=1.0,
                input_types=["float"],
                type="float",
                tooltip="Fraction of denoising steps (0.0–1.0) at which ControlNet conditioning stops applying.",
            )
        )

    def remove_input_parameters(self) -> None:
        super().remove_input_parameters()
        self._node.remove_parameter_element_by_name("control_guidance_start")
        self._node.remove_parameter_element_by_name("control_guidance_end")

    def get_kwargs(self) -> dict[str, Any]:
        kwargs = super().get_kwargs()
        kwargs["control_guidance_start"] = float(self._node.get_parameter_value("control_guidance_start"))
        kwargs["control_guidance_end"] = float(self._node.get_parameter_value("control_guidance_end"))
        return kwargs


class StableDiffusionControlNetNodesParameterType(ControlNetNodesParameterType):
    @property
    def model_repo_ids(self) -> list[str]:
        repo_ids = [
            "xinsir/controlnet-union-sdxl-1.0",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "xinsir/controlnet-scribble-sdxl-1.0",
            "xinsir/controlnet-depth-sdxl-1.0",
            "xinsir/controlnet-canny-sdxl-1.0",
            "xinsir/controlnet-tile-sdxl-1.0",
        ]
        return repo_ids

    def add_input_parameters(self) -> None:
        super().add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="control_guidance_start",
                default_value=0.0,
                input_types=["float"],
                type="float",
                tooltip="Fraction of denoising steps (0.0–1.0) at which ControlNet conditioning begins to apply.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="control_guidance_end",
                default_value=1.0,
                input_types=["float"],
                type="float",
                tooltip="Fraction of denoising steps (0.0–1.0) at which ControlNet conditioning stops applying.",
            )
        )

    def remove_input_parameters(self) -> None:
        super().remove_input_parameters()
        self._node.remove_parameter_element_by_name("control_guidance_start")
        self._node.remove_parameter_element_by_name("control_guidance_end")

    def get_kwargs(self) -> dict[str, Any]:
        kwargs = {}
        kwargs["control_image"] = self.get_control_image()
        kwargs["controlnet_conditioning_scale"] = float(self._node.get_parameter_value("controlnet_conditioning_scale"))
        kwargs["control_guidance_start"] = float(self._node.get_parameter_value("control_guidance_start"))
        kwargs["control_guidance_end"] = float(self._node.get_parameter_value("control_guidance_end"))
        return kwargs


class ZImageControlNetNodesParameterType(ControlNetNodesParameterType):
    REPO_FILES: ClassVar = [
        ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union", "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
        ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1", "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"),
    ]

    @property
    def model_repo_ids(self) -> list[str]:
        return [repo for repo, _ in self.REPO_FILES]

    def __init__(self, node: BaseNode):
        self._node = node
        self._model_repo_parameter = HuggingFaceRepoFileParameter(
            node,
            repo_files=self.REPO_FILES,
            parameter_name=CONTROLNET_MODEL_PARAMETER_NAME,
        )
