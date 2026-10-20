import diffusers  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.standard_parameters.flux_parameters import FluxPipelineParameters


class FluxFillPipelineParameters(FluxPipelineParameters):
    """FluxFillPipeline variant — only differs in model repo and pipeline class."""

    _pipeline_cls = diffusers.FluxFillPipeline  # type: ignore[reportAttributeAccessIssue]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node, list_all_models=list_all_models)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "black-forest-labs/FLUX.1-Fill-dev",
            ],
            parameter_name="model",
            list_all_models=list_all_models,
        )
