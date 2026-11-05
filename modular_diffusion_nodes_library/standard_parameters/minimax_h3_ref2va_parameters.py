from __future__ import annotations

from typing import ClassVar

from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters


class MiniMaxH3Ref2VAPipelineParameters(MiniMaxH3PipelineParameters):
    _workflow: ClassVar[str] = "ref2va"

    @classmethod
    def pipeline_name(cls) -> str:
        return "MiniMaxH3Ref2VAModularPipeline"
