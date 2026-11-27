from __future__ import annotations

from typing import ClassVar

from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters


class MiniMaxH3Ref2VAPipelineParameters(MiniMaxH3PipelineParameters):
    _workflow: ClassVar[str] = "ref2va"
    # ref2va loads `transformer_ref`; `transformer` is the fl2va/t2va partition.
    _unloaded_component_slots: ClassVar[frozenset[str]] = frozenset({"transformer"})

    @classmethod
    def pipeline_name(cls) -> str:
        return "MiniMaxH3Ref2VAModularPipeline"
