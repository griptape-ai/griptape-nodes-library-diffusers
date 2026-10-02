import logging

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value

logger = logging.getLogger("modular_diffusers_nodes_library")


class ReleasePipelineNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip=(
                    "🤗 Diffusion pipeline to release from memory. The configuration is left intact, so a "
                    "node that needs this pipeline again rebuilds it."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self._create_status_parameters(
            result_details_tooltip="Whether the pipeline was being held, and what was released.",
            result_details_placeholder="Release details appear here after execution.",
        )

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        self._clear_execution_status()
        try:
            artifact = normalize_diffusion_pipeline_value(
                self.get_parameter_value("pipeline"), node_name=self.name, raise_on_invalid=True
            )
            if artifact is None:
                msg = f"{self.name}: No pipeline connected, so there is nothing to release."
                raise ValueError(msg)
            if not artifact.config_hash:
                msg = (
                    f"{self.name}: The connected pipeline has no configuration hash, which means it was "
                    f"never built. There is nothing to release."
                )
                raise ValueError(msg)

            released = self.local_objects.drop(self.local_objects.key_for(artifact.config_hash))
        except Exception as e:
            logger.exception("%s: Failed to release pipeline", self.name)
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        # A pipeline that is already gone is the state this node was asked to produce, so a miss is a
        # success. That keeps the node usable in a graph that runs more than once.
        if released:
            details = f"Released '{artifact.pipeline_name}' and freed the memory it held."
        else:
            details = f"'{artifact.pipeline_name}' was not being held. Nothing to release."
        self._set_status_results(was_successful=True, result_details=details)
