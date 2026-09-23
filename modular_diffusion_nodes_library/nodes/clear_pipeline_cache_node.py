import logging

from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode

logger = logging.getLogger("modular_diffusers_nodes_library")


class ClearPipelineCacheNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._create_status_parameters(
            result_details_tooltip="Details about the cache clearing operation.",
            result_details_placeholder="Cache clearing details appear here after execution.",
        )

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        self._clear_execution_status()
        try:
            # Scoped to this library rather than the whole worker: the object store is shared with
            # whatever else is hosted here, and emptying a co-tenant's objects is not this node's business.
            pipeline_count = self.local_objects.drop_all()
            self._set_status_results(
                was_successful=True,
                result_details=f"Cleared {pipeline_count} pipeline(s) from cache.",
            )
        except Exception as e:
            logger.exception("%s: Failed to clear pipeline cache", self.name)
            self._set_status_results(
                was_successful=False,
                result_details=str(e),
            )
            self._handle_failure_exception(e)
