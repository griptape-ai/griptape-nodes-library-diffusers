import logging

from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode

from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache

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
            stats = model_cache.get_cache_stats()
            pipeline_count = int(stats.get("cached_pipelines", 0))
            model_cache.clear_pipeline_cache()
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
