from __future__ import annotations

import logging
from typing import Any

from modular_diffusion_nodes_library.utils.pipeline_utils import clear_diffusion_pipeline

logger = logging.getLogger("modular_diffusers_nodes_library")


class ModelCache:
    def __init__(self) -> None:
        self._pipeline_cache: dict[str, Any] = {}

    def from_pretrained(self, cls: Any, *args, **kwargs) -> Any:
        return cls.from_pretrained(*args, **kwargs)

    def has_pipeline(self, config_hash: str) -> bool:
        """Check if a pipeline with the given config hash exists in the cache."""
        return config_hash in self._pipeline_cache

    def get_pipeline(self, config_hash: str) -> Any | None:
        """Get cached pipeline by config hash."""
        return self._pipeline_cache.get(config_hash)

    def take_pipeline(self, config_hash: str) -> Any | None:
        """Get and remove a pipeline from the cache by config hash.
        It is up to the caller to clear the pipeline from memory after taking it from the cache."""
        return self._pipeline_cache.pop(config_hash, None)

    def add_pipeline(self, config_hash: str, pipeline: Any) -> None:
        """Add a pipeline to the cache with the given config hash."""
        self._pipeline_cache[config_hash] = pipeline

    def get_or_build_pipeline(self, config_hash: str, builder_func: Any) -> Any:
        """Get cached pipeline or build new one if not exists."""
        if config_hash not in self._pipeline_cache:
            logger.info("No cached pipeline found with config hash: %s", config_hash)
            # TODO: Support multiple pipelines via Resource Manager: https://github.com/griptape-ai/griptape-nodes/issues/2237
            self.clear_pipeline_cache()
            logger.info("Building new pipeline with config hash: %s", config_hash)
            self._pipeline_cache[config_hash] = builder_func()
        else:
            logger.info("Using cached pipeline with config hash: %s", config_hash)
        return self._pipeline_cache[config_hash]

    def remove_pipeline(self, config_hash: str) -> None:
        """Remove a specific pipeline from the cache."""
        if config_hash in self._pipeline_cache:
            logger.info("Removing pipeline from cache with config hash: %s", config_hash)
            pipe = self._pipeline_cache.pop(config_hash)
            clear_diffusion_pipeline(pipe)

    def clear_pipeline_cache(self) -> None:
        """Clear all cached pipelines."""
        logger.info("Clearing pipeline cache")
        for config_hash, pipe in self._pipeline_cache.items():
            logger.info("Clearing pipeline with config hash: %s", config_hash)
            clear_diffusion_pipeline(pipe)
        self._pipeline_cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get statistics about the pipeline cache."""
        return {
            "cached_pipelines": len(self._pipeline_cache),
            "cache_keys": list(self._pipeline_cache.keys()),
        }


model_cache = ModelCache()
