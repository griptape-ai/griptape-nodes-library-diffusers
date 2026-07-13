from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch  # type: ignore[reportMissingImports]
from diffusers import GGUFQuantizationConfig  # type: ignore[reportMissingImports]
from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES  # type: ignore[reportMissingImports]
from diffusers.loaders.single_file_utils import (  # type: ignore[reportMissingImports]
    infer_diffusers_model_type,
    load_single_file_checkpoint,
)

from modular_diffusion_nodes_library.component_loading.config_resolver import resolve_config_dir
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import (
    MODEL_TYPE_TO_PIPELINE_TYPE,
    get_component_class,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


@dataclass(frozen=True)
class HFRepoRef:
    repo_id: str
    revision: str | None = None
    subfolder: str | None = None


class ComponentSourceType(StrEnum):
    HF_REPO = "hf_repo"
    SINGLE_FILE = "single_file"
    LOCAL_DIR = "local_dir"


@dataclass(frozen=True)
class ComponentArtifact:
    """Lazy descriptor for a single pipeline component.

    Produced by component loader nodes, consumed by the pipeline builder
    at build time. Does not hold the loaded component itself.
    """

    load_id: str

    source_type: ComponentSourceType
    component: str  # slot name, e.g. "transformer", "vae"
    torch_dtype: str = "bfloat16"

    # HF_REPO
    repo_ref: HFRepoRef | None = None

    # SINGLE_FILE / LOCAL_DIR
    file_path: str | None = None
    config_source: str | None = None  # local path OR HF repo_id

    @property
    def is_quantized(self) -> bool:
        """True if the weights use an embedded quantization format (e.g. GGUF)."""
        return self.file_path is not None and self.file_path.lower().endswith(".gguf")

    @property
    def is_quantized(self) -> bool:
        """True if the weights use an embedded quantization format (e.g. GGUF)."""
        return self.file_path is not None and self.file_path.lower().endswith(".gguf")

    def materialize(self, *, pipeline_cls: type) -> Any:
        """Load this component from its descriptor.

        ``pipeline_cls`` is the diffusers pipeline class (e.g. ``FluxPipeline``)
        used to derive the concrete component class and to validate that the weights
        are compatible with the target pipeline.
        """
        if self.source_type == ComponentSourceType.HF_REPO:
            return self._materialize_hf_repo(pipeline_cls=pipeline_cls)
        if self.source_type == ComponentSourceType.SINGLE_FILE:
            return self._materialize_single_file(pipeline_cls=pipeline_cls)
        if self.source_type == ComponentSourceType.LOCAL_DIR:
            return self._materialize_local_dir(pipeline_cls=pipeline_cls)

        msg = (
            f"Attempted to materialize component '{self.load_id}'. "
            f"Failed because source type '{self.source_type}' is not supported."
        )
        raise NotImplementedError(msg)

    def _materialize_hf_repo(self, *, pipeline_cls: type) -> Any:
        if not self.repo_ref:
            msg = (
                f"Attempted to materialize component '{self.load_id}'. "
                f"Failed because repo_ref is required for HF_REPO source type."
            )
            raise ValueError(msg)

        component_cls = get_component_class(pipeline_cls, self.component)

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.repo_ref.repo_id,
            "local_files_only": True,
        }
        if self.repo_ref.revision is not None:
            kwargs["revision"] = self.repo_ref.revision
        if self.repo_ref.subfolder is not None:
            kwargs["subfolder"] = self.repo_ref.subfolder
        kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        return component_cls.from_pretrained(**kwargs)

    def _materialize_single_file(self, *, pipeline_cls: type) -> Any:
        if not self.file_path:
            msg = (
                f"Attempted to materialize component '{self.load_id}'. "
                f"Failed because file_path is required for SINGLE_FILE source type."
            )
            raise ValueError(msg)

        checkpoint = load_single_file_checkpoint(self.file_path)
        try:
            model_type = infer_diffusers_model_type(checkpoint)
        finally:
            del checkpoint

        logger.info(
            "Inferred model_type='%s' for component '%s' at materialize time.",
            model_type,
            self.load_id,
        )

        component_cls = get_component_class(pipeline_cls, self.component)

        # Load with config if model_type is recognized OR user provided explicit config_source.
        if model_type in MODEL_TYPE_TO_PIPELINE_TYPE or self.config_source:
            # Validate pipeline compatibility for recognized model types.
            if model_type in MODEL_TYPE_TO_PIPELINE_TYPE:
                expected_pipeline_type = MODEL_TYPE_TO_PIPELINE_TYPE[model_type]
                if expected_pipeline_type != pipeline_cls.__name__:
                    logger.warning(
                        "Component '%s': checkpoint model_type='%s' maps to pipeline '%s' "
                        "but builder requested pipeline_type='%s'. Proceeding with builder's "
                        "pipeline_type — ensure weights are compatible.",
                        self.load_id,
                        model_type,
                        expected_pipeline_type,
                        pipeline_cls.__name__,
                    )

            config_dir = resolve_config_dir(model_type, component_cls, self.config_source)
            kwargs: dict[str, Any] = {
                "pretrained_model_link_or_path_or_dict": self.file_path,
                "config": str(config_dir),
                "torch_dtype": getattr(torch, self.torch_dtype),
                "local_files_only": True,
            }
            if self.is_quantized:
                kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=getattr(torch, self.torch_dtype))
            return component_cls.from_single_file(**kwargs)

        # No fallback available — explicit config required.
        msg = (
            f"Attempted to materialize component '{self.load_id}'. "
            f"Failed with model_type='{model_type}' because it is not registered and "
            f"'{component_cls.__name__}' cannot reconstruct its config from checkpoint keys alone. "
            f"Set 'Config Source' to a local config directory or HuggingFace repo_id for this component."
        )
        raise ValueError(msg)

    def _materialize_local_dir(self, *, pipeline_cls: type) -> Any:  # noqa: ARG002
        msg = (
            f"Attempted to materialize component '{self.load_id}'. "
            f"Failed because LOCAL_DIR source type is not yet implemented."
        )
        raise NotImplementedError(msg)
