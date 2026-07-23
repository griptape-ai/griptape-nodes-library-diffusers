from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, override

import torch  # type: ignore[reportMissingImports]
from diffusers import GGUFQuantizationConfig  # type: ignore[reportMissingImports]
from diffusers.loaders.single_file_utils import (  # type: ignore[reportMissingImports]
    infer_diffusers_model_type,
    load_single_file_checkpoint,
)

from modular_diffusion_nodes_library.component_loading.component_slots import slot_component_kind
from modular_diffusion_nodes_library.component_loading.config_resolver import loadable_class_name, resolve_config_dir
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import (
    MODEL_TYPE_TO_PIPELINE_TYPE,
    get_component_class,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


def _pipeline_default_model_type(pipeline_cls: type) -> str | None:
    """Return a canonical ``model_type`` string for ``pipeline_cls``, or ``None``."""
    for model_type, pipeline_type in MODEL_TYPE_TO_PIPELINE_TYPE.items():
        if pipeline_type == pipeline_cls.__name__:
            return model_type
    return None


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
class ComponentArtifact(ABC):
    """Descriptor for a single pipeline component.

    Produced by component loader nodes, consumed by the pipeline builder
    at build time. Does not hold the loaded component itself.

    Subclasses specialize by component kind (model weights, tokenizer,
    text encoder, ...).
    """

    load_id: str
    source_type: ComponentSourceType
    component: str  # slot name, e.g. "transformer", "vae", "tokenizer"
    torch_dtype: str = "bfloat16"

    @property
    def is_quantized(self) -> bool:
        """True if the underlying weights use an embedded quantization format."""
        return False

    @abstractmethod
    def materialize(self, *, pipeline_cls: type, slot: str | None = None) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class ModelComponentArtifact(ComponentArtifact):
    """Descriptor for a model component (Transformer, UNet, VAE, Text Encoder) loaded via diffusers."""

    # HF_REPO
    repo_ref: HFRepoRef | None = None

    # SINGLE_FILE / LOCAL_DIR
    file_path: str | None = None
    config_source: str | None = None  # local path OR HF repo_id

    @property
    @override
    def is_quantized(self) -> bool:
        """True if the weights use an embedded quantization format (e.g. GGUF)."""
        return self.file_path is not None and self.file_path.lower().endswith(".gguf")

    @override
    def materialize(self, *, pipeline_cls: type, slot: str | None = None) -> Any:
        """Load this component from its descriptor.

        ``pipeline_cls`` is the diffusers pipeline class (e.g. ``FluxPipeline``)
        used to derive the concrete component class and to validate that the weights
        are compatible with the target pipeline.

        ``slot`` is the actual pipeline slot being filled (e.g. ``"text_encoder_2"``).
        When provided it takes precedence over ``self.component`` for class lookup,
        so an artifact created for one slot can be correctly loaded into another
        (e.g. a generic text-encoder artifact wired to the ``text_encoder_2`` port).
        """
        effective_slot = slot if slot is not None else self.component
        try:
            if self.source_type == ComponentSourceType.HF_REPO:
                return self._materialize_hf_repo(pipeline_cls=pipeline_cls, effective_slot=effective_slot)
            if self.source_type == ComponentSourceType.SINGLE_FILE:
                return self._materialize_single_file(pipeline_cls=pipeline_cls, effective_slot=effective_slot)
            if self.source_type == ComponentSourceType.LOCAL_DIR:
                return self._materialize_local_dir(pipeline_cls=pipeline_cls, effective_slot=effective_slot)

            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because source type '{self.source_type}' is not supported."
            )
            raise NotImplementedError(msg)
        except Exception as e:
            # Add component context to any materialization error
            component_cls = get_component_class(pipeline_cls, effective_slot)
            source_info = self._describe_source()
            msg = (
                f"Failed to load {self.component} as {component_cls.__name__} from {source_info}. Original error: {e!s}"
            )
            raise type(e)(msg) from e

    def _describe_source(self) -> str:
        """Return a human-readable description of this component's source."""
        if self.source_type == ComponentSourceType.HF_REPO and self.repo_ref:
            parts = [f"HF repo '{self.repo_ref.repo_id}'"]
            if self.repo_ref.subfolder:
                parts.append(f"subfolder '{self.repo_ref.subfolder}'")
            if self.repo_ref.revision:
                parts.append(f"revision '{self.repo_ref.revision}'")
            return ", ".join(parts)
        if self.source_type == ComponentSourceType.SINGLE_FILE and self.file_path:
            return f"single file '{self.file_path}'"
        if self.source_type == ComponentSourceType.LOCAL_DIR and self.file_path:
            return f"local directory '{self.file_path}'"
        return f"{self.source_type} (details unavailable)"

    def _materialize_hf_repo(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        if not self.repo_ref:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because repo_ref is required for HF_REPO source type."
            )
            raise ValueError(msg)

        component_cls = get_component_class(pipeline_cls, effective_slot)

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.repo_ref.repo_id,
            "local_files_only": True,
        }
        if self.repo_ref.revision:
            kwargs["revision"] = self.repo_ref.revision
        if self.repo_ref.subfolder:
            kwargs["subfolder"] = self.repo_ref.subfolder
        # Tokenizer classes do not accept torch_dtype in from_pretrained.
        if slot_component_kind(effective_slot) != "tokenizer":
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        logger.info(
            "Materializing %s (%s) from HF_REPO repo='%s' subfolder='%s' revision='%s'.",
            self.component,
            component_cls.__name__,
            self.repo_ref.repo_id,
            self.repo_ref.subfolder,
            self.repo_ref.revision,
        )
        return component_cls.from_pretrained(**kwargs)

    def _materialize_single_file(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        if not self.file_path:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because file_path is required for SINGLE_FILE source type."
            )
            raise ValueError(msg)

        component_cls = get_component_class(pipeline_cls, effective_slot)

        # Non-diffusers components (e.g. Qwen2_5_VLForConditionalGeneration) are absent from
        # SINGLE_FILE_LOADABLE_CLASSES and cannot use diffusers' from_single_file path.
        # Route them through from_pretrained with the gguf_file kwarg instead.
        if loadable_class_name(component_cls) is None:
            return self._materialize_transformers_from_gguf(component_cls)

        checkpoint = load_single_file_checkpoint(self.file_path)
        inferred_model_type = infer_diffusers_model_type(checkpoint)

        # If inferring model type fails to return a recognised type
        # fall back to the target pipeline's canonical model_type so the
        # config lookup uses the right bundled/cached config.
        if inferred_model_type in MODEL_TYPE_TO_PIPELINE_TYPE:
            model_type = inferred_model_type
        else:
            model_type = _pipeline_default_model_type(pipeline_cls) or inferred_model_type

        logger.info(
            "Inferred model_type='%s' (effective='%s') for %s (%s) from single file '%s'.",
            inferred_model_type,
            model_type,
            self.component,
            component_cls.__name__,
            self.file_path,
        )

        kwargs: dict[str, Any] = {
            "pretrained_model_link_or_path_or_dict": checkpoint,
            "torch_dtype": getattr(torch, self.torch_dtype),
            "local_files_only": True,
        }
        if self.is_quantized:
            kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=getattr(torch, self.torch_dtype))

        # Load with config if model_type is recognized OR user provided explicit config_source.
        if model_type in MODEL_TYPE_TO_PIPELINE_TYPE or self.config_source:
            expected_pipeline_type = MODEL_TYPE_TO_PIPELINE_TYPE.get(inferred_model_type)
            if expected_pipeline_type is not None and expected_pipeline_type != pipeline_cls.__name__:
                logger.warning(
                    "%s: checkpoint model_type='%s' maps to pipeline '%s' "
                    "but builder requested pipeline_type='%s'. Proceeding with builder's "
                    "pipeline_type — ensure weights are compatible.",
                    self.component,
                    inferred_model_type,
                    expected_pipeline_type,
                    pipeline_cls.__name__,
                )

            config_dir = resolve_config_dir(model_type, component_cls, self.config_source)
            component = component_cls.from_single_file(config=str(config_dir), **kwargs)
            kwargs.pop("pretrained_model_link_or_path_or_dict", None)
            del checkpoint
            return component

        # No explicit config provided and none bundled.
        msg = (
            f"Attempted to materialize {self.component}. "
            f"Failed with model_type='{model_type}' pipeline_cls='{pipeline_cls.__name__}' because "
            f"the checkpoint's model_type is unregistered and no bundled/cached config could be "
            f"located for the target pipeline. "
            f"Set 'Config Source' to a local config directory or HuggingFace repo_id for this component."
        )
        raise ValueError(msg)

    def _materialize_transformers_from_gguf(self, component_cls: type) -> Any:
        """Load a non-diffusers component (e.g. Qwen2_5_VLForConditionalGeneration) from a GGUF file.

        Called when the component class is not registered in SINGLE_FILE_LOADABLE_CLASSES.
        """
        file_path = Path(self.file_path)
        if not file_path.is_file():
            msg = (
                f"Attempted to materialize {self.component} as {component_cls.__name__}. "
                f"Failed with file_path='{self.file_path}' because it is not a file. "
                f"Provide the path to the GGUF file directly (e.g. /path/to/model-Q4_K_M.gguf); "
                f"its parent directory must contain a config.json."
            )
            raise FileNotFoundError(msg)

        if not self.is_quantized:
            msg = (
                f"Attempted to materialize {self.component} as {component_cls.__name__}. "
                f"Failed because {component_cls.__name__} is not a diffusers model and does not "
                f"support single-file loading for non-GGUF files. "
                f"Use a .gguf file, or switch to Local Folder or HuggingFace Repo source type."
            )
            raise ValueError(msg)

        kwargs: dict[str, Any] = {
            "gguf_file": file_path.name,
            "local_files_only": True,
            "torch_dtype": getattr(torch, self.torch_dtype),
        }
        return component_cls.from_pretrained(str(file_path.parent), **kwargs)

    def _materialize_local_dir(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        if not self.file_path:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because file_path is required for LOCAL_DIR source type."
            )
            raise ValueError(msg)

        folder = Path(self.file_path)
        if not folder.is_dir():
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed with file_path='{self.file_path}' because it is not an existing directory."
            )
            raise FileNotFoundError(msg)

        is_tokenizer = slot_component_kind(effective_slot) == "tokenizer"
        # Tokenizer folders use tokenizer_config.json; model component folders use config.json.
        if is_tokenizer:
            if not (folder / "tokenizer_config.json").is_file():
                msg = (
                    f"Attempted to materialize {self.component}. "
                    f"Failed with file_path='{self.file_path}' because it does not contain a "
                    f"'tokenizer_config.json'. Pick a diffusers-format tokenizer folder "
                    f"(e.g. '.../FLUX.1-dev/tokenizer/')."
                )
                raise FileNotFoundError(msg)
        else:
            if not (folder / "config.json").is_file():
                msg = (
                    f"Attempted to materialize {self.component}. "
                    f"Failed with file_path='{self.file_path}' because it does not contain a 'config.json'. "
                    f"Pick a diffusers-format component folder (e.g. '.../FLUX.1-dev/transformer/')."
                )
                raise FileNotFoundError(msg)

        component_cls = get_component_class(pipeline_cls, effective_slot)

        logger.info(
            "Materializing %s (%s) from LOCAL_DIR path='%s'.",
            self.component,
            component_cls.__name__,
            self.file_path,
        )
        kwargs: dict[str, Any] = {"local_files_only": True}
        # Tokenizer classes do not accept torch_dtype in from_pretrained.
        if not is_tokenizer:
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        return component_cls.from_pretrained(self.file_path, **kwargs)
