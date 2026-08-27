from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.component_loading.config_resolver import (
    is_hf_config_cached,
    resolve_hf_repo_config_subfolder,
)
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

logger = logging.getLogger("modular_diffusers_nodes_library")


@dataclass(frozen=True)
class TokenizerComponentArtifact(ModelComponentArtifact):
    """Descriptor for a tokenizer component loaded via diffusers."""

    def __post_init__(self) -> None:
        if self.source_type == ComponentSourceType.SINGLE_FILE:
            msg = (
                f"{self.component}: single-file loading is not supported for tokenizer components. "
                f"Use Local Folder or HuggingFace Repo source type."
            )
            raise ValueError(msg)

    @override
    def _materialize_hf_repo(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        if not self.repo_ref:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because repo_ref is required for HF_REPO source type."
            )
            raise ValueError(msg)

        component_cls = get_component_class(pipeline_cls, effective_slot)

        subfolder = self.repo_ref.subfolder or ""
        if not subfolder:
            resolved_subfolder = resolve_hf_repo_config_subfolder(
                self.repo_ref.repo_id,
                effective_slot,
                self.component,
                revision=self.repo_ref.revision,
                config_filename="tokenizer_config.json",
            )
            if resolved_subfolder is None:
                msg = (
                    f"Attempted to materialize {self.component}. "
                    f"Failed because no 'tokenizer_config.json' was found in the HuggingFace cache "
                    f"for repo='{self.repo_ref.repo_id}'. "
                    f"Check the repo name or set an explicit subfolder."
                )
                raise FileNotFoundError(msg)
            subfolder = resolved_subfolder
        elif not is_hf_config_cached(self.repo_ref.repo_id, subfolder, "tokenizer_config.json", self.repo_ref.revision):
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because subfolder='{subfolder}' in repo='{self.repo_ref.repo_id}' "
                f"does not contain a cached 'tokenizer_config.json'. "
                f"Check the subfolder name or leave it blank for auto-detection."
            )
            raise FileNotFoundError(msg)

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.repo_ref.repo_id,
            "local_files_only": True,
        }
        if self.repo_ref.revision:
            kwargs["revision"] = self.repo_ref.revision
        if subfolder:
            kwargs["subfolder"] = subfolder

        logger.info(
            "Materializing %s (%s) from HF_REPO repo='%s' subfolder='%s' revision='%s'.",
            self.component,
            component_cls.__name__,
            self.repo_ref.repo_id,
            subfolder,
            self.repo_ref.revision,
        )
        return component_cls.from_pretrained(**kwargs)

    @override
    def _materialize_local_dir(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        if not self.file_path:
            msg = (
                f"Attempted to materialize {self.component}. Failed because path is required for LOCAL_DIR source type."
            )
            raise ValueError(msg)

        folder = Path(self.file_path)
        if not folder.is_dir():
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed with path='{self.file_path}' because it is not an existing directory."
            )
            raise FileNotFoundError(msg)

        if not (folder / "tokenizer_config.json").is_file():
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed with path='{self.file_path}' because it does not contain a "
                f"'tokenizer_config.json'. Pick a diffusers-format tokenizer folder "
                f"(e.g. '.../FLUX.1-dev/tokenizer/')."
            )
            raise FileNotFoundError(msg)

        component_cls = get_component_class(pipeline_cls, effective_slot)

        logger.info(
            "Materializing %s (%s) from LOCAL_DIR path='%s'.",
            self.component,
            component_cls.__name__,
            self.file_path,
        )
        return component_cls.from_pretrained(self.file_path, local_files_only=True)
