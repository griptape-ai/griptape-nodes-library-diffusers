from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import torch  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ModelComponentArtifact
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class


@dataclass(frozen=True)
class TextEncoderComponentArtifact(ModelComponentArtifact):
    """Descriptor for a text encoder component loaded via diffusers."""

    @override
    def _materialize_single_file(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        if not self.file_path:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because file_path is required for SINGLE_FILE source type."
            )
            raise ValueError(msg)
        component_cls = get_component_class(pipeline_cls, effective_slot)

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
