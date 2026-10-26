from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ModelComponentArtifact
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class


@dataclass(frozen=True)
class TextEncoderComponentArtifact(ModelComponentArtifact):
    """Descriptor for a text encoder component loaded via diffusers."""

    @override
    def _materialize_single_file(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        import torch  # type: ignore[reportMissingImports]

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
                f"its parent directory must contain a config.json unless config_source is provided."
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
        config = None
        if self.config_source:
            from transformers import AutoConfig

            try:
                config = AutoConfig.from_pretrained(self.config_source, local_files_only=True)
            except (OSError, ValueError) as config_exc:
                msg = (
                    f"Attempted to materialize {self.component} as {component_cls.__name__} from '{file_path}'. "
                    f"Failed because config source '{self.config_source}' could not be loaded."
                )
                raise ValueError(msg) from config_exc
            # With gguf_file set, Transformers treats a string config path as a GGUF checkpoint.
            kwargs["config"] = config
        try:
            return component_cls.from_pretrained(str(file_path.parent), **kwargs)
        except (ValueError, OSError, RuntimeError) as exc:
            if "not supported yet" not in str(exc):
                raise
            return self._materialize_gguf_hf_keys(component_cls, file_path, exc, config)

    def _materialize_gguf_hf_keys(
        self,
        component_cls: type,
        file_path: Path,
        official_exc: Exception,
        config: Any | None = None,
    ) -> Any:
        """Load a city96-style GGUF whose tensors still use Hugging Face key names.

        transformers `from_pretrained(..., gguf_file=)` only accepts llama.cpp
        architectures. A converted CLIP still has `text_model.*` / `encoder.*` keys.
        """
        import torch  # type: ignore[reportMissingImports]
        from diffusers.models.model_loading_utils import load_gguf_checkpoint
        from transformers import AutoConfig

        from modular_diffusion_nodes_library.artifact_utils.packed_quant_io import align_state_dict_to_module_keys

        if config is None:
            config_source = str(file_path.parent)
            try:
                config = AutoConfig.from_pretrained(config_source, local_files_only=True)
            except (OSError, ValueError) as config_exc:
                msg = (
                    f"Attempted to materialize {self.component} as {component_cls.__name__} from '{file_path}'. "
                    f"Failed because transformers rejected the GGUF ({official_exc}) and config source "
                    f"'{config_source}' could not be loaded."
                )
                raise ValueError(msg) from config_exc
        builder = getattr(component_cls, "from_config", None) or component_cls._from_config
        model = builder(config)
        state = load_gguf_checkpoint(str(file_path))
        target = set(model.state_dict())
        aligned = align_state_dict_to_module_keys(state, target)
        model.load_state_dict({key: aligned[key] for key in target}, strict=True)
        return model.to(getattr(torch, self.torch_dtype))
