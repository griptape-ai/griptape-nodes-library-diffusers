from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from griptape_nodes.exe_types.node_types import BaseNode
from huggingface_hub import snapshot_download

from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoParameter,
)
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

if TYPE_CHECKING:
    from diffusers import ComponentsManager, ModularPipeline  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")

# The transformer is 61.7 GB in bfloat16 and the Qwen3-VL conditioner another 62.1 GB, so nothing
# fits alongside the accelerator's own working set. The margin is the model card's recommendation
# for a single 80 GB card.
AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN = "12GB"


class MiniMaxH3PipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.minimax_h3.modular_pipeline:MiniMaxH3ModularPipeline"
    _component_slots: ClassVar[list[str]] = []
    _workflow: ClassVar[str] = "fl2va"
    # fl2va loads `transformer`; `transformer_ref` is ref2va's partition.
    _unloaded_component_slots: ClassVar[frozenset[str]] = frozenset({"transformer_ref"})
    # Reloaded from disk when needed, so the 62 GB conditioner isn't parked in RAM next to the transformer.
    # A reload restores the checkpoint as saved, so later changes to the weights are lost (see
    # TODO(disk-backed-optimisations)). Safe today only because nothing modifies `text_encoder` after loading:
    # `is_prequantized` skips quantization and LoRAs are unsupported. Adding either means reapplying it per reload.
    _disk_backed_component_names: ClassVar[frozenset[str]] = frozenset({"text_encoder"})

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        """MiniMaxH3ModularPipeline does not support overrides."""
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = ConnectableHuggingFaceRepoParameter(
            node,
            repo_ids=["MiniMaxAI/MiniMax-H3"],
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        return self.build_data(repo_id, revision)

    @classmethod
    def requires_device_map(cls) -> bool:
        # Used here as an opt-out from post-hoc pipeline optimization, not literally to request an
        # accelerate device_map. MiniMax-H3 is a ModularPipeline, on which
        # `optimize_diffusion_pipeline` would call `.to(device)` (a guaranteed OOM at ~124 GB) while
        # both `enable_*_cpu_offload` calls are hasattr-gated and silently skipped. Placement is
        # instead owned by `build_pipeline_from_build_data` via the ComponentsManager below.
        return True

    @classmethod
    def build_data(cls, repo_id: str, revision: str) -> dict[str, Any]:
        return {"repo_id": repo_id, "revision": revision}

    @classmethod
    def is_prequantized_repo(cls, repo_id: str) -> bool:
        return True

    def is_prequantized(self) -> bool:
        # Suppresses quantization and layerwise casting, which fire before the requires_device_map
        # short-circuit. Neither is safe to apply on top of the ComponentsManager offload hooks.
        return True

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> ModularPipeline:  # type: ignore[reportAttributeAccessIssue]
        # `from_pretrained` resolves the component specs but loads no weights; `load_components`
        # fetches the components for the selected workflow.
        from diffusers import ComponentsManager, ModularPipeline  # type: ignore[reportMissingImports]

        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            build_data["repo_id"],
            revision=build_data["revision"],
            components_manager=manager,
            local_files_only=True,
        )
        cls._load_components_with_disk_backed_offload(pipe, manager, build_data["repo_id"], build_data["revision"])  # type: ignore[reportArgumentType]
        return pipe  # type: ignore[reportReturnType]

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> ModularPipeline:
        # Skips the base class's override routing; there are no overrides here.
        return cls._build_pipeline_from_repo(build_data, {})

    @classmethod
    def _load_components_with_disk_backed_offload(
        cls, pipe: ModularPipeline, manager: ComponentsManager, repo_id: str, revision: str
    ) -> None:
        """Load the workflow's components, keeping `_disk_backed_component_names` on disk instead of in RAM."""
        import torch  # type: ignore[reportMissingImports]

        from modular_diffusion_nodes_library.utils.disk_backed_offload import (
            DiskBackedComponent,
            disk_backed_component_from_spec,
            empty_component_from_spec,
            enable_disk_backed_offload,
        )

        dtype = torch.bfloat16  # shared by the disk-backed components and `load_components`
        disk_backed_components: dict[str, DiskBackedComponent] = {}
        for component_name in cls._disk_backed_component_names:
            spec = pipe.get_component_spec(component_name)
            spec.revision = revision
            disk_backed_components[component_name] = disk_backed_component_from_spec(spec, dtype=dtype)
            # Registered empty so `load_components` skips it; `update_components` would drop the spec's repo fields.
            pipe.register_components(**{component_name: empty_component_from_spec(spec, dtype=dtype)})

        # The processor reads from the cached folder: its folder has no config.json, which transformers
        # cannot resolve offline by repo id. `allow_patterns` limits the offline completeness check to
        # the processor, so a partial download of the other workflow's weights doesn't raise.
        snapshot_dir = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=True,
            allow_patterns="processor/*",
        )

        pipe.load_components(
            workflow=cls._workflow,
            dtype=dtype,
            revision=revision,
            local_files_only=True,
            pretrained_model_name_or_path={"processor": snapshot_dir},
        )
        enable_disk_backed_offload(
            manager,
            get_best_device(),
            disk_backed_components,
            memory_reserve_margin=AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN,
        )
