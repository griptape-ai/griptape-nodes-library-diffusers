"""Pipeline-aware owner for the media generation conditioning surface.

`ModularDiffusionConditioningParameters` owns the optional `pipeline` input
on a conditioning node and dynamically swaps the underlying
`MediaGenConditioningParameter` based on the connected pipeline class.

Registry-driven design:
  * No pipeline connected  → flexible image-or-video config.
  * Supported pipeline     → the `CONDITIONING_CONFIG` ClassVar on the
    pipeline's runtime-params class.
  * Unsupported pipeline   → defaults to flexible config but
    `validate_before_node_run` blocks the run with a clear error.

The conditioning output parameter is added once at construction time and
never removed across swaps — its shape is invariant.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    IncomingConnection,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
    OutgoingConnection,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    PRESET_FIRST,
    PRESET_FIRST_LAST,
    PRESET_FIRST_MIDDLE_LAST,
    FlexibleImageConfig,
    HybridImageConfig,
    MediaGenConditioningConfig,
    VideoConditioningConfig,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    MediaGenConditioningPayload,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.parameter import (
    MediaGenConditioningParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_params_registry import get_runtime_params_class
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

_PARAM_PIPELINE = "pipeline"


def _default_config() -> MediaGenConditioningConfig:
    return MediaGenConditioningConfig(
        image=HybridImageConfig(
            presets=(PRESET_FIRST_MIDDLE_LAST, PRESET_FIRST_LAST, PRESET_FIRST),
            flexible=FlexibleImageConfig(),
            default_choice="Custom",
        ),
        video=VideoConditioningConfig(),
        default_mode=ConditioningMode.IMAGE,
    )


def _runtime_conditioning_config(pipeline_class: str | None) -> MediaGenConditioningConfig | None:
    """Get CONDITIONING_CONFIG for pipeline_class, or None if unknown/unsupported."""
    if pipeline_class is None:
        return None
    runtime_cls = get_runtime_params_class(pipeline_class)
    if runtime_cls is None:
        return None
    return runtime_cls.CONDITIONING_CONFIG


def _config_for_pipeline(pipeline_class: str | None) -> MediaGenConditioningConfig:
    """Pick the conditioning config for `pipeline_class`, falling back to the default UI."""
    if pipeline_class is None:
        return _default_config()
    config = _runtime_conditioning_config(pipeline_class)
    if config is None:
        # Unsupported — keep default UI shape; validate_before_node_run blocks the run.
        return _default_config()
    return config


class ModularDiffusionConditioningParameters:
    """Pipeline-aware conditioning surface owner. Host node calls add_input/output_parameters(),
    on_parameter_value_change(), validate_before_node_run(), and build_conditioning_payload().
    """

    def __init__(self, node: BaseNode) -> None:
        self._node = node
        self._active_config: MediaGenConditioningConfig = _default_config()
        self._conditioning_parameter = MediaGenConditioningParameter(node, self._active_config)

    # ------------------------------------------------------------------
    # Public hook surface (called by the host node)
    # ------------------------------------------------------------------

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name=_PARAM_PIPELINE,
                type="Pipeline Config",
                tooltip=(
                    "Optional 🤗 Diffusion pipeline. When connected, the conditioning surface "
                    "swaps to the configuration tailored to that pipeline."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )
        self._conditioning_parameter.add_output_parameters()

    def add_input_parameters(self) -> None:
        self._conditioning_parameter.add_input_parameters()

    def on_parameter_value_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        *,
        initial_setup: bool,
    ) -> None:
        """Handle parameter changes. Pipeline changes use soft update (on workflow load) or hard swap (post-load)."""
        if param_name == _PARAM_PIPELINE:
            if initial_setup:
                self._soft_update_config(new_value)
                return
            self._handle_pipeline_change(old_value, new_value)
            return

        self._conditioning_parameter.on_parameter_value_change(
            param_name, old_value, new_value, initial_setup=initial_setup
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        pipeline_class = self._current_pipeline_class()
        if pipeline_class is not None and _runtime_conditioning_config(pipeline_class) is None:
            msg = (
                f"{self._node.name}: Pipeline class '{pipeline_class}' does not support media generation conditioning."
            )
            errors.append(ValueError(msg))
        downstream_errors = self._conditioning_parameter.validate_before_node_run()
        if downstream_errors:
            errors.extend(downstream_errors)
        return errors or None

    def build_conditioning_payload(self) -> MediaGenConditioningPayload:
        return self._conditioning_parameter.build_conditioning_payload()

    # ------------------------------------------------------------------
    # Private — pipeline-swap handling
    # ------------------------------------------------------------------

    def _soft_update_config(self, pipeline_artifact: Any) -> None:
        """Update config without removing params (for workflow load)."""
        pipeline_class = self._pipeline_class_of(pipeline_artifact)
        new_config = _config_for_pipeline(pipeline_class)

        if new_config == self._active_config:
            return
        self._active_config = new_config
        self._conditioning_parameter.update_config(new_config)

    def _handle_pipeline_change(self, old_value: Any, new_value: Any) -> None:
        old_class = self._pipeline_class_of(old_value)
        new_class = self._pipeline_class_of(new_value)
        if old_class == new_class:
            return
        new_config = _config_for_pipeline(new_class)
        if new_config == self._active_config:
            return

        saved_incoming, saved_outgoing = self._save_surface_connections()

        self._conditioning_parameter.remove_input_parameters()
        self._active_config = new_config
        self._conditioning_parameter = MediaGenConditioningParameter(self._node, new_config)
        self._conditioning_parameter.add_input_parameters()

        self._restore_surface_connections(saved_incoming, saved_outgoing)

    def _current_pipeline_class(self) -> str | None:
        return self._pipeline_class_of(self._node.get_parameter_value(_PARAM_PIPELINE))

    @staticmethod
    def _pipeline_class_of(value: Any) -> str | None:
        if isinstance(value, DiffusionPipelineArtifact):
            return value.pipeline_name
        return None

    def _save_surface_connections(self) -> tuple[list[IncomingConnection], list[OutgoingConnection]]:
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self._node.name))
        if not isinstance(result, ListConnectionsForNodeResultSuccess):
            return [], []

        incoming = [
            conn for conn in result.incoming_connections if conn.target_parameter_name.startswith(("image_", "video"))
        ]
        outgoing = [
            conn for conn in result.outgoing_connections if conn.source_parameter_name.startswith("conditioning")
        ]
        return incoming, outgoing

    def _restore_surface_connections(
        self,
        saved_incoming: list[IncomingConnection],
        saved_outgoing: list[OutgoingConnection],
    ) -> None:
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self._node.name))
        existing_incoming: set[tuple[str, str, str]] = set()
        existing_outgoing: set[tuple[str, str, str]] = set()
        if isinstance(result, ListConnectionsForNodeResultSuccess):
            existing_incoming = {
                (conn.source_node_name, conn.source_parameter_name, conn.target_parameter_name)
                for conn in result.incoming_connections
            }
            existing_outgoing = {
                (conn.source_parameter_name, conn.target_node_name, conn.target_parameter_name)
                for conn in result.outgoing_connections
            }

        for conn in saved_incoming:
            if self._node.get_parameter_by_name(conn.target_parameter_name) is None:
                continue
            key = (conn.source_node_name, conn.source_parameter_name, conn.target_parameter_name)
            if key in existing_incoming:
                continue
            GriptapeNodes.handle_request(
                CreateConnectionRequest(
                    source_node_name=conn.source_node_name,
                    source_parameter_name=conn.source_parameter_name,
                    target_node_name=self._node.name,
                    target_parameter_name=conn.target_parameter_name,
                )
            )

        for conn in saved_outgoing:
            if self._node.get_parameter_by_name(conn.source_parameter_name) is None:
                continue
            key = (conn.source_parameter_name, conn.target_node_name, conn.target_parameter_name)
            if key in existing_outgoing:
                continue
            GriptapeNodes.handle_request(
                CreateConnectionRequest(
                    source_node_name=self._node.name,
                    source_parameter_name=conn.source_parameter_name,
                    target_node_name=conn.target_node_name,
                    target_parameter_name=conn.target_parameter_name,
                )
            )
