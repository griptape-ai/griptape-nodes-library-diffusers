"""Pipeline-aware owner for the media generation conditioning surface.

`ModularDiffusionConditioningParameters` owns the optional `pipeline` input
on a conditioning node and dynamically swaps the underlying
`MediaGenConditioningParameter` based on the connected pipeline class.

Registry-driven design:
  * No pipeline connected  → today's flexible image-or-video config.
  * Supported pipeline     → the `CONDITIONING_CONFIG` ClassVar on the
    pipeline's runtime-params class.
  * Unsupported pipeline   → defaults to flexible config but
    `validate_before_node_run` blocks the run with a clear error.

The conditioning output parameter is added once at construction time and
never removed across swaps — its shape is invariant.
"""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

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
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.parameter import (
    MediaGenConditioningParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_params_registry import get_runtime_params_class
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

logger = logging.getLogger("diffusers_nodes_library")


_PARAM_PIPELINE = "pipeline"


def _default_config() -> MediaGenConditioningConfig:
    return MediaGenConditioningConfig(
        image=HybridImageConfig(
            presets=(PRESET_FIRST_MIDDLE_LAST, PRESET_FIRST_LAST, PRESET_FIRST),
            flexible=FlexibleImageConfig(),
            default_choice=PRESET_FIRST_MIDDLE_LAST.display_name,
        ),
        video=VideoConditioningConfig(),
        default_mode=ConditioningMode.IMAGE,
    )


def _runtime_conditioning_config(pipeline_class: str | None) -> MediaGenConditioningConfig | None:
    """Return the runtime-params `CONDITIONING_CONFIG` for `pipeline_class`, or None.

    Returns None when the pipeline class is unknown or the runtime-params class
    has not opted in (i.e. its `CONDITIONING_CONFIG` ClassVar is None).
    """
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
    """Owns the pipeline-aware conditioning surface on a host BaseNode.

    The host node:
      1. Constructs this owner in `__init__` and calls
         `add_output_parameters()` then `add_input_parameters()`.
      2. Overrides `set_parameter_value` to capture the prior value via
         `get_parameter_value(param_name)` BEFORE calling
         `super().set_parameter_value(...)`, then calls
         `on_parameter_value_change(name, old_value, new_value, *, initial_setup)`
         AFTER super has written the new value.
      3. Delegates `validate_before_node_run` and the conditioning output
         payload write in `process` to this owner.
    """

    def __init__(self, node: BaseNode) -> None:
        self._node = node
        self._active_config: MediaGenConditioningConfig = _default_config()
        self._conditioning_parameter = MediaGenConditioningParameter(node, self._active_config)

    # ------------------------------------------------------------------
    # Public hook surface (called by the host node)
    # ------------------------------------------------------------------

    def add_output_parameters(self) -> None:
        # Output socket is shape-invariant across pipeline swaps.
        self._conditioning_parameter.add_output_parameters()

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name=_PARAM_PIPELINE,
                type="Pipeline Config",
                tooltip=(
                    "Optional 🤗 Diffusion pipeline. When connected, the conditioning surface "
                    "swaps to the configuration tailored to that pipeline."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self._conditioning_parameter.add_input_parameters()

    def on_parameter_value_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        *,
        initial_setup: bool,
    ) -> None:
        """React to a parameter value change after the node has stored it.

        Pipeline-change handling lives here (not in `after_value_set`) so it
        runs through the same code path as every other reactive change.
        Initial-setup writes (workflow load) are skipped so saved per-param
        values aren't clobbered by destructive rebuilds during restoration.
        """
        if initial_setup:
            return

        if param_name == _PARAM_PIPELINE:
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

    def build_conditioning_payload(self) -> dict:
        return self._conditioning_parameter.build_conditioning_payload()

    # ------------------------------------------------------------------
    # Private — pipeline-swap handling
    # ------------------------------------------------------------------

    def _handle_pipeline_change(self, old_value: Any, new_value: Any) -> None:
        old_class = self._pipeline_class_of(old_value)
        new_class = self._pipeline_class_of(new_value)
        if old_class == new_class:
            return
        new_config = _config_for_pipeline(new_class)
        if new_config == self._active_config:
            return
        self._conditioning_parameter.remove_input_parameters()
        self._active_config = new_config
        self._conditioning_parameter = MediaGenConditioningParameter(self._node, new_config)
        self._conditioning_parameter.add_input_parameters()

    def _current_pipeline_class(self) -> str | None:
        return self._pipeline_class_of(self._node.get_parameter_value(_PARAM_PIPELINE))

    @staticmethod
    def _pipeline_class_of(value: Any) -> str | None:
        if isinstance(value, DiffusionPipelineArtifact):
            return value.pipeline_name
        return None
