from typing import ClassVar

from griptape_nodes.exe_types.node_types import BaseNode
from PIL.Image import Image

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    FlexibleImageConfig,
    MediaGenConditioningConfig,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.flux2_runtime_parameters import (
    Flux2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
)


class Flux2DevPipelineRuntimeParameters(Flux2PipelineRuntimeParameters):
    """Flux2 dev — adds optional image conditioning on top of the base prompt/guidance params."""

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(
            min_count=1,
            max_count=8,
            expose_strength=False,
            expose_frame_index=False,
        ),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="reference_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="Reference images that guide the generation. Connect a Media Gen Conditioning node or an image directly.",
            badge_title="Reference images",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply reference images. "
                "The model uses them as visual references alongside the text prompt. "
                "Only **image** payloads are accepted; **video** payloads are not allowed.\n\n"
                "**Tip:** You can also connect an image directly — without a Media Gen Conditioning node — "
                "for single-image conditioning."
            ),
        )

    def _add_input_parameters(self) -> None:
        super()._add_input_parameters()
        self._media_gen_conditioning_param.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        super()._remove_input_parameters()

    def _get_pipe_kwargs(self) -> dict:
        kwargs = super()._get_pipe_kwargs()
        conditioning_kwargs = self._media_gen_conditioning_param.get_pipe_kwargs()
        raw = conditioning_kwargs.get(MediaGenConditioningKey.OUTPUT)
        images = self._resolve_reference_images(raw)
        if images:
            kwargs["image"] = images
        return kwargs

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        conditioning_errors = self._media_gen_conditioning_param.validate_before_node_run()
        if conditioning_errors:
            errors.extend(conditioning_errors)
        return errors or None

    @staticmethod
    def _resolve_reference_images(raw: object) -> list[Image] | None:
        payloads = normalize_to_payloads(raw)
        if payloads is None:
            return None
        return _build_images_from_payloads(payloads)


def _build_images_from_payloads(payloads: list[MediaGenConditioningPayload]) -> list[Image]:
    images: list[Image] = []
    for payload in payloads:
        if payload.mode is not ConditioningMode.IMAGE:
            raise ValueError(
                f"Attempted to build Flux2 image conditioning. "
                f"Failed with mode '{payload.mode.value}' because only image conditioning is accepted."
            )
        for entry in payload.entries:
            images.append(resolve_conditioning_image(entry.artifact))
    return images
