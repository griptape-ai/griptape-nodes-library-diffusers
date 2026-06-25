import logging
from typing import ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    PRESET_FIRST,
    PRESET_FIRST_LAST,
    PRESET_FIRST_MIDDLE_LAST,
    FlexibleImageConfig,
    HybridImageConfig,
    MediaGenConditioningConfig,
    VideoConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

logger = logging.getLogger("diffusers_nodes_library")

_SOURCE_VIDEO_KEY = "vace_source_video"
_MASK_KEY = "vace_mask"
_REFERENCE_IMAGES_KEY = "vace_reference_images"


class WanVacePipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=HybridImageConfig(
            presets=(PRESET_FIRST_MIDDLE_LAST, PRESET_FIRST_LAST, PRESET_FIRST),
            flexible=FlexibleImageConfig(expose_strength=False),
            default_choice=PRESET_FIRST_MIDDLE_LAST.display_name,
        ),
        video=VideoConditioningConfig(expose_strength=False),
        default_mode=ConditioningMode.IMAGE,
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._source_media_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="source_media",
            output_key=_SOURCE_VIDEO_KEY,
            accepted_modes=(ConditioningMode.IMAGE, ConditioningMode.VIDEO),
            tooltip="Source media for VACE conditioning. Black areas in the mask will be conditioned on this source; white areas will be generated from the prompt.",
            badge_title="Source media",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply the source media for VACE conditioning. "
                "The source media defines the content to condition on: black areas in the mask are filled from this "
                "video, white areas are generated freely from the prompt.\n\n"
                "**IMAGE payload** — individual images are placed at their frame positions (`first`, `last`, or an "
                "integer index); unspecified frames are filled with neutral gray `(128, 128, 128)` — those frames "
                "contribute little to conditioning when paired with a white (generate) mask.\n\n"
                "**VIDEO payload** — frames are placed starting at the payload's `frame_index` (default 0); "
                "any frames before that position or after the video ends are filled with gray. This supports "
                "extension workflows: e.g. supply a short clip at `frame_index=N/2` and the model generates the rest.\n\n"
                "If no mask is connected, the driver auto-derives one: frame positions covered by the source are "
                "marked black (preserve), uncovered positions are marked white (generate).\n\n"
                "**Tip:** You can also connect an image or video directly — without a Media Gen Conditioning node — "
                "for single-item conditioning at frame index **0**."
            ),
        )
        self._mask_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="mask",
            output_key=_MASK_KEY,
            accepted_modes=(ConditioningMode.IMAGE, ConditioningMode.VIDEO),
            tooltip="Per-frame binary mask. Black (0) = condition on source media (preserve); white (255) = generate from prompt (edit). If omitted, the mask is derived automatically from source_media. Requires source_media.",
            badge_title="Mask",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply the per-frame mask for selective VACE editing.\n\n"
                "**Black (0)** = condition on the source video (keep / preserve).\n"
                "**White (255)** = generate freely from the prompt (edit / inpaint).\n\n"
                "**IMAGE payload** — individual mask images are placed at their frame positions; unspecified frames "
                "are filled with white (255 = generate). Common with a VIDEO source: place black mask images at the "
                "specific frames you want to preserve from the video, leave the rest white.\n\n"
                "**VIDEO payload** — frames are placed starting at the payload's `frame_index` (default 0) and "
                "converted to grayscale; unspecified frames are filled with white (255 = generate). Unusual when "
                "combined with an IMAGE source, but valid.\n\n"
                "⚠️ **Frame alignment:** a black mask at a position where `source_media` has only gray fill "
                "(128, 128, 128) will preserve the fill rather than any meaningful source content. This can be "
                "intentional — e.g. to force-preserve a blank region — but is usually a mismatch. Check that your "
                "mask positions correspond to frames that actually have source content.\n\n"
                "**If not connected**, the mask is derived automatically from `source_media`: frame positions "
                "covered by the source are marked black (preserve); uncovered positions are marked white (generate). "
                "⚠️ Requires `source_media` to be connected — connecting mask without source_media raises an error "
                "at generation time.\n\n"
                "**Tip:** You can also connect an image or video directly — without a Media Gen Conditioning node — "
                "for a single mask at frame index **0**."
            ),
        )
        self._reference_images_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="reference_images",
            output_key=_REFERENCE_IMAGES_KEY,
            accepted_modes=(ConditioningMode.IMAGE, ConditioningMode.VIDEO),
            tooltip="Optional reference images for VACE. VIDEO payloads are automatically split into individual frames. Requires source_media.",
            badge_title="Reference images",
            badge_message=(
                "Connect one or more **Media Gen Conditioning** nodes here to supply optional reference images for "
                "VACE generation. Reference images provide additional visual guidance for the generated content — "
                "for example, a reference photo of a character to inpaint into the masked region.\n\n"
                "**IMAGE payload** — each image is used directly as a reference.\n\n"
                "**VIDEO payload** — the video is automatically broken into individual frames, each treated as a "
                "separate reference image. Multiple conditioning nodes can be connected.\n\n"
                "⚠️ Requires `source_media` to be connected.\n\n"
                "**Tip:** You can also connect an image or video directly — without a Media Gen Conditioning node — "
                "for single-item conditioning at frame index **0** and strength **1.0**."
            ),
        )

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide the video generation.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide the video generation. Ignored when guidance_scale is less than 1.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=5.0,
                type="float",
                tooltip=(
                    "Guidance scale as defined in Classifier-Free Diffusion Guidance. Higher values encourage the "
                    "model to follow the prompt more closely, usually at the expense of lower visual quality. "
                    "Guidance is enabled by setting guidance_scale > 1."
                ),
            )
        )
        self._node.add_parameter(
            Parameter(
                name="conditioning_scale",
                default_value=1.0,
                type="float",
                tooltip=(
                    "Conditioning scale applied when adding the VACE control latent stream to the denoising latent "
                    "stream at each control layer. A single float applies uniformly to all layers; a list applies "
                    "per-layer (one value per vace_layers entry in the transformer config). Higher values increase "
                    "source video influence; lower values allow freer generation."
                ),
            )
        )
        self._source_media_param.add_input_parameters()
        self._mask_param.add_input_parameters()
        self._reference_images_param.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._source_media_param.remove_input_parameters()
        self._mask_param.remove_input_parameters()
        self._reference_images_param.remove_input_parameters()
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._node.remove_parameter_element_by_name("conditioning_scale")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "conditioning_scale": self._node.get_parameter_value("conditioning_scale"),
            **self._source_media_param.get_pipe_kwargs(),
            **self._mask_param.get_pipe_kwargs(),
            **self._reference_images_param.get_pipe_kwargs(),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        for param in (self._source_media_param, self._mask_param, self._reference_images_param):
            param_errors = param.validate_before_node_run()
            if param_errors:
                errors.extend(param_errors)

        source_media = self._node.get_parameter_value("source_media")
        mask = self._node.get_parameter_value("mask")
        reference_images = self._node.get_parameter_value("reference_images")

        if mask is not None and source_media is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because `mask` is connected without `source_media`. "
                    "Connect a source media conditioning or disconnect the mask."
                )
            )
        if reference_images and source_media is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because `reference_images` is connected without `source_media`. "
                    "Connect a source media conditioning or disconnect the reference images."
                )
            )

        return errors or None
