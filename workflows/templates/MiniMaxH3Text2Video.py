# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "MiniMax-H3 Text to Video (GGUF Transformer)"
# schema_version = "0.21.0"
# engine_version_created_with = "0.111.1"
# node_libraries_referenced = [["Griptape Nodes Library", "0.72.0"], ["Griptape Modular Diffusion Nodes Library", "0.9.0"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "LoadComponent"], ["Griptape Modular Diffusion Nodes Library", "NoiseLatentNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Nodes Library", "Note"]]
# description = "Generate video with jointly generated audio using MiniMax-H3 and a GGUF transformer override. Includes component setup notes; the GGUF checkpoint must be supplied separately."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/MiniMaxH3Text2Video.png"
# is_griptape_provided = true
# is_template = true
# is_internal = false
# creation_date = 2026-11-24T11:36:45Z
# last_modified_date = 2026-11-24T11:36:45Z
#
# ///

from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import (
    RegisterLibraryFromFileRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    BasePipelineIdentity,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.nodes.load_component_node import _compute_load_id
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(
            library_name="Griptape Nodes Library", perform_discovery_if_not_found=True
        )
    )
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(
            library_name="Griptape Modular Diffusion Nodes Library",
            perform_discovery_if_not_found=True,
        )
    )
    transformer_path = expand_path_macros(
        "{workflow_dir}/assets/models/minimax-h3/minimax_h3_fl2va-Q4_K_M.gguf"
    )
    transformer = ModelComponentArtifact(
        load_id=_compute_load_id(
            source_type="Single File",
            component="transformer",
            path=transformer_path,
            config_source="",
            repo_id="",
            revision="",
            subfolder="",
        ),
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        file_path=transformer_path,
    )
    optimization_kwargs = {
        "memory_optimization_strategy": "Manual",
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": True,
        "transformer_layerwise_casting": False,
        "cpu_offload_strategy": "None",
        "quantization_mode": "None",
    }
    # Validation can build a downstream node's saved recipe before its upstream nodes run.
    pipeline = DiffusionPipelineArtifact(
        pipeline_name="MiniMaxH3ModularPipeline",
        config_hash=BasePipelineIdentity(
            pipeline_name="MiniMaxH3ModularPipeline",
            config_kwargs={
                "provider": "MiniMax-H3",
                "model": "MiniMaxAI/MiniMax-H3",
                "component_override_transformer": transformer.load_id,
            },
            loras={},
            optimization_kwargs=optimization_kwargs,
            torch_dtype="bfloat16",
        ).cache_key(),
        builder_module="modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters",
        builder_class_name="MiniMaxH3PipelineParameters",
        build_data={
            "base_repo_id": "MiniMaxAI/MiniMax-H3",
            "revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
            "_component_overrides": {
                "transformer": transformer,
            },
        },
        optimization_kwargs=optimization_kwargs,
        is_prequantized=True,
        supports_layerwise_casting=False,
    )
    flow0_name = (
        await GriptapeNodes.ahandle_request(
            CreateFlowRequest(
                parent_flow_name=None,
                flow_name="ControlFlow_1",
                set_as_new_context=False,
                metadata={},
            )
        )
    ).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadComponent",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Load Transformer (GGUF)",
                    metadata={
                        "position": {"x": 0, "y": 0},
                        "size": {"width": 600, "height": 560},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LatentDiffusionPipelineBuilderNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Modular Diffusion Pipeline Builder",
                    metadata={
                        "position": {"x": 760, "y": 0},
                        "size": {"width": 600, "height": 1120},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="pipeline_type",
                    tooltip="Specific pipeline variant within the selected provider (e.g. base, Fill, Edit). Determines which checkpoints and runtime parameters are exposed.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={},
                    mode_allowed_output=False,
                    traits=[
                        {
                            "trait_name": "Options",
                            "trait_module": "griptape_nodes.traits.options",
                            "trait_state": {
                                "choices": ["MiniMaxH3ModularPipeline"],
                                "show_search": True,
                                "search_filter": "",
                                "allow_custom": False,
                            },
                        }
                    ],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="model",
                    default_value="MiniMaxAI/MiniMax-H3",
                    tooltip="model",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "hide_label": False,
                        "hide_property": False,
                        "display_name": "model",
                        "data": [
                            {
                                "name": "MiniMaxAI/MiniMax-H3",
                                "icon": "check-circle",
                                "subtitle": "Downloaded",
                            }
                        ],
                        "dropdown_row_icons": True,
                        "dropdown_row_subtitles": True,
                    },
                    mode_allowed_output=False,
                    traits=[
                        {
                            "trait_name": "Options",
                            "trait_module": "griptape_nodes.traits.options",
                            "trait_state": {
                                "choices": ["MiniMaxAI/MiniMax-H3"],
                                "show_search": True,
                                "search_filter": "",
                                "allow_custom": False,
                            },
                        },
                        {
                            "trait_name": "Button",
                            "trait_module": "griptape_nodes.traits.button",
                            "trait_state": {
                                "label": "",
                                "variant": "secondary",
                                "size": "icon",
                                "state": "normal",
                                "icon": "list-restart",
                                "icon_class": None,
                                "icon_position": None,
                                "full_width": False,
                                "loading_label": None,
                                "loading_icon": None,
                                "loading_icon_class": None,
                                "tooltip": "Refresh model list",
                                "button_link": None,
                            },
                        },
                    ],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="model_download",
                    tooltip="Open Model Manager to download the selected model",
                    type="button",
                    input_types=["str", "any"],
                    output_type="str",
                    ui_options={
                        "hide": True,
                        "hide_label": False,
                        "hide_property": False,
                    },
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    traits=[
                        {
                            "trait_name": "Button",
                            "trait_module": "griptape_nodes.traits.button",
                            "trait_state": {
                                "label": "Open Model Manager to Download",
                                "variant": "secondary",
                                "size": "default",
                                "state": "normal",
                                "icon": "download",
                                "icon_class": None,
                                "icon_position": None,
                                "full_width": True,
                                "loading_label": None,
                                "loading_icon": None,
                                "loading_icon_class": None,
                                "tooltip": "Open Model Manager to download the selected model",
                                "button_link": None,
                            },
                        }
                    ],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="attention_slicing",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="vae_slicing",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="vae_tiling",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="transformer_layerwise_casting",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="cpu_offload_strategy",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="quantization_mode",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="component_audio_vae",
                    tooltip="Optional override for the Audio VAE component. Connect a Load Component node to replace the default.",
                    type="VaeComponentArtifact",
                    input_types=["VaeComponentArtifact"],
                    output_type="VaeComponentArtifact",
                    ui_options={"display_name": "Audio VAE"},
                    mode_allowed_property=False,
                    is_user_defined=False,
                    parent_element_name="component_overrides",
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="component_audio_scheduler",
                    tooltip="Optional override for the Audio Scheduler component. Connect a Load Component node to replace the default.",
                    type="SchedulerComponentArtifact",
                    input_types=["SchedulerComponentArtifact"],
                    output_type="SchedulerComponentArtifact",
                    ui_options={"display_name": "Audio Scheduler"},
                    mode_allowed_property=False,
                    is_user_defined=False,
                    parent_element_name="component_overrides",
                    initial_setup=True,
                )
            )
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="NoiseLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Create Noise Latents",
                    metadata={
                        "position": {"x": 1520, "y": 0},
                        "size": {"width": 600, "height": 500},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="num_frames",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DiffusionPipelineGenerateLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Generate Media Latents",
                    metadata={
                        "position": {"x": 2280, "y": 0},
                        "size": {"width": 600, "height": 1080},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="add_noise",
                    ui_options={"hide": False},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="prompt",
                    default_value="",
                    tooltip="The prompt to guide generation of the video and its soundtrack.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={},
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="conditioning_images",
                    tooltip="First/last keyframes for keyframe-to-video generation, from a Reference Media node.",
                    type="media_gen_conditioning",
                    input_types=["media_gen_conditioning", "ImageUrlArtifact"],
                    output_type="media_gen_conditioning",
                    ui_options={},
                    mode_allowed_property=False,
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="num_inference_steps",
                    default_value=20,
                    tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={},
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="randomize_seed",
                    default_value=False,
                    tooltip="randomize the seed on each run",
                    type="bool",
                    input_types=["any"],
                    output_type="bool",
                    ui_options={"hide_label": False, "hide_property": False},
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="seed",
                    default_value=42,
                    tooltip="the seed to use for the generation",
                    type="int",
                    input_types=["any"],
                    output_type="int",
                    ui_options={"hide_label": False, "hide_property": False},
                    traits=[],
                    initial_setup=True,
                )
            )
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeDecodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Decode Media",
                    metadata={
                        "position": {"x": 3040, "y": 0},
                        "size": {"width": 600, "height": 920},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="latent_tensor",
                    tooltip="Latent tensor to decode with the pipeline VAE.",
                    type="LatentArtifact",
                    input_types=["LatentArtifact"],
                    output_type="LatentArtifact",
                    ui_options={"display_name": "input_latent"},
                    mode_allowed_property=False,
                    mode_allowed_output=False,
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="fps",
                    default_value=24,
                    tooltip="Frames per second for video output.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={"min": 1, "max": 120},
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="output_video",
                    tooltip="Generated video.",
                    type="VideoUrlArtifact",
                    input_types=["VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    traits=[],
                    initial_setup=True,
                )
            )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="component_output",
                target_node_name=node2_name,
                target_parameter_name="component_transformer",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="pipeline",
                target_node_name=node3_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="pipeline",
                target_node_name=node4_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="output_latent",
                target_node_name=node4_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="pipeline",
                target_node_name=node5_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="output_latent",
                target_node_name=node5_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="component",
                    node_name=node0_name,
                    value="Transformer",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="source_type",
                    node_name=node0_name,
                    value="Single File",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="file_path",
                    node_name=node0_name,
                    value="{workflow_dir}/assets/models/minimax-h3/minimax_h3_fl2va-Q4_K_M.gguf",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="config_source",
                    node_name=node0_name,
                    value="",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="folder_path",
                    node_name=node0_name,
                    value="",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="repo_id",
                    node_name=node0_name,
                    value="",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="revision",
                    node_name=node0_name,
                    value="main",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="subfolder",
                    node_name=node0_name,
                    value="",
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node2_name,
                    value="MiniMax-H3",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node2_name,
                    value="MiniMaxH3ModularPipeline",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node2_name,
                    value="MiniMaxAI/MiniMax-H3",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node2_name,
                    value="Manual",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node2_name,
                    value=True,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node2_name,
                    value=True,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_tiling",
                    node_name=node2_name,
                    value=True,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node2_name,
                    value=False,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node2_name,
                    value="None",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node2_name,
                    value="None",
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node2_name,
                    value=[],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node3_name,
                    value=pipeline,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node3_name,
                    value=False,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node3_name,
                    value=42,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node3_name,
                    value=672,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node3_name,
                    value=384,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node3_name,
                    value=124,
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node4_name,
                    value=pipeline,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node4_name,
                    value=False,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node4_name,
                    value=0,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node4_name,
                    value=-1,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node4_name,
                    value=False,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node4_name,
                    value=(
                        "A curious little dormouse cosmonaut in a puffy silver spacesuit, with oversized aviator "
                        "goggles tucked inside its round glass helmet, gently plants a tiny flag shaped like a "
                        "leaf on a moon made of pale cheese. It takes a delighted little hop, floating slowly "
                        "above the dusty surface before landing beside its walnut-shell spacecraft. The camera "
                        "glides low alongside the mouse, revealing sparkling stars and a blue marble Earth "
                        "overhead. Warm rim light, vivid colors, whimsical hand-painted 2D animation. Sound: "
                        "soft radio crackles and the mouse's excited squeak over its helmet radio, with a "
                        "gentle electronic chime from the spacecraft. No dialogue or music."
                    ),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node4_name,
                    value=20,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node4_name,
                    value=False,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node4_name,
                    value=42,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node4_name,
                    value=[],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node5_name,
                    value=pipeline,
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node5_name,
                    value=24,
                    initial_setup=True,
                    is_output=False,
                )
            )
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="MiniMax-H3 Setup",
                    metadata={
                        "position": {"x": 760, "y": -587.391472113794},
                        "size": {"width": 600, "height": 527},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node6_name,
                value="## MiniMax-H3 with a GGUF transformer override\n\n1. Download the full MiniMaxAI/MiniMax-H3 model through Model Manager. The base repo supplies the text encoder, VAE, audio VAE, schedulers, tokenizer, processor, and the GGUF transformer config.\n2. Place a compatible MiniMax-H3 GGUF transformer at `assets/models/minimax-h3/minimax_h3_fl2va-Q4_K_M.gguf` beside this workflow, or select your own file in Load Transformer (GGUF).\n\nRun left to right: load the transformer override, build MiniMax-H3, create noise, generate, and decode the video with its soundtrack. Edit the prompt in Generate Media Latents. The noise seed controls the initial latents because Add Noise is disabled.",
                initial_setup=True,
                is_output=False,
            )
        )
        for node_name, parameter_name, value in [
            (node0_name, "component_output", transformer),
            (node2_name, "component_transformer", transformer),
            (node2_name, "pipeline", pipeline),
        ]:
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name=parameter_name,
                    node_name=node_name,
                    value=value,
                    initial_setup=True,
                    is_output=False,
                )
            )
            if parameter_name in {"component_output", "pipeline"}:
                await GriptapeNodes.ahandle_request(
                    SetParameterValueRequest(
                        parameter_name=parameter_name,
                        node_name=node_name,
                        value=value,
                        initial_setup=True,
                        is_output=True,
                    )
                )
