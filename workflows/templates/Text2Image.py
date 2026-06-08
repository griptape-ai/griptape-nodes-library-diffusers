# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "Text2Image"
# schema_version = "0.17.0"
# engine_version_created_with = "0.83.0"
# node_libraries_referenced = [["Griptape Modular Diffusion Nodes Library", "0.1.0"], ["Griptape Nodes Advanced Media Library", "0.72.1"], ["Griptape Nodes Library", "0.78.0"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "NoiseLatentNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "TextInput"]]
# description = "T2I workflow using the Modular Diffusion Library Nodes"
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/Text2Image.webp"
# is_griptape_provided = true
# is_template = true
# creation_date = 2026-05-28T13:32:30.492359Z
# last_modified_date = 2026-06-01T11:05:50.498339Z
#
# ///

import pickle
from griptape_nodes.node_library.library_registry import IconVariant, NodeDeprecationMetadata, NodeMetadata
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(
            library_name="Griptape Modular Diffusion Nodes Library", perform_discovery_if_not_found=True
        )
    )
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
    )
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {
        "6ce7092a-267f-4d3d-bf1b-c0db6545827f": pickle.loads(
            b"\x80\x04\x95\x1a\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0eZImagePipeline\x94\x8c\x0bconfig_hash\x94\x8cQZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\x94\x8c\x0f_builder_module\x94\x8cFmodular_diffusion_nodes_library.standard_parameters.z_image_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x18ZImagePipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94\x8c\rbase_revision\x94\x8c(f332072aa78be7aecdf3ee76d5c247082da564a6\x94u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h\x1eu\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x89\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "0acb3923-7a12-4200-92e5-fc9538a5ed59": pickle.loads(b"\x80\x04\x89."),
        "69eeb8fd-371c-4836-b883-0619a7ff7a18": pickle.loads(b"\x80\x04K*."),
        "03e0ca63-e4a7-4ae5-b2e9-01d49186cee3": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x04."),
        "df572ea8-0a34-401d-9e24-84350168a121": pickle.loads(b"\x80\x04K)."),
        "e422e707-236c-4162-b670-35f32300929a": pickle.loads(b"\x80\x04K\x14."),
        "3414702c-a9f1-4c46-87fd-6bdfb05bc9e2": pickle.loads(
            b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07Z-Image\x94."
        ),
        "ad144a6c-f157-40ed-a045-9fdd5c1e77c0": pickle.loads(
            b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eZImagePipeline\x94."
        ),
        "683aa0ed-9200-495a-aa17-da8dd5cdb8ad": pickle.loads(
            b"\x80\x04\x95\x1c\x00\x00\x00\x00\x00\x00\x00\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94."
        ),
        "89c307a5-6d32-4e86-b433-fd29c0d61b50": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tAutomatic\x94."
        ),
        "4e0a75a4-d349-4c12-bcbb-b8344f1092b9": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04None\x94."
        ),
        "9e014ef5-00b4-4a44-8c10-2377d7600fb0": pickle.loads(b"\x80\x04]\x94."),
        "d0686a76-5442-40b0-8787-ebff1b281940": pickle.loads(
            b"\x80\x04\x95l\x06\x00\x00\x00\x00\x00\x00Xe\x06\x00\x00Building pipeline...\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nNo cached pipeline found. Building new pipeline.\nCreating new pipeline instance...\nLoading pipeline took 32.14 seconds\nConfiguring LoRAs took 0.00 milliseconds\nApplying optimizations took 99.71 milliseconds\nPipeline creation complete.\nPipeline building/caching took 32.24 seconds\nPipeline building complete.\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: LTXPipeline-f5b01a2e40e29232c5a85a4789a3ec7eb6daaa8ea709d1da160b456cb91e4b09-0\nPipeline configuration hash: LTXPipeline-f5b01a2e40e29232c5a85a4789a3ec7eb6daaa8ea709d1da160b456cb91e4b09-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\n\x94."
        ),
        "1100f365-2eb2-4b41-b0d0-00f4f3cc0e14": pickle.loads(b"\x80\x04K\x00."),
        "91b42564-29c9-4901-bc75-0fa093035bc6": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xff\xff\xff\xff."
        ),
        "ec2a56f8-526b-4ad5-82f7-bf4f14628439": pickle.loads(
            b"\x80\x04\x952\x00\x00\x00\x00\x00\x00\x00\x8c.A calico cat on a cobble stone wall, minecraft\x94."
        ),
        "131a647d-c3d9-4cb9-ac7c-a939259eb0d1": pickle.loads(b"\x80\x04]\x94."),
        "a2bcdc65-b664-44bd-8d94-0fee5578c2f0": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00."
        ),
        "85de3f2b-1fc1-45da-ab06-80d88b3d9155": pickle.loads(
            b"\x80\x04\x95\\\x03\x00\x00\x00\x00\x00\x00XU\x03\x00\x00Completed inference step 2 of 20. 1.14 s/it\nCompleted inference step 3 of 20. 1.14 s/it\nCompleted inference step 4 of 20. 1.14 s/it\nCompleted inference step 5 of 20. 1.14 s/it\nCompleted inference step 6 of 20. 1.14 s/it\nCompleted inference step 7 of 20. 1.14 s/it\nCompleted inference step 8 of 20. 1.14 s/it\nCompleted inference step 9 of 20. 1.14 s/it\nCompleted inference step 10 of 20. 1.14 s/it\nCompleted inference step 11 of 20. 1.14 s/it\nCompleted inference step 12 of 20. 1.14 s/it\nCompleted inference step 13 of 20. 1.14 s/it\nCompleted inference step 14 of 20. 1.15 s/it\nCompleted inference step 15 of 20. 1.15 s/it\nCompleted inference step 16 of 20. 1.15 s/it\nCompleted inference step 17 of 20. 1.15 s/it\nCompleted inference step 18 of 20. 1.15 s/it\nCompleted inference step 19 of 20. 1.15 s/it\nCompleted inference step 20 of 20. 1.15 s/it\nDone.\n\x94."
        ),
        "533120f8-c085-4280-9685-11a38148ec55": pickle.loads(
            b"\x80\x04\x95\xc9\x01\x00\x00\x00\x00\x00\x00X\xc2\x01\x00\x00### Modular Diffusion Pipeline Builder\nJust like the `Diffusion Pipline Builder` from the `Griptape-Nodes-Advanced-Media-Library`, the only difference is that the models listed here are setup differently to allow for more advanced workflows\n\nNote: Please make sure you do not mix and match between the standard and the modular diffusion nodes as workflows will not work.\n\nDo not also mix and match model pipelines as they will produce garbage results\x94."
        ),
        "eba87e6c-dc04-41da-b400-25993102f474": pickle.loads(
            b"\x80\x04\x95\xb3\x00\x00\x00\x00\x00\x00\x00\x8c\xaf### Create Noise Latents\nThis node creates an empty latent and fills it the pipelines' expected noise pattern based on the scheduling options supplied in the model repository.\x94."
        ),
        "22886988-aba5-47c4-93e0-347ee7ef5a03": pickle.loads(
            b"\x80\x04\x95\x17\x01\x00\x00\x00\x00\x00\x00X\x10\x01\x00\x00### Generate Media Latents\nThis node is where all the diffusion will take place, it dynamically populates based on the pipeline supplied. \n\nNote: \nIf `add noise` is disabled, the seed will have no effect. You will have to change the seed on the `Create Noise Latents` node\x94."
        ),
        "51fdc8e9-ebf8-4512-9ccb-abf260092e4e": pickle.loads(
            b"\x80\x04\x95_\x00\x00\x00\x00\x00\x00\x00\x8c[### Decode Media\nThis node takes any output latent and converts it to an image or a video. \x94."
        ),
    }
    # Create the Flow, then do work within it as context.
    flow0_name = (
        await GriptapeNodes.ahandle_request(
            CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
        )
    ).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="NoiseLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Create Noise Latents",
                    metadata={
                        "position": {"x": 1080, "y": 585.6666666666667},
                        "tempId": "placing-1779964813546-enk7vp",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Create",
                            description="Create an initial noise latent tensor for a selected 🧨 Diffusers pipeline.",
                            display_name="Create Noise Latents",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "NoiseLatentNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 460},
                        "category": "ModularDiffusion/Create",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(parameter_name="num_frames", ui_options={"hide": True}, initial_setup=True)
            )
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LatentDiffusionPipelineBuilderNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Modular Diffusion Pipeline Builder",
                    metadata={
                        "position": {"x": 335.00000000000006, "y": 585.6666666666667},
                        "tempId": "placing-1779964818220-alxx5",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Pipeline",
                            description="Build and cache latent-compatible 🧨 Diffusers Pipelines for reuse across latent execution nodes.",
                            display_name="Modular Diffusion Pipeline Builder",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "LatentDiffusionPipelineBuilderNode",
                        "showaddparameter": False,
                        "size": {"width": 601, "height": 552},
                        "category": "ModularDiffusion/Pipeline",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="pipeline_type",
                    tooltip="Type of diffusion pipeline to build",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"simple_dropdown": ["ZImagePipeline"], "show_search": True, "search_filter": ""},
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="model",
                    default_value="Tongyi-MAI/Z-Image-Turbo",
                    tooltip="model",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "simple_dropdown": ["Tongyi-MAI/Z-Image-Turbo"],
                        "show_search": True,
                        "search_filter": "",
                        "display_name": "model",
                    },
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="attention_slicing", ui_options={"hide": True}, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="vae_slicing", ui_options={"hide": True}, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="transformer_layerwise_casting", ui_options={"hide": True}, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="cpu_offload_strategy",
                    ui_options={
                        "simple_dropdown": ["None", "Model", "Sequential"],
                        "show_search": True,
                        "search_filter": "",
                        "hide": True,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="quantization_mode",
                    ui_options={
                        "simple_dropdown": ["None", "fp8", "int8", "int4"],
                        "show_search": True,
                        "search_filter": "",
                        "hide": True,
                    },
                    initial_setup=True,
                )
            )
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DiffusionPipelineGenerateLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Generate Media Latents (Modular Diffusion Pipeline)",
                    metadata={
                        "position": {"x": 2010, "y": 585.6666666666667},
                        "tempId": "placing-1779964858968-b27m",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Processing",
                            description="Generate latents via 🧨 Diffusers Pipelines.",
                            display_name="Generate Media Latents (Modular Diffusion Pipeline)",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "DiffusionPipelineGenerateLatentNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 824},
                        "category": "ModularDiffusion/Processing",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="prompt",
                    default_value="",
                    tooltip="The prompt or prompts to guide the image generation.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={},
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
                    initial_setup=True,
                )
            )
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input",
                    metadata={
                        "position": {"x": 1175.0000000000005, "y": 1090.0000000000002},
                        "tempId": "placing-1779964889986-r9iy82",
                        "library_node_metadata": NodeMetadata(
                            category="text",
                            description="TextInput node",
                            display_name="Text Input",
                            tags=None,
                            icon="text-cursor",
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "TextInput",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 236},
                        "category": "text",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeDecodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Decode Media",
                    metadata={
                        "position": {"x": 2800.000000000001, "y": 585.6666666666667},
                        "tempId": "placing-1779964950485-15mbmf",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Encode\\Decode",
                            description="Decode VAE latent images or video via 🧨 Diffusers Pipelines.",
                            display_name="Decode Media",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "VaeDecodeNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 436},
                        "category": "ModularDiffusion/Encode\\Decode",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="latent_tensor",
                    tooltip="Latent tensor to decode with the pipeline VAE.",
                    type="LatentArtifact",
                    input_types=["LatentArtifact"],
                    output_type="LatentArtifact",
                    ui_options={},
                    mode_allowed_property=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="output_image",
                    tooltip="Decoded image from the latent tensor.",
                    type="ImageArtifact",
                    input_types=["ImageArtifact"],
                    output_type="ImageArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    initial_setup=True,
                )
            )
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note",
                    metadata={
                        "position": {"x": 335.00000000000006, "y": 204.20578421631666},
                        "tempId": "placing-1779965174462-1ykfce",
                        "library_node_metadata": NodeMetadata(
                            category="misc",
                            description="Create a note node to provide helpful context in your workflow",
                            display_name="Note",
                            tags=None,
                            icon="notepad-text",
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 338},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_1",
                    metadata={
                        "position": {"x": 1080, "y": 204.20578421631666},
                        "tempId": "placing-1779965174462-1ykfce",
                        "library_node_metadata": NodeMetadata(
                            category="misc",
                            description="Create a note node to provide helpful context in your workflow",
                            display_name="Note",
                            tags=None,
                            icon="notepad-text",
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 299},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_2",
                    metadata={
                        "position": {"x": 2010, "y": 204.20578421631666},
                        "tempId": "placing-1779965174462-1ykfce",
                        "library_node_metadata": NodeMetadata(
                            category="misc",
                            description="Create a note node to provide helpful context in your workflow",
                            display_name="Note",
                            tags=None,
                            icon="notepad-text",
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 295},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_3",
                    metadata={
                        "position": {"x": 2800.000000000001, "y": 204.20578421631666},
                        "tempId": "placing-1779965174462-1ykfce",
                        "library_node_metadata": NodeMetadata(
                            category="misc",
                            description="Create a note node to provide helpful context in your workflow",
                            display_name="Note",
                            tags=None,
                            icon="notepad-text",
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 603, "height": 291},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="pipeline",
                target_node_name=node0_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="pipeline",
                target_node_name=node2_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="output_latent",
                target_node_name=node2_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="pipeline",
                target_node_name=node4_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="text",
                target_node_name=node2_name,
                target_parameter_name="prompt",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="output_latent",
                target_node_name=node4_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["6ce7092a-267f-4d3d-bf1b-c0db6545827f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["69eeb8fd-371c-4836-b883-0619a7ff7a18"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["03e0ca63-e4a7-4ae5-b2e9-01d49186cee3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["03e0ca63-e4a7-4ae5-b2e9-01d49186cee3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["df572ea8-0a34-401d-9e24-84350168a121"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["e422e707-236c-4162-b670-35f32300929a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["6ce7092a-267f-4d3d-bf1b-c0db6545827f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["6ce7092a-267f-4d3d-bf1b-c0db6545827f"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["3414702c-a9f1-4c46-87fd-6bdfb05bc9e2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ad144a6c-f157-40ed-a045-9fdd5c1e77c0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["683aa0ed-9200-495a-aa17-da8dd5cdb8ad"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["89c307a5-6d32-4e86-b433-fd29c0d61b50"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4e0a75a4-d349-4c12-bcbb-b8344f1092b9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4e0a75a4-d349-4c12-bcbb-b8344f1092b9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["9e014ef5-00b4-4a44-8c10-2377d7600fb0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["d0686a76-5442-40b0-8787-ebff1b281940"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["6ce7092a-267f-4d3d-bf1b-c0db6545827f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["1100f365-2eb2-4b41-b0d0-00f4f3cc0e14"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["91b42564-29c9-4901-bc75-0fa093035bc6"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["ec2a56f8-526b-4ad5-82f7-bf4f14628439"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["e422e707-236c-4162-b670-35f32300929a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["0acb3923-7a12-4200-92e5-fc9538a5ed59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["69eeb8fd-371c-4836-b883-0619a7ff7a18"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["131a647d-c3d9-4cb9-ac7c-a939259eb0d1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["a2bcdc65-b664-44bd-8d94-0fee5578c2f0"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["85de3f2b-1fc1-45da-ab06-80d88b3d9155"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["ec2a56f8-526b-4ad5-82f7-bf4f14628439"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["ec2a56f8-526b-4ad5-82f7-bf4f14628439"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["6ce7092a-267f-4d3d-bf1b-c0db6545827f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["533120f8-c085-4280-9685-11a38148ec55"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["eba87e6c-dc04-41da-b400-25993102f474"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["22886988-aba5-47c4-93e0-347ee7ef5a03"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["51fdc8e9-ebf8-4512-9ccb-abf260092e4e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
