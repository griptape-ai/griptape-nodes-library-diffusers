# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "Image2Image"
# schema_version = "0.18.0"
# engine_version_created_with = "0.83.0"
# node_libraries_referenced = [["Griptape Modular Diffusion Nodes Library", "0.1.0"], ["Griptape Nodes Library", "0.78.0"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Modular Diffusion Nodes Library", "VaeEncodeNode"], ["Griptape Nodes Library", "CompareImages"], ["Griptape Nodes Library", "LoadImage"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "TextInput"]]
# description = "I2I workflow using the Modular Diffusion Library Nodes"
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/Image2Image.webp"
# is_griptape_provided = true
# is_template = true
# creation_date = 2026-05-28T13:32:30.492359Z
# last_modified_date = 2026-06-01T08:10:37.889434Z
#
# ///

import pickle
from griptape.artifacts.image_url_artifact import ImageUrlArtifact
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
        "b9a41d3a-08bf-45df-9d89-653fef978612": pickle.loads(
            b"\x80\x04\x95\x1a\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0eZImagePipeline\x94\x8c\x0bconfig_hash\x94\x8cQZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\x94\x8c\x0f_builder_module\x94\x8cFmodular_diffusion_nodes_library.standard_parameters.z_image_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x18ZImagePipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94\x8c\rbase_revision\x94\x8c(f332072aa78be7aecdf3ee76d5c247082da564a6\x94u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h\x1eu\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x89\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "9569da73-4dd4-4ee3-972d-1f96f9f748b6": pickle.loads(
            b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07Z-Image\x94."
        ),
        "501b7463-0d34-4f7a-94e3-62c552d08808": pickle.loads(
            b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eZImagePipeline\x94."
        ),
        "62049c5d-c183-4f46-a038-90d32f2f8735": pickle.loads(
            b"\x80\x04\x95\x1c\x00\x00\x00\x00\x00\x00\x00\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94."
        ),
        "429f534b-0300-4acf-813e-9f3a61893314": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tAutomatic\x94."
        ),
        "4bf2e4c6-da16-4399-9c49-852e93c5d321": pickle.loads(b"\x80\x04\x89."),
        "ef816338-b6c5-4392-b437-30901b6dd8d9": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04None\x94."
        ),
        "ac859ac1-e20d-4888-a83f-906b31c5f101": pickle.loads(b"\x80\x04]\x94."),
        "e9dbc9b9-c9dc-4516-a064-f64450641334": pickle.loads(
            b"\x80\x04\x95\xc0\x01\x00\x00\x00\x00\x00\x00X\xb9\x01\x00\x00Building pipeline...\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nNo cached pipeline found. Building new pipeline.\nCreating new pipeline instance...\nLoading pipeline took 46.68 seconds\nConfiguring LoRAs took 0.00 milliseconds\nApplying optimizations took 207.13 milliseconds\nPipeline creation complete.\nPipeline building/caching took 46.89 seconds\nPipeline building complete.\n\x94."
        ),
        "382befde-96fa-40e8-8633-180c60f897e1": pickle.loads(b"\x80\x04\x88."),
        "f4433a15-314b-4ce8-8d12-3cacaeabc7e5": pickle.loads(b"\x80\x04K\x05."),
        "97d9408b-cd40-491b-bd46-8ee1760cf993": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xff\xff\xff\xff."
        ),
        "fd99bf09-a1be-425d-9295-3724028eaed8": pickle.loads(
            b"\x80\x04\x95A\x00\x00\x00\x00\x00\x00\x00\x8c=A red ferarri on the streets of italy, sports car, old street\x94."
        ),
        "3d722be3-8c96-4187-9617-c9653b3adbb4": pickle.loads(b"\x80\x04K\x14."),
        "ccf4470e-612f-4319-8f1d-45c146d04da9": pickle.loads(b"\x80\x04K*."),
        "eab84b4f-7b76-411b-8546-69dde952abdd": pickle.loads(b"\x80\x04]\x94."),
        "c162ed12-ae19-4988-91d7-190ef107742e": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xe8\x00\x00\x00\x00\x00\x00."
        ),
        "4b00678f-9969-4891-8a51-c99aabfa1ca2": pickle.loads(
            b"\x80\x04\x95{\x02\x00\x00\x00\x00\x00\x00Xt\x02\x00\x00Completed inference step 2 of 20. 1.72 s/it\nCompleted inference step 3 of 20. 1.73 s/it\nCompleted inference step 4 of 20. 1.76 s/it\nCompleted inference step 5 of 20. 1.78 s/it\nCompleted inference step 6 of 20. 1.79 s/it\nCompleted inference step 7 of 20. 1.80 s/it\nCompleted inference step 8 of 20. 1.79 s/it\nCompleted inference step 9 of 20. 1.79 s/it\nCompleted inference step 10 of 20. 1.79 s/it\nCompleted inference step 11 of 20. 1.79 s/it\nCompleted inference step 12 of 20. 1.79 s/it\nCompleted inference step 13 of 20. 1.79 s/it\nCompleted inference step 14 of 20. 1.79 s/it\nCompleted inference step 15 of 20. 1.79 s/it\nDone.\n\x94."
        ),
        "4845e13c-e0fb-47f6-86bf-64b820d7487c": pickle.loads(
            b"\x80\x04\x95\xc9\x01\x00\x00\x00\x00\x00\x00X\xc2\x01\x00\x00### Modular Diffusion Pipeline Builder\nJust like the `Diffusion Pipline Builder` from the `Griptape-Nodes-Advanced-Media-Library`, the only difference is that the models listed here are setup differently to allow for more advanced workflows\n\nNote: Please make sure you do not mix and match between the standard and the modular diffusion nodes as workflows will not work.\n\nDo not also mix and match model pipelines as they will produce garbage results\x94."
        ),
        "ccc94858-cea6-440c-8d81-21bd8c092d7d": pickle.loads(
            b"\x80\x04\x95'\x01\x00\x00\x00\x00\x00\x00X \x01\x00\x00### Encode Media\nThis node converts the supplied image from pixel space to latent space using the pipeline to make sure it is in the right tensor dimensions. \n\nNote: This node doesn't `add noise` to the latent_tensor so if you would like to add noise it would have to be in the next step.\x94."
        ),
        "2c99b756-a31f-4f28-b87f-32faec504413": pickle.loads(
            b'\x80\x04\x95\xf8\x01\x00\x00\x00\x00\x00\x00X\xf1\x01\x00\x00### Generate Media Latents\nThis node is where all the diffusion will take place, it dynamically populates based on the pipeline supplied. In this specific usecase the `start_step` is more than 0 meaning that more of the original image will be used in the diffusion process. The higher the `start_step` the less changes to the overall original image.\n\nNote: \nIf `add noise` is disabled, the seed will have no effect and the output will be rubbish as there will be nothing for the model to "denoise"\x94.'
        ),
        "b282fac2-8c0c-4214-8124-224ea892f67f": pickle.loads(
            b"\x80\x04\x95_\x00\x00\x00\x00\x00\x00\x00\x8c[### Decode Media\nThis node takes any output latent and converts it to an image or a video. \x94."
        ),
        "8f265ace-153b-4d6c-9684-115bb1fda3e4": pickle.loads(
            b"\x80\x04\x95u\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c f50675a96ba948e1b5c2e32c2fb41ac5\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0a\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8cX{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/orange_car.jpg\x94ub."
        ),
        "3740dcbc-a9dd-44f7-82b1-9f4d7bf8003b": pickle.loads(
            b"\x80\x04\x95u\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 62f627d322bb499aa517d680465bd002\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0a\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8cX{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/orange_car.jpg\x94ub."
        ),
        "9327bbc7-a049-4148-9231-c207a3f88999": pickle.loads(
            b"\x80\x04\x95\x5c\x00\x00\x00\x00\x00\x00\x00\x8cX{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/orange_car.jpg\x94."
        ),
        "da237655-ad3e-4378-8624-92713301e9f4": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04none\x94."
        ),
        "27f240ed-6cd0-4955-a202-a92d856cecae": pickle.loads(
            b"\x80\x04\x950\x00\x00\x00\x00\x00\x00\x00\x8c,<Results will appear when the node executes>\x94."
        ),
        "2cb47618-ec2e-44f0-8742-87f80101c39b": pickle.loads(
            b"\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08mask.png\x94."
        ),
        "19dd5ea0-aa38-4098-bdcf-2f5544446aa4": pickle.loads(
            b"\x80\x04\x95\xfd\x01\x00\x00\x00\x00\x00\x00}\x94(\x8c\x0dinput_image_1\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 48d82f1bfb5340aeaaac4c09ea98f6a9\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0c\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x15{outputs}/image_5.png\x94ub\x8c\x0dinput_image_2\x94h\x04)\x81\x94}\x94(h\x07h\x08h\x09h\x0ah\x0b\x8c 62f627d322bb499aa517d680465bd002\x94h\x0dNh\x0e}\x94h\x10h\x1ah\x11h\x12h\x13h\x14h\x15\x8cX{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/orange_car.jpg\x94ubu."
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
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
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
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DiffusionPipelineGenerateLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Generate Media Latents (Modular Diffusion Pipeline)",
                    metadata={
                        "position": {"x": 2010, "y": 585.6666666666667},
                        "tempId": "placing-1779964858968-b27m",
                        "library_node_metadata": {
                            "category": "ModularDiffusion/Processing",
                            "description": "Generate latents via 🧨 Diffusers Pipelines.",
                        },
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "DiffusionPipelineGenerateLatentNode",
                        "showaddparameter": False,
                        "size": {"width": 601, "height": 916},
                        "category": "ModularDiffusion/Processing",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
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
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input",
                    metadata={
                        "position": {"x": 1080, "y": 1055.514619322945},
                        "tempId": "placing-1779964889986-r9iy82",
                        "library_node_metadata": {"category": "text", "description": "TextInput node"},
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
        node3_name = (
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
        with GriptapeNodes.ContextManager().node(node3_name):
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
        node4_name = (
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
        node5_name = (
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
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_2",
                    metadata={
                        "position": {"x": 2010, "y": 204.20578421631666},
                        "tempId": "placing-1779965174462-1ykfce",
                        "library_node_metadata": {
                            "category": "misc",
                            "description": "Create a note node to provide helpful context in your workflow",
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 601, "height": 303},
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
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadImage",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Load Image",
                    metadata={
                        "position": {"x": 335.00000000000006, "y": 1269.629342076291},
                        "tempId": "placing-1780299347598-6z5dwh",
                        "library_node_metadata": {"category": "image", "description": "Loads an image from disk"},
                        "library": "Griptape Nodes Library",
                        "node_type": "LoadImage",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 540},
                        "category": "image",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeEncodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Encode Media",
                    metadata={
                        "position": {"x": 1080, "y": 585.6666666666667},
                        "tempId": "placing-1780299414389-yv7tt",
                        "library_node_metadata": {
                            "category": "ModularDiffusion/Encode\\Decode",
                            "description": "Encode images or video into VAE latent space via 🧨 Diffusers Pipelines.",
                        },
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "VaeEncodeNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 436},
                        "category": "ModularDiffusion/Encode\\Decode",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image",
                    tooltip="Input image to encode into VAE latent space.",
                    type="ImageArtifact",
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    output_type="ImageArtifact",
                    ui_options={},
                    mode_allowed_property=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CompareImages",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Compare Images",
                    metadata={
                        "position": {"x": 3588.2906490928776, "y": 585.6666666666667},
                        "tempId": "placing-1780300398466-dpdr77",
                        "library_node_metadata": {
                            "category": "image",
                            "description": "Can be used to compare two images",
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "CompareImages",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 588},
                        "category": "image",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="pipeline",
                target_node_name=node3_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="pipeline",
                target_node_name=node9_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node8_name,
                source_parameter_name="image",
                target_node_name=node9_name,
                target_parameter_name="image",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node9_name,
                source_parameter_name="latent_tensor",
                target_node_name=node1_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node9_name,
                source_parameter_name="pipeline",
                target_node_name=node1_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="output_latent",
                target_node_name=node3_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="text",
                target_node_name=node1_name,
                target_parameter_name="prompt",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="output_image",
                target_node_name=node10_name,
                target_parameter_name="Image_1",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node8_name,
                source_parameter_name="image",
                target_node_name=node10_name,
                target_parameter_name="Image_2",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["b9a41d3a-08bf-45df-9d89-653fef978612"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["b9a41d3a-08bf-45df-9d89-653fef978612"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["9569da73-4dd4-4ee3-972d-1f96f9f748b6"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["501b7463-0d34-4f7a-94e3-62c552d08808"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["62049c5d-c183-4f46-a038-90d32f2f8735"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["429f534b-0300-4acf-813e-9f3a61893314"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["ef816338-b6c5-4392-b437-30901b6dd8d9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["ef816338-b6c5-4392-b437-30901b6dd8d9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["ac859ac1-e20d-4888-a83f-906b31c5f101"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["e9dbc9b9-c9dc-4516-a064-f64450641334"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["b9a41d3a-08bf-45df-9d89-653fef978612"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["382befde-96fa-40e8-8633-180c60f897e1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["f4433a15-314b-4ce8-8d12-3cacaeabc7e5"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["97d9408b-cd40-491b-bd46-8ee1760cf993"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["fd99bf09-a1be-425d-9295-3724028eaed8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["3d722be3-8c96-4187-9617-c9653b3adbb4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ccf4470e-612f-4319-8f1d-45c146d04da9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["eab84b4f-7b76-411b-8546-69dde952abdd"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["c162ed12-ae19-4988-91d7-190ef107742e"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4b00678f-9969-4891-8a51-c99aabfa1ca2"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["fd99bf09-a1be-425d-9295-3724028eaed8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["fd99bf09-a1be-425d-9295-3724028eaed8"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["b9a41d3a-08bf-45df-9d89-653fef978612"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["4845e13c-e0fb-47f6-86bf-64b820d7487c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["ccc94858-cea6-440c-8d81-21bd8c092d7d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["2c99b756-a31f-4f28-b87f-32faec504413"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["b282fac2-8c0c-4214-8124-224ea892f67f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["8f265ace-153b-4d6c-9684-115bb1fda3e4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["3740dcbc-a9dd-44f7-82b1-9f4d7bf8003b"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["9327bbc7-a049-4148-9231-c207a3f88999"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["9327bbc7-a049-4148-9231-c207a3f88999"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mask_channel",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["da237655-ad3e-4378-8624-92713301e9f4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["4bf2e4c6-da16-4399-9c49-852e93c5d321"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["27f240ed-6cd0-4955-a202-a92d856cecae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["27f240ed-6cd0-4955-a202-a92d856cecae"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mask_output_file",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["2cb47618-ec2e-44f0-8742-87f80101c39b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["b9a41d3a-08bf-45df-9d89-653fef978612"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["3740dcbc-a9dd-44f7-82b1-9f4d7bf8003b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="Image_2",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["3740dcbc-a9dd-44f7-82b1-9f4d7bf8003b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="Compare",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["19dd5ea0-aa38-4098-bdcf-2f5544446aa4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="Compare",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["19dd5ea0-aa38-4098-bdcf-2f5544446aa4"],
                    initial_setup=True,
                    is_output=True,
                )
            )
