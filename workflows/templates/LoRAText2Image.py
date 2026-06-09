# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "LoRAText2Image"
# schema_version = "0.18.0"
# engine_version_created_with = "0.83.0"
# node_libraries_referenced = [["Griptape Modular Diffusion Nodes Library", "0.1.0"], ["Griptape Nodes Library", "0.78.0"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "LoadLora"], ["Griptape Modular Diffusion Nodes Library", "NoiseLatentNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "TextInput"]]
# description = "LoRA assisted T2I workflow using the Modular Diffusion Library Nodes"
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/LoRAText2Image.webp"
# is_griptape_provided = true
# is_template = true
# creation_date = 2026-06-02T11:25:09.945711Z
# last_modified_date = 2026-06-02T11:27:54.321902Z
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
        "87a60ee9-e93d-4732-8623-156672d313bf": pickle.loads(
            b"\x80\x04\x95\x1a\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0eZImagePipeline\x94\x8c\x0bconfig_hash\x94\x8cQZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\x94\x8c\x0f_builder_module\x94\x8cFmodular_diffusion_nodes_library.standard_parameters.z_image_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x18ZImagePipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94\x8c\rbase_revision\x94\x8c(f332072aa78be7aecdf3ee76d5c247082da564a6\x94u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h\x1eu\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x89\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "ea00b0ee-f23e-4bf1-aacd-46fe17357312": pickle.loads(b"\x80\x04\x89."),
        "03a1e568-fbdc-4053-a1eb-1f8ab6abf0e9": pickle.loads(b"\x80\x04K*."),
        "a388271f-ce7a-4215-990a-4fd603877087": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x04."),
        "96ecf226-73ad-451e-a76f-07da4a5c9290": pickle.loads(b"\x80\x04K)."),
        "41cb4d01-63af-4488-923d-2ebe165f7fc4": pickle.loads(b"\x80\x04K\x14."),
        "00bcd269-812b-4436-8bbb-7704d95a8586": pickle.loads(
            b"\x80\x04\x95(\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0eZImagePipeline\x94\x8c\x0bconfig_hash\x94\x8cQZImagePipeline-e926eec1ee70a4bffa7e2abb95ea4e422bfb5df85ee101ccd5dd2ca0af63e3b5-0\x94\x8c\x0f_builder_module\x94\x8cFmodular_diffusion_nodes_library.standard_parameters.z_image_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x18ZImagePipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94\x8c\rbase_revision\x94\x8c(f332072aa78be7aecdf3ee76d5c247082da564a6\x94u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h\x1fu\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x89\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "1a136e5e-85bf-4e25-8fa7-06f8e9116f7d": pickle.loads(
            b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07Z-Image\x94."
        ),
        "a638a0ef-7fb7-4433-81bf-e9f9899aae1e": pickle.loads(
            b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eZImagePipeline\x94."
        ),
        "9c1d6b33-21fb-49f2-b817-c0dbe91a8df9": pickle.loads(
            b"\x80\x04\x95\x1c\x00\x00\x00\x00\x00\x00\x00\x8c\x18Tongyi-MAI/Z-Image-Turbo\x94."
        ),
        "99988cce-e3cc-4ce8-85f4-491d4f0fe72d": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tAutomatic\x94."
        ),
        "0fa38b83-f8bf-41cf-9ef1-cf0d7b2bc6df": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04None\x94."
        ),
        "a4613396-9bce-4746-a418-3f4efefd7a10": pickle.loads(
            b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00]\x94(}\x94}\x94e."
        ),
        "c0a0b8a3-dcf0-458b-afe0-d677a9f52917": pickle.loads(b"\x80\x04}\x94."),
        "b7831946-1145-45eb-bac0-5b4284173fbe": pickle.loads(b"\x80\x04}\x94."),
        "d0547b89-086a-45e1-8ff5-16706d45f837": pickle.loads(
            b"\x80\x04\x95r\x0c\x00\x00\x00\x00\x00\x00Xk\x0c\x00\x00Building pipeline...\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nNo cached pipeline found. Building new pipeline.\nCreating new pipeline instance...\nLoading pipeline took 32.14 seconds\nConfiguring LoRAs took 0.00 milliseconds\nApplying optimizations took 99.71 milliseconds\nPipeline creation complete.\nPipeline building/caching took 32.24 seconds\nPipeline building complete.\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: LTXPipeline-f5b01a2e40e29232c5a85a4789a3ec7eb6daaa8ea709d1da160b456cb91e4b09-0\nPipeline configuration hash: LTXPipeline-f5b01a2e40e29232c5a85a4789a3ec7eb6daaa8ea709d1da160b456cb91e4b09-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-da779454ca5607a327614a7e8ed65785d2c7a9d16d44e70c62ed7be82ca0b1ef-0\nPipeline configuration hash: WanPipeline-210ad82cdc82aab9187d2f9d8f08ccb822fdc988b86932a23b90858f733abb4d-0\nPipeline configuration hash: WanPipeline-210ad82cdc82aab9187d2f9d8f08ccb822fdc988b86932a23b90858f733abb4d-0\nPipeline configuration hash: WanPipeline-210ad82cdc82aab9187d2f9d8f08ccb822fdc988b86932a23b90858f733abb4d-0\nPipeline configuration hash: WanPipeline-210ad82cdc82aab9187d2f9d8f08ccb822fdc988b86932a23b90858f733abb4d-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\nPipeline configuration hash: ZImagePipeline-a250e83e223278501a757394e50a83c85ed3682b0adc9a23c67ab93f79fbacbe-0\n\x94."
        ),
        "b79af807-42df-4351-aa04-ec24d4f97758": pickle.loads(b"\x80\x04K\x00."),
        "e95ed38e-5077-47bb-84af-005724533aa9": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xff\xff\xff\xff."
        ),
        "c9fa4df8-25fc-4894-b4c5-de4596c2fbf0": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "f5a8b119-b9d4-4e83-85d8-0cfbf7f18766": pickle.loads(b"\x80\x04]\x94."),
        "b60b98b8-1447-4f2a-86d9-a7fe33acbcfa": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00."
        ),
        "48a560bd-aba5-4fdc-b55d-0a8787149e97": pickle.loads(
            b"\x80\x04\x95\\\x03\x00\x00\x00\x00\x00\x00XU\x03\x00\x00Completed inference step 2 of 20. 1.14 s/it\nCompleted inference step 3 of 20. 1.14 s/it\nCompleted inference step 4 of 20. 1.14 s/it\nCompleted inference step 5 of 20. 1.14 s/it\nCompleted inference step 6 of 20. 1.14 s/it\nCompleted inference step 7 of 20. 1.14 s/it\nCompleted inference step 8 of 20. 1.14 s/it\nCompleted inference step 9 of 20. 1.14 s/it\nCompleted inference step 10 of 20. 1.14 s/it\nCompleted inference step 11 of 20. 1.14 s/it\nCompleted inference step 12 of 20. 1.14 s/it\nCompleted inference step 13 of 20. 1.14 s/it\nCompleted inference step 14 of 20. 1.15 s/it\nCompleted inference step 15 of 20. 1.15 s/it\nCompleted inference step 16 of 20. 1.15 s/it\nCompleted inference step 17 of 20. 1.15 s/it\nCompleted inference step 18 of 20. 1.15 s/it\nCompleted inference step 19 of 20. 1.15 s/it\nCompleted inference step 20 of 20. 1.15 s/it\nDone.\n\x94."
        ),
        "99539809-379c-4450-baf1-7a1da015b5d0": pickle.loads(
            b"\x80\x04\x952\x00\x00\x00\x00\x00\x00\x00\x8c.A calico cat on a cobble stone wall, minecraft\x94."
        ),
        "9e5bc547-145d-4a44-97c9-0a287a4e7522": pickle.loads(
            b"\x80\x04\x95\xc9\x01\x00\x00\x00\x00\x00\x00X\xc2\x01\x00\x00### Modular Diffusion Pipeline Builder\nJust like the `Diffusion Pipline Builder` from the `Griptape-Nodes-Advanced-Media-Library`, the only difference is that the models listed here are setup differently to allow for more advanced workflows\n\nNote: Please make sure you do not mix and match between the standard and the modular diffusion nodes as workflows will not work.\n\nDo not also mix and match model pipelines as they will produce garbage results\x94."
        ),
        "ede63f7e-a235-4971-9bc5-ab5a7a84a103": pickle.loads(
            b"\x80\x04\x95\xb3\x00\x00\x00\x00\x00\x00\x00\x8c\xaf### Create Noise Latents\nThis node creates an empty latent and fills it the pipelines' expected noise pattern based on the scheduling options supplied in the model repository.\x94."
        ),
        "9cb7a9a6-216f-4d1f-8741-a6a1fe08f81a": pickle.loads(
            b"\x80\x04\x95\x17\x01\x00\x00\x00\x00\x00\x00X\x10\x01\x00\x00### Generate Media Latents\nThis node is where all the diffusion will take place, it dynamically populates based on the pipeline supplied. \n\nNote: \nIf `add noise` is disabled, the seed will have no effect. You will have to change the seed on the `Create Noise Latents` node\x94."
        ),
        "af44c5a8-7466-4283-ba38-5aa2117c82ca": pickle.loads(
            b"\x80\x04\x95_\x00\x00\x00\x00\x00\x00\x00\x8c[### Decode Media\nThis node takes any output latent and converts it to an image or a video. \x94."
        ),
        "c59e4e82-2d67-4f99-929a-7ca1786bb923": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xe8\x00\x00\x00\x00\x00\x00."
        ),
        "c3787676-43c5-4182-942f-278a946a3f20": pickle.loads(
            b"\x80\x04\x95i\x00\x00\x00\x00\x00\x00\x00\x8ce### Load LoRa\nSelect a LoRA that is compatible with your model and guide your diffusion with a theme.\x94."
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
                    tooltip="Specific pipeline variant within the selected provider (e.g. base, Fill, Edit). Determines which checkpoints and runtime parameters are exposed.",
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
                AlterParameterDetailsRequest(parameter_name="vae_tiling", ui_options={"hide": True}, initial_setup=True)
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
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="loras_ParameterListUniqueParamID_19b5c8e52332437f9f335a75ba988fea",
                    default_value=[],
                    tooltip="loras",
                    type="loras",
                    input_types=["loras", "dict"],
                    output_type="loras",
                    ui_options={},
                    mode_allowed_property=False,
                    parent_container_name="loras",
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="loras_ParameterListUniqueParamID_40b26117720e4e9ab2f691911b84a392",
                    default_value=[],
                    tooltip="loras",
                    type="loras",
                    input_types=["loras", "dict"],
                    output_type="loras",
                    ui_options={},
                    mode_allowed_property=False,
                    parent_container_name="loras",
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
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadLora",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Load LoRA",
                    metadata={
                        "position": {"x": -653.9980879541114, "y": 585.6666666666667},
                        "tempId": "placing-1780310228732-oghtdw",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Pipeline",
                            description="Load a LoRA file from a local path and expose it as a `loras` output for use with the Modular Diffusion Pipeline Builder.",
                            display_name="Load LoRA",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "LoadLora",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 284},
                        "category": "ModularDiffusion/Pipeline",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadLora",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Load LoRA_1",
                    metadata={
                        "position": {"x": -653.9980879541114, "y": 869.6666666666667},
                        "tempId": "placing-1780310239619-4pp7n9",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Pipeline",
                            description="Load a LoRA file from a local path and expose it as a `loras` output for use with the Modular Diffusion Pipeline Builder.",
                            display_name="Load LoRA",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "LoadLora",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 284},
                        "category": "ModularDiffusion/Pipeline",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_4",
                    metadata={
                        "position": {"x": -653.9980879541114, "y": 334.5767641659942},
                        "tempId": "placing-1780310371498-liw5t9",
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
                        "size": {"width": 600, "height": 192},
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
                source_node_name=node9_name,
                source_parameter_name="loras",
                target_node_name=node1_name,
                target_parameter_name="loras_ParameterListUniqueParamID_19b5c8e52332437f9f335a75ba988fea",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="loras",
                target_node_name=node1_name,
                target_parameter_name="loras_ParameterListUniqueParamID_40b26117720e4e9ab2f691911b84a392",
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
                    value=top_level_unique_values_dict["87a60ee9-e93d-4732-8623-156672d313bf"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["03a1e568-fbdc-4053-a1eb-1f8ab6abf0e9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["a388271f-ce7a-4215-990a-4fd603877087"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["a388271f-ce7a-4215-990a-4fd603877087"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["96ecf226-73ad-451e-a76f-07da4a5c9290"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["41cb4d01-63af-4488-923d-2ebe165f7fc4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["00bcd269-812b-4436-8bbb-7704d95a8586"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["00bcd269-812b-4436-8bbb-7704d95a8586"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["1a136e5e-85bf-4e25-8fa7-06f8e9116f7d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["a638a0ef-7fb7-4433-81bf-e9f9899aae1e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["9c1d6b33-21fb-49f2-b817-c0dbe91a8df9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["99988cce-e3cc-4ce8-85f4-491d4f0fe72d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_tiling",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["0fa38b83-f8bf-41cf-9ef1-cf0d7b2bc6df"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["0fa38b83-f8bf-41cf-9ef1-cf0d7b2bc6df"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["a4613396-9bce-4746-a418-3f4efefd7a10"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras_ParameterListUniqueParamID_19b5c8e52332437f9f335a75ba988fea",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["c0a0b8a3-dcf0-458b-afe0-d677a9f52917"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras_ParameterListUniqueParamID_40b26117720e4e9ab2f691911b84a392",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["b7831946-1145-45eb-bac0-5b4284173fbe"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["d0547b89-086a-45e1-8ff5-16706d45f837"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["87a60ee9-e93d-4732-8623-156672d313bf"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["b79af807-42df-4351-aa04-ec24d4f97758"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["e95ed38e-5077-47bb-84af-005724533aa9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["c9fa4df8-25fc-4894-b4c5-de4596c2fbf0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["41cb4d01-63af-4488-923d-2ebe165f7fc4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["ea00b0ee-f23e-4bf1-aacd-46fe17357312"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["03a1e568-fbdc-4053-a1eb-1f8ab6abf0e9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["f5a8b119-b9d4-4e83-85d8-0cfbf7f18766"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["b60b98b8-1447-4f2a-86d9-a7fe33acbcfa"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["48a560bd-aba5-4fdc-b55d-0a8787149e97"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["c9fa4df8-25fc-4894-b4c5-de4596c2fbf0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["99539809-379c-4450-baf1-7a1da015b5d0"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["87a60ee9-e93d-4732-8623-156672d313bf"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["9e5bc547-145d-4a44-97c9-0a287a4e7522"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["ede63f7e-a235-4971-9bc5-ab5a7a84a103"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["9cb7a9a6-216f-4d1f-8741-a6a1fe08f81a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["af44c5a8-7466-4283-ba38-5aa2117c82ca"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="weight",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["c59e4e82-2d67-4f99-929a-7ca1786bb923"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["c0a0b8a3-dcf0-458b-afe0-d677a9f52917"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="trigger_phrase",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["c9fa4df8-25fc-4894-b4c5-de4596c2fbf0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="weight",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["c59e4e82-2d67-4f99-929a-7ca1786bb923"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["b7831946-1145-45eb-bac0-5b4284173fbe"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="trigger_phrase",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["c9fa4df8-25fc-4894-b4c5-de4596c2fbf0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["c3787676-43c5-4182-942f-278a946a3f20"],
                    initial_setup=True,
                    is_output=False,
                )
            )
