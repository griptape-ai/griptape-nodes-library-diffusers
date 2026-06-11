# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "FirstAndLastFrameImage2Video"
# schema_version = "0.18.0"
# engine_version_created_with = "0.83.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.78.0"], ["Griptape Modular Diffusion Nodes Library", "0.1.0"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "LatentUpsamplerNode"], ["Griptape Modular Diffusion Nodes Library", "MediaGenConditioningNode"], ["Griptape Modular Diffusion Nodes Library", "NoiseLatentNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Nodes Library", "Group"], ["Griptape Nodes Library", "IntegerInput"], ["Griptape Nodes Library", "LoadImage"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "RescaleImage"], ["Griptape Nodes Library", "TextInput"]]
# description = "First and Last Frame I2V workflow using the Modular Diffusion Library Nodes"
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/FirstAndLastFrameImage2Video.gif"
# is_griptape_provided = true
# is_template = true
# creation_date = 2026-06-02T10:39:56.644559Z
# last_modified_date = 2026-06-02T10:51:57.684579Z
#
# ///

import pickle
from griptape.artifacts.image_url_artifact import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
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
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
    )
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(
            library_name="Griptape Modular Diffusion Nodes Library", perform_discovery_if_not_found=True
        )
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
        "a892317d-e2fa-40a1-b0b8-01bebc44addb": pickle.loads(
            b"\x80\x04\x95A\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x17WanImageToVideoPipeline\x94\x8c\x0bconfig_hash\x94\x8cZWanImageToVideoPipeline-beee334b7caa70e05dbec720d1c196fb0236ce0100a6ef4fbec7a68cc088d2a2-0\x94\x8c\x0f_builder_module\x94\x8cFmodular_diffusion_nodes_library.standard_parameters.wan_i2v_parameters\x94\x8c\x13_builder_class_name\x94\x8c!WanImageToVideoPipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x07repo_id\x94\x8c Wan-AI/Wan2.2-I2V-A14B-Diffusers\x94\x8c\x08revision\x94\x8c(596658fd9ca6b7b71d5057529bbf319ecbc61d74\x94u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h\x1fu\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x88ub."
        ),
        "e8a55714-10d8-4bad-a597-51d36f24c205": pickle.loads(b"\x80\x04\x89."),
        "7537d046-6ec1-4b8d-ac5c-8413a226a53b": pickle.loads(b"\x80\x04K*."),
        "456e5f78-7efa-4698-86f8-bcc67ba0422f": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x05."),
        "5f19c64b-b5b4-4a4a-a0ce-2215b4299bde": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\xd0\x02."),
        "0aa65c92-583a-4035-8758-9e4fcd15c70a": pickle.loads(b"\x80\x04KQ."),
        "f00c67c7-f72e-4a38-a386-ce4147a0fcb1": pickle.loads(b"\x80\x04K\x14."),
        "bdc4fb33-509f-45d7-b66f-6901c1c68661": pickle.loads(
            b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03WAN\x94."
        ),
        "e2bf76c0-12f6-48ff-99d2-f5e263058601": pickle.loads(
            b"\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x17WanImageToVideoPipeline\x94."
        ),
        "50227083-d069-4397-ae82-88629804ad58": pickle.loads(
            b"\x80\x04\x95$\x00\x00\x00\x00\x00\x00\x00\x8c Wan-AI/Wan2.2-I2V-A14B-Diffusers\x94."
        ),
        "ebde2e93-795f-43f6-9262-08ec0dd64906": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tAutomatic\x94."
        ),
        "97576349-86c8-4797-b696-d18370b741c0": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04None\x94."
        ),
        "06ca7d08-18ad-46d5-af16-9dfaf9e2689e": pickle.loads(b"\x80\x04]\x94."),
        "e11f2893-64ea-4a42-b9c3-79631d9fd38f": pickle.loads(
            b"\x80\x04\x95\xc6\x01\x00\x00\x00\x00\x00\x00X\xbf\x01\x00\x00Building pipeline...\nPipeline configuration hash: WanImageToVideoPipeline-beee334b7caa70e05dbec720d1c196fb0236ce0100a6ef4fbec7a68cc088d2a2-0\nNo cached pipeline found. Building new pipeline.\nCreating new pipeline instance...\nLoading pipeline took 1.69 minutes\nConfiguring LoRAs took 0.00 milliseconds\nApplying optimizations took 23.20 milliseconds\nPipeline creation complete.\nPipeline building/caching took 1.69 minutes\nPipeline building complete.\n\x94."
        ),
        "132c418e-9643-4f92-bbba-b2a426326fe3": pickle.loads(b"\x80\x04K\x19."),
        "c1821f3d-46e3-45a4-98a2-55ea5152c134": pickle.loads(
            b"\x80\x04\x95G\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.video_url_artifact\x94\x8c\x10VideoUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94h\x01\x8c\x0bmodule_name\x94h\x00\x8c\x02id\x94\x8c 396a23596f6e4d31bec1276ed856cfea\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x08\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8cahttp://localhost:8124/workspace/staticfiles/256b705f-9322-4c5f-b94a-e30603bc9623.mp4?t=1780397360\x94ub."
        ),
        "74618a79-e1bc-4ad3-80a4-260c35394ae5": pickle.loads(
            b"\x80\x04\x95\xc9\x01\x00\x00\x00\x00\x00\x00X\xc2\x01\x00\x00### Modular Diffusion Pipeline Builder\nJust like the `Diffusion Pipline Builder` from the `Griptape-Nodes-Advanced-Media-Library`, the only difference is that the models listed here are setup differently to allow for more advanced workflows\n\nNote: Please make sure you do not mix and match between the standard and the modular diffusion nodes as workflows will not work.\n\nDo not also mix and match model pipelines as they will produce garbage results\x94."
        ),
        "7b867c28-48c2-49d1-b779-158a551350ae": pickle.loads(
            b"\x80\x04\x95\xb3\x00\x00\x00\x00\x00\x00\x00\x8c\xaf### Create Noise Latents\nThis node creates an empty latent and fills it the pipelines' expected noise pattern based on the scheduling options supplied in the model repository.\x94."
        ),
        "63ababb5-f635-49f8-8ab1-97a5925e978f": pickle.loads(
            b"\x80\x04\x95\x97\x01\x00\x00\x00\x00\x00\x00X\x90\x01\x00\x00### Generate Media Latents\nThis node is where all the diffusion will take place, it dynamically populates based on the pipeline supplied. \n\nIn this workflow, there is a requirement to plugin the Media Generation Conditioning node in the additional parameters to work\n\nNote: \nIf `add noise` is disabled, the seed will have no effect. You will have to change the seed on the `Create Noise Latents` node\x94."
        ),
        "a6f51233-992b-492d-b2f3-79ac9d1ba6a8": pickle.loads(
            b"\x80\x04\x95_\x00\x00\x00\x00\x00\x00\x00\x8c[### Decode Media\nThis node takes any output latent and converts it to an image or a video. \x94."
        ),
        "67f3dd75-9d58-4b7b-a708-57f5338336f6": pickle.loads(
            b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05image\x94."
        ),
        "b1baed1e-71c1-48e5-9bfc-8f53f2560984": pickle.loads(b"\x80\x04K\x02."),
        "c1764a7b-6282-4b91-9311-de88d14d1218": pickle.loads(
            b"\x80\x04\x95@\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 86761c177c2a41e7ac7fe5a44bc303c8\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\n\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c#{outputs}/Resize Image_output_6.png\x94ub."
        ),
        "cdf16e89-a770-4351-be51-2d9a18338190": pickle.loads(b"\x80\x04K\x00."),
        "cf30316e-ffbb-448f-b5c7-50a8b9c5a7a2": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00."
        ),
        "440641d3-27ca-4234-8e8c-f561b044bfe4": pickle.loads(
            b"\x80\x04\x95\x0c\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94h\x01\x8c\x0bmodule_name\x94h\x00\x8c\x02id\x94\x8c e4cae25f85834e59aa49c9e5993a3128\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x08\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c&{outputs}/Resize Image_1_output_28.png\x94ub."
        ),
        "3bb76746-cb74-4c2c-a977-78bfcd4fe834": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xff\xff\xff\xff."
        ),
        "8624ee43-f3a5-4a44-a991-542e5ea8d444": pickle.loads(
            b"\x80\x04\x95y\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 90e2ed4dc276465695ef1578b1da996d\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0a\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x5c{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0001.png\x94ub."
        ),
        "abe01153-519c-41a9-8e29-521795a1efcb": pickle.loads(
            b"\x80\x04\x95y\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 449736dce98c465e9fbbbd1326cffa54\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0a\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x5c{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0001.png\x94ub."
        ),
        "7605646b-831f-4677-8135-d19a036c2d3a": pickle.loads(
            b"\x80\x04\x95`\x00\x00\x00\x00\x00\x00\x00\x8c\x5c{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0001.png\x94."
        ),
        "29eff951-17cc-415d-aded-9db01c00049e": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04none\x94."
        ),
        "2bfa74fe-9db3-408e-b8e6-202a8c763163": pickle.loads(
            b"\x80\x04\x950\x00\x00\x00\x00\x00\x00\x00\x8c,<Results will appear when the node executes>\x94."
        ),
        "6570eb60-0f30-424b-b2b0-5d46ec4be206": pickle.loads(
            b"\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08mask.png\x94."
        ),
        "cfbd5a76-6cf2-489f-a0f3-e99c421a8f91": pickle.loads(
            b"\x80\x04\x95\x14\x00\x00\x00\x00\x00\x00\x00\x8c\x10width and height\x94."
        ),
        "59a0322c-bc33-49f2-8437-8ff19f3ae0c1": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\xe8\x03."),
        "23675c0f-6790-492e-86c8-c47d220fb92b": pickle.loads(b"\x80\x04Kd."),
        "cb6aa17f-402a-4f36-a839-4fbad879765c": pickle.loads(
            b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03fit\x94."
        ),
        "cb2c224a-5293-49a5-8d87-e262b8aefd38": pickle.loads(
            b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07#000000\x94."
        ),
        "0e9806cb-0282-4fea-81c6-ed7e26cd90b8": pickle.loads(
            b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07lanczos\x94."
        ),
        "1e0300a4-124c-4cde-b091-b01e7a9cf740": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04auto\x94."
        ),
        "5ad0e916-d32c-44b8-b363-89ccad81e262": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00\x8c\x0295\x94."
        ),
        "6f1ab114-fe3a-4ef1-938d-36265d693542": pickle.loads(
            b"\x80\x04\x95\x0e\x00\x00\x00\x00\x00\x00\x00\x8c\noutput.png\x94."
        ),
        "e2dea525-e3fb-4878-901d-7a0bd19f9847": pickle.loads(
            b"\x80\x04\x95\xf4\x00\x00\x00\x00\x00\x00\x00\x8c\xf0Detected image format: PNG\nProcessing image: 3840x2160, mode: RGB, format: PNG\n[Processing image rescaling..]\n[Started image processing..]\nimage rescaling\nSuccessfully processed image with suffix: _rescaled.png\n[Finished image processing.]\n\x94."
        ),
        "1d41add8-79fa-47e8-bf66-c2ff75172cb1": pickle.loads(
            b"\x80\x04\x95y\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c eed3767596e54f598aa15796c01f70b5\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0a\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x5c{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0136.png\x94ub."
        ),
        "8072c7b4-cd79-4bbe-991e-0f875863bd35": pickle.loads(
            b"\x80\x04\x95y\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c a9ef1123e92641b1968e438afb4d41fe\x94\x8c\x09reference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0a\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x5c{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0136.png\x94ub."
        ),
        "1441eaf2-deda-4a3f-9a7d-bb7f4151bdc7": pickle.loads(
            b"\x80\x04\x95`\x00\x00\x00\x00\x00\x00\x00\x8c\x5c{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0136.png\x94."
        ),
        "53aac130-e307-4afc-a82c-fcab34289844": pickle.loads(b"\x80\x04\x88."),
        "13d2c731-a03f-4f4c-ac6d-f6347dd6c5fc": pickle.loads(
            b"\x80\x04\x95\x9a\x00\x00\x00\x00\x00\x00\x00\x8c\x96SUCCESS: Image loaded successfully from image parameter ({project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/green_tea.0136.png)\x94."
        ),
        "4c92305b-3ba1-4e39-b1bd-cc847b677a71": pickle.loads(
            b"\x80\x04\x95F\x00\x00\x00\x00\x00\x00\x00\x8cBSUCCESS: Successfully processed image: image rescaling (3840x2160)\x94."
        ),
        "1e6233a4-57f4-4f2a-8800-0290e102a3b3": pickle.loads(
            b"\x80\x04\x95\x85\x01\x00\x00\x00\x00\x00\x00X~\x01\x00\x00### Media Generation Conditioning\n\nConditions the model to take given images/video to help diffuse the image into a given style or reference.\n\nSet the index to a frame number and tweak the strength to ensure that the frame is either used in its entirety or to just use it as a reference to guide the diffusion. In this case it is first and last frame set to the strength value of 1.\x94."
        ),
        "b5b6ce26-d0d8-412b-a968-36eb06d68769": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04LTX2\x94."
        ),
        "e7f60b48-1d7f-41de-8b43-66f3f4a7e806": pickle.loads(
            b"\x80\x04\x95-\x00\x00\x00\x00\x00\x00\x00\x8c)dg845/LTX-2.3-Spatial-Upsampler-Diffusers\x94."
        ),
        "872a311a-469f-490f-b2dd-de268e960db7": pickle.loads(
            b"\x80\x04\x95}\x00\x00\x00\x00\x00\x00\x00\x8cyFor other video models such as LTX, upsampling the latent media is possible to enhance the image quality after diffusion.\x94."
        ),
        "3e0994f2-be07-4822-bc37-93c5f04728be": pickle.loads(
            b"\x80\x04\x95G\x00\x00\x00\x00\x00\x00\x00\x8cCA cup of green tea on a saucer, the camera pans over the top of it.\x94."
        ),
        "ce53b9d6-7cae-4c8f-8f3c-16aa78c84685": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "b5f2010b-32d1-4dc1-b32e-85f89209801f": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\x14\x00\x00\x00\x00\x00\x00."
        ),
        "b783292b-db5b-48a4-9fc5-325e10832730": pickle.loads(
            b"\x80\x04\x95C\x02\x00\x00\x00\x00\x00\x00]\x94}\x94\x8c\x16media_gen_conditioning\x94}\x94(\x8c\x04mode\x94\x8c\x05image\x94\x8c\x06images\x94]\x94(}\x94(h\x05\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 86761c177c2a41e7ac7fe5a44bc303c8\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x13\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c#{outputs}/Resize Image_output_6.png\x94ub\x8c\x0bframe_index\x94K\x00\x8c\x08strength\x94G?\xf0\x00\x00\x00\x00\x00\x00u}\x94(h\x05h\x0b)\x81\x94}\x94(h\x0eh\nh\x10h\th\x12\x8c e4cae25f85834e59aa49c9e5993a3128\x94h\x14Nh\x15}\x94h\x17h#h\x18\x8c\x06strict\x94h\x1a\x8c\x05utf-8\x94h\x1c\x8c&{outputs}/Resize Image_1_output_28.png\x94ubh\x1eJ\xff\xff\xff\xffh\x1fG?\xf0\x00\x00\x00\x00\x00\x00ueusa."
        ),
        "1c2f821c-5393-4666-82a7-162a3b35be87": pickle.loads(
            b"\x80\x04\x95o\x03\x00\x00\x00\x00\x00\x00Xh\x03\x00\x00Completed inference step 2 of 20. 19.30 s/it\nCompleted inference step 3 of 20. 19.46 s/it\nCompleted inference step 4 of 20. 19.33 s/it\nCompleted inference step 5 of 20. 19.27 s/it\nCompleted inference step 6 of 20. 19.25 s/it\nCompleted inference step 7 of 20. 19.26 s/it\nCompleted inference step 8 of 20. 19.30 s/it\nCompleted inference step 9 of 20. 19.35 s/it\nCompleted inference step 10 of 20. 19.37 s/it\nCompleted inference step 11 of 20. 19.40 s/it\nCompleted inference step 12 of 20. 19.43 s/it\nCompleted inference step 13 of 20. 19.48 s/it\nCompleted inference step 14 of 20. 19.52 s/it\nCompleted inference step 15 of 20. 19.56 s/it\nCompleted inference step 16 of 20. 19.59 s/it\nCompleted inference step 17 of 20. 19.62 s/it\nCompleted inference step 18 of 20. 19.65 s/it\nCompleted inference step 19 of 20. 19.66 s/it\nCompleted inference step 20 of 20. 19.67 s/it\nDone.\n\x94."
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
                AlterParameterDetailsRequest(
                    parameter_name="num_frames", ui_options={"hide": False}, initial_setup=True
                )
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
                    resolution="resolved",
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
                    ui_options={
                        "simple_dropdown": ["WanPipeline", "WanImageToVideoPipeline"],
                        "show_search": True,
                        "search_filter": "",
                    },
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="model",
                    default_value="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                    tooltip="model",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "simple_dropdown": ["Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
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
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeDecodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Decode Media",
                    metadata={
                        "position": {"x": 2800.000000000001, "y": 585.6666666666667},
                        "tempId": "placing-1779964950485-15mbmf",
                        "library_node_metadata": {
                            "category": "ModularDiffusion/Encode\\Decode",
                            "description": "Decode VAE latent images or video via 🧨 Diffusers Pipelines.",
                        },
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "VaeDecodeNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 749},
                        "category": "ModularDiffusion/Encode\\Decode",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
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
                    parameter_name="fps",
                    default_value=25,
                    tooltip="Frames per second for video output.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={"min": 1, "max": 120},
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="output_video",
                    tooltip="Generated video",
                    type="VideoUrlArtifact",
                    input_types=["VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    initial_setup=True,
                )
            )
        node3_name = (
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
        node4_name = (
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
        node5_name = (
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
        node6_name = (
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
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="MediaGenConditioningNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Media Generation Conditioning",
                    metadata={
                        "position": {"x": 1080, "y": 1837.5047644635415},
                        "tempId": "placing-1780314090525-206c3y",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Conditioning",
                            description="Conditioning node for media generation pipelines. Supports image or video conditioning inputs with configurable strength.",
                            display_name="Media Generation Conditioning",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "MediaGenConditioningNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 1873},
                        "category": "ModularDiffusion/Conditioning",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image_0",
                    tooltip="Conditioning image 1.",
                    type="ImageUrlArtifact",
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    output_type="ImageUrlArtifact",
                    ui_options={},
                    mode_allowed_property=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image_0_frame_index",
                    default_value=0,
                    tooltip="Frame index in the output video where conditioning image 1 is applied.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image_0_strength",
                    default_value=1.0,
                    tooltip="Strength for conditioning image 1.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image_1",
                    tooltip="Conditioning image 2.",
                    type="ImageUrlArtifact",
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    output_type="ImageUrlArtifact",
                    ui_options={},
                    mode_allowed_property=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image_1_frame_index",
                    default_value=0,
                    tooltip="Frame index in the output video where conditioning image 2 is applied.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="image_1_strength",
                    default_value=1.0,
                    tooltip="Strength for conditioning image 2.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
                    initial_setup=True,
                )
            )
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadImage",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Load Image",
                    metadata={
                        "position": {"x": 301.97072090846575, "y": 307.79132692119947},
                        "tempId": "placing-1780314140499-x1cqzk",
                        "library_node_metadata": NodeMetadata(
                            category="image",
                            description="Loads an image from disk",
                            display_name="Load Image",
                            tags=None,
                            icon="image-up",
                            color=None,
                            group="Input/Output",
                            deprecation=None,
                            is_node_group=None,
                        ),
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
                    node_type="RescaleImage",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Resize Image",
                    metadata={
                        "position": {"x": 977.5647933277094, "y": 307.79132692119947},
                        "tempId": "placing-1780314915181-ffb74d",
                        "library_node_metadata": NodeMetadata(
                            category="image",
                            description='Resize images with separate parameters for target size (pixels) and percentage scale, plus resample filter options. Previously named "Rescale Image".',
                            display_name="Resize Image",
                            tags=None,
                            icon="image-upscale",
                            color=None,
                            group="edit",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "RescaleImage",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 928},
                        "category": "image",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="percentage_scale",
                    ui_options={
                        "slider": {"min_val": 1, "max_val": 500},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": True,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="target_width",
                    ui_options={
                        "slider": {"min_val": 1, "max_val": 8000},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="target_height",
                    ui_options={
                        "slider": {"min_val": 1, "max_val": 8000},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="fit_mode",
                    ui_options={
                        "simple_dropdown": ["fit", "fill", "stretch"],
                        "show_search": True,
                        "search_filter": "",
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="background_color",
                    ui_options={
                        "color_picker": {"format": "hex"},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadImage",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Load Image_1",
                    metadata={
                        "position": {"x": 301.97072090846575, "y": 1344.7393162941023},
                        "tempId": "placing-1780314953633-bbuu7",
                        "library_node_metadata": NodeMetadata(
                            category="image",
                            description="Loads an image from disk",
                            display_name="Load Image",
                            tags=None,
                            icon="image-up",
                            color=None,
                            group="Input/Output",
                            deprecation=None,
                            is_node_group=None,
                        ),
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
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="RescaleImage",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Resize Image_1",
                    metadata={
                        "position": {"x": 977.5647933277094, "y": 1344.7393162941023},
                        "tempId": "placing-1780314971445-hewgxv",
                        "library_node_metadata": NodeMetadata(
                            category="image",
                            description='Resize images with separate parameters for target size (pixels) and percentage scale, plus resample filter options. Previously named "Rescale Image".',
                            display_name="Resize Image",
                            tags=None,
                            icon="image-upscale",
                            color=None,
                            group="edit",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "RescaleImage",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 928},
                        "category": "image",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="percentage_scale",
                    ui_options={
                        "slider": {"min_val": 1, "max_val": 500},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": True,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="target_width",
                    ui_options={
                        "slider": {"min_val": 1, "max_val": 8000},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="target_height",
                    ui_options={
                        "slider": {"min_val": 1, "max_val": 8000},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="fit_mode",
                    ui_options={
                        "simple_dropdown": ["fit", "fill", "stretch"],
                        "show_search": True,
                        "search_filter": "",
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="background_color",
                    ui_options={
                        "color_picker": {"format": "hex"},
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
        node12_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="IntegerInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Width",
                    metadata={
                        "position": {"x": 301.97072090846575, "y": 890.7319678865354},
                        "tempId": "placing-1780315022991-6yldx",
                        "library_node_metadata": NodeMetadata(
                            category="number",
                            description="Create an integer value",
                            display_name="Integer Input",
                            tags=None,
                            icon=None,
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "IntegerInput",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 196},
                        "category": "number",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node13_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="IntegerInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Height",
                    metadata={
                        "position": {"x": 301.97072090846575, "y": 1086.7319678865358},
                        "tempId": "placing-1780315035244-yc0a8n",
                        "library_node_metadata": NodeMetadata(
                            category="number",
                            description="Create an integer value",
                            display_name="Integer Input",
                            tags=None,
                            icon=None,
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "IntegerInput",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 196},
                        "category": "number",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node14_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_4",
                    metadata={
                        "position": {"x": 1080, "y": 1474.7486522065524},
                        "tempId": "placing-1780316417085-4zdqwx",
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
                        "size": {"width": 600, "height": 306},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node15_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LatentUpsamplerNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Latent Upsampler",
                    metadata={
                        "position": {"x": 2010, "y": 1872.595957916338},
                        "tempId": "placing-1780318101220-89hy1e",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Processing",
                            description="Spatially upsample a latent tensor using a latent upsampler model.",
                            display_name="Latent Upsampler",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "LatentUpsamplerNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 320},
                        "category": "ModularDiffusion/Processing",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node15_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="upsampler_model",
                    default_value="dg845/LTX-2.3-Spatial-Upsampler-Diffusers",
                    tooltip="upsampler_model",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "simple_dropdown": ["dg845/LTX-2.3-Spatial-Upsampler-Diffusers"],
                        "show_search": True,
                        "search_filter": "",
                        "display_name": "upsampler_model",
                    },
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
        node16_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_5",
                    metadata={
                        "position": {"x": 2010, "y": 1670.02759606089},
                        "tempId": "placing-1780318113352-rjgewq",
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
        node17_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input",
                    metadata={
                        "position": {"x": 1115.5103455235624, "y": 1108.7393162941023},
                        "tempId": "placing-1780319167338-kgf4jj",
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
        node18_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DiffusionPipelineGenerateLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Generate Media Latents (Modular Diffusion Pipeline)",
                    metadata={
                        "library_node_metadata": {
                            "category": "ModularDiffusion/Processing",
                            "description": "Generate latents via 🧨 Diffusers Pipelines.",
                        },
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "DiffusionPipelineGenerateLatentNode",
                        "position": {"x": 2010, "y": 585.6666666666667},
                        "size": {"width": 600, "height": 976},
                        "showaddparameter": False,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node18_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="prompt",
                    default_value="",
                    tooltip="The prompt or prompts to guide the video generation from the input image.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="negative_prompt",
                    default_value="",
                    tooltip="The prompt or prompts not to guide the video generation. Ignored when guidance_scale < 1.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="guidance_scale",
                    default_value=5.0,
                    tooltip="Higher guidance scale encourages the model to generate videos more closely linked to the text prompt, usually at the expense of lower quality. Guidance is enabled by setting guidance_scale > 1.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
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
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="output_video",
                    tooltip="Generated video output.",
                    type="VideoUrlArtifact",
                    input_types=["VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="additional_parameters_ParameterListUniqueParamID_92f8f5e8342f4aa798989802d98f2992",
                    default_value=[],
                    tooltip="Extra pipeline-specific kwargs (e.g. media generation conditioning).",
                    type="additional_parameters",
                    input_types=["additional_parameters", "dict"],
                    output_type="additional_parameters",
                    ui_options={},
                    mode_allowed_property=False,
                    parent_container_name="additional_parameters",
                    initial_setup=True,
                )
            )
        node19_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Group",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Image conditioning setup",
                    metadata={
                        "position": {"x": -642.5647933277094, "y": 1494.7274989891105},
                        "tempId": "placing-1780316392101-e304iq",
                        "library_node_metadata": NodeMetadata(
                            category="misc",
                            description="Create a group node to organize your workflow",
                            display_name="Group",
                            tags=None,
                            icon="group",
                            color=None,
                            group="create",
                            deprecation=None,
                            is_node_group=True,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "Group",
                        "is_node_group": True,
                        "executable": False,
                        "hideaddparameter": True,
                        "showConnectionsCollapsed": True,
                        "group_settings_params": ["description"],
                        "size": {"width": 1643, "height": 2395},
                        "expanded_dimensions": {"width": 1643, "height": 2395},
                    },
                    node_names_to_add=[node8_name, node9_name, node10_name, node11_name, node12_name, node13_name],
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
                source_node_name=node8_name,
                source_parameter_name="image",
                target_node_name=node9_name,
                target_parameter_name="input_image",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node9_name,
                source_parameter_name="output",
                target_node_name=node7_name,
                target_parameter_name="image_0",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="image",
                target_node_name=node11_name,
                target_parameter_name="input_image",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="output",
                target_node_name=node7_name,
                target_parameter_name="image_1",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="integer",
                target_node_name=node9_name,
                target_parameter_name="target_width",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="integer",
                target_node_name=node9_name,
                target_parameter_name="target_height",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="integer",
                target_node_name=node11_name,
                target_parameter_name="target_width",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="integer",
                target_node_name=node11_name,
                target_parameter_name="target_height",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="integer",
                target_node_name=node0_name,
                target_parameter_name="width",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="integer",
                target_node_name=node0_name,
                target_parameter_name="height",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="pipeline",
                target_node_name=node18_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="output_latent",
                target_node_name=node18_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node18_name,
                source_parameter_name="pipeline",
                target_node_name=node2_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node17_name,
                source_parameter_name="text",
                target_node_name=node18_name,
                target_parameter_name="prompt",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node7_name,
                source_parameter_name="conditioning",
                target_node_name=node18_name,
                target_parameter_name="additional_parameters_ParameterListUniqueParamID_92f8f5e8342f4aa798989802d98f2992",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node18_name,
                source_parameter_name="output_latent",
                target_node_name=node2_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["a892317d-e2fa-40a1-b0b8-01bebc44addb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["7537d046-6ec1-4b8d-ac5c-8413a226a53b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["456e5f78-7efa-4698-86f8-bcc67ba0422f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["5f19c64b-b5b4-4a4a-a0ce-2215b4299bde"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["0aa65c92-583a-4035-8758-9e4fcd15c70a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["f00c67c7-f72e-4a38-a386-ce4147a0fcb1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["a892317d-e2fa-40a1-b0b8-01bebc44addb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["a892317d-e2fa-40a1-b0b8-01bebc44addb"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["bdc4fb33-509f-45d7-b66f-6901c1c68661"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["e2bf76c0-12f6-48ff-99d2-f5e263058601"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["50227083-d069-4397-ae82-88629804ad58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["ebde2e93-795f-43f6-9262-08ec0dd64906"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_tiling",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["97576349-86c8-4797-b696-d18370b741c0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["97576349-86c8-4797-b696-d18370b741c0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["06ca7d08-18ad-46d5-af16-9dfaf9e2689e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["e11f2893-64ea-4a42-b9c3-79631d9fd38f"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["a892317d-e2fa-40a1-b0b8-01bebc44addb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["132c418e-9643-4f92-bbba-b2a426326fe3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_video",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["c1821f3d-46e3-45a4-98a2-55ea5152c134"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["74618a79-e1bc-4ad3-80a4-260c35394ae5"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["7b867c28-48c2-49d1-b779-158a551350ae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["63ababb5-f635-49f8-8ab1-97a5925e978f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["a6f51233-992b-492d-b2f3-79ac9d1ba6a8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mode",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["67f3dd75-9d58-4b7b-a708-57f5338336f6"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_images",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["b1baed1e-71c1-48e5-9bfc-8f53f2560984"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_0",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["c1764a7b-6282-4b91-9311-de88d14d1218"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_0_frame_index",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["cdf16e89-a770-4351-be51-2d9a18338190"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_0_strength",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["cf30316e-ffbb-448f-b5c7-50a8b9c5a7a2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_1",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["440641d3-27ca-4234-8e8c-f561b044bfe4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_1_frame_index",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["3bb76746-cb74-4c2c-a977-78bfcd4fe834"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_1_strength",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["cf30316e-ffbb-448f-b5c7-50a8b9c5a7a2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["8624ee43-f3a5-4a44-a991-542e5ea8d444"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["abe01153-519c-41a9-8e29-521795a1efcb"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["7605646b-831f-4677-8135-d19a036c2d3a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["7605646b-831f-4677-8135-d19a036c2d3a"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mask_channel",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["29eff951-17cc-415d-aded-9db01c00049e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["2bfa74fe-9db3-408e-b8e6-202a8c763163"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["2bfa74fe-9db3-408e-b8e6-202a8c763163"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mask_output_file",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["6570eb60-0f30-424b-b2b0-5d46ec4be206"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="input_image",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["abe01153-519c-41a9-8e29-521795a1efcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resize_mode",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["cfbd5a76-6cf2-489f-a0f3-e99c421a8f91"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_size",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["59a0322c-bc33-49f2-8437-8ff19f3ae0c1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="percentage_scale",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["23675c0f-6790-492e-86c8-c47d220fb92b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_width",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["456e5f78-7efa-4698-86f8-bcc67ba0422f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_height",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["5f19c64b-b5b4-4a4a-a0ce-2215b4299bde"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fit_mode",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["cb6aa17f-402a-4f36-a839-4fbad879765c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="background_color",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["cb2c224a-5293-49a5-8d87-e262b8aefd38"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resample_filter",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["0e9806cb-0282-4fea-81c6-ed7e26cd90b8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_format",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["1e0300a4-124c-4cde-b091-b01e7a9cf740"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quality",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["5ad0e916-d32c-44b8-b363-89ccad81e262"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["c1764a7b-6282-4b91-9311-de88d14d1218"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["6f1ab114-fe3a-4ef1-938d-36265d693542"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["2bfa74fe-9db3-408e-b8e6-202a8c763163"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["2bfa74fe-9db3-408e-b8e6-202a8c763163"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["e2dea525-e3fb-4878-901d-7a0bd19f9847"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1d41add8-79fa-47e8-bf66-c2ff75172cb1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["8072c7b4-cd79-4bbe-991e-0f875863bd35"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1441eaf2-deda-4a3f-9a7d-bb7f4151bdc7"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1441eaf2-deda-4a3f-9a7d-bb7f4151bdc7"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mask_channel",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["29eff951-17cc-415d-aded-9db01c00049e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["53aac130-e307-4afc-a82c-fcab34289844"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["53aac130-e307-4afc-a82c-fcab34289844"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["13d2c731-a03f-4f4c-ac6d-f6347dd6c5fc"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["13d2c731-a03f-4f4c-ac6d-f6347dd6c5fc"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mask_output_file",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["6570eb60-0f30-424b-b2b0-5d46ec4be206"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="input_image",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["8072c7b4-cd79-4bbe-991e-0f875863bd35"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resize_mode",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["cfbd5a76-6cf2-489f-a0f3-e99c421a8f91"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_size",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["59a0322c-bc33-49f2-8437-8ff19f3ae0c1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="percentage_scale",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["23675c0f-6790-492e-86c8-c47d220fb92b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_width",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["456e5f78-7efa-4698-86f8-bcc67ba0422f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_height",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["5f19c64b-b5b4-4a4a-a0ce-2215b4299bde"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fit_mode",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["cb6aa17f-402a-4f36-a839-4fbad879765c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="background_color",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["cb2c224a-5293-49a5-8d87-e262b8aefd38"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resample_filter",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["0e9806cb-0282-4fea-81c6-ed7e26cd90b8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_format",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["1e0300a4-124c-4cde-b091-b01e7a9cf740"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quality",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["5ad0e916-d32c-44b8-b363-89ccad81e262"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["440641d3-27ca-4234-8e8c-f561b044bfe4"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["6f1ab114-fe3a-4ef1-938d-36265d693542"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["53aac130-e307-4afc-a82c-fcab34289844"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["53aac130-e307-4afc-a82c-fcab34289844"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["4c92305b-3ba1-4e39-b1bd-cc847b677a71"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["4c92305b-3ba1-4e39-b1bd-cc847b677a71"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["e2dea525-e3fb-4878-901d-7a0bd19f9847"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="integer",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["456e5f78-7efa-4698-86f8-bcc67ba0422f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="integer",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["456e5f78-7efa-4698-86f8-bcc67ba0422f"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node13_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="integer",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["5f19c64b-b5b4-4a4a-a0ce-2215b4299bde"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="integer",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["5f19c64b-b5b4-4a4a-a0ce-2215b4299bde"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node14_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node14_name,
                    value=top_level_unique_values_dict["1e6233a4-57f4-4f2a-8800-0290e102a3b3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node15_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node15_name,
                    value=top_level_unique_values_dict["b5b6ce26-d0d8-412b-a968-36eb06d68769"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="upsampler_model",
                    node_name=node15_name,
                    value=top_level_unique_values_dict["e7f60b48-1d7f-41de-8b43-66f3f4a7e806"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node16_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node16_name,
                    value=top_level_unique_values_dict["872a311a-469f-490f-b2dd-de268e960db7"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node17_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node17_name,
                    value=top_level_unique_values_dict["3e0994f2-be07-4822-bc37-93c5f04728be"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node17_name,
                    value=top_level_unique_values_dict["3e0994f2-be07-4822-bc37-93c5f04728be"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node18_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["a892317d-e2fa-40a1-b0b8-01bebc44addb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["cdf16e89-a770-4351-be51-2d9a18338190"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["3bb76746-cb74-4c2c-a977-78bfcd4fe834"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["3e0994f2-be07-4822-bc37-93c5f04728be"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="negative_prompt",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["ce53b9d6-7cae-4c8f-8f3c-16aa78c84685"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_scale",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["b5f2010b-32d1-4dc1-b32e-85f89209801f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["f00c67c7-f72e-4a38-a386-ce4147a0fcb1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["e8a55714-10d8-4bad-a597-51d36f24c205"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["7537d046-6ec1-4b8d-ac5c-8413a226a53b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["b783292b-db5b-48a4-9fc5-325e10832730"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["cf30316e-ffbb-448f-b5c7-50a8b9c5a7a2"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["1c2f821c-5393-4666-82a7-162a3b35be87"],
                    initial_setup=True,
                    is_output=True,
                )
            )
