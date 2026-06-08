# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "LTX23-HDR-Text2Video-Upsample-Two-Stage"
# schema_version = "0.18.0"
# engine_version_created_with = "0.85.4"
# node_libraries_referenced = [["Griptape Nodes Library", "0.75.0"], ["Griptape Modular Diffusion Nodes Library", "0.1.0"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DecodeHdrNode"], ["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "EmptyLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "LatentUpsamplerNode"], ["Griptape Modular Diffusion Nodes Library", "LoadLora"], ["Griptape Modular Diffusion Nodes Library", "LoraActivationPipelineNode"], ["Griptape Modular Diffusion Nodes Library", "MediaGenConditioningNode"], ["Griptape Modular Diffusion Nodes Library", "NoiseLatentNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Nodes Library", "TextInput"]]
# description = "Two-stage LTX-2.3 T2V workflow with spatial latent upsampling and HDR IC-LoRA, producing tone-mapped video plus a raw linear EXR sequence."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/LTX23-HDR-Text2Video-Upsample-Two-Stage.png"
# is_griptape_provided = true
# is_template = true
# creation_date = 2026-06-04T00:00:00.920106Z
# last_modified_date = 2026-06-04T00:00:00.920106Z
#
# ///

import pickle

from griptape_nodes.node_library.library_registry import NodeMetadata
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
        "f5d6f13a-c2ce-43c3-adfe-89f98f054926": pickle.loads(
            b"\x80\x04\x958\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "709818f7-75cc-4171-907d-4235c3069ddf": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04LTX2\x94."
        ),
        "1e031b4a-3c13-4947-b7ae-1e173021e197": pickle.loads(
            b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cLTX2Pipeline\x94."
        ),
        "8e639bfd-85d3-4c04-8352-81a7694b1891": pickle.loads(
            b"\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94."
        ),
        "0ce709d1-c8c3-4328-8f5f-0bfd88a8d543": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tAutomatic\x94."
        ),
        "1e4d883d-52ee-4c09-92a3-a66fc46efc5c": pickle.loads(b"\x80\x04\x89."),
        "01e61129-a4f5-434e-bffe-ad2e73e0fcef": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04None\x94."
        ),
        "c73f0805-479a-4a57-a6bf-7933b8d32744": pickle.loads(b"\x80\x04]\x94."),
        "824f4f37-b5d3-46fa-be55-bbf0a2b2714c": pickle.loads(
            b"\x80\x04\x95Z\x01\x00\x00\x00\x00\x00\x00XS\x01\x00\x00Building pipeline...\nPipeline configuration hash: LTX2Pipeline-891e4d37fa3abfdce0b084b3d1fd7fff8a5bcc1069d3fbb5cdc8785892e2bc72-0\nUsing cached pipeline.\nPipeline building/caching took 0.00 milliseconds\nPipeline building complete.\nPipeline configuration hash: LTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\n\x94."
        ),
        "310c47b1-03f8-4b6a-a0d0-295f84477d47": pickle.loads(
            b"\x80\x04\x958\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "5b3bcdb5-759a-447f-aeca-dba8cf9fdb3d": pickle.loads(b"\x80\x04K*."),
        "88bca94d-8553-4ee1-8cc3-be93eea2811b": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x03."),
        "78833574-ee15-4800-9c45-dc7bf28007a9": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x02."),
        "4957f5a1-c540-46af-8d7b-9f96a94dafce": pickle.loads(b"\x80\x04Ky."),
        "cb5054f4-b2d3-456e-a4b5-c4e4c55dd474": pickle.loads(b"\x80\x04K\x14."),
        "d4055550-9d1d-42e0-9e0f-0a024502afa8": pickle.loads(
            b"\x80\x04\x95\xd2\x01\x00\x00\x00\x00\x00\x00X\xcb\x01\x00\x00A slow, majestic pan inside a massive, dimly lit Gothic cathedral at midnight. Moonlight streams through a colossal stained-glass rose window above the altar, making the intricate glass facets glow with intense, blinding ruby, cobalt, and emerald colors. The surrounding stone columns and filigree are plunged into deep, textured shadows. On the altar, a dozen beeswax candles flicker, casting bright, pulsing specular highlights on a polished silver chalice.\x94."
        ),
        "50fa2e6b-0e3d-4f26-a025-fa48304d5646": pickle.loads(
            b'\x80\x04\x95k\x05\x00\x00\x00\x00\x00\x00Xd\x05\x00\x00blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, unreadable text on shirt or hat, incorrect lettering on cap ("PNTR"), incorrect t-shirt slogan ("JUST DO IT"), missing microphone, misplaced microphone, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, smiling, laughing, exaggerated sadness, wrong gaze direction, eyes looking at camera, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, missing sniff sounds, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, missing door or shelves, missing shallow depth of field, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts. music\x94.'
        ),
        "21d1bbc2-a4ef-4f18-858b-5d0cf4355271": pickle.loads(
            b"\x80\x04\x95-\x00\x00\x00\x00\x00\x00\x00\x8c)dg845/LTX-2.3-Spatial-Upsampler-Diffusers\x94."
        ),
        "9db99aa5-5969-4530-9aa1-0eb9d2f7c7de": pickle.loads(b"\x80\x04K\x00."),
        "b58239c3-c88f-4eea-b0b7-d9d599e1187b": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xff\xff\xff\xff."
        ),
        "6ce85e10-5ee5-4136-9364-82128887e113": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00."
        ),
        "004a187e-260b-41a4-a048-b726f7f4c429": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\x08\x00\x00\x00\x00\x00\x00."
        ),
        "57cea271-f5f4-4d4c-b474-3e85a1d6da87": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xe6ffffff."
        ),
        "1e828835-8d61-41d1-8070-9ab369b12b73": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\x1c\x00\x00\x00\x00\x00\x00."
        ),
        "867b7797-308e-40b4-a39f-f8bd9ed38698": pickle.loads(b"\x80\x04]\x94."),
        "b8e65bd4-b663-4d33-98f7-df0c0b110160": pickle.loads(
            b"\x80\x04\x95\xe0\x06\x00\x00\x00\x00\x00\x00X\xd9\x06\x00\x00Completed inference step 2 of 40. 4.32 s/it\nCompleted inference step 3 of 40. 4.28 s/it\nCompleted inference step 4 of 40. 4.28 s/it\nCompleted inference step 5 of 40. 4.28 s/it\nCompleted inference step 6 of 40. 4.29 s/it\nCompleted inference step 7 of 40. 4.33 s/it\nCompleted inference step 8 of 40. 4.33 s/it\nCompleted inference step 9 of 40. 4.35 s/it\nCompleted inference step 10 of 40. 4.37 s/it\nCompleted inference step 11 of 40. 4.38 s/it\nCompleted inference step 12 of 40. 4.39 s/it\nCompleted inference step 13 of 40. 4.40 s/it\nCompleted inference step 14 of 40. 4.41 s/it\nCompleted inference step 15 of 40. 4.42 s/it\nCompleted inference step 16 of 40. 4.42 s/it\nCompleted inference step 17 of 40. 4.43 s/it\nCompleted inference step 18 of 40. 4.44 s/it\nCompleted inference step 19 of 40. 4.44 s/it\nCompleted inference step 20 of 40. 4.45 s/it\nCompleted inference step 21 of 40. 4.45 s/it\nCompleted inference step 22 of 40. 4.46 s/it\nCompleted inference step 23 of 40. 4.46 s/it\nCompleted inference step 24 of 40. 4.47 s/it\nCompleted inference step 25 of 40. 4.47 s/it\nCompleted inference step 26 of 40. 4.47 s/it\nCompleted inference step 27 of 40. 4.47 s/it\nCompleted inference step 28 of 40. 4.48 s/it\nCompleted inference step 29 of 40. 4.48 s/it\nCompleted inference step 30 of 40. 4.48 s/it\nCompleted inference step 31 of 40. 4.48 s/it\nCompleted inference step 32 of 40. 4.49 s/it\nCompleted inference step 33 of 40. 4.49 s/it\nCompleted inference step 34 of 40. 4.49 s/it\nCompleted inference step 35 of 40. 4.49 s/it\nCompleted inference step 36 of 40. 4.49 s/it\nCompleted inference step 37 of 40. 4.49 s/it\nCompleted inference step 38 of 40. 4.50 s/it\nCompleted inference step 39 of 40. 4.50 s/it\nCompleted inference step 40 of 40. 4.50 s/it\nDone.\n\x94."
        ),
        "cdbd8985-40f1-4ab7-81d0-9df2c3cd2810": pickle.loads(
            b"\x80\x04\x95\xce\x04\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89\x8c\x1c_extra_runtime_adapter_steps\x94]\x94\x8c6modular_diffusion_nodes_library.utils.lora_apply_utils\x94\x8c\x1eLoraPipelineRuntimeAdapterStep\x94\x93\x94)\x81\x94}\x94h\x15}\x94\x8c\x95C:\\Users\\ladipo.baruwa\\GriptapeNodes\\libraries\\griptape-nodes-library-modular-diffusion\\workflows\\assets\\lora\\ltx-2.3-22b-ic-lora-hdr-0.9.safetensors\x94\x8c/modular_diffusion_nodes_library.utils.lora_spec\x94\x8c\x08LoraSpec\x94\x93\x94)\x81\x94}\x94(\x8c\x04path\x94h-\x8c\x06weight\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x0etrigger_phrase\x94Nubssbaub."
        ),
        "863623c0-13b3-4280-ac69-bfae84c6e300": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x06."),
        "3c58ba54-dc51-46a6-8b90-21ff352c4ecf": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x04."),
        "801aacb1-5e0b-4043-9a23-1f5390c3c3bd": pickle.loads(
            b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05video\x94."
        ),
        "c529ff6d-5fa5-4eaf-97fd-8b429b3f7fb4": pickle.loads(
            b"\x80\x04\x95\xce\x04\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89\x8c\x1c_extra_runtime_adapter_steps\x94]\x94\x8c6modular_diffusion_nodes_library.utils.lora_apply_utils\x94\x8c\x1eLoraPipelineRuntimeAdapterStep\x94\x93\x94)\x81\x94}\x94h\x15}\x94\x8c\x95C:\\Users\\ladipo.baruwa\\GriptapeNodes\\libraries\\griptape-nodes-library-modular-diffusion\\workflows\\assets\\lora\\ltx-2.3-22b-ic-lora-hdr-0.9.safetensors\x94\x8c/modular_diffusion_nodes_library.utils.lora_spec\x94\x8c\x08LoraSpec\x94\x93\x94)\x81\x94}\x94(\x8c\x04path\x94h-\x8c\x06weight\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x0etrigger_phrase\x94Nubssbaub."
        ),
        "c4e83959-ed8a-42fe-9065-97213d5d8a67": pickle.loads(
            b"\x80\x04\x95\xcb\x00\x00\x00\x00\x00\x00\x00]\x94}\x94(\x8c\x04path\x94\x8c\x95C:\\Users\\ladipo.baruwa\\GriptapeNodes\\libraries\\griptape-nodes-library-modular-diffusion\\workflows\\assets\\lora\\ltx-2.3-22b-ic-lora-hdr-0.9.safetensors\x94\x8c\x06weight\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x0etrigger_phrase\x94Nua."
        ),
        "d5f2a5f3-8096-4d68-9231-f43de667df54": pickle.loads(
            b"\x80\x04\x95\xc8\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x04path\x94\x8c\x95C:\\Users\\ladipo.baruwa\\GriptapeNodes\\libraries\\griptape-nodes-library-modular-diffusion\\workflows\\assets\\lora\\ltx-2.3-22b-ic-lora-hdr-0.9.safetensors\x94\x8c\x06weight\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x0etrigger_phrase\x94Nu."
        ),
        "75f48310-8775-4e32-87fd-6555a2b0fa6b": pickle.loads(
            b"\x80\x04\x954\x02\x00\x00\x00\x00\x00\x00X-\x02\x00\x00Building pipeline...\nPipeline configuration hash: LTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\nUsing cached pipeline.\nPipeline building/caching took 0.00 milliseconds\nPipeline building complete.\nPipeline configuration hash: LTX2Pipeline-891e4d37fa3abfdce0b084b3d1fd7fff8a5bcc1069d3fbb5cdc8785892e2bc72-0\nPipeline configuration hash: LTX2Pipeline-891e4d37fa3abfdce0b084b3d1fd7fff8a5bcc1069d3fbb5cdc8785892e2bc72-0\nPipeline configuration hash: LTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\n\x94."
        ),
        "cacc941c-7603-43dc-a176-31dca85c041f": pickle.loads(
            b"\x80\x04\x95z\x00\x00\x00\x00\x00\x00\x00\x8cv{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/lora/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors\x94."
        ),
        "595d6132-c70e-4d6f-b1b3-320a0eee1e59": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "fb3be385-3085-4c47-927b-704b35815040": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00."
        ),
        "df4a1803-78f3-4a2e-a4df-c5c7a129db52": pickle.loads(b"\x80\x04K\x01."),
        "16029530-fa18-4189-8f0f-5995497160e4": pickle.loads(
            b"\x80\x04\x95\x80\x00\x00\x00\x00\x00\x00\x00\x8c|{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/lora/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors\x94."
        ),
        "3e0a105d-c6e9-438f-b833-644dd3f56783": pickle.loads(b"\x80\x04K\x08."),
        "d5841411-0df9-4ad6-9cb1-8d32de272e98": pickle.loads(b"\x80\x04]\x94."),
        "4953712a-0d5e-427d-95e3-b89cd9523528": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xc9\x99\x99\x99\x99\x99\x9a."
        ),
        "17dc8cc5-8447-4992-a259-1253dab0b9d4": pickle.loads(
            b"\x80\x04\x95H\x01\x00\x00\x00\x00\x00\x00XA\x01\x00\x00Completed inference step 2 of 40. 21.19 s/it\nCompleted inference step 3 of 40. 21.74 s/it\nCompleted inference step 4 of 40. 22.03 s/it\nCompleted inference step 5 of 40. 22.16 s/it\nCompleted inference step 6 of 40. 22.25 s/it\nCompleted inference step 7 of 40. 22.31 s/it\nCompleted inference step 8 of 40. 22.31 s/it\nDone.\n\x94."
        ),
        "39ed6d6e-c86f-43e4-98c2-02e16d8bffcb": pickle.loads(b"\x80\x04K\x19."),
        "f0040d27-0e9e-4be6-bc10-2576ceb1c19e": pickle.loads(
            b"\x80\x04\x95o\x00\x00\x00\x00\x00\x00\x00\x8ck{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/church_candles/church_candles.exr\x94."
        ),
        "ab599fba-68b7-4747-acd9-f4532cf2b4cd": pickle.loads(b"\x80\x04\x88."),
        "d7153365-e6ce-4d9a-b7d9-239abf8ad6e6": pickle.loads(
            b"\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x0baces_filmic\x94."
        ),
        "ff6cfae1-121b-4b04-9aee-e1f6d37f62f7": pickle.loads(
            b"\x80\x04\x956\x00\x00\x00\x00\x00\x00\x00\x8c2LTX2.3 HDR IC LoRA requires an empty input Latent.\x94."
        ),
        "b78f6e9e-7980-4c59-903a-0ed764fac97d": pickle.loads(
            b'\x80\x04\x95\x91\x00\x00\x00\x00\x00\x00\x00\x8c\x8dLoRA Pipeline will ensure that the selected LoRAs are activated on the pipeline during media generation in the "Generate Media Latents" node.\x94.'
        ),
        "7c77da5a-3f98-4387-ad21-11b0bf42b328": pickle.loads(
            b"\x80\x04\x959\x00\x00\x00\x00\x00\x00\x00\x8c5LTX2.3 HDR IC LoRA requires video conditioning input.\x94."
        ),
        "bb901318-c050-4fe2-8c91-773b07680181": pickle.loads(
            b"\x80\x04\x95\xa2\x00\x00\x00\x00\x00\x00\x00\x8c\x9eThe upsampled media is for comparison only, it is not fed into the SDR-to-HDR stage. To use this, you probably want to run a refinement pass, without the LoRA\x94."
        ),
        "3f872933-604d-4a7f-be03-1650a1b359d9": pickle.loads(
            b"\x80\x04\x95\xa6\x00\x00\x00\x00\x00\x00\x00\x8c\xa2No prompt required, it uses the text embeddings: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/blob/main/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors\x94."
        ),
        "bc8e0057-0fd7-4a74-90bf-58604fa2c8a6": pickle.loads(
            b"\x80\x04\x95\x80\x00\x00\x00\x00\x00\x00\x00\x8c|LTX2.3 HDR LoRA: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/blob/main/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors\x94."
        ),
        "3bbe9835-9577-4fa9-9d44-8c07d25559fc": pickle.loads(
            b"\x80\x04\x95\x9d\x00\x00\x00\x00\x00\x00\x00\x8c\x99You can use also use the distilled model in this stage: dg845/LTX-2.3-Distilled-Diffusers. You would have to connect a new pipeline builder to this node.\x94."
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
                        "position": {"x": 56.269616140404196, "y": 105.63674200565657},
                        "tempId": "placing-1780537992973-nknim4",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Pipeline",
                            description="Build and cache latent-compatible 🧨 Diffusers Pipelines. Any LoRAs wired into this node are fused permanently into the cached weights — changing them rebuilds the pipeline. For dynamic adapters that swap per generation, use the LoRA Pipeline node instead.",
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
                        "size": {"width": 600, "height": 716},
                        "category": "ModularDiffusion/Pipeline",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="pipeline_type",
                    tooltip="Specific pipeline variant within the selected provider (e.g. base, Fill, Edit). Determines which checkpoints and runtime parameters are exposed.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"simple_dropdown": ["LTX2Pipeline"], "show_search": True, "search_filter": ""},
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="model",
                    default_value="dg845/LTX-2.3-Diffusers",
                    tooltip="model",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "button_label": "",
                        "variant": "secondary",
                        "size": "icon",
                        "state": "normal",
                        "full_width": False,
                        "button_icon": "list-restart",
                        "iconPosition": "left",
                        "simple_dropdown": ["dg845/LTX-2.3-Diffusers", "dg845/LTX-2.3-Distilled-Diffusers"],
                        "show_search": True,
                        "search_filter": "",
                        "hide_label": False,
                        "hide_property": False,
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
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="NoiseLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Create Noise Latents",
                    metadata={
                        "position": {"x": 827.9015422265763, "y": 105.63674200565657},
                        "tempId": "placing-1780538017550-gq95a2",
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
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="num_frames", ui_options={"hide": False}, initial_setup=True
                )
            )
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input",
                    metadata={
                        "position": {"x": 827.9015422265763, "y": 611.4383164520079},
                        "tempId": "placing-1780540214322-ogsnx",
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
                        "size": {"width": 600, "height": 334},
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
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input_2",
                    metadata={
                        "position": {"x": 827.9015422265763, "y": 1038.334053445911},
                        "tempId": "placing-1780540287387-mm1kz7",
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
                        "size": {"width": 601, "height": 345},
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
                    node_type="LatentUpsamplerNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Latent Upsampler",
                    metadata={
                        "position": {"x": 132.83870537035318, "y": 130.86884482621818},
                        "tempId": "placing-1780540364475-xrlaj",
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
                        "size": {"width": 600, "height": 328},
                        "category": "ModularDiffusion/Processing",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="upsampler_model",
                    default_value="dg845/LTX-2.3-Spatial-Upsampler-Diffusers",
                    tooltip="upsampler_model",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "button_label": "",
                        "variant": "secondary",
                        "size": "icon",
                        "state": "normal",
                        "full_width": False,
                        "button_icon": "list-restart",
                        "iconPosition": "left",
                        "simple_dropdown": ["dg845/LTX-2.3-Spatial-Upsampler-Diffusers"],
                        "show_search": True,
                        "search_filter": "",
                        "hide_label": False,
                        "hide_property": False,
                        "display_name": "upsampler_model",
                    },
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DiffusionPipelineGenerateLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Generate Media Latents (Modular Diffusion Pipeline)",
                    metadata={
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
                        "position": {"x": 1638.96602838905, "y": 105.63674200565657},
                        "size": {"width": 600, "height": 1229},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="use_stage_2",
                    default_value=False,
                    tooltip="If True, run stage 2 (refinement). If False, run stage 1 (initial generation).",
                    type="bool",
                    input_types=["bool"],
                    output_type="bool",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="stg_scale",
                    default_value=0.0,
                    tooltip="Spatio-Temporal Guidance (STG) scale for the video modality. STG moves the sample away from a perturbed version of the denoising model output, requiring one additional forward pass. 0.0 disables STG. For LTX-2.3 a value of 1.0 is suggested.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="modality_scale",
                    default_value=1.0,
                    tooltip="Modality isolation guidance scale for the video modality. Moves the sample away from a version generated without audio-to-video and video-to-audio cross attention, using a CFG-like estimate. Requires one additional forward pass. 1.0 disables modality guidance. For LTX-2.3 a value of 3.0 is suggested.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="guidance_rescale",
                    default_value=0.0,
                    tooltip="Guidance rescale factor for the video modality. Proposed by 'Common Diffusion Noise Schedules and Sample Steps are Flawed' to fix overexposure when using zero terminal SNR. 0.0 disables rescaling.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_guidance_scale",
                    default_value=1.0,
                    tooltip="CFG guidance scale for the audio modality. The same CFG update rule applies as for video, but video and audio can use different values. For LTX-2.3 the authors suggest 7.0 for audio alongside 3.0 for video.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_stg_scale",
                    default_value=0.0,
                    tooltip="Spatio-Temporal Guidance scale for the audio modality. Uses the same STG update rule as the video modality. For LTX-2.3 a value of 1.0 is suggested for both video and audio.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_modality_scale",
                    default_value=1.0,
                    tooltip="Modality isolation guidance scale for the audio modality. Uses the same guidance rule as the video modality scale. For LTX-2.3 a value of 3.0 is suggested for both video and audio.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_guidance_rescale",
                    default_value=0.0,
                    tooltip="Guidance rescale factor for the audio modality. Applies the same overexposure-correction logic as guidance_rescale. Defaults to the video guidance_rescale value when not set explicitly.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="text_embeddings_path",
                    tooltip="Optional safetensors file with LTX2 HDR text embeddings. Used only when HDR IC-LoRA is active and overrides prompt text.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "fileSystemPicker": {
                            "allowFiles": True,
                            "allowDirectories": False,
                            "multiple": False,
                            "workspaceOnly": False,
                            "allowCreate": False,
                            "allowRename": False,
                            "fileTypes": [".safetensors"],
                            "initialPath": "C:\\Users\\ladipo.baruwa\\GriptapeNodes",
                        },
                        "hide": True,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="prompt",
                    default_value="",
                    tooltip="The prompt or prompts to guide video generation. If not defined, a prompt_embeds tensor must be passed instead.",
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
                    tooltip="The prompt or prompts not to guide video generation. Ignored when not using guidance (i.e. when guidance_scale is less than 1).",
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
                    default_value=1.0,
                    tooltip="Classifier-Free Guidance scale for the video modality. Higher values steer generation more strongly toward the prompt, usually at the expense of visual quality. A separate audio_guidance_scale controls the audio modality. For LTX-2.3 the authors suggest 3.0 for video.",
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
                    default_value=8,
                    tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={"hide": True},
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
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="EmptyLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Create Empty Latents",
                    metadata={
                        "position": {"x": 1468.4714370275533, "y": 239.79782184044856},
                        "tempId": "placing-1780541404744-gp9ggb",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Create",
                            description="Create a zero-filled (empty) latent tensor for a selected 🧨 Diffusers pipeline.",
                            display_name="Create Empty Latents",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "EmptyLatentNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 372},
                        "category": "ModularDiffusion/Create",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="num_frames", ui_options={"hide": False}, initial_setup=True
                )
            )
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="MediaGenConditioningNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Media Generation Conditioning",
                    metadata={
                        "position": {"x": 1447.9505276084883, "y": 1145.7803108648968},
                        "tempId": "placing-1780541429149-p3kiri",
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
                        "size": {"width": 600, "height": 595},
                        "category": "ModularDiffusion/Conditioning",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="num_images",
                    ui_options={"slider": {"min_val": 0, "max_val": 8}, "step": 1, "hide": True},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="video",
                    tooltip="Conditioning video.",
                    type="VideoUrlArtifact",
                    input_types=["VideoArtifact", "VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_property=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="frame_index",
                    default_value=0,
                    tooltip="Frame index in the output video where conditioning video is applied.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="video_strength",
                    default_value=1.0,
                    tooltip="Strength for the conditioning video.",
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
                    node_type="LoraActivationPipelineNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="LoRA Pipeline",
                    metadata={
                        "position": {"x": 753.2569287114065, "y": 239.79782184044856},
                        "tempId": "placing-1780541817132-yoxh17",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Pipeline",
                            description="Layers non-fused LoRA adapters on top of an existing pipeline. The base pipeline stays untouched, so the same model can power multiple branches (one with adapters, one without). Adapters are loaded and activated per generation, then released — ideal for in-context (IC) LoRAs, distillation/acceleration LoRAs, slider LoRAs, or any workflow that swaps adapters between runs.",
                            display_name="LoRA Pipeline",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "LoraActivationPipelineNode",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 408},
                        "category": "ModularDiffusion/Pipeline",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="loras_ParameterListUniqueParamID_b06cc6d3773b4ec0919aec630d257cc8",
                    default_value=[],
                    tooltip="One or more LoRA payloads from Load LoRA nodes. Adapters are activated dynamically at generation time and deactivated afterward; the base pipeline weights are never permanently modified.",
                    type="loras",
                    input_types=["loras"],
                    output_type="loras",
                    ui_options={},
                    mode_allowed_property=False,
                    parent_container_name="loras",
                    initial_setup=True,
                )
            )
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadLora",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Load LoRA",
                    metadata={
                        "position": {"x": 50, "y": 415.7978218404486},
                        "tempId": "placing-1780541887401-90dk5",
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Pipeline",
                            description="Load a LoRA file from a local path and expose it as a `loras` output. Wire to the Modular Diffusion Pipeline Builder to fuse the LoRA into the cached weights, or to a LoRA Pipeline node to activate it dynamically per generation.",
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
                    node_type="DiffusionPipelineGenerateLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Generate Media Latents (Modular Diffusion Pipeline)_1",
                    metadata={
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
                        "position": {"x": 2240.0034097647676, "y": 239.79782184044856},
                        "size": {"width": 637, "height": 1581},
                        "showaddparameter": False,
                        "category": "ModularDiffusion/Processing",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="use_stage_2",
                    default_value=False,
                    tooltip="If True, run stage 2 (refinement). If False, run stage 1 (initial generation).",
                    type="bool",
                    input_types=["bool"],
                    output_type="bool",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="stg_scale",
                    default_value=0.0,
                    tooltip="Spatio-Temporal Guidance (STG) scale for the video modality. STG moves the sample away from a perturbed version of the denoising model output, requiring one additional forward pass. 0.0 disables STG. For LTX-2.3 a value of 1.0 is suggested.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="modality_scale",
                    default_value=1.0,
                    tooltip="Modality isolation guidance scale for the video modality. Moves the sample away from a version generated without audio-to-video and video-to-audio cross attention, using a CFG-like estimate. Requires one additional forward pass. 1.0 disables modality guidance. For LTX-2.3 a value of 3.0 is suggested.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="guidance_rescale",
                    default_value=0.0,
                    tooltip="Guidance rescale factor for the video modality. Proposed by 'Common Diffusion Noise Schedules and Sample Steps are Flawed' to fix overexposure when using zero terminal SNR. 0.0 disables rescaling.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_guidance_scale",
                    default_value=1.0,
                    tooltip="CFG guidance scale for the audio modality. The same CFG update rule applies as for video, but video and audio can use different values. For LTX-2.3 the authors suggest 7.0 for audio alongside 3.0 for video.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_stg_scale",
                    default_value=0.0,
                    tooltip="Spatio-Temporal Guidance scale for the audio modality. Uses the same STG update rule as the video modality. For LTX-2.3 a value of 1.0 is suggested for both video and audio.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_modality_scale",
                    default_value=1.0,
                    tooltip="Modality isolation guidance scale for the audio modality. Uses the same guidance rule as the video modality scale. For LTX-2.3 a value of 3.0 is suggested for both video and audio.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="audio_guidance_rescale",
                    default_value=0.0,
                    tooltip="Guidance rescale factor for the audio modality. Applies the same overexposure-correction logic as guidance_rescale. Defaults to the video guidance_rescale value when not set explicitly.",
                    type="float",
                    input_types=["float"],
                    output_type="float",
                    ui_options={},
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="text_embeddings_path",
                    tooltip="Optional safetensors file with LTX2 HDR text embeddings. Used only when HDR IC-LoRA is active and overrides prompt text.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "fileSystemPicker": {
                            "allowFiles": True,
                            "allowDirectories": False,
                            "multiple": False,
                            "workspaceOnly": False,
                            "allowCreate": False,
                            "allowRename": False,
                            "fileTypes": [".safetensors"],
                            "initialPath": "C:\\Users\\ladipo.baruwa\\GriptapeNodes",
                        },
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="prompt",
                    default_value="",
                    tooltip="The prompt or prompts to guide video generation. If not defined, a prompt_embeds tensor must be passed instead.",
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
                    tooltip="The prompt or prompts not to guide video generation. Ignored when not using guidance (i.e. when guidance_scale is less than 1).",
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
                    default_value=1.0,
                    tooltip="Classifier-Free Guidance scale for the video modality. Higher values steer generation more strongly toward the prompt, usually at the expense of visual quality. A separate audio_guidance_scale controls the audio modality. For LTX-2.3 the authors suggest 3.0 for video.",
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
                    default_value=8,
                    tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.",
                    type="int",
                    input_types=["int"],
                    output_type="int",
                    ui_options={"hide": True},
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
                    parameter_name="additional_parameters_ParameterListUniqueParamID_7ee4d766fdef422c9fdcf9903ed44087",
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
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeDecodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Decode Media",
                    metadata={
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
                        "position": {"x": 858.6356639008663, "y": 108.86884482621812},
                        "size": {"width": 600, "height": 590},
                        "showaddparameter": False,
                        "category": "ModularDiffusion/Encode\\Decode",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node11_name):
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
                    tooltip="Generated video.",
                    type="VideoUrlArtifact",
                    input_types=["VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    initial_setup=True,
                )
            )
        node12_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DecodeHdrNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Decode HDR Latents",
                    metadata={
                        "library_node_metadata": NodeMetadata(
                            category="ModularDiffusion/Encode\\Decode",
                            description="Decode HDR VAE latent, apply tone mapping, and optionally save a raw EXR sequence.",
                            display_name="Decode HDR Latents",
                            tags=None,
                            icon=None,
                            color=None,
                            group="diffusion",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Modular Diffusion Nodes Library",
                        "node_type": "DecodeHdrNode",
                        "position": {"x": 3102.2680937839386, "y": 239.79782184044856},
                        "size": {"width": 600, "height": 1014},
                        "showaddparameter": False,
                        "category": "ModularDiffusion/Encode\\Decode",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node12_name):
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
                    parameter_name="exr_output_folder",
                    tooltip="File or folder path for saving the raw linear HDR EXR sequence. Leave empty to skip saving EXRs.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "fileSystemPicker": {
                            "allowFiles": True,
                            "allowDirectories": True,
                            "multiple": False,
                            "workspaceOnly": False,
                            "allowCreate": False,
                            "allowRename": False,
                            "fileTypes": [".exr"],
                            "initialPath": "C:\\Users\\ladipo.baruwa\\GriptapeNodes",
                        }
                    },
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="save_exr_as_half_float",
                    default_value=True,
                    tooltip="Save EXR as float16 — 2.5x smaller files with negligible quality loss",
                    type="bool",
                    input_types=["bool"],
                    output_type="bool",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_output=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="tone_mapping",
                    default_value="aces_filmic",
                    tooltip="Tone mapping function applied to linear HDR frames before encoding to SDR MP4.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "simple_dropdown": ["clip", "reinhard", "aces_filmic", "cv2_reinhard", "cv2_mantiuk"],
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
                    tooltip="Generated video.",
                    type="VideoUrlArtifact",
                    input_types=["VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    initial_setup=True,
                )
            )
        node13_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeDecodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Decode Media_1",
                    metadata={
                        "position": {"x": 50, "y": 1145.7803108648968},
                        "tempId": "placing-1780586179349-408xg4",
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
                        "size": {"width": 610, "height": 571},
                        "category": "ModularDiffusion/Encode\\Decode",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node13_name):
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
                    tooltip="Generated video.",
                    type="VideoUrlArtifact",
                    input_types=["VideoUrlArtifact"],
                    output_type="VideoUrlArtifact",
                    ui_options={},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    initial_setup=True,
                )
            )
        node14_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note",
                    metadata={
                        "position": {"x": 1467.3757850801449, "y": 629.2021781595514},
                        "tempId": "placing-1780666908298-2hcvsi",
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
                        "size": {"width": 600, "height": 136},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node15_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_1",
                    metadata={
                        "position": {"x": 754.3398174416643, "y": 662},
                        "tempId": "placing-1780667404209-mrzl8",
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
                        "size": {"width": 600, "height": 135},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node16_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_2",
                    metadata={
                        "position": {"x": 1447.9505276084883, "y": 1770.457033699852},
                        "tempId": "placing-1780666908298-2hcvsi",
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
                        "size": {"width": 600, "height": 136},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node17_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_3",
                    metadata={
                        "position": {"x": 867.3757850801449, "y": 723.1633240048466},
                        "tempId": "placing-1780667404209-mrzl8",
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
                        "size": {"width": 600, "height": 135},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node18_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_4",
                    metadata={
                        "position": {"x": 2240.0034097647676, "y": 71.75487108234},
                        "tempId": "placing-1780667404209-mrzl8",
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
                        "size": {"width": 636, "height": 148},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node19_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_5",
                    metadata={
                        "position": {"x": 50, "y": 719.5042726015475},
                        "tempId": "placing-1780667404209-mrzl8",
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
                        "size": {"width": 604, "height": 148},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node20_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_6",
                    metadata={
                        "position": {"x": 754.3398174416643, "y": 817.565719019887},
                        "tempId": "placing-1780667404209-mrzl8",
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
                        "size": {"width": 600, "height": 135},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node21_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Group",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Latent Upsampling",
                    metadata={
                        "position": {"x": 2477.796660386473, "y": 167.79782184044856},
                        "tempId": "placing-1780666736511-s8muj5",
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
                        "size": {"width": 1576, "height": 913},
                        "expanded_dimensions": {"width": 1576, "height": 913},
                        "color": "#f59e0b",
                    },
                    node_names_to_add=[node4_name, node11_name, node17_name],
                )
            )
        ).node_name
        node22_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Group",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text-to-Video",
                    metadata={
                        "position": {"x": -42.986154547379556, "y": 167.79782184044856},
                        "tempId": "placing-1780665641108-1p574",
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
                        "size": {"width": 2294, "height": 1445},
                        "expanded_dimensions": {"width": 2294, "height": 1445},
                    },
                    node_names_to_add=[node0_name, node1_name, node2_name, node3_name, node5_name],
                )
            )
        ).node_name
        node23_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Group",
                    specific_library_name="Griptape Nodes Library",
                    node_name="SDR-to-HDR",
                    metadata={
                        "position": {"x": 4319.100541514291, "y": 36.86884482621812},
                        "tempId": "placing-1780666672631-ezl05w",
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
                        "size": {"width": 3784, "height": 1956},
                        "color": "#10b981",
                        "expanded_dimensions": {"width": 3784, "height": 1956},
                    },
                    node_names_to_add=[
                        node6_name,
                        node7_name,
                        node8_name,
                        node9_name,
                        node10_name,
                        node12_name,
                        node13_name,
                        node15_name,
                        node14_name,
                        node16_name,
                        node19_name,
                        node20_name,
                        node18_name,
                    ],
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
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
                target_node_name=node5_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="pipeline",
                target_node_name=node5_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node9_name,
                source_parameter_name="loras",
                target_node_name=node8_name,
                target_parameter_name="loras_ParameterListUniqueParamID_b06cc6d3773b4ec0919aec630d257cc8",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node8_name,
                source_parameter_name="lora_pipeline",
                target_node_name=node6_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node6_name,
                source_parameter_name="output_latent",
                target_node_name=node10_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node6_name,
                source_parameter_name="pipeline",
                target_node_name=node10_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="output_latent",
                target_node_name=node11_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node5_name,
                source_parameter_name="pipeline",
                target_node_name=node11_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="pipeline",
                target_node_name=node12_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="pipeline",
                target_node_name=node8_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="output_video",
                target_node_name=node7_name,
                target_parameter_name="video",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="pipeline",
                target_node_name=node13_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="text",
                target_node_name=node5_name,
                target_parameter_name="prompt",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="text",
                target_node_name=node5_name,
                target_parameter_name="negative_prompt",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node5_name,
                source_parameter_name="output_latent",
                target_node_name=node4_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node5_name,
                source_parameter_name="output_latent",
                target_node_name=node13_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node7_name,
                source_parameter_name="conditioning",
                target_node_name=node10_name,
                target_parameter_name="additional_parameters_ParameterListUniqueParamID_7ee4d766fdef422c9fdcf9903ed44087",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="output_latent",
                target_node_name=node12_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["f5d6f13a-c2ce-43c3-adfe-89f98f054926"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["f5d6f13a-c2ce-43c3-adfe-89f98f054926"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["709818f7-75cc-4171-907d-4235c3069ddf"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["1e031b4a-3c13-4947-b7ae-1e173021e197"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["8e639bfd-85d3-4c04-8352-81a7694b1891"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["0ce709d1-c8c3-4328-8f5f-0bfd88a8d543"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_tiling",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["01e61129-a4f5-434e-bffe-ad2e73e0fcef"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["01e61129-a4f5-434e-bffe-ad2e73e0fcef"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["c73f0805-479a-4a57-a6bf-7933b8d32744"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["824f4f37-b5d3-46fa-be55-bbf0a2b2714c"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["310c47b1-03f8-4b6a-a0d0-295f84477d47"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["5b3bcdb5-759a-447f-aeca-dba8cf9fdb3d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["88bca94d-8553-4ee1-8cc3-be93eea2811b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["78833574-ee15-4800-9c45-dc7bf28007a9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4957f5a1-c540-46af-8d7b-9f96a94dafce"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["cb5054f4-b2d3-456e-a4b5-c4e4c55dd474"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["d4055550-9d1d-42e0-9e0f-0a024502afa8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["d4055550-9d1d-42e0-9e0f-0a024502afa8"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["50fa2e6b-0e3d-4f26-a025-fa48304d5646"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["50fa2e6b-0e3d-4f26-a025-fa48304d5646"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["709818f7-75cc-4171-907d-4235c3069ddf"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="upsampler_model",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["21d1bbc2-a4ef-4f18-858b-5d0cf4355271"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["310c47b1-03f8-4b6a-a0d0-295f84477d47"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["9db99aa5-5969-4530-9aa1-0eb9d2f7c7de"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["b58239c3-c88f-4eea-b0b7-d9d599e1187b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="use_stage_2",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="stg_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="modality_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["004a187e-260b-41a4-a048-b726f7f4c429"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_rescale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["57cea271-f5f4-4d4c-b474-3e85a1d6da87"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_guidance_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["1e828835-8d61-41d1-8070-9ab369b12b73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_stg_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_modality_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["004a187e-260b-41a4-a048-b726f7f4c429"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_guidance_rescale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["57cea271-f5f4-4d4c-b474-3e85a1d6da87"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["d4055550-9d1d-42e0-9e0f-0a024502afa8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="negative_prompt",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["50fa2e6b-0e3d-4f26-a025-fa48304d5646"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["004a187e-260b-41a4-a048-b726f7f4c429"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["cb5054f4-b2d3-456e-a4b5-c4e4c55dd474"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["5b3bcdb5-759a-447f-aeca-dba8cf9fdb3d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["867b7797-308e-40b4-a39f-f8bd9ed38698"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["b8e65bd4-b663-4d33-98f7-df0c0b110160"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["cdbd8985-40f1-4ab7-81d0-9df2c3cd2810"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["863623c0-13b3-4280-ac69-bfae84c6e300"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["3c58ba54-dc51-46a6-8b90-21ff352c4ecf"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["4957f5a1-c540-46af-8d7b-9f96a94dafce"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["cb5054f4-b2d3-456e-a4b5-c4e4c55dd474"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mode",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["801aacb1-5e0b-4043-9a23-1f5390c3c3bd"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_images",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["9db99aa5-5969-4530-9aa1-0eb9d2f7c7de"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="frame_index",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["9db99aa5-5969-4530-9aa1-0eb9d2f7c7de"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="video_strength",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["310c47b1-03f8-4b6a-a0d0-295f84477d47"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="lora_pipeline",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["c529ff6d-5fa5-4eaf-97fd-8b429b3f7fb4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="lora_pipeline",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["c529ff6d-5fa5-4eaf-97fd-8b429b3f7fb4"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["c4e83959-ed8a-42fe-9065-97213d5d8a67"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras_ParameterListUniqueParamID_b06cc6d3773b4ec0919aec630d257cc8",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["d5f2a5f3-8096-4d68-9231-f43de667df54"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["75f48310-8775-4e32-87fd-6555a2b0fa6b"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="file_path",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["cacc941c-7603-43dc-a176-31dca85c041f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="weight",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["d5f2a5f3-8096-4d68-9231-f43de667df54"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["d5f2a5f3-8096-4d68-9231-f43de667df54"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="trigger_phrase",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["595d6132-c70e-4d6f-b1b3-320a0eee1e59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["cdbd8985-40f1-4ab7-81d0-9df2c3cd2810"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["9db99aa5-5969-4530-9aa1-0eb9d2f7c7de"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["b58239c3-c88f-4eea-b0b7-d9d599e1187b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="use_stage_2",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="stg_scale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["fb3be385-3085-4c47-927b-704b35815040"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="modality_scale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_rescale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["fb3be385-3085-4c47-927b-704b35815040"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_guidance_scale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["df4a1803-78f3-4a2e-a4df-c5c7a129db52"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_stg_scale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["fb3be385-3085-4c47-927b-704b35815040"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_modality_scale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_guidance_rescale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["9db99aa5-5969-4530-9aa1-0eb9d2f7c7de"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text_embeddings_path",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["16029530-fa18-4189-8f0f-5995497160e4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["595d6132-c70e-4d6f-b1b3-320a0eee1e59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="negative_prompt",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["595d6132-c70e-4d6f-b1b3-320a0eee1e59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_scale",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["6ce85e10-5ee5-4136-9364-82128887e113"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["3e0a105d-c6e9-438f-b833-644dd3f56783"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["1e4d883d-52ee-4c09-92a3-a66fc46efc5c"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["5b3bcdb5-759a-447f-aeca-dba8cf9fdb3d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["d5841411-0df9-4ad6-9cb1-8d32de272e98"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["4953712a-0d5e-427d-95e3-b89cd9523528"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["17dc8cc5-8447-4992-a259-1253dab0b9d4"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["310c47b1-03f8-4b6a-a0d0-295f84477d47"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["39ed6d6e-c86f-43e4-98c2-02e16d8bffcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["cdbd8985-40f1-4ab7-81d0-9df2c3cd2810"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="exr_output_folder",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["f0040d27-0e9e-4be6-bc10-2576ceb1c19e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="save_exr_as_half_float",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["ab599fba-68b7-4747-acd9-f4532cf2b4cd"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="tone_mapping",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["d7153365-e6ce-4d9a-b7d9-239abf8ad6e6"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["39ed6d6e-c86f-43e4-98c2-02e16d8bffcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node13_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["310c47b1-03f8-4b6a-a0d0-295f84477d47"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["39ed6d6e-c86f-43e4-98c2-02e16d8bffcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node14_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node14_name,
                    value=top_level_unique_values_dict["ff6cfae1-121b-4b04-9aee-e1f6d37f62f7"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node15_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node15_name,
                    value=top_level_unique_values_dict["b78f6e9e-7980-4c59-903a-0ed764fac97d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node16_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node16_name,
                    value=top_level_unique_values_dict["7c77da5a-3f98-4387-ad21-11b0bf42b328"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node17_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node17_name,
                    value=top_level_unique_values_dict["bb901318-c050-4fe2-8c91-773b07680181"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node18_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node18_name,
                    value=top_level_unique_values_dict["3f872933-604d-4a7f-be03-1650a1b359d9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node19_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node19_name,
                    value=top_level_unique_values_dict["bc8e0057-0fd7-4a74-90bf-58604fa2c8a6"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node20_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node20_name,
                    value=top_level_unique_values_dict["3bbe9835-9577-4fa9-9d44-8c07d25559fc"],
                    initial_setup=True,
                    is_output=False,
                )
            )
