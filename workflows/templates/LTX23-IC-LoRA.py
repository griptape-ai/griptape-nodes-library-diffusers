# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "LTX23-IC-LoRA"
# schema_version = "0.18.0"
# engine_version_created_with = "0.85.4"
# node_libraries_referenced = [["Griptape Nodes Library", "0.75.0"], ["Griptape Modular Diffusion Nodes Library", "0.1.1"]]
# node_types_used = [["Griptape Modular Diffusion Nodes Library", "DiffusionPipelineGenerateLatentNode"], ["Griptape Modular Diffusion Nodes Library", "EmptyLatentNode"], ["Griptape Modular Diffusion Nodes Library", "LatentDiffusionPipelineBuilderNode"], ["Griptape Modular Diffusion Nodes Library", "LoadLora"], ["Griptape Modular Diffusion Nodes Library", "LoraActivationPipelineNode"], ["Griptape Modular Diffusion Nodes Library", "MediaGenConditioningNode"], ["Griptape Modular Diffusion Nodes Library", "VaeDecodeNode"], ["Griptape Nodes Library", "LoadVideo"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "TextInput"]]
# description = "LTX-2.3 IC-LoRA workflow that conditions text-to-video generation on a reference video via an in-context LoRA adapter, producing an output clip that follows the reference's structure or motion."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-diffusers/main/workflows/templates/LTX23-IC-LoRA.png"
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
        "ae9ee4df-4cda-4a65-896d-a5de6412adcb": pickle.loads(
            b"\x80\x04\x95u\x04\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89\x8c\x1c_extra_runtime_adapter_steps\x94]\x94\x8c6modular_diffusion_nodes_library.utils.lora_apply_utils\x94\x8c\x1eLoraPipelineRuntimeAdapterStep\x94\x93\x94)\x81\x94}\x94h\x15}\x94\x8c<W:\\lora\\ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors\x94\x8c/modular_diffusion_nodes_library.utils.lora_spec\x94\x8c\x08LoraSpec\x94\x93\x94)\x81\x94}\x94(\x8c\x04path\x94h-\x8c\x06weight\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x0etrigger_phrase\x94Nubssbaub."
        ),
        "f139ce36-b96f-4ea9-a9e4-ee0128521840": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x06."),
        "b08753e0-b1cb-428d-9714-1cb438c9335f": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x04."),
        "fcfce2b6-e69d-4178-8c5e-41668a6c5ae5": pickle.loads(b"\x80\x04Ky."),
        "417cb7fd-2ec8-4fa7-959f-1c2d1a0b0c1e": pickle.loads(b"\x80\x04K\x14."),
        "c020cf10-ddad-455e-b769-fce42af617fa": pickle.loads(
            b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05video\x94."
        ),
        "a176d177-1dbe-44e8-bc83-d4c18aa64921": pickle.loads(b"\x80\x04K\x00."),
        "5fc76d36-d4fb-4958-809c-f3fdb8850c3d": pickle.loads(
            b"\x80\x04\x95v\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.video_url_artifact\x94\x8c\x10VideoUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10VideoUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.video_url_artifact\x94\x8c\x02id\x94\x8c e65499ab76d34708be6632ae4f6fdff0\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\n\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8cY{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/canny_video.mp4\x94ub."
        ),
        "437d8e39-099b-46fd-ac90-846029ebac58": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xf0\x00\x00\x00\x00\x00\x00."
        ),
        "9c491fce-ee8a-4ccc-9e87-153c600f1016": pickle.loads(
            b"\x80\x04\x95\x87\x00\x00\x00\x00\x00\x00\x00\x8c\x83{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/lora/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors\x94."
        ),
        "7dd0c927-1331-4963-b32d-9b9dff5bcdf9": pickle.loads(
            b"\x80\x04\x95L\x00\x00\x00\x00\x00\x00\x00}\x94\x8c<W:\\lora\\ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors\x94G?\xf0\x00\x00\x00\x00\x00\x00s."
        ),
        "cb4861ef-a59f-4c50-a9ae-03b004e7a943": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "140211d0-aeaa-4e84-be91-a75b73f05e1f": pickle.loads(
            b'\x80\x04\x95\x8e\x05\x00\x00\x00\x00\x00\x00X\x87\x05\x00\x00grainy, blurry, low quality, flickering, static noise, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, grainy texture, poor lighting, motion blur, distorted proportions, unnatural skin tones, deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, wrong hand count, artifacts around text, unreadable text on shirt or hat, incorrect lettering on cap ("PNTR"), incorrect t-shirt slogan ("JUST DO IT"), missing microphone, misplaced microphone, inconsistent perspective, camera shake, incorrect depth of field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, smiling, laughing, exaggerated sadness, wrong gaze direction, eyes looking at camera, mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, off-sync audio, missing sniff sounds, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, missing door or shelves, missing shallow depth of field, flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts. music\x94.'
        ),
        "2f584e0d-90c6-4bb2-a441-1991abaa289f": pickle.loads(
            b"\x80\x04\x95z\x01\x00\x00\x00\x00\x00\x00Xs\x01\x00\x00A large, translucent jellyfish drifts gracefully through a beautiful, vibrant blue ocean background, illuminated by soft, ethereal light filtering from above. Its rounded umbrella bell gently expands and contracts in a rhythmic pulse, while its long, delicate tentacles and frilly oral arms trail and sway elegantly in the current amidst tiny drifting plankton particles.\x94."
        ),
        "d380d95e-ea8e-4af6-a0f1-fd38fd973bae": pickle.loads(b"\x80\x04\x89."),
        "181637de-83b4-4acf-bb41-bd52ba647d72": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00J\xff\xff\xff\xff."
        ),
        "44b0a5ef-a55a-4238-98e5-505286ac6c44": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00."
        ),
        "223f33a5-cfce-4968-89b1-d7111751333b": pickle.loads(b"\x80\x04K(."),
        "e470ba87-ab58-4eeb-abc8-3947ca73d2b1": pickle.loads(b"\x80\x04K*."),
        "a987c762-0e35-4a33-89fb-5cc87a721042": pickle.loads(b"\x80\x04]\x94."),
        "835936a9-a01f-4a23-aa07-11c086d395d6": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xc9\x99\x99\x99\x99\x99\x9a."
        ),
        "b206d4f9-ec3a-4417-8fec-cd5bda0464b4": pickle.loads(
            b"\x80\x04\x95A\x01\x00\x00\x00\x00\x00\x00X:\x01\x00\x00Completed inference step 2 of 40. 0.95 s/it\nCompleted inference step 3 of 40. 0.97 s/it\nCompleted inference step 4 of 40. 0.97 s/it\nCompleted inference step 5 of 40. 0.97 s/it\nCompleted inference step 6 of 40. 0.97 s/it\nCompleted inference step 7 of 40. 0.97 s/it\nCompleted inference step 8 of 40. 0.97 s/it\nDone.\n\x94."
        ),
        "7bb1858c-c47e-4dcb-9c0c-d69290ccef59": pickle.loads(
            b"\x80\x04\x958\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "9ef9dab5-1019-4757-a7ee-7cc5fe5a16c8": pickle.loads(
            b"\x80\x04\x95u\x04\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89\x8c\x1c_extra_runtime_adapter_steps\x94]\x94\x8c6modular_diffusion_nodes_library.utils.lora_apply_utils\x94\x8c\x1eLoraPipelineRuntimeAdapterStep\x94\x93\x94)\x81\x94}\x94h\x15}\x94\x8c<W:\\lora\\ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors\x94\x8c/modular_diffusion_nodes_library.utils.lora_spec\x94\x8c\x08LoraSpec\x94\x93\x94)\x81\x94}\x94(\x8c\x04path\x94h-\x8c\x06weight\x94G?\xf0\x00\x00\x00\x00\x00\x00\x8c\x0etrigger_phrase\x94Nubssbaub."
        ),
        "9f2888fb-61d5-4645-b94b-bfd61a78de12": pickle.loads(
            b"\x80\x04\x95O\x00\x00\x00\x00\x00\x00\x00]\x94}\x94\x8c<W:\\lora\\ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors\x94G?\xf0\x00\x00\x00\x00\x00\x00sa."
        ),
        "fc39e8ec-7c7d-4c1e-b822-30e77b2244e9": pickle.loads(
            b"\x80\x04\x95\x02\x03\x00\x00\x00\x00\x00\x00X\xfb\x02\x00\x00Pipeline configuration hash: LTX2Pipeline-896e6ce4ad4a67cc6031fd69fd19a2ae62b7b7646f4e6a46753729f905ee7e77-0\nPipeline configuration hash: LTX2Pipeline-896e6ce4ad4a67cc6031fd69fd19a2ae62b7b7646f4e6a46753729f905ee7e77-0\nPipeline configuration hash: LTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\nPipeline configuration hash: FluxPipeline-87440644b8f5e9a9cb83c5797ef0097df7f5e0e7d251e06dafa64fa5e360d57f-0\nPipeline configuration hash: LTX2Pipeline-7f3190ed0d0d7967d5a9cf80dea1c82f5b3527721614d91cb05b5ac3f7fc2f35-0\nPipeline configuration hash: LTX2Pipeline-891e4d37fa3abfdce0b084b3d1fd7fff8a5bcc1069d3fbb5cdc8785892e2bc72-0\nPipeline configuration hash: LTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\n\x94."
        ),
        "857ec647-6ca5-4efe-b827-4ba74c214ebe": pickle.loads(
            b"\x80\x04\x958\x03\x00\x00\x00\x00\x00\x00\x8c@modular_diffusion_nodes_library.artifact_utils.pipeline_artifact\x94\x8c\x19DiffusionPipelineArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\rpipeline_name\x94\x8c\x0cLTX2Pipeline\x94\x8c\x0bconfig_hash\x94\x8cOLTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\x94\x8c\x0f_builder_module\x94\x8cCmodular_diffusion_nodes_library.standard_parameters.ltx2_parameters\x94\x8c\x13_builder_class_name\x94\x8c\x16LTX2PipelineParameters\x94\x8c\x0b_build_data\x94}\x94(\x8c\x0cbase_repo_id\x94\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94\x8c\rbase_revision\x94\x8c(581689133ea7f534f2df5ce7551563398910a915\x94\x8c\x0cis_distilled\x94\x88u\x8c\x11_build_data_error\x94N\x8c\x06_loras\x94}\x94\x8c\x14_optimization_kwargs\x94}\x94(\x8c\x1cmemory_optimization_strategy\x94\x8c\tAutomatic\x94\x8c\x11attention_slicing\x94\x89\x8c\x0bvae_slicing\x94\x89\x8c\nvae_tiling\x94\x89\x8c\x1dtransformer_layerwise_casting\x94\x89\x8c\x14cpu_offload_strategy\x94\x8c\x04None\x94\x8c\x11quantization_mode\x94h u\x8c\x10_is_prequantized\x94\x89\x8c\x1b_supports_layerwise_casting\x94\x88\x8c\x14_requires_device_map\x94\x89ub."
        ),
        "b2a6f3a5-02d7-41b4-b802-fc56be1b5a37": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04LTX2\x94."
        ),
        "d419600b-0420-4a10-b494-7d9a391f9639": pickle.loads(
            b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cLTX2Pipeline\x94."
        ),
        "8ba3e856-55a2-4a4d-bd57-77b891552fa4": pickle.loads(
            b"\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c!dg845/LTX-2.3-Distilled-Diffusers\x94."
        ),
        "92ff4a90-0abc-46b0-bb07-0c779ec50515": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tAutomatic\x94."
        ),
        "fabf822c-c6ee-418a-9303-e61150b7d828": pickle.loads(
            b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04None\x94."
        ),
        "d177aa75-1e6a-4efa-ae3b-459ffd6c79d5": pickle.loads(b"\x80\x04]\x94."),
        "7d2ec1de-d322-4fe3-a6c6-6fd8db72c21c": pickle.loads(
            b"\x80\x04\x95\x02\x03\x00\x00\x00\x00\x00\x00X\xfb\x02\x00\x00Pipeline configuration hash: FluxPipeline-87440644b8f5e9a9cb83c5797ef0097df7f5e0e7d251e06dafa64fa5e360d57f-0\nPipeline configuration hash: LTX2Pipeline-7f3190ed0d0d7967d5a9cf80dea1c82f5b3527721614d91cb05b5ac3f7fc2f35-0\nPipeline configuration hash: LTX2Pipeline-7f3190ed0d0d7967d5a9cf80dea1c82f5b3527721614d91cb05b5ac3f7fc2f35-0\nPipeline configuration hash: LTX2Pipeline-7f3190ed0d0d7967d5a9cf80dea1c82f5b3527721614d91cb05b5ac3f7fc2f35-0\nPipeline configuration hash: LTX2Pipeline-7f3190ed0d0d7967d5a9cf80dea1c82f5b3527721614d91cb05b5ac3f7fc2f35-0\nPipeline configuration hash: LTX2Pipeline-891e4d37fa3abfdce0b084b3d1fd7fff8a5bcc1069d3fbb5cdc8785892e2bc72-0\nPipeline configuration hash: LTX2Pipeline-48930e275720e49d101623519a65ecff024cf20f66b77a3c5702b81dc71cad48-0\n\x94."
        ),
        "d3c4fd45-2d1f-4767-9016-27376fe5379a": pickle.loads(
            b"\x80\x04\x95\xa9\x00\x00\x00\x00\x00\x00\x00\x8c\xa5Some LoRAs to try:\n- https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control\n- https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control\x94."
        ),
        "9abe6dd2-2208-46a9-813a-ffff35d37be3": pickle.loads(
            b"\x80\x04\x95:\x01\x00\x00\x00\x00\x00\x00X3\x01\x00\x00It is important to read the model card or documentation of the LoRA to ensure the reference media is positionally aligned. LTX-2.3 22B IC-LoRA Union Control was trained with downscaled reference latents by a factor of 2, therefore the resolution of the input video should be 1/2 that of the generated media.\x94."
        ),
        "51679734-9078-4265-8e60-8deea3ed450b": pickle.loads(
            b"\x80\x04\x95\x97\x00\x00\x00\x00\x00\x00\x00\x8c\x93The resolution of the reference must be correct in relation to the reference video signal. Read the LoRA model card or documentation for more info.\x94."
        ),
        "052f29b7-0773-4003-90de-c3909e6c4c1d": pickle.loads(b"\x80\x04K\x19."),
        "819a0121-dff6-4e37-8b2b-b329197ed555": pickle.loads(
            b'\x80\x04\x95\xba\x00\x00\x00\x00\x00\x00\x00\x8c\xb6LTX2.3 distilled model uses predefined schedules depending on "use stage 2" parameter settings.\n- Stage 1 i.e. use_stage_2 = False: 8 steps\n- Stage 2 i.e. use_stage_2 = True: 4 steps\x94.'
        ),
        "f2a34255-b240-4b96-9565-865542b779d8": pickle.loads(
            b"\x80\x04\x95]\x00\x00\x00\x00\x00\x00\x00\x8cY{project_dir}/libraries/griptape-nodes-library-diffusers/workflows/assets/canny_video.mp4\x94."
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
                    node_type="EmptyLatentNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Create Empty Latents",
                    metadata={
                        "position": {"x": 1820.5005495162532, "y": 224.51830512809192},
                        "tempId": "placing-1778844888915-td4n5",
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
                        "size": {"width": 600, "height": 416},
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
                    node_type="MediaGenConditioningNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Media Generation Conditioning",
                    metadata={
                        "position": {"x": 1836.0085941183756, "y": 1664.523005562608},
                        "tempId": "placing-1778844934103-f39nfy",
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
                        "size": {"width": 630, "height": 626},
                        "category": "ModularDiffusion/Conditioning",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
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
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadLora",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Load LoRA",
                    metadata={
                        "position": {"x": 365.71370776486015, "y": 1198.0631022055145},
                        "tempId": "placing-1778845073058-z3ek27",
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
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input_1",
                    metadata={
                        "position": {"x": 1820.5005495162532, "y": 1025.248989472945},
                        "tempId": "placing-1778845418608-0xjoc",
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
                        "size": {"width": 634, "height": 570},
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
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Text Input_2",
                    metadata={
                        "position": {"x": 1820.5005495162532, "y": 691.5985623262355},
                        "tempId": "placing-1778877644448-i6s5km",
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
                        "size": {"height": 295, "width": 600},
                        "category": "text",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
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
                        "position": {"x": 2567.731232392216, "y": 206.01830512809192},
                        "size": {"width": 600, "height": 1435},
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
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="additional_parameters_ParameterListUniqueParamID_40499afa1c2547f4bcf47aadb1a9c994",
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
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoraActivationPipelineNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="LoRA Pipeline_1",
                    metadata={
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
                        "position": {"x": 1103.389781149374, "y": 224.51830512809192},
                        "size": {"width": 602, "height": 728},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="loras_ParameterListUniqueParamID_c1f3b769a62e4b9b83f88c4b21e89be4",
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
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LatentDiffusionPipelineBuilderNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="Modular Diffusion Pipeline Builder_1",
                    metadata={
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
                        "position": {"x": 365.71370776486015, "y": 224.51830512809192},
                        "size": {"width": 600, "height": 918},
                        "showaddparameter": False,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node7_name):
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
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note",
                    metadata={
                        "position": {"x": 380.1286130136938, "y": 1530.9350110755265},
                        "tempId": "placing-1780673783875-0di1zt",
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
                        "size": {"width": 601, "height": 211},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_1",
                    metadata={
                        "position": {"x": 1150.1200952649253, "y": 1435.018305128092},
                        "tempId": "placing-1780673783875-0di1zt",
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
                        "size": {"width": 625, "height": 207},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_2",
                    metadata={
                        "position": {"x": 1820.5005495162532, "y": 70.01830512809192},
                        "tempId": "placing-1780673783875-0di1zt",
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
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="VaeDecodeNode",
                    specific_library_name="Griptape Modular Diffusion Nodes Library",
                    node_name="VAE Decode Latents",
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
                        "position": {"x": 3294.480963561947, "y": 206.01830512809192},
                        "size": {"width": 600, "height": 765},
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
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_3",
                    metadata={
                        "position": {"x": 2567.731232392216, "y": -49.586747372652226},
                        "tempId": "placing-1780673783875-0di1zt",
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
                        "size": {"width": 600, "height": 231},
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node13_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LoadVideo",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Load Video",
                    metadata={
                        "library_node_metadata": NodeMetadata(
                            category="video",
                            description="Loads video files into your workflow",
                            display_name="Load Video",
                            tags=None,
                            icon="file-video",
                            color=None,
                            group="Input/Output",
                            deprecation=None,
                            is_node_group=None,
                        ),
                        "library": "Griptape Nodes Library",
                        "node_type": "LoadVideo",
                        "position": {"x": 1150.1200952649253, "y": 1664.523005562608},
                        "size": {"width": 629, "height": 625},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="output_latent",
                target_node_name=node5_name,
                target_parameter_name="input_latent",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="pipeline",
                target_node_name=node5_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node6_name,
                source_parameter_name="lora_pipeline",
                target_node_name=node0_name,
                target_parameter_name="pipeline",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="loras",
                target_node_name=node6_name,
                target_parameter_name="loras_ParameterListUniqueParamID_c1f3b769a62e4b9b83f88c4b21e89be4",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node7_name,
                source_parameter_name="pipeline",
                target_node_name=node6_name,
                target_parameter_name="pipeline",
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
                source_node_name=node13_name,
                source_parameter_name="video",
                target_node_name=node1_name,
                target_parameter_name="video",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
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
                source_node_name=node1_name,
                source_parameter_name="conditioning",
                target_node_name=node5_name,
                target_parameter_name="additional_parameters_ParameterListUniqueParamID_40499afa1c2547f4bcf47aadb1a9c994",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node5_name,
                source_parameter_name="output_latent",
                target_node_name=node11_name,
                target_parameter_name="latent_tensor",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["ae9ee4df-4cda-4a65-896d-a5de6412adcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="width",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["f139ce36-b96f-4ea9-a9e4-ee0128521840"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="height",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["b08753e0-b1cb-428d-9714-1cb438c9335f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_frames",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["fcfce2b6-e69d-4178-8c5e-41668a6c5ae5"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["417cb7fd-2ec8-4fa7-959f-1c2d1a0b0c1e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="mode",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["c020cf10-ddad-455e-b769-fce42af617fa"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_images",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["a176d177-1dbe-44e8-bc83-d4c18aa64921"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="video",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["5fc76d36-d4fb-4958-809c-f3fdb8850c3d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="frame_index",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["a176d177-1dbe-44e8-bc83-d4c18aa64921"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="video_strength",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["437d8e39-099b-46fd-ac90-846029ebac58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="file_path",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["9c491fce-ee8a-4ccc-9e87-153c600f1016"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="weight",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["437d8e39-099b-46fd-ac90-846029ebac58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["7dd0c927-1331-4963-b32d-9b9dff5bcdf9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["7dd0c927-1331-4963-b32d-9b9dff5bcdf9"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="trigger_phrase",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["cb4861ef-a59f-4c50-a9ae-03b004e7a943"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["140211d0-aeaa-4e84-be91-a75b73f05e1f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["140211d0-aeaa-4e84-be91-a75b73f05e1f"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["2f584e0d-90c6-4bb2-a441-1991abaa289f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["2f584e0d-90c6-4bb2-a441-1991abaa289f"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["ae9ee4df-4cda-4a65-896d-a5de6412adcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="add_noise",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="start_step",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["a176d177-1dbe-44e8-bc83-d4c18aa64921"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="end_step",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["181637de-83b4-4acf-bb41-bd52ba647d72"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="return_fully_denoised",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="use_stage_2",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="stg_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["44b0a5ef-a55a-4238-98e5-505286ac6c44"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="modality_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["437d8e39-099b-46fd-ac90-846029ebac58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_rescale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["44b0a5ef-a55a-4238-98e5-505286ac6c44"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_guidance_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["437d8e39-099b-46fd-ac90-846029ebac58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_stg_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["44b0a5ef-a55a-4238-98e5-505286ac6c44"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_modality_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["437d8e39-099b-46fd-ac90-846029ebac58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="audio_guidance_rescale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["44b0a5ef-a55a-4238-98e5-505286ac6c44"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["2f584e0d-90c6-4bb2-a441-1991abaa289f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="negative_prompt",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["140211d0-aeaa-4e84-be91-a75b73f05e1f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="guidance_scale",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["437d8e39-099b-46fd-ac90-846029ebac58"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="num_inference_steps",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["223f33a5-cfce-4968-89b1-d7111751333b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="randomize_seed",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="seed",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["e470ba87-ab58-4eeb-abc8-3947ca73d2b1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_parameters",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["a987c762-0e35-4a33-89fb-5cc87a721042"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="progress",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["835936a9-a01f-4a23-aa07-11c086d395d6"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["b206d4f9-ec3a-4417-8fec-cd5bda0464b4"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["7bb1858c-c47e-4dcb-9c0c-d69290ccef59"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="lora_pipeline",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["9ef9dab5-1019-4757-a7ee-7cc5fe5a16c8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="lora_pipeline",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["9ef9dab5-1019-4757-a7ee-7cc5fe5a16c8"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["9f2888fb-61d5-4645-b94b-bfd61a78de12"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras_ParameterListUniqueParamID_c1f3b769a62e4b9b83f88c4b21e89be4",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["7dd0c927-1331-4963-b32d-9b9dff5bcdf9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node6_name,
                    value=top_level_unique_values_dict["fc39e8ec-7c7d-4c1e-b822-30e77b2244e9"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["857ec647-6ca5-4efe-b827-4ba74c214ebe"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["857ec647-6ca5-4efe-b827-4ba74c214ebe"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="provider",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["b2a6f3a5-02d7-41b4-b802-fc56be1b5a37"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline_type",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["d419600b-0420-4a10-b494-7d9a391f9639"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["8ba3e856-55a2-4a4d-bd57-77b891552fa4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="memory_optimization_strategy",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["92ff4a90-0abc-46b0-bb07-0c779ec50515"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="attention_slicing",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_slicing",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="vae_tiling",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="transformer_layerwise_casting",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["d380d95e-ea8e-4af6-a0f1-fd38fd973bae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="cpu_offload_strategy",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["fabf822c-c6ee-418a-9303-e61150b7d828"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quantization_mode",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["fabf822c-c6ee-418a-9303-e61150b7d828"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="loras",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["d177aa75-1e6a-4efa-ae3b-459ffd6c79d5"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="logs",
                    node_name=node7_name,
                    value=top_level_unique_values_dict["7d2ec1de-d322-4fe3-a6c6-6fd8db72c21c"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node8_name,
                    value=top_level_unique_values_dict["d3c4fd45-2d1f-4767-9016-27376fe5379a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node9_name,
                    value=top_level_unique_values_dict["9abe6dd2-2208-46a9-813a-ffff35d37be3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node10_name,
                    value=top_level_unique_values_dict["51679734-9078-4265-8e60-8deea3ed450b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="pipeline",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["ae9ee4df-4cda-4a65-896d-a5de6412adcb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node11_name,
                    value=top_level_unique_values_dict["052f29b7-0773-4003-90de-c3909e6c4c1d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node12_name,
                    value=top_level_unique_values_dict["819a0121-dff6-4e37-8b2b-b329197ed555"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node13_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="video",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["5fc76d36-d4fb-4958-809c-f3fdb8850c3d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="video",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["5fc76d36-d4fb-4958-809c-f3fdb8850c3d"],
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["f2a34255-b240-4b96-9565-865542b779d8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node13_name,
                    value=top_level_unique_values_dict["f2a34255-b240-4b96-9565-865542b779d8"],
                    initial_setup=True,
                    is_output=True,
                )
            )
