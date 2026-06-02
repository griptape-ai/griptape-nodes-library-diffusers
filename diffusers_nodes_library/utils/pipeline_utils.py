import contextlib
import gc
import logging

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from diffusers_nodes_library.utils.torch_utils import (
    get_best_device,
    get_free_cuda_memory,
    get_max_memory_footprint,
    get_total_memory_footprint,
    should_enable_attention_slicing,
    to_human_readable_size,
)

logger = logging.getLogger("modular_diffusers_nodes_library")

# Best guess for memory optimization with 20% headroom
# https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator#caveats-with-this-calculator
MEMORY_HEADROOM_FACTOR = 1.2


def _remove_pipeline_hooks(pipe: DiffusionPipeline) -> None:
    """Detach diffusers/accelerate hooks that can keep pipeline modules strongly referenced."""
    if not hasattr(pipe, "remove_all_hooks"):
        return

    with contextlib.suppress(AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        pipe.remove_all_hooks()  # type: ignore[reportAttributeAccessIssue]

    # Offload-related bookkeeping can hold references through hook objects.
    for attr_name in ("_all_hooks", "final_offload_hook", "_offload_gpu_id", "_offload_device", "hf_device_map"):
        if hasattr(pipe, attr_name):
            setattr(pipe, attr_name, None)


def build_scheduler_with_overrides(
    scheduler_class: type[diffusers.SchedulerMixin],  # type: ignore[reportAttributeAccessIssue]
    base_repo_id: str,
    base_revision: str | None,
    config_overrides: dict | None,
    subfolder: str = "scheduler",
) -> diffusers.SchedulerMixin:  # type: ignore[reportAttributeAccessIssue]
    """Load the model's shipped scheduler config and merge user overrides on top."""
    base_scheduler = scheduler_class.from_pretrained(
        pretrained_model_name_or_path=base_repo_id,
        revision=base_revision,
        subfolder=subfolder,
        local_files_only=True,
    )
    merged_config = dict(base_scheduler.config)
    merged_config.update(config_overrides or {})
    return scheduler_class.from_config(merged_config)


def detect_offload_method(pipe: DiffusionPipeline) -> str | None:
    """Detect which CPU offload strategy is active on a pipeline.

    Returns 'sequential', 'model', or None.
    """
    for _name, component in pipe.components.items():
        if component is None or not hasattr(component, "_hf_hook"):
            continue
        hook_class_name = type(component._hf_hook).__name__
        if hook_class_name == "AlignDevicesHook":
            return "sequential"
        if hook_class_name == "CpuOffload":
            return "model"
    return None


def create_pipe_variant(
    source_pipe: DiffusionPipeline,
    target_class: type[DiffusionPipeline],
    torch_dtype: torch.dtype | None = None,
) -> DiffusionPipeline:
    """Create a pipeline variant via from_pipe, preserving offload strategy.

    Detects the offload method on source_pipe, creates the target pipeline
    via target_class.from_pipe(), and re-applies the same offload to it.
    The offload device is propagated from the source pipeline.
    """
    offload_method = detect_offload_method(source_pipe)
    offload_device = getattr(source_pipe, "_offload_device", None)

    new_pipe = target_class.from_pipe(source_pipe, torch_dtype=torch_dtype)

    offload_kwargs = {"device": offload_device} if offload_device is not None else {}
    if offload_method == "sequential":
        new_pipe.enable_sequential_cpu_offload(**offload_kwargs)
    elif offload_method == "model":
        new_pipe.enable_model_cpu_offload(**offload_kwargs)

    return new_pipe


def get_pipeline_component_names(pipe: DiffusionPipeline) -> list[str]:
    """Get component names dynamically from pipeline."""
    component_names = []

    for attr_name in dir(pipe):
        if not attr_name.startswith("_"):
            try:
                attr = getattr(pipe, attr_name)
                if hasattr(attr, "to") and callable(attr.to) and hasattr(attr, "parameters"):
                    component_names.append(attr_name)
            except Exception:
                logger.debug("Error accessing attribute %s of pipeline: %s", attr_name, pipe)
                continue

    if not component_names:
        logger.warning("Could not determine pipeline component names dynamically, using defaults.")
        component_names = ["vae", "text_encoder", "text_encoder_2", "transformer", "controlnet"]

    logger.debug("Detected pipeline components: %s", component_names)
    return component_names


def _check_cuda_memory_sufficient(
    pipe: DiffusionPipeline,
) -> bool:
    """Check if CUDA device has sufficient memory for the pipeline."""
    model_memory = MEMORY_HEADROOM_FACTOR * get_total_memory_footprint(pipe, get_pipeline_component_names(pipe))
    return model_memory <= get_free_cuda_memory()


def _check_mps_memory_sufficient(
    pipe: DiffusionPipeline,
) -> bool:
    """Check if MPS device has sufficient memory for the pipeline."""
    model_memory = get_total_memory_footprint(pipe, get_pipeline_component_names(pipe))
    recommended_max_memory = torch.mps.recommended_max_memory()
    free_memory = recommended_max_memory - torch.mps.current_allocated_memory()
    return model_memory <= free_memory


def _log_memory_info(
    pipe: DiffusionPipeline,
    device: torch.device,
) -> None:
    """Log memory information for the device."""
    model_memory = MEMORY_HEADROOM_FACTOR * get_total_memory_footprint(pipe, get_pipeline_component_names(pipe))

    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        free_memory = total_memory - torch.cuda.memory_allocated(device)
        logger.info("Total memory on %s: %s", device, to_human_readable_size(total_memory))
        logger.info("Free memory on %s: %s", device, to_human_readable_size(free_memory))
    elif device.type == "mps":
        recommended_max_memory = torch.mps.recommended_max_memory()
        free_memory = recommended_max_memory - torch.mps.current_allocated_memory()
        logger.info("Recommended max memory on %s: %s", device, to_human_readable_size(recommended_max_memory))
        logger.info("Free memory on %s: %s", device, to_human_readable_size(free_memory))

    logger.info("Require memory for diffusion pipeline: %s", to_human_readable_size(model_memory))


def _quantize_diffusion_pipeline(
    pipe: DiffusionPipeline,
    quantization_mode: str,
    device: torch.device,
) -> None:
    """Uses optimum.quanto to quantize the pipeline components."""
    from optimum.quanto import freeze, qfloat8, qint4, qint8, quantize  # type: ignore[reportMissingImports]

    logger.info("Applying quantization: %s", quantization_mode)
    _log_memory_info(pipe, device)
    quant_map = {"fp8": qfloat8, "int8": qint8, "int4": qint4}
    quant_type = quant_map[quantization_mode]

    component_names = get_pipeline_component_names(pipe)
    logger.debug("Quantizing components: %s", component_names)
    for name in component_names:
        component = getattr(pipe, name, None)
        if component is not None:
            logger.debug("Quantizing %s with %s", name, quantization_mode)
            quantize(component, weights=quant_type, exclude=["proj_out"])
            logger.debug("Freezing %s", name)
            freeze(component)
            logger.debug("Quantizing completed for %s.", name)
    logger.info("Quantization complete.")


def _automatic_optimize_diffusion_pipeline(  # noqa: C901 PLR0911 PLR0912 PLR0915
    pipe: DiffusionPipeline,
    device: torch.device,
    *,
    is_prequantized: bool = False,
    supports_layerwise_casting: bool = True,
    requires_device_map: bool = False,
) -> None:
    """Optimize pipeline memory footprint with incremental VRAM checking."""
    if device.type == "cuda":
        _log_memory_info(pipe, device)

        if hasattr(pipe, "enable_attention_slicing") and should_enable_attention_slicing(device):
            logger.info("Enabling attention slicing")
            pipe.enable_attention_slicing()

        if hasattr(pipe, "enable_vae_slicing"):
            logger.info("Enabling vae slicing")
            pipe.enable_vae_slicing()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "use_slicing"):
            logger.info("Enabling vae slicing")
            pipe.vae.enable_slicing()

        # Enable VAE tiling for low-VRAM decode
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            logger.info("Enabling VAE tiling")
            pipe.vae.enable_tiling()

        # Pipelines loaded with device_map already have device placement handled.
        # Skip .to(device) and CPU offload strategies as they're incompatible.
        if requires_device_map:
            logger.info("Pipeline loaded with device_map, skipping device placement optimizations")
            return

        if _check_cuda_memory_sufficient(pipe):
            logger.info("Sufficient memory. Moving pipeline to %s", device)
            pipe.to(device)
            return

        if is_prequantized:
            logger.info("Pipeline is pre-quantized; skipping fp8 layerwise casting for transformer")
        elif not supports_layerwise_casting:
            logger.info("Pipeline does not support layerwise casting; skipping fp8 layerwise casting for transformer")
        elif not hasattr(pipe, "transformer"):
            logger.info("Pipeline has no transformer (e.g. UNet-based); skipping layerwise casting")
        else:
            logger.warning("Insufficient memory. Enabling fp8 layerwise caching for transformer")
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            _log_memory_info(pipe, device)
            if _check_cuda_memory_sufficient(pipe):
                logger.info("Sufficient memory after fp8 optimization. Moving pipeline to %s", device)
                pipe.to(device)
                return

        logger.info("Insufficient memory. Trying model offloading techniques.")
        free_cuda_memory = torch.cuda.mem_get_info()[0]
        max_memory_footprint_with_headroom = MEMORY_HEADROOM_FACTOR * get_max_memory_footprint(
            pipe, get_pipeline_component_names(pipe)
        )
        logger.info("Free CUDA memory: %s", to_human_readable_size(free_cuda_memory))
        logger.info(
            "Pipeline estimated max memory footprint: %s",
            to_human_readable_size(max_memory_footprint_with_headroom),
        )
        if max_memory_footprint_with_headroom < free_cuda_memory and hasattr(pipe, "enable_model_cpu_offload"):
            logger.info("Enabling model cpu offload")
            pipe.enable_model_cpu_offload()
            _log_memory_info(pipe, device)
            if _check_cuda_memory_sufficient(pipe):
                logger.info("Sufficient memory after model cpu offload")
                return
        elif hasattr(pipe, "enable_sequential_cpu_offload"):
            logger.info("Enabling sequential cpu offload")
            pipe.enable_sequential_cpu_offload()
            _log_memory_info(pipe, device)
            if _check_cuda_memory_sufficient(pipe):
                logger.info("Sufficient memory after sequential cpu offload")
                return

        # Final check after all optimizations
        if not _check_cuda_memory_sufficient(pipe):
            logger.warning("Memory may still be insufficient after all optimizations, but will try anyway")

        # Intentionally not calling pipe.to(device) here because sequential_cpu_offload
        # manages device placement automatically

    elif device.type == "mps":
        _log_memory_info(pipe, device)

        # Pipelines loaded with device_map already have device placement handled.
        if requires_device_map:
            logger.info("Pipeline loaded with device_map, skipping device placement optimizations")
            return

        if _check_mps_memory_sufficient(pipe):
            logger.info("Sufficient memory on %s for Pipeline.", device)
            logger.info("Moving pipeline to %s", device)
            pipe.to(device)
            return

        logger.warning("Insufficient memory on %s for Pipeline.", device)
        if hasattr(pipe, "enable_vae_slicing"):
            logger.info("Enabling vae slicing")
            pipe.enable_vae_slicing()

        # Final check after VAE slicing
        if not _check_mps_memory_sufficient(pipe):
            logger.warning("Memory may still be insufficient after optimizations, but will try anyway")

        # Intentionally not calling pipe.to(device) here when memory is insufficient
        # to avoid potential OOM errors
    return


def _manual_optimize_diffusion_pipeline(  # noqa: C901 PLR0912 PLR0913
    pipe: DiffusionPipeline,
    device: torch.device,
    *,
    attention_slicing: bool,
    vae_slicing: bool,
    vae_tiling: bool,
    transformer_layerwise_casting: bool,
    cpu_offload_strategy: str,
    quantization_mode: str,
    is_prequantized: bool = False,
    supports_layerwise_casting: bool = True,
    requires_device_map: bool = False,
) -> None:
    if quantization_mode != "None":
        if is_prequantized:
            logger.info("Pipeline is pre-quantized; skipping quantization step")
        else:
            _quantize_diffusion_pipeline(pipe, quantization_mode, device)
    if attention_slicing and hasattr(pipe, "enable_attention_slicing"):
        logger.info("Enabling attention slicing")
        pipe.enable_attention_slicing()
    if vae_slicing:
        if hasattr(pipe, "enable_vae_slicing"):
            logger.info("Enabling vae slicing")
            pipe.enable_vae_slicing()
        elif hasattr(pipe, "vae") and hasattr(pipe.vae, "use_slicing"):
            logger.info("Enabling vae slicing")
            pipe.vae.enable_slicing()
        elif hasattr(pipe, "vae"):
            logger.debug("VAE does not support slicing (e.g., AutoencoderKLTemporalDecoder), skipping")
    if vae_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    if transformer_layerwise_casting and hasattr(pipe, "transformer"):
        if is_prequantized:
            logger.info("Pipeline is pre-quantized; skipping fp8 layerwise casting for transformer")
        elif not supports_layerwise_casting:
            logger.info("Pipeline does not support layerwise casting; skipping fp8 layerwise casting for transformer")
        else:
            logger.info("Enabling fp8 layerwise casting for transformer")
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )

    # Pipelines loaded with device_map already have device placement handled.
    # Skip .to(device) and CPU offload strategies as they're incompatible.
    if requires_device_map:
        logger.info("Pipeline loaded with device_map, skipping device placement optimizations")
        return

    if cpu_offload_strategy == "Sequential":
        if hasattr(pipe, "enable_sequential_cpu_offload"):
            logger.info("Enabling sequential cpu offload")
            pipe.enable_sequential_cpu_offload()
        else:
            logger.warning("Pipeline does not support sequential cpu offload")
    elif cpu_offload_strategy == "Model":
        if hasattr(pipe, "enable_model_cpu_offload"):
            logger.info("Enabling model cpu offload")
            pipe.enable_model_cpu_offload()
        else:
            logger.warning("Pipeline does not support model cpu offload")
    elif cpu_offload_strategy == "None":
        pipe.to(device)


def optimize_diffusion_pipeline(  # noqa: PLR0913
    pipe: DiffusionPipeline,
    *,
    memory_optimization_strategy: str = "Manual",
    attention_slicing: bool = False,
    vae_slicing: bool = False,
    vae_tiling: bool = False,
    transformer_layerwise_casting: bool = False,
    cpu_offload_strategy: str = "None",
    quantization_mode: str = "None",
    is_prequantized: bool = False,
    supports_layerwise_casting: bool = True,
    requires_device_map: bool = False,
) -> None:
    """Optimize pipeline performance and memory."""
    device = get_best_device()

    if memory_optimization_strategy == "Automatic":
        _automatic_optimize_diffusion_pipeline(
            pipe,
            device,
            is_prequantized=is_prequantized,
            supports_layerwise_casting=supports_layerwise_casting,
            requires_device_map=requires_device_map,
        )
    else:
        _manual_optimize_diffusion_pipeline(
            pipe=pipe,
            device=device,
            attention_slicing=attention_slicing,
            vae_slicing=vae_slicing,
            vae_tiling=vae_tiling,
            transformer_layerwise_casting=transformer_layerwise_casting,
            cpu_offload_strategy=cpu_offload_strategy,
            quantization_mode=quantization_mode,
            is_prequantized=is_prequantized,
            supports_layerwise_casting=supports_layerwise_casting,
            requires_device_map=requires_device_map,
        )

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cuda, "sdp_kernel"):
            torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            )
    except Exception:
        logger.debug("sdp_kernel not supported, continuing without")


def clear_diffusion_pipeline(
    pipe: DiffusionPipeline,
) -> None:
    """Clear pipeline from memory."""
    _remove_pipeline_hooks(pipe)

    for component_name in get_pipeline_component_names(pipe):
        if hasattr(pipe, component_name):
            component = getattr(pipe, component_name)
            if component is not None:
                with contextlib.suppress(NotImplementedError):
                    component.to("cpu")
                del component
                setattr(pipe, component_name, None)

    del pipe
    cleanup_memory_caches()


def cleanup_memory_caches() -> None:
    """Clear memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
