"""Directory utilities for the advanced media library."""

import logging

from griptape_nodes.retained_mode.managers.os_manager import OSManager

from modular_diffusion_nodes_library.utils.config_utils import get_config_value, get_workspace_path

logger = logging.getLogger("modular_diffusers_nodes_library")


def cleanup_directory_if_enabled(directory_path: str) -> None:
    """Trim `directory_path` under the static files root, oldest first, when cleanup is enabled.

    Call this immediately before writing a new file into that directory, so the space it frees is
    still free when the write happens.

    Reads these configuration values:
    - modular_diffusion_library.enable_directory_cleanup: whether to clean up at all
    - modular_diffusion_library.max_directory_size_gb: the size to trim down to
    - static_files_directory: the root the directory sits under
    """
    if not get_config_value("modular_diffusion_library.enable_directory_cleanup"):
        return

    static_files_directory = get_config_value("static_files_directory", default="staticfiles")
    path = get_workspace_path() / static_files_directory / directory_path
    max_size_gb = get_config_value("modular_diffusion_library.max_directory_size_gb")
    # `OSManager.cleanup_directory_if_needed` is a static method: it touches only the filesystem,
    # which a worker shares, and holds none of the manager state the facade guard exists to protect.
    # Imported directly because reaching it through the facade raises in a worker, and no request
    # wraps it.
    OSManager.cleanup_directory_if_needed(full_directory_path=path, max_size_gb=max_size_gb)


def get_intermediates_directory_path() -> str:
    """Get the configured intermediates directory name for the advanced media library.

    This function retrieves the directory name where intermediate files (such as
    preview images during AI generation) are stored. The directory name is
    configured via the 'modular_diffusion_library.temp_folder_name' setting.

    Returns:
        str: The configured intermediates directory name, or "intermediates" if not configured.
            This is a directory name (not a full path) that will be used relative to
            the static files directory.

    Note:
        If the configuration value is not found, a warning is logged and the default
        "intermediates" directory name is returned.
    """
    # Get configured temp folder name, default to "intermediates"
    temp_folder_name = get_config_value("modular_diffusion_library.temp_folder_name")
    if temp_folder_name is None:
        logger.warning(
            "Configuration value 'modular_diffusion_library.temp_folder_name' not found, using default 'intermediates'"
        )
        temp_folder_name = "intermediates"
    return temp_folder_name
