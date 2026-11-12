from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import cast

logger = logging.getLogger("modular_diffusers_nodes_library")


def configure_triton_compiler(namespace: str) -> None:
    """Point Triton's Windows compiler at the tcc bundled with the execution dependency."""
    if sys.platform != "win32":
        return

    try:
        import triton  # type: ignore[reportMissingImports]
        from triton.runtime.build import get_cc  # type: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError(
            f"{namespace}: Attempted to configure Triton's Windows compiler. Failed because "
            "the declared 'triton-windows' execution dependency is not importable."
        ) from error

    existing_compiler = os.environ.get("CC")
    if existing_compiler:
        logger.debug("Keeping existing CC compiler setting: %s", existing_compiler)
        return

    tcc_path = Path(cast(str, triton.__file__)).parent / "runtime" / "tcc" / "tcc.exe"
    if not tcc_path.is_file():
        raise RuntimeError(
            f"{namespace}: Attempted to configure Triton's Windows compiler. Failed because "
            f"the bundled tcc compiler was not found at '{tcc_path}'."
        )

    os.environ["CC"] = str(tcc_path)

    # Triton resolves the compiler once per process and caches it, so a build that ran before this
    # point would otherwise keep using whatever compiler it found.
    get_cc.cache_clear()


def configure_compile_cache_dirs() -> None:
    """Point Triton/Inductor's compile caches at a short path under the user's temp directory.

    The default location (``%TEMP%\\torchinductor_<user>\\triton\\...``) embeds the username twice,
    and a generated kernel path from it routinely exceeds Windows' 260-character MAX_PATH, failing
    with ``FileNotFoundError``. If the short directory cannot be created, the variable is left as it
    was and Triton uses its default location.
    """
    if sys.platform != "win32":
        return

    # Inductor writes its default into the environment the first time it runs, so that value is not
    # a user choice and still has to be replaced.
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    inductor_default_cache_dir = default_cache_dir()

    temp_dir = Path(tempfile.gettempdir())
    for env_var, dirname in (
        ("TRITON_CACHE_DIR", "gtn_triton"),
        ("TORCHINDUCTOR_CACHE_DIR", "gtn_inductor"),
    ):
        existing_dir = os.environ.get(env_var)
        if existing_dir and existing_dir != inductor_default_cache_dir:
            logger.debug("Keeping existing %s setting: %s", env_var, existing_dir)
            continue
        cache_dir = temp_dir / dirname
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            logger.warning(
                "Attempted to set %s. Failed to create '%s' because of %s; Triton will use its default cache location.",
                env_var,
                cache_dir,
                error,
            )
            continue
        os.environ[env_var] = str(cache_dir)
