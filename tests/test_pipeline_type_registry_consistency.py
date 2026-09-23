"""Keep `MODEL_TYPE_TO_PIPELINE_TYPE` in sync with the diffusers version this library pins.

`model_type` strings are produced by `diffusers.loaders.single_file_utils.infer_diffusers_model_type`,
so a key that diffusers renames or drops upstream silently stops matching any checkpoint. This test
is the guard on that drift; it lives here rather than as an import-time check in the registry because
importing `diffusers` to validate a table of strings would put diffusers in the edit-time
environment, and the registry is reached whenever the orchestrator imports a node module.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "diffusers",
    reason="Reads real diffusers classes; run `make test/exec` for the execution environment.",
)

from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import MODEL_TYPE_TO_PIPELINE_TYPE


def test_every_model_type_is_known_to_diffusers() -> None:
    unknown = sorted(set(MODEL_TYPE_TO_PIPELINE_TYPE) - set(DIFFUSERS_DEFAULT_PIPELINE_PATHS))
    assert not unknown, (
        f"These model_type keys are not present in "
        f"diffusers.loaders.single_file_utils.DIFFUSERS_DEFAULT_PIPELINE_PATHS: {unknown}. "
        f"Either diffusers renamed or removed them upstream, or they are typos."
    )
