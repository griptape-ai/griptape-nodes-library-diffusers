"""MiniMax-H3 loads every component at the revision selected in the builder, not at refs/main."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="Needs the execution environment; run `make test/exec`.")
diffusers = pytest.importorskip("diffusers", reason="Needs the execution environment; run `make test/exec`.")

from modular_diffusion_nodes_library.standard_parameters import minimax_h3_parameters  # noqa: E402
from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import (  # noqa: E402
    MiniMaxH3PipelineParameters,
)

REPO = "MiniMaxAI/MiniMax-H3"
COMPONENTS = [
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
    "transformer",
]


def _pipe() -> MagicMock:
    pipe = MagicMock()
    pipe.pretrained_component_names = COMPONENTS
    pipe.get_component_spec.side_effect = lambda name: SimpleNamespace(
        pretrained_model_name_or_path=REPO, revision=None
    )
    return pipe


def _run_build(build: str, revision: str | None) -> MagicMock:
    pipe = _pipe()
    with (
        patch.object(diffusers, "ComponentsManager", return_value=MagicMock()),
        patch.object(diffusers.ModularPipeline, "from_pretrained", return_value=pipe),
        patch.object(minimax_h3_parameters, "get_best_device", return_value=torch.device("cpu")),
    ):
        build_data = {"repo_id": REPO, "revision": revision}
        if build == "_build_pipeline_from_repo":
            MiniMaxH3PipelineParameters._build_pipeline_from_repo(build_data, {})
        else:
            MiniMaxH3PipelineParameters.build_pipeline_from_build_data(build_data)
    return pipe


BUILDS = ["_build_pipeline_from_repo", "build_pipeline_from_build_data"]


@pytest.mark.parametrize("build", BUILDS)
def test_components_load_at_the_selected_revision(build: str) -> None:
    pipe = _run_build(build, "commit-hash")

    pipe.load_components.assert_called_once()
    assert pipe.load_components.call_args.kwargs["revision"] == dict.fromkeys(COMPONENTS, "commit-hash")


@pytest.mark.parametrize("build", BUILDS)
def test_no_revision_passes_no_revision_override(build: str) -> None:
    pipe = _run_build(build, None)

    assert "revision" not in pipe.load_components.call_args.kwargs
