from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: E402
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline  # noqa: E402
from transformers import AutoModel, CLIPTextModel, Qwen3Config, Qwen3Model  # noqa: E402

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentSourceType  # noqa: E402
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import (  # noqa: E402
    TextEncoderComponentArtifact,
)
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import (  # noqa: E402
    get_loadable_component_class,
)


def test_single_file_key_diagnostic_returns_none_when_config_inspection_fails() -> None:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
        ModelComponentArtifact,
    )

    class _BrokenConfigComponent:
        @staticmethod
        def load_config(_config_path: str) -> dict[str, object]:
            msg = "invalid config"
            raise ValueError(msg)

    comparison = ModelComponentArtifact._compare_keys_with_model(  # noqa: SLF001
        component_cls=_BrokenConfigComponent,
        loadable_name="CLIPTextModel",
        config_path=Path("config.json"),
        checkpoint_keys=["weight"],
    )

    assert comparison is None


def test_any_single_file_failure_with_a_config_is_reported_as_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from diffusers import FluxTransformer2DModel
    from diffusers.loaders import single_file_utils

    from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
        ModelComponentArtifact,
    )
    from modular_diffusion_nodes_library.component_loading.config_resolver import ConfigLoadError

    config_dir = tmp_path / "transformer"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{}")
    monkeypatch.setattr(single_file_utils, "load_single_file_checkpoint", lambda _path: {})
    monkeypatch.setattr(single_file_utils, "infer_diffusers_model_type", lambda _checkpoint: "flux")

    def _from_single_file(cls: type, *_args: object, **_kwargs: object) -> None:
        msg = "Unexpected x_embedder shape: (1, 2) when loading model."
        raise ValueError(msg)

    monkeypatch.setattr(FluxTransformer2DModel, "from_single_file", classmethod(_from_single_file))
    artifact = ModelComponentArtifact(
        load_id="test",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        file_path=str(tmp_path / "flux.safetensors"),
        config_source=str(config_dir),
    )

    with pytest.raises(ConfigLoadError, match="Could not load the transformer"):
        artifact.materialize(pipeline_cls=FluxPipeline, slot="transformer")


def test_generic_pretrained_model_slot_resolves_to_auto_model() -> None:
    assert get_loadable_component_class(ZImagePipeline, "text_encoder") is AutoModel


def test_concrete_slot_keeps_its_annotated_class() -> None:
    assert get_loadable_component_class(FluxPipeline, "text_encoder") is CLIPTextModel


def test_z_image_text_encoder_loads_from_local_dir(tmp_path: Path) -> None:

    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    Qwen3Model(config).save_pretrained(tmp_path)
    artifact = TextEncoderComponentArtifact(
        load_id="z-image-text-encoder",
        source_type=ComponentSourceType.LOCAL_DIR,
        component="text_encoder",
        file_path=str(tmp_path),
        torch_dtype="float32",
    )

    text_encoder = artifact.materialize(pipeline_cls=ZImagePipeline)

    assert isinstance(text_encoder, Qwen3Model)


def test_single_file_config_comes_from_the_base_repo_before_the_default_repo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub
    from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS

    from modular_diffusion_nodes_library.component_loading.config_resolver import resolve_config_path

    default_repo = DIFFUSERS_DEFAULT_PIPELINE_PATHS["ltx2-dev"]["pretrained_model_name_or_path"]
    cached_repos = {"org/base", default_repo}

    def fake_cache(repo_id: str, filename: str, revision: str | None = None) -> str | None:  # noqa: ARG001
        if repo_id in cached_repos and filename == "transformer/config.json":
            return str(tmp_path / repo_id / filename)
        return None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_cache)

    def resolve() -> Path:
        return resolve_config_path("ltx2-dev", None, pipeline_slot="transformer", base_repo_id="org/base").path

    assert resolve() == tmp_path / "org/base/transformer"

    cached_repos.discard("org/base")
    assert resolve() == tmp_path / default_repo / "transformer"


def test_model_type_guess_is_limited_to_the_pipelines_own_types() -> None:
    from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
    from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

    from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
        ModelComponentArtifact,
    )

    class _Tensor:
        shape = (1, 1)

    # LTX-2 keys that diffusers guesses to be LTX-Video 0.9.7.
    checkpoint = {"patchify_proj.weight": _Tensor(), "transformer_blocks.47.scale_shift_table": _Tensor()}
    artifact = ModelComponentArtifact(load_id="x", source_type=ComponentSourceType.SINGLE_FILE, component="transformer")

    _, for_ltx2 = artifact._infer_single_file_model_type(pipeline_cls=LTX2Pipeline, checkpoint=checkpoint)  # noqa: SLF001
    _, for_ltx = artifact._infer_single_file_model_type(pipeline_cls=LTXPipeline, checkpoint=checkpoint)  # noqa: SLF001

    assert for_ltx2 == "ltx2-dev"
    assert for_ltx == "ltx-video-0.9.7"


def test_model_type_guess_names_a_known_variant_the_pipeline_does_not_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import diffusers.loaders.single_file_utils as single_file_utils

    from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
        ModelComponentArtifact,
    )

    artifact = ModelComponentArtifact(load_id="x", source_type=ComponentSourceType.SINGLE_FILE, component="transformer")

    def infer_with(model_type: str) -> str:
        monkeypatch.setattr(single_file_utils, "infer_diffusers_model_type", lambda _checkpoint: model_type)
        return artifact._infer_single_file_model_type(pipeline_cls=FluxPipeline, checkpoint={})[1]  # noqa: SLF001

    # Flux Fill is a Flux variant of its own: its config, not Flux dev's, describes the file.
    assert infer_with("flux-fill") == "flux-fill"
    # A guess no pipeline of this library registers says nothing about the file, so the pipeline's own type stands.
    assert infer_with("v1") == "flux-dev"
