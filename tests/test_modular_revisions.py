"""pinned_revisions maps each component to the selected revision of the repo its index entry loads from."""

from types import SimpleNamespace

from modular_diffusion_nodes_library.component_loading.modular_revisions import pinned_revisions

BASE = "org/base"
OTHER = "org/upsampler"
INDEX_HASH = "1234567890123456789012345678901234567890"


class _FakePipe:
    """Components keyed to (repo, revision in the index entry)."""

    def __init__(self, entries: dict[str, tuple[str, str | None]]) -> None:
        self._entries = entries
        self.pretrained_component_names = list(entries)

    def get_component_spec(self, name: str) -> SimpleNamespace:
        repo, revision = self._entries[name]
        return SimpleNamespace(pretrained_model_name_or_path=repo, revision=revision)


def test_base_repo_components_get_the_selected_revision() -> None:
    pipe = _FakePipe({"transformer": (BASE, None), "vae": (BASE, None)})

    assert pinned_revisions(pipe, {BASE: "abc"}) == {"transformer": "abc", "vae": "abc"}


def test_an_index_entry_pinned_to_a_hash_is_kept() -> None:
    # The repo author chose that commit for this component; the selected snapshot doesn't override it.
    pipe = _FakePipe({"transformer": (BASE, None), "vae": (BASE, INDEX_HASH)})

    assert pinned_revisions(pipe, {BASE: "abc"}) == {"transformer": "abc"}


def test_a_forced_revision_wins_over_the_index_entry() -> None:
    # Flux3: the user picked the prompt upsampler's revision explicitly.
    pipe = _FakePipe({"transformer": (BASE, None), "prompt_upsampler": (OTHER, INDEX_HASH)})

    pinned = pinned_revisions(pipe, {BASE: "abc"}, forced_revisions={OTHER: "def"})

    assert pinned == {"transformer": "abc", "prompt_upsampler": "def"}


def test_components_from_an_unselected_repo_keep_their_own_entry() -> None:
    pipe = _FakePipe({"transformer": (BASE, None), "text_encoder": ("someone/encoder", None)})

    assert pinned_revisions(pipe, {BASE: "abc"}) == {"transformer": "abc"}


def test_a_none_revision_pins_nothing() -> None:
    pipe = _FakePipe({"transformer": (BASE, None)})

    assert pinned_revisions(pipe, {BASE: None}, forced_revisions={OTHER: None}) == {}
