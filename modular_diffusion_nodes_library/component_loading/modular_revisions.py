"""Pin a ModularPipeline's components to the revisions the user selected.

`ModularPipeline.from_pretrained(repo, revision=X)` reads `modular_model_index.json` at X, but each component then
loads at the revision in its index entry. Published indexes say `revision: None`, which means "whatever `refs/main`
points at in the HF cache": the wrong snapshot when `refs/main` points elsewhere, and a component silently left
`None` (`load_components` only logs) when there is no `refs/main`. Pass the result of `pinned_revisions` to
`load_components(revision=...)` so every component loads from the snapshot the user picked.

Kept free of heavy imports: it only calls methods on the pipeline it is given, so parameter modules the
orchestrator imports can import it at module scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def pinned_revisions(
    pipe: Any,
    revisions_by_repo: Mapping[str, str | None],
    forced_revisions: Mapping[str, str | None] | None = None,
) -> dict[str, str]:
    """Map each component to the selected revision of the repo its index entry loads from.

    `revisions_by_repo` holds the repos whose snapshot the build selected, normally the base repo. A component
    from one of them gets the selected revision only if its index entry has none: an entry pinned to a hash is
    the repo author's choice for that snapshot, so it is kept.

    `forced_revisions` holds repos a component is redirected to with a revision the user picked explicitly
    (Flux3's prompt upsampler). Those always win over the index entry.

    A component whose repo is in neither, or whose selected revision is None, keeps its own index entry.
    """
    forced = forced_revisions or {}
    pinned: dict[str, str] = {}
    for name in pipe.pretrained_component_names:
        spec = pipe.get_component_spec(name)
        repo = spec.pretrained_model_name_or_path
        forced_revision = forced.get(repo)
        if forced_revision is not None:
            pinned[name] = forced_revision
            continue
        revision = revisions_by_repo.get(repo)
        if revision is not None and spec.revision is None:
            pinned[name] = revision
    return pinned
