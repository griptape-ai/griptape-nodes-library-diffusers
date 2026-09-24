"""Directory cleanup is decided in one place.

The policy is four coupled reads -- is cleanup enabled, what is the static files root, what is the
size ceiling, which directory -- and it was written out twice, in `directory_utils` and in
`pillow_utils`. The copies had already drifted: one defaulted `static_files_directory` to
"staticfiles" and the other did not, so a missing key sent the second down a path with `None` in it.

The duplication also cost real work. `publish_output_image_preview_latents` cleaned the intermediates
directory, then passed that same directory to `pil_to_image_artifact`, which cleaned it again.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "modular_diffusion_nodes_library"

#: The engine call that performs the trim. Every cleanup decision has to funnel into one caller.
CLEANUP_CALL = "cleanup_directory_if_needed"

#: Config keys that only mean something together. A second reader of these is a second policy.
POLICY_KEYS = (
    "modular_diffusion_library.enable_directory_cleanup",
    "modular_diffusion_library.max_directory_size_gb",
)


def _call_sites(attribute: str) -> list[str]:
    sites = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == attribute:
                sites.append(f"{path.relative_to(PACKAGE.parent)}:{node.lineno}")
    return sites


def _files_reading(key: str) -> set[str]:
    found = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        if f'"{key}"' in path.read_text():
            found.add(str(path.relative_to(PACKAGE.parent)))
    return found


def test_only_one_place_trims_a_directory() -> None:
    sites = _call_sites(CLEANUP_CALL)

    assert len(sites) == 1, (
        f"`{CLEANUP_CALL}` is called from {len(sites)} places, so the cleanup policy exists in "
        f"{len(sites)} copies and they will drift:\n  " + "\n  ".join(sites)
    )


def test_only_one_module_decides_whether_to_clean() -> None:
    for key in POLICY_KEYS:
        readers = _files_reading(key)
        assert len(readers) == 1, (
            f"'{key}' is read in {len(readers)} modules. It is only meaningful alongside the other "
            f"cleanup settings, so a second reader is a second policy:\n  " + "\n  ".join(sorted(readers))
        )
