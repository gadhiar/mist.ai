"""The pack registry: which ontology packs exist, and which one is active.

`active_pack()` is what the normalizer, the validator and the scorer's id rails run
with when they are given no pack. The active version is `MIST_ONTOLOGY_VERSION` when it
is set, and otherwise the newest pack that is not deprecated.

A pack ships as a package of files under `backend/knowledge/ontologies/packs/`, and the
loader (`backend.knowledge.ontologies.pack.load_pack`) builds it from them. A later
version adds a pack package and a line in `_BUILDERS`, and no engine edit.

The registry is a directory of packs and nothing more. A pack is built once, and what
the loader refuses (`PackLoadError`) is refused here. A pack's pinned cases are not run
here: the rail runner runs them the first time it is handed the pack, and the normalizer
when it is built. So this module imports nothing from the runner, and reading a pack
here does not load it: the extraction service must never import the runner.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import cache, partial

from backend.knowledge.ontologies.pack import Pack, PackLoadError, load_pack
from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0

ENV_VAR = "MIST_ONTOLOGY_VERSION"

_PACKS = "backend.knowledge.ontologies.packs"

# Each version the registry knows, and what builds its pack.
_BUILDERS: dict[str, Callable[[], Pack]] = {
    ONTOLOGY_V1_0_0.version: partial(load_pack, f"{_PACKS}.v1_4_0"),
}


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) if part.isdigit() else 0 for part in version.split("."))


def available_versions() -> tuple[str, ...]:
    """Every version the registry can build, oldest first."""
    return tuple(sorted(_BUILDERS, key=_version_key))


@cache
def get_pack(version: str) -> Pack:
    """The pack of one ontology version, built once.

    The pack's pinned cases are not run here; the rail runner and the normalizer run
    them.

    Args:
        version: A version `available_versions` lists.

    Raises:
        PackLoadError: The version is unknown, or the loader refuses the pack.
    """
    builder = _BUILDERS.get(version)
    if builder is None:
        known = ", ".join(available_versions())
        raise PackLoadError(f"no ontology pack for version {version} (known: {known})")
    return builder()


def active_pack() -> Pack:
    """The pack the engine runs with: `MIST_ONTOLOGY_VERSION`, or the newest in use.

    Raises:
        PackLoadError: The named version is unknown, every pack is deprecated, or the
            pack is refused at load.
    """
    named = os.environ.get(ENV_VAR, "").strip()
    if named:
        return get_pack(named)
    for version in reversed(available_versions()):
        pack = get_pack(version)
        if not pack.manifest.deprecated:
            return pack
    raise PackLoadError("no ontology pack is in use: every registered pack is deprecated")
