"""The release line pins a pack's two hashes outside the pack, and the loader holds to it.

`version_stamps.PACK_RELEASE` names the pack version and both hashes the running
extraction version was released with. The loader refuses a pack of that version whose
hashes differ, so a pack cannot change under a version string, and it leaves a pack of
any other version alone.
"""

from __future__ import annotations

import dataclasses

import pytest

from backend.knowledge import version_stamps
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import Pack, PackLoadError, build_pack
from backend.knowledge.version_stamps import (
    EXTRACTION_VERSION,
    ONTOLOGY_VERSION,
    PACK_RELEASE,
    PackRelease,
)
from tests.unit.knowledge.ontologies.fixtures.base_pack import conventions, manifest, plugins, types


def _build(**overrides) -> Pack:
    parts = {
        "manifest": manifest.MANIFEST,
        "node_types": types.NODE_TYPES,
        "edge_types": types.EDGE_TYPES,
        "extractable_node_types": types.EXTRACTABLE_NODE_TYPES,
        "extractable_edge_types": types.EXTRACTABLE_EDGE_TYPES,
        "rules": conventions.RULES,
        "plugins": plugins.PLUGINS,
    }
    parts.update(overrides)
    return build_pack(**parts)


def _release_for(pack: Pack, **changes) -> PackRelease:
    """A release line that names `pack` as it is, then `changes`."""
    line = PackRelease(
        extraction_version=EXTRACTION_VERSION,
        pack_version=pack.version,
        pack_prompt_hash=pack.pack_prompt_hash,
        pack_rails_hash=pack.pack_rails_hash,
    )
    return dataclasses.replace(line, **changes)


@pytest.fixture
def fresh_registry(monkeypatch):
    """The registry with no pack built, restored to that state afterwards."""
    monkeypatch.delenv(registry.ENV_VAR, raising=False)
    registry.get_pack.cache_clear()
    yield registry
    registry.get_pack.cache_clear()


# --- the line ---------------------------------------------------------------------------


def test_the_release_line_is_frozen_with_slots():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PACK_RELEASE.pack_rails_hash = "edited"  # type: ignore[misc]
    assert not hasattr(PACK_RELEASE, "__dict__")


def test_the_release_line_is_for_the_running_extraction_version_and_ontology():
    assert PACK_RELEASE.extraction_version == EXTRACTION_VERSION
    assert PACK_RELEASE.pack_version == ONTOLOGY_VERSION


def test_the_release_line_holds_two_sha256_hashes():
    for value in (PACK_RELEASE.pack_prompt_hash, PACK_RELEASE.pack_rails_hash):
        assert len(value) == 64 and all(char in "0123456789abcdef" for char in value)
    assert PACK_RELEASE.pack_prompt_hash != PACK_RELEASE.pack_rails_hash


def test_the_pack_in_use_by_default_is_the_pack_the_release_line_names(fresh_registry):
    pack = fresh_registry.active_pack()

    assert (pack.version, pack.pack_prompt_hash, pack.pack_rails_hash) == (
        PACK_RELEASE.pack_version,
        PACK_RELEASE.pack_prompt_hash,
        PACK_RELEASE.pack_rails_hash,
    )


# --- the loader holds to it -------------------------------------------------------------


def test_a_pack_of_the_released_version_with_other_hashes_is_refused():
    found = _build()
    released_version = dataclasses.replace(manifest.MANIFEST, version=PACK_RELEASE.pack_version)

    with pytest.raises(PackLoadError) as refusal:
        _build(manifest=released_version)

    message = str(refusal.value)
    assert f"pack {PACK_RELEASE.pack_version}" in message
    assert f"extraction version {EXTRACTION_VERSION}" in message
    assert (
        f"pack_prompt_hash expected {PACK_RELEASE.pack_prompt_hash}, "
        f"found {found.pack_prompt_hash}"
    ) in message
    assert (
        f"pack_rails_hash expected {PACK_RELEASE.pack_rails_hash}, found {found.pack_rails_hash}"
    ) in message


@pytest.mark.parametrize("moved", ["pack_prompt_hash", "pack_rails_hash"])
def test_either_hash_differing_alone_is_a_refusal(monkeypatch, moved):
    pack = _build()
    monkeypatch.setattr(version_stamps, "PACK_RELEASE", _release_for(pack, **{moved: "0" * 64}))

    with pytest.raises(PackLoadError, match=f"{moved} expected {'0' * 64}, found"):
        _build()


def test_a_pack_whose_hashes_are_the_lines_loads(monkeypatch):
    pack = _build()
    monkeypatch.setattr(version_stamps, "PACK_RELEASE", _release_for(pack))

    again = _build()

    assert (again.pack_prompt_hash, again.pack_rails_hash) == (
        pack.pack_prompt_hash,
        pack.pack_rails_hash,
    )


def test_a_pack_of_a_version_the_line_does_not_name_loads_whatever_its_hashes():
    assert manifest.MANIFEST.version != PACK_RELEASE.pack_version

    changed = _build(rules=conventions.RULES[:-1])

    assert changed.pack_prompt_hash != _build().pack_prompt_hash
