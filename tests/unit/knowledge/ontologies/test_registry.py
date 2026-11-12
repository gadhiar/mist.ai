"""The pack registry, and the v1.4.0 pack it builds from the pack's files."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from backend.knowledge.extraction import id_rails
from backend.knowledge.extraction.pre_llm import plan_vocabulary
from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS
from backend.knowledge.ontologies import (
    ALL_NODE_TYPE_NAMES,
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
    registry,
)
from backend.knowledge.ontologies.pack import PackLoadError, build_pack, load_pack
from backend.knowledge.ontologies.packs.v1_4_0 import conventions as pack_conventions
from backend.knowledge.ontologies.packs.v1_4_0 import manifest as pack_manifest
from backend.knowledge.ontologies.packs.v1_4_0 import plugins as pack_plugins

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")


@pytest.fixture
def fresh_registry(monkeypatch):
    """The registry with no pack built, restored to that state afterwards."""
    monkeypatch.delenv(registry.ENV_VAR, raising=False)
    registry.get_pack.cache_clear()
    yield registry
    registry.get_pack.cache_clear()


def _rebuilt(**manifest_changes):
    """The base fixture pack again, with a changed manifest."""
    return build_pack(
        manifest=dataclasses.replace(BASE.manifest, **manifest_changes),
        node_types=BASE.node_types,
        edge_types=BASE.edge_types,
        extractable_node_types=BASE.extractable_node_types,
        extractable_edge_types=BASE.extractable_edge_types,
        rules=BASE.rules,
        plugins=BASE.plugins,
    )


# --- which pack is active -------------------------------------------------------------


def test_the_active_pack_is_todays_ontology_under_its_own_version(fresh_registry):
    pack = fresh_registry.active_pack()

    assert pack.version == ONTOLOGY_V1_0_0.version
    assert fresh_registry.available_versions() == (ONTOLOGY_V1_0_0.version,)


def test_the_registry_reads_the_version_and_writes_no_literal():
    source = Path(registry.__file__).read_text(encoding="utf-8")

    assert ONTOLOGY_V1_0_0.version not in source
    assert "ONTOLOGY_V1_0_0.version: partial(load_pack," in source


def test_the_registry_builds_each_pack_under_the_version_its_manifest_states(fresh_registry):
    for version in fresh_registry.available_versions():
        assert fresh_registry.get_pack(version).version == version


def test_the_registry_builds_todays_pack_from_the_packs_files(fresh_registry):
    pack = fresh_registry.get_pack(ONTOLOGY_V1_0_0.version)

    assert pack.manifest is pack_manifest.MANIFEST
    assert pack.rules == pack_conventions.RULES
    assert pack.plugins == pack_plugins.PLUGINS
    assert pack.labelling == (
        Path(pack_manifest.__file__).with_name("labelling.md").read_text(encoding="utf-8")
    )


def test_the_registry_holds_no_pack_of_its_own_making():
    # The adapter, which stated the ontology as pack rules from the modules that held
    # it, is gone: the registry builds nothing and reads no rule, type or id rail.
    source = Path(registry.__file__).read_text(encoding="utf-8")

    assert "_adapter" not in source
    assert "build_pack" not in source
    assert "id_rails" not in source


def test_a_pack_is_built_once(fresh_registry):
    first = fresh_registry.active_pack()

    assert fresh_registry.active_pack() is first
    assert fresh_registry.get_pack(ONTOLOGY_V1_0_0.version) is first


def test_the_environment_names_the_active_version(fresh_registry, monkeypatch):
    monkeypatch.setitem(fresh_registry._BUILDERS, "0.1.0", lambda: BASE)
    monkeypatch.setenv(fresh_registry.ENV_VAR, "0.1.0")

    assert fresh_registry.active_pack() is BASE


def test_an_unknown_version_is_refused(fresh_registry, monkeypatch):
    monkeypatch.setenv(fresh_registry.ENV_VAR, "9.9.9")

    with pytest.raises(PackLoadError, match="no ontology pack for version 9.9.9"):
        fresh_registry.active_pack()


def test_the_default_is_the_newest_pack_that_is_not_deprecated(fresh_registry, monkeypatch):
    newest = _rebuilt(version="99.0.0", deprecated=True)
    newer = _rebuilt(version="98.2.0")
    monkeypatch.setitem(fresh_registry._BUILDERS, "99.0.0", lambda: newest)
    monkeypatch.setitem(fresh_registry._BUILDERS, "98.2.0", lambda: newer)
    monkeypatch.setitem(fresh_registry._BUILDERS, "98.10.0", lambda: BASE)

    assert fresh_registry.available_versions()[-3:] == ("98.2.0", "98.10.0", "99.0.0")
    assert fresh_registry.active_pack() is BASE


def test_the_registry_hands_out_a_pack_without_running_its_pinned_cases(
    fresh_registry, monkeypatch
):
    # The registry is a directory of packs. The rail runner refuses a pack whose pinned
    # case does not hold (`test_rails.py`), and so does the normalizer when it is built
    # (`test_normalizer_pack.py`); the extraction service runs neither.
    rule = next(rule for rule in BASE.rules if rule.id == "mooring-number")
    wrong = dataclasses.replace(rule.pinned_case, expected_entities=())
    rules = tuple(
        dataclasses.replace(r, pinned_case=wrong) if r.id == rule.id else r for r in BASE.rules
    )
    unheld = dataclasses.replace(BASE, rules=rules)
    monkeypatch.setitem(fresh_registry._BUILDERS, "0.1.0", lambda: unheld)

    assert fresh_registry.get_pack("0.1.0") is unheld


def test_the_registry_imports_nothing_from_the_rail_runner():
    source = Path(registry.__file__).read_text(encoding="utf-8")

    assert "extraction.rails" not in source
    assert "import rails" not in source


# --- the pack is today's ontology ---------------------------------------------------------


# The v1.4.0 ontology's names, each list hashed in its order: what the ontology's
# modules listed at 058e52c, before the pack held them.
_NODE_NAMES = "f4a5929ad6c4ca36e5d034a4c3c2fa8c3e6e8a8616ff8bb88bc3947173ec3e27"
_EDGE_NAMES = "d4f9197c14bc43fb75ba81950261b17a139c6b2b8cab67c406d86dd60f2459dd"
_EXTRACTABLE_NODE_NAMES = "a225b428fb92ba42a6b46f186209a876dd3a1fb8aaa1b9b83605c55bee93d6a9"
_EXTRACTABLE_EDGE_NAMES = "ad545e1ef9fec6f20a75071bcc5ee446be76d0c744b1cbc345893aec718fdc6a"


def _names_digest(names) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _pairs_digest(pairs) -> str:
    """The pairs of an alias vocabulary, hashed in their order."""
    text = json.dumps([list(pair) for pair in pairs], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_pack_holds_todays_types_and_edges():
    pack = registry.active_pack()

    assert _names_digest(node.type_name for node in pack.node_types) == _NODE_NAMES
    assert _names_digest(edge.type_name for edge in pack.edge_types) == _EDGE_NAMES
    assert _names_digest(pack.extractable_node_types) == _EXTRACTABLE_NODE_NAMES
    assert _names_digest(pack.extractable_edge_types) == _EXTRACTABLE_EDGE_NAMES


def test_the_packs_pair_table_is_the_one_the_validator_derived_before():
    all_types = set(ALL_NODE_TYPE_NAMES)
    before = {}
    for name in EXTRACTABLE_RELATIONSHIP_TYPES:
        sources = set(EDGE_TYPES_BY_NAME[name].allowed_source_types)
        targets = set(EDGE_TYPES_BY_NAME[name].allowed_target_types)
        before[name] = (
            None if sources >= all_types else sources,
            None if targets >= all_types else targets,
        )

    assert before == RELATIONSHIP_CONSTRAINTS
    assert list(RELATIONSHIP_CONSTRAINTS) == EXTRACTABLE_RELATIONSHIP_TYPES
    assert all(
        isinstance(end, set) for ends in RELATIONSHIP_CONSTRAINTS.values() for end in ends if end
    )
    assert registry.active_pack().pair_table.keys() == before.keys()


def test_the_packs_manifest_is_what_the_hierarchy_holds_and_the_id_rails_held():
    manifest = registry.active_pack().manifest

    assert {r.name: r.replacement for r in manifest.retired_types} == {
        "Topic": "Concept",
        "Milestone": "Event",
    }
    assert dict(manifest.supertypes) == {"Abstraction": "Concept"}
    assert (manifest.fallback_type, manifest.user_id) == ("Abstraction", "user")
    defaults = {r.name: r.property_defaults for r in manifest.retired_types}
    assert defaults == {"Topic": (), "Milestone": (("event_type", "milestone"),)}
    assert sum(1 for value in defaults.values() if value) == 1


def test_the_packs_rules_are_the_vocabularies_the_pack_states():
    pack = registry.active_pack()
    reserved = {
        name: (rule.params.id, rule.params.type)
        for rule in pack.rules_of("reserved_identity")
        for name in rule.params.names
    }
    by_match = {rule.params.match: rule.params for rule in pack.rules_of("alias_vocabulary")}

    assert reserved == {
        name: ("mist-identity", "MistIdentity")
        for name in (
            "mist",
            "mist.ai",
            "mist ai",
            "mist-ai",
            "the ai",
            "the-ai",
            "the assistant",
            "the-assistant",
        )
    }
    assert len(by_match["id"].aliases) == 32
    assert _pairs_digest(by_match["id"].aliases) == (
        "3492a6a04cde8509d0e5f7c8f975371c145ad05a1469a9033215bb4a180122e9"
    )
    assert by_match["name"].aliases == (("c++", "c-plus-plus"), ("c#", "c-sharp"))
    assert by_match["id"].types == by_match["name"].types == ()
    assert pack.rules_of("id_exempt")[0].params.types == (
        "MistIdentity",
        "User",
        "Date",
        "Metric",
    )


def test_the_packs_plan_vocabulary_is_todays_extractable_lists():
    words = plan_vocabulary(registry.active_pack())

    assert _names_digest(words.entity_types) == _EXTRACTABLE_NODE_NAMES
    assert _names_digest(words.predicates) == _EXTRACTABLE_EDGE_NAMES
    assert _names_digest(words.fan_out_predicates) == _EXTRACTABLE_EDGE_NAMES


def test_the_packs_only_relation_rails_are_the_team_rules():
    # ADR-054's v14 amendment, register rows 1 to 3: the one convention a relation rail
    # of this pack carries is the named team's.
    assert [rule.id for rule in registry.active_pack().rules_in("relation")] == [
        "team-fan-out",
        "team-part-of",
    ]


def test_the_pack_exposes_both_hashes_and_they_are_stable(fresh_registry):
    first = fresh_registry.active_pack()
    fresh_registry.get_pack.cache_clear()
    second = fresh_registry.active_pack()

    assert second is not first
    assert (first.pack_prompt_hash, first.pack_rails_hash) == (
        second.pack_prompt_hash,
        second.pack_rails_hash,
    )
    assert len(first.pack_prompt_hash) == len(first.pack_rails_hash) == 64
    assert first.pack_prompt_hash != first.pack_rails_hash


def test_the_pack_hashes_to_the_pinned_values(fresh_registry):
    # What today's ontology hashes to. The rails hash covers the source of the module
    # that holds the pack's plug-ins, so an edit there or to the ontology moves a value
    # on purpose; a loader change must not.
    pack = fresh_registry.active_pack()

    assert (pack.pack_prompt_hash, pack.pack_rails_hash) == (
        "944c7fca0f474d70627a98836ba79d5fdf895dacda34550a3fcb732f0b1b078d",
        "1c08415cf78642fc15a69d1badf3e0ebe1ef1fb69edb867094431f99d6be3f28",
    )


def test_the_id_rails_run_with_the_active_pack(fresh_registry, monkeypatch):
    monkeypatch.setitem(fresh_registry._BUILDERS, "0.1.0", lambda: BASE)
    monkeypatch.setenv(fresh_registry.ENV_VAR, "0.1.0")
    entity = {"id": "the-north-mooring", "name": "North Mooring No. 7", "type": "Mooring"}

    id_rails.normalize_entity(entity)

    assert entity["id"] == "mooring-7"
    assert id_rails.stored_entity_id("the-flotsam", "Driftwood") == "flotsam"
    assert id_rails.reserved_identity({"id": "hm", "name": "The Harbourmaster"}) == (
        "harbourmaster",
        "Deckhand",
    )
