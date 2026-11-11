"""The pack registry, and the adapter pack that states today's ontology as pack rules."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from backend.knowledge.extraction import id_rails
from backend.knowledge.extraction.pre_llm import PlanWords, plan_vocabulary
from backend.knowledge.extraction.validator import RELATIONSHIP_CONSTRAINTS
from backend.knowledge.ontologies import (
    ALL_EDGE_TYPES,
    ALL_NODE_TYPE_NAMES,
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
    registry,
)
from backend.knowledge.ontologies.hierarchy import ANCHOR_CHILD, RETIRED_TYPE_MAP
from backend.knowledge.ontologies.pack import PackLoadError, build_pack, load_pack

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
    assert "version=ONTOLOGY_V1_0_0.version" in source


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


def test_a_pack_whose_pinned_case_does_not_hold_is_refused(fresh_registry, monkeypatch):
    rule = next(rule for rule in BASE.rules if rule.id == "mooring-number")
    wrong = dataclasses.replace(rule.pinned_case, expected_entities=())
    rules = tuple(
        dataclasses.replace(r, pinned_case=wrong) if r.id == rule.id else r for r in BASE.rules
    )
    monkeypatch.setitem(
        fresh_registry._BUILDERS, "0.1.0", lambda: dataclasses.replace(BASE, rules=rules)
    )

    with pytest.raises(PackLoadError, match="pinned case.*do not hold: mooring-number"):
        fresh_registry.get_pack("0.1.0")


# --- the adapter pack is today's ontology -------------------------------------------------


def test_the_adapter_holds_todays_types_and_edges():
    pack = registry.active_pack()

    assert [node.type_name for node in pack.node_types] == ALL_NODE_TYPE_NAMES
    assert pack.edge_types == tuple(ALL_EDGE_TYPES)
    assert list(pack.extractable_node_types) == EXTRACTABLE_NODE_TYPES
    assert list(pack.extractable_edge_types) == EXTRACTABLE_RELATIONSHIP_TYPES


def test_the_adapters_pair_table_is_the_one_the_validator_derived_before():
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


def test_the_adapters_manifest_is_read_from_the_hierarchy_and_the_id_rails():
    manifest = registry.active_pack().manifest

    assert {r.name: r.replacement for r in manifest.retired_types} == RETIRED_TYPE_MAP
    assert dict(manifest.supertypes) == ANCHOR_CHILD
    assert manifest.fallback_type == id_rails.FALLBACK_TYPE
    assert manifest.user_id == id_rails.USER_ID
    defaults = {r.name: r.property_defaults for r in manifest.retired_types}
    assert defaults == id_rails.RETIRED_PROPERTY_DEFAULTS
    assert sum(1 for value in defaults.values() if value) == 1


def test_the_adapters_rules_are_the_id_rails_constants():
    pack = registry.active_pack()
    reserved = {
        name: (rule.params.id, rule.params.type)
        for rule in pack.rules_of("reserved_identity")
        for name in rule.params.names
    }
    by_match = {rule.params.match: rule.params for rule in pack.rules_of("alias_vocabulary")}

    assert reserved == id_rails.RESERVED_NAMES
    assert dict(by_match["id"].aliases) == id_rails.STATIC_ALIASES
    assert dict(by_match["name"].aliases) == id_rails.PRE_CANON_ALIASES
    assert by_match["id"].types == by_match["name"].types == ()
    assert pack.rules_of("id_exempt")[0].params.types == id_rails.ID_EXEMPT_TYPES
    assert sorted(id_rails.ID_EXEMPT_TYPES) == sorted(id_rails._N1_EXEMPT_TYPES)


def test_the_adapters_plan_vocabulary_is_todays_extractable_lists():
    assert plan_vocabulary(registry.active_pack()) == PlanWords(
        entity_types=tuple(EXTRACTABLE_NODE_TYPES),
        predicates=tuple(EXTRACTABLE_RELATIONSHIP_TYPES),
        fan_out_predicates=tuple(EXTRACTABLE_RELATIONSHIP_TYPES),
    )


def test_the_adapter_adds_and_removes_no_edge():
    assert registry.active_pack().rules_in("relation") == ()


def test_the_adapter_exposes_both_hashes_and_they_are_stable(fresh_registry):
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


def test_the_adapter_hashes_to_the_pinned_values(fresh_registry):
    # What today's ontology hashed to when the adapter's hashes were first pinned. The
    # rails hash covers the source of `id_rails.py`, where the adapter's plug-ins live,
    # so an edit there or to the ontology moves a value on purpose; a loader change
    # must not.
    pack = fresh_registry.active_pack()

    assert (pack.pack_prompt_hash, pack.pack_rails_hash) == (
        "afb06c63e3281cff788d140b7db0fc1bee75cd3ac80461db661b5f406581e03e",
        "4781cce194608f35e864998ae4946c1a5b4070e310f6d7d756aed2582edd73ab",
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
