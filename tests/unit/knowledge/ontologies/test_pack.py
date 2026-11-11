"""The pack loader: what it builds, what it refuses, and what each hash covers.

Every case runs on the fixture packs under `fixtures/`, whose types and rules are
invented, so nothing here depends on the ontology the product ships.
"""

from __future__ import annotations

import dataclasses

import pytest

from backend.errors import MistError
from backend.knowledge.ontologies.pack import (
    ID,
    KINDS,
    PAYLOAD,
    RELATION,
    ConventionRule,
    FanOut,
    GainedEdge,
    IdExempt,
    Pack,
    PackLoadError,
    PlanTrigger,
    Plugin,
    PluginRule,
    PromptFragment,
    TypedId,
    build_pack,
    load_pack,
)
from tests.unit.knowledge.ontologies.fixtures.base_pack import conventions, manifest, plugins, types

BASE = "tests.unit.knowledge.ontologies.fixtures.base_pack"
PET = "tests.unit.knowledge.ontologies.fixtures.pet_pack"


def _build(**overrides) -> Pack:
    parts = {
        "manifest": manifest.MANIFEST,
        "node_types": types.NODE_TYPES,
        "edge_types": types.EDGE_TYPES,
        "extractable_node_types": types.EXTRACTABLE_NODE_TYPES,
        "extractable_edge_types": types.EXTRACTABLE_EDGE_TYPES,
        "rules": conventions.RULES,
        "plugins": plugins.PLUGINS,
        "labelling": "rows",
    }
    parts.update(overrides)
    return build_pack(**parts)


def _rule(rule_id: str) -> ConventionRule:
    return next(rule for rule in conventions.RULES if rule.id == rule_id)


def _with(rule_id: str, **changes) -> tuple[ConventionRule, ...]:
    return tuple(
        dataclasses.replace(rule, **changes) if rule.id == rule_id else rule
        for rule in conventions.RULES
    )


# --- loading -------------------------------------------------------------------------


def test_the_two_type_fixture_pack_loads():
    pack = load_pack(BASE)

    assert pack.version == "0.1.0"
    assert sorted(pack.type_names) == ["Deckhand", "Mooring"]
    assert [rule.kind for rule in pack.rules_in(PAYLOAD)] == ["reserved_identity"]
    assert "F1." in pack.labelling


def test_the_fixture_pack_carries_a_rule_of_every_kind():
    pack = load_pack(BASE)

    assert {rule.kind for rule in pack.rules} == set(KINDS)


def test_the_worked_example_loads_with_its_type_edge_and_convention():
    pack = load_pack(PET)

    assert "Pet" in pack.type_names
    assert pack.pair_table["HAS_PET"] == (frozenset({"Deckhand"}), frozenset({"Pet"}))
    roster = pack.rules_of("fan_out")[-1]
    assert roster.id == "pet-roster"
    assert roster.params == FanOut(
        trigger=PlanTrigger(predicate="HAS_PET", member_type="Pet"),
        gains=(
            GainedEdge(
                edge="HAS_PET",
                source="user",
                target="member",
                unless_member_has=True,
                properties=(("temporal_status", "current"), ("confidence", 0.9)),
            ),
        ),
    )
    assert roster.pinned_case is not None


def test_the_loader_applies_retired_types_to_an_edges_allowed_sets():
    pack = load_pack(BASE)

    assert pack.edges_by_name["CREWS_WITH"].allowed_source_types == ("Deckhand",)


def test_an_edge_that_allows_every_type_maps_to_any():
    pack = load_pack(BASE)

    assert pack.pair_table["ALONGSIDE"] == (None, None)
    assert pack.pair_table["MOORED_AT"] == (frozenset({"Deckhand"}), frozenset({"Mooring"}))


def test_a_pack_is_frozen():
    pack = load_pack(BASE)

    with pytest.raises(dataclasses.FrozenInstanceError):
        pack.labelling = "other"  # type: ignore[misc]


def test_a_missing_pack_module_is_a_load_error():
    with pytest.raises(PackLoadError, match="types.py does not import"):
        load_pack("tests.unit.knowledge.ontologies.fixtures.no_such_pack")


def test_the_load_error_is_a_mist_error():
    assert issubclass(PackLoadError, MistError)


# --- load-time refusals ----------------------------------------------------------------


def test_a_rule_naming_an_unknown_type_is_refused():
    rules = _with("mooring-exempt", params=IdExempt(types=("Mooring", "Lighthouse")))

    with pytest.raises(PackLoadError, match="mooring-exempt: unknown type.*Lighthouse"):
        _build(rules=rules)


def test_a_rule_naming_an_unknown_edge_is_refused():
    crew = _rule("crew-fan-out")
    gains = (GainedEdge(edge="SALUTES", source="member", target="anchor"),)
    rules = _with("crew-fan-out", params=dataclasses.replace(crew.params, gains=gains))

    with pytest.raises(PackLoadError, match="crew-fan-out: unknown edge.*SALUTES"):
        _build(rules=rules)


def test_a_rule_of_an_unknown_kind_is_refused():
    with pytest.raises(PackLoadError, match="mooring-exempt: unknown kind id_skip"):
        _build(rules=_with("mooring-exempt", kind="id_skip"))


def test_a_rule_naming_a_plugin_that_is_not_registered_is_refused():
    rules = _with("mooring-number", params=TypedId(type="Mooring", plugin="berth-letter"))

    with pytest.raises(PackLoadError, match="plug-in berth-letter is not registered"):
        _build(rules=rules)


def test_a_fan_out_endpoint_plugin_that_is_not_registered_is_refused():
    registered = tuple(plugin for plugin in plugins.PLUGINS if plugin.name != "skipper-lodging")

    with pytest.raises(PackLoadError, match="plug-in skipper-lodging is not registered"):
        _build(plugins=registered)


def test_a_plugin_with_no_pinned_case_is_refused():
    with pytest.raises(PackLoadError, match="plug-in mooring-number has no pinned case"):
        _build(rules=_with("mooring-number", pinned_case=None))


def test_a_registered_plugin_no_rule_names_is_refused_as_unpinned():
    spare = Plugin("spare", ID, plugins.mooring_number)

    with pytest.raises(PackLoadError, match="no pinned case: spare"):
        _build(plugins=(*plugins.PLUGINS, spare))


def test_a_pair_table_that_does_not_build_is_refused():
    broken = dataclasses.replace(types.MOORED_AT, allowed_target_types=("Mooring", "Quay"))
    edges = tuple(broken if edge is types.MOORED_AT else edge for edge in types.EDGE_TYPES)

    with pytest.raises(PackLoadError, match="pair table does not build.*MOORED_AT.*Quay"):
        _build(edge_types=edges)


def test_an_extractable_edge_without_a_definition_does_not_build():
    with pytest.raises(PackLoadError, match="pair table does not build.*TOWS"):
        _build(extractable_edge_types=(*types.EXTRACTABLE_EDGE_TYPES, "TOWS"))


def test_a_phase_the_kind_does_not_run_in_is_refused():
    with pytest.raises(PackLoadError, match="fan_out does not run in the id phase"):
        _build(rules=_with("crew-fan-out", phase=ID))


def test_a_plugin_registered_for_another_phase_is_refused():
    rules = _with("no-solo-crew", phase=PAYLOAD, params=PluginRule(plugin="no-solo-crew"))

    with pytest.raises(PackLoadError, match="registered for the relation phase, not payload"):
        _build(rules=rules)


def test_a_relation_rule_that_names_no_prompt_rule_is_refused():
    with pytest.raises(PackLoadError, match="skipper-outranks: a relation rule names"):
        _build(rules=_with("skipper-outranks", prompt_rule=None))


def test_a_gained_edge_without_the_member_is_refused():
    crew = _rule("crew-fan-out")
    gains = (GainedEdge(edge="SERVES_UNDER", source="user", target="anchor"),)
    rules = _with("crew-fan-out", params=dataclasses.replace(crew.params, gains=gains))

    with pytest.raises(PackLoadError, match="the member at one end"):
        _build(rules=rules)


def test_a_manifest_naming_an_unknown_fallback_type_is_refused():
    broken = dataclasses.replace(manifest.MANIFEST, fallback_type="Flotsam")

    with pytest.raises(PackLoadError, match="manifest: unknown type.*Flotsam"):
        _build(manifest=broken)


def test_a_rule_id_used_twice_is_refused():
    with pytest.raises(PackLoadError, match="used twice: mooring-exempt"):
        _build(rules=(*conventions.RULES, _rule("mooring-exempt")))


# --- the two hashes --------------------------------------------------------------------


def _sha256_shaped(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def test_a_pack_exposes_two_sha256_hashes_that_are_stable_across_loads():
    first, second = load_pack(BASE), _build(labelling=load_pack(BASE).labelling)

    assert _sha256_shaped(first.pack_prompt_hash)
    assert _sha256_shaped(first.pack_rails_hash)
    assert first.pack_prompt_hash != first.pack_rails_hash
    assert (first.pack_prompt_hash, first.pack_rails_hash) == (
        second.pack_prompt_hash,
        second.pack_rails_hash,
    )


def test_a_post_llm_rule_change_moves_only_the_rails_hash():
    base = _build()
    changed = _build(rules=_with("mooring-exempt", params=IdExempt(types=("Mooring", "Deckhand"))))

    assert changed.pack_rails_hash != base.pack_rails_hash
    assert changed.pack_prompt_hash == base.pack_prompt_hash


def test_a_pre_llm_rule_change_moves_only_the_prompt_hash():
    base = _build()
    changed = _build(rules=_with("crew-rule-text", params=PromptFragment(text="F1. Reworded.")))

    assert changed.pack_prompt_hash != base.pack_prompt_hash
    assert changed.pack_rails_hash == base.pack_rails_hash


def test_a_type_change_moves_only_the_prompt_hash():
    described = dataclasses.replace(types.MOORING, description="A buoy.")
    changed = _build(node_types=(types.DECKHAND, described))

    assert changed.pack_prompt_hash != _build().pack_prompt_hash
    assert changed.pack_rails_hash == _build().pack_rails_hash


def test_the_fallback_type_moves_only_the_rails_hash():
    changed = _build(manifest=dataclasses.replace(manifest.MANIFEST, fallback_type="Mooring"))

    assert changed.pack_rails_hash != _build().pack_rails_hash
    assert changed.pack_prompt_hash == _build().pack_prompt_hash


def test_a_prompt_visible_manifest_field_moves_only_the_prompt_hash():
    changed = _build(manifest=dataclasses.replace(manifest.MANIFEST, prompt_token_budget=4000))

    assert changed.pack_prompt_hash != _build().pack_prompt_hash
    assert changed.pack_rails_hash == _build().pack_rails_hash


def test_a_plugin_change_moves_only_the_rails_hash():
    def mooring_number(entity, entity_id, target):
        return None

    swapped = tuple(
        (
            Plugin(plugin.name, plugin.phase, mooring_number)
            if plugin.name == "mooring-number"
            else plugin
        )
        for plugin in plugins.PLUGINS
    )
    changed = _build(plugins=swapped)

    assert changed.pack_rails_hash != _build().pack_rails_hash
    assert changed.pack_prompt_hash == _build().pack_prompt_hash


def test_the_order_of_the_rules_in_a_phase_is_part_of_the_rails_hash():
    names = [rule.id for rule in conventions.RULES]
    first, second = names.index("mooring-number"), names.index("rank-aliases")
    reordered = list(conventions.RULES)
    reordered[first], reordered[second] = reordered[second], reordered[first]

    assert _build(rules=tuple(reordered)).pack_rails_hash != _build().pack_rails_hash


def test_the_labelling_text_moves_neither_hash():
    base, changed = _build(labelling="rows"), _build(labelling="rows, reworded")

    assert changed.labelling != base.labelling
    assert changed.pack_prompt_hash == base.pack_prompt_hash
    assert changed.pack_rails_hash == base.pack_rails_hash


def test_a_pinned_case_moves_neither_hash():
    case = _rule("mooring-number").pinned_case
    other = dataclasses.replace(case, scope="third-party")
    changed = _build(rules=_with("mooring-number", pinned_case=other))

    assert changed.pack_prompt_hash == _build().pack_prompt_hash
    assert changed.pack_rails_hash == _build().pack_rails_hash


def test_the_rails_hash_is_computed_over_the_rules_by_phase():
    relation_only = tuple(rule for rule in conventions.RULES if rule.phase != RELATION)
    kept = tuple(plugin for plugin in plugins.PLUGINS if plugin.phase != RELATION)

    without_relation = _build(rules=relation_only, plugins=kept)

    assert without_relation.pack_rails_hash != _build().pack_rails_hash
    assert without_relation.pack_prompt_hash == _build().pack_prompt_hash
