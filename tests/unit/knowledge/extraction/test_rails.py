"""The rail runner: every post-LLM kind, run on the fixture packs.

The fixture packs under `tests/unit/knowledge/ontologies/fixtures/` name invented types,
so a case that passes here shows the runner reading a rule, not knowing an ontology.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json

import pytest

from backend.knowledge.extraction import id_rails, rails
from backend.knowledge.extraction.rails import (
    RailRow,
    check_pinned_cases,
    edge_is_live,
    rail_row,
    run_rails,
)
from backend.knowledge.ontologies.pack import (
    ENDPOINT_FIXED,
    ENDPOINT_MEMBER,
    ENDPOINT_PLUGIN_PREFIX,
    RELATION,
    ConventionRule,
    EdgeTrigger,
    FanOut,
    GainedEdge,
    Pack,
    PackLoadError,
    PinnedCase,
    PlanTrigger,
    Plugin,
    build_pack,
    load_pack,
)
from tests.unit.knowledge.ontologies.fixtures.pet_pack.conventions import PET_ROSTER_CASE

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")


def _run(entities, relationships=(), pack: Pack = BASE, **row):
    return run_rails(
        rail_row(copy.deepcopy(list(entities)), copy.deepcopy(list(relationships)), **row), pack
    )


def _one(entity: dict, pack: Pack = BASE) -> dict:
    entities, _ = _run([entity], pack=pack)
    return entities[0]


def _hand(entity_id: str, **extra) -> dict:
    return {"id": entity_id, "name": entity_id, "type": "Deckhand", **extra}


def _edge(source: str, target: str, edge_type: str, **properties) -> dict:
    relationship = {"source": source, "target": target, "type": edge_type}
    if properties:
        relationship["properties"] = properties
    return relationship


def _case_row(case) -> RailRow:
    return rail_row(
        copy.deepcopy(list(case.entities)),
        copy.deepcopy(list(case.relationships)),
        scope=case.scope,
        batch_record=copy.deepcopy(case.batch_record),
        turn_plan=copy.deepcopy(case.turn_plan),
    )


# --- the runner ---------------------------------------------------------------------------


def test_run_rails_takes_the_cache_row_and_the_pack_and_nothing_else():
    assert list(inspect.signature(run_rails).parameters) == ["cache_row", "pack"]


def test_the_phases_are_separate_steps_in_the_runners_order():
    assert [name for name, _ in rails.PHASES] == ["payload", "relation", "id", "fallback"]


def test_the_row_holds_the_payload_the_scope_the_batch_record_and_the_plan():
    assert list(RailRow.__dataclass_fields__) == [
        "entities",
        "relationships",
        "scope",
        "batch_record",
        "turn_plan",
    ]


def test_a_row_without_a_batch_record_or_a_plan_is_one_batch_with_no_plan():
    row = rail_row([], [])

    assert (row.scope, row.batch_record, row.turn_plan) == (None, (), None)


def test_a_stored_plan_record_is_read_through_to_its_payload():
    plan = {"entities": [], "templates": [], "back_references": []}

    wrapped = rail_row([], [], turn_plan={"payload": plan, "finish_reason": "stop"})

    assert wrapped.turn_plan == plan
    assert rail_row([], [], turn_plan=plan).turn_plan == plan
    assert rail_row([], [], turn_plan={"payload": None}).turn_plan is None


@pytest.mark.parametrize("pack", [BASE, PET], ids=["base", "pet"])
def test_every_pinned_case_of_the_fixture_packs_holds(pack):
    assert check_pinned_cases(pack) == []


def _unheld_pet() -> Pack:
    """The pet pack again, with a `pet-roster` case that expects no relationship."""
    roster = next(rule for rule in PET.rules if rule.id == "pet-roster")
    wrong = dataclasses.replace(roster.pinned_case, expected_relationships=())
    rules = tuple(
        dataclasses.replace(rule, pinned_case=wrong) if rule.id == "pet-roster" else rule
        for rule in PET.rules
    )
    return dataclasses.replace(PET, rules=rules)


@pytest.fixture
def case_checks(monkeypatch) -> list[Pack]:
    """The packs whose pinned cases the runner runs during the test, one entry per run."""
    checked: list[Pack] = []
    real = rails.check_pinned_cases

    def counted(pack: Pack) -> list[str]:
        checked.append(pack)
        return real(pack)

    monkeypatch.setattr(rails, "check_pinned_cases", counted)
    return checked


def test_a_pinned_case_that_does_not_hold_is_named():
    assert check_pinned_cases(_unheld_pet()) == ["pet-roster"]


def test_run_rails_refuses_a_pack_whose_pinned_case_does_not_hold_at_every_call(case_checks):
    unheld = _unheld_pet()

    for _ in range(2):
        with pytest.raises(PackLoadError, match="pinned case.*do not hold: pet-roster"):
            run_rails(rail_row([_hand("The Cook")], []), unheld)

    assert case_checks == [unheld]


def test_run_rails_checks_the_pinned_cases_of_a_pack_once(case_checks):
    pack = dataclasses.replace(PET)

    for _ in range(3):
        entities, _ = run_rails(rail_row([_hand("The Cook")], []), pack)

    assert entities[0]["id"] == "cook"
    assert case_checks == [pack]


def test_the_single_entity_and_derived_id_steps_refuse_such_a_pack_too(case_checks):
    unheld = _unheld_pet()

    with pytest.raises(PackLoadError, match="pinned case.*do not hold: pet-roster"):
        rails.normalize_entity(_hand("The Cook"), unheld)
    with pytest.raises(PackLoadError, match="pinned case.*do not hold: pet-roster"):
        rails.derived_ids([_hand("The Cook")], [], unheld)

    assert case_checks == [unheld]


def test_checking_the_pinned_cases_does_not_go_through_the_guard(case_checks):
    # `check_pinned_cases` runs the runner on the pinned inputs. Were that the guarded
    # entry point, the check would call itself.
    pack = dataclasses.replace(PET)

    assert rails.check_pinned_cases(pack) == []
    assert case_checks == [pack]


@pytest.mark.parametrize("pack", [BASE, PET], ids=["base", "pet"])
def test_run_rails_is_deterministic_to_the_byte(pack):
    outputs = set()
    for _ in range(2):
        for rule in pack.rules:
            if rule.pinned_case is not None:
                entities, relationships = run_rails(_case_row(rule.pinned_case), pack)
                outputs.add((rule.id, json.dumps([entities, relationships])))

    assert len(outputs) == sum(1 for rule in pack.rules if rule.pinned_case is not None)


def test_run_rails_changes_the_rows_dicts_in_place_and_returns_the_kept_ones():
    entity = _hand("The Cook")
    row = rail_row([entity], [])

    entities, _ = run_rails(row, BASE)

    assert entities[0] is entity
    assert entity["id"] == "cook"


# --- the user anchor and the manifest's data ------------------------------------------------


@pytest.mark.parametrize("raw_id", ["user", "User", "USER"])
def test_the_user_anchor_keeps_its_id_and_its_type(raw_id):
    entity = _one({"id": raw_id, "name": "The Harbourmaster", "type": "Wrong"})

    assert (entity["id"], entity["type"]) == ("user", "Wrong")


def test_a_retired_type_is_replaced_and_gains_its_property_defaults():
    entity = _one({"id": "jem", "type": "Stowaway"})

    assert (entity["type"], entity["properties"]) == ("Deckhand", {"rank": "none"})


def test_a_retired_types_property_default_does_not_overwrite():
    entity = _one({"id": "jem", "type": "Stowaway", "properties": {"rank": "mate"}})

    assert entity["properties"] == {"rank": "mate"}


def test_an_unknown_type_takes_the_fallback_type_last():
    entity = _one({"id": "the-flotsam", "type": "Driftwood"})

    assert (entity["id"], entity["type"]) == ("flotsam", "Deckhand")


# --- reserved_identity ----------------------------------------------------------------------


def test_a_reserved_display_name_sets_the_id_and_the_type():
    entity = _one({"id": "hm", "name": "The Harbourmaster", "type": "Mooring"})

    assert (entity["id"], entity["type"]) == ("harbourmaster", "Deckhand")


def test_a_reserved_id_slug_sets_the_id_and_the_type():
    entity = _one({"id": "Harbour Master", "name": "Old Tam", "type": "Mooring"})

    assert (entity["id"], entity["type"]) == ("harbourmaster", "Deckhand")


def test_endpoints_follow_a_reserved_entity():
    _, relationships = _run(
        [{"id": "hm", "name": "the harbourmaster", "type": "Deckhand"}, _hand("nell")],
        [_edge("nell", "hm", "KNOWS_HAND")],
    )

    assert relationships == [_edge("nell", "harbourmaster", "KNOWS_HAND")]


# --- id_exempt, typed_id, alias_vocabulary ---------------------------------------------------


def test_an_exempt_type_keeps_its_leading_word_and_another_type_loses_it():
    assert _one({"id": "the-far-one", "name": "the far one", "type": "Mooring"})["id"] == (
        "the-far-one"
    )
    assert _one(_hand("the-cook"))["id"] == "cook"


def test_a_typed_id_is_built_by_its_plugin():
    entity = _one({"id": "the-north-mooring", "name": "North Mooring No. 7", "type": "Mooring"})

    assert entity["id"] == "mooring-7"


def test_a_typed_id_plugin_that_returns_none_keeps_the_id():
    assert _one({"id": "Far Mooring", "name": "Far Mooring", "type": "Mooring"})["id"] == (
        "far-mooring"
    )


def test_a_typed_id_applies_to_its_type_only():
    assert _one(_hand("deckhand 7"))["id"] == "deckhand-7"


def test_a_name_alias_replaces_the_slug():
    assert _one({"id": "skipper-one", "name": " Cap'n ", "type": "Mooring"})["id"] == "captain"


def test_an_id_alias_applies_to_its_types_only():
    assert _one({"id": "Bosun", "name": "Bosun", "type": "Deckhand"})["id"] == "boatswain"
    assert _one({"id": "Bosun", "name": "Bosun", "type": "Mooring"})["id"] == "bosun"


def test_the_engines_registry_reads_the_id_the_packs_rules_gave(monkeypatch):
    monkeypatch.setitem(id_rails.CANONICAL_REGISTRY, "the-boatswain", ("deck-boss", "Mooring"))

    entity = _one({"id": "the-boatswain", "name": "x", "type": "Deckhand"})

    assert (entity["id"], entity["type"]) == ("deck-boss", "Mooring")


# --- derived_edge ----------------------------------------------------------------------------


def test_an_id_phase_derived_edge_builds_the_id_from_its_one_target():
    entities, relationships = _run(
        [_hand("Tolliver"), {"id": "mooring 12", "name": "Mooring 12", "type": "Mooring"}],
        [_edge("Tolliver", "mooring 12", "MOORED_AT")],
    )

    assert [entity["id"] for entity in entities] == ["tolliver-at-mooring-12", "mooring-12"]
    assert entities[0]["name"] == "Tolliver mooring-12"
    assert relationships == [_edge("tolliver-at-mooring-12", "mooring-12", "MOORED_AT")]


def test_an_id_phase_derived_edge_with_two_targets_changes_nothing():
    entities, _ = _run(
        [
            _hand("tolliver"),
            {"id": "m1", "name": "Mooring 1", "type": "Mooring"},
            {"id": "m2", "name": "Mooring 2", "type": "Mooring"},
        ],
        [_edge("tolliver", "m1", "MOORED_AT"), _edge("tolliver", "m2", "MOORED_AT")],
    )

    assert entities[0]["id"] == "tolliver"


def test_an_id_phase_derived_edge_ignores_a_target_its_plugin_refuses():
    entities, _ = _run(
        [_hand("tolliver"), {"id": "the quay", "name": "The quay", "type": "Mooring"}],
        [_edge("tolliver", "the quay", "MOORED_AT")],
    )

    assert entities[0]["id"] == "tolliver"


def test_a_relation_phase_derived_edge_builds_the_node_and_the_edge_from_a_property():
    entities, relationships = _run([_hand("Hale", properties={"home_mooring": "mooring-9"})])

    assert entities[1] == {"id": "mooring-9", "name": "mooring-9", "type": "Mooring"}
    assert relationships == [_edge("hale", "mooring-9", "LODGES_AT", temporal_status="current")]


def test_a_relation_phase_derived_edge_reuses_a_node_the_payload_holds():
    entities, relationships = _run(
        [
            _hand("hale", properties={"home_mooring": "mooring-9"}),
            {"id": "mooring-9", "name": "Mooring 9", "type": "Mooring"},
        ],
        [_edge("hale", "mooring-9", "LODGES_AT", confidence=0.6)],
    )

    assert len(entities) == 2
    assert relationships == [_edge("hale", "mooring-9", "LODGES_AT", confidence=0.6)]


# --- fan_out: an edge pattern in the payload -------------------------------------------------


def _crew(member_properties: dict, scope: str | None = "user-scope"):
    return _run(
        [_hand("user"), _hand("oswin"), _hand("brannoch")],
        [_edge("oswin", "brannoch", "CREWS_WITH", **member_properties)],
        scope=scope,
    )


def test_an_edge_trigger_gives_each_member_its_edges_and_copies_the_members_edge():
    _, relationships = _crew({"confidence": 0.8, "temporal_status": "current", "end_date": None})

    copied = {"temporal_status": "current", "confidence": 0.8}
    assert relationships[1:] == [
        _edge("oswin", "brannoch", "SERVES_UNDER", **copied),
        _edge("user", "oswin", "KNOWS_HAND", **copied),
    ]


@pytest.mark.parametrize(
    "properties",
    [
        {"temporal_status": "past"},
        {"negated": True},
        {"assertion_kind": "cease"},
        {"assertion_kind": "retract"},
    ],
)
def test_an_edge_trigger_adds_nothing_from_an_edge_that_is_not_a_live_assertion(properties):
    _, relationships = _crew(properties)

    assert len(relationships) == 1


def _crew_twice(first: dict, second: dict) -> list[dict]:
    """The crew payload with the member's trigger edge written twice."""
    _, relationships = _run(
        [_hand("user"), _hand("oswin"), _hand("brannoch")],
        [
            _edge("oswin", "brannoch", "CREWS_WITH", **first),
            _edge("oswin", "brannoch", "CREWS_WITH", **second),
        ],
        scope="user-scope",
    )
    return relationships


def test_an_edge_trigger_adds_nothing_when_the_members_first_trigger_edge_is_past():
    # The first copy of an edge is the one that is stored, so an edge built on the live
    # second copy would stand on an edge the graph holds as past.
    relationships = _crew_twice(
        {"temporal_status": "past"}, {"temporal_status": "current", "confidence": 0.9}
    )

    assert relationships == [_edge("oswin", "brannoch", "CREWS_WITH", temporal_status="past")]


def test_an_edge_trigger_adds_nothing_when_a_later_trigger_edge_of_the_member_is_past():
    relationships = _crew_twice(
        {"temporal_status": "current", "confidence": 0.9}, {"temporal_status": "past"}
    )

    assert relationships == [
        _edge("oswin", "brannoch", "CREWS_WITH", temporal_status="current", confidence=0.9)
    ]


def test_a_gained_edge_with_a_scope_condition_needs_that_scope():
    for scope in ("third-party", None):
        _, relationships = _crew({}, scope=scope)

        assert [r["type"] for r in relationships] == ["CREWS_WITH", "SERVES_UNDER"]


def test_an_edge_trigger_reads_only_edges_between_its_endpoint_types():
    _, relationships = _run(
        [_hand("oswin"), {"id": "mooring-2", "name": "Mooring 2", "type": "Mooring"}],
        [_edge("oswin", "mooring-2", "CREWS_WITH")],
    )

    assert len(relationships) == 1


def test_a_fan_out_adds_no_edge_the_payload_already_holds_and_none_to_itself():
    _, relationships = _run(
        [_hand("user"), _hand("brannoch")],
        [_edge("user", "brannoch", "CREWS_WITH"), _edge("user", "brannoch", "SERVES_UNDER")],
        scope="user-scope",
    )

    assert relationships == [
        _edge("user", "brannoch", "CREWS_WITH"),
        _edge("user", "brannoch", "SERVES_UNDER"),
    ]


def test_a_fan_out_endpoint_plugin_names_the_endpoint_from_the_payload():
    _, relationships = _run(
        [
            _hand("oswin"),
            _hand("brannoch"),
            {"id": "mooring-3", "name": "Mooring 3", "type": "Mooring"},
        ],
        [_edge("oswin", "brannoch", "CREWS_WITH"), _edge("brannoch", "mooring-3", "LODGES_AT")],
    )

    assert _edge("oswin", "mooring-3", "LODGES_AT") in relationships


@pytest.mark.parametrize(
    "properties",
    [{"temporal_status": "past"}, {"negated": True}, {"assertion_kind": "cease"}],
)
def test_a_fan_out_endpoint_plugin_is_shown_live_edges_only(properties):
    _, relationships = _run(
        [
            _hand("oswin"),
            _hand("brannoch"),
            {"id": "mooring-3", "name": "Mooring 3", "type": "Mooring"},
        ],
        [
            _edge("oswin", "brannoch", "CREWS_WITH"),
            _edge("brannoch", "mooring-3", "LODGES_AT", **properties),
        ],
    )

    assert [r for r in relationships if r["source"] == "oswin" and r["type"] == "LODGES_AT"] == []


_SEEN_GROUPS: list = []


def _record_group(group, row):
    """A fan_out endpoint plug-in that keeps the group it is shown and names no endpoint."""
    _SEEN_GROUPS.append(group)


def test_a_fan_out_group_lists_only_the_members_whose_trigger_edge_is_live():
    watch = ConventionRule(
        id="crew-watch",
        kind="fan_out",
        phase=RELATION,
        params=FanOut(
            trigger=EdgeTrigger(edge="CREWS_WITH"),
            gains=(
                GainedEdge(
                    edge="KNOWS_HAND",
                    source=ENDPOINT_MEMBER,
                    target=ENDPOINT_PLUGIN_PREFIX + "record-group",
                ),
            ),
        ),
        prompt_rule="Rule F5",
        pinned_case=PinnedCase((), (), (), ()),
    )
    pack = _with_rule(BASE, watch, Plugin("record-group", RELATION, _record_group))
    # The first run of a pack checks its pinned cases, and the plug-in would record the
    # groups of those too: check them here, then start the record empty.
    rails.require_pinned_cases(pack)
    _SEEN_GROUPS.clear()

    _run(
        [_hand("oswin"), _hand("wend"), _hand("tamsin"), _hand("brannoch")],
        [
            _edge("oswin", "brannoch", "CREWS_WITH"),
            _edge("wend", "brannoch", "CREWS_WITH", temporal_status="past"),
            _edge("tamsin", "brannoch", "CREWS_WITH", temporal_status="past"),
            _edge("tamsin", "brannoch", "CREWS_WITH"),
        ],
        pack=pack,
    )

    assert [(group.member, group.members) for group in _SEEN_GROUPS] == [("oswin", ("oswin",))]


def test_the_fixture_crew_case_fires_the_fan_out_the_plugin_and_the_suppression():
    case = next(rule for rule in BASE.rules if rule.id == "crew-fan-out").pinned_case

    entities, relationships = run_rails(_case_row(case), BASE)

    assert entities == list(case.expected_entities)
    assert relationships == list(case.expected_relationships)
    assert "OUTRANKS" in {r["type"] for r in case.relationships}
    assert "OUTRANKS" not in {r["type"] for r in relationships}


# --- fan_out: a lead-in template in the row's plan ---------------------------------------------


def test_the_pet_roster_gives_the_user_each_pet_of_the_lead_ins_batches():
    entities, relationships = run_rails(_case_row(PET_ROSTER_CASE), PET)

    assert entities == list(PET_ROSTER_CASE.expected_entities)
    assert relationships == list(PET_ROSTER_CASE.expected_relationships)
    assert _edge("user", "pip", "HAS_PET") not in relationships


def _pet_row(**changes) -> RailRow:
    return _case_row(dataclasses.replace(PET_ROSTER_CASE, **changes))


def test_a_plan_trigger_adds_nothing_without_a_plan():
    _, relationships = run_rails(_pet_row(turn_plan=None), PET)

    assert relationships == list(PET_ROSTER_CASE.relationships)


def test_a_plan_trigger_adds_nothing_without_a_batch_record():
    _, relationships = run_rails(_pet_row(batch_record=None), PET)

    assert relationships == list(PET_ROSTER_CASE.relationships)


def test_a_plan_trigger_needs_a_template_that_names_its_predicate():
    plan = copy.deepcopy(PET_ROSTER_CASE.turn_plan)
    plan["templates"][0]["predicate"] = "KNOWS_HAND"

    _, relationships = run_rails(_pet_row(turn_plan=plan), PET)

    assert relationships == list(PET_ROSTER_CASE.relationships)


def test_a_plan_trigger_fires_on_a_templates_fan_out_predicate():
    plan = copy.deepcopy(PET_ROSTER_CASE.turn_plan)
    plan["templates"][0]["predicate"] = "KNOWS_HAND"
    plan["templates"][0]["fan_out"] = [{"source": "user", "predicate": "HAS_PET", "target": "each"}]

    _, relationships = run_rails(_pet_row(turn_plan=plan), PET)

    assert relationships == list(PET_ROSTER_CASE.expected_relationships)


def test_a_plan_trigger_matches_the_lead_in_as_a_heading_writes_it():
    plan = copy.deepcopy(PET_ROSTER_CASE.turn_plan)
    plan["templates"][0]["lead_in"] = "## our  PETS"

    _, relationships = run_rails(_pet_row(turn_plan=plan), PET)

    assert relationships == list(PET_ROSTER_CASE.expected_relationships)


def test_a_plan_trigger_adds_the_user_anchor_when_the_payload_lacks_it():
    entities = [e for e in PET_ROSTER_CASE.entities if e["id"] != "user"]
    relationships = [r for r in PET_ROSTER_CASE.relationships if r["source"] != "user"]

    out_entities, out_relationships = run_rails(
        _pet_row(entities=tuple(entities), relationships=tuple(relationships)), PET
    )

    assert out_entities[-1] == {"id": "user", "name": "user", "type": "Deckhand"}
    assert {(r["source"], r["target"]) for r in out_relationships if r["source"] == "user"} == {
        ("user", "biscuit"),
        ("user", "mango"),
    }


def _with_rule(pack: Pack, rule: ConventionRule, *plugins: Plugin) -> Pack:
    """`pack` again with one more rule, and the plug-ins that rule names."""
    return build_pack(
        manifest=pack.manifest,
        node_types=pack.node_types,
        edge_types=pack.edge_types,
        extractable_node_types=pack.extractable_node_types,
        extractable_edge_types=pack.extractable_edge_types,
        rules=(*pack.rules, rule),
        plugins=(*pack.plugins, *plugins),
    )


# The pet pack, plus a rule that gives each pet of a lead-in to the entity the plan's
# template names as its fixed endpoint.
KENNEL = _with_rule(
    PET,
    ConventionRule(
        id="kennel-roster",
        kind="fan_out",
        phase=RELATION,
        params=FanOut(
            trigger=PlanTrigger(predicate="HAS_PET", member_type="Pet"),
            gains=(GainedEdge(edge="HAS_PET", source=ENDPOINT_FIXED, target=ENDPOINT_MEMBER),),
        ),
        prompt_rule="N19",
    ),
)


def _kennel_owners(fixed_endpoint: str) -> set[str]:
    plan = copy.deepcopy(PET_ROSTER_CASE.turn_plan)
    plan["templates"][0]["fixed_endpoint"] = fixed_endpoint
    _, relationships = run_rails(_pet_row(turn_plan=plan), KENNEL)
    return {r["target"] for r in relationships if r["source"] == "corwen"}


@pytest.mark.parametrize("written", ["corwen", "Corwen"], ids=["as-the-id", "as-the-name"])
def test_a_plan_trigger_gives_each_member_an_edge_from_the_templates_fixed_endpoint(written):
    assert _kennel_owners(written) == {"sable", "biscuit", "mango"}


@pytest.mark.parametrize("written", ["", "nobody-in-the-payload"], ids=["none", "unknown"])
def test_a_fixed_endpoint_that_names_no_payload_entity_adds_no_edge(written):
    assert _kennel_owners(written) == {"sable"}


def test_the_pet_convention_is_unknown_to_the_base_pack():
    _, relationships = run_rails(_case_row(PET_ROSTER_CASE), BASE)

    assert relationships == list(PET_ROSTER_CASE.relationships)


# --- edge_suppress and the plug-in rule --------------------------------------------------------


def test_an_edge_outside_the_suppressed_pattern_is_kept():
    _, relationships = _run([_hand("nell"), _hand("bo")], [_edge("nell", "bo", "OUTRANKS")])

    assert relationships == [_edge("nell", "bo", "OUTRANKS")]


def test_an_edge_in_the_suppressed_pattern_is_dropped():
    _, relationships = _run(
        [_hand("nell"), _hand("bo")],
        [
            _edge("bo", "nell", "CREWS_WITH", temporal_status="past"),
            _edge("nell", "bo", "OUTRANKS"),
        ],
    )

    assert [r["type"] for r in relationships] == ["CREWS_WITH"]


def test_a_plugin_rule_is_the_whole_rule():
    _, relationships = _run([_hand("fenwick")], [_edge("fenwick", "fenwick", "CREWS_WITH")])

    assert relationships == []


def test_edge_is_live_reads_both_levels_of_an_edge():
    assert edge_is_live({"type": "X"})
    assert edge_is_live({"properties": {"assertion_kind": "ASSERT", "temporal_status": "future"}})
    assert not edge_is_live({"assertion_kind": "cease"})
    assert not edge_is_live({"negated": True})
    assert not edge_is_live({"properties": {"temporal_status": "past"}})
