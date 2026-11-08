"""List assembly: how a structured turn's frame and item outputs become one Stage 2 payload.

The list path (`backend.extraction_service.engine`, T19) calls `assemble` once every frame
and item call has returned. These tests pin the template expansion, the checkbox rules,
the item references, the merge, and the id rails that derive each entity's id from its
source span. Every turn is synthetic, with invented names.
"""

from __future__ import annotations

import copy

from backend.extraction_contract.models import ExtractionPayload
from backend.knowledge.extraction.list_assembly import assemble
from backend.knowledge.extraction.list_frames import (
    FrameOutput,
    GroupFrame,
    ItemEntity,
    ItemOutput,
    ItemTemplate,
    ListEntity,
    ListFacts,
    ListRelationship,
    Template,
    group_units,
)
from backend.knowledge.extraction.structure_splitter import split_structured

_USER = {"id": "user", "name": "User", "type": "User"}
_MIST = {"id": "mist-identity", "name": "MIST", "type": "MistIdentity"}


def _turn(utterance: str):
    return group_units(split_structured(utterance))


def _group(gid: str, *templates: tuple[str, str, str], item_type=None, unchecked="plan"):
    return GroupFrame(gid, tuple(Template(*t) for t in templates), item_type, unchecked)


def _frame(*groups: GroupFrame, frame_entities=(), entities=(), relationships=()):
    return FrameOutput(
        tuple(frame_entities), tuple(groups), ListFacts(tuple(entities), tuple(relationships))
    )


def _rel(source: str, type_: str, target: str, **properties) -> ListRelationship:
    return ListRelationship(source, target, type_, properties)


def _item(
    item_id: str,
    span: str | None,
    etype: str = "Technology",
    *,
    applies: bool = True,
    status: str = "current",
    kind: str = "assert",
    start: str | None = None,
    entities=(),
    relationships=(),
    activity: bool = False,
) -> ItemOutput:
    entity = None if span is None else ItemEntity(span, span, etype, activity)
    return ItemOutput(
        item_id,
        entity,
        ItemTemplate(applies, kind, status, start, None),
        tuple(entities),
        tuple(relationships),
    )


def _props(status: str = "current", kind: str = "assert", start=None, end=None) -> dict:
    return {
        "temporal_status": status,
        "start_date": start,
        "end_date": end,
        "assertion_kind": kind,
    }


def _edge(source: str, type_: str, target: str, properties=None) -> dict:
    return {
        "source": source,
        "target": target,
        "type": type_,
        "properties": _props() if properties is None else properties,
    }


def _triples(payload: ExtractionPayload) -> list[tuple[str, str, str]]:
    return [(r["source"], r["type"], r["target"]) for r in payload.relationships]


def _ids(payload: ExtractionPayload) -> list[str]:
    return [e["id"] for e in payload.entities]


def _run(utterance, frame, items, fallbacks=None):
    groups, prose = _turn(utterance)
    return assemble(frame, items, groups, prose, utterance, fallbacks or {})


class TestTemplates:
    def test_each_item_gets_its_groups_templates_with_its_temporal_fields(self):
        utterance = "Tools I use at Thrennic:\n- Quorvane\n- Ostrel"
        frame = _frame(
            _group("G1", ("user", "USES", "item"), ("item", "USED_FOR", "f1")),
            frame_entities=[ListEntity("f1", "Thrennic", "Thrennic", "Organization")],
            relationships=[_rel("user", "WORKS_AT", "f1")],
        )
        items = [
            _item("G1.1", "Quorvane"),
            _item("G1.2", "Ostrel", status="past", kind="cease", start="2025-01-01"),
        ]

        payload, warnings = _run(utterance, frame, items)

        thrennic = {"id": "thrennic", "name": "Thrennic", "type": "Organization"}
        assert payload.entities == [
            _USER,
            thrennic,
            {"id": "quorvane", "name": "Quorvane", "type": "Technology"},
            {"id": "ostrel", "name": "Ostrel", "type": "Technology"},
        ]
        past = _props("past", "cease", "2025-01-01")
        assert payload.relationships == [
            _edge("user", "WORKS_AT", "thrennic", {}),
            _edge("user", "USES", "quorvane"),
            _edge("quorvane", "USED_FOR", "thrennic"),
            _edge("user", "USES", "ostrel", past),
            _edge("ostrel", "USED_FOR", "thrennic", past),
        ]
        assert warnings == []

    def test_applies_false_drops_the_templates_but_keeps_the_items_own_edges(self):
        utterance = "Tools I use:\n- Quorvane\n- Brellix, though I'm only starting on it"
        frame = _frame(_group("G1", ("user", "USES", "item")))
        items = [
            _item("G1.1", "Quorvane"),
            _item(
                "G1.2", "Brellix", applies=False, relationships=[_rel("user", "LEARNING", "item")]
            ),
        ]

        payload, warnings = _run(utterance, frame, items)

        assert _triples(payload) == [
            ("user", "USES", "quorvane"),
            ("user", "LEARNING", "brellix"),
        ]
        assert warnings == []

    def test_a_nested_group_reaches_its_parent_item_and_mist(self):
        utterance = "Tools MIST runs on:\n- Ostrel:\n  - Velquist"
        frame = _frame(
            _group("G1", ("mist", "IMPLEMENTED_WITH", "item")),
            _group("G2", ("item", "DEPENDS_ON", "parent")),
        )
        items = [_item("G1.1", "Ostrel"), _item("G2.1", "Velquist")]

        payload, warnings = _run(utterance, frame, items)

        assert _ids(payload) == ["mist-identity", "ostrel", "velquist"]
        assert payload.entities[0] == _MIST
        assert _triples(payload) == [
            ("mist-identity", "IMPLEMENTED_WITH", "ostrel"),
            ("velquist", "DEPENDS_ON", "ostrel"),
        ]
        assert warnings == []

    def test_the_item_type_falls_back_to_the_groups(self):
        utterance = "Stack:\n- Quorvane"
        frame = _frame(_group("G1", ("user", "USES", "item"), item_type="Technology"))
        item = _item("G1.1", "Quorvane")
        item = ItemOutput(item.item, ItemEntity("Quorvane", "Quorvane", ""), item.template, (), ())

        payload, _ = _run(utterance, frame, [item])

        assert payload.entities[1]["type"] == "Technology"

    def test_a_metric_keeps_value_unit_and_quantity(self):
        utterance = "Service numbers:\n- p95 latency: 120 ms"
        metric = ListEntity(
            "e1", "120 ms", "120 ms", "Metric", value="120", unit="ms", quantity="p95 latency"
        )
        frame = _frame(
            _group("G1"), frame_entities=[ListEntity("f1", "Ostrel", "Ostrel", "Project")]
        )
        items = [
            _item("G1.1", None, entities=[metric], relationships=[_rel("f1", "HAS_METRIC", "e1")])
        ]

        payload, _ = _run(utterance, frame, items)

        record = payload.entities[1]
        assert record["type"] == "Metric"
        assert record["properties"] == {"value": "120", "unit": "ms", "quantity": "p95 latency"}


class TestCheckboxes:
    _PLAN = "Sprint:\n- [x] ship Kestrin\n- [ ] migrate Ostby"

    def test_a_checked_item_states_its_fact_and_an_unchecked_plan_item_states_nothing(self):
        frame = _frame(_group("G1", ("user", "WORKS_ON", "item"), unchecked="plan"))
        items = [
            _item("G1.1", "Kestrin", "Project"),
            _item(
                "G1.2",
                "Ostby",
                "Project",
                relationships=[_rel("item", "DEPENDS_ON", "e1")],
                entities=[ListEntity("e1", "Brill", "Brill", "Technology")],
            ),
        ]

        payload, warnings = _run(self._PLAN, frame, items)

        assert _triples(payload) == [("user", "WORKS_ON", "kestrin")]
        assert _ids(payload) == ["user", "kestrin"]
        assert warnings == []

    def test_an_unchecked_goal_item_gets_exactly_has_goal(self):
        utterance = "Goals for May:\n- [x] learn Kestrin\n- [ ] visit Ostby"
        frame = _frame(_group("G1", ("user", "EXPERIENCED", "item"), unchecked="goal"))
        items = [
            _item("G1.1", "learn Kestrin", "Goal", status="past"),
            _item(
                "G1.2",
                "visit Ostby",
                "Goal",
                status="future",
                relationships=[_rel("user", "INTERESTED_IN", "item")],
            ),
        ]

        payload, warnings = _run(utterance, frame, items)

        assert payload.relationships == [
            _edge("user", "EXPERIENCED", "learn-kestrin", _props("past")),
            _edge("user", "HAS_GOAL", "visit-ostby", _props("future")),
        ]
        assert warnings == []

    def test_an_unchecked_plan_item_used_by_a_frame_fact_keeps_its_entity(self):
        frame = _frame(
            _group("G1", ("user", "WORKS_ON", "item")),
            relationships=[_rel("user", "INTERESTED_IN", "item:G1.2")],
        )
        items = [_item("G1.1", "Kestrin", "Project"), _item("G1.2", "Ostby", "Project")]

        payload, _ = _run(self._PLAN, frame, items)

        assert _triples(payload) == [
            ("user", "INTERESTED_IN", "ostby"),
            ("user", "WORKS_ON", "kestrin"),
        ]


class TestReferences:
    _EDITORS = "Editors I tried:\n- Kestrel\n- Tallow\n- both of them, daily"

    def test_item_references_resolve_in_frame_facts_and_item_relationships(self):
        frame = _frame(
            _group("G1", ("user", "USES", "item")),
            relationships=[_rel("user", "PREFERS", "item:G1.2")],
        )
        items = [
            # G1.1 refers forward to G1.2, which no call has returned yet when G1.1 is read.
            _item("G1.1", "Kestrel", relationships=[_rel("item", "RELATED_TO", "item:G1.2")]),
            _item("G1.2", "Tallow"),
            # "both of them": the item makes no entity and points at the two earlier items.
            _item(
                "G1.3",
                None,
                applies=False,
                relationships=[
                    _rel("user", "HAS_HABIT", "item:G1.1"),
                    _rel("user", "HAS_HABIT", "item:G1.2"),
                ],
            ),
        ]

        payload, warnings = _run(self._EDITORS, frame, items)

        assert _ids(payload) == ["user", "tallow", "kestrel"]
        assert _triples(payload) == [
            ("user", "PREFERS", "tallow"),
            ("user", "USES", "kestrel"),
            ("kestrel", "RELATED_TO", "tallow"),
            ("user", "USES", "tallow"),
            ("user", "HAS_HABIT", "kestrel"),
            ("user", "HAS_HABIT", "tallow"),
        ]
        # PREFERS and both HAS_HABIT edges sit on a pair a USES template already holds.
        assert warnings == ["list_item_edge_on_template_pair"] * 3

    def test_an_unresolved_reference_or_unknown_role_is_dropped_with_a_warning(self):
        frame = _frame(
            _group("G1", ("user", "USES", "item")),
            relationships=[
                _rel("user", "PREFERS", "item:G7.1"),
                _rel("item", "RELATED_TO", "user"),
                _rel("user", "PREFERS", "e4"),
            ],
        )
        items = [
            _item("G1.1", "Kestrel", relationships=[_rel("item", "RELATED_TO", "item:G1.3")]),
            _item("G1.2", "Tallow", relationships=[_rel("parent", "RELATED_TO", "item")]),
            _item("G1.3", None, applies=False),
        ]

        payload, warnings = _run(self._EDITORS, frame, items)

        assert _triples(payload) == [("user", "USES", "kestrel"), ("user", "USES", "tallow")]
        assert warnings == ["list_item_ref_unresolved"] * 5

    def test_an_item_edge_with_another_predicate_on_a_template_pair_is_kept_and_flagged(self):
        utterance = "Editors:\n- Kestrel, my favourite"
        frame = _frame(_group("G1", ("user", "USES", "item")))
        items = [
            _item(
                "G1.1",
                "Kestrel",
                relationships=[_rel("user", "PREFERS", "item"), _rel("user", "USES", "item")],
            )
        ]

        payload, warnings = _run(utterance, frame, items)

        assert _triples(payload) == [("user", "USES", "kestrel"), ("user", "PREFERS", "kestrel")]
        assert warnings == ["list_item_edge_on_template_pair"]


class TestMerge:
    _TRIPS = "Trips:\n- Velmont\n- Ostby\n- Brill"

    def test_a_repeated_id_leaves_one_record_and_a_type_conflict_warns(self):
        date = ListEntity("e1", "2026-03-12", "2026-03-12", "Date", value="2026-03-12")
        frame = _frame(_group("G1", ("user", "EXPERIENCED", "item"), item_type="Event"))
        items = [
            _item(
                "G1.1",
                "Velmont",
                "Event",
                entities=[date],
                relationships=[_rel("item", "OCCURRED_ON", "e1")],
            ),
            _item(
                "G1.2",
                "Ostby",
                "Event",
                entities=[date],
                relationships=[_rel("item", "OCCURRED_ON", "e1")],
            ),
            _item("G1.3", "Velmont", "Location"),
        ]

        payload, warnings = _run(self._TRIPS, frame, items)

        assert _ids(payload) == ["user", "velmont", "2026-03-12", "ostby"]
        assert payload.entities[1]["type"] == "Event"
        assert _triples(payload) == [
            ("user", "EXPERIENCED", "velmont"),
            ("velmont", "OCCURRED_ON", "2026-03-12"),
            ("user", "EXPERIENCED", "ostby"),
            ("ostby", "OCCURRED_ON", "2026-03-12"),
        ]
        assert warnings == ["list_merge_type_conflict"]

    def test_an_item_fallback_payload_joins_in_its_items_place(self):
        frame = _frame(_group("G1", ("user", "EXPERIENCED", "item"), item_type="Event"))
        fallback = ExtractionPayload(
            entities=[_USER, {"id": "ostby-fair", "name": "Ostby fair", "type": "Event"}],
            relationships=[_edge("user", "EXPERIENCED", "ostby-fair", {"confidence": 0.9})],
        )
        before = copy.deepcopy(fallback)
        items = [_item("G1.1", "Velmont", "Event"), _item("G1.3", "Brill", "Event")]

        payload, warnings = _run(self._TRIPS, frame, items, {"G1.2": fallback})

        assert _ids(payload) == ["user", "velmont", "ostby-fair", "brill"]
        assert _triples(payload) == [
            ("user", "EXPERIENCED", "velmont"),
            ("user", "EXPERIENCED", "ostby-fair"),
            ("user", "EXPERIENCED", "brill"),
        ]
        assert fallback == before
        assert warnings == []


class TestPurity:
    def test_the_same_inputs_give_the_same_payload_and_are_not_changed(self):
        utterance = "Tools I use at Thrennic:\n- Quorvane\n- Ostrel"
        frame = _frame(
            _group("G1", ("user", "USES", "item")),
            frame_entities=[ListEntity("f1", "Thrennic", "Thrennic", "Organization")],
            relationships=[_rel("user", "WORKS_AT", "f1", confidence=0.9)],
        )
        items = [_item("G1.1", "Quorvane"), _item("G1.2", "Ostrel")]
        frame_before = copy.deepcopy(frame)

        first = _run(utterance, frame, items)
        first[0].entities[0]["name"] = "changed"
        first[0].relationships[0]["properties"]["confidence"] = 0.1
        second = _run(utterance, frame, items)

        assert second[0].entities[0] == _USER
        assert second[0].relationships[0]["properties"] == {"confidence": 0.9}
        assert frame == frame_before
        third = _run(utterance, frame, items)
        assert third[0] == second[0] and third[1] == second[1]
