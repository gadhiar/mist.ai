"""List assembly: how a structured turn's frame and item outputs become one Stage 2 payload.

The list path (`backend.extraction_service.engine`, T19) calls `assemble` once every frame
and item call has returned. These tests pin the template expansion, the checkbox rules,
the item references, the merge, and the id rails that derive each entity's id from its
source span. Every turn is synthetic, with invented names.
"""

from __future__ import annotations

import copy
import difflib
import sys
from pathlib import Path

import pytest

from backend.extraction_contract.models import ExtractionPayload
from backend.knowledge.extraction import list_assembly
from backend.knowledge.extraction.canonical_id import canonical_metric_id
from backend.knowledge.extraction.list_assembly import assemble, list_entity_id
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
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.structure_splitter import split_structured
from tests.mocks.embeddings import FakeEmbeddingGenerator

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval_harness.score_extraction_run import (  # noqa: E402
    _produced_entity_match_id,
    canonical_id,
)

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


# -- id rails ---------------------------------------------------------------------------------


def _id(span: str, etype: str, utterance: str, name: str | None = None, **kwargs):
    return list_entity_id(span, span if name is None else name, etype, utterance, **kwargs)


class TestSnap:
    def test_an_exact_match_on_word_boundaries_wins_over_one_inside_a_word(self):
        utterance = "Tools:\n- KESTRELvane fork\n- Kestrel"
        frame = _frame(_group("G1", ("user", "USES", "item")))
        items = [_item("G1.2", "kestrel")]

        payload, warnings = _run(utterance, frame, items)

        # The name keeps the utterance's casing, which shows where the span snapped.
        assert payload.entities[1] == {"id": "kestrel", "name": "Kestrel", "type": "Technology"}
        assert warnings == []

    def test_an_exact_match_inside_a_word_snaps_to_the_span(self):
        assert _id("Ilse", "Person", "I met Ilse's team") == ("ilse", [])
        assert _id("brindle", "Technology", "my Brindlecraft kit") == ("brindle", [])

    def test_a_fuzzy_window_at_the_threshold_snaps_and_one_below_does_not(self):
        span = "Ostrelvane Quindarow"
        at, below = "Ostrelvane Quindaxyz", "Ostrelvane Quindxyzq"
        ratio = difflib.SequenceMatcher(None, span.lower(), at.lower()).ratio()
        assert ratio == 0.85
        assert difflib.SequenceMatcher(None, span.lower(), below.lower()).ratio() < 0.85

        assert _id(span, "Project", f"We ship {at} soon.") == ("ostrelvane-quindaxyz", [])
        assert _id(span, "Project", f"We ship {below} soon.") == (
            "ostrelvane-quindarow",
            ["list_span_unmatched"],
        )

    def test_a_typo_in_the_span_snaps_to_the_utterances_words(self):
        assert _id("Quorvanne dashboard", "Project", "the Quorvane dashboard, daily") == (
            "quorvane-dashboard",
            [],
        )


class TestRails:
    @pytest.mark.parametrize(
        "etype", ["Person", "Organization", "Technology", "Project", "Goal", "Location"]
    )
    def test_a_trailing_possessive_is_dropped_for_every_type(self, etype):
        assert _id("Marta's", etype, "That is Marta's.") == ("marta", [])
        assert _id("Marta’s", etype, "That is Marta’s.") == ("marta", [])

    def test_a_plural_is_kept_as_written(self):
        assert _id("Editors", "Concept", "Editors I tried") == ("editors", [])

    @pytest.mark.parametrize(
        ("span", "expected"),
        [
            ("the Brennic dashboard", "brennic-dashboard"),
            ("An Ostrel board", "ostrel-board"),
            ("my old laptop", "old-laptop"),
            ("Ilse and our band", "ilse-and-band"),
            ("Hall of the Mountain", "hall-of-the-mountain"),
        ],
    )
    def test_n1_drops_leading_articles_and_possessive_pronouns(self, span, expected):
        assert _id(span, "Project", f"About {span} today") == (expected, [])

    @pytest.mark.parametrize(
        ("span", "expected"),
        [
            ("finish the thesis", "finish-thesis"),
            ("learn my Kestrin basics", "learn-kestrin-basics"),
        ],
    )
    def test_n2_a_goal_drops_articles_and_possessive_pronouns_anywhere(self, span, expected):
        assert _id(span, "Goal", f"I want to {span}") == (expected, [])

    def test_a_metric_id_comes_from_canonical_metric_id_with_the_quantity(self, monkeypatch):
        kwargs = {"value": "120", "unit": "ms", "quantity": "p95 latency"}
        utterance = "p95 latency: 120 ms"

        assert _id("120 ms", "Metric", utterance, **kwargs) == ("120-ms-p95-latency", [])
        assert canonical_metric_id("120", "ms", "p95 latency") == "120-ms-p95-latency"
        seen = []
        monkeypatch.setattr(
            list_assembly,
            "canonical_metric_id",
            lambda *args: seen.append(args) or "from-the-builder",
        )
        assert _id("120 ms", "Metric", utterance, **kwargs) == ("from-the-builder", [])
        assert seen == [("120", "ms", "p95 latency")]

    def test_a_date_id_is_its_iso_value(self):
        assert _id("the 12th", "Date", "on the 12th", value="2026-03-12") == ("2026-03-12", [])

    def test_user_and_mist_are_fixed(self):
        assert _id("I", "User", "I use it") == ("user", [])
        assert _id("you", "MistIdentity", "you run on it") == ("mist-identity", [])

    @pytest.mark.parametrize(
        ("name", "etype", "activity", "expected"),
        [
            ("Marta's birthday dinner on the 12th", "Event", False, "birthday-dinner"),
            ("calling my parents", "Abstraction", True, "calling-parents"),
            ("the team offsite in March", "Event", False, "team-offsite"),
            ("launch party on 2026-05-01", "Event", False, "launch-party"),
            ("our standup every Monday", "Event", False, "standup"),
            ("Ostby fair on March 3rd, 2026", "Event", False, "ostby-fair"),
        ],
    )
    def test_an_event_or_activity_drops_dates_articles_and_possessors(
        self, name, etype, activity, expected
    ):
        assert _id(name, etype, name, activity=activity) == (expected, [])

    @pytest.mark.parametrize(
        "name", ["Node.js", "José", "Brill_Tallow  Kit", "Ostrel/Velquist", "--Kestrin--"]
    )
    def test_the_rails_last_step_equals_the_scorer_and_the_normalizer(self, name):
        normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator())
        rail_id, warnings = _id(name, "Technology", name)
        assert rail_id == canonical_id(name) == normalizer._canonicalize(name)
        assert warnings == []

    def test_node_js_and_jose_give_their_rail_ids(self):
        assert _id("Node.js", "Technology", "Node.js")[0] == "nodejs"
        assert _id("José", "Person", "José")[0] == "jos"


# -- names and the normalizer -----------------------------------------------------------------


def _names_turn():
    utterance = (
        "Weekend:\n- Marta's birthday dinner on the 12th at Velmont Hall\n"
        "- Ilse's review of Node.js\n- José's demo"
    )
    date = ListEntity("e1", "the 12th", "the 12th", "Date", value="2026-03-12")
    hall = ListEntity("e2", "Velmont Hall", "Velmont Hall", "Location")
    node = ListEntity("e1", "Node.js", "Node.js", "Technology")
    frame = _frame(_group("G1", ("user", "EXPERIENCED", "item"), item_type="Event"))
    dinner = ItemOutput(
        "G1.1",
        ItemEntity("Marta's birthday dinner", "Marta's birthday dinner on the 12th", "Event"),
        ItemTemplate(True, "assert", "past"),
        (date, hall),
        (_rel("item", "OCCURRED_ON", "e1"), _rel("item", "RELATED_TO", "e2")),
    )
    review = _item(
        "G1.2",
        "Ilse's",
        "Person",
        applies=False,
        entities=[node],
        relationships=[_rel("item", "USES", "e1")],
    )
    demo = _item(
        "G1.3",
        "José's",
        "Person",
        applies=False,
        relationships=[_rel("user", "KNOWS_PERSON", "item")],
    )
    return utterance, frame, [dinner, review, demo]


async def _normalized_ids(payload: ExtractionPayload):
    result = ExtractionResult(
        entities=copy.deepcopy(payload.entities),
        relationships=copy.deepcopy(payload.relationships),
    )
    normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator(), executor=None)
    result = await normalizer.normalize(result)
    triples = [(r["source"], r["type"], r["target"]) for r in result.relationships]
    return [e["id"] for e in result.entities], triples


class TestNames:
    def test_each_name_is_its_ids_phrase_after_the_drops(self):
        payload, warnings = _run(*_names_turn())

        assert payload.entities == [
            _USER,
            {"id": "birthday-dinner", "name": "birthday dinner", "type": "Event"},
            {"id": "2026-03-12", "name": "2026-03-12", "type": "Date"},
            {"id": "velmont-hall", "name": "Velmont Hall", "type": "Location"},
            {"id": "ilse", "name": "Ilse", "type": "Person"},
            {"id": "nodejs", "name": "Node.js", "type": "Technology"},
            {"id": "jos", "name": "José", "type": "Person"},
        ]
        assert warnings == []

    def test_a_metric_keeps_the_models_name(self):
        metric = ListEntity(
            "e1", "120 ms", "120 ms", "Metric", value="120", unit="ms", quantity="p95 latency"
        )
        utterance = "Ostrel:\n- p95 latency: 120 ms"
        frame = _frame(
            _group("G1"), frame_entities=[ListEntity("f1", "Ostrel", "Ostrel", "Project")]
        )
        items = [
            _item("G1.1", None, entities=[metric], relationships=[_rel("f1", "HAS_METRIC", "e1")])
        ]

        payload, _ = _run(utterance, frame, items)

        assert payload.entities[1]["id"] == "120-ms-p95-latency"
        assert payload.entities[1]["name"] == "120 ms"

    @pytest.mark.asyncio
    async def test_the_normalizer_keeps_every_assembled_id(self):
        payload, _ = _run(*_names_turn())

        ids, triples = await _normalized_ids(payload)

        assert ids == _ids(payload)
        assert triples == _triples(payload)

    @pytest.mark.asyncio
    async def test_the_normalizers_own_maps_still_apply(self):
        utterance = "Plans:\n- Postgres\n- Training for 10k"
        frame = _frame(_group("G1", ("user", "USES", "item")))
        items = [_item("G1.1", "Postgres"), _item("G1.2", "Training for 10k", "Project")]
        payload, _ = _run(utterance, frame, items)
        assert _ids(payload) == ["user", "postgres", "training-for-10k"]

        ids, _ = await _normalized_ids(payload)

        # An alias and the version pattern change the id, as they do for every payload.
        assert ids == ["user", "postgresql", "training-for"]


# -- lever rows ---------------------------------------------------------------------------------


def _f_row():
    """F: the lead-in sets the subject and predicate once for its items."""
    utterance = "Tools I use at Thrennic:\n- Quorvane\n- Ostrel"
    frame = _frame(
        _group("G1", ("user", "USES", "item"), item_type="Technology"),
        frame_entities=[ListEntity("f1", "Thrennic", "Thrennic", "Organization")],
        relationships=[_rel("user", "WORKS_AT", "f1")],
    )
    items = [_item("G1.1", "Quorvane"), _item("G1.2", "Ostrel")]
    expected = (
        ["user", "thrennic", "quorvane", "ostrel"],
        [
            ("user", "WORKS_AT", "thrennic"),
            ("user", "USES", "quorvane"),
            ("user", "USES", "ostrel"),
        ],
        [],
    )
    return (utterance, frame, items), expected


def _b_row():
    """B: a back-reference resolved to the earlier items ("both")."""
    utterance = "Editors I tried:\n- Kestrel\n- Tallow\n- Brill\nI kept both of the first two."
    frame = _frame(
        _group("G1", ("user", "USES", "item"), item_type="Technology"),
        relationships=[_rel("user", "PREFERS", "item:G1.1"), _rel("user", "PREFERS", "item:G1.2")],
    )
    items = [_item("G1.1", "Kestrel"), _item("G1.2", "Tallow"), _item("G1.3", "Brill")]
    expected = (
        ["user", "kestrel", "tallow", "brill"],
        [
            ("user", "PREFERS", "kestrel"),
            ("user", "PREFERS", "tallow"),
            ("user", "USES", "kestrel"),
            ("user", "USES", "tallow"),
            ("user", "USES", "brill"),
        ],
        ["list_item_edge_on_template_pair"] * 2,
    )
    return (utterance, frame, items), expected


def _r_row():
    """R: an id rail only: an article, a typo in the span, and the Metric quantity."""
    utterance = "Ostrel at Thrennic:\n- the Quorvane dashboard\n- p95 latency: 120 ms\n- p99 latency: 120 ms"
    frame = _frame(
        _group("G1"),
        frame_entities=[
            ListEntity("f1", "Ostrel", "Ostrel", "Project"),
            ListEntity("f2", "Thrennic", "Thrennic", "Organization"),
        ],
        relationships=[_rel("f1", "PART_OF", "f2")],
    )

    def metric(quantity: str) -> ListEntity:
        return ListEntity(
            "e1", "120 ms", "120 ms", "Metric", value="120", unit="ms", quantity=quantity
        )

    items = [
        _item(
            "G1.1",
            "the Quorvanne dashboard",
            "Project",
            applies=False,
            relationships=[_rel("f1", "DEPENDS_ON", "item")],
        ),
        _item(
            "G1.2",
            None,
            applies=False,
            entities=[metric("p95 latency")],
            relationships=[_rel("f1", "HAS_METRIC", "e1")],
        ),
        _item(
            "G1.3",
            None,
            applies=False,
            entities=[metric("p99 latency")],
            relationships=[_rel("f1", "HAS_METRIC", "e1")],
        ),
    ]
    expected = (
        ["ostrel", "thrennic", "quorvane-dashboard", "120-ms-p95-latency", "120-ms-p99-latency"],
        [
            ("ostrel", "PART_OF", "thrennic"),
            ("ostrel", "DEPENDS_ON", "quorvane-dashboard"),
            ("ostrel", "HAS_METRIC", "120-ms-p95-latency"),
            ("ostrel", "HAS_METRIC", "120-ms-p99-latency"),
        ],
        [],
    )
    return (utterance, frame, items), expected


def _m_row():
    """M: an item split: one item holds an event, a date, a place and a person."""
    utterance = "Weekend:\n- Marta's birthday dinner on the 12th at Velmont Hall"
    frame = _frame(_group("G1", ("user", "EXPERIENCED", "item"), item_type="Event"))
    dinner = ItemOutput(
        "G1.1",
        ItemEntity("Marta's birthday dinner", "Marta's birthday dinner on the 12th", "Event"),
        ItemTemplate(True, "assert", "past"),
        (
            ListEntity("e1", "the 12th", "the 12th", "Date", value="2026-03-12"),
            ListEntity("e2", "Velmont Hall", "Velmont Hall", "Location"),
            ListEntity("e3", "Marta's", "Marta", "Person"),
        ),
        (
            _rel("item", "OCCURRED_ON", "e1"),
            _rel("item", "RELATED_TO", "e2"),
            _rel("e3", "RELATED_TO", "item"),
        ),
    )
    expected = (
        ["user", "birthday-dinner", "2026-03-12", "velmont-hall", "marta"],
        [
            ("user", "EXPERIENCED", "birthday-dinner"),
            ("birthday-dinner", "OCCURRED_ON", "2026-03-12"),
            ("birthday-dinner", "RELATED_TO", "velmont-hall"),
            ("marta", "RELATED_TO", "birthday-dinner"),
        ],
        [],
    )
    return (utterance, frame, [dinner]), expected


_LEVER_ROWS = {"F": _f_row, "B": _b_row, "R": _r_row, "M": _m_row}


class TestLeverRows:
    @pytest.mark.parametrize("row", sorted(_LEVER_ROWS))
    def test_a_turn_shaped_like_the_row_assembles_to_the_expected_payload(self, row):
        turn, (ids, triples, warnings) = _LEVER_ROWS[row]()

        payload, got = _run(*turn)

        assert (_ids(payload), _triples(payload), got) == (ids, triples, warnings)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("row", sorted(_LEVER_ROWS))
    async def test_the_normalizer_keeps_every_id_of_the_row(self, row):
        turn, (ids, triples, _) = _LEVER_ROWS[row]()
        payload, _ = _run(*turn)

        assert await _normalized_ids(payload) == (ids, triples)


class TestValueOnlyMetric:
    """A Metric with a value but no unit or quantity, which Pass 4 does not rebuild."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("span", "expected"),
        [("180 queued jobs", "180-queued-jobs"), ("queue of 180 jobs", "180-queue-of-jobs")],
    )
    async def test_its_id_survives_the_normalizer_and_matches_the_scorer(self, span, expected):
        utterance = f"Ostrel:\n- {span}"
        metric = ListEntity("e1", span, "queue depth", "Metric", value="180")
        frame = _frame(
            _group("G1"), frame_entities=[ListEntity("f1", "Ostrel", "Ostrel", "Project")]
        )
        items = [
            _item(
                "G1.1",
                None,
                applies=False,
                entities=[metric],
                relationships=[_rel("f1", "HAS_METRIC", "e1")],
            )
        ]

        payload, warnings = _run(utterance, frame, items)

        record = payload.entities[1]
        assert record == {
            "id": expected,
            "name": span,
            "type": "Metric",
            "properties": {"value": "180", "unit": None, "quantity": None},
        }
        assert warnings == []
        assert await _normalized_ids(payload) == (
            ["ostrel", expected],
            [("ostrel", "HAS_METRIC", expected)],
        )
        assert _produced_entity_match_id(expected, "Metric", record["properties"]) == expected


# -- Metric items -------------------------------------------------------------------------------


def _metric_item(
    item_id: str,
    span: str,
    name: str,
    *,
    value: str | None = None,
    unit: str | None = None,
    quantity: str | None = None,
) -> ItemOutput:
    return ItemOutput(
        item_id,
        ItemEntity(span, name, "Metric", value=value, unit=unit, quantity=quantity),
        ItemTemplate(True, "assert", "current"),
        (),
        (),
    )


def _metric_items_turn():
    """A durable group whose items are Metrics of the frame's project, one per row."""
    utterance = "Ostrel numbers:\n- p95 latency: 120 ms\n- p99 latency: 120 ms\n- 3 deploys a week"
    frame = _frame(
        _group("G1", ("f1", "HAS_METRIC", "item"), item_type="Metric"),
        frame_entities=[ListEntity("f1", "Ostrel", "Ostrel", "Project")],
    )
    items = [
        _metric_item(
            "G1.1", "p95 latency: 120 ms", "120 ms", value="120", unit="ms", quantity="p95 latency"
        ),
        _metric_item(
            "G1.2", "p99 latency: 120 ms", "120 ms", value="120", unit="ms", quantity="p99 latency"
        ),
        _metric_item("G1.3", "3 deploys a week", "deploys", value="3", quantity="deploys a week"),
    ]
    return utterance, frame, items


class TestMetricItems:
    """An item whose own entity is a Metric takes the Metric id rail, as a keyed Metric does."""

    def test_its_id_is_built_from_its_value_unit_and_quantity(self):
        payload, warnings = _run(*_metric_items_turn())

        assert payload.entities[1:] == [
            {
                "id": "120-ms-p95-latency",
                "name": "120 ms",
                "type": "Metric",
                "properties": {"value": "120", "unit": "ms", "quantity": "p95 latency"},
            },
            {
                "id": "120-ms-p99-latency",
                "name": "120 ms",
                "type": "Metric",
                "properties": {"value": "120", "unit": "ms", "quantity": "p99 latency"},
            },
            {
                "id": canonical_metric_id("3", None, "deploys a week"),
                "name": "deploys",
                "type": "Metric",
                "properties": {"value": "3", "unit": None, "quantity": "deploys a week"},
            },
        ]
        assert _triples(payload) == [
            ("ostrel", "HAS_METRIC", "120-ms-p95-latency"),
            ("ostrel", "HAS_METRIC", "120-ms-p99-latency"),
            ("ostrel", "HAS_METRIC", canonical_metric_id("3", None, "deploys a week")),
        ]
        assert warnings == []

    def test_its_id_comes_from_canonical_metric_id(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            list_assembly,
            "canonical_metric_id",
            lambda *args: seen.append(args) or f"built-{len(seen)}",
        )

        payload, _ = _run(*_metric_items_turn())

        assert _ids(payload) == ["ostrel", "built-1", "built-2", "built-3"]
        assert seen == [
            ("120", "ms", "p95 latency"),
            ("120", "ms", "p99 latency"),
            ("3", None, "deploys a week"),
        ]

    @pytest.mark.parametrize(
        ("span", "value", "expected"),
        [
            # No value: the span rail, then `canonical_metric_id_from_id`.
            ("fast cold starts", None, None),
            # A value alone, which Pass 4 does not rebuild: the span rail as well.
            ("180 queued jobs", "180", "180-queued-jobs"),
        ],
    )
    def test_with_no_value_or_a_value_alone_it_keeps_the_span_rail(self, span, value, expected):
        utterance = f"Ostrel:\n- {span}"
        frame = _frame(
            _group("G1", ("f1", "HAS_METRIC", "item"), item_type="Metric"),
            frame_entities=[ListEntity("f1", "Ostrel", "Ostrel", "Project")],
        )
        items = [_metric_item("G1.1", span, "queue depth", value=value)]

        payload, warnings = _run(utterance, frame, items)

        rail_id, rail_warnings = list_entity_id(
            span, "queue depth", "Metric", utterance, value=value
        )
        assert expected is None or rail_id == expected
        assert payload.entities[1] == {
            "id": rail_id,
            "name": span,
            "type": "Metric",
            "properties": {"value": value, "unit": None, "quantity": None},
        }
        assert warnings == rail_warnings == []

    @pytest.mark.asyncio
    async def test_the_normalizer_keeps_every_assembled_metric_item_id(self):
        payload, _ = _run(*_metric_items_turn())

        ids, triples = await _normalized_ids(payload)

        assert ids == _ids(payload)
        assert triples == _triples(payload)

    def test_the_same_inputs_give_the_same_payload_and_are_not_changed(self):
        utterance, frame, items = _metric_items_turn()
        before = copy.deepcopy((frame, items))

        first = _run(utterance, frame, items)
        second = _run(utterance, frame, items)

        assert first == second
        assert (frame, items) == before


# -- durable groups -----------------------------------------------------------------------------


class TestDurableGroupWithNoTemplate:
    """A durable group with no template: no fact is shared, and each item states its own."""

    def test_each_applying_item_keeps_its_own_edges(self):
        utterance = "Service latencies:\n- Quorvane: 120 ms\n- Ostrel: 80 ms"
        group = GroupFrame("G1", (), "Technology", "plan", durable=True)
        frame = _frame(group)

        def row(item_id: str, name: str, value: str) -> ItemOutput:
            metric = ListEntity(
                "e1", f"{value} ms", f"{value} ms", "Metric", value=value, unit="ms"
            )
            return _item(
                item_id,
                name,
                entities=[metric],
                relationships=[_rel("item", "HAS_METRIC", "e1")],
            )

        items = [row("G1.1", "Quorvane", "120"), row("G1.2", "Ostrel", "80")]

        payload, warnings = _run(utterance, frame, items)

        assert group.durable and group.templates == ()
        assert _ids(payload) == [
            "quorvane",
            canonical_metric_id("120", "ms", None),
            "ostrel",
            canonical_metric_id("80", "ms", None),
        ]
        assert payload.relationships == [
            _edge("quorvane", "HAS_METRIC", canonical_metric_id("120", "ms", None), {}),
            _edge("ostrel", "HAS_METRIC", canonical_metric_id("80", "ms", None), {}),
        ]
        assert warnings == []
