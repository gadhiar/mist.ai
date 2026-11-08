"""List frames: how a structured turn's units group, render and parse for the list path.

The list path (`backend.extraction_service.engine`, T19) sends a structured turn to one
frame call and a few item calls. These tests pin the grouping those calls see, the render
they read, and the strict parsers that turn their outputs into frames and item records.
Every input is synthetic, with invented names.
"""

from __future__ import annotations

import copy
import json

import pytest

from backend.errors import ExtractionValidationError
from backend.extraction_service.list_schemas import EXTRACTABLE_TYPE_ENUM, PREDICATE_ENUM, ItemSlot
from backend.knowledge.extraction.list_frames import (
    FrameOutput,
    GroupFrame,
    ItemEntity,
    ItemOutput,
    ItemTemplate,
    ListEntity,
    ListGroup,
    ListOutputError,
    ListRelationship,
    Template,
    group_units,
    item_slots,
    parse_frame_output,
    parse_item_output,
    render_frame_input,
)
from backend.knowledge.extraction.list_prompts import LIST_SYSTEM_PROMPT, render_frames_for_items
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.structure_splitter import InputUnit, split_structured
from backend.knowledge.extraction.validator import ExtractionValidator


def _item(text: str, *context: str) -> InputUnit:
    return InputUnit(text=text, context=tuple(context), kind="list_item")


def _texts(group: ListGroup) -> list[str]:
    return [u.text for u in group.items]


# --- grouping ---------------------------------------------------------------------------------


def test_a_flat_list_under_a_lead_in_is_one_group_with_numbered_items():
    units = split_structured("My kit:\n- Quorvane for notes\n- Brellix\n- Tamsic")

    groups, prose = group_units(units)

    assert prose == ()
    assert [g.gid for g in groups] == ["G1"]
    assert groups[0].context == ("My kit",)
    assert _texts(groups[0]) == ["Quorvane for notes", "Brellix", "Tamsic"]
    assert groups[0].item_ids == ("G1.1", "G1.2", "G1.3")
    assert groups[0].parent is None


def test_a_nested_list_makes_a_child_group_linked_to_its_parent_item():
    units = split_structured("Stack:\n- Quorvane\n  - Brellix plugin\n  - Tamsic plugin\n- Ostrel")

    groups, _ = group_units(units)

    assert [(g.gid, g.context, _texts(g)) for g in groups] == [
        ("G1", ("Stack",), ["Quorvane", "Ostrel"]),
        ("G2", ("Stack", "Quorvane"), ["Brellix plugin", "Tamsic plugin"]),
    ]
    assert groups[1].parent == ("G1", "G1.1")


def test_a_sibling_after_a_nested_child_stays_in_its_group():
    units = split_structured("- Quorvane\n  - Brellix\n- Ostrel\n  - Tamsic\n- Pindle")

    groups, _ = group_units(units)

    assert [(g.gid, _texts(g), g.parent) for g in groups] == [
        ("G1", ["Quorvane", "Ostrel", "Pindle"], None),
        ("G2", ["Brellix"], ("G1", "G1.1")),
        ("G3", ["Tamsic"], ("G1", "G1.2")),
    ]


def test_a_parent_item_ending_in_a_colon_is_found():
    units = split_structured("Servers:\n- Marrowgate:\n  - Velquist in front of it")

    groups, _ = group_units(units)

    assert groups[1].context == ("Servers", "Marrowgate")
    assert groups[1].parent == ("G1", "G1.1")


def test_a_checked_parent_item_is_found():
    units = split_structured("- [x] Quorvane:\n  - [ ] Brellix")

    groups, _ = group_units(units)

    assert groups[1].parent == ("G1", "G1.1")


def test_checkbox_items_keep_their_marks_in_one_group():
    units = split_structured("Goals this year:\n- [x] learn to weld\n- [ ] run a half marathon")

    groups, _ = group_units(units)

    assert _texts(groups[0]) == ["[x] learn to weld", "[ ] run a half marathon"]


def test_table_rows_under_a_lead_in_are_one_group():
    text = "Team:\n| Name | Role |\n|---|---|\n| Iselde | lead |\n| Tomasz | tester |"

    groups, prose = group_units(split_structured(text))

    assert prose == ()
    assert [(g.gid, g.context, _texts(g)) for g in groups] == [
        ("G1", ("Team",), ["Name: Iselde; Role: lead", "Name: Tomasz; Role: tester"]),
    ]


def test_prose_between_two_lists_ends_the_run_and_is_returned_in_order():
    text = (
        "We moved offices last week.\n\n"
        "- Quorvane\n- Brellix\n\n"
        "Then the second list.\n\n"
        "- Ostrel\n\n"
        "That was all."
    )

    groups, prose = group_units(split_structured(text))

    assert [(g.gid, _texts(g)) for g in groups] == [
        ("G1", ["Quorvane", "Brellix"]),
        ("G2", ["Ostrel"]),
    ]
    assert [p.text for p in prose] == [
        "We moved offices last week.",
        "Then the second list.",
        "That was all.",
    ]


def test_items_under_two_headings_are_two_groups():
    text = "# Home\n- Quorvane\n# Work\n- Brellix\n- Ostrel"

    groups, _ = group_units(split_structured(text))

    assert [(g.context, _texts(g)) for g in groups] == [
        (("Home",), ["Quorvane"]),
        (("Work",), ["Brellix", "Ostrel"]),
    ]


def test_hand_built_units_group_by_context_and_nesting():
    units = (
        _item("Quorvane", "Tools"),
        _item("Brellix", "Tools", "Quorvane"),
        _item("Ostrel", "Tools"),
        InputUnit(text="Nothing else.", context=(), kind="prose"),
        _item("Pindle", "Tools"),
    )

    groups, prose = group_units(units)

    assert [(g.gid, _texts(g)) for g in groups] == [
        ("G1", ["Quorvane", "Ostrel"]),
        ("G2", ["Brellix"]),
        ("G3", ["Pindle"]),
    ]
    assert [p.text for p in prose] == ["Nothing else."]


def test_an_all_prose_turn_has_no_groups():
    groups, prose = group_units(split_structured("I like plain sentences."))

    assert groups == ()
    assert [p.text for p in prose] == ["I like plain sentences."]


# --- render -----------------------------------------------------------------------------------


def test_the_render_lists_prose_then_each_group_with_its_context_once():
    text = (
        "We should finish the move this week.\n\n"
        "Things I run at work:\n"
        "- Quorvane for editing\n"
        "- Marrowgate:\n"
        "  - Velquist in front of it"
    )
    groups, prose = group_units(split_structured(text))

    assert render_frame_input(groups, prose) == (
        "P1 We should finish the move this week.\n"
        'G1 under "Things I run at work"\n'
        "  G1.1 Quorvane for editing\n"
        "  G1.2 Marrowgate:\n"
        'G2 under "Things I run at work" > "Marrowgate" (parent G1.2)\n'
        "  G2.1 Velquist in front of it"
    )


def test_a_group_without_context_and_prose_under_a_heading_render_plainly():
    units = (
        InputUnit(text="Quick note\non two lines", context=("Log",), kind="prose"),
        _item("Quorvane"),
    )
    groups, prose = group_units(units)

    assert render_frame_input(groups, prose) == (
        'P1 under "Log": Quick note on two lines\nG1\n  G1.1 Quorvane'
    )


def test_the_render_is_deterministic():
    text = "Stack:\n- Quorvane\n  - Brellix\n- Ostrel\n\nDone."
    first = render_frame_input(*group_units(split_structured(text)))

    assert all(render_frame_input(*group_units(split_structured(text))) == first for _ in range(5))


# --- strict parsers ---------------------------------------------------------------------------

# The turn `_FRAME` and `_ITEMS` answer: G1 holds G1.1 and G1.2, G2 holds G2.1.
_TURN = "Tools at Thrennic:\n- Quorvane\n- Brellix\n\nPeople:\n- Iselde"
_GROUPS = group_units(split_structured(_TURN))[0]

_FRAME = {
    "frame_entities": [
        {"key": "f1", "span": "Thrennic", "name": "Thrennic", "type": "Organization"}
    ],
    "groups": [
        {
            "group": "G1",
            "durable": True,
            "templates": [{"subject": "user", "predicate": "USES", "object": "item"}],
            "item_type": "Technology",
            "unchecked": "plan",
        },
        {
            "group": "G2",
            "durable": True,
            "templates": [{"subject": "item", "predicate": "MEMBER_OF", "object": "f1"}],
            "item_type": "Person",
            "unchecked": "goal",
        },
    ],
    "facts": {
        "entities": [
            {
                "key": "e1",
                "span": "40 ms",
                "name": "40 ms",
                "type": "Metric",
                "value": "40",
                "unit": "ms",
                "quantity": "lag",
            }
        ],
        "relationships": [
            {"source": "user", "target": "f1", "type": "WORKS_AT"},
            {
                "source": "item:G1.1",
                "target": "e1",
                "type": "HAS_METRIC",
                "properties": {"confidence": 0.9, "temporal_status": "current"},
            },
        ],
    },
}

_ITEMS = {
    "items": [
        {
            "item": "G1.1",
            "template": {"applies": True, "assertion_kind": "assert", "temporal_status": "current"},
            "entity": {"span": "Quorvane", "name": "Quorvane", "type": "Technology"},
            "entities": [],
            "relationships": [],
        },
        {
            "item": "G1.2",
            "template": {
                "applies": False,
                "assertion_kind": "cease",
                "temporal_status": "past",
                "end_date": "2031-01-05",
            },
            "entity": None,
            "entities": [
                {
                    "key": "e1",
                    "span": "rowing",
                    "name": "rowing",
                    "type": "Concept",
                    "activity": True,
                }
            ],
            "relationships": [{"source": "user", "target": "e1", "type": "HAS_HABIT"}],
        },
    ]
}


def _frame(**changes: object) -> str:
    frame = copy.deepcopy(_FRAME)
    frame.update(changes)
    return json.dumps(frame)


def _parse_frame(text: str) -> FrameOutput:
    return parse_frame_output(text, _GROUPS)


def _parse_items(text: str, expected: list[str]) -> tuple[ItemOutput, ...]:
    return parse_item_output(text, expected, _parse_frame(json.dumps(_FRAME)), _GROUPS, {})


def test_list_output_error_is_an_extraction_validation_error():
    assert issubclass(ListOutputError, ExtractionValidationError)


def test_a_well_formed_frame_parses_into_frames_in_the_turns_group_order():
    frame = _parse_frame(_frame(groups=_FRAME["groups"][::-1]))

    assert frame == FrameOutput(
        frame_entities=(ListEntity("f1", "Thrennic", "Thrennic", "Organization"),),
        groups=(
            GroupFrame("G1", (Template("user", "USES", "item"),), "Technology", "plan"),
            GroupFrame("G2", (Template("item", "MEMBER_OF", "f1"),), "Person", "goal"),
        ),
        facts=frame.facts,
    )
    assert frame.facts.entities == (
        ListEntity("e1", "40 ms", "40 ms", "Metric", value="40", unit="ms", quantity="lag"),
    )
    assert frame.facts.relationships == (
        ListRelationship("user", "f1", "WORKS_AT"),
        ListRelationship(
            "item:G1.1", "e1", "HAS_METRIC", {"confidence": 0.9, "temporal_status": "current"}
        ),
    )


def test_a_frame_in_a_code_fence_parses():
    assert _parse_frame(f"```json\n{_frame()}\n```").groups[0].group == "G1"


def test_a_group_frame_built_without_durable_is_durable():
    assert GroupFrame("G1", (), None, "plan").durable is True


def _second_group(**changes: object) -> str:
    group = {"group": "G2", "durable": True, "templates": [], "item_type": None}
    return _with(["groups", 1], group | {"unchecked": "plan"} | changes)


def test_a_group_that_states_nothing_durable_parses_with_no_templates():
    frame = _parse_frame(_second_group(durable=False))

    assert frame.groups[1] == GroupFrame("G2", (), None, "plan", durable=False)


def test_a_durable_group_with_no_shared_template_parses():
    frame = _parse_frame(_second_group())

    assert frame.groups[1] == GroupFrame("G2", (), None, "plan", durable=True)


def test_a_group_that_is_not_durable_but_has_templates_is_refused():
    with pytest.raises(ListOutputError, match=r"groups\[0\].*G1.*durable"):
        _parse_frame(_with(["groups", 0, "durable"], False))


def _with(path: list, value: object) -> str:
    frame = copy.deepcopy(_FRAME)
    target = frame
    for step in path[:-1]:
        target = target[step]
    if value is _DROP:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return json.dumps(frame)


_DROP = object()
_TEMPLATE = {"subject": "user", "predicate": "USES", "object": "item"}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("not json", id="not-json"),
        pytest.param("[]", id="not-an-object"),
        pytest.param(_with(["facts"], _DROP), id="missing-facts"),
        pytest.param(_with(["groups", 0, "unchecked"], _DROP), id="missing-unchecked"),
        pytest.param(_with(["groups", 0, "item_type"], _DROP), id="missing-item-type"),
        pytest.param(_with(["groups", 0, "durable"], _DROP), id="missing-durable"),
        pytest.param(_with(["groups", 0, "durable"], "yes"), id="durable-a-string"),
        pytest.param(_with(["frame_entities", 0, "span"], _DROP), id="missing-entity-span"),
        pytest.param(_with(["groups"], "G1"), id="groups-not-a-list"),
        pytest.param(_with(["groups", 0, "templates"], "user USES item"), id="templates-a-string"),
        pytest.param(_with(["facts", "entities", 0, "value"], 40), id="value-a-number"),
        pytest.param(_with(["groups", 0, "templates"], [_TEMPLATE] * 4), id="four-templates"),
        pytest.param(_with(["groups", 0, "group"], "G3"), id="unknown-group"),
        pytest.param(_with(["groups", 1], _DROP), id="missing-group"),
        pytest.param(_with(["groups", 1, "group"], "G1"), id="repeated-group"),
        pytest.param(_with(["groups", 0, "item_type"], "Gadget"), id="unknown-type"),
        pytest.param(_with(["groups", 0, "unchecked"], "maybe"), id="unknown-unchecked"),
        pytest.param(_with(["groups", 0, "templates", 0, "predicate"], "OWNS"), id="unknown-pred"),
        pytest.param(_with(["groups", 0, "templates", 0, "subject"], "f9"), id="unknown-role"),
        pytest.param(_with(["groups", 0, "extra"], 1), id="extra-key"),
        pytest.param(_with(["facts", "relationships", 0, "type"], "OWNS"), id="unknown-edge"),
        pytest.param(
            _with(["facts", "relationships", 1, "properties", "negated"], "no"),
            id="property-wrong-type",
        ),
        pytest.param(_with(["facts", "entities", 0, "key"], "f1"), id="key-reused"),
    ],
)
def test_the_frame_parser_refuses_a_malformed_output(text):
    with pytest.raises(ListOutputError):
        _parse_frame(text)


def test_a_well_formed_item_output_parses_in_the_order_asked_for():
    items = _parse_items(json.dumps(_ITEMS), ["G1.2", "G1.1"])

    assert [i.item for i in items] == ["G1.2", "G1.1"]
    assert items[1] == ItemOutput(
        item="G1.1",
        entity=ItemEntity("Quorvane", "Quorvane", "Technology"),
        template=ItemTemplate(True, "assert", "current"),
        entities=(),
        relationships=(),
    )
    assert items[0].entity is None
    assert items[0].template == ItemTemplate(False, "cease", "past", end_date="2031-01-05")
    assert items[0].entities == (ListEntity("e1", "rowing", "rowing", "Concept", activity=True),)
    assert items[0].relationships == (ListRelationship("user", "e1", "HAS_HABIT"),)


def _items_with(path: list, value: object) -> str:
    items = copy.deepcopy(_ITEMS)
    target = items
    for step in path[:-1]:
        target = target[step]
    if value is _DROP:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return json.dumps(items)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("{}", id="missing-items"),
        pytest.param(_items_with(["items", 0, "template"], _DROP), id="missing-template"),
        pytest.param(_items_with(["items", 0, "entity"], _DROP), id="missing-entity"),
        pytest.param(_items_with(["items", 0, "template", "applies"], "yes"), id="applies-str"),
        pytest.param(_items_with(["items", 0, "entity"], "Quorvane"), id="entity-a-string"),
        pytest.param(_items_with(["items", 0, "entities"], {}), id="entities-not-a-list"),
        pytest.param(_items_with(["items", 0, "item"], "G9.1"), id="unknown-item"),
        pytest.param(_items_with(["items", 1], _DROP), id="missing-item"),
        pytest.param(_items_with(["items", 1, "item"], "G1.1"), id="repeated-item"),
        pytest.param(_items_with(["items", 0, "entity", "type"], "Gadget"), id="unknown-type"),
        pytest.param(
            _items_with(["items", 0, "template", "temporal_status"], "soon"), id="unknown-status"
        ),
        pytest.param(_items_with(["items", 1, "entities", 0, "key"], "item"), id="key-a-role"),
        pytest.param(_items_with(["items", 0, "entity", "value"], 40), id="value-a-number"),
        pytest.param(_items_with(["items", 0, "entity", "unit"], ["ms"]), id="unit-a-list"),
        pytest.param(_items_with(["items", 0, "entity", "extra"], "x"), id="entity-extra-key"),
    ],
)
def test_the_item_parser_refuses_a_malformed_output(text):
    with pytest.raises(ListOutputError):
        _parse_items(text, ["G1.1", "G1.2"])


def test_an_item_entity_carries_a_metrics_value_unit_and_quantity():
    metric = {"span": "40 ms", "name": "40 ms", "type": "Metric"}
    metric |= {"value": "40", "unit": "ms", "quantity": "lag"}
    text = _items_with(["items", 0], copy.deepcopy(_ITEMS["items"][0]) | {"entity": metric})
    text = text.replace('"applies": true', '"applies": false')

    items = _parse_items(text, ["G1.1", "G1.2"])

    assert items[0].entity == ItemEntity(
        "40 ms", "40 ms", "Metric", value="40", unit="ms", quantity="lag"
    )


def test_an_item_entity_without_metric_keys_leaves_them_none():
    entity = _parse_items(json.dumps(_ITEMS), ["G1.1", "G1.2"])[0].entity

    assert entity is not None
    assert (entity.value, entity.unit, entity.quantity) == (None, None, None)


# --- endpoints resolved at parse time ---------------------------------------------------------

# A nested turn: G1 holds G1.1 Quorvane and G1.2 Ostrel; G2 holds G2.1 Brellix, under G1.1.
_NESTED_TURN = "Stack at Thrennic:\n- Quorvane:\n  - Brellix\n- Ostrel"
_NESTED = group_units(split_structured(_NESTED_TURN))[0]
_THRENNIC = {"key": "f1", "span": "Thrennic", "name": "Thrennic", "type": "Organization"}
_CONCEPT = {"key": "e1", "span": "notes", "name": "notes", "type": "Concept"}


def _group_out(gid: str, *templates: str, item_type: str | None = "Technology", **more) -> dict:
    """A group frame; each template is written `subject PREDICATE object`."""
    group: dict = {"group": gid, "durable": True}
    group["templates"] = [
        dict(zip(("subject", "predicate", "object"), t.split())) for t in templates
    ]
    return group | {"item_type": item_type, "unchecked": "plan"} | more


def _edge(source: str, target: str, type_: str = "RELATED_TO") -> dict:
    return {"source": source, "target": target, "type": type_}


def _nested_frame(
    g1: tuple[str, ...] = ("user USES item",),
    g2: tuple[str, ...] = ("item DEPENDS_ON parent",),
    *,
    frame_entities: tuple[dict, ...] = (_THRENNIC,),
    entities: tuple[dict, ...] = (),
    relationships: tuple[dict, ...] = (),
    **group_changes: dict,
) -> str:
    groups = [_group_out("G1", *g1), _group_out("G2", *g2)]
    for n, gid in enumerate(("G1", "G2")):
        groups[n] |= group_changes.get(gid, {})
    facts = {"entities": list(entities), "relationships": list(relationships)}
    return json.dumps({"frame_entities": list(frame_entities), "groups": groups, "facts": facts})


def _nested_parse(text: str) -> FrameOutput:
    return parse_frame_output(text, _NESTED)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(
            _nested_frame(frame_entities=({**_THRENNIC, "key": "e1"},)),
            r"frame_entities\[0\]\.key: 'e1' is not a frame key, f1 to f20",
            id="F2-frame-key-of-the-facts-set",
        ),
        pytest.param(
            _nested_frame(entities=({**_CONCEPT, "key": "f2"},)),
            r"facts\.entities\[0\]\.key: 'f2' is not an entity key, e1 to e20",
            id="F2-facts-key-of-the-frame-set",
        ),
        pytest.param(
            _nested_frame(("e1 USES item",), entities=(_CONCEPT,)),
            r"groups\[0\]\.templates\[0\]\.subject \(G1\): 'e1' is not",
            id="F1-template-names-a-facts-key",
        ),
        pytest.param(
            _nested_frame(("user USES Quorvane",)),
            r"groups\[0\]\.templates\[0\]\.object \(G1\): 'Quorvane' is not",
            id="U7-template-names-a-name",
        ),
        pytest.param(
            _nested_frame(("user USES item", "item DEPENDS_ON parent")),
            r"groups\[0\]\.templates\[1\]\.object \(G1\): G1 is a top-level group",
            id="U4-parent-in-a-top-level-group",
        ),
        pytest.param(
            _nested_frame(relationships=(_edge("item", "f1", "WORKS_AT"),)),
            r"facts\.relationships\[0\]\.source: 'item' is not",
            id="U1-fact-names-item",
        ),
        pytest.param(
            _nested_frame(relationships=(_edge("user", "parent"),)),
            r"facts\.relationships\[0\]\.target: 'parent' is not",
            id="U1-fact-names-parent",
        ),
        pytest.param(
            _nested_frame(relationships=(_edge("user", "item:G3.1"),)),
            r"facts\.relationships\[0\]\.target: 'item:G3\.1' is not",
            id="U3-fact-names-an-item-outside-the-turn",
        ),
        pytest.param(
            _nested_frame(relationships=(_edge("user", "Thrennic", "WORKS_AT"),)),
            r"facts\.relationships\[0\]\.target: 'Thrennic' is not",
            id="U7-fact-names-a-name",
        ),
        pytest.param(
            _nested_frame(relationships=(_edge("user", "f2", "WORKS_AT"),)),
            r"facts\.relationships\[0\]\.target: 'f2' is not",
            id="U7-fact-names-an-undeclared-key",
        ),
        pytest.param(
            _nested_frame(G1={"item_type": None}),
            r"groups\[0\] \(G1\): a template names item, so item_type must be set",
            id="item-type-null-under-an-item-template",
        ),
    ],
)
def test_the_frame_parser_refuses_an_end_that_does_not_resolve(text, message):
    with pytest.raises(ListOutputError, match=message):
        _nested_parse(text)


def test_the_frame_parser_resolves_every_kind_of_end():
    text = _nested_frame(
        ("user USES item",),
        ("item DEPENDS_ON parent", "mist USES item"),
        entities=(_CONCEPT,),
        relationships=(
            _edge("user", "f1", "WORKS_AT"),
            _edge("mist", "e1"),
            _edge("item:G1.2", "item:G2.1", "DEPENDS_ON"),
        ),
    )

    frame = _nested_parse(text)

    assert [len(g.templates) for g in frame.groups] == [1, 2]
    assert len(frame.facts.relationships) == 3


def test_a_null_item_type_is_kept_when_no_template_names_item():
    frame = _nested_parse(_nested_frame(("user WORKS_AT f1",), G1={"item_type": None}))

    assert frame.groups[0].item_type is None


def _item_out(item_id: str, entity: str | None = "Technology", **changes: object) -> dict:
    """An item record whose entity, unless null, is named after its id with that type."""
    record: dict = {"item": item_id}
    record["template"] = {
        "applies": False,
        "assertion_kind": "assert",
        "temporal_status": "current",
    }
    record["entity"] = (
        None if entity is None else {"span": item_id, "name": item_id, "type": entity}
    )
    return record | {"entities": [], "relationships": []} | changes


def _known(item_id: str, entity: str | None = "Technology") -> ItemOutput:
    """An item output of an earlier batch."""
    own = None if entity is None else ItemEntity(item_id, item_id, entity)
    return ItemOutput(item_id, own, ItemTemplate(False, "assert", "current"), (), ())


def _nested_items(
    *records: dict, earlier: tuple[ItemOutput, ...] = (), frame: str | None = None
) -> tuple[ItemOutput, ...]:
    return parse_item_output(
        json.dumps({"items": list(records)}),
        [r["item"] for r in records],
        _nested_parse(frame or _nested_frame()),
        _NESTED,
        {o.item: o for o in earlier},
    )


# G2 shares no template that names parent, so G1.1's entity is not required in its own call.
_NO_PARENT_TEMPLATE = _nested_frame(g2=("user USES item",))


@pytest.mark.parametrize(
    ("records", "earlier", "message"),
    [
        pytest.param(
            [_item_out("G1.2", entities=[{**_CONCEPT, "key": "f2"}])],
            (),
            r"items\[0\] \(G1\.2\)\.entities\[0\]\.key: 'f2' is not an entity key, e1 to e20",
            id="own-key-of-the-frame-set",
        ),
        pytest.param(
            [_item_out("G1.2", relationships=[_edge("item", "Thrennic", "WORKS_AT")])],
            (),
            r"items\[0\] \(G1\.2\)\.relationships\[0\]\.target: 'Thrennic' is not",
            id="U7-a-name",
        ),
        pytest.param(
            [_item_out("G1.2", relationships=[_edge("item", "e1")])],
            (),
            r"items\[0\] \(G1\.2\)\.relationships\[0\]\.target: 'e1' is not",
            id="U7-an-undeclared-own-key",
        ),
        pytest.param(
            [_item_out("G1.2", relationships=[_edge("item", "item:G4.2")])],
            (),
            r"items\[0\] \(G1\.2\)\.relationships\[0\]\.target: 'item:G4\.2' is not",
            id="U3-an-item-outside-the-turn",
        ),
        pytest.param(
            [_item_out("G1.2", None, relationships=[_edge("user", "item", "USES")])],
            (),
            r"items\[0\] \(G1\.2\)\.relationships\[0\]\.target: item has no entity",
            id="U2-item-with-a-null-entity",
        ),
        pytest.param(
            [_item_out("G1.2", relationships=[_edge("item", "parent", "DEPENDS_ON")])],
            (),
            r"items\[0\] \(G1\.2\)\.relationships\[0\]\.target: G1\.2 is in a top-level group",
            id="U4-parent-in-a-top-level-group",
        ),
        pytest.param(
            [_item_out("G2.1", relationships=[_edge("item", "parent", "DEPENDS_ON")])],
            ("G1.1",),
            r"items\[0\] \(G2\.1\)\.relationships\[0\]\.target: parent G1\.1 has no entity",
            id="U5-parent-of-an-earlier-batch-with-a-null-entity",
        ),
        pytest.param(
            [
                _item_out("G1.1", None),
                _item_out("G2.1", relationships=[_edge("item", "parent", "DEPENDS_ON")]),
            ],
            (),
            r"items\[1\] \(G2\.1\)\.relationships\[0\]\.target: parent G1\.1 has no entity",
            id="U5-parent-in-the-batch-with-a-null-entity",
        ),
        pytest.param(
            [_item_out("G2.1", relationships=[_edge("item", "item:G1.2", "DEPENDS_ON")])],
            ("G1.2",),
            r"items\[0\] \(G2\.1\)\.relationships\[0\]\.target: item:G1\.2 has no entity",
            id="U6-item-of-an-earlier-batch-with-a-null-entity",
        ),
        pytest.param(
            [
                _item_out("G1.2", None),
                _item_out("G2.1", relationships=[_edge("item:G1.2", "item", "DEPENDS_ON")]),
            ],
            (),
            r"items\[1\] \(G2\.1\)\.relationships\[0\]\.source: item:G1\.2 has no entity",
            id="U6-item-in-the-batch-with-a-null-entity",
        ),
    ],
)
def test_the_item_parser_refuses_an_edge_end_that_does_not_resolve(records, earlier, message):
    known = tuple(_known(item_id, None) for item_id in earlier)

    with pytest.raises(ListOutputError, match=message):
        _nested_items(*records, earlier=known, frame=_NO_PARENT_TEMPLATE)


def test_the_item_parser_resolves_every_kind_of_end():
    own = [{**_CONCEPT, "key": "e1"}]
    edges = [
        _edge("user", "item", "USES"),
        _edge("mist", "e1"),
        _edge("item", "f1"),
        _edge("item", "parent", "DEPENDS_ON"),
        _edge("item", "item:G1.2", "DEPENDS_ON"),
        _edge("item", "item:G1.1", "DEPENDS_ON"),
    ]
    earlier = (_known("G1.1"),)

    (item,) = _nested_items(_item_out("G2.1", entities=own, relationships=edges), earlier=earlier)

    assert len(item.relationships) == 6


def test_an_edge_to_an_item_not_yet_parsed_is_left_to_assembly():
    """A later batch's item may still name its entity, so its call is not refused here."""
    (item,) = _nested_items(_item_out("G1.2", relationships=[_edge("item", "item:G2.1")]))

    assert item.relationships[0].target == "item:G2.1"


# --- template types against the relationship constraints ---------------------------------------

_KIT = group_units(split_structured("Kit:\n- Quorvane\n- Brellix"))[0]
_GOALS_TURN = "Goals:\n- [ ] learn Kestrin\n- [x] visit Ostby"
# G1.1 is an unchecked box and the parent of G2, so two sources limit its entity.
_GOAL_PARENT_TURN = "Goals:\n- [ ] learn Kestrin:\n  - Brellix"
_TECH = {"key": "f1", "span": "Quorvane", "name": "Quorvane", "type": "Technology"}


def _frame_text(*groups: dict, frame_entities=(), relationships=()) -> str:
    facts = {"entities": [], "relationships": list(relationships)}
    return json.dumps(
        {"frame_entities": list(frame_entities), "groups": list(groups), "facts": facts}
    )


@pytest.mark.parametrize(
    ("group", "frame_entities", "message"),
    [
        pytest.param(
            _group_out("G1", "user EXPERIENCED item", item_type="Document"),
            (),
            r"groups\[0\]\.templates\[0\] \(G1\): EXPERIENCED does not admit Document at "
            r"object \(item\); it admits Event$",
            id="EXPERIENCED-over-Document-items",
        ),
        pytest.param(
            _group_out("G1", "user HAS_GOAL item", item_type="Skill"),
            (),
            r"HAS_GOAL does not admit Skill at object \(item\); it admits Goal$",
            id="HAS_GOAL-over-Skill-items",
        ),
        pytest.param(
            _group_out("G1", "item USES f1"),
            (_THRENNIC,),
            r"USES does not admit Technology at subject \(item\); it admits User, Person, "
            r"Organization, MistIdentity$",
            id="USES-from-a-Technology-item-to-an-Organization",
        ),
        pytest.param(
            _group_out("G1", "user WORKS_WITH item"),
            (),
            r"WORKS_WITH does not admit User at subject \(user\) with Technology at object "
            r"\(item\) in either orientation; it admits Organization, Technology, "
            r"MistIdentity with Technology$",
            id="WORKS_WITH-in-neither-orientation",
        ),
    ],
)
def test_a_template_whose_predicate_does_not_admit_its_end_types_is_refused(
    group, frame_entities, message
):
    with pytest.raises(ListOutputError, match=message):
        parse_frame_output(_frame_text(group, frame_entities=frame_entities), _KIT)


def test_a_parent_end_is_typed_by_the_parent_groups_item_type():
    text = _nested_frame(("user WORKS_AT item",), G1={"item_type": "Organization"})

    with pytest.raises(
        ListOutputError,
        match=r"groups\[1\]\.templates\[0\] \(G2\): DEPENDS_ON does not admit Organization "
        r"at object \(parent\); it admits Technology$",
    ):
        _nested_parse(text)


@pytest.mark.parametrize(
    ("template", "item_type", "frame_entities"),
    [
        pytest.param("f1 WORKS_WITH item", "Organization", (_TECH,), id="WORKS_WITH-reversed"),
        pytest.param("item WORKS_WITH f1", "Organization", (_TECH,), id="WORKS_WITH-stated"),
        pytest.param("user RELATED_TO item", "Date", (), id="RELATED_TO-any-target"),
        pytest.param("item RELATED_TO mist", "Metric", (), id="RELATED_TO-any-source"),
    ],
)
def test_a_template_that_fits_in_either_orientation_or_any_type_parses(
    template, item_type, frame_entities
):
    group = _group_out("G1", template, item_type=item_type)

    frame = parse_frame_output(_frame_text(group, frame_entities=frame_entities), _KIT)

    assert frame.groups[0].templates[0].predicate == template.split()[1]


_ONE = group_units(split_structured("Kit:\n- Quorvane"))[0]


def _validator_keeps(predicate: str, source_type: str, target_type: str) -> bool:
    entities = [{"id": "a", "name": "a", "type": source_type}]
    entities.append({"id": "b", "name": "b", "type": target_type})
    edge = {"source": "a", "target": "b", "type": predicate, "properties": {"confidence": 0.9}}
    result = ExtractionValidator().validate(ExtractionResult(entities, [edge]))
    return len(result.relationships) == 1


def _parser_keeps(subject: str, predicate: str, object_: str, entities: list[dict]) -> bool:
    group = _group_out("G1", f"{subject} {predicate} {object_}", item_type=None)
    try:
        parse_frame_output(_frame_text(group, frame_entities=entities), _ONE)
    except ListOutputError:
        return False
    return True


def _typed(key: str, etype: str) -> dict:
    return {"key": key, "span": key, "name": key, "type": etype}


@pytest.mark.parametrize("predicate", PREDICATE_ENUM)
def test_the_template_type_check_agrees_with_the_validator(predicate):
    types = list(EXTRACTABLE_TYPE_ENUM)
    disagree = [
        (s, o)
        for s in types
        for o in types
        if _parser_keeps("f1", predicate, "f2", [_typed("f1", s), _typed("f2", o)])
        != _validator_keeps(predicate, s, o)
    ]
    for role, role_type in (("user", "User"), ("mist", "MistIdentity")):
        disagree += [
            (role, o)
            for o in types
            if _parser_keeps(role, predicate, "f1", [_typed("f1", o)])
            != _validator_keeps(predicate, role_type, o)
        ]

    assert disagree == []


# --- item slots -------------------------------------------------------------------------------


def _slots(turn: str, *groups: dict, frame_entities=(), relationships=()) -> dict:
    list_groups = group_units(split_structured(turn))[0]
    text = _frame_text(*groups, frame_entities=frame_entities, relationships=relationships)
    frame = parse_frame_output(text, list_groups)
    ids = [item for group in list_groups for item in group.item_ids]
    return {slot.item: slot for slot in item_slots(frame, list_groups, ids)}


def test_an_item_template_gives_its_items_the_types_it_admits():
    slots = _slots("Kit:\n- Quorvane", _group_out("G1", "user USES item"))

    assert slots == {"G1.1": ItemSlot("G1.1", False, ("Technology",), False, None)}


def test_an_unchecked_goal_box_requires_a_goal_and_takes_no_template():
    goal = _group_out("G1", "user HAS_GOAL item", item_type="Goal", unchecked="goal")

    slots = _slots(_GOALS_TURN, goal)

    assert slots == {
        "G1.1": ItemSlot("G1.1", False, None, True, ("Goal",)),
        "G1.2": ItemSlot("G1.2", False, ("Goal",), False, None),
    }


def test_an_unchecked_plan_box_has_no_slot_type():
    plan = _group_out("G1", "user HAS_GOAL item", item_type="Goal", unchecked="plan")

    assert _slots(_GOALS_TURN, plan)["G1.1"] == ItemSlot("G1.1", False, None, False, None)


def test_the_parent_of_a_group_whose_template_names_parent_is_required():
    frame = _nested_parse(_nested_frame())

    slots = {s.item: s for s in item_slots(frame, _NESTED, ["G1.1", "G1.2", "G2.1"])}

    assert slots == {
        "G1.1": ItemSlot("G1.1", False, ("Technology",), True, ("Technology",)),
        "G1.2": ItemSlot("G1.2", False, ("Technology",), False, None),
        "G2.1": ItemSlot(
            "G2.1", True, ("Organization", "Technology", "Project", "MistIdentity"), False, None
        ),
    }


def test_an_item_a_frame_fact_names_is_required_with_no_type_limit():
    edge = _edge("user", "item:G1.2", "PREFERS")

    slots = _slots(
        "Kit:\n- Quorvane\n- Brellix", _group_out("G1", "user USES item"), relationships=[edge]
    )

    assert slots["G1.2"] == ItemSlot("G1.2", False, ("Technology",), True, None)
    assert slots["G1.1"].required is False


def test_a_group_that_states_nothing_durable_gives_no_slot_type():
    errands = _group_out("G1", item_type=None, durable=False)

    assert _slots("Errands:\n- Pindle", errands) == {
        "G1.1": ItemSlot("G1.1", False, None, False, None)
    }


def test_two_sources_on_one_item_give_the_intersection_of_their_types():
    goal = _group_out("G1", "user HAS_GOAL item", item_type="Goal", unchecked="goal")
    docs = _group_out("G2", "parent REFERENCES_DOCUMENT item", item_type="Document")

    slots = _slots(_GOAL_PARENT_TURN, goal, docs)

    assert slots["G1.1"] == ItemSlot("G1.1", False, None, True, ("Goal",))
    assert slots["G2.1"] == ItemSlot("G2.1", True, ("Document",), False, None)


def test_an_undirected_template_admits_the_union_over_its_orientations():
    group = _group_out("G1", "f1 WORKS_WITH item", item_type="Organization")

    slots = _slots("Kit:\n- Quorvane", group, frame_entities=[_TECH])

    assert slots["G1.1"].applying_types == ("Organization", "Technology", "MistIdentity")


def test_a_frame_that_leaves_a_slot_no_type_is_refused():
    goal = _group_out("G1", item_type=None, unchecked="goal")
    child = _group_out("G2", "item DEPENDS_ON parent")

    with pytest.raises(
        ListOutputError, match=r"item G1\.1: no type meets every set its slot needs.*Goal.*"
    ) as refused:
        _slots(_GOAL_PARENT_TURN, goal, child)

    assert "Technology" in str(refused.value)


# --- item entities against their slots --------------------------------------------------------


def _applying(record: dict) -> dict:
    return record | {"template": record["template"] | {"applies": True}}


_PREFERS_G12 = _nested_frame(relationships=(_edge("user", "item:G1.2", "PREFERS"),))
_GOALS = group_units(split_structured(_GOALS_TURN))[0]


@pytest.mark.parametrize(
    ("record", "frame", "message"),
    [
        pytest.param(
            _applying(_item_out("G1.2", None)),
            None,
            r"items\[0\] \(G1\.2\)\.entity: .*requires an entity",
            id="U2-applies-with-a-null-entity",
        ),
        pytest.param(
            _applying(_item_out("G1.2", "Concept")),
            None,
            r"items\[0\] \(G1\.2\)\.entity\.type: Concept is not admitted here; give one of "
            r"Technology, or applies false",
            id="2b-applies-with-an-unadmitted-type",
        ),
        pytest.param(
            _item_out("G1.1", None),
            None,
            r"items\[0\] \(G1\.1\)\.entity: .*requires an entity",
            id="U5-a-parent-with-a-null-entity",
        ),
        pytest.param(
            _item_out("G1.1", "Concept"),
            None,
            r"items\[0\] \(G1\.1\)\.entity\.type: Concept is not admitted here; give one of "
            r"Technology$",
            id="a-parent-of-an-unadmitted-type",
        ),
        pytest.param(
            _item_out("G1.2", None),
            _PREFERS_G12,
            r"items\[0\] \(G1\.2\)\.entity: .*requires an entity",
            id="U6-an-item-a-fact-names-with-a-null-entity",
        ),
    ],
)
def test_the_item_parser_refuses_an_entity_its_slot_does_not_admit(record, frame, message):
    with pytest.raises(ListOutputError, match=message):
        _nested_items(record, frame=frame)


def test_an_unchecked_goal_box_of_another_type_is_refused():
    goal = _group_out("G1", "user HAS_GOAL item", item_type="Goal", unchecked="goal")
    frame = parse_frame_output(_frame_text(goal), _GOALS)

    with pytest.raises(ListOutputError, match=r"Skill is not admitted here; give one of Goal$"):
        parse_item_output(
            json.dumps({"items": [_item_out("G1.1", "Skill")]}), ["G1.1"], frame, _GOALS, {}
        )


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(_item_out("G1.2", None), id="not-applying-null"),
        pytest.param(_item_out("G1.2", "Concept"), id="not-applying-any-type"),
        pytest.param(_applying(_item_out("G1.1")), id="applying-and-required"),
        pytest.param(_item_out("G2.1", "Skill"), id="a-child-not-applying"),
    ],
)
def test_the_item_parser_accepts_an_entity_its_slot_admits(record):
    (item,) = _nested_items(record)

    assert item.item == record["item"]


# --- the list prompt's examples ---------------------------------------------------------------


def _list_examples() -> list[str]:
    rendered = LIST_SYSTEM_PROMPT.format(reference_date="2031-02-03")
    return rendered.split("## LIST EXAMPLES", 1)[1].split("### List example ")[1:]


def test_each_list_example_parses_and_its_frames_are_the_render_of_its_frame_output():
    examples = _list_examples()

    assert len(examples) >= 2
    for example in examples:
        utterance = json.loads(example.split("Utterance: ", 1)[1].split("\n", 1)[0])
        groups, _ = group_units(split_structured(utterance))
        frame_text = example.split("FRAME output:\n", 1)[1].split("\n", 1)[0]
        frames = example.split("Frames:\n", 1)[1].split("\nITEMS ", 1)[0]
        items_text = example.split(" output:\n")[-1].split("\n", 1)[0]

        frame = parse_frame_output(frame_text, groups)
        expected = [i for g in groups for i in g.item_ids]
        items = parse_item_output(items_text, expected, frame, groups, {})

        assert frames == render_frames_for_items(frame, groups)
        assert len(items) == sum(len(g.items) for g in groups)


def test_each_list_example_writes_its_keys_in_the_schemas_order():
    for example in _list_examples():
        frame_text = example.split("FRAME output:\n", 1)[1].split("\n", 1)[0]
        items_text = example.split(" output:\n")[-1].split("\n", 1)[0]

        for group in json.loads(frame_text)["groups"]:
            assert list(group) == ["group", "durable", "templates", "item_type", "unchecked"]
            assert group["durable"] is True
        for item in json.loads(items_text)["items"]:
            assert list(item) == ["item", "template", "entity", "entities", "relationships"]
