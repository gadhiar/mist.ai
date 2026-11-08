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
    parse_frame_output,
    parse_item_output,
    render_frame_input,
)
from backend.knowledge.extraction.list_prompts import LIST_SYSTEM_PROMPT, render_frames_for_items
from backend.knowledge.extraction.structure_splitter import InputUnit, split_structured


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
    return parse_frame_output(text, [g.gid for g in _GROUPS])


def _parse_items(text: str, expected: list[str]) -> tuple[ItemOutput, ...]:
    return parse_item_output(text, expected)


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

        frame = parse_frame_output(frame_text, [g.gid for g in groups])
        items = parse_item_output(items_text, [i for g in groups for i in g.item_ids])

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
