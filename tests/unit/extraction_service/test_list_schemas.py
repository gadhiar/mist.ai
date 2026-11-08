"""List schemas and list prompts: the request a list-path frame or item call sends.

The list path (T19) sends one system prompt for both stages, a stage-naming user message,
and a strict JSON schema per stage. These tests pin what the model is told: the enums equal
the ontology's lists and the prompt's, every schema object declares its keys in one fixed
order that the prompt's list output section repeats, the main prompt's blocks are carried
verbatim, the prompt fits its size cap, and no gold entity or utterance line of the dev
sets is taught by the list prompt.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from backend.extraction_service.list_schemas import (
    EXTRACTABLE_TYPE_ENUM,
    LIST_FRAME_OUTPUT_SCHEMA,
    LIST_ITEM_OUTPUT_SCHEMA,
    PREDICATE_ENUM,
    ItemSlot,
    list_frame_output_schema,
    list_item_output_schema,
)
from backend.knowledge.extraction.list_frames import (
    FrameOutput,
    GroupFrame,
    ListEntity,
    ListFacts,
    ListGroup,
    ListOutputError,
    Template,
    group_units,
    item_slots,
    parse_frame_output,
    parse_item_output,
    render_frame_input,
)
from backend.knowledge.extraction.list_prompts import (
    LIST_FRAME_REPAIR_INSTRUCTION,
    LIST_FRAME_USER_TEMPLATE,
    LIST_ITEM_REPAIR_INSTRUCTION,
    LIST_ITEM_USER_TEMPLATE,
    LIST_SYSTEM_PROMPT,
    render_frames_for_items,
)
from backend.knowledge.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from backend.knowledge.extraction.structure_splitter import split_structured
from backend.knowledge.ontologies import EXTRACTABLE_NODE_TYPES, EXTRACTABLE_RELATIONSHIP_TYPES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_GOLD = _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl"
_SIXTY = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"
_ANCHOR_IDS = {"user", "mist-identity"}

LIST_PROMPT_MAX_CHARS = 42_000

_ENTITY_KEYS = ["key", "span", "name", "type", "activity", "value", "unit", "quantity"]
_PROPERTY_KEYS = [
    "confidence",
    "temporal_status",
    "start_date",
    "end_date",
    "temporal_expression",
    "context",
    "negated",
    "assertion_kind",
]
_RELATIONSHIP_KEYS = ["source", "target", "type", "properties"]

# Every object of each schema, by path, with its keys in declared order and how many lead
# as required. The usually-null keys are optional and last, which keeps an output short.
_FRAME_OBJECTS = {
    "": (["frame_entities", "groups", "facts"], 3),
    "frame_entities[]": (["key", "span", "name", "type"], 4),
    "groups[]": (["group", "durable", "templates", "item_type", "unchecked"], 5),
    "groups[].templates[]": (["subject", "predicate", "object"], 3),
    "facts": (["entities", "relationships"], 2),
    "facts.entities[]": (_ENTITY_KEYS, 4),
    "facts.relationships[]": (_RELATIONSHIP_KEYS, 3),
    "facts.relationships[].properties": (_PROPERTY_KEYS, 0),
}
_ITEM_OBJECTS = {
    "": (["items"], 1),
    # `template` before `entity`: the item states whether the shared fact applies before
    # it names its entity, so a built schema's `applies` branch can then constrain it.
    "items[]": (["item", "template", "entity", "entities", "relationships"], 5),
    "items[].entity": (["span", "name", "type", "activity", "value", "unit", "quantity"], 3),
    "items[].template": (
        ["applies", "assertion_kind", "temporal_status", "start_date", "end_date"],
        3,
    ),
    "items[].entities[]": (_ENTITY_KEYS, 4),
    "items[].relationships[]": (_RELATIONSHIP_KEYS, 3),
    "items[].relationships[].properties": (_PROPERTY_KEYS, 0),
}


def _objects(schema: dict, path: str = "") -> dict[str, dict]:
    """Every object schema inside `schema`, keyed by its path."""
    found: dict[str, dict] = {}
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            found.update(_objects(option, path))
    elif schema.get("type") == "object":
        found[path] = schema
        for key, sub in schema["properties"].items():
            found.update(_objects(sub, f"{path}.{key}" if path else key))
    elif schema.get("type") == "array":
        found.update(_objects(schema["items"], f"{path}[]"))
    return found


def _key_walk(schema: dict, *, marked: bool = False) -> list[str]:
    """Every key of `schema`, depth first, in declared order; `?key` when optional and `marked`."""
    if "anyOf" in schema:
        return [key for option in schema["anyOf"] for key in _key_walk(option, marked=marked)]
    if schema.get("type") == "array":
        return _key_walk(schema["items"], marked=marked)
    if schema.get("type") != "object":
        return []
    keys: list[str] = []
    for key, sub in schema["properties"].items():
        optional = marked and key not in schema["required"]
        keys.append(f"?{key}" if optional else key)
        keys.extend(_key_walk(sub, marked=marked))
    return keys


def _section(start: str, end: str) -> str:
    return LIST_SYSTEM_PROMPT[LIST_SYSTEM_PROMPT.index(start) : LIST_SYSTEM_PROMPT.index(end)]


def _block(start: str, end: str | None) -> str:
    begin = EXTRACTION_SYSTEM_PROMPT.index(start)
    stop = EXTRACTION_SYSTEM_PROMPT.index(end) if end else len(EXTRACTION_SYSTEM_PROMPT)
    return EXTRACTION_SYSTEM_PROMPT[begin:stop]


def _prompt_list(heading: str) -> list[str]:
    """The comma-separated names under one of the main prompt's `### Allowed` headings."""
    after = EXTRACTION_SYSTEM_PROMPT.split(heading, 1)[1].split("\n", 1)[1]
    text = after.split("\n\n", 1)[0]
    return [name.strip() for name in text.replace("\n", " ").split(",") if name.strip()]


# --- enums ------------------------------------------------------------------------------------


def test_the_enums_equal_the_ontology_lists():
    assert list(EXTRACTABLE_TYPE_ENUM) == list(EXTRACTABLE_NODE_TYPES)
    assert list(PREDICATE_ENUM) == list(EXTRACTABLE_RELATIONSHIP_TYPES)
    assert len(EXTRACTABLE_TYPE_ENUM) == 21
    assert len(PREDICATE_ENUM) == 39


def test_the_enums_equal_the_prompts_lists():
    assert sorted(_prompt_list("### Allowed Entity Types")) == sorted(EXTRACTABLE_TYPE_ENUM)
    assert sorted(_prompt_list("### Allowed Relationship Types")) == sorted(PREDICATE_ENUM)


def _enums(schema: dict) -> dict[str, list]:
    """The `enum` of every type and predicate field in `schema`, keyed by its path."""
    found = {}
    for path, obj in _objects(schema).items():
        for key in ("type", "item_type", "predicate"):
            if key in obj["properties"]:
                found[f"{path}.{key}"] = obj["properties"][key]["enum"]
    return found


@pytest.mark.parametrize(
    ("schema", "count"),
    [(LIST_FRAME_OUTPUT_SCHEMA, 5), (LIST_ITEM_OUTPUT_SCHEMA, 3)],
    ids=["frame", "item"],
)
def test_every_type_and_predicate_field_carries_the_enum(schema, count):
    enums = _enums(schema)

    assert len(enums) == count
    for path, enum in enums.items():
        if path.endswith("relationships[].type") or path.endswith(".predicate"):
            assert enum == list(PREDICATE_ENUM), path
        elif path.endswith(".item_type"):
            assert enum == [*EXTRACTABLE_TYPE_ENUM, None], path
        else:
            assert enum == list(EXTRACTABLE_TYPE_ENUM), path


# --- schema shape -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("schema", "expected"),
    [(LIST_FRAME_OUTPUT_SCHEMA, _FRAME_OBJECTS), (LIST_ITEM_OUTPUT_SCHEMA, _ITEM_OBJECTS)],
    ids=["frame", "item"],
)
def test_each_object_declares_its_keys_in_the_pinned_order_required_first(schema, expected):
    objects = _objects(schema)

    assert sorted(objects) == sorted(expected)
    for path, (keys, required) in expected.items():
        obj = objects[path]
        assert list(obj["properties"]) == keys, path
        assert obj["required"] == keys[:required], path
        assert obj["additionalProperties"] is False, path


def test_a_group_holds_at_most_three_templates():
    group = _objects(LIST_FRAME_OUTPUT_SCHEMA)["groups[]"]

    assert group["properties"]["templates"]["maxItems"] == 3


def test_a_group_states_durable_as_a_boolean_and_an_item_entity_a_metrics_strings():
    group = _objects(LIST_FRAME_OUTPUT_SCHEMA)["groups[]"]
    entity = _objects(LIST_ITEM_OUTPUT_SCHEMA)["items[].entity"]

    assert group["properties"]["durable"] == {"type": "boolean"}
    for key in ("value", "unit", "quantity"):
        assert entity["properties"][key] == {"type": "string"}, key


def test_the_schemas_serialise_as_json():
    for schema in (LIST_FRAME_OUTPUT_SCHEMA, LIST_ITEM_OUTPUT_SCHEMA):
        assert json.loads(json.dumps(schema)) == schema


# --- per-call schemas -------------------------------------------------------------------------
#
# Each built schema narrows its static superset: the same objects, keys, key order and
# required lists, each enum or const inside the superset's values. Synthetic turns cover
# a flat list, a nested one, checkboxes as plans and as goals, a table and a mixed turn.


def _group(gid: str, *templates: str, item_type="Technology", **more) -> dict:
    """A group frame; each template is written `subject PREDICATE object`."""
    group: dict = {"group": gid, "durable": True}
    group["templates"] = [
        dict(zip(("subject", "predicate", "object"), t.split())) for t in templates
    ]
    return group | {"item_type": item_type, "unchecked": "plan"} | more


def _edge(source: str, target: str, type_: str = "RELATED_TO") -> dict:
    return {"source": source, "target": target, "type": type_}


_THRENNIC = {"key": "f1", "span": "Thrennic", "name": "Thrennic", "type": "Organization"}
_OSTBY = {"key": "e1", "span": "Ostby", "name": "Ostby", "type": "Location"}


def _frame(*groups: dict, frame_entities=(), entities=(), relationships=()) -> dict:
    facts = {"entities": list(entities), "relationships": list(relationships)}
    return {"frame_entities": list(frame_entities), "groups": list(groups), "facts": facts}


_TURNS = {
    "flat": (
        "Kit at Thrennic:\n- Quorvane\n- Brellix",
        _frame(
            _group("G1", "user USES item"),
            frame_entities=[_THRENNIC],
            relationships=[_edge("user", "f1", "WORKS_AT")],
        ),
    ),
    "nested": (
        "Stack:\n- Quorvane:\n  - Brellix\n- Ostrel",
        _frame(_group("G1", "user USES item"), _group("G2", "item DEPENDS_ON parent")),
    ),
    "checkbox-plan": (
        "Chores:\n- [ ] sweep the yard\n- [x] fix the gate",
        _frame(_group("G1", item_type=None, durable=False)),
    ),
    "checkbox-goal": (
        "Goals:\n- [ ] learn Kestrin\n- [x] visit Ostby",
        _frame(_group("G1", "user HAS_GOAL item", item_type="Goal", unchecked="goal")),
    ),
    "table": (
        "Services:\n| Service | Latency |\n|---|---|\n| Quorvane | 40 ms |\n| Brellix | 90 ms |",
        _frame(_group("G1")),
    ),
    "mixed": (
        "I moved to Ostby last week.\n\nTools:\n- Quorvane\n- Brellix",
        _frame(
            _group("G1", "user USES item"),
            entities=[_OSTBY],
            relationships=[_edge("user", "e1"), _edge("user", "item:G1.2", "PREFERS")],
        ),
    ),
}


def _turn(name: str) -> tuple[tuple[ListGroup, ...], dict, FrameOutput]:
    text, frame = _TURNS[name]
    groups = group_units(split_structured(text))[0]
    return groups, frame, parse_frame_output(json.dumps(frame), groups)


def _item_ids(groups) -> list[str]:
    return [item for group in groups for item in group.item_ids]


def _frame_schema(groups) -> dict:
    nested = {g.gid for g in groups if g.parent is not None}
    return list_frame_output_schema([g.gid for g in groups], nested, _item_ids(groups))


def _item_schema(groups, frame: FrameOutput, batch: list[str]) -> dict:
    keys = [e.key for e in frame.frame_entities]
    return list_item_output_schema(item_slots(frame, groups, batch), keys, _item_ids(groups))


def _narrows(built: dict, sup: dict, path: str = "$") -> list[str]:
    """Where `built` fails to narrow `sup`; empty when it does."""
    if "anyOf" in built and "anyOf" not in sup:
        return [p for option in built["anyOf"] for p in _narrows(option, sup, path)]
    if "anyOf" in sup:
        options = built.get("anyOf", [built])
        if all(any(not _narrows(o, s, path) for s in sup["anyOf"]) for o in options):
            return []
        return [f"{path}: narrows no option of the superset's anyOf"]
    if built.get("type") != sup.get("type"):
        return [f"{path}: type {built.get('type')!r} is not {sup.get('type')!r}"]
    if sup["type"] == "object":
        if list(built["properties"]) != list(sup["properties"]):
            return [f"{path}: keys {list(built['properties'])}"]
        if built["required"] != sup["required"] or built["additionalProperties"] is not False:
            return [f"{path}: required or additionalProperties differ"]
        return [
            p
            for key, sub in built["properties"].items()
            for p in _narrows(sub, sup["properties"][key], f"{path}.{key}")
        ]
    if sup["type"] == "array":
        entries = built.get("prefixItems", [built.get("items")])
        if "maxItems" in sup and built.get("maxItems", float("inf")) > sup["maxItems"]:
            return [f"{path}: maxItems above the superset's"]
        return [p for n, e in enumerate(entries) for p in _narrows(e, sup["items"], f"{path}[{n}]")]
    extra = set(built) - {"type", "enum", "const", "minLength"}
    values = built.get("enum", [built["const"]] if "const" in built else None)
    if extra:
        return [f"{path}: keys {sorted(extra)} beyond the superset's"]
    if "enum" in sup and (values is None or not set(values) <= set(sup["enum"])):
        return [f"{path}: values {values} outside the superset's enum"]
    return []


def _walk(schema: object):
    """Every dict inside `schema`, itself included."""
    if isinstance(schema, dict):
        yield schema
        for value in schema.values():
            yield from _walk(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _walk(value)


@pytest.mark.parametrize("name", sorted(_TURNS))
def test_each_built_schema_narrows_its_static_superset(name):
    groups, _frame_out, frame = _turn(name)

    assert _narrows(_frame_schema(groups), LIST_FRAME_OUTPUT_SCHEMA) == []
    for batch in (_item_ids(groups), _item_ids(groups)[:1]):
        assert _narrows(_item_schema(groups, frame, batch), LIST_ITEM_OUTPUT_SCHEMA) == []


@pytest.mark.parametrize("name", sorted(_TURNS))
def test_equal_inputs_give_equal_schemas_and_equal_text(name):
    groups, _frame_out, frame = _turn(name)
    batch = _item_ids(groups)

    first = (_frame_schema(groups), _item_schema(groups, frame, batch))
    again = (_frame_schema(groups), _item_schema(groups, frame, batch))

    assert first == again
    assert [json.dumps(s) for s in first] == [json.dumps(s) for s in again]


@pytest.mark.parametrize("name", sorted(_TURNS))
def test_no_built_array_carries_items_beside_prefix_items(name):
    groups, _frame_out, frame = _turn(name)

    for schema in (_frame_schema(groups), _item_schema(groups, frame, _item_ids(groups))):
        tuples = [s for s in _walk(schema) if "prefixItems" in s]
        assert tuples
        for array in tuples:
            assert "items" not in array
            assert array["minItems"] == array["maxItems"] == len(array["prefixItems"])


@pytest.mark.parametrize("name", sorted(_TURNS))
def test_in_every_built_item_schema_template_comes_before_entity(name):
    groups, _frame_out, frame = _turn(name)

    records = [
        s
        for s in _walk(_item_schema(groups, frame, _item_ids(groups)))
        if s.get("type") == "object" and "entity" in s.get("properties", {})
    ]

    assert records
    for record in records:
        keys = list(record["properties"])
        assert keys.index("template") < keys.index("entity")


def test_a_built_item_entry_is_two_branches_when_a_template_names_item():
    groups, _frame_out, frame = _turn("flat")

    (entry,) = _item_schema(groups, frame, ["G1.1"])["properties"]["items"]["prefixItems"]

    applies = [b["properties"]["template"]["properties"]["applies"] for b in entry["anyOf"]]
    entities = [b["properties"]["entity"] for b in entry["anyOf"]]
    assert applies == [{"type": "boolean", "const": True}, {"type": "boolean", "const": False}]
    assert entities[0]["properties"]["type"]["enum"] == ["Technology"]
    assert [option.get("type") for option in entities[1]["anyOf"]] == ["object", "null"]


@pytest.mark.parametrize(
    "slot",
    [
        pytest.param(ItemSlot("G1.1", False, (), False, None), id="no-applying-type"),
        pytest.param(ItemSlot("G1.1", False, None, True, ()), id="no-required-type"),
        pytest.param(ItemSlot("G1.1", False, ("Goal",), True, ("Skill",)), id="sets-do-not-meet"),
    ],
)
def test_an_item_schema_builder_given_an_empty_type_set_raises(slot):
    with pytest.raises(ValueError, match="G1.1"):
        list_item_output_schema([slot], [], ["G1.1"])


# --- built schemas against outputs (jsonschema, Draft 2020-12) ---------------------------------


def _valid(schema: dict, output: dict) -> bool:
    return Draft202012Validator(schema).is_valid(output)


def _valid_items(slots) -> dict:
    """An output the parser accepts for `slots`: every entity named, of an admitted type."""
    records = []
    for slot in slots:
        applies = slot.applying_types is not None
        types = list(slot.applying_types or slot.required_types or ["Concept"])
        if applies and slot.required_types is not None:
            types = [t for t in types if t in slot.required_types]
        edges = [_edge("item", "e1")] + ([_edge("item", "parent")] if slot.nested else [])
        record = {"item": slot.item}
        record["template"] = {"applies": applies, "assertion_kind": "assert"}
        record["template"]["temporal_status"] = "current"
        record["entity"] = {"span": slot.item, "name": slot.item, "type": types[0]}
        record["entities"] = [{"key": "e1", "span": "notes", "name": "notes", "type": "Concept"}]
        records.append(record | {"relationships": edges})
    return {"items": records}


@pytest.mark.parametrize("name", sorted(_TURNS))
def test_every_parser_valid_output_validates_against_its_built_schema(name):
    groups, frame_out, frame = _turn(name)
    batch = _item_ids(groups)
    items = _valid_items(item_slots(frame, groups, batch))

    parse_item_output(json.dumps(items), batch, frame, groups, {})

    assert _valid(_frame_schema(groups), frame_out)
    assert _valid(_item_schema(groups, frame, batch), items)


def _nested_frame() -> dict:
    """The nested turn's frame with a frame entity and a facts entity, a fresh copy."""
    frame = copy.deepcopy(_TURNS["nested"][1])
    frame["frame_entities"] = [dict(_THRENNIC)]
    frame["facts"]["entities"] = [dict(_OSTBY)]
    return frame


def _changed(path: list, value: object) -> dict:
    frame = _nested_frame()
    target = frame
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return frame


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(_changed(["groups", 0, "templates", 0, "subject"], "e1"), id="F1-facts-key"),
        pytest.param(_changed(["frame_entities", 0, "key"], "e1"), id="F2-frame-key-e1"),
        pytest.param(_changed(["facts", "entities", 0, "key"], "f2"), id="F2-facts-key-f2"),
        pytest.param(_changed(["groups", 0, "templates", 0, "object"], "Quorvane"), id="F3-name"),
        pytest.param(_changed(["frame_entities", 0, "span"], ""), id="F4-empty-span"),
        pytest.param(_changed(["facts", "entities", 0, "name"], ""), id="F4-empty-name"),
        pytest.param(_changed(["groups"], _nested_frame()["groups"][:1]), id="F5-missing-group"),
        pytest.param(_changed(["groups", 1, "group"], "G1"), id="F6-repeated-group"),
        pytest.param(_changed(["groups", 1, "group"], "G9"), id="F6-invented-group"),
        pytest.param(
            _changed(["groups"], _nested_frame()["groups"][::-1]), id="groups-out-of-order"
        ),
        pytest.param(
            _changed(["groups", 0, "templates", 0, "subject"], "parent"), id="U4-template"
        ),
        pytest.param(
            _changed(["facts", "relationships"], [_edge("item", "f1")]), id="U1-fact-item"
        ),
        pytest.param(
            _changed(["facts", "relationships"], [_edge("user", "parent")]), id="U1-fact-parent"
        ),
        pytest.param(
            _changed(["facts", "relationships"], [_edge("user", "item:G9.1")]), id="U3-fact"
        ),
        pytest.param(
            _changed(["facts", "relationships"], [_edge("user", "Thrennic")]), id="U7-fact-name"
        ),
    ],
)
def test_the_built_frame_schema_rejects_the_shapes_it_removes(frame):
    groups = group_units(split_structured(_TURNS["nested"][0]))[0]

    assert _valid(_frame_schema(groups), _nested_frame())
    assert not _valid(_frame_schema(groups), frame)


def _nested_items() -> dict:
    groups, _frame_out, frame = _turn("nested")
    return _valid_items(item_slots(frame, groups, _item_ids(groups)))


def _items_changed(path: list, value: object) -> dict:
    items = _nested_items()
    target = items
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value
    return items


@pytest.mark.parametrize(
    "items",
    [
        pytest.param(_items_changed(["items"], _nested_items()["items"][:2]), id="missing-item"),
        pytest.param(_items_changed(["items", 1, "item"], "G1.1"), id="repeated-item"),
        pytest.param(_items_changed(["items", 1, "item"], "G7.1"), id="invented-item"),
        pytest.param(_items_changed(["items", 0, "entities", 0, "key"], "f1"), id="own-key-f1"),
        pytest.param(
            _items_changed(["items", 0, "relationships", 0, "target"], "Quorvane"), id="U7-name"
        ),
        pytest.param(
            _items_changed(["items", 0, "relationships", 0, "target"], "f2"), id="U7-undeclared"
        ),
        pytest.param(
            _items_changed(["items", 0, "relationships", 0, "target"], "item:G7.1"), id="U3"
        ),
        pytest.param(_items_changed(["items", 0, "relationships", 0, "target"], "parent"), id="U4"),
        pytest.param(_items_changed(["items", 1, "entity"], None), id="U2-applies-null"),
        pytest.param(
            _items_changed(["items", 1, "entity", "type"], "Concept"), id="2b-applies-type"
        ),
        pytest.param(_items_changed(["items", 0, "entity", "span"], ""), id="empty-span"),
    ],
)
def test_the_built_item_schema_rejects_the_shapes_it_removes(items):
    groups, _frame_out, frame = _turn("nested")
    schema = _item_schema(groups, frame, _item_ids(groups))

    assert _valid(schema, _nested_items())
    assert not _valid(schema, items)


def test_a_required_parent_with_no_entity_is_rejected_in_either_branch():
    groups, _frame_out, frame = _turn("nested")
    items = _items_changed(["items", 0, "entity"], None)
    items["items"][0]["template"]["applies"] = False

    assert not _valid(_item_schema(groups, frame, _item_ids(groups)), items)


@pytest.mark.parametrize("name", ["nested", "checkbox-goal", "checkbox-plan", "mixed", "table"])
def test_the_item_grammar_and_the_parser_agree_on_each_items_entity(name):
    """For every slot, `applies` and entity type or null: the schema and parser agree."""
    groups, _frame_out, frame = _turn(name)
    disagree = []
    for slot in item_slots(frame, groups, _item_ids(groups)):
        schema = _item_schema(groups, frame, [slot.item])
        for applies in (True, False):
            for etype in (None, *EXTRACTABLE_TYPE_ENUM):
                record = {"item": slot.item}
                record["template"] = {"applies": applies, "assertion_kind": "assert"}
                record["template"]["temporal_status"] = "current"
                record["entity"] = None if etype is None else {"span": "x", "name": "x"}
                if etype is not None:
                    record["entity"]["type"] = etype
                output = {"items": [record | {"entities": [], "relationships": []}]}
                try:
                    parse_item_output(json.dumps(output), [slot.item], frame, groups, {})
                    parsed = True
                except ListOutputError:
                    parsed = False
                if parsed != _valid(schema, output):
                    disagree.append((slot.item, applies, etype))

    assert disagree == []


# --- the list output section ------------------------------------------------------------------


def _schema_line(stage: str) -> str:
    section = _section("## LIST OUTPUT SCHEMA", "## EXTRACTION RULES")
    return section.split(f"Stage {stage}:\n", 1)[1].split("\n", 1)[0]


@pytest.mark.parametrize(
    ("stage", "schema"),
    [("FRAME", LIST_FRAME_OUTPUT_SCHEMA), ("ITEMS", LIST_ITEM_OUTPUT_SCHEMA)],
)
def test_the_list_output_section_names_the_keys_in_schema_order(stage, schema):
    named = re.findall(r'"(\??[a-z_]+)":', _schema_line(stage))

    assert [key.lstrip("?") for key in named] == _key_walk(schema)
    assert named == _key_walk(schema, marked=True)


# --- the system prompt ------------------------------------------------------------------------


def test_the_list_prompt_starts_with_the_main_prompt_up_to_its_output_schema():
    prefix = EXTRACTION_SYSTEM_PROMPT[: EXTRACTION_SYSTEM_PROMPT.index("## OUTPUT SCHEMA")]

    assert LIST_SYSTEM_PROMPT.startswith(prefix)
    assert "## OUTPUT SCHEMA" not in LIST_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("## EXTRACTION RULES", "## REFERENCE DATE"),
        ("## REFERENCE DATE", "## EXAMPLES"),
        ("## EXAMPLES", None),
    ],
)
def test_the_list_prompt_holds_the_main_prompts_block_verbatim(start, end):
    assert _block(start, end) in LIST_SYSTEM_PROMPT


def test_the_list_prompt_orders_its_sections_as_the_design_states():
    order = [
        "## ONTOLOGY CONSTRAINTS",
        "## LIST OUTPUT SCHEMA",
        "## EXTRACTION RULES",
        "## LIST RULES",
        "## REFERENCE DATE",
        "## EXAMPLES",
        "## LIST EXAMPLES",
    ]

    positions = [LIST_SYSTEM_PROMPT.index(heading) for heading in order]

    assert positions == sorted(positions)


def test_the_list_prompt_fits_its_cap():
    assert len(LIST_SYSTEM_PROMPT) <= LIST_PROMPT_MAX_CHARS


def test_the_list_prompt_is_a_template_of_the_reference_date_only():
    rendered = LIST_SYSTEM_PROMPT.format(reference_date="2031-02-03")

    assert "Today's date: 2031-02-03" in rendered
    assert "{reference_date}" not in rendered


def test_the_list_examples_parse_as_the_list_outputs():
    rendered = LIST_SYSTEM_PROMPT.format(reference_date="2031-02-03")
    examples = rendered.split("## LIST EXAMPLES", 1)[1]
    outputs = [json.loads(line) for line in examples.splitlines() if line.startswith("{")]

    frames = [o for o in outputs if "frame_entities" in o]
    items = [o for o in outputs if "items" in o]
    assert frames and items
    assert len(frames) + len(items) == len(outputs)
    for output in frames:
        assert _key_walk_value(output) <= set(_key_walk(LIST_FRAME_OUTPUT_SCHEMA))
    for output in items:
        assert _key_walk_value(output) <= set(_key_walk(LIST_ITEM_OUTPUT_SCHEMA))


def _list_examples() -> list[str]:
    rendered = LIST_SYSTEM_PROMPT.format(reference_date="2031-02-03")
    return rendered.split("## LIST EXAMPLES", 1)[1].split("### List example ")[1:]


def test_each_list_examples_units_are_the_render_of_its_utterance():
    examples = _list_examples()

    assert len(examples) >= 2
    for example in examples:
        utterance = json.loads(example.split("Utterance: ", 1)[1].split("\n", 1)[0])
        units = example.split("Units:\n", 1)[1].split("\nFRAME output:", 1)[0]
        assert units == render_frame_input(*group_units(split_structured(utterance)))


def _key_walk_value(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _key_walk_value(v)}
    if isinstance(value, list):
        return {k for v in value for k in _key_walk_value(v)}
    return set()


# --- user templates and repair instructions ---------------------------------------------------


def test_the_frame_user_template_names_its_stage_and_takes_the_turn():
    message = LIST_FRAME_USER_TEMPLATE.format(
        context="(no prior context)",
        subject_scope="user-scope",
        utterance="- Quorvane",
        render="G1\n  G1.1 Quorvane",
    )

    assert message.startswith("Stage: FRAME\n")
    assert "Subject scope: user-scope" in message
    assert "G1.1 Quorvane" in message


def test_the_item_user_template_names_its_stage_and_the_items_to_extract():
    message = LIST_ITEM_USER_TEMPLATE.format(
        context="(no prior context)",
        subject_scope="user-scope",
        utterance="- Quorvane",
        render="G1\n  G1.1 Quorvane",
        frames="G1: each item is the object of user USES",
        items="G1.1",
    )

    assert message.startswith("Stage: ITEMS\n")
    assert "G1: each item is the object of user USES" in message
    assert message.rstrip().endswith("Output:")


def test_each_repair_instruction_names_its_own_outputs_keys():
    frame_keys = ["frame_entities", "groups", "facts", "group", "templates", "item_type"]
    item_keys = ["items", "item", "entity", "template", "entities", "relationships"]

    for key in frame_keys:
        assert f"{key}" in LIST_FRAME_REPAIR_INSTRUCTION, key
    for key in item_keys:
        assert f"{key}" in LIST_ITEM_REPAIR_INSTRUCTION, key
    assert "frame_entities" not in LIST_ITEM_REPAIR_INSTRUCTION
    assert '"items"' not in LIST_FRAME_REPAIR_INSTRUCTION


# --- frames for the item calls ----------------------------------------------------------------


def test_the_decided_frames_render_as_one_sentence_a_group():
    text = "Tools at Thrennic:\n- Quorvane\n- Ostrel:\n  - Velquist\n\nErrands:\n- [ ] Pindle"
    groups, _ = group_units(split_structured(text))
    frame = FrameOutput(
        frame_entities=(
            ListEntity(key="f1", span="Thrennic", name="Thrennic", type="Organization"),
        ),
        groups=(
            GroupFrame("G1", (Template("user", "USES", "item"),), "Technology", "plan"),
            GroupFrame(
                "G2",
                (Template("item", "DEPENDS_ON", "parent"), Template("f1", "USES", "item")),
                None,
                "plan",
            ),
            GroupFrame("G3", (), None, "goal", durable=False),
        ),
        facts=ListFacts(entities=(), relationships=()),
    )

    assert render_frames_for_items(frame, groups) == (
        "f1 Thrennic (Organization)\n"
        "G1: each item is the object of user USES; item type Technology\n"
        "G2: each item is the subject of DEPENDS_ON parent G1.2; "
        "each item is the object of f1 USES\n"
        "G3: no template, nothing durable; unchecked items are goals"
    )


@pytest.mark.parametrize(
    ("durable", "sentence"),
    [
        pytest.param(True, "G1: no shared template; extract each item's own facts", id="durable"),
        pytest.param(False, "G1: no template, nothing durable", id="not-durable"),
    ],
)
def test_a_group_with_no_template_renders_whether_it_is_durable(durable, sentence):
    """A table whose fact is in each row is durable with no template; an errand list is not."""
    groups, _ = group_units(split_structured("Brindle runs:\n- Corvel\n- Ashwick"))
    frame = FrameOutput(
        frame_entities=(),
        groups=(GroupFrame("G1", (), None, "plan", durable=durable),),
        facts=ListFacts(entities=(), relationships=()),
    )

    assert render_frames_for_items(frame, groups) == sentence


def test_list_rule_5_is_keyed_to_a_group_that_is_not_durable():
    rules = _section("## LIST RULES", "## REFERENCE DATE")
    rule = rules.split("\n5. ", 1)[1].split("\n", 1)[0]

    assert "durable false" in rule
    assert "durable true" in rule


# --- no gold content in the list prompt -------------------------------------------------------


def _term_pattern(term: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in re.split(r"[\s\-_.]+", term) if p]
    return re.compile(r"\b" + r"[\s\-_.]*".join(parts) + r"\b", re.IGNORECASE)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("path", [_DEV_GOLD, _SIXTY], ids=["dev-lists", "sixty"])
def test_no_gold_entity_id_occurs_in_the_list_prompt_beyond_the_main_prompt(path):
    ids = {
        entity["id"]
        for row in _rows(path)
        for entity in row["expected_entities"]
        if entity["id"] not in _ANCHOR_IDS
    }

    hits = sorted(
        i
        for i in ids
        if _term_pattern(i).search(LIST_SYSTEM_PROMPT)
        and not _term_pattern(i).search(EXTRACTION_SYSTEM_PROMPT)
    )

    assert hits == []


def test_no_dev_utterance_line_occurs_in_the_list_prompt_beyond_the_main_prompt():
    lines = {
        line.strip()
        for row in _rows(_DEV_GOLD)
        for line in row["utterance"].splitlines()
        if len(line.split()) >= 3
    }

    hits = sorted(
        line
        for line in lines
        if line.lower() in LIST_SYSTEM_PROMPT.lower()
        and line.lower() not in EXTRACTION_SYSTEM_PROMPT.lower()
    )

    assert hits == []
