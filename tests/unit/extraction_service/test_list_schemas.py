"""List schemas and list prompts: the request a list-path frame or item call sends.

The list path (T19) sends one system prompt for both stages, a stage-naming user message,
and a strict JSON schema per stage. These tests pin what the model is told: the enums equal
the ontology's lists and the prompt's, every schema object declares its keys in one fixed
order that the prompt's list output section repeats, the main prompt's blocks are carried
verbatim, the prompt fits its size cap, and no gold entity or utterance line of the dev
sets is taught by the list prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.extraction_service.list_schemas import (
    EXTRACTABLE_TYPE_ENUM,
    LIST_FRAME_OUTPUT_SCHEMA,
    LIST_ITEM_OUTPUT_SCHEMA,
    PREDICATE_ENUM,
)
from backend.knowledge.extraction.list_frames import (
    FrameOutput,
    GroupFrame,
    ListEntity,
    ListFacts,
    Template,
    group_units,
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
    "groups[]": (["group", "templates", "item_type", "unchecked"], 4),
    "groups[].templates[]": (["subject", "predicate", "object"], 3),
    "facts": (["entities", "relationships"], 2),
    "facts.entities[]": (_ENTITY_KEYS, 4),
    "facts.relationships[]": (_RELATIONSHIP_KEYS, 3),
    "facts.relationships[].properties": (_PROPERTY_KEYS, 0),
}
_ITEM_OBJECTS = {
    "": (["items"], 1),
    "items[]": (["item", "entity", "template", "entities", "relationships"], 5),
    "items[].entity": (["span", "name", "type", "activity"], 3),
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


def test_the_schemas_serialise_as_json():
    for schema in (LIST_FRAME_OUTPUT_SCHEMA, LIST_ITEM_OUTPUT_SCHEMA):
        assert json.loads(json.dumps(schema)) == schema


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
            GroupFrame("G3", (), None, "goal"),
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
