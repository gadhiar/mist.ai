"""The list path's system prompt, user templates and repair instructions.

One system prompt serves a structured turn's frame call and its item calls, so
llama-server keeps one cached prompt prefix across the turn; the user message names the
stage. `LIST_SYSTEM_PROMPT` is built from `EXTRACTION_SYSTEM_PROMPT` at import, so the
ontology, the extraction rules and the prose examples the list path teaches are the
prose path's, character for character:

1. The main prompt from its start up to `## OUTPUT SCHEMA`: the header and ONTOLOGY
   CONSTRAINTS.
2. `## LIST OUTPUT SCHEMA`, generated from the two list schemas so it names their keys in
   the order the grammar emits them.
3. The main prompt's `## EXTRACTION RULES`.
4. `## LIST RULES`, the labelling guide's list rules written as instructions.
5. The main prompt's `## REFERENCE DATE` and `## EXAMPLES`, after a line saying the
   examples show which facts to extract.
6. `## LIST EXAMPLES`, synthetic, with invented names.

Like the main prompt, the result is a `str.format` template of `reference_date` alone:
every brace this module adds is doubled.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.extraction_service.list_schemas import (
    EXTRACTABLE_TYPE_ENUM,
    LIST_FRAME_OUTPUT_SCHEMA,
    LIST_ITEM_OUTPUT_SCHEMA,
    PREDICATE_ENUM,
)

from .list_frames import FrameOutput, GroupFrame, ListGroup, Template
from .prompts import EXTRACTION_SYSTEM_PROMPT

_SCALAR_HINTS = {"string": "string", "boolean": "true|false", "number": "number", "null": "null"}


def _enum_hint(values: Sequence[str | None]) -> str:
    names = [value for value in values if value is not None]
    if names == list(EXTRACTABLE_TYPE_ENUM):
        hint = "EntityType"
    elif names == list(PREDICATE_ENUM):
        hint = "RELATIONSHIP_TYPE"
    else:
        hint = '"' + "|".join(names) + '"'
    return f"{hint} or null" if None in values else hint


def _skeleton(schema: dict) -> str:
    """One line naming every key of `schema` in declared order, `?` before an optional one."""
    if "anyOf" in schema:
        return " or ".join(_skeleton(option) for option in schema["anyOf"])
    if "enum" in schema:
        return _enum_hint(schema["enum"])
    if schema["type"] == "object":
        required = set(schema["required"])
        fields = (
            f'"{"" if key in required else "?"}{key}": {_skeleton(sub)}'
            for key, sub in schema["properties"].items()
        )
        return "{" + ", ".join(fields) + "}"
    if schema["type"] == "array":
        return f"[{_skeleton(schema['items'])}]"
    return _SCALAR_HINTS[schema["type"]]


def _escaped(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _block(start: str, end: str | None) -> str:
    begin = EXTRACTION_SYSTEM_PROMPT.index(start)
    stop = EXTRACTION_SYSTEM_PROMPT.index(end) if end else len(EXTRACTION_SYSTEM_PROMPT)
    return EXTRACTION_SYSTEM_PROMPT[begin:stop]


_HEADER = EXTRACTION_SYSTEM_PROMPT[: EXTRACTION_SYSTEM_PROMPT.index("## OUTPUT SCHEMA")]
_EXTRACTION_RULES = _block("## EXTRACTION RULES", "## REFERENCE DATE")
_REFERENCE_DATE = _block("## REFERENCE DATE", "## EXAMPLES")
_EXAMPLES = _block("## EXAMPLES", None)

_LIST_OUTPUT = f"""## LIST OUTPUT SCHEMA
A list turn is extracted in two stages, and the user message names the stage. Return that stage's JSON object with its keys in the order shown. A key marked ? is optional: leave it out when it has no value.

Stage FRAME:
{_skeleton(LIST_FRAME_OUTPUT_SCHEMA)}

Stage ITEMS:
{_skeleton(LIST_ITEM_OUTPUT_SCHEMA)}

- frame_entities: what a lead-in, heading or parent item names, keyed f1, f2.
- groups: one per group in Units. templates: at most 3 facts every item of the group states; subject and object are roles: user, mist (MIST itself), item (each item's own entity), parent (the item a nested group is under) or a frame key. item_type: the type every item shares, or null. unchecked: see LIST RULES 3.
- facts: facts of the prose units and of the frame itself, entities keyed e1, e2.
- items: one record per item id the user message asks for. template: whether the group's templates hold for this item (applies), with their assertion_kind, temporal_status and dates. entity: the item's own entity, or null. entities and relationships: only facts the templates do not already state.
- An endpoint is a role, a frame or entity key, or another item as item:G1.2.

"""

_LIST_RULES = """## LIST RULES
Units is the utterance cut into prose units (P1, P2) and groups (G1, G2) of items (G1.1, G1.2), each group under the heading, lead-in or parent item it follows.
1. Read each item as a sentence that completes the line above it: "Weekend kit:" then "- Brindlecraft" reads "I use Brindlecraft". That line sets the subject or the predicate, which FRAME writes once as the group's templates. It is an entity only when it names one, such as an organization or a project.
2. An item that refers back ("the second one", "both of them") points at the earlier item as item:G1.2 and makes no new entity. An option that was only listed, which nobody chose, gets no edge: applies false.
3. Checkboxes: [x] states a done fact. [ ] is a plan and gets no edge; unchecked is plan. When the lead-in names goals, unchecked is goal, and each [ ] item becomes user HAS_GOAL item.
4. Tables: the header row names the columns, and each row is one fact about its first column, whose entity is the row's entity.
5. A list with nothing durable (an errand list) gets durable false and no templates, and its items return entity null, applies false and no facts. A table whose fact is in each row gets durable true and no templates, and each item states its own facts.
6. Copy every span exactly from the utterance. A Goal span is the goal's verb phrase. An Event's name, and an activity's (activity true: a habit or recurring activity), is the bare noun, with no date and no possessor. A Metric gives value, unit and quantity. A Date's value is its ISO date, resolved against the reference date.
7. Where a rule returns an empty extraction, return the stage's object with empty lists.

"""

_EXAMPLES_INTRO = (
    "The EXAMPLES below show which facts to extract. They use the prose output; "
    "the list stages carry the same facts in their own keys.\n\n"
)

_LIST_EXAMPLES = """
## LIST EXAMPLES

### List example 1: an organization in the lead-in, a nested group, an item the frame does not hold for, a back-reference
Subject scope: user-scope
Utterance: "Tools I use at Thrennic:\\n- Quorvane\\n- Brellix, though I'm only starting on it\\n- Ostrel:\\n  - Velquist\\nI like the first one most."
Units:
P1 I like the first one most.
G1 under "Tools I use at Thrennic"
  G1.1 Quorvane
  G1.2 Brellix, though I'm only starting on it
  G1.3 Ostrel:
G2 under "Tools I use at Thrennic" > "Ostrel" (parent G1.3)
  G2.1 Velquist
FRAME output:
{"frame_entities": [{"key": "f1", "span": "Thrennic", "name": "Thrennic", "type": "Organization"}], "groups": [{"group": "G1", "durable": true, "templates": [{"subject": "user", "predicate": "USES", "object": "item"}], "item_type": "Technology", "unchecked": "plan"}, {"group": "G2", "durable": true, "templates": [{"subject": "user", "predicate": "USES", "object": "item"}, {"subject": "item", "predicate": "DEPENDS_ON", "object": "parent"}], "item_type": "Technology", "unchecked": "plan"}], "facts": {"entities": [], "relationships": [{"source": "user", "target": "f1", "type": "WORKS_AT"}, {"source": "user", "target": "item:G1.1", "type": "PREFERS"}]}}
Frames:
f1 Thrennic (Organization)
G1: each item is the object of user USES; item type Technology
G2: each item is the object of user USES; each item is the subject of DEPENDS_ON parent G1.3; item type Technology
ITEMS G1.1, G1.2, G1.3, G2.1 output:
{"items": [{"item": "G1.1", "template": {"applies": true, "assertion_kind": "assert", "temporal_status": "current"}, "entity": {"span": "Quorvane", "name": "Quorvane", "type": "Technology"}, "entities": [], "relationships": []}, {"item": "G1.2", "template": {"applies": false, "assertion_kind": "assert", "temporal_status": "current"}, "entity": {"span": "Brellix", "name": "Brellix", "type": "Technology"}, "entities": [], "relationships": [{"source": "user", "target": "item", "type": "LEARNING"}]}, {"item": "G1.3", "template": {"applies": true, "assertion_kind": "assert", "temporal_status": "current"}, "entity": {"span": "Ostrel", "name": "Ostrel", "type": "Technology"}, "entities": [], "relationships": []}, {"item": "G2.1", "template": {"applies": true, "assertion_kind": "assert", "temporal_status": "current"}, "entity": {"span": "Velquist", "name": "Velquist", "type": "Technology"}, "entities": [], "relationships": []}]}

### List example 2: a table of events with dates, and unchecked goals
Subject scope: user-scope
Utterance: "Races I ran:\\n| Race | Date |\\n|---|---|\\n| Halvard Fell Race | 2026-04-12 |\\n\\nGoals for May:\\n- [ ] learn Kestrin\\n- [ ] visit Ostby"
Units:
G1 under "Races I ran"
  G1.1 Race: Halvard Fell Race; Date: 2026-04-12
G2 under "Goals for May"
  G2.1 [ ] learn Kestrin
  G2.2 [ ] visit Ostby
FRAME output:
{"frame_entities": [], "groups": [{"group": "G1", "durable": true, "templates": [{"subject": "user", "predicate": "EXPERIENCED", "object": "item"}], "item_type": "Event", "unchecked": "plan"}, {"group": "G2", "durable": true, "templates": [{"subject": "user", "predicate": "HAS_GOAL", "object": "item"}], "item_type": "Goal", "unchecked": "goal"}], "facts": {"entities": [], "relationships": []}}
Frames:
G1: each item is the object of user EXPERIENCED; item type Event
G2: each item is the object of user HAS_GOAL; item type Goal; unchecked items are goals
ITEMS G1.1, G2.1, G2.2 output:
{"items": [{"item": "G1.1", "template": {"applies": true, "assertion_kind": "assert", "temporal_status": "past"}, "entity": {"span": "Halvard Fell Race", "name": "Halvard Fell Race", "type": "Event"}, "entities": [{"key": "e1", "span": "2026-04-12", "name": "2026-04-12", "type": "Date", "value": "2026-04-12"}], "relationships": [{"source": "item", "target": "e1", "type": "OCCURRED_ON"}]}, {"item": "G2.1", "template": {"applies": true, "assertion_kind": "assert", "temporal_status": "future"}, "entity": {"span": "learn Kestrin", "name": "learn Kestrin", "type": "Goal"}, "entities": [], "relationships": []}, {"item": "G2.2", "template": {"applies": true, "assertion_kind": "assert", "temporal_status": "future"}, "entity": {"span": "visit Ostby", "name": "visit Ostby", "type": "Goal"}, "entities": [], "relationships": []}]}
"""

LIST_SYSTEM_PROMPT = (
    _HEADER
    + _escaped(_LIST_OUTPUT)
    + _EXTRACTION_RULES
    + _escaped(_LIST_RULES)
    + _REFERENCE_DATE
    + _escaped(_EXAMPLES_INTRO)
    + _EXAMPLES
    + _escaped(_LIST_EXAMPLES)
)

LIST_FRAME_USER_TEMPLATE = """Stage: FRAME
Context:
{context}
Subject scope: {subject_scope}
Utterance: "{utterance}"

Units:
{render}

Output:"""

LIST_ITEM_USER_TEMPLATE = """Stage: ITEMS
Context:
{context}
Subject scope: {subject_scope}
Utterance: "{utterance}"

Units:
{render}

Frames:
{frames}

Extract these items: {items}

Output:"""

LIST_FRAME_REPAIR_INSTRUCTION = (
    "Your previous output was not valid JSON or did not match the FRAME output. Return ONLY "
    'a JSON object {"frame_entities": [...], "groups": [...], "facts": {"entities": [...], '
    '"relationships": [...]}}, with one groups entry for each group id in Units, each with '
    "group, durable, templates (at most 3), item_type and unchecked. No prose, no markdown "
    "code fences, no explanation."
)

LIST_ITEM_REPAIR_INSTRUCTION = (
    "Your previous output was not valid JSON or did not match the ITEMS output. Return ONLY "
    'a JSON object {"items": [...]}, with one record for each item id asked for, each with '
    "item, template, entity, entities and relationships. No prose, no markdown code fences, "
    "no explanation."
)

LIST_REFUSAL_TEMPLATE = "The output was refused: {error}"


def list_repair_text(instruction: str, error: Exception) -> str:
    """A list stage's repair text, then the line naming why the parser refused the output.

    At temperature 0 a repair that names no reason tends to repeat the refused output,
    so the list calls name it; the fixed instructions stay constants that the prompt
    hash covers.

    Args:
        instruction: The stage's fixed repair instruction.
        error: The parser's refusal.
    """
    return instruction + "\n" + LIST_REFUSAL_TEMPLATE.format(error=error)


def _role(role: str, group: ListGroup | None) -> str:
    if role == "parent" and group is not None and group.parent is not None:
        return f"parent {group.parent[1]}"
    return role


def _template_text(template: Template, group: ListGroup | None) -> str:
    subject = _role(template.subject, group)
    obj = _role(template.object, group)
    if template.object == "item" and template.subject != "item":
        return f"each item is the object of {subject} {template.predicate}"
    if template.subject == "item" and template.object != "item":
        return f"each item is the subject of {template.predicate} {obj}"
    return f"{subject} {template.predicate} {obj}"


def _has_checkbox(group: ListGroup | None) -> bool:
    return group is not None and any(u.text.startswith(("[ ] ", "[x] ")) for u in group.items)


def _frame_sentence(frame: GroupFrame, group: ListGroup | None) -> str:
    """The group's decisions in one sentence.

    A group with no template says whether it is durable: an errand list states nothing,
    while a table whose fact is in each row states each item's own facts, and the item
    call must not read the two alike.
    """
    if frame.templates:
        parts = [_template_text(t, group) for t in frame.templates]
    elif frame.durable:
        parts = ["no shared template; extract each item's own facts"]
    else:
        parts = ["no template, nothing durable"]
    if frame.item_type is not None:
        parts.append(f"item type {frame.item_type}")
    if _has_checkbox(group):
        parts.append(f"unchecked items are {frame.unchecked}s")
    return f"{frame.group}: " + "; ".join(parts)


def render_frames_for_items(frame: FrameOutput, groups: Sequence[ListGroup]) -> str:
    """The frame call's decisions as the item calls read them.

    One line per frame entity, then one sentence per group, in the frame's group order,
    such as "G1: each item is the object of user USES; item type Technology". The
    template edges are built by code, so an item call that reads this returns only what
    the frame does not already state.
    """
    by_gid = {group.gid: group for group in groups}
    lines = [f"{entity.key} {entity.name} ({entity.type})" for entity in frame.frame_entities]
    lines.extend(_frame_sentence(g, by_gid.get(g.group)) for g in frame.groups)
    return "\n".join(lines)
