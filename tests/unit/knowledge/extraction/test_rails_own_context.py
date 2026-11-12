"""The own-context rail: an id loses the slug of the context its batch was read under.

A list item is extracted with the lead-in, headings and parent items above it as its
context, and the model sometimes writes that context in front of the item's id. The
batch record names each batch's units, their contexts and the entities the batch
returned, so the rail knows which context an id may carry.

A line is dropped only when something is left, when it names no entity of the row's
payload, and when what is left is not the user anchor's id or a reserved identity's
name or id. The longest line that fits is tried, and after a drop the id is read
again; a line that is refused ends the strip.

The cases run on the fixture packs, whose types are invented, so a case that passes
shows the runner reading the pack's rules and not knowing a type or a name. The cases
that name the v1.4.0 pack run on it, because they pin what that pack's reserved names
and its own example give. Every batch record is built by hand. `_batch` builds one in
the contract model's shape; the first two cases under "a record that says less" are
written as bare dicts, because they leave out what the model requires. Every name is
invented.
"""

from __future__ import annotations

import copy
import json

import pytest

from backend.extraction_contract.models import BatchRecordEntry, ExtractionPayload
from backend.knowledge.extraction.rails import normalize_entity, rail_row, run_rails
from backend.knowledge.ontologies.pack import (
    ID,
    PAYLOAD,
    ConventionRule,
    IdExempt,
    Pack,
    ReservedIdentity,
    build_pack,
    load_pack,
)

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")
V1_4_0 = load_pack("backend.knowledge.ontologies.packs.v1_4_0")


def _with_rules(pack: Pack, rules: tuple[ConventionRule, ...]) -> Pack:
    """`pack` again with `rules` for its rules."""
    return build_pack(
        manifest=pack.manifest,
        node_types=pack.node_types,
        edge_types=pack.edge_types,
        extractable_node_types=pack.extractable_node_types,
        extractable_edge_types=pack.extractable_edge_types,
        rules=rules,
        plugins=pack.plugins,
    )


def _with_rule(pack: Pack, rule: ConventionRule) -> Pack:
    """`pack` again with one more rule."""
    return _with_rules(pack, (*pack.rules, rule))


# The base pack with no alias rule, so no id below is stored under another word.
NO_ALIAS = _with_rules(BASE, tuple(rule for rule in BASE.rules if rule.kind != "alias_vocabulary"))


def _run(entities, relationships=(), pack: Pack = BASE, **row):
    return run_rails(
        rail_row(copy.deepcopy(list(entities)), copy.deepcopy(list(relationships)), **row), pack
    )


def _hand(entity_id: str, name: str | None = None) -> dict:
    return {"id": entity_id, "name": name or entity_id, "type": "Deckhand"}


def _edge(source: str, target: str, edge_type: str) -> dict:
    return {"source": source, "target": target, "type": edge_type}


def _batch(contexts, entities, *, shown=()) -> dict:
    """One Stage 2 call's record: a unit per entry of `contexts`, and what it returned.

    Args:
        contexts: Each unit's context lines, outermost first.
        entities: The entities the call returned.
        shown: The earlier units the call was given for reference.
    """
    entry = BatchRecordEntry(
        units=[
            {"text": f"item {number}", "context": list(context), "kind": "list_item"}
            for number, context in enumerate(contexts, start=1)
        ],
        context=list(shown),
        payload=ExtractionPayload(entities=copy.deepcopy(list(entities)), relationships=[]),
        prompt_tokens=120,
        completion_tokens=40,
        finish="stop",
    )
    return entry.model_dump(mode="json")


def _ids(entities: list[dict]) -> list[str]:
    return [entity["id"] for entity in entities]


def _stored(entity: dict, *batches: dict, pack: Pack = BASE) -> str:
    """The id one entity is stored under, given the row's batch record."""
    entities, _ = _run([entity], pack=pack, batch_record=list(batches))
    return entities[0]["id"]


def _row_ids(row: tuple) -> list[str]:
    """The ids a row of `_ROWS`'s shape is stored under: (entities, edges, record, pack)."""
    entities, relationships, record, pack = row
    stored, _ = _run(entities, relationships, pack=pack, batch_record=record)
    return _ids(stored)


NIGHT_WATCH = ("Night watch",)

# --- the shape: a lead-in's slug in front of each item's id ----------------------------------

_ROSTER = [
    _hand("user", "User"),
    _hand("night-watch-tolliver", "Tolliver"),
    _hand("night-watch-fenwick", "Fenwick"),
    _hand("night-watch-hale", "Hale"),
    _hand("night-watch-the-cook", "the cook"),
    {"id": "mooring 12", "name": "Mooring 12", "type": "Mooring"},
]
_ROSTER_EDGES = [
    _edge("user", "night-watch-tolliver", "KNOWS_HAND"),
    _edge("user", "night-watch-fenwick", "KNOWS_HAND"),
    _edge("user", "night-watch-hale", "KNOWS_HAND"),
    _edge("user", "night-watch-the-cook", "KNOWS_HAND"),
    _edge("night-watch-tolliver", "mooring 12", "MOORED_AT"),
]
# One batch of four list items, all under the one lead-in.
_ROSTER_RECORD = [_batch([NIGHT_WATCH] * 4, _ROSTER)]

# The rows of the cases below, each as (entities, edges, batch record, pack). The twice
# test at the end of the file runs every one of them under its own pack.
_ROSTER_ROW = (_ROSTER, _ROSTER_EDGES, _ROSTER_RECORD, BASE)

# The target: the lead-in is in each item's name as well, and is no entity of the row.
_DAWN = [
    _hand("user", "User"),
    _hand("dawn-crew-marlow", "Dawn crew Marlow"),
    _hand("dawn-crew-pettigrew", "Dawn crew Pettigrew"),
]
_DAWN_EDGES = [
    _edge("user", "dawn-crew-marlow", "KNOWS_HAND"),
    _edge("user", "dawn-crew-pettigrew", "KNOWS_HAND"),
]
_DAWN_ROW = (_DAWN, _DAWN_EDGES, [_batch([("Dawn crew",)] * 2, _DAWN)], BASE)

# Two headings that are entities of the row, each with one item under it. The headings
# came from a batch with no context, the items from a batch under their own heading.
_STORES = [
    _hand("redis", "Redis"),
    _hand("postgres", "Postgres"),
    _hand("redis-cluster", "Redis cluster"),
    _hand("postgres-cluster", "Postgres cluster"),
]
_STORES_RECORD = [
    _batch([()], _STORES[:2]),
    _batch([("Redis",)], _STORES[2:3]),
    _batch([("Postgres",)], _STORES[3:]),
]
_STORES_ROW = (_STORES, [], _STORES_RECORD, NO_ALIAS)

# The pack's own example of an Event id that keeps its owner, under a heading that
# names the owner. The example's payload holds no such Person; this row adds one.
_HOUSEWARMING = [
    {"id": "user", "name": "User", "type": "User"},
    {"id": "maelle", "name": "Maelle", "type": "Person"},
    {
        "id": "maelle-housewarming-in-dunmarrow",
        "name": "Maelle housewarming in Dunmarrow",
        "type": "Event",
    },
]
_HOUSEWARMING_ROW = (_HOUSEWARMING, [], [_batch([("Maelle",)], _HOUSEWARMING)], V1_4_0)

# The longest line that fits names an entity of the row; the shorter line does not.
_NODES = [
    _hand("redis-cluster", "Redis cluster"),
    _hand("redis-cluster-node-1", "Redis cluster node 1"),
]
_NODES_ROW = (_NODES, [], [_batch([("Redis", "Redis cluster")], _NODES)], BASE)

# The id is the longest line and nothing more, and a shorter line starts it.
_PLAN = [_hand("trip-plan", "Trip plan")]
_PLAN_ROW = (_PLAN, [], [_batch([("Trip", "Trip plan")], _PLAN)], BASE)

# An id that carries both lines of its context, the heading and then the lead-in.
_VISIT = [_hand("trip-plan-day-1-museum-visit", "Museum visit")]
_VISIT_ROW = (_VISIT, [], [_batch([("Trip plan", "Day 1")], _VISIT)], BASE)

# One batch that returned an id with its context twice, and the same id with it once.
_DOUBLED = [
    _hand("redis-redis-cluster", "Redis cluster"),
    _hand("redis-cluster", "Redis cluster"),
]
_DOUBLED_ROW = (_DOUBLED, [], [_batch([("Redis",)], _DOUBLED)], BASE)

# Ids that a strip would leave as a name the v1.4.0 pack reserves, or as the id it
# maps those names to. Neither entity is itself reserved: its name and its whole id
# are not reserved names.
_TOOLS = [
    {"id": "tools-mist", "name": "Tools MIST", "type": "Technology"},
    {"id": "tools-mist-identity", "name": "Tools MIST identity", "type": "Technology"},
    {"id": "tools-lanternfish", "name": "Tools Lanternfish", "type": "Technology"},
]
_TOOLS_ROW = (_TOOLS, [], [_batch([("Tools",)], _TOOLS)], V1_4_0)

# The same under the base pack: the user anchor's id, a reserved name, and the id the
# pack maps that name to.
_KEPT = [
    _hand("night-watch-user", "Night watch user"),
    _hand("night-watch-harbour-master", "Night watch harbour master"),
    _hand("night-watch-harbourmaster", "Night watch harbourmaster"),
]
_KEPT_ROW = (_KEPT, [], [_batch([NIGHT_WATCH], _KEPT)], BASE)


def test_a_lead_ins_slug_in_front_of_each_items_id_is_stripped_and_the_edges_follow():
    entities, relationships = _run(_ROSTER, _ROSTER_EDGES, batch_record=_ROSTER_RECORD)

    # The strip comes first: the leading-word rail then drops `the`, and the rule that
    # builds an id from an edge builds it from the stripped id.
    assert _ids(entities) == [
        "user",
        "tolliver-at-mooring-12",
        "fenwick",
        "hale",
        "cook",
        "mooring-12",
    ]
    assert relationships == [
        _edge("user", "tolliver-at-mooring-12", "KNOWS_HAND"),
        _edge("user", "fenwick", "KNOWS_HAND"),
        _edge("user", "hale", "KNOWS_HAND"),
        _edge("user", "cook", "KNOWS_HAND"),
        _edge("tolliver-at-mooring-12", "mooring-12", "MOORED_AT"),
    ]


def test_an_items_id_loses_a_lead_in_that_is_in_its_name_and_names_no_entity():
    entities, relationships = _run(*_DAWN_ROW[:2], batch_record=_DAWN_ROW[2])

    # The rail builds the id; the name is the model's and keeps the lead-in.
    assert entities == [
        _hand("user", "User"),
        _hand("marlow", "Dawn crew Marlow"),
        _hand("pettigrew", "Dawn crew Pettigrew"),
    ]
    assert relationships == [
        _edge("user", "marlow", "KNOWS_HAND"),
        _edge("user", "pettigrew", "KNOWS_HAND"),
    ]


def test_a_row_without_a_batch_record_is_not_stripped():
    entities, _ = _run(_ROSTER, _ROSTER_EDGES)

    assert _ids(entities) == [
        "user",
        "night-watch-tolliver-at-mooring-12",
        "night-watch-fenwick",
        "night-watch-hale",
        "night-watch-the-cook",
        "mooring-12",
    ]


def test_one_entity_normalized_alone_has_no_record_and_is_not_stripped():
    entity = _hand("night-watch-fenwick", "Fenwick")

    normalize_entity(entity, BASE)

    assert entity["id"] == "night-watch-fenwick"


# --- what is never stripped, and an id the strip would empty ----------------------------------


def test_an_id_that_is_the_context_and_nothing_more_is_kept():
    watch = _hand("Night watch")

    assert _stored(watch, _batch([NIGHT_WATCH], [watch])) == "night-watch"


def test_a_type_an_id_exempt_rule_names_is_not_stripped():
    quay = {"id": "night-watch-far-quay", "name": "Far quay", "type": "Mooring"}

    assert _stored(quay, _batch([NIGHT_WATCH], [quay])) == "night-watch-far-quay"


def test_the_exempt_types_are_read_from_the_packs_rules():
    biscuit = {"id": "our-pets-biscuit", "name": "Biscuit", "type": "Pet"}
    record = _batch([("Our pets",)], [biscuit])
    exempting = _with_rule(
        PET,
        ConventionRule(
            id="pet-exempt", kind="id_exempt", phase=ID, params=IdExempt(types=("Pet",))
        ),
    )

    assert _stored(biscuit, record, pack=PET) == "biscuit"
    assert _stored(biscuit, record, pack=exempting) == "our-pets-biscuit"


def test_a_settled_entity_is_not_stripped():
    # The reserved id `night-warden` opens with the lead-in `Night` and a hyphen, so a
    # strip that reached a settled entity would store it as `warden`.
    warden_pack = _with_rule(
        BASE,
        ConventionRule(
            id="night-warden",
            kind="reserved_identity",
            phase=PAYLOAD,
            params=ReservedIdentity(
                names=("the night warden",), id="night-warden", type="Deckhand"
            ),
        ),
    )
    warden = {"id": "night-warden", "name": "The night warden", "type": "Mooring"}
    record = _batch([("Night",)], [warden])

    entities, _ = _run([warden], pack=warden_pack, batch_record=[record])

    assert entities == [{"id": "night-warden", "name": "The night warden", "type": "Deckhand"}]
    # The same id on an entity the pack does not reserve is stripped.
    assert _stored(_hand("night-warden", "Warden"), record) == "warden"


@pytest.mark.parametrize(
    ("lead_in", "written", "kept"),
    [
        ("???", "??? ", "???-"),
        ("???", "???  ", "???--"),
        ("Список", "Список ", "список-"),
        ("Список", "список-", "список-"),
    ],
    ids=["one-space", "two-spaces", "no-ascii-letter", "a-written-hyphen"],
)
def test_an_id_that_is_the_context_and_a_hyphen_is_kept(lead_in, written, kept):
    # A slug with no ASCII letter or digit is the lowered text with a hyphen for each
    # space, so it can end in a hyphen: dropping the context and its hyphen would
    # leave an empty id, or only hyphens.
    entity = _hand(written, lead_in)

    assert _stored(entity, _batch([(lead_in,)], [entity])) == kept


# --- a context that names an entity of the row ------------------------------------------------


def test_a_heading_that_is_an_entity_of_the_row_is_kept_in_its_items_ids():
    assert _row_ids(_STORES_ROW) == ["redis", "postgres", "redis-cluster", "postgres-cluster"]
    # With neither heading in the row, both items lose theirs and become one entity.
    assert _row_ids((_STORES[2:], [], _STORES_RECORD[1:], NO_ALIAS)) == ["cluster"]


def test_the_v1_4_0_event_id_keeps_an_owner_that_is_a_person_of_the_row():
    assert _row_ids(_HOUSEWARMING_ROW) == ["user", "maelle", "maelle-housewarming-in-dunmarrow"]
    # The example's own payload holds no such Person, and its id loses the heading.
    event_only = [_HOUSEWARMING[0], _HOUSEWARMING[2]]
    record = [_batch([("Maelle",)], event_only)]
    assert _row_ids((event_only, [], record, V1_4_0)) == ["user", "housewarming-in-dunmarrow"]


@pytest.mark.parametrize(
    ("named", "line", "written"),
    [
        (_hand("Night Watch", "Night watch"), "Night watch", "night-watch-tolliver"),
        (
            {"id": "Mooring 12", "name": "Mooring 12", "type": "Mooring"},
            "Mooring 12",
            "mooring-12-tolliver",
        ),
        (
            {"id": "harbour-master", "name": "The harbourmaster", "type": "Deckhand"},
            "Harbour master",
            "harbour-master-tolliver",
        ),
        (_hand("user", "User"), "User", "user-tolliver"),
    ],
    ids=["written-in-another-case", "an-exempt-type", "a-reserved-identity", "the-user-anchor"],
)
def test_every_entity_of_the_row_counts_as_named_by_the_slug_of_its_written_id(
    named, line, written
):
    # The named entity is in the row and in no batch: the row's payload is what is read.
    hand = _hand(written, "Tolliver")
    record = [_batch([(line,)], [hand])]

    entities, _ = _run([named, hand], batch_record=record)

    assert entities[-1]["id"] == written
    assert _stored(hand, *record) == "tolliver"


# --- the order: the longest line, again after a drop, and no shorter line after a refusal ------


def test_a_longest_line_that_names_an_entity_ends_the_strip():
    # `Redis` alone would leave `cluster-node-1`; it is not tried after the refusal.
    assert _row_ids(_NODES_ROW) == ["redis-cluster", "redis-cluster-node-1"]


def test_an_id_that_is_the_longest_line_is_kept_though_a_shorter_line_starts_it():
    assert _row_ids(_PLAN_ROW) == ["trip-plan"]


def test_an_id_that_carries_two_lines_of_its_context_loses_both():
    assert _row_ids(_VISIT_ROW) == ["museum-visit"]


def test_a_context_written_twice_is_dropped_twice():
    doubled = _hand("night-watch-night-watch-tolliver", "Tolliver")

    assert _stored(doubled, _batch([NIGHT_WATCH], [doubled])) == "tolliver"
    # Both of this batch's ids are stored as one: the id with the context twice, and
    # the id with it once.
    assert _row_ids(_DOUBLED_ROW) == ["cluster"]


# --- a strip that would leave a reserved name, a reserved id or the user anchor's id -----------


def test_the_v1_4_0_reserved_name_and_its_id_are_never_what_a_strip_leaves():
    assert _row_ids(_TOOLS_ROW) == ["tools-mist", "tools-mist-identity", "lanternfish"]


def test_the_user_anchors_id_and_a_reserved_name_or_id_are_read_from_the_pack():
    assert _row_ids(_KEPT_ROW) == [
        "night-watch-user",
        "night-watch-harbour-master",
        "night-watch-harbourmaster",
    ]
    # A pack with no `reserved_identity` rule reserves nothing; the anchor is its manifest's.
    unreserved = _with_rules(
        BASE, tuple(rule for rule in BASE.rules if rule.kind != "reserved_identity")
    )
    assert _row_ids((*_KEPT_ROW[:3], unreserved)) == [
        "night-watch-user",
        "harbour-master",
        "harbourmaster",
    ]


# --- which contexts may strip which entity ------------------------------------------------------


def test_a_batch_strips_only_the_entities_its_own_payload_holds():
    tolliver = _hand("night-watch-tolliver", "Tolliver")
    fenwick = _hand("night-watch-fenwick", "Fenwick")
    hale = _hand("night-watch-hale", "Hale")
    record = [_batch([NIGHT_WATCH], [tolliver]), _batch([("Day watch",)], [fenwick])]

    entities, _ = _run([tolliver, fenwick, hale], batch_record=record)

    # `fenwick` came from the day watch's batch, and `hale` from no batch at all.
    assert _ids(entities) == ["tolliver", "night-watch-fenwick", "night-watch-hale"]


def test_an_entity_two_batches_hold_is_read_under_both_batches_contexts():
    fenwick = _hand("day-watch-fenwick", "Fenwick")
    record = [_batch([NIGHT_WATCH], [fenwick]), _batch([("Day watch",)], [fenwick])]

    assert _stored(fenwick, *record) == "fenwick"


def test_any_unit_of_the_batch_may_strip_an_entity_of_the_batch():
    # The record is per batch, not per entity: it does not say which unit an entity
    # came from, so the context of each of the batch's units is tried.
    tolliver = _hand("night-watch-tolliver", "Tolliver")
    fenwick = _hand("day-watch-fenwick", "Fenwick")
    record = _batch([NIGHT_WATCH, ("Day watch",)], [tolliver, fenwick])

    entities, _ = _run([tolliver, fenwick], batch_record=[record])

    assert _ids(entities) == ["tolliver", "fenwick"]


def test_every_line_of_a_units_context_is_tried_a_heading_as_well_as_the_lead_in():
    under_heading = _hand("harbour-roster-fenwick", "Fenwick")
    under_lead_in = _hand("night-watch-hale", "Hale")
    record = _batch([("Harbour roster", "Night watch")], [under_heading, under_lead_in])

    entities, _ = _run([under_heading, under_lead_in], batch_record=[record])

    assert _ids(entities) == ["fenwick", "hale"]


def test_the_earlier_units_a_batch_was_shown_for_reference_strip_nothing():
    hale = _hand("night-watch-hale", "Hale")
    record = _batch([()], [hale], shown=["Night watch", "Night watch > Tolliver"])

    assert _stored(hale, record) == "night-watch-hale"


# --- how the slug is matched --------------------------------------------------------------------


def test_the_context_is_slugged_as_an_id_is():
    tolliver = _hand("night-watch-week-2-tolliver", "Tolliver")

    assert _stored(tolliver, _batch([("Night Watch (week 2)",)], [tolliver])) == "tolliver"


def test_the_longest_context_that_fits_is_the_one_stripped():
    tolliver = _hand("night-watch-tolliver", "Tolliver")

    assert _stored(tolliver, _batch([("Night", "Night watch")], [tolliver])) == "tolliver"


def test_a_context_is_stripped_only_as_whole_words():
    gale = _hand("nightingale-crew", "Nightingale crew")

    assert _stored(gale, _batch([("Night",)], [gale])) == "nightingale-crew"


def test_a_context_inside_or_at_the_end_of_an_id_is_kept():
    inside = _hand("tolliver-night-watch", "Tolliver")

    assert _stored(inside, _batch([NIGHT_WATCH], [inside])) == "tolliver-night-watch"


def test_a_context_that_opens_with_a_leading_word_is_matched_as_written():
    cook = _hand("the-night-watch-cook", "Cook")

    assert _stored(cook, _batch([("The night watch",)], [cook])) == "cook"


def test_the_packs_id_rules_read_the_stripped_id():
    # `bosun` is an alias of the pack, looked up by the id.
    bosun = _hand("night-watch-bosun", "Bosun")

    assert _stored(bosun, _batch([NIGHT_WATCH], [bosun])) == "boatswain"


# --- a record that says less ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        [{"units": []}],
        [{"units": [{"text": "Fenwick", "kind": "list_item"}], "context": [], "payload": None}],
        [_batch([()], [_hand("night-watch-fenwick", "Fenwick")])],
        [_batch([("***",)], [_hand("night-watch-fenwick", "Fenwick")])],
    ],
    ids=["no-payload-key", "no-payload", "no-context", "a-context-with-no-word"],
)
def test_a_record_that_names_no_context_for_the_entity_strips_nothing(record):
    assert _stored(_hand("night-watch-fenwick", "Fenwick"), *record) == "night-watch-fenwick"


# --- determinism and replay -------------------------------------------------------------------------


def test_the_roster_row_gives_the_bytes_stated_here():
    # The bytes are written out, so a run in any process, under any hash seed, is held
    # to the same ones: key order and entity order included.
    stated = json.dumps(
        [
            [
                {"id": "user", "name": "User", "type": "Deckhand"},
                {
                    "id": "tolliver-at-mooring-12",
                    "name": "Tolliver mooring-12",
                    "type": "Deckhand",
                },
                {"id": "fenwick", "name": "Fenwick", "type": "Deckhand"},
                {"id": "hale", "name": "Hale", "type": "Deckhand"},
                {"id": "cook", "name": "the cook", "type": "Deckhand"},
                {"id": "mooring-12", "name": "Mooring 12", "type": "Mooring"},
            ],
            [
                {"source": "user", "target": "tolliver-at-mooring-12", "type": "KNOWS_HAND"},
                {"source": "user", "target": "fenwick", "type": "KNOWS_HAND"},
                {"source": "user", "target": "hale", "type": "KNOWS_HAND"},
                {"source": "user", "target": "cook", "type": "KNOWS_HAND"},
                {"source": "tolliver-at-mooring-12", "target": "mooring-12", "type": "MOORED_AT"},
            ],
        ]
    )

    assert json.dumps(_run(_ROSTER, _ROSTER_EDGES, batch_record=_ROSTER_RECORD)) == stated


@pytest.mark.parametrize(
    "row",
    [
        _ROSTER_ROW,
        _DAWN_ROW,
        _STORES_ROW,
        _HOUSEWARMING_ROW,
        _NODES_ROW,
        _PLAN_ROW,
        _VISIT_ROW,
        _DOUBLED_ROW,
        _TOOLS_ROW,
        _KEPT_ROW,
    ],
    ids=[
        "the-roster",
        "the-target",
        "headings-that-are-entities",
        "the-v1-4-0-event",
        "a-refused-longest-line",
        "an-id-that-is-the-longest-line",
        "two-lines",
        "a-context-written-twice",
        "the-v1-4-0-reserved-names",
        "the-anchor-and-the-reserved-names",
    ],
)
def test_a_second_run_over_a_rows_own_output_changes_nothing(row):
    # True of these rows, not of every row: `rails._own_contexts` states the two shapes
    # in which a second run differs.
    entities, relationships, record, pack = row
    first = _run(entities, relationships, pack=pack, batch_record=record)

    assert _run(*first, pack=pack, batch_record=record) == first
