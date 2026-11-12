"""The near-duplicate merge: two entities of one type under one name, one edit apart, become one.

The model writes one entity under two spellings of its id in two batches of a turn. The
merge joins the later into the earlier when the two have one type, the slugs of their
ids are within one edit, and their names are equal, read in one case and with single
spaces. It refuses a type an `id_exempt` rule names, an edit that adds, drops or
replaces a digit, and an edit that replaces a one-character token.

The cases run on the fixture packs, whose types are invented, so a case that passes
shows the runner reading the pack's `id_exempt` rules and not knowing a type. No rule
of those packs joins a pair below. The cases on today's ontology are in
`test_id_rails.py`. Every name is invented.
"""

from __future__ import annotations

import copy
import json

import pytest

from backend.knowledge.extraction.rails import rail_row, run_rails
from backend.knowledge.ontologies.pack import (
    ID,
    ConventionRule,
    IdExempt,
    Pack,
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


def _hand(entity_id: str, name: str, **extra) -> dict:
    return {"id": entity_id, "name": name, "type": "Deckhand", **extra}


def _pet(entity_id: str, name: str) -> dict:
    return {"id": entity_id, "name": name, "type": "Pet"}


def _edge(source: str, target: str, edge_type: str, **properties) -> dict:
    relationship = {"source": source, "target": target, "type": edge_type}
    if properties:
        relationship["properties"] = properties
    return relationship


def _ids(entities: list[dict]) -> list[str]:
    return [entity["id"] for entity in entities]


TAMSIN = _hand("tamsin", "Tamsin")
# The same hand under a second spelling of the id: the model's name for her is the same.
TAMSYN = _hand("tamsyn", "Tamsin")
WEND = _hand("wend", "Wend")


# --- what merges --------------------------------------------------------------------------------


def test_two_ids_one_letter_apart_under_one_name_are_one_entity_the_earlier():
    # One hand, written `tamsin` in one batch and `tamsn` in the next.
    entities, relationships = _run(
        [TAMSIN, WEND, _hand("tamsn", "Tamsin")], [_edge("wend", "tamsn", "KNOWS_HAND")]
    )

    assert entities == [TAMSIN, WEND]
    assert relationships == [_edge("wend", "tamsin", "KNOWS_HAND")]


@pytest.mark.parametrize(
    "later_id",
    ["tamsyn", "tamsinn", "tamsi", "amsin"],
    ids=["a-letter-changed", "a-letter-added", "the-last-letter-lost", "the-first-letter-lost"],
)
def test_one_edit_is_one_letter_changed_added_or_lost(later_id):
    entities, _ = _run([TAMSIN, _hand(later_id, "Tamsin")])

    assert entities == [TAMSIN]


@pytest.mark.parametrize(
    ("first_name", "second_name"),
    [
        ("Tamsin Reed", "TAMSIN REED"),
        ("Tamsin Reed", " Tamsin   Reed "),
        ("tamsin\treed", "Tamsin Reed"),
    ],
    ids=["case", "spaces", "a-tab-and-case"],
)
def test_names_that_differ_in_case_or_spacing_alone_are_one_name(first_name, second_name):
    entities, _ = _run([_hand("tamsin-reed", first_name), _hand("tamsyn-reed", second_name)])

    assert entities == [_hand("tamsin-reed", first_name)]


@pytest.mark.parametrize(
    ("first_id", "second_id"),
    [("ky", "ko"), ("jem", "jen"), ("wend", "wand")],
    ids=["two-characters", "three-characters", "four-characters"],
)
def test_two_short_ids_one_letter_apart_under_one_name_are_merged(first_id, second_id):
    entities, _ = _run([_hand(first_id, "Wend"), _hand(second_id, "Wend")])

    assert _ids(entities) == [first_id]


def test_a_digit_and_a_one_character_token_elsewhere_in_both_ids_do_not_stop_a_merge():
    # The edit is inside `tamsin`; the `2` and the `b` are the same in both ids.
    entities, _ = _run([_hand("watch-2-b-tamsin", "Tamsin"), _hand("watch-2-b-tamsyn", "Tamsin")])

    assert _ids(entities) == ["watch-2-b-tamsin"]


def test_ids_are_compared_as_slugs():
    # As written the two ids are several edits apart; their slugs are one.
    entities, relationships = _run(
        [_hand("Tamsin", "Tamsin"), WEND, _hand(" TAMSYN", "Tamsin")],
        [_edge("wend", " TAMSYN", "KNOWS_HAND")],
    )

    assert _ids(entities) == ["tamsin", "wend"]
    assert relationships == [_edge("wend", "tamsin", "KNOWS_HAND")]


def test_a_missing_name_is_read_as_the_id():
    # `tamsin` has no name, so its id is its name, and `tamsyn` is named `Tamsin`.
    entities, _ = _run([{"id": "tamsin", "type": "Deckhand"}, TAMSYN])

    assert entities == [{"id": "tamsin", "type": "Deckhand"}]


def test_the_earlier_entity_in_payload_order_is_kept_whole_and_the_later_one_is_dropped():
    # Payload order decides, not which spelling is right or which entity says more.
    second = _hand("tamsin", "Tamsin", properties={"rank": "mate"})

    entities, _ = _run([TAMSYN, second])

    assert entities == [TAMSYN]


# --- what stays two, under two names ------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_hand("mario", "Mario"), _hand("maria", "Maria")),
        (_hand("olsen", "Olsen"), _hand("olson", "Olson")),
        (_hand("mysql", "MySQL"), _hand("mssql", "MSSQL")),
        (_hand("phase-i", "Phase I"), _hand("phase-ii", "Phase II")),
        (_hand("python-3-11", "Python 3.11"), _hand("python-3-12", "Python 3.12")),
        (_hand("phase-1", "Phase 1"), _hand("phase-2", "Phase 2")),
        (_hand("watch-1", "Watch 1"), _hand("watch-2", "Watch 2")),
        (_hand("js", "JS"), _hand("ts", "TS")),
        (_hand("nextjs", "Next.js"), _hand("nuxtjs", "Nuxt.js")),
        ({"id": "tamsin", "type": "Deckhand"}, {"id": "tamsyn", "type": "Deckhand"}),
    ],
    ids=[
        "two-given-names",
        "two-family-names",
        "two-databases",
        "two-roman-numerals",
        "two-versions",
        "two-numbered-phases",
        "two-numbered-watches",
        "two-short-names",
        "two-frameworks",
        "no-names",
    ],
)
def test_two_ids_one_edit_apart_under_two_names_stay_two(first, second):
    entities, _ = _run([first, second])

    assert entities == [first, second]


def test_two_names_whose_ids_slug_alike_are_not_merged_and_the_dedupe_step_joins_them():
    # `c` and `c#` slug to one id. The merge leaves them, since their names differ: the
    # edge between them is still there, where the merge would have dropped it. The
    # dedupe step then keeps the first entity stored under the id. A pack that tells
    # the two apart gives the second its own id before that (`test_id_rails.py`).
    entities, relationships = _run(
        [_hand("c", "C"), _hand("c#", "C#")], [_edge("c", "c#", "OUTRANKS")]
    )

    assert entities == [_hand("c", "C")]
    assert relationships == [_edge("c", "c", "OUTRANKS")]


def test_one_name_is_not_enough_when_the_ids_are_further_apart():
    entities, _ = _run([_hand("cook-tamsin", "Tamsin"), TAMSYN])

    assert _ids(entities) == ["cook-tamsin", "tamsyn"]


@pytest.mark.parametrize(
    "later_id", ["tomsyn", "tasmin"], ids=["two-letters-changed", "two-letters-swapped"]
)
def test_ids_two_edits_apart_under_one_name_are_not_merged(later_id):
    entities, _ = _run([TAMSIN, _hand(later_id, "Tamsin")])

    assert _ids(entities) == ["tamsin", later_id]


# --- what stays two, under one name: a digit, and a one-character token ---------------------------


@pytest.mark.parametrize(
    ("first_id", "second_id", "name"),
    [
        ("python-3-11", "python-3-12", "Python"),
        ("sprint-1", "sprint-12", "Sprint"),
        ("sprint-12", "sprint-1", "Sprint"),
        ("crate7", "crates", "Crate"),
        ("crates", "crate7", "Crate"),
    ],
    ids=[
        "a-digit-replaced",
        "a-digit-added",
        "a-digit-dropped",
        "a-digit-for-a-letter",
        "a-letter-for-a-digit",
    ],
)
def test_an_edit_that_adds_drops_or_replaces_a_digit_is_not_merged(first_id, second_id, name):
    entities, _ = _run([_hand(first_id, name), _hand(second_id, name)])

    assert _ids(entities) == [first_id, second_id]


@pytest.mark.parametrize(
    ("first_id", "second_id", "name"),
    [
        ("team-a", "team-b", "Team"),
        ("sam-r", "sam-t", "Sam"),
        ("k-shed", "m-shed", "Shed"),
        ("watch-a-crew", "watch-b-crew", "Crew"),
        ("k", "m", "Berth"),
    ],
    ids=["at-the-end", "an-initial", "at-the-start", "in-the-middle", "the-whole-id"],
)
def test_an_edit_that_replaces_a_one_character_token_is_not_merged(first_id, second_id, name):
    entities, _ = _run([_hand(first_id, name), _hand(second_id, name)])

    assert _ids(entities) == [first_id, second_id]


@pytest.mark.parametrize(
    ("first_id", "second_id", "name"),
    [("teama", "teamb", "Team"), ("phase-i", "phase-ii", "Phase"), ("team-a", "team-ab", "Team")],
    ids=["a-mark-inside-a-token", "a-letter-added-to-a-mark", "a-letter-added-after-a-mark"],
)
def test_a_mark_that_is_not_a_token_of_its_own_or_is_not_replaced_is_merged(
    first_id, second_id, name
):
    # On record: the two refusals stop a number and a replaced one-character token, and
    # nothing else. These pairs may be two things, and under one name they are merged.
    entities, _ = _run([_hand(first_id, name), _hand(second_id, name)])

    assert _ids(entities) == [first_id]


# --- the types, and what takes no part ------------------------------------------------------------


def test_two_types_are_never_merged():
    entities, _ = _run([_hand("corwen", "Corwen"), _pet("corwin", "Corwen")], pack=PET)

    assert _ids(entities) == ["corwen", "corwin"]


def test_a_type_an_id_exempt_rule_names_is_never_merged():
    quays = [
        {"id": "far-quay", "name": "Far Quay", "type": "Mooring"},
        {"id": "far-quai", "name": "Far Quay", "type": "Mooring"},
    ]

    entities, _ = _run(quays)

    assert entities == quays


def test_the_exempt_types_are_read_from_the_packs_rules():
    pets = [_pet("biscuit", "Biscuit"), _pet("biscuet", "Biscuit")]
    exempting = build_pack(
        manifest=PET.manifest,
        node_types=PET.node_types,
        edge_types=PET.edge_types,
        extractable_node_types=PET.extractable_node_types,
        extractable_edge_types=PET.extractable_edge_types,
        rules=(
            *PET.rules,
            ConventionRule(
                id="pet-exempt", kind="id_exempt", phase=ID, params=IdExempt(types=("Pet",))
            ),
        ),
        plugins=PET.plugins,
    )

    assert _ids(_run(pets, pack=PET)[0]) == ["biscuit"]
    assert _ids(_run(pets, pack=exempting)[0]) == ["biscuit", "biscuet"]


def test_the_user_anchor_takes_no_part():
    entities, _ = _run([_hand("user", "User"), _hand("users", "User")])

    assert _ids(entities) == ["user", "users"]


def test_a_reserved_identity_takes_no_part():
    # The first is reserved by its id, so the two can share a name that is not reserved.
    entities, _ = _run([_hand("harbour-master", "Brannoch"), _hand("harbour-masters", "Brannoch")])

    assert _ids(entities) == ["harbourmaster", "harbour-masters"]


def test_a_retired_type_is_read_as_its_replacement():
    entities, _ = _run(
        [
            {"id": "jem", "name": "Jem", "type": "Stowaway"},
            {"id": "jen", "name": "Jem", "type": "Deckhand"},
        ]
    )

    assert entities == [
        {"id": "jem", "name": "Jem", "type": "Deckhand", "properties": {"rank": "none"}}
    ]


def test_an_entity_with_no_id_field_takes_no_part():
    # No endpoint can name it, so it can neither be kept for another nor be remapped.
    entities, _ = _run([{"name": "Tamsin", "type": "Deckhand"}, TAMSYN])

    assert _ids(entities) == ["tamsin", "tamsyn"]


def test_an_entity_that_repeats_a_written_id_follows_the_entity_that_first_wrote_it():
    again = _hand("tamsyn", "Tamsin of the night watch")

    merged_away, _ = _run([TAMSIN, TAMSYN, again])
    kept, _ = _run([TAMSYN, WEND, again])

    assert merged_away == [TAMSIN]
    assert kept == [TAMSYN, WEND]


def test_ids_that_come_within_one_edit_only_as_stored_are_not_merged():
    # The merge reads the ids the model wrote, before the id phase drops a leading word.
    entities, _ = _run([_hand("the-tamsin", "Tamsin"), TAMSYN])

    assert _ids(entities) == ["tamsin", "tamsyn"]


# --- the tie rules ------------------------------------------------------------------------------


def test_a_chain_whose_ends_are_two_edits_apart_keeps_both_ends_when_an_end_comes_first():
    # `marlow` goes to `marlo`. `marlowe` is one edit from `marlow`, which is gone, and
    # two from `marlo`: an id is merged only into a kept id it is one edit from.
    entities, relationships = _run(
        [_hand("marlo", "Marlow"), _hand("marlow", "Marlow"), _hand("marlowe", "Marlow"), WEND],
        [_edge("wend", "marlow", "KNOWS_HAND"), _edge("wend", "marlowe", "KNOWS_HAND")],
    )

    assert _ids(entities) == ["marlo", "marlowe", "wend"]
    assert relationships == [
        _edge("wend", "marlo", "KNOWS_HAND"),
        _edge("wend", "marlowe", "KNOWS_HAND"),
    ]


def test_a_chain_whose_ends_are_two_edits_apart_is_one_entity_when_the_middle_comes_first():
    entities, _ = _run(
        [_hand("marlow", "Marlow"), _hand("marlo", "Marlow"), _hand("marlowe", "Marlow")]
    )

    assert _ids(entities) == ["marlow"]


def test_an_entity_within_one_edit_of_two_kept_entities_goes_to_the_earlier():
    entities, relationships = _run(
        [TAMSIN, _hand("tomsyn", "Tamsin"), WEND, _hand("tomsin", "Tamsin")],
        [_edge("wend", "tomsin", "KNOWS_HAND")],
    )

    assert _ids(entities) == ["tamsin", "tomsyn", "wend"]
    assert relationships == [_edge("wend", "tamsin", "KNOWS_HAND")]


# --- the edges of a merged entity -----------------------------------------------------------------


def test_a_merged_entitys_edges_are_remapped_and_then_deduplicated_first_wins():
    entities, relationships = _run(
        [TAMSIN, WEND, TAMSYN],
        [
            _edge("wend", "tamsin", "KNOWS_HAND", confidence=0.9),
            _edge("wend", "tamsyn", "KNOWS_HAND", confidence=0.4),
            _edge("tamsyn", "wend", "OUTRANKS"),
        ],
    )

    assert _ids(entities) == ["tamsin", "wend"]
    assert relationships == [
        _edge("wend", "tamsin", "KNOWS_HAND", confidence=0.9),
        _edge("tamsin", "wend", "OUTRANKS"),
    ]


def test_an_edge_between_the_two_merged_entities_is_dropped():
    # The two are one entity now, so the edge would join it to itself.
    entities, relationships = _run(
        [TAMSIN, WEND, TAMSYN],
        [_edge("tamsin", "tamsyn", "KNOWS_HAND"), _edge("wend", "tamsyn", "KNOWS_HAND")],
    )

    assert _ids(entities) == ["tamsin", "wend"]
    assert relationships == [_edge("wend", "tamsin", "KNOWS_HAND")]


def test_an_edge_the_model_wrote_from_an_entity_to_itself_is_not_the_merges_to_drop():
    _, relationships = _run([TAMSIN, WEND], [_edge("wend", "wend", "KNOWS_HAND")])

    assert relationships == [_edge("wend", "wend", "KNOWS_HAND")]


# --- the merge runs in the payload phase, before the relation rules -------------------------------


def test_a_fan_out_reads_the_kept_id_of_a_merged_member():
    _, relationships = _run(
        [TAMSIN, _hand("brannoch", "Brannoch"), TAMSYN],
        [_edge("tamsyn", "brannoch", "CREWS_WITH")],
    )

    assert relationships == [
        _edge("tamsin", "brannoch", "CREWS_WITH"),
        _edge("tamsin", "brannoch", "SERVES_UNDER"),
    ]


def test_a_batch_that_holds_the_merged_id_places_the_kept_entity_under_its_lead_in():
    # `pip` is written under the neighbours' lead-in and `pipp` under the roster's. They
    # are one pet under one name, which the roster's batch therefore holds.
    pip = PET_ROSTER_CASE.entities[-1]
    pipp = _pet("pipp", pip["name"])
    record = copy.deepcopy(PET_ROSTER_CASE.batch_record)
    record[0]["payload"]["entities"].append(pipp)
    entities = (pip, *PET_ROSTER_CASE.entities[:-1], pipp)

    out_entities, relationships = _run(
        entities,
        PET_ROSTER_CASE.relationships,
        pack=PET,
        scope=PET_ROSTER_CASE.scope,
        batch_record=record,
        turn_plan=PET_ROSTER_CASE.turn_plan,
    )

    assert _ids(out_entities) == ["pip", "user", "biscuit", "mango", "corwen", "sable"]
    assert [r["target"] for r in relationships if r["source"] == "user"] == [
        "biscuit",
        "pip",
        "mango",
    ]


# --- determinism ----------------------------------------------------------------------------------

_MERGING_PAYLOADS = {
    "a-pair": (
        [TAMSIN, WEND, TAMSYN],
        [_edge("wend", "tamsyn", "KNOWS_HAND"), _edge("tamsin", "tamsyn", "OUTRANKS")],
    ),
    "a-chain": (
        [_hand("marlo", "Marlow"), _hand("marlow", "Marlow"), _hand("marlowe", "Marlow"), WEND],
        [_edge("wend", "marlow", "KNOWS_HAND"), _edge("wend", "marlowe", "KNOWS_HAND")],
    ),
    "a-crew": (
        [TAMSIN, _hand("brannoch", "Brannoch"), TAMSYN],
        [_edge("tamsyn", "brannoch", "CREWS_WITH")],
    ),
}


@pytest.mark.parametrize("name", sorted(_MERGING_PAYLOADS))
def test_the_same_row_gives_the_same_bytes(name):
    entities, relationships = _MERGING_PAYLOADS[name]

    assert json.dumps(_run(entities, relationships)) == json.dumps(_run(entities, relationships))
