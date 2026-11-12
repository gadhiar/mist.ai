"""The near-duplicate merge: two entities of one type under one name, one edit apart, become one.

The model writes one entity under two spellings of its id in two batches of a turn. The
merge joins two written ids when they have one type, their slugs are within one edit,
and their names are equal, read in one case and with single spaces. It keeps the
spelling a fixed order reads first, never the payload's: the longer slug, then the
smaller slug, then the smaller written id. It refuses a type an `id_exempt` rule names,
an edit that adds, drops or replaces a digit, and an edit that replaces a one-character
token. A pin of which spelling is kept runs in every order of its entities.

The cases run on the fixture packs, whose types are invented, so a case that passes
shows the runner reading the pack's `id_exempt` rules and not knowing a type. No rule
of those packs joins a pair below. The cases on today's ontology are in
`test_id_rails.py`. Every name is invented.
"""

from __future__ import annotations

import copy
import json
from itertools import permutations

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
from tests.unit.knowledge.extraction.test_rails_own_context import NO_ALIAS
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


def _every_order(entities, relationships=(), pack: Pack = BASE) -> set:
    """What the row stores in each order of its entities: sorted ids, sorted edges."""
    return {
        (
            tuple(sorted(_ids(stored))),
            tuple(sorted((r["source"], r["target"], r["type"]) for r in edges)),
        )
        for stored, edges in (
            _run(order, relationships, pack=pack) for order in permutations(entities)
        )
    }


TAMSIN = _hand("tamsin", "Tamsin")
# The same hand under a second spelling of the id: the model's name for her is the same.
TAMSYN = _hand("tamsyn", "Tamsin")
WEND = _hand("wend", "Wend")


# --- what merges --------------------------------------------------------------------------------


def test_two_ids_one_letter_apart_under_one_name_are_one_entity_the_longer_spelling():
    # One hand, written `tamsin` in one batch and `tamsn` in the next.
    entities, relationships = _run(
        [TAMSIN, WEND, _hand("tamsn", "Tamsin")], [_edge("wend", "tamsn", "KNOWS_HAND")]
    )

    assert entities == [TAMSIN, WEND]
    assert relationships == [_edge("wend", "tamsin", "KNOWS_HAND")]


@pytest.mark.parametrize(
    ("other_id", "kept"),
    [("tamsyn", "tamsin"), ("tamsinn", "tamsinn"), ("tamsi", "tamsin"), ("amsin", "tamsin")],
    ids=["a-letter-changed", "a-letter-added", "the-last-letter-lost", "the-first-letter-lost"],
)
def test_one_edit_is_one_letter_changed_added_or_lost(other_id, kept):
    # The fixed order keeps the longer slug, and of two of one length the smaller.
    other = _hand(other_id, "Tamsin")

    assert _ids(_run([TAMSIN, other])[0]) == [kept]
    assert _ids(_run([other, TAMSIN])[0]) == [kept]


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
    ("first_id", "second_id", "kept"),
    [("ky", "ko", "ko"), ("jem", "jen", "jem"), ("wend", "wand", "wand")],
    ids=["two-characters", "three-characters", "four-characters"],
)
def test_two_short_ids_one_letter_apart_under_one_name_are_merged(first_id, second_id, kept):
    # Of two spellings of one length the smaller slug is kept.
    entities, _ = _run([_hand(first_id, "Wend"), _hand(second_id, "Wend")])

    assert _ids(entities) == [kept]


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


def test_the_spelling_the_fixed_order_reads_first_is_kept_whole_and_the_other_is_dropped():
    # The order decides, not the payload's order or which entity says more: `tamsin` is
    # the smaller of two slugs of one length, and the property of `tamsyn` is lost.
    other = _hand("tamsyn", "Tamsin", properties={"rank": "mate"})

    assert _run([other, TAMSIN])[0] == [TAMSIN]
    assert _run([TAMSIN, other])[0] == [TAMSIN]


# --- which spelling is kept: a fixed order, the same in every order of the payload -----------------


def test_a_letter_dropped_inside_a_token_keeps_the_longer_spelling_in_every_order():
    # The shape of a typo the model makes: one letter lost inside a word. The longer
    # spelling holds every letter either one wrote.
    pair = [_hand("brannoch", "Brannoch"), _hand("branoch", "Brannoch")]
    edges = [_edge("wend", "brannoch", "KNOWS_HAND"), _edge("wend", "branoch", "OUTRANKS")]

    assert _every_order([*pair, WEND], edges) == {
        (
            ("brannoch", "wend"),
            (("wend", "brannoch", "KNOWS_HAND"), ("wend", "brannoch", "OUTRANKS")),
        )
    }


def test_two_spellings_of_one_length_keep_the_smaller_slug_in_every_order():
    edges = [_edge("wend", "tamsyn", "KNOWS_HAND"), _edge("wend", "tamsin", "OUTRANKS")]

    assert _every_order([TAMSYN, TAMSIN, WEND], edges) == {
        (("tamsin", "wend"), (("wend", "tamsin", "KNOWS_HAND"), ("wend", "tamsin", "OUTRANKS")))
    }


def test_two_written_ids_with_one_slug_keep_the_smaller_written_id_in_both_orders():
    # `Tamsin` sorts before `tamsin`, so the entity written `Tamsin` is kept, and with it
    # its property.
    upper = {"id": "Tamsin", "name": "Tamsin", "type": "Deckhand", "properties": {"rank": "mate"}}
    lower = _hand("tamsin", "Tamsin")
    stored = [
        {"id": "tamsin", "name": "Tamsin", "type": "Deckhand", "properties": {"rank": "mate"}}
    ]

    assert _run([upper, lower])[0] == stored
    assert _run([lower, upper])[0] == stored


def test_a_written_id_repeated_under_two_names_is_read_under_both_in_every_order():
    # `tamsyn` shares the name `Tamsin` with one writer of `tamsin`, whichever writer of
    # `tamsin` the payload lists first, so the two ids are one entity in every order.
    entities = [
        _hand("tamsin", "Tamsin"),
        _hand("tamsin", "Tamsin Hale"),
        _hand("tamsyn", "Tamsin"),
        WEND,
    ]
    edges = [_edge("wend", "tamsyn", "KNOWS_HAND"), _edge("wend", "tamsin", "KNOWS_HAND")]

    assert _every_order(entities, edges, pack=NO_ALIAS) == {
        (("tamsin", "wend"), (("wend", "tamsin", "KNOWS_HAND"),))
    }


def test_a_written_id_one_of_whose_writers_is_of_an_exempt_type_takes_no_part():
    entities = [
        {"id": "tamsin", "name": "Tamsin", "type": "Mooring"},
        _hand("tamsin", "Tamsin"),
        _hand("tamsyn", "Tamsin"),
        WEND,
    ]
    edges = [_edge("wend", "tamsyn", "KNOWS_HAND"), _edge("wend", "tamsin", "KNOWS_HAND")]

    assert _every_order(entities, edges, pack=NO_ALIAS) == {
        (
            ("tamsin", "tamsyn", "wend"),
            (("wend", "tamsin", "KNOWS_HAND"), ("wend", "tamsyn", "KNOWS_HAND")),
        )
    }


def test_a_written_id_whose_writers_are_of_two_types_takes_no_part():
    # The pack's alias stores the Deckhand `bosun` as `boatswain`, and the `Galley` one,
    # a type the pack does not define, keeps `bosun`. Merging `bosun` into `bosuns`
    # would drop a writer of another type. Which stored id the edge to `bosun` names
    # follows the id phase, so the pin is on the entity ids.
    entities = [
        _hand("bosun", "Bosun"),
        {"id": "bosun", "name": "Bosun", "type": "Galley"},
        _hand("bosuns", "Bosun"),
        WEND,
    ]
    edges = [_edge("wend", "bosun", "KNOWS_HAND")]

    stored = {tuple(sorted(_ids(_run(order, edges)[0]))) for order in permutations(entities)}

    assert stored == {("boatswain", "bosun", "bosuns", "wend")}


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
    ("first_id", "second_id", "name", "kept"),
    [
        ("teama", "teamb", "Team", "teama"),
        ("phase-i", "phase-ii", "Phase", "phase-ii"),
        ("team-a", "team-ab", "Team", "team-ab"),
    ],
    ids=["a-mark-inside-a-token", "a-letter-added-to-a-mark", "a-letter-added-after-a-mark"],
)
def test_a_mark_that_is_not_a_token_of_its_own_or_is_not_replaced_is_merged(
    first_id, second_id, name, kept
):
    # On record: the two refusals stop a number and a replaced one-character token, and
    # nothing else. These pairs may be two things, and under one name they are merged.
    entities, _ = _run([_hand(first_id, name), _hand(second_id, name)])

    assert _ids(entities) == [kept]


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

    # Under the pack itself the two are one, the smaller of two slugs of one length.
    assert _ids(_run(pets, pack=PET)[0]) == ["biscuet"]
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


def test_a_retired_writer_is_read_as_its_replacement_type_in_both_orders():
    # The base pack retires Stowaway for Deckhand, so a Stowaway `tamsyn` reads as a
    # Deckhand: it shares a reading with the Deckhand `tamsin` and merges into it.
    stowaway = {"id": "tamsyn", "name": "Tamsin", "type": "Stowaway"}

    assert _every_order([stowaway, TAMSIN]) == {(("tamsin",), ())}
    # One id written by a Stowaway and by a Deckhand is of one type, and takes part.
    assert _every_order([stowaway, TAMSYN, TAMSIN]) == {(("tamsin",), ())}


def test_an_entity_with_no_id_field_takes_no_part():
    # No endpoint can name it, so it can neither be kept for another nor be remapped.
    entities, _ = _run([{"name": "Tamsin", "type": "Deckhand"}, TAMSYN])

    assert _ids(entities) == ["tamsin", "tamsyn"]


def test_every_writer_of_a_written_id_follows_that_id():
    # Dropped with it when it is merged, and otherwise left to the dedupe step.
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


def test_a_chain_whose_ends_are_two_edits_apart_keeps_both_ends_when_an_end_is_read_first():
    # The longest, `marlowe`, is read first and kept, and `marlow` goes to it. `marlo`
    # is one edit from `marlow`, which is gone, and two from `marlowe`: an id is merged
    # only into a kept id it is one edit from. So in every order:
    chain = [_hand("marlo", "Marlow"), _hand("marlow", "Marlow"), _hand("marlowe", "Marlow")]
    edges = [_edge("wend", "marlow", "KNOWS_HAND"), _edge("wend", "marlowe", "KNOWS_HAND")]

    assert _every_order([*chain, WEND], edges) == {
        (("marlo", "marlowe", "wend"), (("wend", "marlowe", "KNOWS_HAND"),))
    }


def test_a_chain_whose_ends_are_two_edits_apart_is_one_entity_when_the_middle_is_read_first():
    # `marlow` is the longest, so it is read first; each end is one edit from it.
    chain = [_hand("marlo", "Marlow"), _hand("marlow", "Marlow"), _hand("marow", "Marlow")]

    assert _every_order(chain) == {(("marlow",), ())}


def test_an_entity_within_one_edit_of_two_kept_entities_goes_to_the_earlier():
    # `kerinn` and `kerrin` are two edits apart, so both are kept, `kerinn` first as
    # the smaller of one length. `kerin` is read last, as the shortest, and is one
    # edit from each: it goes to the earlier in that order.
    spellings = [_hand("kerrin", "Kerrin"), _hand("kerinn", "Kerrin"), _hand("kerin", "Kerrin")]

    assert _every_order([*spellings, WEND], [_edge("wend", "kerin", "KNOWS_HAND")]) == {
        (("kerinn", "kerrin", "wend"), (("wend", "kerinn", "KNOWS_HAND"),))
    }


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
    # `pip` is written under the neighbours' lead-in and `pi` under the roster's. They
    # are one pet under one name, kept as the longer `pip`, which the roster's batch
    # therefore holds.
    pip = PET_ROSTER_CASE.entities[-1]
    pi = _pet("pi", pip["name"])
    record = copy.deepcopy(PET_ROSTER_CASE.batch_record)
    record[0]["payload"]["entities"].append(pi)
    entities = (pip, *PET_ROSTER_CASE.entities[:-1], pi)

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
