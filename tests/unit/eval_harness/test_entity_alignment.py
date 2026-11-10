"""The entity aligner, on hand-built probes with invented names.

Each test builds its gold and produced entity lists by hand. No gold file in the repository
and no run output is opened.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness import entity_alignment as ea  # noqa: E402
from scripts.eval_harness.entity_alignment import (  # noqa: E402
    AlignedPair,
    EntityRef,
    align_entities,
    candidate_pairs,
    event_date_equal,
    match_relations,
    metric_equal,
)

Triple = tuple[str, str, str]


def _ref(
    entity_id: str,
    entity_type: str | None = "Concept",
    *,
    alternatives: tuple[str, ...] = (),
    occurred_on: str | None = None,
) -> EntityRef:
    return EntityRef(entity_id, entity_type, alternatives, occurred_on)


def _aligned(
    gold: list[EntityRef],
    produced: list[EntityRef],
    accepted: tuple[tuple[str, str], ...] = (),
) -> set[Triple]:
    return {
        (p.gold_id, p.produced_id, p.kind)
        for p in align_entities(gold, produced, accepted=accepted)
    }


def _candidates(gold: list[EntityRef], produced: list[EntityRef]) -> set[Triple]:
    return {(p.gold_id, p.produced_id, p.kind) for p in candidate_pairs(gold, produced)}


# ---------------------------------------------------------------------------
# The kinds
# ---------------------------------------------------------------------------


def test_the_kinds_their_order_and_their_sets() -> None:
    assert ea.ALIGNMENT_KINDS == (
        "exact",
        "alternative",
        "event-date",
        "metric",
        "n1",
        "reorder",
        "typo",
        "paraphrase",
    )
    assert frozenset({"exact", "alternative", "event-date", "metric"}) == (
        ea.EXCLUSION_EXEMPT_KINDS
    )
    assert frozenset({"typo", "paraphrase"}) == ea.PENDING_KINDS
    assert frozenset({"RELATED_TO", "WORKS_WITH"}) == ea.UNDIRECTED_PREDICATES


KIND_CASES = [
    ("exact", _ref("harbor-plan", "Project"), _ref("harbor-plan", "Project")),
    (
        "alternative",
        _ref("kestrel-board", "Organization", alternatives=("kestrel-committee",)),
        _ref("kestrel-committee", "Organization"),
    ),
    (
        "event-date",
        _ref("quill-review", "Event", occurred_on="2026-03-14"),
        _ref("quill-review-2026-03-14", "Event", occurred_on="2026-03-14"),
    ),
    ("metric", _ref("42-ms-p50-latency", "Metric"), _ref("42-p50-latency-ms", "Metric")),
    ("n1", _ref("acme-corp", "Organization"), _ref("the-acme-corp", "Organization")),
    ("reorder", _ref("day-trip-to-varlo", "Event"), _ref("varlo-day-trip", "Event")),
    ("typo", _ref("kestrel-board", "Project"), _ref("kestral-board", "Project")),
    ("paraphrase", _ref("quarry-notes", "Document"), _ref("quarry-notes-draft", "Document")),
]


@pytest.mark.parametrize(("kind", "gold", "produced"), KIND_CASES, ids=[c[0] for c in KIND_CASES])
def test_each_kind_is_offered(kind: str, gold: EntityRef, produced: EntityRef) -> None:
    assert _candidates([gold], [produced]) == {(gold.id, produced.id, kind)}


@pytest.mark.parametrize(
    ("gold", "produced", "kind"),
    [
        # A Metric id in another order is a metric pair before it is a reorder pair.
        (_ref("42-ms-p50-latency", "Metric"), _ref("42-p50-latency-ms", "Metric"), "metric"),
        # A leading article is an n1 pair before it is a paraphrase pair.
        (_ref("acme-corp", "Organization"), _ref("the-acme-corp", "Organization"), "n1"),
        # An alternative that is also a typo is an alternative.
        (
            _ref("kestrel-board", "Project", alternatives=("kestral-board",)),
            _ref("kestral-board", "Project"),
            "alternative",
        ),
    ],
)
def test_a_pair_takes_its_highest_kind(gold: EntityRef, produced: EntityRef, kind: str) -> None:
    assert _candidates([gold], [produced]) == {(gold.id, produced.id, kind)}


def test_exact_outranks_another_gold_entitys_alternative() -> None:
    gold = [_ref("orbit", alternatives=("lumen",)), _ref("lumen")]
    assert _aligned(gold, [_ref("lumen")]) == {("lumen", "lumen", "exact")}


def test_global_ranking_keeps_an_exact_match_from_an_earlier_gold_entitys_metric() -> None:
    # `12-latency-ms` sorts first and is a metric pair of the produced id, which is
    # `12-ms-latency`'s exact match; ranking by kind first gives it to the exact one.
    gold = [_ref("12-latency-ms", "Metric"), _ref("12-ms-latency", "Metric")]
    produced = [_ref("12-ms-latency", "Metric")]
    assert ("12-latency-ms", "12-ms-latency", "metric") in _candidates(gold, produced)
    assert _aligned(gold, produced) == {("12-ms-latency", "12-ms-latency", "exact")}


def test_a_produced_id_another_gold_entity_matches_by_a_rail_kind_is_reserved() -> None:
    # `the-harbor-crew` is `harbor-team`'s alternative, so `harbor-crew` may not take it by
    # n1, though `harbor-team` takes its exact match and leaves the alternative unused.
    gold = [
        _ref("harbor-team", "Organization", alternatives=("the-harbor-crew",)),
        _ref("harbor-crew", "Organization"),
    ]
    produced = [_ref("harbor-team", "Organization"), _ref("the-harbor-crew", "Organization")]
    assert ("harbor-crew", "the-harbor-crew", "n1") not in _candidates(gold, produced)
    assert _aligned(gold, produced) == {("harbor-team", "harbor-team", "exact")}


def test_one_to_one() -> None:
    gold = [_ref("acme-corp", "Organization")]
    produced = [_ref("the-acme-corp", "Organization"), _ref("my-acme-corp", "Organization")]
    assert _aligned(gold, produced) == {("acme-corp", "my-acme-corp", "n1")}


# ---------------------------------------------------------------------------
# The exclusions on kinds 5 to 8
# ---------------------------------------------------------------------------


def test_merged_id_is_excluded() -> None:
    gold = [_ref("alice", "Person"), _ref("bob", "Person")]
    assert _candidates(gold, [_ref("alice-bob", "Person")]) == set()


def test_one_shared_word_never_aligns() -> None:
    gold = [_ref("acme-corp", "Organization")]
    assert _candidates(gold, [_ref("acme-labs", "Organization")]) == set()


def test_differing_digits_never_align() -> None:
    gold = [_ref("p50-latency", "Metric")]
    assert _candidates(gold, [_ref("p95-latency", "Metric")]) == set()


@pytest.mark.parametrize(
    ("gold", "produced"),
    [
        (_ref("acme-corp", "Organization"), _ref("the-acme-corp", "Person")),
        (_ref("day-trip-to-varlo", "Event"), _ref("varlo-day-trip", "Person")),
        (_ref("kestrel-board", "Project"), _ref("kestral-board", "Person")),
        (_ref("quarry-notes", "Document"), _ref("quarry-notes-draft", "Person")),
    ],
    ids=["n1", "reorder", "typo", "paraphrase"],
)
def test_an_incompatible_type_excludes_kinds_5_to_8(gold: EntityRef, produced: EntityRef) -> None:
    assert _candidates([gold], [produced]) == set()


def test_an_exact_id_aligns_across_differing_types() -> None:
    # Exact scoring never reads entity types, so aligned scoring must not lose what it finds.
    gold = [_ref("dark-mode", "Preference")]
    assert _aligned(gold, [_ref("dark-mode", "Concept")]) == {("dark-mode", "dark-mode", "exact")}


def test_a_gold_alternative_aligns_across_differing_types() -> None:
    gold = [_ref("kestrel-board", "Organization", alternatives=("kestrel-committee",))]
    assert _aligned(gold, [_ref("kestrel-committee", "Concept")]) == {
        ("kestrel-board", "kestrel-committee", "alternative")
    }


def test_an_untyped_produced_id_is_compatible() -> None:
    gold = [_ref("acme-corp", "Organization")]
    assert _candidates(gold, [_ref("the-acme-corp", None)]) == {
        ("acme-corp", "the-acme-corp", "n1")
    }


def test_a_dated_event_id_is_not_excluded_as_merged_id() -> None:
    # The date's own gold entity holds the id's extra words; event-date is checked first.
    gold = [
        _ref("quill-review", "Event", occurred_on="2026-03-14"),
        _ref("2026-03-14", "Date"),
    ]
    produced = [_ref("quill-review-2026-03-14", "Event", occurred_on="2026-03-14")]
    assert _aligned(gold, produced) == {("quill-review", "quill-review-2026-03-14", "event-date")}


def test_a_metric_id_is_not_excluded_as_merged_id() -> None:
    # `request-queue` holds the produced id's extra words, but metric is checked first.
    gold = [_ref("9500-rps", "Metric"), _ref("request-queue", "Concept")]
    produced = [_ref("9500-requests-per-second", "Metric")]
    assert _aligned(gold, produced) == {("9500-rps", "9500-requests-per-second", "metric")}


# ---------------------------------------------------------------------------
# metric, n1 and event-date
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gold_id", "produced_id", "equal"),
    [
        ("38-ms-p50-latency", "38-p50-latency-ms", True),
        ("9500-rps", "9500-requests-per-second", True),
        ("120-milliseconds-p95", "120-ms-p95", True),
        ("0-2-percent-error-rate", "02-percent-error-rate", True),
        ("38-ms-p50-latency", "39-ms-p50-latency", False),
        ("38", "38", False),
        ("71-percent-cpu", "71-percent-peak-cpu", False),
        ("ms-latency", "latency-ms", False),
    ],
)
def test_metric_equal(gold_id: str, produced_id: str, equal: bool) -> None:
    assert metric_equal(gold_id, produced_id) is equal


def test_a_strict_subset_of_measure_words_is_offered_as_pending_paraphrase() -> None:
    gold = [_ref("71-percent-cpu", "Metric")]
    produced = [_ref("71-percent-peak-cpu", "Metric")]
    assert _candidates(gold, produced) == {("71-percent-cpu", "71-percent-peak-cpu", "paraphrase")}
    assert _aligned(gold, produced) == set()


@pytest.mark.parametrize(
    ("gold_id", "produced_id"),
    [("acme-corp", "the-acme-corp"), ("reading-list", "my-reading-list"), ("the-lab", "lab")],
)
def test_n1_drops_a_leading_article_or_possessive(gold_id: str, produced_id: str) -> None:
    assert _candidates([_ref(gold_id)], [_ref(produced_id)]) == {(gold_id, produced_id, "n1")}


def test_n1_keeps_an_article_inside_the_id() -> None:
    gold = [_ref("watering-plants", "Activity")]
    produced = [_ref("watering-the-plants", "Activity")]
    assert _candidates(gold, produced) == {("watering-plants", "watering-the-plants", "paraphrase")}


@pytest.mark.parametrize(
    ("gold", "produced", "equal"),
    [
        (
            _ref("quill-review", "Event", occurred_on="2026-03-14"),
            _ref("quill-review-2026-03-14", "Event", occurred_on="2026-03-14"),
            True,
        ),
        (
            _ref("quill-review-2026-03-14", "Event", occurred_on="2026-03-14"),
            _ref("quill-review", "Event", occurred_on="2026-03-14"),
            True,
        ),
        # Gold names no date for the trip, so a dated produced id is another occurrence.
        (
            _ref("varlo-trip", "Event"),
            _ref("varlo-trip-2026-12-10", "Event", occurred_on="2026-12-10"),
            False,
        ),
        (
            _ref("varlo-trip", "Event", occurred_on="2026-12-10"),
            _ref("varlo-trip-2026-12-10", "Event"),
            False,
        ),
        (
            _ref("varlo-trip", "Event", occurred_on="2026-12-11"),
            _ref("varlo-trip-2026-12-10", "Event", occurred_on="2026-12-10"),
            False,
        ),
    ],
)
def test_event_date_equal(gold: EntityRef, produced: EntityRef, equal: bool) -> None:
    assert event_date_equal(gold, produced) is equal


def test_a_gold_event_with_no_date_never_aligns_with_a_dated_id() -> None:
    gold = [_ref("varlo-trip", "Event")]
    produced = [_ref("varlo-trip-2026-12-10", "Event", occurred_on="2026-12-10")]
    assert _aligned(gold, produced) == set()


# ---------------------------------------------------------------------------
# Pending kinds
# ---------------------------------------------------------------------------


def test_a_pending_pair_is_listed_not_counted_until_accepted() -> None:
    gold = [_ref("kestrel-board", "Project")]
    produced = [_ref("kestral-board", "Project")]
    assert _candidates(gold, produced) == {("kestrel-board", "kestral-board", "typo")}
    assert _aligned(gold, produced) == set()
    accepted = (("kestrel-board", "kestral-board"),)
    assert _aligned(gold, produced, accepted) == {("kestrel-board", "kestral-board", "typo")}


def test_an_unaccepted_pending_pair_does_not_hold_its_produced_id() -> None:
    # `harbor-plan` sorts first and its typo pair outranks `plan`'s paraphrase pair, so it
    # would take the produced id if an unaccepted pair stayed a candidate.
    gold = [_ref("harbor-plan", "Project"), _ref("plan", "Project")]
    produced = [_ref("harbr-plan", "Project")]
    assert _candidates(gold, produced) == {
        ("harbor-plan", "harbr-plan", "typo"),
        ("plan", "harbr-plan", "paraphrase"),
    }
    accepted = (("plan", "harbr-plan"),)
    assert _aligned(gold, produced, accepted) == {("plan", "harbr-plan", "paraphrase")}


def test_a_rejected_pending_pair_leaves_the_id_to_an_n1_or_reorder_match() -> None:
    gold = [_ref("kestrel-board", "Project"), _ref("board-kestral", "Project")]
    produced = [_ref("kestral-board", "Project")]
    assert _aligned(gold, produced) == {("board-kestral", "kestral-board", "reorder")}


# ---------------------------------------------------------------------------
# Relations over the alignment
# ---------------------------------------------------------------------------


def _pairs(*rows: Triple) -> tuple[AlignedPair, ...]:
    return tuple(AlignedPair(g, p, k) for g, p, k in rows)  # type: ignore[arg-type]


def test_relations_match_through_the_alignment() -> None:
    pairs = _pairs(("user", "user", "exact"), ("acme-corp", "the-acme-corp", "n1"))
    result = match_relations(
        [("user", "WORKS_AT", "acme-corp")],
        [("user", "WORKS_AT", "the-acme-corp"), ("user", "WORKS_AT", "zenith")],
        pairs,
    )
    assert result.found == frozenset({("user", "WORKS_AT", "acme-corp")})
    assert result.missed == frozenset()
    assert result.extra == frozenset({("user", "WORKS_AT", "zenith")})


def test_a_predicate_alternative_matches() -> None:
    pairs = _pairs(("user", "user", "exact"), ("orbit", "orbit", "exact"))
    gold = [("user", "USES", "orbit")]
    produced = [("user", "WORKS_ON", "orbit")]
    assert match_relations(gold, produced, pairs).found == frozenset()
    result = match_relations(
        gold, produced, pairs, predicate_alternatives={("user", "USES", "orbit"): ("WORKS_ON",)}
    )
    assert result.found == frozenset(gold)
    assert result.extra == frozenset()


@pytest.mark.parametrize("predicate", sorted(ea.UNDIRECTED_PREDICATES))
def test_an_undirected_predicate_matches_either_way(predicate: str) -> None:
    pairs = _pairs(("alice", "alice", "exact"), ("bob", "bob", "exact"))
    result = match_relations([("alice", predicate, "bob")], [("bob", predicate, "alice")], pairs)
    assert result.found == frozenset({("alice", predicate, "bob")})


def test_a_directed_predicate_does_not_match_reversed() -> None:
    pairs = _pairs(("alice", "alice", "exact"), ("bob", "bob", "exact"))
    result = match_relations([("alice", "KNOWS", "bob")], [("bob", "KNOWS", "alice")], pairs)
    assert result.found == frozenset()
    assert result.missed == frozenset({("alice", "KNOWS", "bob")})
    assert result.extra == frozenset({("bob", "KNOWS", "alice")})


def test_each_produced_key_matches_at_most_one_gold_key() -> None:
    pairs = _pairs(("alice", "alice", "exact"), ("bob", "bob", "exact"))
    gold = [("alice", "WORKS_WITH", "bob"), ("bob", "WORKS_WITH", "alice")]
    result = match_relations(gold, [("alice", "WORKS_WITH", "bob")], pairs)
    assert result.found == frozenset({("alice", "WORKS_WITH", "bob")})
    assert result.missed == frozenset({("bob", "WORKS_WITH", "alice")})


def test_an_unaligned_endpoint_is_never_found() -> None:
    pairs = _pairs(("user", "user", "exact"))
    result = match_relations([("user", "USES", "orbit")], [("user", "USES", "orbit")], pairs)
    assert result.found == frozenset()
    assert result.extra == frozenset({("user", "USES", "orbit")})


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_input_order_does_not_change_the_pairs(seed: int) -> None:
    gold = [
        _ref("user", "User"),
        _ref("acme-corp", "Organization"),
        _ref("12-latency-ms", "Metric"),
        _ref("12-ms-latency", "Metric"),
        _ref("harbor-plan", "Project"),
        _ref("plan", "Project"),
        _ref("quill-review", "Event", occurred_on="2026-03-14"),
    ]
    produced = [
        _ref("user", "User"),
        _ref("the-acme-corp", "Organization"),
        _ref("my-acme-corp", "Organization"),
        _ref("12-ms-latency", "Metric"),
        _ref("harbr-plan", "Project"),
        _ref("quill-review-2026-03-14", "Event", occurred_on="2026-03-14"),
    ]
    accepted = (("plan", "harbr-plan"),)
    expected = align_entities(gold, produced, accepted=accepted)
    rng = random.Random(seed)
    shuffled_gold, shuffled_produced = gold[:], produced[:]
    rng.shuffle(shuffled_gold)
    rng.shuffle(shuffled_produced)
    assert align_entities(shuffled_gold, produced, accepted=accepted) == expected
    assert align_entities(gold, shuffled_produced, accepted=accepted) == expected
    assert align_entities(shuffled_gold, shuffled_produced, accepted=accepted) == expected
    assert candidate_pairs(shuffled_gold, shuffled_produced) == candidate_pairs(gold, produced)
