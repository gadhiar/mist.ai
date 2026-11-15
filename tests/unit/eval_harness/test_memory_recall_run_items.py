"""`run_items`: each run item carries the fetch rank of the injected candidates it stands for.

The scorer's MRR reads that rank (plan v4, A26), so the block is built by hand here with the
three shapes where rank and block position part: a list whose facts share one rank, a note
served in two chunks, and a note ranked above the facts the block places before it.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from backend.chat.context_budget import ApproximateTokenCounter
from backend.memory.block import FactList, PlacedList
from backend.memory.fact_index import IndexedFact
from backend.memory.models import (
    Candidate,
    CandidateDisposition,
    CandidateSource,
    FactSource,
    FetchTimings,
    MemoryCandidate,
    MemoryItem,
    MemoryResult,
    MemoryStage,
    NoteSource,
)
from backend.memory.ranking import ScoredCandidate
from scripts.eval_harness.memory_recall import FIXTURE_DIR, load_probes, parse_run_record
from scripts.eval_harness.memory_recall_run import run_items, run_record


def _fact(fact_key: str, object_name: str) -> IndexedFact:
    return IndexedFact(
        row_key=f"{fact_key}-v1",
        fact_key=fact_key,
        subject_id="user",
        subject_name="Wren",
        subject_type="User",
        rel_type="HAS_HABIT",
        object_id=f"obj-{fact_key}",
        object_name=object_name,
        object_type="Habit",
        text=f"Wren has the habit: {object_name}.",
        valid_from=None,
        valid_to=None,
        recorded_at="2026-01-01T00:00:00+00:00",
        confidence=0.9,
        source_utterance_id="utt-1",
        content_hash=hashlib.sha256(fact_key.encode()).hexdigest(),
    )


def _fact_source(fact: IndexedFact) -> FactSource:
    return FactSource(
        fact_key=fact.fact_key,
        row_key=fact.row_key,
        recorded_at=fact.recorded_at,
        source_utterance_id=fact.source_utterance_id,
    )


def _scored(fact: IndexedFact, score: float) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=Candidate(
            kind="fact",
            item_id=fact.fact_key,
            text=fact.text,
            when=None,
            source=_fact_source(fact),
            ranks={"fts": 1},
            confidence=fact.confidence,
        ),
        score=score,
        relevance=score,
        reranked=True,
        time_term=0.0,
        window_applied=False,
        confidence=fact.confidence,
    )


def _fact_item(ref: str, fact: IndexedFact, score: float) -> MemoryItem:
    return MemoryItem(
        ref=ref,
        kind="fact",
        item_id=fact.fact_key,
        text=fact.text,
        when=None,
        score=score,
        source=_fact_source(fact),
    )


def _note_item(ref: str, path: str, heading: str, score: float) -> MemoryItem:
    return MemoryItem(
        ref=ref,
        kind="note",
        item_id=f"{path}#{heading}",
        text=f"{heading} body",
        when=None,
        score=score,
        source=NoteSource(path=path, heading=heading, authored_by="user"),
    )


def _candidate(
    source: CandidateSource, rank: int, disposition: CandidateDisposition, item: int | None
) -> MemoryCandidate:
    return MemoryCandidate(
        source=source,
        ref=f"{source}:{rank}",
        alt_ref=f"{source}:{rank}:alt" if source == "graph" else None,
        rank=rank,
        score=1.0 / rank,
        item=item,
        disposition=disposition,
    )


def _result(items: tuple[MemoryItem, ...], candidates: tuple[MemoryCandidate, ...]) -> MemoryResult:
    timings = FetchTimings()
    stage = MemoryStage(
        ran=True,
        skip_reason=None,
        level="full",
        tier="top",
        block_version=1,
        budget_tokens=500,
        tokens=120,
        degraded_reason=None,
        prefetch_reuse="none",
        reranked=len(candidates),
        suppressed=0,
        narrative_suppressed=0,
        timings=timings,
        candidates=candidates,
    )
    return MemoryResult(
        items=items,
        block_text="<memory>...</memory>",
        tokens=120,
        budget_tokens=500,
        degraded=False,
        degraded_reason=None,
        timings=timings,
        reranked=len(candidates),
        suppressed=0,
        narrative_suppressed=0,
        stage=stage,
    )


# The block in layout order: facts, then notes. m1 is a list of three siblings, two of them
# scored (ranks 3 and 5); m2 a fact at rank 4; m3 a note the fetcher ranked first, placed after
# both facts; m4 and m5 two chunks of one note, at ranks 6 and 2.
_WALK = _fact("fact-walk", "lunchtime walk")
_READ = _fact("fact-read", "reading before bed")
_SWIM = _fact("fact-swim", "morning swim")
_TEA = _fact("fact-tea", "green tea")
_LIST = PlacedList(
    ref="m1",
    facts=FactList(facts=(_WALK, _READ, _SWIM), scored=(_scored(_WALK, 0.7), _scored(_READ, 0.6))),
)
_ITEMS = (
    _fact_item("m1", _WALK, 0.7),
    _fact_item("m2", _TEA, 0.65),
    _note_item("m3", "notes/trip.md", "Plan", 0.9),
    _note_item("m4", "notes/garden.md", "Beds", 0.5),
    _note_item("m5", "notes/garden.md", "Seeds", 0.8),
)
# Injected first, then by rank, as the fetcher keeps them; ranks 7 and 8 were dropped.
_CANDIDATES = (
    _candidate("vault", 1, "injected", 3),
    _candidate("vault", 2, "injected", 5),
    _candidate("graph", 3, "injected", 1),
    _candidate("graph", 4, "injected", 2),
    _candidate("graph", 5, "injected", 1),
    _candidate("vault", 6, "injected", 4),
    _candidate("graph", 7, "dropped_budget", None),
    _candidate("vault", 8, "dropped_min_score", None),
)

_EXPECTED = [
    {"ref": "m1", "kind": "fact", "id": "fact-walk", "score": 0.7, "rank": 3},
    {"ref": "m1", "kind": "fact", "id": "fact-read", "score": 0.7, "rank": 3},
    {"ref": "m1", "kind": "fact", "id": "fact-swim", "score": 0.7, "rank": 3},
    {"ref": "m2", "kind": "fact", "id": "fact-tea", "score": 0.65, "rank": 4},
    {"ref": "m3", "kind": "note", "id": "notes/trip.md", "score": 0.9, "rank": 1},
    {"ref": "m4", "kind": "note", "id": "notes/garden.md", "score": 0.5, "rank": 2},
]


def test_each_item_carries_the_lowest_rank_it_stands_for() -> None:
    items = run_items(_result(_ITEMS, _CANDIDATES), [_LIST])

    assert items == _EXPECTED


def test_the_scorer_accepts_the_record_with_its_ranks() -> None:
    probe = load_probes(FIXTURE_DIR / "probes.jsonl")[0]
    record = run_record(
        probe,
        _result(_ITEMS, _CANDIDATES),
        [_LIST],
        counter=ApproximateTokenCounter(),
        cpu_load_procs=0,
        embedder_device="cuda",
    )

    parsed = parse_run_record(record)

    assert record["embedder_device"] == "cuda"

    assert [(item.ref, item.id, item.rank) for item in parsed.items] == [
        (entry["ref"], entry["id"], entry["rank"]) for entry in _EXPECTED
    ]


def test_the_record_carries_the_scored_shortlist_size() -> None:
    # The stage keeps 50 candidates at most; the ones past the cap count as well.
    result = _result(_ITEMS, _CANDIDATES)
    truncated = replace(result, stage=replace(result.stage, truncated=3))
    probe = load_probes(FIXTURE_DIR / "probes.jsonl")[0]

    record = run_record(
        probe,
        truncated,
        [_LIST],
        counter=ApproximateTokenCounter(),
        cpu_load_procs=0,
        embedder_device="cpu",
    )

    assert record["shortlisted"] == len(_CANDIDATES) + 3


def test_an_item_without_an_injected_candidate_is_refused() -> None:
    # m2's only candidate is gone from the stage, so no rank can be written for it.
    candidates = tuple(c for c in _CANDIDATES if c.item != 2)

    with pytest.raises(ValueError, match="m2"):
        run_items(_result(_ITEMS, candidates), [_LIST])
