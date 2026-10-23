"""Data types of the memory read path, the fact key, and the turn-log trace.

`MemoryTrace` is the contract home for the keys of `retrieval_context.memory` in the turn log;
`fact_key` is the one place a fact's identity is computed, so the index, the corrections and
the CLI cannot disagree about which fact an id names.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME

ItemKind = Literal["fact", "episode", "exchange", "note"]
SummaryState = Literal["raw", "summarized"]
DegradedReason = Literal["deadline", "index_error", "reranker_unavailable", "context_budget"]


def fact_key(subject_id: str, rel_type: str, object_id: str) -> str:
    """Identity of a fact across its versions: `sha256(subject_id|rel_type|object_id)`.

    For a relation type the ontology marks `directional=False`, the two ids are put in the order
    reconciliation stores them (`EdgeAssertion.from_rel_dict` swaps them when the target sorts
    before the source), so both orientations of one undirected fact share a key. Ids are
    canonical names, so the key survives a rebuild that replays the same extraction cache. A
    relation type the ontology does not know keeps the stated order.

    Args:
        subject_id: Canonical id of the edge's source entity.
        rel_type: Relationship type, as stored on the edge.
        object_id: Canonical id of the edge's target entity.

    Returns:
        The hex SHA-256 digest of the pipe-joined triple.
    """
    definition = EDGE_TYPES_BY_NAME.get(rel_type)
    if definition is not None and not definition.directional and object_id < subject_id:
        subject_id, object_id = object_id, subject_id
    return hashlib.sha256(f"{subject_id}|{rel_type}|{object_id}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class FactSource:
    """Where a fact item came from: the fact, the version and the utterance that stated it."""

    fact_key: str
    row_key: str
    recorded_at: str | None
    source_utterance_id: str | None


@dataclass(frozen=True, slots=True)
class EpisodeSource:
    """Where an episode item came from, with whether its summary exists yet."""

    episode_id: str
    conversation_id: str
    first_event_id: str
    last_event_id: str
    summary_state: SummaryState


@dataclass(frozen=True, slots=True)
class ExchangeSource:
    """Where an exchange item came from: one recorded turn."""

    event_id: str
    conversation_id: str | None
    episode_id: str | None


@dataclass(frozen=True, slots=True)
class NoteSource:
    """Where a vault note item came from; `authored_by` sets the note's confidence."""

    path: str
    heading: str | None
    authored_by: str | None


ItemSource = FactSource | EpisodeSource | ExchangeSource | NoteSource

_SOURCE_BY_KIND: dict[str, type] = {
    "fact": FactSource,
    "episode": EpisodeSource,
    "exchange": ExchangeSource,
    "note": NoteSource,
}


def _check_source(kind: str, source: object) -> None:
    expected = _SOURCE_BY_KIND.get(kind)
    if expected is None:
        raise ValueError(f"unknown item kind {kind!r}")
    if not isinstance(source, expected):
        raise ValueError(
            f"a {kind} item needs a {expected.__name__} source, got {type(source).__name__}"
        )


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    """One turn's request to the fetcher.

    `now` is the handler's fact-time seam, so a replay fetches against the same instant it
    records. `budget_tokens`, when set, lowers the configured block budget for this turn.
    """

    session_id: str
    conversation_id: str | None
    query_text: str
    now: datetime
    recent_user_turns: tuple[str, ...]
    turn_index: int
    conversation_start: bool = False
    budget_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """One item placed in the block, numbered by `ref` (m1..mN in block order).

    For a fact, `item_id` is the `fact_key`, so a correction names the fact rather than one of
    its versions; `__post_init__` enforces that and that `source` matches `kind`.
    """

    ref: str
    kind: ItemKind
    item_id: str
    text: str
    when: datetime | None
    score: float
    source: ItemSource

    def __post_init__(self) -> None:
        """Reject an item whose source does not match its kind or whose fact id is not its key."""
        _check_source(self.kind, self.source)
        if isinstance(self.source, FactSource) and self.item_id != self.source.fact_key:
            raise ValueError(
                f"a fact item's item_id must be its fact_key, got {self.item_id!r} for "
                f"{self.source.fact_key!r}"
            )


@dataclass(frozen=True, slots=True)
class FetchTimings:
    """Where one fetch spent its time, in milliseconds, and whether stage 1 was prefetched.

    `prefetch_hit` lives here because MemoryResult's fields are fixed by the interface and
    whether stage 1 came from the prefetch cache is part of how the fetch spent its time.
    """

    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    rerank_ms: float = 0.0
    queue_wait_ms: float = 0.0
    total_ms: float = 0.0
    prefetch_hit: bool = False


@dataclass(frozen=True, slots=True)
class MemoryResult:
    """What `fetch` returns; it never raises, so failures arrive as `degraded` results.

    `degraded` and `degraded_reason` are set together, so a consumer can trust either one.
    """

    items: tuple[MemoryItem, ...]
    block_text: str | None
    tokens: int
    budget_tokens: int
    degraded: bool
    degraded_reason: DegradedReason | None
    timings: FetchTimings
    reranked: int
    suppressed: int
    narrative_suppressed: int

    def __post_init__(self) -> None:
        """Keep `degraded` and `degraded_reason` consistent."""
        if self.degraded != (self.degraded_reason is not None):
            raise ValueError(
                "degraded and degraded_reason must be set together, got "
                f"degraded={self.degraded} degraded_reason={self.degraded_reason!r}"
            )


@dataclass(frozen=True, slots=True)
class MemoryTrace:
    """The typed source of `retrieval_context.memory` in the turn log.

    `to_json` emits exactly the keys the design doc lists; this class is their contract home,
    so a key change here is a turn-log contract change.
    """

    block_version: int
    role: str
    budget_tokens: int
    tokens: int
    delta_tokens_est: int | None
    degraded: bool
    degraded_reason: str | None
    tz: str
    reranked: int
    timings: FetchTimings
    items: tuple[MemoryItem, ...]
    suppressed: int
    narrative_suppressed: int

    @classmethod
    def from_result(
        cls,
        result: MemoryResult,
        *,
        block_version: int,
        role: str,
        tz: str,
        delta_tokens_est: int | None,
    ) -> MemoryTrace:
        """Build the trace of one turn from its fetch result and the turn's prompt facts.

        Args:
            result: The fetch result, after any handler change such as a dropped block.
            block_version: Version of the block format the turn used.
            role: Block placement used (`user_prefix` or `system_tail`).
            tz: Name of the zone temporal phrases were read in.
            delta_tokens_est: Tokens after the longest common prefix with the previous turn's
                message list, or None when there is no previous turn to compare.
        """
        return cls(
            block_version=block_version,
            role=role,
            budget_tokens=result.budget_tokens,
            tokens=result.tokens,
            delta_tokens_est=delta_tokens_est,
            degraded=result.degraded,
            degraded_reason=result.degraded_reason,
            tz=tz,
            reranked=result.reranked,
            timings=result.timings,
            items=result.items,
            suppressed=result.suppressed,
            narrative_suppressed=result.narrative_suppressed,
        )

    def to_json(self) -> dict[str, object]:
        """The JSON object written as `retrieval_context.memory`."""
        return {
            "block_version": self.block_version,
            "role": self.role,
            "budget_tokens": self.budget_tokens,
            "tokens": self.tokens,
            "delta_tokens_est": self.delta_tokens_est,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "tz": self.tz,
            "prefetch_hit": self.timings.prefetch_hit,
            "reranked": self.reranked,
            "timings_ms": {
                "stage1": self.timings.stage1_ms,
                "stage2": self.timings.stage2_ms,
                "rerank": self.timings.rerank_ms,
                "queue_wait": self.timings.queue_wait_ms,
                "total": self.timings.total_ms,
            },
            "items": [
                {"ref": item.ref, "kind": item.kind, "id": item.item_id, "score": item.score}
                for item in self.items
            ],
            "suppressed": self.suppressed,
            "narrative_suppressed": self.narrative_suppressed,
        }


@dataclass(frozen=True, slots=True)
class FactRow:
    """One latest-belief version of a fact, as the graph source yields it for the index.

    Keyed by `row_key` (the edge's `version_key`, or a structural key for an edge without one)
    and grouped by `fact_key` when read. The bounds and `recorded_at` stay the strings the graph
    stores (`valid_from` may be the `-inf` sentinel), because read-time currency compares them
    the way graph_store's predicate does. The index adds the rendered text and content hash.
    """

    row_key: str
    fact_key: str
    subject_id: str
    subject_name: str
    rel_type: str
    object_id: str
    object_name: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str | None
    confidence: float
    source_utterance_id: str | None


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A temporal phrase read as a closed-open interval [start, end) in the configured zone.

    "last time" names the previous conversation rather than a span of the clock, so it has no
    bounds and sets `previous_conversation`; every other window has both bounds, aware.
    """

    start: datetime | None
    end: datetime | None
    phrase: str
    previous_conversation: bool = False

    def __post_init__(self) -> None:
        """Require aware, ordered bounds on a dated window."""
        if self.previous_conversation and self.start is None and self.end is None:
            return
        if self.start is None or self.end is None:
            raise ValueError(f"a dated window needs both bounds, got {self.start}..{self.end}")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("window bounds must be timezone-aware")
        if self.end <= self.start:
            raise ValueError(f"window end {self.end} must follow start {self.start}")

    def contains(self, instant: datetime) -> bool:
        """Whether `instant` falls in [start, end); always False for `previous_conversation`."""
        if self.start is None or self.end is None:
            return False
        return self.start <= instant < self.end


@dataclass(frozen=True, slots=True)
class Candidate:
    """An item offered by one or more candidate pools, before fusion and rerank.

    Attributes:
        kind: Item kind; `source` must match it.
        item_id: Stable id (the `fact_key` for a fact), used to merge pools and break ties.
        text: Text the reranker scores and the block shows.
        when: Instant recency is measured from (a fact's `recorded_at`, an episode's end).
        source: Provenance carried through to the MemoryItem.
        ranks: 1-based rank of the item in each pool that returned it, by pool name.
        confidence: Edge confidence for a fact; None for other kinds, whose confidence ranking
            derives from the source.
        span_start: Start of the span an episode covers, for the window-match term.
        span_end: End of that span.
    """

    kind: ItemKind
    item_id: str
    text: str
    when: datetime | None
    source: ItemSource
    ranks: Mapping[str, int]
    confidence: float | None = None
    span_start: datetime | None = None
    span_end: datetime | None = None

    def __post_init__(self) -> None:
        """Reject a candidate whose source does not match its kind."""
        _check_source(self.kind, self.source)
