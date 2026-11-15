"""Data types of the memory read path, the fact key, the serving profile and the memory stage.

`fact_key` is the one place a fact's identity is computed, so the index, the corrections and
the CLI cannot disagree about which fact an id names. `ServingProfile` is how the serving side
runs the turn, which sets the block budget. `MemoryStage` carries what the fetch did from the
fetcher to the trace builder: the fields of the trace's `memory` key, refs and numbers only,
never text.
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
DegradedReason = Literal[
    "deadline", "index_error", "index_rebuilding", "reranker_unavailable", "context_budget"
]
ServingLevel = Literal["full", "low", "off"]
ServingTier = Literal["top", "interaction"]
CandidateSource = Literal["graph", "vault", "episode", "exchange"]
CandidateDisposition = Literal[
    "injected", "dropped_min_score", "dropped_budget", "dropped_no_fit", "dropped_suppressed"
]
StageSkipReason = Literal["off", "zero_budget", "error"]
PrefetchReuse = Literal["block", "stage1", "none"]

_DEGRADED_REASONS: tuple[str, ...] = (
    "deadline",
    "index_error",
    "index_rebuilding",
    "reranker_unavailable",
    "context_budget",
)
_SERVING_LEVELS: tuple[str, ...] = ("full", "low", "off")
_SERVING_TIERS: tuple[str, ...] = ("top", "interaction")
_CANDIDATE_SOURCES: tuple[str, ...] = ("graph", "vault", "episode", "exchange")
_CANDIDATE_DISPOSITIONS: tuple[str, ...] = (
    "injected",
    "dropped_min_score",
    "dropped_budget",
    "dropped_no_fit",
    "dropped_suppressed",
)
_STAGE_SKIP_REASONS: tuple[str, ...] = ("off", "zero_budget", "error")
_PREFETCH_REUSES: tuple[str, ...] = ("block", "stage1", "none")


def _require_choice(field_name: str, value: object, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{field_name} must be one of {choices}, got {value!r}")


def _require_non_negative(field_name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{field_name} must not be negative, got {value}")


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
class ServingProfile:
    """How the serving side runs this turn: its power level and the model tier answering it.

    The level sets the block budget (`MemoryConfig.block_tokens_for`): full, low, or off, where
    the fetch is skipped. The tier is recorded in the trace so a fetch's cost can be read per
    tier; it does not change the budget, because one tier can run at more than one level.
    """

    level: ServingLevel
    tier: ServingTier

    def __post_init__(self) -> None:
        """Reject a level or tier the read path has no budget for."""
        _require_choice("level", self.level, _SERVING_LEVELS)
        _require_choice("tier", self.tier, _SERVING_TIERS)


# What a turn gets when no serving resolver is wired: today's main model at full power.
DEFAULT_SERVING = ServingProfile(level="full", tier="top")


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    """One turn's request to the fetcher.

    `now` is the handler's fact-time seam, so a replay fetches against the same instant it
    records. `budget_tokens`, when set, lowers the configured block budget for this turn.
    `serving` is the profile the turn runs under, recorded in the stage.
    """

    session_id: str
    conversation_id: str | None
    query_text: str
    now: datetime
    recent_user_turns: tuple[str, ...]
    turn_index: int
    conversation_start: bool = False
    budget_tokens: int | None = None
    serving: ServingProfile = DEFAULT_SERVING


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

    `prefetch_hit` says whether any prefetched work was reused, which is part of how the fetch
    spent its time; `MemoryStage.prefetch_reuse` says which, and the stage checks the two agree.
    The trace renders `total_ms` as `critical_path`: the fetch's time from the call, the span
    `MemoryConfig.deadline_ms` bounds.
    """

    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    rerank_ms: float = 0.0
    queue_wait_ms: float = 0.0
    total_ms: float = 0.0
    prefetch_hit: bool = False


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    """One candidate the fetch considered, as the trace records it: a ref, never text.

    Attributes:
        source: Where the candidate came from: `graph`, `vault`, `episode` or `exchange`.
        ref: The ref the tools would give the same memory (`graph_ref`, `vault_ref`, or an
            `episode:` or `exchange:` hash), so a tool's ref joins the candidate.
        alt_ref: For a graph candidate, the ref of the reversed orientation,
            `graph_ref(object_id, predicate, subject_id)`. The tools' Cypher returns edges in
            path order and puts the user first whatever the stored direction, so a tool ref
            matches either this or `ref`. None for every other source.
        rank: 1-based rank after fusion and rerank.
        score: Final score.
        item: 1-based position in the block for an injected candidate, else None.
        disposition: What became of it: `injected`, or the reason it was dropped.
    """

    source: CandidateSource
    ref: str
    alt_ref: str | None
    rank: int
    score: float
    item: int | None
    disposition: CandidateDisposition

    def __post_init__(self) -> None:
        """Keep `alt_ref` to graph candidates and a block position to injected ones."""
        _require_choice("source", self.source, _CANDIDATE_SOURCES)
        _require_choice("disposition", self.disposition, _CANDIDATE_DISPOSITIONS)
        if (self.alt_ref is not None) != (self.source == "graph"):
            raise ValueError(
                f"alt_ref is set for graph candidates only, got source={self.source!r} "
                f"alt_ref={self.alt_ref!r}"
            )
        if (self.item is not None) != (self.disposition == "injected"):
            raise ValueError(
                f"item is set for injected candidates only, got disposition="
                f"{self.disposition!r} item={self.item!r}"
            )
        if self.rank < 1:
            raise ValueError(f"rank must be at least 1, got {self.rank}")
        if self.item is not None and self.item < 1:
            raise ValueError(f"item must be at least 1, got {self.item}")


@dataclass(frozen=True, slots=True)
class MemoryStage:
    """What one turn's fetch did, carried from the fetcher to the trace builder.

    The fields are the trace's `memory` key; `timings` renders as `timings_ms`. A stage that
    did not run (`ran` false) names why in `skip_reason`; a turn with memory disabled has no
    stage at all. No field holds text, so nothing the user said reaches the trace through it.
    Candidates arrive in the order the fetcher keeps them; capping them for the trace is the
    trace builder's, which counts the rest in `truncated`.
    """

    ran: bool
    skip_reason: StageSkipReason | None
    level: ServingLevel
    tier: ServingTier
    block_version: int
    budget_tokens: int
    tokens: int
    degraded_reason: DegradedReason | None
    prefetch_reuse: PrefetchReuse
    reranked: int
    suppressed: int
    narrative_suppressed: int
    timings: FetchTimings
    candidates: tuple[MemoryCandidate, ...]
    truncated: int = 0

    def __post_init__(self) -> None:
        """Reject a stage whose fields contradict each other or the trace's value sets."""
        if self.ran == (self.skip_reason is not None):
            raise ValueError(
                "a stage that ran has no skip_reason and one that did not names it, got "
                f"ran={self.ran} skip_reason={self.skip_reason!r}"
            )
        if self.skip_reason is not None:
            _require_choice("skip_reason", self.skip_reason, _STAGE_SKIP_REASONS)
        _require_choice("level", self.level, _SERVING_LEVELS)
        _require_choice("tier", self.tier, _SERVING_TIERS)
        if self.degraded_reason is not None:
            _require_choice("degraded_reason", self.degraded_reason, _DEGRADED_REASONS)
        _require_choice("prefetch_reuse", self.prefetch_reuse, _PREFETCH_REUSES)
        if self.timings.prefetch_hit != (self.prefetch_reuse != "none"):
            raise ValueError(
                f"prefetch_reuse={self.prefetch_reuse!r} disagrees with "
                f"timings.prefetch_hit={self.timings.prefetch_hit}"
            )
        for name in (
            "budget_tokens",
            "tokens",
            "reranked",
            "suppressed",
            "narrative_suppressed",
            "truncated",
        ):
            _require_non_negative(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class MemoryResult:
    """What `fetch` returns; it never raises, so failures arrive as `degraded` results.

    `degraded` and `degraded_reason` are set together, so a consumer can trust either one.
    `stage` is what the trace records of the fetch.
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
    stage: MemoryStage

    def __post_init__(self) -> None:
        """Keep `degraded` and `degraded_reason` consistent."""
        if self.degraded != (self.degraded_reason is not None):
            raise ValueError(
                "degraded and degraded_reason must be set together, got "
                f"degraded={self.degraded} degraded_reason={self.degraded_reason!r}"
            )


@dataclass(frozen=True, slots=True)
class FactRow:
    """One latest-belief version of a fact, as the graph source yields it for the index.

    Keyed by `row_key` (the edge's `version_key`, or a structural key for an edge without one)
    and grouped by `fact_key` when read. The bounds and `recorded_at` stay the strings the graph
    stores (`valid_from` may be the `-inf` sentinel), because read-time currency compares them
    the way graph_store's predicate does. The index adds the rendered text and content hash.

    `subject_type` and `object_type` are the nodes' `entity_type`, None when a node has none.
    They have no default, so every producer states them: the index's type terms, and with them
    every type-scoped question, depend on them.
    """

    row_key: str
    fact_key: str
    subject_id: str
    subject_name: str
    subject_type: str | None
    rel_type: str
    object_id: str
    object_name: str
    object_type: str | None
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
        when_is_day: True when `when` names a local day rather than an instant: a note whose
            `last_updated` is a date, kept as that day's midnight in the configured zone so
            recency is measured as before. The block then dates the item by its day alone,
            because a part of the day read off midnight is a time nobody said.
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
    when_is_day: bool = False

    def __post_init__(self) -> None:
        """Reject a candidate whose source does not match its kind."""
        _check_source(self.kind, self.source)
