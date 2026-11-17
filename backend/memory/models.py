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
# What a candidate can be: a served kind, or a facet, which the fetcher expands into its member
# facts before the block, so no served item is one.
CandidateKind = Literal["fact", "episode", "exchange", "note", "facet"]
SummaryState = Literal["raw", "summarized"]
DegradedReason = Literal[
    "deadline",
    "index_error",
    "index_rebuilding",
    "reranker_unavailable",
    "context_budget",
    "need_unknown",
]
ServingLevel = Literal["full", "low", "off"]
ServingTier = Literal["top", "interaction"]
CandidateSource = Literal["graph", "vault", "episode", "exchange", "facet"]
CandidateDisposition = Literal[
    "injected",
    "dropped_rank",
    "dropped_budget",
    "dropped_no_fit",
    "dropped_suppressed",
    "dropped_no_members",
    "dropped_late",
    "dropped_band",
]
StageSkipReason = Literal["off", "zero_budget", "error", "no_need"]
PrefetchReuse = Literal["block", "stage1", "none"]
NeedMode = Literal["on", "shadow"]
NeedBand = Literal["closed", "band", "open"]
NeedDegraded = Literal["deadline", "need_unknown"]

_DEGRADED_REASONS: tuple[str, ...] = (
    "deadline",
    "index_error",
    "index_rebuilding",
    "reranker_unavailable",
    "context_budget",
    "need_unknown",
)
_SERVING_LEVELS: tuple[str, ...] = ("full", "low", "off")
_SERVING_TIERS: tuple[str, ...] = ("top", "interaction")
_CANDIDATE_SOURCES: tuple[str, ...] = ("graph", "vault", "episode", "exchange", "facet")
_CANDIDATE_DISPOSITIONS: tuple[str, ...] = (
    "injected",
    "dropped_rank",
    "dropped_budget",
    "dropped_no_fit",
    "dropped_suppressed",
    "dropped_no_members",
    "dropped_late",
    "dropped_band",
)
# A facet's trace ref: `facet:<facet_key>`, so no tool ref can collide with it.
FACET_REF_PREFIX = "facet:"
_STAGE_SKIP_REASONS: tuple[str, ...] = ("off", "zero_budget", "error", "no_need")
_PREFETCH_REUSES: tuple[str, ...] = ("block", "stage1", "none")
_NEED_MODES: tuple[str, ...] = ("on", "shadow")
_NEED_BANDS: tuple[str, ...] = ("closed", "band", "open")
_NEED_DEGRADED: tuple[str, ...] = ("deadline", "need_unknown")


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


@dataclass(frozen=True, slots=True)
class FacetSource:
    """Where a facet candidate came from: the group of facts it stands for.

    A facet is a subject, a relation and an object type held by two or more facts; the fetcher
    serves it as its current members, never as itself.
    """

    facet_key: str
    subject_id: str
    rel_type: str
    object_type: str


ItemSource = FactSource | EpisodeSource | ExchangeSource | NoteSource

_ITEM_SOURCE_BY_KIND: dict[str, type] = {
    "fact": FactSource,
    "episode": EpisodeSource,
    "exchange": ExchangeSource,
    "note": NoteSource,
}
_SOURCE_BY_KIND: dict[str, type] = {**_ITEM_SOURCE_BY_KIND, "facet": FacetSource}


def _check_source(kind: str, source: object, kinds: Mapping[str, type] = _SOURCE_BY_KIND) -> None:
    expected = kinds.get(kind)
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
    its versions; `__post_init__` enforces that and that `source` matches `kind`. A facet is
    never an item: the fetcher serves its member facts instead.
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
        if self.kind == "facet":
            raise ValueError("a facet is served as its member facts, never as an item")
        _check_source(self.kind, self.source, _ITEM_SOURCE_BY_KIND)
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
        source: Where the candidate came from: `graph`, `vault`, `episode`, `exchange` or
            `facet`.
        ref: The ref the tools would give the same memory (`graph_ref`, `vault_ref`, or an
            `episode:` or `exchange:` hash), so a tool's ref joins the candidate; a facet,
            which no tool returns, is `facet:<facet_key>`.
        alt_ref: For a graph candidate, the ref of the reversed orientation,
            `graph_ref(object_id, predicate, subject_id)`. The tools' Cypher returns edges in
            path order and puts the user first whatever the stored direction, so a tool ref
            matches either this or `ref`. None for every other source.
        rank: 1-based place in the fetch's order: the candidate's position when it was
            ordered, and after every ordered candidate for one dropped before the order.
        score: Its order score, `1/(60 + pool_rank) + 1/(60 + ce_rank)`, the second term 0
            when not reranked.
        item: 1-based position in the block for an injected candidate, else None.
        disposition: What became of it: `injected`, or the reason it was dropped;
            `dropped_band` for one packing placed and a band turn's cut then dropped, and for
            a selected facet none of whose members is injected and one of which the cut
            dropped.
        pool_rank: 1-based position in the fused order; None from a producer that has none.
        ce_rank: 1-based place among the reranked candidates by cross-encoder probability,
            ties by pool rank; None when the cross-encoder did not score it.
        p_ce: The cross-encoder's probability; None when it did not score it.
    """

    source: CandidateSource
    ref: str
    alt_ref: str | None
    rank: int
    score: float
    item: int | None
    disposition: CandidateDisposition
    pool_rank: int | None = None
    ce_rank: int | None = None
    p_ce: float | None = None

    def __post_init__(self) -> None:
        """Keep `alt_ref` to graph candidates and a block position to injected ones."""
        _require_choice("source", self.source, _CANDIDATE_SOURCES)
        _require_choice("disposition", self.disposition, _CANDIDATE_DISPOSITIONS)
        if (self.alt_ref is not None) != (self.source == "graph"):
            raise ValueError(
                f"alt_ref is set for graph candidates only, got source={self.source!r} "
                f"alt_ref={self.alt_ref!r}"
            )
        if self.source == "facet" and not self.ref.startswith(FACET_REF_PREFIX):
            raise ValueError(
                f"a facet candidate's ref starts {FACET_REF_PREFIX!r}, got {self.ref!r}"
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
        if self.pool_rank is not None and self.pool_rank < 1:
            raise ValueError(f"pool_rank must be at least 1, got {self.pool_rank}")
        if self.ce_rank is not None and self.ce_rank < 1:
            raise ValueError(f"ce_rank must be at least 1, got {self.ce_rank}")
        if self.p_ce is not None:
            _require_probability("p_ce", self.p_ce)


def _band_of(p: float, theta_low: float, theta_high: float) -> str:
    if p < theta_low:
        return "closed"
    if p < theta_high:
        return "band"
    return "open"


def _require_probability(field_name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:  # also false for NaN
        raise ValueError(f"{field_name} must be in [0, 1], got {value!r}")


@dataclass(frozen=True, slots=True)
class NeedTrace:
    """What the need gate decided for one fetch: numbers and bank ids, never text.

    In `on` mode the fetch acts on `band`; in `shadow` mode the gate decides and is traced, and
    the fetch is served as open. A gate vector that missed the deadline or failed leaves the
    gate closed with `degraded` naming why, which is what `on` acts on; a shadow fetch records
    it here and leaves its own `degraded_reason` to the rest of the fetch.

    Attributes:
        mode: `on` or `shadow`. With the gate off the fetch has no `NeedTrace`.
        band: `closed` below `theta_low`, `band` from `theta_low` up to `theta_high`, `open`
            at `theta_high` and above; `closed` when `degraded` is set.
        p: The turn's need probability: the query's, or the follow-up context's when that is
            higher. None only when `degraded` is set before the query vector was decided.
        theta_low: The gate model's lower threshold.
        theta_high: The gate model's upper threshold.
        band_k: The gate model's entry count for a band turn.
        need_ids: The nearest need exemplars of the decision `p` came from, nearest first.
        no_need_ids: The same for no-need exemplars.
        context: Whether the follow-up context's probability was the higher and moved the
            band, so that the context, not the query alone, decided the band.
        context_p: The follow-up context's probability, when the gate waited for it and it
            arrived; None otherwise.
        degraded: `deadline` for a gate vector that missed the deadline, `need_unknown` for an
            embedder error or a vector the gate refused; None when the gate decided.
    """

    mode: NeedMode
    band: NeedBand
    p: float | None
    theta_low: float
    theta_high: float
    band_k: int
    need_ids: tuple[str, ...]
    no_need_ids: tuple[str, ...]
    context: bool
    context_p: float | None
    degraded: NeedDegraded | None

    def __post_init__(self) -> None:
        """Reject a decision whose band, probabilities or degradation contradict each other."""
        _require_choice("mode", self.mode, _NEED_MODES)
        _require_choice("band", self.band, _NEED_BANDS)
        if self.degraded is not None:
            _require_choice("degraded", self.degraded, _NEED_DEGRADED)
        _require_probability("theta_low", self.theta_low)
        _require_probability("theta_high", self.theta_high)
        if self.theta_low > self.theta_high:
            raise ValueError(
                f"theta_low must not exceed theta_high, got {self.theta_low} > {self.theta_high}"
            )
        if self.band_k < 1:
            raise ValueError(f"band_k must be at least 1, got {self.band_k}")
        if self.p is not None:
            _require_probability("p", self.p)
        if self.context_p is not None:
            _require_probability("context_p", self.context_p)
        if self.degraded is not None:
            if self.band != "closed":
                raise ValueError(
                    f"a degraded decision is closed, got band={self.band!r} "
                    f"degraded={self.degraded!r}"
                )
            if self.context or self.context_p is not None:
                raise ValueError("a degraded decision has no context probability")
            return
        if self.p is None:
            raise ValueError("p is set whenever the gate decided")
        expected = _band_of(self.p, self.theta_low, self.theta_high)
        if self.band != expected:
            raise ValueError(
                f"band must follow p: p={self.p} gives {expected!r}, got {self.band!r}"
            )
        if self.context and self.context_p != self.p:
            raise ValueError(
                f"a context that moved the band gives the turn's p, got context_p="
                f"{self.context_p!r} p={self.p}"
            )
        if self.context_p is not None and self.context_p > self.p:
            raise ValueError(
                f"p is the higher of the query's and the context's, got context_p="
                f"{self.context_p} p={self.p}"
            )


@dataclass(frozen=True, slots=True)
class MemoryStage:
    """What one turn's fetch did, carried from the fetcher to the trace builder.

    The fields are the trace's `memory` key; `timings` renders as `timings_ms`. A stage that
    did not run (`ran` false) names why in `skip_reason`; a turn with memory disabled has no
    stage at all. No field holds text, so nothing the user said reaches the trace through it.
    Candidates arrive in the order the fetcher keeps them; capping them for the trace is the
    trace builder's, which counts the rest in `truncated`. `need` is set whenever the need
    gate was asked; a closed `on` gate skips the stage with `no_need`.
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
    need: NeedTrace | None = None

    def __post_init__(self) -> None:
        """Reject a stage whose fields contradict each other or the trace's value sets."""
        if self.ran == (self.skip_reason is not None):
            raise ValueError(
                "a stage that ran has no skip_reason and one that did not names it, got "
                f"ran={self.ran} skip_reason={self.skip_reason!r}"
            )
        if self.skip_reason is not None:
            _require_choice("skip_reason", self.skip_reason, _STAGE_SKIP_REASONS)
        if self.skip_reason == "no_need":
            need = self.need
            if need is None:
                raise ValueError("a no_need skip carries the gate's decision")
            if need.mode != "on":
                raise ValueError("a shadow decision never skips the stage (no_need)")
            if need.band != "closed" or need.degraded is not None:
                raise ValueError(
                    "a no_need skip is a closed decision; a degraded one runs with its reason, "
                    f"got band={need.band!r} degraded={need.degraded!r}"
                )
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
        kind: Candidate kind; `source` must match it. A `facet` stands for a group of facts
            and is expanded into them before the block.
        item_id: Stable id (the `fact_key` for a fact, the `facet_key` for a facet), used to
            merge pools and break ties.
        text: Text the block shows; the reranker reads it when there is no `view_text`. A
            facet's is its view.
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
        view_text: The item's retrieval view (backend/memory/retrieval_view.py), such as the
            fact view the index embedded, which the reranker reads; None when the item has no
            view apart from `text`, as for a note, whose view is N0, its content.
    """

    kind: CandidateKind
    item_id: str
    text: str
    when: datetime | None
    source: ItemSource | FacetSource
    ranks: Mapping[str, int]
    confidence: float | None = None
    span_start: datetime | None = None
    span_end: datetime | None = None
    when_is_day: bool = False
    view_text: str | None = None

    def __post_init__(self) -> None:
        """Reject a candidate whose source does not match its kind."""
        _check_source(self.kind, self.source)
