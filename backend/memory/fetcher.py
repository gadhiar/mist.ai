"""The per-turn memory fetch: candidate pools, rerank, sibling lists and the budgeted block.

`MemoryFetcher.fetch` turns one `MemoryRequest` into a `MemoryResult` under a critical-path
deadline (`MemoryConfig.deadline_ms`, 50 ms by default) counted from the call. The work falls
in two stages (Design point 8):

- Stage 1 needs no transcript: the index state (`index_rebuilding` while a rebuild runs), the
  fact and facet entity anchors of the last two user utterances, and the suppression set of the
  corrections. It is cached per session for 30 s, so a prefetch started at speech start takes
  it off the critical path.
- Stage 2 depends on the words: the FTS, vector and entity-anchored fact and facet pools, the
  vault note pool, RRF fusion into the pool order, the shortlist's drops (suppressed facts,
  facets with fewer than two current members), the cross-encoder on the pairs that fit the
  time left, the order by rank, selection of the first `select_k`, expansion of sibling lists
  (plan v2 V7) and facets into their facts, and packing in that order.

The need gate, when one is given, decides in stage 2, beside the pools and before the
cross-encoder, how likely the turn needs memory. In `on` mode a closed decision ends the fetch
with no block (`skip_reason="no_need"`) and cancels the pools; a band or open one proceeds. A
gate vector that misses the deadline closes it as `deadline`, an embedder error or a refused
vector as `need_unknown`. In `shadow` mode the gate waits and decides at the same point, is
traced, and the fetch is served as open. A closed or band query that follows a fetch which
served memory is also judged by the previous user turn joined to it, taking the higher
probability. The decision is traced as `MemoryStage.need`.

Blocking work runs on the injected executors: SQLite on `memory-index`, the query embedding and
the rerank on `memory-model`, the gate's own embeddings on `memory-gate`. A thread cannot be
cancelled, so at the deadline the fetcher stops waiting, discards whatever arrives late and
returns what has finished, marked `degraded_reason="deadline"`. The time a call spent queued
behind earlier work is counted as `queue_wait_ms`.

`prefetch(session_id, text)` runs the whole fetch for the latest words of a session, one in
flight per session, and keeps the last completed result with its normalised text. At the
endpoint `fetch` reuses that result when the final words normalise to the same text
(`prefetch_reuse="block"`), awaits a running prefetch of the same text within the deadline,
reuses a cached stage 1 when the words are close enough (Jaccard 0.8 on the token sets,
`stage1`), and otherwise runs fresh (`none`).

`fetch` never raises into the turn: an index failure returns an empty result marked
`index_error`, a reranker failure keeps the pool order marked `reranker_unavailable`, and any
other contained failure returns an empty stage marked `skip_reason="error"`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from backend.errors import MemoryIndexError, MemoryRerankError, MistError
from backend.interfaces import EmbeddingProvider
from backend.knowledge.models import graph_ref, vault_ref

from .block import BLOCK_VERSION, FactList, MemoryBlockBuilder, PlacedList
from .config import MemoryConfig
from .fact_index import FactGroup, IndexedFacet, IndexedFact
from .fts import words
from .models import (
    FACET_REF_PREFIX,
    Candidate,
    CandidateDisposition,
    CandidateSource,
    DegradedReason,
    FacetSource,
    FactSource,
    FetchTimings,
    MemoryCandidate,
    MemoryRequest,
    MemoryResult,
    MemoryStage,
    NeedDegraded,
    NeedMode,
    NeedTrace,
    NoteSource,
    PrefetchReuse,
    ServingProfile,
    StageSkipReason,
)
from .need_gate import NeedDecision, NeedGate, context_text
from .ranking import (
    FusedCandidate,
    ScoredCandidate,
    confidence_term,
    fuse,
    order_candidates,
    order_score,
)
from .rerank import NullReranker, Reranker
from .sources import (
    ANCHOR_TURNS,
    DEFAULT_FACT_POOL_SIZE,
    ENTITY_POOL,
    FILE_HEADING,
    FTS_POOL,
    VECTOR_POOL,
)
from .temporal import parse_window

logger = logging.getLogger(__name__)

T = TypeVar("T")
A = TypeVar("A")

# Executor names the runtime builds and the factory passes in (Design point 8).
INDEX_EXECUTOR = "memory-index"
MODEL_EXECUTOR = "memory-model"
GATE_EXECUTOR = "memory-gate"

# How the fetch acts on the need gate's band; with the gate off the fetcher gets no gate.
_NEED_MODES: tuple[str, ...] = ("on", "shadow")

# The bands after which a session's next fetch may be a follow-up of one that served memory.
_SERVED_BANDS = frozenset({"band", "open"})

# How long a cached stage 1 and a completed prefetch stay valid.
STAGE1_TTL_S = 30.0

# Jaccard similarity of the token sets at or above which a prefetched stage 1 is reused.
STAGE1_REUSE_JACCARD = 0.8

# A prefetch runs off the critical path, so it gets a wider budget than the endpoint's deadline:
# enough to rerank the whole shortlist on CPU, bounded so a stuck executor cannot hold a session.
DEFAULT_PREFETCH_DEADLINE_MS = 1000.0

# Most candidates the stage keeps, the injected ones first (plan v2 V5).
STAGE_CANDIDATE_CAP = 50

# A session with no fetch or prefetch for this long is forgotten.
_SESSION_IDLE_S = 600.0

# Failures `fetch` contains rather than lets reach the turn. MistError covers every I/O error
# the read path raises (index, sidecar, embedder, reranker); the builtin types cover a fault in
# a fake or a third-party library that does not wrap its errors. Each is logged with its type.
_CONTAINED_ERRORS: tuple[type[BaseException], ...] = (
    MistError,
    sqlite3.Error,
    OSError,
    RuntimeError,
    ValueError,
    LookupError,
    ArithmeticError,
    TypeError,
    AttributeError,
)

# What an embedder can raise: EmbeddingError (a MistError) from the production embedder, and
# the raw library errors FactIndex also maps (fact_index.py, `_embed`).
_EMBEDDER_ERRORS: tuple[type[BaseException], ...] = (MistError, RuntimeError, OSError, ValueError)

_SOURCE_BY_KIND: dict[str, CandidateSource] = {
    "fact": "graph",
    "note": "vault",
    "episode": "episode",
    "exchange": "exchange",
    "facet": "facet",
}

# Task names of the facet pools; their hits fuse under the fact pools' names.
_FACET_FTS = "facet_fts"
_FACET_VECTOR = "facet_vector"
_FACET_ENTITY = "facet_entity"

# One degraded reason per result; the more fundamental failure is reported.
_DEGRADED_PRIORITY: tuple[DegradedReason, ...] = (
    "index_error",
    "index_rebuilding",
    "need_unknown",
    "deadline",
    "reranker_unavailable",
)


class FactSearch(Protocol):
    """The reads the fetcher makes on the fact index; `FactIndex` satisfies it."""

    def search_fts(self, query_text: str, *, limit: int = 20) -> list[FactGroup]:
        """Facts sharing a content word with `query_text`, best first."""
        ...

    def search_vector(self, embedding: Sequence[float], *, limit: int = 20) -> list[FactGroup]:
        """Facts nearest to `embedding`, nearest first."""
        ...

    def search_names(self, texts: Sequence[str], *, limit: int = 20) -> list[FactGroup]:
        """Facts whose subject or object name occurs in one of `texts`."""
        ...

    def siblings(
        self, subject_id: str, rel_type: str, source_utterance_id: str, *, now: datetime
    ) -> tuple[IndexedFact, ...]:
        """The current facts one utterance stated with one subject and relation."""
        ...

    def search_facets_fts(self, query_text: str, *, limit: int = 20) -> list[IndexedFacet]:
        """Facets whose view, subject name or type terms share a content word with the query."""
        ...

    def search_facets_vector(
        self, embedding: Sequence[float], *, limit: int = 20
    ) -> list[IndexedFacet]:
        """Facets whose view is nearest to `embedding`, nearest first."""
        ...

    def search_facet_names(self, texts: Sequence[str], *, limit: int = 20) -> list[IndexedFacet]:
        """Facets whose subject's name occurs whole in one of `texts`."""
        ...

    def facet_members_many(
        self, facet_keys: Iterable[str], *, now: datetime
    ) -> dict[str, tuple[IndexedFact, ...]]:
        """The members of each facet current at `now`, newest recorded first, in one read."""
        ...


class NoteSearch(Protocol):
    """The vault note pool; `sources.VaultCandidates` satisfies it."""

    async def gather(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        *,
        profile_path: str | None = None,
    ) -> list[Candidate]:
        """Note candidates for the query, best first."""
        ...


class SuppressionSource(Protocol):
    """The fact keys corrections suppress now (Phase C); read in stage 1, blocking."""

    def suppressed_fact_keys(self) -> frozenset[str]:
        """Every `fact_key` a correction currently suppresses."""
        ...


class FetchClock(Protocol):
    """Monotonic seconds for the deadline and the wall clock for a prefetch's `now`.

    `tests.fakes.memory_fakes.FakeClock` satisfies it; production passes `SystemClock`.
    """

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin that never goes backwards."""
        ...

    def now(self) -> datetime:
        """The current instant, timezone-aware."""
        ...


class SystemClock:
    """The process clocks: `time.monotonic` and the UTC wall clock."""

    def monotonic(self) -> float:
        """The process's monotonic clock, in seconds."""
        return time.monotonic()

    def now(self) -> datetime:
        """The current UTC instant."""
        return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class FetcherExecutors:
    """The three executors the fetcher submits blocking work to; the runtime owns them.

    Attributes:
        index: `memory-index`, for SQLite reads (two threads, shared with the refresher).
        model: `memory-model`, for the query embedding and the rerank (one thread).
        gate: `memory-gate`, for the need gate's own embeddings: a gate embedder's query
            vector and the follow-up context's (one thread). They are CPU work of their own,
            so they never queue behind the cross-encoder or ahead of it.
    """

    index: Executor
    model: Executor
    gate: Executor


@dataclass(frozen=True, slots=True)
class FetchOutcome:
    """A fetch's result and the fact lists its block placed.

    The result is what the turn and the trace use. `lists` lets an offline reader, such as
    the recall harness, flatten a list item back into its facts: the result names a list by
    its earliest member only, and a sibling only the expansion brought in is not a
    candidate.

    `fused`, `ranked` and `selected` let the same reader attribute a miss to a stage: the
    result's stage keeps refs, ranks and scores, not the score terms, and at most
    `STAGE_CANDIDATE_CAP` candidates. All three are empty when stage 2 did not run.

    Attributes:
        result: What `fetch` returns.
        lists: The fact lists the block placed.
        fused: Every fused candidate in pool order, before the `rerank_top_n` cut, so its
            first `rerank_top_n` are the shortlist before its drops.
        ranked: The shortlist after the drops (suppressed facts, facets without members or
            whose lookup was late), in position order; the stage's candidate of rank `i` is
            `ranked[i - 1]` for `i` up to its length.
        selected: What selection and expansion bring, in packing order: the first
            `select_k` of `ranked`, the selected facets included, and every fact expansion
            placed. Each fact sits at the earlier of its own position and that of the facet or
            fact that brought it; the block packs a list at its earliest member's position, so
            a list member here may sit later than its list does in the block.
    """

    result: MemoryResult
    lists: tuple[PlacedList, ...] = ()
    fused: tuple[FusedCandidate, ...] = ()
    ranked: tuple[ScoredCandidate, ...] = ()
    selected: tuple[ScoredCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class _Expansion:
    """What selection and expansion hand the block, and what the trace needs of them.

    Attributes:
        items: The block's input: candidates alone and fact lists.
        selected: `FetchOutcome.selected`.
        facets: Each selected facet's key, mapped to the fact keys of the members it brought.
    """

    items: list[ScoredCandidate | FactList]
    selected: tuple[ScoredCandidate, ...]
    facets: dict[tuple[str, str], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class _Stage1:
    """What stage 1 found: the recent turns' fact and facet anchors and the suppressed keys."""

    anchors: tuple[FactGroup, ...]
    suppressed: frozenset[str]
    facet_anchors: tuple[IndexedFacet, ...] = ()


@dataclass(frozen=True, slots=True)
class _Shape:
    """The request fields besides the words that decide a block, so a reuse is only taken when
    they match: anchor texts, the effective budget, the serving profile and the reserve flag.
    """

    anchors: tuple[str, ...]
    budget: int
    level: str
    tier: str
    conversation_start: bool


@dataclass(slots=True)
class _Stage1Entry:
    """A cached or running stage 1 for one session."""

    shape_anchors: tuple[str, ...]
    text: str | None
    started: float
    future: asyncio.Future[_Stage1]
    generation: int


@dataclass(slots=True)
class _Completed:
    """The last completed prefetch of a session."""

    norm: str
    shape: _Shape
    outcome: FetchOutcome
    finished: float
    generation: int


@dataclass(slots=True)
class _Running:
    """The prefetch in flight for a session."""

    norm: str
    shape: _Shape
    task: asyncio.Task[FetchOutcome | None]


@dataclass(slots=True)
class _Context:
    """What a prefetch borrows from the session's last fetch: it has only the words."""

    conversation_id: str | None = None
    recent_user_turns: tuple[str, ...] = ()
    turn_index: int = 0
    conversation_start: bool = True
    budget_tokens: int | None = None
    serving_level: str = "full"
    serving_tier: str = "top"


@dataclass(slots=True)
class _Session:
    context: _Context = field(default_factory=_Context)
    stage1: _Stage1Entry | None = None
    running: _Running | None = None
    completed: _Completed | None = None
    touched: float = 0.0
    # Work stamped with a generation below this was started before the session's last
    # invalidate and is never reused.
    floor: int = 0
    # The session's last fetch served memory: its gate's band was band or open (in shadow
    # mode, the band the gate decided), so the next one may be a follow-up of it.
    served: bool = False


@dataclass(slots=True)
class _Ledger:
    """Per-fetch accounting: queue waits, stage times and the degraded reasons met."""

    queue_wait_s: float = 0.0
    stage1_s: float = 0.0
    stage2_s: float = 0.0
    rerank_s: float = 0.0
    reasons: set[str] = field(default_factory=set)

    def reason(self) -> DegradedReason | None:
        for reason in _DEGRADED_PRIORITY:
            if reason in self.reasons:
                return reason
        return None


class _LateError(Exception):
    """A call did not finish before the deadline, or finished after it; its result is dropped.

    Private and raised only inside the fetcher, which catches it at each await.
    """


class _GoneError(Exception):
    """Work a fetch was waiting on was cancelled by a newer prefetch, not by the fetch's caller."""


class _IndexFailedError(Exception):
    """The index failed mid-fetch; the fetch returns an empty `index_error` result."""


class MemoryFetcher:
    """Runs the memory fetch for each turn and the speculative prefetch between turns.

    Every collaborator is injected (the mist.ai DI rule); `backend/factories.py` wires the real
    ones and tests pass fakes.

    Args:
        index: The fact index's reads (`FactIndex`).
        embedder: Embeds the query; the instance the retriever already shares in production.
        reranker: Scores query-text pairs (`CrossEncoderReranker`, `HttpReranker`, or
            `NullReranker` when the model is off or failed to load).
        block_builder: Packs and renders the block; built with the same `config`.
        config: The memory settings: deadline, `rerank_top_n`, `select_k`, `list_max_items`,
            `facets`, the zone, and the budget per serving level.
        executors: The `memory-index`, `memory-model` and `memory-gate` executors.
        need_gate: The need gate, or None when `MIST_MEMORY_NEED_GATE` is off: every fetch is
            then open and traces no `need`. When the gate `shares_query_vector`, it decides on
            the fetch's own query vector and its embedder is the fetch's model; the fetcher
            then serializes its calls into that model across the model and gate threads.
        need_mode: `on` acts on the gate's band; `shadow` decides, traces and serves as open.
            Read only when `need_gate` is set.
        need_context: Whether a closed or band query after a fetch that served memory is also
            judged by the previous user turn joined to it.
        clock: Monotonic seconds for the deadline and the wall clock for a prefetch's `now`.
        index_state: Returns the index's degraded reason, `index_db.degraded_reason` on a
            connection of its own (`runtime.IndexStateProbe`); checked on every fetch.
        notes: The vault note pool, or None when no sidecar is configured.
        suppression: The corrections' suppression set (Phase C), or None for none.
        profile_path: The always-injected user profile note, kept out of the note pool.
        pool_size: Facts in each of the FTS, vector and entity pools.
        prefetch_deadline_ms: The budget of a prefetch's whole fetch.

    Raises:
        ValueError: `pool_size` or `prefetch_deadline_ms` is not positive, or `need_mode` is
            not `on` or `shadow` while a gate is given.
    """

    def __init__(
        self,
        *,
        index: FactSearch,
        embedder: EmbeddingProvider,
        reranker: Reranker,
        block_builder: MemoryBlockBuilder,
        config: MemoryConfig,
        executors: FetcherExecutors,
        need_gate: NeedGate | None,
        need_mode: NeedMode,
        need_context: bool,
        clock: FetchClock,
        index_state: Callable[[], DegradedReason | None],
        notes: NoteSearch | None = None,
        suppression: SuppressionSource | None = None,
        profile_path: str | None = None,
        pool_size: int = DEFAULT_FACT_POOL_SIZE,
        prefetch_deadline_ms: float = DEFAULT_PREFETCH_DEADLINE_MS,
    ) -> None:
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")
        if not prefetch_deadline_ms > 0:
            raise ValueError(f"prefetch_deadline_ms must be positive, got {prefetch_deadline_ms}")
        if need_gate is not None and need_mode not in _NEED_MODES:
            raise ValueError(f"need_mode must be one of {_NEED_MODES}, got {need_mode!r}")
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._builder = block_builder
        self._config = config
        self._executors = executors
        self._need_gate = need_gate
        self._need_mode = need_mode
        self._need_context = need_context
        # Under a shared gate embedder the context embedding (gate thread) and the fetch's own
        # query embedding (model thread) call one model object, which nothing else serializes
        # (EmbeddingGenerator.generate_embedding takes no lock); this lock does.
        self._shared_embed_lock = (
            threading.Lock() if need_gate is not None and need_gate.shares_query_vector else None
        )
        self._clock = clock
        self._index_state = index_state
        self._notes = notes
        self._suppression = suppression
        self._profile_path = profile_path
        self._pool_size = pool_size
        self._prefetch_deadline_s = prefetch_deadline_ms / 1000.0
        self._sessions: dict[str, _Session] = {}
        self._generation = 0
        # The reranker is a stand-in only when one was asked for: `off` is a choice, not a fault.
        self._rerank_unavailable = (
            isinstance(reranker, NullReranker) and config.rerank_backend != "off"
        )

    # -- public API -----------------------------------------------------------------------------

    async def fetch(self, request: MemoryRequest) -> MemoryResult:
        """The memory block for one turn, within the deadline; never raises into the turn."""
        return (await self.fetch_outcome(request)).result

    async def fetch_outcome(self, request: MemoryRequest) -> FetchOutcome:
        """`fetch`, with the fact lists the block placed, for offline readers.

        Returns:
            The outcome; its result is what `fetch` returns.
        """
        started = self._clock.monotonic()
        ledger = _Ledger()
        try:
            outcome = await self._fetch(request, started, ledger)
        except _CONTAINED_ERRORS as exc:
            logger.error(
                "memory fetch failed with %s; the turn gets no memory block",
                type(exc).__name__,
                exc_info=exc,
            )
            outcome = FetchOutcome(
                result=self._skipped(request, "error", self._timings(started, ledger, None))
            )
        self._remember(request, outcome)
        return outcome

    async def prefetch(self, session_id: str, partial_text: str | None = None) -> None:
        """Start the fetch for a session's latest words, and return at once.

        With no words (speech start on the threaded path) only stage 1 runs and is cached. With
        words, the whole fetch runs; a prefetch already running for other words is cancelled,
        so one is in flight per session and the latest words win. The request borrows the
        session's last fetch for everything but the words (anchors, budget, serving). The work
        runs as tasks named `memory-prefetch-<session_id>` (the whole fetch) and
        `memory-stage1-<session_id>`.
        """
        try:
            self._prefetch(session_id, partial_text)
        except _CONTAINED_ERRORS as exc:
            logger.warning("memory prefetch did not start: %s", type(exc).__name__, exc_info=exc)

    def invalidate(self, session_id: str | None = None) -> None:
        """Drop the cached stage 1 and prefetched results, of one session or of all.

        A correction calls it with no argument, so a forgotten fact cannot leak through a
        cached suppression set or a block packed before it. A prefetch still running finishes
        but is not kept.
        """
        self._generation += 1
        targets = (
            list(self._sessions.values())
            if session_id is None
            else [s for s in (self._sessions.get(session_id),) if s is not None]
        )
        for session in targets:
            session.floor = self._generation
            session.stage1 = None
            session.completed = None
            session.running = None  # left to finish; its generation keeps it from being stored

    # -- the fetch ------------------------------------------------------------------------------

    async def _fetch(self, request: MemoryRequest, started: float, ledger: _Ledger) -> FetchOutcome:
        budget = self._budget(request)
        if request.serving.level == "off":
            return FetchOutcome(
                result=self._skipped(request, "off", self._timings(started, ledger, None))
            )
        if budget == 0:
            return FetchOutcome(
                result=self._skipped(request, "zero_budget", self._timings(started, ledger, None))
            )
        deadline = started + self._config.deadline_ms / 1000.0
        session = self._session(request.session_id)
        norm = normalise(request.query_text)
        shape = self._shape(request)

        try:
            state = await self._call(
                self._executors.index, self._index_state, deadline=deadline, ledger=ledger
            )
        except _LateError:
            return self._empty(request, started, ledger, "deadline", "none")
        except MemoryIndexError as exc:
            logger.warning("memory index state could not be read: %s", exc)
            return self._empty(request, started, ledger, "index_error", "none")
        if state is not None:
            return self._empty(request, started, ledger, state, "none")

        completed = session.completed
        if (
            completed is not None
            and completed.norm == norm
            and completed.shape == shape
            and completed.generation >= session.floor
            and self._clock.monotonic() - completed.finished <= STAGE1_TTL_S
        ):
            return self._reused(completed.outcome, started, ledger)

        running = session.running
        if (
            running is not None
            and running.norm == norm
            and running.shape == shape
            and not running.task.cancelled()
        ):
            try:
                prefetched = await self._await(running.task, deadline=deadline)
            except _LateError:
                return self._empty(request, started, ledger, "deadline", "none")
            except _GoneError:
                prefetched = None  # newer words replaced it; run this fetch's own
            # An invalidate while waiting clears `session.running`; its block predates it.
            if prefetched is not None and session.running is running:
                return self._reused(prefetched, started, ledger)

        reuse: PrefetchReuse = "none"
        entry = session.stage1
        stage1_started = self._clock.monotonic()
        if entry is not None and self._stage1_reusable(session, entry, shape, request.query_text):
            reuse = "stage1"
            try:
                stage1 = await self._await(entry.future, deadline=deadline)
            except _LateError:
                ledger.stage1_s = self._clock.monotonic() - stage1_started
                return self._empty(request, started, ledger, "deadline", reuse)
            except MemoryIndexError:
                return self._empty(request, started, ledger, "index_error", reuse)
        else:
            try:
                stage1 = await self._stage1(shape.anchors, deadline=deadline, ledger=ledger)
            except _LateError:
                ledger.stage1_s = self._clock.monotonic() - stage1_started
                return self._empty(request, started, ledger, "deadline", reuse)
            except MemoryIndexError as exc:
                logger.warning("memory stage 1 failed: %s", exc)
                return self._empty(request, started, ledger, "index_error", reuse)
        ledger.stage1_s = self._clock.monotonic() - stage1_started

        try:
            return await self._stage2(
                request, stage1, deadline, started, ledger, reuse, served=session.served
            )
        except _IndexFailedError:
            return self._empty(request, started, ledger, "index_error", reuse)

    async def _stage1(
        self, anchors: tuple[str, ...], *, deadline: float, ledger: _Ledger | None
    ) -> _Stage1:
        """Run stage 1 on the index executor: the recent turns' anchors and the suppression set.

        The facet anchors, the facets whose subject the recent turns name, are read only when
        `facets` is on.
        """

        def work() -> _Stage1:
            groups = (
                tuple(self._index.search_names(list(anchors), limit=self._pool_size))
                if anchors
                else ()
            )
            facet_anchors = (
                tuple(self._index.search_facet_names(list(anchors), limit=self._pool_size))
                if anchors and self._config.facets
                else ()
            )
            suppressed = (
                self._suppression.suppressed_fact_keys()
                if self._suppression is not None
                else frozenset()
            )
            return _Stage1(
                anchors=groups, suppressed=frozenset(suppressed), facet_anchors=facet_anchors
            )

        return await self._call(self._executors.index, work, deadline=deadline, ledger=ledger)

    async def _stage2(
        self,
        request: MemoryRequest,
        stage1: _Stage1,
        deadline: float,
        started: float,
        ledger: _Ledger,
        reuse: PrefetchReuse,
        *,
        served: bool,
    ) -> FetchOutcome:
        now = request.now
        window = parse_window(request.query_text, now=now, tz=self._config.tz)
        stage2_started = self._clock.monotonic()
        query = request.query_text
        embed_task: asyncio.Task[list[float]] | None = None
        need_task: asyncio.Task[NeedTrace] | None = None
        if query.strip():
            # The fetch's own query embedding first, then the gate's work, so the gate's
            # vectors are submitted beside it at the start of the stage.
            embed_task = asyncio.ensure_future(
                self._call(
                    self._executors.model,
                    self._embed_query,
                    query,
                    deadline=deadline,
                    ledger=ledger,
                )
            )
            if self._need_gate is not None:
                need_task = asyncio.ensure_future(
                    self._decide_need(request, embed_task, deadline, ledger, served=served)
                )
        pools_task = asyncio.ensure_future(
            self._pools(request, stage1, deadline, ledger, embed_task)
        )
        need: NeedTrace | None = None
        try:
            if need_task is not None:
                # The gate decides before the cross-encoder in both modes, so a shadow fetch
                # does all the work a gated one does.
                need = await need_task
                if self._need_mode == "on" and need.band == "closed":
                    pools_task.cancel()
                    await asyncio.gather(pools_task, return_exceptions=True)
                    ledger.stage2_s = self._clock.monotonic() - stage2_started
                    return self._closed(request, started, ledger, reuse, need)
            candidates, versions = await pools_task
        finally:
            for task in (need_task, pools_task, embed_task):
                if task is not None and not task.done():
                    task.cancel()
            if embed_task is not None and embed_task.done() and not embed_task.cancelled():
                embed_task.exception()  # retrieved, so asyncio does not report it as never read
        ledger.stage2_s = self._clock.monotonic() - stage2_started

        fused = fuse(candidates)
        shortlisted, dropped, members = await self._drop_before_rerank(
            fused[: self._config.rerank_top_n], stage1.suppressed, now, deadline, ledger
        )
        scores = await self._rerank(request.query_text, shortlisted, deadline, ledger)
        ranked = order_candidates(shortlisted, scores, now=now, window=window)

        expansion = await self._expand(
            ranked, members, versions, stage1.suppressed, now, deadline, ledger
        )
        block = self._builder.build(expansion.items, request=request, window=window)

        placed: dict[tuple[str, str], tuple[CandidateDisposition, int | None]] = {}
        for outcome in block.outcomes:
            placed[_key(outcome.candidate.candidate)] = (outcome.disposition, outcome.item)
        stage_candidates: list[MemoryCandidate] = []
        for scored in ranked:
            key = _key(scored.candidate)
            if key in expansion.facets:
                disposition, item = _facet_disposition(expansion.facets[key], placed)
            elif key in placed:
                disposition, item = placed[key]
            else:
                disposition, item = "dropped_rank", None
            stage_candidates.append(
                _stage_candidate(scored, scored.position, disposition, item, versions.get(key[1]))
            )
        # Candidates that left the shortlist before the order are traced after it, by pool rank.
        for rank, (fused_candidate, reason) in enumerate(dropped, start=len(ranked) + 1):
            stage_candidates.append(
                _stage_candidate(
                    _unordered(fused_candidate),
                    rank,
                    reason,
                    None,
                    versions.get(fused_candidate.candidate.item_id),
                )
            )
        suppressed_count = sum(1 for _, reason in dropped if reason == "dropped_suppressed")

        if self._rerank_unavailable:
            ledger.reasons.add("reranker_unavailable")
        timings = self._timings(started, ledger, reuse)
        reason = ledger.reason()
        ordered = sorted(stage_candidates, key=lambda c: (c.disposition != "injected", c.rank))
        stage = MemoryStage(
            ran=True,
            skip_reason=None,
            level=request.serving.level,
            tier=request.serving.tier,
            block_version=BLOCK_VERSION,
            budget_tokens=block.budget_tokens,
            tokens=block.tokens,
            degraded_reason=reason,
            prefetch_reuse=reuse,
            reranked=len(scores),
            suppressed=suppressed_count,
            narrative_suppressed=0,
            timings=timings,
            candidates=tuple(ordered[:STAGE_CANDIDATE_CAP]),
            truncated=max(0, len(ordered) - STAGE_CANDIDATE_CAP),
            need=need,
        )
        result = MemoryResult(
            items=block.items,
            block_text=block.text,
            tokens=block.tokens,
            budget_tokens=block.budget_tokens,
            degraded=reason is not None,
            degraded_reason=reason,
            timings=timings,
            reranked=len(scores),
            suppressed=suppressed_count,
            narrative_suppressed=0,
            stage=stage,
        )
        return FetchOutcome(
            result=result,
            lists=block.lists,
            fused=tuple(fused),
            ranked=tuple(ranked),
            selected=expansion.selected,
        )

    async def _drop_before_rerank(
        self,
        head: Sequence[FusedCandidate],
        suppressed: frozenset[str],
        now: datetime,
        deadline: float,
        ledger: _Ledger,
    ) -> tuple[
        list[FusedCandidate],
        list[tuple[FusedCandidate, CandidateDisposition]],
        dict[str, tuple[IndexedFact, ...]],
    ]:
        """Take out of the shortlist what must not take a rerank pair or a slot.

        A suppressed fact leaves as `dropped_suppressed`. The members of every shortlisted
        facet are read in one batched lookup on the index executor; a facet left with fewer
        than two current members no correction suppresses leaves as `dropped_no_members`. A
        lookup that misses the deadline removes every facet as `dropped_late` and marks the
        fetch `deadline`, so the facts are still served. The shortlist is not refilled.

        Returns:
            The kept shortlist in pool order, the dropped candidates with their dispositions in
            pool order, and each kept facet's members, newest recorded first.

        Raises:
            _IndexFailedError: The member lookup failed.
        """
        dropped: dict[tuple[str, str], CandidateDisposition] = {}
        facet_keys: list[str] = []
        for fused in head:
            candidate = fused.candidate
            if candidate.kind == "fact" and candidate.item_id in suppressed:
                dropped[_key(candidate)] = "dropped_suppressed"
            elif candidate.kind == "facet":
                facet_keys.append(candidate.item_id)
        members: dict[str, tuple[IndexedFact, ...]] = {}
        if facet_keys:
            try:
                found = await self._call(
                    self._executors.index,
                    lambda: self._index.facet_members_many(facet_keys, now=now),
                    deadline=deadline,
                    ledger=ledger,
                )
            except _LateError:
                ledger.reasons.add("deadline")
                dropped.update((("facet", key), "dropped_late") for key in facet_keys)
            except MemoryIndexError as exc:
                logger.warning("memory facet member lookup failed: %s", exc)
                raise _IndexFailedError from exc
            else:
                for key in facet_keys:
                    current = tuple(
                        fact for fact in found.get(key, ()) if fact.fact_key not in suppressed
                    )
                    if len(current) < 2:
                        dropped[("facet", key)] = "dropped_no_members"
                    else:
                        members[key] = current
        kept = [fused for fused in head if _key(fused.candidate) not in dropped]
        gone = [
            (fused, dropped[_key(fused.candidate)])
            for fused in head
            if _key(fused.candidate) in dropped
        ]
        return kept, gone, members

    async def _pools(
        self,
        request: MemoryRequest,
        stage1: _Stage1,
        deadline: float,
        ledger: _Ledger,
        embed_task: asyncio.Task[list[float]] | None,
    ) -> tuple[list[Candidate], dict[str, IndexedFact]]:
        """Run the fact, facet and note pools concurrently and turn their hits into candidates.

        The facet FTS, vector and entity pools run only when `facets` is on; the facet entity
        pool is topped up with stage 1's facet anchors as the fact entity pool is with its
        fact anchors. Each pool has the deadline to itself; one that misses it is left out and
        the fetch is marked `deadline`, so the candidates of the pools that finished still
        rank. An index failure in any pool fails the fetch (`index_error`). An embedder failure
        costs the vector and note pools only. `embed_task` is the query's embedding, which stage 2
        starts and owns because the need gate may decide on the same vector; None for a blank
        query, which has no pools.
        """
        query = request.query_text
        if embed_task is None or not query.strip():
            return [], {}
        own_embedding = embed_task
        index = self._executors.index
        limit = self._pool_size

        async def fts() -> list[FactGroup]:
            return await self._call(
                index,
                lambda: self._index.search_fts(query, limit=limit),
                deadline=deadline,
                ledger=ledger,
            )

        async def names() -> list[FactGroup]:
            return await self._call(
                index,
                lambda: self._index.search_names([query], limit=limit),
                deadline=deadline,
                ledger=ledger,
            )

        async def embedding() -> list[float] | None:
            try:
                return await self._await(own_embedding, deadline=deadline)
            except _EMBEDDER_ERRORS as exc:
                logger.warning(
                    "memory query embedding failed with %s; the vector and note pools are "
                    "skipped",
                    type(exc).__name__,
                )
                return None

        async def vector() -> list[FactGroup]:
            vec = await embedding()
            if vec is None:
                return []
            try:
                return await self._call(
                    index,
                    lambda: self._index.search_vector(vec, limit=limit),
                    deadline=deadline,
                    ledger=ledger,
                )
            except ValueError as exc:
                logger.warning("memory vector pool skipped: %s", exc)
                return []

        async def notes() -> list[Candidate]:
            if self._notes is None:
                return []
            vec = await embedding()
            if vec is None:
                return []
            try:
                return await asyncio.wait_for(
                    self._notes.gather(query, vec, profile_path=self._profile_path),
                    timeout=self._remaining(deadline),
                )
            except TimeoutError as exc:
                raise _LateError from exc
            except MistError as exc:
                logger.warning("memory note pool skipped: %s: %s", type(exc).__name__, exc)
                return []

        async def facet_fts() -> list[IndexedFacet]:
            return await self._call(
                index,
                lambda: self._index.search_facets_fts(query, limit=limit),
                deadline=deadline,
                ledger=ledger,
            )

        async def facet_names() -> list[IndexedFacet]:
            return await self._call(
                index,
                lambda: self._index.search_facet_names([query], limit=limit),
                deadline=deadline,
                ledger=ledger,
            )

        async def facet_vector() -> list[IndexedFacet]:
            vec = await embedding()
            if vec is None:
                return []
            try:
                return await self._call(
                    index,
                    lambda: self._index.search_facets_vector(vec, limit=limit),
                    deadline=deadline,
                    ledger=ledger,
                )
            except ValueError as exc:
                logger.warning("memory facet vector pool skipped: %s", exc)
                return []

        tasks: dict[str, asyncio.Future[list]] = {
            FTS_POOL: asyncio.ensure_future(fts()),
            ENTITY_POOL: asyncio.ensure_future(names()),
            VECTOR_POOL: asyncio.ensure_future(vector()),
            "notes": asyncio.ensure_future(notes()),
        }
        if self._config.facets:
            tasks[_FACET_FTS] = asyncio.ensure_future(facet_fts())
            tasks[_FACET_ENTITY] = asyncio.ensure_future(facet_names())
            tasks[_FACET_VECTOR] = asyncio.ensure_future(facet_vector())
        try:
            await asyncio.wait(tasks.values())
        finally:
            # A closed gate cancels the pools; those already finished are read here so asyncio
            # does not report their exceptions as never retrieved.
            for task in tasks.values():
                if not task.done():
                    task.cancel()
                elif not task.cancelled():
                    task.exception()
        # Each pool's hits by task name: fact groups, facets or note candidates.
        found: dict[str, list] = {}
        for name, task in tasks.items():
            exc = task.exception()
            if exc is None:
                found[name] = task.result()
            elif isinstance(exc, _LateError):
                ledger.reasons.add("deadline")
            elif isinstance(exc, MemoryIndexError):
                logger.warning("memory %s pool failed: %s", name, exc)
                raise _IndexFailedError from exc
            else:
                raise exc

        query_anchors = _with_anchors(
            found.get(ENTITY_POOL, []), stage1.anchors, limit, key=lambda group: group.fact_key
        )
        pools: tuple[tuple[str, list[FactGroup]], ...] = (
            (FTS_POOL, list(found.get(FTS_POOL, []))),
            (VECTOR_POOL, list(found.get(VECTOR_POOL, []))),
            (ENTITY_POOL, query_anchors),
        )
        groups: dict[str, FactGroup] = {}
        ranks: dict[str, dict[str, int]] = {}
        for pool, hits in pools:
            for rank, group in enumerate(hits, start=1):
                groups.setdefault(group.fact_key, group)
                ranks.setdefault(group.fact_key, {})[pool] = rank
        candidates: list[Candidate] = []
        versions: dict[str, IndexedFact] = {}
        for key, group in groups.items():
            version = group.current_at(request.now)
            if version is None:
                continue
            versions[key] = version
            candidates.append(_fact_candidate(version, ranks[key]))
        if self._config.facets:
            # A facet's pools carry the fact pools' names, so a facet found first by its pools
            # fuses as a fact found first by the same pools would.
            facet_anchors = _with_anchors(
                found.get(_FACET_ENTITY, []),
                stage1.facet_anchors,
                limit,
                key=lambda facet: facet.facet_key,
            )
            facet_pools: tuple[tuple[str, list[IndexedFacet]], ...] = (
                (FTS_POOL, list(found.get(_FACET_FTS, []))),
                (VECTOR_POOL, list(found.get(_FACET_VECTOR, []))),
                (ENTITY_POOL, facet_anchors),
            )
            facets: dict[str, IndexedFacet] = {}
            facet_ranks: dict[str, dict[str, int]] = {}
            for pool, facet_hits in facet_pools:
                for rank, facet in enumerate(facet_hits, start=1):
                    facets.setdefault(facet.facet_key, facet)
                    facet_ranks.setdefault(facet.facet_key, {})[pool] = rank
            candidates.extend(
                _facet_candidate(facet, facet_ranks[key]) for key, facet in facets.items()
            )
        candidates.extend(found.get("notes", []))
        return candidates, versions

    async def _rerank(
        self,
        query: str,
        shortlisted: Sequence[FusedCandidate],
        deadline: float,
        ledger: _Ledger,
    ) -> tuple[float, ...]:
        """Score the head of the shortlist that fits the time left, on each view or text.

        A candidate the cross-encoder did not reach sorts by its pool rank alone; with no
        scores at all the order is the pool order.
        """
        if not shortlisted:
            return ()
        remaining_ms = self._remaining(deadline) * 1000.0
        if remaining_ms <= 0:
            ledger.reasons.add("deadline")
            return ()
        count = min(len(shortlisted), max(0, self._reranker.pairs_within(remaining_ms)))
        if count == 0:
            return ()
        texts = [fused.candidate.view_text or fused.candidate.text for fused in shortlisted[:count]]
        rerank_started = self._clock.monotonic()
        try:
            scores = await self._call(
                self._executors.model,
                self._reranker.score,
                query,
                texts,
                remaining_ms,
                deadline=deadline,
                ledger=ledger,
            )
        except _LateError:
            ledger.reasons.add("deadline")
            return ()
        except MemoryRerankError as exc:
            logger.warning("memory rerank failed, keeping the pool order: %s", exc)
            ledger.reasons.add("reranker_unavailable")
            return ()
        finally:
            ledger.rerank_s = self._clock.monotonic() - rerank_started
        values = tuple(float(score) for score in scores)
        if len(values) > count or any(
            math.isnan(value) or not 0.0 <= value <= 1.0 for value in values
        ):
            logger.warning(
                "memory reranker returned %d scores for %d pairs or a score outside [0, 1]; "
                "keeping the pool order",
                len(values),
                count,
            )
            ledger.reasons.add("reranker_unavailable")
            return ()
        return values

    async def _expand(
        self,
        ranked: list[ScoredCandidate],
        members: dict[str, tuple[IndexedFact, ...]],
        versions: dict[str, IndexedFact],
        suppressed: frozenset[str],
        now: datetime,
        deadline: float,
        ledger: _Ledger,
    ) -> _Expansion:
        """Select the first `select_k` candidates and expand them into the block's items.

        A facet counts as one. Facts sharing `subject_id`, `rel_type` and
        `source_utterance_id` are siblings: when two or more are selected alone, every current
        sibling no correction suppresses joins them (plan v2 V7). A selected facet brings its
        members, up to `list_max_items`: those also selected alone first, in sort order, then
        the rest newest recorded first. A fact expansion brings takes the position and order
        score of the candidate that brought it, or keeps its own when that is earlier, so the
        block packs it where its bringer would be packed. Each fact is placed once: facts of
        one subject, relation and utterance that a sibling lookup or a facet brought make one
        `FactList`, capped at `list_max_items`; the members past the cap that are candidates
        stay single items, and facet members of other utterances are single dated facts.

        Raises:
            _IndexFailedError: The sibling lookup failed.
        """
        cap = self._config.list_max_items
        chosen = ranked[: self._config.select_k]
        by_key = {_key(scored.candidate): scored for scored in ranked}
        # Each placed fact, with its candidate; None for a sibling only the lookup brought,
        # which is listed but is not a candidate and gets no packing outcome.
        placed: dict[str, ScoredCandidate | None] = {}
        facts: dict[str, IndexedFact] = {}
        others: list[ScoredCandidate] = []
        for scored in chosen:
            candidate = scored.candidate
            if candidate.kind == "fact":
                placed[candidate.item_id] = scored
                version = versions.get(candidate.item_id)
                if version is not None:
                    facts[candidate.item_id] = version
            elif candidate.kind != "facet":
                others.append(scored)
        alone = set(placed)

        groups: dict[tuple[str, str, str], list[ScoredCandidate]] = {}
        for scored in chosen:
            version = (
                facts.get(scored.candidate.item_id) if scored.candidate.kind == "fact" else None
            )
            if version is None or version.source_utterance_id is None:
                continue
            sibling_key = (version.subject_id, version.rel_type, version.source_utterance_id)
            groups.setdefault(sibling_key, []).append(scored)
        listed = {key: group for key, group in groups.items() if len(group) >= 2}
        listable: set[tuple[str, str, str]] = set()
        if listed:

            def lookup() -> dict[tuple[str, str, str], tuple[IndexedFact, ...]]:
                return {
                    key: self._index.siblings(key[0], key[1], key[2], now=now) for key in listed
                }

            try:
                found = await self._call(
                    self._executors.index, lookup, deadline=deadline, ledger=ledger
                )
            except _LateError:
                ledger.reasons.add("deadline")
                found = {}
            except MemoryIndexError as exc:
                logger.warning("memory sibling lookup failed: %s", exc)
                raise _IndexFailedError from exc
            for sibling_key, group in sorted(
                listed.items(), key=lambda pair: min(_position_order(s) for s in pair[1])
            ):
                if sibling_key not in found:
                    continue
                listable.add(sibling_key)
                bringer = min(group, key=_position_order)
                siblings = [fact for fact in found[sibling_key] if fact.fact_key not in suppressed]
                # Siblings the fetch ranked join before those it never saw.
                joining = sorted(
                    (fact for fact in siblings if ("fact", fact.fact_key) in by_key),
                    key=lambda fact: _position_order(by_key[("fact", fact.fact_key)]),
                ) + [fact for fact in siblings if ("fact", fact.fact_key) not in by_key]
                room = cap - len(group)
                for fact in joining:
                    if room <= 0:
                        break
                    if fact.fact_key in placed:
                        continue
                    entry = by_key.get(("fact", fact.fact_key))
                    placed[fact.fact_key] = None if entry is None else _brought(entry, bringer)
                    facts[fact.fact_key] = fact
                    room -= 1

        facet_members: dict[tuple[str, str], tuple[str, ...]] = {}
        for scored in chosen:
            if scored.candidate.kind != "facet":
                continue
            current = members.get(scored.candidate.item_id, ())
            first = sorted(
                (fact for fact in current if fact.fact_key in alone),
                key=lambda fact: _position_order(by_key[("fact", fact.fact_key)]),
            )
            taken = (first + [fact for fact in current if fact.fact_key not in alone])[:cap]
            for fact in taken:
                key = fact.fact_key
                existing = placed.get(key)
                if existing is None:
                    entry = by_key.get(("fact", key))
                    placed[key] = (
                        _brought(entry, scored)
                        if entry is not None
                        else _member_candidate(fact, scored)
                    )
                    facts.setdefault(key, fact)
                elif existing.position > scored.position:
                    placed[key] = replace(existing, position=scored.position, score=scored.score)
                version = facts[key]
                if version.source_utterance_id is not None:
                    listable.add(
                        (version.subject_id, version.rel_type, version.source_utterance_id)
                    )
            facet_members[_key(scored.candidate)] = tuple(fact.fact_key for fact in taken)

        items: list[ScoredCandidate | FactList] = list(others)
        grouped: dict[tuple[str, str, str], list[str]] = {}
        for key, entry in placed.items():
            version = facts.get(key)
            group_key = (
                None
                if version is None or version.source_utterance_id is None
                else (version.subject_id, version.rel_type, version.source_utterance_id)
            )
            if group_key is not None and group_key in listable:
                grouped.setdefault(group_key, []).append(key)
            elif entry is not None:
                items.append(entry)
        for keys in grouped.values():
            ranked_keys = sorted(
                (key for key in keys if placed[key] is not None),
                key=lambda key: _position_order(placed[key]),  # type: ignore[arg-type]
            )
            ordered = ranked_keys + [key for key in keys if placed[key] is None]
            in_list = ordered[:cap]
            listed_scored = tuple(
                entry for entry in (placed[key] for key in in_list) if entry is not None
            )
            rest = ordered[cap:]
            if len(in_list) >= 2 and listed_scored:
                items.append(
                    FactList(
                        facts=tuple(sorted((facts[key] for key in in_list), key=_list_order)),
                        scored=listed_scored,
                    )
                )
            else:
                rest = ordered
            items.extend(entry for entry in (placed[key] for key in rest) if entry is not None)

        selected: list[ScoredCandidate] = [
            scored for scored in chosen if scored.candidate.kind == "facet"
        ]
        for item in items:
            selected.extend(item.scored if isinstance(item, FactList) else (item,))
        return _Expansion(
            items=items,
            selected=tuple(sorted(selected, key=_position_order)),
            facets=facet_members,
        )

    # -- the need gate --------------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        """The fetch's own query embedding, on the model thread."""
        lock = self._shared_embed_lock
        if lock is None:
            return self._embedder.generate_embedding(text)
        with lock:
            return self._embedder.generate_embedding(text)

    def _gate_embed(self, gate: NeedGate, text: str) -> list[float]:
        """One of the gate's own embeddings, on the gate thread."""
        lock = self._shared_embed_lock
        if lock is None:
            return gate.embed_query(text)
        with lock:
            return gate.embed_query(text)

    async def _decide_need(
        self,
        request: MemoryRequest,
        own_embedding: asyncio.Task[list[float]],
        deadline: float,
        ledger: _Ledger,
        *,
        served: bool,
    ) -> NeedTrace:
        """The gate's decision for the turn, within the fetch's deadline.

        The gate's vectors are submitted at once: the query's on the gate thread (or the
        fetch's own vector, when the gate shares it), then the follow-up context's when the
        session's last fetch served memory and the request carries a recent turn. The query
        decides first. An open query never waits for the context, whose result is discarded;
        a closed or band one waits for it and takes the higher probability and its band. A
        vector that misses the deadline closes the gate as `deadline`, an embedder error or a
        vector `decide` refuses as `need_unknown`.
        """
        gate = self._need_gate
        if gate is None:
            raise RuntimeError("the need gate is decided only when one is given")
        query = request.query_text
        mode = self._need_mode
        executor = self._executors.gate
        query_vector: asyncio.Future[list[float]] = (
            own_embedding
            if gate.shares_query_vector
            else asyncio.ensure_future(
                self._call(
                    executor, self._gate_embed, gate, query, deadline=deadline, ledger=ledger
                )
            )
        )
        previous = _previous_turn(request) if self._need_context and served else None
        context_vector: asyncio.Future[list[float]] | None = None
        if previous is not None:
            context_vector = asyncio.ensure_future(
                self._call(
                    executor,
                    self._gate_embed,
                    gate,
                    context_text(previous, query),
                    deadline=deadline,
                    ledger=ledger,
                )
            )
        try:
            try:
                decision = gate.decide(await self._await(query_vector, deadline=deadline))
            except _LateError:
                return _degraded_need(gate, mode, "deadline", None)
            except _EMBEDDER_ERRORS as exc:
                logger.warning("memory need gate could not score the query: %s", exc)
                return _degraded_need(gate, mode, "need_unknown", None)
            if decision.band == "open" or context_vector is None:
                return _need_trace(mode, decision, None)
            try:
                context = gate.decide(await self._await(context_vector, deadline=deadline))
            except _LateError:
                return _degraded_need(gate, mode, "deadline", decision)
            except _EMBEDDER_ERRORS as exc:
                logger.warning("memory need gate could not score the follow-up context: %s", exc)
                return _degraded_need(gate, mode, "need_unknown", decision)
            return _need_trace(mode, decision, context)
        finally:
            for vector in (query_vector, context_vector):
                if vector is not None and vector is not own_embedding:
                    if not vector.done():
                        vector.cancel()
                    elif not vector.cancelled():
                        vector.exception()  # retrieved, so asyncio does not report it

    def _closed(
        self,
        request: MemoryRequest,
        started: float,
        ledger: _Ledger,
        reuse: PrefetchReuse,
        need: NeedTrace,
    ) -> FetchOutcome:
        """The outcome of a fetch the `on` gate closed: skipped for `no_need`, or degraded."""
        if need.degraded is None:
            timings = self._timings(started, ledger, reuse)
            return FetchOutcome(
                result=self._skipped(request, "no_need", timings, reuse=reuse, need=need)
            )
        ledger.reasons.add(need.degraded)
        reason = ledger.reason() or need.degraded
        return self._empty(request, started, ledger, reason, reuse, need=need)

    # -- prefetch -------------------------------------------------------------------------------

    def _prefetch(self, session_id: str, partial_text: str | None) -> None:
        session = self._session(session_id)
        context = session.context
        anchors = _anchor_texts(context.recent_user_turns)
        now = self._clock.monotonic()
        if partial_text is None or not partial_text.strip():
            entry = session.stage1
            if (
                entry is None
                or entry.generation < session.floor
                or entry.shape_anchors != anchors
                or now - entry.started > STAGE1_TTL_S
            ):
                session.stage1 = self._start_stage1(session_id, anchors, None)
            return
        request = MemoryRequest(
            session_id=session_id,
            conversation_id=context.conversation_id,
            query_text=partial_text,
            now=self._clock.now(),
            recent_user_turns=context.recent_user_turns,
            turn_index=context.turn_index,
            conversation_start=context.conversation_start,
            budget_tokens=context.budget_tokens,
            serving=_serving(context),
        )
        norm = normalise(partial_text)
        shape = self._shape(request)
        running = session.running
        if running is not None and not running.task.done():
            if running.norm == norm and running.shape == shape:
                return
            running.task.cancel()
        completed = session.completed
        if (
            completed is not None
            and completed.norm == norm
            and completed.shape == shape
            and completed.generation >= session.floor
            and now - completed.finished <= STAGE1_TTL_S
        ):
            session.running = None
            return
        session.stage1 = self._start_stage1(session_id, anchors, partial_text)
        task = asyncio.get_running_loop().create_task(
            self._prefetch_run(session, norm, shape, request, self._generation),
            name=f"memory-prefetch-{session_id}",
        )
        session.running = _Running(norm=norm, shape=shape, task=task)

    def _start_stage1(
        self, session_id: str, anchors: tuple[str, ...], text: str | None
    ) -> _Stage1Entry:
        started = self._clock.monotonic()
        deadline = started + self._prefetch_deadline_s
        future = asyncio.get_running_loop().create_task(
            self._stage1(anchors, deadline=deadline, ledger=None),
            name=f"memory-stage1-{session_id}",
        )
        future.add_done_callback(_consume)
        return _Stage1Entry(
            shape_anchors=anchors,
            text=text,
            started=started,
            future=future,
            generation=self._generation,
        )

    async def _prefetch_run(
        self,
        session: _Session,
        norm: str,
        shape: _Shape,
        request: MemoryRequest,
        generation: int,
    ) -> FetchOutcome | None:
        """The whole fetch for a prefetch's words, kept for reuse if it completes cleanly."""
        started = self._clock.monotonic()
        deadline = started + self._prefetch_deadline_s
        ledger = _Ledger()
        entry = session.stage1
        try:
            if entry is not None and entry.text == request.query_text:
                stage1 = await self._await(entry.future, deadline=deadline)
            else:
                stage1 = await self._stage1(shape.anchors, deadline=deadline, ledger=ledger)
            state = await self._call(
                self._executors.index, self._index_state, deadline=deadline, ledger=ledger
            )
            if state is not None:
                return None
            outcome = await self._stage2(
                request, stage1, deadline, started, ledger, "none", served=session.served
            )
        except (_LateError, _GoneError, _IndexFailedError):
            return None
        except _CONTAINED_ERRORS as exc:
            logger.warning("memory prefetch failed: %s", type(exc).__name__, exc_info=exc)
            return None
        reason = outcome.result.degraded_reason
        if reason is not None and reason != "reranker_unavailable":
            return None
        if generation >= session.floor:
            session.completed = _Completed(
                norm=norm,
                shape=shape,
                outcome=outcome,
                finished=self._clock.monotonic(),
                generation=generation,
            )
        return outcome

    # -- helpers --------------------------------------------------------------------------------

    def _stage1_reusable(
        self, session: _Session, entry: _Stage1Entry, shape: _Shape, query: str
    ) -> bool:
        if entry.generation < session.floor or entry.shape_anchors != shape.anchors:
            return False
        if self._clock.monotonic() - entry.started > STAGE1_TTL_S:
            return False
        if entry.future.cancelled() or (
            entry.future.done() and entry.future.exception() is not None
        ):
            return False
        if entry.text is None:
            return True
        return jaccard(entry.text, query) >= STAGE1_REUSE_JACCARD

    async def _call(
        self,
        executor: Executor,
        fn: Callable[..., T],
        *args: object,
        deadline: float,
        ledger: _Ledger | None,
    ) -> T:
        """Run `fn` on `executor`, waiting no later than `deadline`.

        Raises:
            _LateError: The call did not finish in time, or finished after the deadline by the
                injected clock; its result is dropped.
        """
        submitted = self._clock.monotonic()
        began: list[float] = []

        def timed() -> T:
            began.append(self._clock.monotonic())
            return fn(*args)

        future = asyncio.get_running_loop().run_in_executor(executor, timed)
        try:
            result = await asyncio.wait_for(future, timeout=self._remaining(deadline))
        except TimeoutError as exc:
            raise _LateError from exc
        finally:
            if ledger is not None:
                start = began[0] if began else self._clock.monotonic()
                ledger.queue_wait_s += max(0.0, start - submitted)
        if self._clock.monotonic() > deadline:
            raise _LateError
        return result

    async def _await(self, awaitable: asyncio.Future[T], *, deadline: float) -> T:
        """Wait for work another call started, no later than `deadline`, without cancelling it.

        Raises:
            _LateError: The work did not finish in time.
            _GoneError: The work was cancelled while this call still runs.
        """
        try:
            result = await asyncio.wait_for(
                asyncio.shield(awaitable), timeout=self._remaining(deadline)
            )
        except TimeoutError as exc:
            raise _LateError from exc
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if awaitable.cancelled() and (current is None or current.cancelling() == 0):
                raise _GoneError from None
            raise
        if self._clock.monotonic() > deadline:
            raise _LateError
        return result

    def _remaining(self, deadline: float) -> float:
        return max(0.0, deadline - self._clock.monotonic())

    def _budget(self, request: MemoryRequest) -> int:
        """The serving level's budget, lowered by the request's, as the block packs it."""
        budget = self._config.block_tokens_for(request.serving)
        if request.budget_tokens is not None:
            budget = min(budget, request.budget_tokens)
        return max(0, budget)

    def _shape(self, request: MemoryRequest) -> _Shape:
        return _Shape(
            anchors=_anchor_texts(request.recent_user_turns),
            budget=self._budget(request),
            level=request.serving.level,
            tier=request.serving.tier,
            conversation_start=request.conversation_start,
        )

    def _session(self, session_id: str) -> _Session:
        now = self._clock.monotonic()
        for key in [
            key
            for key, session in self._sessions.items()
            if key != session_id
            and now - session.touched > _SESSION_IDLE_S
            and (session.running is None or session.running.task.done())
        ]:
            del self._sessions[key]
        session = self._sessions.get(session_id)
        if session is None:
            session = _Session()
            self._sessions[session_id] = session
        session.touched = now
        return session

    def _remember(self, request: MemoryRequest, outcome: FetchOutcome) -> None:
        """Keep what the session's next fetch and prefetch borrow.

        The words become a recent turn, and whether the fetch served memory decides whether
        the next one may be judged as its follow-up.
        """
        session = self._session(request.session_id)
        need = outcome.result.stage.need
        session.served = need is not None and need.band in _SERVED_BANDS
        turns = (*request.recent_user_turns, request.query_text)
        session.context = _Context(
            conversation_id=request.conversation_id,
            recent_user_turns=tuple(turns[-ANCHOR_TURNS:]),
            turn_index=request.turn_index + 1,
            conversation_start=False,
            budget_tokens=request.budget_tokens,
            serving_level=request.serving.level,
            serving_tier=request.serving.tier,
        )

    def _timings(
        self, started: float, ledger: _Ledger, reuse: PrefetchReuse | None
    ) -> FetchTimings:
        return FetchTimings(
            stage1_ms=ledger.stage1_s * 1000.0,
            stage2_ms=ledger.stage2_s * 1000.0,
            rerank_ms=ledger.rerank_s * 1000.0,
            queue_wait_ms=ledger.queue_wait_s * 1000.0,
            total_ms=(self._clock.monotonic() - started) * 1000.0,
            prefetch_hit=reuse is not None and reuse != "none",
        )

    def _reused(self, outcome: FetchOutcome, started: float, ledger: _Ledger) -> FetchOutcome:
        """A prefetched outcome served at the endpoint: its block, this fetch's timings."""
        timings = self._timings(started, ledger, "block")
        result = outcome.result
        stage = replace(result.stage, prefetch_reuse="block", timings=timings)
        return replace(outcome, result=replace(result, timings=timings, stage=stage))

    def _empty(
        self,
        request: MemoryRequest,
        started: float,
        ledger: _Ledger,
        reason: DegradedReason,
        reuse: PrefetchReuse,
        *,
        need: NeedTrace | None = None,
    ) -> FetchOutcome:
        """A stage that ran but serves no block, for `reason`."""
        timings = self._timings(started, ledger, reuse)
        budget = self._budget(request)
        stage = MemoryStage(
            ran=True,
            skip_reason=None,
            level=request.serving.level,
            tier=request.serving.tier,
            block_version=BLOCK_VERSION,
            budget_tokens=budget,
            tokens=0,
            degraded_reason=reason,
            prefetch_reuse=reuse,
            reranked=0,
            suppressed=0,
            narrative_suppressed=0,
            timings=timings,
            candidates=(),
            need=need,
        )
        return FetchOutcome(
            result=MemoryResult(
                items=(),
                block_text=None,
                tokens=0,
                budget_tokens=budget,
                degraded=True,
                degraded_reason=reason,
                timings=timings,
                reranked=0,
                suppressed=0,
                narrative_suppressed=0,
                stage=stage,
            )
        )

    def _skipped(
        self,
        request: MemoryRequest,
        reason: StageSkipReason,
        timings: FetchTimings,
        *,
        reuse: PrefetchReuse = "none",
        need: NeedTrace | None = None,
    ) -> MemoryResult:
        """A stage that did not run, for `reason`; no block and no degradation."""
        budget = self._budget(request)
        stage = MemoryStage(
            ran=False,
            skip_reason=reason,
            level=request.serving.level,
            tier=request.serving.tier,
            block_version=BLOCK_VERSION,
            budget_tokens=budget,
            tokens=0,
            degraded_reason=None,
            prefetch_reuse=reuse,
            reranked=0,
            suppressed=0,
            narrative_suppressed=0,
            timings=timings,
            candidates=(),
            need=need,
        )
        return MemoryResult(
            items=(),
            block_text=None,
            tokens=0,
            budget_tokens=budget,
            degraded=False,
            degraded_reason=None,
            timings=timings,
            reranked=0,
            suppressed=0,
            narrative_suppressed=0,
            stage=stage,
        )


def normalise(text: str) -> str:
    """`text` with whitespace runs collapsed and case folded: what a prefetch is matched by."""
    return " ".join(text.split()).casefold()


def jaccard(left: str, right: str) -> float:
    """Jaccard similarity of the two texts' folded token sets; 1 when both have none."""
    a = set(words(left))
    b = set(words(right))
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _anchor_texts(turns: Sequence[str]) -> tuple[str, ...]:
    return tuple(text for text in turns[-ANCHOR_TURNS:] if text and text.strip())


def _previous_turn(request: MemoryRequest) -> str | None:
    """The last non-blank recent user turn, which a follow-up is judged by; None for none."""
    for turn in reversed(request.recent_user_turns):
        if turn and turn.strip():
            return turn
    return None


def _need_trace(mode: NeedMode, query: NeedDecision, context: NeedDecision | None) -> NeedTrace:
    """The turn's decision: the query's, or the context's when its probability is higher.

    A higher probability never lowers the band, so the turn's band is the higher one's.
    `context` marks a context that also moved the band; the nearest exemplars come from the
    decision the probability came from.
    """
    chosen = query
    moved = False
    if context is not None and context.p > query.p:
        chosen = context
        moved = context.band != query.band
    return NeedTrace(
        mode=mode,
        band=chosen.band,
        p=chosen.p,
        theta_low=chosen.theta_low,
        theta_high=chosen.theta_high,
        band_k=chosen.band_k,
        need_ids=chosen.need_ids,
        no_need_ids=chosen.no_need_ids,
        context=moved,
        context_p=None if context is None else context.p,
        degraded=None,
    )


def _degraded_need(
    gate: NeedGate, mode: NeedMode, reason: NeedDegraded, query: NeedDecision | None
) -> NeedTrace:
    """A closed decision for a gate vector that was late or failed.

    It keeps the query's probability and exemplars when the query was decided and only the
    follow-up context's vector failed; the thresholds are the gate model's.
    """
    model = gate.model
    return NeedTrace(
        mode=mode,
        band="closed",
        p=None if query is None else query.p,
        theta_low=model.theta_low,
        theta_high=model.theta_high,
        band_k=model.band_k,
        need_ids=() if query is None else query.need_ids,
        no_need_ids=() if query is None else query.no_need_ids,
        context=False,
        context_p=None,
        degraded=reason,
    )


def _serving(context: _Context) -> ServingProfile:
    return ServingProfile(level=context.serving_level, tier=context.serving_tier)


def _key(candidate: Candidate) -> tuple[str, str]:
    return (candidate.kind, candidate.item_id)


def _position_order(scored: ScoredCandidate) -> tuple[int, str, str]:
    """The block's packing order: position, then item id and kind."""
    return (scored.position, scored.candidate.item_id, scored.candidate.kind)


def _list_order(fact: IndexedFact) -> tuple[str, str, str]:
    """`FactIndex.siblings` order, so a list names its facts as the utterance's list does."""
    return (fact.recorded_at or "", fact.object_name.casefold(), fact.row_key)


def _brought(entry: ScoredCandidate, bringer: ScoredCandidate) -> ScoredCandidate:
    """A ranked candidate expansion places: at its bringer's position, or its own if earlier."""
    if entry.position <= bringer.position:
        return entry
    return replace(entry, position=bringer.position, score=bringer.score)


def _member_candidate(fact: IndexedFact, facet: ScoredCandidate) -> ScoredCandidate:
    """A facet member the pools never returned, placed at the facet's position and score."""
    candidate = _fact_candidate(fact, {})
    return ScoredCandidate(
        candidate=candidate,
        score=facet.score,
        position=facet.position,
        pool_rank=facet.pool_rank,
        ce_rank=None,
        p_ce=None,
        time_term=facet.time_term,
        window_applied=facet.window_applied,
        confidence=confidence_term(candidate),
    )


def _unordered(fused: FusedCandidate) -> ScoredCandidate:
    """A candidate dropped before the order, scored by its pool rank alone for the trace."""
    candidate = fused.candidate
    return ScoredCandidate(
        candidate=candidate,
        score=order_score(fused.pool_rank, None),
        position=0,
        pool_rank=fused.pool_rank,
        ce_rank=None,
        p_ce=None,
        time_term=0.0,
        window_applied=False,
        confidence=confidence_term(candidate),
    )


def _facet_disposition(
    member_keys: Sequence[str],
    placed: dict[tuple[str, str], tuple[CandidateDisposition, int | None]],
) -> tuple[CandidateDisposition, int | None]:
    """A selected facet's disposition: injected at its first placed member's item, or not."""
    items = [
        item
        for disposition, item in (
            placed.get(("fact", key), ("dropped_rank", None)) for key in member_keys
        )
        if disposition == "injected" and item is not None
    ]
    if items:
        return "injected", min(items)
    return "dropped_budget", None


def _with_anchors(
    hits: Sequence[A], anchors: Sequence[A], limit: int, *, key: Callable[[A], str]
) -> list[A]:
    """The query's entity hits topped up to `limit` with the recent turns' anchors, once each."""
    merged = list(hits)
    seen = {key(hit) for hit in merged}
    for anchor in anchors:
        if len(merged) >= limit:
            break
        if key(anchor) not in seen:
            merged.append(anchor)
            seen.add(key(anchor))
    return merged


def _consume(future: asyncio.Future[object]) -> None:
    """Retrieve a background stage 1's exception, so asyncio does not log it as unread."""
    if not future.cancelled():
        future.exception()


def _instant(value: str | None) -> datetime | None:
    """A stored `recorded_at` as an aware instant, UTC when naive; None when unreadable."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _fact_candidate(version: IndexedFact, ranks: dict[str, int]) -> Candidate:
    """The candidate a fact offers: the version current at `now`, under its `fact_key`."""
    return Candidate(
        kind="fact",
        item_id=version.fact_key,
        text=version.text,
        when=_instant(version.recorded_at),
        source=FactSource(
            fact_key=version.fact_key,
            row_key=version.row_key,
            recorded_at=version.recorded_at,
            source_utterance_id=version.source_utterance_id,
        ),
        ranks=dict(ranks),
        confidence=version.confidence,
        view_text=version.view_text,
    )


def _facet_candidate(facet: IndexedFacet, ranks: dict[str, int]) -> Candidate:
    """The candidate a facet offers: its view as both the text and the reranker's view.

    It has no time, so its recency is 0, and no confidence of its own.
    """
    return Candidate(
        kind="facet",
        item_id=facet.facet_key,
        text=facet.view_text,
        when=None,
        source=FacetSource(
            facet_key=facet.facet_key,
            subject_id=facet.subject_id,
            rel_type=facet.rel_type,
            object_type=facet.object_type,
        ),
        ranks=dict(ranks),
        view_text=facet.view_text,
    )


def _stage_candidate(
    scored: ScoredCandidate,
    rank: int,
    disposition: CandidateDisposition,
    item: int | None,
    version: IndexedFact | None,
) -> MemoryCandidate:
    """The trace's record of one candidate: refs the tools would give it, never its text.

    `rank` is the candidate's position in the order, or for one dropped before the order, a
    place after every ordered candidate. A facet's ref is `facet:<facet_key>`, with no
    `alt_ref`.
    """
    candidate = scored.candidate
    source = _SOURCE_BY_KIND[candidate.kind]
    alt_ref: str | None = None
    if candidate.kind == "facet":
        ref = f"{FACET_REF_PREFIX}{candidate.item_id}"
    elif candidate.kind == "fact" and version is not None:
        ref = graph_ref(version.subject_id, version.rel_type, version.object_id)
        alt_ref = graph_ref(version.object_id, version.rel_type, version.subject_id)
    elif isinstance(candidate.source, NoteSource):
        heading = candidate.source.heading if candidate.source.heading is not None else FILE_HEADING
        ref = vault_ref(candidate.source.path, heading, candidate.text)
    else:
        digest = hashlib.sha256(candidate.item_id.encode("utf-8")).hexdigest()[:20]
        ref = f"{candidate.kind}:{digest}"
    if source == "graph" and alt_ref is None:
        alt_ref = ref
    return MemoryCandidate(
        source=source,
        ref=ref,
        alt_ref=alt_ref,
        rank=rank,
        score=scored.score,
        item=item,
        disposition=disposition,
        pool_rank=scored.pool_rank,
        ce_rank=scored.ce_rank,
        p_ce=scored.p_ce,
    )


__all__ = [
    "GATE_EXECUTOR",
    "INDEX_EXECUTOR",
    "MODEL_EXECUTOR",
    "FactSearch",
    "FetchClock",
    "FetchOutcome",
    "FetcherExecutors",
    "MemoryFetcher",
    "NoteSearch",
    "SuppressionSource",
    "SystemClock",
    "jaccard",
    "normalise",
]
