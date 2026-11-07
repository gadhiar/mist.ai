"""The per-turn memory fetch: candidate pools, rerank, sibling lists and the budgeted block.

`MemoryFetcher.fetch` turns one `MemoryRequest` into a `MemoryResult` under a critical-path
deadline (`MemoryConfig.deadline_ms`, 50 ms by default) counted from the call. The work falls
in two stages (Design point 8):

- Stage 1 needs no transcript: the index state (`index_rebuilding` while a rebuild runs), the
  entity anchors of the last two user utterances, and the suppression set of the corrections.
  It is cached per session for 30 s, so a prefetch started at speech start takes it off the
  critical path.
- Stage 2 depends on the words: the FTS, vector and entity-anchored fact pools, the vault note
  pool, RRF fusion, the cross-encoder on the pairs that fit the time left, the weighted score,
  the `min_score` threshold, suppression, sibling expansion (plan v2 V7) and packing.

Blocking work runs on the injected executors: SQLite on `memory-index`, the query embedding and
the rerank on `memory-model`. A thread cannot be cancelled, so at the deadline the fetcher stops
waiting, discards whatever arrives late and returns what has finished, marked
`degraded_reason="deadline"`. The time a call spent queued behind earlier work is counted as
`queue_wait_ms`.

`prefetch(session_id, text)` runs the whole fetch for the latest words of a session, one in
flight per session, and keeps the last completed result with its normalised text. At the
endpoint `fetch` reuses that result when the final words normalise to the same text
(`prefetch_reuse="block"`), awaits a running prefetch of the same text within the deadline,
reuses a cached stage 1 when the words are close enough (Jaccard 0.8 on the token sets,
`stage1`), and otherwise runs fresh (`none`).

`fetch` never raises into the turn: an index failure returns an empty result marked
`index_error`, a reranker failure keeps the RRF scores marked `reranker_unavailable`, and any
other contained failure returns an empty stage marked `skip_reason="error"`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import sqlite3
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from backend.errors import MemoryIndexError, MemoryRerankError, MistError
from backend.interfaces import EmbeddingProvider
from backend.knowledge.models import graph_ref, vault_ref

from .block import BLOCK_VERSION, FactList, MemoryBlockBuilder, PlacedList
from .config import MemoryConfig
from .fact_index import FactGroup, IndexedFact
from .fts import words
from .models import (
    Candidate,
    CandidateDisposition,
    CandidateSource,
    DegradedReason,
    FactSource,
    FetchTimings,
    MemoryCandidate,
    MemoryRequest,
    MemoryResult,
    MemoryStage,
    NoteSource,
    PrefetchReuse,
    ServingProfile,
    StageSkipReason,
)
from .ranking import FusedCandidate, ScoredCandidate, score_shortlist, shortlist
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

# Executor names the runtime builds and the factory passes in (Design point 8).
INDEX_EXECUTOR = "memory-index"
MODEL_EXECUTOR = "memory-model"

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
}

# One degraded reason per result; the more fundamental failure is reported.
_DEGRADED_PRIORITY: tuple[DegradedReason, ...] = (
    "index_error",
    "index_rebuilding",
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
    """The two executors the fetcher submits blocking work to; the runtime owns them.

    Attributes:
        index: `memory-index`, for SQLite reads (two threads, shared with the refresher).
        model: `memory-model`, for the query embedding and the rerank (one thread).
    """

    index: Executor
    model: Executor


@dataclass(frozen=True, slots=True)
class FetchOutcome:
    """A fetch's result and the fact lists its block placed.

    The result is what the turn and the trace use. `lists` lets an offline reader, such as
    the recall harness, flatten a list item back into its facts: the result names a list by
    its best-scored member only, and a sibling only the expansion brought in is not a
    candidate.
    """

    result: MemoryResult
    lists: tuple[PlacedList, ...] = ()


@dataclass(frozen=True, slots=True)
class _Stage1:
    """What stage 1 found: the anchors of the recent turns and the suppressed fact keys."""

    anchors: tuple[FactGroup, ...]
    suppressed: frozenset[str]


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
        config: The memory settings: deadline, `min_score`, `rerank_top_n`,
            `list_max_items`, the zone, and the budget per serving level.
        executors: The `memory-index` and `memory-model` executors.
        clock: Monotonic seconds for the deadline and the wall clock for a prefetch's `now`.
        index_state: Returns the index's degraded reason, `index_db.degraded_reason` on a
            connection of its own (`runtime.IndexStateProbe`); checked on every fetch.
        notes: The vault note pool, or None when no sidecar is configured.
        suppression: The corrections' suppression set (Phase C), or None for none.
        profile_path: The always-injected user profile note, kept out of the note pool.
        pool_size: Facts in each of the FTS, vector and entity pools.
        prefetch_deadline_ms: The budget of a prefetch's whole fetch.

    Raises:
        ValueError: `pool_size` or `prefetch_deadline_ms` is not positive.
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
        self._index = index
        self._embedder = embedder
        self._reranker = reranker
        self._builder = block_builder
        self._config = config
        self._executors = executors
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
        self._remember(request)
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
            return await self._stage2(request, stage1, deadline, started, ledger, reuse)
        except _IndexFailedError:
            return self._empty(request, started, ledger, "index_error", reuse)

    async def _stage1(
        self, anchors: tuple[str, ...], *, deadline: float, ledger: _Ledger | None
    ) -> _Stage1:
        """Run stage 1 on the index executor: the recent turns' anchors and the suppression set."""

        def work() -> _Stage1:
            groups = (
                tuple(self._index.search_names(list(anchors), limit=self._pool_size))
                if anchors
                else ()
            )
            suppressed = (
                self._suppression.suppressed_fact_keys()
                if self._suppression is not None
                else frozenset()
            )
            return _Stage1(anchors=groups, suppressed=frozenset(suppressed))

        return await self._call(self._executors.index, work, deadline=deadline, ledger=ledger)

    async def _stage2(
        self,
        request: MemoryRequest,
        stage1: _Stage1,
        deadline: float,
        started: float,
        ledger: _Ledger,
        reuse: PrefetchReuse,
    ) -> FetchOutcome:
        now = request.now
        window = parse_window(request.query_text, now=now, tz=self._config.tz)
        stage2_started = self._clock.monotonic()
        candidates, versions = await self._pools(request, stage1, deadline, ledger)
        ledger.stage2_s = self._clock.monotonic() - stage2_started

        shortlisted = shortlist(candidates, top_n=self._config.rerank_top_n)
        scores = await self._rerank(request.query_text, shortlisted, deadline, ledger)
        unthresholded = replace(self._config, min_score=0.0)
        ranked = score_shortlist(shortlisted, scores, now=now, window=window, config=unthresholded)

        kept: list[ScoredCandidate] = []
        below: dict[tuple[str, str], ScoredCandidate] = {}
        suppressed: set[tuple[str, str]] = set()
        for scored in ranked:
            key = _key(scored.candidate)
            if scored.score < self._config.min_score:
                below[key] = scored
            elif scored.candidate.kind == "fact" and scored.candidate.item_id in stage1.suppressed:
                suppressed.add(key)
            else:
                kept.append(scored)

        items = await self._expand(kept, below, versions, stage1.suppressed, now, deadline, ledger)
        block = self._builder.build(items, request=request, window=window)

        placed: dict[tuple[str, str], tuple[CandidateDisposition, int | None]] = {}
        for outcome in block.outcomes:
            placed[_key(outcome.candidate.candidate)] = (outcome.disposition, outcome.item)
        stage_candidates: list[MemoryCandidate] = []
        for rank, scored in enumerate(ranked, start=1):
            key = _key(scored.candidate)
            if key in placed:
                disposition, item = placed[key]
            elif key in suppressed:
                disposition, item = "dropped_suppressed", None
            else:
                disposition, item = "dropped_min_score", None
            stage_candidates.append(
                _stage_candidate(scored, rank, disposition, item, versions.get(key[1]))
            )

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
            suppressed=len(suppressed),
            narrative_suppressed=0,
            timings=timings,
            candidates=tuple(ordered[:STAGE_CANDIDATE_CAP]),
            truncated=max(0, len(ordered) - STAGE_CANDIDATE_CAP),
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
            suppressed=len(suppressed),
            narrative_suppressed=0,
            stage=stage,
        )
        return FetchOutcome(result=result, lists=block.lists)

    async def _pools(
        self, request: MemoryRequest, stage1: _Stage1, deadline: float, ledger: _Ledger
    ) -> tuple[list[Candidate], dict[str, IndexedFact]]:
        """Run the fact and note pools concurrently and turn their hits into candidates.

        Each pool has the deadline to itself; one that misses it is left out and the fetch is
        marked `deadline`, so the candidates of the pools that finished still rank. An index
        failure in any pool fails the fetch (`index_error`). An embedder failure costs the
        vector and note pools only.
        """
        query = request.query_text
        if not query.strip():
            return [], {}
        index = self._executors.index
        model = self._executors.model
        limit = self._pool_size

        embed_task = asyncio.ensure_future(
            self._call(
                model, self._embedder.generate_embedding, query, deadline=deadline, ledger=ledger
            )
        )

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
                return await self._await(embed_task, deadline=deadline)
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

        tasks = {
            FTS_POOL: asyncio.ensure_future(fts()),
            ENTITY_POOL: asyncio.ensure_future(names()),
            VECTOR_POOL: asyncio.ensure_future(vector()),
            "notes": asyncio.ensure_future(notes()),
        }
        try:
            await asyncio.wait(tasks.values())
        finally:
            for task in (*tasks.values(), embed_task):
                if not task.done():
                    task.cancel()
        found: dict[str, object] = {}
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
        if embed_task.done() and not embed_task.cancelled():
            embed_task.exception()  # retrieved, so asyncio does not report it as never read

        query_anchors: list[FactGroup] = list(found.get(ENTITY_POOL, []))  # type: ignore[arg-type]
        seen = {group.fact_key for group in query_anchors}
        for group in stage1.anchors:
            if len(query_anchors) >= limit:
                break
            if group.fact_key not in seen:
                query_anchors.append(group)
                seen.add(group.fact_key)
        pools: tuple[tuple[str, list[FactGroup]], ...] = (
            (FTS_POOL, list(found.get(FTS_POOL, []))),  # type: ignore[arg-type]
            (VECTOR_POOL, list(found.get(VECTOR_POOL, []))),  # type: ignore[arg-type]
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
        candidates.extend(found.get("notes", []))  # type: ignore[arg-type]
        return candidates, versions

    async def _rerank(
        self,
        query: str,
        shortlisted: Sequence[FusedCandidate],
        deadline: float,
        ledger: _Ledger,
    ) -> tuple[float, ...]:
        """Score the head of the shortlist that fits the time left; RRF scores stand otherwise."""
        if not shortlisted:
            return ()
        remaining_ms = self._remaining(deadline) * 1000.0
        if remaining_ms <= 0:
            ledger.reasons.add("deadline")
            return ()
        count = min(len(shortlisted), max(0, self._reranker.pairs_within(remaining_ms)))
        if count == 0:
            return ()
        texts = [fused.candidate.text for fused in shortlisted[:count]]
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
            logger.warning("memory rerank failed, keeping the RRF scores: %s", exc)
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
                "keeping the RRF scores",
                len(values),
                count,
            )
            ledger.reasons.add("reranker_unavailable")
            return ()
        return values

    async def _expand(
        self,
        kept: list[ScoredCandidate],
        below: dict[tuple[str, str], ScoredCandidate],
        versions: dict[str, IndexedFact],
        suppressed: frozenset[str],
        now: datetime,
        deadline: float,
        ledger: _Ledger,
    ) -> list[ScoredCandidate | FactList]:
        """Group the kept facts of one list utterance into one item (plan v2 V7).

        Facts sharing `subject_id`, `rel_type` and `source_utterance_id` are siblings. When at
        least two clear `min_score`, every current sibling that no correction suppresses joins
        them, up to `list_max_items`: the scored members first, best score first, then the
        other siblings in the index's sibling order. A sibling ranking dropped below
        `min_score` that joins a list is placed as one of its scored members, so it gets the
        list's disposition instead of a drop that the block contradicts. Scored members past
        the cap stay single items.
        """
        groups: dict[tuple[str, str, str], list[ScoredCandidate]] = {}
        for scored in kept:
            version = (
                versions.get(scored.candidate.item_id) if scored.candidate.kind == "fact" else None
            )
            if version is None or version.source_utterance_id is None:
                continue
            sibling_key = (version.subject_id, version.rel_type, version.source_utterance_id)
            groups.setdefault(sibling_key, []).append(scored)
        listed = {key: members for key, members in groups.items() if len(members) >= 2}
        if not listed:
            return list(kept)

        def lookup() -> dict[tuple[str, str, str], tuple[IndexedFact, ...]]:
            return {key: self._index.siblings(key[0], key[1], key[2], now=now) for key in listed}

        try:
            found = await self._call(
                self._executors.index, lookup, deadline=deadline, ledger=ledger
            )
        except _LateError:
            ledger.reasons.add("deadline")
            return list(kept)
        except MemoryIndexError as exc:
            logger.warning("memory sibling lookup failed: %s", exc)
            raise _IndexFailedError from exc

        cap = self._config.list_max_items
        in_lists: set[str] = set()
        lists: list[FactList] = []
        for sibling_key, members in listed.items():
            siblings = [
                fact for fact in found.get(sibling_key, ()) if fact.fact_key not in suppressed
            ]
            order = {fact.fact_key: position for position, fact in enumerate(siblings)}
            scored_members = sorted(members, key=_score_order)
            for fact in siblings:
                extra = below.get(("fact", fact.fact_key))
                if extra is not None and fact.fact_key not in in_lists:
                    scored_members.append(extra)
            chosen = scored_members[:cap]
            chosen_keys = {member.candidate.item_id for member in chosen}
            fill = [fact for fact in siblings if fact.fact_key not in chosen_keys]
            facts_by_key = {fact.fact_key: fact for fact in siblings}
            for member in chosen:
                key = member.candidate.item_id
                if key not in facts_by_key and key in versions:
                    facts_by_key[key] = versions[key]
            room = cap - len(chosen)
            keys = [member.candidate.item_id for member in chosen] + [
                fact.fact_key for fact in fill[: max(0, room)]
            ]
            if len(keys) < 2:
                continue
            keys.sort(key=lambda key: order.get(key, len(order)))
            lists.append(
                FactList(facts=tuple(facts_by_key[key] for key in keys), scored=tuple(chosen))
            )
            in_lists.update(keys)
            for member in chosen:
                below.pop(_key(member.candidate), None)
        items: list[ScoredCandidate | FactList] = [
            scored
            for scored in kept
            if not (scored.candidate.kind == "fact" and scored.candidate.item_id in in_lists)
        ]
        items.extend(lists)
        return items

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
            outcome = await self._stage2(request, stage1, deadline, started, ledger, "none")
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

    def _remember(self, request: MemoryRequest) -> None:
        """Keep what the session's next prefetch borrows: its words become a recent turn."""
        session = self._session(request.session_id)
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
        return FetchOutcome(
            result=replace(result, timings=timings, stage=stage), lists=outcome.lists
        )

    def _empty(
        self,
        request: MemoryRequest,
        started: float,
        ledger: _Ledger,
        reason: DegradedReason,
        reuse: PrefetchReuse,
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
        self, request: MemoryRequest, reason: StageSkipReason, timings: FetchTimings
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
            prefetch_reuse="none",
            reranked=0,
            suppressed=0,
            narrative_suppressed=0,
            timings=timings,
            candidates=(),
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


def _serving(context: _Context) -> ServingProfile:
    return ServingProfile(level=context.serving_level, tier=context.serving_tier)


def _key(candidate: Candidate) -> tuple[str, str]:
    return (candidate.kind, candidate.item_id)


def _score_order(scored: ScoredCandidate) -> tuple[float, str, str]:
    return (-scored.score, scored.candidate.item_id, scored.candidate.kind)


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
    )


def _stage_candidate(
    scored: ScoredCandidate,
    rank: int,
    disposition: CandidateDisposition,
    item: int | None,
    version: IndexedFact | None,
) -> MemoryCandidate:
    """The trace's record of one candidate: refs the tools would give it, never its text."""
    candidate = scored.candidate
    source = _SOURCE_BY_KIND[candidate.kind]
    alt_ref: str | None = None
    if candidate.kind == "fact" and version is not None:
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
    )


__all__ = [
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
