"""The extraction dispatcher: one consumer draining the backlog in log order.

WHAT IT DOES
------------
One asyncio loop, one job in flight. For the head of the backlog
(`store.BacklogScan.head`, replay order) it either

- applies it, when the extraction cache already holds its decision
  (apply-pending: a result that arrived before a crash, or a gate skip), or
- dispatches it, when the cache has nothing (inference-pending): checks
  `/v1/info` against the active epoch, runs Gates 0/2/3, sends one
  `ExtractRequest`, caches the raw result DURABLY, and then applies it.

The apply step is `ExtractionPipeline.apply_cached_turn` in both cases -- the
live path and the crash-recovery path are the same code. Turn N is fully
applied (graph write, Stage 9 operations, `applied` marker) before turn N+1 is
dispatched, because Stage 9's prompt context (`existing_internal_entities`) is
read from the graph at dispatch time (`ExtractionPipeline.
build_derivation_context`).

FAILURES
--------
- The backend's WRITER stamps disagree with the active epoch: state
  `stalled`, nothing applied or dispatched. Curation writes a stamp triple
  taken from `KnowledgeConfig` (`ontology_version`, `extraction_version`,
  `compose_model_hash(config)`), not from the epoch ledger, onto each
  relationship edge reconciliation CREATES (the `ON CREATE SET` of
  `_apply_append` and `_apply_structural` in `curation/reconciliation.py`)
  and onto each EXTRACTED_FROM edge it creates or matches
  (`curation/graph_writer.py`, `_extracted_from_clause`). Reconciliation's
  `ON MATCH` updates, closes and reinforces set no stamps, and other edges
  curation writes, such as LEARNED_FROM and ABOUT, carry none. `build_extraction_dispatcher`
  derives `writer_stamps` with the same function `build_curation_pipeline`
  uses (`factories.writer_stamps_from_config`). Outside an open cutover,
  each step compares all three fields with the active epoch before it scans
  the backlog. On any difference the step returns without calling
  `/v1/info`, sending a job, writing the cache, curating or writing an apply
  marker, and the reason (both stamp triples) is logged. The epoch's first
  activation (`_ensure_activation`) runs before the guard and is not
  guarded: when the epoch has no activation row yet, it records the floor,
  marks the turns already cached under the epoch applied, and writes an
  `extraction_legacy_turns` row for each logged turn with no cache row
  under the epoch (`BacklogStore.ensure_activation`,
  `EventStore.record_extraction_activation`). It writes nothing to the
  graph. The fix is to set
  `MIST_MODEL_HASH` (or deploy the code whose versions match the epoch) and
  restart the backend. The cutover fill is not guarded: it writes candidate
  cache rows and applies nothing to the live graph.
- Service unreachable, or `model_loading`: not the job's fault. State
  `unreachable`, exponential reconnect backoff, no attempt counted.
- `/v1/info` (or a reply's stamps) disagree with the active epoch, or an
  `epoch_mismatch` error: state `epoch_mismatch`, nothing dispatched, no
  attempt counted. The dispatcher sends nothing unless
  `info.extraction_version == epoch.extraction_version` and the
  backend-composed hash of `info.model_hash` equals `epoch.model_hash`
  (composed through `compose_model_hash`, never inline).
- `contract_mismatch`, an incompatible `contract_version`, or an unparsable
  `/v1/info`: state `stalled`.
- Job-attributable failures -- retryable `upstream_llm`, `timeout`, or a reply
  that fails validation -- are counted. After `max_attempts` of them (or at
  once, for a NON-retryable `upstream_llm`/`timeout`) the turn is
  dead-lettered: cached as an `extraction_failed` skip and applied as a no-op,
  so the next turn proceeds. `python -m backend.extraction_backlog.admin
  retry-dead-letters` puts it back: it deletes the row and leaves the
  `applied` marker, and `_dispatch` clears that stale marker before it caches
  the new result (`BacklogStore.retry_dead_letter`).

EPOCH CUTOVER (T2b)
-------------------
While an `epoch_cutover` row is open (`BacklogStore.open_cutover`), each step
is a FILL step (`_fill_step`) instead: the target is the candidate's stamps,
every logged turn without a candidate cache row is gated, inferred and cached
in log order, and nothing is applied to the live graph. See `cutover.py` for
the lifecycle and `CUTOVER.md` for the runbook.

CRASH SAFETY
------------
The service result is written to the extraction cache (entities,
relationships, scope, Stage 9 operations) before anything touches the graph,
so a crash after that point re-applies from the cache without calling the
service again. Apply progress is two markers (`curated`, then `applied`; see
`ExtractionPipeline.ApplyProgress`). What a crash can still repeat is the
curation of ONE turn, when it lands inside `curate_and_store` or between its
return and the `curated` marker. Re-running curation for an already-curated
turn converges on the canonical surface -- relationship appends MERGE on
`version_key` and the reconciliation engine's `turn_already_applied` probe
turns a replayed edge into a no-op (`curation/reconciliation.py`,
`_fetch_existing`), entity writes MERGE on id with longest-wins text fields
(`curation/graph_writer.py`, `_upsert_entity`). The `ON MATCH` confidence
reinforcement is skipped when this event's EXTRACTED_FROM edge to the entity
already exists (plan v2, `graph_writer.py` `_upsert_entity`), so a replayed
turn does not reinforce twice. The window where a kill between an entity's
upsert and its EXTRACTED_FROM write still reinforced on replay is closed: one
Cypher statement per entity writes the entity, its guarded reinforce, the
EXTRACTED_FROM edge and, for a new entity, its `new_fact` LearningEvent, so a
kill leaves all of them or none. `TestKillAtAnyStatementThenReplay` in
`tests/unit/knowledge/curation/test_graph_writer_replay_guard.py` kills before
every statement of a turn and checks the replay against a single apply.
Node `confidence` is excluded from `canonical_graph_form`
(`canonical_serialize.NODE_ONLY_EXCLUDED_FIELDS`), so a canonical comparison
cannot see a double reinforce.

`stop()` is graceful: a job still waiting on the service is abandoned (nothing
was cached, so the turn stays inference-pending and is re-dispatched with a new
`job_id`); an apply in progress is allowed to finish.

OBSERVABILITY
-------------
Every job logs one INFO line with `request_id`, `job_id`, `turn_id`,
`event_id`, `attempt`, `duration_ms`, `error_code` and `outcome`, and every
service call is recorded in the `extraction_attempts` table. `snapshot()`
returns the contract's `ExtractionStatus`, with its `cutover` block filled
while a cutover is open. `add_state_listener` callbacks run on every state
transition and every cutover transition the dispatcher makes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.errors import MistError
from backend.extraction_contract.models import (
    CONTRACT_VERSION,
    CutoverStatus,
    DerivationInput,
    ErrorCode,
    ExpectStamps,
    ExtractionStatus,
    ExtractRequest,
    ExtractResponse,
    HistoryMessage,
    InfoResponse,
    LastJob,
    ServiceStatus,
    is_compatible,
)
from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.extraction.pipeline import (
    ApplyReport,
    ExtractionPipeline,
    TurnToApply,
)
from backend.knowledge.extraction_cache import SKIP_EXTRACTION_FAILED

from . import telemetry
from .cutover import log_transition
from .errors import (
    ExtractionInferenceError,
    InferenceResponseInvalidError,
    InferenceServiceError,
    InferenceUnreachableError,
)
from .inference import ExtractionInference
from .settings import DispatcherSettings
from .store import (
    BacklogStore,
    Cutover,
    Epoch,
    PendingTurn,
    age_ms,
    compose_epoch_model_hash,
)

logger = logging.getLogger(__name__)

ApplyListener = Callable[[ApplyReport], Awaitable[None]]
# Called synchronously, on the event loop, as `listener(previous, current)`
# after every state transition (and after every cutover transition, with
# previous == current). Must not block and must not raise.
StateListener = Callable[[str, str], None]

STATE_IDLE = "idle"
STATE_WORKING = "working"
STATE_STALLED = "stalled"
STATE_UNREACHABLE = "unreachable"
STATE_EPOCH_MISMATCH = "epoch_mismatch"
STATE_DISABLED = "disabled"

# Failures the dispatcher's own loop contains rather than dying on: storage
# (SQLite) and the MIST error tree (Neo4j, curation, inference). Anything else
# is a bug, propagates, and ends the loop with state `stalled` (see
# `_on_loop_done`).
_CONTAINED_ERRORS = (MistError, sqlite3.Error, OSError)


@dataclass(frozen=True, slots=True)
class _Wait:
    """How long the loop sleeps after a step, and whether `wake()` cuts it short."""

    seconds: float
    interruptible: bool


def _now_ms(now: datetime) -> int:
    return int(now.timestamp() * 1000)


class ExtractionDispatcher:
    """Single-writer consumer of the extraction backlog."""

    def __init__(
        self,
        *,
        store: BacklogStore,
        pipeline: ExtractionPipeline,
        inference: ExtractionInference | None,
        settings: DispatcherSettings,
        embedding_model_name: str,
        writer_stamps: RebuildStamps,
        clock: Callable[[], datetime] | None = None,
        on_stop: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize the dispatcher. Nothing runs until `start()`.

        Args:
            store: Backlog view over the event store and the extraction cache.
            pipeline: The backend's extraction pipeline, used for Gates 0/2/3,
                the Stage 9 context, and `apply_cached_turn`. Its Stage 1.5/2
                LLM stages are never called from here.
            inference: Where Stages 1.5/2/9 run. None only in `off` mode.
            settings: Mode, retry and backoff configuration.
            embedding_model_name: The backend's embedding model identity, folded
                into the service's bare model hash for the epoch comparison.
            writer_stamps: The stamp triple curation sets on the edges
                reconciliation creates and on EXTRACTED_FROM edges (see
                FAILURES in the module docstring). `build_extraction_dispatcher` derives it from
                `KnowledgeConfig` with `writer_stamps_from_config`, the
                function `build_curation_pipeline` uses. Outside a cutover
                the dispatcher stalls rather than apply or dispatch while it
                differs from the active epoch's stamps.
            clock: Wall clock (tz-aware). Defaults to `datetime.now(UTC)`.
            on_stop: Awaited once at the end of `stop()`, e.g. to close an
                HTTP client the factory built for this dispatcher.

        Raises:
            ValueError: `inference` is None while `settings.mode` is `service`.
        """
        if settings.mode == "service" and inference is None:
            raise ValueError("an ExtractionInference is required in 'service' mode")
        self._store = store
        self._pipeline = pipeline
        self._inference = inference
        self._settings = settings
        self._embedding_model_name = embedding_model_name
        self._writer_stamps = writer_stamps
        self._writer_mismatch_logged_epoch: int | None = None
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._on_stop = on_stop

        self._state = STATE_DISABLED if settings.mode == "off" else STATE_IDLE
        self._task: asyncio.Task | None = None
        self._stopping = False
        self._wake_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._changed = asyncio.Event()
        self._phase = "idle"  # 'idle' | 'inference' | 'apply'
        self._listeners: list[ApplyListener] = []
        self._state_listeners: list[StateListener] = []
        self._consecutive_unreachable = 0
        self._consecutive_step_errors = 0
        self._activated_epoch_id: int | None = None
        self._legacy_unextracted = 0
        self._last_job: LastJob | None = None
        self._service = ServiceStatus(
            reachable=False,
            location_label=None,
            model_id=None,
            extraction_version=None,
            contract_version=None,
            last_health_ms=None,
        )

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """`service` or `off`."""
        return self._settings.mode

    @property
    def state(self) -> str:
        """Current `ExtractionStatus.state` value."""
        return self._state

    @property
    def running(self) -> bool:
        """True while the loop task is alive."""
        return self._task is not None and not self._task.done()

    @property
    def legacy_unextracted(self) -> int:
        """Turns logged before first activation with no cache row (never dispatched)."""
        return self._legacy_unextracted

    def add_apply_listener(self, listener: ApplyListener) -> None:
        """Register a coroutine called after every applied turn.

        The listener must not raise; a `MistError` it raises is logged and
        swallowed so one listener cannot stop the backlog.
        """
        self._listeners.append(listener)

    def add_state_listener(self, listener: StateListener) -> None:
        """Register a callback run on every state (and cutover) transition.

        Called synchronously from the loop as `listener(previous, current)`;
        the server uses it to push `extraction_status` on a transition rather
        than only on its timer. A `MistError`, `sqlite3.Error`, `OSError` or
        `ValueError` it raises is logged and swallowed.
        """
        self._state_listeners.append(listener)

    async def start(self) -> None:
        """Activate the backlog for the current epoch and, in `service` mode, run it.

        In `off` mode this records the first-activation floor (so turns logged
        while extraction is off are NOT classified legacy when it is turned on
        later) and returns with state `disabled`; no loop runs.
        """
        if self._task is not None:
            return
        self._stopping = False
        self._stop_event.clear()
        epoch = self._store.active_epoch()
        if epoch is not None:
            self._ensure_activation(epoch)
        if self._settings.mode == "off":
            self._set_state(STATE_DISABLED)
            logger.info("Extraction dispatcher disabled (MIST_EXTRACTION_INFERENCE=off)")
            return
        self._task = asyncio.create_task(self._run(), name="extraction-dispatcher")
        self._task.add_done_callback(self._on_loop_done)
        logger.info("Extraction dispatcher started (service=%s)", self._settings.service_url)

    async def stop(self, timeout: float = 30.0) -> None:
        """Stop the loop.

        A job waiting on the service is abandoned by cancellation -- nothing
        was cached for it, so it is re-dispatched later with a new `job_id`.
        An apply in progress is allowed to finish (it is not cancelled unless
        it outlives `timeout`).
        """
        self._stopping = True
        self._stop_event.set()
        self._wake_event.set()
        task = self._task
        if task is None:
            await self._run_on_stop()
            return
        if self._phase == "inference":
            task.cancel()
        done, _pending = await asyncio.wait({task}, timeout=timeout)
        if not done:
            logger.warning(
                "Extraction dispatcher did not stop within %.0fs; cancelling mid-%s",
                timeout,
                self._phase,
            )
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._task = None
        await self._run_on_stop()
        logger.info("Extraction dispatcher stopped")

    async def _run_on_stop(self) -> None:
        if self._on_stop is None:
            return
        on_stop, self._on_stop = self._on_stop, None
        await on_stop()

    def wake(self) -> None:
        """Tell the loop a turn was logged. Cheap; safe to call on every turn."""
        self._wake_event.set()

    async def drain(self, timeout: float) -> bool:
        """Wait until the backlog is empty or the dispatcher cannot make progress.

        Returns True when nothing is pending (inference- or apply-pending).
        Returns False at `timeout`, or at once when the loop is not running or
        its state is not `working`/`idle` (unreachable, stalled,
        epoch_mismatch, disabled) -- waiting would not help. Also at once while
        a cutover is open: the dispatcher is filling the candidate and applies
        nothing for the active epoch until promotion, so the active backlog
        cannot drain.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._changed.clear()
            empty = self._backlog_empty()
            if empty:
                return True
            if self._store.open_cutover() is not None:
                return False
            if not self.running or self._state not in (STATE_WORKING, STATE_IDLE):
                return False
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._changed.wait(), timeout=remaining)
            except TimeoutError:
                return False

    def snapshot(self) -> ExtractionStatus:
        """The contract's `ExtractionStatus` for this dispatcher, computed now."""
        epoch = self._store.active_epoch()
        backlog_depth = apply_pending = dead_lettered = legacy_unextracted = 0
        oldest: int | None = None
        if epoch is not None:
            scan = self._store.scan(epoch)
            backlog_depth = scan.backlog_depth
            apply_pending = scan.apply_pending
            dead_lettered = scan.dead_lettered
            legacy_unextracted = scan.legacy_unextracted
            oldest = age_ms(scan.oldest_pending_timestamp, self._clock())
        cutover_status = None
        cutover = self._store.open_cutover()
        if cutover is not None:
            fill = self._store.fill_scan(cutover)
            cutover_status = CutoverStatus(
                state=cutover.state,  # type: ignore[arg-type]
                target_extraction_version=cutover.extraction_version,
                target_model_hash=cutover.model_hash,
                covered=fill.covered,
                total=fill.total,
            )
        return ExtractionStatus(
            state=self._state,  # type: ignore[arg-type]
            backlog_depth=backlog_depth,
            apply_pending=apply_pending,
            dead_lettered=dead_lettered,
            oldest_pending_age_ms=oldest,
            unrecorded_turns=telemetry.unrecorded_turns(),
            legacy_unextracted=legacy_unextracted,
            service=self._service,
            last_job=self._last_job,
            cutover=cutover_status,
        )

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _on_loop_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._set_state(STATE_STALLED)
            logger.error("Extraction dispatcher loop died: %s", exc, exc_info=exc)

    async def _run(self) -> None:
        while not self._stopping:
            self._wake_event.clear()
            try:
                wait = await self._step()
                self._consecutive_step_errors = 0
            except _CONTAINED_ERRORS as exc:
                # A storage or graph failure outside the service taxonomy (e.g.
                # Neo4j down during apply). Nothing is marked, so the same turn
                # is retried after the backoff; the backlog order is unchanged.
                self._phase = "idle"
                self._consecutive_step_errors += 1
                logger.error(
                    "Extraction dispatcher step failed (retry %d): %s",
                    self._consecutive_step_errors,
                    exc,
                    exc_info=True,
                )
                wait = _Wait(self._settings.backoff_s(self._consecutive_step_errors), False)
            self._notify()
            if wait is not None and not self._stopping:
                await self._sleep(wait)

    async def _sleep(self, wait: _Wait) -> None:
        self._phase = "idle"
        event = self._wake_event if wait.interruptible else self._stop_event
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout=wait.seconds)

    async def _step(self) -> _Wait | None:
        """Process at most one turn. Returns how long to wait, or None to continue."""
        epoch = self._store.active_epoch()
        if epoch is None:
            self._set_state(STATE_STALLED, reason="the epoch ledger is empty")
            return _Wait(self._settings.stall_recheck_s, True)
        self._ensure_activation(epoch)

        cutover = self._store.open_cutover()
        if cutover is not None:
            return await self._fill_step(cutover)

        # Writer-stamp guard: everything below can apply to the live graph.
        # Curation sets its own config-derived triple on the edges listed
        # under FAILURES in the module docstring, and `self._writer_stamps`
        # stands for that triple, so the steps below run only when it equals
        # the active epoch's stamps.
        mismatch = self._writer_stamp_mismatch(epoch)
        if mismatch is not None:
            if self._writer_mismatch_logged_epoch != epoch.epoch_id:
                self._writer_mismatch_logged_epoch = epoch.epoch_id
                logger.error("Extraction dispatcher refuses to apply: %s", mismatch)
            self._set_state(STATE_STALLED, reason=mismatch)
            return _Wait(self._settings.stall_recheck_s, True)

        scan = self._store.scan(epoch)
        head = scan.head
        if head is None:
            self._set_state(STATE_IDLE)
            return _Wait(self._settings.idle_poll_s, True)

        if head.cached:
            self._set_state(STATE_WORKING)
            await self._apply(head, epoch, request_id="", attempt=0, started=time.perf_counter())
            return None

        self._phase = "inference"
        info_or_wait = await self._check_service(epoch)
        if isinstance(info_or_wait, _Wait):
            return info_or_wait
        self._set_state(STATE_WORKING)
        return await self._dispatch(head, epoch, info_or_wait)

    async def _fill_step(self, cutover: Cutover) -> _Wait | None:
        """One step of an open cutover: infer the next uncovered turn under the candidate.

        The inference target is the CANDIDATE's stamps (`cutover.target`): the
        dispatcher sends a job only when `/v1/info` matches them, runs Gates
        0/2/3 and inference for every logged turn in log order from the
        beginning, and caches the result under the candidate's key. It applies
        NOTHING -- no curation, no Stage 9 operations, no `applied` marker --
        so the live graph stays exactly as it was. The active epoch's
        apply-pending turns therefore stay pending for the life of the
        cutover; the service is serving the candidate model and could not
        extract for the active epoch anyway.

        When every logged turn has a candidate row the cutover moves
        'filling' -> 'ready'. It keeps filling after that: a turn logged later
        is uncovered again, and is inferred in its place in the log.
        """
        candidate = cutover.target
        fill = self._store.fill_scan(cutover)
        head = fill.head
        if head is None:
            if cutover.state == "filling" and self._store.transition_cutover(
                cutover,
                from_states=("filling",),
                to_state="ready",
                updated_at=self._clock().isoformat(),
            ):
                log_transition(cutover, "filling", "ready", covered=fill.covered, total=fill.total)
                self._notify()
                self._emit_state(self._state, self._state)
            self._set_state(STATE_IDLE)
            return _Wait(self._settings.idle_poll_s, True)

        self._phase = "inference"
        info_or_wait = await self._check_service(candidate)
        if isinstance(info_or_wait, _Wait):
            return info_or_wait
        self._set_state(STATE_WORKING)
        return await self._dispatch(head, candidate, info_or_wait, apply=False)

    # ------------------------------------------------------------------
    # Service / epoch checks
    # ------------------------------------------------------------------

    def _writer_stamp_mismatch(self, epoch: Epoch) -> str | None:
        """None when the writer stamps equal `epoch`'s on all three fields, else why not."""
        writer = self._writer_stamps
        if (
            writer.ontology_version == epoch.ontology_version
            and writer.extraction_version == epoch.extraction_version
            and writer.model_hash == epoch.model_hash
        ):
            return None
        return (
            f"backend writer stamps (ontology_version={writer.ontology_version!r}, "
            f"extraction_version={writer.extraction_version!r}, "
            f"model_hash={writer.model_hash!r}) != {epoch.label} "
            f"(ontology_version={epoch.ontology_version!r}, "
            f"extraction_version={epoch.extraction_version!r}, "
            f"model_hash={epoch.model_hash!r}); nothing is applied or dispatched until "
            "they match. Set MIST_MODEL_HASH (or deploy the code whose ontology and "
            "extraction versions match the epoch), then restart the backend"
        )

    def _composed(self, bare_model_hash: str) -> str:
        return compose_epoch_model_hash(bare_model_hash, self._embedding_model_name)

    def _epoch_matches(self, extraction_version: str, bare_model_hash: str, epoch: Epoch) -> bool:
        return (
            extraction_version == epoch.extraction_version
            and self._composed(bare_model_hash) == epoch.model_hash
        )

    async def _check_service(self, epoch: Epoch) -> InfoResponse | _Wait:
        assert self._inference is not None  # service mode, enforced in __init__
        try:
            info = await self._inference.info()
        except InferenceUnreachableError as exc:
            return self._unreachable(str(exc))
        except (InferenceServiceError, InferenceResponseInvalidError) as exc:
            self._service = self._service.model_copy(update={"reachable": True})
            self._set_state(STATE_STALLED, reason=f"/v1/info unusable: {exc}")
            return _Wait(self._settings.stall_recheck_s, True)

        self._consecutive_unreachable = 0
        self._service = ServiceStatus(
            reachable=True,
            location_label=info.location_label,
            model_id=info.model_hash,
            extraction_version=info.extraction_version,
            contract_version=info.contract_version,
            last_health_ms=_now_ms(self._clock()),
        )
        if not is_compatible(info.contract_version):
            self._set_state(
                STATE_STALLED,
                reason=(
                    f"service contract_version {info.contract_version!r} is incompatible "
                    f"with {CONTRACT_VERSION!r}"
                ),
            )
            return _Wait(self._settings.stall_recheck_s, True)
        if not self._epoch_matches(info.extraction_version, info.model_hash, epoch):
            self._set_state(
                STATE_EPOCH_MISMATCH,
                reason=(
                    f"service (extraction_version={info.extraction_version!r}, "
                    f"model_hash={self._composed(info.model_hash)!r}) != "
                    f"{epoch.label} (extraction_version={epoch.extraction_version!r}, "
                    f"model_hash={epoch.model_hash!r})"
                ),
            )
            return _Wait(self._settings.stall_recheck_s, True)
        return info

    def _unreachable(self, reason: str) -> _Wait:
        self._consecutive_unreachable += 1
        self._service = self._service.model_copy(update={"reachable": False})
        self._set_state(STATE_UNREACHABLE, reason=reason)
        return _Wait(self._settings.backoff_s(self._consecutive_unreachable), True)

    # ------------------------------------------------------------------
    # One job
    # ------------------------------------------------------------------

    async def _dispatch(
        self, head: PendingTurn, epoch: Epoch, info: InfoResponse, *, apply: bool = True
    ) -> _Wait | None:
        """Gate, infer and cache one turn under `epoch`; apply it when `apply`.

        `apply=False` is the cutover fill: the result is cached under the
        candidate's stamps and nothing touches the graph. `last_job` is not
        updated for such a job -- `LastJob.outcome` has no value for "cached,
        not applied" -- and its log line says `outcome=extracted` (or
        `skipped`).
        """
        assert self._inference is not None
        started = time.perf_counter()
        turn = self._store.get_turn(head.event_id)
        if turn is None:  # pragma: no cover -- the scan just listed it
            raise MistError(f"turn {head.event_id} vanished from the event log")
        utterance = str(turn["user_utterance"])

        if apply and self._store.clear_stale_marker(head, epoch):
            # The head has no cache row, so any marker it has is stale (a
            # retried dead letter, or a lost cache row). It must go before the
            # new row is cached, or a crash in between reads as done.
            logger.info(
                "Cleared a stale apply marker for %s (event %s, no cache row under %s); "
                "re-extracting it",
                head.turn_id,
                head.event_id,
                epoch.label,
            )

        decision = self._pipeline.evaluate_dispatch_gates(utterance)
        if decision.skip_reason is not None:
            self._store.put_skip(
                head.event_id, epoch, skip_reason=decision.skip_reason, created_at=head.timestamp
            )
            if apply:
                await self._apply(head, epoch, request_id="", attempt=0, started=started)
            else:
                self._log_job(
                    head,
                    request_id="-",
                    job_id="-",
                    attempt=0,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    error_code=decision.skip_reason,
                    outcome="skipped",
                )
            return None

        derivation_ctx = await self._pipeline.build_derivation_context(utterance)
        derivation = None
        if derivation_ctx is not None:
            # `assistant_response` is the turn's LOGGED reply. The in-process
            # path sends "" here (`pipeline._run_internal_derivation`, the TODO
            # beside `assistant_response=""`); sending the real reply is a
            # deliberate improvement, since the derivation prompt asks about
            # MIST's behaviour and the reply is that behaviour.
            derivation = DerivationInput(
                signal_types=list(derivation_ctx.signal_types),
                matched_patterns=list(derivation_ctx.matched_patterns),
                existing_internal_entities=derivation_ctx.existing_internal_entities,
                assistant_response=str(turn["system_response"]),
            )

        history = self._store.session_history(
            head.session_id,
            through_turn_index=head.turn_index,
            limit=self._settings.history_messages,
        )
        job_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        request = ExtractRequest(
            contract_version=CONTRACT_VERSION,
            job_id=job_id,
            event_id=head.event_id,
            turn_id=head.turn_id,
            request_id=request_id,
            session_id=head.session_id,
            recorded_at=head.timestamp,
            turn_index=head.turn_index,
            utterance=utterance,
            conversation_history=[HistoryMessage(**m) for m in history],
            # The service compares `expect.model_hash` with its BARE model hash
            # (`extraction_service/app.py`, the EPOCH_MISMATCH branch), so the
            # bare value goes on the wire; the composed comparison against the
            # epoch already passed in `_check_service`.
            expect=ExpectStamps(
                extraction_version=epoch.extraction_version, model_hash=info.model_hash
            ),
            derivation=derivation,
        )

        attempt = self._store.next_attempt_number(head.event_id, epoch)
        started_at = self._clock().isoformat()
        call_start = time.perf_counter()
        try:
            response = await self._inference.extract(request)
            self._validate_response(request, response, epoch)
        except ExtractionInferenceError as exc:
            duration_ms = (time.perf_counter() - call_start) * 1000
            return self._on_job_error(
                exc,
                head=head,
                epoch=epoch,
                attempt=attempt,
                job_id=job_id,
                request_id=request_id,
                started_at=started_at,
                duration_ms=duration_ms,
            )
        duration_ms = (time.perf_counter() - call_start) * 1000

        # Durable BEFORE apply: from here on a crash re-applies from this row.
        self._store.put_extracted(
            head.event_id,
            epoch,
            created_at=head.timestamp,
            entities=response.payload.entities,
            relationships=response.payload.relationships,
            scope=response.scope.label,
            scope_confidence=response.scope.confidence,
            derivation=(
                None
                if response.derivation is None
                else {"operations": response.derivation.operations}
            ),
            service_stamps={
                "stamps": response.stamps.model_dump(mode="json"),
                "timings_ms": response.timings_ms.model_dump(mode="json"),
                "attempts": response.attempts,
                "warnings": list(response.warnings),
                "contract_version": response.contract_version,
                "job_id": job_id,
                "request_id": request_id,
                "location_label": info.location_label,
            },
        )
        self._store.record_attempt(
            turn=head,
            epoch=epoch,
            attempt=attempt,
            job_id=job_id,
            request_id=request_id,
            started_at=started_at,
            finished_at=self._clock().isoformat(),
            duration_ms=duration_ms,
            error_code=None,
            outcome="extracted",
            counted=False,
        )
        self._pipeline.note_extraction_completed(utterance, decision.embedding)
        if not apply:
            self._log_job(
                head,
                request_id=request_id,
                job_id=job_id,
                attempt=attempt,
                duration_ms=duration_ms,
                error_code=None,
                outcome="extracted",
            )
            return None
        await self._apply(
            head,
            epoch,
            request_id=request_id,
            attempt=attempt,
            started=started,
            job_id=job_id,
        )
        return None

    def _validate_response(
        self, request: ExtractRequest, response: ExtractResponse, epoch: Epoch
    ) -> None:
        """Checks the contract model cannot make on its own.

        Raises:
            InferenceResponseInvalidError: The reply is for another job, or its
                contract version is incompatible. Job-attributable.
            InferenceServiceError: The reply's stamps disagree with the epoch
                (the service changed between `/v1/info` and the reply). Coded
                `epoch_mismatch`, so it stalls rather than counts.
        """
        if response.job_id != request.job_id:
            raise InferenceResponseInvalidError(
                f"reply is for job {response.job_id!r}, expected {request.job_id!r}"
            )
        if not is_compatible(response.contract_version):
            raise InferenceResponseInvalidError(
                f"reply contract_version {response.contract_version!r} is incompatible"
            )
        stamps = response.stamps
        if not self._epoch_matches(stamps.extraction_version, stamps.model_hash, epoch):
            raise InferenceServiceError(
                f"reply stamps (extraction_version={stamps.extraction_version!r}, "
                f"model_hash={stamps.model_hash!r}) do not match {epoch.label}",
                code=ErrorCode.EPOCH_MISMATCH,
                retryable=False,
                http_status=200,
            )

    def _on_job_error(
        self,
        exc: ExtractionInferenceError,
        *,
        head: PendingTurn,
        epoch: Epoch,
        attempt: int,
        job_id: str,
        request_id: str,
        started_at: str,
        duration_ms: float,
    ) -> _Wait | None:
        """Classify a failed extract call, record it, and decide what happens next."""

        def record(outcome: str, error_code: str, counted: bool) -> None:
            self._store.record_attempt(
                turn=head,
                epoch=epoch,
                attempt=attempt,
                job_id=job_id,
                request_id=request_id,
                started_at=started_at,
                finished_at=self._clock().isoformat(),
                duration_ms=duration_ms,
                error_code=error_code,
                outcome=outcome,
                counted=counted,
            )
            self._log_job(
                head,
                request_id=request_id,
                job_id=job_id,
                attempt=attempt,
                duration_ms=duration_ms,
                error_code=error_code,
                outcome=outcome,
            )

        if isinstance(exc, InferenceUnreachableError) or exc.code is ErrorCode.MODEL_LOADING:
            code = "unreachable" if exc.code is None else exc.code.value
            record("deferred", code, counted=False)
            return self._unreachable(str(exc))
        if exc.code is ErrorCode.EPOCH_MISMATCH:
            record("deferred", exc.code.value, counted=False)
            self._set_state(STATE_EPOCH_MISMATCH, reason=str(exc))
            return _Wait(self._settings.stall_recheck_s, True)
        if exc.code is ErrorCode.CONTRACT_MISMATCH:
            record("deferred", exc.code.value, counted=False)
            self._set_state(STATE_STALLED, reason=str(exc))
            return _Wait(self._settings.stall_recheck_s, True)

        # Job-attributable: upstream_llm, timeout, or an invalid reply.
        code = exc.code.value if exc.code is not None else "invalid_response"
        failures = self._store.counted_failures(head.event_id, epoch) + 1
        dead = failures >= self._settings.max_attempts or not exc.retryable
        if not dead:
            record("failed", code, counted=True)
            logger.warning(
                "Extraction job failed for %s (%d/%d): %s",
                head.turn_id,
                failures,
                self._settings.max_attempts,
                exc,
            )
            return _Wait(self._settings.backoff_s(failures), False)

        record("dead_lettered", code, counted=True)
        logger.warning(
            "Extraction dead-lettered %s (event %s) after %d job failure(s): %s",
            head.turn_id,
            head.event_id,
            failures,
            exc,
        )
        self._store.put_skip(
            head.event_id, epoch, skip_reason=SKIP_EXTRACTION_FAILED, created_at=head.timestamp
        )
        # Returning None loops straight back: the scan now sees the turn as
        # apply-pending, and `_step` applies the skip as a no-op plus marker.
        return None

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    async def _apply(
        self,
        head: PendingTurn,
        epoch: Epoch,
        *,
        request_id: str,
        attempt: int,
        started: float,
        job_id: str = "",
    ) -> ApplyReport | None:
        """Apply the turn's cache row under `epoch`; None when the row is gone.

        The row can vanish between the scan and this read: `retry-dead-letters`
        deletes an `extraction_failed` row from another process. Nothing is
        marked then, and the next scan finds the turn inference-pending.
        """
        turn = self._store.get_turn(head.event_id)
        if turn is None:  # pragma: no cover -- the scan just listed it
            raise MistError(f"turn {head.event_id} vanished from the event log")
        cached = self._store.get_cached(head.event_id, epoch)
        if cached is None:
            logger.info(
                "Cache row for %s (event %s) was removed before its apply (a dead-letter "
                "retry); rescanning",
                head.turn_id,
                head.event_id,
            )
            return None
        if not request_id:
            stamps = cached.get("service_stamps") or {}
            request_id = str(stamps.get("request_id") or "")
            job_id = job_id or str(stamps.get("job_id") or "")

        progress = self._store.progress(head, epoch, now_iso=lambda: self._clock().isoformat())
        self._phase = "apply"
        report = await self._pipeline.apply_cached_turn(
            TurnToApply(
                event_id=head.event_id,
                session_id=head.session_id,
                user_utterance=str(turn["user_utterance"]),
                recorded_at=head.timestamp,
            ),
            cached,
            progress,
        )
        self._phase = "idle"

        if report.skipped:
            outcome = (
                "dead_lettered"
                if cached.get("skip_reason") == SKIP_EXTRACTION_FAILED
                else "skipped"
            )
        elif report.stage_errors:
            outcome = "failed"
            logger.warning(
                "Applied %s with curation stage errors: %s", head.turn_id, report.stage_errors
            )
        else:
            outcome = "applied"
        duration_ms = (time.perf_counter() - started) * 1000
        finished = self._clock()
        self._last_job = LastJob(
            event_id=head.event_id,
            turn_id=head.turn_id,
            request_id=request_id,
            duration_ms=duration_ms,
            outcome=outcome,  # type: ignore[arg-type]
            finished_ms=_now_ms(finished),
        )
        self._log_job(
            head,
            request_id=request_id or "-",
            job_id=job_id or "-",
            attempt=attempt,
            duration_ms=duration_ms,
            error_code=cached.get("skip_reason"),
            outcome=outcome,
        )
        for listener in self._listeners:
            try:
                await listener(report)
            except MistError as exc:
                logger.warning("Extraction apply listener failed (ignored): %s", exc)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_activation(self, epoch: Epoch) -> None:
        if self._activated_epoch_id == epoch.epoch_id:
            return
        activation = self._store.ensure_activation(epoch, now_iso=self._clock().isoformat())
        self._activated_epoch_id = epoch.epoch_id
        self._legacy_unextracted = activation.legacy_unextracted
        if activation.created_now:
            logger.info(
                "Extraction backlog activated for epoch %d: %d logged turn(s), %d already "
                "applied by the in-process path, %d legacy-unextracted (not dispatched; the "
                "epoch cutover re-extracts them)",
                epoch.epoch_id,
                activation.turns_at_activation,
                activation.marked_applied,
                activation.legacy_unextracted,
            )
        elif activation.legacy_unextracted:
            logger.info(
                "Extraction backlog epoch %d: %d legacy-unextracted turn(s) excluded "
                "(activation floor recorded %s)",
                epoch.epoch_id,
                activation.legacy_unextracted,
                activation.activated_at,
            )

    def _backlog_empty(self) -> bool:
        epoch = self._store.active_epoch()
        if epoch is None:
            return True
        scan = self._store.scan(epoch)
        return scan.head is None

    def _set_state(self, state: str, *, reason: str | None = None) -> None:
        if state == self._state:
            return
        previous, self._state = self._state, state
        if reason:
            logger.info("Extraction dispatcher %s -> %s: %s", previous, state, reason)
        else:
            logger.info("Extraction dispatcher %s -> %s", previous, state)
        self._notify()
        self._emit_state(previous, state)

    def _emit_state(self, previous: str, current: str) -> None:
        for listener in self._state_listeners:
            try:
                listener(previous, current)
            except (MistError, sqlite3.Error, OSError, ValueError) as exc:
                logger.warning("Extraction state listener failed (ignored): %s", exc)

    def _notify(self) -> None:
        self._changed.set()

    @staticmethod
    def _log_job(
        head: PendingTurn,
        *,
        request_id: str,
        job_id: str,
        attempt: int,
        duration_ms: float,
        error_code: str | None,
        outcome: str,
    ) -> None:
        logger.info(
            "extraction job request_id=%s job_id=%s turn_id=%s event_id=%s attempt=%d "
            "duration_ms=%.1f error_code=%s outcome=%s",
            request_id,
            job_id,
            head.turn_id,
            head.event_id,
            attempt,
            duration_ms,
            error_code or "-",
            outcome,
        )
