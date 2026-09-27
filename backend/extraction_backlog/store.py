"""The extraction backlog, as a view over the event log and the extraction cache.

There is no queue table. For the ACTIVE epoch (the latest `epoch_ledger` row):

- a turn is INFERENCE-PENDING when the extraction cache has no row for
  `cache_key(event_id, epoch.extraction_version, epoch.model_hash)`, marker or not;
- a turn is APPLY-PENDING when it has a cache row but no `applied` marker in
  `extraction_applied`;
- a turn is DONE when it has a cache row AND its marker says `applied`;
- a turn is LEGACY when it was logged before the backlog first activated for
  this epoch and had no cache row then. Legacy turns are never dispatched:
  applying them after later turns would violate log order. The epoch cutover
  (T2b) re-extracts the whole log, which covers them.

Order is replay order, `ORDER BY timestamp, session_id, turn_index`
(`EventStore.list_turn_keys_in_replay_order`), the order a rebuild replays in.
Every logged turn is eligible whatever its origin -- the pre-T2a live path
extracted every recorded turn, and so does the backlog.

The event store and the cache are separate SQLite files
(`backend.factories.production_cache_path` puts the cache beside the event
store), so the join happens here in Python: one ordered key scan of the log and
one stamp-filtered id scan of the cache per call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backend.event_store.store import EventStore
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_EXTRACTION_FAILED,
    ExtractionCache,
)
from backend.knowledge.version_stamps import compose_model_hash

STAGE_CURATED = "curated"
STAGE_APPLIED = "applied"


@dataclass(frozen=True, slots=True)
class _EmbeddingIdentity:
    model_name: str


@dataclass(frozen=True, slots=True)
class _StampIdentity:
    """The two attributes `compose_model_hash` reads, for a service-reported hash.

    `compose_model_hash` takes a `KnowledgeConfig`-shaped object; this shim lets
    the backlog compose the SERVICE's bare model hash with the BACKEND's
    embedding model exactly as the epoch row was composed, without re-building
    the string inline (the function's docstring explains why callers must not).
    """

    model_hash: str
    embedding: _EmbeddingIdentity


def compose_epoch_model_hash(bare_model_hash: str, embedding_model_name: str) -> str:
    """The epoch-side (composed) model hash for a bare service model hash."""
    return compose_model_hash(
        _StampIdentity(
            model_hash=bare_model_hash,
            embedding=_EmbeddingIdentity(model_name=embedding_model_name),
        )
    )


@dataclass(frozen=True, slots=True)
class Epoch:
    """The stamp triple the backlog reads and writes the cache under.

    Either a ledger epoch, or a cutover CANDIDATE (`cutover_id` set). A
    candidate's `epoch_id` is `-cutover_id`: the id its service attempts are
    recorded under in `extraction_attempts`, a namespace no ledger row can
    occupy because ledger ids are AUTOINCREMENT and start at 1.
    """

    epoch_id: int
    ontology_version: str
    extraction_version: str
    model_hash: str
    cutover_id: int | None = None

    @property
    def label(self) -> str:
        """`epoch N` or `cutover N candidate`, for log lines."""
        if self.cutover_id is not None:
            return f"cutover {self.cutover_id} candidate"
        return f"epoch {self.epoch_id}"

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Epoch:
        """Build from an `epoch_ledger` row dict."""
        return cls(
            epoch_id=int(row["epoch_id"]),
            ontology_version=str(row["ontology_version"]),
            extraction_version=str(row["extraction_version"]),
            model_hash=str(row["model_hash"]),
        )


@dataclass(frozen=True, slots=True)
class Cutover:
    """One `epoch_cutover` row: a candidate epoch and where it is in its lifecycle."""

    cutover_id: int
    ontology_version: str
    extraction_version: str
    model_hash: str
    bare_model_hash: str
    requested_at: str
    state: str
    source_epoch_id: int | None
    rebuild_job_id: str | None
    rebuilt_through_event_id: str | None
    check_report: dict[str, Any] | None
    promoted_epoch_id: int | None
    updated_at: str

    @property
    def target(self) -> Epoch:
        """The candidate's stamps as an `Epoch` (epoch_id = -cutover_id)."""
        return Epoch(
            epoch_id=-self.cutover_id,
            ontology_version=self.ontology_version,
            extraction_version=self.extraction_version,
            model_hash=self.model_hash,
            cutover_id=self.cutover_id,
        )

    def epoch_dict(self) -> dict[str, Any]:
        """The candidate as the epoch dict `LogRegenerator.rebuild` takes.

        `activated_at` is the cutover's `requested_at`, a constant: the
        regenerator stamps the seed-apply with it, and two rebuilds must stamp
        identically or the rebuild-twice gate fails on clock noise.
        """
        return {
            "epoch_id": -self.cutover_id,
            "ontology_version": self.ontology_version,
            "extraction_version": self.extraction_version,
            "model_hash": self.model_hash,
            "activated_at": self.requested_at,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Cutover:
        """Build from an `epoch_cutover` row dict."""
        report = row.get("check_report")
        return cls(
            cutover_id=int(row["cutover_id"]),
            ontology_version=str(row["ontology_version"]),
            extraction_version=str(row["extraction_version"]),
            model_hash=str(row["model_hash"]),
            bare_model_hash=str(row["bare_model_hash"]),
            requested_at=str(row["requested_at"]),
            state=str(row["state"]),
            source_epoch_id=(
                None if row.get("source_epoch_id") is None else int(row["source_epoch_id"])
            ),
            rebuild_job_id=row.get("rebuild_job_id"),
            rebuilt_through_event_id=row.get("rebuilt_through_event_id"),
            check_report=None if report is None else json.loads(report),
            promoted_epoch_id=(
                None if row.get("promoted_epoch_id") is None else int(row["promoted_epoch_id"])
            ),
            updated_at=str(row["updated_at"]),
        )


@dataclass(frozen=True, slots=True)
class FillScan:
    """How far a cutover candidate's re-extraction of the log has got."""

    head: PendingTurn | None
    covered: int
    total: int


@dataclass(frozen=True, slots=True)
class PendingTurn:
    """One not-yet-applied turn, in replay order."""

    event_id: str
    session_id: str
    turn_index: int
    timestamp: str
    cached: bool
    curated: bool

    @property
    def turn_id(self) -> str:
        """`{session_id}:{turn_index}`, the id the contract's `turn_id` carries."""
        return f"{self.session_id}:{self.turn_index}"


@dataclass(frozen=True, slots=True)
class BacklogScan:
    """The backlog for one epoch at one instant."""

    pending: tuple[PendingTurn, ...]
    backlog_depth: int
    apply_pending: int
    dead_lettered: int
    legacy_unextracted: int

    @property
    def head(self) -> PendingTurn | None:
        """The next turn to process, or None when nothing is pending."""
        return self.pending[0] if self.pending else None

    @property
    def oldest_pending_timestamp(self) -> str | None:
        """Logged timestamp of the oldest pending turn (the head, by ordering)."""
        return self.pending[0].timestamp if self.pending else None


@dataclass(frozen=True, slots=True)
class Activation:
    """The first-activation floor for an epoch."""

    epoch_id: int
    activated_at: str
    turns_at_activation: int
    marked_applied: int
    legacy_unextracted: int
    created_now: bool


class TurnProgress:
    """`ApplyProgress` for one turn under one epoch, backed by `extraction_applied`."""

    def __init__(
        self, store: EventStore, *, event_id: str, epoch_id: int, curated: bool, now_iso
    ) -> None:
        self._store = store
        self._event_id = event_id
        self._epoch_id = epoch_id
        self._curated = curated
        self._now_iso = now_iso

    @property
    def curated(self) -> bool:
        """True when the turn's marker already says `curated`."""
        return self._curated

    def mark_curated(self) -> None:
        """Write the `curated` marker."""
        self._store.mark_extraction_stage(
            event_id=self._event_id,
            epoch_id=self._epoch_id,
            stage=STAGE_CURATED,
            updated_at=self._now_iso(),
        )
        self._curated = True

    def mark_applied(self) -> None:
        """Write the `applied` marker."""
        self._store.mark_extraction_stage(
            event_id=self._event_id,
            epoch_id=self._epoch_id,
            stage=STAGE_APPLIED,
            updated_at=self._now_iso(),
        )


class BacklogStore:
    """Backlog queries plus applied/attempts access over one event store and cache."""

    def __init__(self, event_store: EventStore, cache: ExtractionCache) -> None:
        self._events = event_store
        self._cache = cache

    @property
    def event_store(self) -> EventStore:
        """The event store (the log) this backlog reads."""
        return self._events

    @property
    def cache(self) -> ExtractionCache:
        """The extraction cache this backlog reads and writes."""
        return self._cache

    # -- epoch / activation ----------------------------------------------------

    def active_epoch(self) -> Epoch | None:
        """The latest epoch in the ledger, or None when the ledger is empty."""
        row = self._events.get_current_epoch()
        return Epoch.from_row(row) if row is not None else None

    def get_activation(self, epoch: Epoch) -> Activation | None:
        """The stored first-activation floor for `epoch`, or None before activation."""
        row = self._events.get_extraction_activation(epoch.epoch_id)
        if row is None:
            return None
        return Activation(
            epoch_id=int(row["epoch_id"]),
            activated_at=str(row["activated_at"]),
            turns_at_activation=int(row["turns_at_activation"]),
            marked_applied=int(row["marked_applied"]),
            legacy_unextracted=int(row["legacy_unextracted"]),
            created_now=False,
        )

    def ensure_activation(self, epoch: Epoch, *, now_iso: str) -> Activation:
        """Record the first-activation floor for `epoch`, once.

        Every turn logged so far with a cache row under the epoch was extracted
        by the pre-T2a live path, which applied it in the same task that cached
        it (`ExtractionPipeline.extract_from_utterance` records Stage 2 and then
        curates), so it is marked applied. Every turn logged so far with NO row
        is legacy-unextracted and is listed, not dispatched.

        Stable across restarts: the floor is written once per epoch in one
        transaction (`EventStore.record_extraction_activation`), and a later
        call returns the stored floor unchanged.
        """
        existing = self.get_activation(epoch)
        if existing is not None:
            return existing
        keys = self._events.list_turn_keys_in_replay_order()
        cached = self._cache.event_ids_for(epoch.extraction_version, epoch.model_hash)
        applied = [k["event_id"] for k in keys if k["event_id"] in cached]
        legacy = [k["event_id"] for k in keys if k["event_id"] not in cached]
        created = self._events.record_extraction_activation(
            epoch_id=epoch.epoch_id,
            activated_at=now_iso,
            turns_at_activation=len(keys),
            applied_event_ids=applied,
            legacy_event_ids=legacy,
        )
        stored = self.get_activation(epoch)
        if stored is None:  # pragma: no cover -- written or already present above
            raise RuntimeError(f"activation floor for epoch {epoch.epoch_id} not readable")
        if not created:
            return stored
        return Activation(
            epoch_id=stored.epoch_id,
            activated_at=stored.activated_at,
            turns_at_activation=stored.turns_at_activation,
            marked_applied=stored.marked_applied,
            legacy_unextracted=stored.legacy_unextracted,
            created_now=True,
        )

    # -- epoch cutover ---------------------------------------------------------

    def open_cutover(self) -> Cutover | None:
        """The open cutover (filling, ready or checked), or None."""
        row = self._events.get_open_epoch_cutover()
        return Cutover.from_row(row) if row is not None else None

    def get_cutover(self, cutover_id: int) -> Cutover | None:
        """One cutover by id, or None."""
        row = self._events.get_epoch_cutover(cutover_id)
        return Cutover.from_row(row) if row is not None else None

    def list_cutovers(self) -> list[Cutover]:
        """Every cutover, oldest first."""
        return [Cutover.from_row(row) for row in self._events.list_epoch_cutovers()]

    def fill_scan(self, cutover: Cutover) -> FillScan:
        """The candidate's coverage of the log: the first uncovered turn, and counts.

        A turn is covered when the extraction cache has a row for it under the
        candidate's stamps -- extracted, gate-skipped or dead-lettered. The
        head is the first uncovered turn in replay order (the same
        `list_turn_keys_in_replay_order` the active backlog uses), so the
        candidate re-extracts the whole log from the beginning, in log order,
        and a turn logged mid-fill is reached in its place.
        """
        keys = self._events.list_turn_keys_in_replay_order()
        cached = self._cache.event_ids_for(cutover.extraction_version, cutover.model_hash)
        head: PendingTurn | None = None
        covered = 0
        for key in keys:
            if key["event_id"] in cached:
                covered += 1
            elif head is None:
                head = PendingTurn(
                    event_id=str(key["event_id"]),
                    session_id=str(key["session_id"]),
                    turn_index=int(key["turn_index"]),
                    timestamp=str(key["timestamp"]),
                    cached=False,
                    curated=False,
                )
        return FillScan(head=head, covered=covered, total=len(keys))

    def begin_cutover(
        self,
        *,
        ontology_version: str,
        extraction_version: str,
        model_hash: str,
        bare_model_hash: str,
        source_epoch_id: int,
        requested_at: str,
    ) -> Cutover | None:
        """Open a candidate in state 'filling'. None when one is already open."""
        cutover_id = self._events.begin_epoch_cutover(
            ontology_version=ontology_version,
            extraction_version=extraction_version,
            model_hash=model_hash,
            bare_model_hash=bare_model_hash,
            source_epoch_id=source_epoch_id,
            requested_at=requested_at,
        )
        return None if cutover_id is None else self.get_cutover(cutover_id)

    def transition_cutover(
        self,
        cutover: Cutover,
        *,
        from_states: tuple[str, ...],
        to_state: str,
        updated_at: str,
        rebuild_job_id: str | None = None,
        rebuilt_through_event_id: str | None = None,
        check_report: dict[str, Any] | None = None,
        write_check: bool = False,
    ) -> bool:
        """Compare-and-set the cutover's state.

        With `write_check`, the three check columns are written too (None ->
        NULL), which is how a failed re-check clears a stale
        `rebuilt_through_event_id`.
        """
        fields = None
        if write_check:
            fields = {
                "rebuild_job_id": rebuild_job_id,
                "rebuilt_through_event_id": rebuilt_through_event_id,
                "check_report": None if check_report is None else json.dumps(check_report),
            }
        return self._events.transition_epoch_cutover(
            cutover.cutover_id,
            from_states=from_states,
            to_state=to_state,
            updated_at=updated_at,
            fields=fields,
        )

    def promote_cutover(self, cutover: Cutover, *, activated_at: str) -> dict[str, Any]:
        """Append the candidate to the ledger with a first-hand activation (one transaction).

        See `EventStore.promote_epoch_cutover`.
        """
        return self._events.promote_epoch_cutover(
            cutover_id=cutover.cutover_id, activated_at=activated_at
        )

    # -- scanning --------------------------------------------------------------

    def scan(self, epoch: Epoch) -> BacklogScan:
        """The backlog for `epoch`; a marker counts only beside a cache row (see retry)."""
        keys = self._events.list_turn_keys_in_replay_order()
        cached = self._cache.event_ids_for(epoch.extraction_version, epoch.model_hash)
        applied = self._events.get_extraction_applied(epoch.epoch_id)
        legacy = self._events.get_extraction_legacy_turns(epoch.epoch_id)

        pending: list[PendingTurn] = []
        inference_pending = 0
        apply_pending = 0
        for key in keys:
            event_id = key["event_id"]
            stage = applied.get(event_id) if event_id in cached else None
            if stage == STAGE_APPLIED or event_id in legacy:
                continue
            is_cached = event_id in cached
            if is_cached:
                apply_pending += 1
            else:
                inference_pending += 1
            pending.append(
                PendingTurn(
                    event_id=event_id,
                    session_id=str(key["session_id"]),
                    turn_index=int(key["turn_index"]),
                    timestamp=str(key["timestamp"]),
                    cached=is_cached,
                    curated=stage == STAGE_CURATED,
                )
            )
        dead_lettered = sum(1 for reason in cached.values() if reason == SKIP_EXTRACTION_FAILED)
        return BacklogScan(
            pending=tuple(pending),
            backlog_depth=inference_pending,
            apply_pending=apply_pending,
            dead_lettered=dead_lettered,
            legacy_unextracted=len(legacy),
        )

    # -- per-turn reads / writes -----------------------------------------------

    def get_turn(self, event_id: str) -> dict[str, Any] | None:
        """The full logged turn row, or None."""
        return self._events.get_turn(event_id)

    def session_history(self, session_id: str, *, through_turn_index: int, limit: int) -> list:
        """The last `limit` user/assistant messages of a session, through a turn.

        Rebuilt from the log. The in-process path passes
        `session.get_history(max_history)` taken after the turn's own assistant
        message was added (`conversation_handler.py`, the create_task call
        follows `session.add_message("assistant", ...)`), so the window ends
        with this turn's user and assistant messages; this reproduces that
        window from logged turns. It cannot reproduce in-memory-only history
        (e.g. messages of a session that predates a process restart are in the
        log here but were not in memory there).
        """
        messages: list[dict[str, str]] = []
        for turn in self._events.get_turns(session_id):
            if int(turn["turn_index"]) > through_turn_index:
                break
            messages.append({"role": "user", "content": turn["user_utterance"]})
            messages.append({"role": "assistant", "content": turn["system_response"]})
        return messages[-limit:] if limit > 0 else []

    def get_cached(self, turn_event_id: str, epoch: Epoch) -> dict[str, Any] | None:
        """The turn's cached decision under the epoch's stamps, or None."""
        return self._cache.get(turn_event_id, epoch.extraction_version, epoch.model_hash)

    def put_skip(self, event_id: str, epoch: Epoch, *, skip_reason: str, created_at: str) -> None:
        """Record a gate skip or a dead-letter under the epoch's stamps."""
        self._cache.put(
            event_id,
            epoch.ontology_version,
            epoch.extraction_version,
            epoch.model_hash,
            outcome=OUTCOME_SKIPPED,
            skip_reason=skip_reason,
            created_at=created_at,
        )

    def put_extracted(
        self,
        event_id: str,
        epoch: Epoch,
        *,
        created_at: str,
        entities: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        scope: str | None,
        scope_confidence: float | None,
        derivation: dict[str, Any] | None,
        service_stamps: dict[str, Any] | None,
    ) -> None:
        """Record the service's raw Stage-2 result (and Stage 9 ops) under the epoch."""
        self._cache.put(
            event_id,
            epoch.ontology_version,
            epoch.extraction_version,
            epoch.model_hash,
            outcome=OUTCOME_EXTRACTED,
            created_at=created_at,
            entities=entities,
            relationships=relationships,
            scope=scope,
            scope_confidence=scope_confidence,
            derivation=derivation,
            service_stamps=service_stamps,
        )

    def progress(self, turn: PendingTurn, epoch: Epoch, *, now_iso) -> TurnProgress:
        """The turn's durable apply markers (`ApplyProgress`)."""
        return TurnProgress(
            self._events,
            event_id=turn.event_id,
            epoch_id=epoch.epoch_id,
            curated=turn.curated,
            now_iso=now_iso,
        )

    # -- attempts --------------------------------------------------------------

    def next_attempt_number(self, event_id: str, epoch: Epoch) -> int:
        """1-based number for the next service call (retired attempts included)."""
        return self._events.count_extraction_attempts(event_id, epoch.epoch_id) + 1

    def counted_failures(self, event_id: str, epoch: Epoch) -> int:
        """Job-attributable failures since the turn last (re-)entered the backlog."""
        return self._events.count_counted_extraction_failures(event_id, epoch.epoch_id)

    def record_attempt(
        self,
        *,
        turn: PendingTurn,
        epoch: Epoch,
        attempt: int,
        job_id: str,
        request_id: str,
        started_at: str,
        finished_at: str,
        duration_ms: float,
        error_code: str | None,
        outcome: str,
        counted: bool,
    ) -> None:
        """Append one service call to `extraction_attempts`."""
        self._events.append_extraction_attempt(
            event_id=turn.event_id,
            epoch_id=epoch.epoch_id,
            attempt=attempt,
            job_id=job_id,
            request_id=request_id,
            turn_id=turn.turn_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            error_code=error_code,
            outcome=outcome,
            counted=counted,
        )

    # -- dead letters ----------------------------------------------------------

    def list_dead_letters(self, epoch: Epoch) -> list[str]:
        """Event ids dead-lettered under `epoch`, in replay order."""
        cached = self._cache.event_ids_for(epoch.extraction_version, epoch.model_hash)
        return [
            k["event_id"]
            for k in self._events.list_turn_keys_in_replay_order()
            if cached.get(k["event_id"]) == SKIP_EXTRACTION_FAILED
        ]

    def clear_stale_marker(self, turn: PendingTurn, epoch: Epoch) -> bool:
        """Delete the apply marker of a turn that has no cache row under `epoch`.

        The dispatcher calls this for every inference-pending head before it
        caches a new result; a marker there can only be stale. Otherwise a
        crash between that cache write and the apply would leave a cache row
        beside the old `applied` marker, which `scan` reads as done, and the
        turn would never be applied. True when a marker was deleted.
        """
        return self._events.delete_extraction_applied(turn.event_id, epoch.epoch_id)

    def retry_dead_letter(self, event_id: str, epoch: Epoch) -> bool:
        r"""Put a dead-lettered turn back into the backlog.

        Retires its attempts so the failure count restarts, then deletes its
        `extraction_failed` cache row. It does NOT touch the turn's apply
        marker. Refuses (returns False, changes nothing) for a turn that is
        not dead-lettered.

        The turn is then applied OUT OF LOG ORDER: every later turn has already
        been applied, and this one lands after them. Graph state that depends on
        order (dedup winners, supersession) can therefore differ from a rebuild
        until the next epoch cutover re-extracts the log.

        WHY THE MARKER IS LEFT ALONE. This runs in the operator's CLI process,
        usually while the dispatcher runs in the backend. The cache row and the
        markers live in different SQLite files (`ExtractionCache(
        production_cache_path(config))` beside `EventStore(db_path=...)`, see
        `backend/factories.py`), so no transaction can cover both, and a
        dispatcher step can land between any two writes made here. Deleting
        the marker first and the row last (the previous order) was crash-safe
        but not race-safe: a scan between the two saw a skip row with no
        marker, applied it as a no-op and wrote `applied`; the row delete then
        left the turn marked applied with no row -- skipped by the old `scan`,
        refused by this method, stranded.

        So the rule moved into `scan`: a turn with no cache row is
        inference-pending whatever its marker says, and the dispatcher -- the
        only writer of markers after activation -- deletes the stale marker
        (`clear_stale_marker`) before it caches the new result. That is sound
        because an `applied` marker always has a cache row beside it, except
        after this method. Every writer of `applied`:

        - first activation marks only turns with a row:
          `grep -nF 'in cached]' backend/extraction_backlog/store.py` ->
          `ensure_activation`'s `applied` list;
        - the dispatcher marks only after reading the row it applies:
          `ExtractionDispatcher._apply` returns before `apply_cached_turn`
          when `get_cached` is None;
        - cutover promotion marks turns through `rebuilt_through_event_id`,
          and `check_cutover` refuses unless every logged turn has a
          candidate row: `grep -n 'if fill.head is not None'
          backend/extraction_backlog/cutover.py`;
        - dead-lettering writes the `extraction_failed` row, then applies it
          (`ExtractionDispatcher._on_job_error`, `put_skip` then return None).

        And nothing but this method deletes a cache row:
        `grep -rn 'cache.delete(\|DELETE FROM extraction_cache' backend/` ->
        this method and `ExtractionCache.delete` itself.

        The alternative, a durable retry-request record the dispatcher
        consumes, needs a new table in the event store or the cache, both
        outside this package; the record would change who writes, not what
        the scan must tolerate.

        Every interleaving with a running dispatcher then ends with the turn
        re-dispatched and applied once. Before the row delete the turn is
        still a dead letter (marker present or re-written as a no-op); after
        it the turn is inference-pending. The dispatcher never writes a row
        for a turn that has one, and this method only deletes an
        `extraction_failed` row, so neither undoes the other's write. The
        "only" is enforced in the delete itself, not just by the read above
        (`ExtractionCache.delete(..., only_skip_reason=...)` is one
        conditional `DELETE`): with two retries of one turn racing, the first
        can free the turn and the dispatcher re-extract it before the second
        deletes, and an unconditional delete would then remove the real row
        and cause a second extraction and apply.

        Crash ordering: attempts are retired FIRST and the row deleted LAST,
        as the commit point. A crash before the row delete leaves a dead
        letter this method accepts again (retiring is idempotent); a crash
        after it leaves an inference-pending turn.

        Consequence: if cache rows ever went missing for another reason (a
        lost or replaced cache file), their turns become inference-pending
        and are re-extracted in log order; the dispatcher logs each stale
        marker it clears.
        """
        cached = self.get_cached(event_id, epoch)
        if cached is None or cached.get("skip_reason") != SKIP_EXTRACTION_FAILED:
            return False
        self._events.retire_extraction_attempts(event_id, epoch.epoch_id)
        return self._cache.delete(
            event_id,
            epoch.extraction_version,
            epoch.model_hash,
            only_skip_reason=SKIP_EXTRACTION_FAILED,
        )


def age_ms(timestamp_iso: str | None, now: datetime) -> int | None:
    """Milliseconds from a logged ISO timestamp to `now`, None for None."""
    if timestamp_iso is None:
        return None
    then = datetime.fromisoformat(timestamp_iso)
    if then.tzinfo is None or now.tzinfo is None:
        then = then.replace(tzinfo=None)
        now = now.replace(tzinfo=None)
    return max(0, int((now - then).total_seconds() * 1000))
