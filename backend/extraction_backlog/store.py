"""The extraction backlog, as a view over the event log and the extraction cache.

There is no queue table. For the ACTIVE epoch (the latest `epoch_ledger` row):

- a turn is INFERENCE-PENDING when the extraction cache has no row for
  `cache_key(event_id, epoch.extraction_version, epoch.model_hash)`;
- a turn is APPLY-PENDING when it has a cache row but no `applied` marker in
  `extraction_applied`;
- a turn is DONE when its marker says `applied`;
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

STAGE_CURATED = "curated"
STAGE_APPLIED = "applied"


@dataclass(frozen=True, slots=True)
class Epoch:
    """The stamp triple the backlog reads and writes the cache under."""

    epoch_id: int
    ontology_version: str
    extraction_version: str
    model_hash: str

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

    # -- scanning --------------------------------------------------------------

    def scan(self, epoch: Epoch) -> BacklogScan:
        """Compute the backlog for `epoch` from the log, the cache and the markers."""
        keys = self._events.list_turn_keys_in_replay_order()
        cached = self._cache.event_ids_for(epoch.extraction_version, epoch.model_hash)
        applied = self._events.get_extraction_applied(epoch.epoch_id)
        legacy = self._events.get_extraction_legacy_turns(epoch.epoch_id)

        pending: list[PendingTurn] = []
        inference_pending = 0
        apply_pending = 0
        for key in keys:
            event_id = key["event_id"]
            stage = applied.get(event_id)
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

    def retry_dead_letter(self, event_id: str, epoch: Epoch) -> bool:
        """Put a dead-lettered turn back into the backlog.

        Deletes its `extraction_failed` cache row and its applied marker for the
        epoch, and retires its attempts so the failure count restarts. Refuses
        (returns False, changes nothing) for a turn that is not dead-lettered.

        The turn is then applied OUT OF LOG ORDER: every later turn has already
        been applied, and this one lands after them. Graph state that depends on
        order (dedup winners, supersession) can therefore differ from a rebuild
        until the next epoch cutover re-extracts the log.
        """
        cached = self.get_cached(event_id, epoch)
        if cached is None or cached.get("skip_reason") != SKIP_EXTRACTION_FAILED:
            return False
        self._cache.delete(event_id, epoch.extraction_version, epoch.model_hash)
        self._events.delete_extraction_applied(event_id, epoch.epoch_id)
        self._events.retire_extraction_attempts(event_id, epoch.epoch_id)
        return True


def age_ms(timestamp_iso: str | None, now: datetime) -> int | None:
    """Milliseconds from a logged ISO timestamp to `now`, None for None."""
    if timestamp_iso is None:
        return None
    then = datetime.fromisoformat(timestamp_iso)
    if then.tzinfo is None or now.tzinfo is None:
        then = then.replace(tzinfo=None)
        now = now.replace(tzinfo=None)
    return max(0, int((now - then).total_seconds() * 1000))
