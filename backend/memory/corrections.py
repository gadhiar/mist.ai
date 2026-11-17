"""The correction log and the suppression set the read path builds from it (Phase C).

A correction is the user's forget or correct of one fact. `CorrectionService` records it as
one append-only row of the event store's `memory_corrections` table, stamped by the injected
clock or at the `recorded_at` its caller passes (the issuing turn's, for a correction a tool
queued mid-turn), and then calls each recorded listener with the record.

Recording takes effect at read time, before any graph write: `LoggedSuppression` reads the log
into a `Suppression`, which the fetcher reads in stage 1 and applies before the cross-encoder.
A fetch that runs its own stage 1 reads the log, so the correction reaches it at once. What a
prefetch left (its cached stage 1, its finished result, or a run still in flight for the same
words) holds the set read before the correction and is reused for up to 30 s, until a recorded
listener calls `fetcher.invalidate()` (registering one is T34's); the fetcher module's
docstring says when each is reused. What it suppresses
(Design point 12, with plan v2's and v5's changes):

- every version of the corrected `fact_key` recorded at or before the correction, so the same
  fact said again later is served;
- a version the issuing turn asserted (`source_utterance_id` equal to the correction's
  `source_event_id`) when it is the same fact, by key or by the (subject name, relation,
  object name) triple, so the turn that says "forget that" cannot re-assert it while every
  other fact it produced is still served;
- a vault note whose whole text holds the corrected object name as a whole word and which was
  recorded no later than the correction.

Applying a correction to the graph is T22's; the log's rows never change, and each epoch's
progress through them lives in `memory_corrections_applied`.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from backend.errors import MemoryCorrectionError, MistError

from .fact_index import IndexedFact

logger = logging.getLogger(__name__)

CorrectionAction = Literal["forget", "correct"]
CorrectionSurface = Literal["ws", "tool", "cli"]

CORRECTION_ACTIONS: tuple[str, ...] = ("forget", "correct")
CORRECTION_SURFACES: tuple[str, ...] = ("ws", "tool", "cli")

# `corr-` and a uuid4's 32 hex digits.
_CORRECTION_ID = re.compile(r"^corr-[0-9a-f]{32}$")


@dataclass(frozen=True, slots=True)
class CorrectedFact:
    """The fact a correction names, by the ids and names the read path served it under.

    Attributes:
        subject_id: Canonical id of the fact's subject.
        subject_name: The subject's name.
        relation: The relationship type.
        object_id: Canonical id of the fact's object.
        object_name: The object's name; the narrative rule matches notes on it.
    """

    subject_id: str
    subject_name: str
    relation: str
    object_id: str
    object_name: str


@dataclass(frozen=True, slots=True)
class CorrectionRecord:
    """One `memory_corrections` row, as recorded and as read back.

    Attributes:
        correction_id: `corr-<uuid4 hex>`.
        recorded_at: When the correction takes effect, aware and in UTC.
        action: `forget` or `correct`.
        fact_key: The corrected fact's key, the `item_id` it was served under.
        replacement_object_name: A correct's new object name; None for a forget.
        surface: Where the user asked: `ws`, `tool` or `cli`.
        session_id: The session it was asked in, if any.
        conversation_id: The conversation it was asked in, if any.
        turn_index: The issuing turn's index, if any.
        source_event_id: The event id of the turn that issued it, if one was appended.
        origin: Provenance, as a session's `origin` is (`real` for genuine use).
    """

    correction_id: str
    recorded_at: datetime
    action: CorrectionAction
    fact_key: str
    subject_id: str
    subject_name: str
    relation: str
    object_id: str
    object_name: str
    replacement_object_name: str | None
    surface: CorrectionSurface
    session_id: str | None
    conversation_id: str | None
    turn_index: int | None
    source_event_id: str | None
    origin: str

    def to_row(self) -> dict[str, Any]:
        """The row `EventStore.append_memory_correction` takes.

        `recorded_at` is written as ISO-8601 UTC with microseconds, one fixed width, so the
        log's text order is its time order.
        """
        row = {name: getattr(self, name) for name in self.__slots__}
        row["recorded_at"] = self.recorded_at.astimezone(UTC).isoformat(timespec="microseconds")
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> CorrectionRecord:
        """The record of one row `EventStore.list_memory_corrections` returned.

        Raises:
            MemoryCorrectionError: `recorded_at` is not an ISO-8601 instant.
        """
        try:
            recorded_at = datetime.fromisoformat(str(row["recorded_at"]))
        except ValueError as exc:
            raise MemoryCorrectionError(
                f"correction {row.get('correction_id')!r} has an unreadable recorded_at"
            ) from exc
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.replace(tzinfo=UTC)
        values = {name: row[name] for name in cls.__slots__ if name != "recorded_at"}
        return cls(recorded_at=recorded_at.astimezone(UTC), **values)


RecordedListener = Callable[[CorrectionRecord], None]


class CorrectionLog(Protocol):
    """The event-store calls the correction log needs; `EventStore` satisfies it."""

    def append_memory_correction(self, row: Mapping[str, Any]) -> None:
        """Append one row to `memory_corrections`."""
        ...

    def list_memory_corrections(
        self, *, origins: tuple[str, ...] | None = None
    ) -> list[dict[str, Any]]:
        """Every logged correction in log order, optionally of some origins only."""
        ...


class CorrectionService:
    """Records the user's forgets and corrects in the correction log.

    Every collaborator is injected (the mist.ai DI rule); the factory passes the event store
    and `build_now_fn()`, so a fixed-clock run stamps corrections the way it stamps turns.

    Args:
        event_store: The log's store.
        now_fn: The clock a correction is stamped by when its caller passes no `recorded_at`.
    """

    def __init__(self, event_store: CorrectionLog, now_fn: Callable[[], datetime]) -> None:
        self._log = event_store
        self._now_fn = now_fn
        self._listeners: list[RecordedListener] = []

    def add_recorded_listener(self, listener: RecordedListener) -> None:
        """Call `listener` with each new record, synchronously, after its row is appended.

        A listener must return at once. One that raises a MistError is logged and the others
        still run; any other exception propagates to the caller after the row is logged. A
        listener of another shape is wrapped in an adapter, never registered bare: the
        fetcher's `invalidate(session_id=None)` registered bare would take the record as a
        session id and drop nothing.
        """
        self._listeners.append(listener)

    def forget(
        self,
        item_id: str,
        surface: CorrectionSurface,
        *,
        fact: CorrectedFact,
        session_id: str | None = None,
        conversation_id: str | None = None,
        turn_index: int | None = None,
        source_event_id: str | None = None,
        origin: str = "real",
        recorded_at: datetime | None = None,
        correction_id: str | None = None,
    ) -> CorrectionRecord:
        """Record a forget of the fact served as `item_id`.

        Args:
            item_id: The fact's `fact_key`.
            surface: Where the user asked.
            fact: The fact's ids and names.
            session_id: The session it was asked in, if any.
            conversation_id: The conversation it was asked in, if any.
            turn_index: The issuing turn's index, if any.
            source_event_id: The issuing turn's event id, once it is appended.
            origin: Provenance, as the session's.
            recorded_at: When it takes effect: the issuing turn's `recorded_at` for a
                correction a tool queued; None reads `now_fn`. Must be timezone-aware.
            correction_id: An existing `corr-<32 hex>` id to keep, as a fixture's; None
                makes a new one.

        Returns:
            The record, as logged.

        Raises:
            MemoryCorrectionError: An argument is invalid, or the store fails; nothing is
                logged.
        """
        return self._record(
            "forget",
            item_id,
            None,
            surface,
            fact=fact,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
            source_event_id=source_event_id,
            origin=origin,
            recorded_at=recorded_at,
            correction_id=correction_id,
        )

    def correct(
        self,
        item_id: str,
        replacement_object: str,
        surface: CorrectionSurface,
        *,
        fact: CorrectedFact,
        session_id: str | None = None,
        conversation_id: str | None = None,
        turn_index: int | None = None,
        source_event_id: str | None = None,
        origin: str = "real",
        recorded_at: datetime | None = None,
        correction_id: str | None = None,
    ) -> CorrectionRecord:
        """Record that the fact served as `item_id` has `replacement_object` as its object.

        The arguments, result and errors are `forget`'s, with `replacement_object`, which must
        not be blank.
        """
        return self._record(
            "correct",
            item_id,
            replacement_object,
            surface,
            fact=fact,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
            source_event_id=source_event_id,
            origin=origin,
            recorded_at=recorded_at,
            correction_id=correction_id,
        )

    def _record(
        self,
        action: CorrectionAction,
        item_id: str,
        replacement: str | None,
        surface: CorrectionSurface,
        *,
        fact: CorrectedFact,
        session_id: str | None,
        conversation_id: str | None,
        turn_index: int | None,
        source_event_id: str | None,
        origin: str,
        recorded_at: datetime | None,
        correction_id: str | None,
    ) -> CorrectionRecord:
        """Validate, stamp, append the row, then call the listeners."""
        if surface not in CORRECTION_SURFACES:
            raise MemoryCorrectionError(
                f"surface must be one of {CORRECTION_SURFACES}, got {surface!r}"
            )
        if not item_id.strip():
            raise MemoryCorrectionError("item_id must name the corrected fact's key")
        if not fact.object_name.strip() or not fact.subject_name.strip():
            raise MemoryCorrectionError("the corrected fact needs its subject and object names")
        if action == "correct" and (replacement is None or not replacement.strip()):
            raise MemoryCorrectionError("a correct needs a non-blank replacement object")
        if correction_id is not None and not _CORRECTION_ID.match(correction_id):
            raise MemoryCorrectionError(
                f"correction_id must be corr- and 32 hex digits, got {correction_id!r}"
            )
        at = recorded_at if recorded_at is not None else self._now_fn()
        if at.tzinfo is None:
            raise MemoryCorrectionError("recorded_at must be timezone-aware")
        record = CorrectionRecord(
            correction_id=correction_id or f"corr-{uuid.uuid4().hex}",
            recorded_at=at.astimezone(UTC),
            action=action,
            fact_key=item_id,
            subject_id=fact.subject_id,
            subject_name=fact.subject_name,
            relation=fact.relation,
            object_id=fact.object_id,
            object_name=fact.object_name,
            replacement_object_name=replacement.strip() if replacement is not None else None,
            surface=surface,
            session_id=session_id,
            conversation_id=conversation_id,
            turn_index=turn_index,
            source_event_id=source_event_id,
            origin=origin,
        )
        try:
            self._log.append_memory_correction(record.to_row())
        except (sqlite3.Error, ValueError) as exc:
            raise MemoryCorrectionError(
                f"correction {record.correction_id} could not be logged: {exc}"
            ) from exc
        logger.info(
            "memory correction %s recorded: %s of %s (%s)",
            record.correction_id,
            action,
            item_id,
            surface,
        )
        for listener in tuple(self._listeners):
            try:
                listener(record)
            except MistError as exc:
                logger.error(
                    "memory correction listener failed for %s: %s: %s",
                    record.correction_id,
                    type(exc).__name__,
                    exc,
                )
        return record


@dataclass(frozen=True, slots=True)
class _Named:
    """A corrected object name as a whole-word pattern, with its latest correction's time."""

    pattern: re.Pattern[str]
    recorded_at: datetime


@dataclass(frozen=True, slots=True)
class Suppression:
    """What the corrections suppress at one read; empty by default.

    Attributes:
        fact_keys: Each corrected `fact_key`, mapped to its latest correction's time; a
            version of it recorded at or before that time is suppressed.
        issued_keys: (`source_event_id`, `fact_key`) of each correction a turn issued.
        issued_triples: (`source_event_id`, subject name, relation, object name) of each
            correction a turn issued, names case-folded.
        names: Each corrected object name, for the narrative rule.
    """

    fact_keys: Mapping[str, datetime] = field(default_factory=dict)
    issued_keys: frozenset[tuple[str, str]] = frozenset()
    issued_triples: frozenset[tuple[str, str, str, str]] = frozenset()
    names: tuple[_Named, ...] = ()

    @classmethod
    def of(cls, records: Iterable[CorrectionRecord]) -> Suppression:
        """The suppression set of every record, forgets and corrects alike."""
        fact_keys: dict[str, datetime] = {}
        issued_keys: set[tuple[str, str]] = set()
        issued_triples: set[tuple[str, str, str, str]] = set()
        named: dict[str, datetime] = {}
        for record in records:
            at = record.recorded_at
            fact_keys[record.fact_key] = max(at, fact_keys.get(record.fact_key, at))
            if record.source_event_id is not None:
                issued_keys.add((record.source_event_id, record.fact_key))
                issued_triples.add(
                    (
                        record.source_event_id,
                        record.subject_name.casefold(),
                        record.relation,
                        record.object_name.casefold(),
                    )
                )
            name = " ".join(record.object_name.split()).casefold()
            if name:
                named[name] = max(at, named.get(name, at))
        return cls(
            fact_keys=fact_keys,
            issued_keys=frozenset(issued_keys),
            issued_triples=frozenset(issued_triples),
            names=tuple(_Named(_whole_words(name), at) for name, at in named.items()),
        )

    @property
    def narrates(self) -> bool:
        """Whether any correction names an object a note could hold."""
        return bool(self.names)

    def suppresses_fact(self, fact: IndexedFact) -> bool:
        """Whether a fact version is suppressed.

        It is when its key was corrected and it was recorded at or before the latest such
        correction (a `recorded_at` that is missing or unreadable counts as before), or when
        the turn that issued a correction asserted it and it is that correction's fact, by
        key or by the (subject name, relation, object name) triple.
        """
        line = self.fact_keys.get(fact.fact_key)
        if line is not None:
            at = _instant(fact.recorded_at)
            if at is None or at <= line:
                return True
        utterance = fact.source_utterance_id
        if utterance is None:
            return False
        if (utterance, fact.fact_key) in self.issued_keys:
            return True
        triple = (
            utterance,
            fact.subject_name.casefold(),
            fact.rel_type,
            fact.object_name.casefold(),
        )
        return triple in self.issued_triples

    def suppresses_note(self, text: str, when: datetime | None) -> bool:
        """Whether a note's whole text is suppressed.

        It is when it holds a corrected object name as a whole word, in any case (both are
        case-folded) and with any whitespace between the name's words, and the note was
        recorded (`when`, its `last_updated` or mtime as the note pool reads it) no later than
        that name's latest correction. A note with no time counts as earlier.
        """
        folded = text.casefold()
        for named in self.names:
            if (when is None or when <= named.recorded_at) and named.pattern.search(folded):
                return True
        return False


NO_SUPPRESSION = Suppression()


class LoggedSuppression:
    """The fetcher's suppression source: the correction log read into a `Suppression`.

    The fetcher calls `suppression` in stage 1 on its `memory-index` executor, so the read is
    blocking. Stage 1 is cached per session; a recorded listener that calls
    `fetcher.invalidate()` drops the cache, so the next fetch reads the log again.

    Args:
        log: The event store holding the log.
        origins: Only corrections of these origins suppress; None takes every origin.
    """

    def __init__(self, log: CorrectionLog, *, origins: tuple[str, ...] | None = None) -> None:
        self._log = log
        self._origins = origins

    def suppression(self) -> Suppression:
        """The suppression set of every correction logged now.

        Raises:
            MemoryCorrectionError: The log could not be read.
        """
        try:
            rows = self._log.list_memory_corrections(origins=self._origins)
        except sqlite3.Error as exc:
            raise MemoryCorrectionError(f"the correction log could not be read: {exc}") from exc
        return Suppression.of(CorrectionRecord.from_row(row) for row in rows)


def _whole_words(name: str) -> re.Pattern[str]:
    """`name`, already case-folded, as a pattern matching it only as whole words.

    It is matched against case-folded text, not with `re.IGNORECASE`: folding maps "ß" to
    "ss", which a case-insensitive match never does, so the name and the text are folded alike.
    """
    words = r"\s+".join(re.escape(word) for word in name.split())
    return re.compile(rf"(?<!\w){words}(?!\w)")


def _instant(value: str | None) -> datetime | None:
    """A stored `recorded_at` as an aware instant, UTC when naive; None when unreadable."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


__all__ = [
    "CORRECTION_ACTIONS",
    "CORRECTION_SURFACES",
    "NO_SUPPRESSION",
    "CorrectedFact",
    "CorrectionAction",
    "CorrectionLog",
    "CorrectionRecord",
    "CorrectionService",
    "CorrectionSurface",
    "LoggedSuppression",
    "RecordedListener",
    "Suppression",
]
