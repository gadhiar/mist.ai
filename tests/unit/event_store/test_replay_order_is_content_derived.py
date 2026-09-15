"""Replay order must be a function of the log, not of the database file. MIS-138.

## The gap these tests close

`get_all_turns_for_reextraction` ordered by `e.rowid ASC`, and its own comment said
why: *"Use rowid for stable ordering since event_id is a UUID"* -- correct about
`event_id` (`store.py:203` assigns `str(uuid.uuid4())`, so ordering by it is an
arbitrary permutation) and wrong about the remedy.

`schema.sql:20` declares `event_id TEXT PRIMARY KEY` with no explicit
`INTEGER PRIMARY KEY`, so the table's rowid is an implicit physical row number, and
SQLite documents that `VACUUM` may renumber rowids for exactly such tables.
**Replay order was therefore a property of the database FILE, not of the logged
events** -- and ADR-023 section 2 claims the entity subgraph is a function of the
log.

It is not cosmetic. `EntityDeduplicator._find_existing` resolves each incoming
entity against whatever is already in the target graph
(`curation/deduplication.py:130-172`), so processing order decides which entity wins
`display_name`, `description`, `entity_type`, and the alias union. A rebuild that
replays in a different order than live accumulated produces different facts and the
gate reports it as non-determinism.

The order is now `(timestamp, session_id, turn_index)` -- all three in the log,
none in the file. `timestamp` leads because that is the order the LIVE path applied
turns in (arrival order), which is what a rebuild has to reproduce;
`(session_id, turn_index)` is the deterministic tiebreak for equal timestamps.

## Why `to_row` normalises to UTC, and why that is load-bearing here

SQLite orders TEXT lexicographically. For ISO-8601 that equals chronological order
ONLY when every value carries the same offset. `2026-01-01T10:00:00+05:00` (05:00Z)
sorts AFTER `2026-01-01T06:00:00+00:00` (06:00Z) as a string while being earlier as
an instant, so a single mixed-offset writer silently misorders the replay.

Every writer happens to pass UTC today -- `datetime.now(UTC)` on the live path, and
`load_hydration_clock` normalises with `.astimezone(UTC)` and refuses naive values.
But that is a property of the CALLERS, and the ordering correctness of the store
should not depend on auditing them. `ConversationTurnEvent.to_row` now normalises,
so the canonical stored format is a property of the store. `TestOffsetsDoNotMisorder`
is the test that would have caught the string-comparison assumption.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def _store() -> EventStore:
    store = EventStore(":memory:")
    store.initialize()
    return store


def _turn(session_id: str, turn_index: int, timestamp: datetime) -> ConversationTurnEvent:
    return ConversationTurnEvent(
        session_id=session_id,
        turn_index=turn_index,
        timestamp=timestamp,
        user_utterance=f"u{turn_index}",
        system_response=f"r{turn_index}",
    )


def _append(store: EventStore, session_id: str, turn_index: int, timestamp: datetime) -> str:
    return store.append_turn(_turn(session_id, turn_index, timestamp))


def _replayed(store: EventStore) -> list[tuple[str, int]]:
    """(session_id, turn_index) in the order a rebuild would replay them."""
    return [
        (t["session_id"], t["turn_index"])
        for t in store.get_all_turns_for_reextraction(origins=None)
    ]


class TestOrderIsContentDerivedNotInsertionDerived:
    """Insert out of chronological order; replay must still be chronological."""

    def test_later_timestamp_inserted_first_still_replays_second(self):
        store = _store()
        store.start_session("s1")
        # Inserted newest-first, so rowid order is the REVERSE of the real timeline.
        _append(store, "s1", 1, _BASE + timedelta(minutes=5))
        _append(store, "s1", 0, _BASE)

        assert _replayed(store) == [("s1", 0), ("s1", 1)]

    def test_interleaved_sessions_replay_in_wall_clock_order(self):
        """Live applied these interleaved; a per-session order would not reproduce it."""
        store = _store()
        store.start_session("sA")
        store.start_session("sB")
        _append(store, "sA", 0, _BASE)
        _append(store, "sB", 0, _BASE + timedelta(minutes=1))
        _append(store, "sA", 1, _BASE + timedelta(minutes=2))
        _append(store, "sB", 1, _BASE + timedelta(minutes=3))

        assert _replayed(store) == [("sA", 0), ("sB", 0), ("sA", 1), ("sB", 1)]

    def test_order_survives_insertion_in_a_scrambled_sequence(self):
        store = _store()
        store.start_session("s1")
        for idx in (3, 0, 4, 2, 1):
            _append(store, "s1", idx, _BASE + timedelta(minutes=idx))

        assert _replayed(store) == [("s1", i) for i in range(5)]


class TestTiesAreBrokenDeterministically:
    """Equal timestamps must not leave the order up to the file."""

    def test_same_timestamp_different_sessions_orders_by_session_id(self):
        store = _store()
        store.start_session("sB")
        store.start_session("sA")
        _append(store, "sB", 0, _BASE)
        _append(store, "sA", 0, _BASE)

        assert _replayed(store) == [("sA", 0), ("sB", 0)]

    def test_same_timestamp_same_session_orders_by_turn_index(self):
        store = _store()
        store.start_session("s1")
        _append(store, "s1", 2, _BASE)
        _append(store, "s1", 0, _BASE)
        _append(store, "s1", 1, _BASE)

        assert _replayed(store) == [("s1", 0), ("s1", 1), ("s1", 2)]


class TestStoredTimestampsAreCanonicalUTC:
    """What makes the SQL lexicographic ordering provably correct."""

    def test_a_non_utc_offset_is_stored_as_utc(self):
        store = _store()
        store.start_session("s1")
        plus_five = timezone(timedelta(hours=5))
        event_id = _append(store, "s1", 0, datetime(2026, 1, 1, 10, 0, 0, tzinfo=plus_five))

        # Reading the raw column deliberately: the point is the stored TEXT, which
        # is what SQLite's ORDER BY compares. A decoded datetime would hide it.
        conn = store._get_connection()
        stored = conn.execute(
            "SELECT timestamp FROM conversation_turn_events WHERE event_id = ?", (event_id,)
        ).fetchone()[0]

        assert stored.endswith("+00:00"), f"stored offset not canonical: {stored!r}"
        assert datetime.fromisoformat(stored) == datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)

    def test_the_instant_round_trips_unchanged(self):
        """Normalising must move the offset, never the instant."""
        store = _store()
        store.start_session("s1")
        plus_five = timezone(timedelta(hours=5))
        original = datetime(2026, 1, 1, 10, 0, 0, tzinfo=plus_five)
        _append(store, "s1", 0, original)

        # Turns come back with `timestamp` as the raw ISO string, so parse to
        # compare instants -- aware-datetime equality is offset-independent and
        # string equality is not, which is the whole distinction under test.
        [turn] = store.get_all_turns_for_reextraction(origins=None)
        assert datetime.fromisoformat(turn["timestamp"]) == original


class TestOffsetsDoNotMisorder:
    """The test that would have caught the string-comparison assumption."""

    def test_mixed_offsets_order_by_instant_not_by_string(self):
        store = _store()
        store.start_session("s1")
        plus_five = timezone(timedelta(hours=5))

        # 10:00+05:00 is 05:00Z -- EARLIER than 06:00Z, but sorts LATER as a string.
        _append(store, "s1", 0, datetime(2026, 1, 1, 10, 0, 0, tzinfo=plus_five))
        _append(store, "s1", 1, datetime(2026, 1, 1, 6, 0, 0, tzinfo=UTC))

        # Compared as INSTANTS, not as strings. Asserting the strings are sorted
        # would pass trivially once they are canonical, and would have passed for
        # the wrong reason before -- the defect is an ordering that disagrees with
        # the timeline, which only parsed values can express.
        instants = [
            datetime.fromisoformat(t["timestamp"])
            for t in store.get_all_turns_for_reextraction(origins=None)
        ]
        assert instants == sorted(instants), "replay order disagrees with the instants"
        assert instants[0] == datetime(2026, 1, 1, 5, 0, 0, tzinfo=UTC)


class TestResumeCursorAgreesWithTheOrder:
    """`after_event_id` must mean "after, in replay order", not "after, by rowid"."""

    def test_resume_returns_the_chronological_remainder(self):
        store = _store()
        store.start_session("s1")
        # Insert newest-first so rowid order and replay order disagree.
        third = _append(store, "s1", 2, _BASE + timedelta(minutes=2))
        first = _append(store, "s1", 0, _BASE)
        _append(store, "s1", 1, _BASE + timedelta(minutes=1))

        after_first = store.get_all_turns_for_reextraction(after_event_id=first, origins=None)
        assert [(t["session_id"], t["turn_index"]) for t in after_first] == [("s1", 1), ("s1", 2)]

        after_third = store.get_all_turns_for_reextraction(after_event_id=third, origins=None)
        assert after_third == []

    def test_resume_excludes_the_cursor_turn_itself(self):
        store = _store()
        store.start_session("s1")
        _append(store, "s1", 0, _BASE)
        second = _append(store, "s1", 1, _BASE + timedelta(minutes=1))

        remainder = store.get_all_turns_for_reextraction(after_event_id=second, origins=None)
        assert remainder == []
