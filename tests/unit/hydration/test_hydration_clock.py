"""B2: authored per-turn timestamps, keyed rather than sequential.

`_record_turn_event` stamps `recorded_at`, which becomes the fact-time
authority for every bitemporal edge the turn produces. Before this it read
`datetime.now(UTC)` DIRECTLY -- bypassing the `_now_fn` DI seam that already
existed on the same class, three lines under a comment reading "never
wall-clock at write time, design 4.2".

Under hydration that means the golden log's authored gap ladder (2025-09 ->
2026-07-15) collapses onto the moment the run happened. The failure direction
is what makes it serious: both sides of `live == rebuilt` read the timestamp
back out of the event store, so the gate goes GREEN while valid-time intervals,
supersession ordering, the currency triple and R1.5 staleness were all
evaluated against a timeline that never existed.

Every test below is either about producing the authored timeline, or about
refusing rather than guessing when it cannot.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from backend.chat.hydration_clock import (
    HydrationClock,
    HydrationClockError,
    load_hydration_clock,
)

_ROWS = [
    {"session_id": "golden-2025-09", "turn_index": 0, "timestamp": "2025-09-02T08:00:00+00:00"},
    {"session_id": "golden-2025-09", "turn_index": 1, "timestamp": "2025-09-02T08:05:00+00:00"},
    {"session_id": "golden-2026-07", "turn_index": 0, "timestamp": "2026-07-15T10:00:00+00:00"},
]


def _corpus(tmp_path, rows=None, name="corpus.jsonl"):
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(r) for r in (rows if rows is not None else _ROWS)),
        encoding="utf-8",
    )
    return path


class TestLoad:
    def test_loads_every_turn_keyed_by_session_and_index(self, tmp_path):
        clock = load_hydration_clock(_corpus(tmp_path))
        assert len(clock.timestamps) == 3
        assert clock.recorded_at_for("golden-2025-09", 1) == datetime(2025, 9, 2, 8, 5, tzinfo=UTC)

    def test_the_authored_gap_ladder_survives(self, tmp_path):
        """The property the whole item exists for: turns are not simultaneous."""
        clock = load_hydration_clock(_corpus(tmp_path))
        first = clock.recorded_at_for("golden-2025-09", 0)
        last = clock.recorded_at_for("golden-2026-07", 0)
        assert (last - first).days > 300

    def test_blank_and_comment_lines_are_skipped(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text(f"# header\n\n{json.dumps(_ROWS[0])}\n\n", encoding="utf-8")
        assert len(load_hydration_clock(path).timestamps) == 1

    def test_the_real_golden_log_loads(self):
        """Non-vacuity against the actual corpus, not only a synthetic one.

        The clock's whole contract is that it is total over the corpus the
        hydration run drives. A fixture-only test would pass while the real
        file lacked a field the loader requires.
        """
        clock = load_hydration_clock("data/golden-log/golden-log.jsonl")
        assert len(clock.timestamps) >= 80
        span = max(clock.timestamps.values()) - min(clock.timestamps.values())
        assert span.days > 300, "the authored ladder must span months, not minutes"


class TestLoadRefusals:
    """Every one of these fails the run rather than producing a partial clock."""

    def test_missing_file(self, tmp_path):
        with pytest.raises(HydrationClockError, match="not found"):
            load_hydration_clock(tmp_path / "nope.jsonl")

    def test_empty_corpus(self, tmp_path):
        with pytest.raises(HydrationClockError, match="zero turns"):
            load_hydration_clock(_corpus(tmp_path, rows=[]))

    def test_malformed_json(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text("{not json}", encoding="utf-8")
        with pytest.raises(HydrationClockError, match="not valid JSON"):
            load_hydration_clock(path)

    @pytest.mark.parametrize("field", ["session_id", "turn_index", "timestamp"])
    def test_missing_required_field(self, tmp_path, field):
        row = dict(_ROWS[0])
        del row[field]
        with pytest.raises(HydrationClockError, match=field):
            load_hydration_clock(_corpus(tmp_path, rows=[row]))

    def test_unparseable_timestamp(self, tmp_path):
        row = dict(_ROWS[0], timestamp="last tuesday")
        with pytest.raises(HydrationClockError, match="ISO 8601"):
            load_hydration_clock(_corpus(tmp_path, rows=[row]))

    def test_naive_timestamp_is_refused(self, tmp_path):
        """`recorded_at` is a bitemporal authority; a naive value is ambiguous."""
        row = dict(_ROWS[0], timestamp="2025-09-02T08:00:00")
        with pytest.raises(HydrationClockError, match="naive"):
            load_hydration_clock(_corpus(tmp_path, rows=[row]))

    def test_duplicate_key_is_refused(self, tmp_path):
        """Two timestamps for one turn means the corpus defines no timeline.

        The alternative is last-writer-wins, which picks one silently.
        """
        rows = [_ROWS[0], dict(_ROWS[0], timestamp="2025-09-02T09:00:00+00:00")]
        with pytest.raises(HydrationClockError, match="duplicates key"):
            load_hydration_clock(_corpus(tmp_path, rows=rows))

    def test_non_integer_turn_index(self, tmp_path):
        row = dict(_ROWS[0], turn_index="first")
        with pytest.raises(HydrationClockError, match="not an integer"):
            load_hydration_clock(_corpus(tmp_path, rows=[row]))


class TestLookup:
    def test_a_missing_key_raises_and_does_not_fall_back(self):
        """The fail-closed arm, and the reason for keying at all.

        A miss means the event store's `turn_count` has diverged from the
        corpus's `turn_index`. Stamping wall-clock there would bake the
        divergence into every bitemporal edge the turn produces, and the run
        would complete looking fine.
        """
        clock = HydrationClock(
            timestamps={("s", 0): datetime(2025, 9, 2, tzinfo=UTC)}, source_path="x"
        )
        with pytest.raises(HydrationClockError, match="turn_index=7"):
            clock.recorded_at_for("s", 7)

    def test_a_wrong_session_raises(self):
        clock = HydrationClock(
            timestamps={("s", 0): datetime(2025, 9, 2, tzinfo=UTC)}, source_path="x"
        )
        with pytest.raises(HydrationClockError, match="other"):
            clock.recorded_at_for("other", 0)

    def test_order_does_not_matter(self, tmp_path):
        """The property a sequential clock cannot offer.

        A pop-the-next implementation would return a different answer depending
        on the order lookups happened in, and one extra call anywhere would
        shift every later turn silently.
        """
        clock = load_hydration_clock(_corpus(tmp_path))
        forward = [
            clock.recorded_at_for("golden-2025-09", 0),
            clock.recorded_at_for("golden-2025-09", 1),
        ]
        reverse = [
            clock.recorded_at_for("golden-2025-09", 1),
            clock.recorded_at_for("golden-2025-09", 0),
        ]
        assert forward == list(reversed(reverse))

    def test_repeated_lookups_are_stable(self, tmp_path):
        """Extra `now_fn`-shaped calls must not advance anything."""
        clock = load_hydration_clock(_corpus(tmp_path))
        seen = {clock.recorded_at_for("golden-2025-09", 0) for _ in range(5)}
        assert len(seen) == 1
