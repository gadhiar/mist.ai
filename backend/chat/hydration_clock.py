"""B2: authored per-turn timestamps for a hydration run.

`_record_turn_event` stamps `recorded_at` on the conversation turn, and that
instant becomes the fact-time authority for every bitemporal edge the turn
produces. Under hydration it must be the corpus's AUTHORED timestamp, not the
wall clock, or the golden log's gap ladder (2025-09 -> 2026-07-15) collapses
onto the moment the run happened.

That would not merely be wrong -- it would be wrong in the direction that
passes. Both sides of `live == rebuilt` read the timestamp back out of the
event store, so the gate goes GREEN while valid-time intervals, supersession
ordering, the currency triple and R1.5 staleness were all evaluated on a
timeline that never existed.

Why keyed and not sequential
----------------------------

The obvious implementation pops the next timestamp on each call. It couples to
call ORDER: one extra `now_fn()` anywhere in a turn shifts every subsequent
turn by one, silently, and the run still completes. A wrong-but-plausible
timeline fails green for the same reason above.

Keying on `(session_id, turn_index)` removes the coupling entirely. Order does
not matter, extra calls do not matter, and a key that is not in the corpus
RAISES rather than falling back to wall-clock -- so a desync between the event
store's `turn_count` and the corpus's `turn_index` surfaces on the first turn
that disagrees instead of corrupting the timeline quietly.

Why not MIST_FIXED_CLOCK
------------------------

It shares the `_now_fn` seam and nothing else. `build_now_fn` returns
`lambda: fixed_dt` -- one constant forever -- which applied to 87 turns is
precisely the degenerate timeline this module exists to prevent. It also has a
live eval contract (`resolve_fixed_rendered_at`: the value "round-trips
byte-for-byte into the seeded `users/<id>.md` Provenance block"), and
overloading it to mean "per-turn sequence" would break that. Sibling
implementations behind one seam, not one implementation doing two jobs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class HydrationClockError(RuntimeError):
    """Raised when the hydration clock cannot supply an authored timestamp."""


@dataclass(frozen=True, slots=True)
class HydrationClock:
    """Maps `(session_id, turn_index)` to the corpus's authored timestamp.

    Immutable and total over the corpus it was built from: every lookup either
    returns an authored instant or raises. There is deliberately no default.
    """

    timestamps: dict[tuple[str, int], datetime]
    source_path: str

    def recorded_at_for(self, session_id: str, turn_index: int) -> datetime:
        """Return the authored timestamp for one turn.

        Raises:
            HydrationClockError: when the key is absent. This is the fail-closed
                arm and it is the point of the class -- a miss means the event
                store's `turn_count` has diverged from the corpus's
                `turn_index`, and any fallback would bake that divergence into
                the graph as a plausible-looking timeline.
        """
        key = (session_id, turn_index)
        stamp = self.timestamps.get(key)
        if stamp is None:
            raise HydrationClockError(
                f"No authored timestamp for session={session_id!r} "
                f"turn_index={turn_index}. The hydration clock is total over "
                f"{self.source_path!r} ({len(self.timestamps)} turns) and has no "
                "fallback: a miss means the event store's turn_count has diverged "
                "from the corpus's turn_index, and stamping wall-clock here would "
                "bake that divergence into every bitemporal edge this turn "
                "produces. Refusing."
            )
        return stamp


def load_hydration_clock(path: str | Path) -> HydrationClock:
    """Build a clock from a JSONL corpus carrying session_id/turn_index/timestamp.

    The golden log already carries all three per line, so no new fixture format
    is introduced.

    Raises:
        HydrationClockError: for a missing file, a malformed line, a missing or
            unparsable field, a non-UTC-convertible timestamp, a duplicate key,
            or an empty corpus. Every one of these fails the run rather than
            producing a partial clock -- a clock with holes is worse than no
            clock, because its holes only surface as a timeline that looks fine.
    """
    source = Path(path)
    if not source.is_file():
        raise HydrationClockError(f"Hydration clock corpus not found: {source}")

    timestamps: dict[tuple[str, int], datetime] = {}
    for lineno, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HydrationClockError(f"{source}:{lineno} is not valid JSON: {exc}") from exc

        missing = [f for f in ("session_id", "turn_index", "timestamp") if f not in row]
        if missing:
            raise HydrationClockError(
                f"{source}:{lineno} is missing {', '.join(missing)}. The clock keys on "
                "session_id and turn_index and cannot infer either."
            )

        try:
            stamp = datetime.fromisoformat(str(row["timestamp"]))
        except ValueError as exc:
            raise HydrationClockError(
                f"{source}:{lineno} timestamp {row['timestamp']!r} is not ISO 8601: {exc}"
            ) from exc
        if stamp.tzinfo is None:
            raise HydrationClockError(
                f"{source}:{lineno} timestamp {row['timestamp']!r} is naive. `recorded_at` "
                "is the fact-time authority for bitemporal edges and must be tz-aware; a "
                "naive value would be interpreted differently by different readers."
            )
        stamp = stamp.astimezone(UTC)

        try:
            key = (str(row["session_id"]), int(row["turn_index"]))
        except (TypeError, ValueError) as exc:
            raise HydrationClockError(
                f"{source}:{lineno} turn_index {row['turn_index']!r} is not an integer."
            ) from exc

        if key in timestamps:
            raise HydrationClockError(
                f"{source}:{lineno} duplicates key session={key[0]!r} turn_index={key[1]}. "
                "Two authored timestamps for one turn means the corpus cannot define a "
                "timeline; the last-writer-wins alternative would pick one silently."
            )
        timestamps[key] = stamp

    if not timestamps:
        raise HydrationClockError(
            f"{source} yielded zero turns. An empty clock raises on EVERY lookup, which "
            "would read as a hydration failure rather than a corpus problem -- refusing "
            "here so the cause is named once, at load."
        )

    logger.info(
        "Hydration clock loaded: %d turns from %s (%s .. %s)",
        len(timestamps),
        source,
        min(timestamps.values()).isoformat(),
        max(timestamps.values()).isoformat(),
    )
    return HydrationClock(timestamps=timestamps, source_path=str(source))
