"""Build the golden log artifact and materialize it into an ISOLATED store.

Two stages, deliberately separate:

1. `build_golden_turns` + `render_artifact` produce `data/golden-log/golden-log.jsonl`
   from the authored schedule plus the gold corpus. Pure, deterministic, byte-identical on
   re-run -- which a SQLite file could never be, so the checked-in artifact is the thing
   idempotence is claimed about.
2. `materialize_isolated` creates a fresh `EventStore` + `ExtractionCache` under a caller
   supplied directory and loads the artifact into them via `append_turn` and `put`.

The golden log is NEVER written to the live event store. `assert_isolated_root` makes that
mechanical rather than documentary: the materializer creates the store files itself, and
refuses a root under any live data directory.

Version stamps are read from `backend.knowledge.version_stamps` (ontology, extraction) and
`KnowledgeConfig` (model hash) at materialize time. There is deliberately no parameter for
them and no literal here: a generator that can state its own stamp triple reintroduces, by a
new door, exactly the drift that collapsing the four authorities removed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from backend.errors import MistError
from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.extraction_cache import ExtractionCache
from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION

from .translate import GOLD_UTTERANCE_FIELD, load_gold_corpus, translate_gold_record

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_LOG_DIR = REPO_ROOT / "data" / "golden-log"
SCHEDULE_PATH = GOLDEN_LOG_DIR / "turn-schedule.yaml"
ARTIFACT_PATH = GOLDEN_LOG_DIR / "golden-log.jsonl"

# Event ids are derived from the schedule id (which for a gold turn IS the gold tag), never
# generated, so the cache key is stable across regeneration.
EVENT_ID_PREFIX = "golden-"

# The golden log models the EXTRACTION side of a turn. No assistant replies are authored,
# and none are read by the replay -- `LogRegenerator` reads event_id, session_id and
# timestamp only. Stating that here beats inventing 87 plausible-looking replies.
SYSTEM_RESPONSE = "[golden log fixture: no system response authored]"

INPUT_MODALITY = "text"
# R1.4 Task 3's provenance discriminator. The golden log is fixture traffic, so it is
# never counted as genuine usage by anything that filters on origin.
SESSION_ORIGIN = "test"


class GoldenLogError(MistError):
    """Raised when the golden log cannot be built or materialized."""


@dataclass(frozen=True, slots=True)
class GoldenTurn:
    """One authored turn: the event-store row plus its extraction-cache payload."""

    event_id: str
    session_id: str
    timestamp: str
    turn_index: int
    utterance: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    def to_artifact_row(self) -> dict[str, Any]:
        """Serializable form written to `golden-log.jsonl`."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "turn_index": self.turn_index,
            "utterance": self.utterance,
            "entities": self.entities,
            "relationships": self.relationships,
        }

    @classmethod
    def from_artifact_row(cls, row: dict[str, Any]) -> GoldenTurn:
        """Rebuild a turn from an artifact row."""
        return cls(
            event_id=row["event_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            turn_index=row["turn_index"],
            utterance=row["utterance"],
            entities=row["entities"],
            relationships=row["relationships"],
        )


@dataclass(frozen=True, slots=True)
class MaterializedGoldenLog:
    """An isolated event store + extraction cache holding the golden log."""

    event_store: EventStore
    extraction_cache: ExtractionCache
    epoch: dict[str, Any]
    turn_count: int


def session_id_for(timestamp: str) -> str:
    """Derive the session id for an authored timestamp.

    One session per calendar month. Deterministic by rule rather than authored per turn,
    so the schedule stays about the content that matters (the gaps).
    """
    return f"golden-{timestamp[:7]}"


def event_id_for(schedule_id: str) -> str:
    """Derive the event id for a schedule entry. Never generated."""
    return f"{EVENT_ID_PREFIX}{schedule_id}"


def load_schedule(path: Path = SCHEDULE_PATH) -> dict[str, Any]:
    """Load and structurally validate the authored turn schedule.

    Raises:
        GoldenLogError: On a missing key, a duplicate id, a duplicate timestamp, or turns
            that are not in timestamp order (rowid order must match timestamp order --
            the replay reads rowid order and stamps `recorded_at` from the timestamp).
    """
    schedule = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("gold_corpus", "log_end", "turns"):
        if key not in schedule:
            raise GoldenLogError(f"{path}: schedule is missing required key {key!r}")

    turns = schedule["turns"]
    if not turns:
        raise GoldenLogError(f"{path}: schedule has no turns; a log that replays nothing")

    seen_ids: set[str] = set()
    previous: datetime | None = None
    for entry in turns:
        entry_id = entry.get("id")
        if not entry_id:
            raise GoldenLogError(f"{path}: a turn has no id; event ids are derived from it")
        if entry_id in seen_ids:
            raise GoldenLogError(f"{path}: duplicate turn id {entry_id!r}")
        seen_ids.add(entry_id)

        at = datetime.fromisoformat(entry["at"])
        if previous is not None and at <= previous:
            raise GoldenLogError(
                f"{path}: turn {entry_id!r} at {entry['at']} is not after the previous turn; "
                "turns must be listed in strictly increasing timestamp order"
            )
        previous = at

    return schedule


def build_golden_turns(
    *, schedule_path: Path = SCHEDULE_PATH, repo_root: Path = REPO_ROOT
) -> list[GoldenTurn]:
    """Build every turn of the golden log from the schedule plus the gold corpus.

    A schedule entry with a `record` is authored inline (in GOLD shape, so it goes through
    the same translation as the corpus); an entry without one must resolve to a gold tag.

    Raises:
        GoldenLogError: When a schedule entry references a tag the gold corpus does not
            contain, or when the schedule yields no turns.
    """
    schedule = load_schedule(schedule_path)
    gold = load_gold_corpus(repo_root / schedule["gold_corpus"])

    turns: list[GoldenTurn] = []
    turn_index_by_session: dict[str, int] = {}
    for entry in schedule["turns"]:
        entry_id = entry["id"]
        timestamp = entry["at"]

        if "record" in entry:
            record = dict(entry["record"])
            record.setdefault("tag", entry_id)
            utterance = entry.get("utterance", "")
        else:
            record = gold.get(entry_id)
            if record is None:
                raise GoldenLogError(
                    f"schedule entry {entry_id!r} has no inline record and no matching tag "
                    f"in {schedule['gold_corpus']}"
                )
            utterance = entry.get("utterance") or record.get(GOLD_UTTERANCE_FIELD, "")

        entities, relationships = translate_gold_record(record)
        session_id = session_id_for(timestamp)
        turn_index = turn_index_by_session.get(session_id, 0)
        turn_index_by_session[session_id] = turn_index + 1

        turns.append(
            GoldenTurn(
                event_id=event_id_for(entry_id),
                session_id=session_id,
                timestamp=timestamp,
                turn_index=turn_index,
                utterance=utterance,
                entities=entities,
                relationships=relationships,
            )
        )

    if not turns:
        raise GoldenLogError("golden log built zero turns; a log that replays nothing")
    return turns


def render_artifact(turns: list[GoldenTurn]) -> str:
    """Render turns as deterministic JSONL. Same input => byte-identical output."""
    lines = [
        json.dumps(turn.to_artifact_row(), sort_keys=True, separators=(",", ":")) for turn in turns
    ]
    return "\n".join(lines) + "\n"


def write_artifact(turns: list[GoldenTurn], path: Path = ARTIFACT_PATH) -> None:
    """Write the artifact with LF endings so re-running is byte-identical on any platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_artifact(turns), encoding="utf-8", newline="\n")


def load_artifact(path: Path = ARTIFACT_PATH) -> list[GoldenTurn]:
    """Load the checked-in artifact back into turns."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise GoldenLogError(f"{path}: artifact holds zero turns; a log that replays nothing")
    return [GoldenTurn.from_artifact_row(row) for row in rows]


def _live_data_roots() -> list[Path]:
    """Directories that hold live state and must never receive a golden-log store."""
    candidates = [
        REPO_ROOT / "data",
        Path("/app/data"),
        Path.home() / ".mist",
        REPO_ROOT / "mist-memory",
    ]
    return [c.resolve() for c in candidates if c.exists()]


def assert_isolated_root(root: Path) -> None:
    """Refuse a root that is, or sits under, a live data directory.

    The golden log is a fixture. R1.4 corrected the projection identity precisely so seed
    and log stay separate channels, and a live seed run stripped the production graph on
    2026-07-31 -- so "never written to live" is enforced here rather than documented.

    Raises:
        GoldenLogError: When `root` resolves into a live data directory.
    """
    resolved = root.resolve()
    for live in _live_data_roots():
        if resolved == live or live in resolved.parents:
            raise GoldenLogError(
                f"refusing to materialize the golden log at {resolved}: it is under the live "
                f"data directory {live}. The golden log is a repo fixture and must be "
                "materialized into an isolated store (pytest tmp_path or equivalent)."
            )


def materialize_isolated(
    turns: list[GoldenTurn], *, root: Path, activated_at: str | None = None
) -> MaterializedGoldenLog:
    """Create an isolated event store + cache under `root` and load the golden log.

    Writes one session row per distinct session id, one turn row per turn, and one
    `ExtractionCache` entry per turn keyed on the stamp triple the epoch carries -- so a
    subsequent `LogRegenerator.rebuild` against this epoch has 100% cache coverage.

    The epoch is appended to THIS store, never read from the live ledger. Live `epoch_id 1`
    captured a pre-collapse `extraction_version` and `ensure_initial_epoch` is idempotent so
    it will not self-correct; a fixture must not depend on that row either way.

    Args:
        turns: The golden log. Must be non-empty.
        root: Directory to create `events.db` and `extraction-cache.db` under.
        activated_at: Epoch activation timestamp. Defaults to the first turn's timestamp so
            no wall-clock read enters a deterministic fixture.

    Returns:
        MaterializedGoldenLog with the store, the cache, the epoch row, and the turn count.

    Raises:
        GoldenLogError: On zero turns, a non-isolated root, or a cache that did not end up
            covering every turn.
    """
    if not turns:
        raise GoldenLogError(
            "refusing to materialize zero turns: a golden log that replays nothing would "
            "satisfy every downstream assertion vacuously"
        )
    assert_isolated_root(root)
    root.mkdir(parents=True, exist_ok=True)

    event_store = EventStore(str(root / "events.db"))
    event_store.initialize()

    # Stamps come from the single authority. No parameter, no literal: a generator that can
    # state its own triple reintroduces the drift the collapse removed.
    model_hash = KnowledgeConfig.from_env().model_hash
    epoch_id = event_store.append_epoch(
        ontology_version=ONTOLOGY_VERSION,
        extraction_version=EXTRACTION_VERSION,
        model_hash=model_hash,
        activated_at=activated_at or turns[0].timestamp,
    )
    epoch = event_store.get_current_epoch()
    if epoch is None or int(epoch["epoch_id"]) != epoch_id:
        raise GoldenLogError("epoch append did not produce a readable current epoch")

    for session_id in dict.fromkeys(turn.session_id for turn in turns):
        event_store.start_session(session_id, input_modality=INPUT_MODALITY, origin=SESSION_ORIGIN)

    extraction_cache = ExtractionCache(str(root / "extraction-cache.db"))
    extraction_cache.initialize()

    for turn in turns:
        event = ConversationTurnEvent(
            session_id=turn.session_id,
            turn_index=turn.turn_index,
            timestamp=datetime.fromisoformat(turn.timestamp),
            user_utterance=turn.utterance,
            system_response=SYSTEM_RESPONSE,
            event_id=turn.event_id,
        )
        event_store.append_turn(event)
        extraction_cache.put(
            turn.event_id,
            epoch["ontology_version"],
            epoch["extraction_version"],
            epoch["model_hash"],
            entities=turn.entities,
            relationships=turn.relationships,
            created_at=turn.timestamp,
        )

    uncached = [
        turn.event_id
        for turn in turns
        if extraction_cache.get(
            turn.event_id,
            epoch["ontology_version"],
            epoch["extraction_version"],
            epoch["model_hash"],
        )
        is None
    ]
    if uncached:
        raise GoldenLogError(
            f"{len(uncached)} of {len(turns)} turns are uncached after materialize; a "
            f"rebuild would raise ColdCacheError. First: {uncached[:3]}"
        )

    return MaterializedGoldenLog(
        event_store=event_store,
        extraction_cache=extraction_cache,
        epoch=epoch,
        turn_count=len(turns),
    )


def main(argv: list[str] | None = None) -> int:
    """Regenerate `data/golden-log/golden-log.jsonl` from the authored schedule."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schedule", type=Path, default=SCHEDULE_PATH)
    parser.add_argument("--output", type=Path, default=ARTIFACT_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the artifact on disk differs from a fresh render.",
    )
    args = parser.parse_args(argv)

    turns = build_golden_turns(schedule_path=args.schedule)
    rendered = render_artifact(turns)

    if args.check:
        on_disk = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if on_disk != rendered:
            print(f"{args.output} is stale: re-run without --check to regenerate.")
            return 1
        print(f"{args.output} is current ({len(turns)} turns).")
        return 0

    write_artifact(turns, args.output)
    print(f"Wrote {len(turns)} turns to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
