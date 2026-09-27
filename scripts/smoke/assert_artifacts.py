"""Assert the four artifacts the smoke conversation should have produced.

Each assertion prints PASS, FAIL or INCONCLUSIVE with the discriminating
evidence inline. The distinction between FAIL and INCONCLUSIVE is the point of
this script: a FAIL says the pipeline did not do something it was exercised to
do, and an INCONCLUSIVE says it was never exercised, or that the run cannot say
which. Reporting the second as the first sends people to debug working code.

THE FOUR
--------
A1  Five turn rows and one session row in `smoke-state/event_store.db`.
A2  At least one `:__Entity__` with an `EXTRACTED_FROM` edge to this session's
    `ConversationContext` in the smoke Neo4j.
A3  A session note under `smoke-state/vault/sessions/`.
A4  A `self_reflection` row in `curation_job_runs` with `examined > 0`.

A2 HAS A TRAP, AND IT IS THE MOST IMPORTANT THING IN THIS FILE
---------------------------------------------------------------
There is NO success log line for a conversational entity write.
`backend/knowledge/curation/graph_writer.py:198` gates its only log on
`if source_metadata is not None and result.document_provenance_edges > 0` --
the DOCUMENT provenance branch. The conversational branch is the `else` at
`graph_writer.py:194-196`, which increments `result.provenance_edges_created`
and logs nothing. So do not grep the backend log for an extraction success
message: there is not one to find, and its absence is not evidence.

The Cypher count is the only positive evidence. `:__Entity__` carries no
session_id property either -- `graph_writer.py:243-274` sets `id`,
`entity_type`, `display_name`, `provenance` and the rest, and no session key --
so the join goes through the `ConversationContext` anchor that
`graph_writer.py:210-216` MERGEs on `conversation_id`.

WHERE THE BACKEND LOG LIVES, AND WHY YOU MUST CAPTURE IT EARLY
---------------------------------------------------------------
Two streams, with different levels, and the smoke stack keeps neither:

- Console, INFO and above (`backend/server.py:72`
  `logging.basicConfig(level=logging.INFO, ...)`). Reachable with
  `docker logs mist-backend-smoke`. Carries the `Extraction skipped (` lines
  (`backend/knowledge/extraction/pipeline.py:642,650,676,687`, all
  `logger.info`) and both scheduler lines
  (`backend/knowledge/curation/scheduler.py:265-272` and `:277-280`).
- File, DEBUG and above (`backend/server.py:75-80`, a `FileHandler` on
  `/app/logs/mist-backend.log` with `setLevel(logging.DEBUG)`). Reachable only
  with `docker exec mist-backend-smoke cat /app/logs/mist-backend.log`. Carries
  the two A3 discriminators, both `logger.debug`:
  `Session note written for %s at %s` (`conversation_handler.py:1970`) and
  `Session %s below synthesis threshold` (`:1962`).

`docker-compose.live-path-smoke.yml:324-330` deliberately bind-mounts no
`/app/logs`, so the file log lives in the container's writable layer and dies
with the container. The Phase 2 `--force-recreate` replaces that container --
so the Phase 1 log, which is where A1, A2 and A3's evidence is, must be
captured to the host BEFORE Phase 2. RUNBOOK.md has the two commands. Pass the
captured files here with `--backend-log`.

USAGE
-----
From the repository root:

    python -m scripts.smoke.assert_artifacts --snapshot-vault <dir>/vault-before.json \
        --smoke-state ./smoke-state

    python -m scripts.smoke.assert_artifacts \
        --session-id <uuid from the driver> \
        --smoke-state ./smoke-state \
        --transcript <dir>/transcript.jsonl \
        --vault-before <dir>/vault-before.json \
        --backend-log <dir>/phase1-console.log \
        --backend-log <dir>/phase1-debug.log

EXIT CODES
----------
    0  no FAIL (all PASS, or a mix of PASS and INCONCLUSIVE)
    1  at least one FAIL
    2  the invocation itself was wrong (bad session id, missing smoke-state)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# scripts/smoke/assert_artifacts.py -> scripts/smoke -> scripts -> repo root.
# Inserted because `python scripts/smoke/assert_artifacts.py` puts scripts/smoke
# -- not the repository root -- at sys.path[0], which would make the
# `scripts.smoke.baseline` import below fail. RUNBOOK.md uses the `-m` form,
# which does not need this; the insert keeps the direct-path form working rather
# than failing in a way that reads like a missing file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke.baseline import (  # noqa: E402 -- must follow the sys.path insert
    open_readonly,
    parse_cypher_plain,
    run_command,
)

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BAD_INVOCATION = 2

EXPECTED_TURNS = 5

#: Session ids are minted with `str(uuid.uuid4())` at `backend/server.py:787`.
#: Validated against this before being interpolated into Cypher: cypher-shell is
#: invoked without a shell, but the id still becomes part of a query string, and
#: a value restricted to hex digits and hyphens cannot close a quote.
SESSION_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                           r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")

#: `backend/vault/writer.py:538` renders `<root>/sessions/<date>-<slug>.md`,
#: with the slug validated as lowercase kebab-case at `writer.py:536`.
SESSION_NOTE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")

#: The four skip gates, all `logger.info`. `too short` is
#: `backend/knowledge/extraction/pipeline.py:642`, `rate-limited` is `:650`,
#: `significance ... < ...` is `:676-681` (the only one that prints numbers),
#: `duplicate` is `:687`.
EXTRACTION_SKIP_RE = re.compile(r"Extraction skipped \((?P<reason>[^)]*)\) for '(?P<utterance>[^']*)'")

SKIP_GATE_CITATIONS = {
    "too short": "backend/knowledge/extraction/pipeline.py:640-643 (fewer than 3 words)",
    "rate-limited": "backend/knowledge/extraction/pipeline.py:648-651 (rate limiter)",
    "duplicate": "backend/knowledge/extraction/pipeline.py:685-688 (input dedup)",
    "significance": "backend/knowledge/extraction/pipeline.py:674-682 (score below threshold)",
}

SCHEDULER_STARTED = "Curation scheduler started with"
SCHEDULER_NOT_STARTED = "Curation scheduler NOT started:"
NOTE_WRITTEN_MARKER = "Session note written for"
BELOW_THRESHOLD_MARKER = "below synthesis threshold"
NO_TURNS_MARKER = "but no event-store turns were recorded"


class InvocationError(ValueError):
    """The arguments given to this script cannot be acted on."""


@dataclass(frozen=True, slots=True)
class Verdict:
    """One assertion's result."""

    assertion: str
    title: str
    status: str
    headline: str
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TranscriptFacts:
    """What the driver transcript establishes about the conversation.

    Attributes:
        present: Whether a readable transcript with a summary record was found.
            False makes A1 and A3 INCONCLUSIVE rather than FAIL, because
            without it neither "the turns completed" nor "the client
            disconnected cleanly" can be established.
        terminal_counts: Terminal frames by type. `stream_complete` is the one
            A1 turns on.
    """

    present: bool
    schema_version: int | None
    session_id: str | None
    disconnected_at: str | None
    terminal_counts: dict[str, int]
    turns_planned: int | None
    aborted: str | None
    closed: bool
    error: str | None


@dataclass(frozen=True, slots=True)
class SkipRecord:
    """One `Extraction skipped (...)` line, with its gate identified."""

    reason: str
    gate: str
    citation: str
    utterance: str


# ---------------------------------------------------------------------------
# Readers -- pure parsers, unit-tested against recorded fixtures
# ---------------------------------------------------------------------------


def parse_transcript(text: str) -> TranscriptFacts:
    """Read the driver's JSONL transcript.

    Prefers the `summary` record `drive_turns.py` writes last, and falls back to
    counting `turn_outcome` records when the run was killed before the summary
    was written. The fallback exists because a transcript that stops mid-run
    still establishes how many turns completed, which is exactly the fact A1
    needs.

    Args:
        text: The transcript file's contents.

    Returns:
        The facts, with `present=False` and `error` set when nothing usable
        was found.
    """
    summary: dict[str, Any] | None = None
    outcomes: list[dict[str, Any]] = []
    session_id: str | None = None
    schema_version: int | None = None
    closed = False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("record")
        if kind == "header":
            schema_version = record.get("schema_version")
        elif kind == "session":
            session_id = record.get("session_id")
        elif kind == "turn_outcome":
            outcomes.append(record)
        elif kind == "note" and record.get("event") == "closed":
            # `closed` means CLEANLY closed. drive_turns.py records the note
            # even when the close itself raised, carrying `close_error`; A3
            # reads this field as evidence that the server's disconnect
            # `finally` (backend/server.py:1008-1017) was reached in good
            # order, which a failed close does not establish.
            closed = not record.get("close_error")
        elif kind == "summary":
            summary = record

    if summary is None and not outcomes:
        return TranscriptFacts(
            present=False,
            schema_version=schema_version,
            session_id=session_id,
            disconnected_at=None,
            terminal_counts={},
            turns_planned=None,
            aborted=None,
            closed=closed,
            error="no summary and no turn_outcome records; the transcript is empty "
            "or is not one drive_turns.py wrote",
        )

    if summary is not None:
        counts = summary.get("terminal_counts") or {}
        return TranscriptFacts(
            present=True,
            schema_version=summary.get("schema_version", schema_version),
            session_id=summary.get("session_id") or session_id,
            disconnected_at=summary.get("disconnected_at"),
            terminal_counts={k: int(v) for k, v in counts.items() if isinstance(v, int)},
            turns_planned=summary.get("turns_planned"),
            aborted=summary.get("aborted"),
            closed=closed,
            error=None,
        )

    fallback_counts: dict[str, int] = {}
    for outcome in outcomes:
        key = str(outcome.get("terminal_type"))
        fallback_counts[key] = fallback_counts.get(key, 0) + 1
    fallback_counts["total"] = len(outcomes)
    return TranscriptFacts(
        present=True,
        schema_version=schema_version,
        session_id=session_id,
        disconnected_at=None,
        terminal_counts=fallback_counts,
        turns_planned=None,
        aborted="transcript has no summary record; the driver did not finish",
        closed=closed,
        error=None,
    )


def parse_extraction_skips(log_text: str) -> list[SkipRecord]:
    """Extract every `Extraction skipped (...)` line, identifying its gate.

    A per-turn skip is normal behaviour, not a defect: the significance gate at
    `backend/knowledge/extraction/pipeline.py:674` is a scored decision and
    legitimately declines some utterances. That is why A2 asserts at least ONE
    turn produced entities rather than all five.

    Args:
        log_text: Concatenated backend log text.

    Returns:
        One record per matching line, in file order.
    """
    records: list[SkipRecord] = []
    for match in EXTRACTION_SKIP_RE.finditer(log_text):
        reason = match.group("reason")
        gate = next((name for name in SKIP_GATE_CITATIONS if reason.startswith(name)), "unknown")
        records.append(
            SkipRecord(
                reason=reason,
                gate=gate,
                citation=SKIP_GATE_CITATIONS.get(
                    gate, "no citation: this skip reason is not one of the four known gates"
                ),
                utterance=match.group("utterance"),
            )
        )
    return records


def find_lines(log_text: str, marker: str) -> list[str]:
    """Return every log line containing `marker`, stripped.

    Args:
        log_text: Concatenated backend log text.
        marker: Substring to match.

    Returns:
        Matching lines, in file order.
    """
    return [line.strip() for line in log_text.splitlines() if marker in line]


# ---------------------------------------------------------------------------
# Adjudicators -- pure functions over already-gathered evidence
# ---------------------------------------------------------------------------


def adjudicate_a1(
    turn_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
    transcript: TranscriptFacts,
    session_id: str,
    *,
    expected_turns: int = EXPECTED_TURNS,
) -> Verdict:
    """A1: turn rows and the session row.

    Deliberately makes no assertion about `ended_at`. That column is written
    only by `EventStore.end_session` (`backend/event_store/store.py:156`), whose
    sole caller is `ConversationHandler.clear_session`
    (`backend/chat/conversation_handler.py:2874`), which has zero production
    callers -- documented at `backend/server.py:262-265`. The WebSocket
    disconnect path calls `end_session` on the HANDLER
    (`backend/server.py:1015`), which writes the vault note and never touches
    `ended_at`. So `ended_at` stays NULL after a perfectly normal disconnect,
    and asserting on it would fail for a reason unrelated to the pipeline.

    Args:
        turn_rows: Rows from `conversation_turn_events`, all sessions.
        session_rows: Rows from `conversation_sessions`, all sessions.
        transcript: What the driver observed.
        session_id: The session under test.
        expected_turns: How many turns were planned.

    Returns:
        The verdict.
    """
    mine = [row for row in turn_rows if row.get("session_id") == session_id]
    my_sessions = [row for row in session_rows if row.get("session_id") == session_id]
    completes = transcript.terminal_counts.get("stream_complete", 0)

    evidence = [
        f"transcript stream_complete frames: {completes} "
        f"(all terminals: {transcript.terminal_counts or 'none recorded'})",
        f"conversation_turn_events rows for this session: {len(mine)} "
        f"(whole table: {len(turn_rows)})",
        f"conversation_sessions rows for this session: {len(my_sessions)} "
        f"(whole table: {len(session_rows)})",
    ]
    for row in mine:
        evidence.append(
            f"  turn_index={row.get('turn_index')} "
            f"user_utterance[:60]={str(row.get('user_utterance'))[:60]!r}"
        )
    for row in my_sessions:
        evidence.append(
            f"  session origin={row.get('origin')!r} "
            f"input_modality={row.get('input_modality')!r} "
            f"turn_count={row.get('turn_count')!r} "
            f"ended_at={row.get('ended_at')!r} (ended_at is NOT asserted; see docstring)"
        )

    if not transcript.present:
        return Verdict(
            "A1", "turn rows", INCONCLUSIVE,
            "no usable driver transcript, so it cannot be established whether the "
            "turns ever completed",
            tuple(evidence + [f"transcript problem: {transcript.error}"]),
        )

    if completes < expected_turns:
        return Verdict(
            "A1", "turn rows", INCONCLUSIVE,
            f"only {completes} of {expected_turns} turns reached stream_complete, so "
            "A1 was never fully exercised. This is not a FAIL: a turn that did not "
            "complete never reached _record_turn_event "
            "(backend/chat/conversation_handler.py:2279) in the first place.",
            tuple(
                evidence
                + [
                    "The transcript is the discriminator here. Read its turn_outcome "
                    "records for the terminal type of each turn before looking at the "
                    "memory pipeline at all.",
                    f"driver reported abort: {transcript.aborted!r}",
                ]
            ),
        )

    if not mine:
        return Verdict(
            "A1", "turn rows", FAIL,
            f"the driver recorded {completes} stream_complete frames but "
            "conversation_turn_events holds no row for this session. The turns ran "
            "and were not written.",
            tuple(
                evidence
                + [
                    "_record_turn_event (backend/chat/conversation_handler.py:2279) has "
                    "exactly ONE gate: `if self.event_store is None: return None, None` "
                    "at :2306-2307. Start there -- check whether the smoke backend built "
                    "an event store at all, and whether ENABLE_KNOWLEDGE_INTEGRATION "
                    "was true (docker-compose.live-path-smoke.yml:256; with knowledge "
                    "off, backend/voice_models/model_manager.py:501-530 takes a path "
                    "that never calls _record_turn_event).",
                ]
            ),
        )

    problems: list[str] = []
    if len(mine) != expected_turns:
        problems.append(f"expected {expected_turns} turn rows, found {len(mine)}")
    # Filtered to ints before sorting: a NULL turn_index would make `sorted`
    # raise on comparing None with int. Dropping it still fails the comparison
    # below, which is the correct outcome, and it fails as a reported mismatch
    # rather than as a traceback.
    indices = sorted(
        row.get("turn_index") for row in mine if isinstance(row.get("turn_index"), int)
    )
    if indices != list(range(expected_turns)):
        problems.append(
            f"turn_index values are {indices}, expected {list(range(expected_turns))} "
            "(backend/chat/conversation_handler.py:2320-2322 starts at 0 and takes each "
            "later index from the session's turn_count)"
        )
    if len(my_sessions) != 1:
        problems.append(f"expected exactly 1 session row, found {len(my_sessions)}")
    else:
        row = my_sessions[0]
        if row.get("origin") != "real":
            problems.append(
                f"origin is {row.get('origin')!r}, expected 'real' "
                "(docker-compose.live-path-smoke.yml:261 sets MIST_SESSION_ORIGIN=real)"
            )
        if row.get("input_modality") != "text":
            problems.append(
                f"input_modality is {row.get('input_modality')!r}, expected 'text' "
                "(backend/chat/conversation_handler.py:2317-2319 hardcodes it)"
            )
        if row.get("turn_count") != expected_turns:
            problems.append(
                f"turn_count is {row.get('turn_count')!r}, expected {expected_turns}"
            )

    if problems:
        return Verdict(
            "A1", "turn rows", FAIL,
            "turn rows exist but do not match the expected shape",
            tuple(evidence + [f"mismatch: {problem}" for problem in problems]),
        )

    return Verdict(
        "A1", "turn rows", PASS,
        f"{len(mine)} turn rows and 1 session row, origin='real', "
        f"input_modality='text', turn_count={expected_turns}",
        tuple(evidence),
    )


def adjudicate_a2(
    entity_count: int | None,
    edge_count: int | None,
    entity_rows: list[dict[str, str]],
    skips: list[SkipRecord],
    a1_status: str,
    log_available: bool,
    cypher_error: str | None,
) -> Verdict:
    """A2: at least one entity anchored to this session's ConversationContext.

    Args:
        entity_count: `count(DISTINCT e)`, or None when the query failed.
        edge_count: `count(r)`, or None when the query failed.
        entity_rows: A sample of the entities, for evidence.
        skips: Parsed `Extraction skipped (` lines.
        a1_status: A1's status, because "extraction never ran" is only a FAIL
            when the turns were recorded in the first place.
        log_available: Whether any backend log text was readable. Without it,
            "no skip line" means "nobody looked", not "no skip happened".
        cypher_error: The Cypher failure, if the query did not run.

    Returns:
        The verdict.
    """
    evidence = [
        "There is NO success log line for a conversational entity write. "
        "backend/knowledge/curation/graph_writer.py:198 gates its only log on "
        "`source_metadata is not None and result.document_provenance_edges > 0`, "
        "the DOCUMENT branch. The conversational branch (graph_writer.py:194-196) "
        "logs nothing. The Cypher count below is the only positive evidence.",
        f"entities: {entity_count}, EXTRACTED_FROM edges: {edge_count}",
    ]
    for row in entity_rows:
        evidence.append(
            f"  id={row.get('id')!r} type={row.get('entity_type')!r} "
            f"provenance={row.get('provenance')!r}"
        )
    for skip in skips:
        evidence.append(
            f"  skip gate={skip.gate!r} detail={skip.reason!r} -> {skip.citation}; "
            f"utterance={skip.utterance!r}"
        )

    if cypher_error is not None:
        return Verdict(
            "A2", "entity with EXTRACTED_FROM", INCONCLUSIVE,
            "the Cypher count did not run, so there is no positive evidence either way",
            tuple(evidence + [f"cypher error: {cypher_error}"]),
        )

    if entity_count and entity_count > 0:
        return Verdict(
            "A2", "entity with EXTRACTED_FROM", PASS,
            f"{entity_count} entity/entities anchored to this session's "
            f"ConversationContext via {edge_count} EXTRACTED_FROM edge(s)",
            tuple(
                evidence
                + [
                    f"{len(skips)} turn(s) were skipped by an extraction gate. That is "
                    "normal: the significance gate is a scored decision, so A2 requires "
                    "at least ONE turn to produce entities, not all five."
                ]
            ),
        )

    if skips:
        return Verdict(
            "A2", "entity with EXTRACTED_FROM", INCONCLUSIVE,
            f"no entities, and {len(skips)} utterance(s) were declined by an extraction "
            "gate. A per-turn skip is normal behaviour, not a defect -- but every turn "
            "being skipped means extraction ran and chose not to write.",
            tuple(
                evidence
                + [
                    "Read the gate names above. `significance` prints its score and the "
                    "threshold; the threshold default is 0.3 "
                    "(backend/knowledge/config.py:151), overridable by "
                    "SIGNIFICANCE_THRESHOLD (config.py:169)."
                ]
            ),
        )

    if not log_available:
        return Verdict(
            "A2", "entity with EXTRACTED_FROM", INCONCLUSIVE,
            "no entities, and no backend log was readable, so it cannot be told apart "
            "from every turn being declined by a gate",
            tuple(
                evidence
                + [
                    "Capture the Phase 1 log and re-run this assertion. If the smoke "
                    "backend was already recreated for Phase 2, that log is gone with "
                    "its container (docker-compose.live-path-smoke.yml:324-330 mounts "
                    "no /app/logs) and this stays INCONCLUSIVE for this run."
                ]
            ),
        )

    if a1_status == PASS:
        return Verdict(
            "A2", "entity with EXTRACTED_FROM", FAIL,
            "turns were recorded (A1 PASS), no entities were written, and the backend "
            "log carries no `Extraction skipped (` line. Extraction never ran.",
            tuple(
                evidence
                + [
                    "Extraction is spawned per turn at "
                    "backend/chat/conversation_handler.py:1662-1674, gated on "
                    "`if event_id:`. A1 passing means event ids existed, so start at "
                    "that spawn and at whether the extraction pipeline was constructed "
                    "at all.",
                    "end_session drains those tasks before writing the note "
                    "(conversation_handler.py:1920), so 'the background task had not "
                    "finished' is not available as an explanation after a clean close.",
                ]
            ),
        )

    return Verdict(
        "A2", "entity with EXTRACTED_FROM", INCONCLUSIVE,
        f"no entities and no skip lines, but A1 is {a1_status}: the turns were not "
        "established as recorded, so extraction had nothing to run on",
        tuple(evidence),
    )


def adjudicate_a3(
    before: set[str],
    after: dict[str, float],
    note_texts: dict[str, str],
    corpus_facts: list[str],
    transcript: TranscriptFacts,
    log_text: str,
    log_available: bool,
    session_id: str,
) -> Verdict:
    """A3: a session note appeared under `smoke-state/vault/sessions/`.

    Attribution between the two writers is done with the DEBUG file log, and
    falls back to mtime when that log is not available. The reasoning:

    - `ConversationHandler.end_session` logs `Session note written for %s at %s`
      at `backend/chat/conversation_handler.py:1970` on success. That is the
      only positive attribution available.
    - `SessionNoteCatchup` has NO success log line at any level: its write at
      `backend/vault/session_catchup.py:243-245` is followed by a bare `return`
      at `:250`, and only the failure path logs (`:247`). So catch-up cannot be
      confirmed directly; it is inferred when the note exists and the
      `end_session` line does not.
    - Both lines are `logger.debug`, so neither reaches `docker logs` -- the
      console handler is INFO (`backend/server.py:72`). They exist only in
      `/app/logs/mist-backend.log` (`backend/server.py:75-80`).

    Timing is a secondary signal, not proof: catch-up defers while a connection
    is open (`session_catchup.py:121`, wired to `active_connections` per
    `backend/server.py:260-272`) and then ticks every 300 seconds
    (`session_catchup.py:53`, the default `run_forever` takes at
    `backend/server.py:317`).

    Args:
        before: Session-note filenames present before the conversation.
        after: Session-note filenames present now, mapped to mtime.
        note_texts: New notes' contents, keyed by filename.
        corpus_facts: Strings from the corpus to grep for.
        transcript: What the driver observed.
        log_text: Concatenated backend log.
        log_available: Whether any log text was readable.
        session_id: The session under test, for the log line match.

    Returns:
        The verdict.
    """
    new_notes = sorted(name for name in after if name not in before and SESSION_NOTE_RE.match(name))
    written_lines = [line for line in find_lines(log_text, NOTE_WRITTEN_MARKER) if session_id in line]
    threshold_lines = [
        line for line in find_lines(log_text, BELOW_THRESHOLD_MARKER) if session_id in line
    ]
    no_turn_lines = [line for line in find_lines(log_text, NO_TURNS_MARKER) if session_id in line]

    evidence = [
        f"session-note files before: {sorted(before) or 'none'}",
        f"session-note files after : {sorted(after) or 'none'}",
        f"new notes: {new_notes or 'none'}",
        f"driver disconnected_at: {transcript.disconnected_at}",
        f"clean close recorded by driver: {transcript.closed}",
        f"`{NOTE_WRITTEN_MARKER}` lines for this session: {written_lines or 'none'}",
        f"`{BELOW_THRESHOLD_MARKER}` lines for this session: {threshold_lines or 'none'}",
    ]
    if no_turn_lines:
        evidence.append(f"`{NO_TURNS_MARKER}` lines: {no_turn_lines}")
    if not log_available:
        evidence.append(
            "NO backend log was readable. Both attribution lines are logger.debug "
            "(conversation_handler.py:1962,1970) and reach only "
            "/app/logs/mist-backend.log, never `docker logs` (console handler is INFO "
            "at backend/server.py:72). Attribution below is by mtime alone and is a "
            "likelihood, not a proof."
        )

    if new_notes:
        attribution = _attribute_note(
            new_notes, after, written_lines, transcript.disconnected_at, log_available
        )
        evidence.append(f"attribution: {attribution}")
        matched = sorted(
            {
                fact
                for fact in corpus_facts
                for text in note_texts.values()
                if fact.lower() in text.lower()
            }
        )
        evidence.append(f"corpus facts present in the new note(s): {matched or 'none'}")
        if not matched:
            return Verdict(
                "A3", "session note", INCONCLUSIVE,
                f"a note appeared ({', '.join(new_notes)}) but it names none of the "
                "corpus facts, so it cannot be shown to be this conversation's note",
                tuple(evidence + [f"looked for: {corpus_facts}"]),
            )
        return Verdict(
            "A3", "session note", PASS,
            f"{len(new_notes)} new session note ({', '.join(new_notes)}) naming "
            f"{len(matched)} corpus fact(s)",
            tuple(evidence),
        )

    if threshold_lines:
        return Verdict(
            "A3", "session note", INCONCLUSIVE,
            "no note, and the synthesizer returned None: end_session hit `continue` at "
            "backend/chat/conversation_handler.py:1961-1963. That is a legitimate 'not "
            "worth remembering' decision, not a pipeline failure.",
            tuple(
                evidence
                + [
                    "The minimum-turns floor is 1 "
                    "(backend/chat/session_synthesizer.py:19 `_MIN_TURNS_FOR_SYNTHESIS = 1`, "
                    "checked at :106), so with five recorded turns the floor is not what "
                    "declined it -- the synthesizer's own judgement is."
                ]
            ),
        )

    if no_turn_lines:
        return Verdict(
            "A3", "session note", INCONCLUSIVE,
            "no note, and end_session found no event-store turns for this session "
            "(backend/chat/conversation_handler.py:1952-1958). A3 depends on A1; fix "
            "that first.",
            tuple(evidence),
        )

    if not transcript.present or not transcript.closed:
        return Verdict(
            "A3", "session note", INCONCLUSIVE,
            "no note, and the driver did not record a clean close, so the disconnect "
            "hook cannot be shown to have run",
            tuple(
                evidence
                + [
                    "The note is written from the WebSocket `finally` at "
                    "backend/server.py:1008-1017. Without a recorded close there is no "
                    "evidence that block was reached."
                ]
            ),
        )

    if not log_available:
        return Verdict(
            "A3", "session note", INCONCLUSIVE,
            "no note and no readable backend log. The two discriminating lines are "
            "DEBUG-only, so their absence here carries no information.",
            tuple(evidence),
        )

    return Verdict(
        "A3", "session note", FAIL,
        "the driver closed cleanly, the backend log carries no 'below synthesis "
        "threshold' line for this session, and no note appeared",
        tuple(
            evidence
            + [
                "Path that should have produced it: backend/server.py:1015 calls "
                "ConversationHandler.end_session "
                "(backend/chat/conversation_handler.py:1885), which drains extraction "
                "tasks (:1920), synthesizes (:1960) and writes via "
                "VaultWriter.write_session_note (:1965). A write failure there is "
                "swallowed per Invariant 6 but logged as a WARNING at :1972 -- grep for "
                "'Session note write failed'.",
                "expected location: smoke-state/vault/sessions/YYYY-MM-DD-<slug>.md "
                "(backend/vault/writer.py:538), slug derived from the FIRST utterance "
                "(backend/chat/conversation_handler.py:2433-2434).",
            ]
        ),
    )


def _attribute_note(
    new_notes: list[str],
    mtimes: dict[str, float],
    written_lines: list[str],
    disconnected_at: str | None,
    log_available: bool,
) -> str:
    """Decide which writer produced the note, and say how confident that is."""
    if written_lines:
        return (
            "the disconnect hook (ConversationHandler.end_session). PROVEN: "
            "conversation_handler.py:1970 logged the write, and SessionNoteCatchup has "
            "no equivalent success line (session_catchup.py:243-250 returns without "
            "logging)."
        )
    if log_available:
        return (
            "startup catch-up (SessionNoteCatchup, backend/vault/session_catchup.py:113). "
            "INFERRED: the DEBUG log was readable and carries no "
            f"'{NOTE_WRITTEN_MARKER}' line for this session, which end_session would "
            "have emitted at conversation_handler.py:1970 had it written the note."
        )
    if disconnected_at is None:
        return "UNDETERMINED: no log and no disconnect timestamp to compare mtimes against."

    try:
        disconnect_epoch = datetime.fromisoformat(disconnected_at).timestamp()
    except ValueError:
        return f"UNDETERMINED: disconnected_at {disconnected_at!r} is not an ISO timestamp."

    gaps = {name: mtimes[name] - disconnect_epoch for name in new_notes if name in mtimes}
    if not gaps:
        return "UNDETERMINED: no mtime was read for the new note(s)."
    closest = min(gaps.values(), key=abs)
    verdict = (
        "the disconnect hook" if abs(closest) <= 120 else "startup catch-up, or something later"
    )
    return (
        f"{verdict}. LIKELIHOOD ONLY, not proof: the note's mtime is {closest:+.1f}s from "
        "the driver's disconnect. Catch-up defers while a connection is open "
        "(session_catchup.py:121) and then ticks every 300s (session_catchup.py:53), so a "
        "note within seconds of disconnect is most likely the hook -- but the DEBUG log "
        "that would settle it was not available."
    )


def adjudicate_a4(
    rows: list[dict[str, Any]],
    log_text: str,
    log_available: bool,
    *,
    expected_examined: int = EXPECTED_TURNS,
) -> Verdict:
    """A4: a scheduled `self_reflection` run that examined something.

    Args:
        rows: `curation_job_runs` rows for `self_reflection`, newest first.
        log_text: Concatenated backend log from the PHASE 2 container.
        log_available: Whether any log text was readable.
        expected_examined: How many turns the job should have seen.

    Returns:
        The verdict.
    """
    started = find_lines(log_text, SCHEDULER_STARTED)
    not_started = find_lines(log_text, SCHEDULER_NOT_STARTED)

    evidence = [f"self_reflection rows found: {len(rows)}"]
    for row in rows:
        evidence.append(
            f"  run_id={row.get('run_id')} trigger_source={row.get('trigger_source')!r} "
            f"started_at={row.get('started_at')!r} outcome={row.get('outcome')!r} "
            f"examined={row.get('examined')!r} produced={row.get('produced')!r}"
        )
        if row.get("error"):
            evidence.append(f"    error={row.get('error')!r}")
    evidence.append(f"`{SCHEDULER_STARTED}` lines: {started or 'none'}")
    evidence.append(f"`{SCHEDULER_NOT_STARTED}` lines: {not_started or 'none'}")

    if not rows:
        if not_started:
            return Verdict(
                "A4", "curation with examined > 0", INCONCLUSIVE,
                "no self_reflection row, and the backend logged that the scheduler did "
                "not start. The job never ran, so nothing about `examined` was tested.",
                tuple(
                    evidence
                    + [
                        "The two reasons are distinguishable in that line: "
                        "MIST_HYDRATION_ISOLATION set (scheduler.py:260-270) or "
                        "MIST_CURATION_SCHEDULER_ENABLED off (scheduler.py:271-273). "
                        "The compose file sets MIST_CURATION_SCHEDULER_ENABLED from "
                        "${MIST_SMOKE_SCHEDULER:-0} "
                        "(docker-compose.live-path-smoke.yml:321), so Phase 2 must be "
                        "run with MIST_SMOKE_SCHEDULER=1.",
                    ]
                ),
            )
        if started:
            return Verdict(
                "A4", "curation with examined > 0", INCONCLUSIVE,
                "the scheduler started but wrote no self_reflection row. Something "
                "between the loop and the ledger did not complete.",
                tuple(
                    evidence
                    + [
                        "'it has not fired yet' is NOT an available answer: "
                        "backend/knowledge/curation/scheduler.py:305-311 reads "
                        "`last_run.get(config.name, 0.0)`, which makes every enabled job "
                        "due on the loop's FIRST pass. The loop sleeps 60s between passes "
                        "(scheduler.py:326-327), so allow one minute after Phase 2 comes "
                        "up, then look again.",
                        "If the row is still absent, check whether the job raised: "
                        "_execute_and_record turns an exception into an `outcome='failed'` "
                        "row (scheduler.py:188-198), so a total absence means the ledger "
                        "write itself did not happen (scheduler.py:214 returns early when "
                        "the recorder is None).",
                    ]
                ),
            )
        return Verdict(
            "A4", "curation with examined > 0", INCONCLUSIVE,
            "no self_reflection row, and no backend log to say whether the scheduler "
            "started",
            tuple(
                evidence
                + [
                    "Discriminate with one grep of the Phase 2 smoke backend log: "
                    f"'{SCHEDULER_STARTED}' (scheduler.py:277-280) versus "
                    f"'{SCHEDULER_NOT_STARTED}' (scheduler.py:265-272). Both are "
                    "logger.info, so `docker logs mist-backend-smoke` is enough.",
                ]
                + ([] if log_available else ["no log text was readable at all"])
            ),
        )

    scheduled = [row for row in rows if row.get("trigger_source") == "scheduled"]
    if not scheduled:
        return Verdict(
            "A4", "curation with examined > 0", INCONCLUSIVE,
            "self_reflection rows exist but none has trigger_source='scheduled'. The "
            "experiment tests the SCHEDULER; a manual row came from `run_all_once` "
            "(backend/knowledge/curation/scheduler.py:158) and does not answer it.",
            tuple(evidence),
        )

    row = scheduled[0]
    outcome = row.get("outcome")
    examined = row.get("examined")

    if outcome != "completed":
        return Verdict(
            "A4", "curation with examined > 0", FAIL,
            f"the newest scheduled self_reflection run has outcome={outcome!r}",
            tuple(
                evidence
                + [
                    "backend/knowledge/curation/scheduler.py:188-198 writes "
                    "outcome='failed' with the exception text in `error` when the job "
                    "raised. That text is the next thing to read."
                ]
            ),
        )

    if not examined:
        return Verdict(
            "A4", "curation with examined > 0", FAIL,
            f"the run completed and examined={examined!r}. THIS IS THE BUG THE "
            "EXPERIMENT EXISTS TO FIND, and it is the live stack's exact symptom: a "
            "job that returns zeros WITHOUT LOOKING.",
            tuple(
                evidence
                + [
                    "`examined` is ReflectionResult.events_processed "
                    "(backend/knowledge/curation/run_record.py:83). That counter is "
                    "incremented per turn at "
                    "backend/knowledge/curation/self_reflection.py:89, after a `continue` "
                    "on an empty user_utterance at :72-73.",
                    "The input is `get_turns_since(now - 24h)` "
                    "(self_reflection.py:60-61), which is "
                    "`SELECT * FROM conversation_turn_events WHERE timestamp >= ?` "
                    "(backend/event_store/store.py:315-322). With A1 PASSing, five rows "
                    "exist -- so compare their `timestamp` values against the run's "
                    "`started_at` and check the 24-hour window and the timestamp format.",
                    "backend/event_store/schema.sql:95-96: examined = 0 means the job "
                    "looked at nothing. That is a different failure from producing "
                    "nothing, which is what the column exists to separate.",
                ]
            ),
        )

    headline = f"scheduled self_reflection run completed with examined={examined}"
    extra: list[str] = []
    if examined != expected_examined:
        extra.append(
            f"NOTE: examined={examined}, expected {expected_examined} (one per turn). "
            "The job counts every turn in the 24-hour window, not only this session's "
            "(self_reflection.py:60-61 has no session filter), so a higher number means "
            "the smoke store holds turns from an earlier run."
        )
    return Verdict("A4", "curation with examined > 0", PASS, headline, tuple(evidence + extra))


# ---------------------------------------------------------------------------
# Gatherers -- the I/O half
# ---------------------------------------------------------------------------


def read_backend_log(
    container: str | None, files: list[Path]
) -> tuple[str, list[str]]:
    """Assemble backend log text from captured files and, if possible, the container.

    Args:
        container: Container to pull from live, or None to skip.
        files: Host files the operator captured earlier.

    Returns:
        (concatenated text, one description line per source attempted).
    """
    chunks: list[str] = []
    sources: list[str] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            sources.append(f"{path}: UNREADABLE ({exc})")
            continue
        chunks.append(text)
        sources.append(f"{path}: {len(text.splitlines())} line(s)")

    if container:
        console = run_command(["docker", "logs", container])
        if console.ok:
            chunks.append(console.stdout)
            chunks.append(console.stderr)
            sources.append(
                f"docker logs {container}: "
                f"{len(console.stdout.splitlines()) + len(console.stderr.splitlines())} line(s), "
                "INFO and above (backend/server.py:72)"
            )
        else:
            sources.append(f"docker logs {container}: UNAVAILABLE ({console.stderr.strip()})")

        debug = run_command(["docker", "exec", container, "cat", "/app/logs/mist-backend.log"])
        if debug.ok:
            chunks.append(debug.stdout)
            sources.append(
                f"docker exec {container} cat /app/logs/mist-backend.log: "
                f"{len(debug.stdout.splitlines())} line(s), DEBUG and above "
                "(backend/server.py:75-80)"
            )
        else:
            sources.append(
                f"docker exec {container} cat /app/logs/mist-backend.log: UNAVAILABLE "
                f"({debug.stderr.strip()})"
            )

    return "\n".join(chunks), sources


def read_event_store(db_path: Path) -> dict[str, Any]:
    """Read the three tables A1 and A4 need out of the smoke event store.

    Args:
        db_path: `smoke-state/event_store.db`.

    Returns:
        `{"status": "ok", "turns": [...], "sessions": [...], "curation": [...]}`
        or `{"status": "unavailable", "error": ...}`.
    """
    if not db_path.exists():
        return {"status": "unavailable", "error": f"no such file: {db_path}"}
    conn = open_readonly(db_path)
    try:
        return {
            "status": "ok",
            "turns": _query(
                conn,
                "SELECT session_id, turn_index, substr(user_utterance, 1, 60) AS user_utterance "
                "FROM conversation_turn_events ORDER BY turn_index",
            ),
            "sessions": _query(
                conn,
                "SELECT session_id, origin, turn_count, input_modality, ended_at "
                "FROM conversation_sessions",
            ),
            "curation": _query(
                conn,
                "SELECT run_id, job_name, trigger_source, started_at, outcome, examined, "
                "produced, error FROM curation_job_runs WHERE job_name = 'self_reflection' "
                "ORDER BY started_at DESC LIMIT 5",
            ),
        }
    finally:
        conn.close()


def _query(conn: Any, sql: str) -> list[dict[str, Any]]:
    """Run one SELECT and return dict rows."""
    cursor = conn.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def query_smoke_graph(
    container: str, user: str, password: str, database: str, session_id: str
) -> dict[str, Any]:
    """Count entities anchored to this session, and sample them.

    The session id is interpolated into the query text rather than passed as a
    bound parameter, because reaching cypher-shell through `docker exec` makes
    parameter quoting the fragile part. `SESSION_ID_RE` has already restricted
    it to hex digits and hyphens by the time this runs, so it cannot close the
    quote it sits inside.

    Args:
        container: Smoke Neo4j container name.
        user: Neo4j username.
        password: Neo4j password.
        database: Neo4j database name.
        session_id: Validated session id.

    Returns:
        `{"entities": int, "edges": int, "rows": [...]}` or `{"error": ...}`.
    """
    count_query = (
        "MATCH (e:__Entity__)-[r:EXTRACTED_FROM]->"
        f"(ctx:ConversationContext {{conversation_id: '{session_id}'}}) "
        "RETURN count(DISTINCT e) AS entities, count(r) AS edges;"
    )
    sample_query = (
        "MATCH (e:__Entity__)-[:EXTRACTED_FROM]->"
        f"(ctx:ConversationContext {{conversation_id: '{session_id}'}}) "
        "RETURN e.id AS id, e.entity_type AS entity_type, e.provenance AS provenance "
        "ORDER BY id LIMIT 25;"
    )

    def _run(query: str) -> tuple[list[dict[str, str]], str | None]:
        result = run_command(
            [
                "docker", "exec", container, "cypher-shell",
                "-u", user, "-p", password, "-d", database,
                "--format", "plain", query,
            ]
        )
        if not result.ok:
            return [], result.stderr.strip() or result.stdout.strip() or "non-zero exit"
        return parse_cypher_plain(result.stdout), None

    counts, error = _run(count_query)
    if error is not None:
        return {"error": error}
    if not counts:
        return {"error": "the count query returned no rows"}
    try:
        entities = int(counts[0].get("entities", ""))
        edges = int(counts[0].get("edges", ""))
    except ValueError:
        return {"error": f"unparseable counts: {counts[0]!r}"}

    rows, sample_error = _run(sample_query)
    return {"entities": entities, "edges": edges, "rows": rows, "sample_error": sample_error}


def list_session_notes(vault_root: Path) -> dict[str, float]:
    """List `<vault>/sessions/*.md` with mtimes.

    Args:
        vault_root: `smoke-state/vault`.

    Returns:
        filename -> mtime. Empty when the directory does not exist.
    """
    sessions = vault_root / "sessions"
    if not sessions.is_dir():
        return {}
    notes: dict[str, float] = {}
    for path in sorted(sessions.glob("*.md")):
        try:
            notes[path.name] = path.stat().st_mtime
        except OSError:
            continue
    return notes


def load_corpus_facts(turns_path: Path) -> list[str]:
    """Read the corpus facts A3 greps the note for.

    Args:
        turns_path: `scripts/smoke/turns.json`.

    Returns:
        The `corpus_facts` list, or an empty list when it cannot be read.
    """
    try:
        raw = json.loads(turns_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    facts = raw.get("corpus_facts", [])
    return [str(fact) for fact in facts] if isinstance(facts, list) else []


def render(verdict: Verdict) -> str:
    """Format one verdict for stdout."""
    lines = [
        "-" * 78,
        f"{verdict.status}  {verdict.assertion} -- {verdict.title}",
        f"  {verdict.headline}",
    ]
    lines.extend(f"    {line}" for line in verdict.evidence)
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI."""
    parser = argparse.ArgumentParser(
        prog="assert_artifacts.py",
        description="Assert the four artifacts of the live-path smoke run.",
    )
    parser.add_argument("--session-id", help="The server-minted session id from the driver.")
    parser.add_argument(
        "--smoke-state",
        type=Path,
        required=True,
        help="The smoke state root, normally ./smoke-state.",
    )
    parser.add_argument("--transcript", type=Path, help="The driver's JSONL transcript.")
    parser.add_argument(
        "--vault-before",
        type=Path,
        help="JSON written earlier by --snapshot-vault. Absent means 'treat the "
        "sessions directory as having been empty', which is true for a fresh "
        "smoke-state but is stated rather than assumed silently.",
    )
    parser.add_argument(
        "--snapshot-vault",
        type=Path,
        help="Write the current session-note listing here and exit. Run this BEFORE "
        "the conversation so A3 has a real before-list.",
    )
    parser.add_argument(
        "--backend-log",
        type=Path,
        action="append",
        default=[],
        help="A captured backend log file. Repeatable. Capture the Phase 1 log before "
        "the Phase 2 recreate -- see this module's docstring.",
    )
    parser.add_argument(
        "--backend-container",
        default="mist-backend-smoke",
        help="Also read logs live from this container. Pass an empty string to skip.",
    )
    parser.add_argument("--neo4j-container", default="mist-neo4j-smoke")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument(
        "--turns",
        type=Path,
        default=Path(__file__).resolve().parent / "turns.json",
        help="Corpus file, read for the A3 fact list.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to `sys.argv[1:]`.

    Returns:
        A process exit code; see the module docstring.
    """
    args = build_arg_parser().parse_args(argv)
    vault_root = args.smoke_state / "vault"

    if args.snapshot_vault:
        listing = list_session_notes(vault_root)
        args.snapshot_vault.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot_vault.write_text(json.dumps(listing, indent=2), encoding="utf-8")
        print(f"[assert] vault session-note listing ({len(listing)} file(s)) -> "
              f"{args.snapshot_vault}")
        return EXIT_OK

    try:
        session_id = _require_session_id(args.session_id)
    except InvocationError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return EXIT_BAD_INVOCATION

    transcript = (
        parse_transcript(args.transcript.read_text(encoding="utf-8", errors="replace"))
        if args.transcript and args.transcript.exists()
        else TranscriptFacts(
            present=False,
            schema_version=None,
            session_id=None,
            disconnected_at=None,
            terminal_counts={},
            turns_planned=None,
            aborted=None,
            closed=False,
            error=f"transcript not provided or not found: {args.transcript}",
        )
    )

    log_text, log_sources = read_backend_log(args.backend_container or None, args.backend_log)
    log_available = bool(log_text.strip())

    store = read_event_store(args.smoke_state / "event_store.db")
    graph = query_smoke_graph(
        args.neo4j_container,
        args.neo4j_user,
        args.neo4j_password,
        args.neo4j_database,
        session_id,
    )

    before_names: set[str]
    if args.vault_before and args.vault_before.exists():
        before_names = set(json.loads(args.vault_before.read_text(encoding="utf-8")))
    else:
        before_names = set()
    after = list_session_notes(vault_root)
    note_texts = {
        name: (vault_root / "sessions" / name).read_text(encoding="utf-8", errors="replace")
        for name in after
        if name not in before_names
    }

    print("=" * 78)
    print("LIVE-PATH SMOKE: ARTIFACT ASSERTIONS")
    print("=" * 78)
    print(f"  session_id     : {session_id}")
    print(f"  smoke-state    : {args.smoke_state}")
    print(f"  transcript     : {args.transcript} (usable: {transcript.present})")
    if transcript.session_id and transcript.session_id != session_id:
        print(
            f"  WARNING: the transcript's session_id is {transcript.session_id}, which is "
            f"not the --session-id given. Every assertion below joins on --session-id."
        )
    for source in log_sources:
        print(f"  log source     : {source}")
    if not log_available:
        print("  log source     : NONE readable")
    if store.get("status") != "ok":
        print(f"  event store    : UNAVAILABLE ({store.get('error')})")

    a1 = adjudicate_a1(
        store.get("turns", []), store.get("sessions", []), transcript, session_id
    )
    a2 = adjudicate_a2(
        graph.get("entities"),
        graph.get("edges"),
        graph.get("rows", []),
        parse_extraction_skips(log_text),
        a1.status,
        log_available,
        graph.get("error"),
    )
    a3 = adjudicate_a3(
        before_names,
        after,
        note_texts,
        load_corpus_facts(args.turns),
        transcript,
        log_text,
        log_available,
        session_id,
    )
    a4 = adjudicate_a4(store.get("curation", []), log_text, log_available)

    verdicts = [a1, a2, a3, a4]
    for verdict in verdicts:
        print(render(verdict))

    print("=" * 78)
    print("  " + "  ".join(f"{v.assertion}={v.status}" for v in verdicts))
    failed = [v.assertion for v in verdicts if v.status == FAIL]
    if failed:
        print(f"  RESULT: FAIL ({', '.join(failed)})")
        print("=" * 78)
        return EXIT_FAILED
    inconclusive = [v.assertion for v in verdicts if v.status == INCONCLUSIVE]
    if inconclusive:
        print(
            f"  RESULT: NO FAILURES, but {', '.join(inconclusive)} could not be decided. "
            "An INCONCLUSIVE is not a pass -- read its evidence before concluding "
            "anything about the pipeline."
        )
    else:
        print("  RESULT: all four artifacts observed")
    print("=" * 78)
    return EXIT_OK


def _require_session_id(value: str | None) -> str:
    """Validate the session id argument.

    Args:
        value: What the operator passed.

    Returns:
        The validated id.

    Raises:
        InvocationError: When it is missing or not a UUID.
    """
    if not value:
        raise InvocationError(
            "--session-id is required. The SERVER mints it (backend/server.py:787) and "
            "sends it on the session_started frame; drive_turns.py prints it as "
            "SMOKE_SESSION_ID=<uuid> and records it in the transcript's `session` and "
            "`summary` records."
        )
    if not SESSION_ID_RE.match(value):
        raise InvocationError(
            f"--session-id {value!r} is not a uuid4. The server mints it with "
            "`str(uuid.uuid4())` at backend/server.py:787, so a client-chosen value "
            "such as 'default' joins to nothing. Take it from the driver's output."
        )
    return value


if __name__ == "__main__":
    sys.exit(main())
