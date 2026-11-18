"""Drive one real conversation over the smoke stack's WebSocket and record it.

This is the instrument, not the experiment. It speaks five turns to
`ws://localhost:8003/ws` (the smoke backend published by
`docker-compose.live-path-smoke.yml:160`), writes every frame it sees to a JSONL
transcript, closes cleanly, and exits. It asserts nothing about the memory
pipeline -- `assert_artifacts.py` does that, and the transcript this script
writes is the evidence it reads.

WHAT THE TRANSCRIPT IS FOR
--------------------------
Two of the four artifact assertions are only decidable with it. A1 (turn rows)
distinguishes "the pipeline failed to write" from "the turns never completed"
by counting `stream_complete` frames here; A3 (session note) attributes the
note to the disconnect hook or to startup catch-up by comparing file mtimes
against the disconnect timestamp recorded here. A missing transcript turns both
FAILs into INCONCLUSIVEs, so the transcript is flushed after every record
rather than at exit.

PROTOCOL, verified against source
---------------------------------
The wire contract is the realtime protocol, 2.0.0
(`backend/realtime/protocol.py`).

- The endpoint is `@app.websocket("/ws")`, `websocket_endpoint` in
  `backend/server.py`. It hands the socket to one `RealtimeSession`
  (`backend/realtime/session.py`), whose `_on_text` parses each text message
  with `parse_client_message` and posts a `TextMessage` to the session's
  `TurnController` (`backend/realtime/turn.py`).
- A text turn is exactly `{"type": "text", "text": "<utterance>"}`
  (`TextMessage`, `protocol.py:398-402`). An empty `text` is not skipped: it
  opens a turn with an empty utterance, which is why `build_text_frame`
  refuses one here.
- The SERVER mints the session id: `session_id = str(uuid.uuid4())` in
  `websocket_endpoint`, once per connection. A client cannot supply one. The
  first frame sent is `session_started` and its `session_id` is the join key
  for every later assertion, so this script reads it from that frame and
  refuses to proceed without it.
- Routing is per session: a turn's frames go to its own session's socket only
  (`grep -rn message_queue backend/realtime` -> no hits); only process-wide
  events such as heartbeats reach every client through the broadcaster.
- A typed turn opens with `stream_start{turn_id, channel, origin}`, channel
  and origin both `text` and no `audio_tag` (`_open_turn` and `_channel_for`
  in `turn.py`), then streams `stream_token` frames, then ends its stream with
  exactly one of `stream_complete` or `stream_cancelled`. A failed generation
  sends a `voice_decision` `degraded` `turn_error` and still ends with
  `stream_complete`. The turn is committed AFTER its stream terminal, and
  `state_cycle` idle and then `turn_metrics{turn_id, outcome, recorded}`
  follow the commit (`TurnController._complete` and `_cut`). A text sent
  before that commit cuts the turn as `new_input` (`TurnController._on_text`),
  so the driver waits for `turn_metrics` before sending the next turn.
- An `error` frame never names a turn: a protocol error carries
  `context: {"field": ...}` (`RealtimeSession._protocol_error`), the server's
  own errors carry `context: None`. One that arrives before `stream_start`
  means the text was refused and no turn will open.
- A typed turn is never spoken (`_channel_for` speaks only a `voice` turn),
  so TTS_ENABLED does not change what this script sees.

ISOLATION
---------
`assert_ws_target_not_live` (`backend/knowledge/eval_isolation.py`) runs
BEFORE any socket is opened. Its hardcoded denylist `LIVE_WS_ENDPOINTS`
refuses `mist-backend:8001`, `localhost:8001` and `127.0.0.1:8001`, and no
environment variable can widen it. Its allowlist arm defaults to the DEV
endpoints (`DEFAULT_DEV_WS_ENDPOINTS`), so the smoke port must be
named explicitly via MIST_DEV_WS_HOSTS; see `_guard_message` for the exact
value and RUNBOOK.md for where it is set.

CLEAN CLOSE
-----------
The session note is written from `websocket_endpoint`'s `finally` block
(`backend/server.py`): once the session has committed its turn and `run` has
returned, `_end_handler_session` calls `ConversationHandler.end_session`
(`backend/chat/conversation_handler.py:3394`). That call also DRAINS the
per-turn background extraction tasks (`_drain_extraction_tasks`,
`conversation_handler.py:3429`) before synthesising, which is what makes
"assert after a clean disconnect" free of the "a background task had not
finished yet" ambiguity. A killed client still reaches the server's `finally`
when the session's receive loop sees the disconnect -- but a killed
DRIVER skips its own transcript flush and summary, so this script closes the
socket, awaits closure, and returns rather than being interrupted.

EXIT CODES
----------
The exit code reports whether the INSTRUMENT worked, not whether the pipeline
did. A conversation in which every turn returned `error` still exits 0, because
the instrument did its job and `assert_artifacts.py` is the judge of the result.

    0  conversation ran to the end and the transcript was flushed
    2  `assert_ws_target_not_live` refused the target
    3  `websockets` could not be imported on this host
    4  the corpus file is missing or malformed
    5  the connection could not be opened
    6  a turn timed out, or the connection dropped mid-conversation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

# scripts/smoke/drive_turns.py -> scripts/smoke -> scripts -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TURNS_PATH = Path(__file__).resolve().parent / "turns.json"

#: Bump when the transcript record shape changes. `assert_artifacts.py` reads
#: this field into `TranscriptFacts.schema_version` (`assert_artifacts.py:235`,
#: `:253`, `:268`, `:285`) and currently COMPARES IT TO NOTHING:
#: `grep -rn "schema_version" scripts/smoke/` shows no equality test and no
#: refusal path. So bumping this alone does not make an old reader reject a new
#: transcript -- a v2 transcript fed to a v1 reader would count zero
#: `stream_complete` frames and report A1 INCONCLUSIVE for an instrument
#: reason. Anyone changing the record shape must add the version check at the
#: same time. Recorded as a known gap rather than described as a guard it is
#: not: an earlier revision of this comment claimed the refusal already
#: existed, which would have let exactly that silent misread through.
#:
#: The protocol 2 port kept version 1: it only ADDED fields to `turn_outcome`
#: and `summary` records, and every field a v1 reader takes (`terminal_type`,
#: `terminal_counts`, `session_id`, `disconnected_at`) keeps its name and meaning.
TRANSCRIPT_SCHEMA_VERSION = 1

#: How a turn can end, as `TurnOutcome.terminal_type` and the summary's
#: `terminal_counts` name it. `stream_complete` and `stream_cancelled` are the
#: turn's stream terminal frames. `error` is a turn whose `turn_metrics` outcome
#: is `error` (its stream still ends with `stream_complete`, after a `degraded
#: turn_error` decision), or a text the server refused before any turn opened.
TERMINAL_FRAME_TYPES = frozenset({"error", "stream_cancelled", "stream_complete"})

#: The stream terminal frames. Neither ends the turn: the turn commits after
#: it, and `turn_metrics` follows the commit.
STREAM_TERMINAL_TYPES = frozenset({"stream_cancelled", "stream_complete"})

#: What a typed turn's `stream_start` carries: `_on_text` opens the turn with
#: origin `text` (`backend/realtime/turn.py:2182`), and `_channel_for`
#: (`turn.py:2510-2520`) speaks only a `voice` turn, so the channel is `text`
#: and `audio_tag` is absent whatever TTS_ENABLED says.
EXPECTED_STREAM_START = (("origin", "text"), ("channel", "text"))

EXIT_OK = 0
EXIT_GUARD_REFUSED = 2
EXIT_NO_WEBSOCKETS = 3
EXIT_BAD_CORPUS = 4
EXIT_CONNECT_FAILED = 5
EXIT_CONVERSATION_INCOMPLETE = 6


class CorpusError(ValueError):
    """The turns file is missing, malformed, or describes an unsendable turn."""


@dataclass(frozen=True, slots=True)
class TurnSpec:
    """One planned utterance."""

    index: int
    text: str
    purpose: str


@dataclass(frozen=True, slots=True)
class TurnPlan:
    """The whole corpus: what to say, and what the assertions will look for."""

    corpus_id: str
    turns: tuple[TurnSpec, ...]
    corpus_facts: tuple[str, ...]
    retrieval_probe_terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    """How one turn ended, as observed on the wire.

    Attributes:
        terminal_type: One of `TERMINAL_FRAME_TYPES`.
        turn_id: The id `turn_metrics` carried, or None when the turn never
            opened (a refused text, ended by an `error`).
        turn_id_matched: Whether the stream terminal's id equals the one
            `stream_start` opened the turn with.
        text: `full_text` for `stream_complete`, `partial_text` for
            `stream_cancelled`, `message` for an `error` that ended the wait.
            Three fields, one attribute, because what the caller wants is
            "what did the server end up saying".
        error_kind: The first `degraded` decision's reason for this turn
            (`turn_error` for a failed generation), or the refusing `error`
            frame's `kind`.
        origin, channel, audio_tag: As `stream_start` carried them.
        stream_start_problems: Where `stream_start` differed from a typed
            turn's (`EXPECTED_STREAM_START`, no `audio_tag`).
        cancel_reason: `stream_cancelled.reason`, else None.
        outcome, recorded: As `turn_metrics` carried them; None when the turn
            never opened.
        session_error_kinds: Session-level `error` frames that arrived while
            the turn ran; they do not end it.
        foreign_frames: Frames naming another turn id. Turn frames go to their
            own session's socket only, so anything but 0 is a defect.
    """

    turn_index: int
    utterance: str
    terminal_type: str
    turn_id: str | None
    turn_id_matched: bool
    text: str
    duration_ms: int | None
    tool_calls_used: int | None
    error_kind: str | None
    token_count: int
    elapsed_s: float
    origin: str | None
    channel: str | None
    audio_tag: int | None
    stream_start_problems: tuple[str, ...]
    cancel_reason: str | None
    outcome: str | None
    recorded: bool | None
    session_error_kinds: tuple[str, ...]
    foreign_frames: int


# ---------------------------------------------------------------------------
# Pure logic. No sockets, no third-party imports -- all of this is unit-tested
# in tests/unit/smoke/test_smoke_driver.py.
# ---------------------------------------------------------------------------


def load_turn_plan(path: Path) -> TurnPlan:
    """Parse and validate the corpus file.

    Validation is not ceremony here. A turn under the 3-word floor
    (`ExtractionPipeline._is_too_short`, `backend/knowledge/extraction/pipeline.py`)
    would be silently skipped for extraction, and an EMPTY turn is worse: the
    server does not skip it (`TextMessage.text` is any string,
    `backend/realtime/protocol.py:398-402`), so it would open a real turn with
    an empty utterance that the corpus never meant to say.

    Args:
        path: Location of `turns.json`.

    Returns:
        The parsed plan, turns in file order.

    Raises:
        CorpusError: If the file is absent, is not valid JSON, has no turns, or
            contains a turn whose text is empty or not a string.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CorpusError(f"corpus file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorpusError(f"corpus file {path} is not valid JSON: {exc}") from exc

    entries = raw.get("turns")
    if not isinstance(entries, list) or not entries:
        raise CorpusError(f"corpus file {path} has no 'turns' list")

    turns: list[TurnSpec] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise CorpusError(f"turn {position} in {path} is not an object")
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise CorpusError(
                f"turn {position} in {path} has empty or non-string 'text'. "
                "The server does not skip an empty text (TextMessage, "
                "backend/realtime/protocol.py:398-402): it would open a turn with "
                "an empty utterance."
            )
        turns.append(
            TurnSpec(
                index=int(entry.get("index", position)),
                text=text,
                purpose=str(entry.get("purpose", "unspecified")),
            )
        )

    return TurnPlan(
        corpus_id=str(raw.get("corpus_id", "")),
        turns=tuple(turns),
        corpus_facts=tuple(str(f) for f in raw.get("corpus_facts", ())),
        retrieval_probe_terms=tuple(str(f) for f in raw.get("retrieval_probe_terms", ())),
    )


def build_text_frame(text: str) -> dict[str, str]:
    """Build the one inbound frame shape this script sends.

    Exactly two keys. `TextMessage` (`backend/realtime/protocol.py:398-402`)
    has `type` and `text` and nothing else, and its model ignores extra fields
    (`extra="ignore"` on `_WireModel`), so any extra field would be silently
    discarded -- and a field that looks like it is doing something while being
    discarded is how a session_id comes to be believed in. The server mints its
    own session id in `websocket_endpoint` (`backend/server.py`); there is no
    client-supplied id to send.

    Args:
        text: The utterance. Must be non-empty after stripping.

    Returns:
        `{"type": "text", "text": text}`.

    Raises:
        CorpusError: If `text` is empty or whitespace-only.
    """
    if not text.strip():
        raise CorpusError(
            "refusing to send an empty text frame: the server would open a turn "
            "with an empty utterance (TextMessage, backend/realtime/protocol.py:398-402)"
        )
    return {"type": "text", "text": text}


def parse_frame(payload: str | bytes) -> dict[str, Any] | None:
    """Decode one received frame, returning None for anything undecodable.

    Binary frames are not decoded. On this path they should not occur at all:
    downlink audio frames carry a `tts`-channel turn's `audio_tag`, and a typed
    turn has the `text` channel (`_channel_for`, `backend/realtime/turn.py`);
    the smoke stack also sets MIST_VOICE_TTS_CHAIN=off, so no TTS engine is
    loaded (`docker-compose.live-path-smoke.yml`). They are
    counted rather than treated as an error, so an unexpected one shows up in
    the transcript instead of aborting the run.

    Args:
        payload: What `recv()` returned.

    Returns:
        The decoded object when it is a JSON object, else None.
    """
    if isinstance(payload, bytes | bytearray):
        return None
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def session_id_from_frame(frame: dict[str, Any]) -> str | None:
    """Extract the server-minted session id from a `session_started` frame.

    Args:
        frame: A decoded inbound frame.

    Returns:
        The session id when this is a `session_started` frame carrying a
        non-empty one (`SessionStartedMessage`, `backend/realtime/protocol.py`),
        else None.
    """
    if frame.get("type") != "session_started":
        return None
    session_id = frame.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def stream_start_problems(frame: dict[str, Any]) -> tuple[str, ...]:
    """Where a `stream_start` differs from a typed turn's.

    Args:
        frame: A decoded `stream_start` frame.

    Returns:
        One line per unexpected field; empty for a typed turn's opener.
    """
    problems = [
        f"{key} {frame.get(key)!r}, expected {expected!r}"
        for key, expected in EXPECTED_STREAM_START
        if frame.get(key) != expected
    ]
    if frame.get("audio_tag") is not None:
        problems.append(
            f"audio_tag {frame.get('audio_tag')!r}, expected none: a text-channel turn "
            "sends no audio"
        )
    return tuple(problems)


def terminal_turn_id(frame: dict[str, Any]) -> str | None:
    """Read the turn id off a terminal frame.

    `stream_complete` and `stream_cancelled` carry `turn_id` at the top level.
    An `error` never names a turn in protocol 2: its `context` is
    `{"field": ...}` for a message that failed to parse
    (`RealtimeSession._protocol_error`, `backend/realtime/session.py:707-720`)
    and None for the server's own errors (`_ws_error`,
    `backend/server.py:2345`, and the not-ready error in `websocket_endpoint`).

    Args:
        frame: A decoded terminal frame.

    Returns:
        The turn id, or None when the frame carries none.
    """
    if frame.get("type") == "error":
        return None
    turn_id = frame.get("turn_id")
    return turn_id if isinstance(turn_id, str) else None


def terminal_text(frame: dict[str, Any]) -> str:
    """The server's final text for a turn, from whichever field carries it.

    Args:
        frame: A decoded terminal frame.

    Returns:
        `full_text` for `stream_complete`, `partial_text` for
        `stream_cancelled`, `message` for `error`. Empty string when the field
        is absent or not a string.
    """
    field = {
        "stream_complete": "full_text",
        "stream_cancelled": "partial_text",
        "error": "message",
    }.get(str(frame.get("type")), "")
    value = frame.get(field)
    return value if isinstance(value, str) else ""


class TurnRecorder:
    """The per-turn frame state machine.

    One instance per turn. Frames are fed in arrival order; `consume` returns
    None until the turn has ended, then returns its `TurnOutcome`. A turn ends
    at the `turn_metrics` naming it, not at its stream terminal: `_complete`
    and `_cut` commit the turn between the two (`backend/realtime/turn.py:3088-3093`,
    `:3280-3293`), and a text sent before the commit cuts the turn as
    `new_input` (`TurnController._on_text`, `turn.py:2167-2182`). The one
    exception is an `error` before any `stream_start`: the server refused the
    text and no turn will open. Feeding the recorder after that raises, because
    silently overwriting the first outcome would hide a second end.
    """

    def __init__(self, spec: TurnSpec, *, started_at: float | None = None) -> None:
        self._spec = spec
        self._started_at = time.monotonic() if started_at is None else started_at
        self._start: dict[str, Any] | None = None
        self._stream_turn_id: str | None = None
        self._tokens: list[str] = []
        self._terminal: dict[str, Any] | None = None
        self._degraded: list[str] = []
        self._session_errors: list[str] = []
        self._foreign = 0
        self._done = False

    @property
    def stream_turn_id(self) -> str | None:
        """The turn id `stream_start` opened this turn with, if it has arrived."""
        return self._stream_turn_id

    @property
    def streamed_text(self) -> str:
        """Concatenation of every `stream_token` seen so far.

        Kept alongside the terminal's `full_text` because they can differ:
        `_send_complete` (`backend/realtime/turn.py:2916-2930`) sends the
        generation's `final_response` when it has one.
        """
        return "".join(self._tokens)

    def consume(self, frame: dict[str, Any], *, now: float | None = None) -> TurnOutcome | None:
        """Feed one frame.

        Args:
            frame: A decoded inbound frame.
            now: Monotonic clock override, for tests.

        Returns:
            The outcome when `frame` ends the turn, else None.

        Raises:
            RuntimeError: If fed a frame after the turn has already ended.
        """
        if self._done:
            raise RuntimeError(
                f"turn {self._spec.index} already terminated; a further frame "
                f"({frame.get('type')!r}) would overwrite its outcome"
            )

        frame_type = frame.get("type")

        if frame_type == "error":
            kind = frame.get("kind") if isinstance(frame.get("kind"), str) else "unknown"
            if self._stream_turn_id is None:
                return self._end(frame, outcome=None, recorded=None, error_kind=kind, now=now)
            self._session_errors.append(kind)
            return None

        if frame_type == "stream_start" and self._stream_turn_id is None:
            turn_id = frame.get("turn_id")
            if isinstance(turn_id, str):
                self._stream_turn_id = turn_id
                self._start = frame
            return None

        turn_id = frame.get("turn_id")
        if self._stream_turn_id is None or not isinstance(turn_id, str):
            return None
        if turn_id != self._stream_turn_id:
            self._foreign += 1
            return None

        if frame_type == "stream_token":
            token = frame.get("token")
            if isinstance(token, str):
                self._tokens.append(token)
        elif frame_type == "voice_decision" and frame.get("kind") == "degraded":
            self._degraded.append(str(frame.get("reason")))
        elif frame_type in STREAM_TERMINAL_TYPES and self._terminal is None:
            self._terminal = frame
        elif frame_type == "turn_metrics":
            outcome = frame.get("outcome") if isinstance(frame.get("outcome"), str) else None
            recorded = frame.get("recorded") if isinstance(frame.get("recorded"), bool) else None
            error_kind = self._degraded[0] if self._degraded else None
            return self._end(
                self._terminal, outcome=outcome, recorded=recorded, error_kind=error_kind, now=now
            )
        return None

    def _end(
        self,
        terminal: dict[str, Any] | None,
        *,
        outcome: str | None,
        recorded: bool | None,
        error_kind: str | None,
        now: float | None,
    ) -> TurnOutcome:
        self._done = True
        terminal = terminal or {}
        if outcome == "error" or terminal.get("type") == "error":
            terminal_type = "error"
        else:
            terminal_type = str(terminal.get("type", "none"))
        start = self._start or {}
        terminal_id = terminal_turn_id(terminal)
        cancel_reason = terminal.get("reason")
        elapsed = (time.monotonic() if now is None else now) - self._started_at
        return TurnOutcome(
            turn_index=self._spec.index,
            utterance=self._spec.text,
            terminal_type=terminal_type,
            turn_id=self._stream_turn_id,
            turn_id_matched=terminal_id is not None and terminal_id == self._stream_turn_id,
            text=terminal_text(terminal),
            duration_ms=_as_int(terminal.get("duration_ms")),
            tool_calls_used=_as_int(terminal.get("tool_calls_used")),
            error_kind=error_kind,
            token_count=len(self._tokens),
            elapsed_s=elapsed,
            origin=start.get("origin") if isinstance(start.get("origin"), str) else None,
            channel=start.get("channel") if isinstance(start.get("channel"), str) else None,
            audio_tag=_as_int(start.get("audio_tag")),
            stream_start_problems=stream_start_problems(start) if start else (),
            cancel_reason=cancel_reason if isinstance(cancel_reason, str) else None,
            outcome=outcome,
            recorded=recorded,
            session_error_kinds=tuple(self._session_errors),
            foreign_frames=self._foreign,
        )


def _as_int(value: Any) -> int | None:
    """Coerce a frame field to int, or None when it is not an integer."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def summarise_outcomes(outcomes: list[TurnOutcome]) -> dict[str, int]:
    """Count terminals by type.

    This is the count A1 turns on: `assert_artifacts.py` calls fewer than five
    `stream_complete` terminals INCONCLUSIVE rather than FAIL, because a turn
    that never completed never exercised `_record_turn_event`
    (`backend/chat/conversation_handler.py:3888`) as a completed turn. A
    failed turn counts as `error`, not `stream_complete`, though protocol 2
    ends its stream with `stream_complete` (see `TERMINAL_FRAME_TYPES`).

    Args:
        outcomes: Every turn outcome, in order.

    Returns:
        A dict with one key per terminal type plus `total`. Terminal types with
        no occurrences are present with value 0, so a consumer never has to
        distinguish "absent key" from "zero".
    """
    counts = {name: 0 for name in sorted(TERMINAL_FRAME_TYPES)}
    for outcome in outcomes:
        counts[outcome.terminal_type] = counts.get(outcome.terminal_type, 0) + 1
    counts["total"] = len(outcomes)
    return counts


def _guard_message(ws_url: str, reason: str) -> str:
    """Build the refusal message, naming the variable the operator must set."""
    return (
        f"REFUSED: {ws_url} did not pass assert_ws_target_not_live "
        f"(backend/knowledge/eval_isolation.py).\n"
        f"  Guard said: {reason}\n"
        "  If the target IS the smoke stack (host port 8003, published by "
        "docker-compose.live-path-smoke.yml:160), the allowlist arm is what "
        "refused it: it defaults to the DEV endpoints "
        "(DEFAULT_DEV_WS_ENDPOINTS in eval_isolation.py). Export exactly:\n"
        "      MIST_DEV_WS_HOSTS=localhost:8003,127.0.0.1:8003\n"
        "  The denylist arm (LIVE_WS_ENDPOINTS) refuses localhost:8001, "
        "127.0.0.1:8001 and mist-backend:8001 -- the LIVE backend -- and no "
        "environment variable widens it. If that is what refused you, the URL is "
        "wrong; do not try to work around it."
    )


def guard_ws_target(ws_url: str) -> None:
    """Refuse a target that is, or might be, the live backend.

    Called before the socket is opened. The guard is a pure function and makes
    no network call of its own (`assert_ws_target_not_live`'s docstring,
    `backend/knowledge/eval_isolation.py`),
    so a driver aimed at live is refused without ever contacting live.

    Args:
        ws_url: The `--url` the operator passed.

    Raises:
        SystemExit: With `EXIT_GUARD_REFUSED` when the guard refuses.
    """
    from backend.knowledge.eval_isolation import (
        EvalIsolationError,
        assert_ws_target_not_live,
    )

    try:
        assert_ws_target_not_live(ws_url)
    except EvalIsolationError as exc:
        print(_guard_message(ws_url, str(exc)), file=sys.stderr)
        raise SystemExit(EXIT_GUARD_REFUSED) from exc


def resolve_websockets() -> tuple[Any, str]:
    """Import `websockets` lazily and report the version that got resolved.

    Lazy, and inside a function rather than at module scope, for two reasons.
    First, `tests/unit/smoke/test_smoke_driver.py` imports this module to test
    the pure logic above, and that must not depend on `websockets` being
    installed. Second, this script runs on the HOST, whose Python is not the
    container's: the container image was measured at websockets 16.0 (`python -c
    "import websockets; print(websockets.__version__)"`), but the host's
    interpreter is a different installation entirely, so the version is printed
    into the transcript rather than assumed.

    Returns:
        (module, version string).

    Raises:
        SystemExit: With `EXIT_NO_WEBSOCKETS` and an actionable message when the
            import fails.
    """
    try:
        import websockets
    except ImportError as exc:
        print(
            "FAILED: this script needs the `websockets` package and the host "
            f"Python at {sys.executable} does not have it.\n"
            f"  Import error: {exc}\n"
            "  Install it into the interpreter you are running this with:\n"
            f"      {sys.executable} -m pip install websockets\n"
            "  Do not switch to `python3` on this machine -- that name opens the "
            "Windows Store prompt. Use `python`.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_NO_WEBSOCKETS) from exc

    return websockets, str(getattr(websockets, "__version__", "unknown"))


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


class Transcript:
    """Append-only JSONL record of the run.

    Flushed after every record. The transcript is the discriminator for two of
    the four assertions, so losing the tail of it to a buffer would convert a
    decidable FAIL into an INCONCLUSIVE.

    Record shapes, all carrying `t` (ISO-8601 UTC) and `record`:
        `header`       -- schema version, url, corpus id, websockets version
        `session`      -- the server-minted session id
        `sent`         -- one outbound text frame
        `frame`        -- one inbound frame, verbatim, with the turn it arrived
                          under (`turn_index`, null before the first turn)
        `turn_outcome` -- one per terminated turn
        `note`         -- driver-level events (timeouts, binary frames, close)
        `summary`      -- terminal counts, disconnect timestamp, probe answer
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, record: str, **fields: Any) -> None:
        """Append one record and flush.

        Args:
            record: The record kind.
            **fields: Record-specific payload.
        """
        line = {"t": datetime.now(UTC).isoformat(), "record": record, **fields}
        self._stream.write(json.dumps(line, default=str) + "\n")
        self._stream.flush()


# ---------------------------------------------------------------------------
# The conversation
# ---------------------------------------------------------------------------


async def _await_session_started(ws: Any, transcript: Transcript, timeout: float) -> str:
    """Read frames until `session_started` arrives, and return its session id.

    The handshake is the FIRST frame the server sends (`websocket_endpoint`,
    `backend/server.py`), but this loops rather than reading exactly
    one frame so an unexpected leading frame is recorded instead of crashing the
    run.

    Args:
        ws: The open connection.
        transcript: Where to record what arrives.
        timeout: Seconds to wait in total.

    Returns:
        The server-minted session id.

    Raises:
        TimeoutError: If no `session_started` arrives within `timeout`.
    """
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no session_started frame within {timeout:.0f}s")
        payload = await asyncio.wait_for(ws.recv(), timeout=remaining)
        frame = parse_frame(payload)
        if frame is None:
            transcript.write("note", event="undecodable_frame", turn_index=None)
            continue
        transcript.write("frame", turn_index=None, frame=frame)
        session_id = session_id_from_frame(frame)
        if session_id is not None:
            return session_id


async def _run_turn(
    ws: Any,
    spec: TurnSpec,
    transcript: Transcript,
    timeout: float,
) -> TurnOutcome:
    """Send one utterance and read frames until its turn ends.

    The turn ends at its `turn_metrics`, after the commit, not at its stream
    terminal; returning earlier would let the next turn's text cut this one as
    `new_input` (see `TurnRecorder`).

    Args:
        ws: The open connection.
        spec: The turn to send.
        transcript: Where to record what arrives.
        timeout: Seconds to wait for the turn to end.

    Returns:
        The turn's outcome.

    Raises:
        TimeoutError: If the turn does not end within `timeout`.
    """
    frame_out = build_text_frame(spec.text)
    await ws.send(json.dumps(frame_out))
    transcript.write("sent", turn_index=spec.index, frame=frame_out, purpose=spec.purpose)

    recorder = TurnRecorder(spec)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"turn {spec.index} produced no turn_metrics within {timeout:.0f}s "
                f"(saw {len(recorder.streamed_text)} streamed characters)"
            )
        payload = await asyncio.wait_for(ws.recv(), timeout=remaining)
        frame = parse_frame(payload)
        if frame is None:
            kind = "binary_frame" if isinstance(payload, bytes | bytearray) else "undecodable"
            transcript.write("note", event=kind, turn_index=spec.index)
            continue
        transcript.write("frame", turn_index=spec.index, frame=frame)
        outcome = recorder.consume(frame)
        if outcome is not None:
            transcript.write(
                "turn_outcome",
                turn_index=outcome.turn_index,
                terminal_type=outcome.terminal_type,
                turn_id=outcome.turn_id,
                turn_id_matched=outcome.turn_id_matched,
                utterance=outcome.utterance,
                text=outcome.text,
                streamed_text=recorder.streamed_text,
                duration_ms=outcome.duration_ms,
                tool_calls_used=outcome.tool_calls_used,
                error_kind=outcome.error_kind,
                token_count=outcome.token_count,
                elapsed_s=round(outcome.elapsed_s, 3),
                origin=outcome.origin,
                channel=outcome.channel,
                audio_tag=outcome.audio_tag,
                stream_start_problems=list(outcome.stream_start_problems),
                cancel_reason=outcome.cancel_reason,
                outcome=outcome.outcome,
                recorded=outcome.recorded,
                session_error_kinds=list(outcome.session_error_kinds),
                foreign_frames=outcome.foreign_frames,
            )
            for problem in outcome.stream_start_problems:
                print(f"[smoke] turn {spec.index}: unexpected stream_start: {problem}")
            if outcome.foreign_frames:
                print(
                    f"[smoke] turn {spec.index}: {outcome.foreign_frames} frames named "
                    "another turn; turn frames should reach their own session only"
                )
            return outcome


async def run_conversation(
    ws_url: str,
    plan: TurnPlan,
    transcript: Transcript,
    websockets: Any,
    ws_version: str,
    *,
    connect_timeout: float,
    turn_timeout: float,
) -> int:
    """Open the connection, speak every turn, close cleanly.

    Args:
        ws_url: Already guarded by `guard_ws_target`.
        plan: The corpus.
        transcript: Where to record the run.
        websockets: The resolved `websockets` module.
        ws_version: Its version string, recorded in the transcript header.
        connect_timeout: Seconds allowed for the TCP+WS handshake.
        turn_timeout: Seconds allowed per turn for its `turn_metrics`.

    Returns:
        A process exit code.
    """
    transcript.write(
        "header",
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        ws_url=ws_url,
        corpus_id=plan.corpus_id,
        turn_count=len(plan.turns),
        websockets_version=ws_version,
        python=sys.version.split()[0],
        executable=sys.executable,
    )
    print(f"[smoke] websockets {ws_version} (host python {sys.version.split()[0]})")
    print(f"[smoke] connecting to {ws_url}")

    try:
        connection = await websockets.connect(ws_url, open_timeout=connect_timeout)
    except Exception as exc:  # noqa: BLE001 -- every connect failure is one report
        transcript.write("note", event="connect_failed", error=str(exc))
        print(
            f"FAILED: could not open {ws_url}: {exc}\n"
            "  Check the smoke backend is up and healthy:\n"
            '      docker ps --filter name=mist-backend-smoke --format "{{.Status}}"\n'
            "  Its healthcheck has a 120s start_period "
            "(docker-compose.live-path-smoke.yml:381): the backend builds its "
            "runtime and warms its models before it accepts a connection, so a "
            "cold start is not instant.",
            file=sys.stderr,
        )
        return EXIT_CONNECT_FAILED

    outcomes: list[TurnOutcome] = []
    exit_code = EXIT_OK
    session_id: str | None = None
    failure: str | None = None
    # Pre-bound so the summary record below always has a value, even if the
    # close itself raises: A3 attributes the session note by comparing its mtime
    # to this timestamp, and a summary with the field missing would make that
    # attribution impossible rather than merely imprecise.
    disconnected_at = datetime.now(UTC).isoformat()

    try:
        session_id = await _await_session_started(connection, transcript, connect_timeout)
        transcript.write("session", session_id=session_id)
        print(f"[smoke] SMOKE_SESSION_ID={session_id}")
        print("[smoke] The server minted that id (websocket_endpoint). Every")
        print("[smoke] assertion joins on it. Copy it now.")

        for spec in plan.turns:
            print(f"[smoke] turn {spec.index} ({spec.purpose}): sending")
            outcome = await _run_turn(connection, spec, transcript, turn_timeout)
            outcomes.append(outcome)
            print(
                f"[smoke] turn {spec.index}: {outcome.terminal_type} "
                f"after {outcome.elapsed_s:.1f}s, {outcome.token_count} tokens"
            )
    except (TimeoutError, asyncio.TimeoutError) as exc:
        failure = f"timeout: {exc}"
        exit_code = EXIT_CONVERSATION_INCOMPLETE
    except Exception as exc:  # noqa: BLE001 -- a dropped connection is a report
        failure = f"{type(exc).__name__}: {exc}"
        exit_code = EXIT_CONVERSATION_INCOMPLETE
    finally:
        if failure is not None:
            transcript.write("note", event="conversation_aborted", error=failure)
            print(f"[smoke] conversation aborted: {failure}", file=sys.stderr)
        # Clean close, always. `websocket_endpoint`'s `finally`
        # (backend/server.py) writes the session note on disconnect;
        # closing rather than dropping the process is what lets THIS side record
        # when that happened.
        #
        # The close is itself guarded, because a raise here would skip the
        # summary record below -- and A1 and A3 both read that record. A failed
        # close is worth recording; it is not worth losing the transcript over.
        close_error: str | None = None
        try:
            await connection.close()
            await connection.wait_closed()
        except Exception as exc:  # noqa: BLE001 -- recorded, not propagated
            close_error = f"{type(exc).__name__}: {exc}"
            exit_code = EXIT_CONVERSATION_INCOMPLETE
        disconnected_at = datetime.now(UTC).isoformat()
        transcript.write(
            "note",
            event="closed",
            close_code=getattr(connection, "close_code", None),
            close_error=close_error,
        )

    counts = summarise_outcomes(outcomes)
    probe = next(
        (o.text for o, s in zip(outcomes, plan.turns) if s.purpose == "retrieval-probe"),
        "",
    )
    transcript.write(
        "summary",
        schema_version=TRANSCRIPT_SCHEMA_VERSION,
        session_id=session_id,
        disconnected_at=disconnected_at,
        terminal_counts=counts,
        turns_recorded=sum(1 for o in outcomes if o.recorded),
        turns_planned=len(plan.turns),
        retrieval_probe_answer=probe,
        aborted=failure,
    )

    _print_summary(plan, session_id, counts, probe, disconnected_at)
    return exit_code


def _print_summary(
    plan: TurnPlan,
    session_id: str | None,
    counts: dict[str, int],
    probe: str,
    disconnected_at: str,
) -> None:
    """Print the operator-facing summary, probe answer included."""
    print("")
    print("=" * 72)
    print("SMOKE CONVERSATION SUMMARY")
    print("=" * 72)
    print(f"  session_id      : {session_id}")
    print(f"  disconnected_at : {disconnected_at}")
    print(f"  turns planned   : {len(plan.turns)}")
    print(f"  stream_complete : {counts.get('stream_complete', 0)}")
    print(f"  stream_cancelled: {counts.get('stream_cancelled', 0)}")
    print(f"  error           : {counts.get('error', 0)}")
    print("")
    print("RETRIEVAL PROBE (turn 5). Not one of the four artifacts -- a free read")
    print("on retrieval quality. Whether it names the corpus terms is a finding")
    print(f"worth reporting. Looking for: {', '.join(plan.retrieval_probe_terms)}")
    print("-" * 72)
    print(probe if probe else "(no answer recorded)")
    print("-" * 72)
    if probe:
        hits = [term for term in plan.retrieval_probe_terms if term.lower() in probe.lower()]
        missing = [term for term in plan.retrieval_probe_terms if term not in hits]
        print(f"  terms present: {hits or 'none'}")
        print(f"  terms absent : {missing or 'none'}")
    print("=" * 72)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI.

    `--url` has no default on purpose, matching the reasoning in
    `assert_ws_target_not_live`'s docstring
    (`backend/knowledge/eval_isolation.py:323-324`): a default is a value nobody
    checked, and live and smoke differ by two digits on the host.
    """
    parser = argparse.ArgumentParser(
        prog="drive_turns.py",
        description="Speak the smoke corpus to the smoke backend and transcribe the run.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="WebSocket URL, e.g. ws://localhost:8003/ws. No default: live and "
        "smoke differ only by port on the host.",
    )
    parser.add_argument(
        "--transcript",
        required=True,
        type=Path,
        help="Path for the JSONL transcript. assert_artifacts.py reads this.",
    )
    parser.add_argument(
        "--turns",
        type=Path,
        default=DEFAULT_TURNS_PATH,
        help=f"Corpus file (default: {DEFAULT_TURNS_PATH}).",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=60.0,
        help="Seconds for the handshake and the session_started frame (default: 60).",
    )
    parser.add_argument(
        "--turn-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait per turn for its turn_metrics (default: 300).",
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

    try:
        plan = load_turn_plan(args.turns)
    except CorpusError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return EXIT_BAD_CORPUS

    guard_ws_target(args.url)
    # Resolved BEFORE the transcript is opened, so a host without `websockets`
    # fails with the install instruction instead of leaving a zero-byte
    # transcript that later reads as "the conversation produced nothing".
    websockets, ws_version = resolve_websockets()

    args.transcript.parent.mkdir(parents=True, exist_ok=True)
    with args.transcript.open("w", encoding="utf-8") as handle:
        transcript = Transcript(handle)
        exit_code = asyncio.run(
            run_conversation(
                args.url,
                plan,
                transcript,
                websockets,
                ws_version,
                connect_timeout=args.connect_timeout,
                turn_timeout=args.turn_timeout,
            )
        )
    print(f"[smoke] transcript written to {args.transcript}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
