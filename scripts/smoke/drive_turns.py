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
- The endpoint is `@app.websocket("/ws")` at `backend/server.py:739`. There is
  no pydantic model on input: `backend/server.py:804` does a raw
  `receive_json()` and `:806` dispatches on `data.get("type")`.
- A text turn is exactly `{"type": "text", "text": "<utterance>"}`
  (`backend/server.py:856-858`). Any other field is ignored. An empty `text` is
  skipped by `continue` at `backend/server.py:859`, which is why
  `build_text_frame` refuses an empty utterance here rather than sending one
  and waiting forever for a terminal that cannot come.
- The SERVER mints the session id: `session_id = str(uuid.uuid4())` at
  `backend/server.py:787`, once per connection. A client cannot supply one. The
  first frame sent is `session_started` (`backend/server.py:788-799`) and its
  `session_id` is the join key for every later assertion, so this script reads
  it from that frame and refuses to proceed without it.
- Exactly three terminal frame types exist per turn, mutually exclusive, at
  `backend/voice_processor.py:620-674`: `error` (`:631`), `stream_cancelled`
  (`:646`), `stream_complete` (`:665`). Non-terminal frames also seen on this
  path: `stream_start` (`:547`), `stream_token` (`:601`), and -- with
  TTS_ENABLED=false, which `docker-compose.live-path-smoke.yml:285` sets -- a
  `state_cycle` idle at `backend/voice_processor.py:696-698`.
- The text branch calls `_process_conversation_turn` directly
  (`backend/server.py:872`), the same function the voice path converges on at
  `backend/voice_processor.py:483`. TTS is gated at `:564`, `:609` and `:677`,
  so turning TTS off does not change turn recording.

ISOLATION
---------
`assert_ws_target_not_live` (`backend/knowledge/eval_isolation.py:308`) runs
BEFORE any socket is opened. Its hardcoded denylist `LIVE_WS_ENDPOINTS`
(`eval_isolation.py:120-126`) refuses `mist-backend:8001`, `localhost:8001` and
`127.0.0.1:8001`, and no environment variable can widen it
(`eval_isolation.py:345-355`). Its allowlist arm defaults to the DEV endpoints
(`DEFAULT_DEV_WS_ENDPOINTS`, `eval_isolation.py:79`), so the smoke port must be
named explicitly via MIST_DEV_WS_HOSTS; see `_guard_message` for the exact
value and RUNBOOK.md for where it is set.

CLEAN CLOSE
-----------
The session note is written from the WebSocket `finally` block at
`backend/server.py:1008-1017`, which calls `ConversationHandler.end_session`
(`backend/chat/conversation_handler.py:1885`). That call also DRAINS the
per-turn background extraction tasks (`conversation_handler.py:1920`) before
synthesising, which is what makes "assert after a clean disconnect" free of the
"a background task had not finished yet" ambiguity. A killed client still
triggers the server's `finally` via `WebSocketDisconnect` -- but a killed
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
TRANSCRIPT_SCHEMA_VERSION = 1

#: The three mutually exclusive per-turn terminals, `backend/voice_processor.py`
#: :631 (error), :646 (stream_cancelled), :665 (stream_complete).
TERMINAL_FRAME_TYPES = frozenset({"error", "stream_cancelled", "stream_complete"})

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
        terminal_type: One of `error`, `stream_cancelled`, `stream_complete`.
        turn_id: The id the terminal frame carried, or None when it carried
            none (an `error` whose `context` is None -- see `terminal_turn_id`).
        turn_id_matched: Whether that id equals the one `stream_start` opened
            the turn with. False means the terminal did not belong to this
            turn's stream, which is the shape a protocol-level `error` takes.
        text: `full_text` for `stream_complete`
            (`backend/voice_processor.py:667`), `partial_text` for
            `stream_cancelled` (`:648`), `message` for `error` (`:633`). Three
            different fields, one attribute, because what the caller wants is
            "what did the server end up saying".
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


# ---------------------------------------------------------------------------
# Pure logic. No sockets, no third-party imports -- all of this is unit-tested
# in tests/unit/smoke/test_smoke_driver.py.
# ---------------------------------------------------------------------------


def load_turn_plan(path: Path) -> TurnPlan:
    """Parse and validate the corpus file.

    Validation is not ceremony here. A turn under the 3-word floor at
    `backend/knowledge/extraction/pipeline.py:640` would be silently skipped for
    extraction, and an EMPTY turn is worse: `backend/server.py:859` does
    `continue` on a falsy `text`, so the server would never emit a terminal
    frame and the driver would sit on its per-turn timeout for nothing.

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
                "backend/server.py:859 skips a falsy text with `continue`, so such "
                "a turn would produce no terminal frame and the driver would hang "
                "until its per-turn timeout."
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

    Exactly two keys. `backend/server.py:856-858` reads `data.get("type")` and
    `data.get("text", "")` and nothing else from a text message, so any extra
    field would be silently discarded -- and a field that looks like it is doing
    something while being discarded is how a session_id comes to be believed in.
    The server mints its own session id at `backend/server.py:787`; there is no
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
            "refusing to send an empty text frame: backend/server.py:859 skips it "
            "with `continue`, so no terminal frame would ever arrive"
        )
    return {"type": "text", "text": text}


def parse_frame(payload: str | bytes) -> dict[str, Any] | None:
    """Decode one received frame, returning None for anything undecodable.

    Binary frames are not decoded. On this path they should not occur at all:
    the binary MIST audio frames are emitted by `_tts_consumer`, which only runs
    when TTS is enabled (`backend/voice_processor.py:564`), and
    `docker-compose.live-path-smoke.yml:285` sets TTS_ENABLED=false. They are
    counted rather than treated as an error, so an unexpected one shows up in
    the transcript instead of aborting the run.

    Args:
        payload: What `recv()` returned.

    Returns:
        The decoded object when it is a JSON object, else None.
    """
    if isinstance(payload, (bytes, bytearray)):
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
        non-empty one (`backend/server.py:788-791`), else None.
    """
    if frame.get("type") != "session_started":
        return None
    session_id = frame.get("session_id")
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def is_terminal_frame(frame: dict[str, Any]) -> bool:
    """Whether this frame ends a turn.

    Args:
        frame: A decoded inbound frame.

    Returns:
        True for `error`, `stream_cancelled` and `stream_complete`.
    """
    return frame.get("type") in TERMINAL_FRAME_TYPES


def terminal_turn_id(frame: dict[str, Any]) -> str | None:
    """Read the turn id off a terminal frame, whichever shape it uses.

    `stream_complete` and `stream_cancelled` carry `turn_id` at the top level
    (`backend/voice_processor.py:647`, `:666`). `error` carries it nested, as
    `context.turn_id` (`backend/voice_processor.py:635`).

    An `error` whose `context` is None is NOT a turn terminal: that shape is the
    connection-level validation error built at `backend/server.py:809-816` and
    `:824-832`, and the not-ready error at `backend/server.py:769-775`. It
    returns None here, and the caller records it with `turn_id_matched=False`.

    Args:
        frame: A decoded terminal frame.

    Returns:
        The turn id, or None when the frame carries none.
    """
    if frame.get("type") == "error":
        context = frame.get("context")
        if isinstance(context, dict):
            turn_id = context.get("turn_id")
            return turn_id if isinstance(turn_id, str) else None
        return None
    turn_id = frame.get("turn_id")
    return turn_id if isinstance(turn_id, str) else None


def terminal_text(frame: dict[str, Any]) -> str:
    """The server's final text for a turn, from whichever field carries it.

    Args:
        frame: A decoded terminal frame.

    Returns:
        `full_text` for `stream_complete` (`backend/voice_processor.py:667`),
        `partial_text` for `stream_cancelled` (`:648`), `message` for `error`
        (`:633`). Empty string when the field is absent or not a string.
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
    None until a terminal frame arrives, then returns the `TurnOutcome` for that
    turn. Feeding it further frames after that is a programming error and
    raises, because a second terminal for one turn would mean the three-way
    exclusivity at `backend/voice_processor.py:620-674` had broken and silently
    overwriting the first outcome would hide it.
    """

    def __init__(self, spec: TurnSpec, *, started_at: float | None = None) -> None:
        self._spec = spec
        self._started_at = time.monotonic() if started_at is None else started_at
        self._stream_turn_id: str | None = None
        self._tokens: list[str] = []
        self._done = False

    @property
    def stream_turn_id(self) -> str | None:
        """The turn id `stream_start` opened this turn with, if it has arrived."""
        return self._stream_turn_id

    @property
    def streamed_text(self) -> str:
        """Concatenation of every `stream_token` seen so far.

        Kept alongside the terminal's `full_text` because they can differ:
        `backend/voice_processor.py:617` trims the response to its last complete
        sentence before building `stream_complete`, after the tokens have
        already gone out at `:601`.
        """
        return "".join(self._tokens)

    def consume(self, frame: dict[str, Any], *, now: float | None = None) -> TurnOutcome | None:
        """Feed one frame.

        Args:
            frame: A decoded inbound frame.
            now: Monotonic clock override, for tests.

        Returns:
            The outcome when `frame` is a terminal, else None.

        Raises:
            RuntimeError: If fed a frame after the turn has already terminated.
        """
        if self._done:
            raise RuntimeError(
                f"turn {self._spec.index} already terminated; a second frame "
                f"({frame.get('type')!r}) would overwrite its outcome"
            )

        frame_type = frame.get("type")

        if frame_type == "stream_start":
            turn_id = frame.get("turn_id")
            if isinstance(turn_id, str):
                self._stream_turn_id = turn_id
            return None

        if frame_type == "stream_token":
            token = frame.get("token")
            if isinstance(token, str):
                self._tokens.append(token)
            return None

        if not is_terminal_frame(frame):
            return None

        self._done = True
        turn_id = terminal_turn_id(frame)
        elapsed = (time.monotonic() if now is None else now) - self._started_at
        return TurnOutcome(
            turn_index=self._spec.index,
            utterance=self._spec.text,
            terminal_type=str(frame_type),
            turn_id=turn_id,
            turn_id_matched=turn_id is not None and turn_id == self._stream_turn_id,
            text=terminal_text(frame),
            duration_ms=_as_int(frame.get("duration_ms")),
            tool_calls_used=_as_int(frame.get("tool_calls_used")),
            error_kind=frame.get("kind") if isinstance(frame.get("kind"), str) else None,
            token_count=len(self._tokens),
            elapsed_s=elapsed,
        )


def _as_int(value: Any) -> int | None:
    """Coerce a frame field to int, or None when it is not an integer."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def summarise_outcomes(outcomes: list[TurnOutcome]) -> dict[str, int]:
    """Count terminals by type.

    This is the count A1 turns on: `assert_artifacts.py` calls fewer than five
    `stream_complete` terminals INCONCLUSIVE rather than FAIL, because a turn
    that never completed never exercised `_record_turn_event`
    (`backend/chat/conversation_handler.py:2279`) at all.

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
        f"(backend/knowledge/eval_isolation.py:308).\n"
        f"  Guard said: {reason}\n"
        "  If the target IS the smoke stack (host port 8003, published by "
        "docker-compose.live-path-smoke.yml:160), the allowlist arm is what "
        "refused it: it defaults to the DEV endpoints "
        "(DEFAULT_DEV_WS_ENDPOINTS, eval_isolation.py:79). Export exactly:\n"
        "      MIST_DEV_WS_HOSTS=localhost:8003,127.0.0.1:8003\n"
        "  The denylist arm (eval_isolation.py:120-126) refuses localhost:8001, "
        "127.0.0.1:8001 and mist-backend:8001 -- the LIVE backend -- and no "
        "environment variable widens it. If that is what refused you, the URL is "
        "wrong; do not try to work around it."
    )


def guard_ws_target(ws_url: str) -> None:
    """Refuse a target that is, or might be, the live backend.

    Called before the socket is opened. The guard is a pure function and makes
    no network call of its own (`backend/knowledge/eval_isolation.py:318-320`),
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

    The handshake is the FIRST frame the server sends
    (`backend/server.py:788-799`), but this loops rather than reading exactly
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
    """Send one utterance and read frames until its turn terminates.

    Args:
        ws: The open connection.
        spec: The turn to send.
        transcript: Where to record what arrives.
        timeout: Seconds to wait for a terminal frame.

    Returns:
        The turn's outcome.

    Raises:
        TimeoutError: If no terminal frame arrives within `timeout`.
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
                f"turn {spec.index} produced no terminal frame within {timeout:.0f}s "
                f"(saw {len(recorder.streamed_text)} streamed characters)"
            )
        payload = await asyncio.wait_for(ws.recv(), timeout=remaining)
        frame = parse_frame(payload)
        if frame is None:
            kind = "binary_frame" if isinstance(payload, (bytes, bytearray)) else "undecodable"
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
        turn_timeout: Seconds allowed per turn for a terminal frame.

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
            "(docker-compose.live-path-smoke.yml:370); Whisper still loads with "
            "TTS off (backend/voice_models/model_manager.py:119-120), so a cold "
            "start is not instant.",
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
        print("[smoke] The server minted that id (backend/server.py:787). Every")
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
        # Clean close, always. The server's `finally`
        # (backend/server.py:1008-1017) writes the session note on disconnect;
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
        help="Seconds to wait per turn for a terminal frame (default: 300).",
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
