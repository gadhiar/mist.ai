"""Reachability tests for the WebSocket session id (P6), on the realtime `/ws` path.

The defect these pin: `server.websocket_endpoint` minted a fresh UUID per connection and shipped
it in the `session_started` handshake, but nothing carried it inward, so every turn of every
connection landed in one session literally named `default`.

These tests assert the session id ARRIVES at the turn responder, driving the real chain:

    server.websocket_endpoint          (real)
      -> SessionRegistry.create_session (real)
      -> RealtimeSession                (real)
      -> TurnController                 (real)
      -> TurnResponder.open_turn        <- assertion point

Only the genuine I/O boundaries are faked: the WebSocket, the conversation handler that is the
runtime's responder, and the speech engines. The helpers here (`FakeWebSocket`,
`RecordingConversationHandler`, `install_runtime`, `run_endpoint`) are what the other `/ws`
tests drive the endpoint with.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from backend.realtime.budget import BargeInConfig, VoiceSettings
from backend.realtime.power import PowerGate
from backend.realtime.registry import LogGate, SessionRegistry
from backend.realtime.runtime import voice_capabilities
from backend.realtime.session import InferenceLanes, SessionServices, SpeechEngines
from backend.request_context import current_session_id
from tests.fakes.speech import (
    EnergyVad,
    FakeSynthesizer,
    ScriptedNameSpotter,
    ScriptedTranscriber,
    ScriptedWakeVerifier,
)
from tests.unit.realtime.test_session import CALIBRATION, FakeIngress, FakeSocket, voice_start
from tests.unit.realtime.turn_fakes import (
    TEST_PLAYBACK,
    TEST_SEGMENTER,
    FakeClock,
    FakeResponder,
    IdCounter,
    InlineLane,
    floor_placement,
    settle,
)

# -- The `/ws` rig, shared with the other endpoint tests -------------------------------------


class RecordingConversationHandler(FakeResponder):
    """The runtime's conversation handler: the turn responder, plus the server's own calls.

    The server reaches the handler for `end_session` on disconnect and for `decide_proposal`;
    every session reaches it as its `TurnResponder`. `end_session` is logged in the
    responder's `log`, so a test can read it against the turn's `commit:<outcome>`.
    """

    def __init__(self, *scripts: object, clock: FakeClock | None = None) -> None:
        super().__init__(*scripts, clock=clock)  # type: ignore[arg-type]
        self.ended_sessions: list[str | None] = []

    @property
    def session_ids(self) -> list[str]:
        """The session id of each turn the responder opened, in order."""
        return [request.session_id for request in self.requests]

    async def end_session(self, session_id: str | None = None) -> None:
        """Record which session ended: `None` (end every session) must be distinguishable."""
        self.ended_sessions.append(session_id)
        self.log.append("end_session")


class FakeWebSocket(FakeSocket):
    """The realtime fake socket plus the Starlette handshake surface the endpoint uses."""

    def __init__(self, origin: str | None = None) -> None:
        super().__init__()
        self.headers: dict[str, str] = {} if origin is None else {"origin": origin}
        self.accepted = False

    async def accept(self, headers: object = None) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, object]) -> None:
        await self._write(payload)

    @property
    def frames(self) -> list[dict[str, object]]:
        """Every JSON message sent, in order: the binary frames left out."""
        return [item for item in self.sent if isinstance(item, dict)]

    @property
    def handshake_session_id(self) -> str:
        """The session id the server minted and published to the client."""
        for frame in self.frames:
            if frame.get("type") == "session_started":
                return str(frame["session_id"])
        raise AssertionError(f"no session_started frame was sent; got {self.sent}")


@dataclass
class EndpointRig:
    """What `install_runtime` put in place of `lifespan`'s runtime and registry."""

    handler: RecordingConversationHandler
    registry: SessionRegistry
    engines: SpeechEngines
    ingresses: list[FakeIngress] = field(default_factory=list)
    clock: FakeClock = field(default_factory=FakeClock)


def install_runtime(
    monkeypatch,
    handler: RecordingConversationHandler | None = None,
    *,
    log_gate: LogGate | None = None,
    power_gate: PowerGate | None = None,
    llm_provider: object | None = None,
) -> EndpointRig:
    """Install a runtime and a session registry as `lifespan` leaves them, over fakes.

    The handler is both the runtime's `conversation_handler` and every session's responder, as
    in production; the speech engines are fakes on inline lanes. `power_gate` is every
    session's gate, and `llm_provider` the runtime's base provider, which the server's tiers
    read.
    """
    from backend import server

    handler = handler or RecordingConversationHandler()
    clock = handler.clock or FakeClock()
    handler.clock = clock
    ids = IdCounter()
    engines = SpeechEngines(
        vad=EnergyVad(),
        turn_end=None,
        transcriber=ScriptedTranscriber(["hello there"], ms_per_word=20),
        synthesizer=FakeSynthesizer(),
        spotter=ScriptedNameSpotter(hit_lag_ms=180, calibration=CALIBRATION),
        verifier=ScriptedWakeVerifier(),
    )
    ingresses: list[FakeIngress] = []

    def ingress_factory(**kwargs: object) -> FakeIngress:
        ingress = FakeIngress(kwargs)
        ingresses.append(ingress)
        return ingress

    services = SessionServices(
        responder=handler,  # type: ignore[arg-type]
        engines=engines,
        lanes=InferenceLanes(
            vad=InlineLane(),
            eot=InlineLane(),
            stt=InlineLane(),
            tts=InlineLane(),
            kws=InlineLane(),
            verify=InlineLane(),
        ),
        settings=VoiceSettings(
            playback=TEST_PLAYBACK,
            segmenter=TEST_SEGMENTER,
            barge_in=BargeInConfig(arm_delay_ms=0),
        ),
        placement=floor_placement,
        clock_factory=lambda: clock,
        utterance_ids=lambda: ids,
        ingress_factory=ingress_factory,
        on_capability_message=server._on_capability_message,
        power_gate=power_gate,
    )
    registry = SessionRegistry(services, log_gate=log_gate)
    runtime = SimpleNamespace(
        conversation_handler=handler,
        responder=handler,
        llm_provider=llm_provider,
        agent_available=True,
        engines=engines,
        services=services,
        gpu=None,
        voice_capabilities=lambda: voice_capabilities(engines),
    )
    monkeypatch.setattr(server, "runtime", runtime)
    monkeypatch.setattr(server, "session_registry", registry)
    monkeypatch.setattr(server, "log_handler", None)
    return EndpointRig(handler, registry, engines, ingresses, clock)


async def drain_capability_calls() -> None:
    """Wait until every capability call the server started has finished, then settle."""
    from backend import server

    while True:
        pending = [task for tasks in server._capability_tasks.values() for task in tasks]
        if not pending:
            break
        await asyncio.wait(pending, timeout=5)
    await settle()


class Connection:
    """One `/ws` connection served by the real endpoint as a task."""

    def __init__(self, ws: FakeWebSocket, task: asyncio.Task[None]) -> None:
        self.ws = ws
        self.task = task

    async def send(self, *messages: dict[str, object] | str) -> None:
        """Deliver each message, letting the session and any capability call finish it."""
        for message in messages:
            self.ws.text(message)
            await settle()
            await drain_capability_calls()

    async def leave(self) -> None:
        """Disconnect, and wait for the endpoint to return."""
        self.ws.disconnect()
        await asyncio.wait_for(self.task, timeout=5)


async def connect(ws: FakeWebSocket | None = None) -> Connection:
    """Open one connection on the installed runtime; the handshake is sent before this returns."""
    from backend import server

    ws = ws or FakeWebSocket()
    task = asyncio.get_running_loop().create_task(server.websocket_endpoint(ws))
    await settle()
    return Connection(ws, task)


async def run_endpoint(
    monkeypatch,
    handler: RecordingConversationHandler | None,
    inbound: list[dict[str, object]],
    *,
    ws: FakeWebSocket | None = None,
) -> FakeWebSocket:
    """Serve one connection that sends `inbound` and then disconnects; return its socket.

    Installs the runtime first when the test has not.
    """
    from backend import server

    if server.session_registry is None or handler is not None:
        install_runtime(monkeypatch, handler)
    connection = await connect(ws)
    await connection.send(*inbound)
    await connection.leave()
    return connection.ws


@pytest.fixture(autouse=True)
def reset_session_context():
    """Keep a leaked context var from one test satisfying the next."""
    token = current_session_id.set(None)
    yield
    current_session_id.reset(token)


@pytest.fixture(autouse=True)
def no_runtime(monkeypatch):
    """Each test installs its own runtime; none leaks from another module."""
    from backend import server

    monkeypatch.setattr(server, "runtime", None)
    monkeypatch.setattr(server, "session_registry", None)


# -- Disconnect ------------------------------------------------------------------------------


class TestDisconnectEndsOnlyItsOwnSession:
    """A disconnect must end THIS connection's session, not every session.

    `end_session(session_id=None)` ends every session the handler tracks: one client
    disconnecting would write every other open connection's vault note early and evict their
    paths. The assertion is on the value that ARRIVED, because the endpoint swallows the
    handler's failures from this call.
    """

    @pytest.mark.asyncio
    async def test_disconnect_scopes_end_session_to_the_handshake_id(self, monkeypatch):
        handler = RecordingConversationHandler()

        ws = await run_endpoint(monkeypatch, handler, [{"type": "text", "text": "hi"}])

        assert handler.ended_sessions == [ws.handshake_session_id], (
            "disconnect did not scope end_session to this connection's session. "
            f"Got {handler.ended_sessions!r}; None means 'end every tracked "
            "session', which ends other live connections' sessions too."
        )

    @pytest.mark.asyncio
    async def test_disconnect_never_ends_every_session(self, monkeypatch):
        """The specific regression: `None` must never reach `end_session`."""
        handler = RecordingConversationHandler()

        await run_endpoint(monkeypatch, handler, [{"type": "text", "text": "hi"}])

        assert (
            None not in handler.ended_sessions
        ), "end_session was called with None, which ends EVERY tracked session"


# -- Turns -----------------------------------------------------------------------------------


class TestSessionIdReachesHandler:
    """The id minted at the WS handshake must arrive at the responder."""

    @pytest.mark.asyncio
    async def test_text_turn_delivers_handshake_session_id(self, monkeypatch):
        handler = RecordingConversationHandler()

        ws = await run_endpoint(monkeypatch, handler, [{"type": "text", "text": "hi"}])

        assert handler.session_ids == [
            ws.handshake_session_id
        ], "session id from the session_started handshake never reached the responder"

    @pytest.mark.asyncio
    async def test_voice_turn_delivers_handshake_session_id(self, monkeypatch):
        """An utterance from the voice uplink opens its turn under the handshake id too."""
        rig = install_runtime(monkeypatch)
        connection = await connect()
        await connection.send(voice_start())
        ingress = rig.ingresses[-1]
        utterance = ingress.new_id()

        ingress.emit(ingress.started(utterance))
        ingress.emit(ingress.ended(utterance, "hello"))
        await settle()
        await connection.leave()

        assert [r.origin for r in rig.handler.requests] == ["voice"]
        assert rig.handler.session_ids == [connection.ws.handshake_session_id]

    @pytest.mark.asyncio
    async def test_separate_connections_do_not_share_one_session(self, monkeypatch):
        """Two connections produce two distinct session ids at the responder."""
        handler = RecordingConversationHandler()

        first = await run_endpoint(monkeypatch, handler, [{"type": "text", "text": "one"}])
        second = await run_endpoint(monkeypatch, handler, [{"type": "text", "text": "two"}])

        assert first.handshake_session_id != second.handshake_session_id
        assert handler.session_ids == [
            first.handshake_session_id,
            second.handshake_session_id,
        ], "connections collapsed into a shared session"
