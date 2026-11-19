"""The process's realtime sessions: routing, log streaming, voice activity and wake arbitration.

One registry per process. It builds each `RealtimeSession` and is its host: it forgets a session
once the session has committed its turn, gates log streaming on the union of what subscribers
asked for, joins every session's recently played text for the addressee gate's echo guard, and
owns the one `WakeArbiter` every controller shares, counting voice sessions only, so a connected
client that never started voice does not make a lone session wait out the window.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Protocol

from .session import Endpoint, EndpointKind, RealtimeSession, SessionServices, SessionSocket
from .turn import WakeArbiter

logger = logging.getLogger(__name__)

CallLater = Callable[[float, Callable[[], None]], object]


class LogGate(Protocol):
    """The streaming log handler's switches (`backend/log_handler.py`)."""

    def set_streaming(self, enabled: bool, levels: list[str] | None) -> None:
        """Stream records at `levels` (None: every level), or stop streaming."""
        ...

    def set_logger_level(self, logger_name: str, level: str) -> None:
        """Gate one logger's streamed records at `level`."""
        ...


def _loop_call_later(delay: float, callback: Callable[[], None]) -> object:
    return asyncio.get_running_loop().call_later(delay, callback)


class SessionRegistry:
    """Every live session, and what they share.

    Args:
        services: What every session is built with.
        log_gate: The streaming log handler, or None when log streaming is not wired.
        call_later: The arbiter's timer; the running loop's `call_later` by default.
    """

    def __init__(
        self,
        services: SessionServices,
        *,
        log_gate: LogGate | None = None,
        call_later: CallLater | None = None,
    ) -> None:
        self._services = services
        self._log_gate = log_gate
        self._sessions: dict[str, RealtimeSession] = {}
        self._log_levels: dict[str, frozenset[str] | None] = {}
        self._listeners: list[Callable[[bool], None]] = []
        self._activity = False
        self.arbiter = WakeArbiter(
            window_ms=services.settings.barge_in.wake_arbitration_ms,
            sessions=self.voice_session_count,
            call_later=call_later or _loop_call_later,
        )

    # -- Sessions ------------------------------------------------------------------------------

    def create_session(
        self,
        *,
        session_id: str,
        websocket: SessionSocket,
        endpoint: Endpoint,
        endpoint_kind: EndpointKind | None = None,
    ) -> RealtimeSession:
        """Build a session for an accepted socket and register it; the caller awaits `run`.

        `endpoint` is `desktop` or a room device's name. `endpoint_kind` defaults to `desktop`
        for the `desktop` endpoint and to `room` for any other, whose session accepts only the
        device's message types (`ROOM_CLIENT_MESSAGE_TYPES`).
        """
        session = RealtimeSession(
            session_id=session_id,
            websocket=websocket,
            endpoint=endpoint,
            endpoint_kind=endpoint_kind,
            services=self._services,
            host=self,
        )
        self.register(session)
        return session

    def register(self, session: RealtimeSession) -> None:
        """Add a session; its id must not be live already."""
        if session.session_id in self._sessions:
            raise ValueError(f"session {session.session_id} is already registered")
        self._sessions[session.session_id] = session

    def unregister(self, session: RealtimeSession) -> None:
        """Forget a session and its log subscription."""
        if self._sessions.get(session.session_id) is not session:
            return
        del self._sessions[session.session_id]
        if self._log_levels.pop(session.session_id, False) is not False:
            self._update_log_gate()
        self._recheck_activity()

    def session_closed(self, session: RealtimeSession) -> None:
        """The session committed its turn and stopped."""
        self.unregister(session)

    def count(self) -> int:
        """How many sessions are connected."""
        return len(self._sessions)

    def voice_session_count(self) -> int:
        """How many sessions have voice on: the arbiter's session count."""
        return sum(1 for session in self._sessions.values() if session.voice_on)

    def endpoint_present(self, endpoint: Endpoint) -> bool:
        """Whether a session bound to `endpoint` is connected."""
        return any(session.endpoint == endpoint for session in self._sessions.values())

    def endpoint_of(self, session_id: str) -> Endpoint | None:
        """The endpoint a live session is bound to, or None for one that is not connected."""
        session = self._sessions.get(session_id)
        return None if session is None else session.endpoint

    # -- Outbound ------------------------------------------------------------------------------

    def send_to(self, session_id: str, message: object) -> bool:
        """Queue a message on one session's `control` lane; False when it is not connected."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.send_control(message)
        return True

    def broadcast(self, message: object, *, endpoint: Endpoint | None = None) -> None:
        """Queue a process-wide message on every session's `bulk` lane, or one endpoint's."""
        for session in list(self._sessions.values()):
            if endpoint is None or session.endpoint == endpoint:
                session.send_bulk(message)

    def broadcast_voice_state(self) -> None:
        """Send each session its own `voice_state`, after a power level or engine change."""
        for session in list(self._sessions.values()):
            session.send_voice_state()

    # -- Logs ----------------------------------------------------------------------------------

    def publish_log(self, line: str, level: str) -> None:
        """Queue a log line on the `bulk` lane of each session subscribed at `level`."""
        level = level.upper()
        for session_id, levels in self._log_levels.items():
            if levels is None or level in levels:
                session = self._sessions.get(session_id)
                if session is not None:
                    session.send_bulk(line)

    def set_log_subscription(
        self, session: RealtimeSession, levels: list[str] | None, enabled: bool
    ) -> None:
        """Record a session's `subscribe_logs`, and gate the handler on every subscription."""
        if enabled:
            self._log_levels[session.session_id] = (
                None if levels is None else frozenset(level.upper() for level in levels)
            )
        else:
            self._log_levels.pop(session.session_id, None)
        self._update_log_gate()

    def set_logger_level(self, logger_name: str, level: str) -> None:
        """Apply a session's `log_config` to the handler."""
        if self._log_gate is not None:
            self._log_gate.set_logger_level(logger_name, level)

    def _update_log_gate(self) -> None:
        gate = self._log_gate
        if gate is None:
            return
        subscriptions = list(self._log_levels.values())
        if not subscriptions:
            gate.set_streaming(False, None)
        elif any(levels is None for levels in subscriptions):
            gate.set_streaming(True, None)
        else:
            union: set[str] = set()
            for levels in subscriptions:
                union.update(levels or ())
            gate.set_streaming(True, sorted(union))

    # -- Echo text -----------------------------------------------------------------------------

    def recent_tts_text(self, window_ms: int) -> str:
        """Every session's text played within `window_ms`, so one room's speech is not a wake."""
        texts = (session.recent_tts_text(window_ms) for session in self._sessions.values())
        return " ".join(text for text in texts if text)

    # -- Voice activity ------------------------------------------------------------------------

    def voice_activity(self) -> bool:
        """Whether any session has an open utterance or a held, thinking or speaking turn."""
        return any(session.voice_activity for session in self._sessions.values())

    def subscribe_voice_activity(self, callback: Callable[[bool], None]) -> Callable[[], None]:
        """Call `callback` with each change of `voice_activity`; returns the unsubscribe."""
        self._listeners.append(callback)

        def unsubscribe() -> None:
            if callback in self._listeners:
                self._listeners.remove(callback)

        return unsubscribe

    def voice_activity_changed(self, session: RealtimeSession) -> None:
        """A session's activity may have changed; tell the listeners if the total did."""
        self._recheck_activity()

    def _recheck_activity(self) -> None:
        active = self.voice_activity()
        if active == self._activity:
            return
        self._activity = active
        for callback in list(self._listeners):
            callback(active)
