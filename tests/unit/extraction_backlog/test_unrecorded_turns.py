"""T2b part B: turns answered without knowledge integration are counted and warned about.

The fallback moved from the deleted threaded voice path to `DirectResponder` (T12); these
are the same assertions against it. Each turn's stream counts once; opening and committing it
count nothing.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from backend.chat.stream_events import Complete, Token
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest
from backend.extraction_backlog import telemetry
from backend.realtime.degraded import FALLBACK_SYSTEM_PROMPT, DirectResponder
from backend.request_context import current_request_id

TELEMETRY_LOGGER = "backend.extraction_backlog.telemetry"

pytestmark = pytest.mark.asyncio


class _FakeProvider:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request, *, stream=False):
        self.requests.append(request)
        yield SimpleNamespace(content="Hello", partial=True)
        yield SimpleNamespace(content=" there.", partial=True)
        yield SimpleNamespace(content=None, partial=False)


@pytest.fixture(autouse=True)
def _zero_unrecorded():
    telemetry.reset_unrecorded_turns()
    yield
    telemetry.reset_unrecorded_turns()


def _warnings(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == TELEMETRY_LOGGER and r.levelno == logging.WARNING
    ]


def _request(text: str) -> TurnRequest:
    return TurnRequest(session_id="s1", user_text=text, origin="text", channel="text")


async def _turn(responder: DirectResponder, text: str) -> tuple[str, list[object]]:
    scope = responder.open_turn(_request(text))
    events = [event async for event in responder.stream_turn(scope)]
    tokens = "".join(e.text for e in events if isinstance(e, Token))
    return tokens, events


class TestFallbackTurns:
    async def test_each_fallback_turn_counts_once_and_warns_once(self, caplog):
        # Arrange
        provider = _FakeProvider()
        responder = DirectResponder(provider)
        token = current_request_id.set("turn-41")

        # Act
        try:
            with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
                first, events = await _turn(responder, "hi")
                second, _ = await _turn(responder, "and again")
        finally:
            current_request_id.reset(token)

        # Assert
        assert first == second == "Hello there."
        assert events[-1] == Complete(final_response="Hello there.")
        assert len(provider.requests) == 2
        assert telemetry.unrecorded_turns() == 2
        records = _warnings(caplog)
        assert len(records) == 2
        assert all("request_id=turn-41" in r.getMessage() for r in records)
        assert "unrecorded_total=2" in records[-1].getMessage()

    async def test_a_missing_request_id_is_logged_as_a_dash(self, caplog):
        responder = DirectResponder(_FakeProvider())
        token = current_request_id.set(None)
        try:
            with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
                await _turn(responder, "hi")
        finally:
            current_request_id.reset(token)

        assert "request_id=- " in _warnings(caplog)[0].getMessage()

    async def test_the_request_keeps_the_fallback_prompt_and_its_token_limit(self):
        provider = _FakeProvider()

        await _turn(DirectResponder(provider), "what is up")

        request = provider.requests[0]
        assert request.messages == [
            {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": "what is up"},
        ]
        assert request.max_tokens == 400
        assert request.top_p == 0.9
        assert FALLBACK_SYSTEM_PROMPT.startswith("You are M.I.S.T, a helpful voice assistant")
        assert FALLBACK_SYSTEM_PROMPT.endswith("thorough when needed.")


class TestOpenAndCommit:
    async def test_opening_and_committing_neither_count_nor_warn(self, caplog):
        responder = DirectResponder(_FakeProvider())

        with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
            scope = responder.open_turn(_request("hi"))
            result = responder.commit_turn(
                scope,
                TurnDelivery(
                    outcome="completed", delivered_text="", stt_model=None, tts_model=None
                ),
            )
            responder.prefetch("s1", "partial")

        assert result == CommitResult(recorded=False, event_id=None, error=None)
        assert scope.dispatch_gate.is_set()
        assert telemetry.unrecorded_turns() == 0
        assert _warnings(caplog) == []
