"""T2b part B: turns answered without knowledge integration are counted and warned about.

`ModelManager.generate_llm_response` is exercised for real; `__init__` is
bypassed (it loads Whisper/TTS and builds a live KnowledgeIntegration), the same
approach `tests/unit/test_session_id_propagation.py` takes.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from backend.extraction_backlog import telemetry
from backend.request_context import current_request_id
from backend.voice_models.model_manager import ModelManager

TELEMETRY_LOGGER = "backend.extraction_backlog.telemetry"


class _FakeProvider:
    def __init__(self) -> None:
        self.requests = []

    def generate_sync(self, request, stream=True):
        self.requests.append(request)
        yield SimpleNamespace(content="Hello")
        yield SimpleNamespace(content=" there.")


class _FakeKnowledge:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self.calls = []

    def is_enabled(self) -> bool:
        return self._enabled

    def generate_response_streaming(self, user_text, event_loop=None):
        self.calls.append(user_text)
        yield "From the graph."


def _manager(knowledge) -> tuple[ModelManager, _FakeProvider]:
    mm = object.__new__(ModelManager)
    provider = _FakeProvider()
    mm.knowledge = knowledge
    mm._llm_provider = provider
    mm.event_loop = None
    return mm, provider


@pytest.fixture(autouse=True)
def _zero_unrecorded():
    telemetry.reset_unrecorded_turns()
    yield
    telemetry.reset_unrecorded_turns()


def _warnings(caplog) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == TELEMETRY_LOGGER and r.levelno == logging.WARNING
    ]


class TestFallbackBranch:
    @pytest.mark.parametrize(
        "knowledge", [None, _FakeKnowledge(enabled=False)], ids=["no_knowledge", "knowledge_off"]
    )
    def test_each_fallback_turn_counts_once_and_warns_once(self, knowledge, caplog):
        # Arrange
        mm, provider = _manager(knowledge)
        token = current_request_id.set("turn-41")

        # Act
        try:
            with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
                first = "".join(mm.generate_llm_response("hi"))
                second = "".join(mm.generate_llm_response("and again"))
        finally:
            current_request_id.reset(token)

        # Assert
        assert first == second == "Hello there."
        assert len(provider.requests) == 2
        assert telemetry.unrecorded_turns() == 2
        records = _warnings(caplog)
        assert len(records) == 2
        assert all("request_id=turn-41" in r.getMessage() for r in records)
        assert "unrecorded_total=2" in records[-1].getMessage()

    def test_a_missing_request_id_is_logged_as_a_dash(self, caplog):
        mm, _provider = _manager(None)
        token = current_request_id.set(None)
        try:
            with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
                list(mm.generate_llm_response("hi"))
        finally:
            current_request_id.reset(token)

        assert "request_id=- " in _warnings(caplog)[0].getMessage()


class TestKnowledgeBranch:
    def test_the_knowledge_path_neither_counts_nor_warns(self, caplog):
        knowledge = _FakeKnowledge(enabled=True)
        mm, provider = _manager(knowledge)

        with caplog.at_level(logging.WARNING, logger=TELEMETRY_LOGGER):
            text = "".join(mm.generate_llm_response("what do I use?"))

        assert text == "From the graph."
        assert knowledge.calls == ["what do I use?"]
        assert provider.requests == []
        assert telemetry.unrecorded_turns() == 0
        assert _warnings(caplog) == []
