"""Tests for Stage 1.5 SubjectScopeClassifier.

Follows tests/CLAUDE.md: explicit fakes (FakeLLM) via DI, no MagicMock,
arrange/act/assert structure, class grouping by concern, descriptive
snake_case names.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime

import pytest

from backend.knowledge.config import ScopeClassifierConfig
from backend.knowledge.extraction.preprocessor import PreProcessedInput
from backend.knowledge.extraction.scope_classifier import (
    ScopeResult,
    SubjectScopeClassifier,
)
from backend.llm.instrumented_provider import get_llm_call_context
from backend.llm.models import LLMRequest, LLMResponse
from backend.llm.provider import StreamingLLMProvider
from tests.mocks.ollama import FakeLLM

# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

REF_DATE = datetime(2026, 4, 21)


def build_pre_processed(utterance: str) -> PreProcessedInput:
    """Build a PreProcessedInput with no conversation context."""
    return PreProcessedInput(
        original_text=utterance,
        resolved_text=utterance,
        conversation_context=[],
        reference_date=REF_DATE,
        turn_index=0,
        metadata={},
    )


def _json_response(scope: str, confidence: float, reasoning: str = "short note") -> str:
    """Serialize the expected classifier output shape."""
    return (
        '{"scope": "' + scope + '", '
        '"confidence": ' + str(confidence) + ", "
        '"reasoning": "' + reasoning + '"}'
    )


class _CaptureLLM(StreamingLLMProvider):
    """FakeLLM-style recorder that also captures the llm_call_context at invoke time.

    The real InstrumentedStreamingLLMProvider wrapper reads the ContextVar;
    for unit tests we expose it directly so we can assert the classifier
    set `call_site="extraction.scope_classifier"` inside the context.
    """

    model: str = "capture-llm"

    def __init__(self, *, response_content: str) -> None:
        self._response = response_content
        self.calls: list[LLMRequest] = []
        self.captured_contexts: list[dict] = []

    async def generate(self, request, *, stream: bool = False):  # pragma: no cover
        yield LLMResponse(content=self._response, partial=False)

    def generate_sync(self, request, *, stream: bool = False):  # pragma: no cover
        yield LLMResponse(content=self._response, partial=False)

    async def invoke(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        self.captured_contexts.append(get_llm_call_context())
        return LLMResponse(content=self._response, partial=False)


# ---------------------------------------------------------------------------
# Classification happy paths
# ---------------------------------------------------------------------------


class TestClassificationScopes:
    """Verify the classifier surfaces each of the three real scopes."""

    @pytest.mark.asyncio
    async def test_classifies_user_utterance_as_user_scope(self):
        # Arrange
        llm = FakeLLM(default_response=_json_response("user-scope", 0.95))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("I learned Python last year.")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "user-scope"
        assert result.confidence == pytest.approx(0.95)
        assert result.reasoning == "short note"

    @pytest.mark.asyncio
    async def test_classifies_mist_self_reference_as_system_scope(self):
        # Arrange
        llm = FakeLLM(default_response=_json_response("system-scope", 0.9))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("MIST uses LanceDB for vector search.")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "system-scope"
        assert result.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_classifies_external_entity_as_third_party(self):
        # Arrange
        llm = FakeLLM(default_response=_json_response("third-party", 0.8))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("Slalom is a consulting firm.")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "third-party"
        assert result.confidence == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Failure paths -- never raise, always "unknown"
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Classifier must never propagate exceptions. Failures collapse to unknown."""

    @pytest.mark.asyncio
    async def test_returns_unknown_on_malformed_json(self):
        # Arrange
        llm = FakeLLM(default_response="this is not json at all {{{")
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("Something ambiguous.")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "unknown"
        assert result.confidence == 0.0
        assert result.reasoning == "classification_failed"

    @pytest.mark.asyncio
    async def test_returns_unknown_on_invalid_scope_value(self):
        """LLM returns JSON but with a scope outside the allowed set."""
        # Arrange
        llm = FakeLLM(default_response='{"scope": "cosmic", "confidence": 0.7, "reasoning": "x"}')
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("x")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "unknown"
        assert result.reasoning == "classification_failed"

    @pytest.mark.asyncio
    async def test_returns_unknown_on_llm_exception(self):
        """Raising LLM invoke must not crash the classifier."""

        # Arrange
        class RaisingLLM(StreamingLLMProvider):
            model = "raiser"

            async def generate(self, request, *, stream: bool = False):  # pragma: no cover
                raise RuntimeError("connection refused")
                yield  # noqa: unreachable

            def generate_sync(self, request, *, stream: bool = False):  # pragma: no cover
                raise RuntimeError("connection refused")
                yield  # noqa: unreachable

            async def invoke(self, request):
                raise RuntimeError("connection refused")

        classifier = SubjectScopeClassifier(llm=RaisingLLM())
        pre = build_pre_processed("I use Rust.")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "unknown"
        assert result.confidence == 0.0
        assert result.reasoning == "classification_failed"

    @pytest.mark.asyncio
    async def test_returns_unknown_on_timeout(self):
        """An invoke that exceeds the configured timeout must yield unknown."""

        # Arrange
        class HangingLLM(StreamingLLMProvider):
            model = "hanger"

            async def generate(self, request, *, stream: bool = False):  # pragma: no cover
                await asyncio.sleep(10)
                yield LLMResponse(content="{}", partial=False)

            def generate_sync(self, request, *, stream: bool = False):  # pragma: no cover
                yield LLMResponse(content="{}", partial=False)

            async def invoke(self, request):
                await asyncio.sleep(10)
                return LLMResponse(content="{}", partial=False)

        config = ScopeClassifierConfig(timeout_seconds=0.05)
        classifier = SubjectScopeClassifier(llm=HangingLLM(), config=config)
        pre = build_pre_processed("anything")

        # Act
        result = await classifier.classify(pre)

        # Assert
        assert result.scope == "unknown"
        assert result.reasoning == "classification_failed"


# ---------------------------------------------------------------------------
# Request shape + observability
# ---------------------------------------------------------------------------


class TestRequestContract:
    """The LLMRequest sent by the classifier has the spec-mandated fields."""

    @pytest.mark.asyncio
    async def test_forwards_max_tokens_96_to_llm_request(self):
        # Arrange
        llm = FakeLLM(default_response=_json_response("user-scope", 0.9))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("I use Python.")

        # Act
        await classifier.classify(pre)

        # Assert
        assert len(llm.calls) == 1
        request = llm.calls[0]
        assert request.max_tokens == 96
        assert request.temperature == 0.0
        assert request.json_mode is True

    @pytest.mark.asyncio
    async def test_forwards_custom_max_tokens_from_config(self):
        """Config overrides default max_tokens."""
        # Arrange
        llm = FakeLLM(default_response=_json_response("user-scope", 0.9))
        classifier = SubjectScopeClassifier(llm=llm, config=ScopeClassifierConfig(max_tokens=128))
        pre = build_pre_processed("x")

        # Act
        await classifier.classify(pre)

        # Assert
        assert llm.calls[0].max_tokens == 128

    @pytest.mark.asyncio
    async def test_populates_elapsed_ms(self):
        # Arrange
        llm = FakeLLM(default_response=_json_response("user-scope", 0.9))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("I use Rust.")

        # Act
        result = await classifier.classify(pre)

        # Assert -- FakeLLM is ~instant but perf_counter delta must be >= 0
        assert result.elapsed_ms >= 0.0

    @pytest.mark.asyncio
    async def test_sends_system_prompt_with_scope_definitions(self):
        """The classifier's system message must explain the three scopes."""
        # Arrange
        llm = FakeLLM(default_response=_json_response("user-scope", 0.9))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("anything")

        # Act
        await classifier.classify(pre)

        # Assert
        messages = llm.calls[0].messages
        assert messages[0]["role"] == "system"
        system_text = messages[0]["content"]
        for label in ("user-scope", "system-scope", "third-party"):
            assert label in system_text

    @pytest.mark.asyncio
    async def test_llm_call_context_tagged_with_call_site(self):
        """Classifier wraps the invoke in llm_call_context(call_site=...)."""
        # Arrange
        llm = _CaptureLLM(response_content=_json_response("user-scope", 0.9))
        classifier = SubjectScopeClassifier(llm=llm)
        pre = build_pre_processed("I use Rust.")

        # Act
        await classifier.classify(pre)

        # Assert
        assert len(llm.captured_contexts) == 1
        assert llm.captured_contexts[0].get("call_site") == "extraction.scope_classifier"


# ---------------------------------------------------------------------------
# ScopeResult dataclass shape
# ---------------------------------------------------------------------------


class TestScopeResultDataclass:
    """ScopeResult is a frozen slotted dataclass -- enforce immutability."""

    def test_scope_result_is_frozen(self):
        # Arrange
        r = ScopeResult(scope="user-scope", confidence=0.9)

        # Act + Assert
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.scope = "system-scope"  # type: ignore[misc]

    def test_scope_result_defaults(self):
        # Arrange + Act
        r = ScopeResult(scope="unknown", confidence=0.0)

        # Assert
        assert r.reasoning is None
        assert r.elapsed_ms == 0.0
