"""Stage 1.5 subject-scope classifier for the extraction pipeline.

Inserts a terse LLM call between pre-processing (Stage 1) and ontology
extraction (Stage 2). Tags each utterance with one of:

  - user-scope   -- a first-person claim about the user, their work, or
                    their world. ("I learned Python")
  - system-scope -- a claim about MIST itself (identity, traits,
                    capabilities, preferences). ("MIST uses LanceDB")
  - third-party  -- a claim about some external entity that is neither
                    the user nor MIST. ("Slalom is a consulting firm")
  - unknown      -- classification failed or timed out. The pipeline
                    continues with no scope gating.

The scope is written to ``PreProcessedInput.metadata["subject_scope"]`` and
``PreProcessedInput.metadata["subject_scope_confidence"]`` so Stage 2 can
weight the extraction prompt accordingly. Closes Bug I/J extraction drift.

Target latency: <150ms at temperature 0 with max_tokens=96.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from backend.interfaces import LLMProvider
from backend.knowledge.config import ScopeClassifierConfig
from backend.knowledge.extraction.preprocessor import PreProcessedInput
from backend.llm.instrumented_provider import llm_call_context
from backend.llm.models import LLMRequest

logger = logging.getLogger(__name__)


ScopeLabel = Literal["user-scope", "system-scope", "third-party", "unknown"]

_VALID_SCOPES: frozenset[str] = frozenset({"user-scope", "system-scope", "third-party", "unknown"})


SCOPE_CLASSIFIER_SYSTEM_PROMPT = """You are a subject-scope classifier for a knowledge extraction pipeline. Your only job is to label each utterance by WHO the claim is about.

Output ONLY valid JSON. No explanations. No markdown.

## SCOPES (pick exactly one)

- "user-scope": First-person claims the speaker makes about themselves, their work, their life, or their world. The subject is the user. Examples: "I learned Python", "My team uses React", "I prefer Vim".

- "system-scope": Claims about MIST itself -- its identity, traits, capabilities, preferences, or implementation. The subject is MIST (the AI). Examples: "MIST uses LanceDB for vector search", "MIST is implemented with llama.cpp", "MIST is curious by nature".

- "third-party": Claims about external entities that are neither the user nor MIST -- companies, public figures, public technologies in the abstract, news, etc. Examples: "Slalom is a consulting firm", "Rust is memory-safe", "OpenAI released GPT-5".

## OUTPUT

{"scope": "user-scope" | "system-scope" | "third-party", "confidence": 0.0-1.0, "reasoning": "<=15 words"}

Confidence: 0.9+ for unambiguous, 0.6-0.8 when mixed, below 0.5 for genuinely ambiguous. If the utterance mentions both user and MIST, pick the dominant subject."""


@dataclass(frozen=True, slots=True)
class ScopeResult:
    """Result of Stage 1.5 subject-scope classification.

    Immutable by design -- downstream stages only read fields.
    """

    scope: ScopeLabel
    confidence: float
    reasoning: str | None = None
    elapsed_ms: float = 0.0


class SubjectScopeClassifier:
    """Classifies an utterance's subject scope via a terse LLM call.

    Accepts an ``LLMProvider`` (Protocol) so tests can inject a ``FakeLLM``
    without touching factories. On any error -- LLM exception, timeout, or
    unparsable JSON -- returns ``ScopeResult(scope="unknown", confidence=0.0,
    reasoning="classification_failed")``. The pipeline never fails closed on
    classifier error; Stage 2 runs with scope "unknown".
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: ScopeClassifierConfig | None = None,
    ) -> None:
        """Initialize the scope classifier.

        Args:
            llm: LLM provider for the classification call. Any implementation
                of the LLMProvider Protocol satisfies this (production uses
                StreamingLLMProvider; tests use FakeLLM).
            config: Scope classifier configuration. When None, defaults to
                ``ScopeClassifierConfig()`` (enabled, temperature 0.0,
                max_tokens 96, timeout 5s).
        """
        self._llm = llm
        self._config = config or ScopeClassifierConfig()

    async def classify(self, pre_processed: PreProcessedInput) -> ScopeResult:
        """Classify the subject scope of a pre-processed utterance.

        Runs a single LLM call with JSON-mode output and parses the
        ``{"scope": ..., "confidence": ..., "reasoning": ...}`` response.
        Wrapped in ``llm_call_context(call_site="extraction.scope_classifier")``
        so Cluster 5 observability tags the call.

        Args:
            pre_processed: Output from Stage 1 PreProcessor. Reads
                ``original_text``; ignores ``conversation_context`` so the
                classification is stable across turn position.

        Returns:
            ScopeResult with scope, confidence in [0.0, 1.0], optional
            reasoning string, and elapsed wall-clock time in milliseconds.
            Returns ``ScopeResult(scope="unknown", confidence=0.0,
            reasoning="classification_failed", elapsed_ms=...)`` on any
            failure path.
        """
        start = time.perf_counter()

        user_message = f'Utterance: "{pre_processed.original_text}"\n\nOutput:'
        request = LLMRequest(
            messages=[
                {"role": "system", "content": SCOPE_CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            json_mode=True,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

        try:
            with llm_call_context(call_site="extraction.scope_classifier"):
                response = await asyncio.wait_for(
                    self._llm.invoke(request),
                    timeout=self._config.timeout_seconds,
                )
            raw = response.content or ""
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning(
                "Scope classifier timed out after %.1fms (limit %.1fs) for utterance=%r",
                elapsed,
                self._config.timeout_seconds,
                pre_processed.original_text[:120],
            )
            return ScopeResult(
                scope="unknown",
                confidence=0.0,
                reasoning="classification_failed",
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.warning("Scope classifier LLM call failed after %.1fms: %s", elapsed, exc)
            return ScopeResult(
                scope="unknown",
                confidence=0.0,
                reasoning="classification_failed",
                elapsed_ms=elapsed,
            )

        scope, confidence, reasoning = self._parse_output(raw)
        elapsed = (time.perf_counter() - start) * 1000

        if scope == "unknown":
            logger.debug("Scope classifier could not parse output (%.1fms): %r", elapsed, raw[:200])
        else:
            logger.debug(
                "Scope classifier: scope=%s confidence=%.2f elapsed=%.1fms",
                scope,
                confidence,
                elapsed,
            )

        return ScopeResult(
            scope=scope,
            confidence=confidence,
            reasoning=reasoning,
            elapsed_ms=elapsed,
        )

    @staticmethod
    def _parse_output(raw: str) -> tuple[ScopeLabel, float, str | None]:
        """Parse the classifier's JSON output.

        Attempts direct JSON parse, then falls back to regex-extracting the
        first JSON object from the string. Returns ``("unknown", 0.0,
        "classification_failed")`` on any parse failure or invalid schema.
        """
        if not raw or not raw.strip():
            return "unknown", 0.0, "classification_failed"

        data: dict | None = None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                    if isinstance(parsed, dict):
                        data = parsed
                except json.JSONDecodeError:
                    data = None

        if data is None:
            return "unknown", 0.0, "classification_failed"

        scope_raw = data.get("scope")
        if not isinstance(scope_raw, str) or scope_raw not in _VALID_SCOPES:
            return "unknown", 0.0, "classification_failed"

        confidence_raw = data.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reasoning_raw = data.get("reasoning")
        reasoning = reasoning_raw if isinstance(reasoning_raw, str) else None

        # Literal-narrowing: scope_raw is one of the four valid literals.
        return scope_raw, confidence, reasoning  # type: ignore[return-value]
