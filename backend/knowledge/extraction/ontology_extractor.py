"""Ontology-constrained knowledge extractor.

Stage 2: Single LLM call with ontology-constrained prompt. Replaces the
previous LLMGraphTransformer + PropertyEnricher two-pass approach.
Runs against the configured StreamingLLMProvider (Gemma 4 E4B via
llama-server in production) with JSON-structured output.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from backend.errors import ExtractionValidationError
from backend.interfaces import LLMProvider
from backend.llm.models import LLMRequest

if TYPE_CHECKING:
    from backend.knowledge.curation.graph_writer import SourceMetadata

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.extraction.preprocessor import PreProcessedInput
from backend.knowledge.extraction.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
from backend.knowledge.ontologies import (
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
)

logger = logging.getLogger(__name__)


def render_extraction_messages(pre_processed: PreProcessedInput) -> list[dict]:
    """Render the Stage 2 chat messages for a pre-processed utterance.

    Pure function -- no I/O, no LLM call. Shared by the in-process
    `OntologyConstrainedExtractor` and the extraction service.

    Args:
        pre_processed: Output from Stage 1 PreProcessor, with Stage 1.5's
            `subject_scope` already written into `metadata` when enabled.

    Returns:
        A two-message list: system prompt (with reference date substituted),
        then the user message (context, subject scope, utterance).
    """
    context_str = (
        "\n".join(pre_processed.conversation_context)
        if pre_processed.conversation_context
        else "(no prior context)"
    )

    system_prompt = EXTRACTION_SYSTEM_PROMPT.format(
        reference_date=pre_processed.reference_date.strftime("%Y-%m-%d"),
    )

    # `subject_scope` is written by Stage 1.5 SubjectScopeClassifier into
    # pre_processed.metadata["subject_scope"]. Falls back to "unknown" so
    # the template substitution never fails closed when Stage 1.5 is
    # disabled or missing.
    subject_scope = pre_processed.metadata.get("subject_scope", "unknown")
    user_message = EXTRACTION_USER_TEMPLATE.format(
        context=context_str,
        utterance=pre_processed.original_text,
        subject_scope=subject_scope,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


def _try_parse_json_object(raw: str) -> dict | None:
    """Direct JSON parse, then regex-extract-first-object fallback.

    Returns None (never raises) when both strategies fail.
    """
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _is_well_shaped_extraction(result: dict) -> bool:
    """True iff `result` has list-typed `entities` and `relationships` keys."""
    return isinstance(result.get("entities"), list) and isinstance(
        result.get("relationships"), list
    )


def parse_extraction_output(raw: str, *, strict: bool) -> dict:
    """Parse Stage 2 LLM output into `{"entities": [...], "relationships": [...]}`.

    Non-strict (`strict=False`, the in-process pipeline's historical
    behavior): falls back to an empty result -- `{"entities": [],
    "relationships": []}` -- on empty input, unparsable JSON, or a parsed
    value that is not well-shaped. The pipeline never raises on bad LLM
    output; Stage 2 already logs and degrades to "nothing extracted".

    Strict (`strict=True`, the extraction service): raises instead of
    degrading, so a caller can tell "valid output with zero entities"
    apart from "unparsable output" and report a typed failure -- never
    silently cache an empty extraction as a real "nothing found" decision.

    Args:
        raw: Raw string output from the LLM.
        strict: When True, raise on empty/unparsable/malformed input
            instead of returning an empty result.

    Returns:
        Parsed dict with "entities" and "relationships" keys (non-strict
        may return other keys the model emitted; strict enforces list
        typing on the two required keys).

    Raises:
        ExtractionValidationError: `strict=True` and `raw` is empty, not
            parseable as a JSON object, or missing list-typed `entities`/
            `relationships` keys.
    """
    if not raw or not raw.strip():
        if strict:
            raise ExtractionValidationError("Empty extraction output from LLM")
        logger.warning("Empty LLM output, returning empty result")
        return {"entities": [], "relationships": []}

    result = _try_parse_json_object(raw)
    if result is None:
        if strict:
            raise ExtractionValidationError(
                f"Could not parse extraction output as a JSON object: {raw[:200]!r}"
            )
        logger.warning("Failed to parse LLM output as JSON: %s", raw[:200])
        return {"entities": [], "relationships": []}

    if strict and not _is_well_shaped_extraction(result):
        raise ExtractionValidationError(
            f"Extraction output missing list-typed entities/relationships: {raw[:200]!r}"
        )

    return result


@dataclass
class ExtractionResult:
    """Result of a single extraction pass.

    Contains structured entities and relationships parsed from the LLM
    output, plus diagnostic metadata.
    """

    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    raw_llm_output: str = ""
    extraction_time_ms: float = 0.0
    source_utterance: str = ""
    source_metadata: SourceMetadata | None = None


class OntologyConstrainedExtractor:
    """Single-LLM-call extractor with ontology constraints.

    Replaces the LLMGraphTransformer + PropertyEnricher pipeline with
    a single structured prompt that enforces allowed entity types,
    relationship types, and property schemas.
    """

    # Derived from the ontology registry (Inv-A6): hand-maintained mirrors
    # drifted silently as types were added across Cluster 1 / post-MVP /
    # v1.1.0. One source of truth; the prompt enumeration is the only
    # remaining mirror and is pinned by the extraction-version drift guard.
    ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset(EXTRACTABLE_NODE_TYPES)

    ALLOWED_RELATIONSHIP_TYPES: frozenset[str] = frozenset(EXTRACTABLE_RELATIONSHIP_TYPES)

    def __init__(self, config: KnowledgeConfig, llm: LLMProvider) -> None:
        """Initialize the extractor.

        Args:
            config: Knowledge system configuration.
            llm: LLM provider for structured extraction calls.
        """
        self.config = config
        self._llm = llm

    async def extract(self, pre_processed: PreProcessedInput) -> ExtractionResult:
        """Run ontology-constrained extraction via a single LLM call.

        Args:
            pre_processed: Output from PreProcessor containing the utterance,
                conversation context, and reference date.

        Returns:
            ExtractionResult with parsed entities and relationships.
        """
        messages = render_extraction_messages(pre_processed)

        start_time = time.perf_counter()
        try:
            request = LLMRequest(
                messages=messages,
                json_mode=True,
                temperature=self.config.llm.temperature,
                max_tokens=2048,
            )
            from backend.llm.instrumented_provider import llm_call_context

            with llm_call_context(call_site="extraction.ontology"):
                response = await self._llm.invoke(request)
            raw_output = response.content
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("LLM extraction call failed after %.1fms: %s", elapsed_ms, e)
            return ExtractionResult(
                raw_llm_output="",
                extraction_time_ms=elapsed_ms,
                source_utterance=pre_processed.original_text,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info("LLM extraction completed in %.1fms", elapsed_ms)

        # Parse the JSON output. Non-strict: falls back to an empty result
        # on unparsable/malformed output rather than raising -- the
        # in-process pipeline's historical behavior.
        parsed = parse_extraction_output(raw_output, strict=False)

        entities = parsed.get("entities", [])
        relationships = parsed.get("relationships", [])

        logger.info(
            "Extracted %d entities and %d relationships from: %s",
            len(entities),
            len(relationships),
            pre_processed.original_text[:80],
        )

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            raw_llm_output=raw_output,
            extraction_time_ms=elapsed_ms,
            source_utterance=pre_processed.original_text,
        )
