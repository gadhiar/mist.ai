"""Ontology-constrained knowledge extractor.

Stage 2: Single LLM call with ontology-constrained prompt. Replaces the
previous LLMGraphTransformer + PropertyEnricher two-pass approach.
Uses Qwen 2.5 7B via Ollama with format="json" for structured output.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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

logger = logging.getLogger(__name__)


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

    ALLOWED_ENTITY_TYPES: frozenset[str] = frozenset(
        {
            "User",
            "Person",
            "Organization",
            "Technology",
            "Skill",
            "Project",
            "Concept",
            "Topic",
            "Event",
            "Goal",
            "Preference",
            "Location",
            # Cluster 1: MIST self-model entity type (13th).
            "MistIdentity",
        }
    )

    ALLOWED_RELATIONSHIP_TYPES: frozenset[str] = frozenset(
        {
            "USES",
            "KNOWS",
            "WORKS_ON",
            "WORKS_AT",
            "INTERESTED_IN",
            "HAS_GOAL",
            "PREFERS",
            "DISLIKES",
            "EXPERT_IN",
            "LEARNING",
            "STRUGGLES_WITH",
            "DECIDED",
            "EXPERIENCED",
            "IS_A",
            "PART_OF",
            "RELATED_TO",
            "DEPENDS_ON",
            "USED_FOR",
            "WORKS_WITH",
            "KNOWS_PERSON",
            "MEMBER_OF",
            # Cluster 1: MIST-scope relationships for system-scope extractions.
            "IMPLEMENTED_WITH",
            "MIST_HAS_CAPABILITY",
            "MIST_HAS_TRAIT",
            "MIST_HAS_PREFERENCE",
        }
    )

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
        # Build context string from conversation history
        context_str = (
            "\n".join(pre_processed.conversation_context)
            if pre_processed.conversation_context
            else "(no prior context)"
        )

        # Format the system prompt with reference date
        system_prompt = EXTRACTION_SYSTEM_PROMPT.format(
            reference_date=pre_processed.reference_date.strftime("%Y-%m-%d"),
        )

        # Format the user message. `subject_scope` is written by Stage 1.5
        # SubjectScopeClassifier into pre_processed.metadata["subject_scope"].
        # Falls back to "unknown" when Stage 1.5 is disabled or missing so
        # the template substitution never fails closed.
        subject_scope = pre_processed.metadata.get("subject_scope", "unknown")
        user_message = EXTRACTION_USER_TEMPLATE.format(
            context=context_str,
            utterance=pre_processed.original_text,
            subject_scope=subject_scope,
        )

        start_time = time.perf_counter()
        try:
            request = LLMRequest(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
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

        # Parse the JSON output
        parsed = self._parse_json_output(raw_output)

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

    def _parse_json_output(self, raw: str) -> dict:
        """Parse LLM output as JSON with fallback strategies.

        Attempts:
        1. Direct JSON parse of the full string.
        2. Regex extraction of the first JSON object.
        3. Returns empty result on total failure.

        Args:
            raw: Raw string output from the LLM.

        Returns:
            Parsed dict with "entities" and "relationships" keys.
        """
        if not raw or not raw.strip():
            logger.warning("Empty LLM output, returning empty result")
            return {"entities": [], "relationships": []}

        # Strategy 1: Direct parse
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: Find JSON object via regex (handles leading/trailing text)
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    logger.debug("Parsed JSON via regex fallback")
                    return result
        except json.JSONDecodeError:
            pass

        # Strategy 3: Give up
        logger.warning("Failed to parse LLM output as JSON: %s", raw[:200])
        return {"entities": [], "relationships": []}
