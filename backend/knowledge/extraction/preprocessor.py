"""Pre-processing stage for the extraction pipeline.

Stage 1: Assembles context from conversation history and prepares input
for the LLM extraction call. No LLM call, target <10ms.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Bug K guard: directive-language patterns that commonly prefix prompt
# injection attempts. Matches at word boundaries, case-insensitive. This is
# a SIGNAL -- the pipeline decides what to do (warn, drop extraction, etc.).
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ignore_previous",
        re.compile(
            r"\bignore\s+(previous|prior|above|all)\s+(instructions?|prompts?)\b", re.IGNORECASE
        ),
    ),
    (
        "forget_directive",
        re.compile(r"\bforget\s+(what|everything|the\s+(last|previous|above))\b", re.IGNORECASE),
    ),
    (
        "instead_treat",
        re.compile(r"\binstead[,\s]+(treat|classify|extract|label)\b", re.IGNORECASE),
    ),
    (
        "override_system",
        re.compile(
            r"\b(override|disregard|skip)\s+(the\s+)?(system|rules?|guidelines?)\b", re.IGNORECASE
        ),
    ),
    ("role_switch", re.compile(r"\byou\s+are\s+now\s+a\b", re.IGNORECASE)),
    (
        "new_instructions",
        re.compile(r"\b(new\s+instructions?|updated\s+rules?)\s*(:|follow|are)\b", re.IGNORECASE),
    ),
]


def _detect_injection(utterance: str) -> str | None:
    """Return the name of the first matching injection pattern, or None."""
    for pattern_name, pattern in _INJECTION_PATTERNS:
        if pattern.search(utterance):
            return pattern_name
    return None


@dataclass
class PreProcessedInput:
    """Input prepared for LLM extraction.

    Contains the original utterance, conversation context formatted
    for the prompt, and metadata for downstream stages.

    Metadata keys currently written or read by the pipeline:
      - ``injection_warning`` (bool) + ``pattern`` (str): Bug K guard set
        by ``_detect_injection`` on directive-language utterances.
      - ``subject_scope`` (str): one of "user-scope", "system-scope",
        "third-party", "unknown". Set by Stage 1.5 SubjectScopeClassifier
        and consumed by OntologyConstrainedExtractor to weight the
        extraction prompt. Absent when Stage 1.5 is disabled.
      - ``subject_scope_confidence`` (float in [0.0, 1.0]): classifier
        confidence paired with ``subject_scope``.
    """

    original_text: str
    resolved_text: str  # Same as original (LLM resolves coreferences inline)
    conversation_context: list[str]  # Last N exchanges as "[role]: content"
    reference_date: datetime
    turn_index: int
    metadata: dict = field(default_factory=dict)


class PreProcessor:
    """Heuristic pre-processing. No LLM call.

    Assembles conversation context and packages input for the
    OntologyConstrainedExtractor. Keeps the last 3 exchanges
    (6 messages) for coreference resolution by the LLM.
    """

    MAX_CONTEXT_MESSAGES: int = 6  # 3 exchanges (user + assistant each)

    def pre_process(
        self,
        utterance: str,
        conversation_history: list[dict[str, str]],
        reference_date: datetime,
        turn_index: int = 0,
    ) -> PreProcessedInput:
        """Assemble context for extraction.

        Takes the raw utterance and recent conversation history and
        packages them into a PreProcessedInput for the extractor.

        Args:
            utterance: The current user utterance to extract from.
            conversation_history: List of {"role": str, "content": str} dicts
                representing prior conversation turns.
            reference_date: Current date for temporal resolution.
            turn_index: Position of this turn in the conversation.

        Returns:
            PreProcessedInput ready for the extraction stage.
        """
        context_turns: list[str] = []
        for msg in conversation_history[-self.MAX_CONTEXT_MESSAGES :]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_turns.append(f"[{role}]: {content}")

        metadata: dict = {}
        injection_pattern = _detect_injection(utterance)
        if injection_pattern is not None:
            metadata["injection_warning"] = True
            metadata["pattern"] = injection_pattern
            logger.warning(
                "Directive-injection pattern '%s' detected in utterance; "
                "flagged for downstream policy. utterance=%r",
                injection_pattern,
                utterance[:200],
            )

        return PreProcessedInput(
            original_text=utterance,
            resolved_text=utterance,
            conversation_context=context_turns,
            reference_date=reference_date,
            turn_index=turn_index,
            metadata=metadata,
        )
