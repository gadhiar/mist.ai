"""Pre-processing stage for the extraction pipeline.

Stage 1: Assembles context from conversation history and prepares input
for the LLM extraction call. No LLM call, target <10ms.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PreProcessedInput:
    """Input prepared for LLM extraction.

    Contains the original utterance, conversation context formatted
    for the prompt, and metadata for downstream stages.
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

        return PreProcessedInput(
            original_text=utterance,
            resolved_text=utterance,
            conversation_context=context_turns,
            reference_date=reference_date,
            turn_index=turn_index,
            metadata={},
        )
