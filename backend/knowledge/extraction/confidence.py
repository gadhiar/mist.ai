"""Confidence scoring stage for the extraction pipeline.

Stage 3: Post-LLM heuristic adjustment of confidence scores based on
hedge words, third-party attribution, and other linguistic signals.
No LLM call, target <5ms.
"""

import logging
import re

from backend.knowledge.extraction.ontology_extractor import ExtractionResult

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Adjusts extraction confidence based on linguistic signals.

    Applies two classes of adjustments:
    - Hedge penalties: words like "maybe", "I think", "probably" reduce
      confidence by a fixed amount. Only the worst (largest) penalty applies.
    - Third-party cap: statements attributed to others ("my coworker says",
      "I heard that") are capped at 0.80.

    Final confidence is clamped to [0.1, 0.99].
    """

    # Each tuple is (compiled regex pattern, confidence penalty).
    # Penalty is subtracted from the LLM-assigned confidence.
    # Only the single worst (most negative) hedge penalty applies.
    HEDGE_PATTERNS: list[tuple[re.Pattern, float]] = [
        (re.compile(r"\b(?:maybe|perhaps|possibly|might)\b", re.IGNORECASE), -0.20),
        (re.compile(r"\b(?:I think|I guess|I suppose|I believe)\b", re.IGNORECASE), -0.15),
        (re.compile(r"\b(?:probably|likely)\b", re.IGNORECASE), -0.10),
        (re.compile(r"\b(?:not sure|uncertain|unsure)\b", re.IGNORECASE), -0.25),
        (re.compile(r"\b(?:sort of|kind of|kinda|sorta)\b", re.IGNORECASE), -0.10),
    ]

    THIRD_PARTY_PATTERNS: list[re.Pattern] = [
        re.compile(
            r"\b(?:my\s+)?(?:coworker|colleague|friend|boss|manager|teammate)\s+"
            r"(?:says?|said|told|mentioned|thinks?|thought)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:I heard|I read|I saw)\s+that\b", re.IGNORECASE),
        re.compile(r"\b(?:apparently|reportedly|supposedly)\b", re.IGNORECASE),
    ]

    THIRD_PARTY_CAP: float = 0.80
    CONFIDENCE_MIN: float = 0.10
    CONFIDENCE_MAX: float = 0.99

    def adjust_confidence(self, extraction: ExtractionResult) -> ExtractionResult:
        """Adjust confidence scores on all relationships in the extraction.

        Modifies the ExtractionResult in place and returns it.

        Args:
            extraction: The ExtractionResult from the ontology extractor.

        Returns:
            The same ExtractionResult with adjusted confidence values.
        """
        source_text = extraction.source_utterance

        # Determine hedge penalty (worst match only)
        hedge_penalty = 0.0
        for pattern, penalty in self.HEDGE_PATTERNS:
            if pattern.search(source_text) and penalty < hedge_penalty:
                # penalty is negative, so we want the most negative (worst)
                hedge_penalty = penalty

        # Determine if third-party attribution is present
        is_third_party = any(pattern.search(source_text) for pattern in self.THIRD_PARTY_PATTERNS)

        if hedge_penalty == 0.0 and not is_third_party:
            # No adjustments needed
            return extraction

        adjustments_made = 0
        for rel in extraction.relationships:
            props = rel.get("properties", {})
            if props is None:
                props = {}
                rel["properties"] = props

            original_conf = props.get("confidence", 0.9)
            if not isinstance(original_conf, int | float):
                try:
                    original_conf = float(original_conf)
                except (ValueError, TypeError):
                    original_conf = 0.9

            adjusted = original_conf + hedge_penalty

            if is_third_party and adjusted > self.THIRD_PARTY_CAP:
                adjusted = self.THIRD_PARTY_CAP

            # Clamp
            adjusted = max(self.CONFIDENCE_MIN, min(self.CONFIDENCE_MAX, adjusted))

            if adjusted != original_conf:
                props["confidence"] = round(adjusted, 2)
                adjustments_made += 1

        if adjustments_made > 0:
            logger.debug(
                "Adjusted confidence on %d relationships (hedge=%.2f, third_party=%s)",
                adjustments_made,
                hedge_penalty,
                is_third_party,
            )

        return extraction
