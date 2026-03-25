"""Knowledge routing -- boundary logic for graph-worthy vs vector-only content.

Classifies incoming content into one of four destinations:
- DISCARD: empty filler, acknowledgments, system messages
- MCP_ONLY: transient tool output that can be re-invoked
- GRAPH_AND_VECTOR: content with extractable entities/relationships
- VECTOR_ONLY: everything else (default fallback)

The router is a lightweight pre-check (regex + heuristics, no LLM).
It does NOT overlap with SignalDetector, which gates internal/self-model
derivation. The router gates the external knowledge extraction pipeline.
"""

from __future__ import annotations

import logging
import re

from backend.knowledge.models import ContentSourceType, RoutingDecision, RoutingDestination
from backend.knowledge.ontologies import EXTRACTABLE_NODE_TYPES

logger = logging.getLogger(__name__)

# ===================================================================
# Entity type names lowercased for density heuristic
# ===================================================================

_ENTITY_TYPE_NAMES_LOWER: frozenset[str] = frozenset(
    name.lower() for name in EXTRACTABLE_NODE_TYPES
)

# ===================================================================
# Step 1 -- Discard patterns (empty filler / acknowledgments)
# ===================================================================

_DISCARD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*$"),
    re.compile(
        r"^(?:ok|okay|sure|thanks|thank you|got it|yes|no|yep|yup|nope"
        r"|hmm|hm|huh|ah|oh|alright|right|cool|fine|k|kk|mhm|uh huh"
        r"|sounds good|understood|noted|will do|perfect)[.!?]*$",
        re.IGNORECASE,
    ),
)

# ===================================================================
# Step 3 -- Graph-worthy signal patterns (conversational content)
# ===================================================================

# Preference signals: user expressing likes, dislikes, favorites.
# NOTE: These detect *personal preferences about things*, not preferences
# about MIST's behavior (which SignalDetector handles via _PREFERENCE_PATTERNS).
_PREFERENCE_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi prefer\b", re.IGNORECASE),
    re.compile(r"\bi (?:really )?(?:like|love|enjoy|hate|dislike|despise)\b", re.IGNORECASE),
    re.compile(r"\bmy (?:favorite|favourite)\b", re.IGNORECASE),
)

# Correction signals: user correcting factual information.
_CORRECTION_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bactually[,.]?\s+(?:it|that|the|i|my|we)\b", re.IGNORECASE),
    re.compile(r"\bno[,.]?\s+it(?:'s| is)\b", re.IGNORECASE),
    re.compile(r"\bi meant\b", re.IGNORECASE),
    re.compile(r"\bcorrection[:.]\b", re.IGNORECASE),
)

# Relationship signals: user mentioning personal relationships.
_RELATIONSHIP_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bmy (?:wife|husband|partner|spouse|boss|manager|colleague|coworker"
        r"|friend|brother|sister|mom|mother|dad|father|son|daughter|team)\b",
        re.IGNORECASE,
    ),
)

# Skill signals: user stating capabilities or tool usage.
_SKILL_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bi(?:'m| am) (?:good|great|experienced) at\b", re.IGNORECASE),
    re.compile(r"\bi (?:know|use|work with|specialize in)\b", re.IGNORECASE),
)

# Structural signals: entity relationships (works at, part of, etc.).
_STRUCTURAL_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bworks? at\b", re.IGNORECASE),
    re.compile(r"\bpart of\b", re.IGNORECASE),
    re.compile(r"\brelated to\b", re.IGNORECASE),
    re.compile(r"\bdepends on\b", re.IGNORECASE),
    re.compile(r"\bbelongs to\b", re.IGNORECASE),
    re.compile(r"\bbuilt (?:with|using|on)\b", re.IGNORECASE),
)

# All graph signal categories with their patterns and base confidence.
_GRAPH_SIGNAL_CATEGORIES: tuple[tuple[str, tuple[re.Pattern[str], ...], float], ...] = (
    ("preference", _PREFERENCE_SIGNAL_PATTERNS, 0.85),
    ("correction", _CORRECTION_SIGNAL_PATTERNS, 0.80),
    ("relationship", _RELATIONSHIP_SIGNAL_PATTERNS, 0.90),
    ("skill", _SKILL_SIGNAL_PATTERNS, 0.80),
    ("structural", _STRUCTURAL_SIGNAL_PATTERNS, 0.75),
)

# Entity density threshold -- if probable entity mentions / word count
# exceeds this, content is likely graph-worthy even without explicit signals.
_ENTITY_DENSITY_THRESHOLD: float = 0.15


class KnowledgeRouter:
    """Classifies content into routing destinations.

    Four-step decision cascade:
    1. Discard empty/filler content
    2. Fast-path by source type (MCP, document, system)
    3. Graph-worthy signal detection (conversational content)
    4. Fallback to vector-only

    Instantiation compiles all regex patterns once. Thread-safe after init.
    """

    def __init__(self) -> None:
        # Patterns are module-level constants compiled at import time.
        # Nothing extra to initialize, but the constructor exists so
        # the class can be extended with configuration later.
        pass

    def classify(
        self,
        content: str,
        source_type: str,
        metadata: dict | None = None,
    ) -> RoutingDecision:
        """Classify content into a routing destination.

        Args:
            content: The text content to classify.
            source_type: A `ContentSourceType` value string.
            metadata: Optional metadata dict. Recognized keys:
                - transient (bool): MCP output that is ephemeral.
                - re_invocable (bool): MCP output that can be fetched again.

        Returns:
            RoutingDecision with destination, reason, and confidence.
        """
        if metadata is None:
            metadata = {}

        # -- Step 1: Discard empty/filler content -----------------------
        decision = self._check_discard(content)
        if decision is not None:
            logger.debug("Routing DISCARD: %s", decision.reason)
            return decision

        # -- Step 2: Source-type fast path -------------------------------
        decision = self._check_source_type(source_type, metadata)
        if decision is not None:
            logger.debug(
                "Routing %s (source-type fast path): %s", decision.destination, decision.reason
            )
            return decision

        # -- Step 3: Graph-worthy signal detection ----------------------
        decision = self._check_graph_signals(content)
        if decision is not None:
            logger.debug("Routing GRAPH_AND_VECTOR: %s", decision.reason)
            return decision

        # -- Step 4: Fallback -------------------------------------------
        word_count = len(content.split())
        confidence = 0.5 if word_count < 20 else 0.6
        decision = RoutingDecision(
            destination=RoutingDestination.VECTOR_ONLY.value,
            reason="no graph-worthy signals detected",
            confidence=confidence,
        )
        logger.debug("Routing VECTOR_ONLY (fallback): %s", decision.reason)
        return decision

    # ---------------------------------------------------------------
    # Step 1: Discard check
    # ---------------------------------------------------------------

    @staticmethod
    def _check_discard(content: str) -> RoutingDecision | None:
        """Return a DISCARD decision if content is empty or acknowledgment filler."""
        stripped = content.strip()
        if not stripped:
            return RoutingDecision(
                destination=RoutingDestination.DISCARD.value,
                reason="empty or whitespace-only content",
                confidence=1.0,
            )

        for pattern in _DISCARD_PATTERNS:
            if pattern.fullmatch(stripped):
                return RoutingDecision(
                    destination=RoutingDestination.DISCARD.value,
                    reason=f"acknowledgment filler: '{stripped}'",
                    confidence=0.90,
                )

        return None

    # ---------------------------------------------------------------
    # Step 2: Source-type fast path
    # ---------------------------------------------------------------

    @staticmethod
    def _check_source_type(source_type: str, metadata: dict) -> RoutingDecision | None:
        """Return a decision based on source type, skipping signal detection."""
        if source_type == ContentSourceType.SYSTEM.value:
            return RoutingDecision(
                destination=RoutingDestination.DISCARD.value,
                reason="system message",
                confidence=0.95,
            )

        if source_type == ContentSourceType.MCP_TOOL_OUTPUT.value:
            if metadata.get("transient") or metadata.get("re_invocable"):
                return RoutingDecision(
                    destination=RoutingDestination.MCP_ONLY.value,
                    reason="transient/re-invocable MCP tool output",
                    confidence=0.90,
                )
            return RoutingDecision(
                destination=RoutingDestination.VECTOR_ONLY.value,
                reason="MCP tool output (persistent, no graph signals checked)",
                confidence=0.70,
            )

        if source_type == ContentSourceType.DOCUMENT_CHUNK.value:
            return RoutingDecision(
                destination=RoutingDestination.VECTOR_ONLY.value,
                reason="document chunk (indexed for RAG retrieval)",
                confidence=0.85,
            )

        if source_type == ContentSourceType.REFERENCE_LOOKUP.value:
            return RoutingDecision(
                destination=RoutingDestination.VECTOR_ONLY.value,
                reason="reference lookup result",
                confidence=0.80,
            )

        # Conversation and unknown types fall through to signal detection.
        return None

    # ---------------------------------------------------------------
    # Step 3: Graph-worthy signal detection
    # ---------------------------------------------------------------

    def _check_graph_signals(self, content: str) -> RoutingDecision | None:
        """Detect graph-worthy signals in conversational content."""
        matched_categories: list[str] = []
        max_confidence: float = 0.0

        for category, patterns, base_confidence in _GRAPH_SIGNAL_CATEGORIES:
            for pattern in patterns:
                if pattern.search(content):
                    matched_categories.append(category)
                    if base_confidence > max_confidence:
                        max_confidence = base_confidence
                    break  # One match per category is sufficient.

        # Entity density heuristic
        density = self._entity_density(content)
        if density >= _ENTITY_DENSITY_THRESHOLD:
            matched_categories.append("entity_density")
            density_confidence = min(0.70 + (density - _ENTITY_DENSITY_THRESHOLD) * 2, 0.95)
            if density_confidence > max_confidence:
                max_confidence = density_confidence

        if matched_categories:
            # Boost confidence slightly when multiple categories match.
            if len(matched_categories) > 1:
                max_confidence = min(max_confidence + 0.05, 0.95)

            return RoutingDecision(
                destination=RoutingDestination.GRAPH_AND_VECTOR.value,
                reason=f"graph signals: {', '.join(matched_categories)}",
                confidence=round(max_confidence, 2),
            )

        return None

    # ---------------------------------------------------------------
    # Entity density heuristic
    # ---------------------------------------------------------------

    @staticmethod
    def _entity_density(content: str) -> float:
        """Estimate entity density as probable entity mentions / word count.

        Uses capitalized multi-word sequences and known entity type names
        from the ontology as signals for probable entities.
        """
        words = content.split()
        if not words:
            return 0.0

        entity_mentions = 0

        # Count capitalized words that are not sentence starters.
        # Simple heuristic: words after position 0 that start with uppercase
        # and are not common English words.
        _COMMON_WORDS: frozenset[str] = frozenset(
            {
                "I",
                "The",
                "A",
                "An",
                "This",
                "That",
                "It",
                "Is",
                "Are",
                "Was",
                "Were",
                "But",
                "And",
                "Or",
                "If",
                "So",
                "My",
                "We",
                "They",
                "He",
                "She",
                "You",
                "In",
                "On",
                "At",
                "To",
                "For",
                "Of",
                "With",
                "Not",
                "Do",
                "Does",
                "Did",
                "Has",
                "Have",
                "Had",
                "Be",
                "Been",
                "Can",
                "Will",
                "Would",
                "Should",
                "Could",
                "May",
                "Just",
                "Also",
                "Very",
                "Really",
                "Then",
                "Now",
                "Here",
                "There",
                "When",
                "Where",
                "How",
                "What",
                "Who",
                "Which",
                "Some",
                "All",
                "Each",
                "Every",
                "No",
            }
        )

        for i, word in enumerate(words):
            cleaned = word.strip(".,;:!?()[]\"'")
            if not cleaned:
                continue

            # Known ontology type name mentioned explicitly
            if cleaned.lower() in _ENTITY_TYPE_NAMES_LOWER:
                entity_mentions += 1
                continue

            # Capitalized word not at sentence start and not common
            if i > 0 and cleaned[0].isupper() and cleaned not in _COMMON_WORDS:
                entity_mentions += 1

        return entity_mentions / len(words)
