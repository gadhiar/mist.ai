"""Query intent classifier for hybrid retrieval routing.

Classifies user queries into one of four intent types to determine
which retrieval backends should handle the query:

    factual   -> vector store (document/content recall)
    relational -> graph store (entity/relationship traversal)
    hybrid    -> both vector + graph
    live      -> MCP tool invocation (real-time state)

Mirrors the SignalDetector pattern: compiled regex, no LLM, pure heuristic.
"""

import re

from backend.knowledge.config import QueryIntentConfig
from backend.knowledge.models import QueryIntent

# ---------------------------------------------------------------------------
# FACTUAL patterns -- content/document recall
# ---------------------------------------------------------------------------
_FACTUAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+(?:is|are|was|were)\s+the\b", re.I),
    re.compile(r"\b(?:describe|explain|summarize)\b", re.I),
    re.compile(r"\bwhat\s+does\s+.+?\s+(?:say|do|mean)\b", re.I),
    re.compile(r"\b(?:benefits|advantages|drawbacks)\s+of\b", re.I),
    re.compile(r"\bwhen\s+was\s+.+?\s+created\b", re.I),
    re.compile(r"\bwhat\s+did\s+.+?\s+say\s+about\b", re.I),
    re.compile(r"\b(?:compare|contrast)\b", re.I),
    re.compile(r"\b(?:learned|experience|know|problems)\s+(?:about|with)\b", re.I),
    re.compile(r"\b(?:concepts|architecture|goals)\b", re.I),
    re.compile(r"\bwhat\s+are\s+the\s+(?:benefits|advantages|drawbacks)\b", re.I),
]

# ---------------------------------------------------------------------------
# RELATIONAL patterns -- graph entity/relationship queries
# ---------------------------------------------------------------------------
_RELATIONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bwhat\s+(?:do|does|did)\s+I\b", re.I),
    re.compile(r"\b(?:I|we)\s+(?:use|know|work|learn|have|am|contribute)\b", re.I),
    re.compile(r"\b(?:my|our)\b", re.I),
    re.compile(r"\bhow\s+is\s+.+?\s+related\s+to\b", re.I),
    re.compile(r"\bwho\s+(?:do|does|did)\s+I\b", re.I),
    re.compile(r"\bwhat\s+skills\s+does\b", re.I),
    re.compile(r"\bdepends?\s+on\b", re.I),
    re.compile(r"\bdo\s+I\s+know\b", re.I),
    re.compile(r"\bwhat\s+am\s+I\b", re.I),
    re.compile(r"\bwhat\s+are\s+my\b", re.I),
    re.compile(r"\bam\s+I\s+(?:an?\s+)?(?:expert|proficient)\b", re.I),
    re.compile(r"\b(?:relate\s+to\s+my|based\s+on\s+my|require)\b", re.I),
    re.compile(r"\bwhat\s+projects\s+am\s+I\b", re.I),
    re.compile(r"\bwhat\s+technologies\b", re.I),
]

# ---------------------------------------------------------------------------
# LIVE patterns -- real-time / MCP tool queries
# ---------------------------------------------------------------------------

# Strong live markers: any single match is sufficient to classify as live.
_STRONG_LIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bright\s+now\b", re.I),
    re.compile(r"\brunning\b", re.I),
    re.compile(r"\bMIS-\d+", re.I),
    re.compile(r"\bin\s+Linear\b", re.I),
    re.compile(r"\bin\s+GitHub\b", re.I),
    re.compile(r"\bon\s+the\s+repo\b", re.I),
    re.compile(r"\bgit\s+branch\b", re.I),
    re.compile(r"\bcommit\s+on\b", re.I),
    re.compile(r"\bthis\s+sprint\b", re.I),
    re.compile(r"\bhow\s+much\s+.+?\s+(?:available|free|used)\b", re.I),
]

# Regular live patterns: need score >= 2 or co-occurrence with strong marker.
_LIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bstatus\s+of\b", re.I),
    re.compile(r"\bopen\s+(?:pull\s+requests?|PRs?|issues?|tickets?)\b", re.I),
    re.compile(
        r"\b(?:closed|merged|assigned)\b.*?\b(?:this\s+sprint|this\s+week|today)\b",
        re.I,
    ),
    re.compile(r"\bwhat\s+(?:is|are)\s+the\s+(?:current|latest)\b", re.I),
    re.compile(r"\bshow\s+me\s+the\s+(?:latest|recent)\b", re.I),
    re.compile(r"\bcurrent\b", re.I),
    re.compile(r"\btasks?\s+(?:are\s+)?assigned\s+to\b", re.I),
    re.compile(r"\blatest\s+commit\b", re.I),
]

# Store mapping per intent type.
_STORE_MAP: dict[str, tuple[str, ...]] = {
    "factual": ("vector",),
    "relational": ("graph",),
    "hybrid": ("vector", "graph"),
    "live": ("mcp",),
}


class QueryClassifier:
    """Classifies query intent for hybrid retrieval routing.

    Uses compiled regex patterns to score each intent category
    independently, then selects the winning intent based on
    priority rules: live -> hybrid -> single winner -> fallback.
    """

    def __init__(self, config: QueryIntentConfig | None = None) -> None:
        self._config = config or QueryIntentConfig()

    def classify(self, query: str) -> QueryIntent:
        """Classify a query into an intent type.

        Args:
            query: The user's natural language query.

        Returns:
            QueryIntent with intent type, confidence, and suggested stores.
        """
        factual_score = self._score_patterns(query, _FACTUAL_PATTERNS)
        relational_score = self._score_patterns(query, _RELATIONAL_PATTERNS)
        live_score, has_strong_live = self._score_live(query)

        # Priority 1: Live -- requires strong marker or score >= 2
        if has_strong_live or live_score >= 2:
            return self._build_result("live", live_score)

        # Priority 2: Hybrid -- both factual AND relational signals present
        if factual_score >= 1 and relational_score >= 1:
            combined = factual_score + relational_score
            return self._build_result("hybrid", combined)

        # Priority 3: Single winner
        if factual_score > relational_score:
            return self._build_result("factual", factual_score)
        if relational_score > factual_score:
            return self._build_result("relational", relational_score)

        # Priority 4: Tie or no matches -- fallback to hybrid with min confidence
        if factual_score == relational_score and factual_score > 0:
            return self._build_result("hybrid", factual_score)

        return QueryIntent(
            intent="hybrid",
            confidence=self._config.min_confidence,
            suggested_stores=_STORE_MAP["hybrid"],
        )

    def _score_patterns(self, query: str, patterns: list[re.Pattern[str]]) -> int:
        """Count the number of matching patterns in a query."""
        return sum(1 for p in patterns if p.search(query))

    def _score_live(self, query: str) -> tuple[int, bool]:
        """Score live patterns and detect strong live markers.

        Returns:
            Tuple of (total_live_score, has_strong_marker).
        """
        strong_count = sum(1 for p in _STRONG_LIVE_PATTERNS if p.search(query))
        weak_count = sum(1 for p in _LIVE_PATTERNS if p.search(query))
        return strong_count + weak_count, strong_count > 0

    def _build_result(self, intent: str, match_count: int) -> QueryIntent:
        """Build a QueryIntent with computed confidence."""
        raw = self._config.confidence_base + (match_count - 1) * self._config.confidence_per_match
        confidence = min(1.0, max(0.0, raw))
        return QueryIntent(
            intent=intent,
            confidence=confidence,
            suggested_stores=_STORE_MAP[intent],
        )
