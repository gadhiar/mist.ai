"""Tool usage tracking and pattern detection for skill derivation.

Records tool calls and detects recurring usage patterns that indicate
user skills. Patterns are consumed by SkillDerivationJob to create
Skill and MistCapability entities in the knowledge graph.
"""

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from backend.knowledge.config import SkillDerivationConfig


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Single tool invocation record."""

    tool_name: str
    tool_type: str
    context: str
    success: bool
    timestamp: datetime
    session_id: str
    event_id: str


@dataclass(frozen=True, slots=True)
class ToolUsagePattern:
    """Detected recurring tool usage pattern."""

    tool_type: str
    tool_names: frozenset[str]
    context_summary: str
    occurrence_count: int
    success_count: int
    first_seen: datetime
    last_seen: datetime


# Static mapping from tool name substrings/patterns to logical types.
_TOOL_TYPE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"search|grep|find|lookup|query", re.IGNORECASE), "search"),
    (re.compile(r"file|read|write|edit|delete|move|copy|mkdir", re.IGNORECASE), "file_management"),
    (re.compile(r"exec|run|shell|bash|terminal|compile|build", re.IGNORECASE), "code_execution"),
]


def classify_tool_type(tool_name: str) -> str:
    """Classify a tool name into a logical tool type.

    Uses substring pattern matching against known categories.

    Args:
        tool_name: Name of the tool (e.g. "file_search", "bash_exec").

    Returns:
        One of "search", "file_management", "code_execution", or "general".
    """
    for pattern, tool_type in _TOOL_TYPE_PATTERNS:
        if pattern.search(tool_name):
            return tool_type
    return "general"


# Regex to strip non-content tokens for Jaccard comparison.
_WORD_RE = re.compile(r"[a-z]+")


def _content_words(text: str) -> set[str]:
    """Extract lowercased content words from text.

    Strips punctuation and non-alphabetic tokens so that Jaccard
    similarity operates on meaningful words only.
    """
    return set(_WORD_RE.findall(text.lower()))


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Compute Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ToolUsageTracker:
    """Tracks tool call records and detects usage patterns.

    Maintains a bounded window of recent records and groups them by
    tool type and context similarity to surface recurring patterns.

    Args:
        config: Skill derivation configuration controlling window size,
            thresholds, and lookback period.
    """

    def __init__(self, config: SkillDerivationConfig) -> None:
        self._config = config
        # Use a plain list with manual trimming so we can iterate freely.
        self._records: list[ToolCallRecord] = []
        self._max_records: int = config.window_size

    def record(self, record: ToolCallRecord) -> None:
        """Append a tool call record, evicting the oldest if at capacity."""
        self._records.append(record)
        if len(self._records) > self._max_records:
            self._records = self._records[-self._max_records :]

    def detect_patterns(self) -> list[ToolUsagePattern]:
        """Detect recurring tool usage patterns from recent records.

        Algorithm:
            1. Filter records to those within ``lookback_days``.
            2. Group by ``tool_type``.
            3. Within each type group, sub-group by context similarity
               using Jaccard similarity on lowercased content words.
            4. Return patterns where ``occurrence_count >= skill_threshold``.

        Returns:
            List of detected ToolUsagePattern instances.
        """
        cutoff = datetime.now(UTC) - timedelta(days=self._config.lookback_days)
        recent = [r for r in self._records if r.timestamp >= cutoff]

        if not recent:
            return []

        # Group by tool_type
        by_type: dict[str, list[ToolCallRecord]] = defaultdict(list)
        for rec in recent:
            by_type[rec.tool_type].append(rec)

        patterns: list[ToolUsagePattern] = []

        for tool_type, records in by_type.items():
            clusters = self._cluster_by_context(records)
            for cluster in clusters:
                if len(cluster) < self._config.skill_threshold:
                    continue
                patterns.append(self._build_pattern(tool_type, cluster))

        return patterns

    def _cluster_by_context(self, records: Sequence[ToolCallRecord]) -> list[list[ToolCallRecord]]:
        """Sub-group records by context similarity.

        Uses single-linkage clustering: a record joins the first cluster
        whose representative context has Jaccard similarity (on lowercased
        content words) >= ``similarity_threshold``.

        Returns:
            List of record clusters.
        """
        clusters: list[tuple[set[str], list[ToolCallRecord]]] = []

        for rec in records:
            words = _content_words(rec.context)
            merged = False
            for cluster_words, cluster_records in clusters:
                if _jaccard_similarity(words, cluster_words) >= self._config.similarity_threshold:
                    cluster_records.append(rec)
                    cluster_words.update(words)
                    merged = True
                    break
            if not merged:
                clusters.append((set(words), [rec]))

        return [recs for _, recs in clusters]

    @staticmethod
    def _build_pattern(tool_type: str, cluster: list[ToolCallRecord]) -> ToolUsagePattern:
        """Build a ToolUsagePattern from a cluster of records."""
        tool_names = frozenset(r.tool_name for r in cluster)
        # Use the most common context as the summary.
        context_counts: dict[str, int] = defaultdict(int)
        for r in cluster:
            context_counts[r.context] += 1
        context_summary = max(context_counts, key=context_counts.get)  # type: ignore[arg-type]

        timestamps = [r.timestamp for r in cluster]
        success_count = sum(1 for r in cluster if r.success)

        return ToolUsagePattern(
            tool_type=tool_type,
            tool_names=tool_names,
            context_summary=context_summary,
            occurrence_count=len(cluster),
            success_count=success_count,
            first_seen=min(timestamps),
            last_seen=max(timestamps),
        )
