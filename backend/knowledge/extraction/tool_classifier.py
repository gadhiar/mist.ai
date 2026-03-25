"""Tool output classification for MCP tool results.

Supplements KnowledgeRouter with granular, per-provider routing decisions.
KnowledgeRouter's Step 2 MCP fast-path remains the generic fallback for
tool output that does not pass through this classifier.

Decision cascade:
1. Parse tool name -> extract provider + action
2. Anti-bloat gates: SHA-256 dedup (1h window), rate limiting
3. Discard patterns: write confirmations, ack-only responses
4. Provider-specific rules (Linear, GitHub, filesystem, web)
5. Content length gate: <50 chars -> discard for vector
6. Entity count gate: graph requires 1+ entity signal
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque

from backend.knowledge.models import RoutingDestination, ToolClassification

logger = logging.getLogger(__name__)

# ===================================================================
# Discard patterns -- write confirmations and trivial ack responses
# ===================================================================

_DISCARD_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*$"),
    re.compile(
        r"^(?:created successfully|ok|done|success|updated|deleted|saved|"
        r"file written|file saved|file created|branch created|issue created|"
        r"comment added|label created|document created|attachment created|"
        r"pull request created|merged successfully|pushed successfully"
        r")[.!]?\s*$",
        re.IGNORECASE,
    ),
)

# ===================================================================
# Entity signal patterns for graph-worthiness heuristic
# ===================================================================

_ENTITY_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"),  # Multi-word proper nouns
    re.compile(r"\b(?:works? at|part of|related to|depends on|belongs to)\b", re.IGNORECASE),
    re.compile(r"\b(?:assignee|reporter|creator|owner|author|member)\b", re.IGNORECASE),
    re.compile(r"\b(?:team|project|milestone|sprint|cycle|label)\b", re.IGNORECASE),
)

# ===================================================================
# Provider-specific action rules
# ===================================================================
# Maps (provider, action) -> (destination, reason, base_confidence).
# Actions not listed fall through to content-based heuristics.

_PROVIDER_ACTION_RULES: dict[tuple[str, str], tuple[str, str, float]] = {
    # -- Linear --
    ("linear", "get_issue"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear issue with entities/relationships",
        0.80,
    ),
    ("linear", "list_issues"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear issue list with entities",
        0.75,
    ),
    ("linear", "get_project"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear project metadata",
        0.80,
    ),
    ("linear", "get_team"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear team metadata",
        0.80,
    ),
    ("linear", "get_user"): (RoutingDestination.GRAPH_AND_VECTOR.value, "linear user entity", 0.80),
    ("linear", "get_milestone"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear milestone metadata",
        0.75,
    ),
    ("linear", "list_projects"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear project list",
        0.70,
    ),
    ("linear", "list_teams"): (RoutingDestination.GRAPH_AND_VECTOR.value, "linear team list", 0.70),
    ("linear", "list_users"): (RoutingDestination.GRAPH_AND_VECTOR.value, "linear user list", 0.70),
    ("linear", "list_cycles"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear cycle list",
        0.70,
    ),
    ("linear", "list_milestones"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "linear milestone list",
        0.70,
    ),
    ("linear", "list_issue_labels"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear labels reference data",
        0.65,
    ),
    ("linear", "list_issue_statuses"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear statuses reference data",
        0.65,
    ),
    ("linear", "list_project_labels"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear project labels reference",
        0.65,
    ),
    ("linear", "list_comments"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear comments text content",
        0.70,
    ),
    ("linear", "list_documents"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear document list",
        0.65,
    ),
    ("linear", "get_document"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear document content",
        0.75,
    ),
    ("linear", "get_issue_status"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear status lookup",
        0.65,
    ),
    ("linear", "search_documentation"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear documentation search results",
        0.75,
    ),
    ("linear", "get_attachment"): (
        RoutingDestination.VECTOR_ONLY.value,
        "linear attachment metadata",
        0.60,
    ),
    ("linear", "save_issue"): (RoutingDestination.DISCARD.value, "linear write operation", 0.90),
    ("linear", "save_comment"): (RoutingDestination.DISCARD.value, "linear write operation", 0.90),
    ("linear", "save_project"): (RoutingDestination.DISCARD.value, "linear write operation", 0.90),
    ("linear", "save_milestone"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "create_issue_label"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "create_document"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "create_attachment"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "update_document"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "delete_comment"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "delete_attachment"): (
        RoutingDestination.DISCARD.value,
        "linear write operation",
        0.90,
    ),
    ("linear", "extract_images"): (
        RoutingDestination.DISCARD.value,
        "linear binary extraction",
        0.90,
    ),
    # -- GitHub --
    ("github", "get_file_contents"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github file content for RAG",
        0.75,
    ),
    ("github", "search_code"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github code search results",
        0.75,
    ),
    ("github", "get_pull_request_files"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github PR file diff",
        0.70,
    ),
    ("github", "get_pull_request_comments"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github PR comments",
        0.70,
    ),
    ("github", "get_pull_request_reviews"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github PR reviews",
        0.70,
    ),
    ("github", "list_pull_requests"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "github PR list with entities",
        0.75,
    ),
    ("github", "get_pull_request"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "github PR with entities",
        0.80,
    ),
    ("github", "get_pull_request_status"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github PR status",
        0.65,
    ),
    ("github", "get_issue"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "github issue with entities",
        0.80,
    ),
    ("github", "list_issues"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "github issue list with entities",
        0.75,
    ),
    ("github", "list_commits"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github commit history",
        0.70,
    ),
    ("github", "search_issues"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "github issue search results",
        0.75,
    ),
    ("github", "search_repositories"): (
        RoutingDestination.VECTOR_ONLY.value,
        "github repository search",
        0.70,
    ),
    ("github", "search_users"): (
        RoutingDestination.GRAPH_AND_VECTOR.value,
        "github user search",
        0.70,
    ),
    ("github", "create_or_update_file"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "create_pull_request"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "create_pull_request_review"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "create_repository"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "create_branch"): (RoutingDestination.DISCARD.value, "github write operation", 0.90),
    ("github", "create_issue"): (RoutingDestination.DISCARD.value, "github write operation", 0.90),
    ("github", "add_issue_comment"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "update_issue"): (RoutingDestination.DISCARD.value, "github write operation", 0.90),
    ("github", "update_pull_request_branch"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "fork_repository"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    ("github", "push_files"): (RoutingDestination.DISCARD.value, "github write operation", 0.90),
    ("github", "merge_pull_request"): (
        RoutingDestination.DISCARD.value,
        "github write operation",
        0.90,
    ),
    # -- Filesystem --
    ("filesystem", "read_file"): (
        RoutingDestination.VECTOR_ONLY.value,
        "filesystem file content",
        0.75,
    ),
    ("filesystem", "read_text_file"): (
        RoutingDestination.VECTOR_ONLY.value,
        "filesystem text content",
        0.75,
    ),
    ("filesystem", "read_multiple_files"): (
        RoutingDestination.VECTOR_ONLY.value,
        "filesystem multi-file content",
        0.75,
    ),
    ("filesystem", "read_media_file"): (
        RoutingDestination.DISCARD.value,
        "filesystem binary content",
        0.90,
    ),
    ("filesystem", "search_files"): (
        RoutingDestination.VECTOR_ONLY.value,
        "filesystem search results",
        0.65,
    ),
    ("filesystem", "list_directory"): (
        RoutingDestination.VECTOR_ONLY.value,
        "filesystem directory listing",
        0.60,
    ),
    ("filesystem", "list_directory_with_sizes"): (
        RoutingDestination.VECTOR_ONLY.value,
        "filesystem directory listing",
        0.60,
    ),
    ("filesystem", "list_allowed_directories"): (
        RoutingDestination.DISCARD.value,
        "filesystem config info",
        0.85,
    ),
    ("filesystem", "get_file_info"): (
        RoutingDestination.DISCARD.value,
        "filesystem metadata only",
        0.80,
    ),
    ("filesystem", "write_file"): (
        RoutingDestination.DISCARD.value,
        "filesystem write operation",
        0.90,
    ),
    ("filesystem", "edit_file"): (
        RoutingDestination.DISCARD.value,
        "filesystem write operation",
        0.90,
    ),
    ("filesystem", "create_directory"): (
        RoutingDestination.DISCARD.value,
        "filesystem write operation",
        0.90,
    ),
    ("filesystem", "move_file"): (
        RoutingDestination.DISCARD.value,
        "filesystem write operation",
        0.90,
    ),
    # directory_tree handled dynamically in _classify_directory_tree
}

# Providers whose tool names use the mcp__<provider-slug>__<action> format.
# The slug may contain hyphens (e.g. "filesystem-dev") which we normalize.
_KNOWN_PROVIDERS: frozenset[str] = frozenset(
    {
        "linear",
        "github",
        "filesystem",
    }
)

# ===================================================================
# Rate-limit constants
# ===================================================================

_ORCHESTRATOR_RATE_LIMIT: int = 10  # calls per minute
_AGENT_RATE_LIMIT: int = 5  # calls per minute
_RATE_WINDOW_SECONDS: float = 60.0
_DEDUP_WINDOW_SECONDS: float = 3600.0  # 1 hour

# Minimum content length for vector storage
_MIN_VECTOR_CONTENT_LENGTH: int = 50


class ToolOutputClassifier:
    """Classifies MCP tool output into routing destinations.

    Supplements KnowledgeRouter with provider-aware, per-action rules and
    anti-bloat gates (deduplication, rate limiting). KnowledgeRouter's
    generic MCP fast-path remains the fallback for content that does not
    pass through this classifier.

    Not thread-safe -- designed for single-threaded async event loop usage
    consistent with the rest of the extraction pipeline.
    """

    def __init__(self) -> None:
        # SHA-256 -> timestamp for deduplication within 1h window
        self._dedup_cache: dict[str, float] = {}
        # Timestamps of recent classifications, per source level
        self._rate_log: dict[str, deque[float]] = {
            "orchestrator": deque(),
            "agent": deque(),
        }

    def classify(
        self,
        tool_name: str,
        tool_output: str,
        source_level: str = "orchestrator",
        metadata: dict | None = None,
    ) -> ToolClassification:
        """Classify a tool output into a routing destination.

        Args:
            tool_name: Full MCP tool name (e.g. "mcp__linear__get_issue").
            tool_output: The string output returned by the tool.
            source_level: "orchestrator" or "agent".
            metadata: Optional extra metadata (currently unused, reserved).

        Returns:
            ToolClassification with destination, reason, and provider info.
        """
        # -- Step 1: Parse tool name ------------------------------------
        provider, action = self._parse_tool_name(tool_name)

        # -- Compute content hash (used by dedup and returned in result) -
        content_hash = hashlib.sha256(tool_output.encode("utf-8")).hexdigest()

        # -- Step 2: Anti-bloat gates -----------------------------------
        # 2a. Dedup: same output within 1h window
        dedup_result = self._check_dedup(content_hash, provider, action, source_level)
        if dedup_result is not None:
            return dedup_result

        # 2b. Rate limiting
        rate_result = self._check_rate_limit(provider, action, source_level, content_hash)
        if rate_result is not None:
            return rate_result

        # Record this call for future rate-limit checks
        self._record_call(source_level)

        # -- Step 3: Discard patterns -----------------------------------
        discard_result = self._check_discard_patterns(
            tool_output,
            provider,
            action,
            source_level,
            content_hash,
        )
        if discard_result is not None:
            return discard_result

        # -- Step 4: Provider-specific rules ----------------------------
        provider_result = self._check_provider_rules(
            provider,
            action,
            tool_output,
            source_level,
            content_hash,
        )
        if provider_result is not None:
            return provider_result

        # -- Step 5: Web provider heuristic (unknown providers) ---------
        if provider == "web" or provider == "unknown":
            return self._classify_web_or_unknown(
                tool_output,
                provider,
                action,
                source_level,
                content_hash,
            )

        # -- Step 6: Content length gate --------------------------------
        if len(tool_output) < _MIN_VECTOR_CONTENT_LENGTH:
            confidence = 0.70 if source_level == "orchestrator" else 0.55
            return ToolClassification(
                destination=RoutingDestination.DISCARD.value,
                reason=f"content too short for vector storage ({len(tool_output)} chars)",
                confidence=confidence,
                tool_provider=provider,
                tool_action=action,
                source_level=source_level,
                content_hash=content_hash,
            )

        # -- Step 7: Fallback to vector_only ----------------------------
        confidence = 0.60 if source_level == "orchestrator" else 0.45
        return ToolClassification(
            destination=RoutingDestination.VECTOR_ONLY.value,
            reason="no provider-specific rule matched; fallback to vector",
            confidence=confidence,
            tool_provider=provider,
            tool_action=action,
            source_level=source_level,
            content_hash=content_hash,
        )

    # ---------------------------------------------------------------
    # Tool name parsing
    # ---------------------------------------------------------------

    @staticmethod
    def _parse_tool_name(tool_name: str) -> tuple[str, str]:
        """Extract (provider, action) from an MCP tool name.

        Expected format: ``mcp__<provider-slug>__<action>``
        Provider slugs may contain hyphens (e.g. "filesystem-dev").
        We strip the suffix after the first hyphen to normalize
        "filesystem-dev" -> "filesystem".

        Returns:
            Tuple of (provider, action). Falls back to ("unknown", tool_name)
            if the name does not match the expected format.
        """
        parts = tool_name.split("__")
        if len(parts) >= 3 and parts[0] == "mcp":
            raw_provider = parts[1]
            # Normalize: "filesystem-dev" -> "filesystem"
            provider = raw_provider.split("-")[0]
            action = "__".join(parts[2:])  # Rejoin in case action has "__"
            return provider, action

        return "unknown", tool_name

    # ---------------------------------------------------------------
    # Anti-bloat: dedup
    # ---------------------------------------------------------------

    def _check_dedup(
        self,
        content_hash: str,
        provider: str,
        action: str,
        source_level: str,
    ) -> ToolClassification | None:
        """Return DISCARD if identical output was seen within the dedup window."""
        now = time.monotonic()
        self._evict_stale_dedup(now)

        if content_hash in self._dedup_cache:
            return ToolClassification(
                destination=RoutingDestination.DISCARD.value,
                reason="duplicate_within_1h",
                confidence=0.95,
                tool_provider=provider,
                tool_action=action,
                source_level=source_level,
                content_hash=content_hash,
            )

        # Register hash
        self._dedup_cache[content_hash] = now
        return None

    def _evict_stale_dedup(self, now: float) -> None:
        """Remove dedup entries older than the window."""
        cutoff = now - _DEDUP_WINDOW_SECONDS
        stale = [h for h, ts in self._dedup_cache.items() if ts < cutoff]
        for h in stale:
            del self._dedup_cache[h]

    # ---------------------------------------------------------------
    # Anti-bloat: rate limiting
    # ---------------------------------------------------------------

    def _check_rate_limit(
        self,
        provider: str,
        action: str,
        source_level: str,
        content_hash: str,
    ) -> ToolClassification | None:
        """Return DISCARD if the source level has exceeded its rate limit."""
        now = time.monotonic()
        limit = _ORCHESTRATOR_RATE_LIMIT if source_level == "orchestrator" else _AGENT_RATE_LIMIT
        log = self._rate_log.get(source_level)
        if log is None:
            log = deque()
            self._rate_log[source_level] = log

        # Evict entries outside the window
        cutoff = now - _RATE_WINDOW_SECONDS
        while log and log[0] < cutoff:
            log.popleft()

        if len(log) >= limit:
            return ToolClassification(
                destination=RoutingDestination.DISCARD.value,
                reason="rate_limit_exceeded",
                confidence=0.90,
                tool_provider=provider,
                tool_action=action,
                source_level=source_level,
                content_hash=content_hash,
            )

        return None

    def _record_call(self, source_level: str) -> None:
        """Record a classification call for rate limiting."""
        now = time.monotonic()
        log = self._rate_log.get(source_level)
        if log is None:
            log = deque()
            self._rate_log[source_level] = log
        log.append(now)

    # ---------------------------------------------------------------
    # Step 3: Discard patterns
    # ---------------------------------------------------------------

    @staticmethod
    def _check_discard_patterns(
        tool_output: str,
        provider: str,
        action: str,
        source_level: str,
        content_hash: str,
    ) -> ToolClassification | None:
        """Discard trivial write confirmations and ack-only responses."""
        stripped = tool_output.strip()
        for pattern in _DISCARD_OUTPUT_PATTERNS:
            if pattern.fullmatch(stripped):
                return ToolClassification(
                    destination=RoutingDestination.DISCARD.value,
                    reason=f"trivial output pattern: '{stripped[:60]}'",
                    confidence=0.90,
                    tool_provider=provider,
                    tool_action=action,
                    source_level=source_level,
                    content_hash=content_hash,
                )
        return None

    # ---------------------------------------------------------------
    # Step 4: Provider-specific rules
    # ---------------------------------------------------------------

    def _check_provider_rules(
        self,
        provider: str,
        action: str,
        tool_output: str,
        source_level: str,
        content_hash: str,
    ) -> ToolClassification | None:
        """Apply provider-specific action rules from the lookup table."""
        # Special case: directory_tree depth heuristic
        if provider == "filesystem" and action == "directory_tree":
            return self._classify_directory_tree(
                tool_output,
                source_level,
                content_hash,
            )

        rule = _PROVIDER_ACTION_RULES.get((provider, action))
        if rule is None:
            return None

        destination, reason, base_confidence = rule
        confidence = self._adjust_confidence(base_confidence, source_level)

        # For graph_and_vector at agent level, require entity signals
        if destination == RoutingDestination.GRAPH_AND_VECTOR.value and source_level == "agent":
            entity_count = self._count_entity_signals(tool_output)
            if entity_count < 2:
                destination = RoutingDestination.VECTOR_ONLY.value
                reason = f"{reason} (agent: insufficient entity signals, {entity_count} found)"
                confidence = min(confidence, 0.60)

        return ToolClassification(
            destination=destination,
            reason=reason,
            confidence=round(confidence, 2),
            tool_provider=provider,
            tool_action=action,
            source_level=source_level,
            content_hash=content_hash,
        )

    def _classify_directory_tree(
        self,
        tool_output: str,
        source_level: str,
        content_hash: str,
    ) -> ToolClassification:
        """Classify directory_tree output based on depth/size."""
        line_count = tool_output.count("\n") + 1
        # Deep trees (many lines) suggest structural info worth graphing
        if line_count > 30:
            confidence = self._adjust_confidence(0.70, source_level)
            return ToolClassification(
                destination=RoutingDestination.GRAPH_AND_VECTOR.value,
                reason=f"filesystem directory_tree with structural depth ({line_count} lines)",
                confidence=round(confidence, 2),
                tool_provider="filesystem",
                tool_action="directory_tree",
                source_level=source_level,
                content_hash=content_hash,
            )
        confidence = self._adjust_confidence(0.60, source_level)
        return ToolClassification(
            destination=RoutingDestination.VECTOR_ONLY.value,
            reason=f"filesystem directory_tree shallow ({line_count} lines)",
            confidence=round(confidence, 2),
            tool_provider="filesystem",
            tool_action="directory_tree",
            source_level=source_level,
            content_hash=content_hash,
        )

    # ---------------------------------------------------------------
    # Step 5: Web / unknown provider heuristic
    # ---------------------------------------------------------------

    @staticmethod
    def _classify_web_or_unknown(
        tool_output: str,
        provider: str,
        action: str,
        source_level: str,
        content_hash: str,
    ) -> ToolClassification:
        """Classify web or unknown provider output by content length."""
        if len(tool_output) > 200:
            confidence = 0.65 if source_level == "orchestrator" else 0.50
            return ToolClassification(
                destination=RoutingDestination.VECTOR_ONLY.value,
                reason=f"{provider} output with substantial content ({len(tool_output)} chars)",
                confidence=confidence,
                tool_provider=provider,
                tool_action=action,
                source_level=source_level,
                content_hash=content_hash,
            )
        confidence = 0.70 if source_level == "orchestrator" else 0.55
        return ToolClassification(
            destination=RoutingDestination.DISCARD.value,
            reason=f"{provider} output too short ({len(tool_output)} chars)",
            confidence=confidence,
            tool_provider=provider,
            tool_action=action,
            source_level=source_level,
            content_hash=content_hash,
        )

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _adjust_confidence(base: float, source_level: str) -> float:
        """Apply source-level confidence adjustment.

        Agent-sourced outputs get a -0.15 penalty reflecting lower
        trust in agentic tool call relevance.
        """
        if source_level == "agent":
            return max(base - 0.15, 0.10)
        return base

    @staticmethod
    def _count_entity_signals(text: str) -> int:
        """Count entity signal pattern matches in text."""
        count = 0
        for pattern in _ENTITY_SIGNAL_PATTERNS:
            count += len(pattern.findall(text))
        return count
