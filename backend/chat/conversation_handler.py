"""Conversation Handler with MCP-like Tool Access.

Enables LLM to autonomously:
- Query knowledge graph for context
- Extract and store new knowledge
- Think and search database freely
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.chat.context_budget import ContextBudgetPlanner
from backend.chat.mist_context import MistContext
from backend.chat.slop_detector import SlopDetector
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.models import (
    ConversationSession,
    RetrievalFilters,
    RetrievalResult,
    RetrievedFact,
)
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm import LLMRequest, StreamingLLMProvider
from backend.llm.instrumented_provider import llm_call_context
from backend.llm.models import LLMResponse
from backend.llm.models import ToolCall as LLMToolCall
from backend.vault.conventions import ConventionsLoader

if TYPE_CHECKING:
    from backend.debug_jsonl_logger import DebugJSONLLogger, TurnRecord
    from backend.interfaces import VaultWriterProtocol
    from backend.knowledge.curation.graph_regenerator import RebuildResult
    from backend.knowledge.extraction.pipeline import ExtractionPipeline
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
    from backend.vault.invalidation_bus import InvalidationBus

logger = logging.getLogger(__name__)


class _ToolNotFoundError(Exception):
    """Raised by _dispatch_tool when the LLM invokes an unregistered tool.

    The observability wrap turns this into a tool_call_completed event with
    a 'ToolNotFound' error label. Module-private (leading underscore) so
    nothing outside the conversation pipeline can rely on it.
    """

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool not found: {tool_name}")
        self.tool_name = tool_name


def _summarize_tool_args(tool_name: str, arguments: dict[str, Any]) -> str:
    """One-line human-readable args summary for tool_call_started events.

    ADR-017 Wave 2: tool-call observability. Per-tool heuristics keep
    summaries short and informative for the FE chrome's inline tool-call
    indicator. Generic fallback for unknown tools enumerates kwargs.
    """
    if tool_name == "query_knowledge_graph":
        query = str(arguments.get("query", ""))
        limit = arguments.get("limit", 20)
        verbosity = arguments.get("verbosity", "compact")
        return f"query={query[:40]!r} limit={limit} v={verbosity!r}"
    if tool_name == "query_vault":
        query = str(arguments.get("query", ""))
        limit = arguments.get("limit", 5)
        hint = arguments.get("display_hint", "auto")
        return f"query={query[:40]!r} limit={limit} hint={hint!r}"
    if tool_name == "frontend.switch_form":
        form = arguments.get("form", "")
        return f"form={form!r}"
    if tool_name == "frontend.summon_cards":
        header = str(arguments.get("header", ""))
        cards = arguments.get("cards", [])
        n = len(cards) if isinstance(cards, list) else 0
        return f"header={header[:30]!r} cards={n}"
    if tool_name == "frontend.dismiss_cards":
        return ""
    return ", ".join(f"{k}={str(v)[:30]!r}" for k, v in arguments.items())[:120]


def _summarize_tool_result(tool_name: str, result: str) -> str:
    """One-line human-readable result summary for tool_call_completed events.

    Detection order: error prefix > empty-results prefix > tool-specific
    heuristic > length fallback.
    """
    if not result:
        return "empty"
    if result.startswith("Tool error:") or result.startswith("Error"):
        return "error"
    if result.startswith("No information found"):
        return "0 facts"
    if tool_name == "query_knowledge_graph":
        # Compact mode result starts with "Focal:"; extract neighbor count.
        if result.startswith("Focal:"):
            if "Related (" in result:
                try:
                    count = int(result.split("Related (", 1)[1].split(")", 1)[0])
                    return f"focal + {count} related"
                except (ValueError, IndexError):
                    return "focal entity"
            return "focal entity"
        # Full mode result has '\n- ' fact bullets.
        n_facts = result.count("\n- ")
        return f"{n_facts} facts" if n_facts > 0 else "results retrieved"
    if tool_name == "query_vault":
        if result.startswith("No vault content"):
            return "0 chunks"
        # Each chunk is rendered as a prose block separated by blank lines
        n_chunks = result.count("\n\n") + 1
        return f"{n_chunks} chunks"
    if tool_name == "frontend.switch_form":
        return result if result and len(result) < 80 else "form switched"
    if tool_name == "frontend.summon_cards":
        return "cards displayed"
    if tool_name == "frontend.dismiss_cards":
        return "dismissed"
    return f"{len(result)} chars"


_VALID_CARD_PATTERNS: frozenset[str] = frozenset({"lines", "dots", "schematic", "photo"})

# ADR-017 Wave 2 (vault_results): valid display hints the LLM can pass via
# the query_vault tool to recommend FE presentation mode. 'auto' lets the
# FE decide based on its own UX rules (count + content length); 'cards'
# and 'panel' are explicit overrides.
_VALID_VAULT_DISPLAY_HINTS: frozenset[str] = frozenset({"auto", "cards", "panel"})

# vault_results event: per-result snippet truncation length.
_VAULT_SNIPPET_MAX_CHARS: int = 200

# ADR-017 Wave 2: graph_subgraph layout caps.
_GRAPH_SUBGRAPH_NEIGHBOR_CAP: int = 6
_GRAPH_SUBGRAPH_DISTANT_MIN: int = 5
_GRAPH_SUBGRAPH_DISTANT_MAX: int = 8
_GRAPH_SUBGRAPH_DISTANT_RADIUS: float = 1.6


def _derive_note_title_from_path(path: str) -> str:
    """Best-effort human-readable title from a vault relative path.

    Strategy: take the filename stem, strip date prefixes, replace
    separators with spaces, title-case. Returns the path as-is if
    derivation fails. Cheap; no disk reads.

    Future enhancement: enrich at the sidecar layer so the chunk carries
    frontmatter.title directly when present (currently dropped at the
    retriever boundary). Tracked alongside the ADR-021 citation contract
    work since both surfaces benefit from richer chunk metadata.
    """
    if not path:
        return "Untitled"
    # Filename stem, no directory, no extension
    stem = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if stem.endswith(".md"):
        stem = stem[:-3]
    # Strip a leading YYYY-MM-DD- date prefix if present
    if len(stem) >= 11 and stem[4] == "-" and stem[7] == "-" and stem[10] == "-":
        stem = stem[11:]
    # Replace separators with spaces and title-case
    pretty = stem.replace("-", " ").replace("_", " ").strip()
    return pretty.title() if pretty else "Untitled"


def _chunk_id_for(path: str, section: str | None) -> str:
    """Stable chunk identifier from (path, section).

    Used by the FE to track a chunk across emits (e.g., re-querying the
    same content) and as a click-handle for navigation. SHA-1 truncated
    to 12 hex chars is plenty of entropy for an FE-side identifier.
    """
    import hashlib

    key = f"{path}::{section or ''}"
    # nosec B324: SHA-1 here is a non-security keyed identifier, not a digest of secrets.
    return hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _format_kg_compact_for_llm(facts: list[RetrievedFact]) -> str:
    """Compact LLM-facing summary of knowledge-graph retrieval results.

    Trades detail for context budget. The LLM sees the focal entity, a
    deduplicated list of neighbor labels (up to 6 to match the
    graph_subgraph layout cap), and a nudge to chain a follow-up query
    for entity-specific detail. The FE has the full graph rendered
    (graph_subgraph event carries focal + neighbors + edges + distant),
    so the user can browse the wider context while MIST stays focused
    on the focal in her reasoning prompt.

    Caller selects this via verbosity='compact' (the default). Pass
    verbosity='full' to get the legacy formatted_context with every
    fact verbatim -- meaningfully larger but useful when MIST needs to
    answer detailed questions in one turn.
    """
    if not facts:
        return "No graph results."
    top_fact = max(facts, key=lambda f: f.similarity_score)
    focal = top_fact.subject
    focal_type = top_fact.subject_type

    seen_neighbors: list[str] = []
    for f in facts:
        if f.subject == focal and f.object not in seen_neighbors:
            seen_neighbors.append(f.object)
            if len(seen_neighbors) >= 6:
                break

    lines = [f"Focal: {focal} ({focal_type})"]
    if seen_neighbors:
        lines.append(f"Related ({len(seen_neighbors)}): {', '.join(seen_neighbors)}")
    lines.append("(Graph rendered for user. Chain another query for entity-specific detail.)")
    return "\n".join(lines)


def _build_vault_results_payload(
    query: str,
    facts: list[RetrievedFact],
    display_hint: str,
    total_results: int,
) -> dict[str, Any]:
    """Build an ADR-017 vault_results event payload from VaultNote facts.

    Each VaultNote fact carries the underlying chunk in its `properties`
    map (path, text, sources). This helper extracts the FE-bound shape
    so the FE can render either as cards (snippet + note_title) or as
    an expandable panel (snippet + full_text + note_path).

    `display_hint` is the LLM-supplied presentation recommendation; the
    FE may honor or override per its own UX rules.
    """
    results: list[dict[str, Any]] = []
    for fact in facts:
        props = fact.properties or {}
        path = str(props.get("path", "") or "")
        # Sidecar returns "(file)" for file-level (heading-less) chunks; map back to None.
        section_raw = fact.object if fact.object else None
        section: str | None = None if section_raw in (None, "(file)") else str(section_raw)
        # `text` is the canonical key; `content` retained as deprecated alias.
        content = str(props.get("text") or props.get("content") or "")
        snippet = content[:_VAULT_SNIPPET_MAX_CHARS]
        if len(content) > _VAULT_SNIPPET_MAX_CHARS:
            snippet = snippet + "..."
        sources_raw = props.get("sources") or []
        sources = [str(s) for s in sources_raw if isinstance(s, str)]
        results.append(
            {
                "chunk_id": _chunk_id_for(path, section),
                "note_path": path,
                "note_title": _derive_note_title_from_path(path),
                "section": section,
                "snippet": snippet,
                "full_text": content,
                # Emit the distance-derived similarity carried from the sidecar
                # (Task 2), NOT fact.similarity_score (the RRF fusion score,
                # which read as a misleading uniform ~2% in the FE). None for
                # FTS-only hits; the FE renders null as a "lexical" indicator.
                "similarity": (
                    float(props["display_similarity"])
                    if props.get("display_similarity") is not None
                    else None
                ),
                "sources": sources,
            }
        )
    return {
        "type": "vault_results",
        "query": query,
        "total_results": total_results,
        "display_hint": display_hint,
        "results": results,
    }


def _build_graph_subgraph_payload(
    result: RetrievalResult,
    seed: str,
) -> dict[str, Any] | None:
    """Build an ADR-017 graph_subgraph event payload from a retrieval result.

    Returns None when result has no facts so callers can omit the emit
    (the FE keeps prior graphData when no event arrives).

    Layout:
    - focal: top-confidence fact's subject at the origin.
    - neighbors: up to _GRAPH_SUBGRAPH_NEIGHBOR_CAP distinct object entities
      across facts whose subject matches the focal id. Placed angularly on
      the unit circle (i / N around 2 pi). Weight is the per-fact similarity
      score (falls back to 1.0 if score is zero or negative).
    - distant: 5-8 background points at radius 1.6, seeded by `seed` so
      the FE re-bake is stable for identical inputs within a turn.

    `seed` should incorporate session_id and turn_index so different turns
    produce different distant fields and identical dispatches within one
    turn produce identical placements.
    """
    if not result.facts:
        return None

    top_fact = max(result.facts, key=lambda f: f.similarity_score)
    focal_id = top_fact.subject
    focal_kind = top_fact.subject_type

    neighbors_by_id: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for fact in result.facts:
        if fact.subject != focal_id:
            continue
        if fact.object in neighbors_by_id:
            continue
        if len(neighbors_by_id) >= _GRAPH_SUBGRAPH_NEIGHBOR_CAP:
            break
        neighbors_by_id[fact.object] = {
            "id": fact.object,
            "label": fact.object,
            "kind": fact.object_type,
            # ADR-017 line 118 requires `meta` on graph nodes. We do not yet
            # carry a per-entity description; emit empty string so the FE
            # never receives undefined for a documented field.
            "meta": "",
        }
        weight = float(fact.similarity_score) if fact.similarity_score > 0 else 1.0
        edges.append({"from": focal_id, "to": fact.object, "weight": weight})

    n = len(neighbors_by_id)
    neighbors: list[dict[str, Any]] = []
    for i, data in enumerate(neighbors_by_id.values()):
        angle = (2.0 * math.pi * i / n) if n > 0 else 0.0
        neighbors.append({**data, "x": math.cos(angle), "y": math.sin(angle)})

    # Non-security: deterministic UI layout seeded by caller (session+turn).
    # B311 is suppressed because pseudo-randomness is exactly what we want here.
    rng = random.Random(seed)  # nosec B311
    n_distant = rng.randint(_GRAPH_SUBGRAPH_DISTANT_MIN, _GRAPH_SUBGRAPH_DISTANT_MAX)
    distant: list[dict[str, float]] = []
    for _ in range(n_distant):
        theta = rng.uniform(0.0, 2.0 * math.pi)
        distant.append(
            {
                "x": _GRAPH_SUBGRAPH_DISTANT_RADIUS * math.cos(theta),
                "y": _GRAPH_SUBGRAPH_DISTANT_RADIUS * math.sin(theta),
            }
        )

    return {
        "type": "graph_subgraph",
        "focal": {
            "x": 0.0,
            "y": 0.0,
            "label": focal_id,
            "kind": focal_kind,
            # ADR-017 line 118 documents `meta` on the focal too.
            "meta": "",
        },
        "neighbors": neighbors,
        "edges": edges,
        "distant": distant,
    }


KNOWLEDGE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": (
                "Search the typed knowledge graph for structured facts and relationships"
                " about the user. The graph is your reasoning substrate -- typed entities"
                " and edges you can traverse for inference, multi-hop reasoning, and"
                " relational lookups.\n\n"
                "USE for: questions about specific entities/relationships/typed facts;"
                " multi-hop reasoning over the user's stack, projects, or learning;"
                " how-to or debugging questions whose answer depends on the user's"
                " libraries/tools/projects; explicit graph queries.\n\n"
                "DO NOT USE for: pure conversational filler (greetings, thanks);"
                " general-knowledge with no user-specific anchor; questions already"
                " answered by the vault prose in context; purely creative tasks"
                " without user-specific reasoning.\n\n"
                "Result format: default 'compact' returns only the focal entity"
                " plus the labels of related neighbors (count + names) for context-"
                "budget efficiency. The full graph is rendered for the user on the"
                " graph form. Chain another query_knowledge_graph or query_vault for"
                " entity-specific detail. Pass verbosity='full' when you must see"
                " every fact verbatim in one turn (avoid this for browsing-style"
                " questions; it bloats context fast)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": ("What to search for (e.g. 'Python', 'my projects')"),
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Optional filter by entity types (e.g. ['Technology'])"),
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": ("Optional filter by relationships (e.g. ['USES'])"),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum facts to retrieve (default 20)",
                        "default": 20,
                    },
                    "verbosity": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "default": "compact",
                        "description": (
                            "Result detail level. 'compact' (default) = focal +"
                            " neighbor labels; 'full' = every fact verbatim."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_vault",
            "description": (
                "Search the vault (your prose notes: sessions, decisions, identity,"
                " user fact sheets) via hybrid semantic + keyword retrieval.\n\n"
                "USE for: questions about past conversations, prior decisions, the"
                " user's notes on a topic, anything written down in prose. The"
                " vault is the canonical user-approved history.\n\n"
                "DO NOT USE for: typed-graph entity questions (use"
                " query_knowledge_graph for those); current-state lookups against"
                " live data sources; questions already answered by the prose in"
                " your current context.\n\n"
                "Pair with frontend.switch_form when displaying results would"
                " benefit from a panel form."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (natural language)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max chunks to return (1-10, default 5)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                    "display_hint": {
                        "type": "string",
                        "enum": list(_VALID_VAULT_DISPLAY_HINTS),
                        "default": "auto",
                        "description": (
                            "Suggested FE presentation. 'auto' lets the FE decide"
                            " based on count + content length; 'cards' for short"
                            " tile view; 'panel' for expandable long-form review."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "frontend.switch_form",
            "description": (
                "Switch the visible FE form. USE when the form needs to match"
                " what you're doing: 'graph' for showing knowledge graph results,"
                " 'cloud' for ambient or casual conversation, 'ring' for the"
                " default focused conversation view. Always provide a brief"
                " reason for the switch (one sentence) so the user has context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "form": {
                        "type": "string",
                        "enum": ["ring", "cloud", "graph"],
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short explanation of why switching (1 sentence)",
                    },
                },
                "required": ["form", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "frontend.summon_cards",
            "description": (
                "Display a panel of cards to the user. Use when the user asks to see"
                " options, results, or visual choices laid out as a panel. Cards are"
                " short labeled tiles in a 2x2 or 1xN layout depending on count. The"
                " pattern field selects the visual texture: 'lines' (default), 'dots',"
                " 'schematic', or 'photo'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "header": {
                        "type": "string",
                        "description": "Short panel title (1-4 words)",
                    },
                    "cards": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "label": {"type": "string"},
                                "pattern": {
                                    "type": "string",
                                    "enum": list(_VALID_CARD_PATTERNS),
                                    "default": "lines",
                                },
                            },
                            "required": ["id", "label"],
                        },
                        "minItems": 1,
                        "maxItems": 8,
                    },
                },
                "required": ["header", "cards"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "frontend.dismiss_cards",
            "description": (
                "Dismiss any panel of cards currently visible. Use when the user"
                " indicates they're done with the panel or want to move on."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Module-level system prompt templates (Cluster 3)
#
# Consolidated into one body + an optional identity header (Fix O).
#
# _STATIC_IDENTITY_HEADER — prepended when mist_context=None (legacy fallback)
#   so the prompt is self-contained without a MistContext persona block.
#
# _STATIC_SYSTEM_TEMPLATE_BODY — shared body used in both paths.
#   When mist_context is provided the persona block already introduces MIST,
#   so the header is omitted.
# ---------------------------------------------------------------------------

_STATIC_IDENTITY_HEADER = (
    "You are MIST, a conversational AI assistant with a personal knowledge graph.\n\n"
)

_STATIC_SYSTEM_TEMPLATE_BODY = """\
=== MEMORY ARCHITECTURE ===

You have two memory layers with distinct semantic roles.

NOTES (vault prose; relevant excerpts auto-surfaced below):
  - HISTORICAL and FACTUAL substrate -- past sessions, identity
    notes, user fact sheets, decisions, prior conversations
  - The relevant prose for this turn is already in the context below.
    Auto-retrieval covers most history-and-recall queries.
  - When the user asks about past sessions, decisions, or notes that
    the excerpts below do not fully answer, call query_vault for
    deeper search over the full note corpus.

KNOWLEDGE GRAPH (typed triples; tool-only access):
  - REASONING substrate -- structured entities, relationships, and
    inferred beliefs you can traverse and query
  - Access only via the query_knowledge_graph tool
  - Use when the question requires multi-hop reasoning, typed-fact
    lookup, or relational queries that prose cannot reliably answer

=== TOOL USAGE RULES ===

USE query_knowledge_graph when the question requires REASONING over
user-specific structured knowledge:

- User asks about specific entities, relationships, or typed facts
  ("what tech do I use for ML?", "which projects involve data?")
- User explicitly asks to query the graph
- Question requires multi-hop reasoning or traversal over the user's
  structure -- including how-to questions, debugging, or complex
  reasoning whose answer depends on the user's stack, libraries,
  tools, projects, or learning
- Vault prose is insufficient and the answer needs typed-fact lookup

USE query_vault when the question needs PROSE or HISTORY recall beyond
the excerpts already injected below:

- User asks what was discussed, decided, or written in past sessions
  and the injected excerpts do not contain the answer
- User references a note, decision, or session by name or date that
  the excerpts below do not cover
- Do NOT use query_vault for recall of specific facts you stated
  (decisions, names, tools, employers, dates) -- those are typed facts;
  use query_knowledge_graph. query_vault is for narrative/prose recall.

=== TOOL USAGE INVARIANTS ===

DO NOT call query_knowledge_graph OR query_vault when:

- The message is purely conversational -- greetings, acknowledgements,
  social closings ("Hi", "Good morning", "Thanks", "Sounds good",
  "That's helpful")
- The question is general-knowledge with no user-specific anchor
  (Python syntax in the abstract, how a public protocol works,
  capital of France, generic best practices)
- The vault prose below already contains a complete answer
- The question is purely creative with no user-specific reasoning
  required ("write me a poem", "tell a joke")

When unsure whether a tool is warranted, ASK YOURSELF: does the answer depend on user-specific knowledge that the prose below alone
cannot give? If yes, call the right tool (typed facts -> graph;
prose/history -> vault). If no, answer from prose or general
knowledge.
  Examples: "which database did I choose" -> query_knowledge_graph;
  "who recommended FastAPI" -> query_knowledge_graph; "what did we
  write in yesterday's session note" -> query_vault. Hedged or
  temporal phrasings of a fact ("have I decided X yet", "did I
  decide anything recently", "what was that tool I wanted to try
  again", "am I on track with my goals") are still typed-fact
  lookups -> query_knowledge_graph, NOT query_vault.

Knowledge from conversations is captured automatically -- you do not
need to extract it manually.

=== GUIDELINES ===

- Be conversational and natural
- Cite sources from the prose below when helpful
- When you can't decide, prefer the answer that minimizes fabrication
  risk -- call the tool when the answer would otherwise rely on
  inferring user-specific facts the prose does not state
"""


# ---------------------------------------------------------------------------
# User-profile always-inject (ADR-010)
#
# The curated users/<user_id>.md is the source of truth ABOUT the user
# (authored_by: user). When the user is known, its body is injected into
# EVERY turn's context as an always-present block -- exactly the way
# ConventionsLoader always injects vault-root MIST.md. This is independent of
# retrieval similarity/intent: a meta-query like "what do you know about me?"
# embeds closer to MIST's own first-person identity prose than to the user's
# third-person profile, so a similarity-gated retriever ranks the profile
# below the cutoff and never surfaces it. The user's own fact sheet must
# never be subject to a similarity gate; this block guarantees it is present.
#
# The auto-inject retrieval path de-duplicates against this block (the
# profile chunk is dropped from the "Relevant prose from your vault" assembly)
# so the body appears exactly once.
# ---------------------------------------------------------------------------


def _format_user_profile_block(body: str) -> str:
    """Wrap the curated user-profile body in an LLM-visible labeled block.

    Mirrors `ConventionsLoader.format_for_prompt`: a clearly delimited header
    so the model treats the content as the canonical user fact sheet rather
    than as one ranked retrieval hit among many.
    """
    return f"=== WHAT YOU KNOW ABOUT THE USER (user profile) ===\n\n{body.strip()}\n"


class ConversationHandler:
    """Handles conversations with knowledge graph integration.

    Uses MCP-like tool access pattern:
    - LLM decides autonomously when to query or extract
    - Tools available: query_knowledge_graph, extract_knowledge
    - No separate intent classification (LLM is smart enough)
    """

    def __init__(
        self,
        config: KnowledgeConfig,
        graph_store: GraphStore,
        extraction_pipeline: ExtractionPipeline,
        retriever: KnowledgeRetriever,
        llm_provider: StreamingLLMProvider,
        conventions_loader: ConventionsLoader,
        tool_usage_tracker: ToolUsageTracker | None = None,
        debug_logger: DebugJSONLLogger | None = None,
        budget_planner: ContextBudgetPlanner | None = None,
        vault_writer: VaultWriterProtocol | None = None,
        invalidation_bus: InvalidationBus | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        """Initialize conversation handler.

        Args:
            config: Knowledge system configuration
            graph_store: Neo4j graph store
            extraction_pipeline: Pipeline for automatic knowledge extraction.
            retriever: Pre-built knowledge retriever (supports hybrid retrieval).
            llm_provider: LLM inference provider (StreamingLLMProvider).
            conventions_loader: ConventionsLoader for vault-root MIST.md auto-load
                (ADR-014). Inserted as a user message in every turn's prompt between
                system messages and conversation history.
            tool_usage_tracker: Optional tracker for recording tool calls for
                skill derivation. When None, tool usage is not recorded.
            debug_logger: Optional DebugJSONLLogger for per-turn structured
                observability. Produced by `DebugJSONLLogger.from_env()`; when
                `MIST_DEBUG_JSONL` is unset the logger yields no-op records.
            budget_planner: Optional ContextBudgetPlanner (Cluster 6). When
                None and `config.context_budget.enabled` is True, one is
                constructed from `config.context_budget`. When disabled, the
                handler falls back to legacy pre-Cluster-6 message assembly.
            vault_writer: Optional VaultWriterProtocol (Cluster 8 Phase 5).
                When set, every successful turn appends to the vault session
                note. None preserves legacy pre-Cluster-8 behavior.
            invalidation_bus: Optional InvalidationBus (Phase 3 Task 21). When
                set, the handler subscribes `_on_vault_rebuild` to receive
                rebuild-completion events from the filewatcher and evict stale
                `_mist_context_cache` entries. When None, no subscription is
                registered and the cache is not driven by vault edits.
            now_fn: Injectable clock returning a tz-aware datetime. Used at the
                C-pattern user-snapshot writeback (`_maybe_refresh_user_vault`)
                to stamp `rendered_at`. Defaults to `lambda: datetime.now(UTC)`
                (real wall-clock -- unchanged production behavior). The replay
                path injects a FIXED clock so the user-snapshot timestamp is
                reproducible and the greedy chat reply does not diverge run to
                run (wired in `backend.factories.build_conversation_handler`).
        """
        self.config = config
        # Injectable clock (DI seam). Default = real wall-clock so production
        # behavior is unchanged; the replay path supplies a fixed value.
        self._now_fn: Callable[[], datetime] = now_fn or (lambda: datetime.now(UTC))
        self.graph_store = graph_store
        self._extraction_pipeline = extraction_pipeline
        self.retriever = retriever
        self._conventions_loader = conventions_loader
        self._tool_usage_tracker = tool_usage_tracker
        self._debug_logger = debug_logger

        # Cluster 8 Phase 5: optional vault layer write integration. When set,
        # every successful turn appends to the vault session note via
        # VaultWriter.append_turn_to_session. None means vault layer disabled
        # (legacy pre-Cluster-8 behavior preserved).
        self._vault_writer = vault_writer

        # Maps external session_id -> pre-allocated vault note path. Filled
        # lazily on first turn via vault_writer.session_path(...). Stable
        # for the session lifetime.
        self._vault_paths: dict[str, str] = {}

        # Tracks turn count per session for vault writes (independent of
        # event_store turn numbering -- vault numbering starts at 1 per session).
        self._vault_turn_counts: dict[str, int] = {}

        # In-flight background extraction tasks (task -> session_id). Strong
        # references keep them GC-safe; end_session drains a session's tasks
        # before the Summary/status flip, and aclose() drains everything at
        # shutdown so cancellation cannot land mid commit-protocol (which
        # would retire a belief without writing its successor).
        self._extraction_tasks: dict[asyncio.Task, str] = {}
        # Always-on extraction-failure counter (independent of the
        # MIST_DEBUG_JSONL gate) so a persistent failure is countable.
        self._extraction_failures: int = 0

        # Cluster 6: budget-aware context assembly. Planner constructed from
        # config when not injected; legacy behavior preserved when disabled.
        if budget_planner is not None:
            self._budget_planner: ContextBudgetPlanner | None = budget_planner
        elif getattr(config, "context_budget", None) and config.context_budget.enabled:
            self._budget_planner = ContextBudgetPlanner(config.context_budget)
        else:
            self._budget_planner = None

        # LLM provider (replaces ChatOllama)
        self._provider = llm_provider

        # Tool configuration
        self._tool_schemas = KNOWLEDGE_TOOL_SCHEMAS

        # Active sessions
        self.sessions: dict[str, ConversationSession] = {}

        # Event store (Layer 1) -- append-only conversation log
        self.event_store: EventStore | None = None
        es_config = config.event_store
        if es_config.enabled:
            try:
                self.event_store = EventStore(db_path=es_config.db_path)
                self.event_store.initialize()
                logger.info("Event store enabled at %s", self.event_store.db_path)
            except Exception as e:
                logger.error("Failed to initialize event store: %s", e, exc_info=True)
                self.event_store = None

        # Maps external session_id -> event store session_id
        self._es_session_ids: dict[str, str] = {}

        # Cluster 3: MistContext cached per session for persona injection.
        # Populated on first handle_message for a given session; stable until
        # clear_session or process restart.
        self._mist_context_cache: dict[str, MistContext] = {}

        # Phase 3 Task 21: optional InvalidationBus subscription.
        # When set, _on_vault_rebuild is called after each vault-file rebuild
        # (filewatcher -> GraphRegenerator -> bus) to evict stale cache entries.
        self._invalidation_bus: InvalidationBus | None = invalidation_bus
        if invalidation_bus is not None:
            invalidation_bus.subscribe(self._on_vault_rebuild)

        # Cluster 3: response post-filter for slop patterns
        self._slop_detector = SlopDetector()
        self._slop_max_regen_attempts = 2

        # ADR-017 Wave 2: per-turn FE-bound event buffer. Reset at the top of
        # every handle_message; appended during tool dispatch
        # (tool_call_started/completed, cards_summon/dismiss, graph_subgraph);
        # drained by handle_message_streaming as WSEvent yields before Token
        # chars so the bridge forwards them to the canonical message_queue.
        self._turn_ws_events: list[dict[str, Any]] = []

        # ADR-017 Wave 2: per-turn context used by tool handlers that need
        # session/turn awareness (graph_subgraph distant-points RNG seed).
        # Set at the top of handle_message; read inside dispatch handlers.
        self._current_session_id: str | None = None
        self._current_turn_index: int = 0

        logger.info("ConversationHandler initialized with model: %s", llm_provider.model)

    async def _handle_query_knowledge_graph(
        self,
        query: str,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        limit: int = 20,
        verbosity: str = "compact",
    ) -> str:
        """Execute the query_knowledge_graph tool.

        ADR-017 Wave 2: on successful retrieval with at least one fact,
        appends a graph_subgraph WS event to the per-turn buffer so the
        FE graph form can render the focal + neighbors. Empty retrievals
        skip the emit (the FE keeps prior graphData).

        LLM-context split (BE/FE differentiation):
        - 'compact' (default): the LLM-facing result is the focal entity
          plus a deduped neighbor-label list (up to 6). Optimizes for
          context budget. MIST chains a follow-up query for entity-specific
          detail.
        - 'full': the LLM-facing result is the legacy formatted_context
          with every fact verbatim. Use when answering a single-turn
          detailed question where chaining isn't practical.

        In both modes the FE receives the same graph_subgraph payload.
        """
        if verbosity not in ("compact", "full"):
            raise ValueError(f"invalid verbosity: {verbosity!r}; must be 'compact' or 'full'")
        try:
            filters = None
            if entity_types or relationship_types:
                filters = RetrievalFilters(
                    entity_types=entity_types,
                    relationship_types=relationship_types,
                )

            result = await self.retriever.retrieve(
                query=query, user_id="User", limit=limit, filters=filters
            )

            if result.total_facts == 0:
                return (
                    f"No information found for query: '{query}'. "
                    "You may want to ask the user about this topic."
                )

            # ADR-017 Wave 2: emit graph_subgraph alongside the textual
            # tool result so the FE renders the focal + depth-1 neighborhood
            # for this query. Seed RNG with session+turn for layout stability.
            graph_payload = _build_graph_subgraph_payload(
                result,
                seed=f"{self._current_session_id}:{self._current_turn_index}",
            )
            if graph_payload is not None:
                self._turn_ws_events.append(graph_payload)

            if verbosity == "full":
                return result.formatted_context
            return _format_kg_compact_for_llm(result.facts)

        except Exception as e:
            logger.error("Error querying knowledge graph: %s", e)
            return f"Error searching knowledge graph: {e!s}"

    async def _handle_query_vault(
        self,
        query: str,
        limit: int = 5,
        display_hint: str = "auto",
    ) -> str:
        """Execute the query_vault tool.

        Routes the query through the existing retriever with
        force_intent='historical' so it lands on the vault sidecar
        exclusively (ADR-010 invariant 4 -- prose queries target vault,
        not graph). Emits a vault_results WS event with structured chunk
        data for FE rendering while returning the LLM-facing formatted
        text from the retriever.
        """
        if display_hint not in _VALID_VAULT_DISPLAY_HINTS:
            raise ValueError(
                f"invalid display_hint: {display_hint!r};"
                f" must be one of {sorted(_VALID_VAULT_DISPLAY_HINTS)}"
            )
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1 or limit > 10:
            raise ValueError(f"limit must be between 1 and 10; got {limit}")

        try:
            result = await self.retriever.retrieve(
                query=query,
                user_id="User",
                limit=limit,
                force_intent="historical",
            )

            # Zero-result short-circuit: skip the vault_results emit so the FE
            # keeps prior state, and signal absence to the LLM. Has to run
            # before the emit branch to avoid the "FE sees results panel +
            # LLM tells user no results" inconsistency.
            if result.total_facts == 0:
                return (
                    f"No vault content found for query: '{query}'."
                    " You may want to ask the user about this topic."
                )

            # The curated profile is ALWAYS-injected as its own block; the
            # tool path must mirror the auto-inject dedup or the body appears
            # twice in one prompt (deep review phase1-conversation-2). The
            # sidecar keeps the profile indexed on purpose -- only the
            # injection layer dedups.
            profile_path = self._resolve_user_profile_path()
            if profile_path is not None:
                is_profile = self._profile_fact_matcher(profile_path)
                kept = [f for f in result.facts if not is_profile(f)]
                if len(kept) != len(result.facts):
                    result.facts = kept
                    result.total_facts = len(kept)
                    if not kept:
                        return (
                            f"No vault content found for query: '{query}'."
                            " You may want to ask the user about this topic."
                        )
                    result.formatted_context = self.retriever.format_context(
                        kept, query, intent=result.intent
                    )

            vault_facts = [f for f in result.facts if f.subject == "VaultNote"]
            if vault_facts:
                self._turn_ws_events.append(
                    _build_vault_results_payload(
                        query=query,
                        facts=vault_facts,
                        display_hint=display_hint,
                        total_results=result.total_facts,
                    )
                )
            return result.formatted_context
        except Exception as e:
            logger.error("Error querying vault: %s", e)
            return f"Error searching vault: {e!s}"

    async def _handle_switch_form(self, form: str, reason: str) -> str:
        """Execute the frontend.switch_form tool.

        Appends a form_switch WS event with the target form + reason per
        the ADR-017 canonical shape. Returns a short summary to the LLM
        so the final-pass response can reference the switch contextually.
        """
        if form not in ("ring", "cloud", "graph"):
            raise ValueError(f"invalid form: {form!r}; must be one of ['ring', 'cloud', 'graph']")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        self._turn_ws_events.append(
            {
                "type": "form_switch",
                "form": form,
                "reason": reason,
            }
        )
        return f"Switched to {form} form"

    async def _handle_summon_cards(self, header: str, cards: list[dict[str, Any]]) -> str:
        """Execute the frontend.summon_cards tool.

        Normalizes each card's pattern (default 'lines'), validates against
        the closed enum, then appends a cards_summon WS event to the per-turn
        buffer per ADR-017 Wave 2. Returns a short string to the LLM so the
        final-pass response can reference what was displayed.
        """
        if not isinstance(cards, list) or not cards:
            raise ValueError("cards must be a non-empty list")
        normalized: list[dict[str, str]] = []
        for raw in cards:
            if not isinstance(raw, dict):
                raise ValueError(f"card must be an object; got {type(raw).__name__}")
            card_id = raw.get("id")
            label = raw.get("label")
            if not isinstance(card_id, str) or not card_id:
                raise ValueError("card.id must be a non-empty string")
            if not isinstance(label, str) or not label:
                raise ValueError("card.label must be a non-empty string")
            pattern = raw.get("pattern", "lines")
            if pattern not in _VALID_CARD_PATTERNS:
                raise ValueError(
                    f"invalid card.pattern: {pattern!r};"
                    f" must be one of {sorted(_VALID_CARD_PATTERNS)}"
                )
            normalized.append({"id": card_id, "label": label, "pattern": pattern})
        self._turn_ws_events.append(
            {"type": "cards_summon", "panel": {"header": header, "cards": normalized}}
        )
        return f"Displayed {len(normalized)} cards"

    async def _handle_dismiss_cards(self) -> str:
        """Execute the frontend.dismiss_cards tool.

        Appends a cards_dismiss WS event (empty payload per ADR-017) to the
        per-turn buffer. Returns a short string to the LLM.
        """
        self._turn_ws_events.append({"type": "cards_dismiss"})
        return "Dismissed cards panel"

    async def _dispatch_tool(self, tool_call: LLMToolCall) -> str:
        """Dispatch a tool call to its handler.

        Raises ToolNotFoundError when the LLM calls a tool name not in the
        handler registry. The observability wrap catches and emits as a
        tool_call_completed with error set; the LLM-facing result is the
        sentinel string so the LLM can self-correct on the next turn.
        """
        handlers = {
            "query_knowledge_graph": self._handle_query_knowledge_graph,
            "query_vault": self._handle_query_vault,
            "frontend.switch_form": self._handle_switch_form,
            "frontend.summon_cards": self._handle_summon_cards,
            "frontend.dismiss_cards": self._handle_dismiss_cards,
        }
        handler = handlers.get(tool_call.name)
        if handler is None:
            raise _ToolNotFoundError(tool_call.name)
        return await handler(**tool_call.arguments)

    async def _dispatch_tool_with_observability(self, tool_call: LLMToolCall) -> str:
        """Dispatch a tool call with FE-bound observability events.

        Appends tool_call_started before _dispatch_tool runs and
        tool_call_completed after (success or failure) into the per-turn
        WS event buffer. Returns the tool result string; on exception,
        substitutes a "Tool error: ..." sentinel and records a sanitized
        error label on the completed event (type name + message; no
        repr() that could leak module paths or internal state).

        Duration via perf_counter to avoid wall-clock drift, in integer
        milliseconds. Events drain via handle_message_streaming as WSEvent
        yields per ADR-017 Wave 2.
        """
        # Propagate the provider-assigned id so FE events, message history,
        # and JSONL records all join on ONE tool_call_id (ADR-017 UUID
        # convention); mint only when the provider sent none.
        tool_call_id = tool_call.id or str(uuid.uuid4())
        self._turn_ws_events.append(
            {
                "type": "tool_call_started",
                "tool_call_id": tool_call_id,
                "name": tool_call.name,
                "args_summary": _summarize_tool_args(tool_call.name, tool_call.arguments),
            }
        )
        start = time.perf_counter()
        error: str | None = None
        try:
            tool_result = await self._dispatch_tool(tool_call)
        except _ToolNotFoundError as exc:
            tool_result = (
                f"Tool not found: {exc.tool_name}." " Pick a tool from the registered catalog."
            )
            error = f"ToolNotFound: {exc.tool_name}"
        except Exception as exc:
            tool_result = f"Tool error: {type(exc).__name__}: {exc}"
            # Sanitized FE-facing error: type name + message only. repr(exc)
            # would leak module paths, internal kwargs, and Python frame
            # detail that the FE has no business displaying.
            error = f"{type(exc).__name__}: {exc}"
        duration_ms = int((time.perf_counter() - start) * 1000)
        self._turn_ws_events.append(
            {
                "type": "tool_call_completed",
                "tool_call_id": tool_call_id,
                "name": tool_call.name,
                "duration_ms": duration_ms,
                "result_summary": _summarize_tool_result(tool_call.name, tool_result),
                "error": error,
            }
        )
        return tool_result

    def get_or_create_session(self, session_id: str, user_id: str = "User") -> ConversationSession:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id, user_id=user_id)
            logger.info(f"Created new session: {session_id}")

        return self.sessions[session_id]

    async def _get_or_fetch_mist_context(self, session_id: str) -> MistContext:
        """Return cached MistContext for the session or fetch + cache on miss.

        Fresh persona retrieval once per session lifetime; stable thereafter.
        Clear by calling clear_session(session_id) or restarting the handler.

        Failures from retrieve_mist_context (missing method, graph errors, etc.)
        fall back to an empty MistContext with a warning log so existing tests
        and callers with minimally-mocked retrievers remain green.
        """
        cached = self._mist_context_cache.get(session_id)
        if cached is not None:
            return cached
        try:
            ctx = await self.retriever.retrieve_mist_context()
        except AttributeError as e:
            # Legacy retriever without the method — fall back to empty persona.
            logger.warning(
                "retrieve_mist_context unavailable on retriever for session %s: %s; "
                "using empty persona.",
                session_id,
                e,
            )
            ctx = MistContext(
                display_name="MIST",
                pronouns="she/her",
                self_concept="",
                traits=[],
                capabilities=[],
                preferences=[],
            )
        self._mist_context_cache[session_id] = ctx
        return ctx

    async def _on_vault_rebuild(self, event: RebuildResult) -> None:
        """Evict mist_context cache entries affected by a vault rebuild.

        Subscribed to InvalidationBus on __init__ (when bus is provided).
        Coordination guarantee: filewatcher publishes AFTER graph rebuild
        completes, so the next mist_context fetch reads correct re-derived state.

        Eviction rules:
        - identity/mist.md -> clear ALL active sessions (persona changed)
        - users/<user>.md  -> clear sessions whose user_id matches the stem
        - other paths      -> no-op (sessions/*, decisions/*, etc.)
        """
        parts = event.path.parts

        if "identity" in parts and event.path.name == "mist.md":
            # Persona definition changed -- all cached contexts are stale.
            cleared_count = len(self._mist_context_cache)
            self._mist_context_cache.clear()
            logger.info(
                "_on_vault_rebuild: identity/mist.md rebuilt; cleared all %d session caches",
                cleared_count,
            )
            return

        if "users" in parts:
            user_id = event.path.stem
            # Targeted eviction: keep caches for sessions belonging to other users.
            # user_id is resolved from self.sessions, which is populated by
            # get_or_create_session (called on every handle_message). Sessions
            # with no entry in self.sessions have no associated user_id and are
            # left untouched (conservative: don't evict what we can't classify).
            stale_sids = {
                sid
                for sid, session in self.sessions.items()
                if session.user_id == user_id and sid in self._mist_context_cache
            }
            for sid in stale_sids:
                del self._mist_context_cache[sid]
            if stale_sids:
                logger.info(
                    "_on_vault_rebuild: users/%s.md rebuilt; evicted %d session cache(s): %s",
                    user_id,
                    len(stale_sids),
                    sorted(stale_sids),
                )
            return

        # Other paths (sessions/*, decisions/*) do not affect mist_context.
        logger.debug("_on_vault_rebuild: path=%s; no mist_context eviction needed", event.path)

    def _build_request(
        self,
        *,
        call_site: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> LLMRequest:
        """Construct an LLMRequest and dump kwargs on Pydantic validation failure.

        Cluster 5 pre-validation observability: when the Pydantic BaseModel
        validator raises (e.g. a future Bug C-class tool_calls schema drift),
        this method emits a `phase: "llm_request_raw"` record containing the
        raw kwargs BEFORE re-raising. The record is gated on
        `MIST_DEBUG_LLM_REQUESTS=1`; when the gate is off this is a cheap
        no-op pass-through to `LLMRequest(**kwargs)`.

        `call_site` is a short string identifying the construction location
        (e.g. "chat.initial", "chat.final", "chat.regen"). It is only used as
        metadata on the debug record.
        """
        try:
            return LLMRequest(**kwargs)
        except Exception as exc:  # noqa: BLE001 — want to dump for any validation failure
            if self._debug_logger is not None:
                # Best-effort safe serialization: messages and tools may contain
                # objects that don't JSON-serialize cleanly, but _emit uses
                # default=str to stringify non-serializable values.
                self._debug_logger.record_llm_request_dump(
                    request_dict=kwargs,
                    error_message=repr(exc),
                    call_site=call_site,
                    session_id=session_id,
                )
            raise

    async def _post_filter_response(
        self,
        initial_response: str,
        messages: list[dict],
        session_id: str,
    ) -> str:
        """Scan LLM response for critical slop. On detection, regenerate with a strict
        rider up to _slop_max_regen_attempts times. Fallback after cap:
        SlopDetector.strip_fixable mechanical cleanup + WARNING log.

        Regeneration uses conversation_temperature - 0.2 (floor 0.3) to tighten
        constraint-following without going fully deterministic, and tools=None
        because style correction does not need tool calls.
        """
        current_response = initial_response if initial_response is not None else ""

        for _attempt in range(self._slop_max_regen_attempts):
            findings = self._slop_detector.detect(current_response, severity_floor="critical")
            if not findings:
                return current_response

            violation_names = sorted({f.pattern_name for f in findings})
            rider_content = (
                f"Your previous response contained slop patterns: {', '.join(violation_names)}. "
                f"Regenerate the same semantic answer without these patterns. "
                f"Remember the HARD RULES: no emoji, no symbols, no arrows, plain text only."
            )

            rider_messages = [
                *messages,
                {"role": "assistant", "content": current_response},
                {"role": "user", "content": rider_content},
            ]
            rider_temp = max(0.3, round(self.config.llm.conversation_temperature - 0.2, 10))
            rider_request = self._build_request(
                call_site="chat.regen",
                session_id=session_id,
                messages=rider_messages,
                tools=None,
                temperature=rider_temp,
                max_tokens=self.config.llm.conversation_max_tokens,
            )
            with llm_call_context(
                session_id=session_id,
                call_site="chat.regen",
                pass_num=_attempt + 1,
            ):
                rider_response = await self._provider.invoke(rider_request)
            current_response = rider_response.content or ""

        # Cap reached with critical findings still present
        logger.warning(
            "Slop post-filter cap reached (session=%s); falling back to strip_fixable",
            session_id,
        )
        return self._slop_detector.strip_fixable(current_response)

    async def handle_message(
        self, user_message: str, session_id: str, user_id: str = "User", max_history: int = 10
    ) -> str:
        """Handle a user message with autonomous tool use.

        LLM decides autonomously whether to:
        1. Query knowledge graph for context
        2. Extract knowledge from user message
        3. Both
        4. Neither (just respond)

        Args:
            user_message: User's message
            session_id: Session identifier
            user_id: User identifier
            max_history: Maximum conversation history to include

        Returns:
            Assistant's response
        """
        # ADR-017 Wave 2: clear per-turn FE-bound event buffer. Events
        # (tool_call_*, cards_*, graph_subgraph) accumulate here during this
        # turn and are drained by handle_message_streaming as WSEvent yields.
        self._turn_ws_events = []
        self._current_session_id = session_id
        self._current_turn_index += 1

        # Get or create session
        session = self.get_or_create_session(session_id, user_id)

        # Add user message to history
        session.add_message("user", user_message)

        # Auto-inject runs against the vault sidecar only, per ADR-010
        # invariant 4 ("No semantic read cross-pollination. Reasoning
        # queries target graph. Prose/history queries target vault
        # sidecar."). The vault carries canonical user-approved prose
        # (sessions, decisions, identity, user fact sheets) and
        # MIST-architecture documents. Graph traversal is the explicit
        # `query_knowledge_graph` tool's job.
        #
        # Pre-fix history: auto-inject was running against the merged
        # graph + vector + sidecar pipeline, which leaked graph hits
        # into pass 1's system prompt as "Relevant knowledge from your
        # graph (query: <user query>)". The model treated that framing
        # as a definitive search result, producing FN (skipped tool when
        # off-topic graph hits looked personal) and FP (tool-called when
        # any user-related graph fact surfaced for an unrelated query).
        # Bias compounded as the graph filled with extracted noise.
        #
        # `force_intent="historical"` bypasses the query classifier and
        # routes the retriever exclusively to the vault sidecar. Graph
        # is left untouched here — reserved for the tool path.
        retrieval_result: RetrievalResult | None = None
        auto_inject_enabled = self.config.auto_inject_docs and len(user_message.split()) >= 3
        if auto_inject_enabled:
            try:
                retrieval_result = await self.retriever.retrieve(
                    query=user_message,
                    user_id=user_id,
                    limit=self.config.auto_inject_limit,
                    similarity_threshold=self.config.auto_inject_threshold,
                    session_id=session_id,
                    force_intent="historical",
                )
                logger.info(
                    "[AUTO-RAG] Vault-only retrieval: chunks=%d (intent=%s)",
                    retrieval_result.document_chunks_used,
                    retrieval_result.intent,
                )
            except Exception as e:
                logger.error("[AUTO-RAG] Error during vault-only retrieval: %s", e)
                retrieval_result = None

        mist_context = await self._get_or_fetch_mist_context(session_id)
        messages = self._build_messages(
            session,
            max_history,
            retrieval_result=retrieval_result,
            mist_context=mist_context,
            max_output_tokens=self.config.llm.conversation_max_tokens,
        )

        try:
            # LLM autonomously decides to use tools
            logger.info(f"Processing message in session {session_id}")

            request = self._build_request(
                call_site="chat.initial",
                session_id=session_id,
                messages=messages,
                tools=self._tool_schemas,
                temperature=self.config.llm.conversation_temperature,
                max_tokens=self.config.llm.conversation_max_tokens,
            )
            _llm_start_1 = time.time()
            with llm_call_context(
                session_id=session_id,
                call_site="chat.initial",
                pass_num=1,
            ):
                response = await self._provider.invoke(request)
            _llm_duration_1_ms = (time.time() - _llm_start_1) * 1000

            # Check if LLM made tool calls
            tool_calls = []
            tool_results = []
            final_response: LLMResponse | None = None
            _llm_duration_2_ms: float = 0.0

            if response.tool_calls:
                logger.info("[TOOLS] LLM made %d tool calls", len(response.tool_calls))

                # Execute tool calls
                for tc in response.tool_calls:
                    logger.info("[TOOLS] Executing tool: %s", tc.name)
                    logger.info("[TOOLS]   Args: %s", tc.arguments)

                    # ADR-017 Wave 2: dispatch with observability wrap. Emits
                    # tool_call_started before _dispatch_tool runs and
                    # tool_call_completed after (success or failure) into the
                    # per-turn buffer drained by handle_message_streaming.
                    tool_result = await self._dispatch_tool_with_observability(tc)

                    # Log the result (truncated if too long)
                    result_preview = (
                        tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    )
                    logger.info("[TOOLS]   Result: %s", result_preview)

                    tool_calls.append({"name": tc.name, "args": tc.arguments})
                    tool_results.append(
                        {
                            "name": tc.name,
                            "result": tool_result,
                            "tool_call_id": tc.id,
                        }
                    )

                    # Record tool usage for skill derivation
                    if self._tool_usage_tracker is not None:
                        from datetime import UTC

                        from backend.knowledge.extraction.tool_usage_tracker import (
                            ToolCallRecord,
                            classify_tool_type,
                        )

                        self._tool_usage_tracker.record(
                            ToolCallRecord(
                                tool_name=tc.name,
                                tool_type=classify_tool_type(tc.name),
                                context=str(tc.arguments)[:500],
                                success=not tool_result.startswith("Tool not found:"),
                                timestamp=datetime.now(UTC),
                                session_id=session_id,
                                event_id="",
                            )
                        )

                # Build assistant message with tool_calls for correlation
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content or "",
                    "tool_calls": [tc.to_openai_dict() for tc in response.tool_calls],
                }
                messages.append(assistant_msg)

                for result in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "content": result["result"],
                            "tool_call_id": result["tool_call_id"],
                        }
                    )

                # Get final response with tool results
                logger.info("[TOOLS] Generating final response with tool results...")
                final_request = self._build_request(
                    call_site="chat.final",
                    session_id=session_id,
                    messages=messages,
                    temperature=self.config.llm.conversation_temperature,
                    max_tokens=self.config.llm.conversation_max_tokens,
                )
                _llm_start_2 = time.time()
                with llm_call_context(
                    session_id=session_id,
                    call_site="chat.final",
                    pass_num=2,
                ):
                    final_response = await self._provider.invoke(final_request)
                _llm_duration_2_ms = (time.time() - _llm_start_2) * 1000
                assistant_message = final_response.content
                logger.info(
                    "[TOOLS] Final response: %s...",
                    assistant_message[:100],
                )

            else:
                # No-tool path: pass 1 declined to tool-call, the response
                # content is the final answer. Vault-only auto-inject was
                # already provided in pass 1's context; graph access is
                # reserved for the tool path that pass 1 declined to take.
                # Single LLM call — no pass 2 on this branch.
                assistant_message = response.content

            # Cluster 3: slop post-filter before storing/returning.
            # Uses the current messages list as context for any regeneration.
            if assistant_message is not None:
                assistant_message = await self._post_filter_response(
                    initial_response=assistant_message,
                    messages=messages,
                    session_id=session_id,
                )

            # Add assistant response to history
            session.add_message(
                "assistant",
                assistant_message,
                tool_calls=tool_calls if tool_calls else None,
                tool_results=tool_results if tool_results else None,
            )

            # --- Step 0: Vault path pre-allocation (ADR-010 Cluster 8 Phase 6) ---
            # Compute the vault session note path BEFORE any downstream write
            # so the path can be threaded through extraction -> curation ->
            # graph_writer for the load-bearing DERIVED_FROM edge. Pure
            # path computation, no I/O. Returns None when vault layer is
            # disabled, in which case extraction proceeds without vault-note
            # provenance (legacy pre-Phase-6 behavior).
            #
            # Phase 9: pass the user message so the slug can be derived from
            # the first utterance content rather than the opaque session_id.
            # On subsequent turns the cached path is reused regardless.
            vault_note_path = self._get_or_allocate_vault_path(
                session_id, first_utterance=user_message
            )

            # --- Event Store Write (Layer 1) ---
            # Synchronous, <5ms target. Happens BEFORE any async extraction.
            event_id, recorded_at = self._record_turn_event(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                context_window=messages,
                retrieval_result=retrieval_result,
                tool_calls=tool_calls if tool_calls else None,
            )

            # --- Vault Write deferred to extraction completion ---
            # ADR-011 bucket 2 (rebuild substrate): the vault append happens
            # inside `_extract_knowledge_async`, gated on extraction yielding
            # at least one entity OR one relationship. This implements the
            # 2026-05-06 canonical-vault-pattern decision to skip per-turn
            # appends for zero-extraction turns ("Hi"/"Thanks") - those
            # produce no graph state to anchor via DERIVED_FROM, so a vault
            # note for them is pure noise. Substantive turns still anchor
            # cleanly; the rebuild contract is preserved.

            # Debug JSONL: record this turn and attach the TurnRecord to the
            # background extraction task so the extraction phase flushes a
            # second JSONL line keyed by event_id.
            turn_record = None
            if self._debug_logger is not None and event_id:
                turn_record = self._debug_logger.begin_turn(
                    event_id=event_id,
                    session_id=session_id,
                    user_id=user_id,
                    utterance=user_message,
                )
                if retrieval_result is not None:
                    turn_record.record_retrieval(retrieval_result)
                turn_record.record_llm_response(response, pass_num=1, timing_ms=_llm_duration_1_ms)
                if final_response is not None:
                    turn_record.record_llm_response(
                        final_response, pass_num=2, timing_ms=_llm_duration_2_ms
                    )
                turn_record.flush_turn()

            # Fire-and-forget background extraction (also performs the
            # conditional vault append per ADR-011 bucket 2). Tracked so
            # end_session/aclose can drain instead of abandoning to GC.
            if event_id and len(user_message.split()) >= 3:
                task = asyncio.create_task(
                    self._extract_knowledge_async(
                        utterance=user_message,
                        assistant_message=assistant_message,
                        conversation_history=session.get_history(max_history),
                        event_id=event_id,
                        session_id=session_id,
                        turn_record=turn_record,
                        vault_note_path=vault_note_path,
                        recorded_at=recorded_at,
                    )
                )
                self._extraction_tasks[task] = session_id
                task.add_done_callback(lambda t: self._extraction_tasks.pop(t, None))

            return assistant_message

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            error_msg = f"I encountered an error: {str(e)}"
            session.add_message("assistant", error_msg)
            # Record the error turn to event store
            self._record_turn_event(
                session_id=session_id,
                user_message=user_message,
                assistant_message=error_msg,
            )
            return error_msg

    async def handle_message_streaming(
        self,
        user_message: str,
        session_id: str,
        user_id: str = "User",
        max_history: int = 10,
    ) -> AsyncIterator[StreamEvent]:
        """Streaming canonical conversation pipeline.

        v1: wraps handle_message and fake-streams the result character-by-character.
        All canonical pipeline behavior (retrieval, mist context injection, tool
        dispatch, slop filter, vault append, EventStore record, fire-and-forget
        extraction) is inherited unchanged from handle_message.

        Yields Token events for each character of the final response, then a
        single terminal Complete event carrying the full final_response and a
        duration_ms metric. Pattern matches Claude's per-turn shape: input goes
        in, the LLM executes whatever it needs, response streams out, Complete
        terminates. Caller (text client / voice TTS layer) decides presentation.

        Thinking and Filler events are reserved for future iterations and are
        not emitted by v1.

        v2 plan: invert relationship — make handle_message_streaming the
        canonical generator and have handle_message join its output to a string.
        Add provider-level streaming with tool_calls support to recover the
        ~5s LLM-side streaming benefit currently lost in v1's fake-stream
        approach.

        Args:
            user_message: User's message.
            session_id: Session identifier.
            user_id: User identifier.
            max_history: Maximum conversation history to include.

        Yields:
            Token events (one per character) followed by a terminal Complete event.
        """
        start = time.monotonic()
        response: str = ""
        try:
            response = await self.handle_message(
                user_message=user_message,
                session_id=session_id,
                user_id=user_id,
                max_history=max_history,
            )
        finally:
            # ADR-017 Wave 2: drain per-turn FE-bound events. The drain runs
            # under finally so it fires even if handle_message is ever
            # refactored to raise (current handle_message catches internally,
            # but this guards against regression). FE always sees the events
            # that were buffered before the failure point so tool_call_started
            # never orphans a missing tool_call_completed.
            # Count dispatched tools BEFORE clearing the heterogeneous buffer
            # (graph_subgraph / vault / cards events accumulate here too) so
            # stream_complete.tool_calls_used reflects reality instead of the
            # dataclass default 0 (deep review febe-observability-1).
            tool_calls_used = sum(
                1 for p in self._turn_ws_events if p.get("type") == "tool_call_started"
            )
            for event_payload in self._turn_ws_events:
                yield WSEvent(payload=event_payload)
            self._turn_ws_events = []

        duration_ms = (time.monotonic() - start) * 1000

        for char in response:
            yield Token(text=char)

        yield Complete(
            final_response=response,
            tool_calls_used=tool_calls_used,
            duration_ms=duration_ms,
        )

    async def _extract_knowledge_async(
        self,
        utterance: str,
        conversation_history: list[dict[str, str]],
        event_id: str,
        session_id: str,
        assistant_message: str = "",
        turn_record: TurnRecord | None = None,
        vault_note_path: str | None = None,
        recorded_at: str | None = None,
    ) -> None:
        """Fire-and-forget background extraction.

        Called via asyncio.create_task after every user turn.
        Failures are logged but never propagated.

        If `turn_record` is supplied, records extraction outcome + graph writes
        to the per-turn JSONL debug log (phase: "extraction", keyed by event_id).

        `vault_note_path` (ADR-010 Cluster 8 Phase 6): pre-allocated vault session
        note path for the current turn. Forwarded to `extract_from_utterance` so
        every entity written by the curation graph writer carries a DERIVED_FROM
        edge to its source vault note. None preserves legacy pre-Phase-6 behavior
        (no vault-note provenance edges).

        `assistant_message` (ADR-011 bucket 2 - 2026-05-06): the assistant's
        finalized response for this turn. After extraction completes, if the
        result yielded at least one entity OR one relationship, the vault
        append fires here (replacing the unconditional handle_message append).
        Empty default keeps backwards compat for callers that don't yet thread
        the response through.
        """
        _ex_start = time.time()
        try:
            # Propagate session_id + event_id to every nested extraction LLM
            # call (extraction.ontology, extraction.scope_classifier,
            # extraction.internal_derivation). The inner llm_call_context
            # blocks in those extractors only set call_site, and the
            # ContextVar merges with inner-precedence -- so this outer set
            # populates session_id + event_id on the emitted llm_call records
            # without changing extractor signatures. asyncio.create_task
            # captures the current ContextVar copy via copy_context(), so
            # this works even though extraction is fire-and-forget.
            with llm_call_context(session_id=session_id, event_id=event_id):
                result = await self._extraction_pipeline.extract_from_utterance(
                    utterance=utterance,
                    conversation_history=conversation_history,
                    event_id=event_id,
                    session_id=session_id,
                    vault_note_path=vault_note_path,
                    recorded_at=recorded_at,
                )
            # Log results at debug level. Result may be ValidationResult
            # (has .entities/.relationships) or CurationResult (has .write_result
            # with counts). Handle both without importing concrete types.
            if hasattr(result, "entities"):
                # ValidationResult path (curation disabled)
                entity_count = len(result.entities)
                rel_count = len(result.relationships)
            elif hasattr(result, "write_result"):
                # CurationResult path (curation enabled). Relationship counts
                # live on reconcile_result since the C2 cutover.
                wr = result.write_result
                entity_count = wr.entities_created + wr.entities_updated
                rr = result.reconcile_result
                rel_count = rr.appended + rr.structural
            else:
                entity_count = 0
                rel_count = 0

            if entity_count or rel_count:
                logger.debug(
                    "Background extraction: %d entities, %d relationships from '%s'",
                    entity_count,
                    rel_count,
                    utterance[:60],
                )

            if turn_record is not None:
                turn_record.record_extraction(
                    result,
                    duration_ms=(time.time() - _ex_start) * 1000,
                    parse_ok=True,
                )
                turn_record.flush_extraction()

            # ADR-011 bucket 2: conditional per-turn vault append. Skip the
            # session-note write when extraction yielded zero entities AND
            # zero relationships - those turns produce no graph state to
            # anchor via DERIVED_FROM, so a vault note for them is pure noise.
            # Substantive turns still write; the rebuild contract is preserved
            # because every entity that gets a DERIVED_FROM edge has its
            # vault note created here.
            await self._maybe_append_session_turn(
                session_id=session_id,
                user_message=utterance,
                assistant_message=assistant_message,
                extraction_result=result,
            )

            # ADR-011 bucket 1 / C-pattern: re-render users/<user_id>.md when
            # extraction touched user-scope (User entity or User-source/target
            # edge). Fire-and-forget; failures are logged but never propagate.
            # Skipped when vault layer disabled or extraction touched no
            # user-scope state. The graph snapshot reflects post-curation state
            # because curation completed before this point.
            await self._maybe_refresh_user_vault(result)
        except Exception as e:
            # Always-on structured signal: counter + ids + full traceback.
            # With the debug-JSONL gate off (default), this containment is the
            # ONLY record of a persistent extraction failure (Neo4j down, LLM
            # schema regression) while the user-visible reply keeps flowing.
            self._extraction_failures += 1
            logger.error(
                "Background extraction failed (non-fatal) "
                "[session=%s event=%s failure_count=%d]: %s",
                session_id,
                event_id,
                self._extraction_failures,
                e,
                exc_info=True,
            )
            if turn_record is not None:
                turn_record.record_extraction(
                    None,
                    duration_ms=(time.time() - _ex_start) * 1000,
                    parse_ok=False,
                )
                turn_record.flush_extraction()

    async def end_session(self, session_id: str | None = None) -> None:
        """Mark one (or all) active session(s) as completed.

        Gap #1 / ADR-011 bucket 2: invoked on session-end signal (WebSocket
        disconnect, idle timeout, explicit end). For each target session:

        1. Generate a MIST-authored end-of-session synthesis (gap #1b)
        2. Append the synthesis as `## Summary` above the sentinel
        3. Flip frontmatter `status: in-progress` -> `status: completed`

        Failure-isolated per Invariant 6: failures in synthesis or write are
        logged but never propagated to the caller. Synthesis errors do not
        block the status flip — a session note is closed even if MIST
        couldn't summarize it.

        Args:
            session_id: Specific session to end. When None, ends every
                tracked session_id.
        """
        if self._vault_writer is None:
            return

        targets: list[str]
        if session_id is None:
            targets = list(self._vault_paths.keys())
        else:
            targets = [session_id] if session_id in self._vault_paths else []

        for sid in targets:
            path = self._vault_paths.get(sid)
            if not path:
                continue
            # Step 0: drain the session's in-flight extraction tasks so a
            # background _maybe_append_session_turn cannot land turn blocks
            # AFTER the Summary/status flip (FE heartbeat-loss reconnects
            # routinely hit this window).
            await self._drain_extraction_tasks(session_id=sid)
            # Step 1+2: synthesis (gap #1b). Failure must not block status flip.
            try:
                synthesis = await self._generate_session_synthesis(sid)
                if synthesis:
                    await self._vault_writer.append_session_synthesis(path, synthesis)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Session synthesis failed for %s (non-fatal): %s", sid, exc)
            # Step 3: status flip (gap #1a). Always attempts.
            try:
                await self._vault_writer.mark_session_completed(path)
                logger.debug("Session %s marked completed at %s", sid, path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "end_session status flip failed for %s "
                    "(non-fatal, swallowed per Invariant 6): %s",
                    sid,
                    exc,
                )
            # A resumed conversation must allocate a FRESH note: appending
            # turns (and a second Summary on the next disconnect) to a
            # completed note corrupts the session record.
            self._vault_paths.pop(sid, None)
            self._vault_turn_counts.pop(sid, None)

    async def _drain_extraction_tasks(
        self, session_id: str | None = None, timeout: float = 60.0
    ) -> None:
        """Await in-flight extraction tasks (optionally one session's).

        Bounded by `timeout` so a hung llama-server cannot block shutdown;
        tasks still running after the bound are cancelled (their writes are
        MERGE-idempotent and replay convergently on a later re-extraction).
        """
        tasks = [
            t
            for t, sid in list(self._extraction_tasks.items())
            if not t.done() and (session_id is None or sid == session_id)
        ]
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for t in pending:
            t.cancel()
        if pending:
            logger.warning(
                "Cancelled %d extraction task(s) still running after %.0fs drain timeout",
                len(pending),
                timeout,
            )

    async def aclose(self) -> None:
        """Drain all in-flight extraction tasks (server shutdown hook).

        Mirrors GraphRegenerator.aclose (Phase 5.5 Fix A): without the drain,
        loop teardown cancels extraction mid commit-protocol -- which can
        permanently retire a belief without writing its successor -- and
        silently drops the turn's vault append and DERIVED_FROM anchoring.
        """
        await self._drain_extraction_tasks(session_id=None)

    async def _generate_session_synthesis(self, session_id: str) -> str | None:
        """Build a session-end synthesis via one LLM call.

        Reads the session's accumulated turn history and asks the LLM to
        produce a markdown body with subsections: What Was Accomplished,
        Decisions Made, Next Actions, Context for Next Session. Mirrors the
        end-of-session protocol the user (Raj) follows in his own
        Claude+Obsidian workflow.

        Returns the markdown body (without a leading `## Summary` header --
        the writer adds that). Returns None when the session has no turns
        worth synthesizing (one-turn sessions, empty content).
        """
        session = self.sessions.get(session_id)
        if session is None or len(session.messages) < 2:
            return None  # no substantive content to summarize

        transcript_parts: list[str] = []
        for msg in session.messages:
            role = msg.role.upper()
            transcript_parts.append(f"**{role}:** {msg.content}\n")
        transcript = "".join(transcript_parts)

        prompt = (
            "You are MIST writing a session-end summary for the user's "
            "persistent memory vault. The conversation just ended. Read the "
            "transcript and produce a concise markdown body with the four "
            "subsections below. If a subsection has no content, write "
            "`(none)`. Do NOT include a leading `## Summary` header -- only "
            "the subsection content.\n\n"
            "### What Was Accomplished\n"
            "<bullet list of substantive accomplishments, each one line>\n\n"
            "### Decisions Made\n"
            "<bullet list of explicit decisions, each one line, or (none)>\n\n"
            "### Next Actions\n"
            "<bullet list of action items the user or MIST committed to, each "
            "one line, or (none)>\n\n"
            "### Context for Next Session\n"
            "<one to three sentences of prose summarizing what would be "
            "useful to remember when picking this conversation back up>\n\n"
            "---\n\n"
            "TRANSCRIPT:\n\n" + transcript
        )

        try:
            request = LLMRequest(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm.conversation_temperature,
                max_tokens=self.config.llm.conversation_max_tokens,
                top_p=0.9,
            )
            with llm_call_context(
                session_id=session_id,
                call_site="session.synthesis",
                pass_num=1,
            ):
                response = await self._provider.invoke(request)
            content = (response.content or "").strip()
            return content or None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Session synthesis LLM call failed for %s: %s", session_id, exc)
            return None

    async def _maybe_append_session_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        extraction_result: Any,
    ) -> None:
        """ADR-011 bucket 2: conditional per-turn vault session-note append.

        Skipped when extraction yielded zero entities AND zero relationships.
        Such turns ("Hi", "Thanks") produce no graph state, so a session-note
        block for them is pure noise. The rebuild contract is preserved
        because every extracted entity's DERIVED_FROM edge points to a vault
        note that DOES exist (lazily created on first append).

        Failure-isolated per Invariant 6: vault write errors are logged but
        never propagate.
        """
        if self._vault_writer is None:
            return
        if not assistant_message:
            return  # nothing to write
        entities = getattr(extraction_result, "validated_entities", None)
        relationships = getattr(extraction_result, "validated_relationships", None)
        if entities is None and hasattr(extraction_result, "entities"):
            entities = extraction_result.entities
        if relationships is None and hasattr(extraction_result, "relationships"):
            relationships = extraction_result.relationships
        entities = entities or []
        relationships = relationships or []
        if not entities and not relationships:
            logger.debug(
                "Skipping vault append for session %s: extraction yielded "
                "zero entities and zero relationships (ADR-011 bucket 2)",
                session_id,
            )
            return
        try:
            await self._write_to_vault(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Conditional vault append failed for session %s "
                "(non-fatal, swallowed per Invariant 6): %s",
                session_id,
                exc,
            )

    async def _maybe_refresh_user_vault(self, extraction_result: Any) -> None:
        """C-pattern user.md auto-render trigger (ADR-011 bucket 1).

        Inspects the extraction result for any User-scope state change. If
        present and vault layer is enabled, queries a 1-hop graph snapshot
        of the User entity + outbound neighbors, renders the markdown body,
        and calls VaultWriter.upsert_user. Fire-and-forget; never raises.

        Skipped when:
        - Vault writer is None (vault layer disabled)
        - Extraction touched zero user-scope entities/relationships
        - Result type doesn't expose validated_entities/relationships
          (defensive fallback for unexpected pipeline shapes)
        """
        if self._vault_writer is None:
            return
        # Curation-enabled path returns CurationResult with
        # validated_entities/relationships (added 2026-05-06). The
        # validation-only path returns ValidationResult which has the same
        # field names. We accept both via duck-typing.
        entities = getattr(extraction_result, "validated_entities", None)
        relationships = getattr(extraction_result, "validated_relationships", None)
        if entities is None and hasattr(extraction_result, "entities"):
            entities = extraction_result.entities
        if relationships is None and hasattr(extraction_result, "relationships"):
            relationships = extraction_result.relationships
        if entities is None and relationships is None:
            return
        entities = entities or []
        relationships = relationships or []

        from backend.vault.user_snapshot import (
            extraction_touched_user_scope,
            query_user_snapshot,
            render_user_snapshot_body,
        )

        user_id = self._user_id_for_vault()
        if not extraction_touched_user_scope(entities, relationships, user_id=user_id):
            return

        try:
            from backend.knowledge.storage.graph_executor import GraphExecutor

            executor = GraphExecutor(self.graph_store.connection)
            # Injected clock (DI seam): production default is wall-clock; the
            # replay path supplies a FIXED instant so the snapshot's rendered_at
            # -- and therefore the currency-filter $now bound below -- is
            # reproducible across runs (see ConversationHandler.now_fn).
            rendered_at = self._now_fn().isoformat()
            snapshot = await query_user_snapshot(executor, user_id, rendered_at)
            body_md = render_user_snapshot_body(snapshot)
            # Write to the SEPARATE machine-owned derived snapshot file
            # (users/<user_id>-graph-snapshot.md), NOT the hand-curated
            # users/<user_id>.md. The curated profile is user-authoritative
            # (ADR-010 Invariant 5) and must never be clobbered by the
            # C-pattern writeback.
            await self._vault_writer.upsert_user_snapshot(
                user_id=user_id, body_markdown=body_md, rendered_at=rendered_at
            )
            logger.debug(
                "User vault refreshed (C-pattern): user_id=%s, %d edge types in snapshot",
                user_id,
                len(snapshot.edges_by_type),
            )
        except Exception as exc:  # noqa: BLE001
            # ADR-010 Invariant 6: vault write failure is recoverable; never
            # propagate to the conversation pipeline.
            logger.warning(
                "User vault refresh failed (non-fatal, swallowed per Invariant 6): %s",
                exc,
            )

    def _user_id_for_vault(self) -> str:
        """Return the canonical user_id for vault writes.

        Returns "user" by default to match the seed-bootstrap convention
        (`scripts/seed_data.yaml` creates the User entity with `id: "user"`,
        and `users/user.md` is what bootstrap_vault_from_seed writes to disk).
        `VaultConfig.default_user_id` is honored ONLY if it has been
        explicitly set to something other than the dataclass default "raj"
        (which is vestigial and inconsistent with seed behavior). This keeps
        single-user MIST predictable until VaultConfig is reconciled.
        """
        vault_config = getattr(self.config, "vault", None)
        if vault_config is not None:
            default_uid = getattr(vault_config, "default_user_id", None)
            if default_uid and default_uid not in ("raj",):
                return default_uid
        return "user"

    def _resolve_user_profile_path(self) -> Path | None:
        """Locate the curated `users/<user_id>.md` profile file on disk.

        Resolution order:
        1. Exact-case match `users/<resolved_user_id>.md` under the vault root.
        2. Case-insensitive fallback: the first file in `users/` whose stem
           equals the resolved user_id ignoring case. This handles the casing
           nuance where a request carries user_id "User" while the seed writes
           `users/user.md` to disk (`_user_id_for_vault` already lower-cases
           to "user" by default, but a configured `default_user_id` may differ
           in case from the on-disk filename).

        Returns the resolved `Path`, or None when the vault layer is disabled,
        no matching file exists, or the vault root / users dir is absent.
        Never raises.
        """
        try:
            vault_cfg = self.config.vault
            # Profile injection is a vault feature: with the layer disabled
            # the default root still points at the REAL mist-memory, and an
            # ungated read leaks the live profile into vault-less contexts
            # (surfaced by the budget-accounting fix: unit tests were
            # silently injecting the live curated profile).
            if vault_cfg is None or not vault_cfg.enabled:
                return None
            vault_root = Path(vault_cfg.root)
        except (AttributeError, TypeError):
            return None

        user_id = self._user_id_for_vault()
        users_dir = vault_root / "users"

        exact = users_dir / f"{user_id}.md"
        if exact.is_file():
            return exact

        # Case-insensitive fallback within users/.
        try:
            if not users_dir.is_dir():
                return None
            target = user_id.casefold()
            for candidate in sorted(users_dir.glob("*.md")):
                if candidate.stem.casefold() == target:
                    return candidate
        except OSError:
            return None
        return None

    def _load_user_profile_block(self) -> tuple[Path | None, str | None]:
        """Load + format the known user's curated profile for always-injection.

        Reads `users/<resolved_user_id>.md` (see `_resolve_user_profile_path`),
        strips the YAML frontmatter via `parse_frontmatter`, and wraps the
        markdown body in the labeled profile block. The profile is ~400 words,
        so this is a small always-on read; the OS page cache makes the per-turn
        cost negligible.

        Returns `(resolved_path, formatted_block)` on success, or
        `(None, None)` when the profile file is absent, unreadable, or has an
        empty body. Graceful by contract: callers inject nothing on None and
        never error (ADR-010 Invariant 6 -- vault-read failure is recoverable).
        """
        path = self._resolve_user_profile_path()
        if path is None:
            return None, None
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read user profile %s (non-fatal): %s", path, exc)
            return None, None

        # Lazy import to avoid a module-level vault dependency in the hot path.
        from backend.vault.models import parse_frontmatter

        _frontmatter, body = parse_frontmatter(raw)
        if not body or not body.strip():
            return None, None
        return path, _format_user_profile_block(body)

    def _profile_fact_matcher(self, profile_path: Path):
        """Build a predicate matching facts sourced from the curated profile.

        Path matching is tolerant of the sidecar storing vault-RELATIVE paths
        (e.g. "users/user.md") while `profile_path` is absolute: we compare
        the normalized POSIX tail "users/<filename>". Shared by the
        auto-inject dedup AND the query_vault tool path so the profile body
        can never appear twice in one prompt.
        """
        try:
            vault_root = Path(self.config.vault.root)
            profile_rel = profile_path.relative_to(vault_root).as_posix()
        except (AttributeError, TypeError, ValueError):
            # profile_path not under vault_root (unexpected); fall back to the
            # users/<name> tail which is what the sidecar stores anyway.
            profile_rel = f"users/{profile_path.name}"

        profile_name = profile_path.name

        def _is_profile_fact(fact: RetrievedFact) -> bool:
            raw_path = str((fact.properties or {}).get("path", "") or "")
            if not raw_path:
                return False
            norm = raw_path.replace("\\", "/")
            return (
                norm == profile_rel
                or norm.endswith(f"/users/{profile_name}")
                or (norm == f"users/{profile_name}")
            )

        return _is_profile_fact

    def _dedup_profile_from_retrieval(
        self, retrieval_result: RetrievalResult, profile_path: Path
    ) -> None:
        """Drop the user-profile chunk from a retrieval result, in place.

        The curated profile is always-injected as its own block, so any copy
        the auto-inject retrieved must be removed to avoid a duplicate body.
        Filters `retrieval_result.facts` by source path, then re-renders
        `formatted_context` from the surviving facts via the retriever's
        canonical formatter so the budget-planner path (re-renders from facts)
        and the legacy path (reads formatted_context) stay consistent.
        """
        _is_profile_fact = self._profile_fact_matcher(profile_path)

        kept = [f for f in retrieval_result.facts if not _is_profile_fact(f)]
        if len(kept) == len(retrieval_result.facts):
            return  # profile chunk not in the auto-inject; nothing to dedup.

        dropped = len(retrieval_result.facts) - len(kept)
        retrieval_result.facts = kept
        retrieval_result.total_facts = len(kept)
        # document_chunks_used is best-effort metadata; decrement so it stays
        # consistent with the surviving vault chunks (floor at 0).
        retrieval_result.document_chunks_used = max(
            0, retrieval_result.document_chunks_used - dropped
        )
        # Re-render the LLM-facing context from the surviving facts via the
        # retriever's canonical public renderer (the exact function that
        # produced the original), preserving the historical "Relevant prose"
        # framing.
        retrieval_result.formatted_context = self.retriever.format_context(
            kept, retrieval_result.query, intent=retrieval_result.intent
        )
        logger.debug(
            "[USER-PROFILE] Deduped %d profile chunk(s) from auto-inject (path=%s)",
            dropped,
            profile_path,
        )

    def _record_turn_event(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        context_window: list[dict[str, str]] | None = None,
        retrieval_result: RetrievalResult | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> tuple[str | None, str | None]:
        """Record a conversation turn to the event store.

        Synchronous write, targets <5ms. Failures are logged but never
        propagated -- the event store must not break the conversation flow.

        Args:
            session_id: External session identifier.
            user_message: Raw user utterance.
            assistant_message: Raw system response.
            context_window: The full message list sent to the LLM.
            retrieval_result: Hybrid retrieval result from auto-RAG, if any.
            tool_calls: Tool calls made during this turn, if any.

        Returns:
            (event_id, recorded_at_iso) on success -- recorded_at is the event's
            UTC-aware timestamp, the fact-time authority for bitemporal edges
            (C1). (None, None) when the event store is disabled or on failure.
        """
        if self.event_store is None:
            return None, None

        try:
            # Ensure an event store session exists for this session_id
            if session_id not in self._es_session_ids:
                es_session_id = self.event_store.start_session(input_modality="text")
                self._es_session_ids[session_id] = es_session_id

            es_session_id = self._es_session_ids[session_id]

            # Determine turn_index from session turn_count
            es_session = self.event_store.get_session(es_session_id)
            turn_index = es_session.turn_count if es_session else 0

            # Build retrieval_context from RetrievalResult if present
            retrieval_context = None
            if retrieval_result and retrieval_result.total_facts > 0:
                retrieval_context = {
                    "intent": retrieval_result.intent,
                    "requires_mcp": retrieval_result.requires_mcp,
                    "fact_count": retrieval_result.total_facts,
                    "document_chunks_used": retrieval_result.document_chunks_used,
                }

            # UTC-aware fact-time: this exact instant becomes recorded_at on
            # every bitemporal edge the turn produces (never wall-clock at
            # write time, design 4.2).
            recorded_at = datetime.now(UTC)
            event = ConversationTurnEvent(
                session_id=es_session_id,
                turn_index=turn_index,
                timestamp=recorded_at,
                user_utterance=user_message,
                system_response=assistant_message,
                context_window=context_window,
                retrieval_context=retrieval_context,
                tool_calls=tool_calls,
                llm_model=self._provider.model,
                llm_parameters={"temperature": 0.7},
                ontology_version=self.config.ontology_version,
            )

            event_id = self.event_store.append_turn(event)
            logger.debug("Recorded turn event %s for session %s", event_id, session_id)
            return event_id, recorded_at.isoformat()

        except Exception as e:
            # Log but never propagate -- event store failure must not
            # break the conversation.
            logger.error(
                "Failed to record turn event for session %s: %s",
                session_id,
                e,
                exc_info=True,
            )
            return None, None

    def _get_or_allocate_vault_path(
        self,
        session_id: str,
        first_utterance: str | None = None,
    ) -> str | None:
        """Return the pre-allocated vault session-note path for `session_id`.

        ADR-010 Cluster 8 Phase 6 Step 0: pure path computation done once per
        session lifetime, returned synchronously so the path can be threaded
        through downstream writes (event store, vault append, extraction
        pipeline -> curation -> graph writer DERIVED_FROM emission) before
        any of them dispatch.

        Returns None when the vault layer is disabled (`vault_writer is None`),
        which causes downstream callers to skip vault-note provenance and
        retain legacy pre-Phase-6 behavior. Path computation never raises;
        on slug-derivation edge cases it falls back to the kebab-case
        sanitizer with `"session"` as the ultimate fallback.

        Phase 9 slug improvement: when `first_utterance` is supplied on the
        first call for this session, the slug is derived from significant
        words in the utterance (stopwords filtered, top tokens kebab-joined)
        instead of sanitizing the opaque `session_id`. This produces
        human-readable session-note filenames like
        `2026-04-22-vault-architecture.md` rather than
        `2026-04-22-2dc1-...-id.md`. The slug is fixed at first allocation
        and never changes for the session.

        Args:
            session_id: External session identifier.
            first_utterance: Optional first user message in the session,
                used to derive a content-meaningful slug. When None, falls
                back to sanitizing `session_id` (Phase 5/6 behavior).

        Returns:
            Absolute vault note path, or None if vault layer is disabled.
        """
        if self._vault_writer is None:
            return None

        cached = self._vault_paths.get(session_id)
        if cached is not None:
            return cached

        from datetime import UTC, datetime

        today = datetime.now(UTC).date().isoformat()
        if first_utterance is not None:
            slug = self._derive_session_slug_from_utterance(first_utterance, session_id)
        else:
            slug = self._derive_session_slug(session_id)
        path = self._vault_writer.session_path(today, slug)
        self._vault_paths[session_id] = path
        # Initialize the per-session vault turn counter only on first allocation.
        # Seed from disk if a session note already exists at `path` so backend
        # restart does not reset turn numbering for an ongoing session (e.g.,
        # session_id="default" reused across restarts -- the V6 unified-path
        # validation 2026-05-06 surfaced this as gap #4). When peek is not
        # available on the writer (e.g., legacy fakes), default to 0.
        if session_id not in self._vault_turn_counts:
            peek = getattr(self._vault_writer, "peek_turn_count", None)
            self._vault_turn_counts[session_id] = peek(path) if callable(peek) else 0
        return path

    async def _write_to_vault(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> str | None:
        """Append the current turn to the vault session note.

        Failure-isolated per ADR-010 Invariant 6: vault write errors are
        logged but never propagate. Returns the vault note path on success,
        None on failure or when vault layer is disabled.

        Path allocation is delegated to `_get_or_allocate_vault_path` so the
        same path can be reused by Phase 6's extraction-pipeline plumbing
        without recomputation.

        Args:
            session_id: External session identifier.
            user_message: Raw user utterance.
            assistant_message: Final assistant response.

        Returns:
            Absolute vault note path on success, None on failure / disabled.
        """
        if self._vault_writer is None:
            return None

        try:
            vault_path = self._get_or_allocate_vault_path(session_id)
            if vault_path is None:
                return None

            self._vault_turn_counts[session_id] += 1
            turn_index = self._vault_turn_counts[session_id]

            return await self._vault_writer.append_turn_to_session(
                session_id=session_id,
                turn_index=turn_index,
                user_text=user_message,
                mist_text=assistant_message,
                vault_note_path=vault_path,
            )
        except Exception as exc:  # noqa: BLE001
            # ADR-010 Invariant 6: vault write failure is recoverable from
            # event store. Log and continue.
            logger.warning(
                "Vault write failed for session %s (turn write swallowed per Invariant 6): %s",
                session_id,
                exc,
            )
            return None

    def _derive_session_slug(self, session_id: str) -> str:
        """Sanitize a session_id into a vault-compatible kebab-case slug.

        Used as a fallback when the first utterance is not available
        (e.g. legacy callers, tests). Phase 9 introduced
        `_derive_session_slug_from_utterance` as the preferred path.
        """
        import re

        slug = re.sub(r"[^a-z0-9-]+", "-", session_id.lower()).strip("-")
        if not slug:
            slug = "session"
        # Truncate to a reasonable length
        return slug[:50]

    # Stopwords filtered out of utterance-derived slugs. Mirrors the small
    # set used by ExtractionPipeline._compute_significance so slug derivation
    # uses the same notion of "significant" tokens.
    _SLUG_STOPWORDS: frozenset[str] = frozenset(
        {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "and",
            "or",
            "but",
            "if",
            "while",
            "about",
            "up",
            "it",
            "its",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "they",
            "them",
            "their",
            "this",
            "that",
            "these",
            "those",
            "what",
            "which",
            "who",
            "whom",
            "how",
            "when",
            "where",
            "why",
            "tell",
            "let",
            "please",
            "want",
            "need",
            "so",
            "just",
            "very",
            "much",
            "some",
            "any",
            "all",
            "no",
            "not",
        }
    )

    def _derive_session_slug_from_utterance(
        self,
        utterance: str,
        session_id: str,
    ) -> str:
        """Derive a kebab-case slug from significant words in the first utterance.

        ADR-010 "Session Slug Generation" specifies extracting the highest-
        confidence Project/Concept/Topic/Goal entity from the first 3
        utterances. Doing that synchronously at Step 0 would require either
        blocking the response on extraction or renaming the file mid-session
        (filewatcher / sidecar disruption). This method takes the pragmatic
        middle ground: derive significant non-stopword tokens from the first
        utterance and use them as the slug. The full entity-extraction-driven
        approach is documented as future work pending VaultWriter atomic-rename
        + filewatcher coordination support.

        Algorithm:
        1. Lowercase + tokenize (alphanumeric runs).
        2. Filter stopwords (see `_SLUG_STOPWORDS`) and short tokens (< 3 chars).
        3. Take the first 5 surviving tokens (preserve utterance order so
           subject-verb-object reading is preserved).
        4. Kebab-join.
        5. Cap at 50 chars.
        6. Fallback to a UUID-derived 8-char hex suffix on `session_id`
           when no significant tokens survive (matches ADR-010 fallback).

        Examples:
        - "Tell me about the vault architecture for MIST" -> "vault-architecture-mist"
        - "What's up?" -> "session-<hex8>"  (no significant tokens)
        - "Hi" -> "session-<hex8>"  (single short token)

        Args:
            utterance: Raw first user utterance for the session.
            session_id: Used only to seed the deterministic UUID fallback.

        Returns:
            Kebab-case slug, max 50 chars.
        """
        import hashlib
        import re

        tokens = re.findall(r"[a-z0-9]+", utterance.lower())
        significant = [t for t in tokens if len(t) >= 3 and t not in self._SLUG_STOPWORDS][:5]

        # 4-char session-id hash suffix guarantees per-session uniqueness even
        # when two sessions open with similar utterances. Stable per session_id.
        digest4 = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:4]

        if not significant:
            # No content tokens -- longer hash for the full identifier.
            digest8 = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]
            return f"session-{digest8}"

        # Cap content portion to leave room for the hash suffix while staying
        # under the 50-char total budget.
        content_slug = "-".join(significant)
        max_content_len = 50 - len(digest4) - 1  # -1 for the joining hyphen
        return f"{content_slug[:max_content_len]}-{digest4}"

    def _format_document_context(self, doc_results: list[dict[str, Any]]) -> str:
        """Format document search results for injection into context.

        Args:
            doc_results: List of document chunks from search_documents()

        Returns:
            Formatted string for system context
        """
        if not doc_results:
            return ""

        lines = ["=== MIST Documentation (Relevant Excerpts) ===\n"]

        for i, result in enumerate(doc_results, 1):
            source = result.get("source_title", "Unknown Document")
            text = result.get("text", "")
            similarity = result.get("similarity", 0.0)

            lines.append(f"[{i}] From: {source} (relevance: {similarity:.2f})")
            lines.append(f"{text}\n")

        lines.append("=" * 50)

        return "\n".join(lines)

    def _build_messages(
        self,
        session: ConversationSession,
        max_history: int,
        retrieval_result: RetrievalResult | None = None,
        mist_context: MistContext | None = None,
        max_output_tokens: int = 400,
    ) -> list[dict[str, str]]:
        """Build message list for LLM.

        Ordering:
          1. Persona block from MistContext (when provided) -- identity + HARD RULES
          2. Static system template -- tool availability, strategy, guidelines
          3. Retrieval context (when auto-RAG produced facts) — budget-pruned
          4. Live-data advisory (when retrieval requires MCP)
          5. Conversation history — budget-pruned via history strategy

        Cluster 6: when `self._budget_planner` is not None, a BudgetPlan is
        computed before composition. The planner prunes retrieval context by
        fact-priority score and history by strategy (sliding-window default)
        so the total prompt fits within
        `config.context_budget.context_window - max_output_tokens - reserves`.

        Args:
            session: Conversation session
            max_history: Maximum history messages to include (upper bound;
                the budget planner may prune further).
            retrieval_result: Optional hybrid retrieval result from auto-RAG
            mist_context: Optional MistContext to prepend as persona block.
                When None, falls back to the legacy full static template.
            max_output_tokens: Expected output budget for the coming LLM call.
                Subtracted from the total budget so pruning leaves headroom.

        Returns:
            List of messages for LLM, budget-compliant when planner is active.
        """
        # Compute persona + static text first so the planner can account for them.
        persona_text = mist_context.as_system_prompt_block() if mist_context is not None else None
        if mist_context is not None:
            static_template = _STATIC_SYSTEM_TEMPLATE_BODY
        else:
            static_template = _STATIC_IDENTITY_HEADER + _STATIC_SYSTEM_TEMPLATE_BODY

        # Always-inject the known user's curated profile (ADR-010). Resolved
        # once here so the same path drives BOTH the dedup below and the
        # injection further down. `profile_block` is the labeled context block
        # (None when no profile file exists -- graceful skip); `profile_path`
        # is the resolved on-disk file used to dedup the auto-inject copy.
        profile_path, profile_block = self._load_user_profile_block()

        # Dedup against the auto-inject: drop the profile's own chunk from the
        # retrieval result so the body is injected exactly once (at the
        # always-present block above), never duplicated inside the "Relevant
        # prose from your vault" assembly. Mutating `retrieval_result` in place
        # (facts + formatted_context) ensures BOTH the budget-planner path
        # (which re-renders from facts) and the legacy path (which reads
        # formatted_context) observe the deduped result consistently. The
        # sidecar INDEX is untouched -- the profile stays retrievable for
        # content queries (e.g. "Slalom"); only the injection layer dedups.
        if profile_path is not None and retrieval_result is not None:
            self._dedup_profile_from_retrieval(retrieval_result, profile_path)

        # Live-data advisory is a fixed-cost segment when present.
        live_advisory_text: str | None = None
        if retrieval_result and retrieval_result.requires_mcp and retrieval_result.suggested_tools:
            live_advisory_text = (
                "=== LIVE DATA ADVISORY ===\n"
                "This query appears to request real-time information. Consider using\n"
                "available tools for current data rather than relying on stored knowledge.\n"
                "Suggested tools: %s" % ", ".join(retrieval_result.suggested_tools)
            )

        raw_history = session.get_history(max_history)

        # Pre-load the conventions block so the planner can charge it as a
        # fixed segment (it is appended unconditionally below). Formatted
        # form is what gets counted -- that is what lands in the message.
        conventions_content = self._conventions_loader.load_vault_root()
        conventions_message = (
            self._conventions_loader.format_for_prompt(conventions_content)
            if conventions_content is not None
            else None
        )

        # --- Cluster 6: budget planning ---
        if self._budget_planner is not None:
            plan = self._budget_planner.plan(
                persona_text=persona_text,
                static_text=static_template,
                retrieval_result=retrieval_result,
                live_advisory_text=live_advisory_text,
                history=raw_history,
                tools=self._tool_schemas,
                max_output_tokens=max_output_tokens,
                # Always-injected blocks below MUST be charged as fixed cost
                # or they silently eat the 8K quality envelope past the
                # 256-token safety margin (deep review phase1-conversation-1).
                extra_fixed_texts=[
                    t for t in (conventions_message, profile_block) if t is not None
                ],
            )
            retrieval_text = plan.pruned_retrieval_text
            history = plan.pruned_history
            if not plan.fits:
                logger.warning(
                    "[BUDGET] Context budget exceeded: fixed_cost=%d total_budget=%d. "
                    "Degrading to minimal prompt (no retrieval, no history).",
                    plan.fixed_cost,
                    plan.total_budget,
                )
            elif plan.facts_dropped or len(history) < len(raw_history):
                logger.info(
                    "[BUDGET] Pruned: retrieval=%d used / %d budget (%d facts kept, %d dropped) | "
                    "history=%d used / %d budget (%d kept / %d raw) | fixed_cost=%d total=%d",
                    plan.retrieval_used,
                    plan.retrieval_budget,
                    plan.facts_kept,
                    plan.facts_dropped,
                    plan.history_used,
                    plan.history_budget,
                    len(history),
                    len(raw_history),
                    plan.fixed_cost,
                    plan.total_budget,
                )
        else:
            # Legacy behavior (budget disabled): full retrieval text, no history pruning.
            retrieval_text = (
                retrieval_result.formatted_context
                if retrieval_result and retrieval_result.total_facts > 0
                else None
            )
            history = raw_history

        # --- Compose messages ---
        # Ordering rationale (KV-cache discipline, G6 in parity audit v2.1):
        # llama.cpp's KV cache is prefix-based -- once a token mismatches, all
        # subsequent positions must re-process. So stable content goes at the
        # front; variable content goes at the tail. Order:
        #   1. Persona block       (stable within session; cached MistContext)
        #   2. Static template     (stable across all sessions)
        #   3. MIST.md             (stable across turns; mtime-cached)
        #   4. Retrieval text      (VARIES per turn -- breaks cache here)
        #   5. Live-data advisory  (conditional, varies)
        #   6. History             (variable, growing tail)
        # Pre-2026-05-25 the order was 1,2,4,5,3,6 which broke MIST.md cache
        # reuse every turn even though its content was unchanged.
        messages: list[dict[str, str]] = []

        # 1. Persona block (Cluster 3)
        if persona_text is not None:
            messages.append({"role": "system", "content": persona_text})

        # 2. Static system template
        messages.append({"role": "system", "content": static_template})

        # 3. Vault conventions (ADR-014): MIST.md auto-load as user message.
        # Mirrors Claude Code's CLAUDE.md user-message-after-system-prompt
        # position. Placed before retrieval to preserve KV-cache reuse across
        # turns (parity audit v2.1 G6). Omitted when no conventions file
        # exists (vault-less or test contexts).
        if conventions_message is not None:
            messages.append({"role": "user", "content": conventions_message})

        # 3b. User profile (ADR-010): always-inject the known user's curated
        # users/<user_id>.md body, independent of retrieval similarity/intent
        # -- the analog of the MIST.md conventions always-inject above. Placed
        # in the stable prefix (before variable retrieval/advisory/history) so
        # it does not break KV-cache reuse across turns, and deduped against
        # the auto-inject earlier so the body appears exactly once. Omitted
        # when no profile file exists (graceful skip).
        if profile_block is not None:
            logger.debug("[USER-PROFILE] Injecting curated profile from %s", profile_path)
            messages.append({"role": "user", "content": profile_block})

        # 4. Retrieval context (pruned by planner when active)
        if retrieval_text:
            if retrieval_result:
                logger.info(
                    "[AUTO-RAG] Injecting retrieval context: intent=%s, facts=%d, chunks=%d",
                    retrieval_result.intent,
                    retrieval_result.total_facts,
                    retrieval_result.document_chunks_used,
                )
            messages.append({"role": "system", "content": retrieval_text})

        # 5. Live-data advisory
        if live_advisory_text:
            messages.append({"role": "system", "content": live_advisory_text})

        # 6. Conversation history
        messages.extend(history)

        return messages

    def clear_session(self, session_id: str):
        """Clear a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            # End event store session
            if self.event_store and session_id in self._es_session_ids:
                try:
                    self.event_store.end_session(self._es_session_ids[session_id])
                except Exception as e:
                    logger.error("Failed to end event store session: %s", e)
                del self._es_session_ids[session_id]
            # Evict cached MistContext so the next session gets a fresh fetch.
            self._mist_context_cache.pop(session_id, None)
            logger.info(f"Cleared session: {session_id}")

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a session."""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "started_at": session.started_at.isoformat(),
            "message_count": len(session.messages),
            "last_message": session.messages[-1].content if session.messages else None,
        }
