"""Conversation Handler with MCP-like Tool Access.

Enables LLM to autonomously:
- Query knowledge graph for context
- Extract and store new knowledge
- Think and search database freely
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from backend.chat.context_budget import ContextBudgetPlanner
from backend.chat.mist_context import MistContext
from backend.chat.slop_detector import SlopDetector
from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.models import ConversationSession, RetrievalFilters, RetrievalResult
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm import LLMRequest, StreamingLLMProvider
from backend.llm.instrumented_provider import llm_call_context
from backend.llm.models import ToolCall as LLMToolCall

if TYPE_CHECKING:
    from backend.debug_jsonl_logger import DebugJSONLLogger, TurnRecord
    from backend.interfaces import VaultWriterProtocol
    from backend.knowledge.extraction.pipeline import ExtractionPipeline
    from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker

logger = logging.getLogger(__name__)

KNOWLEDGE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge_graph",
            "description": (
                "Search the knowledge graph for relevant information about the user. "
                "Use when: user asks about past info, preferences, or knowledge; "
                "you need context about entities, technologies, projects; "
                "you want to personalize based on what you know."
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
                },
                "required": ["query"],
            },
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

_STATIC_SYSTEM_TEMPLATE_BODY = """=== CONTEXT PROVIDED ===

You receive relevant context automatically with each query (see below).
This may include graph facts, document excerpts, or both depending on query type.
Knowledge from conversations is captured automatically -- you do not need to extract it manually.

=== AVAILABLE TOOLS ===

You have one tool at your disposal:

1. **query_knowledge_graph(query: str, limit: int = 20)**
   - Search the personal knowledge graph for user-specific information
   - Use when: User asks about THEIR preferences, skills, projects, past conversations
   - Returns: Facts you've learned about the user (entities + relationships)
   - Example: "What programming languages do I know?" -> use this tool

=== TOOL USAGE STRATEGY ===

**For user questions:**
- Technical questions (how/what/why about MIST) -> Use auto-provided context (already below)
- Personal questions (about user's info) -> Use query_knowledge_graph tool
- Both types -> Use context + query_knowledge_graph

**Autonomous Decision Making:**
- You decide when to use the tool - no one tells you when
- Context is automatically provided - use it! Cite sources when helpful.
- Only call query_knowledge_graph when you need personal user context

=== GUIDELINES ===

- Be conversational and natural
- Cite documentation sources when answering technical questions
- Use tools to enhance responses, not replace conversation
- Combine auto-provided context with your conversational abilities
- Query the knowledge graph when personal context matters

Remember: Context is already provided below. Think about whether you need personal user context from the knowledge graph."""


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
        tool_usage_tracker: ToolUsageTracker | None = None,
        debug_logger: DebugJSONLLogger | None = None,
        budget_planner: ContextBudgetPlanner | None = None,
        vault_writer: VaultWriterProtocol | None = None,
    ) -> None:
        """Initialize conversation handler.

        Args:
            config: Knowledge system configuration
            graph_store: Neo4j graph store
            extraction_pipeline: Pipeline for automatic knowledge extraction.
            retriever: Pre-built knowledge retriever (supports hybrid retrieval).
            llm_provider: LLM inference provider (StreamingLLMProvider).
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
        """
        self.config = config
        self.graph_store = graph_store
        self._extraction_pipeline = extraction_pipeline
        self.retriever = retriever
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

        # Cluster 3: response post-filter for slop patterns
        self._slop_detector = SlopDetector()
        self._slop_max_regen_attempts = 2

        logger.info("ConversationHandler initialized with model: %s", llm_provider.model)

    async def _handle_query_knowledge_graph(
        self,
        query: str,
        entity_types: list[str] | None = None,
        relationship_types: list[str] | None = None,
        limit: int = 20,
    ) -> str:
        """Execute the query_knowledge_graph tool."""
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

            return result.formatted_context

        except Exception as e:
            logger.error("Error querying knowledge graph: %s", e)
            return f"Error searching knowledge graph: {e!s}"

    async def _dispatch_tool(self, tool_call: LLMToolCall) -> str:
        """Dispatch a tool call to its handler."""
        handlers = {
            "query_knowledge_graph": self._handle_query_knowledge_graph,
        }
        handler = handlers.get(tool_call.name)
        if handler is None:
            return f"Tool not found: {tool_call.name}"
        return await handler(**tool_call.arguments)

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
        # Get or create session
        session = self.get_or_create_session(session_id, user_id)

        # Add user message to history
        session.add_message("user", user_message)

        # AUTO-INJECT: Hybrid retrieval (configurable)
        retrieval_result: RetrievalResult | None = None
        auto_inject_limit = self.config.auto_inject_limit
        auto_inject_threshold = self.config.auto_inject_threshold
        auto_inject_enabled = self.config.auto_inject_docs

        if auto_inject_enabled:
            # Skip auto-injection for very short messages
            if len(user_message.split()) >= 3:
                logger.info("[AUTO-RAG] Retrieving context for: '%s...'", user_message[:50])
                try:
                    retrieval_result = await self.retriever.retrieve(
                        query=user_message,
                        user_id=user_id,
                        limit=auto_inject_limit,
                        similarity_threshold=auto_inject_threshold,
                        session_id=session_id,
                    )
                    logger.info(
                        "[HYBRID-RAG] Intent: %s, confidence: %.2f, facts: %d, chunks: %d",
                        retrieval_result.intent,
                        retrieval_result.config_used.get("similarity_threshold", 0.0),
                        retrieval_result.total_facts,
                        retrieval_result.document_chunks_used,
                    )
                except Exception as e:
                    logger.error("[AUTO-RAG] Error during hybrid retrieval: %s", e)
                    retrieval_result = None
            else:
                logger.debug("[AUTO-RAG] Skipping retrieval for short message")

        # Build conversation with system prompt and optional retrieval context
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

            if response.tool_calls:
                logger.info("[TOOLS] LLM made %d tool calls", len(response.tool_calls))

                # Execute tool calls
                for tc in response.tool_calls:
                    logger.info("[TOOLS] Executing tool: %s", tc.name)
                    logger.info("[TOOLS]   Args: %s", tc.arguments)

                    # Dispatch to handler
                    tool_result = await self._dispatch_tool(tc)

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
                # No tool calls, use response directly
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
            event_id = self._record_turn_event(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                context_window=messages,
                retrieval_result=retrieval_result,
                tool_calls=tool_calls if tool_calls else None,
            )

            # --- Vault Write (Layer 2, ADR-010 Cluster 8 Phase 5) ---
            # Append turn to vault session note. Failure-isolated; logs and
            # continues. Pre-extraction so the file mtime change reaches the
            # filewatcher and triggers sidecar reindex without blocking the
            # user response.
            await self._write_to_vault(
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
            )

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
                if response.tool_calls:
                    turn_record.record_llm_response(
                        final_response, pass_num=2, timing_ms=_llm_duration_2_ms
                    )
                turn_record.flush_turn()

            # Fire-and-forget background extraction
            if event_id and len(user_message.split()) >= 3:
                asyncio.create_task(
                    self._extract_knowledge_async(
                        utterance=user_message,
                        conversation_history=session.get_history(max_history),
                        event_id=event_id,
                        session_id=session_id,
                        turn_record=turn_record,
                        vault_note_path=vault_note_path,
                    )
                )

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

    async def _extract_knowledge_async(
        self,
        utterance: str,
        conversation_history: list[dict[str, str]],
        event_id: str,
        session_id: str,
        turn_record: TurnRecord | None = None,
        vault_note_path: str | None = None,
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
        """
        _ex_start = time.time()
        try:
            result = await self._extraction_pipeline.extract_from_utterance(
                utterance=utterance,
                conversation_history=conversation_history,
                event_id=event_id,
                session_id=session_id,
                vault_note_path=vault_note_path,
            )
            # Log results at debug level. Result may be ValidationResult
            # (has .entities/.relationships) or CurationResult (has .write_result
            # with counts). Handle both without importing concrete types.
            if hasattr(result, "entities"):
                # ValidationResult path (curation disabled)
                entity_count = len(result.entities)
                rel_count = len(result.relationships)
            elif hasattr(result, "write_result"):
                # CurationResult path (curation enabled)
                wr = result.write_result
                entity_count = wr.entities_created + wr.entities_updated
                rel_count = wr.relationships_created
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
        except Exception as e:
            logger.error("Background extraction failed (non-fatal): %s", e)
            if turn_record is not None:
                turn_record.record_extraction(
                    None,
                    duration_ms=(time.time() - _ex_start) * 1000,
                    parse_ok=False,
                )
                turn_record.flush_extraction()

    def _record_turn_event(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        context_window: list[dict[str, str]] | None = None,
        retrieval_result: RetrievalResult | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> str | None:
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
            The event_id on success, None if event store is disabled or on failure.
        """
        if self.event_store is None:
            return None

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

            event = ConversationTurnEvent(
                session_id=es_session_id,
                turn_index=turn_index,
                timestamp=datetime.now(),
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
            return event_id

        except Exception as e:
            # Log but never propagate -- event store failure must not
            # break the conversation.
            logger.error(
                "Failed to record turn event for session %s: %s",
                session_id,
                e,
                exc_info=True,
            )
            return None

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
        # _write_to_vault increments it on every successful append.
        self._vault_turn_counts.setdefault(session_id, 0)
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
        messages: list[dict[str, str]] = []

        # 1. Persona block (Cluster 3)
        if persona_text is not None:
            messages.append({"role": "system", "content": persona_text})

        # 2. Static system template
        messages.append({"role": "system", "content": static_template})

        # 3. Retrieval context (pruned by planner when active)
        if retrieval_text:
            if retrieval_result:
                logger.info(
                    "[AUTO-RAG] Injecting retrieval context: intent=%s, facts=%d, chunks=%d",
                    retrieval_result.intent,
                    retrieval_result.total_facts,
                    retrieval_result.document_chunks_used,
                )
            messages.append({"role": "system", "content": retrieval_text})

        # 4. Live-data advisory
        if live_advisory_text:
            messages.append({"role": "system", "content": live_advisory_text})

        # 5. Conversation history
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
