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

from backend.chat.mist_context import MistContext
from backend.chat.slop_detector import SlopDetector
from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.models import ConversationSession, RetrievalFilters, RetrievalResult
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm import LLMRequest, StreamingLLMProvider
from backend.llm.models import ToolCall as LLMToolCall

if TYPE_CHECKING:
    from backend.debug_jsonl_logger import DebugJSONLLogger, TurnRecord
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
# _STATIC_SYSTEM_TEMPLATE_WITH_IDENTITY  — legacy path (mist_context=None)
#   Includes the "You are MIST..." opener so the prompt is self-contained.
#
# _STATIC_SYSTEM_TEMPLATE_WITHOUT_IDENTITY — persona path (mist_context provided)
#   Omits the opener because the MistContext persona block already introduces MIST.
# ---------------------------------------------------------------------------

_STATIC_SYSTEM_TEMPLATE_WITH_IDENTITY = """You are MIST, a conversational AI assistant with a personal knowledge graph.

=== CONTEXT PROVIDED ===

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

_STATIC_SYSTEM_TEMPLATE_WITHOUT_IDENTITY = """=== CONTEXT PROVIDED ===

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
        """
        self.config = config
        self.graph_store = graph_store
        self._extraction_pipeline = extraction_pipeline
        self.retriever = retriever
        self._tool_usage_tracker = tool_usage_tracker
        self._debug_logger = debug_logger

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
                {"role": "system", "content": rider_content},
            ]
            rider_temp = max(0.3, round(self.config.llm.conversation_temperature - 0.2, 10))
            rider_request = LLMRequest(
                messages=rider_messages,
                tools=None,
                temperature=rider_temp,
                max_tokens=400,
            )
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
        )

        try:
            # LLM autonomously decides to use tools
            logger.info(f"Processing message in session {session_id}")

            request = LLMRequest(
                messages=messages,
                tools=self._tool_schemas,
                temperature=self.config.llm.conversation_temperature,
                max_tokens=400,
            )
            _llm_start_1 = time.time()
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
                final_request = LLMRequest(
                    messages=messages,
                    temperature=self.config.llm.conversation_temperature,
                    max_tokens=400,
                )
                _llm_start_2 = time.time()
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
    ) -> None:
        """Fire-and-forget background extraction.

        Called via asyncio.create_task after every user turn.
        Failures are logged but never propagated.

        If `turn_record` is supplied, records extraction outcome + graph writes
        to the per-turn JSONL debug log (phase: "extraction", keyed by event_id).
        """
        _ex_start = time.time()
        try:
            result = await self._extraction_pipeline.extract_from_utterance(
                utterance=utterance,
                conversation_history=conversation_history,
                event_id=event_id,
                session_id=session_id,
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
    ) -> list[dict[str, str]]:
        """Build message list for LLM.

        Ordering:
          1. Persona block from MistContext (when provided) -- identity + HARD RULES
          2. Static system template -- tool availability, strategy, guidelines
          3. Retrieval context (when auto-RAG produced facts)
          4. Live-data advisory (when retrieval requires MCP)
          5. Conversation history

        Args:
            session: Conversation session
            max_history: Maximum history messages to include
            retrieval_result: Optional hybrid retrieval result from auto-RAG
            mist_context: Optional MistContext to prepend as persona block.
                When None, falls back to the legacy full static template
                (includes "You are MIST..." opener) so existing callers are
                unaffected.

        Returns:
            List of messages for LLM
        """
        messages: list[dict[str, str]] = []

        # 1. Persona block (Cluster 3)
        if mist_context is not None:
            messages.append({"role": "system", "content": mist_context.as_system_prompt_block()})
            # Use the template without the redundant "You are MIST..." opener
            # because the persona block already introduces MIST.
            static_template = _STATIC_SYSTEM_TEMPLATE_WITHOUT_IDENTITY
        else:
            # Fallback: no persona retrieved; keep the legacy full template.
            static_template = _STATIC_SYSTEM_TEMPLATE_WITH_IDENTITY

        # 2. Static system template
        messages.append({"role": "system", "content": static_template})

        # 3. Retrieval context (unchanged logic)
        if retrieval_result and retrieval_result.total_facts > 0:
            context_str = retrieval_result.formatted_context
            logger.info(
                "[AUTO-RAG] Injecting retrieval context: intent=%s, facts=%d, chunks=%d",
                retrieval_result.intent,
                retrieval_result.total_facts,
                retrieval_result.document_chunks_used,
            )
            messages.append({"role": "system", "content": context_str})

            # 4. Live-data advisory when query requires MCP tools
            if retrieval_result.requires_mcp and retrieval_result.suggested_tools:
                advisory = (
                    "=== LIVE DATA ADVISORY ===\n"
                    "This query appears to request real-time information. Consider using\n"
                    "available tools for current data rather than relying on stored knowledge.\n"
                    "Suggested tools: %s" % ", ".join(retrieval_result.suggested_tools)
                )
                messages.append({"role": "system", "content": advisory})

        # 5. Conversation history
        history = session.get_history(max_history)
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
