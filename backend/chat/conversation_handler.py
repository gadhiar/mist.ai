"""
Conversation Handler with MCP-like Tool Access

Enables LLM to autonomously:
- Query knowledge graph for context
- Extract and store new knowledge
- Think and search database freely
"""

import logging
from typing import Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.extraction.entity_extractor import EntityExtractor
from backend.knowledge.models import ConversationSession, RetrievalFilters
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    Handles conversations with knowledge graph integration.

    Uses MCP-like tool access pattern:
    - LLM decides autonomously when to query or extract
    - Tools available: query_knowledge_graph, extract_knowledge
    - No separate intent classification (LLM is smart enough)
    """

    def __init__(
        self, config: KnowledgeConfig, graph_store: GraphStore, model_name: str = "qwen2.5:7b"
    ):
        """
        Initialize conversation handler.

        Args:
            config: Knowledge system configuration
            graph_store: Neo4j graph store
            model_name: Ollama model to use
        """
        self.config = config
        self.graph_store = graph_store

        # Initialize components
        self.retriever = KnowledgeRetriever(config, graph_store)
        self.extractor = EntityExtractor(config.extraction)

        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
        )

        # Bind tools to LLM
        self._setup_tools()

        # Active sessions
        self.sessions: dict[str, ConversationSession] = {}

        logger.info(f"ConversationHandler initialized with model: {model_name}")

    def _setup_tools(self):
        """Setup LLM tools for MCP-like access"""

        # Store references for tool implementations
        retriever = self.retriever
        extractor = self.extractor
        graph_store = self.graph_store

        @tool
        async def query_knowledge_graph(
            query: str,
            entity_types: list[str] | None = None,
            relationship_types: list[str] | None = None,
            limit: int = 20,
        ) -> str:
            """Search the knowledge graph for relevant information about the user.

            Use this tool when:
            - User asks about their past information, preferences, or knowledge
            - You need context about entities, technologies, projects, or relationships
            - You want to personalize your response based on what you know

            Args:
                query: What to search for (e.g., "Python programming", "my projects", "backend technologies")
                entity_types: Optional filter by entity types (e.g., ["Technology", "Project"])
                relationship_types: Optional filter by relationships (e.g., ["USES", "WORKS_ON"])
                limit: Maximum facts to retrieve (default 20)

            Returns:
                Natural language context with relevant facts from the knowledge graph
            """
            try:
                filters = None
                if entity_types or relationship_types:
                    filters = RetrievalFilters(
                        entity_types=entity_types, relationship_types=relationship_types
                    )

                result = await retriever.retrieve(
                    query=query, user_id="User", limit=limit, filters=filters
                )

                if result.total_facts == 0:
                    return f"No information found for query: '{query}'. You may want to ask the user about this topic."

                return result.formatted_context

            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")
                return f"Error searching knowledge graph: {str(e)}"

        @tool
        async def extract_knowledge(text: str, context: str | None = None) -> str:
            """Extract and store new knowledge from text into the knowledge graph.

            Use this tool when:
            - User shares information about themselves (skills, preferences, projects, etc.)
            - You learn something new that should be remembered
            - User mentions technologies, tools, people, or relationships

            Args:
                text: The text to extract knowledge from (typically user's message)
                context: Optional context about the conversation

            Returns:
                Confirmation of what was learned and stored
            """
            try:
                import uuid
                from datetime import datetime

                # Ensure conversation event exists
                conversation_id = context or "default"
                graph_store.store_conversation_event(
                    conversation_id=conversation_id, user_id="User"
                )

                # Generate unique utterance ID
                utterance_id = str(uuid.uuid4())

                # Store the utterance
                graph_store.store_utterance(
                    utterance_id=utterance_id,
                    conversation_id=conversation_id,
                    text=text,
                    metadata={"source": "conversation", "timestamp": datetime.now().isoformat()},
                )

                # Extract entities from text
                graph_docs = await extractor.extract_from_utterance(
                    utterance=text, conversation_history=[], metadata={"utterance_id": utterance_id}
                )

                # Store extracted entities
                if graph_docs and graph_docs[0].nodes:
                    graph_store.store_extracted_entities(
                        graph_document=graph_docs[0],
                        utterance_id=utterance_id,
                        ontology_version=None,
                    )

                    entity_names = [n.id for n in graph_docs[0].nodes]
                    rel_count = len(graph_docs[0].relationships)

                    return f"Learned and stored: {len(entity_names)} entities ({', '.join(entity_names[:5])}{'...' if len(entity_names) > 5 else ''}) with {rel_count} relationships."
                else:
                    return "No new knowledge extracted from the text."

            except Exception as e:
                logger.error(f"Error extracting knowledge: {e}")
                return f"Error storing knowledge: {str(e)}"

        @tool
        async def extract_knowledge_from_document(
            chunk_id: str, reason: str | None = None
        ) -> str:
            """Extract entities from a specific document chunk into the knowledge graph.

            Use this tool when:
            - You found important information in documentation that should be remembered
            - User asks you to "remember" or "learn" something from the docs
            - You want to convert document facts into queryable knowledge graph entities
            - Information will be queried frequently (extract once, query fast forever)

            This is SELECTIVE extraction - only extract when truly useful.
            Most doc info is already available via auto-injected context.

            Args:
                chunk_id: The chunk_id from auto-provided documentation
                reason: Optional explanation of why you're extracting this

            Returns:
                Confirmation of entities extracted and stored
            """
            try:
                # Get the chunk text
                query = """
                MATCH (c:DocumentChunk {chunk_id: $chunk_id})
                RETURN c.text as text, c.chunk_id as chunk_id
                """
                results = graph_store.connection.execute_query(query, {"chunk_id": chunk_id})

                if not results:
                    return f"Chunk not found: {chunk_id}"

                chunk_text = results[0]["text"]

                # Extract entities from chunk
                graph_docs = await extractor.extract_from_utterance(
                    utterance=chunk_text,
                    conversation_history=[],
                    metadata={
                        "source": "document_extraction",
                        "chunk_id": chunk_id,
                        "reason": reason or "LLM-selected extraction",
                    },
                )

                # Store extracted entities
                if graph_docs and graph_docs[0].nodes:
                    graph_store.store_extracted_entities(
                        graph_document=graph_docs[0], chunk_id=chunk_id, ontology_version=None
                    )

                    entity_names = [n.id for n in graph_docs[0].nodes]
                    rel_count = len(graph_docs[0].relationships)

                    return f"Extracted from document: {len(entity_names)} entities ({', '.join(entity_names[:5])}{'...' if len(entity_names) > 5 else ''}) with {rel_count} relationships. These are now queryable via query_knowledge_graph."
                else:
                    return "No entities extracted from this chunk."

            except Exception as e:
                logger.error(f"Error extracting from document chunk: {e}")
                return f"Error extracting knowledge from document: {str(e)}"

        # Bind tools to LLM
        self.tools = [query_knowledge_graph, extract_knowledge, extract_knowledge_from_document]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        logger.info(
            "Tools bound to LLM: query_knowledge_graph, extract_knowledge, extract_knowledge_from_document"
        )

    def get_or_create_session(self, session_id: str, user_id: str = "User") -> ConversationSession:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id, user_id=user_id)
            logger.info(f"Created new session: {session_id}")

        return self.sessions[session_id]

    async def handle_message(
        self, user_message: str, session_id: str, user_id: str = "User", max_history: int = 10
    ) -> str:
        """
        Handle a user message with autonomous tool use.

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

        # AUTO-INJECT: Search documentation (configurable)
        doc_results = []
        auto_inject_limit = self.config.auto_inject_limit
        auto_inject_threshold = self.config.auto_inject_threshold
        auto_inject_enabled = self.config.auto_inject_docs

        if auto_inject_enabled:
            # Skip auto-injection for very short messages
            if len(user_message.split()) >= 3:
                logger.info(f"[AUTO-RAG] Searching documentation for: '{user_message[:50]}...'")
                try:
                    doc_results = await self.retriever.search_documents(
                        query=user_message,
                        limit=auto_inject_limit,
                        similarity_threshold=auto_inject_threshold,
                    )
                    logger.info(f"[AUTO-RAG] Found {len(doc_results)} relevant document chunks")
                except Exception as e:
                    logger.error(f"[AUTO-RAG] Error searching documents: {e}")
                    doc_results = []
            else:
                logger.debug("[AUTO-RAG] Skipping search for short message")

        # Build conversation with system prompt and optional doc context
        messages = self._build_messages(session, max_history, doc_context=doc_results)

        try:
            # LLM autonomously decides to use tools
            logger.info(f"Processing message in session {session_id}")

            response = await self.llm_with_tools.ainvoke(messages)

            # Check if LLM made tool calls
            tool_calls = []
            tool_results = []

            if hasattr(response, "tool_calls") and response.tool_calls:
                logger.info(f"[TOOLS] LLM made {len(response.tool_calls)} tool calls")

                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_call_id = tool_call.get("id", f"call_{tool_name}")

                    logger.info(f"[TOOLS] Executing tool: {tool_name}")
                    logger.info(f"[TOOLS]   Args: {tool_args}")

                    # Find and execute tool
                    tool_result = await self._execute_tool(tool_name, tool_args)

                    # Log the result (truncated if too long)
                    result_preview = (
                        tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    )
                    logger.info(f"[TOOLS]   Result: {result_preview}")

                    tool_calls.append({"name": tool_name, "args": tool_args})
                    tool_results.append(
                        {"name": tool_name, "result": tool_result, "tool_call_id": tool_call_id}
                    )

                # If tools were called, get final response with tool results
                messages.append({"role": "assistant", "content": response.content or ""})

                for result in tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "content": result["result"],
                            "name": result["name"],
                            "tool_call_id": result["tool_call_id"],
                        }
                    )

                # Get final response
                logger.info("[TOOLS] Generating final response with tool results...")
                final_response = await self.llm.ainvoke(messages)
                assistant_message = final_response.content
                logger.info(f"[TOOLS] Final response: {assistant_message[:100]}...")

            else:
                # No tool calls, use response directly
                assistant_message = response.content

            # Add assistant response to history
            session.add_message(
                "assistant",
                assistant_message,
                tool_calls=tool_calls if tool_calls else None,
                tool_results=tool_results if tool_results else None,
            )

            return assistant_message

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            error_msg = f"I encountered an error: {str(e)}"
            session.add_message("assistant", error_msg)
            return error_msg

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.ainvoke(tool_args)

        return f"Tool not found: {tool_name}"

    def _format_document_context(self, doc_results: list[dict[str, Any]]) -> str:
        """
        Format document search results for injection into context.

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
        doc_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, str]]:
        """
        Build message list for LLM with system prompt, doc context, and history.

        Args:
            session: Conversation session
            max_history: Maximum history messages to include
            doc_context: Optional auto-injected document results

        Returns:
            List of messages for LLM
        """

        system_prompt = """You are MIST, a conversational AI assistant with a personal knowledge graph.

=== CONTEXT PROVIDED ===

You receive relevant MIST documentation automatically with each query (see below).
This documentation is provided via semantic search and contains the most relevant information.

=== AVAILABLE TOOLS ===

You have three tools at your disposal:

1. **query_knowledge_graph(query: str, limit: int = 20)**
   - Search the personal knowledge graph for user-specific information
   - Use when: User asks about THEIR preferences, skills, projects, past conversations
   - Returns: Facts you've learned about the user (entities + relationships)
   - Example: "What programming languages do I know?" -> use this tool

2. **extract_knowledge(text: str, context: str = None)**
   - Store new information the user shares about themselves
   - Use when: User tells you about their skills, preferences, projects, interests
   - Effect: Adds entities to knowledge graph from conversation
   - Example: "I'm learning React" -> use this tool to remember

3. **extract_knowledge_from_document(chunk_id: str, reason: str = None)**
   - Extract entities from auto-provided documentation chunks into the knowledge graph
   - Use when: User asks you to "remember" or "learn" something from the docs
   - Use SPARINGLY: Only when user explicitly asks or info will be queried frequently
   - Effect: Converts document facts into queryable graph entities
   - Note: chunk_id can be found in the auto-provided documentation context

=== TOOL USAGE STRATEGY ===

**For user questions:**
- Technical questions (how/what/why about MIST) -> Use auto-provided docs (already in context)
- Personal questions (about user's info) -> Use query_knowledge_graph tool
- Both types -> Use docs + query_knowledge_graph

**For user statements:**
- Shares personal info -> Use extract_knowledge tool
- Asks to "remember" doc info -> Use extract_knowledge_from_document tool (rarely needed)

**Autonomous Decision Making:**
- You decide when to use tools - no one tells you when
- You can use multiple tools, one tool, or no tools per response
- Think before responding: "What context do I need?"
- Documentation is automatically provided - use it! Cite sources when helpful.
- Only call query_knowledge_graph when you need personal user context

=== GUIDELINES ===

- Be conversational and natural
- Cite documentation sources when answering technical questions
- Use tools to enhance responses, not replace conversation
- Combine auto-provided docs with your conversational abilities
- Query the knowledge graph when personal context matters

Remember: Documentation is already provided below. Think about whether you need personal user context from the knowledge graph."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add documentation context if available
        if doc_context:
            doc_context_str = self._format_document_context(doc_context)
            if doc_context_str:
                # Log what's being injected for debugging
                logger.info(
                    f"[AUTO-RAG] Injecting {len(doc_context)} document chunks into context:"
                )
                for i, chunk in enumerate(doc_context, 1):
                    title = chunk.get("source_title", "Unknown")
                    text_preview = (
                        chunk.get("text", "")[:100] + "..."
                        if len(chunk.get("text", "")) > 100
                        else chunk.get("text", "")
                    )
                    similarity = chunk.get("similarity", 0.0)
                    logger.info(
                        f"[AUTO-RAG]   [{i}] {title} (sim={similarity:.3f}): {text_preview}"
                    )

                messages.append({"role": "system", "content": doc_context_str})

        # Add conversation history
        history = session.get_history(max_history)
        messages.extend(history)

        return messages

    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

    def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        """Get information about a session"""
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
