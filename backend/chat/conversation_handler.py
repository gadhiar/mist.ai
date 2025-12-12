"""
Conversation Handler with MCP-like Tool Access

Enables LLM to autonomously:
- Query knowledge graph for context
- Extract and store new knowledge
- Think and search database freely
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.extraction.entity_extractor import EntityExtractor
from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.models import (
    ConversationSession,
    Message,
    RetrievalFilters
)
from backend.knowledge.config import ExtractionConfig

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
        self,
        config: ExtractionConfig,
        graph_store: GraphStore,
        model_name: str = "qwen2.5:7b"
    ):
        """
        Initialize conversation handler.

        Args:
            config: Extraction configuration
            graph_store: Neo4j graph store
            model_name: Ollama model to use
        """
        self.config = config
        self.graph_store = graph_store

        # Initialize components
        self.retriever = KnowledgeRetriever(config, graph_store)
        self.extractor = EntityExtractor(config)

        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
        )

        # Bind tools to LLM
        self._setup_tools()

        # Active sessions
        self.sessions: Dict[str, ConversationSession] = {}

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
            entity_types: Optional[List[str]] = None,
            relationship_types: Optional[List[str]] = None,
            limit: int = 20
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
                        entity_types=entity_types,
                        relationship_types=relationship_types
                    )

                result = await retriever.retrieve(
                    query=query,
                    user_id="User",
                    limit=limit,
                    filters=filters
                )

                if result.total_facts == 0:
                    return f"No information found for query: '{query}'. You may want to ask the user about this topic."

                return result.formatted_context

            except Exception as e:
                logger.error(f"Error querying knowledge graph: {e}")
                return f"Error searching knowledge graph: {str(e)}"

        @tool
        async def extract_knowledge(text: str, context: Optional[str] = None) -> str:
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
                    conversation_id=conversation_id,
                    user_id="User"
                )

                # Generate unique utterance ID
                utterance_id = str(uuid.uuid4())

                # Store the utterance
                graph_store.store_utterance(
                    utterance_id=utterance_id,
                    conversation_id=conversation_id,
                    text=text,
                    metadata={"source": "conversation", "timestamp": datetime.now().isoformat()}
                )

                # Extract entities from text
                graph_docs = await extractor.extract_from_utterance(
                    utterance=text,
                    conversation_history=[],
                    metadata={"utterance_id": utterance_id}
                )

                # Store extracted entities
                if graph_docs and graph_docs[0].nodes:
                    graph_store.store_extracted_entities(
                        graph_document=graph_docs[0],
                        utterance_id=utterance_id,
                        ontology_version=None
                    )

                    entity_names = [n.id for n in graph_docs[0].nodes]
                    rel_count = len(graph_docs[0].relationships)

                    return f"Learned and stored: {len(entity_names)} entities ({', '.join(entity_names[:5])}{'...' if len(entity_names) > 5 else ''}) with {rel_count} relationships."
                else:
                    return "No new knowledge extracted from the text."

            except Exception as e:
                logger.error(f"Error extracting knowledge: {e}")
                return f"Error storing knowledge: {str(e)}"

        # Bind tools to LLM
        self.tools = [query_knowledge_graph, extract_knowledge]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        logger.info("Tools bound to LLM: query_knowledge_graph, extract_knowledge")

    def get_or_create_session(
        self,
        session_id: str,
        user_id: str = "User"
    ) -> ConversationSession:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                user_id=user_id
            )
            logger.info(f"Created new session: {session_id}")

        return self.sessions[session_id]

    async def handle_message(
        self,
        user_message: str,
        session_id: str,
        user_id: str = "User",
        max_history: int = 10
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

        # Build conversation with system prompt
        messages = self._build_messages(session, max_history)

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
                    result_preview = tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                    logger.info(f"[TOOLS]   Result: {result_preview}")

                    tool_calls.append({"name": tool_name, "args": tool_args})
                    tool_results.append({
                        "name": tool_name,
                        "result": tool_result,
                        "tool_call_id": tool_call_id
                    })

                # If tools were called, get final response with tool results
                messages.append({"role": "assistant", "content": response.content or ""})

                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "content": result["result"],
                        "name": result["name"],
                        "tool_call_id": result["tool_call_id"]
                    })

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
                tool_results=tool_results if tool_results else None
            )

            return assistant_message

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            error_msg = f"I encountered an error: {str(e)}"
            session.add_message("assistant", error_msg)
            return error_msg

    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool by name"""
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.ainvoke(tool_args)

        return f"Tool not found: {tool_name}"

    def _build_messages(
        self,
        session: ConversationSession,
        max_history: int
    ) -> List[Dict[str, str]]:
        """Build message list for LLM with system prompt and history"""

        system_prompt = """You are MIST, a conversational AI assistant with a personal knowledge graph about the user.

You have access to two powerful tools:

1. **query_knowledge_graph**: Search for information you've learned about the user
   - Use when user asks about their past information, preferences, skills, projects, etc.
   - Use to personalize responses based on accumulated knowledge
   - Think: "Do I need context from what I've learned before?"

2. **extract_knowledge**: Store new information the user shares
   - Use when user tells you about themselves (technologies, projects, skills, preferences, etc.)
   - Use to remember important information for future conversations
   - Think: "Is this something I should remember?"

**How to use tools autonomously:**
- You decide when to use tools - no one tells you when
- You can use both tools, one tool, or no tools per response
- Think before responding: "Do I need to query? Do I need to learn?"
- Use tools to provide personalized, context-aware responses

**Guidelines:**
- Be conversational and natural
- Use tools to enhance responses, not replace conversation
- If you don't have information, query the knowledge graph
- If user shares information, extract it
- Combine retrieved knowledge with your conversational abilities

Remember: You're autonomous. Think about what you need to do, then do it."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        history = session.get_history(max_history)
        messages.extend(history)

        return messages

    def clear_session(self, session_id: str):
        """Clear a conversation session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "started_at": session.started_at.isoformat(),
            "message_count": len(session.messages),
            "last_message": session.messages[-1].content if session.messages else None
        }
