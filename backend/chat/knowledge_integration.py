"""Knowledge Graph Integration for Voice Processor.

Provides a bridge between existing voice system and knowledge-augmented conversation.
"""

import asyncio
import contextlib
import logging
from collections.abc import Generator

from backend.chat.conversation_handler import ConversationHandler
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


class KnowledgeIntegration:
    """Integrates knowledge graph with existing voice system.

    Wraps ConversationHandler to work with existing streaming architecture.
    """

    def __init__(self, config: KnowledgeConfig):
        """Initialize knowledge integration.

        Args:
            config: Complete knowledge system configuration
        """
        self.enabled = False
        self.conversation_handler: ConversationHandler | None = None
        self.current_session_id = "default"

        try:
            # Initialize graph store with knowledge config
            graph_store = GraphStore(config)

            # Initialize conversation handler with full knowledge config
            self.conversation_handler = ConversationHandler(
                config=config, graph_store=graph_store, model_name=config.llm.model
            )

            self.enabled = True
            logger.info(" Knowledge integration enabled")

        except Exception as e:
            logger.warning(f"Knowledge integration disabled: {e}")
            logger.warning("Falling back to standard LLM (no knowledge graph)")

    def generate_response_streaming(
        self,
        user_text: str,
        session_id: str | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> Generator[str, None, None]:
        """Generate LLM response with knowledge integration (streaming).

        Note: Current implementation returns complete response, not streaming.
        Future: Can implement true streaming with tool results.

        Args:
            user_text: User's message
            session_id: Optional session ID (uses default if not provided)

        Yields:
            Response tokens
        """
        if not self.enabled or not self.conversation_handler:
            logger.warning("Knowledge integration not available, cannot generate response")
            yield "I'm sorry, the knowledge system is not available right now."
            return

        try:
            # Use session ID if provided
            sid = session_id or self.current_session_id

            # Get response (async) - use provided event loop or existing one
            if event_loop is not None:
                # Use provided event loop (from voice_processor)
                future = asyncio.run_coroutine_threadsafe(
                    self.conversation_handler.handle_message(
                        user_message=user_text, session_id=sid
                    ),
                    event_loop,
                )
                response = future.result(timeout=120)
            else:
                # No event loop provided, create one (for testing)
                import nest_asyncio

                with contextlib.suppress(BaseException):
                    nest_asyncio.apply()

                response = asyncio.run(
                    self.conversation_handler.handle_message(user_message=user_text, session_id=sid)
                )

            # Yield complete response
            # TODO: Future enhancement - stream tokens as they're generated
            yield response

        except Exception as e:
            logger.error(f"Error in knowledge integration: {e}", exc_info=True)
            yield f"I encountered an error: {str(e)}"

    def set_session_id(self, session_id: str):
        """Set the current session ID."""
        self.current_session_id = session_id
        logger.info(f"Session ID set to: {session_id}")

    def clear_session(self, session_id: str | None = None):
        """Clear a conversation session."""
        if self.conversation_handler:
            sid = session_id or self.current_session_id
            self.conversation_handler.clear_session(sid)

    def is_enabled(self) -> bool:
        """Check if knowledge integration is enabled."""
        return self.enabled
