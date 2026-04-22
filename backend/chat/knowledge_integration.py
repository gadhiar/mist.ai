"""Knowledge Graph Integration for Voice Processor.

Provides a bridge between existing voice system and knowledge-augmented conversation.
"""

import asyncio
import contextlib
import logging
from collections.abc import Generator
from pathlib import Path

import yaml

from backend.chat.conversation_handler import ConversationHandler
from backend.factories import build_conversation_handler
from backend.knowledge.config import KnowledgeConfig
from backend.llm import LLMRequest, StreamingLLMProvider

logger = logging.getLogger(__name__)


class KnowledgeIntegration:
    """Integrates knowledge graph with existing voice system.

    Wraps ConversationHandler to work with existing streaming architecture.
    """

    def __init__(
        self,
        config: KnowledgeConfig,
        llm_provider: StreamingLLMProvider | None = None,
        vault_writer=None,
        vault_sidecar=None,
    ):
        """Initialize knowledge integration.

        Args:
            config: Complete knowledge system configuration
            llm_provider: Optional pre-built provider. When None, built from config.
            vault_writer: Optional pre-started VaultWriter (Cluster 8 Phase 5).
                Passed directly into build_conversation_handler so the
                voice-path handler shares the server-owned writer instead of
                auto-building a second (unstarted) instance.
            vault_sidecar: Optional initialized VaultSidecarIndex (Cluster 8
                Phase 9). Forwarded into build_conversation_handler so the
                retriever's `historical` and three-way `hybrid` RRF paths
                route to the sidecar. None preserves pre-Phase-9 behavior.
        """
        self.enabled = False
        self.conversation_handler: ConversationHandler | None = None
        self.current_session_id = "default"
        self._llm_provider = llm_provider
        self._config = config

        try:
            if llm_provider is None:
                from backend.factories import build_llm_provider

                llm_provider = build_llm_provider(config)
                self._llm_provider = llm_provider

            self.conversation_handler = build_conversation_handler(
                config=config,
                llm_provider=llm_provider,
                vault_writer=vault_writer,
                vault_sidecar=vault_sidecar,
            )

            self.enabled = True
            logger.info("Knowledge integration enabled")

        except Exception as e:
            logger.warning("Knowledge integration disabled: %s", e)
            logger.warning("Falling back to standard LLM (no knowledge graph)")

    def set_voice_profile(self, profile_name: str) -> None:
        """Set the active voice profile for personality templating."""
        self._voice_profile = profile_name
        logger.info("Voice profile set to: %s", profile_name)

    def _load_personality(self, profile_name: str) -> dict:
        """Load personality config for a voice profile."""
        config_path = (
            Path(__file__).parent.parent.parent
            / "voice_profiles"
            / profile_name
            / "personality.yaml"
        )
        if not config_path.exists():
            logger.debug("No personality config at %s", config_path)
            return {}
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    def _build_voice_system_prompt(self, personality: dict) -> str:
        """Build voice system prompt from personality config."""
        if not personality:
            return (
                "You are M.I.S.T, a helpful voice assistant and friend to your "
                "creator, Raj Gadhia.\n\n"
                "Response Guidelines:\n"
                "- For simple questions or greetings: 1-3 sentences\n"
                "- For detailed requests: provide complete, thorough responses\n"
                "- Use a warm, friendly tone suitable for spoken conversation\n"
                "- Prioritize correctness, accuracy, and thoroughness\n"
                "- Don't artificially truncate content the user explicitly "
                "requested"
            )

        name = personality.get("name", "M.I.S.T")
        style = personality.get("speaking_style", "").strip()
        openers = personality.get("characteristic_openers", [])
        mannerisms = personality.get("mannerisms", [])

        parts = [
            f"You are {name}, a personal AI assistant and friend to " "your creator, Raj Gadhia."
        ]

        if style:
            parts.append(f"\nSpeaking style: {style}")

        if openers:
            opener_list = "\n".join(f"- {o}" for o in openers)
            parts.append(
                "\nWhen responding, begin with a brief acknowledgment or "
                "opening phrase before your main answer. Examples of "
                f"characteristic openers you use:\n{opener_list}"
            )

        if mannerisms:
            mannerism_list = "\n".join(f"- {m}" for m in mannerisms)
            parts.append(f"\nAdditional guidelines:\n{mannerism_list}")

        parts.append(
            "\nResponse Guidelines:\n"
            "- For simple questions or greetings: 1-3 sentences\n"
            "- For detailed requests: provide complete, thorough responses\n"
            "- Prioritize correctness, accuracy, and thoroughness\n"
            "- Don't artificially truncate content the user explicitly "
            "requested"
        )

        return "\n".join(parts)

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

            # Yield complete response (non-streaming fallback).
            # For token-level streaming, use generate_tokens_streaming().
            yield response

        except Exception as e:
            logger.error(f"Error in knowledge integration: {e}", exc_info=True)
            yield f"I encountered an error: {str(e)}"

    def generate_tokens_streaming(
        self,
        user_text: str,
        session_id: str | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> Generator[str, None, None]:
        """Generate LLM response with true token-level streaming.

        Runs RAG retrieval, builds a voice-optimized prompt, then streams
        tokens via the injected LLM provider. Bypasses ConversationHandler's
        tool-calling chain for lower latency.

        Known limitations vs handle_message():
        - No tool calling (voice mode does not need mid-stream tools)
        - EventStore recording added after stream completes
        - Background extraction skipped (TODO: add in future)

        Args:
            user_text: User's message.
            session_id: Optional session ID.
            event_loop: Event loop for async RAG retrieval.

        Yields:
            Individual tokens from the LLM stream.
        """
        if not self.enabled or not self.conversation_handler:
            yield "I'm sorry, the knowledge system is not available right now."
            return

        try:
            sid = session_id or self.current_session_id
            handler = self.conversation_handler

            # Step 1: RAG retrieval
            retrieval_result = None
            if (
                handler.retriever
                and handler.config.auto_inject_docs
                and len(user_text.split()) >= 3
                and event_loop is not None
            ):
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        handler.retriever.retrieve(
                            query=user_text,
                            user_id="User",
                            limit=handler.config.auto_inject_limit,
                            similarity_threshold=handler.config.auto_inject_threshold,
                        ),
                        event_loop,
                    )
                    retrieval_result = future.result(timeout=30)
                    if retrieval_result and retrieval_result.total_facts > 0:
                        logger.info(
                            "RAG retrieved %d facts for voice streaming",
                            retrieval_result.total_facts,
                        )
                except Exception as e:
                    logger.warning("RAG retrieval failed in voice streaming: %s", e)

            # Step 2: Build voice-optimized messages
            session = handler.get_or_create_session(sid, "User")
            session.add_message("user", user_text)

            voice_profile = getattr(self, "_voice_profile", "friday")
            personality = self._load_personality(voice_profile)
            system_prompt = self._build_voice_system_prompt(personality)

            messages: list[dict[str, str]] = [
                {"role": "system", "content": system_prompt},
            ]

            # Add RAG context if available
            if retrieval_result and retrieval_result.total_facts > 0:
                messages.append(
                    {
                        "role": "system",
                        "content": retrieval_result.formatted_context,
                    }
                )

            # Add conversation history (last 10 turns = 20 messages)
            for msg in session.messages[-20:]:
                messages.append({"role": msg.role, "content": msg.content})

            # Step 3: Stream tokens from LLM provider
            request = LLMRequest(
                messages=messages,
                temperature=self._config.llm.temperature,
                max_tokens=400,
                top_p=0.9,
            )

            full_response = ""
            for chunk in self._llm_provider.generate_sync(request, stream=True):
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            # Step 4: Record to session history
            session.add_message("assistant", full_response)

            # Step 5: Record to EventStore (Layer 1 audit trail)
            if handler.event_store:
                try:
                    handler._record_turn_event(
                        session_id=sid,
                        user_message=user_text,
                        assistant_message=full_response,
                        context_window=messages,
                        retrieval_result=retrieval_result,
                    )
                except Exception as e:
                    logger.warning("EventStore recording failed: %s", e)

        except Exception as e:
            logger.error("Error in streaming knowledge integration: %s", e, exc_info=True)
            yield f"I encountered an error: {e!s}"

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
