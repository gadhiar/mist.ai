"""Knowledge Graph Integration for Voice Processor.

Provides a bridge between existing voice system and knowledge-augmented conversation.
"""

import asyncio
import contextlib
import logging
import queue
from collections.abc import Generator

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.stream_events import Complete, Token
from backend.factories import build_conversation_handler
from backend.knowledge.config import KnowledgeConfig
from backend.llm import StreamingLLMProvider

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
        """Record the active voice profile name (TTS texture selection only).

        Per project convention (`user_mist_personality` memory): VoiceProfiles
        are TTS textures, not personas. Personality lives in the canonical
        identity (graph + vault `mist-identity`) and is injected via
        `handle_message_streaming` -> mist_context regardless of voice profile.
        Stored here for future plumbing if needed; not currently consumed by
        the canonical conversation pipeline.
        """
        self._voice_profile = profile_name
        logger.info("Voice profile set to: %s", profile_name)

    def generate_response_streaming(
        self,
        user_text: str,
        session_id: str | None = None,
        event_loop: asyncio.AbstractEventLoop | None = None,
    ) -> Generator[str, None, None]:
        """Bridge handle_message_streaming (async generator) to sync token iteration.

        The voice_processor worker thread iterates this generator and forwards
        each token to the WebSocket client + SentenceBoundaryDetector + TTS.
        Internally drains `ConversationHandler.handle_message_streaming` on the
        provided event loop via a thread-safe queue.

        All canonical pipeline behavior (retrieval, mist context injection,
        tool dispatch, slop filter, vault append, EventStore record,
        fire-and-forget extraction) is inherited unchanged from
        handle_message_streaming and runs on the event loop.

        Args:
            user_text: User's message.
            session_id: Optional session ID (uses default if not provided).
            event_loop: Asyncio event loop for the streaming generator. Required
                in the live-server / voice path. When None, falls back to
                `asyncio.run` (test/CLI contexts).

        Yields:
            Token text strings, one per Token event from handle_message_streaming.
            Stream ends when the queue sentinel arrives (Complete event seen
            on the producer side).
        """
        if not self.enabled or not self.conversation_handler:
            logger.warning("Knowledge integration not available, cannot generate response")
            yield "I'm sorry, the knowledge system is not available right now."
            return

        sid = session_id or self.current_session_id

        if event_loop is None:
            # Test / CLI fallback: drain the async generator inline.
            try:
                with contextlib.suppress(BaseException):
                    import nest_asyncio

                    nest_asyncio.apply()

                async def _drain_inline() -> list[str]:
                    out: list[str] = []
                    async for event in self.conversation_handler.handle_message_streaming(
                        user_message=user_text, session_id=sid
                    ):
                        if isinstance(event, Token):
                            out.append(event.text)
                    return out

                yield from asyncio.run(_drain_inline())
                return
            except Exception as e:
                logger.error("Error in knowledge integration: %s", e, exc_info=True)
                yield f"I encountered an error: {e!s}"
                return

        # Live-server / voice path: bridge async stream to sync via queue.
        q: queue.Queue = queue.Queue()
        DONE = object()

        async def _drain() -> None:
            try:
                async for event in self.conversation_handler.handle_message_streaming(
                    user_message=user_text, session_id=sid
                ):
                    q.put(event)
            except Exception as exc:  # noqa: BLE001
                logger.error("handle_message_streaming failed: %s", exc, exc_info=True)
                q.put(("__error__", str(exc)))
            finally:
                q.put(DONE)

        asyncio.run_coroutine_threadsafe(_drain(), event_loop)

        while True:
            try:
                item = q.get(timeout=180)
            except queue.Empty:
                logger.error("handle_message_streaming bridge timed out after 180s")
                yield "I encountered an error: response timeout"
                return
            if item is DONE:
                return
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                yield f"I encountered an error: {item[1]}"
                return
            if isinstance(item, Token):
                yield item.text
            elif isinstance(item, Complete):
                # Producer side finished; DONE sentinel will arrive next.
                # We don't yield the Complete event itself (caller wants strings).
                pass

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
