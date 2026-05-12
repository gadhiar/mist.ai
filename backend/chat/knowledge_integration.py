"""Knowledge Graph Integration for Voice Processor.

Provides a bridge between existing voice system and knowledge-augmented conversation.
"""

import asyncio
import contextlib
import logging
import queue
from collections.abc import Generator

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.stream_events import Complete, Token, WSEvent
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
        invalidation_bus=None,
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
            invalidation_bus: Optional InvalidationBus shared with the filewatcher
                (Phase 5.5). Forwarded into build_conversation_handler so
                ConversationHandler can subscribe `_on_vault_rebuild` and
                evict stale `_mist_context_cache` entries on vault edits.
                Must be the same instance returned by build_phase3_components.
                None preserves pre-Phase-5.5 behavior (no cache invalidation).
        """
        self.enabled = False
        self.conversation_handler: ConversationHandler | None = None
        self.current_session_id = "default"
        self._llm_provider = llm_provider
        self._config = config
        self._invalidation_bus = invalidation_bus
        # Bridge side-channel: last Complete event captured per turn so callers
        # can read duration_ms / tool_calls_used after generate_response_streaming
        # returns (the generator yields only strings; ADR-017 stream_complete
        # needs the metadata). Reset at the top of each streaming call.
        self.last_complete: Complete | None = None
        # Bridge side-channel: last error captured per turn as (kind, message)
        # so callers can emit a discriminated ADR-017 error event instead of
        # yielding error strings as fake llm_token chunks. Closes Phase 1
        # fix #3 (synthetic llm_token leak from bridge timeout) cleanly.
        self.last_error: tuple[str, str] | None = None

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
                invalidation_bus=invalidation_bus,
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
    ) -> Generator[str | dict, None, None]:
        """Bridge handle_message_streaming (async generator) to sync iteration.

        The voice_processor worker thread iterates this generator and routes
        each yielded item: strings are tokens (existing stream_token path),
        dicts are pre-formed FE-bound event payloads (ADR-017 Wave 2:
        tool_call_started/completed, cards_summon/dismiss, graph_subgraph)
        forwarded as-is onto the canonical message_queue.

        Internally drains `ConversationHandler.handle_message_streaming` on the
        provided event loop via a thread-safe queue. All canonical pipeline
        behavior (retrieval, mist context injection, tool dispatch, slop
        filter, vault append, EventStore record, fire-and-forget extraction)
        is inherited unchanged from handle_message_streaming.

        Args:
            user_text: User's message.
            session_id: Optional session ID (uses default if not provided).
            event_loop: Asyncio event loop for the streaming generator. Required
                in the live-server / voice path. When None, falls back to
                `asyncio.run` (test/CLI contexts).

        Yields:
            Either a string (token text from a Token event) OR a dict (the
            payload of a WSEvent, ready to json.dumps onto the message_queue).
            Stream ends when the queue sentinel arrives (Complete event seen
            on the producer side).

        Side-effect:
            Sets ``self.last_complete`` to the last ``Complete`` event seen on
            the bridge (or ``None`` if the stream errored or returned no
            Complete). Callers that need ``duration_ms`` / ``tool_calls_used``
            for ADR-017 ``stream_complete`` payloads should read it after the
            generator finishes.
        """
        # Reset bridge side-channels for this turn.
        self.last_complete = None
        self.last_error = None

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

                async def _drain_inline() -> list[str | dict]:
                    out: list[str | dict] = []
                    async for event in self.conversation_handler.handle_message_streaming(
                        user_message=user_text, session_id=sid
                    ):
                        if isinstance(event, Token):
                            out.append(event.text)
                        elif isinstance(event, WSEvent):
                            out.append(event.payload)
                    return out

                yield from asyncio.run(_drain_inline())
                return
            except Exception as e:
                logger.error("Error in knowledge integration: %s", e, exc_info=True)
                # Capture on side-channel; caller emits discriminated error.
                self.last_error = ("server", str(e))
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
                # Phase 1 fix #3: capture timeout on side-channel instead of
                # yielding error text as a fake token. Caller reads last_error
                # after the generator returns and emits a discriminated
                # ADR-017 error event.
                self.last_error = ("timeout", "response timeout")
                return
            if item is DONE:
                return
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__error__":
                # Capture on side-channel; caller emits discriminated error.
                self.last_error = ("server", str(item[1]))
                return
            if isinstance(item, Token):
                yield item.text
            elif isinstance(item, WSEvent):
                # ADR-017 Wave 2: forward FE-bound event payload as a dict.
                # voice_processor distinguishes dict vs str and routes dicts
                # directly onto the message_queue (json.dumps + put).
                yield item.payload
            elif isinstance(item, Complete):
                # Producer side finished; DONE sentinel will arrive next. We
                # don't yield the Complete event itself (caller wants tokens),
                # but capture it on the integration object so callers can read
                # self.last_complete after the generator returns to access
                # duration_ms / tool_calls_used per ADR-017 stream_complete.
                self.last_complete = item

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
