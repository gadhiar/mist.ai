"""The responder that answers turns when knowledge integration is off or failed to start.

Moved from `ModelManager.generate_llm_response`'s fallback branch with its prompt, token limit
and sampling unchanged, behind the `TurnResponder` API the turn controller drives. Such a turn
is answered but never recorded: the event store belongs to the knowledge subsystem that is off,
so each turn is counted and warned about once (`backend/extraction_backlog/telemetry.py`), and
`ExtractionStatus.unrecorded_turns` shows the loss.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import AsyncIterator

from backend.chat.stream_events import Complete, StreamEvent, Token
from backend.chat.turn_scope import CommitResult, TurnDelivery, TurnRequest, TurnScope
from backend.extraction_backlog.telemetry import record_unrecorded_turn
from backend.llm.models import LLMRequest
from backend.llm.provider import StreamingLLMProvider
from backend.request_context import current_request_id

FALLBACK_SYSTEM_PROMPT = """You are M.I.S.T, a helpful voice assistant and friend to your creator, Raj Gadhia.

Response Guidelines:
- For simple questions or greetings: 1-3 sentences (brief and direct)
- For requests asking for multiple items, stories, examples, or detailed explanations: provide complete, thorough responses
- When user says "give me N things/stories/examples", provide all N items fully
- When user asks "explain" or "tell me about", give comprehensive answers
- Use a warm, friendly tone suitable for spoken conversation
- Always prioritize correctness, accuracy, and thoroughness
- Don't artificially truncate content the user explicitly requested

Match your response depth to what the user is asking for - be concise when appropriate, thorough when needed."""  # noqa: E501
FALLBACK_MAX_TOKENS = 400


class DirectResponder:
    """Answers each turn from the LLM alone, with no memory, retrieval or tools.

    Args:
        llm_provider: The interactive LLM provider.
    """

    def __init__(self, llm_provider: StreamingLLMProvider) -> None:
        self._llm_provider = llm_provider
        self._turn_indices = itertools.count()

    def open_turn(self, request: TurnRequest) -> TurnScope:
        """Create the turn's scope; this responder dispatches no tools, so nothing waits."""
        scope = TurnScope(
            session_id=request.session_id,
            turn_index=next(self._turn_indices),
            request=request,
        )
        if not request.held:
            scope.dispatch_gate.set()
        return scope

    async def stream_turn(self, scope: TurnScope) -> AsyncIterator[StreamEvent]:
        """Stream the reply's tokens, then `Complete`; counts the turn as unrecorded once."""
        record_unrecorded_turn(current_request_id.get())
        request = LLMRequest(
            messages=[
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": scope.request.user_text},
            ],
            max_tokens=FALLBACK_MAX_TOKENS,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            top_p=0.9,
        )
        async for chunk in self._llm_provider.generate(request, stream=True):
            if chunk.content:
                scope.stream.emitted += chunk.content
                yield Token(chunk.content)
        yield Complete(final_response=scope.stream.emitted)

    def commit_turn(self, scope: TurnScope, delivery: TurnDelivery) -> CommitResult:
        """Record nothing: with no event store this is not a failure."""
        return CommitResult(recorded=False, event_id=None, error=None)

    def prefetch(self, session_id: str, partial_text: str | None) -> None:
        """Nothing to prefetch without retrieval."""
