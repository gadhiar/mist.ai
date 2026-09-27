"""Tests for ConversationHandler.handle_message_streaming (T3/v2 canonical stream).

T3 inverted the v1 relationship: `handle_message_streaming` is now the
canonical generator -- it streams the provider's real SSE
(`StreamingLLMProvider.generate(..., stream=True)`) through
`_stream_llm_pass`, gated to sentence boundaries so the slop filter always
inspects a complete sentence -- and `handle_message` is a thin consumer
that joins the terminal `Complete` event. v1 wrapped `handle_message` and
fake-streamed the finished reply one character at a time; these tests now
validate the real contract: token granularity (per gated sentence, not
per character), real-time delivery (a token reaches the consumer before a
later, slower chunk is even produced), tool-call turns, the sentence-gated
slop filter, and equivalence with `handle_message`'s return value.
"""

import asyncio

import pytest

from backend.chat.conversation_handler import ConversationHandler
from backend.chat.stream_events import Complete, StreamEvent, Token, WSEvent
from backend.errors import LLMConnectionError
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.graph_store import GraphStore
from backend.llm.models import ToolCall as LLMToolCall
from tests.mocks.config import build_test_config
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.mocks.ollama import FakeLLM
from tests.mocks.streaming_llm import FakeStreamingLLMProvider, ScriptedPass
from tests.unit.conftest import make_test_conventions_loader


def _make_retriever(config, gs):
    return KnowledgeRetriever(config=config, graph_store=gs)


class FakeExtractionPipeline:
    async def extract_from_utterance(self, **kwargs):
        from backend.knowledge.extraction.validator import ValidationResult

        return ValidationResult(valid=True, entities=[], relationships=[])


def _make_handler(default_response: str = "Hello there.") -> ConversationHandler:
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config()
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=FakeExtractionPipeline(),
        retriever=_make_retriever(config, gs),
        llm_provider=FakeLLM(default_response=default_response),
        conventions_loader=make_test_conventions_loader(),
    )


def _make_streaming_handler(provider: FakeStreamingLLMProvider) -> ConversationHandler:
    conn = FakeNeo4jConnection()
    gs = GraphStore(conn, FakeEmbeddingGenerator())
    config = build_test_config()
    return ConversationHandler(
        config=config,
        graph_store=gs,
        extraction_pipeline=FakeExtractionPipeline(),
        retriever=_make_retriever(config, gs),
        llm_provider=provider,
        conventions_loader=make_test_conventions_loader(),
    )


class TestStreamingContract:
    @pytest.mark.asyncio
    async def test_yields_at_least_one_event(self):
        handler = _make_handler("Hi.")
        events = []
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s1"
        ):
            events.append(event)
        assert len(events) >= 1, "stream must yield at least one event"
        assert all(isinstance(e, StreamEvent) for e in events)

    @pytest.mark.asyncio
    async def test_terminates_with_complete(self):
        handler = _make_handler("Hello.")
        events = []
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s2"
        ):
            events.append(event)
        assert isinstance(
            events[-1], Complete
        ), f"last event must be Complete, got {type(events[-1]).__name__}"

    @pytest.mark.asyncio
    async def test_only_one_complete_event(self):
        handler = _make_handler("Just one response.")
        complete_count = 0
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s3"
        ):
            if isinstance(event, Complete):
                complete_count += 1
        assert complete_count == 1, "exactly one Complete event must be yielded"

    @pytest.mark.asyncio
    async def test_token_texts_join_to_final_response(self):
        response = "This is the canonical response."
        handler = _make_handler(response)
        token_chars = []
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s4"
        ):
            if isinstance(event, Token):
                token_chars.append(event.text)
            elif isinstance(event, Complete):
                complete = event
        joined = "".join(token_chars)
        assert complete is not None
        assert joined == complete.final_response, (
            f"token concat ({joined!r}) must equal Complete.final_response "
            f"({complete.final_response!r})"
        )

    @pytest.mark.asyncio
    async def test_streaming_matches_non_streaming_response(self):
        """handle_message_streaming.Complete.final_response equals handle_message's return.

        T3/v2: handle_message is now a thin consumer of
        handle_message_streaming (inverted from v1), so this equivalence
        holds by construction rather than by parallel implementation --
        this test guards against a future regression splitting the two.
        """
        response = "Identical content path."
        handler_a = _make_handler(response)
        non_streaming = await handler_a.handle_message(
            user_message="Tell me something useful", session_id="sa"
        )
        handler_b = _make_handler(response)
        streamed_complete: Complete | None = None
        async for event in handler_b.handle_message_streaming(
            user_message="Tell me something useful", session_id="sb"
        ):
            if isinstance(event, Complete):
                streamed_complete = event
        assert streamed_complete is not None
        assert streamed_complete.final_response == non_streaming, (
            "handle_message_streaming Complete.final_response must equal "
            "handle_message return for the same input"
        )

    @pytest.mark.asyncio
    async def test_token_granularity_is_per_sentence_not_per_character(self):
        """T3/v2 gates content on sentence boundaries (SentenceBoundaryDetector),
        not on characters: a two-sentence reply streamed as a single
        provider chunk yields exactly two Token events, not one per
        character.
        """
        provider = FakeStreamingLLMProvider(
            [ScriptedPass(chunks=["This is sentence one. This is sentence two."])]
        )
        handler = _make_streaming_handler(provider)
        token_count = 0
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s5"
        ):
            if isinstance(event, Token):
                token_count += 1
            elif isinstance(event, Complete):
                complete = event
        assert complete is not None
        assert token_count == 2, f"expected one Token per gated sentence, got {token_count}"
        assert token_count < len(complete.final_response)

    @pytest.mark.asyncio
    async def test_complete_carries_duration(self):
        handler = _make_handler("ok.")
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s6"
        ):
            if isinstance(event, Complete):
                complete = event
        assert complete is not None
        assert complete.duration_ms >= 0, "duration_ms must be non-negative"


class TestFirstTokenBeforeCompletion:
    """Acceptance criterion 1 (T3): real streaming, not a fake full-buffer wait."""

    @pytest.mark.asyncio
    async def test_first_sentence_arrives_before_second_sentence_is_produced(self):
        """Pass 1 yields sentence 1, then waits on an asyncio.Event, then
        yields sentence 2. The consumer must be able to receive sentence
        1's Token while the fake provider is still parked on the event --
        i.e. before the test itself sets it. Manual `anext` stepping
        proves this deterministically: if the implementation buffered the
        whole pass before yielding anything (the v1 behavior), the first
        `anext` call below would never return, because nothing sets
        `gate_event` until after it does.
        """
        gate_event = asyncio.Event()
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    chunks=[
                        "This is the first sentence. ",
                        gate_event,
                        "This is the second sentence.",
                    ]
                )
            ]
        )
        handler = _make_streaming_handler(provider)

        stream = handler.handle_message_streaming(
            user_message="Tell me something useful", session_id="s-ttft"
        )

        first_event = await anext(stream)
        while not isinstance(first_event, Token):
            first_event = await anext(stream)

        assert first_event.text.strip() == "This is the first sentence."
        # Pass 1 is still parked inside `await gate_event.wait()` -- proof
        # that reaching this line did not require the second sentence to
        # have streamed yet.
        assert not gate_event.is_set()

        gate_event.set()

        remaining_tokens: list[str] = []
        complete: Complete | None = None
        async for event in stream:
            if isinstance(event, Token):
                remaining_tokens.append(event.text)
            elif isinstance(event, Complete):
                complete = event

        assert complete is not None
        assert "second sentence" in "".join(remaining_tokens)
        assert "second sentence" in complete.final_response


class TestPass1ContentBeforeToolCall:
    """T3 design decision: pass-1 content streamed before a tool-call
    decision is surfaced to the caller live (pass_num=1), not held back
    pending the final chunk. See `_stream_llm_pass`'s docstring for the
    rationale (the provider never mixes tool-call fragments into
    `content`, and holding content back would mean buffering the whole
    pass just to learn whether a tool call follows).
    """

    @pytest.mark.asyncio
    async def test_pass1_content_streams_even_though_a_tool_call_follows(self):
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    chunks=["Let me check that for you."],
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            name="query_knowledge_graph",
                            arguments={"query": "x"},
                        )
                    ],
                ),
                ScriptedPass(chunks=["Here is the answer."]),
            ]
        )
        handler = _make_streaming_handler(provider)
        handler._dispatch_tool = _StubDispatch()

        tokens: list[Token] = []
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="What is my Rust experience?", session_id="s-pass1"
        ):
            if isinstance(event, Token):
                tokens.append(event)
            elif isinstance(event, Complete):
                complete = event

        pass1_texts = [t.text for t in tokens if t.pass_num == 1]
        pass2_texts = [t.text for t in tokens if t.pass_num == 2]
        assert any("Let me check that for you." in t for t in pass1_texts), (
            "pass-1 content must stream to the caller even though a tool "
            f"call follows; got pass1 tokens: {pass1_texts}"
        )
        assert any("Here is the answer." in t for t in pass2_texts)
        assert complete is not None
        assert "Let me check that for you." in complete.final_response
        assert "Here is the answer." in complete.final_response
        assert complete.tool_calls_used == 1


class TestStreamingSlopGate:
    """Acceptance criterion 3 (T3): sentence-gated critical-slop enforcement.

    Replaces the whole-response regen loop (`_post_filter_response`, still
    available directly -- see TestPostFilterRegeneration in
    test_conversation_handler.py) on the streaming path: a clean sentence
    streams verbatim, a sentence with a critical finding streams stripped
    (`SlopDetector.strip_fixable`), and `Complete.final_response` is
    exactly the concatenation of what was emitted -- not a separately
    reconciled value.
    """

    @pytest.mark.asyncio
    async def test_clean_sentence_emitted_verbatim(self):
        provider = FakeStreamingLLMProvider(
            [ScriptedPass(chunks=["This sentence has no slop in it at all."])]
        )
        handler = _make_streaming_handler(provider)
        tokens: list[str] = []
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="hello", session_id="s-slop-clean"
        ):
            if isinstance(event, Token):
                tokens.append(event.text)
            elif isinstance(event, Complete):
                complete = event
        assert complete is not None
        joined = "".join(tokens)
        assert joined.strip() == "This sentence has no slop in it at all."
        assert complete.final_response == joined

    @pytest.mark.asyncio
    async def test_critical_slop_sentence_emitted_stripped(self):
        # Contains an emoji (critical pattern) that strip_fixable removes.
        provider = FakeStreamingLLMProvider(
            [ScriptedPass(chunks=["Great work \U0001f389 on shipping this."])]
        )
        handler = _make_streaming_handler(provider)
        tokens: list[str] = []
        complete: Complete | None = None
        async for event in handler.handle_message_streaming(
            user_message="hello", session_id="s-slop-critical"
        ):
            if isinstance(event, Token):
                tokens.append(event.text)
            elif isinstance(event, Complete):
                complete = event
        assert complete is not None
        joined = "".join(tokens)
        assert "\U0001f389" not in joined
        assert "Great work" in joined
        assert "on shipping this" in joined
        # final_response is exactly the concatenation of what was emitted --
        # not a separately post-filtered copy of it.
        assert complete.final_response == joined


def _capture_recorded_turns(handler: ConversationHandler) -> list[str]:
    """Wrap `_record_turn_event` so a test can read every recorded
    assistant_message (the event-log write) without a real event store.
    """
    recorded: list[str] = []
    original = handler._record_turn_event

    def _capture(*args, **kwargs):
        recorded.append(kwargs["assistant_message"])
        return original(*args, **kwargs)

    handler._record_turn_event = _capture
    return recorded


async def _drain(handler: ConversationHandler, session_id: str, message: str = "hello"):
    tokens: list[Token] = []
    complete: Complete | None = None
    async for event in handler.handle_message_streaming(
        user_message=message, session_id=session_id
    ):
        if isinstance(event, Token):
            tokens.append(event)
        elif isinstance(event, Complete):
            complete = event
    assert complete is not None
    return tokens, complete


class TestErrorTurnIsStreamed:
    """Review finding 1: on a provider failure the user must SEE the error
    text that session history and the event log record. Before the fix the
    except path set `final_text` but yielded no Token, and the sync bridge
    forwards only Tokens to the socket, so the user saw nothing.
    """

    @pytest.mark.asyncio
    async def test_provider_error_before_first_token_streams_error_text(self):
        provider = FakeStreamingLLMProvider(
            [ScriptedPass(chunks=[], error=LLMConnectionError("llama-server unreachable"))]
        )
        handler = _make_streaming_handler(provider)
        recorded = _capture_recorded_turns(handler)

        tokens, complete = await _drain(handler, "s-err-1")

        streamed = "".join(t.text for t in tokens)
        assert streamed == "I encountered an error: llama-server unreachable"
        assert complete.final_response == streamed
        assert recorded == [streamed]
        history = handler.sessions["s-err-1"].messages
        assert history[-1].role == "assistant"
        assert history[-1].content == streamed

    @pytest.mark.asyncio
    async def test_provider_error_mid_stream_appends_error_after_streamed_text(self):
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    chunks=["The first sentence made it out. And then"],
                    error=LLMConnectionError("connection reset"),
                )
            ]
        )
        handler = _make_streaming_handler(provider)
        recorded = _capture_recorded_turns(handler)

        tokens, complete = await _drain(handler, "s-err-2")

        streamed = "".join(t.text for t in tokens)
        assert streamed == (
            "The first sentence made it out.\n\nI encountered an error: connection reset"
        )
        assert complete.final_response == streamed
        assert recorded == [streamed]
        assert handler.sessions["s-err-2"].messages[-1].content == streamed


class TestWhitespacePreserved:
    """Review finding 2: paragraph and list newlines must survive into the
    Token stream, `final_response`, session history and the event log. Before
    the fix `SentenceBoundaryDetector.feed` stripped the whitespace at each
    boundary and `_gate_and_emit` re-joined sentences with a single space.
    """

    TEXT = "First paragraph here.\n\nSecond paragraph here.\n1. item one.\n2. item two."

    @staticmethod
    def _awkward_chunks(text: str) -> list[list[str]]:
        # Splits that land inside the separator run, right after the period,
        # right before the list marker, and one character at a time.
        return [
            [text],
            [
                "First paragraph here.\n",
                "\nSecond paragraph here.",
                "\n1",
                ". item one.\n2. item two.",
            ],
            [
                "First paragraph here",
                ".",
                "\n\nSecond paragraph here.\n",
                "1. item one.\n",
                "2. item two.",
            ],
            list(text),
            [text[i : i + 3] for i in range(0, len(text), 3)],
        ]

    @pytest.mark.asyncio
    async def test_paragraph_and_list_newlines_survive_any_chunking(self):
        for i, chunks in enumerate(self._awkward_chunks(self.TEXT)):
            provider = FakeStreamingLLMProvider([ScriptedPass(chunks=chunks)])
            handler = _make_streaming_handler(provider)
            recorded = _capture_recorded_turns(handler)
            sid = f"s-ws-{i}"

            tokens, complete = await _drain(handler, sid)

            streamed = "".join(t.text for t in tokens)
            assert streamed == self.TEXT, f"chunking {i}: {chunks!r} -> {streamed!r}"
            assert complete.final_response == self.TEXT
            assert recorded == [self.TEXT]
            assert handler.sessions[sid].messages[-1].content == self.TEXT

    @pytest.mark.asyncio
    async def test_slop_stripped_sentence_keeps_surrounding_separators(self):
        text = "Opening line here.\n\nGreat work \U0001f389 on shipping this.\n- next point here."
        provider = FakeStreamingLLMProvider(
            [ScriptedPass(chunks=[text[i : i + 5] for i in range(0, len(text), 5)])]
        )
        handler = _make_streaming_handler(provider)

        tokens, complete = await _drain(handler, "s-ws-slop")

        streamed = "".join(t.text for t in tokens)
        assert "\U0001f389" not in streamed
        assert streamed.startswith("Opening line here.\n\nGreat work")
        assert streamed.endswith("on shipping this.\n- next point here.")
        assert complete.final_response == streamed

    @pytest.mark.asyncio
    async def test_pass2_text_keeps_its_newlines(self):
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    chunks=[],
                    tool_calls=[
                        LLMToolCall(id="call_1", name="query_knowledge_graph", arguments={})
                    ],
                ),
                ScriptedPass(chunks=["Here is the list.\n", "1. alpha one.\n2. beta two."]),
            ]
        )
        handler = _make_streaming_handler(provider)
        handler._dispatch_tool = _StubDispatch()

        tokens, complete = await _drain(handler, "s-ws-pass2")

        assert complete.final_response == "Here is the list.\n1. alpha one.\n2. beta two."
        assert "".join(t.text for t in tokens) == complete.final_response


class _StubDispatch:
    """Callable stand-in for ConversationHandler._dispatch_tool.

    Bypasses the real tool registry so tests that only care about the
    streaming/tool-call plumbing don't also depend on
    query_knowledge_graph's own behavior.
    """

    async def __call__(self, tool_call: LLMToolCall) -> str:
        return "stub tool result"


class TestToolCallTurnStreaming:
    """Acceptance criterion 2 (T3): tool-call turns stream correctly."""

    @pytest.mark.asyncio
    async def test_tool_dispatched_once_and_ws_events_precede_pass2_tokens(self):
        provider = FakeStreamingLLMProvider(
            [
                ScriptedPass(
                    chunks=[],
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            name="query_knowledge_graph",
                            arguments={"query": "user rust experience"},
                        )
                    ],
                ),
                ScriptedPass(chunks=["You have intermediate Rust experience."]),
            ]
        )
        handler = _make_streaming_handler(provider)
        dispatch_calls: list[LLMToolCall] = []

        async def counting_dispatch(tc):
            dispatch_calls.append(tc)
            return "stub tool result"

        handler._dispatch_tool = counting_dispatch

        events: list[StreamEvent] = []
        async for event in handler.handle_message_streaming(
            user_message="What's my experience level with Rust?", session_id="s-tool-turn"
        ):
            events.append(event)

        assert len(dispatch_calls) == 1, "tool must be dispatched exactly once"

        ws_indices = [i for i, e in enumerate(events) if isinstance(e, WSEvent)]
        pass2_token_indices = [
            i for i, e in enumerate(events) if isinstance(e, Token) and e.pass_num == 2
        ]
        assert ws_indices, "expected tool_call_started/completed WSEvents"
        assert pass2_token_indices, "expected pass-2 Token events"
        assert max(ws_indices) < min(pass2_token_indices), (
            "tool_call_started/completed must be yielded before pass 2's " "first token"
        )
        assert [e.payload["type"] for e in events if isinstance(e, WSEvent)] == [
            "tool_call_started",
            "tool_call_completed",
        ]

        complete = next(e for e in events if isinstance(e, Complete))
        assert complete.tool_calls_used == 1
        assert "You have intermediate Rust experience." in complete.final_response
