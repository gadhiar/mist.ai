"""Stream events emitted by the canonical conversation pipeline.

`ConversationHandler.handle_message_streaming` yields a sequence of these
events. Token + Complete are the load-bearing pair emitted by v1; Thinking
and Filler are extensibility hooks reserved for future iterations.

Pattern motivated by Claude's per-turn shape: input goes in, the LLM
executes whatever it needs (retrieval, tool calls, regen), tokens stream
out, a Complete event terminates the stream. Caller (text client / voice
TTS layer) decides presentation. Extraction is a post-stream side effect
fired before Complete yields and is invisible to the caller.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Base class for events yielded by the streaming conversation pipeline."""


@dataclass(frozen=True, slots=True)
class Token(StreamEvent):
    """A token of the user-facing response.

    `pass_num=1` for tokens streamed from the initial LLM call. `pass_num=2`
    for tokens streamed from the post-tool-call final pass (only emitted when
    the LLM invoked tools mid-turn).
    """

    text: str
    pass_num: int = 1


@dataclass(frozen=True, slots=True)
class Thinking(StreamEvent):
    """Internal reasoning step. Reserved for future implementation.

    Default behavior: not emitted. Future: will surface model reasoning blocks
    similar to Claude's thinking content. Caller may display, log, or ignore.
    """

    text: str


@dataclass(frozen=True, slots=True)
class Filler(StreamEvent):
    """Bridge text during pauses (retrieval, tool execution, extended thinking).

    Reserved for future implementation. Default behavior: not emitted. Future:
    will surface short bridge phrases at retrieval entry / tool dispatch /
    extended LLM latency so voice path can TTS them and maintain conversational
    flow during otherwise silent gaps. Examples: "One moment", "Let me check".
    """

    text: str


@dataclass(frozen=True, slots=True)
class Complete(StreamEvent):
    """Terminal stream event. Response generation finished, slop filter applied.

    `final_response` is the post-filter version. If a regen fired during
    slop check, `final_response` reflects the regenerated text rather than
    whatever was streamed in earlier Token events. Callers that displayed
    raw tokens should reconcile to `final_response` on receipt of Complete.

    `tool_calls_used` and `duration_ms` are observability fields.
    """

    final_response: str
    tool_calls_used: int = 0
    duration_ms: float = 0.0
