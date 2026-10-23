"""Exceptions raised by the realtime voice package.

Hierarchy:

    MistError
    +-- RealtimeError
        +-- ProtocolError
        +-- SpeechEngineError
"""

from ..errors import MistError


class RealtimeError(MistError):
    """Base exception for the realtime voice path."""


class ProtocolError(RealtimeError):
    """A binary frame or control message violates the realtime protocol.

    The session answers it with an `error` event of kind `protocol`, so the error carries the
    offending field separately: the client can point at the field without parsing the message.

    Attributes:
        field: The header field, payload or dotted message path that failed, for example
            `magic`, `payload` or `detail.utterance_id`.
        reason: What was wrong with it.
    """

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}")
        self.field = field
        self.reason = reason


class SpeechEngineError(RealtimeError):
    """A speech engine (VAD, turn-end, STT or TTS) failed to load or to run a job."""
