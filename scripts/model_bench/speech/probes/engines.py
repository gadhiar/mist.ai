"""The engine contract every speech adapter meets.

The suites time engines through these protocols only, so a fake with a controlled clock can stand
in for any engine in the unit tests. `numpy` appears in annotations only and is imported under
`TYPE_CHECKING`, so importing this module loads no numpy and no engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy


class SpeechEngine(Protocol):
    """A text-to-speech engine speaking with one of its own stock voices.

    `load` takes a stock voice by name from `voices()` and nothing else: no path and no audio,
    because nothing is cloned in any run. `stream` yields audio chunks as the engine produces
    them; a whole-segment engine yields once.
    """

    name: str
    device: str
    sample_rate: int

    def voices(self) -> list[str]:
        """Return the engine's stock voice names."""
        ...

    def load(self, voice: str) -> None:
        """Load the model and select the stock voice `voice`."""
        ...

    def stream(self, text: str) -> Iterator[numpy.ndarray]:
        """Synthesize `text`, yielding audio chunks at `sample_rate`."""
        ...


class Transcriber(Protocol):
    """A speech-to-text engine that transcribes a whole utterance."""

    def transcribe(self, audio: numpy.ndarray, sample_rate: int) -> str:
        """Return the final text for `audio`."""
        ...


class StreamingTranscriber(Transcriber, Protocol):
    """A transcriber that also takes audio chunk by chunk, as a live microphone delivers it.

    Final-text latency is timed from the last `feed` to the return of `finish`.
    """

    def feed(self, chunk: numpy.ndarray) -> None:
        """Take the next chunk of audio."""
        ...

    def finish(self) -> str:
        """Flush the stream and return the final text."""
        ...


class TurnDetector(Protocol):
    """An end-of-turn model."""

    def predict(self, audio: numpy.ndarray) -> float:
        """Return the probability that the speaker's turn is finished."""
        ...
