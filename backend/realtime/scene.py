"""The scene seam: what the turn controller tells the screen, anchored to the audio clock (D3).

The controller calls a `ScenePort` with intents, each carrying an anchor. It never computes a
sample: when it calls the port, the segment it names may still be in synthesis, and the ledger
records a segment only once a frame of it is queued. So an anchor stays symbolic (a segment's
start, or the stream's end) until the pacer reaches it and calls one of the two
`PlaybackListener` hooks, just before it queues that frame. A port that writes to the client
holds each anchored intent until a hook resolves it, then writes it ahead of the frame, so the
operation always reaches the client before the sample it names.

Stage 1 wires `NullScenePort`, which sends nothing. The port is called on the controller's
mailbox task and the hooks on the turn's play task, both on the event loop thread, so a port
never blocks and takes no lock.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

ScenePhase = Literal["wake", "listening", "thinking", "speaking", "idle"]

ScreenView = Literal["alarms", "timers", "clock", "lights", "music"]
"""The panes a tool result can open, as the capabilities layer names them."""


@dataclass(frozen=True, slots=True)
class Immediate:
    """Run on receipt."""


@dataclass(frozen=True, slots=True)
class AtSegment:
    """Run when segment `segment_index` of the tagged stream starts playing.

    A segment that never plays (it cleaned to empty, or its audio was dropped) calls no hook, so
    a port resolves this at the first `segment_starting` of the tag whose index is
    `segment_index` or more.
    """

    audio_tag: int
    segment_index: int


@dataclass(frozen=True, slots=True)
class AtStreamEnd:
    """Run when the tagged stream's last sample plays; never runs if the stream is cut."""

    audio_tag: int


SceneAnchor = Immediate | AtSegment | AtStreamEnd


@dataclass(frozen=True, slots=True)
class AtAudio:
    """The wire form of a resolved anchor: a 24 kHz sample offset into the tag's stream.

    Only a port that writes to the client produces it; the controller never does.
    """

    audio_tag: int
    sample: int


@runtime_checkable
class ScenePort(Protocol):
    """What the controller calls with scene intents; it also serves as the pacer's listener.

    `cancel_turn` comes in the same synchronous step as the turn's purge and drops every intent
    of the turn still held, so nothing of a cut turn follows its purge. `turn_ended` comes once
    for a turn that ends without a cut, after its last frame was queued or when it queued none;
    a port then writes what it still holds of the turn as `Immediate`.
    """

    def phase(self, phase: ScenePhase, *, turn_id: str | None, at: SceneAnchor) -> None:
        """MIST's visible phase, for a turn or, with `turn_id` None, for no turn."""
        ...

    def tool_view(
        self,
        turn_id: str,
        call_id: str,
        view: ScreenView,
        data: Mapping[str, object] | None,
        *,
        at: SceneAnchor,
    ) -> None:
        """Open the pane a tool call produced; `call_id` is the pane's identity."""
        ...

    def cancel_turn(self, turn_id: str) -> None:
        """Drop the turn's held intents: it was cut."""
        ...

    def turn_ended(self, turn_id: str) -> None:
        """Write the turn's held intents now: it ended without a cut."""
        ...

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        """The pacer's hook: segment `segment_index`'s first frame is next."""
        ...

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        """The pacer's hook: the tag's terminal frame is next."""
        ...


class NullScenePort:
    """Stage 1's port: every method, the hooks included, does nothing."""

    def phase(self, phase: ScenePhase, *, turn_id: str | None, at: SceneAnchor) -> None:
        """Do nothing."""

    def tool_view(
        self,
        turn_id: str,
        call_id: str,
        view: ScreenView,
        data: Mapping[str, object] | None,
        *,
        at: SceneAnchor,
    ) -> None:
        """Do nothing."""

    def cancel_turn(self, turn_id: str) -> None:
        """Do nothing."""

    def turn_ended(self, turn_id: str) -> None:
        """Do nothing."""

    def segment_starting(self, audio_tag: int, segment_index: int, first_sample: int) -> None:
        """Do nothing."""

    def stream_ending(self, audio_tag: int, last_sample: int) -> None:
        """Do nothing."""
