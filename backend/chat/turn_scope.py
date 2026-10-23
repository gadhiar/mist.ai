"""Per-turn state for one conversation turn.

`ConversationHandler` is shared by every session in the process. Before this
module, the FE-bound event buffer, the session id and the turn index lived on
the handler itself, so a turn starting while another was still in flight
cleared the first turn's buffer and overwrote the session id its graph seed
read. A `TurnScope` is created at the start of each turn and passed explicitly
to every method that reads or writes that state, so turns in different
sessions can run concurrently without seeing each other's events.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TurnScope:
    """State owned by exactly one conversation turn.

    Mutable, unlike the request-side value objects, because the pipeline
    appends to it while the turn runs. It is created per turn and never
    shared between turns, so no locking is needed.

    Attributes:
        session_id: Session the turn belongs to.
        turn_index: The handler's turn counter at the start of this turn.
            One counter per handler across all sessions, as before T4, so
            seeds match the pre-T4 handler; a turn in another session
            advances it.
        ws_events: ADR-017 FE-bound event payloads (tool_call_started and
            tool_call_completed, cards_summon and cards_dismiss,
            graph_subgraph, vault_results, form_switch) appended during tool
            dispatch and drained by the streaming pipeline as `WSEvent`s.
    """

    session_id: str
    turn_index: int
    ws_events: list[dict[str, object]] = field(default_factory=list)

    @property
    def graph_seed(self) -> str:
        """Seed for the graph_subgraph distant-point layout.

        Session plus turn index, so a repeated query within one turn lays out
        identically and different turns lay out differently.
        """
        return f"{self.session_id}:{self.turn_index}"

    def emit(self, payload: dict[str, object]) -> None:
        """Append one FE-bound event payload to this turn's buffer."""
        self.ws_events.append(payload)

    def drain_ws_events(self) -> list[dict[str, object]]:
        """Return the buffered payloads in emit order and empty the buffer.

        No await sits between the read and the reset, so a concurrent task
        cannot append to the list being returned.
        """
        events = self.ws_events
        self.ws_events = []
        return events
