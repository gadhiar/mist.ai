"""A closed connection's residue must not reach the next session (T0).

`VoiceProcessor` was a module-level singleton the `/ws` disconnect path reset
through `_reset_voice_state_if_last_connection`. `/ws` now serves one
`RealtimeSession` per connection, so no state outlives its session there
(`tests/unit/realtime/test_session.py`); these tests pin the processor's own
reset until the threaded path is deleted. The sharp case is
`latest_user_input`: a parked utterance that survives a disconnect is replayed
by the next turn to reach the drain in `_process_conversation_turn`'s
`finally`.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from backend.voice_processor import VoiceProcessor


def build_voice_processor() -> VoiceProcessor:
    return VoiceProcessor(
        config=SimpleNamespace(tts_enabled=False),
        message_queue=asyncio.Queue(),
        vault_writer=None,
        vault_sidecar=None,
        invalidation_bus=None,
    )


def test_reset_clears_a_parked_utterance():
    vp = build_voice_processor()
    vp.latest_user_input = ("a dead session's words", "session-gone")

    vp.reset_connection_state()

    assert vp.latest_user_input is None


def test_reset_takes_the_input_lock():
    """The drain at voice_processor.py:733 reads the slot under `input_lock`;
    the reset writes it, so it must take the same lock or it can clear the slot
    between the drain's read and its write-back.
    """
    vp = build_voice_processor()
    vp.latest_user_input = ("parked", "session-gone")
    vp.input_lock.acquire()

    done = threading.Event()
    threading.Thread(target=lambda: (vp.reset_connection_state(), done.set()), daemon=True).start()

    assert not done.wait(timeout=0.5), "reset did not block on input_lock"
    vp.input_lock.release()
    assert done.wait(timeout=5.0)
    assert vp.latest_user_input is None
