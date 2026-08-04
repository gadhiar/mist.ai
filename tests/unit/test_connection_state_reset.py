"""A closed connection's residue must not reach the next session (T0).

`VoiceProcessor` is a module-level singleton (`backend/server.py:92`) and the
disconnect path (`backend/server.py:919-952`) resets none of its state. The
sharp case is `latest_user_input`: a parked utterance survives the disconnect
of the session that produced it, and the next turn to reach the drain in
`_process_conversation_turn`'s `finally` replays it.
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


def test_reset_clears_a_set_interrupt_flag():
    """A stale set flag would cancel the NEXT session's first turn."""
    vp = build_voice_processor()
    vp.interrupt_flag.set()

    vp.reset_connection_state()

    assert not vp.interrupt_flag.is_set()


def test_reset_drains_buffered_audio():
    vp = build_voice_processor()
    vp.audio_queue.put(b"stale frame")
    vp.audio_queue.put(b"another")

    vp.reset_connection_state()

    assert vp.audio_queue.empty()


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


def test_helper_resets_only_when_no_connection_remains(monkeypatch):
    import backend.server as server

    calls: list[int] = []
    fake_vp = SimpleNamespace(reset_connection_state=lambda: calls.append(1))
    monkeypatch.setattr(server, "voice_processor", fake_vp)

    monkeypatch.setattr(server, "active_connections", {object()})
    assert server._reset_voice_state_if_last_connection() is False
    assert calls == []

    monkeypatch.setattr(server, "active_connections", set())
    assert server._reset_voice_state_if_last_connection() is True
    assert calls == [1]


def test_helper_is_safe_before_the_voice_processor_exists(monkeypatch):
    import backend.server as server

    monkeypatch.setattr(server, "voice_processor", None)
    monkeypatch.setattr(server, "active_connections", set())

    assert server._reset_voice_state_if_last_connection() is False


def test_the_disconnect_path_calls_the_helper():
    """Reachability, asserted on the CALL SITE.

    Mechanism A's precedent (2026-08-03): the factories were correct and the
    call site was wrong, so the guard belongs on the call site. Parsing beats
    a substring match here -- a match would pass on the name appearing in a
    comment, which is how a dead wire reads as a live one.
    """
    import ast
    import inspect
    import textwrap

    import backend.server as server

    tree = ast.parse(textwrap.dedent(inspect.getsource(server.websocket_endpoint)))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_reset_voice_state_if_last_connection" in called, (
        "the disconnect path does not call the reset helper; "
        "reset_connection_state is unreachable from production"
    )
