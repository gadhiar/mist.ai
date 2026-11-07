"""The WebSocket and `health_status` behaviour of gaming mode.

Driven through FastAPI's test client with the voice processor stubbed and a
real `GpuYield` over a temporary marker. `TestClient` is constructed without
the `with` block, so the real lifespan, and with it the model stack, never
runs (as in `tests/unit/test_origin_allowlist.py`).

The last test covers the other side of the gate: a voice turn admitted just
before Whisper was released, driven through a real `VoiceProcessor` whose
model manager raises `GpuYieldedError`.
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.config import VoiceConfig
from backend.errors import GpuYieldedError
from backend.gpu_yield import RESUME_FAILED_MESSAGE, YIELDED_MESSAGE, GpuYield


class StubVoiceProcessor:
    """Records the turn methods the WebSocket dispatches to. No models, no VAD."""

    def __init__(self) -> None:
        self.turns: list[tuple[str, object]] = []
        self.vad_resets = 0
        self.interrupt_flag = threading.Event()
        self.models = SimpleNamespace(knowledge=None)

    def process_complete_audio(self, audio_data, sample_rate, session_id=None) -> None:
        self.turns.append(("audio", len(audio_data)))

    def _process_conversation_turn(self, user_text, session_id=None, request_id=None) -> None:
        self.turns.append(("text", user_text))

    def reset_vad(self) -> None:
        self.vad_resets += 1

    def reset_connection_state(self) -> None:
        return None


def _noop() -> None:
    return None


def _failing_resume() -> None:
    raise RuntimeError("no memory for Whisper")


def _gpu(marker: Path, *, on_resume=_noop) -> GpuYield:
    return GpuYield(marker, _noop, on_resume, poll_s=2.0)


def yielded_gpu(tmp_path: Path) -> GpuYield:
    marker = tmp_path / "gpu-yielded"
    marker.write_text('{"v": 1}', encoding="utf-8")
    return _gpu(marker)


def resume_failed_gpu(tmp_path: Path) -> GpuYield:
    marker = tmp_path / "gpu-yielded"
    marker.write_text('{"v": 1}', encoding="utf-8")
    gpu = _gpu(marker, on_resume=_failing_resume)
    marker.unlink()
    asyncio.run(gpu.poll_once())
    assert gpu.state() == "resume_failed"
    return gpu


def normal_gpu(tmp_path: Path) -> GpuYield:
    return _gpu(tmp_path / "gpu-yielded")


def _envelope(message: str) -> dict:
    return {
        "type": "error",
        "kind": "server",
        "message": message,
        "retriable": True,
        "context": None,
    }


@pytest.fixture
def stub(monkeypatch) -> StubVoiceProcessor:
    processor = StubVoiceProcessor()
    monkeypatch.setattr(server, "voice_processor", processor)
    monkeypatch.setattr(server, "_gpu_yield", None)
    monkeypatch.delenv("MIST_ALLOWED_ORIGINS", raising=False)
    return processor


def _session(client: TestClient):
    return client.websocket_connect("/ws")


def _send_turns_and_collect(ws) -> list[dict]:
    """Send one audio and one text message, then an interrupt as a barrier.

    The endpoint handles messages in order, so the interrupt's acknowledgement
    arrives only after both turns were dispatched or refused.
    """
    ws.send_json({"type": "audio", "audio": [0.0, 0.1, 0.2], "sample_rate": 16000})
    ws.send_json({"type": "text", "text": "hello"})
    ws.send_json({"type": "interrupt"})
    replies = []
    while True:
        reply = ws.receive_json()
        replies.append(reply)
        if reply.get("type") == "status":
            return replies


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(yielded_gpu, YIELDED_MESSAGE, id="yielded"),
        pytest.param(resume_failed_gpu, RESUME_FAILED_MESSAGE, id="resume-failed"),
    ],
)
def test_turns_are_refused_with_the_state_message(stub, monkeypatch, tmp_path, build, message):
    monkeypatch.setattr(server, "_gpu_yield", build(tmp_path))

    with _session(TestClient(server.app)) as ws:
        assert ws.receive_json()["type"] == "session_started"
        replies = _send_turns_and_collect(ws)

    assert replies[:2] == [_envelope(message), _envelope(message)]
    assert stub.turns == []


def test_interrupt_and_reset_vad_behave_as_today_while_yielded(stub, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_gpu_yield", yielded_gpu(tmp_path))

    with _session(TestClient(server.app)) as ws:
        ws.receive_json()
        ws.send_json({"type": "interrupt"})
        interrupt_reply = ws.receive_json()
        ws.send_json({"type": "reset_vad"})
        reset_reply = ws.receive_json()

    assert interrupt_reply == {"type": "status", "message": "Interrupt acknowledged"}
    assert stub.interrupt_flag.is_set()
    assert reset_reply == {"type": "status", "message": "VAD reset"}
    assert stub.vad_resets == 1


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(normal_gpu, id="gaming-mode-configured-but-normal"),
        pytest.param(None, id="no-gaming-mode"),
    ],
)
def test_turns_reach_the_voice_processor_when_not_yielded(stub, monkeypatch, tmp_path, build):
    if build is not None:
        monkeypatch.setattr(server, "_gpu_yield", build(tmp_path))

    with _session(TestClient(server.app)) as ws:
        ws.receive_json()
        replies = _send_turns_and_collect(ws)

    assert stub.turns == [("audio", 3), ("text", "hello")]
    assert [r["type"] for r in replies] == ["status"]


async def _first_health_status(monkeypatch) -> dict:
    queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(server, "message_queue", queue)
    task = asyncio.create_task(server.health_status_loop(interval_seconds=0))
    try:
        payload = await asyncio.wait_for(queue.get(), timeout=5.0)
    finally:
        task.cancel()
    return json.loads(payload)


@pytest.mark.parametrize(
    ("build", "expected_llm"),
    [
        pytest.param(yielded_gpu, False, id="yielded"),
        pytest.param(resume_failed_gpu, False, id="resume-failed"),
        pytest.param(normal_gpu, True, id="normal"),
        pytest.param(None, True, id="no-gaming-mode"),
    ],
)
def test_health_status_llm_is_false_while_yielded(stub, monkeypatch, tmp_path, build, expected_llm):
    if build is not None:
        monkeypatch.setattr(server, "_gpu_yield", build(tmp_path))

    payload = asyncio.run(_first_health_status(monkeypatch))

    assert payload == {"type": "health_status", "llm": expected_llm, "agent": False, "local": True}


class ReleasedModels:
    """A model manager whose Whisper was released after the turn passed the gate."""

    def transcribe_audio(self, audio_data, sample_rate):
        raise GpuYieldedError("Whisper is not loaded")


@pytest.mark.asyncio
async def test_speech_turn_past_the_gate_gets_the_yielded_envelope_and_records_no_turn():
    from backend import voice_processor as vp_module

    queue: asyncio.Queue = asyncio.Queue()
    processor = vp_module.VoiceProcessor(VoiceConfig(tts_enabled=False), message_queue=queue)
    processor.models = ReleasedModels()
    processor.loop = asyncio.get_running_loop()
    turns: list[str] = []
    processor._process_conversation_turn = lambda text, **kwargs: turns.append(text)

    await asyncio.to_thread(processor._process_user_speech, [0.0, 0.1], 16000)
    message = json.loads(await asyncio.wait_for(queue.get(), timeout=5.0))

    assert message == _envelope(YIELDED_MESSAGE)
    assert queue.empty()
    assert turns == []
