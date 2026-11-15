"""The WebSocket and `health_status` behaviour of gaming mode.

Driven through FastAPI's test client with the voice processor stubbed and a
real `GpuYield` over a temporary marker. `TestClient` is constructed without
the `with` block, so the real lifespan, and with it the model stack, never
runs (as in `tests/unit/test_origin_allowlist.py`).

With a capability host attached, a `text` message while yielded is offered to
the capability-only turn, and refused as before only when nothing answers it;
`audio` is refused either way.

The `VoiceProcessor` tests cover the other side of the gate: a voice turn
admitted just before Whisper was released, whose model manager raises
`GpuYieldedError`, and the capability-only turn, which streams through the
text turn's output path and never reaches the model path.
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
from backend.chat.stream_events import Complete, Token, WSEvent
from backend.config import VoiceConfig
from backend.errors import GpuYieldedError
from backend.gpu_yield import RESUME_FAILED_MESSAGE, YIELDED_MESSAGE, GpuYield
from backend.request_context import current_turn_complete
from backend.voice_models.model_manager import ModelManager

_TIMER_TEXT = "set a timer for 10 minutes"


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


# --- Capability commands at the off level ----------------------------------------


class CapabilityStubVoiceProcessor(StubVoiceProcessor):
    """Also takes the capability-only turn, answering it as `handled` says."""

    def __init__(self, handled: bool) -> None:
        super().__init__()
        self.handled = handled
        self.capability_turns: list[tuple[str, object]] = []

    def _process_conversation_turn(
        self, user_text, session_id=None, request_id=None, *, capability_only=False
    ) -> bool:
        if capability_only:
            self.capability_turns.append((user_text, session_id))
            return self.handled
        self.turns.append(("text", user_text))
        return True


class SnapshotlessHost:
    """Stands in for an attached capability host; its connect snapshot is empty."""

    async def snapshot_events(self) -> list[dict]:
        return []


def _attach_host(monkeypatch) -> None:
    monkeypatch.setattr(server, "_capability_build", SimpleNamespace(host=SnapshotlessHost()))


def _capability_stub(monkeypatch, *, handled: bool) -> CapabilityStubVoiceProcessor:
    processor = CapabilityStubVoiceProcessor(handled)
    monkeypatch.setattr(server, "voice_processor", processor)
    return processor


def _send_and_collect(ws, *messages: dict) -> list[dict]:
    """Send `messages`, then an interrupt as a barrier, and return the replies before it."""
    for message in messages:
        ws.send_json(message)
    ws.send_json({"type": "interrupt"})
    replies = []
    while True:
        reply = ws.receive_json()
        if reply.get("type") == "status":
            return replies
        replies.append(reply)


def test_a_handled_text_command_while_yielded_gets_no_refusal(stub, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_gpu_yield", yielded_gpu(tmp_path))
    _attach_host(monkeypatch)
    processor = _capability_stub(monkeypatch, handled=True)

    with _session(TestClient(server.app)) as ws:
        session_id = ws.receive_json()["session_id"]
        replies = _send_and_collect(ws, {"type": "text", "text": "set a timer for 10 minutes"})

    assert replies == []
    assert processor.capability_turns == [("set a timer for 10 minutes", session_id)]
    assert processor.turns == []


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(yielded_gpu, YIELDED_MESSAGE, id="yielded"),
        pytest.param(resume_failed_gpu, RESUME_FAILED_MESSAGE, id="resume-failed"),
    ],
)
def test_an_unhandled_text_message_gets_the_state_refusal(
    stub, monkeypatch, tmp_path, build, message
):
    monkeypatch.setattr(server, "_gpu_yield", build(tmp_path))
    _attach_host(monkeypatch)
    processor = _capability_stub(monkeypatch, handled=False)

    with _session(TestClient(server.app)) as ws:
        ws.receive_json()
        replies = _send_and_collect(ws, {"type": "text", "text": "tell me a story"})

    assert replies == [_envelope(message)]
    assert [text for text, _session_id in processor.capability_turns] == ["tell me a story"]
    assert processor.turns == []


def test_audio_while_yielded_is_refused_with_a_host_attached(stub, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_gpu_yield", yielded_gpu(tmp_path))
    _attach_host(monkeypatch)
    processor = _capability_stub(monkeypatch, handled=True)

    with _session(TestClient(server.app)) as ws:
        ws.receive_json()
        replies = _send_and_collect(
            ws, {"type": "audio", "audio": [0.0, 0.1, 0.2], "sample_rate": 16000}
        )

    assert replies == [_envelope(YIELDED_MESSAGE)]
    assert processor.turns == []
    assert processor.capability_turns == []


def test_empty_text_while_yielded_is_refused_with_a_host_attached(stub, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_gpu_yield", yielded_gpu(tmp_path))
    _attach_host(monkeypatch)
    processor = _capability_stub(monkeypatch, handled=True)

    with _session(TestClient(server.app)) as ws:
        ws.receive_json()
        replies = _send_and_collect(ws, {"type": "text", "text": ""})

    assert replies == [_envelope(YIELDED_MESSAGE)]
    assert processor.capability_turns == []


def test_text_with_a_host_when_not_yielded_takes_the_normal_turn(stub, monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_gpu_yield", normal_gpu(tmp_path))
    _attach_host(monkeypatch)
    processor = _capability_stub(monkeypatch, handled=True)

    with _session(TestClient(server.app)) as ws:
        ws.receive_json()
        replies = _send_and_collect(ws, {"type": "text", "text": "hello"})

    assert replies == []
    assert processor.turns == [("text", "hello")]
    assert processor.capability_turns == []


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


# --- The capability-only turn in the voice processor ------------------------------

_STARTED = {"type": "tool_call_started", "tool_call_id": "call-1", "name": "timers.start"}
_COMPLETED = {"type": "tool_call_completed", "tool_call_id": "call-1", "name": "timers.start"}
_REPLY = "Timer set for 10 minutes."


class ScriptedCapabilityHandler:
    """A conversation handler whose capability-only answer is scripted."""

    def __init__(self, answer) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    async def handle_capability_only(self, text: str, session_id: str):
        self.calls.append((text, session_id))
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _answered_events() -> list:
    return [
        WSEvent(payload=dict(_STARTED)),
        WSEvent(payload=dict(_COMPLETED)),
        Token(text=_REPLY),
        Complete(final_response=_REPLY, tool_calls_used=1, duration_ms=3.0),
    ]


def _trim(text: str) -> str:
    return ModelManager.trim_to_last_sentence(None, text)  # type: ignore[arg-type]


def _processor(queue: asyncio.Queue, handler, model_items=None):
    """A real `VoiceProcessor`, text only, over a handler and a scripted model path.

    The model path records each call; with `model_items` None it yields nothing, so a test
    can assert that it never ran.
    """
    from backend import voice_processor as vp_module

    model_calls: list[str] = []

    def generate_llm_response(user_text):
        model_calls.append(user_text)
        for item in model_items or ():
            if isinstance(item, Complete):
                current_turn_complete.set(item)
            else:
                yield item

    processor = vp_module.VoiceProcessor(VoiceConfig(tts_enabled=False), message_queue=queue)
    processor.loop = asyncio.get_running_loop()
    processor.models = SimpleNamespace(
        knowledge=SimpleNamespace(conversation_handler=handler, is_enabled=lambda: True),
        generate_llm_response=generate_llm_response,
        trim_to_last_sentence=_trim,
    )
    return processor, model_calls


async def _messages_until_idle(queue: asyncio.Queue) -> list[dict]:
    messages = []
    while True:
        message = json.loads(await asyncio.wait_for(queue.get(), timeout=5.0))
        messages.append(message)
        if message == {"type": "state_cycle", "state": "idle"}:
            return messages


async def _settle() -> None:
    """Let any put the worker thread scheduled on the loop run."""
    for _ in range(5):
        await asyncio.sleep(0)


def _comparable(messages: list[dict]) -> list[dict]:
    """The messages with the per-turn id and the timing left out."""
    out = []
    for message in messages:
        message = {k: v for k, v in message.items() if k not in ("turn_id", "duration_ms")}
        out.append(message)
    return out


@pytest.mark.asyncio
async def test_a_handled_capability_turn_streams_like_a_text_turn():
    queue: asyncio.Queue = asyncio.Queue()
    handler = ScriptedCapabilityHandler(_answered_events())
    processor, model_calls = _processor(queue, handler)

    answered = await asyncio.to_thread(
        processor._process_conversation_turn, _TIMER_TEXT, session_id="s1", capability_only=True
    )
    off_messages = await _messages_until_idle(queue)

    normal_queue: asyncio.Queue = asyncio.Queue()
    normal, _calls = _processor(
        normal_queue,
        ScriptedCapabilityHandler(None),
        model_items=[dict(_STARTED), dict(_COMPLETED), _REPLY, _answered_events()[-1]],
    )
    await asyncio.to_thread(normal._process_conversation_turn, _TIMER_TEXT, session_id="s1")
    normal_messages = await _messages_until_idle(normal_queue)

    assert answered is True
    assert handler.calls == [(_TIMER_TEXT, "s1")]
    assert model_calls == []
    assert _comparable(off_messages) == _comparable(normal_messages)
    assert [m["type"] for m in off_messages] == [
        "state_cycle",
        "stream_start",
        "tool_call_started",
        "tool_call_completed",
        "state_cycle",
        "stream_token",
        "stream_complete",
        "state_cycle",
    ]
    complete = off_messages[-2]
    assert (complete["full_text"], complete["tool_calls_used"]) == (_REPLY, 1)
    assert not processor.generation_lock.locked()


@pytest.mark.asyncio
async def test_an_unhandled_capability_turn_sends_nothing_and_calls_no_model():
    queue: asyncio.Queue = asyncio.Queue()
    handler = ScriptedCapabilityHandler(None)
    processor, model_calls = _processor(queue, handler)

    answered = await asyncio.to_thread(
        processor._process_conversation_turn,
        "tell me a story",
        session_id="s1",
        capability_only=True,
    )
    await _settle()

    assert answered is False
    assert handler.calls == [("tell me a story", "s1")]
    assert model_calls == []
    assert queue.empty()
    assert not processor.generation_lock.locked()


@pytest.mark.asyncio
async def test_a_capability_turn_with_no_conversation_handler_is_unhandled():
    queue: asyncio.Queue = asyncio.Queue()
    processor, model_calls = _processor(queue, None)
    processor.models.knowledge = None

    answered = await asyncio.to_thread(
        processor._process_conversation_turn, _TIMER_TEXT, session_id="s1", capability_only=True
    )
    await _settle()

    assert answered is False
    assert model_calls == []
    assert queue.empty()
    assert not processor.generation_lock.locked()


@pytest.mark.asyncio
async def test_a_capability_turn_while_a_turn_is_in_flight_is_unhandled_and_not_queued():
    queue: asyncio.Queue = asyncio.Queue()
    handler = ScriptedCapabilityHandler(_answered_events())
    processor, model_calls = _processor(queue, handler)
    processor.generation_lock.acquire()
    try:
        answered = await asyncio.to_thread(
            processor._process_conversation_turn,
            _TIMER_TEXT,
            session_id="s1",
            capability_only=True,
        )
    finally:
        processor.generation_lock.release()
    await _settle()

    assert answered is False
    assert handler.calls == []
    assert processor.latest_user_input is None
    assert model_calls == []
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_capability_turn_does_not_respawn_a_queued_model_turn():
    queue: asyncio.Queue = asyncio.Queue()
    handler = ScriptedCapabilityHandler(_answered_events())
    processor, model_calls = _processor(queue, handler)
    processor.latest_user_input = ("what is the weather", "s1")

    await asyncio.to_thread(
        processor._process_conversation_turn, _TIMER_TEXT, session_id="s1", capability_only=True
    )
    await _messages_until_idle(queue)
    await asyncio.sleep(0.05)

    assert model_calls == []
    assert processor.latest_user_input == ("what is the weather", "s1")


@pytest.mark.asyncio
async def test_a_failing_capability_turn_gets_an_error_without_the_exception_text():
    queue: asyncio.Queue = asyncio.Queue()
    handler = ScriptedCapabilityHandler(RuntimeError("token tok-quince-7741 rejected"))
    processor, model_calls = _processor(queue, handler)

    answered = await asyncio.to_thread(
        processor._process_conversation_turn, _TIMER_TEXT, session_id="s1", capability_only=True
    )
    messages = await _messages_until_idle(queue)

    assert answered is True
    assert model_calls == []
    errors = [m for m in messages if m["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["kind"] == "model"
    assert errors[0]["retriable"] is True
    assert "tok-quince-7741" not in json.dumps(messages)
    assert "stream_start" not in [m["type"] for m in messages]
    assert not processor.generation_lock.locked()
