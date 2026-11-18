"""Gaming mode as the server wires it: the power gate on `/ws`, `health_status`, and the boot.

`lifespan` builds the gate from gaming mode's `GpuYield` (`_build_power`, `_capability_tiers`,
`_build_power_gate`) and every session applies it (T22's `PowerGatedResponder`). These tests
drive the real `websocket_endpoint` with that gate over a real `GpuYield` on a temporary
marker, with the fakes of `tests/unit/test_session_id_propagation.py`.

Protocol 1.4.0's server-side gate and the threaded voice path are gone; each invariant their
tests pinned is pinned here or by T22's realtime tests:

- a typed turn while yielded is refused with the yield's message: here, and
  `realtime/test_session_power.py::TestTypedTurns::test_the_refusal_names_the_yield_while_gpu_yield_is_yielded`;
- `resume_failed` refused with its own message: the realtime reload never leaves `GpuYield` in
  `resume_failed` (`realtime/test_power.py::TestReload::test_a_real_gpu_yield_is_normal_after_a_failing_reload`),
  so `gpu_yield_refusal` names the missing model then
  (`test_session_power.py::TestPowerGate::test_the_yield_refusal_names_the_missing_model_otherwise`);
- a handled capability command while yielded gets no refusal and streams as a typed turn: here,
  and `test_session_power.py::TestTypedTurns::test_a_typed_message_goes_to_the_capability_only_path`;
- an unhandled message gets what its pre-LLM step drained, then the refusal, and calls no model:
  here, and `TestTypedTurns::test_an_unhandled_message_gets_its_drained_events_then_the_refusal`
  (its `FailingResponder` fails any model call);
- audio while yielded is refused: `audio` is no protocol 2 message (T1); voice at `off` is
  `test_session_power.py::TestVoiceAtOff`;
- turns reach the model when not yielded: here, and
  `TestTypedTurns::test_at_off_with_a_provider_a_typed_question_still_reaches_it`;
- `interrupt` while yielded: `realtime/test_session.py::TestDispatch`; `reset_vad` is removed (T1);
- a voice turn past the gate when the engines were released (main's yielded-GPU error):
  `realtime/test_power.py::TestSessionCut::test_a_tts_turn_in_synthesis_is_cut_whole_and_recorded_interrupted`
  and `TestSlots::test_an_empty_slot_refuses_work_with_a_speech_engine_error`;
- the bridge's `generation_lock`, `latest_user_input` respawn and put-completion ordering: removed
  with the bridge; a session's turns are its controller's, one at a time.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend import server
from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import Complete, Token
from backend.factories import build_mist_runtime
from backend.gpu_yield import YIELDED_MESSAGE, GpuYield
from backend.realtime.power import NO_MODEL_MESSAGE
from backend.realtime.runtime import TorchGpuMeter
from tests.unit.realtime.test_runtime import (
    FakeProvider,
    HandlerFactory,
    Loaders,
    inline_lanes,
    knowledge_config,
    settings,
)
from tests.unit.test_session_id_propagation import (  # noqa: F401 -- autouse fixtures
    RecordingConversationHandler,
    connect,
    install_runtime,
    no_runtime,
    reset_session_context,
)

PROVIDER = object()


def _noop() -> None:
    return None


def _marker(tmp_path: Path) -> Path:
    return tmp_path / "gpu-yielded"


def yielded_gpu(tmp_path: Path) -> GpuYield:
    _marker(tmp_path).write_text('{"v": 1}', encoding="utf-8")
    return GpuYield(_marker(tmp_path), _noop, _noop, poll_s=2.0)


def normal_gpu(tmp_path: Path) -> GpuYield:
    return GpuYield(_marker(tmp_path), _noop, _noop, poll_s=2.0)


def _gate(monkeypatch, gpu: GpuYield | None):
    """The gate exactly as `lifespan` builds it, over `gpu`."""
    monkeypatch.setattr(server, "_gpu_yield", gpu)
    power = server._build_power(gpu)
    return server._build_power_gate(power, server._capability_tiers(power), gpu)


def _errors(frames: list[dict]) -> list[dict]:
    return [frame for frame in frames if frame["type"] == "error"]


def _names(frames: list[dict]) -> list[str]:
    return [str(frame["type"]) for frame in frames]


# -- Typed turns through the gate ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_typed_turn_while_yielded_is_refused_with_the_yield_message(monkeypatch, tmp_path):
    handler = RecordingConversationHandler()
    install_runtime(
        monkeypatch,
        handler,
        power_gate=_gate(monkeypatch, yielded_gpu(tmp_path)),
        llm_provider=PROVIDER,
    )
    connection = await connect()

    await connection.send({"type": "text", "text": "tell me a story"})
    session_id = connection.ws.handshake_session_id
    await connection.leave()

    errors = _errors(connection.ws.frames)
    assert [(e["kind"], e["message"], e["retriable"]) for e in errors] == [
        ("server", YIELDED_MESSAGE, True)
    ]
    assert handler.capability_calls == [("tell me a story", session_id)]
    assert handler.requests == [], "the model path was reached while yielded"


@pytest.mark.asyncio
async def test_a_handled_command_while_yielded_streams_with_no_refusal(monkeypatch, tmp_path):
    handler = RecordingConversationHandler()
    handler.capability_results.append(
        CapabilityOnlyTurn(
            handled=True,
            events=(Token("Timer set."), Complete(final_response="Timer set.")),
        )
    )
    install_runtime(
        monkeypatch,
        handler,
        power_gate=_gate(monkeypatch, yielded_gpu(tmp_path)),
        llm_provider=PROVIDER,
    )
    connection = await connect()

    await connection.send({"type": "text", "text": "set a timer for 10 minutes"})
    session_id = connection.ws.handshake_session_id
    await connection.leave()

    frames = connection.ws.frames
    assert _errors(frames) == []
    assert handler.capability_calls == [("set a timer for 10 minutes", session_id)]
    assert handler.requests == []
    names = _names(frames)
    assert names.index("stream_start") < names.index("stream_complete")


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False], ids=["gaming-mode-normal", "no-gaming-mode"])
async def test_a_typed_turn_reaches_the_model_when_not_yielded(monkeypatch, tmp_path, configured):
    handler = RecordingConversationHandler()
    gpu = normal_gpu(tmp_path) if configured else None
    install_runtime(monkeypatch, handler, power_gate=_gate(monkeypatch, gpu), llm_provider=PROVIDER)
    connection = await connect()

    await connection.send({"type": "text", "text": "hello"})
    await connection.leave()

    assert [r.user_text for r in handler.requests] == ["hello"]
    assert handler.capability_calls == []
    assert _errors(connection.ws.frames) == []
    assert "stream_complete" in _names(connection.ws.frames)


@pytest.mark.asyncio
async def test_before_the_runtime_has_a_provider_the_refusal_names_the_missing_model(
    monkeypatch, tmp_path
):
    handler = RecordingConversationHandler()
    install_runtime(monkeypatch, handler, power_gate=_gate(monkeypatch, normal_gpu(tmp_path)))
    connection = await connect()

    await connection.send({"type": "text", "text": "hello"})
    await connection.leave()

    assert [e["message"] for e in _errors(connection.ws.frames)] == [NO_MODEL_MESSAGE]
    assert handler.requests == []


# -- health_status ---------------------------------------------------------------------------


async def _first_health_status(monkeypatch) -> dict:
    queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(server, "message_queue", queue)
    task = asyncio.create_task(server.health_status_loop(interval_seconds=0))
    try:
        payload = await asyncio.wait_for(queue.get(), timeout=5.0)
    finally:
        task.cancel()
    return json.loads(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("build", "expected_llm"),
    [
        pytest.param(yielded_gpu, False, id="yielded"),
        pytest.param(normal_gpu, True, id="normal"),
        pytest.param(None, True, id="no-gaming-mode"),
    ],
)
async def test_health_status_llm_is_false_while_yielded(monkeypatch, tmp_path, build, expected_llm):
    monkeypatch.setattr(server, "_gpu_yield", build(tmp_path) if build is not None else None)
    monkeypatch.setattr(server, "runtime", SimpleNamespace(agent_available=True))

    payload = await _first_health_status(monkeypatch)

    assert payload == {"type": "health_status", "llm": expected_llm, "agent": True, "local": True}


@pytest.mark.asyncio
async def test_health_status_reads_no_llm_and_no_agent_before_the_runtime_is_built(monkeypatch):
    monkeypatch.setattr(server, "_gpu_yield", None)
    monkeypatch.setattr(server, "runtime", None)

    payload = await _first_health_status(monkeypatch)

    assert payload == {"type": "health_status", "llm": False, "agent": False, "local": True}


@pytest.mark.asyncio
async def test_health_status_agent_is_false_when_turns_are_answered_without_memory(monkeypatch):
    monkeypatch.setattr(server, "_gpu_yield", None)
    monkeypatch.setattr(server, "runtime", SimpleNamespace(agent_available=False))

    payload = await _first_health_status(monkeypatch)

    assert (payload["llm"], payload["agent"]) == (True, False)


# -- The boot --------------------------------------------------------------------------------


def _build_at_boot(
    monkeypatch, gpu: GpuYield | None
) -> tuple[Loaders, FakeProvider, HandlerFactory]:
    """`build_mist_runtime` with the gate `lifespan` builds, over fake loaders and provider."""
    no_device = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", no_device)
    loaders = Loaders()
    provider = FakeProvider()
    factory = HandlerFactory()
    build_mist_runtime(
        knowledge_config(),  # type: ignore[arg-type]
        voice_settings=settings(),
        llm_provider=provider,  # type: ignore[arg-type]
        handler_factory=factory,
        engine_loaders=loaders.build(),
        gpu_meter=TorchGpuMeter(lambda: no_device),
        lanes=inline_lanes(),
        power_gate=_gate(monkeypatch, gpu),
        torch_loader=lambda: no_device,
    )
    return loaders, provider, factory


def test_a_boot_with_the_marker_present_builds_no_cuda_engine_and_skips_the_warmup(
    monkeypatch, tmp_path
):
    """Main's `load_all_models(yielded=True)`: no GPU model, no LLM warm-up, embeddings warmed."""
    loaders, provider, factory = _build_at_boot(monkeypatch, yielded_gpu(tmp_path))

    assert loaders.calls, "the CPU engines still load"
    assert not [call for call in loaders.calls if call.endswith("@cuda")]
    assert provider.warmups == []
    assert factory.embedder.warmed == 1


def test_a_boot_without_the_marker_tries_cuda_first_and_warms_the_llm(monkeypatch, tmp_path):
    loaders, provider, factory = _build_at_boot(monkeypatch, normal_gpu(tmp_path))

    assert [call for call in loaders.calls if call.endswith("@cuda")]
    assert len(provider.warmups) == 1
    assert factory.embedder.warmed == 1
