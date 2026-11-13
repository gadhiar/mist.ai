"""The per-session turn controller's core (T11a).

Every test drives the controller only through `post` and `close`, with the recording transport,
`RecordingScenePort`, the fake responder, `FakeSynthesizer` on an inline lane, and the fake clock
from `turn_fakes.py`. The client-facing record is what the recording transport's sender took.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest
import pytest_asyncio

from backend.chat.context_budget import PromptStats
from backend.chat.stream_events import Complete, HandOff, StateHint, Token
from backend.chat.turn_scope import CommitResult
from backend.realtime.budget import BUDGET_PRESETS, TurnConfig
from backend.realtime.playback import PlaybackListener
from backend.realtime.protocol import Frame, MsgType
from backend.realtime.scene import (
    AtSegment,
    AtStreamEnd,
    Immediate,
    NullScenePort,
    ScenePort,
)
from backend.realtime.speech import TextSegment
from backend.realtime.turn import (
    BargeInReport,
    Close,
    Interrupt,
    PlaybackDone,
    PlaybackProgress,
    Text,
    TurnController,
    TurnPlacement,
    WakeRank,
    WakeVerdict,
    judge_over_budget,
)
from tests.fakes.speech import FakeSynthesizer

from .turn_fakes import (
    Call,
    FakeWakeArbiter,
    Harness,
    IdCounter,
    Pause,
    RecordingIngress,
    close_live_controllers,
    dispatch_tool,
    make_harness,
    settle,
    tool_completed,
    tool_started,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _stop_controllers():
    yield
    await close_live_controllers()


TURN_LOGGER = "backend.realtime.turn"


def subsequence(names: list[str], expected: list[str]) -> bool:
    it = iter(names)
    return all(name in it for name in expected)


def frames(h: Harness, msg_type: MsgType) -> list[Frame]:
    return [i for i in h.outbound.client if isinstance(i, Frame) and i.msg_type == msg_type]


def metrics(h: Harness):
    return h.outbound.of("turn_metrics")


async def finish_voice_turn(h: Harness, turn_id: str) -> None:
    h.post(PlaybackDone(turn_id=turn_id, played_ms=1))
    await h.settle()


def tool_task(scope) -> None:
    """Dispatch a tool whose task runs until the test ends it."""
    dispatch_tool(scope)
    scope.tool_tasks.append(asyncio.ensure_future(asyncio.Event().wait()))


# -- The mailbox ---------------------------------------------------------------------


async def test_close_returns_only_after_the_turns_commit():
    pause = Pause()
    h = make_harness([Token("One. "), pause, Token("Two.")])
    h.start_voice()
    await h.speak("hello")

    await h.controller.close()

    assert h.responder.log[-1] == "commit:interrupted"
    assert h.controller.task.done()


async def test_messages_are_handled_in_post_order_by_one_task():
    h = make_harness([Token("A.")], [Token("B.")])
    h.post(Text("first"), Text("second"))
    await h.settle()

    assert [r.user_text for r in h.responder.requests] == ["first", "second"]
    # The first turn was cut by the second before the second opened.
    assert h.responder.log.index("commit:interrupted") < h.responder.log.index("open", 1)


# -- The channel rule ------------------------------------------------------------------


async def test_voice_turn_with_tts_enabled_a_synthesizer_and_playback_uses_tts():
    h = make_harness()
    h.start_voice()
    await h.speak("hello")

    (start,) = h.outbound.of("stream_start")
    assert (start.channel, start.origin) == ("tts", "voice")
    assert start.audio_tag is not None
    assert h.responder.requests[0].channel == "tts"


@pytest.mark.parametrize("missing", ["tts_enabled", "synthesizer", "playback"])
async def test_voice_turn_missing_one_condition_uses_text(missing):
    h = make_harness(
        tts_enabled=missing != "tts_enabled",
        synthesizer=missing != "synthesizer",
    )
    h.start_voice(playback=missing != "playback")
    await h.speak("hello")

    (start,) = h.outbound.of("stream_start")
    assert (start.channel, start.origin) == ("text", "voice")
    assert start.audio_tag is None
    assert h.responder.requests[0].channel == "text"


async def test_typed_turn_uses_text_with_everything_available():
    h = make_harness()
    h.start_voice()
    h.post(Text("hi"))
    await h.settle()

    (start,) = h.outbound.of("stream_start")
    assert (start.channel, start.origin) == ("text", "text")


async def test_voice_turn_on_text_channel_behaves_and_records_as_typed():
    h = make_harness([Token("Hi "), Token("there.")], synthesizer=False)
    h.start_voice()
    await h.speak("hello")

    assert not [i for i in h.outbound.client if isinstance(i, Frame)]
    assert h.responder.commits[0].outcome == "completed"
    assert h.responder.commits[0].delivered_text == "Hi there."
    assert h.responder.commits[0].tts_model is None
    assert subsequence(
        h.outbound.names(), ["stream_start", "stream_complete", "state_cycle:idle", "turn_metrics"]
    )
    assert h.controller.phase == "idle"


# -- Completed turns -------------------------------------------------------------------


async def test_completed_voice_turn_emits_in_order_and_commits_before_the_next_turn():
    pause = Pause()
    h = make_harness([Token("Hello there."), pause], [Token("Again.")])
    h.start_voice()
    await h.speak("hello", partials=["hello th"])
    pause.release()
    await h.settle()

    names = h.outbound.names()
    assert subsequence(
        names,
        [
            "vad_status:speech_started",
            "state_cycle:listen",
            "voice_decision:endpoint",
            "vad_status:speech_ended",
            "transcription",
            "state_cycle:think",
            "stream_start",
            "stream_token",
            "0x01",
            "stream_complete",
            "0x02",
        ],
    )
    assert "state_cycle:idle" not in names and "turn_metrics" not in names
    assert h.responder.commits == []
    (start,) = h.outbound.of("stream_start")
    assert {f.turn_tag for f in frames(h, MsgType.AUDIO_CHUNK)} == {start.audio_tag}

    await finish_voice_turn(h, start.turn_id)

    assert h.outbound.names()[-2:] == ["state_cycle:idle", "turn_metrics"]
    assert h.responder.commits[0].outcome == "completed"
    assert h.responder.commits[0].delivered_text == "Hello there."
    await h.speak("again")
    assert h.responder.log.index("commit:completed") < h.responder.log.index("open", 1)
    assert h.outbound.violations == []


async def test_typed_turn_sends_no_binary_frame_and_commits_at_completion():
    h = make_harness([Token("Hi."), Complete("Hi.")])
    h.post(Text("hello"))
    await h.settle()

    assert not [i for i in h.outbound.client if isinstance(i, Frame)]
    assert h.outbound.names() == [
        "state_cycle:think",
        "stream_start",
        "stream_token",
        "stream_complete",
        "state_cycle:idle",
        "turn_metrics",
    ]
    assert h.responder.commits[0].delivered_text == "Hi."


async def test_ingress_drops_and_overloads_become_their_decisions():
    h = make_harness()
    h.start_voice()
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.dropped(uid, "noise"), h.ingress.overload("vad"))
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("noise", uid)
    (overload,) = h.outbound.of("voice_decision:overload")
    assert (overload.detail.lane, overload.detail.dropped_ms) == ("vad", 120)


async def test_ingress_events_of_another_epoch_are_dropped_and_counted():
    h = make_harness()
    h.start_voice()
    stale = RecordingIngress(h.ids, epoch=9)
    h.post(stale.started(h.ids()))
    await h.settle()

    assert h.outbound.client == []
    assert h.controller.stale_events == 1


# -- Cancellation ------------------------------------------------------------------------


class TokenRecordingSynth(FakeSynthesizer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokens = []

    def synthesize(self, segment, token):
        self.tokens.append(token)
        return super().synthesize(segment, token)


async def test_keyword_interrupt_while_speaking_cuts_before_the_next_token():
    pause = Pause()
    synth = TokenRecordingSynth()
    # Two segments: the pacer holds a segment's last frame until the next segment starts, so
    # the first sentence is fully queued, and can be heard, once the second has begun.
    h = make_harness([Token("One two three. Four five. "), pause, Token("Six.")], synthesizer=synth)
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    assert h.controller.phase == "speaking"
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=700))
    other = h.ids()
    h.post(h.ingress.started(other), h.ingress.partial(other, "stop", heard_at_ms=40))
    await h.settle()

    assert h.responder.closed == [0]
    assert "yield:Token:Six." not in h.responder.log
    assert synth.tokens and all(token.is_set() for token in synth.tokens)
    assert h.outbound.purges[0][0] == turn_id
    names = h.outbound.names()
    assert subsequence(names, ["voice_decision:barge_in", "0x03", "playback_stop"])
    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.detail.keyword, barge.detail.utterance_id) == (
        "keyword",
        "stop",
        other,
    )
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.reason == "barge_in"
    assert cancelled.partial_text == "One two three. Four five. "
    assert cancelled.spoken_text == "One two three."
    assert h.responder.commits[0].outcome == "interrupted"
    assert h.responder.commits[0].delivered_text == "One two three."
    assert h.controller.phase == "idle"
    assert h.outbound.violations == []
    pause.release()
    await h.settle()
    assert "yield:Token:Six." not in h.responder.log


async def test_endpoint_report_interrupts_only_while_speaking():
    pause = Pause()
    h = make_harness([pause, Token("One. "), Pause()])
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    h.post(BargeInReport(turn_id=turn_id, keyword="stop"))
    await h.settle()
    assert h.responder.commits == []

    pause.release()
    await h.settle()
    assert h.controller.phase == "speaking"
    h.post(BargeInReport(turn_id=turn_id, keyword="stop"))
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.detail.phase, barge.detail.utterance_id) == (
        "endpoint_kws",
        "speaking",
        None,
    )
    assert h.responder.commits[0].outcome == "interrupted"


async def test_task_cancel_while_speaking_commits_once_and_reraises():
    h = make_harness([Token("One. "), Pause()])
    h.start_voice()
    await h.speak("hello")
    assert h.controller.phase == "speaking"

    h.controller.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await h.controller.task

    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    assert h.responder.closed == [0]


async def test_close_callers_cancel_while_it_waits_commits_once_and_reraises():
    h = make_harness([Token("One. "), Call(tool_task), Pause()])
    h.start_voice()
    await h.speak("hello")
    closer = asyncio.create_task(h.controller.close())
    await h.settle()
    assert h.responder.commits == []  # the teardown waits on the tool's grace

    closer.cancel()
    await h.settle()
    await h.clock.advance(TurnConfig().tool_cancel_grace_ms)

    with pytest.raises(asyncio.CancelledError):
        await closer
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    assert h.responder.commits[0].tool_effect_unknown is True


# -- The commit rule ----------------------------------------------------------------------


async def test_a_shown_turn_cut_before_any_delivery_commits_an_empty_response():
    pause = Pause()
    h = make_harness([pause, Token("Late.")])
    h.post(Text("hello"))
    await h.settle()
    h.post(Interrupt())
    await h.settle()

    (delivery,) = h.responder.commits
    assert (delivery.outcome, delivery.delivered_text) == ("interrupted", "")
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.reason, cancelled.spoken_text) == ("user_interrupt", None)


async def test_a_second_commit_request_for_a_turn_is_a_no_op():
    h = make_harness([Token("One. "), Pause()])
    h.post(Text("hello"))
    await h.settle()
    h.post(Interrupt(), Interrupt(), Close())
    await h.settle()

    assert len(h.responder.commits) == 1


async def test_a_cancellation_posted_during_teardown_interrupts_neither_wait_nor_commit():
    h = make_harness([Token("One. "), Call(tool_task), Pause()])
    h.post(Text("hello"))
    await h.settle()
    h.post(Interrupt())
    await h.settle()
    h.post(Interrupt(), Close())
    await h.clock.advance(500)
    assert h.responder.commits == []

    await h.clock.advance(TurnConfig().tool_cancel_grace_ms - 500)

    assert len(h.responder.commits) == 1
    assert h.responder.commit_times == [TurnConfig().tool_cancel_grace_ms]
    assert h.responder.commits[0].tool_effect_unknown is True


async def test_event_store_disabled_sends_no_record_failed():
    h = make_harness([Token("Hi.")], commit_result=CommitResult(False, None, None))
    h.post(Text("hello"))
    await h.settle()

    assert h.outbound.of("voice_decision:degraded") == []
    (m,) = metrics(h)
    assert m.recorded is False


async def test_failed_record_sends_degraded_record_failed():
    h = make_harness([Token("Hi.")], commit_result=CommitResult(False, None, "disk full"))
    h.post(Text("hello"))
    await h.settle()

    (degraded,) = h.outbound.of("voice_decision:degraded")
    assert degraded.reason == "record_failed"
    assert subsequence(h.outbound.names(), ["stream_start", "voice_decision:degraded"])
    (m,) = metrics(h)
    assert m.recorded is False


async def test_error_tokens_are_shown_not_spoken_and_degrade_turn_error():
    pause = Pause()
    h = make_harness([Token("Fine. "), Token("I encountered an error: x", is_error=True), pause])
    h.start_voice()
    await h.speak("hello")
    pause.release()
    await h.settle()

    assert [s.text for s in h.synth.segments] == ["Fine."]
    assert h.outbound.tokens() == ["Fine. ", "I encountered an error: x"]
    (degraded,) = h.outbound.of("voice_decision:degraded")
    assert (degraded.reason, degraded.detail.voiced_text) == ("turn_error", None)


# -- The interruption note ------------------------------------------------------------------


async def interrupted_voice_turn(h: Harness, *, played_ms: int | None) -> None:
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    if played_ms is not None:
        h.post(PlaybackProgress(turn_id=turn_id, played_ms=played_ms))
    h.post(Interrupt())
    await h.settle()


async def test_note_after_a_tts_turn_cut_mid_reply_is_mid_reply():
    h = make_harness([Token("One two three. Four five. "), Pause()], [Token("Next.")])
    await interrupted_voice_turn(h, played_ms=700)
    assert h.responder.commits[0].delivered_text == "One two three."
    await h.speak("again")

    assert h.responder.requests[1].interruption_note == "mid_reply"
    assert h.responder.requests[1].user_text == "again"


async def test_note_after_a_tts_turn_cut_before_any_audio_is_before_reply():
    h = make_harness([Pause()], [Token("Next.")])
    await interrupted_voice_turn(h, played_ms=None)
    assert h.responder.commits[0].delivered_text == ""
    await h.speak("again")

    assert h.responder.requests[1].interruption_note == "before_reply"


async def test_note_after_a_text_turn_cut_mid_reply_is_mid_reply():
    h = make_harness([Token("Partial "), Pause()], [Token("Next.")])
    h.post(Text("hello"))
    await h.settle()
    h.post(Interrupt(), Text("again"))
    await h.settle()

    assert h.responder.commits[0].delivered_text == "Partial "
    assert h.responder.requests[1].interruption_note == "mid_reply"


async def test_note_after_a_text_turn_cut_before_any_token_is_before_reply():
    h = make_harness([Pause()], [Token("Next.")])
    h.post(Text("hello"))
    await h.settle()
    h.post(Interrupt(), Text("again"))
    await h.settle()

    assert h.responder.requests[1].interruption_note == "before_reply"


async def test_a_completed_commit_clears_the_note():
    h = make_harness([Token("Part "), Pause()], [Token("Done.")], [Token("Third.")])
    h.post(Text("one"))
    await h.settle()
    h.post(Interrupt(), Text("two"))
    await h.settle()
    h.post(Text("three"))
    await h.settle()

    notes = [r.interruption_note for r in h.responder.requests]
    assert notes == [None, "mid_reply", None]
    assert [r.user_text for r in h.responder.requests] == ["one", "two", "three"]


# -- turn_metrics -----------------------------------------------------------------------------


T1_VECTOR_TIMINGS = {
    "eot_decided": 240,
    "stt_final": 520,
    "context_built": 530,
    "request_sent": 535,
    "first_token": 780,
    "first_segment": 895,
    "first_tts_chunk": 1430,
    "first_frame_sent": 1440,
    "first_played": 1530,
}


async def test_the_t1_vector_timings_give_its_over_budget_list():
    over = judge_over_budget(T1_VECTOR_TIMINGS, BUDGET_PRESETS["floor"], tool=False)
    assert over == ["stt_final_ms", "context_build_ms", "tts_first_chunk_ms", "total_p50_ms"]


async def test_metrics_carry_the_design_keys_and_the_scopes_prompt_stats():
    stats = PromptStats(1800, 640, 220, 2700, 2400, False, "step")

    def set_stats(scope) -> None:
        scope.prompt_stats = stats

    pause = Pause()
    h = make_harness([Call(set_stats), Token("Hello there."), pause])
    h.start_voice()
    await h.speak("hello", heard_at_ms=0)
    turn_id = h.last_turn()
    h.post(PlaybackProgress(turn_id="turn-of-someone-else", played_ms=10))
    pause.release()
    await h.settle()
    await finish_voice_turn(h, turn_id)

    (m,) = metrics(h)
    assert (m.turn_id, m.origin, m.channel, m.kind, m.outcome) == (
        turn_id,
        "voice",
        "tts",
        "plain",
        "completed",
    )
    assert m.recorded is True
    assert {"stt_final", "request_sent", "first_token", "first_segment"} <= set(m.timings)
    assert {"first_tts_chunk", "first_frame_sent", "first_played"} <= set(m.timings)
    assert m.budget.preset == "floor"
    assert m.budget.model_extra["tts_first_chunk_ms"] == 500
    assert m.prompt.prefix_tokens_est == 1800
    assert m.prompt.cached_tokens == 2400
    assert m.prompt.history_compacted is True
    assert m.stale_events == 1
    assert m.ledger_source == "client"
    assert m.engines.stt == "cuda:0"


async def test_tool_turn_is_kind_tool_and_judges_tool_call_ms():
    h = make_harness([Token("Sure. "), Call(dispatch_tool), tool_started(), tool_completed()])
    h.post(Text("set a timer"))
    await h.settle()

    (m,) = metrics(h)
    assert m.kind == "tool"
    assert {"tool_call_parsed", "tool_done"} <= set(m.timings)
    assert judge_over_budget(
        {"first_token": 0, "tool_call_parsed": 451}, BUDGET_PRESETS["floor"], tool=True
    ) == ["tool_call_ms"]
    assert "tool_call_ms" not in judge_over_budget(
        {"first_token": 0, "tool_call_parsed": 451}, BUDGET_PRESETS["floor"], tool=False
    )


async def test_keyword_interrupted_turn_carries_keyword_heard_and_playback_stopped():
    h = make_harness([Token("One two three. "), Pause()])
    h.start_voice()
    await h.speak("hello", heard_at_ms=0)
    await h.clock.advance(1000)
    other = h.ids()
    h.post(h.ingress.started(other), h.ingress.partial(other, "stop", heard_at_ms=200))
    await h.settle()

    (m,) = metrics(h)
    fade = h.outbound.of("playback_stop")[0].fade_ms
    assert m.timings["keyword_heard"] == 200
    assert m.timings["playback_stopped"] == 1000 + fade
    assert "barge_in_reaction_ms" in m.over_budget  # 830 ms against 700


async def test_a_plain_completed_turn_judges_neither_tool_nor_barge_in_rows():
    h = make_harness([Token("Hi.")])
    h.post(Text("hello"))
    await h.settle()

    (m,) = metrics(h)
    assert not {"keyword_heard", "playback_stopped", "tool_call_parsed"} & set(m.timings)
    assert not {"tool_call_ms", "barge_in_reaction_ms"} & set(m.over_budget)


async def test_stale_playback_reports_are_dropped_and_counted():
    pause = Pause()
    h = make_harness([Token("Hi. "), pause])
    h.start_voice()
    await h.speak("hello")
    h.post(PlaybackDone(turn_id="turn-stale", played_ms=5))
    pause.release()
    await h.settle()

    assert h.responder.commits == []  # the stale done completed nothing
    await finish_voice_turn(h, h.last_turn())
    assert metrics(h)[0].stale_events == 1


# -- Placement --------------------------------------------------------------------------------


def slow_first_chunk(h: Harness, ms: int) -> None:
    def advance(seconds: float) -> None:
        h.clock.now += int(seconds * 1000)

    h.synth.sleep = advance
    h.synth.chunk_delay_s = ms / 1000


YIELD_BUDGET = dataclasses.replace(
    BUDGET_PRESETS["floor"], preset="floor_yield", tts_first_chunk_ms=1200
)


async def run_slow_voice_turn(h: Harness) -> object:
    pause = Pause()
    h.responder.add(Token("Hi there."), pause)
    slow_first_chunk(h, 800)
    h.start_voice()
    await h.speak("hello")
    pause.release()
    await h.settle()
    await finish_voice_turn(h, h.last_turn())
    return metrics(h)[0]


async def test_a_yield_placement_judges_against_its_own_budget():
    calls = []

    def provider() -> TurnPlacement:
        calls.append(1)
        return TurnPlacement(mode="yield", stt="cpu", tts="cpu", budget=YIELD_BUDGET)

    m = await run_slow_voice_turn(make_harness(placement=provider))

    assert len(calls) == 1
    assert (m.engines.stt, m.engines.tts, m.budget.preset) == ("cpu", "cpu", "floor_yield")
    assert m.timings["first_tts_chunk"] - m.timings["first_segment"] == 800
    assert "tts_first_chunk_ms" not in m.over_budget


async def test_the_same_turn_on_the_floor_budget_is_over():
    m = await run_slow_voice_turn(make_harness())
    assert "tts_first_chunk_ms" in m.over_budget


async def test_a_placement_change_mid_turn_does_not_change_that_turns_metrics():
    answers = [
        TurnPlacement(mode="yield", stt="cpu", tts="cpu", budget=YIELD_BUDGET),
        TurnPlacement(mode="primary", stt="cuda:0", tts="cuda:0", budget=BUDGET_PRESETS["floor"]),
    ]
    calls = []

    def provider() -> TurnPlacement:
        calls.append(1)
        return answers[min(len(calls) - 1, 1)]

    m = await run_slow_voice_turn(make_harness(placement=provider))
    assert len(calls) == 1
    assert (m.engines.tts, m.budget.preset) == ("cpu", "floor_yield")


# -- Prefetch -----------------------------------------------------------------------------------


async def test_prefetch_on_speech_start_each_partial_and_each_endpoint_candidate():
    h = make_harness([Pause()])
    h.start_voice()
    uid = h.ids()
    h.post(
        h.ingress.started(uid),
        h.ingress.partial(uid, "what"),
        h.ingress.partial(uid, "what time"),
        h.ingress.candidate(uid, "what time is it"),
    )
    await h.settle()

    assert h.responder.prefetches == [
        ("session-1", None),
        ("session-1", "what"),
        ("session-1", "what time"),
        ("session-1", "what time is it"),
    ]


# -- The scene seam -------------------------------------------------------------------------------


async def test_null_scene_port_sends_nothing_and_is_the_default():
    port = NullScenePort()
    assert isinstance(port, ScenePort)
    # PlaybackListener is not runtime_checkable: check its hooks structurally.
    for hook in ("segment_starting", "stream_ending"):
        assert hasattr(PlaybackListener, hook)
        assert callable(getattr(port, hook))
    assert port.phase("idle", turn_id=None, at=Immediate()) is None
    assert port.tool_view("t", "c", "timers", None, at=Immediate()) is None
    assert port.cancel_turn("t") is None
    assert port.turn_ended("t") is None
    assert port.segment_starting(1, 0, 0) is None
    assert port.stream_ending(1, 0) is None

    h = make_harness([Token("Hi."), Pause()], start=False)
    kwargs = {key: value for key, value in h.kwargs.items() if key != "scene"}
    controller = TurnController(**kwargs)
    controller.start()
    controller.post(Text("hello"))
    await settle()
    assert isinstance(controller.scene, NullScenePort)
    assert h.outbound.of("stream_start")
    await controller.close()


async def test_voice_turn_scene_anchors():
    pause = Pause()
    h = make_harness([Token("Hello there."), pause])
    h.start_voice()
    uid = h.ids()
    h.post(h.ingress.started(uid))
    await h.settle()
    assert h.scene.phases() == [("phase", "listening", None, Immediate())]

    seen_at_thinking: list[str] = []
    h.scene.on_call = lambda call: (
        seen_at_thinking.extend(h.outbound.names()) if call[1] == "thinking" else None
    )
    h.post(h.ingress.ended(uid, "hello"))
    await h.settle()
    turn_id = h.last_turn()
    tag = h.outbound.of("stream_start")[0].audio_tag

    assert "stream_start" in seen_at_thinking
    assert h.scene.phases()[1:] == [
        ("phase", "thinking", turn_id, Immediate()),
        ("phase", "speaking", turn_id, AtSegment(tag, 0)),
        ("phase", "idle", turn_id, AtStreamEnd(tag)),
    ]
    assert ("segment_starting", tag, 0, 0) in h.scene.calls  # the port is the pacer's listener

    pause.release()
    await h.settle()
    ending = h.scene.calls.index(("stream_ending", tag, h.scene.calls[-2][2]))
    assert h.scene.calls.count(("turn_ended", turn_id)) == 1
    assert h.scene.calls.index(("turn_ended", turn_id)) > ending
    await finish_voice_turn(h, turn_id)
    assert h.scene.calls.count(("turn_ended", turn_id)) == 1


async def test_text_turn_scene_anchors_are_immediate():
    h = make_harness([Token("Hi.")])
    h.post(Text("hello"))
    await h.settle()
    turn_id = h.last_turn()

    assert h.scene.phases() == [
        ("phase", "thinking", turn_id, Immediate()),
        ("phase", "idle", turn_id, Immediate()),
    ]
    assert h.scene.calls[-1] == ("turn_ended", turn_id)


async def test_a_cut_cancels_the_turns_scene_intents_in_the_purge_step():
    h = make_harness([Token("One two. "), Pause()])
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    purges_at_cancel: list[int] = []
    h.scene.on_call = lambda call: (
        purges_at_cancel.append(len(h.outbound.purges)) if call[0] == "cancel_turn" else None
    )
    h.post(Interrupt())
    await h.settle()

    assert purges_at_cancel == [1]
    after = h.scene.calls[h.scene.calls.index(("cancel_turn", turn_id)) + 1 :]
    assert all(turn_id not in call for call in after)
    assert ("phase", "idle", None, Immediate()) in after
    assert ("turn_ended", turn_id) not in h.scene.calls


async def test_tool_view_is_anchored_to_the_next_segment():
    pause = Pause()
    h = make_harness(
        [
            Token("Setting it. "),
            tool_started(),
            tool_completed(view="timers"),
            Token("Done."),
            pause,
        ]
    )
    h.start_voice()
    await h.speak("set a timer")
    turn_id = h.last_turn()
    tag = h.outbound.of("stream_start")[0].audio_tag

    views = [c for c in h.scene.calls if c[0] == "tool_view"]
    assert views == [
        ("tool_view", turn_id, "call-1", "timers", {"label": "pasta"}, AtSegment(tag, 1))
    ]


async def test_tool_view_on_the_text_channel_is_immediate():
    h = make_harness([Token("Ok. "), tool_started(), tool_completed(view="timers")])
    h.post(Text("set a timer"))
    await h.settle()

    views = [c for c in h.scene.calls if c[0] == "tool_view"]
    assert views[0][-1] == Immediate()


# -- StateHint ------------------------------------------------------------------------------------


async def run_voice_turns(h: Harness, count: int = 1) -> None:
    h.start_voice()
    for _ in range(count):
        await h.speak("hello")
        await finish_voice_turn(h, h.last_turn())


async def test_without_a_hint_every_segment_is_neutral():
    h = make_harness([Token("One. "), Token("Two.")])
    await run_voice_turns(h)
    assert [s.state for s in h.synth.segments] == ["neutral", "neutral"]


async def test_a_hint_sets_the_state_of_later_segments_only_and_not_the_next_turn():
    pause = Pause()
    h = make_harness(
        [StateHint("sleepy"), Token("One. "), pause, StateHint("neutral"), Token("Two.")],
        [Token("Three.")],
    )
    h.start_voice()
    await h.speak("hello")
    assert [s.state for s in h.synth.segments] == ["sleepy"]
    pause.release()
    await h.settle()
    await finish_voice_turn(h, h.last_turn())
    await h.speak("again")

    assert [(s.text, s.state) for s in h.synth.segments] == [
        ("One.", "sleepy"),
        ("Two.", "neutral"),
        ("Three.", "neutral"),
    ]


async def test_a_hint_outside_the_vocabulary_warns_once_without_text(caplog):
    caplog.set_level(logging.DEBUG)
    h = make_harness([StateHint("grumpy"), Token("Secret words. "), StateHint("grumpy")])
    await run_voice_turns(h)

    warnings = [r for r in caplog.records if r.name == TURN_LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Secret" not in warnings[0].getMessage()
    assert [s.state for s in h.synth.segments] == ["neutral"]
    assert h.responder.commits[0].outcome == "completed"


async def test_an_engine_without_the_state_voices_neutral_and_one_info_counts_it(caplog):
    caplog.set_level(logging.INFO)
    synth = FakeSynthesizer(states=frozenset({"neutral"}))
    h = make_harness(
        [StateHint("sleepy"), Token("Quiet words. "), Token("More.")], synthesizer=synth
    )
    await run_voice_turns(h)

    infos = [r for r in caplog.records if r.name == TURN_LOGGER and r.levelno == logging.INFO]
    assert len(infos) == 1
    message = infos[0].getMessage()
    assert "requested=2" in message and "voiced=0" in message and "engine=fake" in message
    assert "Quiet" not in message and "More" not in message
    assert h.responder.commits[0].outcome == "completed"


# -- HandOff ------------------------------------------------------------------------------------------


async def test_handoff_ack_is_the_first_token_and_segment_and_plays_at_once():
    pause = Pause()
    h = make_harness([HandOff("One moment."), pause, Token(" Here it is.")])
    slow_first_chunk(h, 300)
    h.start_voice()
    await h.speak("look it up")
    turn_id = h.last_turn()

    assert h.outbound.tokens() == ["One moment."]
    assert h.synth.segments == [TextSegment("One moment.", 0, 11)]
    assert h.outbound.names().count("state_cycle:speak") == 1

    # The client reports the ack played; the turn waits in thinking.
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=600))
    await h.settle()
    assert h.controller.phase == "thinking"
    after_speak = h.outbound.names()[h.outbound.names().index("state_cycle:speak") :]
    assert "state_cycle:think" in after_speak

    pause.release()
    await h.settle()
    await finish_voice_turn(h, turn_id)

    names = h.outbound.names()
    first_speak = names.index("state_cycle:speak")
    assert subsequence(
        names[first_speak:], ["state_cycle:speak", "state_cycle:think", "state_cycle:speak"]
    )
    assert h.outbound.tokens() == ["One moment.", " Here it is."]
    assert h.responder.commits[0].delivered_text == "One moment. Here it is."
    (m,) = metrics(h)
    assert m.timings["handoff_ack_ms"] == 300


async def test_handoff_after_a_token_or_a_second_handoff_warns_and_changes_nothing(caplog):
    caplog.set_level(logging.WARNING)
    h = make_harness(
        [HandOff("One moment."), HandOff("Again."), Token(" Done.")],
        [Token("First. "), HandOff("Late.")],
    )
    h.post(Text("one"))
    await h.settle()
    h.post(Text("two"))
    await h.settle()

    warnings = [r for r in caplog.records if r.name == TURN_LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert not any("Again" in r.getMessage() or "Late" in r.getMessage() for r in warnings)
    assert h.outbound.tokens() == ["One moment.", " Done.", "First. "]


async def test_a_turn_without_handoff_has_no_handoff_ack_key():
    h = make_harness([Token("Hi.")])
    await run_voice_turns(h)
    assert "handoff_ack_ms" not in metrics(h)[0].timings


# -- The shared fakes -------------------------------------------------------------------------------


async def test_fake_arbiter_grants_a_solo_claim_before_claim_returns():
    verdicts: list[bool] = []
    arbiter = FakeWakeArbiter(solo=True)
    rank = WakeRank(arrival=0.0, false_accepts_per_hour=None, positive_quantile=None, margin=0.1)
    claim = arbiter.claim("s", rank, verdicts.append)
    assert verdicts == [True] and claim == 1

    manual = FakeWakeArbiter()
    first = manual.claim("s", rank, verdicts.append)
    manual.deny(first)
    manual.settle(first, False)
    assert verdicts == [True, False]
    assert manual.settles == [(1, False)]
    assert WakeVerdict(claim_id=1, proceed=True).proceed is True


async def test_ingress_and_controller_ids_come_from_one_counter():
    ids = IdCounter()
    ingress = RecordingIngress(ids)
    opened = ingress.open_utterance("wake", 10)
    minted = ids()
    ingress.set_engaged(True)
    ingress.request_gate_partial(opened.utterance_id, 20)

    assert opened.utterance_id != minted
    assert ids.issued == [opened.utterance_id, minted]
    assert ("set_engaged", True) in ingress.calls
    h = make_harness()
    assert h.kwargs["new_utterance_id"] is h.ids
    assert h.ingress.ids is h.ids
