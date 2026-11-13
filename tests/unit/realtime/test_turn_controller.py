"""The per-session turn controller: its core (T11a), and cuts, purges, records and completion
(T11c).

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
from backend.realtime.ingress import UtteranceEnded
from backend.realtime.playback import LanePurge, PlaybackListener
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
    Reconfigure,
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
    FailingSynthesizer,
    FakeWakeArbiter,
    Harness,
    IdCounter,
    Pause,
    RecordingIngress,
    close_live_controllers,
    describe,
    dispatch_tool,
    make_harness,
    settle,
    tool_completed,
    tool_started,
    voice_settings,
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


# == T11c: cuts, purges, records and completion ======================================================


def is_audio(item: object) -> bool:
    return isinstance(item, Frame) and item.msg_type == MsgType.AUDIO_CHUNK


def tag_frames(h: Harness, tag: int) -> list[str]:
    """The client-facing record's frames of one tag, as `0x01`, `0x02` and `0x03`."""
    return [describe(i) for i in h.outbound.client if isinstance(i, Frame) and i.turn_tag == tag]


def assert_one_terminal(h: Harness, tag: int) -> None:
    """M47: one `0x02` or `0x03` per tag announced, and no frame of the tag after it."""
    seen = tag_frames(h, tag)
    terminals = [name for name in seen if name in ("0x02", "0x03")]
    assert len(terminals) == 1, seen
    assert seen[-1] == terminals[0], seen


def queued_start(h: Harness):
    """The last `stream_start` queued, whether or not the sender took it."""
    return [s.item for s in h.outbound.sent if describe(s.item) == "stream_start"][-1]


def take_tokens(h: Harness, count: int) -> None:
    """Take items, `control` first, until the sender has taken `count` `stream_token`s."""
    h.outbound.take_until(lambda _item: len(h.outbound.of("stream_token")) >= count)


def take_frames(h: Harness, count: int) -> None:
    """Take items until the sender has taken `count` `0x01` frames."""
    h.outbound.take_until(lambda _item: len(frames(h, MsgType.AUDIO_CHUNK)) >= count)


def keyword_cut(h: Harness) -> None:
    other = h.ids()
    h.post(h.ingress.started(other), h.ingress.partial(other, "stop"))


def degraded_tts_failed(h: Harness):
    return [d for d in h.outbound.of("voice_decision:degraded") if d.reason == "tts_failed"]


def turn_lane_names(h: Harness, turn_id: str) -> list[str]:
    """The client-facing record of one turn's `turn`-lane items, in the order taken."""
    names = []
    for item in h.outbound.client:
        if isinstance(item, Frame) or turn_id_of_item(item) == turn_id:
            names.append(describe(item))
    return names


def turn_id_of_item(item: object) -> str | None:
    if isinstance(item, dict):
        return item.get("turn_id")  # type: ignore[return-value]
    return getattr(item, "turn_id", None)


# -- The cut: the B5 order and the frameless variant (v2 :1477-1501, M62) ---------------------------


async def cut_with(h: Harness, how: str) -> None:
    if how == "ui":
        h.post(Interrupt())
    else:
        keyword_cut(h)
    await h.settle()
    h.outbound.take_all()


@pytest.mark.parametrize("how", ["ui", "keyword"])
async def test_b5_a_cut_in_speaking_purges_the_queued_last_frames_and_the_0x02(how):
    h = make_harness([Token("One two three. ")], auto_take=False)
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)
    take_frames(h, 1)
    assert h.controller.phase == "speaking"

    await cut_with(h, how)

    (purged_turn, purge) = h.outbound.purges[0]
    assert purged_turn == start.turn_id
    assert purge.terminal is True
    assert tag_frames(h, start.audio_tag) == ["0x01", "0x03"]
    names = h.outbound.names()
    assert subsequence(names, ["0x01", "0x03", "playback_stop", "stream_cancelled"])
    assert names.index("0x03") < names.index("playback_stop")
    assert "0x02" not in names
    assert h.outbound.violations == []


@pytest.mark.parametrize("how", ["ui", "keyword"])
async def test_a_cut_after_the_sender_took_the_0x02_sends_playback_stop_alone(how):
    h = make_harness([Token("One two three. ")])
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)
    assert "0x02" in tag_frames(h, start.audio_tag)  # the client still plays the tag

    await cut_with(h, how)

    assert h.outbound.purges[0][1].terminal is False
    assert tag_frames(h, start.audio_tag)[-1] == "0x02"
    assert "0x03" not in h.outbound.names()
    assert h.outbound.names().count("playback_stop") == 1
    assert_one_terminal(h, start.audio_tag)
    assert h.outbound.violations == []


@pytest.mark.parametrize("taken", [False, True])
async def test_frameless_variant_an_empty_reply_cut_in_thinking(taken):
    h = make_harness([], auto_take=False)
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)
    assert h.controller.phase == "thinking"  # waiting for `playback_done` of a frameless tag
    assert h.responder.commits == []
    if taken:
        h.outbound.take_all()
        assert tag_frames(h, start.audio_tag) == ["0x02"]

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].terminal is (not taken)
    names = h.outbound.names()
    if taken:
        assert tag_frames(h, start.audio_tag) == ["0x02"]
        assert "0x03" not in names
    else:
        assert tag_frames(h, start.audio_tag) == ["0x03"]
        assert names.index("0x03") < names.index("playback_stop")
    assert names.count("playback_stop") == 1
    assert_one_terminal(h, start.audio_tag)
    assert h.outbound.violations == []


# -- Terminal frames (v2 :1814-1824, M47) ------------------------------------------------------------


async def test_a_completed_turn_ends_its_tag_with_one_0x02():
    h = make_harness([Token("One two. "), Token("Three four.")])
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)
    await finish_voice_turn(h, start.turn_id)

    assert_one_terminal(h, start.audio_tag)
    assert tag_frames(h, start.audio_tag)[-1] == "0x02"
    assert h.responder.commits[0].outcome == "completed"


async def test_an_interrupt_in_thinking_before_any_frame_sends_0x03_with_no_0x01():
    h = make_harness([Pause(), Token("Late.")])
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)
    h.post(Interrupt())
    await h.settle()

    assert tag_frames(h, start.audio_tag) == ["0x03"]
    names = h.outbound.names()
    assert names.index("0x03") < names.index("playback_stop")
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.spoken_text == ""
    assert_one_terminal(h, start.audio_tag)
    assert h.outbound.violations == []


# -- The stream_start step (v2 :1520-1525, M69) -------------------------------------------------------


async def test_the_step_that_starts_a_turn_has_queued_its_stream_start_first():
    h = make_harness(
        [tool_started(), Token("Typed reply. "), tool_completed()],
        [Token("Spoken reply.")],
        auto_take=False,
    )
    h.start_voice()
    at_return: list[list[str]] = []
    handle = h.controller._handle

    async def recording_handle(message: object) -> None:
        await handle(message)
        if isinstance(message, Text | UtteranceEnded):
            turn_id = h.outbound.queued_turn_ids()[-1]
            at_return.append(
                [
                    describe(s.item)
                    for s in h.outbound.sent
                    if s.lane == "turn" and s.turn_id == turn_id
                ]
            )

    h.controller._handle = recording_handle  # type: ignore[method-assign]
    h.post(Text("typed"))
    await h.settle()
    await h.speak("spoken")

    assert at_return == [["stream_start"], ["stream_start"]]
    assert h.outbound.violations == []


# -- A purged stream_start (v2 :1511-1519, M68) ---------------------------------------------------------


async def test_a_cut_that_purged_the_stream_start_logs_record_failed_and_sends_no_degraded(caplog):
    caplog.set_level(logging.ERROR, logger=TURN_LOGGER)
    h = make_harness(
        [Token("One two. "), Pause()],
        auto_take=False,
        commit_result=CommitResult(False, None, "disk full"),
    )
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].stream_start is True
    assert any(
        "record_failed" in r.getMessage() and turn_id in r.getMessage() for r in caplog.records
    )
    assert h.outbound.of("voice_decision:degraded") == []
    assert subsequence(
        h.outbound.names(), ["0x03", "playback_stop", "stream_cancelled", "turn_metrics"]
    )
    assert h.outbound.violations == []


async def test_a_cut_after_the_stream_start_was_taken_sends_record_failed_after_it():
    h = make_harness(
        [Token("One two. "), Pause()],
        auto_take=False,
        commit_result=CommitResult(False, None, "disk full"),
    )
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    h.outbound.take_until(lambda item: describe(item) == "stream_start")

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].stream_start is False
    (degraded,) = h.outbound.of("voice_decision:degraded")
    assert (degraded.reason, degraded.turn_id) == ("record_failed", turn_id)
    assert (degraded.detail.voiced_text, degraded.detail.text_from_token) == (None, None)
    assert subsequence(turn_lane_names(h, turn_id), ["stream_start", "voice_decision:degraded"])
    assert h.outbound.violations == []


async def test_a_typed_turn_whose_stream_start_was_purged_still_gets_stream_cancelled():
    h = make_harness([Token("Partial "), Pause()], auto_take=False)
    h.post(Text("hello"))
    await h.settle()
    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].stream_start is True
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.reason, cancelled.partial_text, cancelled.spoken_text) == (
        "user_interrupt",
        "",
        None,
    )
    assert h.responder.commits[0].delivered_text == ""


# -- The record (v2 :1526-1560, M69) ---------------------------------------------------------------------


async def failed_in_thinking(h: Harness, pause: Pause, *, tokens: int) -> str:
    """Queue the first `tokens` tokens with the TTS lane held, the sender taking them; then the
    synthesizer fails on segment 0, so the turn moves to `text` with no `0x01` sent.
    """
    h.lane.hold()
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    h.outbound.take_all()
    assert len(h.outbound.tokens(turn_id)) == tokens
    h.lane.open()
    await h.settle()
    assert h.controller.phase == "text"
    pause.release()
    await h.settle()
    return turn_id


SIX = [Token("Alpha. "), Token("Beta. "), Token("Gamma. "), Token("Delta. "), Token("Eps. ")]


async def test_record_the_reviewers_order_a_purged_degraded_records_nothing():
    pause = Pause()
    h = make_harness(
        [*SIX[:2], pause, *SIX[2:], Token("Zeta. "), Pause()],
        synthesizer=FailingSynthesizer(fail_call=0),
        auto_take=False,
    )
    turn_id = await failed_in_thinking(h, pause, tokens=2)
    queued = [describe(s.item) for s in h.outbound.sent if s.turn_id == turn_id]
    assert subsequence(queued, ["voice_decision:degraded", "0x02", "stream_token"])

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1] == LanePurge(terminal=True, stream_start=False, first_token=2)
    assert h.outbound.tokens(turn_id) == ["Alpha. ", "Beta. "]
    assert "voice_decision:degraded" not in h.outbound.names()
    assert subsequence(
        h.outbound.names(),
        [
            "stream_start",
            "stream_token",
            "stream_token",
            "0x03",
            "playback_stop",
            "stream_cancelled",
        ],
    )
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.spoken_text == ""
    (delivery,) = h.responder.commits
    assert (delivery.outcome, delivery.delivered_text) == ("interrupted", "")
    assert h.outbound.violations == []


async def test_record_with_the_degraded_0x02_and_token_k_taken_holds_token_k_alone():
    pause = Pause()
    h = make_harness(
        [*SIX[:2], pause, *SIX[2:], Token("Zeta. "), Pause()],
        synthesizer=FailingSynthesizer(fail_call=0),
        auto_take=False,
    )
    turn_id = await failed_in_thinking(h, pause, tokens=2)
    take_tokens(h, 3)
    assert subsequence(h.outbound.names(), ["voice_decision:degraded", "0x02", "stream_token"])

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].first_token == 3
    assert h.outbound.tokens(turn_id) == ["Alpha. ", "Beta. ", "Gamma. "]
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.spoken_text == ""
    assert h.responder.commits[0].delivered_text == "Gamma. "
    assert h.outbound.violations == []


async def test_record_a_typed_turn_holds_the_tokens_the_sender_took():
    h = make_harness([*SIX, Token("Zeta. "), Pause()], auto_take=False)
    h.post(Text("hello"))
    await h.settle()
    take_tokens(h, 3)

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].first_token == 3
    assert h.responder.commits[0].delivered_text == "Alpha. Beta. Gamma. "
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.partial_text == "Alpha. Beta. Gamma. "
    assert h.outbound.violations == []


async def test_record_a_disconnect_with_a_full_lane_sends_nothing_and_caps_the_spoken_prefix():
    pause = Pause()
    h = make_harness([*SIX[:3], pause, *SIX[3:], Token("Zeta.")], auto_take=False)
    h.lane.hold()
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    h.outbound.take_all()  # the stream_start and tokens 0 to 2
    h.lane.open()
    await h.settle()
    take_frames(h, 3)  # the first frames, voicing token 0
    assert h.controller.phase == "speaking"
    pause.release()
    await h.settle()  # tokens 3 to 5, later frames and the 0x02 stay queued
    await h.clock.advance(60_000)  # the estimate now covers every queued frame
    queued_before = len(h.outbound.sent)
    client_before = list(h.outbound.client)

    h.post(Close())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges == [
        (turn_id, LanePurge(terminal=True, stream_start=False, first_token=3))
    ]
    assert h.outbound.sent[queued_before:] == []
    assert h.outbound.client == client_before
    (delivery,) = h.responder.commits
    assert delivery.outcome == "interrupted"
    assert delivery.delivered_text == "Alpha. Beta. Gamma. "
    assert h.outbound.violations == []


async def test_record_a_disconnect_of_a_typed_turn_holds_the_taken_tokens():
    h = make_harness([*SIX, Token("Zeta. "), Pause()], auto_take=False)
    h.post(Text("hello"))
    await h.settle()
    take_tokens(h, 3)
    queued_before = len(h.outbound.sent)

    h.post(Close())
    await h.settle()

    assert h.outbound.purges[0][1].first_token == 3
    assert h.outbound.sent[queued_before:] == []
    assert h.responder.commits[0].delivered_text == "Alpha. Beta. Gamma. "


async def test_record_a_failure_in_speaking_cut_with_its_degraded_purged():
    pause = Pause()
    h = make_harness(
        [
            Token("One two three four five six. "),
            Token("Seven eight. "),
            pause,
            Token("Nine. "),
            Token("Ten. "),
            Pause(),
        ],
        synthesizer=FailingSynthesizer(fail_call=1),
        auto_take=False,
    )
    h.lane.hold()
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    h.outbound.take_all()  # the stream_start and tokens 0 and 1
    h.lane.open()
    await h.settle()  # segment 0 is voiced, segment 1 fails
    (degraded,) = (s.item for s in h.outbound.sent if describe(s.item) == "voice_decision:degraded")
    assert degraded.detail.voiced_text == "One two three four five six."
    assert degraded.detail.text_from_token == 2
    pause.release()
    await h.settle()  # tokens 2 and 3 queue behind the degraded
    take_frames(h, 12)
    assert h.controller.phase == "speaking"
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=480))  # the twelve frames taken

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].first_token == 2
    assert "voice_decision:degraded" not in h.outbound.names()
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.spoken_text == "One two"
    assert len(cancelled.spoken_text) < len(degraded.detail.voiced_text)
    assert h.responder.commits[0].delivered_text == "One two"
    assert h.outbound.violations == []


async def test_record_the_cap_with_no_reports_stops_at_the_last_taken_token():
    h = make_harness([*SIX, Token("Zeta. "), Pause()], auto_take=False)
    h.lane.hold()
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    take_tokens(h, 3)  # the stream_start and tokens 0 to 2 only
    h.lane.open()
    await h.settle()  # then the frames voicing tokens 0 to 5
    await h.clock.advance(60_000)
    ledger = h.controller._turn.ledger
    assert ledger.source == "estimate"
    # The pacer holds the last segment's final frame until the end, so the estimate stops
    # short of token 5; it still runs past the tokens the sender took.
    assert ledger.spoken_text().startswith("Alpha. Beta. Gamma. Delta")

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0][1].first_token == 3
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.spoken_text == "Alpha. Beta. Gamma. "
    assert h.responder.commits[0].delivered_text == "Alpha. Beta. Gamma. "
    assert turn_id in h.outbound.turn_ids()
    assert h.outbound.violations == []


# -- tts_failed (v2 :1561-1590, M65 and M68; :1838-1843) ---------------------------------------------


def assert_degraded_in_stream_order(h: Harness, turn_id: str) -> object:
    """M68: the `degraded` follows exactly `text_from_token` tokens, before the next one, its
    `voiced_text` a prefix of their text, and the tag's frameless `0x02` comes after it.
    """
    names = turn_lane_names(h, turn_id)
    (degraded,) = degraded_tts_failed(h)
    at = names.index("voice_decision:degraded")
    assert names.index("stream_start") < at
    assert names[:at].count("stream_token") == degraded.detail.text_from_token
    before = "".join(h.outbound.tokens(turn_id)[: degraded.detail.text_from_token])
    assert before.startswith(degraded.detail.voiced_text)
    return degraded


async def test_tts_failed_in_thinking_before_any_chunk_moves_the_turn_to_text():
    pause = Pause()
    h = make_harness(
        [Token("Alpha. "), Token("Beta. "), pause, Token("Gamma. "), Pause()],
        synthesizer=FailingSynthesizer(fail_call=0),
    )
    turn_id = await failed_in_thinking(h, pause, tokens=2)
    start = queued_start(h)

    degraded = assert_degraded_in_stream_order(h, turn_id)
    assert (degraded.detail.voiced_text, degraded.detail.text_from_token) == ("", 2)
    assert degraded.detail.engine == "fake"
    assert tag_frames(h, start.audio_tag) == ["0x02"]
    names = turn_lane_names(h, turn_id)
    assert names.index("voice_decision:degraded") < names.index("0x02")
    assert h.outbound.tokens(turn_id) == ["Alpha. ", "Beta. ", "Gamma. "]
    assert h.synth.calls == 1  # the tokens after the switch stream as text only
    assert h.controller.phase == "text"

    sent_before = len(h.outbound.sent)
    other = h.ids()
    h.post(h.ingress.started(other))
    await h.settle()
    assert h.outbound.sent[sent_before:] == []  # no hold, nothing at all

    h.post(Text("new words"))
    await h.settle()
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.reason, cancelled.spoken_text) == ("new_input", "")
    assert h.responder.commits[0].outcome == "interrupted"
    assert h.responder.commits[0].delivered_text == "Gamma. "
    assert h.responder.requests[1].user_text == "new words"
    assert_one_terminal(h, start.audio_tag)
    assert h.outbound.violations == []


async def test_tts_failed_with_a_chunk_the_pacer_had_not_handed_on_drops_it():
    pause = Pause()
    h = make_harness(
        [Token("Alpha. "), pause, Token("Beta.")],
        synthesizer=FailingSynthesizer(fail_call=0, after_chunks=1, chunk_ms=20),
    )
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)

    (degraded,) = degraded_tts_failed(h)
    assert (degraded.detail.voiced_text, degraded.detail.text_from_token) == ("", 1)
    assert tag_frames(h, start.audio_tag) == ["0x02"]
    assert h.controller.phase == "text"
    assert h.controller._turn.ledger.recorded_samples == 0

    # M66, no client reports: the frameless tag's estimate has passed its end, so the turn
    # completes at the responder's completion with no `playback_done`.
    pause.release()
    await h.settle()
    assert h.responder.commits[0].outcome == "completed"
    assert h.responder.commits[0].delivered_text == "Beta."
    assert h.outbound.names()[-2:] == ["state_cycle:idle", "turn_metrics"]
    assert_one_terminal(h, start.audio_tag)


async def test_a_ui_interrupt_in_text_before_the_0x02_is_taken_sends_0x03_then_playback_stop():
    pause = Pause()
    h = make_harness(
        [Token("Alpha. "), pause, Pause()],
        synthesizer=FailingSynthesizer(fail_call=0),
        auto_take=False,
    )
    h.lane.hold()
    h.start_voice()
    await h.speak("hello")
    start = queued_start(h)
    h.outbound.take_all()
    h.lane.open()
    await h.settle()
    assert h.controller.phase == "text"

    h.post(Interrupt())
    await h.settle()
    h.outbound.take_all()

    assert tag_frames(h, start.audio_tag) == ["0x03"]
    names = h.outbound.names()
    assert names.index("0x03") < names.index("playback_stop")
    assert_one_terminal(h, start.audio_tag)
    assert h.outbound.violations == []


async def failed_in_speaking(h: Harness) -> str:
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    assert h.controller.phase == "speaking"
    return turn_id


def speaking_failure(*tail: object) -> Harness:
    """Segment 0 is voiced and segment 1 fails, with tokens 0 and 1 sent before the switch."""
    return make_harness(
        [Token("One two three. "), Token("Four five. "), *tail],
        synthesizer=FailingSynthesizer(fail_call=1),
    )


async def test_tts_failed_after_the_first_0x01_keeps_speaking_until_the_audio_drains():
    pause = Pause()
    h = speaking_failure(pause, Token("Six. "), Pause())
    turn_id = await failed_in_speaking(h)
    start = queued_start(h)

    degraded = assert_degraded_in_stream_order(h, turn_id)
    assert (degraded.detail.voiced_text, degraded.detail.text_from_token) == ("One two three.", 2)
    seen = tag_frames(h, start.audio_tag)
    assert seen[-1] == "0x02" and seen.count("0x01") > 1
    assert_one_terminal(h, start.audio_tag)
    assert h.controller.phase == "speaking"

    h.post(PlaybackDone(turn_id=turn_id, played_ms=10_000))
    await h.settle()
    assert h.controller.phase == "text"
    pause.release()
    await h.settle()
    names = turn_lane_names(h, turn_id)
    token_at = [i for i, name in enumerate(names) if name == "stream_token"]
    assert len(token_at) == 3
    assert token_at[1] < names.index("voice_decision:degraded") < token_at[2]

    h.post(Text("new words"))
    await h.settle()
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.reason, cancelled.spoken_text) == ("new_input", "One two three.")
    assert h.responder.commits[0].outcome == "interrupted"
    assert h.responder.commits[0].delivered_text == "One two three.Six. "
    assert h.outbound.violations == []


async def test_tts_failed_while_speaking_a_keyword_still_interrupts():
    h = speaking_failure(Pause())
    turn_id = await failed_in_speaking(h)
    assert degraded_tts_failed(h)
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=10_000))

    keyword_cut(h)
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.detail.phase) == ("keyword", "speaking")
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.reason, cancelled.spoken_text) == ("barge_in", "One two three.")
    assert h.responder.commits[0].delivered_text == "One two three."


async def test_tts_failed_in_speaking_with_the_responder_done_first_completes_at_playback_done():
    pause = Pause()
    h = speaking_failure(pause, Token("Six."))
    turn_id = await failed_in_speaking(h)
    pause.release()
    await h.settle()
    assert h.controller.phase == "speaking"
    assert h.responder.commits == []

    h.post(PlaybackDone(turn_id=turn_id, played_ms=10_000))
    await h.settle()

    (delivery,) = h.responder.commits
    assert delivery.outcome == "completed"
    assert delivery.delivered_text == "One two three.Six."
    assert h.controller.phase == "idle"
    assert h.outbound.names()[-2:] == ["state_cycle:idle", "turn_metrics"]


# -- Completion (v2 :1591-1612, M66) -------------------------------------------------------------------


async def test_completion_an_early_playback_done_waits_for_the_generation():
    pause = Pause()
    h = make_harness(
        [Token("Alpha. "), Token("Beta. "), pause, Token("Gamma.")],
        synthesizer=FailingSynthesizer(fail_call=0),
    )
    h.lane.hold()
    h.start_voice()
    await h.speak("hello")
    turn_id = queued_start(h).turn_id
    h.lane.open()
    await h.settle()
    assert "0x02" in h.outbound.names()  # the sender took the frameless 0x02

    h.post(PlaybackDone(turn_id=turn_id, played_ms=0))
    h.post(Reconfigure(settings=voice_settings(), refusal=None))
    await h.settle()
    assert h.controller.phase == "text"
    assert h.responder.commits == []
    assert "state_cycle:idle" not in h.outbound.names()
    assert metrics(h) == []
    assert h.host.calls == []  # the reconfigure stays pending

    pause.release()
    await h.settle()
    (delivery,) = h.responder.commits
    assert (delivery.outcome, delivery.delivered_text) == ("completed", "Gamma.")
    assert h.outbound.names()[-2:] == ["state_cycle:idle", "turn_metrics"]
    assert ("phase", "idle", turn_id, Immediate()) in h.scene.calls


async def test_completion_a_drain_first_moves_to_text_and_commits_at_the_completion():
    pause = Pause()
    h = speaking_failure(pause, Token("Six."))
    turn_id = await failed_in_speaking(h)
    h.post(PlaybackDone(turn_id=turn_id, played_ms=10_000))
    await h.settle()
    assert h.controller.phase == "text"
    assert h.responder.commits == []

    pause.release()
    await h.settle()

    (delivery,) = h.responder.commits
    assert (delivery.outcome, delivery.delivered_text) == ("completed", "One two three.Six.")
    assert h.controller.phase == "idle"


# -- stream_cancelled.spoken_text (v2 :1831-1837) --------------------------------------------------------


async def test_spoken_text_is_null_for_a_voice_turn_on_the_text_channel():
    h = make_harness([Token("Hi "), Pause()], synthesizer=False)
    h.start_voice()
    await h.speak("hello")
    h.post(Interrupt())
    await h.settle()

    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.partial_text, cancelled.spoken_text) == ("Hi ", None)


# -- Tool grace (v1 :2723-2726) -----------------------------------------------------------------------------


def slow_tool(h: Harness, ms: int) -> Call:
    """Dispatch a tool that finishes after `ms` on the fake clock and records its result."""

    def start(scope) -> None:
        dispatch_tool(scope)

        async def run() -> str:
            await h.clock.sleep_ms(ms)
            scope.tool_results.append({"name": "set_timer", "result": "set", "tool_call_id": "c"})
            return "set"

        scope.tool_tasks.append(asyncio.ensure_future(run()))

    return Call(start)


def cancelled_completions(h: Harness) -> list[dict]:
    return [
        item
        for item in h.outbound.of("tool_call_completed")
        if isinstance(item, dict) and item.get("error") == "cancelled"
    ]


async def test_tool_grace_a_tool_ending_within_the_grace_records_its_real_result():
    h = make_harness()
    h.responder.add(Token("Setting it. "), slow_tool(h, 1000), tool_started(), Pause())
    h.post(Text("set a timer"))
    await h.settle()
    h.post(Interrupt())
    await h.clock.advance(500)
    assert h.responder.commits == []

    await h.clock.advance(500)

    (delivery,) = h.responder.commits
    assert delivery.tool_effect_unknown is False
    assert h.responder.commit_times == [1000]
    assert h.responder.scopes[0].tool_results[0]["result"] == "set"
    (closing,) = cancelled_completions(h)
    assert closing["tool_call_id"] == "call-1"
    assert subsequence(
        h.outbound.names(), ["stream_cancelled", "tool_call_completed", "turn_metrics"]
    )
    assert h.outbound.violations == []


async def test_tool_grace_a_tool_past_the_grace_is_committed_with_its_effect_unknown():
    h = make_harness()
    h.responder.add(Token("Setting it. "), slow_tool(h, 5000), tool_started(), Pause())
    h.post(Text("set a timer"))
    await h.settle()
    h.post(Interrupt())
    await h.clock.advance(TurnConfig().tool_cancel_grace_ms)

    (delivery,) = h.responder.commits
    assert delivery.tool_effect_unknown is True
    assert h.responder.commit_times == [TurnConfig().tool_cancel_grace_ms]
    (closing,) = cancelled_completions(h)
    assert closing["tool_call_id"] == "call-1"
    await h.clock.advance(5000)
    assert len(h.responder.commits) == 1
    assert len(cancelled_completions(h)) == 1


async def test_tool_grace_only_an_orphan_tool_call_started_is_closed():
    h = make_harness(
        [
            Token("Two tools. "),
            tool_started("call-1"),
            tool_completed("call-1"),
            tool_started("call-2"),
            Pause(),
        ]
    )
    h.post(Text("do two things"))
    await h.settle()
    h.post(Interrupt())
    await h.settle()

    assert [c["tool_call_id"] for c in cancelled_completions(h)] == ["call-2"]
    assert len(h.outbound.of("tool_call_completed")) == 2


async def test_tool_grace_a_disconnect_sends_no_orphan_closing():
    h = make_harness([Token("One. "), tool_started(), Pause()])
    h.post(Text("hello"))
    await h.settle()
    h.post(Close())
    await h.settle()

    assert cancelled_completions(h) == []
    assert len(h.responder.commits) == 1


# -- HandOff cut during its continuation (v6 :1645-1648) ------------------------------------------------


async def handoff_turn(h: Harness) -> str:
    slow_first_chunk(h, 300)
    h.start_voice()
    await h.speak("look it up")
    turn_id = h.last_turn()
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=600))  # the ack has played
    await h.settle()
    assert h.controller.phase == "thinking"
    return turn_id


async def test_a_handoff_turn_cut_during_its_continuation_keeps_the_ack_in_spoken_text():
    pause = Pause()
    h = make_harness(
        [
            HandOff("One moment."),
            pause,
            Token(" Here it is, the answer. "),
            Token("More words follow here. "),
            Pause(),
        ]
    )
    turn_id = await handoff_turn(h)
    pause.release()
    await h.settle()
    assert h.controller.phase == "speaking"
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=1500))

    h.post(Interrupt())
    await h.settle()

    (cancelled,) = h.outbound.of("stream_cancelled")
    joined = "".join(h.outbound.tokens(turn_id))
    assert cancelled.spoken_text.startswith("One moment.")
    assert len(cancelled.spoken_text) > len("One moment.")
    assert joined.startswith(cancelled.spoken_text)
    assert h.responder.commits[0].delivered_text == cancelled.spoken_text
    assert h.outbound.violations == []


async def test_a_handoff_turns_tts_failed_voiced_text_keeps_the_ack():
    pause = Pause()
    h = make_harness(
        [HandOff("One moment."), pause, Token(" Here it is. "), Pause(), Token("More.")],
        synthesizer=FailingSynthesizer(fail_call=1),
    )
    turn_id = await handoff_turn(h)
    pause.release()
    await h.settle()

    degraded = assert_degraded_in_stream_order(h, turn_id)
    joined = "".join(h.outbound.tokens(turn_id))
    assert degraded.detail.voiced_text == "One moment."
    assert degraded.detail.text_from_token == 2
    assert joined.startswith(degraded.detail.voiced_text)
