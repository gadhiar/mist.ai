"""The per-session turn controller: its core (T11a), cuts, purges, records and completion
(T11c), the cross-session `WakeArbiter` with `outranks` (T11e part 2), and "stop" while an alarm
rings with the played text the echo guard reads (T52, D8).

Every test drives the controller only through `post` and `close`, with the recording transport,
`RecordingScenePort`, the fake responder, `FakeSynthesizer` on an inline lane, and the fake clock
from `turn_fakes.py`. The client-facing record is what the recording transport's sender took.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import logging
import random
from collections.abc import Callable

import numpy as np
import pytest
import pytest_asyncio

from backend.chat.context_budget import PromptStats
from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import Complete, HandOff, StateHint, Token, WSEvent
from backend.chat.turn_scope import CommitResult
from backend.realtime import turn as turn_module
from backend.realtime.budget import (
    BUDGET_PRESETS,
    BargeInConfig,
    CharacterConfig,
    TurnConfig,
    WakeConfig,
)
from backend.realtime.character import CharacterStage
from backend.realtime.degraded import DirectResponder
from backend.realtime.errors import SpeechEngineError
from backend.realtime.ingress import UtteranceEnded
from backend.realtime.playback import LanePurge, PlaybackLedger, PlaybackListener
from backend.realtime.protocol import Frame, MsgType
from backend.realtime.scene import (
    AtSegment,
    AtStreamEnd,
    Immediate,
    NullScenePort,
    ScenePort,
)
from backend.realtime.speech import CancelToken, TextSegment
from backend.realtime.turn import (
    BargeInReport,
    Close,
    Interrupt,
    Mute,
    PlaybackDone,
    PlaybackProgress,
    Reconfigure,
    Text,
    TurnController,
    TurnPlacement,
    VoiceStart,
    VoiceStop,
    Wake,
    WakeArbiter,
    WakeRank,
    WakeVerdict,
    judge_over_budget,
    outranks,
)
from tests.fakes.speech import FakeSynthesizer

from .turn_fakes import (
    TEST_PLAYBACK,
    Call,
    FailingSynthesizer,
    FakeAlarm,
    FakeCallLater,
    FakeWakeArbiter,
    Harness,
    IdCounter,
    InlineLane,
    Pause,
    RecordingIngress,
    WaitGate,
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
    # Two segments. The first sentence's frames, its short last frame among them, are queued
    # once its own audio ends; its 14 characters voice 700 ms, so the reported 700 ms covers
    # it whole and none of the second.
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


def audio_sizes(h: Harness) -> list[int]:
    return [len(f.payload) // 2 for f in frames(h, MsgType.AUDIO_CHUNK)]


# "One moment." is 11 characters, 550 ms at 50 ms each: thirteen 40 ms frames (520 ms) and a
# short last frame of 30 ms, 720 samples.
ACK_FRAMES = [960] * 13 + [720]


async def test_a_handoff_ack_counts_as_played_only_once_its_short_last_frame_has():
    pause = Pause()
    h = make_harness([HandOff("One moment."), pause, Token(" Here it is.")])
    h.start_voice()
    await h.speak("look it up")
    turn_id = h.last_turn()
    assert audio_sizes(h) == ACK_FRAMES

    h.post(PlaybackProgress(turn_id=turn_id, played_ms=520))  # every full frame
    await h.settle()
    assert h.controller.phase == "speaking"

    h.post(PlaybackProgress(turn_id=turn_id, played_ms=550))  # and the short last one
    await h.settle()
    assert h.controller.phase == "thinking"
    pause.release()
    await h.settle()


class ReportingSynthesizer(FakeSynthesizer):
    """Runs `on_end` inside the call that finds a segment's audio exhausted."""

    def __init__(self, on_end: Callable[[], None]) -> None:
        super().__init__()
        self.on_end = on_end

    def synthesize(self, segment, token):
        chunks = super().synthesize(segment, token)

        def ending():
            yield from chunks
            self.on_end()

        return ending()


async def test_a_handoff_ack_reported_played_as_its_synthesis_ends_waits_for_its_last_frame():
    # The client reports every queued frame played inside the step that finds the ack's audio
    # exhausted, so the report reaches the mailbox before the play task has queued the short
    # last frame. The ack's synthesis must not count as done until that frame is queued.
    pause = Pause()
    reports: list[str] = []

    def report() -> None:
        if not reports:
            reports.append(h.last_turn())
            h.post(PlaybackProgress(turn_id=reports[0], played_ms=520))

    h = make_harness(
        [HandOff("One moment."), pause, Token(" Here it is.")],
        synthesizer=ReportingSynthesizer(report),
    )
    h.start_voice()
    await h.speak("look it up")

    assert reports and audio_sizes(h) == ACK_FRAMES
    assert h.controller.phase == "speaking"
    h.post(PlaybackProgress(turn_id=reports[0], played_ms=550))
    await h.settle()
    assert h.controller.phase == "thinking"
    pause.release()
    await h.settle()


async def test_a_segments_last_frame_is_queued_while_the_next_waits_on_the_lane():
    pause = Pause()
    h = make_harness([Token("Hello. "), pause, Token("World."), Pause()])
    h.start_voice()
    await h.speak("hello")
    h.lane.hold()

    pause.release()
    await h.settle()

    # "Hello." voices 300 ms: seven full frames and a short one of 20 ms.
    assert h.synth.segments == [TextSegment("Hello.", 0, 6)]
    assert audio_sizes(h) == [960] * 7 + [480]
    assert h.outbound.tokens() == ["Hello. ", "World."]
    h.lane.open()
    await h.settle()
    assert h.synth.segments[1] == TextSegment("World.", 7, 13)
    assert audio_sizes(h) == [960] * 7 + [480] + [960] * 7 + [480]


# -- The character stage (T50b) ---------------------------------------------------------------------

CHARACTER_PLAYBACK = dataclasses.replace(TEST_PLAYBACK, character=CharacterConfig())
DOWNLINK_RATE = 24000
FRAME_BYTES = 960 * 2


class ToneSynthesizer(FakeSynthesizer):
    """`FakeSynthesizer` that records each segment's chunks as the controller pulls them.

    `amplitude` rescales the fake's tone (0.3 of full scale) to another peak, so a test can
    drive the stage past its limiter's knee or keep it well below.
    """

    def __init__(self, *, amplitude: float | None = None) -> None:
        super().__init__()
        self.amplitude = amplitude
        self.emitted: list[list[bytes]] = []

    def synthesize(self, segment, token):
        chunks = super().synthesize(segment, token)
        emitted: list[bytes] = []
        self.emitted.append(emitted)

        def recorded():
            for chunk in chunks:
                if self.amplitude is not None:
                    tone = np.frombuffer(chunk.pcm16, dtype="<i2") * (self.amplitude / 0.3)
                    pcm16 = np.clip(np.rint(tone), -32768, 32767).astype("<i2").tobytes()
                    chunk = dataclasses.replace(chunk, pcm16=pcm16)
                emitted.append(chunk.pcm16)
                yield chunk

        return recorded()


def staged(
    config: CharacterConfig, emitted: list[list[bytes]], *, tails: bool = True
) -> tuple[CharacterStage, list[bytes]]:
    """The stage run offline over the same chunks: each segment's audio, then its tail."""
    stage = CharacterStage(config, sample_rate=DOWNLINK_RATE)
    segments = []
    for chunks in emitted:
        audio = b"".join(stage.process(chunk) for chunk in chunks)
        if tails:
            audio += stage.end_segment()
        segments.append(audio)
    return stage, segments


def tag_payload(h: Harness, tag: int) -> bytes:
    return b"".join(f.payload for f in frames(h, MsgType.AUDIO_CHUNK) if f.turn_tag == tag)


def segment_audio(h: Harness, tag: int) -> list[bytes]:
    """The tag's `0x01` payloads cut where the pacer started each segment.

    Every segment starts on a frame boundary, so no frame holds two segments' audio.
    """
    starts = [c[3] for c in h.scene.calls if c[0] == "segment_starting" and c[1] == tag]
    offsets, at = set(), 0
    for frame in frames(h, MsgType.AUDIO_CHUNK):
        if frame.turn_tag == tag:
            offsets.add(at)
            at += len(frame.payload) // 2
    assert set(starts) <= offsets
    payload = tag_payload(h, tag)
    bounds = [start * 2 for start in starts] + [len(payload)]
    return [payload[a:b] for a, b in zip(bounds, bounds[1:])]


def spy_stages(monkeypatch, *, on_process=None, fail_at: int | None = None) -> list:
    """Build the controller's stages as spies: each counts `process` calls and keeps its tails.

    `fail_at` makes that call of `process` (1-based) raise, and `on_process` runs after each
    call with its number.
    """
    stages: list = []

    class SpyStage(CharacterStage):
        def __init__(self, config, *, sample_rate):
            super().__init__(config, sample_rate=sample_rate)
            self.processed = 0
            self.tails: list[bytes] = []
            stages.append(self)

        def process(self, pcm16):
            self.processed += 1
            if self.processed == fail_at:
                raise ValueError("stage failure")
            out = super().process(pcm16)
            if on_process is not None:
                on_process(self.processed)
            return out

        def end_segment(self):
            tail = super().end_segment()
            self.tails.append(tail)
            return tail

    monkeypatch.setattr(turn_module, "CharacterStage", SpyStage)
    return stages


async def test_with_a_stage_each_segment_is_the_stage_run_offline_and_ends_in_its_tail(
    monkeypatch,
):
    noted: dict[int, int] = {}
    note_synthesized = PlaybackLedger.note_synthesized

    def note(self, segment_index, segment, samples):
        noted[segment_index] = samples
        note_synthesized(self, segment_index, segment, samples)

    monkeypatch.setattr(PlaybackLedger, "note_synthesized", note)
    synth = ToneSynthesizer()
    h = make_harness(
        [Token("Hello. "), Token("World.")], synthesizer=synth, playback=CHARACTER_PLAYBACK
    )
    h.start_voice()
    await h.speak("hello")
    tag = queued_start(h).audio_tag

    assert len(synth.emitted) == 2
    _, expected = staged(CharacterConfig(), synth.emitted)
    tail = CharacterStage(CharacterConfig(), sample_rate=DOWNLINK_RATE).tail_samples
    assert tail > 0
    # Each segment's tail is framed inside it: after its own audio, before the next segment's
    # first frame, and before the `0x02`.
    assert segment_audio(h, tag) == expected
    assert tag_frames(h, tag)[-1] == "0x02"
    for index, chunks in enumerate(synth.emitted):
        engine_samples = sum(len(chunk) for chunk in chunks) // 2
        assert len(expected[index]) // 2 == engine_samples + tail
        # The ledger's synthesized count for the segment includes its tail.
        assert noted[index] == engine_samples + tail


async def test_without_a_stage_every_payload_is_the_engines_byte_for_byte():
    synth = ToneSynthesizer()
    h = make_harness([Token("Hello. "), Token("World.")], synthesizer=synth)
    h.start_voice()
    await h.speak("hello")
    tag = queued_start(h).audio_tag

    assert len(synth.emitted) == 2
    assert segment_audio(h, tag) == [b"".join(chunks) for chunks in synth.emitted]


async def test_a_cut_sends_no_tail(monkeypatch):
    synth = ToneSynthesizer()
    h = make_harness(
        [Token("Hello there. "), Pause()], synthesizer=synth, playback=CHARACTER_PLAYBACK
    )
    # The lane holds the segment's second chunk, so the cut lands inside the segment.
    stages = spy_stages(monkeypatch, on_process=lambda n: h.lane.hold() if n == 1 else None)
    h.start_voice()
    await h.speak("hello")
    tag = queued_start(h).audio_tag
    (stage,) = stages
    assert stage.processed == 1

    h.post(Interrupt())
    await h.settle()
    h.lane.open()
    await h.settle()

    assert stage.processed == 1
    assert stage.tails == []
    first = CharacterStage(CharacterConfig(), sample_rate=DOWNLINK_RATE).process(
        synth.emitted[0][0]
    )
    assert tag_payload(h, tag) == first
    assert tag_frames(h, tag)[-1] == "0x03"
    assert h.outbound.violations == []


async def test_a_stage_that_raises_ends_the_turn_through_the_synthesis_error_path(monkeypatch):
    stages = spy_stages(monkeypatch, fail_at=2)
    synth = ToneSynthesizer()
    h = make_harness([Token("Hello there.")], synthesizer=synth, playback=CHARACTER_PLAYBACK)
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    tag = queued_start(h).audio_tag

    (degraded,) = degraded_tts_failed(h)
    assert degraded.detail.error == "SpeechEngineError"
    # The first chunk's frames went out before the failure, and no frame follows it.
    first = CharacterStage(CharacterConfig(), sample_rate=DOWNLINK_RATE).process(
        synth.emitted[0][0]
    )
    assert tag_payload(h, tag) == first
    names = turn_lane_names(h, turn_id)
    at = names.index("voice_decision:degraded")
    assert "0x01" not in names[at:]
    assert tag_frames(h, tag)[-1] == "0x02"
    assert stages[0].tails == []

    await finish_voice_turn(h, turn_id)
    assert [c.outcome for c in h.responder.commits] == ["completed"]
    assert h.outbound.violations == []


async def test_a_segments_tail_is_queued_while_the_next_waits_and_never_reaches_it():
    pause = Pause()
    synth = ToneSynthesizer()
    h = make_harness(
        [Token("Hello. "), pause, Token("World."), Pause()],
        synthesizer=synth,
        playback=CHARACTER_PLAYBACK,
    )
    h.start_voice()
    await h.speak("hello")
    h.lane.hold()
    pause.release()
    await h.settle()
    tag = queued_start(h).audio_tag

    # Segment 0's tail is queued while segment 1 still waits on the lane.
    assert len(synth.emitted) == 1
    _, first = staged(CharacterConfig(), synth.emitted)
    assert segment_audio(h, tag) == first

    h.lane.open()
    await h.settle()

    assert len(synth.emitted) == 2
    _, expected = staged(CharacterConfig(), synth.emitted)
    _, untailed = staged(CharacterConfig(), synth.emitted, tails=False)
    audio = segment_audio(h, tag)
    assert audio == expected
    # Segment 1's first frame is the stage's after segment 0's tail emptied the copies; run
    # straight on from segment 0, the copies would still read segment 0 there.
    assert audio[1][:FRAME_BYTES] == expected[1][:FRAME_BYTES]
    assert audio[1][:FRAME_BYTES] != untailed[1][:FRAME_BYTES]


def stage_logs(caplog, turn_id: str) -> list[tuple[int, tuple]]:
    """The end-of-reply stage lines: each one's level and arguments; none holds reply text."""
    lines = [
        r for r in caplog.records if r.name == TURN_LOGGER and "character stage" in r.getMessage()
    ]
    for record in lines:
        assert "Hello" not in record.getMessage()
        assert turn_id in record.getMessage()
    return [(record.levelno, record.args) for record in lines]


async def stage_reply(config: CharacterConfig, amplitude: float) -> tuple[Harness, ToneSynthesizer]:
    synth = ToneSynthesizer(amplitude=amplitude)
    h = make_harness(
        [Token("Hello there.")],
        synthesizer=synth,
        playback=dataclasses.replace(TEST_PLAYBACK, character=config),
    )
    h.start_voice()
    await h.speak("hello")
    assert tag_frames(h, queued_start(h).audio_tag)[-1] == "0x02"
    return h, synth


async def test_a_reply_clipped_without_the_limiter_ends_with_one_warning_of_its_count(caplog):
    caplog.set_level(logging.INFO, logger=TURN_LOGGER)
    config = CharacterConfig(limiter=None)
    h, synth = await stage_reply(config, amplitude=0.9)

    reference, _ = staged(config, synth.emitted)
    assert reference.clipped_samples > 0
    assert stage_logs(caplog, h.last_turn()) == [
        (logging.WARNING, (h.last_turn(), reference.clipped_samples))
    ]


async def test_a_reply_the_limiter_held_ends_with_one_info_line_of_its_count(caplog):
    caplog.set_level(logging.INFO, logger=TURN_LOGGER)
    config = CharacterConfig()
    h, synth = await stage_reply(config, amplitude=0.9)

    reference, _ = staged(config, synth.emitted)
    assert (reference.clipped_samples, reference.limited_samples > 0) == (0, True)
    assert stage_logs(caplog, h.last_turn()) == [
        (logging.INFO, (h.last_turn(), reference.limited_samples))
    ]


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(CharacterConfig(), id="limiter"),
        pytest.param(CharacterConfig(limiter=None), id="no-limiter"),
    ],
)
async def test_a_reply_below_the_knee_logs_no_stage_line(caplog, config):
    caplog.set_level(logging.INFO, logger=TURN_LOGGER)
    h, synth = await stage_reply(config, amplitude=0.01)

    reference, _ = staged(config, synth.emitted)
    assert (reference.clipped_samples, reference.limited_samples) == (0, 0)
    assert stage_logs(caplog, h.last_turn()) == []


# -- "Stop" while an alarm rings (D8, T52) -----------------------------------------------------------


def handled(reply: str) -> CapabilityOnlyTurn:
    """A capability-only result the host answered, shaped as `handle_capability_only` makes it."""
    return CapabilityOnlyTurn(handled=True, events=(Token(reply), Complete(final_response=reply)))


async def ringing_reply(
    alarm: FakeAlarm,
    *,
    reply: str = "One two three. Four five. ",
    aec: str = "headset",
    into_ms: int = 300,
) -> tuple[Harness, str]:
    """MIST speaks `reply` with v1's arm delay of 1000 ms, and has played `into_ms` of it."""
    h = make_harness(
        [Token(reply), Pause()], barge_in=BargeInConfig(), alarm_ringing=alarm, aec=aec
    )
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    assert h.controller.phase == "speaking"
    await h.clock.advance(into_ms)
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=into_ms))
    await h.settle()
    return h, turn_id


async def test_stop_while_ringing_300_ms_into_her_speech_cuts_her_and_stops_the_alarm():
    alarm = FakeAlarm()
    h, turn_id = await ringing_reply(alarm)
    h.responder.capability_results.append(handled("The alarm is off."))
    h.responder.on_capability = lambda _text: alarm.stop()
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, "Stop", heard_at_ms=300))
    await h.settle()

    # Cut at once, 700 ms before v1's arm delay would let a keyword through.
    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.turn_id, barge.reason, barge.detail.keyword, barge.detail.utterance_id) == (
        turn_id,
        "keyword",
        "stop",
        other,
    )
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    assert h.responder.capability_calls == []  # the whole utterance goes, at its end

    h.post(h.ingress.ended(other, "Stop."))
    await h.settle()

    assert h.responder.capability_calls == [("Stop.", "session-1")]
    assert alarm.ringing is False
    assert len(h.responder.requests) == 1  # the model opened no turn for it
    answer = h.outbound.of("stream_start")[-1]
    assert answer.turn_id != turn_id
    assert (answer.channel, answer.origin) == ("tts", "voice")
    assert h.outbound.tokens(answer.turn_id) == ["The alarm is off."]
    (transcription,) = h.outbound.of("transcription")[-1:]
    assert (transcription.text, transcription.utterance_id) == ("Stop.", other)

    await finish_voice_turn(h, answer.turn_id)
    # The capability path committed its own turn: the controller commits nothing more.
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    last = metrics(h)[-1]
    assert (last.turn_id, last.origin, last.channel, last.outcome) == (
        answer.turn_id,
        "voice",
        "tts",
        "completed",
    )
    assert h.outbound.violations == []


async def test_stop_the_alarm_while_ringing_reaches_capability_only_whole_and_opens_no_turn():
    alarm = FakeAlarm()
    h, turn_id = await ringing_reply(alarm)
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, "stop the", heard_at_ms=300))
    h.post(h.ingress.ended(other, "Stop the alarm."))
    await h.settle()

    assert h.outbound.of("voice_decision:barge_in")[0].turn_id == turn_id
    assert h.responder.capability_calls == [("Stop the alarm.", "session-1")]
    # Nothing went on as a next turn, and a None result does nothing more.
    assert len(h.responder.requests) == 1
    assert h.outbound.turn_ids() == [turn_id]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.text) == ("keyword", "Stop the alarm.")
    assert h.controller.phase == "idle"


async def test_an_unhandled_stop_sends_the_drained_payloads_and_opens_no_turn():
    alarm = FakeAlarm()
    h, turn_id = await ringing_reply(alarm)
    resolved = {"type": "action_resolved", "action_id": "a-1", "status": "superseded"}
    h.responder.capability_results.append(
        CapabilityOnlyTurn(handled=False, events=(WSEvent(resolved),))
    )
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, "stop", heard_at_ms=300))
    h.post(h.ingress.ended(other, "stop"))
    await h.settle()

    assert h.outbound.turn_ids() == [turn_id]
    assert h.outbound.of("action_resolved") == [resolved]
    assert h.outbound.of("voice_decision:dropped")[0].reason == "keyword"


async def test_a_failed_capability_call_is_logged_and_closes_as_dropped_keyword(caplog):
    alarm = FakeAlarm()
    h, turn_id = await ringing_reply(alarm)

    def fail(_text: str) -> None:
        raise RuntimeError("host down")

    h.responder.on_capability = fail
    other = h.ids()

    with caplog.at_level(logging.ERROR, logger=TURN_LOGGER):
        h.post(h.ingress.started(other), h.ingress.partial(other, "stop", heard_at_ms=300))
        h.post(h.ingress.ended(other, "stop"))
        await h.settle()

    assert "capability-only path failed with RuntimeError" in caplog.text
    assert h.outbound.turn_ids() == [turn_id]
    assert h.outbound.of("voice_decision:dropped")[0].reason == "keyword"


async def test_stop_with_no_alarm_ringing_keeps_v1_and_never_reaches_capability_only():
    alarm = FakeAlarm(ringing=False)
    h, turn_id = await ringing_reply(alarm)
    early = h.ids()

    h.post(h.ingress.started(early), h.ingress.partial(early, "stop", heard_at_ms=300))
    h.post(h.ingress.ended(early, "stop"))
    await h.settle()
    assert h.outbound.of("voice_decision:barge_in") == []  # v1's arm delay holds it back
    assert h.controller.phase == "speaking"

    await h.clock.advance(1000)
    late = h.ids()
    h.post(h.ingress.started(late), h.ingress.partial(late, "stop"), h.ingress.ended(late, "stop"))
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.turn_id, barge.detail.keyword) == (turn_id, "stop")
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert dropped.reason == "keyword"
    assert h.responder.capability_calls == []
    assert len(h.responder.requests) == 1


@pytest.mark.parametrize("ringing", [True, False])
@pytest.mark.parametrize("word", ["cancel", "never mind"])
async def test_cancel_and_never_mind_never_reach_capability_only(word, ringing):
    alarm = FakeAlarm(ringing=ringing)
    h = make_harness([Token("One two three. "), Pause()], alarm_ringing=alarm)
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, word), h.ingress.ended(other, word))
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.turn_id, barge.detail.keyword) == (turn_id, word)
    assert h.outbound.of("voice_decision:dropped")[0].reason == "keyword"
    assert h.responder.capability_calls == []
    assert alarm.ringing is ringing


async def test_stop_heard_in_her_own_speech_within_the_echo_guard_cuts_nothing():
    alarm = FakeAlarm()
    h, turn_id = await ringing_reply(alarm, reply="Stop the oven when it beeps. ", aec="browser")
    assert "Stop" in h.controller.recent_tts_text(BargeInConfig().echo_guard_ms)
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, "stop", heard_at_ms=300))
    h.post(h.ingress.ended(other, "stop"))
    await h.settle()

    assert h.outbound.of("voice_decision:barge_in") == []
    assert h.responder.commits == []
    assert h.responder.capability_calls == []
    assert h.controller.phase == "speaking"
    assert alarm.ringing is True


async def test_stop_while_ringing_and_thinking_cuts_at_once():
    alarm = FakeAlarm()
    h = make_harness([Pause(), Token("Late.")], barge_in=BargeInConfig(), alarm_ringing=alarm)
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    assert h.controller.phase == "thinking"
    other = h.ids()

    h.post(h.ingress.started(other), h.ingress.partial(other, "stop"))
    await h.settle()
    assert h.outbound.of("voice_decision:barge_in")[0].turn_id == turn_id
    assert h.controller.phase == "idle"

    h.post(h.ingress.ended(other, "stop the alarm"))
    await h.settle()
    assert h.responder.capability_calls == [("stop the alarm", "session-1")]
    assert len(h.responder.requests) == 1


async def test_stop_ending_a_held_turn_while_ringing_goes_to_capability_only():
    alarm = FakeAlarm()
    h = make_harness(alarm_ringing=alarm)
    h.responder.capability_results.append(handled("The alarm is off."))
    h.start_voice()
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.candidate(uid, "stop"))
    await h.settle()
    assert h.controller.phase == "held"

    h.post(h.ingress.ended(uid, "stop"))
    await h.settle()

    assert h.responder.capability_calls == [("stop", "session-1")]
    # The held turn was cancelled without trace; only the capability's answer reached the client.
    assert len(h.responder.requests) == 1 and h.responder.commits == []
    (start,) = h.outbound.of("stream_start")
    assert h.outbound.tokens(start.turn_id) == ["The alarm is off."]
    assert h.outbound.of("voice_decision:dropped") == []


async def test_the_direct_responder_has_no_capability_host():
    responder = DirectResponder(llm_provider=None)  # type: ignore[arg-type]

    assert await responder.handle_capability_only("stop", "session-1") is None


# -- The played text the echo guard reads (lead ruling, 17 Nov) ----------------------------------------


async def test_recent_tts_text_is_the_latest_spoken_turns_and_outlives_the_turn():
    h = make_harness([Token("First reply here. ")], [Token("Second.")])
    assert h.controller.recent_tts_text(5000) == ""
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    await h.clock.advance(400)
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=400))
    await h.settle()
    assert "First" in h.controller.recent_tts_text(5000)

    await finish_voice_turn(h, turn_id)
    assert h.controller.phase == "idle"
    assert "First" in h.controller.recent_tts_text(5000)

    # A held turn has played nothing: the turn before it stays the one the guard reads.
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.candidate(uid, "and then"))
    await h.settle()
    assert h.controller.phase == "held"
    assert "First" in h.controller.recent_tts_text(5000)


# -- The shared fakes -------------------------------------------------------------------------------


async def test_inline_lane_fails_a_job_raising_another_error_as_the_real_lane_does():
    def broken() -> None:
        raise ValueError("job failure")

    with pytest.raises(SpeechEngineError, match="worker thread died"):
        await InlineLane().run(broken, token=CancelToken())


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
    # The estimate runs past the tokens the sender took, to the end of the last segment, the
    # one token 4 completed; token 5 waits in the segmenter for the text after it.
    assert ledger.spoken_text() == "Alpha. Beta. Gamma. Delta. Eps."

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


# == T11e part 2: cross-session wake arbitration =====================================================
# `outranks` (v5 :336-351, v9 :127-130) and `WakeArbiter` (v9 :132-151) alone, with a fake
# `call_later` and a session count the test sets.


def calibrated(
    fa: float, quantile: float, *, arrival: float = 0.0, margin: float = 0.1
) -> WakeRank:
    return WakeRank(
        arrival=arrival, false_accepts_per_hour=fa, positive_quantile=quantile, margin=margin
    )


def uncalibrated(margin: float, *, arrival: float = 0.0) -> WakeRank:
    return WakeRank(
        arrival=arrival, false_accepts_per_hour=None, positive_quantile=None, margin=margin
    )


@pytest.mark.parametrize(
    ("better", "worse"),
    [
        # A calibrated rank outranks an uncalibrated one, whatever their margins and arrivals.
        (calibrated(5.0, 0.1, arrival=900.0, margin=0.01), uncalibrated(0.9, arrival=0.0)),
        # Between calibrated ranks the lower false accepts per hour wins, though its quantile,
        # margin and arrival all favour the other.
        (calibrated(0.1, 0.2, arrival=500.0, margin=0.05), calibrated(0.5, 0.9, margin=0.4)),
        # Equal false accepts: the higher positive quantile, though it arrived 120 ms later.
        (calibrated(0.1, 0.8, arrival=120.0, margin=0.05), calibrated(0.1, 0.6, margin=0.4)),
        # Equal on both scales: the earlier arrival, whatever the margins.
        (calibrated(0.1, 0.8, arrival=10.0, margin=0.01), calibrated(0.1, 0.8, arrival=20.0)),
        # Between uncalibrated ranks the larger margin wins, though it arrived later.
        (uncalibrated(0.3, arrival=200.0), uncalibrated(0.2, arrival=0.0)),
        # Equal margins: the earlier arrival.
        (uncalibrated(0.2, arrival=5.0), uncalibrated(0.2, arrival=6.0)),
    ],
    ids=[
        "calibrated-over-uncalibrated",
        "lower-false-accepts",
        "higher-quantile-on-a-false-accept-tie",
        "earlier-arrival-on-a-full-tie",
        "larger-margin-uncalibrated",
        "earlier-arrival-on-a-margin-tie",
    ],
)
async def test_outranks_follows_each_ordering(better, worse):
    assert outranks(better, worse)
    assert not outranks(worse, better)


async def test_outranks_is_false_both_ways_for_equal_ranks():
    for rank in (calibrated(0.1, 0.5, arrival=3.0), uncalibrated(0.2, arrival=3.0)):
        twin = dataclasses.replace(rank)
        assert not outranks(rank, twin)
        assert not outranks(twin, rank)


async def test_a_rank_missing_either_calibrated_scale_ranks_as_uncalibrated():
    half = WakeRank(arrival=0.0, false_accepts_per_hour=0.1, positive_quantile=None, margin=0.9)
    assert outranks(calibrated(9.0, 0.0, arrival=50.0), half)
    assert outranks(half, uncalibrated(0.5))


class Verdicts:
    """Each claim's verdicts, in the order the `verdict` callbacks were called."""

    def __init__(self) -> None:
        self.log: list[tuple[str, bool]] = []

    def to(self, name: str) -> Callable[[bool], None]:
        return lambda proceed: self.log.append((name, proceed))

    def of(self, name: str) -> list[bool]:
        return [proceed for who, proceed in self.log if who == name]


def arbiter_with(
    sessions: int = 2, window_ms: int = 250
) -> tuple[WakeArbiter, FakeCallLater, list[int]]:
    """A real arbiter, its fake `call_later`, and the session count a test may change."""
    count = [sessions]
    later = FakeCallLater()
    wake = WakeArbiter(window_ms=window_ms, sessions=lambda: count[0], call_later=later)
    return wake, later, count


async def test_one_session_with_no_claim_pending_proceeds_before_claim_returns_with_no_timer():
    wake, later, _ = arbiter_with(sessions=1)
    verdicts = Verdicts()

    claim = wake.claim("s1", uncalibrated(0.2), verdicts.to("a"))
    seen_at_return = list(verdicts.log)

    assert seen_at_return == [("a", True)]
    assert later.timers == []
    wake.settle(claim, True)
    second = wake.claim("s1", uncalibrated(0.2), verdicts.to("b"))
    assert verdicts.of("b") == [True]
    assert later.timers == []
    assert second != claim


async def test_one_session_with_a_departed_sessions_claim_pending_waits():
    wake, later, count = arbiter_with(sessions=2)
    verdicts = Verdicts()
    gone = wake.claim("s2", uncalibrated(0.2), verdicts.to("gone"))
    count[0] = 1  # session s2 left with its claim still pending

    stay = wake.claim("s1", uncalibrated(0.1), verdicts.to("stay"))

    assert verdicts.log == []
    assert len(later.timers) == 1  # it joined the window `gone` opened
    later.fire()
    assert verdicts.log == [("gone", True)]
    wake.settle(gone, False)  # the departed session's controller dropped it at disconnect
    assert verdicts.log == [("gone", True), ("stay", True)]
    wake.settle(stay, True)


async def test_one_session_with_a_departed_sessions_winner_unsettled_waits():
    wake, later, count = arbiter_with(sessions=2)
    verdicts = Verdicts()
    gone = wake.claim("s2", uncalibrated(0.2), verdicts.to("gone"))
    later.fire()
    count[0] = 1

    stay = wake.claim("s1", uncalibrated(0.9), verdicts.to("stay"))

    assert verdicts.log == [("gone", True)]
    assert len(later.timers) == 1
    wake.settle(gone, True)
    assert verdicts.of("stay") == [False]
    wake.settle(stay, False)


async def test_a_window_opens_at_the_first_claim_and_its_best_claim_proceeds_at_its_close():
    wake, later, _ = arbiter_with(sessions=2, window_ms=250)
    verdicts = Verdicts()

    wake.claim("s1", uncalibrated(0.1, arrival=0.0), verdicts.to("weak"))
    assert [delay for delay, _ in later.timers] == [0.25]
    wake.claim("s2", uncalibrated(0.3, arrival=120.0), verdicts.to("strong"))

    assert len(later.timers) == 1  # the second claim joined the open window
    assert verdicts.log == []
    later.fire()
    assert verdicts.log == [("strong", True)]  # the weak claim waits for the winner's verdict


async def test_a_claim_after_the_window_closed_waits_while_the_winner_is_unsettled():
    wake, later, _ = arbiter_with()
    verdicts = Verdicts()
    first = wake.claim("s1", uncalibrated(0.2), verdicts.to("first"))
    later.fire()

    wake.claim("s2", uncalibrated(0.9), verdicts.to("late"))

    assert verdicts.log == [("first", True)]
    assert len(later.timers) == 1  # no new window while a winner is unsettled
    wake.settle(first, False)
    assert verdicts.log == [("first", True), ("late", True)]


async def test_accepting_the_winner_sends_every_other_pending_claim_false_and_empties_it():
    wake, later, _ = arbiter_with()
    verdicts = Verdicts()
    winner = wake.claim("s1", uncalibrated(0.5), verdicts.to("winner"))
    wake.claim("s2", uncalibrated(0.2), verdicts.to("loser"))
    later.fire()
    wake.claim("s3", uncalibrated(0.1), verdicts.to("late"))

    wake.settle(winner, True)

    assert verdicts.log == [("winner", True), ("loser", False), ("late", False)]
    # Empty: the next claim opens a new window.
    wake.claim("s1", uncalibrated(0.5), verdicts.to("next"))
    assert len(later.timers) == 2
    assert verdicts.of("next") == []


async def test_rejecting_the_winner_gives_the_next_ranked_claim_its_verdict():
    wake, later, _ = arbiter_with()
    verdicts = Verdicts()
    winner = wake.claim("s1", uncalibrated(0.5), verdicts.to("winner"))
    third = wake.claim("s2", uncalibrated(0.1), verdicts.to("third"))
    second = wake.claim("s3", uncalibrated(0.3), verdicts.to("second"))
    later.fire()

    wake.settle(winner, False)
    assert verdicts.log == [("winner", True), ("second", True)]
    wake.settle(second, False)
    assert verdicts.log[-1] == ("third", True)
    wake.settle(third, False)

    # None remains, so it is empty: the next claim opens a new window.
    assert len(verdicts.log) == 3
    wake.claim("s1", uncalibrated(0.5), verdicts.to("next"))
    assert len(later.timers) == 2


async def test_withdrawing_a_waiting_claim_sends_no_verdict_to_anyone():
    wake, later, _ = arbiter_with()
    verdicts = Verdicts()
    best = wake.claim("s1", uncalibrated(0.9), verdicts.to("best"))
    other = wake.claim("s2", uncalibrated(0.1), verdicts.to("other"))

    wake.settle(best, False)  # withdrawn inside the window: dropped at mute, say
    assert verdicts.log == []
    later.fire()
    assert verdicts.log == [("other", True)]

    late = wake.claim("s1", uncalibrated(0.9), verdicts.to("late"))
    wake.settle(late, False)  # withdrawn while the winner is unsettled
    assert verdicts.log == [("other", True)]
    wake.settle(other, True)
    assert verdicts.log == [("other", True)]


async def test_withdrawing_every_claim_in_a_window_leaves_its_close_nothing_to_grant():
    wake, later, _ = arbiter_with()
    verdicts = Verdicts()
    only = wake.claim("s1", uncalibrated(0.9), verdicts.to("only"))
    wake.settle(only, False)
    later.fire()
    assert verdicts.log == []
    wake.claim("s1", uncalibrated(0.9), verdicts.to("next"))
    assert len(later.timers) == 2


async def test_settle_true_on_a_claim_that_is_not_the_winner_raises():
    wake, later, _ = arbiter_with()
    verdicts = Verdicts()
    winner = wake.claim("s1", uncalibrated(0.9), verdicts.to("winner"))
    waiting = wake.claim("s2", uncalibrated(0.1), verdicts.to("waiting"))

    with pytest.raises(ValueError):
        wake.settle(waiting, True)  # still inside the window
    later.fire()
    with pytest.raises(ValueError):
        wake.settle(waiting, True)  # waiting on the winner
    with pytest.raises(ValueError):
        wake.settle(999, True)  # never claimed
    assert verdicts.log == [("winner", True)]
    wake.settle(winner, True)
    with pytest.raises(ValueError):
        wake.settle(waiting, True)  # it lost
    with pytest.raises(ValueError):
        wake.settle(winner, True)  # already settled


async def test_settle_false_on_a_claim_it_no_longer_holds_is_a_no_op():
    """Decision 3: a controller settles a lost claim too, after its `verdict(False)`."""
    wake, later, count = arbiter_with()
    verdicts = Verdicts()
    winner = wake.claim("s1", uncalibrated(0.9), verdicts.to("winner"))
    loser = wake.claim("s2", uncalibrated(0.1), verdicts.to("loser"))
    later.fire()
    wake.settle(winner, True)
    assert verdicts.log == [("winner", True), ("loser", False)]

    wake.settle(loser, False)
    wake.settle(winner, False)
    wake.settle(999, False)

    assert verdicts.log == [("winner", True), ("loser", False)]
    # It is still empty: a lone session's next claim proceeds at once, with no timer.
    count[0] = 1
    fresh = wake.claim("s1", uncalibrated(0.9), verdicts.to("fresh"))
    assert verdicts.of("fresh") == [True]
    assert len(later.timers) == 1
    wake.settle(fresh, True)


@pytest.mark.parametrize("seed", range(50))
async def test_each_claim_gets_at_most_one_verdict_and_none_after_it_is_settled(seed):
    """Random claims, window closes, settles and session counts, against the arbiter's rules."""
    rng = random.Random(seed)
    count = [rng.choice([1, 2, 3])]
    later = FakeCallLater()
    wake = WakeArbiter(window_ms=250, sessions=lambda: count[0], call_later=later)
    got: dict[int, list[bool]] = {}
    settled: set[int] = set()
    order: list[int] = []

    def record(index: int) -> Callable[[bool], None]:
        def verdict(proceed: bool) -> None:
            assert index >= len(order) or order[index] not in settled, "a verdict after settle"
            got.setdefault(index, []).append(proceed)

        return verdict

    def unsettled_winners() -> list[int]:
        return [i for i, v in got.items() if v == [True] and order[i] not in settled]

    for _ in range(60):
        action = rng.random()
        if action < 0.35:
            rank = uncalibrated(round(rng.random(), 1), arrival=float(rng.randint(0, 3)))
            index = len(order)
            order.append(wake.claim(f"s{rng.randint(1, 3)}", rank, record(index)))
        elif action < 0.55 and later.pending:
            later.fire()
        elif action < 0.65:
            count[0] = rng.choice([1, 2, 3])
        elif order:
            index = rng.randrange(len(order))
            if order[index] in settled:
                continue
            accepted = index in unsettled_winners() and rng.random() < 0.5
            wake.settle(order[index], accepted)
            settled.add(order[index])
        assert len(unsettled_winners()) <= 1, "two unsettled winners"
        assert all(len(v) == 1 for v in got.values()), "a claim with two verdicts"
        waiting = [i for i in range(len(order)) if order[i] not in settled and i not in got]
        # No claim waits on nothing: a window is open or a winner's verdict is still to come.
        assert not waiting or unsettled_winners() or later.pending, "a claim left waiting"


# == T11e part 2: 200 seeded random interleavings (v1 :2733-2734, v2 :1155, v9 :294, :327) ===========
# Each seed drives one controller through a random sequence of every message type: ingress events
# (`NameHeard` from either spotter and the gate partial among them, and events of an old epoch),
# controls, `wake`, verdicts from `FakeWakeArbiter` granted or denied at any point (after the
# claim was settled included), playback reports, the responder's pauses and the clock. The step
# invariants are checked after every step, and the end invariants after a drain and the close.

INTERLEAVING_STEPS = 40
INTERLEAVING_SEEDS = 200
INTERLEAVING_TEXTS = (
    "mist what's up",
    "I missed the bus",
    "stop",
    "stop, mist",
    "what time is it",
    "mist, wait",
)
CLOSING_MESSAGES = ("transcription", "voice_decision:dropped")
EVERY_KIND = frozenset(
    {
        "SpeechStarted",
        "SpeechResumed",
        "TranscriptPartial",
        "TranscriptPartial(gate)",
        "EndpointCandidate",
        "UtteranceEnded",
        "UtteranceDropped",
        "IngressOverload",
        "NameHeard(server)",
        "NameHeard(endpoint)",
        "old epoch",
        "Text",
        "Interrupt",
        "VoiceStart",
        "VoiceStop",
        "Reconfigure",
        "Mute",
        "Wake",
        "WakeVerdict",
        "BargeInReport",
        "PlaybackProgress",
        "PlaybackDone",
    }
)
REACHED: dict[int, tuple[frozenset[str], int]] = {}
"""Each seed run so far: the message kinds it posted and its verdicts for settled claims."""


def named_utterance(item: object) -> str | None:
    """The utterance a client message names, at its top level or in its `detail`."""
    uid = getattr(item, "utterance_id", None)
    if uid is None:
        uid = getattr(getattr(item, "detail", None), "utterance_id", None)
    return uid if isinstance(uid, str) else None


def random_script(rng: random.Random, pauses: list[Pause]) -> list[object]:
    pause = Pause()
    kind = rng.randrange(4)
    if kind == 0:
        return [Token("One two three. "), Token("Four five.")]
    pauses.append(pause)
    if kind == 1:
        return [Token("One. "), pause, Token("Two.")]
    if kind == 2:
        return [pause, Token("Late.")]
    return [
        Token("Setting it. "),
        WaitGate(),
        Call(dispatch_tool),
        tool_started(),
        pause,
        tool_completed(),
        Token("Done."),
    ]


class Interleaving:
    """One seed's run: a controller, a model of the ingress feeding it, and the invariants."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.pauses: list[Pause] = []
        scripts = [random_script(self.rng, self.pauses) for _ in range(INTERLEAVING_STEPS)]
        self.h = make_harness(
            *scripts,
            solo=False,
            synthesizer=self.rng.random() < 0.8,
            wake=WakeConfig(followup_window_ms=self.rng.choice([0, 6000])),
        )
        self.ingresses: list[RecordingIngress] = [self.h.ingress]
        self.ingress: RecordingIngress | None = None
        self.read = 0
        self.applies = 0
        self.open_uid: str | None = None
        self.gate_asks: list[tuple[str, int]] = []
        self.failures: list[str] = []
        self.committed: list[int] = []
        self.stale_verdicts = 0
        self.trace: list[str] = []
        self._instrument()

    # -- Instrumentation ------------------------------------------------------------------------

    def _instrument(self) -> None:
        """Count commits per turn, and watch each verdict for a settled claim as it is handled."""
        responder = self.h.responder
        commit_turn = responder.commit_turn

        def commit(scope, delivery):
            self.committed.append(scope.turn_index)
            return commit_turn(scope, delivery)

        responder.commit_turn = commit  # type: ignore[method-assign]
        handlers = self.h.controller._handlers
        on_verdict = handlers[WakeVerdict]

        async def verdict(message: WakeVerdict) -> None:
            settled = any(cid == message.claim_id for cid, _ in self.h.arbiter.settles)
            before = self._footprint()
            stale = self.h.controller.stale_events
            await on_verdict(message)
            if not settled:
                return
            self.stale_verdicts += 1
            if self._footprint() != before:
                self.failures.append(f"the verdict for settled claim {message.claim_id} acted")
            if self.h.controller.stale_events != stale + 1:
                self.failures.append(f"the verdict for settled claim {message.claim_id} uncounted")

        handlers[WakeVerdict] = verdict

    def _footprint(self) -> tuple[int, int, int, int]:
        """Everything a handled message could leave: sends, scene calls, levers and settles."""
        h = self.h
        levers = sum(len(ingress.calls) for ingress in self.ingresses)
        return (len(h.outbound.sent), len(h.scene.calls), levers, len(h.arbiter.settles))

    # -- The ingress model ------------------------------------------------------------------------

    def adopt(self, ingress: RecordingIngress | None) -> None:
        self.ingress = ingress
        self.read = 0
        self.open_uid = None
        self.gate_asks = []

    def _sync(self) -> None:
        """Follow the levers the controller pulled, and a reconfigure it applied."""
        applies = [call for call in self.h.host.calls if call[0] == "apply"]
        if len(applies) > self.applies:
            self.applies = len(applies)
            start = self.h.host.apply_result
            if start is not None:
                self.adopt(start.ingress)  # type: ignore[arg-type]
        ingress = self.ingress
        if ingress is None:
            return
        for call in ingress.calls[self.read :]:
            if call[0] == "open_utterance":
                self.open_uid = str(call[3])
            elif call[0] == "drop" and call[1] == self.open_uid:
                self.open_uid = None
            elif call[0] == "request_gate_partial":
                self.gate_asks.append((str(call[1]), int(call[2])))  # type: ignore[call-overload]
        self.read = len(ingress.calls)

    def _new_start(self) -> VoiceStart:
        ingress = RecordingIngress(self.h.ids, epoch=len(self.ingresses) + 1)
        self.ingresses.append(ingress)
        return VoiceStart(
            ingress=ingress,  # type: ignore[arg-type]
            policy=self.h.policy,
            settings=voice_settings(),
            epoch=ingress.epoch,
        )

    def _text(self) -> str:
        return self.rng.choice(INTERLEAVING_TEXTS)

    def _now(self) -> int:
        return self.h.clock.now

    # -- Actions: each posts one message and returns its kind, or None when it cannot apply now --

    def _speech(self) -> str | None:
        ingress, uid = self.ingress, self.open_uid
        if ingress is None:
            return None
        if uid is None:
            self.open_uid = uid = self.h.ids()
            self.h.post(ingress.started(uid))
            return "SpeechStarted"
        roll = self.rng.random()
        if roll < 0.3:
            self.h.post(ingress.partial(uid, self._text(), heard_at_ms=self._now()))
            return "TranscriptPartial"
        if roll < 0.45:
            self.h.post(ingress.candidate(uid, self._text(), heard_at_ms=self._now()))
            return "EndpointCandidate"
        if roll < 0.55:
            self.h.post(ingress.resumed(uid))
            return "SpeechResumed"
        self.open_uid = None
        if roll < 0.85:
            self.h.post(ingress.ended(uid, self._text(), heard_at_ms=self._now()))
            return "UtteranceEnded"
        self.h.post(ingress.dropped(uid, self.rng.choice(["noise", "no_speech", "muted"])))
        return "UtteranceDropped"

    def _gate_partial(self) -> str | None:
        if self.ingress is None or not self.gate_asks:
            return None
        uid, through = self.gate_asks.pop(0)
        self.h.post(self.ingress.partial(uid, self._text(), gate=True, through_pos=through))
        return "TranscriptPartial(gate)"

    def _name(self) -> str | None:
        if self.ingress is None:
            return None
        spotter = "endpoint" if self.rng.random() < 0.3 else "server"
        self.h.post(
            self.ingress.heard(
                None if spotter == "endpoint" else self.open_uid,
                from_pos=self.rng.randint(40, 100),
                hit_pos=self.rng.randint(100, 140),
                arrival=float(self._now()),
                score=self.rng.choice([0.6, 0.9]),
                reference_score=self.rng.choice([None, None, None, 0.9]),
                spotter=spotter,
            )
        )
        return f"NameHeard({spotter})"

    def _overload(self) -> str | None:
        if self.ingress is None:
            return None
        self.h.post(self.ingress.overload(self.rng.choice(["vad", "kws", "capture"])))
        return "IngressOverload"

    def _old_epoch(self) -> str | None:
        old = [ingress for ingress in self.ingresses if ingress is not self.ingress]
        if not old:
            return None
        self.h.post(self.rng.choice(old).ended(self.h.ids(), self._text()))
        return "old epoch"

    def _wake(self) -> str | None:
        backlog = self.rng.choice([1500, 1500, 0])
        self.h.post(Wake(keyword="mist", score=0.8, backlog_ms=backlog, arrival=float(self._now())))
        return "Wake"

    def _verdict(self) -> str | None:
        waiting = [claim for claim in self.h.arbiter.claims if not claim.answered]
        if not waiting:
            return None
        claim = self.rng.choice(waiting)
        if self.rng.random() < 0.6:
            self.h.arbiter.grant(claim.claim_id)
        else:
            self.h.arbiter.deny(claim.claim_id)
        return "WakeVerdict"

    def _control(self) -> str | None:
        roll = self.rng.random()
        if roll < 0.25:
            self.h.post(Text("typed"))
            return "Text"
        if roll < 0.4:
            self.h.post(Interrupt())
            return "Interrupt"
        if roll < 0.55:
            self.h.post(Mute(active=self.rng.random() < 0.5))
            return "Mute"
        if roll < 0.7:
            self.h.post(VoiceStop())
            self.adopt(None)
            return "VoiceStop"
        if roll < 0.85:
            start = self._new_start()
            self.h.post(start)
            self.adopt(start.ingress)  # type: ignore[arg-type]
            return "VoiceStart"
        self.h.host.apply_result = self._new_start() if self.rng.random() < 0.7 else None
        refusal = self.rng.choice([None, None, "unavailable"])
        self.h.post(Reconfigure(settings=voice_settings(), refusal=refusal))
        return "Reconfigure"

    def _playback(self) -> str | None:
        turn_ids = self.h.outbound.queued_turn_ids()[-2:] or ["no-turn"]
        turn_id = self.rng.choice(turn_ids)
        played = self.rng.choice([0, 400, 1500, 60_000])
        roll = self.rng.random()
        if roll < 0.4:
            self.h.post(PlaybackProgress(turn_id=turn_id, played_ms=played))
            return "PlaybackProgress"
        if roll < 0.8:
            self.h.post(PlaybackDone(turn_id=turn_id, played_ms=played))
            return "PlaybackDone"
        self.h.post(BargeInReport(turn_id=turn_id, keyword="stop"))
        return "BargeInReport"

    def _release(self) -> str | None:
        reached = [p for p in self.pauses if p.reached.is_set() and not p.event.is_set()]
        if not reached:
            return None
        self.rng.choice(reached).release()
        return "release"

    def _tick(self) -> str | None:
        return "advance"

    # -- Running ----------------------------------------------------------------------------------

    async def step(self) -> None:
        actions: list[Callable[[], str | None]] = [
            self._speech,
            self._gate_partial,
            self._name,
            self._overload,
            self._old_epoch,
            self._wake,
            self._verdict,
            self._control,
            self._playback,
            self._release,
            self._tick,
        ]
        weights = [6, 2, 3, 1, 1, 1, 3, 3, 2, 3, 2]
        for _ in range(20):
            kind = self.rng.choices(actions, weights)[0]()
            if kind is not None:
                break
        else:
            kind = "advance"
        self.trace.append(kind)
        if kind == "advance":
            await self.h.clock.advance(self.rng.choice([20, 300, 1500, 6000]))
        await self.h.settle()
        self._sync()
        self.check()

    def check(self) -> None:
        """The step invariants."""
        h = self.h
        task = h.controller.task
        assert task is not None
        assert not task.done(), f"the mailbox task ended: {task}"
        assert self.failures == []
        assert h.outbound.violations == []
        # Every claim is settled at most once (exactly once by the end).
        settles = collections.Counter(cid for cid, _ in h.arbiter.settles)
        assert all(n == 1 for n in settles.values()), f"a claim settled twice: {settles}"
        # No utterance gets a second closing message, or any message after its closing one.
        closed: set[str] = set()
        for item in h.outbound.client:
            uid = named_utterance(item)
            if uid is None:
                continue
            assert uid not in closed, f"{describe(item)} names {uid} after its closing message"
            if describe(item) in CLOSING_MESSAGES:
                closed.add(uid)
        # No scene intent names a turn after its `cancel_turn`.
        cancelled: set[object] = set()
        for call in h.scene.calls:
            if call[0] == "cancel_turn":
                cancelled.add(call[1])
            elif call[0] == "phase" and call[2] is not None:
                assert call[2] not in cancelled, f"{call} after its turn's cancel_turn"
            elif call[0] == "tool_view":
                assert call[1] not in cancelled, f"{call} after its turn's cancel_turn"

    async def finish(self) -> None:
        """Drain and stop voice, check, close, then check the end invariants."""
        h = self.h
        for pause in self.pauses:
            pause.release()
        await h.clock.advance(20_000)
        h.post(VoiceStop())
        self.adopt(None)
        await h.settle()
        await h.clock.advance(20_000)
        self.check()
        # Every utterance a message named has exactly one closing message.
        named = {uid for uid in map(named_utterance, h.outbound.client) if uid is not None}
        closings = collections.Counter(
            named_utterance(item)
            for item in h.outbound.client
            if describe(item) in CLOSING_MESSAGES
        )
        assert sorted(uid for uid in named if closings[uid] != 1) == []

        closing = asyncio.ensure_future(h.controller.close())
        for _ in range(20):
            if closing.done():
                break
            await h.clock.advance(20_000)
        assert closing.done(), "close did not finish"
        await closing
        task = h.controller.task
        assert task is not None and task.done() and task.exception() is None
        assert self.failures == []
        # No turn task is left running.
        assert [
            t.get_name() for t in asyncio.all_tasks() if t.get_name().startswith("session-1-")
        ] == []
        # Every lane is idle.
        h.outbound.take_all()
        assert h.outbound.control == []
        assert h.outbound.lane.take() is None
        # Every shown turn was committed exactly once, and no other turn was.
        commits = collections.Counter(self.committed)
        assert all(n == 1 for n in commits.values()), f"a turn committed twice: {commits}"
        assert len(commits) == len(set(h.outbound.queued_turn_ids()))
        # Every claim was settled exactly once.
        settles = collections.Counter(cid for cid, _ in h.arbiter.settles)
        assert settles == collections.Counter(c.claim_id for c in h.arbiter.claims)
        # No message names a turn the client never saw start: a held turn sends nothing (v1).
        shown = set(h.outbound.queued_turn_ids())
        for sent in h.outbound.sent:
            item = sent.item
            stale_answer = describe(item) == "voice_decision:barge_in_ignored"
            if sent.turn_id is not None and not stale_answer:
                assert sent.turn_id in shown, f"{describe(item)} names unseen {sent.turn_id}"


async def run_interleaving(seed: int) -> None:
    run = Interleaving(seed)
    run.h.start_voice()
    run.adopt(run.h.ingress)
    try:
        for _ in range(INTERLEAVING_STEPS):
            await run.step()
        await run.finish()
    except AssertionError as error:
        raise AssertionError(f"seed {seed} after {run.trace}: {error}") from error
    REACHED[seed] = (frozenset(run.trace), run.stale_verdicts)


@pytest.mark.parametrize("seed", range(INTERLEAVING_SEEDS))
async def test_seeded_random_interleavings_keep_every_invariant(seed):
    await run_interleaving(seed)


async def test_the_interleavings_post_every_message_type_and_stale_verdicts():
    """The 200 seeds together reach every kind; seeds this process has not run are run here."""
    for seed in range(INTERLEAVING_SEEDS):
        if seed not in REACHED:
            await run_interleaving(seed)
            await close_live_controllers()
    reached = frozenset().union(*(kinds for kinds, _ in REACHED.values()))
    assert sorted(EVERY_KIND - reached) == []
    assert sum(stale for _, stale in REACHED.values()) > 0
