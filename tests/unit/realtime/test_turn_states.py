"""The turn controller's state table rows for `VoiceStart`, `voice_stop` and `Reconfigure`, mute
for a published utterance (T11d, v2 :1704-1717 and :1736-1784, v3 :1681-1682, v8 :239-247), and
the speculative held turn: its start, buffer, release and cancel (v1 :400-423, :2714-2717), M65
from `held` at its release and M67's `release` (v2 :392-408, :1615-1637), and v2's `held` column
(v2 :476-493) where v5 :1174-1189 leaves it. Then the rest of v2's table in `idle`, `thinking`,
`speaking` and `text` that v5 keeps, the keyword row's closing messages (v2 :617-651), ignored
endpoint reports (v2 :1844-1847), and Test strategy's variants that survive v5 (v2 :1117-1130).

T11e part 1 adds the addressee rows (v5 :1172-1190) and their tests (v5 :1668-1744, without the
arbitration bullets at :1709-1722), v4's muted wake, the mute bullets for unpublished candidates
(v8 :248-270), "stop" or the name during a `HandOff` wait (v6 :1649-1650), and the controller's
handling of wake verdicts (v9 :153-163, :296-300), with `FakeWakeArbiter` standing in for
arbitration. T11e part 2 runs the arbitration bullets (v5 :1709-1722) through two controllers
that share one real `WakeArbiter`.

T55c (A51) amends v5's `thinking` cell: a name accepted while a model turn is `thinking` with none
of its audio queued supersedes the turn instead of barging in. T55d keeps a superseded turn that
dispatched no tool and joins its request text to the next turn's, or records it alone where the
superseding utterance starts no turn.

Every test drives the controller only through `post` and `close`, with the fakes from
`turn_fakes.py`. The client-facing record is what the recording transport's sender took.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from dataclasses import dataclass

import pytest
import pytest_asyncio

from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import Complete, HandOff, Token
from backend.realtime.addressee import NameAlignment
from backend.realtime.budget import BargeInConfig, WakeConfig
from backend.realtime.protocol import VoiceDecisionWakeLost, WakeLostDetail
from backend.realtime.scene import Immediate
from backend.realtime.speech import HeardWord, SpotterCalibration
from backend.realtime.turn import (
    BargeInReport,
    Close,
    Interrupt,
    Mute,
    PlaybackDone,
    PlaybackProgress,
    Reconfigure,
    Text,
    VoiceStart,
    VoiceStop,
    Wake,
    WakeArbiter,
    WakeRank,
)

from .turn_fakes import (
    Call,
    FailingSynthesizer,
    FakeAlarm,
    FakeCallLater,
    Harness,
    Pause,
    RecordingIngress,
    WaitGate,
    close_live_controllers,
    describe,
    dispatch_tool,
    make_harness,
    tool_started,
    voice_settings,
)

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def _stop_controllers():
    yield
    await close_live_controllers()


def subsequence(names: list[str], expected: list[str]) -> bool:
    it = iter(names)
    return all(name in it for name in expected)


def calls(ingress: RecordingIngress, name: str) -> list[tuple[object, ...]]:
    return [call for call in ingress.calls if call[0] == name]


def new_start(h: Harness, *, epoch: int = 2, playback: bool = True) -> VoiceStart:
    ingress = RecordingIngress(h.ids, epoch=epoch)
    return VoiceStart(
        ingress=ingress,  # type: ignore[arg-type]
        policy=h.policy,
        settings=voice_settings(playback=playback),
        epoch=epoch,
    )


def request(req: int = 2, refusal: str | None = None) -> Reconfigure:
    return Reconfigure(settings=voice_settings().model_copy(update={"req": req}), refusal=refusal)


async def published(h: Harness, *partials: str) -> str:
    """Publish an utterance in idle and leave it open."""
    uid = h.ids()
    h.post(h.ingress.started(uid))
    for text in partials:
        h.post(h.ingress.partial(uid, text))
    await h.settle()
    return uid


async def in_phase(phase: str) -> tuple[Harness, Pause]:
    """A harness with a turn in `thinking`, `speaking` or `text` (a typed turn)."""
    pause = Pause()
    if phase == "thinking":
        h = make_harness([pause, Token("Late.")])
    else:
        h = make_harness([Token("One two three. Four five. "), pause, Token("Six.")])
    h.start_voice()
    if phase == "text":
        h.post(Text("typed"))
        await h.settle()
    else:
        await h.speak("hello")
    assert h.controller.phase == phase
    return h, pause


# -- voice_stop (v2 :1647-1650, :1792-1795; the row) -------------------------------------------


async def test_voice_stop_in_idle_closes_ingress_and_the_open_utterance_from_the_record():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "turn the", "turn the lights")

    h.post(VoiceStop())
    await h.settle()

    assert calls(h.ingress, "close") == [("close",)]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)
    # The detail comes from the record: the last partial's text, no probabilities.
    assert dropped.detail.text == "turn the lights"
    assert dropped.detail.duration_ms > 0
    assert (dropped.detail.no_speech_prob, dropped.detail.avg_logprob) == (None, None)


async def test_voice_stop_takes_the_detail_from_the_drop_close_returns_when_it_names_it():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "turn")
    h.ingress.close_result = h.ingress.dropped(uid, "noise", text="turn on the")

    h.post(VoiceStop())
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.text) == ("voice_stop", "turn on the")
    assert dropped.detail.duration_ms == 300


async def test_voice_stop_with_no_partial_closes_with_an_empty_detail():
    h = make_harness()
    h.start_voice()
    await published(h)

    h.post(VoiceStop())
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.detail.text, dropped.detail.duration_ms) == ("", 0)


async def test_voice_stop_before_the_first_voice_start_or_a_second_one_calls_no_close():
    h = make_harness()
    h.post(VoiceStop())
    await h.settle()
    assert h.ingress.calls == []

    h.start_voice()
    h.post(VoiceStop(), VoiceStop())
    await h.settle()

    assert calls(h.ingress, "close") == [("close",)]
    assert h.outbound.client == []


@pytest.mark.parametrize("phase", ["thinking", "speaking"])
async def test_voice_stop_while_a_voice_turn_runs_interrupts_it_then_closes_ingress(phase):
    h, pause = await in_phase(phase)
    turn_id = h.last_turn()

    h.post(VoiceStop())
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.turn_id, barge.reason, barge.detail.phase) == (turn_id, "voice_stop", phase)
    assert barge.detail.utterance_id is None
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.reason == "voice_stop"
    assert h.responder.commits[-1].outcome == "interrupted"
    assert calls(h.ingress, "close") == [("close",)]
    assert h.controller.phase == "idle"
    assert h.outbound.violations == []
    pause.release()


async def test_voice_stop_in_speaking_ends_the_tag_with_0x03_before_playback_stop():
    h = make_harness([Token("One two three. ")], auto_take=False)
    h.start_voice()
    await h.speak("hello")
    start = next(s.item for s in h.outbound.sent if getattr(s.item, "type", "") == "stream_start")
    h.outbound.take_until(lambda item: getattr(item, "msg_type", None) is not None)
    assert h.controller.phase == "speaking"

    h.post(VoiceStop())
    await h.settle()
    h.outbound.take_all()

    assert h.outbound.purges[0] == (start.turn_id, h.outbound.purges[0][1])
    assert h.outbound.purges[0][1].terminal is True
    names = h.outbound.names()
    assert subsequence(names, ["0x01", "0x03", "playback_stop", "stream_cancelled"])
    assert "0x02" not in names
    assert h.outbound.violations == []


async def test_voice_stop_in_thinking_with_a_frameless_0x02_queued_sends_0x03():
    h = make_harness([], auto_take=False)
    h.start_voice()
    await h.speak("hello")
    assert h.controller.phase == "thinking"

    h.post(VoiceStop())
    await h.settle()
    h.outbound.take_all()

    names = h.outbound.names()
    assert "0x02" not in names
    assert names.index("0x03") < names.index("playback_stop")


async def test_voice_stop_in_text_closes_ingress_and_the_typed_turn_completes():
    h, pause = await in_phase("text")

    h.post(VoiceStop())
    await h.settle()
    pause.release()
    await h.settle()

    assert calls(h.ingress, "close") == [("close",)]
    assert h.outbound.of("voice_decision:barge_in") == []
    assert h.responder.commits[-1].outcome == "completed"
    assert h.responder.commits[-1].delivered_text == "One two three. Four five. Six."


async def test_after_voice_stop_every_ingress_event_is_dropped_and_counted():
    h = make_harness()
    h.start_voice()
    h.post(VoiceStop())
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.ended(uid, "hello"))
    await h.settle()

    assert h.outbound.client == []
    assert h.controller.stale_events == 2
    assert h.responder.requests == []


# -- The utterance record (v2 :1652-1663) --------------------------------------------------------


async def test_voice_stop_before_a_queued_end_closes_the_utterance_and_drops_the_end():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hello")
    h.post(VoiceStop(), h.ingress.ended(uid, "hello there"))
    await h.settle()

    closing = [m for m in h.outbound.client if getattr(m, "type", "") in ("voice_decision",)]
    assert [(m.kind, m.reason) for m in closing] == [("dropped", "voice_stop")]
    assert h.outbound.of("transcription") == []
    assert h.responder.requests == []
    assert h.controller.stale_events == 1


async def test_an_old_epoch_end_after_voice_stop_and_voice_start_is_dropped_and_counted():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hello")
    h.post(VoiceStop(), new_start(h), h.ingress.ended(uid, "hello there"))
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)
    assert h.outbound.of("transcription") == []
    assert h.controller.stale_events == 1


# -- VoiceStart (v2 :1736-1752) ------------------------------------------------------------------


async def test_voice_start_adopts_the_ingress_and_sends_nothing():
    h = make_harness()
    start = new_start(h)
    h.post(start)
    await h.settle()
    assert h.outbound.client == []

    uid = h.ids()
    h.post(start.ingress.started(uid))  # type: ignore[attr-defined]
    await h.settle()
    assert h.outbound.names() == ["vad_status:speech_started", "state_cycle:listen"]


async def test_voice_start_with_an_ingress_adopted_runs_voice_stop_first():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hi")
    start = new_start(h)

    h.post(start, h.ingress.started(h.ids()))
    await h.settle()

    assert calls(h.ingress, "close") == [("close",)]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)
    # The old ingress's event after the swap publishes nothing.
    assert h.outbound.names().count("vad_status:speech_started") == 1
    assert h.controller.stale_events == 1


@pytest.mark.parametrize("phase", ["thinking", "speaking"])
async def test_voice_start_driven_in_a_voice_turns_phase_cuts_it_then_adopts(phase):
    h, pause = await in_phase(phase)

    h.post(new_start(h))
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert barge.reason == "voice_stop"
    assert calls(h.ingress, "close") == [("close",)]
    assert h.controller.phase == "idle"
    pause.release()


async def test_voice_start_in_text_keeps_the_typed_turn_going():
    h, pause = await in_phase("text")

    h.post(new_start(h))
    await h.settle()
    pause.release()
    await h.settle()

    assert h.responder.commits[-1].outcome == "completed"


async def test_the_next_voice_turn_reads_the_adopted_settings_playback():
    h = make_harness()
    h.start_voice()
    await h.speak("first")
    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    start = new_start(h, playback=False)
    h.post(VoiceStop(), start)
    uid = h.ids()
    ingress = start.ingress
    h.post(ingress.started(uid), ingress.ended(uid, "second"))  # type: ignore[attr-defined]
    await h.settle()

    assert [s.channel for s in h.outbound.of("stream_start")] == ["tts", "text"]


# -- Reconfigure (v2 :1753-1784, M46) ------------------------------------------------------------


async def test_reconfigure_quiescent_in_idle_hands_off_applies_and_closes_in_one_step():
    h = make_harness()
    h.start_voice()
    applied = new_start(h)
    h.host.apply_result = applied
    order: list[str] = []
    original_apply = h.host.apply

    def apply(r, seed):
        order.extend(name for name, *_ in h.ingress.calls)
        return original_apply(r, seed)

    h.host.apply = apply  # type: ignore[method-assign]
    req = request()

    h.post(req)
    await h.settle()

    assert order == ["handoff"]  # handed off before the apply, closed after it
    assert [c[0] for c in h.ingress.calls] == ["handoff", "close"]
    assert h.host.calls[0][1] is req
    assert h.outbound.client == []
    uid = h.ids()
    h.post(applied.ingress.started(uid))  # type: ignore[attr-defined]
    h.post(h.ingress.started(h.ids()))
    await h.settle()
    assert h.outbound.names() == ["vad_status:speech_started", "state_cycle:listen"]
    assert h.controller.stale_events == 1


async def test_reconfigure_while_a_turn_runs_waits_for_it_to_complete():
    h, pause = await in_phase("speaking")
    turn_id = h.last_turn()
    h.host.apply_result = new_start(h)
    h.post(request())
    await h.settle()
    assert h.host.calls == []

    pause.release()
    await h.settle()
    assert h.host.calls == []  # generation done, the tag has not ended
    h.post(PlaybackDone(turn_id=turn_id, played_ms=1))
    await h.settle()

    assert [c[0] for c in h.host.calls] == ["apply"]
    assert h.responder.commits[-1].outcome == "completed"


async def test_reconfigure_in_text_waits_for_the_typed_turn():
    h, pause = await in_phase("text")
    h.post(request())
    await h.settle()
    assert h.host.calls == []
    pause.release()
    await h.settle()
    assert [c[0] for c in h.host.calls] == ["apply"]


async def test_reconfigure_waits_for_an_utterance_in_the_record_to_close():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hm")
    h.post(request())
    await h.settle()
    assert h.host.calls == []

    h.post(h.ingress.dropped(uid, "noise"))
    await h.settle()
    assert [c[0] for c in h.host.calls] == ["apply"]


async def test_reconfigure_waits_while_ingress_holds_an_utterance_the_controller_has_not_seen():
    h = make_harness()
    h.start_voice()
    h.ingress.holding = True
    h.post(request())
    await h.settle()
    assert h.host.calls == []

    uid = h.ids()
    event = h.ingress.dropped(uid, "noise")
    h.ingress.holding = False
    h.post(event)
    await h.settle()
    assert [c[0] for c in h.host.calls] == ["apply"]


async def test_reconfigure_waits_for_an_emitted_event_still_in_the_mailbox():
    h = make_harness()
    h.start_voice()
    queued = h.ingress.overload()  # emitted by ingress, not yet posted
    h.post(request())
    await h.settle()
    assert h.host.calls == []

    h.post(queued)
    await h.settle()
    assert [c[0] for c in h.host.calls] == ["apply"]


async def test_a_dropped_event_still_counts_as_handled():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hm")
    h.post(h.ingress.dropped(uid, "noise"))
    await h.settle()
    stale = h.ingress.partial(uid, "hm late")  # names a closed utterance
    h.post(request())
    await h.settle()
    assert h.host.calls == []

    h.post(stale)
    await h.settle()
    assert h.controller.stale_events == 1
    assert [c[0] for c in h.host.calls] == ["apply"]


async def test_an_old_epoch_event_never_counts_for_the_new_ingress():
    h = make_harness()
    h.start_voice()
    applied = new_start(h)
    applied.ingress.started(h.ids())  # type: ignore[attr-defined] - the new ingress emitted one
    h.host.apply_result = applied
    h.post(request())
    await h.settle()
    second = request(3)
    h.post(h.ingress.overload(), second)  # an old-epoch event, then a second reconfigure
    await h.settle()

    assert [c[0] for c in h.host.calls] == ["apply"]  # the second waits for the new event


async def test_two_servable_reconfigures_while_speaking_refuse_the_first_as_superseded():
    h, pause = await in_phase("speaking")
    h.host.apply_result = new_start(h)
    first, second = request(2), request(3)
    h.post(first, second)
    await h.settle()

    assert h.host.calls == [("refuse", first, "superseded")]
    pause.release()
    await h.settle()
    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    await h.settle()
    assert h.host.calls[1][:2] == ("apply", second)


async def test_a_pending_reconfigure_then_a_refused_one_answers_both_and_applies_nothing():
    h, pause = await in_phase("speaking")
    first, second = request(2), request(3, refusal="unavailable")
    h.post(first, second)
    await h.settle()
    pause.release()
    await h.settle()
    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    await h.settle()

    assert h.host.calls == [("refuse", first, "superseded"), ("refuse", second, "unavailable")]


async def test_a_refused_reconfigure_with_none_pending_is_answered_at_once_while_speaking():
    h, pause = await in_phase("speaking")
    req = request(refusal="unavailable")
    h.post(req)
    await h.settle()

    assert h.host.calls == [("refuse", req, "unavailable")]
    assert h.controller.phase == "speaking"
    pause.release()


async def test_voice_stop_refuses_a_pending_reconfigure_before_its_row_sends_anything():
    h, pause = await in_phase("speaking")
    req = request()
    h.post(req)
    await h.settle()
    sent_at_refuse: list[int] = []
    original = h.host.refuse
    h.host.refuse = lambda r, reason: (  # type: ignore[method-assign]
        sent_at_refuse.append(len(h.outbound.sent)),
        original(r, reason),
    )[1]
    before = len(h.outbound.sent)

    h.post(VoiceStop())
    await h.settle()

    assert h.host.calls == [("refuse", req, "voice_stop")]
    assert sent_at_refuse == [before]
    assert h.outbound.of("voice_decision:barge_in")
    pause.release()


async def test_a_disconnect_with_a_pending_reconfigure_neither_applies_nor_refuses():
    h, pause = await in_phase("speaking")
    h.post(request())
    await h.settle()

    await h.controller.close()

    assert h.host.calls == []


async def test_apply_returning_none_keeps_the_old_ingress_adopted_and_open():
    h = make_harness()
    h.start_voice()
    h.post(request())
    await h.settle()

    assert [c[0] for c in h.host.calls] == ["apply"]
    assert calls(h.ingress, "close") == []
    h.post(h.ingress.started(h.ids()))
    await h.settle()
    assert h.outbound.names() == ["vad_status:speech_started", "state_cycle:listen"]
    h.post(Interrupt())  # a later message applies nothing: nothing is pending
    await h.settle()
    assert [c[0] for c in h.host.calls] == ["apply"]


@pytest.mark.parametrize(
    ("refusal", "reason"), [(None, "voice_stop"), ("unavailable", "unavailable")]
)
async def test_with_no_ingress_adopted_a_reconfigure_is_refused(refusal, reason):
    h = make_harness()
    req = request(refusal=refusal)
    h.post(req)
    await h.settle()

    assert h.host.calls == [("refuse", req, reason)]


# -- Mute for a published utterance (v8 :239-247, :262-263, :271; v3 :1682) ----------------------


@pytest.mark.parametrize("order", ["drop_first", "mute_first"])
async def test_mute_a_published_utterance_gets_one_dropped_muted_and_the_mute_step_sends_nothing(
    order,
):
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hey")
    before = len(h.outbound.client)
    drop = h.ingress.dropped(uid, "muted")
    if order == "drop_first":
        h.post(drop, Mute(active=True))
    else:
        h.post(Mute(active=True))
        await h.settle()
        assert len(h.outbound.client) == before  # the Mute step itself sends nothing
        h.post(drop)
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("muted", uid)
    assert len(h.outbound.client) == before + 1


async def test_mute_leaves_a_turn_in_flight_going_and_unmute_sends_nothing():
    h, pause = await in_phase("speaking")
    turn_id = h.last_turn()
    before = len(h.outbound.client)

    h.post(Mute(active=True))
    await h.settle()
    h.post(Mute(active=False))
    await h.settle()
    assert len(h.outbound.client) == before
    assert h.controller.phase == "speaking"

    pause.release()
    await h.settle()
    h.post(PlaybackDone(turn_id=turn_id, played_ms=1))
    await h.settle()
    assert h.responder.commits[-1].outcome == "completed"


async def test_a_later_unmuted_utterance_opens_normally():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hey")
    h.post(Mute(active=True), h.ingress.dropped(uid, "muted"), Mute(active=False))
    await h.settle()

    await h.speak("what time is it")

    assert [r.user_text for r in h.responder.requests] == ["what time is it"]


# -- The held turn (v1 :400-404, :2714-2717; v2 :392-408, :1615-1637) ------------------------------


async def held(h: Harness, text: str = "what time is it", *partials: str) -> str:
    """Publish an utterance in idle and post its endpoint candidate: the turn starts held."""
    uid = await published(h, *partials)
    h.post(h.ingress.candidate(uid, text))
    await h.settle()
    assert h.controller.phase == "held"
    return uid


def names_since(h: Harness, mark: int, lane: str | None = None) -> list[str]:
    """What was queued to `Outbound` from `mark` on, on one lane or on both."""
    return [describe(s.item) for s in h.outbound.sent[mark:] if lane is None or s.lane == lane]


async def test_an_endpoint_candidate_in_idle_starts_a_held_turn_that_sends_nothing():
    h = make_harness([Token("One two three. "), Token("Four five. "), Pause()])
    h.start_voice()
    uid = await published(h, "what time")
    mark, scene_mark = len(h.outbound.sent), len(h.scene.calls)

    h.post(h.ingress.candidate(uid, "what time is it"))
    await h.settle()

    assert h.controller.phase == "held"
    (req,) = h.responder.requests
    assert (req.user_text, req.origin, req.channel, req.held) == (
        "what time is it",
        "voice",
        "tts",
        True,
    )
    # Generation and synthesis run while held; nothing reaches the client or the screen.
    assert "yield:Token:Four five. " in h.responder.log
    assert [s.text for s in h.synth.segments] == ["One two three.", "Four five."]
    assert h.outbound.sent[mark:] == []
    assert h.scene.calls[scene_mark:] == []
    assert h.controller._turn.ledger.recorded_samples == 0
    assert h.ingress.calls == []


async def test_a_held_turn_dispatches_no_tool_until_its_release():
    h = make_harness(
        [Token("Setting it. "), WaitGate(), Call(dispatch_tool), tool_started(), Pause()]
    )
    h.start_voice()
    uid = await held(h, "set a timer")
    scope = h.responder.scopes[0]
    assert not scope.dispatch_gate.is_set()
    assert not scope.tool_dispatched
    assert "gate" not in h.responder.log

    gate_at: dict[str, bool] = {}
    send_control, send_turn = h.outbound.send_control, h.outbound.send_turn

    def control(item):
        gate_at.setdefault(describe(item), scope.dispatch_gate.is_set())
        send_control(item)

    def turn(turn_id, item, **kwargs):
        gate_at.setdefault(describe(item), scope.dispatch_gate.is_set())
        return send_turn(turn_id, item, **kwargs)

    h.outbound.send_control = control  # type: ignore[method-assign]
    h.outbound.send_turn = turn  # type: ignore[method-assign]
    mark = len(h.outbound.sent)

    h.post(h.ingress.ended(uid, "set a timer"))
    await h.settle()

    # v1's order: `release`, then the dispatch gate set, then the buffered output flushed.
    assert gate_at["voice_decision:release"] is False
    assert gate_at["stream_start"] is True
    assert scope.tool_dispatched
    names = names_since(h, mark)
    assert names.index("stream_start") < names.index("tool_call_started")
    assert h.outbound.violations == []


async def test_utterance_ended_releases_the_held_turn_release_first_then_its_buffer_in_order():
    pause = Pause()
    h = make_harness([Token("One two three. "), pause])
    h.start_voice()
    uid = await held(h)
    ledger = h.controller._turn.ledger
    mark, scene_mark = len(h.outbound.sent), len(h.scene.calls)

    h.post(h.ingress.ended(uid, "what time is it", heard_at_ms=0))
    await h.settle()

    turn_id = h.last_turn()
    names = names_since(h, mark)
    assert names[:6] == [
        "voice_decision:release",
        "voice_decision:endpoint",
        "vad_status:speech_ended",
        "transcription",
        "state_cycle:think",
        "stream_start",
    ]
    assert subsequence(names, ["stream_start", "stream_token", "state_cycle:speak", "0x01"])
    (release,) = h.outbound.of("voice_decision:release")
    assert (release.turn_id, release.detail.utterance_id) == (turn_id, uid)
    (transcription,) = h.outbound.of("transcription")
    assert (transcription.text, transcription.utterance_id) == ("what time is it", uid)
    # The buffered frames are recorded in the ledger as they are queued, at the flush.
    assert ledger.recorded_samples > 0
    # The scene's intents and the pacer's hooks were buffered too, and come in their order.
    assert [c[:2] for c in h.scene.calls[scene_mark:]][:3] == [
        ("phase", "thinking"),
        ("phase", "speaking"),
        ("phase", "idle"),
    ]
    assert h.scene.calls[scene_mark + 3][0] == "segment_starting"
    assert h.controller.phase == "speaking"
    assert h.outbound.violations == []

    pause.release()
    await h.settle()
    h.post(PlaybackDone(turn_id=turn_id, played_ms=1))
    await h.settle()
    (commit,) = h.responder.commits
    assert (commit.outcome, commit.delivered_text) == ("completed", "One two three. ")


async def test_a_held_text_channel_turn_whose_reply_ended_completes_only_after_its_release():
    h = make_harness([Token("It is noon.")], synthesizer=False)
    h.start_voice()
    uid = await held(h)
    assert "yield:Token:It is noon." in h.responder.log
    assert h.responder.commits == []

    h.post(h.ingress.ended(uid, "what time is it"))
    await h.settle()

    names = names_since(h, 0)
    assert names.index("voice_decision:release") < names.index("transcription")
    assert names.index("transcription") < names.index("stream_complete")
    (commit,) = h.responder.commits
    assert (commit.outcome, commit.delivered_text) == ("completed", "It is noon.")
    assert h.controller.phase == "idle"


async def test_speech_resumed_cancels_the_held_turn_without_trace_and_the_utterance_continues():
    h = make_harness([Token("Early. "), Pause()], [Token("Noon. "), Pause()])
    h.start_voice()
    uid = await held(h, "what time")
    mark, scene_mark = len(h.outbound.sent), len(h.scene.calls)

    h.post(h.ingress.resumed(uid))
    await h.settle()

    assert h.controller.phase == "idle"
    assert h.outbound.sent[mark:] == []
    assert h.outbound.purges == []
    assert h.scene.calls[scene_mark:] == []
    assert h.responder.commits == []
    assert h.responder.closed == [0]

    h.post(h.ingress.candidate(uid, "what time is it"), h.ingress.ended(uid, "what time is it"))
    await h.settle()

    assert [r.user_text for r in h.responder.requests] == ["what time", "what time is it"]
    assert [t.text for t in h.outbound.of("transcription")] == ["what time is it"]
    assert len(h.outbound.of("voice_decision:release")) == 1
    assert len(h.outbound.of("stream_start")) == 1
    assert h.outbound.tokens() == ["Noon. "]


async def test_a_held_turn_cancelled_without_trace_leaves_the_interruption_note():
    h = make_harness([Token("Long answer. "), Pause()], [Pause()], [Pause()], synthesizer=False)
    h.start_voice()
    h.post(Text("tell me a story"))
    await h.settle()
    h.post(Interrupt())
    await h.settle()
    assert h.responder.commits[-1].outcome == "interrupted"

    uid = await held(h)
    h.post(h.ingress.resumed(uid))
    await h.settle()
    h.post(h.ingress.candidate(uid, "what time is it"))
    await h.settle()

    notes = [r.interruption_note for r in h.responder.requests]
    assert notes == [None, "mid_reply", "mid_reply"]
    assert len(h.responder.commits) == 1


async def test_an_end_whose_text_differs_from_the_held_turns_starts_a_turn_from_the_end():
    h = make_harness([Pause()], [Pause()])
    h.start_voice()
    uid = await held(h, "turn on")

    h.post(h.ingress.ended(uid, "turn on the lights"))
    await h.settle()

    assert [(r.user_text, r.held) for r in h.responder.requests] == [
        ("turn on", True),
        ("turn on the lights", False),
    ]
    assert h.outbound.of("voice_decision:release") == []
    assert [t.text for t in h.outbound.of("transcription")] == ["turn on the lights"]
    assert len(h.outbound.of("stream_start")) == 1
    assert h.responder.commits == []
    assert h.responder.closed == [0]


# -- M65 from `held`, at its release (v2 :392-401, :1615-1624) ----------------------------------


async def test_tts_failed_while_held_applies_at_the_release_in_its_buffered_order():
    pause = Pause()
    h = make_harness(
        [Token("One two three. "), Token("Four five. "), pause],
        synthesizer=FailingSynthesizer(fail_call=1),
    )
    h.start_voice()
    uid = await held(h)
    ledger = h.controller._turn.ledger
    assert h.synth.calls == 2  # the second segment failed while held
    assert h.controller.phase == "held"
    mark = len(h.outbound.sent)

    h.post(h.ingress.ended(uid, "what time is it"))
    await h.settle()

    assert names_since(h, mark)[0] == "voice_decision:release"
    # The buffered frame is discarded; the `degraded` follows the two tokens, the `0x02` it.
    assert names_since(h, mark, "turn") == [
        "stream_start",
        "stream_token",
        "stream_token",
        "voice_decision:degraded",
        "0x02",
    ]
    (degraded,) = h.outbound.of("voice_decision:degraded")
    assert (degraded.reason, degraded.detail.voiced_text, degraded.detail.text_from_token) == (
        "tts_failed",
        "",
        2,
    )
    assert ledger.recorded_samples == 0
    assert h.controller.phase == "text"

    pause.release()
    await h.settle()
    # The record is the voiced text, "", plus the tokens from `text_from_token` on (M65, M69).
    (commit,) = h.responder.commits
    assert (commit.outcome, commit.delivered_text) == ("completed", "")
    assert h.controller.phase == "idle"
    assert h.outbound.violations == []


async def test_tts_failed_while_held_then_speech_resumed_cancels_it_without_trace():
    h = make_harness(
        [Token("One two three. "), Token("Four five. "), Pause()],
        synthesizer=FailingSynthesizer(fail_call=1),
    )
    h.start_voice()
    uid = await held(h)
    assert h.synth.calls == 2
    mark = len(h.outbound.sent)

    h.post(h.ingress.resumed(uid))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.responder.commits == []
    assert h.controller.phase == "idle"


# -- The `held` column of v2's table (v2 :476-493; v5 :1174-1189 keeps these) -------------------


async def test_held_speech_started_stays_held_and_sends_nothing():
    h = make_harness([Pause()])
    h.start_voice()
    await held(h)
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(h.ids()))
    await h.settle()

    assert h.controller.phase == "held"
    assert h.outbound.sent[mark:] == []
    assert h.ingress.calls == []


async def test_held_endpoint_candidate_is_ignored():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await held(h)
    mark = len(h.outbound.sent)

    h.post(h.ingress.candidate(uid, "what time is it now"))
    await h.settle()

    assert h.controller.phase == "held"
    assert h.outbound.sent[mark:] == []
    assert len(h.responder.requests) == 1
    assert h.ingress.calls == []


async def test_held_utterance_dropped_sends_dropped_and_cancels_without_trace():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await held(h)
    mark = len(h.outbound.sent)

    h.post(h.ingress.dropped(uid, "no_speech", text="what time is it"))
    await h.settle()

    assert names_since(h, mark) == ["voice_decision:dropped"]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("no_speech", uid)
    assert h.responder.commits == []
    assert h.responder.closed == [0]
    assert h.controller.phase == "idle"


async def test_held_stop_keyword_end_interrupts_with_no_new_turn_and_closes_dropped_keyword():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await held(h, "stop")
    mark = len(h.outbound.sent)

    h.post(h.ingress.ended(uid, "stop"))
    await h.settle()

    # The held turn never reached the client: no `barge_in`, no teardown, no `release`.
    assert names_since(h, mark) == ["voice_decision:dropped"]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id, dropped.detail.text) == (
        "keyword",
        uid,
        "stop",
    )
    assert len(h.responder.requests) == 1
    assert h.responder.commits == []
    assert h.controller.phase == "idle"


async def test_held_ui_interrupt_cancels_without_trace():
    h = make_harness([Pause()])
    h.start_voice()
    await held(h)
    mark = len(h.outbound.sent)

    h.post(Interrupt())
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.responder.commits == []
    assert h.responder.closed == [0]
    assert h.controller.phase == "idle"


async def test_held_voice_stop_cancels_without_trace_and_closes_ingress():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await held(h, "what time is it", "what time")
    mark = len(h.outbound.sent)

    h.post(VoiceStop())
    await h.settle()

    assert names_since(h, mark) == ["voice_decision:dropped"]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)
    assert dropped.detail.text == "what time is it"  # the last transcript the record handled
    assert calls(h.ingress, "close") == [("close",)]
    assert h.responder.commits == []
    assert h.controller.phase == "idle"


async def test_held_voice_start_driven_directly_runs_voice_stop_then_adopts():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await held(h)
    start = new_start(h)

    h.post(start)
    await h.settle()
    mark = len(h.outbound.sent)
    h.post(start.ingress.started(h.ids()))  # type: ignore[attr-defined]
    await h.settle()

    assert calls(h.ingress, "close") == [("close",)]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)
    assert h.responder.commits == []
    assert names_since(h, mark) == ["vad_status:speech_started", "state_cycle:listen"]


async def test_held_reconfigure_is_held_pending_until_the_released_turn_completes():
    pause = Pause()
    h = make_harness([Token("Noon. "), pause])
    h.start_voice()
    uid = await held(h)
    h.host.apply_result = new_start(h)

    h.post(request())
    await h.settle()
    assert h.host.calls == []

    h.post(h.ingress.ended(uid, "what time is it"))
    await h.settle()
    pause.release()
    await h.settle()
    assert h.host.calls == []
    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    await h.settle()

    assert [c[0] for c in h.host.calls] == ["apply"]


async def test_held_text_message_cancels_without_trace_drops_the_utterance_then_runs_the_typed_turn():
    h = make_harness([Pause()], [Token("Lights on.")])
    h.start_voice()
    uid = await held(h, "what time is it", "what time")
    mark = len(h.outbound.sent)

    h.post(Text("turn the lights on"), h.ingress.ended(uid, "what time is it"))
    await h.settle()

    assert calls(h.ingress, "drop") == [("drop", uid, "text_input")]
    names = names_since(h, mark)
    assert names[0] == "voice_decision:dropped"
    assert "voice_decision:release" not in names
    assert "transcription" not in names
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id, dropped.detail.text) == (
        "text_input",
        uid,
        "what time is it",
    )
    assert h.controller.stale_events == 1  # the later end names a closed utterance
    assert [(r.user_text, r.origin) for r in h.responder.requests] == [
        ("what time is it", "voice"),
        ("turn the lights on", "text"),
    ]
    (commit,) = h.responder.commits
    assert (commit.outcome, commit.delivered_text) == ("completed", "Lights on.")


async def test_held_disconnect_cancels_without_trace():
    h = make_harness([Pause()])
    h.start_voice()
    await held(h)
    mark = len(h.outbound.sent)

    await h.controller.close()

    assert h.outbound.sent[mark:] == []
    assert h.responder.commits == []
    assert h.responder.closed == [0]


async def test_held_overload_names_no_turn_and_the_turn_stays_held():
    """A held turn sends nothing to the client (v1 :402-404), so no message may name it."""
    h = make_harness([Token("One two. "), Pause()])
    h.start_voice()
    uid = await held(h)
    mark = len(h.outbound.sent)

    h.post(h.ingress.overload("vad", dropped_ms=60))
    await h.settle()

    assert names_since(h, mark) == ["voice_decision:overload"]
    (overload,) = h.outbound.of("voice_decision:overload")
    assert overload.turn_id is None
    assert (overload.detail.lane, overload.detail.dropped_ms) == ("vad", 60)
    assert h.controller.phase == "held"

    # Released, the turn's own messages name it as usual; its `stream_start` comes first.
    h.post(h.ingress.ended(uid, "what time is it"))
    await h.settle()
    turn_id = h.last_turn()
    h.post(h.ingress.overload("kws", dropped_ms=20))
    await h.settle()
    assert h.outbound.of("voice_decision:overload")[-1].turn_id == turn_id


@pytest.mark.parametrize(("backlog_ms", "reason"), [(0, "no_backlog"), (1500, "not_addressed")])
async def test_held_endpoint_wake_rejected_names_no_turn(backlog_ms, reason):
    """As with `overload`, a `wake_rejected` while held must not name the held turn."""
    h = make_harness([Token("One two. "), Pause()])
    h.start_voice()
    await held(h)

    h.post(Wake(keyword="mist", score=0.8, backlog_ms=backlog_ms, arrival=0.0))
    await h.settle()
    if backlog_ms:
        h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint", score=0.8))
        await h.settle()
        ((*_, uid),) = calls(h.ingress, "open_utterance")
        h.post(h.ingress.partial(uid, "I missed the bus", gate=True))
        await h.settle()

    (rejected,) = h.outbound.of("voice_decision:wake_rejected")
    assert (rejected.reason, rejected.turn_id) == (reason, None)
    assert h.outbound.queued_turn_ids() == []


# == The cells of v2's table in idle, thinking, speaking and text (v2 :476-492) ==================
# v5 :1174-1189 keeps these rows; its new and changed rows (`NameHeard`, the gate partial, an
# unpublished candidate's end, `SpeechStarted`, `wake`) are T11e's, and the `ptt` rows are gone.
# Unnamed speech in `thinking` and `speaking` stays unpublished (v5 :421-426).

TURN_PHASES = ["thinking", "speaking", "text"]


async def with_open_utterance(
    phase: str, *partials: str, script: list[object] | None = None
) -> tuple[Harness, Pause, str]:
    """A turn in `phase`, with an utterance published in idle before it opened and still open.

    Through the session a typed turn reaches `text` this way; `thinking` and `speaking` with a
    published utterance open are what a named utterance gives (T11e), driven here directly.
    """
    pause = Pause()
    if script is not None:
        h = make_harness(script)
    elif phase == "thinking":
        h = make_harness([pause, Token("Late.")])
    else:
        h = make_harness([Token("One two three. Four five. "), pause, Token("Six.")])
    h.start_voice()
    uid = await published(h, *partials)
    if phase == "text":
        h.post(Text("typed"))
        await h.settle()
    else:
        await h.speak("hello")
    assert h.controller.phase == phase
    return h, pause, uid


def decisions(h: Harness) -> list[tuple[str, str]]:
    """The client-facing `voice_decision`s, as (kind, reason)."""
    return [
        (m.kind, m.reason) for m in h.outbound.client if getattr(m, "type", "") == "voice_decision"
    ]


# -- EndpointCandidate: ignored outside idle --------------------------------------------------


@pytest.mark.parametrize("phase", TURN_PHASES)
async def test_endpoint_candidate_in_a_turns_phase_is_ignored(phase):
    h, pause, uid = await with_open_utterance(phase, "what")
    mark = len(h.outbound.sent)

    h.post(h.ingress.candidate(uid, "what time is it"))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.phase == phase
    assert len(h.responder.requests) == 1
    assert h.ingress.calls == []
    pause.release()


# -- SpeechResumed: ignored with no held turn ---------------------------------------------------


@pytest.mark.parametrize("phase", ["idle", *TURN_PHASES])
async def test_speech_resumed_with_no_held_turn_is_ignored(phase):
    if phase == "idle":
        h, pause = make_harness(), Pause()
        h.start_voice()
        uid = await published(h, "what")
    else:
        h, pause, uid = await with_open_utterance(phase, "what")
    mark, requests = len(h.outbound.sent), len(h.responder.requests)

    # The published utterance, and unnamed speech nobody published, "as `SpeechStarted`" (v5).
    h.post(h.ingress.resumed(uid), h.ingress.resumed(h.ids()))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.phase == phase
    assert len(h.responder.requests) == requests
    assert h.responder.commits == []
    assert h.ingress.calls == []
    pause.release()


# -- UtteranceEnded: the `text` cell ------------------------------------------------------------


async def test_utterance_ended_in_text_cuts_the_typed_turn_then_starts_a_voice_turn():
    h, pause, uid = await with_open_utterance("text", "what", script=[Token("Typed. "), Pause()])
    typed = h.last_turn()

    h.post(h.ingress.ended(uid, "what time is it"))
    await h.settle()

    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.turn_id, cancelled.reason) == (typed, "new_input")
    assert h.responder.commits[0].outcome == "interrupted"
    (transcription,) = h.outbound.of("transcription")
    assert (transcription.text, transcription.utterance_id) == ("what time is it", uid)
    assert [(r.user_text, r.origin) for r in h.responder.requests] == [
        ("typed", "text"),
        ("what time is it", "voice"),
    ]
    assert h.outbound.violations == []


# -- UtteranceDropped: `dropped` in every phase ---------------------------------------------------


@pytest.mark.parametrize("reason", ["noise", "no_speech", "stt_error"])
async def test_an_ingress_drop_in_idle_closes_with_its_reason(reason):
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hm")

    h.post(h.ingress.dropped(uid, reason, text="hm"))
    await h.settle()

    assert decisions(h) == [("dropped", reason)]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.turn_id, dropped.detail.utterance_id, dropped.detail.text) == (None, uid, "hm")
    assert h.responder.requests == []


@pytest.mark.parametrize("phase", TURN_PHASES)
async def test_utterance_dropped_in_a_turns_phase_sends_dropped_and_the_turn_goes_on(phase):
    h, pause, uid = await with_open_utterance(phase, "hm")
    mark = len(h.outbound.sent)

    h.post(h.ingress.dropped(uid, "no_speech"))
    await h.settle()

    assert names_since(h, mark) == ["voice_decision:dropped"]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("no_speech", uid)
    assert h.controller.phase == phase
    assert h.responder.commits == []
    assert h.ingress.calls == []
    pause.release()


@pytest.mark.parametrize("phase", ["thinking", "speaking"])
async def test_utterance_dropped_of_unnamed_speech_in_a_voice_turn_sends_nothing(phase):
    h, pause = await in_phase(phase)
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.partial(uid, "la la"))
    await h.settle()
    mark = len(h.outbound.sent)

    h.post(h.ingress.dropped(uid, "no_speech"))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.phase == phase
    pause.release()


# -- interrupt (UI stop), text and disconnect: the cells no earlier part tested -----------------


async def test_ui_interrupt_in_idle_is_ignored():
    h = make_harness()
    h.start_voice()
    await published(h, "hm")
    mark = len(h.outbound.sent)

    h.post(Interrupt())
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.phase == "idle"
    assert h.ingress.calls == []


@pytest.mark.parametrize("phase", ["thinking", "speaking"])
async def test_text_message_in_a_voice_turn_cancels_and_commits_then_runs_a_text_turn(phase):
    h, pause = await in_phase(phase)
    voice = h.last_turn()

    h.post(Text("typed"))
    await h.settle()

    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.turn_id, cancelled.reason) == (voice, "new_input")
    assert h.responder.commits[0].outcome == "interrupted"
    assert h.responder.log.index("commit:interrupted") < h.responder.log.index("open", 1)
    assert [r.origin for r in h.responder.requests] == ["voice", "text"]
    assert h.outbound.of("voice_decision:barge_in") == []
    assert h.outbound.violations == []
    pause.release()


async def test_disconnect_in_idle_with_an_utterance_open_sends_nothing_and_calls_no_lever():
    h = make_harness()
    h.start_voice()
    await published(h, "hm")
    mark = len(h.outbound.sent)

    await h.controller.close()

    assert h.outbound.sent[mark:] == []
    assert h.ingress.calls == []
    assert h.responder.commits == []


@pytest.mark.parametrize("phase", TURN_PHASES)
async def test_disconnect_in_a_turns_phase_commits_and_sends_nothing(phase):
    h, pause, _uid = await with_open_utterance(phase, "hm")
    mark = len(h.outbound.sent)

    await h.controller.close()

    assert h.outbound.sent[mark:] == []
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    pause.release()


# -- The keyword verdict row (v2 :483) and its closing messages (v2 :624-632) -------------------


async def test_keyword_in_idle_is_ignored_with_one_barge_in_ignored_idle_and_the_turn_starts():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await published(h, "stop the", "stop the music")

    h.post(h.ingress.ended(uid, "stop the music"))
    await h.settle()

    (ignored,) = h.outbound.of("voice_decision:barge_in_ignored")
    assert (ignored.turn_id, ignored.reason) == (None, "idle")
    assert (ignored.detail.keyword, ignored.detail.text, ignored.detail.utterance_id) == (
        "stop",
        "stop the",
        uid,
    )
    assert h.outbound.of("voice_decision:barge_in") == []
    assert [r.user_text for r in h.responder.requests] == ["stop the music"]


async def test_stop_keyword_ending_in_thinking_interrupts_with_no_turn_and_closes_dropped_keyword():
    h, pause = await in_phase("thinking")
    turn_id = h.last_turn()
    uid = h.ids()

    h.post(h.ingress.started(uid), h.ingress.ended(uid, "Stop."))
    await h.settle()

    assert decisions(h)[-2:] == [("barge_in", "keyword"), ("dropped", "keyword")]
    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.turn_id, barge.detail.phase, barge.detail.utterance_id) == (
        turn_id,
        "thinking",
        uid,
    )
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.detail.utterance_id, dropped.detail.text) == (uid, "Stop.")
    names = h.outbound.names()
    assert names.index("stream_cancelled") < names.index("voice_decision:dropped")
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    assert len(h.responder.requests) == 1
    assert h.outbound.of("transcription")[-1].utterance_id != uid
    assert h.controller.phase == "idle"
    pause.release()


async def test_unnamed_speech_ending_in_thinking_without_a_keyword_changes_nothing():
    h, pause = await in_phase("thinking")
    uid = h.ids()
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(uid), h.ingress.ended(uid, "stop the music"))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.phase == "thinking"
    pause.release()


async def test_keyword_only_utterance_ending_in_speaking_gets_barge_in_then_dropped_keyword():
    h, pause = await in_phase("speaking")
    uid = h.ids()

    h.post(h.ingress.started(uid), h.ingress.ended(uid, "stop"))
    await h.settle()

    assert decisions(h)[-2:] == [("barge_in", "keyword"), ("dropped", "keyword")]
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert dropped.detail.utterance_id == uid
    assert len(h.responder.requests) == 1
    assert h.controller.phase == "idle"
    pause.release()


async def test_text_after_the_keyword_at_the_end_in_speaking_becomes_the_next_turn():
    h = make_harness([Token("One two three. Four five. "), Pause()], [Pause()])
    h.start_voice()
    await h.speak("hello")
    assert h.controller.phase == "speaking"
    uid = h.ids()
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(uid), h.ingress.ended(uid, "Stop, what's the weather?"))
    await h.settle()

    assert h.outbound.of("voice_decision:dropped") == []
    assert h.responder.commits[0].outcome == "interrupted"
    assert [r.user_text for r in h.responder.requests] == ["hello", "what's the weather?"]
    transcription = h.outbound.of("transcription")[-1]
    assert (transcription.text, transcription.utterance_id) == ("what's the weather?", uid)
    names = names_since(h, mark)
    assert subsequence(names, ["voice_decision:barge_in", "stream_cancelled", "transcription"])
    assert h.outbound.violations == []


async def test_a_partial_keyword_cut_in_speaking_then_an_end_in_idle_starts_the_remainder():
    h = make_harness([Token("One two three. Four five. "), Pause()], [Pause()])
    h.start_voice()
    await h.speak("hello")
    uid = h.ids()

    h.post(h.ingress.started(uid), h.ingress.partial(uid, "stop what"))
    await h.settle()
    assert h.controller.phase == "idle"
    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert barge.detail.utterance_id == uid
    mark = len(h.outbound.sent)

    # The barge-in published it: its later partials show, and its keyword is not judged again.
    h.post(h.ingress.partial(uid, "stop what time"), h.ingress.ended(uid, "Stop. What time is it?"))
    await h.settle()

    assert names_since(h, mark)[0] == "transcript_partial"
    assert h.outbound.of("voice_decision:barge_in_ignored") == []
    transcription = h.outbound.of("transcription")[-1]
    assert (transcription.text, transcription.utterance_id) == ("What time is it?", uid)
    assert [r.user_text for r in h.responder.requests] == ["hello", "What time is it?"]


async def test_a_partial_keyword_cut_with_nothing_after_it_ends_in_idle_as_dropped_keyword():
    h, pause = await in_phase("speaking")
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.partial(uid, "stop"))
    await h.settle()
    assert h.controller.phase == "idle"

    h.post(h.ingress.ended(uid, "Stop."))
    await h.settle()

    assert decisions(h)[-1] == ("dropped", "keyword")
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.detail.utterance_id, dropped.detail.text) == (uid, "Stop.")
    assert len(h.responder.requests) == 1
    assert h.outbound.of("transcription")[-1].utterance_id != uid
    pause.release()


async def test_a_keyword_cut_with_nothing_after_it_ending_in_text_leaves_the_typed_turn_going():
    h = make_harness([Token("One two three. Four five. "), Pause()], [Token("Typed. "), Pause()])
    h.start_voice()
    await h.speak("hello")
    uid = h.ids()
    h.post(h.ingress.started(uid), h.ingress.partial(uid, "stop"))
    await h.settle()
    h.post(Text("typed"))
    await h.settle()
    assert h.controller.phase == "text"

    h.post(h.ingress.ended(uid, "stop"))
    await h.settle()

    assert decisions(h)[-1] == ("dropped", "keyword")
    assert h.controller.phase == "text"
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    assert len(h.responder.requests) == 2


async def test_keyword_in_text_does_not_consult_the_policy():
    h, pause = await in_phase("text")
    consulted: list[str] = []

    class Spy:
        def __getattr__(self, name):
            consulted.append(name)
            return getattr(h.policy, name)

    h.controller._policy = Spy()  # the policy VoiceStart adopted, watched
    uid = h.ids()
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(uid), h.ingress.partial(uid, "stop"), h.ingress.ended(uid, "stop"))
    await h.settle()

    assert consulted == []
    assert h.outbound.sent[mark:] == []
    assert h.controller.phase == "text"
    pause.release()


# -- Endpoint barge-in reports: an ignored verdict names no utterance (v2 :1844-1847) -----------


async def test_an_endpoint_report_before_the_arm_delay_is_ignored_not_armed():
    h = make_harness(
        [Token("One two three. Four five. "), Pause()], barge_in=BargeInConfig(arm_delay_ms=5000)
    )
    h.start_voice()
    await h.speak("hello")
    turn_id = h.last_turn()
    assert h.controller.phase == "speaking"

    h.post(BargeInReport(turn_id=turn_id, keyword="Stop"))
    await h.settle()

    (ignored,) = h.outbound.of("voice_decision:barge_in_ignored")
    assert (ignored.turn_id, ignored.reason) == (turn_id, "not_armed")
    assert (ignored.detail.keyword, ignored.detail.text, ignored.detail.utterance_id) == (
        "stop",
        "",
        None,
    )
    assert h.responder.commits == []
    assert h.controller.phase == "speaking"


async def test_an_endpoint_report_for_an_ended_turn_is_ignored_stale_turn_and_counted():
    h = make_harness([Token("One. ")], [Token("Two. Three. "), Pause()])
    h.start_voice()
    await h.speak("hello")
    first = h.last_turn()
    h.post(PlaybackDone(turn_id=first, played_ms=1))
    await h.speak("again")
    assert h.controller.phase == "speaking"
    stale = h.controller.stale_events

    h.post(BargeInReport(turn_id=first, keyword="stop"))
    await h.settle()

    (ignored,) = h.outbound.of("voice_decision:barge_in_ignored")
    assert (ignored.turn_id, ignored.reason, ignored.detail.utterance_id) == (
        first,
        "stale_turn",
        None,
    )
    assert h.controller.stale_events == stale + 1
    assert [c.outcome for c in h.responder.commits] == ["completed"]


# -- Test strategy's variants (v2 :1117-1130) that v5 keeps ----------------------------------------
# The `ptt` variants went with the `ptt` rows (v5 :1179, :1664-1666). The `voice_stop` variants in
# idle and held are part 1's and part 2's; these are the other three phases.


@pytest.mark.parametrize("phase", TURN_PHASES)
async def test_voice_stop_with_an_utterance_open_closes_it_dropped_voice_stop(phase):
    h, pause, uid = await with_open_utterance(phase, "turn the")

    h.post(VoiceStop())
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id, dropped.detail.text) == (
        "voice_stop",
        uid,
        "turn the",
    )
    assert calls(h.ingress, "close") == [("close",)]
    if phase == "text":
        assert decisions(h) == [("dropped", "voice_stop")]
        pause.release()
        await h.settle()
        assert h.responder.commits[-1].outcome == "completed"
    else:
        assert decisions(h)[-2:] == [("barge_in", "voice_stop"), ("dropped", "voice_stop")]
        assert h.responder.commits[-1].outcome == "interrupted"
        pause.release()


async def test_reconfigure_in_idle_with_one_pending_refuses_the_first_as_superseded():
    h = make_harness()
    h.start_voice()
    uid = await published(h, "hm")
    h.host.apply_result = new_start(h)
    first, second = request(2), request(3)

    h.post(first, second)
    await h.settle()
    assert h.host.calls == [("refuse", first, "superseded")]

    h.post(h.ingress.dropped(uid, "noise"))
    await h.settle()
    assert h.host.calls[1][:2] == ("apply", second)


async def test_reconfigure_in_idle_carrying_a_refusal_is_refused_at_once():
    h = make_harness()
    h.start_voice()
    req = request(refusal="unavailable")

    h.post(req)
    await h.settle()

    assert h.host.calls == [("refuse", req, "unavailable")]
    assert calls(h.ingress, "close") == []


# == T11e part 1: the addressee rows ================================================================
# Every test here uses `FakeWakeArbiter`: `solo=True` grants each claim before `claim` returns, as
# the real arbiter does for one session; `solo=False` leaves it waiting for the test's verdict.

HIT = 110
GATE_THROUGH = HIT + 15  # the hit plus `gate_tail_ms` 300, in 20 ms frames


async def name_hit(h: Harness, uid: str | None = None, **heard: object) -> str:
    """Post a server `NameHeard` and run the mailbox; returns the candidate's utterance id."""
    opened = len(calls(h.ingress, "open_utterance"))
    h.post(h.ingress.heard(uid, hit_pos=HIT, **heard))  # type: ignore[arg-type]
    await h.settle()
    if uid is not None:
        return uid
    new = calls(h.ingress, "open_utterance")[opened:]
    assert len(new) == 1, "the name opened no candidate"
    return str(new[0][3])


async def gate(h: Harness, uid: str, text: str) -> None:
    """Post the candidate's gate partial, the decode the controller requested."""
    h.post(h.ingress.partial(uid, text, gate=True, through_pos=GATE_THROUGH))
    await h.settle()


def engaged_calls(h: Harness) -> list[bool]:
    return [bool(call[1]) for call in calls(h.ingress, "set_engaged")]


def ducks(h: Harness) -> list[float]:
    return [item.db for item in h.outbound.of("playback_duck")]


async def speaking_with_next(*scripts: list[object]) -> tuple[Harness, Pause, str]:
    """A voice turn in `speaking`, paused mid-reply, with scripts for the turns after it."""
    pause = Pause()
    h = make_harness([Token("One two three. Four five. "), pause, Token("Six.")], *scripts)
    h.start_voice()
    await h.speak("hello")
    assert h.controller.phase == "speaking"
    return h, pause, h.last_turn()


# -- NameHeard in idle: name first, name last, the rejections (v5 :1174-1175, :1669-1688) ----------


async def test_name_first_sends_nothing_at_the_hit_then_publishes_in_one_step_and_starts_the_turn():
    h = make_harness([Pause()])
    h.start_voice()

    uid = await name_hit(h)

    # Check a, the candidate from `from_pos`, a claim, check b, the gate partial: no `attend`.
    assert h.outbound.names() == []
    assert calls(h.ingress, "open_utterance") == [("open_utterance", "wake", 100, uid)]
    assert [(c.session_id, c.rank.margin) for c in h.arbiter.claims] == [
        ("session-1", pytest.approx(0.4))
    ]
    assert calls(h.ingress, "request_gate_partial") == [("request_gate_partial", uid, GATE_THROUGH)]
    assert h.responder.prefetches == []
    mark = len(h.outbound.client)
    at_wake: list[list[str]] = []
    h.scene.on_call = lambda call: (
        at_wake.append(h.outbound.names()[mark:]) if call[1] == "wake" else None
    )

    await gate(h, uid, "mist what")

    assert h.outbound.names()[mark:] == [
        "vad_status:speech_started",
        "state_cycle:listen",
        "transcript_partial",
    ]
    # The scene `wake` comes in the publishing step, after `listen` went to the client.
    assert at_wake == [["vad_status:speech_started", "state_cycle:listen"]]
    assert h.outbound.of("transcript_partial")[0].text == "mist what"
    assert h.arbiter.settles == [(1, True)]

    h.post(h.ingress.ended(uid, "Mist, what's up?"))
    await h.settle()
    assert [r.user_text for r in h.responder.requests] == ["Mist, what's up?"]
    transcription = h.outbound.of("transcription")[-1]
    assert (transcription.text, transcription.utterance_id) == ("Mist, what's up?", uid)


async def test_name_last_with_a_preamble_opens_at_from_pos_and_the_turn_holds_the_preamble():
    h = make_harness([Pause()])
    h.start_voice()

    uid = await name_hit(h, from_pos=40)
    await gate(h, uid, "so about tomorrow are you there mist")
    h.post(h.ingress.ended(uid, "So about tomorrow... are you there, Mist?"))
    await h.settle()

    assert calls(h.ingress, "open_utterance") == [("open_utterance", "wake", 40, uid)]
    assert h.ingress.first_pos[uid] == 40
    (request,) = h.responder.requests
    assert request.user_text == "So about tomorrow... are you there, Mist?"


async def test_i_missed_the_bus_is_rejected_with_no_trace_of_its_words_or_its_id(caplog):
    caplog.set_level(logging.DEBUG)
    h = make_harness([Pause()])
    h.start_voice()

    uid = await name_hit(h)
    h.post(h.ingress.partial(uid, "I missed the"))
    await gate(h, uid, "I missed the bus")
    h.post(h.ingress.ended(uid, "I missed the bus."))
    await h.settle()

    assert h.outbound.names() == ["voice_decision:wake_rejected"]
    (rejected,) = h.outbound.of("voice_decision:wake_rejected")
    assert (rejected.reason, rejected.turn_id) == ("not_addressed", None)
    assert rejected.detail.model_dump() == {
        "source": "server",
        "score": 0.9,
        "reference_score": None,
    }
    assert "utterance_id" not in rejected.model_dump_json()
    assert calls(h.ingress, "drop") == [("drop", uid, "not_addressed")]
    assert h.responder.prefetches == [] and h.responder.requests == []
    assert h.arbiter.settles == [(1, False)]
    for record in caplog.records:
        message = record.getMessage().lower()
        assert "missed" not in message and "bus" not in message
    for item in h.outbound.client:
        dumped = item.model_dump_json() if hasattr(item, "model_dump_json") else repr(item)
        assert uid not in dumped


# -- The name at the hit (v16 T69): a server wake's gate reads the words heard at its hit -----------

ALIAS_LAST = "what's the time, missed?"


def heard_at_hit(hit_pos: int) -> tuple[HeardWord, ...]:
    """`ALIAS_LAST` timed so that "missed?" ends on the hit, as the spotter hears a name."""
    return (
        HeardWord("what's", hit_pos - 30, hit_pos - 21),
        HeardWord("the", hit_pos - 20, hit_pos - 16),
        HeardWord("time,", hit_pos - 15, hit_pos - 6),
        HeardWord("missed?", hit_pos - 5, hit_pos),
    )


def spy_alignments(h: Harness, monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record the alignment the controller passes the gate's `decide`, verdicts unchanged."""
    gate = h.kwargs["gate"]
    decide = gate.decide  # type: ignore[attr-defined]
    seen: list[object] = []

    def recording(**kwargs: object) -> object:
        seen.append(kwargs.get("alignment"))
        return decide(**kwargs)

    monkeypatch.setattr(gate, "decide", recording)
    return seen


@pytest.mark.parametrize(
    ("speech_end_pos", "accepted"),
    [(HIT, True), (HIT + 5, True), (HIT + 30, False)],
    ids=["speech-ended-at-the-hit", "100-ms-past", "speech-went-on-600-ms"],
)
async def test_a_server_gate_partial_reads_an_alias_at_the_hit_under_the_guard(
    monkeypatch, speech_end_pos, accepted
):
    h = make_harness([Pause()])
    h.start_voice()
    seen = spy_alignments(h, monkeypatch)
    uid = await name_hit(h)
    words = heard_at_hit(HIT)

    h.post(
        dataclasses.replace(
            h.ingress.partial(uid, ALIAS_LAST, gate=True, through_pos=GATE_THROUGH),
            words=words,
            speech_end_pos=speech_end_pos,
        )
    )
    await h.settle()

    assert seen == [NameAlignment(HIT, 20, speech_end_pos, words)]
    assert h.arbiter.settles == [(1, accepted)]
    if accepted:
        assert h.outbound.of("voice_decision:wake_rejected") == []
        assert h.outbound.of("transcript_partial")[0].text == ALIAS_LAST
    else:
        (rejected,) = h.outbound.of("voice_decision:wake_rejected")
        assert rejected.reason == "not_addressed"
        assert calls(h.ingress, "drop") == [("drop", uid, "not_addressed")]


async def test_an_endpoint_wake_is_decided_on_its_text_alone(monkeypatch):
    h = make_harness([Pause()])
    h.start_voice()
    seen = spy_alignments(h, monkeypatch)
    h.post(Wake(keyword="mist", score=0.8, backlog_ms=1500, arrival=0.0))
    await h.settle()
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint", score=0.8))
    await h.settle()
    ((*_, uid),) = calls(h.ingress, "open_utterance")

    # The same words at its hit would wake a server candidate; an endpoint's are not read.
    h.post(
        dataclasses.replace(
            h.ingress.partial(uid, ALIAS_LAST, gate=True),
            words=heard_at_hit(99),
            speech_end_pos=99,
        )
    )
    await h.settle()

    assert seen == [None]
    (rejected,) = h.outbound.of("voice_decision:wake_rejected")
    assert rejected.reason == "not_addressed"


async def test_a_held_end_is_read_with_its_words_and_no_speech_end(monkeypatch):
    h = make_harness([Pause()], solo=False)
    h.start_voice()
    seen = spy_alignments(h, monkeypatch)
    uid = await name_hit(h)
    words = heard_at_hit(HIT)
    h.post(dataclasses.replace(h.ingress.ended(uid, ALIAS_LAST), words=words))
    await h.settle()

    h.arbiter.grant(1)
    await h.settle()

    assert calls(h.ingress, "request_gate_partial") == []
    assert seen == [NameAlignment(HIT, 20, None, words)]
    assert [r.user_text for r in h.responder.requests] == [ALIAS_LAST]
    assert h.arbiter.settles == [(1, True)]


async def test_self_speech_in_her_recent_speech_from_any_session_rejects_before_the_candidate():
    asked: list[int] = []

    def recent(ms: int) -> str:
        asked.append(ms)
        return "The room unit said: ask Mist anything."  # another session's turn ledger

    h = make_harness([Pause()], recent_tts_text=recent)
    h.start_voice()

    h.post(h.ingress.heard(None, hit_pos=HIT))
    await h.settle()

    assert h.outbound.names() == ["voice_decision:wake_rejected"]
    assert h.outbound.of("voice_decision:wake_rejected")[0].reason == "self_speech"
    assert asked == [BargeInConfig().echo_guard_ms]
    assert calls(h.ingress, "open_utterance") == []
    assert h.arbiter.claims == []


async def test_self_media_rejects_at_the_claim_with_no_state_and_before_any_gate_partial():
    h = make_harness([Pause()])
    h.start_voice()

    uid = await name_hit(h, reference_score=0.7)

    assert h.outbound.names() == ["voice_decision:wake_rejected"]
    (rejected,) = h.outbound.of("voice_decision:wake_rejected")
    assert (rejected.reason, rejected.detail.reference_score) == ("self_media", 0.7)
    assert calls(h.ingress, "request_gate_partial") == []
    assert calls(h.ingress, "drop") == [("drop", uid, "not_addressed")]
    assert h.arbiter.settles == [(1, False)]


async def test_scene_wake_then_listening_only_for_an_accepted_name():
    h = make_harness([Pause()])
    h.start_voice()
    rejected = await name_hit(h)
    await gate(h, rejected, "I missed the bus")
    assert h.scene.calls == []

    accepted = await name_hit(h)
    assert h.scene.calls == []
    await gate(h, accepted, "mist, what's up")

    assert h.scene.phases() == [
        ("phase", "wake", None, Immediate()),
        ("phase", "listening", None, Immediate()),
    ]


async def test_the_name_step_sends_no_attend_and_listen_goes_out_at_acceptance():
    h = make_harness([Pause()], solo=False)  # two sessions: the claim waits for its verdict
    h.start_voice()
    h.clock.now = 500

    uid = await name_hit(h, arrival=500.0)

    assert h.outbound.names() == []
    assert h.clock.now == 500
    assert calls(h.ingress, "request_gate_partial") == []
    h.arbiter.grant(1)
    await h.settle()
    assert calls(h.ingress, "request_gate_partial") == [("request_gate_partial", uid, GATE_THROUGH)]
    assert h.outbound.names() == []

    await gate(h, uid, "mist what's up")

    assert h.outbound.names()[:2] == ["vad_status:speech_started", "state_cycle:listen"]
    assert h.outbound.of("state_cycle:attend") == []


async def test_name_heard_while_held_is_ignored():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await held(h)
    mark = len(h.outbound.sent)

    h.post(h.ingress.heard(uid, hit_pos=HIT))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.ingress.calls == []
    assert h.arbiter.claims == []
    assert h.controller.phase == "held"


# -- The held end of an unpublished candidate (v5 :1176, :1184-1186, :1697-1701) -------------------


async def test_a_held_end_waits_for_the_verdict_and_acceptance_replays_it():
    h = make_harness([Token("Tomorrow is clear."), Pause()])
    h.start_voice()
    uid = await name_hit(h, from_pos=40)
    text = "so about tomorrow are you there mist"
    mark = len(h.outbound.client)

    h.post(h.ingress.candidate(uid, text), h.ingress.ended(uid, text))
    await h.settle()
    assert h.outbound.client[mark:] == []
    assert h.responder.prefetches == [] and h.responder.requests == []

    await gate(h, uid, text)

    names = h.outbound.names()[mark:]
    assert names[:3] == ["vad_status:speech_started", "state_cycle:listen", "transcript_partial"]
    assert subsequence(names, ["voice_decision:release", "transcription", "stream_start"])
    (request,) = h.responder.requests
    assert (request.user_text, request.held) == (text, True)
    assert h.controller.phase in ("thinking", "speaking")


async def test_a_held_end_is_discarded_on_rejection():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await name_hit(h)
    text = "so about tomorrow the mist"
    h.post(h.ingress.candidate(uid, text), h.ingress.ended(uid, text))
    await h.settle()

    await gate(h, uid, text)

    assert h.outbound.names() == ["voice_decision:wake_rejected"]
    assert h.responder.requests == [] and h.responder.prefetches == []
    assert h.outbound.of("voice_decision:dropped") == []
    assert h.controller.phase == "idle"


async def test_an_end_held_before_the_verdict_is_decided_on_its_own_text():
    """Ingress ended the candidate before the claim won, so no gate decode can come (v9 :153)."""
    h = make_harness([Pause()], solo=False)
    h.start_voice()
    uid = await name_hit(h)
    h.post(h.ingress.ended(uid, "Mist, what's the time?"))
    await h.settle()

    h.arbiter.grant(1)
    await h.settle()

    assert calls(h.ingress, "request_gate_partial") == []
    assert [r.user_text for r in h.responder.requests] == ["Mist, what's the time?"]
    assert h.arbiter.settles == [(1, True)]


# -- Speaking and thinking (v5 :1174, :1177-1178, :1182-1183, :1692-1694, :1702-1708) --------------


@pytest.mark.parametrize("phase", ["thinking", "speaking"])
async def test_a_rejected_name_unducks_and_sends_no_state_with_nothing_cut(phase):
    h, pause = await in_phase(phase)
    turn_id = h.last_turn()
    uid = h.ids()
    h.post(h.ingress.started(uid))
    await h.settle()
    mark = len(h.outbound.client)

    await name_hit(h, uid)
    await gate(h, uid, "I missed it")

    duck = ["playback_duck"] if phase == "speaking" else []
    assert h.outbound.names()[mark:] == [*duck, "voice_decision:wake_rejected", *duck]
    assert ducks(h) == ([12.0, 0.0] if phase == "speaking" else [])
    assert h.outbound.of("voice_decision:wake_rejected")[0].turn_id == turn_id
    assert h.responder.commits == []
    assert h.controller.phase == phase
    assert calls(h.ingress, "open_utterance") == []  # the candidate is the engaged utterance
    pause.release()


async def test_while_speaking_other_unnamed_speech_sends_nothing_and_does_not_duck():
    h, pause, _ = await speaking_with_next()
    uid = h.ids()
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(uid), h.ingress.partial(uid, "what a day"))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    pause.release()


async def test_while_speaking_mist_wait_ducks_at_the_hit_accepts_and_interrupts_with_wake():
    h, pause, turn_id = await speaking_with_next([Pause()])
    uid = h.ids()
    h.post(h.ingress.started(uid))
    await h.settle()
    mark = len(h.outbound.client)

    await name_hit(h, uid)
    assert h.outbound.names()[mark:] == ["playback_duck"]
    assert ducks(h) == [12.0]

    # "wait" is a stop keyword: the candidate's partials go to the gate, not the keyword policy.
    h.post(h.ingress.partial(uid, "mist wait"))
    await h.settle()
    assert h.outbound.of("voice_decision:barge_in") == []
    await gate(h, uid, "mist, wait")

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.turn_id) == ("wake", turn_id)
    assert (barge.detail.phase, barge.detail.utterance_id) == ("speaking", uid)
    assert h.outbound.of("voice_decision:supersede") == []
    names = h.outbound.names()[mark:]
    assert subsequence(
        names,
        [
            "voice_decision:barge_in",
            "stream_cancelled",
            "vad_status:speech_started",
            "state_cycle:listen",
        ],
    )
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    h.post(h.ingress.ended(uid, "Mist, wait."))
    await h.settle()
    assert [r.user_text for r in h.responder.requests] == ["hello", "Mist, wait."]
    pause.release()


async def test_while_thinking_unnamed_speech_neither_holds_output_nor_supersedes():
    pause = Pause()
    h = make_harness([Token("One. "), pause, Token("Two. "), Pause()])
    h.lane.hold()  # no audio yet: the turn stays in `thinking` while its tokens flow
    h.start_voice()
    await h.speak("hello")
    assert h.controller.phase == "thinking"
    uid = h.ids()
    silent = h.controller.silent_closes

    h.post(h.ingress.started(uid), h.ingress.partial(uid, "what about the"))
    pause.release()
    await h.settle()

    assert h.outbound.tokens() == ["One. ", "Two. "]
    kinds = [kind for kind, _ in decisions(h)]
    assert "hold" not in kinds and "supersede" not in kinds
    assert h.controller.phase == "thinking"

    h.post(h.ingress.ended(uid, "what about the weather"))
    await h.settle()
    assert h.controller.silent_closes == silent + 1
    assert len(h.responder.requests) == 1
    assert h.responder.commits == []
    h.lane.open()


@pytest.mark.parametrize("phase", ["speaking", "text"])
async def test_unnamed_speech_ending_without_a_keyword_closes_silently_and_is_counted(phase):
    h, pause = await in_phase(phase)
    uid = h.ids()
    silent = h.controller.silent_closes
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(uid), h.ingress.ended(uid, "what a lovely day"))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.silent_closes == silent + 1
    assert h.controller.phase == phase
    pause.release()


async def test_a_name_during_a_typed_turn_interrupts_it_after_acceptance():
    h = make_harness([Token("Typed. "), Pause()], [Pause()])
    h.start_voice()
    h.post(Text("typed"))
    await h.settle()
    typed = h.last_turn()
    assert h.controller.phase == "text"

    uid = await name_hit(h)
    assert h.responder.commits == []  # nothing is cut before the verdict
    await gate(h, uid, "mist, what's the time")

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.turn_id, barge.detail.phase) == ("wake", typed, "text")
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    assert h.outbound.names()[-1] == "transcript_partial"


async def test_speech_started_in_text_is_ignored():
    h, pause = await in_phase("text")
    mark = len(h.outbound.sent)

    h.post(h.ingress.started(h.ids()))
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.responder.prefetches == []
    pause.release()


# -- "stop", or the name, during the wait after a `HandOff` ack (v6 :1649-1650) --------------------


@pytest.mark.parametrize("how", ["stop", "name"])
async def test_stop_or_the_name_during_the_handoff_wait_cancels_as_in_thinking(how):
    h = make_harness([HandOff("One moment."), Pause(), Token(" Here it is.")], [Pause()])
    h.start_voice()
    await h.speak("look it up")
    turn_id = h.last_turn()
    h.post(PlaybackProgress(turn_id=turn_id, played_ms=600))
    await h.settle()
    assert h.controller.phase == "thinking"
    uid = h.ids()
    h.post(h.ingress.started(uid))

    if how == "stop":
        h.post(h.ingress.ended(uid, "Stop."))
        await h.settle()
    else:
        await h.settle()
        await name_hit(h, uid)
        await gate(h, uid, "mist, never mind that")

    (barge,) = h.outbound.of("voice_decision:barge_in")
    reason = "keyword" if how == "stop" else "wake"
    assert (barge.reason, barge.turn_id, barge.detail.phase) == (reason, turn_id, "thinking")
    # The ack's audio is queued, so the name barges in (A51 supersedes only before any audio).
    assert h.outbound.of("voice_decision:supersede") == []
    assert h.responder.closed == [0]
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]


# -- The endpoint `wake` (v5 :1179, :1723-1733) ------------------------------------------------------


def endpoint_wake(backlog_ms: int = 1500, *, arrival: float = 1000.0, score: float = 0.8) -> Wake:
    return Wake(keyword="mist", score=score, backlog_ms=backlog_ms, arrival=arrival)


async def test_endpoint_wake_sends_nothing_at_the_message_and_opens_at_the_name_heards_from_pos():
    asked: list[int] = []
    h = make_harness([Pause()], recent_tts_text=lambda ms: asked.append(ms) or "")
    h.start_voice()

    h.post(endpoint_wake(1500))
    await h.settle()

    # Check a at the `wake` message, before any of the backlog has arrived: no `attend`.
    assert asked == [BargeInConfig().echo_guard_ms]
    assert h.outbound.names() == []
    (expect,) = calls(h.ingress, "expect_endpoint_wake")
    assert expect[2] == {"keyword": "mist", "score": 0.8, "backlog_ms": 1500, "arrival": 1000.0}
    assert calls(h.ingress, "open_utterance") == []

    # The backlog's first frame is 25 (75 frames ending at the hit, 99); the chain rule says 60.
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint", score=0.8))
    await h.settle()
    ((_, mode, from_pos, uid),) = calls(h.ingress, "open_utterance")
    assert (mode, from_pos) == ("wake", 60)
    assert calls(h.ingress, "request_gate_partial") == [("request_gate_partial", uid, 99 + 15)]

    h.post(h.ingress.partial(uid, "mist what's the time", gate=True))
    await h.settle()
    assert h.outbound.names()[0:2] == ["vad_status:speech_started", "state_cycle:listen"]
    assert ("phase", "wake", None, Immediate()) in h.scene.calls


async def test_endpoint_wake_ranks_with_its_calibration_and_a_slow_backlog_still_takes_part():
    calibration = SpotterCalibration(
        false_accepts=((0.5, 2.0), (0.9, 0.1)), positive_quantile=((0.5, 0.1), (0.9, 0.9))
    )
    h = make_harness([Pause()], calibration=calibration, solo=False)
    h.start_voice()

    h.post(endpoint_wake(1500, arrival=1000.0, score=0.7))
    await h.settle()

    (claim,) = h.arbiter.claims
    assert claim.rank == WakeRank(
        arrival=1000.0,
        false_accepts_per_hour=pytest.approx(1.05),
        positive_quantile=pytest.approx(0.5),
        margin=pytest.approx(0.2),
        score=0.7,
    )
    # The backlog completes well after `wake_arbitration_ms`; the claim made at `wake` waits.
    await h.clock.advance(400)
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint", score=0.7))
    await h.settle()
    assert calls(h.ingress, "request_gate_partial") == []
    h.arbiter.grant(claim.claim_id)
    await h.settle()
    assert len(calls(h.ingress, "request_gate_partial")) == 1


@pytest.mark.parametrize("backlog_ms", [0, 1499])
async def test_endpoint_wake_with_too_little_backlog_is_rejected_no_backlog(backlog_ms):
    h, pause, turn_id = await speaking_with_next()
    mark = len(h.outbound.client)

    h.post(endpoint_wake(backlog_ms))
    await h.settle()

    assert h.outbound.names()[mark:] == ["voice_decision:wake_rejected"]
    (rejected,) = h.outbound.of("voice_decision:wake_rejected")
    assert (rejected.reason, rejected.turn_id) == ("no_backlog", turn_id)
    assert rejected.detail.model_dump() == {
        "source": "endpoint",
        "score": 0.8,
        "reference_score": None,
    }
    assert calls(h.ingress, "expect_endpoint_wake") == []
    assert calls(h.ingress, "open_utterance") == []
    assert h.arbiter.claims == []
    assert h.responder.commits == [] and h.controller.phase == "speaking"
    pause.release()


async def test_endpoint_wake_while_speaking_interrupts_only_after_acceptance():
    h, pause, turn_id = await speaking_with_next([Pause()])

    h.post(endpoint_wake(1500))
    await h.settle()
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint"))
    await h.settle()
    assert h.responder.commits == []
    ((*_, uid),) = calls(h.ingress, "open_utterance")

    h.post(h.ingress.partial(uid, "mist, stop that", gate=True))
    await h.settle()

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.turn_id, barge.detail.utterance_id) == ("wake", turn_id, uid)
    assert h.outbound.of("voice_decision:supersede") == []
    assert [c.outcome for c in h.responder.commits] == ["interrupted"]
    pause.release()


# -- A name while she thinks supersedes the turn (A51, T55c) ------------------------------------------
# A name accepted while a model turn is in `thinking` with none of its audio queued supersedes it:
# `supersede`, the cut with `stream_cancelled superseded`, `idle` and `turn_metrics`, then the
# name's publish, all in the accepting step. K, the superseded turn, is committed at the cut only
# when it dispatched a tool; otherwise it is kept and joins the next turn (T55d, below).


def transcribed(h: Harness, uid: str) -> list[str]:
    """The texts of the `transcription`s naming `uid`."""
    return [t.text for t in h.outbound.of("transcription") if t.utterance_id == uid]


async def accept_a_name(h: Harness, how: str, text: str) -> str:
    """Accept a name through the server spotter or an endpoint `wake`; the candidate's id."""
    if how == "server":
        uid = h.ids()
        h.post(h.ingress.started(uid))
        await h.settle()
        await name_hit(h, uid)
        await gate(h, uid, text)
        return uid
    h.post(endpoint_wake(1500))
    await h.settle()
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint"))
    await h.settle()
    ((*_, uid),) = calls(h.ingress, "open_utterance")
    h.post(h.ingress.partial(uid, text, gate=True))
    await h.settle()
    return str(uid)


@pytest.mark.parametrize("how", ["server", "endpoint"])
async def test_a_name_accepted_in_thinking_before_any_audio_supersedes_the_turn(how):
    pause = Pause()
    h = make_harness([pause, Token("Late.")], [Token("In Paris it rains."), Pause()])
    h.start_voice()
    await h.speak("mist, what's the weather")
    turn_id = h.last_turn()
    assert h.controller.phase == "thinking"
    mark = len(h.outbound.client)

    uid = await accept_a_name(h, how, "mist, and in Paris")

    (supersede,) = h.outbound.of("voice_decision:supersede")
    assert (supersede.turn_id, supersede.reason) == (turn_id, "continued_speech")
    assert supersede.detail.model_dump() == {"merged": True, "tool_dispatched": False}
    assert h.outbound.of("voice_decision:barge_in") == []
    names = h.outbound.names()[mark:]
    # The pacer's stop ends the announced audio tag (`0x03`, `playback_stop`) within the cut.
    assert names[names.index("voice_decision:supersede") :] == [
        "voice_decision:supersede",
        "0x03",
        "playback_stop",
        "stream_cancelled",
        "state_cycle:idle",
        "turn_metrics",
        "vad_status:speech_started",
        "state_cycle:listen",
        "transcript_partial",
    ]
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert (cancelled.turn_id, cancelled.reason) == (turn_id, "superseded")
    assert (cancelled.partial_text, cancelled.spoken_text) == ("", "")
    # No tool was dispatched, so K is kept for the merge: its metrics say it is not recorded.
    (metrics,) = h.outbound.of("turn_metrics")
    assert (metrics.turn_id, metrics.outcome, metrics.recorded) == (turn_id, "superseded", False)
    assert h.responder.commits == []
    assert h.controller.phase == "idle"
    # The name's prefetch already carries K's words ahead of its own.
    assert h.responder.prefetches[-1] == (
        "session-1",
        "mist, what's the weather mist, and in Paris",
    )

    # U's end starts a non-held turn whose request carries both texts; the client sees U's own.
    h.post(h.ingress.ended(uid, "mist, and in Paris"))
    await h.settle()
    first, second = h.responder.requests
    assert first.user_text == "mist, what's the weather"
    assert second.user_text == "mist, what's the weather mist, and in Paris"
    assert not second.held
    assert second.interruption_note is None  # K is not recorded, so the note is left as it was
    assert transcribed(h, uid) == ["mist, and in Paris"]
    assert h.responder.commits == []
    assert h.outbound.of("voice_decision:supersede") == [supersede]
    pause.release()

    # One commit in all, U's turn's: K's words are recorded once, inside it.
    await h.controller.close()
    assert [entry for entry in h.responder.log if entry.startswith("commit:")] == [
        "commit:interrupted"
    ]


async def test_a_supersede_after_a_dispatched_tool_says_so():
    pause = Pause()
    h = make_harness(
        [WaitGate(), Call(dispatch_tool), tool_started(), pause, Token("Done.")], [Pause()]
    )
    h.start_voice()
    await h.speak("mist, set a timer")
    turn_id = h.last_turn()
    assert h.controller.phase == "thinking"

    uid = await accept_a_name(h, "server", "mist, make it ten minutes")

    (supersede,) = h.outbound.of("voice_decision:supersede")
    assert supersede.turn_id == turn_id
    assert supersede.detail.model_dump() == {"merged": False, "tool_dispatched": True}
    assert h.outbound.of("voice_decision:barge_in") == []
    (cancelled,) = h.outbound.of("stream_cancelled")
    assert cancelled.reason == "superseded"
    # K is committed at the cut, before its metrics, so they say it is recorded.
    (metrics,) = h.outbound.of("turn_metrics")
    assert (metrics.outcome, metrics.kind, metrics.recorded) == ("superseded", "tool", True)
    assert [(c.outcome, c.delivered_text) for c in h.responder.commits] == [("superseded", "")]
    assert h.responder.prefetches[-1] == ("session-1", "mist, make it ten minutes")

    # The turn from the name takes its own text, after K's record.
    h.post(h.ingress.ended(uid, "mist, make it ten minutes"))
    await h.settle()
    first, second = h.responder.requests
    assert (first.user_text, second.user_text) == ("mist, set a timer", "mist, make it ten minutes")
    assert second.interruption_note == "before_reply"
    assert [c.outcome for c in h.responder.commits] == ["superseded"]
    pause.release()


async def test_a_name_on_a_capability_answered_turn_in_thinking_still_barges_in():
    """Its text is already recorded (D8), so it is interrupted, never superseded."""
    alarm = FakeAlarm()
    h = make_harness(alarm_ringing=alarm)
    reply = "The alarm is off."
    h.responder.capability_results.append(
        CapabilityOnlyTurn(handled=True, events=(Token(reply), Complete(final_response=reply)))
    )
    h.lane.hold()  # no audio: the answer stays in `thinking`
    h.start_voice()
    stop = h.ids()
    h.post(h.ingress.started(stop), h.ingress.candidate(stop, "stop"))
    await h.settle()
    h.post(h.ingress.ended(stop, "stop"))
    await h.settle()
    answer = h.last_turn()
    assert h.controller.phase == "thinking"

    await accept_a_name(h, "server", "mist, what's the time")

    (barge,) = h.outbound.of("voice_decision:barge_in")
    assert (barge.reason, barge.turn_id, barge.detail.phase) == ("wake", answer, "thinking")
    assert h.outbound.of("voice_decision:supersede") == []
    assert h.outbound.of("stream_cancelled")[0].reason == "barge_in"
    assert h.outbound.of("turn_metrics")[-1].outcome == "interrupted"
    assert h.responder.commits == []  # the capability-only path recorded it
    h.lane.open()


async def test_a_name_accepted_while_a_turn_is_held_cancels_it_without_trace():
    h = make_harness([Token("One two. "), Pause()], [Pause()])
    h.start_voice()
    await held(h)

    await accept_a_name(h, "endpoint", "mist, what's up")

    assert h.outbound.of("voice_decision:supersede") == []
    assert h.outbound.of("voice_decision:barge_in") == []
    assert h.outbound.of("stream_cancelled") == []
    assert h.outbound.queued_turn_ids() == []
    assert h.responder.commits == []


# -- A superseded question joins the next one (A51, T55d) ------------------------------------------
# With no tool dispatched, K is kept, not committed, at the cut: every turn from U sends the
# responder K's request text, one space, then its own, and K is resolved when a non-held turn from
# U opens or a held one is released. Where U starts no turn, or another turn would be recorded
# first, K is committed alone, `superseded` with "".

K_TEXT = "mist, what's the weather"
U_TEXT = "mist, and in Paris"
MERGED = f"{K_TEXT} {U_TEXT}"


async def kept_turn(h: Harness, u_text: str = U_TEXT) -> str:
    """Start voice, supersede a thinking turn that dispatched no tool; U's utterance id."""
    h.start_voice()
    await h.speak(K_TEXT)
    assert h.controller.phase == "thinking"
    uid = await accept_a_name(h, "server", u_text)
    (supersede,) = h.outbound.of("voice_decision:supersede")
    assert supersede.detail.model_dump() == {"merged": True, "tool_dispatched": False}
    assert h.responder.commits == []
    return uid


def recorded(h: Harness) -> list[str]:
    """The responder's opens, commits and capability calls, in order."""
    return [e for e in h.responder.log if e == "open" or e.startswith(("commit:", "capability:"))]


def commits_of(h: Harness) -> list[tuple[str, str]]:
    return [(c.outcome, c.delivered_text) for c in h.responder.commits]


async def test_with_speculative_start_the_held_turn_from_u_carries_k_through_its_release():
    h = make_harness([Pause(), Token("Late.")], [Token("In Paris it rains. "), Pause()])
    uid = await kept_turn(h)

    h.post(h.ingress.candidate(uid, U_TEXT))
    await h.settle()

    assert h.controller.phase == "held"
    _, held_request = h.responder.requests
    assert (held_request.held, held_request.user_text) == (True, MERGED)
    assert h.responder.prefetches[-1] == ("session-1", MERGED)

    h.post(h.ingress.ended(uid, U_TEXT))
    await h.settle()

    # The release keeps the held request: no second request, and the client sees U's own text.
    (release,) = h.outbound.of("voice_decision:release")
    assert release.detail.utterance_id == uid
    assert [r.user_text for r in h.responder.requests] == [K_TEXT, MERGED]
    assert transcribed(h, uid) == [U_TEXT]
    assert h.responder.commits == []

    await h.controller.close()
    assert recorded(h) == ["open", "open", "commit:interrupted"]


async def test_a_held_turn_from_u_cancelled_by_resumed_speech_leaves_k_for_the_next_one():
    h = make_harness([Pause(), Token("Late.")], [Pause()], [Token("In Paris it rains. "), Pause()])
    uid = await kept_turn(h)
    h.post(h.ingress.candidate(uid, U_TEXT))
    await h.settle()

    h.post(h.ingress.resumed(uid))
    await h.settle()

    assert h.controller.phase == "idle"
    assert h.responder.commits == []
    longer = f"{U_TEXT} tomorrow"
    h.post(h.ingress.candidate(uid, longer))
    await h.settle()
    h.post(h.ingress.ended(uid, longer))
    await h.settle()

    # Each held turn from U carries K once; the second is released.
    _, cancelled, released = h.responder.requests
    assert (cancelled.held, cancelled.user_text) == (True, MERGED)
    assert (released.held, released.user_text) == (True, f"{K_TEXT} {longer}")
    assert len(h.outbound.of("voice_decision:release")) == 1
    assert transcribed(h, uid) == [longer]
    assert h.responder.commits == []

    await h.controller.close()
    assert recorded(h) == ["open", "open", "open", "commit:interrupted"]


@pytest.mark.parametrize("final", ["stop", "mist, stop"])
async def test_mist_stop_with_speculative_start_off_starts_a_turn_from_u_that_merges_k(final):
    """U ends in `idle`, where no keyword verdict cuts anything, so it starts a turn from U.

    "mist, stop" is accepted on the gate's decode; its final transcript may or may not keep the
    name, and either way the end in `idle` starts a turn.
    """
    h = make_harness([Pause(), Token("Late.")], [Pause()])
    uid = await kept_turn(h, "mist, stop")

    h.post(h.ingress.ended(uid, final))
    await h.settle()

    _, second = h.responder.requests
    assert (second.held, second.user_text) == (False, f"{K_TEXT} {final}")
    assert transcribed(h, uid) == [final]
    assert h.outbound.of("voice_decision:dropped") == []
    assert h.responder.commits == []

    await h.controller.close()
    assert recorded(h) == ["open", "open", "commit:interrupted"]


async def test_mist_stop_on_a_held_turn_from_u_whose_final_keeps_the_name_cancels_it():
    """In `held` one leading name then a keyword alone cancels (`KeywordBargeIn._judge_whole`),
    as the keyword alone does: U closes `dropped` `keyword` and K is recorded on its own, once.
    """
    h = make_harness([Pause(), Token("Late.")], [Pause()], [Pause()])
    uid = await kept_turn(h, "mist, stop")

    await _stop_with_the_name_on_the_held_turn(h, uid)

    assert h.outbound.of("voice_decision:release") == []
    assert [r.user_text for r in h.responder.requests] == [K_TEXT, f"{K_TEXT} mist, stop"]
    assert commits_of(h) == [("superseded", "")]
    await h.controller.close()
    assert commits_of(h) == [("superseded", "")]


async def _stop_with_the_name_on_the_held_turn(h: Harness, uid: str) -> None:
    """The name and `stop` with speculative start on, the final transcript keeping the name."""
    h.post(h.ingress.candidate(uid, "mist, stop"))
    await h.settle()
    assert h.controller.phase == "held"
    assert h.responder.requests[-1].user_text == f"{K_TEXT} mist, stop"
    h.post(h.ingress.ended(uid, "mist, stop"))
    await h.settle()
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("keyword", uid)


async def _stop_on_the_held_turn(h: Harness, uid: str) -> None:
    """The name and `stop` with speculative start on, the final transcript the keyword alone."""
    h.post(h.ingress.candidate(uid, "stop"))
    await h.settle()
    assert h.controller.phase == "held"
    assert h.responder.requests[-1].user_text == f"{K_TEXT} stop"
    h.post(h.ingress.ended(uid, "stop"))
    await h.settle()
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("keyword", uid)


async def _mute(h: Harness, uid: str) -> None:
    h.post(h.ingress.dropped(uid, "muted"), Mute(active=True))
    await h.settle()
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("muted", uid)


async def _voice_stop(h: Harness, uid: str) -> None:
    h.post(VoiceStop())
    await h.settle()
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)


async def _voice_start_over_an_adopted_ingress(h: Harness, uid: str) -> None:
    h.post(new_start(h))
    await h.settle()
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("voice_stop", uid)


async def _typed_while_held(h: Harness, uid: str) -> None:
    h.post(h.ingress.candidate(uid, U_TEXT))
    await h.settle()
    assert h.controller.phase == "held"
    h.post(Text("what's on today"))
    await h.settle()
    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert (dropped.reason, dropped.detail.utterance_id) == ("text_input", uid)
    # K is recorded before the typed turn opens.
    assert recorded(h) == ["open", "open", "commit:superseded", "open"]
    assert h.responder.requests[-1].user_text == "what's on today"


async def _close(h: Harness, uid: str) -> None:
    await h.controller.close()


async def _cancel(h: Harness, uid: str) -> None:
    task = h.controller.task
    assert task is not None
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.parametrize(
    "end_u",
    [
        _stop_on_the_held_turn,
        _stop_with_the_name_on_the_held_turn,
        _mute,
        _voice_stop,
        _voice_start_over_an_adopted_ingress,
        _typed_while_held,
        _close,
        _cancel,
    ],
    ids=[
        "mist-stop-speculative",
        "mist-stop-with-the-name",
        "mute",
        "voice_stop",
        "voice_start",
        "typed-while-held",
        "close",
        "cancel",
    ],
)
async def test_k_is_committed_alone_once_where_u_starts_no_turn(end_u):
    h = make_harness([Pause(), Token("Late.")], [Pause()], [Pause()])
    uid = await kept_turn(h, "mist, stop")

    await end_u(h, uid)

    assert commits_of(h) == [("superseded", "")]
    assert h.responder.commits[0].tool_effect_unknown is False
    task = h.controller.task
    if task is not None and not task.done():
        await h.controller.close()
    # Exactly once: nothing later records K again.
    assert [entry for entry in recorded(h) if entry == "commit:superseded"] == ["commit:superseded"]


async def test_an_alarm_stop_word_records_k_before_the_capability_path_is_asked():
    alarm = FakeAlarm()
    h = make_harness([Pause(), Token("Late.")], [Pause()], alarm_ringing=alarm)
    uid = await kept_turn(h, "mist, stop")
    h.post(h.ingress.candidate(uid, "mist, stop"))
    await h.settle()
    assert h.controller.phase == "held"

    h.post(h.ingress.ended(uid, "mist, stop"))
    await h.settle()

    assert h.responder.capability_calls == [("mist, stop", "session-1")]
    assert recorded(h) == ["open", "open", "commit:superseded", "capability:mist, stop"]
    assert commits_of(h) == [("superseded", "")]
    await h.controller.close()
    assert commits_of(h) == [("superseded", "")]


async def test_a_typed_message_while_u_is_open_records_k_first_and_u_then_takes_its_own_text():
    h = make_harness([Pause(), Token("Late.")], [Pause()], [Token("In Paris it rains. "), Pause()])
    uid = await kept_turn(h)

    h.post(Text("what's on today"))
    await h.settle()

    # K is recorded before the typed turn opens, so its record precedes the typed turn's.
    assert recorded(h) == ["open", "commit:superseded", "open"]
    assert h.controller.phase == "text"

    h.post(h.ingress.ended(uid, U_TEXT))
    await h.settle()

    # U's end cuts the typed turn and starts a turn from U with its own text alone.
    assert [r.user_text for r in h.responder.requests] == [K_TEXT, "what's on today", U_TEXT]
    assert h.outbound.of("stream_cancelled")[-1].reason == "new_input"
    assert commits_of(h) == [("superseded", ""), ("interrupted", "")]
    assert transcribed(h, uid) == [U_TEXT]
    await h.controller.close()
    assert [c.outcome for c in h.responder.commits] == ["superseded", "interrupted", "interrupted"]


async def test_a_wake_while_muted_is_dropped_muted_with_a_fresh_id():
    h = make_harness([Pause()])
    h.start_voice()
    first = h.ids()  # an id ingress issued before the wake
    h.post(Mute(active=True), endpoint_wake(1500))
    await h.settle()

    (dropped,) = h.outbound.of("voice_decision:dropped")
    assert h.outbound.names() == ["voice_decision:dropped"]
    detail = dropped.detail
    assert (dropped.reason, detail.duration_ms, detail.text) == ("muted", 0, None)
    assert (detail.no_speech_prob, detail.avg_logprob) == (None, None)
    assert detail.utterance_id == h.ids.issued[-1] != first
    assert h.ids() != detail.utterance_id
    assert calls(h.ingress, "open_utterance") == []
    assert calls(h.ingress, "expect_endpoint_wake") == []
    assert h.arbiter.claims == []
    assert h.scene.calls == []
    assert h.responder.requests == []


@pytest.mark.parametrize("muted", [True, False])
async def test_a_wake_with_no_voice_session_is_ignored(muted):
    h = make_harness([Pause()])
    if muted:
        h.post(Mute(active=True))
    h.post(endpoint_wake(1500))
    await h.settle()

    assert h.outbound.client == []
    assert h.ids.issued == []


# -- The follow-up window, no `attend`, `set_engaged` and the wake timings (v5 :1734-1744) --------


async def named_turn(h: Harness, text: str = "Mist, what's up?") -> str:
    """Accept a name in idle, end the utterance and finish the turn it opens; returns its id."""
    uid = await name_hit(h)
    await gate(h, uid, "mist what's up")
    h.post(h.ingress.ended(uid, text))
    await h.settle()
    turn_id = h.last_turn()
    h.post(PlaybackDone(turn_id=turn_id, played_ms=1))
    await h.settle()
    assert h.controller.phase == "idle"
    return turn_id


async def test_set_engaged_is_true_at_acceptance_and_false_at_the_turns_end_with_no_window():
    h = make_harness([Token("Sure.")])
    h.start_voice()
    uid = await name_hit(h)
    assert engaged_calls(h) == []

    await gate(h, uid, "mist what's up")
    assert engaged_calls(h) == [True]
    h.post(h.ingress.ended(uid, "Mist, what's up?"))
    await h.settle()
    assert engaged_calls(h) == [True]

    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    await h.settle()
    assert engaged_calls(h) == [True, False]


async def test_followup_window_unnamed_speech_two_seconds_after_playback_done_opens_a_turn():
    h = make_harness([Token("Sure.")], [Pause()], wake=WakeConfig(followup_window_ms=6000))
    h.start_voice()
    await named_turn(h)
    assert engaged_calls(h) == [True]
    await h.clock.advance(2000)
    mark = len(h.outbound.client)
    uid = h.ids()

    h.post(h.ingress.started(uid))
    await h.settle()
    assert h.outbound.client[mark:] == []  # unpublished until checks a and b pass
    h.post(h.ingress.partial(uid, "and tomorrow"))
    await h.settle()
    assert h.outbound.names()[mark:] == [
        "vad_status:speech_started",
        "state_cycle:listen",
        "transcript_partial",
    ]
    h.post(h.ingress.ended(uid, "And tomorrow?"))
    await h.settle()

    assert [r.user_text for r in h.responder.requests] == ["Mist, what's up?", "And tomorrow?"]
    assert h.outbound.of("state_cycle:attend") == []  # neither the name nor the follow-up


async def test_followup_window_rejects_silently_when_her_recent_speech_holds_her_name():
    spoken = {"text": ""}
    h = make_harness(
        [Token("Sure.")],
        wake=WakeConfig(followup_window_ms=6000),
        recent_tts_text=lambda _ms: spoken["text"],
    )
    h.start_voice()
    await named_turn(h)
    spoken["text"] = "This is Mist."
    mark = len(h.outbound.client)
    uid = h.ids()

    h.post(h.ingress.started(uid), h.ingress.partial(uid, "this is mist"))
    await h.settle()

    assert h.outbound.client[mark:] == []
    assert calls(h.ingress, "drop") == [("drop", uid, "not_addressed")]
    assert len(h.responder.requests) == 1


async def test_followup_window_lapses_after_its_length_and_ingress_is_dormant_again():
    h = make_harness([Token("Sure.")], wake=WakeConfig(followup_window_ms=6000))
    h.start_voice()
    await named_turn(h)

    await h.clock.advance(5999)
    assert engaged_calls(h) == [True]
    await h.clock.advance(1001)  # 7 s after `playback_done`

    assert engaged_calls(h) == [True, False]
    assert len(h.responder.requests) == 1


async def test_with_no_followup_window_ingress_is_dormant_at_once_after_the_turn():
    h = make_harness([Token("Sure.")])
    h.start_voice()
    await named_turn(h)
    assert engaged_calls(h) == [True, False]

    await h.clock.advance(60_000)
    assert engaged_calls(h) == [True, False]
    assert len(h.responder.requests) == 1


async def test_a_rejected_name_sends_no_attend_and_no_state():
    h = make_harness([Pause()])
    h.start_voice()
    uid = await name_hit(h)

    await gate(h, uid, "I missed the bus")

    assert h.outbound.names() == ["voice_decision:wake_rejected"]


async def test_turn_metrics_carry_the_wake_timings_of_a_name_opened_turn():
    h = make_harness([Token("Sure.")], hit_lag_ms=150)
    h.start_voice()
    h.clock.now = 1040  # the hit's last frame arrived at 1000; the mailbox reached it 40 ms later

    uid = await name_hit(h, arrival=1000.0)
    await h.clock.advance(260)
    await gate(h, uid, "mist what's up")
    h.post(h.ingress.ended(uid, "Mist, what's up?"))
    await h.settle()
    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    await h.settle()

    (metrics,) = h.outbound.of("turn_metrics")
    assert metrics.timings["wake_accept_ms"] == 300
    # The name ended `hit_lag_ms` before the hit's arrival; the screen ends at acceptance.
    assert metrics.timings["wake_screen_ms"] == 150 + 300


async def test_a_turn_not_opened_by_a_name_has_no_wake_timings():
    h = make_harness([Token("Sure.")])
    h.start_voice()
    await h.speak("hello")
    h.post(PlaybackDone(turn_id=h.last_turn(), played_ms=1))
    await h.settle()

    (metrics,) = h.outbound.of("turn_metrics")
    assert "wake_accept_ms" not in metrics.timings and "wake_screen_ms" not in metrics.timings


# -- Mute and unpublished candidates (v8 :248-270) ---------------------------------------------------


async def test_while_speaking_a_name_then_mute_unducks_with_no_state_and_the_reply_goes_on():
    h, pause, turn_id = await speaking_with_next()
    uid = h.ids()
    h.post(h.ingress.started(uid))
    await h.settle()
    await name_hit(h, uid)
    mark = len(h.outbound.client)

    # Ingress's drop for the utterance open at mute is ahead of `Mute` in the mailbox (T12).
    h.post(h.ingress.dropped(uid, "muted"), Mute(active=True))
    await h.settle()

    assert h.outbound.names()[mark:] == ["playback_duck"]
    assert ducks(h) == [12.0, 0.0]
    assert h.outbound.of("voice_decision:wake_rejected") == []
    assert h.outbound.of("voice_decision:dropped") == []
    assert h.arbiter.settles == [(1, False)]
    pause.release()
    await h.settle()
    assert h.outbound.tokens(turn_id)[-1] == "Six."
    assert h.responder.commits == []


async def test_a_waiting_endpoint_wake_abandoned_at_mute_sends_nothing():
    h = make_harness([Pause()])
    h.start_voice()
    h.post(endpoint_wake(1500))
    await h.settle()

    h.post(Mute(active=True))
    await h.settle()

    assert h.outbound.names() == []
    assert h.arbiter.settles == [(1, False)]
    # The backlog's `NameHeard`, should one still come, has no wake waiting for it.
    stale = h.controller.stale_events
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint"))
    await h.settle()
    assert h.controller.stale_events == stale + 1
    assert calls(h.ingress, "open_utterance") == []


async def test_a_name_heard_queued_before_mute_closes_through_the_immediate_drop_with_no_message():
    h = make_harness([Pause()])
    h.start_voice()
    h.post(Mute(active=True))

    uid = await name_hit(h)  # a muted ingress mints the id and posts its drop after this step
    h.post(h.ingress.dropped(uid, "muted"))
    await h.settle()

    assert h.outbound.names() == []
    assert h.arbiter.settles == [(1, False)]
    assert h.responder.requests == []


async def test_an_endpoint_name_heard_with_no_waiting_wake_opens_nothing_and_is_counted():
    h = make_harness([Pause()])
    h.start_voice()
    stale = h.controller.stale_events

    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint"))
    await h.settle()

    assert h.outbound.client == []
    assert h.ingress.calls == []
    assert h.arbiter.claims == []
    assert h.controller.stale_events == stale + 1


# -- Verdict handling (v9 :153-163, :296-300) ---------------------------------------------------------


@pytest.mark.parametrize("drop", ["muted", "noise", "text", "voice_stop", "disconnect"])
async def test_a_dropped_candidate_settles_its_claim_once_with_accepted_false(drop):
    h = make_harness([Pause()], solo=False)
    h.start_voice()
    uid = await name_hit(h)

    if drop in ("muted", "noise"):
        h.post(h.ingress.dropped(uid, drop))
    elif drop == "text":
        h.post(Text("typed"))
    elif drop == "voice_stop":
        h.post(VoiceStop())
    else:
        h.post(Close())
    await h.settle()

    assert h.arbiter.settles == [(1, False)]
    assert h.outbound.of("voice_decision:wake_rejected") == []
    assert h.outbound.of("voice_decision:dropped") == []
    if drop == "text":
        assert calls(h.ingress, "drop") == [("drop", uid, "text_input")]
    else:
        assert calls(h.ingress, "drop") == []


def losses(h: Harness) -> list[VoiceDecisionWakeLost]:
    return h.outbound.of("voice_decision:wake_lost")  # type: ignore[return-value]


async def test_a_lost_claim_closes_its_candidate_with_one_wake_lost_and_settles_once():
    h = make_harness([Pause()], solo=False)
    h.start_voice()
    uid = await name_hit(h, score=0.6)

    h.arbiter.deny(1, winner_score=0.93)
    await h.settle()

    assert h.outbound.names() == ["voice_decision:wake_lost"]
    (message,) = losses(h)
    assert message.turn_id is None
    assert message.reason == "arbitration"
    assert message.detail == WakeLostDetail(winner_score=0.93, score=0.6, utterance_id=uid)
    assert calls(h.ingress, "drop") == [("drop", uid, "wake_lost")]
    assert calls(h.ingress, "request_gate_partial") == []
    assert h.arbiter.settles == [(1, False)]


@pytest.mark.parametrize("phase", ["thinking", "speaking"])
async def test_a_lost_claim_in_a_turn_cuts_nothing_and_sends_no_state(phase):
    h, pause = await in_phase(phase)
    h.arbiter.solo = False
    turn_id = h.last_turn()
    uid = h.ids()
    h.post(h.ingress.started(uid))
    await h.settle()
    mark = len(h.outbound.client)

    await name_hit(h, uid, score=0.6)
    h.arbiter.deny(h.arbiter.claims[-1].claim_id, winner_score=0.9)
    await h.settle()

    duck = ["playback_duck"] if phase == "speaking" else []
    assert h.outbound.names()[mark:] == [*duck, "voice_decision:wake_lost", *duck]
    assert ducks(h) == ([12.0, 0.0] if phase == "speaking" else [])
    (message,) = losses(h)
    assert message.turn_id == turn_id
    assert message.detail == WakeLostDetail(winner_score=0.9, score=0.6, utterance_id=uid)
    assert h.outbound.of("voice_decision:wake_rejected") == []
    assert h.outbound.of("state_cycle:idle") == []
    assert h.outbound.purges == []
    assert h.responder.commits == []
    assert h.controller.phase == phase
    assert calls(h.ingress, "drop") == [("drop", uid, "wake_lost")]
    pause.release()


async def test_an_endpoint_wake_that_loses_before_its_backlog_gets_a_fresh_id():
    h = make_harness([Pause()], solo=False)
    h.start_voice()
    h.post(endpoint_wake(1500, score=0.6))
    await h.settle()
    (claim,) = h.arbiter.claims

    h.arbiter.deny(claim.claim_id, winner_score=0.9)
    await h.settle()

    assert h.outbound.names() == ["voice_decision:wake_lost"]
    (message,) = losses(h)
    assert message.reason == "arbitration"
    assert message.turn_id is None
    assert message.detail.winner_score == 0.9
    assert message.detail.score == 0.6
    uid = message.detail.utterance_id
    # No ingress lever ever named it: the id is fresh, and nothing opened to drop.
    named = [call for call in h.ingress.calls if uid in call]
    assert named == []
    assert calls(h.ingress, "open_utterance") == []
    assert calls(h.ingress, "drop") == []
    assert h.arbiter.settles == [(claim.claim_id, False)]

    # The backlog's `NameHeard` comes after the loss: no wake waits for it, so it is stale.
    stale, mark = h.controller.stale_events, len(h.outbound.sent)
    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint", score=0.6))
    await h.settle()

    assert h.controller.stale_events == stale + 1
    assert h.outbound.sent[mark:] == []
    assert calls(h.ingress, "open_utterance") == []


async def test_a_verdict_for_a_settled_claim_is_ignored_counted_and_sends_nothing():
    h = make_harness([Pause()], solo=False)
    h.start_voice()
    uid = await name_hit(h)
    h.post(h.ingress.dropped(uid, "noise"))
    await h.settle()
    mark, stale = len(h.outbound.sent), h.controller.stale_events

    h.arbiter.grant(1)  # the verdict was already in the mailbox when the drop was handled
    await h.settle()

    assert h.outbound.sent[mark:] == []
    assert h.controller.stale_events == stale + 1
    assert calls(h.ingress, "request_gate_partial") == []
    assert h.arbiter.settles == [(1, False)]


async def test_a_winning_endpoint_wake_requests_its_gate_partial_when_its_name_heard_comes():
    h = make_harness([Pause()])
    h.start_voice()
    h.post(endpoint_wake(1500))
    await h.settle()
    assert [c.answered for c in h.arbiter.claims] == [True]  # the verdict came first
    assert calls(h.ingress, "request_gate_partial") == []

    h.post(h.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint"))
    await h.settle()

    ((*_, uid),) = calls(h.ingress, "open_utterance")
    assert calls(h.ingress, "request_gate_partial") == [("request_gate_partial", uid, 99 + 15)]


async def test_an_accepted_and_a_rejected_claim_are_each_settled_exactly_once():
    h = make_harness([Pause()])
    h.start_voice()
    rejected = await name_hit(h)
    await gate(h, rejected, "I missed the bus")
    accepted = await name_hit(h)
    await gate(h, accepted, "mist, what's up")
    h.post(h.ingress.ended(accepted, "Mist, what's up?"))
    await h.settle()

    assert h.arbiter.settles == [(1, False), (2, True)]


# == T11e part 2: arbitration through two controllers (v5 :1709-1722) ================================
# Two sessions' controllers share one real `WakeArbiter`, whose window a fake `call_later` closes
# when the test fires it. The window is `wake_arbitration_ms`, 250.

FLOOR = SpotterCalibration(
    # Every score from 0.7 up clamps to the false-accept floor, 0.1 an hour.
    false_accepts=((0.5, 3.0), (0.7, 0.1)),
    positive_quantile=((0.5, 0.0), (1.0, 1.0)),
)


@dataclass
class TwoSessions:
    a: Harness
    b: Harness
    arbiter: WakeArbiter
    later: FakeCallLater
    count: list[int]

    async def settle(self) -> None:
        await self.a.settle()

    async def close_window(self) -> None:
        self.later.fire()
        await self.settle()


def two_sessions(
    *,
    cal_a: SpotterCalibration | None = None,
    cal_b: SpotterCalibration | None = None,
    sessions: int = 2,
) -> TwoSessions:
    count = [sessions]
    later = FakeCallLater()
    shared = WakeArbiter(window_ms=250, sessions=lambda: count[0], call_later=later)
    a = make_harness([Pause()], arbiter=shared, session_id="desk", id_prefix="a", calibration=cal_a)
    b = make_harness([Pause()], arbiter=shared, session_id="room", id_prefix="b", calibration=cal_b)
    a.start_voice()
    b.start_voice()
    return TwoSessions(a, b, shared, later, count)


def rejected_reasons(h: Harness) -> list[str]:
    return [item.reason for item in h.outbound.of("voice_decision:wake_rejected")]


def lost_details(h: Harness) -> list[tuple[str, float, float, str]]:
    """Each `wake_lost`'s reason, winner score, own score and utterance id."""
    return [
        (m.reason, m.detail.winner_score, m.detail.score, m.detail.utterance_id) for m in losses(h)
    ]


async def test_two_calibrated_hits_the_lower_false_accept_rate_proceeds_over_a_larger_margin():
    # The two tables disagree with the margins: a's 0.6 (margin 0.1) reads 0.2 false accepts an
    # hour, b's 0.9 (margin 0.4) reads 1.0.
    cal_a = SpotterCalibration(
        false_accepts=((0.5, 0.4), (0.7, 0.0)), positive_quantile=((0.5, 0.2), (0.7, 0.6))
    )
    cal_b = SpotterCalibration(
        false_accepts=((0.8, 1.2), (1.0, 0.8)), positive_quantile=((0.8, 0.6), (1.0, 0.9))
    )
    s = two_sessions(cal_a=cal_a, cal_b=cal_b)
    ua = await name_hit(s.a, score=0.6, arrival=100.0)
    ub = await name_hit(s.b, score=0.9, arrival=0.0)

    # Both wait in the window; neither decodes its gate partial yet.
    assert calls(s.a.ingress, "request_gate_partial") == []
    assert calls(s.b.ingress, "request_gate_partial") == []
    await s.close_window()
    assert calls(s.a.ingress, "request_gate_partial") == [
        ("request_gate_partial", ua, GATE_THROUGH)
    ]
    assert calls(s.b.ingress, "request_gate_partial") == []
    # The loser waits for the winner's verdict: nothing is sent to it yet.
    assert s.b.outbound.names() == []

    await gate(s.a, ua, "mist what's up")

    assert "vad_status:speech_started" in s.a.outbound.names()
    assert lost_details(s.b) == [("arbitration", 0.6, 0.9, ub)]
    assert s.b.outbound.names() == ["voice_decision:wake_lost"]
    assert calls(s.b.ingress, "drop") == [("drop", ub, "wake_lost")]
    assert calls(s.b.ingress, "request_gate_partial") == []


async def test_two_hits_at_the_false_accept_floor_the_higher_quantile_proceeds_though_second():
    # The desktop's server hit arrives first; the room unit's `wake` arrives 120 ms later with
    # the higher positive quantile (0.9 against 0.7), and both read the 0.1 floor.
    s = two_sessions(cal_a=FLOOR, cal_b=FLOOR)
    ua = await name_hit(s.a, score=0.85, arrival=1000.0)
    s.b.post(Wake(keyword="mist", score=0.95, backlog_ms=1500, arrival=1120.0))
    await s.settle()

    await s.close_window()
    assert calls(s.a.ingress, "request_gate_partial") == []
    # The room unit won before its backlog arrived; its gate partial follows its `NameHeard`.
    s.b.post(s.b.ingress.heard(None, from_pos=60, hit_pos=99, spotter="endpoint", score=0.95))
    await s.settle()
    ((*_, ub),) = calls(s.b.ingress, "open_utterance")
    assert calls(s.b.ingress, "request_gate_partial") == [("request_gate_partial", ub, 99 + 15)]

    await gate(s.b, ub, "mist what's the time")

    assert lost_details(s.a) == [("arbitration", 0.95, 0.85, ua)]
    assert rejected_reasons(s.a) == []
    assert calls(s.a.ingress, "drop") == [("drop", ua, "wake_lost")]
    assert rejected_reasons(s.b) == []
    assert losses(s.b) == []


async def test_two_hits_equal_on_both_scales_the_earlier_arrival_proceeds():
    s = two_sessions(cal_a=FLOOR, cal_b=FLOOR)
    ua = await name_hit(s.a, score=0.8, arrival=60.0)  # claimed first, arrived later
    ub = await name_hit(s.b, score=0.8, arrival=10.0)

    await s.close_window()

    assert calls(s.a.ingress, "request_gate_partial") == []
    assert calls(s.b.ingress, "request_gate_partial") == [
        ("request_gate_partial", ub, GATE_THROUGH)
    ]
    await gate(s.b, ub, "mist, are you there")
    assert lost_details(s.a) == [("arbitration", 0.8, 0.8, ua)]
    assert rejected_reasons(s.a) == []
    assert calls(s.a.ingress, "drop") == [("drop", ua, "wake_lost")]


@pytest.mark.parametrize(
    ("cal_a", "score_a", "cal_b", "score_b", "winner"),
    [
        # A calibrated hit outranks an uncalibrated one with the far larger margin.
        (None, 0.99, FLOOR, 0.71, "b"),
        # Two uncalibrated hits rank by margin, whatever their arrival.
        (None, 0.6, None, 0.9, "b"),
        (None, 0.9, None, 0.6, "a"),
    ],
    ids=["calibrated-over-uncalibrated", "uncalibrated-larger-margin-b", "uncalibrated-a"],
)
async def test_calibrated_outranks_uncalibrated_and_uncalibrated_ranks_by_margin(
    cal_a, score_a, cal_b, score_b, winner
):
    s = two_sessions(cal_a=cal_a, cal_b=cal_b)
    ua = await name_hit(s.a, score=score_a, arrival=0.0)
    ub = await name_hit(s.b, score=score_b, arrival=50.0)

    await s.close_window()

    won, lost = (s.a, s.b) if winner == "a" else (s.b, s.a)
    won_uid, lost_uid = (ua, ub) if winner == "a" else (ub, ua)
    won_score, lost_score = (score_a, score_b) if winner == "a" else (score_b, score_a)
    assert calls(won.ingress, "request_gate_partial") == [
        ("request_gate_partial", won_uid, GATE_THROUGH)
    ]
    assert calls(lost.ingress, "request_gate_partial") == []
    await gate(won, won_uid, "mist what's up")
    assert rejected_reasons(won) == []
    assert losses(won) == []
    assert rejected_reasons(lost) == []
    assert lost_details(lost) == [("arbitration", won_score, lost_score, lost_uid)]


async def test_when_the_winner_is_rejected_the_runner_up_requests_its_gate_partial():
    s = two_sessions()
    ua = await name_hit(s.a, score=0.9, arrival=0.0)
    ub = await name_hit(s.b, score=0.7, arrival=30.0)
    await s.close_window()
    assert calls(s.b.ingress, "request_gate_partial") == []

    await gate(s.a, ua, "I missed the bus")
    await s.settle()

    assert rejected_reasons(s.a) == ["not_addressed"]
    assert calls(s.b.ingress, "request_gate_partial") == [
        ("request_gate_partial", ub, GATE_THROUGH)
    ]
    await gate(s.b, ub, "mist what's up")
    assert rejected_reasons(s.b) == []
    assert "vad_status:speech_started" in s.b.outbound.names()
    # The rejected winner's claim is settled, so the runner-up's acceptance sends it no loss.
    assert losses(s.a) == []


async def test_only_the_winners_gate_partial_is_requested_while_both_wait():
    s = two_sessions()
    ua = await name_hit(s.a, score=0.9, arrival=0.0)
    await name_hit(s.b, score=0.7, arrival=30.0)

    # Inside the window neither decodes; after it only the winner does, and the loser keeps
    # waiting through any later events until the winner's verdict.
    assert calls(s.a.ingress, "request_gate_partial") == []
    assert calls(s.b.ingress, "request_gate_partial") == []
    await s.close_window()
    await s.a.clock.advance(1000)
    await s.b.clock.advance(1000)

    assert calls(s.a.ingress, "request_gate_partial") == [
        ("request_gate_partial", ua, GATE_THROUGH)
    ]
    assert calls(s.b.ingress, "request_gate_partial") == []
    assert rejected_reasons(s.b) == []


async def test_with_one_session_no_wait_occurs():
    s = two_sessions(sessions=1)
    s.b.post(VoiceStop())  # one voice session; the other harness takes no part
    await s.settle()

    uid = await name_hit(s.a)

    # The gate partial is requested in the name's own step: the clock never moved, and no window
    # was opened.
    assert s.a.clock.now == 0
    assert s.later.timers == []
    assert calls(s.a.ingress, "request_gate_partial") == [
        ("request_gate_partial", uid, GATE_THROUGH)
    ]


async def test_a_dropped_waiting_candidate_withdraws_its_claim_and_the_other_proceeds_alone():
    s = two_sessions()
    ua = await name_hit(s.a, score=0.9, arrival=0.0)
    ub = await name_hit(s.b, score=0.7, arrival=30.0)

    s.a.post(s.a.ingress.dropped(ua, "noise"))  # the better candidate is dropped in the window
    await s.settle()
    await s.close_window()

    assert calls(s.b.ingress, "request_gate_partial") == [
        ("request_gate_partial", ub, GATE_THROUGH)
    ]
    assert calls(s.a.ingress, "request_gate_partial") == []
    assert s.a.outbound.of("voice_decision:wake_rejected") == []
