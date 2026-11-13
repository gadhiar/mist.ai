"""The turn controller's state table rows for `VoiceStart`, `voice_stop` and `Reconfigure`, mute
for a published utterance (T11d, v2 :1704-1717 and :1736-1784, v3 :1681-1682, v8 :239-247), and
the speculative held turn: its start, buffer, release and cancel (v1 :400-423, :2714-2717), M65
from `held` at its release and M67's `release` (v2 :392-408, :1615-1637), and v2's `held` column
(v2 :476-493) where v5 :1174-1189 leaves it. Then the rest of v2's table in `idle`, `thinking`,
`speaking` and `text` that v5 keeps, the keyword row's closing messages (v2 :617-651), ignored
endpoint reports (v2 :1844-1847), and Test strategy's variants that survive v5 (v2 :1117-1130).

Every test drives the controller only through `post` and `close`, with the fakes from
`turn_fakes.py`. The client-facing record is what the recording transport's sender took.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from backend.chat.stream_events import Token
from backend.realtime.budget import BargeInConfig
from backend.realtime.turn import (
    BargeInReport,
    Interrupt,
    Mute,
    PlaybackDone,
    Reconfigure,
    Text,
    VoiceStart,
    VoiceStop,
)

from .turn_fakes import (
    Call,
    FailingSynthesizer,
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
