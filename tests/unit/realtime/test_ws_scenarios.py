"""T16: end-to-end voice scenarios over `/ws`.

Each scenario runs through Starlette's `TestClient` against the real server lifespan, with
`build_mist_runtime` replaced by the fakes in `conftest.py`: real sessions, real ingress, the real
turn controller, scripted speech engines and a scripted responder. Every scenario ends with
`assert_protocol_rules`, the cross-cutting rules every connection keeps:

- each utterance id a `transcript_partial` names gets exactly one closing message, and nothing
  names it afterwards;
- each `voice_start` gets exactly one `voice_state` with `cause` `answer`, carrying its `req`,
  on that connection, in send order;
- each `audio_tag` a `stream_start` announces gets exactly one `0x02` or `0x03`, no frame of that
  tag follows it, and a cut's `0x03` comes before its `playback_stop`.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from backend import server
from backend.chat.conversation_handler import CapabilityOnlyTurn
from backend.chat.stream_events import Complete, Token
from backend.gpu_yield import MARKER_NAME, YIELDED_MESSAGE, turn_refusal
from backend.realtime.errors import SpeechEngineError
from backend.realtime.protocol import Frame, MsgType
from tests.fakes.scripted_audio import silence, speech_burst

from .conftest import VoiceClient, VoiceStack, voice_start
from .turn_fakes import Pause

AUDIO_COMPLETE = 0x02
AUDIO_FADE = 0x03

# "mist, ..." said after 200 ms of room tone: the name ends 200 ms into a 1.2 s burst.
NAMED = [silence(200), speech_burst(1200), silence(600)]
NAME_AT_MS = 400


# -- The rules every connection keeps ------------------------------------------------------------


def _named_utterance(item: dict[str, object]) -> str | None:
    if "utterance_id" in item:
        return item["utterance_id"]  # type: ignore[return-value]
    detail = item.get("detail")
    if isinstance(detail, dict):
        return detail.get("utterance_id")
    return None


def _closes(item: dict[str, object]) -> str | None:
    if item.get("type") == "transcription":
        return item["utterance_id"]  # type: ignore[return-value]
    if item.get("type") == "voice_decision" and item.get("kind") in ("dropped", "wake_lost"):
        return item["detail"]["utterance_id"]  # type: ignore[index]
    return None


def assert_protocol_rules(client: VoiceClient) -> None:
    items = client.voice
    texts = [(index, item) for index, item in enumerate(items) if isinstance(item, dict)]

    partial_ids = {
        item["utterance_id"] for _, item in texts if item["type"] == "transcript_partial"
    }
    for utterance_id in partial_ids:
        closing = [index for index, item in texts if _closes(item) == utterance_id]
        naming = [index for index, item in texts if _named_utterance(item) == utterance_id]
        assert len(closing) == 1, f"{utterance_id} closed {len(closing)} times"
        assert naming[-1] == closing[0], f"{utterance_id} named after its closing message"

    answers = [
        item["req"]
        for _, item in texts
        if item["type"] == "voice_state" and item["cause"] == "answer"
    ]
    assert answers == client.voice_starts

    tags = [item.get("audio_tag") for _, item in texts if item["type"] == "stream_start"]
    for tag in (tag for tag in tags if tag is not None):
        frames = [
            (index, item)
            for index, item in enumerate(items)
            if isinstance(item, Frame) and item.turn_tag == tag
        ]
        ends = [(index, frame) for index, frame in frames if frame.msg_type in (2, 3)]
        assert len(ends) == 1, f"audio tag {tag} ended {len(ends)} times"
        end_index, end = ends[0]
        assert frames[-1][0] == end_index, f"a frame of audio tag {tag} followed its end"
        if end.msg_type == AUDIO_FADE:
            stops = [
                index
                for index, item in texts
                if item["type"] == "playback_stop" and item["audio_tag"] == tag
            ]
            assert stops and stops[0] > end_index, f"tag {tag}: playback_stop before its 0x03"


class _Recorded:
    """What `assert_protocol_rules` reads of a connection, written by hand."""

    def __init__(self, items: list[object], voice_starts: list[int]) -> None:
        self.voice = items
        self.voice_starts = voice_starts


def _partial(uid: str) -> dict[str, object]:
    return {"type": "transcript_partial", "utterance_id": uid, "text": "x", "stable_chars": 0}


def _closed(uid: str) -> dict[str, object]:
    return {"type": "transcription", "utterance_id": uid, "text": "x"}


def _frame(tag: int, msg_type: int) -> Frame:
    return Frame(MsgType(msg_type), tag, 0, 24000, b"\x00\x00")


_START = {"type": "stream_start", "turn_id": "t", "channel": "tts", "audio_tag": 1}
_STOP = {"type": "playback_stop", "turn_id": "t", "audio_tag": 1, "fade_ms": 30, "reason": "x"}


@pytest.mark.parametrize(
    ("items", "voice_starts"),
    [
        pytest.param([_partial("u1")], [], id="utterance-never-closed"),
        pytest.param(
            [
                _partial("u1"),
                _closed("u1"),
                {"type": "voice_decision", "kind": "dropped", "detail": {"utterance_id": "u1"}},
            ],
            [],
            id="utterance-closed-twice",
        ),
        pytest.param(
            [
                _partial("u1"),
                _closed("u1"),
                {"type": "voice_decision", "kind": "release", "detail": {"utterance_id": "u1"}},
            ],
            [],
            id="utterance-named-after-closing",
        ),
        pytest.param(
            [{"type": "voice_state", "cause": "answer", "req": 2}], [1], id="answer-of-another-req"
        ),
        pytest.param([], [1], id="voice-start-unanswered"),
        pytest.param([_START, _frame(1, 1)], [], id="audio-tag-never-ended"),
        pytest.param([_START, _frame(1, 2), _frame(1, 1)], [], id="frame-after-the-end"),
        pytest.param([_START, _STOP, _frame(1, 3)], [], id="playback-stop-before-the-fade"),
    ],
)
def test_the_protocol_rules_catch_each_violation(items, voice_starts):
    """The rules each scenario ends with are not vacuous: each violation fails them."""
    with pytest.raises(AssertionError):
        assert_protocol_rules(_Recorded(items, voice_starts))  # type: ignore[arg-type]


# -- Helpers -------------------------------------------------------------------------------------


def frames_of(client: VoiceClient, tag: int) -> list[Frame]:
    return [item for item in client.voice if isinstance(item, Frame) and item.turn_tag == tag]


def one(client: VoiceClient, kind: str, **match: object) -> dict[str, object]:
    found = client.of(kind, **match)
    assert len(found) == 1, f"{kind} {match}: {len(found)} in {client.names()}"
    return found[0]


def after(client: VoiceClient, mark: int) -> list[str]:
    """The names of the voice items received after the first `mark` of them."""
    return client.names()[mark:]


def assert_no_second_turn(srv, client: VoiceClient) -> None:
    """The keyword opened no turn: a speculative start it made stayed held and never streamed."""
    assert len(client.of("stream_start")) == 1
    assert all(request.held for request in srv.responder.requests[1:])
    assert len(srv.responder.commits) == 1


def finish(client: VoiceClient) -> dict[str, object]:
    """Report the newest tts turn's playback done; its `turn_metrics`."""
    turn_id = client.of("stream_start")[-1]["turn_id"]
    client.finish_playback(str(turn_id))
    return one(client, "turn_metrics", turn_id=turn_id)


# -- v1 to v5: one session, one voice -------------------------------------------------------------


def test_a_named_question_completes_a_voice_turn_with_tagged_audio(voice_server):
    stack = VoiceStack(transcripts=["mist, what's the weather"])
    stack.responder.add(Token("It is sunny."))
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        metrics = finish(client)

        start = one(client, "stream_start")
        assert (start["channel"], start["origin"]) == ("tts", "voice")
        tagged = frames_of(client, int(start["audio_tag"]))
        assert [frame.msg_type for frame in tagged][-1] == AUDIO_COMPLETE
        assert len(tagged) > 1
        transcription = one(client, "transcription")
        assert transcription["text"] == "mist, what's the weather"
        assert one(client, "endpoint")["detail"]["utterance_id"] == transcription["utterance_id"]
        assert (metrics["outcome"], metrics["origin"], metrics["recorded"]) == (
            "completed",
            "voice",
            True,
        )
        assert [request.user_text for request in srv.responder.requests] == [
            "mist, what's the weather"
        ]
        assert_protocol_rules(client)


def test_the_name_said_last_after_a_pause_keeps_the_preamble(voice_server):
    """v5 scenario 2: a 700 ms pause is inside the join gap, so the chain reaches the preamble."""
    text = "so about the weekend plans, mist?"
    stack = VoiceStack(transcripts=[text])
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        client.say(
            [silence(200), speech_burst(1400), silence(700), speech_burst(400), silence(600)],
            name_at_ms=2700,
        )
        metrics = finish(client)

        assert one(client, "transcription")["text"] == text
        assert [request.user_text for request in srv.responder.requests] == [text]
        assert metrics["outcome"] == "completed"
        # The utterance's audio holds the preamble: it opened before the pause.
        (stream,) = stack.transcriber.streams
        assert stream.audio_ms >= 1400 + 700 + 400
        assert_protocol_rules(client)


def speaking_turn(srv, client: VoiceClient, *steps: object) -> str:
    """Start voice, ask a named question, and return the turn id once its reply streams."""
    srv.responder.add(*steps)
    client.send(voice_start(len(client.voice_starts) + 1))
    client.say(NAMED[:2], name_at_ms=NAME_AT_MS)
    # With a second voice session open, the wake waits for the arbiter's window.
    settle_arbitration(srv)
    client.say(NAMED[2:])
    return str(client.of("stream_start")[-1]["turn_id"])


def test_a_keyword_mid_speech_cuts_the_reply_at_the_ledger_prefix(voice_server):
    """v1 scenario 3 with v2's change: the keyword alone closes with `dropped keyword`."""
    stack = VoiceStack(transcripts=["mist, what's the weather", "stop"])
    pause = Pause()
    with voice_server(stack) as srv, srv.connect() as client:
        turn_id = speaking_turn(
            srv, client, Token("It is sunny and warm today."), pause, Token(" More later.")
        )
        # 600 ms of the 1350 ms segment (27 characters at 50 ms each) has played.
        client.send({"type": "playback_progress", "turn_id": turn_id, "played_ms": 600})
        client.say([speech_burst(400), silence(600)])
        srv.run(pause.release)

        barge = one(client, "barge_in", turn_id=turn_id)
        assert barge["reason"] == "keyword"
        cancelled = one(client, "stream_cancelled", turn_id=turn_id)
        assert cancelled["reason"] == "barge_in"
        assert cancelled["spoken_text"] == "It is sunny"
        tokens = [item["token"] for item in client.of("stream_token", turn_id=turn_id)]
        assert tokens == ["It is sunny and warm today."]
        dropped = one(client, "dropped")
        assert dropped["reason"] == "keyword"
        (commit,) = srv.responder.commits
        assert (commit.outcome, commit.delivered_text) == ("interrupted", "It is sunny")
        assert one(client, "turn_metrics", turn_id=turn_id)["outcome"] == "interrupted"
        assert_no_second_turn(srv, client)
        assert_protocol_rules(client)


def test_other_speech_while_she_speaks_sends_nothing(voice_server):
    """v5 scenario 4: speech that is not a keyword leaves no trace on the wire."""
    stack = VoiceStack(transcripts=["mist, what's the weather", "uh huh right"])
    with voice_server(stack) as srv, srv.connect() as client:
        speaking_turn(srv, client, Token("It is sunny and warm today."))
        mark = len(client.names())
        client.say([speech_burst(600), silence(600)])

        assert after(client, mark) == []
        assert finish(client)["outcome"] == "completed"
        assert len(srv.responder.requests) == 1
        assert_protocol_rules(client)


@pytest.mark.parametrize("second", ["stop", "mist, stop"])
def test_stop_while_thinking_ends_the_turn_with_an_empty_response(voice_server, second):
    """v1 scenario 6 with v2's change: `stop` closes with `dropped keyword`, no new turn.

    Her name before the keyword, unmarked by the spotter so nothing supersedes, ends the same way.
    """
    stack = VoiceStack(transcripts=["mist, what's the weather", second])
    pause = Pause()
    with voice_server(stack) as srv, srv.connect() as client:
        srv.responder.add(pause, Token("Sure."))
        client.send(voice_start(1))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        (turn_id,) = (item["turn_id"] for item in client.of("state_cycle", state="think"))
        client.say([speech_burst(400), silence(600)])
        srv.run(pause.release)

        assert one(client, "barge_in", turn_id=turn_id)["reason"] == "keyword"
        assert one(client, "dropped")["reason"] == "keyword"
        cancelled = one(client, "stream_cancelled", turn_id=turn_id)
        assert cancelled["spoken_text"] == ""
        (commit,) = srv.responder.commits
        assert (commit.outcome, commit.delivered_text) == ("interrupted", "")
        assert_no_second_turn(srv, client)
        assert client.of("stream_token") == []
        assert_protocol_rules(client)


def test_a_typed_turn_sends_no_binary_frame(voice_server):
    stack = VoiceStack()
    stack.responder.add(Token("Hello."))
    with voice_server(stack) as srv, srv.connect() as client:
        client.send({"type": "text", "text": "hello"})

        start = one(client, "stream_start")
        assert (start["channel"], "audio_tag" in start) == ("text", False)
        assert [item for item in client.voice if isinstance(item, Frame)] == []
        assert one(client, "turn_metrics")["outcome"] == "completed"
        assert_protocol_rules(client)


def test_two_connections_stay_apart_and_a_log_line_reaches_the_subscriber_only(voice_server):
    stack = VoiceStack()
    with voice_server(stack) as srv, srv.connect() as first, srv.connect() as second:
        first.send({"type": "subscribe_logs", "enabled": True, "levels": None})
        first.send({"type": "text", "text": "hello"})
        srv.run(lambda: logging.getLogger("backend.realtime").warning("scenario log line"))
        srv.settle()

        assert one(first, "turn_metrics")["outcome"] == "completed"
        assert second.of("stream_start") == second.of("turn_metrics") == []
        assert any("scenario log line" in str(item) for item in first.voice)
        assert not any("scenario log line" in str(item) for item in second.voice)
        for client in (first, second):
            assert_protocol_rules(client)


def test_a_ui_interrupt_during_generation_is_handled(voice_server):
    """v1 scenario 9: the receive loop takes `interrupt` while the responder is mid-stream."""
    stack = VoiceStack()
    pause = Pause()
    stack.responder.add(Token("Part one."), pause, Token(" Part two."))
    with voice_server(stack) as srv, srv.connect() as client:
        client.send({"type": "text", "text": "tell me two things"})
        client.send({"type": "interrupt"})

        assert one(client, "barge_in")["reason"] == "ui"
        assert one(client, "stream_cancelled")["reason"] == "user_interrupt"
        assert one(client, "turn_metrics")["outcome"] == "interrupted"
        assert not pause.event.is_set()
        srv.run(pause.release)
        assert [item["token"] for item in client.of("stream_token")] == ["Part one."]
        assert_protocol_rules(client)


def test_a_reconfigure_waits_for_the_turn_and_voice_stop_cuts_and_closes(voice_server):
    """v1 scenario 10 with v2's change, in wake mode (v5)."""
    stack = VoiceStack(
        transcripts=["mist, what's the weather", "mist, tell me a story", "mist, and then"]
    )
    pause = Pause()
    with voice_server(stack) as srv, srv.connect() as client:
        turn_id = speaking_turn(srv, client, Token("It is sunny."))
        client.send(voice_start(2, aec="native"))
        assert client.of("voice_state", cause="answer", req=2) == []
        turn_metrics = finish(client)

        # The answer waits for the turn to end: it follows the turn's audio, its completion, its
        # `state_cycle idle` (v2 M46) and its `turn_metrics`. The controller queues the idle and
        # the metrics on the turn lane, and the sender fences the answer `apply` readies behind
        # every turn-lane item queued before it (A53).
        names = client.names()
        answer = client.voice.index(one(client, "voice_state", cause="answer", req=2))
        complete = client.voice.index(one(client, "stream_complete", turn_id=turn_id))
        idle = client.voice.index(one(client, "state_cycle", state="idle", turn_id=turn_id))
        metrics = client.voice.index(turn_metrics)
        assert names.index("0x02") < answer and complete < answer
        assert idle < answer and metrics < answer
        assert client.of("stream_cancelled") == []

        # A turn in flight: `voice_stop` cuts it with reason `voice_stop`.
        srv.responder.add(pause, Token("Once upon a time."))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        second = str(client.of("state_cycle", state="think")[-1]["turn_id"])
        client.send({"type": "voice_stop"})
        srv.run(pause.release)
        assert one(client, "barge_in", turn_id=second)["reason"] == "voice_stop"
        assert one(client, "stream_cancelled", turn_id=second)["reason"] == "voice_stop"

        # An utterance open: `voice_stop` closes it with `dropped voice_stop`.
        client.send(voice_start(3))
        client.say([silence(200), speech_burst(1200)], name_at_ms=NAME_AT_MS)
        open_id = client.of("transcript_partial")[-1]["utterance_id"]
        client.send({"type": "voice_stop"})
        dropped = one(client, "dropped", reason="voice_stop")
        assert dropped["detail"]["utterance_id"] == open_id
        assert_protocol_rules(client)


def test_a_disconnect_mid_turn_commits_it_interrupted(voice_server):
    stack = VoiceStack()
    stack.responder.add(Token("Partial answer."), Pause())
    with voice_server(stack) as srv:
        with srv.connect() as client:
            client.send({"type": "text", "text": "tell me something"})
            assert client.of("stream_token")
        srv.settle()

        (commit,) = srv.responder.commits
        assert (commit.outcome, commit.delivered_text) == ("interrupted", "Partial answer.")
        assert_protocol_rules(client)


def test_speculative_starts_are_cancelled_by_resumed_speech_and_released_at_the_end(voice_server):
    """v1 scenario 5, its first two parts.

    The end-of-turn model never says done (0.2), so each pause opens a held speculative turn.
    The first is cancelled when speech resumes, and leaves nothing on the wire; the second is
    released at the silence cap. The third part, the supersede, is the next test.
    """
    text = "mist, what's the weather like"
    stack = VoiceStack(transcripts=[text], eot=(0.2,))
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        client.say(
            [silence(200), speech_burst(1200), silence(400), speech_burst(600), silence(2000)],
            name_at_ms=NAME_AT_MS,
        )
        metrics = finish(client)

        cancelled, released = srv.responder.requests
        assert (cancelled.held, released.held) == (True, True)
        assert cancelled.user_text == released.user_text == text
        release = one(client, "release")
        assert release["reason"] == "silence_cap"
        turn_id = release["turn_id"]
        # Only the released turn reaches the client.
        assert {
            item["turn_id"] for item in client.voice if isinstance(item, dict) and "turn_id" in item
        } == {turn_id}
        assert one(client, "endpoint")["turn_id"] == turn_id
        assert metrics["outcome"] == "completed"
        assert [commit.outcome for commit in srv.responder.commits] == ["completed"]
        assert_protocol_rules(client)


def test_a_name_while_she_thinks_supersedes_the_turn(voice_server):
    """v1 scenario 5, its third part, on v13's A51 (T55c, T55d).

    The responder pauses before the first turn's first token, so none of its audio is queued
    when the second name is accepted: the name supersedes the turn instead of barging in. The
    `supersede` goes out at the cut (Decision 2, option A). No tool was dispatched, so the first
    turn is not recorded on its own: the second turn's request carries both utterances joined by
    one space, its `transcription` its own text, and its commit is the only one.
    """
    stack = VoiceStack(transcripts=["mist, what's the weather", "mist, and in Paris"])
    pause = Pause()
    with voice_server(stack) as srv, srv.connect() as client:
        srv.responder.add(pause, Token("It is sunny."))
        srv.responder.add(Token("In Paris it rains."))
        client.send(voice_start(1))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        (first,) = (item["turn_id"] for item in client.of("state_cycle", state="think"))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        srv.run(pause.release)
        metrics = finish(client)

        supersede = one(client, "supersede", turn_id=first)
        assert (supersede["reason"], supersede["detail"]) == (
            "continued_speech",
            {"merged": True, "tool_dispatched": False},
        )
        cancelled = one(client, "stream_cancelled", turn_id=first)
        assert cancelled["reason"] == "superseded"
        assert client.voice.index(supersede) < client.voice.index(cancelled)
        assert client.of("barge_in") == []
        first_metrics = one(client, "turn_metrics", turn_id=first)
        assert (first_metrics["outcome"], first_metrics["recorded"]) == ("superseded", False)
        assert [request.user_text for request in srv.responder.requests] == [
            "mist, what's the weather",
            "mist, what's the weather mist, and in Paris",
        ]
        assert [item["text"] for item in client.of("transcription")] == [
            "mist, what's the weather",
            "mist, and in Paris",
        ]
        assert [commit.outcome for commit in srv.responder.commits] == ["completed"]
        assert metrics["turn_id"] != first and metrics["outcome"] == "completed"
        assert_protocol_rules(client)


def test_mist_stop_while_she_thinks_supersedes_the_turn_and_starts_none(voice_server):
    """The name heard on "mist, stop" supersedes the thinking turn, and the end, her name then a
    keyword alone, cancels the held turn from it: the first turn is recorded on its own, empty,
    and no second turn reaches the model or the client.
    """
    stack = VoiceStack(transcripts=["mist, what's the weather", "mist, stop"])
    pause = Pause()
    with voice_server(stack) as srv, srv.connect() as client:
        srv.responder.add(pause, Token("It is sunny."))
        srv.responder.add(Pause())
        client.send(voice_start(1))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        (first,) = (item["turn_id"] for item in client.of("state_cycle", state="think"))
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        srv.run(pause.release)

        supersede = one(client, "supersede", turn_id=first)
        assert supersede["detail"] == {"merged": True, "tool_dispatched": False}
        dropped = one(client, "dropped")
        assert dropped["reason"] == "keyword"
        (said_first,) = (item["utterance_id"] for item in client.of("transcription"))
        (said_second,) = {item["utterance_id"] for item in client.of("transcript_partial")} - {
            said_first
        }
        assert dropped["detail"]["utterance_id"] == said_second
        # Only the first utterance's held turn is released; the second's is cancelled unseen.
        assert [item["detail"]["utterance_id"] for item in client.of("release")] == [said_first]
        assert [(c.outcome, c.delivered_text) for c in srv.responder.commits] == [
            ("superseded", "")
        ]
        assert_no_second_turn(srv, client)
        assert client.of("stream_token") == []
        assert_protocol_rules(client)


def test_a_malformed_frame_is_a_protocol_error_and_the_session_goes_on(voice_server):
    stack = VoiceStack()
    with voice_server(stack) as srv, srv.connect() as client:
        client.send_bytes(b"\x00" * 7)
        client.send({"type": "text", "text": "hello"})

        assert one(client, "error")["kind"] == "protocol"
        assert one(client, "turn_metrics")["outcome"] == "completed"
        assert_protocol_rules(client)


# -- v3 and v5: endpoints, mute, isolation ---------------------------------------------------------


def wake(score: float, backlog_ms: int) -> dict[str, object]:
    return {"type": "wake", "keyword": "mist", "score": score, "backlog_ms": backlog_ms}


def settle_arbitration(srv) -> None:
    while srv.stack.call_later.pending:
        srv.fire_arbiter()


def test_endpoint_wakes_arbitrate_barge_in_and_bound_the_chain(voice_server, caplog):
    """v5 scenario 13 over `/ws`, which binds every connection `desktop`.

    Two sessions use the endpoint spotter. A `voice_start` naming `room` is logged and ignored
    (v3). The worse-ranked wake loses arbitration and gets one `wake_lost` with the winner's
    score (v2 :2256).
    """
    question = "mist, what's the time"
    stack = VoiceStack(transcripts=[question] * 4)
    # Each backlog: room tone, then the name, which ends at the backlog's last frame.
    backlog = [silence(1100), speech_burst(400)]
    live = [speech_burst(1000), silence(600)]
    endpoint_start = {"spotter": "endpoint", "endpoint": "room"}
    with voice_server(stack) as srv, srv.connect() as first, srv.connect() as second:
        with caplog.at_level(logging.WARNING, logger="backend.realtime.session"):
            first.send(voice_start(1, **endpoint_start))
            second.send(voice_start(1, **endpoint_start))
        bound = [r for r in caplog.records if "connection is bound to desktop" in r.getMessage()]
        assert len(bound) == 2
        assert one(first, "voice_state", cause="answer")["refused"] is None

        srv.responder.add(Token("It is noon."), Pause())
        first.send(wake(0.9, 1500))
        second.send(wake(0.6, 1500))
        first.say(backlog)
        second.say(backlog, seed=1)
        settle_arbitration(srv)
        first.say(live)
        second.say(live, seed=1)

        lost = one(second, "wake_lost")
        assert lost["reason"] == "arbitration"
        assert lost["detail"]["score"] == 0.6
        assert lost["detail"]["winner_score"] == 0.9
        assert lost["detail"]["utterance_id"] is not None
        assert second.of("wake_rejected") == []
        assert second.of("stream_start") == []
        turn_id = str(one(first, "stream_start")["turn_id"])
        assert one(first, "transcription")["text"] == question

        # The endpoint's own spotter hears a stop keyword: the reply is cut.
        first.send({"type": "barge_in", "turn_id": turn_id, "keyword": "stop"})
        assert one(first, "barge_in", turn_id=turn_id)["detail"]["utterance_id"] is None
        assert one(first, "stream_cancelled", turn_id=turn_id)["reason"] == "barge_in"

        # A wake with no backlog is rejected and nothing else is sent.
        mark = len(first.names())
        first.send(wake(0.9, 0))
        assert after(first, mark) == ["decision:wake_rejected"]
        assert first.of("wake_rejected")[-1]["reason"] == "no_backlog"

        # A 4 s backlog: unrelated speech, a 1.2 s pause, then the name. The pause is past the
        # join gap, so the utterance opens at the name and the unrelated speech is left out.
        srv.responder.add(Token("It is one."))
        streams_before = len(stack.transcriber.streams)
        first.send(wake(0.9, 4000))
        first.say([silence(400), speech_burst(2000), silence(1200), speech_burst(400)], seed=2)
        settle_arbitration(srv)
        first.say(live, seed=3)
        metrics = finish(first)

        assert metrics["outcome"] == "completed"
        assert first.of("transcription")[-1]["text"] == question
        (stream,) = stack.transcriber.streams[streams_before:]
        assert stream.audio_ms < 2000 + 1200
        for client in (first, second):
            assert_protocol_rules(client)


TIMER_REPLY = "Timer set for 10 minutes."


def timer_handled() -> CapabilityOnlyTurn:
    return CapabilityOnlyTurn(
        handled=True, events=[Token(TIMER_REPLY), Complete(final_response=TIMER_REPLY)]
    )


def since(client: VoiceClient, mark: int, kind: str, **match: object) -> list[dict[str, object]]:
    """`client.of(kind, ...)` among the voice items after the first `mark`."""
    fresh = client.voice[mark:]
    return [item for item in client.of(kind, **match) if any(item is f for f in fresh)]


def assert_spoken(client: VoiceClient) -> int:
    """The newest turn went out on the `tts` channel and its audio tag ended with a `0x02`."""
    start = client.of("stream_start")[-1]
    assert start["channel"] == "tts", start
    tag = int(start["audio_tag"])
    tagged = frames_of(client, tag)
    assert [frame.msg_type for frame in tagged].count(AUDIO_COMPLETE) == 1, tagged
    return tag


@pytest.mark.parametrize(
    "devices",
    [pytest.param(("cuda",), id="stt-on-cuda"), pytest.param(("cpu",), id="stt-on-cpu")],
)
def test_gaming_mode_takes_the_level_off_and_back(voice_server, tmp_path, devices):
    """v3 scenario 14: the real `GpuYield` on a marker, `GpuYieldPower`, `SingleModelTiers`.

    TTS has no CUDA arm in the engine table, so it stays on the CPU in both runs; the CUDA run
    puts STT on CUDA with no CPU option, so the release leaves voice nothing to transcribe with.
    """
    on_cuda = devices == ("cuda",)
    spoken = [] if on_cuda else ["mist, set a timer for 10 minutes", "mist, tell me a story"]
    stack = VoiceStack(transcripts=[*spoken, "mist, what's the weather"], stt_devices=devices)
    if not on_cuda:
        stack.torch.cuda.reserved = 0
    responder = stack.responder
    responder.capability_results.extend([timer_handled(), None] * 2)
    marker = tmp_path / MARKER_NAME
    with (
        voice_server(stack, ops_dir=tmp_path) as srv,
        srv.connect() as active,
        srv.connect() as other,
    ):
        active.send(voice_start(1))
        marks = [len(active.voice), len(other.voice)]
        marker.touch()
        srv.poll_gpu_yield()

        for client, mark in zip((active, other), marks, strict=True):
            (placement,) = since(client, mark, "placement")
            assert placement["reason"] == "power_off"
            assert placement["detail"]["released_mib"] == (2048 if on_cuda else 0)
            assert since(client, mark, "voice_state")[-1]["power_level"] == "off"
        state = active.of("voice_state")[-1]
        if on_cuda:
            assert (state["active"], state["refused"]) == (False, "power_off")
        else:
            assert (state["active"], state["refused"]) == (True, None)

        # Typed: a command is handled with no model call, anything else gets the envelope.
        active.send({"type": "text", "text": "set a timer for 10 minutes"})
        active.send({"type": "text", "text": "tell me a story"})
        assert active.of("stream_token")[-1]["token"] == TIMER_REPLY
        assert active.of("error")[-1] == turn_refusal(YIELDED_MESSAGE)
        assert responder.requests == []

        if on_cuda:
            active.send(voice_start(2))
            answer = one(active, "voice_state", cause="answer", req=2)
            assert (answer["active"], answer["refused"]) == (False, "power_off")
        else:
            # Spoken: the command is handled and spoken; the story gets the envelope, spoken.
            active.say(NAMED, name_at_ms=NAME_AT_MS)
            assert finish(active)["outcome"] == "completed"
            assert active.of("stream_token")[-1]["token"] == TIMER_REPLY
            timer_tag = assert_spoken(active)
            active.say(NAMED, name_at_ms=NAME_AT_MS)
            assert finish(active)["outcome"] == "completed"
            assert active.of("error")[-1] == turn_refusal(YIELDED_MESSAGE)
            assert active.of("stream_token")[-1]["token"] == YIELDED_MESSAGE
            assert assert_spoken(active) != timer_tag
            assert responder.requests == []
        assert [call[0] for call in responder.capability_calls] == [
            "set a timer for 10 minutes",
            "tell me a story",
            *spoken,
        ]

        marks = [len(active.voice), len(other.voice)]
        marker.unlink()
        srv.poll_gpu_yield()
        for client, mark in zip((active, other), marks, strict=True):
            (restored,) = since(client, mark, "placement")
            assert restored["reason"] == "power_restored"
            assert restored["detail"]["engines"]["stt"] == devices[0]
            assert since(client, mark, "voice_state")[-1]["power_level"] == "full"
        state = active.of("voice_state")[-1]
        assert (state["active"], state["refused"]) == (True, None)

        # The spoken refusal names MIST; past the echo guard her name is the user's again.
        active.say([silence(2000)])
        turns = len(active.of("stream_start"))
        active.say(NAMED, name_at_ms=NAME_AT_MS)
        assert len(active.of("stream_start")) == turns + 1
        assert finish(active)["outcome"] == "completed"
        assert [request.user_text for request in responder.requests] == ["mist, what's the weather"]
        for client in (active, other):
            assert_protocol_rules(client)


def test_a_failed_reload_retries_until_stt_is_back_on_its_device(voice_server, tmp_path):
    """v3 scenario 14b."""
    stack = VoiceStack(stt_devices=("cuda", "cpu"))
    stack.responder.add(Token("Here is the news."))
    marker = tmp_path / MARKER_NAME
    with voice_server(stack, ops_dir=tmp_path) as srv, srv.connect() as client:
        marker.touch()
        srv.poll_gpu_yield()
        assert one(client, "placement", reason="power_off")["detail"]["engines"]["stt"] == "cpu"

        stack.stt_load_errors.append(SpeechEngineError("the CUDA load failed"))
        marker.unlink()
        srv.poll_gpu_yield()

        gpu_yield = server._gpu_yield
        assert gpu_yield is not None and gpu_yield.state() == "normal"
        (poll,) = srv.run(lambda: [t for t in asyncio.all_tasks() if t.get_name() == "gpu-yield"])
        assert not poll.done()
        failed = one(client, "placement", reason="reload_failed")
        assert failed["detail"]["engines"]["stt"] == "cpu"
        client.send({"type": "text", "text": "what's new"})
        assert one(client, "turn_metrics")["outcome"] == "completed"
        assert [request.user_text for request in stack.responder.requests] == ["what's new"]

        runtime = stack.runtime
        assert runtime is not None and runtime.gpu is not None and runtime.gpu.retrying
        srv.run(stack.retry_now)
        srv.until(lambda: not runtime.gpu.retrying)

        assert client.of("placement")[-1]["reason"] == "power_restored"
        assert client.of("placement")[-1]["detail"]["engines"]["stt"] == "cuda"
        assert len(client.of("placement", reason="reload_failed")) == 1
        assert_protocol_rules(client)


def test_mute_closes_the_open_utterance_and_holds_until_unmute(voice_server):
    """v3 scenario 15."""
    stack = VoiceStack(transcripts=["mist, what's the", "mist, are you there"])
    pause = Pause()
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        client.say([silence(200), speech_burst(1200)], name_at_ms=NAME_AT_MS)
        open_id = client.of("transcript_partial")[-1]["utterance_id"]
        srv.responder.add(Token("Typed reply."), pause, Token(" Done."))
        client.send({"type": "mute", "active": True})

        dropped = one(client, "dropped", reason="muted")
        assert dropped["detail"]["utterance_id"] == open_id
        assert client.of("voice_state")[-1]["muted"] is True

        # Muted: a turn in flight completes, and speech, even with the name, opens nothing.
        client.send({"type": "text", "text": "hello"})
        mark = len(client.names())
        client.say(NAMED, name_at_ms=NAME_AT_MS)
        srv.run(pause.release)
        assert "transcript_partial" not in after(client, mark)
        assert "state:attend" not in after(client, mark)
        typed = one(client, "turn_metrics")
        assert typed["outcome"] == "completed"
        assert all(state["muted"] for state in client.of("voice_state")[1:])

        client.send({"type": "mute", "active": False})
        assert client.of("voice_state")[-1]["muted"] is False
        assert_protocol_rules(client)


def test_turn_payloads_stay_with_their_session_and_alarm_state_reaches_both(voice_server):
    """v3 scenario 16."""
    stack = VoiceStack(transcripts=["mist, what's the weather"])
    with voice_server(stack) as srv, srv.connect() as first, srv.connect() as second:
        speaking_turn(srv, first, Token("It is sunny."))
        finish(first)
        alarm = {
            "type": "alarm_state",
            "rev": 1,
            "alarms": [],
            "ringing": [],
            "missed": [],
            "fallback_after_s": 30.0,
            "missed_grace_s": 60.0,
            "ring_timeout_s": 300.0,
            "snooze_minutes": 9,
        }
        srv.run(server._capability_sink.emit, alarm)

        turn_kinds = {"stream_start", "stream_token", "stream_complete", "turn_metrics"}
        assert {name for name in first.names() if name in turn_kinds} == turn_kinds
        assert [name for name in second.names() if name in turn_kinds] == []
        assert [item for item in second.voice if isinstance(item, Frame)] == []
        for client in (first, second):
            client.drain()
            states = [
                i for i in client.received if isinstance(i, dict) and i["type"] == "alarm_state"
            ]
            assert states == [alarm]
            assert_protocol_rules(client)


def test_a_muted_client_reconnects_muted(voice_server):
    """v3 scenario 17."""
    stack = VoiceStack(transcripts=["mist, what's the weather"])
    with voice_server(stack) as srv:
        with srv.connect() as before:
            before.send(voice_start(1), {"type": "mute", "active": True})
            assert before.of("voice_state")[-1]["muted"] is True
        with srv.connect() as client:
            client.send(voice_start(1, muted=True))
            answer = one(client, "voice_state", cause="answer")
            assert (answer["active"], answer["muted"]) == (True, True)
            mark = len(client.names())
            client.say(NAMED, name_at_ms=NAME_AT_MS)

            assert after(client, mark) == []
            assert stack.transcriber.streams == []
            assert_protocol_rules(client)
        assert_protocol_rules(before)


def test_a_greeting_and_a_leading_name_each_open_a_turn_attend_first(voice_server):
    """v5 scenario 18."""
    texts = ["hey mist, how's it going", "mist, what's up"]
    stack = VoiceStack(transcripts=texts)
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        for text in texts:
            mark = len(client.names())
            client.say(NAMED, name_at_ms=NAME_AT_MS)
            finish(client)
            states = [name for name in after(client, mark) if name.startswith("state:")]
            assert states[:2] == ["state:attend", "state:listen"]
            assert client.of("transcription")[-1]["text"] == text

        assert [request.user_text for request in srv.responder.requests] == texts
        assert [metrics["outcome"] for metrics in client.of("turn_metrics")] == ["completed"] * 2
        assert_protocol_rules(client)


def test_the_name_inside_other_speech_leaves_only_attend_rejected_idle(voice_server, caplog):
    """v5 scenario 19: neither sentence addresses her, so nothing of either travels."""
    texts = ["I missed it", "the mist rolled in"]
    stack = VoiceStack(transcripts=texts)
    caplog.set_level(logging.DEBUG)
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        mark = len(client.names())
        for _ in texts:
            client.say(NAMED, name_at_ms=NAME_AT_MS)

        each = ["state:attend", "decision:wake_rejected", "state:idle"]
        assert after(client, mark) == each * 2
        assert [item["reason"] for item in client.of("wake_rejected")] == ["not_addressed"] * 2
        for word in ("missed", "rolled"):
            assert word not in caplog.text
        assert srv.responder.requests == srv.responder.commits == []
        assert_protocol_rules(client)


MARKER = b"\x7f\x7e" * 8


def test_the_name_in_the_pcs_own_playback_is_self_media(voice_server):
    """v5 scenario 20: the `0x11` frames just before the hit carry the scripted name marker.

    The reference must cover the spotter's whole window (1280 ms) before the hit, so the name
    comes after 1.4 s of room tone.
    """
    stack = VoiceStack(transcripts=["mist, what's the weather"], markers={MARKER: 0.9})
    hit_seq = (1400 + 200) // 20

    def reference(seq: int, pcm: bytes) -> bytes:
        return MARKER + pcm[len(MARKER) :] if hit_seq - 10 < seq <= hit_seq else bytes(len(pcm))

    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1, reference=True))
        mark = len(client.names())
        client.say(
            [silence(1400), speech_burst(1200), silence(600)], name_at_ms=1600, reference=reference
        )

        assert after(client, mark) == ["state:attend", "decision:wake_rejected", "state:idle"]
        assert one(client, "wake_rejected")["reason"] == "self_media"
        assert srv.responder.requests == []
        assert_protocol_rules(client)


def test_her_own_name_in_her_recent_speech_is_self_speech(voice_server):
    """v5 scenario 21: in this session's reply, and in a second session's."""
    stack = VoiceStack(transcripts=["mist, who are you", "mist, who are you", "mist, hi"])
    with voice_server(stack) as srv, srv.connect() as client:
        speaking_turn(srv, client, Token("I'm Mist."))
        finish(client)
        mark = len(client.names())
        client.say(NAMED, name_at_ms=NAME_AT_MS)

        assert after(client, mark) == ["decision:wake_rejected"]
        assert one(client, "wake_rejected")["reason"] == "self_speech"
        assert_protocol_rules(client)

    stack = VoiceStack(transcripts=["mist, who are you", "mist, hi"])
    with voice_server(stack) as srv, srv.connect() as speaker, srv.connect() as listener:
        listener.send(voice_start(1))
        # Hits are scripted by uplink position: room tone first keeps the listener's name hit
        # clear of the speaker's.
        listener.say([silence(2400)])
        speaking_turn(srv, speaker, Token("I'm Mist."))
        finish(speaker)
        mark = len(listener.names())
        listener.say(NAMED, name_at_ms=NAME_AT_MS)

        assert after(listener, mark) == ["decision:wake_rejected"]
        assert one(listener, "wake_rejected")["reason"] == "self_speech"
        for client in (speaker, listener):
            assert_protocol_rules(client)


def test_without_a_name_spotter_voice_is_refused_and_typing_works(voice_server):
    """v5 scenario 22: `MIST_VOICE_KWS_MODEL` unset builds no spotter."""
    stack = VoiceStack(kws=False)
    stack.responder.add(Token("Hello."))
    with voice_server(stack) as srv, srv.connect() as client:
        client.send(voice_start(1))
        answer = one(client, "voice_state", cause="answer")
        assert (answer["active"], answer["refused"]) == (False, "kws unavailable")
        client.send({"type": "text", "text": "hello"})

        assert one(client, "turn_metrics")["outcome"] == "completed"
        assert_protocol_rules(client)
