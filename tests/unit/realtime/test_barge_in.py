"""Tests for `KeywordBargeIn`, which decides what user speech means while MIST speaks or works.

A wrong verdict is something the user hears at once: MIST talking over a "stop", MIST cutting
herself off on her own echo, or a request said over her reply silently lost. So each rule of the
Design doc's Barge-in policy is pinned here by phase: the arm delay and the keyword window in
`speaking`, the echo guard and its `headset` exemption, the stop-keyword-only rule in `held` and
`thinking`, `idle` and `text`, and the endpoint's own keyword reports. Time comes from a fake
clock and the played TTS text from a fake ledger, so every verdict is fixed by the calls made.
"""

import pytest

from backend.realtime.bargein import AecMode, BargeInPolicy, BargeInVerdict, KeywordBargeIn
from backend.realtime.budget import BargeInConfig
from backend.realtime.protocol import (
    BargeInDetail,
    BargeInIgnoredDetail,
    VoiceDecisionBargeIn,
    VoiceDecisionBargeInIgnored,
    VoiceStartMessage,
)
from backend.realtime.speech import Phase, TranscriptUpdate

TURN = "turn-1"
AUDIO_AT = 10_000
ARMED_AT = AUDIO_AT + 1000


class FakeClock:
    """The session clock, set by hand."""

    def __init__(self, now_ms: int = 0) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        return self.now_ms


class FakeEcho:
    """The ledger's `recent_tts_text`, returning set text and recording each window asked for."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.windows: list[int] = []

    def __call__(self, window_ms: int) -> str:
        self.windows.append(window_ms)
        return self.text


def partial(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(text=text, stable_chars=0, is_final=False, audio_ms=600)


def final(text: str) -> TranscriptUpdate:
    return TranscriptUpdate(text=text, stable_chars=len(text), is_final=True, audio_ms=900)


def make_policy(
    *,
    aec: AecMode = "browser",
    echo: FakeEcho | None = None,
    clock: FakeClock | None = None,
    config: BargeInConfig | None = None,
    audio_started: bool = True,
) -> tuple[KeywordBargeIn, FakeEcho, FakeClock]:
    echo = echo if echo is not None else FakeEcho()
    clock = clock if clock is not None else FakeClock(ARMED_AT)
    policy = KeywordBargeIn(
        config if config is not None else BargeInConfig(),
        aec=aec,
        recent_tts_text=echo,
        clock=clock,
    )
    if audio_started:
        policy.on_turn_audio_started(TURN, AUDIO_AT)
    return policy, echo, clock


def interrupt(keyword: str, text: str, remainder: str, phase: Phase | None) -> BargeInVerdict:
    return BargeInVerdict(
        action="interrupt",
        reason="keyword",
        keyword=keyword,
        text=text,
        remainder=remainder,
        phase=phase,
    )


def ignored(reason: str, keyword: str | None, text: str, phase: Phase | None) -> BargeInVerdict:
    return BargeInVerdict(
        action="ignored", reason=reason, keyword=keyword, text=text, remainder="", phase=phase
    )


def test_keyword_barge_in_satisfies_the_policy_protocol() -> None:
    policy: BargeInPolicy = make_policy()[0]
    assert policy.evaluate(partial("stop"), ARMED_AT, "speaking") is not None


def test_aec_mode_matches_the_voice_start_field() -> None:
    """The policy's `aec` type is the wire's, so a client value always reaches the guard."""
    assert AecMode.__args__ == VoiceStartMessage.model_fields["aec"].annotation.__args__


class TestArmDelay:
    """In `speaking` a keyword counts only `arm_delay_ms` after the turn's first audio played."""

    @pytest.mark.parametrize(
        ("now_ms", "expected_reason"),
        [
            (AUDIO_AT, "not_armed"),
            (ARMED_AT - 1, "not_armed"),
            (ARMED_AT, "keyword"),
            (ARMED_AT + 5000, "keyword"),
        ],
    )
    def test_partial_keyword_is_held_back_until_armed(
        self, now_ms: int, expected_reason: str
    ) -> None:
        policy, _, _ = make_policy()
        verdict = policy.evaluate(partial("stop"), now_ms, "speaking")
        assert verdict is not None
        assert verdict.reason == expected_reason
        assert verdict.keyword == "stop"
        assert verdict.action == ("interrupt" if expected_reason == "keyword" else "ignored")

    def test_not_armed_verdict_carries_keyword_and_text(self) -> None:
        policy, _, _ = make_policy()
        verdict = policy.evaluate(partial("Stop, please"), ARMED_AT - 1, "speaking")
        assert verdict == ignored("not_armed", "stop", "Stop, please", "speaking")

    def test_no_audio_yet_is_not_armed(self) -> None:
        policy, _, _ = make_policy(audio_started=False)
        verdict = policy.evaluate(partial("stop"), ARMED_AT + 60_000, "speaking")
        assert verdict == ignored("not_armed", "stop", "stop", "speaking")

    def test_each_turn_arms_from_its_own_first_audio(self) -> None:
        policy, _, _ = make_policy()
        policy.on_turn_audio_started("turn-2", 20_000)
        assert policy.evaluate(partial("stop"), 20_999, "speaking").reason == "not_armed"
        assert policy.evaluate(partial("stop"), 21_000, "speaking").reason == "keyword"

    def test_arm_delay_comes_from_config(self) -> None:
        policy, _, _ = make_policy(config=BargeInConfig(arm_delay_ms=250))
        assert policy.evaluate(partial("stop"), AUDIO_AT + 249, "speaking").reason == "not_armed"
        assert policy.evaluate(partial("stop"), AUDIO_AT + 250, "speaking").reason == "keyword"

    def test_utterance_end_reads_the_injected_clock(self) -> None:
        clock = FakeClock(ARMED_AT - 1)
        policy, _, _ = make_policy(clock=clock)
        assert policy.on_utterance_end(final("stop"), "speaking") == ignored(
            "not_armed", "stop", "stop", "speaking"
        )
        clock.now_ms = ARMED_AT
        assert policy.on_utterance_end(final("stop"), "speaking") == interrupt(
            "stop", "stop", "", "speaking"
        )


class TestKeywordWindow:
    """Keywords and phrases match as whole words within the first `keyword_window_words` words."""

    @pytest.mark.parametrize(
        ("text", "keyword"),
        [
            ("stop", "stop"),
            ("Stop!", "stop"),
            ("STOP.", "stop"),
            ("please stop that", "stop"),
            ("okay, I wait", "wait"),
            ("Cancel that.", "cancel"),
            ("hold on", "hold on"),
            ("Oh, hold on a second", "hold on"),
            ("Never  mind.", "never mind"),
            ("never-mind", "never mind"),
            ("Wait, hold on", "wait"),
            ("okay wait, never mind", "wait"),
        ],
    )
    @pytest.mark.parametrize("make_update", [partial, final], ids=["partial", "final"])
    def test_matches(self, text: str, keyword: str, make_update) -> None:
        policy, _, _ = make_policy()
        verdict = policy.evaluate(make_update(text), ARMED_AT, "speaking")
        assert verdict is not None
        assert (verdict.action, verdict.reason, verdict.keyword) == (
            "interrupt",
            "keyword",
            keyword,
        )

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "stopwatch please",
            "nonstop music",
            "waiting room",
            "hold the line",
            "never again mind",
            "well I think we should stop",
            "so um hold on",
            "what does it mean to stop",
            "stop's",
        ],
    )
    @pytest.mark.parametrize("make_update", [partial, final], ids=["partial", "final"])
    def test_does_not_match(self, text: str, make_update) -> None:
        policy, _, _ = make_policy()
        assert policy.evaluate(make_update(text), ARMED_AT, "speaking") is None

    def test_window_size_comes_from_config(self) -> None:
        policy, _, _ = make_policy(config=BargeInConfig(keyword_window_words=1))
        assert policy.evaluate(partial("stop now"), ARMED_AT, "speaking").keyword == "stop"
        assert policy.evaluate(partial("please stop"), ARMED_AT, "speaking") is None
        assert policy.evaluate(partial("hold on"), ARMED_AT, "speaking") is None

    def test_keywords_come_from_config(self) -> None:
        policy, _, _ = make_policy(config=BargeInConfig(keywords=("enough", "shut up")))
        assert policy.evaluate(partial("stop"), ARMED_AT, "speaking") is None
        assert policy.evaluate(partial("Enough!"), ARMED_AT, "speaking").keyword == "enough"
        assert policy.evaluate(partial("oh shut up"), ARMED_AT, "speaking").keyword == "shut up"

    def test_longer_phrase_wins_at_the_same_start(self) -> None:
        policy, _, _ = make_policy(config=BargeInConfig(keywords=("hold", "hold on")))
        verdict = policy.evaluate(partial("hold on, what"), ARMED_AT, "speaking")
        assert (verdict.keyword, verdict.remainder) == ("hold on", "what")


class TestVerdictText:
    """An interrupt names its keyword and keeps what followed it, which becomes the next turn."""

    @pytest.mark.parametrize(
        ("text", "keyword", "remainder"),
        [
            ("stop", "stop", ""),
            ("Stop.", "stop", ""),
            ("Stop, what's the weather tomorrow?", "stop", "what's the weather tomorrow?"),
            ("Hold on. Set a timer for ten minutes", "hold on", "Set a timer for ten minutes"),
            ("okay wait, never mind", "wait", "never mind"),
            ("please stop -- play jazz instead", "stop", "play jazz instead"),
        ],
    )
    def test_keyword_and_remainder(self, text: str, keyword: str, remainder: str) -> None:
        policy, _, _ = make_policy()
        verdict = policy.evaluate(final(text), ARMED_AT, "speaking")
        assert verdict == interrupt(keyword, text, remainder, "speaking")

    def test_utterance_end_carries_the_same_split(self) -> None:
        policy, _, _ = make_policy()
        verdict = policy.on_utterance_end(final("Stop. Turn the lights off"), "speaking")
        assert verdict == interrupt(
            "stop", "Stop. Turn the lights off", "Turn the lights off", "speaking"
        )

    def test_text_is_stripped(self) -> None:
        policy, _, _ = make_policy()
        verdict = policy.evaluate(partial("  stop it  "), ARMED_AT, "speaking")
        assert (verdict.text, verdict.remainder) == ("stop it", "it")


class TestEchoGuard:
    """A keyword MIST herself just said is her echo, unless the user wears a headset."""

    def test_keyword_in_recent_tts_text_is_ignored(self) -> None:
        policy, echo, _ = make_policy(echo=FakeEcho("Okay, I will stop the music now."))
        verdict = policy.evaluate(partial("stop"), ARMED_AT, "speaking")
        assert verdict == ignored("echo_guard", "stop", "stop", "speaking")
        assert echo.windows == [1500]

    def test_phrase_in_recent_tts_text_is_ignored(self) -> None:
        policy, _, _ = make_policy(echo=FakeEcho("You can say Hold On any time."))
        verdict = policy.evaluate(partial("hold on there"), ARMED_AT, "speaking")
        assert verdict == ignored("echo_guard", "hold on", "hold on there", "speaking")

    def test_window_comes_from_config(self) -> None:
        policy, echo, _ = make_policy(
            echo=FakeEcho("stop"), config=BargeInConfig(echo_guard_ms=700)
        )
        policy.evaluate(partial("stop"), ARMED_AT, "speaking")
        assert echo.windows == [700]

    @pytest.mark.parametrize(
        "tts_text", ["", "The bus stopped at the corner.", "a nonstop flight", "hold the door"]
    )
    def test_other_tts_text_does_not_guard(self, tts_text: str) -> None:
        policy, _, _ = make_policy(echo=FakeEcho(tts_text))
        verdict = policy.evaluate(partial("hold on"), ARMED_AT, "speaking")
        assert (verdict.action, verdict.reason) == ("interrupt", "keyword")
        assert policy.evaluate(partial("stop"), ARMED_AT, "speaking").reason == "keyword"

    def test_headset_skips_the_guard(self) -> None:
        policy, echo, _ = make_policy(aec="headset", echo=FakeEcho("I will stop now"))
        verdict = policy.evaluate(partial("stop"), ARMED_AT, "speaking")
        assert verdict == interrupt("stop", "stop", "", "speaking")
        assert echo.windows == []

    @pytest.mark.parametrize("aec", ["browser", "native", "device", "none"])
    def test_every_other_aec_keeps_the_guard(self, aec: AecMode) -> None:
        policy, _, _ = make_policy(aec=aec, echo=FakeEcho("I will stop now"))
        assert policy.evaluate(partial("stop"), ARMED_AT, "speaking").reason == "echo_guard"

    def test_guard_applies_at_utterance_end(self) -> None:
        policy, _, _ = make_policy(echo=FakeEcho("Never mind, I said."))
        verdict = policy.on_utterance_end(final("never mind"), "speaking")
        assert verdict == ignored("echo_guard", "never mind", "never mind", "speaking")

    def test_not_armed_is_checked_before_the_guard(self) -> None:
        policy, echo, _ = make_policy(echo=FakeEcho("stop"))
        verdict = policy.evaluate(partial("stop"), ARMED_AT - 1, "speaking")
        assert verdict.reason == "not_armed"
        assert echo.windows == []


class TestSpeakingUtteranceEnd:
    """Speech without a keyword while MIST speaks is ignored when it ends, and says so."""

    @pytest.mark.parametrize(
        "text", ["What time is it?", "stopwatch please", "I think we should stop"]
    )
    def test_no_keyword_is_ignored_with_text(self, text: str) -> None:
        policy, echo, _ = make_policy()
        verdict = policy.on_utterance_end(final(text), "speaking")
        assert verdict == ignored("no_keyword", None, text, "speaking")
        assert echo.windows == []

    def test_no_keyword_partial_gives_no_verdict(self) -> None:
        policy, _, _ = make_policy()
        assert policy.evaluate(partial("What time is it"), ARMED_AT, "speaking") is None

    def test_no_keyword_even_before_the_arm_delay(self) -> None:
        policy, _, _ = make_policy(clock=FakeClock(AUDIO_AT))
        verdict = policy.on_utterance_end(final("what time is it"), "speaking")
        assert verdict == ignored("no_keyword", None, "what time is it", "speaking")


@pytest.mark.parametrize("phase", ["held", "thinking"])
class TestHeldAndThinking:
    """No audio has played, so only an utterance that is a stop keyword alone interrupts."""

    @pytest.mark.parametrize("text", ["stop", "Hold on", "stop the music", "what time is it"])
    def test_partials_never_interrupt(self, phase: Phase, text: str) -> None:
        policy, _, _ = make_policy(audio_started=False)
        assert policy.evaluate(partial(text), ARMED_AT, phase) is None
        assert policy.evaluate(final(text), ARMED_AT, phase) is None

    @pytest.mark.parametrize(
        ("text", "keyword"),
        [
            ("stop", "stop"),
            ("Stop.", "stop"),
            ("Hold on!", "hold on"),
            ("never mind", "never mind"),
        ],
    )
    def test_stop_keyword_utterance_interrupts(self, phase: Phase, text: str, keyword: str) -> None:
        policy, _, _ = make_policy(audio_started=False)
        assert policy.on_utterance_end(final(text), phase) == interrupt(keyword, text, "", phase)

    def test_no_arm_delay_and_no_echo_guard(self, phase: Phase) -> None:
        echo = FakeEcho("stop stop stop")
        policy, _, _ = make_policy(echo=echo, clock=FakeClock(AUDIO_AT))
        assert policy.on_utterance_end(final("stop"), phase) == interrupt("stop", "stop", "", phase)
        assert echo.windows == []

    @pytest.mark.parametrize(
        "text", ["stop the music", "please stop", "wait what", "what time is it", "stopwatch", ""]
    )
    def test_other_text_gives_no_verdict(self, phase: Phase, text: str) -> None:
        policy, _, _ = make_policy(audio_started=False)
        assert policy.on_utterance_end(final(text), phase) is None


class TestIdleAndText:
    """With MIST silent a keyword interrupts nothing; in `text` the controller never asks."""

    @pytest.mark.parametrize("text", ["stop", "Hold on, one thing", "okay cancel"])
    def test_idle_keyword_is_ignored(self, text: str) -> None:
        policy, echo, _ = make_policy(echo=FakeEcho(text), clock=FakeClock(0))
        keyword = policy.evaluate(partial(text), 0, "idle")
        assert keyword is not None
        assert (keyword.action, keyword.reason, keyword.text) == ("ignored", "idle", text)
        assert policy.on_utterance_end(final(text), "idle") == keyword
        assert echo.windows == []

    @pytest.mark.parametrize("text", ["what time is it", "set a timer that stops"])
    def test_idle_without_keyword_gives_no_verdict(self, text: str) -> None:
        policy, _, _ = make_policy()
        assert policy.evaluate(partial(text), ARMED_AT, "idle") is None
        assert policy.on_utterance_end(final(text), "idle") is None

    def test_text_phase_gives_no_verdict(self) -> None:
        policy, _, _ = make_policy()
        assert policy.evaluate(partial("stop"), ARMED_AT, "text") is None
        assert policy.on_utterance_end(final("stop"), "text") is None


class TestEndpointReport:
    """An endpoint's own keyword spotter interrupts the turn it heard, once armed."""

    def test_armed_matching_turn_interrupts(self) -> None:
        policy, echo, _ = make_policy(echo=FakeEcho("stop"))
        verdict = policy.on_endpoint_report(TURN, "stop", ARMED_AT)
        assert verdict == BargeInVerdict(
            action="interrupt",
            reason="endpoint_kws",
            keyword="stop",
            text="",
            remainder="",
            phase=None,
        )
        assert echo.windows == []

    def test_before_the_arm_delay_is_not_armed(self) -> None:
        policy, _, _ = make_policy()
        verdict = policy.on_endpoint_report(TURN, "stop", ARMED_AT - 1)
        assert verdict == ignored("not_armed", "stop", "", None)

    @pytest.mark.parametrize("now_ms", [ARMED_AT - 1, ARMED_AT])
    def test_other_turn_is_stale(self, now_ms: int) -> None:
        policy, _, _ = make_policy()
        verdict = policy.on_endpoint_report("turn-0", "stop", now_ms)
        assert verdict == ignored("stale_turn", "stop", "", None)

    def test_report_before_any_turn_audio_is_stale(self) -> None:
        policy, _, _ = make_policy(audio_started=False)
        verdict = policy.on_endpoint_report(TURN, "stop", ARMED_AT)
        assert verdict == ignored("stale_turn", "stop", "", None)

    def test_earlier_turn_is_stale_after_a_new_one_starts(self) -> None:
        policy, _, _ = make_policy()
        policy.on_turn_audio_started("turn-2", 20_000)
        assert policy.on_endpoint_report(TURN, "stop", 30_000).reason == "stale_turn"
        assert policy.on_endpoint_report("turn-2", "stop", 30_000).reason == "endpoint_kws"


class TestVerdict:
    """The verdict maps onto the wire's decision details without translation."""

    @pytest.mark.parametrize(
        ("action", "reason"),
        [
            ("interrupt", "no_keyword"),
            ("interrupt", "idle"),
            ("ignored", "keyword"),
            ("ignored", "endpoint_kws"),
            ("interrupt", "ui"),
        ],
    )
    def test_reason_must_belong_to_action(self, action: str, reason: str) -> None:
        with pytest.raises(ValueError, match=reason):
            BargeInVerdict(
                action=action, reason=reason, keyword=None, text="", remainder="", phase=None
            )

    def test_interrupt_fills_the_barge_in_decision(self) -> None:
        policy, _, _ = make_policy()
        verdict = policy.evaluate(final("Stop, play jazz"), ARMED_AT, "speaking")
        message = VoiceDecisionBargeIn(
            turn_id=TURN,
            reason=verdict.reason,
            detail=BargeInDetail(keyword=verdict.keyword, phase=verdict.phase, utterance_id="u1"),
        )
        assert message.detail.keyword == "stop"

    @pytest.mark.parametrize(
        ("text", "phase", "now_ms", "reason"),
        [
            ("stop", "speaking", ARMED_AT - 1, "not_armed"),
            ("what time", "speaking", ARMED_AT, "no_keyword"),
            ("stop", "idle", ARMED_AT, "idle"),
        ],
    )
    def test_ignored_fills_the_ignored_decision(
        self, text: str, phase: Phase, now_ms: int, reason: str
    ) -> None:
        policy, _, _ = make_policy(clock=FakeClock(now_ms))
        verdict = policy.on_utterance_end(final(text), phase)
        message = VoiceDecisionBargeInIgnored(
            turn_id=TURN,
            reason=verdict.reason,
            detail=BargeInIgnoredDetail(
                keyword=verdict.keyword, text=verdict.text, utterance_id="u1"
            ),
        )
        assert (message.reason, message.detail.text) == (reason, text)
