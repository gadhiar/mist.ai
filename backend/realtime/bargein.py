"""Barge-in policy: what user speech means while MIST speaks, holds a reply or thinks.

The turn controller sees user speech arrive in every phase, but only some of it is meant for the
turn in flight. Deciding which is a policy, kept apart from the controller so that the keyword
rules here (path a, Design doc, Barge-in policy) can later be replaced by an open-microphone
policy behind the same `BargeInPolicy` protocol without touching the controller. A policy only
judges: it returns a `BargeInVerdict` (interrupt, or ignored with a reason) or no verdict, and
the controller acts on it, sends the `voice_decision` and fills in what only it knows, the
`turn_id` and the `utterance_id`.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from .budget import BargeInConfig
from .speech import Phase, TranscriptUpdate

AecMode = Literal["browser", "native", "device", "headset", "none"]
"""The client's echo cancellation, as `voice_start.aec` declares it.

Repeated here because the wire model declares it inline; a test pins the two together.
"""

VerdictAction = Literal["interrupt", "ignored"]

InterruptReason = Literal["keyword", "endpoint_kws"]
"""The `barge_in` reasons a keyword policy can give; the others (`ui`, `wake`, `voice_stop`) are
the controller's own."""

IgnoreReason = Literal["not_armed", "echo_guard", "no_keyword", "stale_turn", "idle"]

_INTERRUPT_REASONS = frozenset({"keyword", "endpoint_kws"})
_IGNORE_REASONS = frozenset({"not_armed", "echo_guard", "no_keyword", "stale_turn", "idle"})

# A word is a run of letters or digits, joined across an apostrophe so that "stop's" is one word
# and never the keyword "stop". Everything else (punctuation, hyphens, spaces) separates words,
# which is how Whisper's "Stop." and "never-mind" still match.
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
# What separates a keyword from the request after it ("Stop, what's the weather"), dropped so the
# remainder reads as a request of its own.
_LEADING_SEPARATORS = re.compile(r"^[\s,.;:!?\-–—…]+")


@dataclass(frozen=True, slots=True)
class BargeInVerdict:
    """A policy's judgement of user speech, carrying what the wire's decision details need.

    `action` `interrupt` becomes `voice_decision` kind `barge_in`, and `ignored` becomes kind
    `barge_in_ignored`; `reason` is the decision's reason unchanged. `keyword` is the configured
    keyword in its normalized form, or `None` when none was heard. `text` is the transcript the
    verdict judged (stripped), empty for an endpoint report, which has none. `remainder` is the
    text after the keyword, which the controller makes the next turn's text when the utterance
    ends; it is empty for every `ignored` verdict. `phase` is the phase the policy was given, and
    `None` for an endpoint report, which arrives without one, so the controller supplies its own.

    Raises:
        ValueError: when `reason` is not one of `action`'s reasons, so a verdict can never put an
            ignore reason on a `barge_in` decision or the reverse.
    """

    action: VerdictAction
    reason: InterruptReason | IgnoreReason
    keyword: str | None
    text: str
    remainder: str
    phase: Phase | None

    def __post_init__(self) -> None:
        allowed = _INTERRUPT_REASONS if self.action == "interrupt" else _IGNORE_REASONS
        if self.reason not in allowed:
            raise ValueError(f"reason {self.reason!r} is not a reason for action {self.action!r}")


class BargeInPolicy(Protocol):
    """Judges user speech for the turn controller, which calls it from its mailbox task only."""

    def on_turn_audio_started(self, turn_id: str, at_ms: int) -> None:
        """Record that `turn_id`'s first audio reached the client at `at_ms` (session clock)."""
        ...

    def evaluate(
        self, update: TranscriptUpdate, now_ms: int, phase: Phase
    ) -> BargeInVerdict | None:
        """Judge a transcript while the utterance goes on; `None` means no verdict yet."""
        ...

    def on_endpoint_report(self, turn_id: str, keyword: str, now_ms: int) -> BargeInVerdict | None:
        """Judge an endpoint's `barge_in` report, heard by its own keyword spotter."""
        ...

    def on_utterance_end(self, update: TranscriptUpdate, phase: Phase) -> BargeInVerdict | None:
        """Judge an utterance's final transcript; `None` leaves it to the phase's own rule."""
        ...


@dataclass(frozen=True, slots=True)
class _Hit:
    """A configured keyword found in a transcript, and where its last word ends in the text."""

    keyword: str
    end_char: int


class KeywordBargeIn:
    """The keyword policy: a configured word or phrase near the start of speech interrupts.

    Open-microphone speech while MIST talks is mostly not addressed to her, and on speakers part
    of it is her own voice leaking through the echo canceller. So in `speaking` only a keyword in
    the first `keyword_window_words` words counts, and only once the turn has played for
    `arm_delay_ms` (the canceller converges by then), and not when MIST herself said that keyword
    within `echo_guard_ms`. A user wearing a headset hears no leak, so `aec` `headset` turns the
    echo guard off. In `held` and `thinking` nothing has played, so neither the arm delay nor the
    echo guard applies, and only a keyword-only utterance, or one name then a keyword-only
    utterance, interrupts: any other speech is a new request, which the controller merges in by
    superseding. The name is allowed so that "mist, stop" means what "stop" means: a final
    transcript may keep the name said before the keyword. Only one leading name with a
    word after it is skipped; a trailing name ("stop, mist") or a greeting before the name
    ("hey mist, stop") still makes a request.

    The policy is per voice session: the session builds one at each `voice_start`, with that
    message's `aec`. It keeps one piece of state, the turn whose audio started last; partial
    verdicts are not deduplicated, because which partials to report is the controller's call.

    Args:
        config: The keywords, the arm delay, the keyword window and the echo guard window.
        aec: The client's declared echo cancellation.
        names: Her configured names (`WakeConfig.names`). Each is held as its own words, since
            a name may hold a hyphen, which separates the words of a transcript ("mist-ai" is
            "mist ai").
        recent_tts_text: The text MIST played in the last `window_ms`, shaped like
            `PlaybackLedger.recent_tts_text` for the current turn's ledger.
        clock: The session's monotonic millisecond clock, read for the arm delay at an
            utterance end, whose call carries no time of its own.
    """

    def __init__(
        self,
        config: BargeInConfig,
        *,
        aec: AecMode,
        names: tuple[str, ...],
        recent_tts_text: Callable[[int], str],
        clock: Callable[[], int],
    ) -> None:
        self._config = config
        self._echo_guarded = aec != "headset"
        self._recent_tts_text = recent_tts_text
        self._clock = clock
        # Keywords as word tuples, normalized the same way as transcripts so that a keyword
        # configured as "Hold  on" still matches "hold on".
        self._keywords = tuple(
            (" ".join(words), words) for words in (_words(k) for k in config.keywords) if words
        )
        self._names = tuple(words for words in (_words(n) for n in names) if words)
        self._turn_id: str | None = None
        self._audio_at_ms = 0

    def on_turn_audio_started(self, turn_id: str, at_ms: int) -> None:
        """Arm the delay from this turn's first audio; any earlier turn's reports become stale."""
        self._turn_id = turn_id
        self._audio_at_ms = at_ms

    def evaluate(
        self, update: TranscriptUpdate, now_ms: int, phase: Phase
    ) -> BargeInVerdict | None:
        """Judge a partial (or a final before the utterance end) as it arrives.

        In `held` and `thinking` a partial never interrupts: the user may be saying "stop the
        music" rather than "stop", and only the end tells them apart. Speech without a keyword
        gets no verdict here, since it is judged once, at its end.
        """
        if phase in ("speaking", "idle"):
            return self._judge_window(update.text, now_ms, phase)
        return None

    def on_endpoint_report(self, turn_id: str, keyword: str, now_ms: int) -> BargeInVerdict | None:
        """Accept an endpoint's keyword for the current turn once it is armed.

        The endpoint ran its own spotter on its own audio, so there is no transcript to window or
        echo-guard; only the turn and the arm delay are checked. A report for any turn but the
        one whose audio started last is stale: it was heard during a turn already over.
        """
        word = " ".join(_words(keyword)) or keyword
        if turn_id != self._turn_id:
            return _ignored("stale_turn", word, "", None)
        if not self._armed(now_ms):
            return _ignored("not_armed", word, "", None)
        return BargeInVerdict(
            action="interrupt",
            reason="endpoint_kws",
            keyword=word,
            text="",
            remainder="",
            phase=None,
        )

    def on_utterance_end(self, update: TranscriptUpdate, phase: Phase) -> BargeInVerdict | None:
        """Judge the final transcript when the utterance ends.

        In `speaking` speech without a keyword is ignored with reason `no_keyword`, so the user
        sees why MIST talked on. In `held` and `thinking` a keyword-only utterance, or one name
        then a keyword-only utterance, interrupts and anything else gets no verdict, which the
        controller treats as a new request.
        """
        if phase in ("speaking", "idle"):
            verdict = self._judge_window(update.text, self._clock(), phase)
            if verdict is None and phase == "speaking":
                return _ignored("no_keyword", None, update.text.strip(), phase)
            return verdict
        if phase in ("held", "thinking"):
            return self._judge_whole(update.text, phase)
        return None

    def _judge_window(self, text: str, now_ms: int, phase: Phase) -> BargeInVerdict | None:
        hit = self._find_in_window(text)
        if hit is None:
            return None
        stripped = text.strip()
        if phase == "idle":
            return _ignored("idle", hit.keyword, stripped, phase)
        if not self._armed(now_ms):
            return _ignored("not_armed", hit.keyword, stripped, phase)
        if self._echo_guarded and self._heard_from_tts(hit.keyword):
            return _ignored("echo_guard", hit.keyword, stripped, phase)
        remainder = _LEADING_SEPARATORS.sub("", text[hit.end_char :]).strip()
        return BargeInVerdict(
            action="interrupt",
            reason="keyword",
            keyword=hit.keyword,
            text=stripped,
            remainder=remainder,
            phase=phase,
        )

    def _judge_whole(self, text: str, phase: Phase) -> BargeInVerdict | None:
        """Interrupt for an utterance that is a keyword alone, or one leading name then one.

        The verdict's `text` is the whole utterance, name included, as the user said it.
        """
        said = _words(text)
        candidates = [said]
        candidates.extend(
            said[len(name) :]
            for name in self._names
            if len(said) > len(name) and said[: len(name)] == name
        )
        for keyword, parts in self._keywords:
            if parts in candidates:
                return BargeInVerdict(
                    action="interrupt",
                    reason="keyword",
                    keyword=keyword,
                    text=text.strip(),
                    remainder="",
                    phase=phase,
                )
        return None

    def _find_in_window(self, text: str) -> _Hit | None:
        """The earliest keyword lying wholly in the window; the longest one at a tied start.

        The window counts words of the transcript, and a phrase must end inside it: "so um hold
        on" has "on" as its fourth word and does not match with a window of three.
        """
        matches = list(_WORD.finditer(text))[: self._config.keyword_window_words]
        words = tuple(m.group().lower() for m in matches)
        for start in range(len(words)):
            best: tuple[str, tuple[str, ...]] | None = None
            for keyword, parts in self._keywords:
                if words[start : start + len(parts)] == parts and (
                    best is None or len(parts) > len(best[1])
                ):
                    best = (keyword, parts)
            if best is not None:
                return _Hit(best[0], matches[start + len(best[1]) - 1].end())
        return None

    def _armed(self, now_ms: int) -> bool:
        return self._turn_id is not None and now_ms >= self._audio_at_ms + self._config.arm_delay_ms

    def _heard_from_tts(self, keyword: str) -> bool:
        played = " ".join(_words(self._recent_tts_text(self._config.echo_guard_ms)))
        return f" {keyword} " in f" {played} "


def _words(text: str) -> tuple[str, ...]:
    """The normalized words of `text`: lower case, punctuation dropped."""
    return tuple(m.group().lower() for m in _WORD.finditer(text))


def _ignored(
    reason: IgnoreReason, keyword: str | None, text: str, phase: Phase | None
) -> BargeInVerdict:
    return BargeInVerdict(
        action="ignored", reason=reason, keyword=keyword, text=text, remainder="", phase=phase
    )
