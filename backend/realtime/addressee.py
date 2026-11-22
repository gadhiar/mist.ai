"""The addressee gate: whether a name hit was said to MIST.

The name spotter answers only "was the name heard". Wake by name would otherwise open a turn for
her own voice saying her name, for a film on the PC saying it, and for every ordinary use of the
word ("the mist rolled in"). The gate turns that hit into a verdict in two steps: `pre_decode`
rejects her own speech and played media, and `verified` accepts the hit when the acoustic
verifier's score reaches the threshold for the spotter's cutoff. No transcript text takes part
in the verdict.

The module is pure: it reads a `WakeConfig`, strings and scores, and imports no controller,
ingress or engine, so it runs, and is tested, without a model loaded. `WakeConfig` is
re-exported so that a caller holding only this module can build one.
"""

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from .budget import WakeConfig

__all__ = [
    "AddresseeGate",
    "GateCheck",
    "GateVerdict",
    "WakeConfig",
    "normalize",
]

GateCheck = Literal["verified", "self_speech", "self_media", "not_addressed"]

_WORD = r"[a-z0-9]+(?:'[a-z0-9]+)*"
_TEXT_TOKEN = re.compile(rf"{_WORD}|[,?.]")
_TYPOGRAPHIC_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """The gate's answer for one name hit, with the scores the wake event reports.

    `check` names the check that decided, accepted or not, so that one rejection reason can be
    told apart from another.
    """

    accepted: bool
    check: GateCheck
    score: float
    reference_score: float | None


def normalize(text: str) -> list[str]:
    """Split a transcript into lower-case words and the punctuation marks ",", "?" and ".".

    Other punctuation is dropped. A typographic apostrophe reads as a plain one, so "what’s"
    from one recogniser equals "what's" from another. `pre_decode` reads her recent speech's
    words through it.
    """
    return _TEXT_TOKEN.findall(text.lower().translate(_TYPOGRAPHIC_APOSTROPHES))


class AddresseeGate:
    """Stage 2 of wake by name: turns a name hit into an accepted or rejected verdict.

    `echo_guard_ms` is how far back the controller asks the registry for her spoken text before
    calling `pre_decode`; the gate holds it so that the self-speech policy lives in one object.

    `thresholds` maps a spotter cutoff to the score the acoustic verifier must reach for a hit
    at that cutoff, None for a cutoff below its table; the caller passes the loaded verifier's
    `threshold_at`. It is a callable rather than the verifier so that this module stays free of
    engines. Without it `verified` accepts nothing, so a gate built with no verifier fails closed.
    """

    class Thresholds(Protocol):
        """A cutoff-to-threshold lookup, such as `WakeVerifier.threshold_at`.

        A protocol rather than `collections.abc.Callable`, because the module's imports are
        pinned to `re`, `dataclasses` and `typing` besides `.budget`.
        """

        def __call__(self, cutoff: float, /) -> float | None:
            """The score a hit at `cutoff` must reach, or None below the table."""
            ...

    def __init__(
        self,
        config: WakeConfig,
        *,
        echo_guard_ms: int,
        thresholds: Thresholds | None = None,
    ) -> None:
        self._config = config
        self._names = frozenset(config.names)
        self._thresholds = thresholds
        self.echo_guard_ms = echo_guard_ms

    def verified(
        self,
        *,
        score: float,
        reference_score: float | None,
        verify_score: float | None,
        cutoff: float,
    ) -> GateVerdict:
        """Accept a hit when the verifier's score reaches the threshold for the spotter's cutoff.

        `score` and `reference_score` are the spotter's, carried into the verdict for the wake
        event. `verify_score` is None when the verifier failed, the window was too short or no
        verifier is loaded, and the threshold is None for a cutoff below the table; each of
        those rejects the hit `not_addressed`, as a score below the threshold does.
        """
        threshold = self._thresholds(cutoff) if self._thresholds is not None else None
        accepted = verify_score is not None and threshold is not None and verify_score >= threshold
        return GateVerdict(
            accepted=accepted,
            check="verified" if accepted else "not_addressed",
            score=score,
            reference_score=reference_score,
        )

    def pre_decode(
        self, *, score: float, reference_score: float | None, recent_tts_text: str
    ) -> GateVerdict | None:
        """Reject a hit her own output explains, or return None when it explains none.

        `self_media` is checked first: a reference score at or above `kws_threshold` means the
        PC played the name, which is direct acoustic evidence, while her recent text only says
        she might have. `self_speech` does not wait for the reference to be absent: a low
        reference clears only this session's playback, and another session's speaker (the room
        unit's, say) can reach this microphone, which is why `recent_tts_text` spans every
        session. Any name token in her recent text counts, whatever its position, because the
        spotter hears the sound of the name, not its grammar.
        """
        if reference_score is not None and reference_score >= self._config.kws_threshold:
            return GateVerdict(
                accepted=False, check="self_media", score=score, reference_score=reference_score
            )
        if any(token in self._names for token in normalize(recent_tts_text)):
            return GateVerdict(
                accepted=False, check="self_speech", score=score, reference_score=reference_score
            )
        return None
