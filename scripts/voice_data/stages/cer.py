"""Character error rate, for the take gate (VC-044) and the agreement of a segment's two ASRs.

Both texts go through the speech bench's `normalise` first, imported unchanged, so a take is
scored the way the bench scores the student: numbers as words, punctuation and case removed.
The normalised words are joined with single spaces and aligned character by character with the
bench's own `align`, so the space between two words counts as a character.
"""

from __future__ import annotations

from scripts.model_bench.speech.wer import align, normalise


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return character edits over the normalised reference's length.

    An empty normalised reference scores the hypothesis's character count, as `wer` does for
    words, so an empty script with a non-empty take never passes a gate.
    """
    ref = " ".join(normalise(reference))
    hyp = " ".join(normalise(hypothesis))
    return align(list(ref), list(hyp)).wer
