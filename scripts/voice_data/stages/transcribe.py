"""Two-ASR transcripts of a cleaned segment, their agreement, and the prosody they feed (Design 5).

The timing ASR (WhisperX) gives word times, and the verbatim ASR (CrisperWhisper) gives the text
with its disfluencies, which is the segment's training text. The agreement is the CER of the
timing text against the verbatim text, by `cer.character_error_rate`, the same measure as the
take gate. A segment whose two transcripts disagree by more than 10% keeps its audio in the seed,
but its text is not trusted for training: its record carries the reason `asr_disagree`, and
`training_text` returns `None` for it.

Transcripts are stored under the role keys of the segment record (`whisperx`, `crisper`), not
under the bindings' model names, so later stages read them without knowing which build ran.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..bindings.protocols import Asr, Transcript, Word
from ..records import Prosody, SegmentRecord
from .cer import character_error_rate

AGREEMENT_CER_MAX = 0.10
ASR_DISAGREE = "asr_disagree"
TIMING_KEY = "whisperx"
VERBATIM_KEY = "crisper"

ProsodyMeasure = Callable[[np.ndarray, int, Sequence[Word]], Prosody]


@dataclass(frozen=True, slots=True)
class Transcription:
    """A segment's two transcripts and the CER between them."""

    timing: Transcript
    verbatim: Transcript
    cer: float

    @property
    def agree(self) -> bool:
        """Whether the transcripts agree closely enough for the text to be used in training."""
        return self.cer <= AGREEMENT_CER_MAX

    def texts(self) -> dict[str, str]:
        """Return both texts under the segment record's role keys."""
        return {TIMING_KEY: self.timing.text, VERBATIM_KEY: self.verbatim.text}


def transcribe(audio: np.ndarray, sample_rate: int, timing: Asr, verbatim: Asr) -> Transcription:
    """Transcribe the clip with both ASRs and measure their agreement.

    The verbatim text is the CER's reference, because it is the text training would use.
    """
    timed = timing.transcribe(audio, sample_rate)
    exact = verbatim.transcribe(audio, sample_rate)
    return Transcription(timed, exact, character_error_rate(exact.text, timed.text))


def annotate(
    record: SegmentRecord,
    audio: np.ndarray,
    sample_rate: int,
    result: Transcription,
    measure: ProsodyMeasure,
) -> SegmentRecord:
    """Return the record with its transcripts, agreement and prosody from the cleaned audio.

    The prosody is measured again on the cleaned clip, and its speech rate comes from the timing
    ASR's word times rather than T7's energy-nucleus estimate.
    """
    reasons = record.reasons if result.agree else (*record.reasons, ASR_DISAGREE)
    return replace(
        record,
        transcripts=result.texts(),
        asr_agreement_cer=round(result.cer, 4),
        prosody=measure(audio, sample_rate, result.timing.words),
        reasons=reasons,
    )


def training_text(record: SegmentRecord) -> str | None:
    """Return the verbatim text; `None` for a dropped segment, a disagreement or no transcript."""
    if not record.kept or ASR_DISAGREE in record.reasons:
        return None
    return record.transcripts.get(VERBATIM_KEY)
