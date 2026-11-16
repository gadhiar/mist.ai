"""Two-ASR transcripts, their agreement and the prosody rate they feed (T8).

The ASRs are the fakes of `fakes.py`; the clean run's own check of the same rules, through the
command, is in `test_clean.py`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from scripts.voice_data.bindings.protocols import Word
from scripts.voice_data.records import Prosody, SegmentRecord
from scripts.voice_data.stages import prosody as prosody_mod
from scripts.voice_data.stages import transcribe

from .fakes import FakeAsr

RATE = 24000
TEXT = "seven quiet lanterns drift above the harbour tonight"


def record(reasons: tuple[str, ...] = ()) -> SegmentRecord:
    return SegmentRecord(
        segment_id="seg-000001", source_id="src-0001", start_s=1.0, end_s=5.0, channel="FC",
        channel_reason=None, separator="none", audio="seed/runs/r1/audio/seg-000001.flac",
        diar_cluster="spk-01", cluster="primary", flag_group=None,
        sim={"wavlm-sv": 0.9, "ecapa": 0.8}, overlap_ms=0, snr_db=30.0, c50_db=40.0,
        transcripts={}, asr_agreement_cer=0.0,
        prosody=Prosody(f0_mean_hz=210.0, f0_std_hz=4.0, rate_sps=0.0, energy_db=-20.0),
        arousal=0.4, state_proposed="neutral", state=None, set="short", kept=True,
        reasons=reasons,
    )  # fmt: skip


def audio(seconds: float = 4.0) -> np.ndarray:
    return np.zeros(int(seconds * RATE), dtype=np.float32)


class WordsSeen:
    """A prosody measure that records the words it was given and rates from them."""

    def __init__(self) -> None:
        self.words: list[tuple[Word, ...]] = []

    def __call__(self, clip: np.ndarray, rate: int, words: Sequence[Word]) -> Prosody:
        self.words.append(tuple(words))
        return Prosody(
            f0_mean_hz=200.0, f0_std_hz=5.0, rate_sps=prosody_mod.rate_from_words(words),
            energy_db=-20.0,
        )  # fmt: skip


def annotate(timing_text: str, verbatim_text: str, reasons: tuple[str, ...] = ()):
    clip = audio()
    result = transcribe.transcribe(
        clip, RATE, FakeAsr("whisperx", timing_text), FakeAsr("crisper", verbatim_text)
    )
    return result, transcribe.annotate(record(reasons), clip, RATE, result, WordsSeen())


def test_both_transcripts_are_recorded_under_their_role_keys() -> None:
    _, out = annotate(TEXT, TEXT)
    assert out.transcripts == {transcribe.TIMING_KEY: TEXT, transcribe.VERBATIM_KEY: TEXT}
    assert transcribe.TIMING_KEY == "whisperx" and transcribe.VERBATIM_KEY == "crisper"


def test_agreeing_transcripts_keep_the_segment_in_training_text() -> None:
    result, out = annotate(TEXT, TEXT)
    assert result.cer == 0.0 and result.agree
    assert out.asr_agreement_cer == 0.0
    assert transcribe.ASR_DISAGREE not in out.reasons
    assert transcribe.training_text(out) == TEXT


def test_cer_above_ten_percent_marks_asr_disagree_and_excludes_training_text() -> None:
    result, out = annotate(TEXT, "seven loud lanterns sink below the river at noon")
    assert result.cer > transcribe.AGREEMENT_CER_MAX
    assert not result.agree
    assert out.reasons == (transcribe.ASR_DISAGREE,)
    assert out.asr_agreement_cer == pytest.approx(result.cer, abs=1e-4)
    assert out.kept  # the audio stays in the seed; only its text is excluded
    assert transcribe.training_text(out) is None


def test_exactly_ten_percent_agrees_and_just_above_does_not() -> None:
    at, _ = annotate("abcdefghiz", "abcdefghij")
    above, _ = annotate("abcdefghzz", "abcdefghij")
    assert at.cer == pytest.approx(0.10) and at.agree
    assert above.cer == pytest.approx(0.20) and not above.agree


def test_agreement_cer_uses_the_verbatim_transcript_as_reference() -> None:
    # One word dropped by the timing ASR is a deletion against the verbatim reference.
    result, _ = annotate("one two three", "one two three four")
    assert result.cer == pytest.approx(5 / len("one two three four"))


def test_earlier_reasons_are_kept() -> None:
    _, out = annotate("abc", "xyz", reasons=("earlier",))
    assert out.reasons == ("earlier", transcribe.ASR_DISAGREE)


def test_word_times_of_the_timing_asr_feed_the_prosody_rate() -> None:
    clip = audio(5.0)
    timing = FakeAsr("whisperx", TEXT)
    result = transcribe.transcribe(clip, RATE, timing, FakeAsr("crisper", "no times here"))
    seen = WordsSeen()
    out = transcribe.annotate(record(), clip, RATE, result, seen)
    expected = timing.transcribe(clip, RATE).words
    assert seen.words == [expected]
    assert out.prosody.rate_sps == pytest.approx(prosody_mod.rate_from_words(expected))
    assert out.prosody.rate_sps > 0.0


def test_a_record_without_the_verbatim_transcript_has_no_training_text() -> None:
    assert transcribe.training_text(record()) is None
