"""Reference choice per state (T8): 2-4 clips, 3-12 s, at the leave-one-out median or above.

The choice runs on `references.Candidate` values here; the clean run that writes the chosen clips
under `targets/<id>/refs/<state>/` is tested end to end in `test_clean.py`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.voice_data.commands import clean as clean_cmd
from scripts.voice_data.records import Prosody
from scripts.voice_data.stages import references
from scripts.voice_data.states import SPEAKING_STATES

MEDIANS = {"wavlm-sv": 0.8, "ecapa": 0.7}


def cand(
    sid: str,
    source: str,
    state: str = "neutral",
    *,
    seconds: float = 5.0,
    snr: float = 30.0,
    sim: tuple[float, float] = (0.9, 0.8),
    rate: float = 4.0,
    transcript: str = "a line both passes agree on",
    disagree: bool = False,
) -> references.Candidate:
    return references.Candidate(
        segment_id=sid,
        source_id=source,
        state=state,
        duration_s=seconds,
        snr_db=snr,
        prosody=Prosody(f0_mean_hz=210.0, f0_std_hz=5.0, rate_sps=rate, energy_db=-20.0),
        sim={"wavlm-sv": sim[0], "ecapa": sim[1]},
        transcript=transcript,
        asr_disagree=disagree,
    )


def chosen_ids(choice: references.StateChoice) -> list[str]:
    return [c.segment_id for c in choice.chosen]


def test_every_speaking_state_gets_a_choice() -> None:
    choices = references.choose([cand("a", "s1"), cand("b", "s2")], MEDIANS)
    assert list(choices) == list(SPEAKING_STATES)
    assert choices["neutral"].short is False
    assert all(choices[s].short for s in SPEAKING_STATES if s != "neutral")


def test_at_most_four_references_per_state() -> None:
    pool = [cand(f"n{i}", f"s{i}", rate=4.0 + 0.1 * i) for i in range(6)]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert len(choice.chosen) == references.MAX_REFS == 4
    assert choice.eligible == 6 and choice.members == 6


def test_two_eligible_candidates_make_two_references() -> None:
    choice = references.choose([cand("a", "s1"), cand("b", "s1")], MEDIANS)["neutral"]
    assert sorted(chosen_ids(choice)) == ["a", "b"]
    assert references.MIN_REFS == 2


def test_distinct_sources_are_preferred() -> None:
    # Three clips of s1 sit nearest the state's prosody; two from other sources still join.
    pool = [
        cand("a1", "s1", rate=4.0),
        cand("a2", "s1", rate=4.0),
        cand("a3", "s1", rate=4.1),
        cand("b1", "s2", rate=5.5),
        cand("c1", "s3", rate=6.0),
    ]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert len(choice.chosen) == 4
    assert {"b1", "c1"} <= set(chosen_ids(choice))
    assert len({c.source_id for c in choice.chosen}) == 3


def test_a_single_source_still_fills_the_state() -> None:
    pool = [cand(f"a{i}", "s1", rate=4.0 + 0.1 * i) for i in range(5)]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert len(choice.chosen) == 4


@pytest.mark.parametrize(
    ("seconds", "eligible"), [(2.9, False), (3.0, True), (12.0, True), (12.1, False)]
)
def test_references_are_3_to_12_seconds(seconds: float, eligible: bool) -> None:
    pool = [cand("x", "s1", seconds=seconds), cand("a", "s2"), cand("b", "s3")]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert ("x" in chosen_ids(choice)) is eligible
    assert all(3.0 <= c.duration_s <= 12.0 for c in choice.chosen)


@pytest.mark.parametrize(
    ("sim", "eligible"),
    [((0.8, 0.7), True), ((0.79, 0.9), False), ((0.9, 0.69), False), ((0.95, 0.95), True)],
)
def test_references_are_at_or_above_the_leave_one_out_median_on_both_embedders(
    sim: tuple[float, float], eligible: bool
) -> None:
    pool = [cand("x", "s1", sim=sim), cand("a", "s2"), cand("b", "s3")]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert ("x" in chosen_ids(choice)) is eligible


def test_a_state_with_fewer_than_two_candidates_is_short_and_never_padded() -> None:
    pool = [
        cand("ok", "s1", "sleepy"),
        cand("short", "s2", "sleepy", seconds=2.0),
        cand("low", "s3", "sleepy", sim=(0.5, 0.5)),
        cand("long", "s1", "sleepy", seconds=20.0),
    ]
    choice = references.choose(pool, MEDIANS)["sleepy"]
    assert choice.short
    assert choice.chosen == ()
    assert choice.eligible == 1 and choice.members == 4


def test_a_state_with_no_segment_is_short() -> None:
    choice = references.choose([cand("a", "s1"), cand("b", "s2")], MEDIANS)["excited"]
    assert choice.short and choice.chosen == () and choice.members == 0


def test_prosody_nearest_the_state_median_is_chosen_first() -> None:
    rates = {"r2": 2.0, "r4a": 4.0, "r4b": 4.1, "r4c": 3.9, "r45": 4.5, "r9": 9.0}
    pool = [cand(sid, f"s-{sid}", rate=r) for sid, r in rates.items()]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert set(chosen_ids(choice)) == {"r4a", "r4b", "r4c", "r45"}
    assert chosen_ids(choice)[0] == "r4a"


def test_higher_snr_breaks_a_prosody_tie() -> None:
    pool = [cand(f"t{i}", f"s{i}", snr=20.0 + i) for i in range(5)]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert chosen_ids(choice) == ["t4", "t3", "t2", "t1"]


def test_the_profile_comes_from_every_member_of_the_state() -> None:
    # The ineligible clips still set the median, so the chosen clips sit near the state's centre.
    pool = [cand(f"x{i}", "s1", rate=8.0, seconds=2.0) for i in range(5)]
    pool += [cand("near", "s2", rate=7.5), cand("mid", "s3", rate=5.0), cand("far", "s4", rate=2.0)]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert chosen_ids(choice)[0] == "near"


# References come only from segments both ASR passes agree on (v2).


def test_an_asr_disagree_candidate_is_never_chosen_even_when_it_ranks_highest() -> None:
    # "x" sits on the state's prosody median and has the highest SNR, so it would rank first.
    pool = [
        cand("x", "s1", rate=4.0, snr=60.0, disagree=True),
        cand("a", "s2", rate=4.0, snr=20.0),
        cand("b", "s3", rate=4.0, snr=20.0),
        cand("c", "s4", rate=4.0, snr=20.0),
    ]
    agreed = [replace(c, asr_disagree=False) if c.segment_id == "x" else c for c in pool]
    assert chosen_ids(references.choose(agreed, MEDIANS)["neutral"])[0] == "x"
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert "x" not in chosen_ids(choice)
    assert sorted(chosen_ids(choice)) == ["a", "b", "c"]
    assert choice.members == 4 and choice.eligible == 3


@pytest.mark.parametrize("transcript", ["", "   ", "\n\t"])
def test_a_candidate_with_no_verbatim_text_is_never_chosen(transcript: str) -> None:
    pool = [cand("x", "s1", snr=60.0, transcript=transcript), cand("a", "s2"), cand("b", "s3")]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert sorted(chosen_ids(choice)) == ["a", "b"]
    assert not references.eligible(pool[0], MEDIANS)


def test_a_state_left_with_one_agreed_candidate_is_short_with_no_references() -> None:
    pool = [
        cand("x", "s1", "calm", snr=60.0, disagree=True),
        cand("y", "s2", "calm", transcript=" "),
        cand("a", "s3", "calm"),
    ]
    choice = references.choose(pool, MEDIANS)["calm"]
    assert choice.short
    assert choice.chosen == ()
    assert choice.eligible == 1 and choice.members == 3


def test_a_disagreeing_member_still_sets_the_state_profile() -> None:
    # Three disagreeing clips at rate 8 pull the median there, so "near" ranks before "far".
    pool = [cand(f"d{i}", "s1", rate=8.0, disagree=True) for i in range(3)]
    pool += [cand("near", "s2", rate=7.5), cand("mid", "s3", rate=5.0), cand("far", "s4", rate=2.0)]
    choice = references.choose(pool, MEDIANS)["neutral"]
    assert chosen_ids(choice)[0] == "near"
    assert choice.members == 6 and choice.eligible == 3


def test_transcript_paths_lie_beside_their_reference() -> None:
    assert clean_cmd.transcript_rel("calm", 2) == "refs/calm/2.private.txt"


def test_a_transcript_is_written_as_its_utf8_bytes_exactly(tmp_path: Path) -> None:
    path = tmp_path / "refs" / "neutral" / "1.private.txt"
    text = "um, so\r\nthe café is — open "
    references.write_transcript(path, text)
    assert path.read_bytes() == text.encode("utf-8")
    assert not list(path.parent.glob("*.part"))


def test_reference_paths_follow_the_target_layout() -> None:
    assert references.reference_rel("calm", 2) == "refs/calm/2.wav"


def test_references_are_written_as_wav_pcm16_mono_24khz(tmp_path: Path) -> None:
    path = tmp_path / "refs" / "neutral" / "1.wav"
    clip = (0.3 * np.sin(np.arange(24000) / 10.0)).astype(np.float32)
    references.write_wav(path, clip, 24000)
    info = sf.info(str(path))
    assert (info.format, info.subtype, info.channels, info.samplerate) == (
        "WAV",
        "PCM_16",
        1,
        24000,
    )
    assert not list(path.parent.glob("*.part"))


def test_write_wav_refuses_another_rate(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        references.write_wav(tmp_path / "x.wav", np.zeros(10, dtype=np.float32), 16000)
