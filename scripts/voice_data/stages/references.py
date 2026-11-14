"""Reference clips per state for the target and every teacher (Design 5).

A state's references are 2-4 of its cleaned segments, each 3-12 s long, at or above the
leave-one-out median SIM on every embedder (each clip against the centroid of the others, as
`score.leave_one_out_scores` defines it), and with a verbatim transcript both ASR passes agree
on. Among those, clips whose prosody sits nearest the state's median come first, the higher SNR
breaking a tie, and each new source is taken before a second clip of a source already chosen. A
state with fewer than two such clips is reported short and gets none: a clip outside the rules is
never added to make up the number.

The transcript rule exists because a teacher conditions on the reference's text as well as its
audio (OmniVoice's `ref_text`, CosyVoice 3's zero-shot prompt), so a text the two ASRs disagree
on, or no text at all, would steer the teacher wrongly. The text is written beside the WAV as
`refs/<state>/<n>.private.txt`, the segment's `crisper` transcript as UTF-8 and nothing else.

The prosody distance is `score.select`'s, so references and take selection agree on what "near
the state's profile" means. The state's profile is the median of every member's prosody, and its
spread their standard deviation, eligible or not: a clip barred by its transcript still shows how
the state sounds.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..records import Prosody
from ..states import SPEAKING_STATES
from . import score

MIN_REFS = 2
MAX_REFS = 4
MIN_S = 3.0
MAX_S = 12.0
RATE = 24000
_FIELDS = ("f0_mean_hz", "f0_std_hz", "rate_sps", "energy_db")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A confirmed, cleaned segment as a reference candidate.

    `sim` is its leave-one-out SIM, `transcript` its verbatim (`crisper`) text, and
    `asr_disagree` whether the two ASR passes disagreed on it.
    """

    segment_id: str
    source_id: str
    state: str
    duration_s: float
    snr_db: float
    prosody: Prosody
    sim: dict[str, float]
    transcript: str
    asr_disagree: bool


@dataclass(frozen=True, slots=True)
class StateChoice:
    """One state's references, in the order chosen, and how many clips it had to choose from."""

    state: str
    chosen: tuple[Candidate, ...]
    members: int
    eligible: int

    @property
    def short(self) -> bool:
        """Whether the state has fewer than `MIN_REFS` eligible clips, and so no references."""
        return self.eligible < MIN_REFS


def eligible(candidate: Candidate, medians: Mapping[str, float]) -> bool:
    """Return whether the clip may be a reference.

    It must have a transcript both ASR passes agree on, be 3-12 s long, and sit at the
    leave-one-out median or above on every embedder.
    """
    if candidate.asr_disagree or not candidate.transcript.strip():
        return False
    if not MIN_S <= candidate.duration_s <= MAX_S:
        return False
    return all(candidate.sim.get(name, float("-inf")) >= m for name, m in medians.items())


def profile(members: Sequence[Candidate]) -> tuple[Prosody, Prosody]:
    """Return the members' prosody median and standard deviation, field by field."""
    values = np.array([[getattr(c.prosody, f) for f in _FIELDS] for c in members], dtype=float)
    centre, spread = np.median(values, axis=0), np.std(values, axis=0)
    return (
        Prosody(**{f: float(v) for f, v in zip(_FIELDS, centre, strict=True)}),
        Prosody(**{f: float(v) for f, v in zip(_FIELDS, spread, strict=True)}),
    )


def rank(pool: Sequence[Candidate], centre: Prosody, spread: Prosody) -> list[Candidate]:
    """Return `pool` nearest the profile first; equal distances go to the higher SNR."""
    by_id = {c.segment_id: c for c in pool}
    remaining = [
        score.TakeScores(take_id=c.segment_id, sim=c.sim, cer=0.0, prosody=c.prosody)
        for c in sorted(pool, key=lambda c: (-c.snr_db, c.segment_id))
    ]
    order: list[Candidate] = []
    while remaining:
        best = score.select(remaining, centre, spread)
        remaining = [t for t in remaining if t.take_id != best.take_id]
        order.append(by_id[best.take_id])
    return order


def _pick(ranked: Sequence[Candidate]) -> tuple[Candidate, ...]:
    chosen: list[Candidate] = []
    sources: set[str] = set()
    for candidate in ranked:
        if len(chosen) < MAX_REFS and candidate.source_id not in sources:
            chosen.append(candidate)
            sources.add(candidate.source_id)
    for candidate in ranked:
        if len(chosen) < MAX_REFS and candidate not in chosen:
            chosen.append(candidate)
    return tuple(chosen)


def choose(
    candidates: Iterable[Candidate],
    medians: Mapping[str, float],
    states: Sequence[str] = SPEAKING_STATES,
) -> dict[str, StateChoice]:
    """Return each state's references, in `states` order; a short state chooses none."""
    by_state: dict[str, list[Candidate]] = {state: [] for state in states}
    for candidate in candidates:
        if candidate.state in by_state:
            by_state[candidate.state].append(candidate)
    choices: dict[str, StateChoice] = {}
    for state, members in by_state.items():
        pool = [c for c in members if eligible(c, medians)]
        chosen: tuple[Candidate, ...] = ()
        if len(pool) >= MIN_REFS:
            chosen = _pick(rank(pool, *profile(members)))
        choices[state] = StateChoice(state, chosen, len(members), len(pool))
    return choices


def reference_rel(state: str, number: int) -> str:
    """Return a reference's path relative to its target folder, as `target.json` records it."""
    return f"refs/{state}/{number}.wav"


def write_transcript(path: Path, text: str) -> None:
    """Write a reference's transcript as its UTF-8 bytes exactly, atomically.

    Bytes rather than text mode, so no newline is translated and a later sha256 of the file is
    the sha256 of the transcript itself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(text.encode("utf-8"))
    os.replace(part, path)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write a reference as WAV PCM16 mono at 24 kHz, atomically.

    Raises:
        ValueError: The audio is not at 24 kHz.
    """
    import soundfile

    if sample_rate != RATE:
        raise ValueError(f"a reference is written at {RATE} Hz, not {sample_rate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    soundfile.write(
        str(part), np.asarray(audio, dtype=np.float32), RATE, format="WAV", subtype="PCM_16"
    )
    os.replace(part, path)
