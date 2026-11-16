"""The rehearsal: public stand-in episodes with ground truth (Design 12, A11).

`rehearsal build` mixes episodes from public speech into a rehearsal root's `rehearsal/`
folder: one Expresso speaker as the target, with two fixed in-show effects, LibriTTS-R speakers as
the others (one of them, the impostor, in every episode), and MUSAN music and noise. Ingest walks
`seed/raw/`, which every package writer refuses, so the lead moves the episodes there by hand (an
H6 host step); the ground truth therefore names each episode by its file's sha256, which ingest
records per source, never by path.

This module holds what the host and the job share: the file layout under `rehearsal/`, the
reading and checking of `sources.json`, the ground-truth record, the impostor's eligibility rule
and the episode formats. It is host side, so it imports the standard library and `scripts.` only;
the audio work is `mix.py`'s, in the `score` image.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath

from ..errors import InputError
from ..states import SPEAKING_STATES

SOURCES_PATH = Path(__file__).with_name("sources.json")
SCHEMA = 1
KIND = "rehearsal"
REHEARSAL_REL = "rehearsal"
EPISODES_REL = f"{REHEARSAL_REL}/episodes"
PLAN_REL = f"{REHEARSAL_REL}/build-plan.json"
BUILD_REL = f"{REHEARSAL_REL}/build.json"
TRUTH_REL = f"{REHEARSAL_REL}/truth.jsonl"
ANCHOR_REL = f"{REHEARSAL_REL}/reference/anchor.wav"
OUTPUTS_NAME = "outputs.json"
ANSWERS_REL = f"{REHEARSAL_REL}/answers"
SCORES_REL = f"{REHEARSAL_REL}/scores"
EFFECTS_PLAN_NAME = "effects-plan.json"
EFFECTS_NAME = "effects.json"
SCORE_NAME = "score.json"
SEED_ANCHOR_REL = "seed/reference/rehearsal-anchor.wav"
STEPS = ("build", "effects")

DEFAULT_EPISODES = 8
DEFAULT_MINUTES = 10.0
DEFAULT_SEED = 20261115
# A shorter episode cannot hold the three fallback and two off-centre target lines.
MIN_MINUTES = 0.5

# Episode formats, as Raj's files are (E-AC-3 5.1 MKV with a small black video stream), plus one
# AAC 5.1 MKV and one stereo MP4, so ingest's and extract's other paths run too.
EAC3_MKV = "eac3-mkv"
AAC_MKV = "aac-mkv"
STEREO_MP4 = "stereo-mp4"
FORMATS = (EAC3_MKV, AAC_MKV, STEREO_MP4)

SPEAKERS = ("target", "impostor", "other")
GEOMETRIES = ("centre", "off-centre", "fallback")
SEXES = ("female", "male")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.()+ -]*")


def episode_formats(count: int) -> list[str]:
    """Return each episode's format: E-AC-3 MKV for all but the last two, then AAC and MP4.

    With fewer than three episodes the formats come in the order E-AC-3, AAC, MP4, so a short
    test build still covers the first ones.
    """
    if count < 3:
        return list(FORMATS[:count])
    return [EAC3_MKV] * (count - 2) + [AAC_MKV, STEREO_MP4]


def episode_name(index: int, fmt: str) -> str:
    """Return episode `index`'s (0-based) file name under `rehearsal/episodes/`."""
    suffix = ".mp4" if fmt == STEREO_MP4 else ".mkv"
    return f"ep{index + 1:02d}{suffix}"


def impostor_turns(minutes: float) -> int:
    """Return the most turns the impostor takes in one episode: two per minute, at least three.

    The impostor speaks with no clip repeated within an episode, so this is also the number of
    clips it needs (`eligible`).
    """
    return max(3, math.ceil(2.0 * minutes))


def eligible(clips: int, minutes: float) -> bool:
    """Return whether a speaker with `clips` clips can be the impostor in episodes of `minutes`.

    The rule: enough clips for its turns in every episode, with no clip repeated within one
    (`impostor_turns`); clips may repeat across episodes, which are scored separately.
    """
    return clips >= impostor_turns(minutes)


# ---------------------------------------------------------------------------
# sources.json
# ---------------------------------------------------------------------------


def _need(condition: bool, what: str) -> None:
    if not condition:
        raise InputError(f"refused: sources.json {what}")


def _rel(*parts: str) -> str:
    for part in parts:
        _need(isinstance(part, str) and bool(part), "holds an empty path")
        for piece in PurePosixPath(part).parts:
            _need(
                piece not in ("..", ".") and _PART.fullmatch(piece) is not None, "holds a bad path"
            )
    return PurePosixPath(*parts).as_posix()


def _files(group: Mapping[str, object], folder: str, what: str) -> list[tuple[str, str]]:
    files = group.get("files")
    _need(isinstance(files, list) and bool(files), f"{what} lists no files")
    found = []
    for entry in files:  # type: ignore[union-attr]
        _need(isinstance(entry, list) and len(entry) == 2, f"{what} has a file that is not a pair")
        path, digest = entry
        _need(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None, f"{what} sha256")
        found.append((_rel(folder, str(group.get("dir")), path), digest))
    _need(found == sorted(found), f"{what} files are not in sorted path order")
    return found


def _group(spec: Mapping[str, object], name: str) -> Mapping[str, object]:
    group = spec.get(name)
    _need(isinstance(group, Mapping), f"has no {name} group")
    for key in ("dataset", "licence", "subset", "folder"):
        _need(isinstance(group.get(key), str) and bool(group.get(key)), f"{name} has no {key}")  # type: ignore[union-attr]
    folder = str(group["folder"])  # type: ignore[index]
    _need(folder.startswith("public/"), f"{name}'s folder is not under public/")
    return group  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Speaker:
    """One LibriTTS-R speaker: id, sex, median F0 and its files (root-relative, sha256)."""

    speaker: str
    sex: str
    f0_median_hz: float
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class Sources:
    """`sources.json`, checked: the target's files per Expresso style, the others, MUSAN."""

    target_speaker: str
    target_sex: str
    states: dict[str, str]
    target: dict[str, tuple[tuple[str, str], ...]]
    others: tuple[Speaker, ...]
    clear_f0_min_hz: float
    music: tuple[tuple[str, str], ...]
    noise: tuple[tuple[str, str], ...]
    raw: dict[str, object]

    def all_files(self) -> dict[str, str]:
        """Return every file as root-relative path to sha256."""
        found: dict[str, str] = {}
        for files in self.target.values():
            found.update(files)
        for speaker in self.others:
            found.update(speaker.files)
        found.update(self.music)
        found.update(self.noise)
        return found

    def clear_same_sex(self) -> list[Speaker]:
        """Return the others of the target's sex whose median F0 leaves no doubt about it."""
        same = [s for s in self.others if s.sex == self.target_sex]
        if self.target_sex == "female":
            return [s for s in same if s.f0_median_hz >= self.clear_f0_min_hz]
        return [s for s in same if s.f0_median_hz < self.clear_f0_min_hz]


MIN_CLEAR_SAME_SEX = 8


def parse_sources(spec: object) -> Sources:
    """Check a parsed `sources.json` and return it as `Sources`.

    Raises:
        InputError: A group or field is missing or malformed, a path repeats, the style map does
            not cover the five speaking states, or fewer than eight others are clearly of the
            target's sex.
    """
    _need(isinstance(spec, Mapping) and spec.get("schema") == SCHEMA, "is not a schema 1 file")
    target = _group(spec, "target")  # type: ignore[arg-type]
    sex = target.get("sex")
    _need(sex in SEXES, "target has no sex")
    states = target.get("states")
    _need(isinstance(states, Mapping), "target has no style-to-state map")
    _need(
        sorted(states.values()) == sorted(SPEAKING_STATES),  # type: ignore[union-attr]
        "target's style map does not cover each speaking state once",
    )
    styles = target.get("styles")
    _need(isinstance(styles, Mapping) and set(styles) == set(states), "target styles do not match")  # type: ignore[arg-type]
    target_files = {
        str(style): tuple(_files(group, str(target["folder"]), f"target {style}"))
        for style, group in styles.items()  # type: ignore[union-attr]
    }
    others = _group(spec, "others")  # type: ignore[arg-type]
    clear = others.get("clear_f0_min_hz")
    _need(isinstance(clear, int | float) and not isinstance(clear, bool), "has no clear F0 bound")
    speakers = []
    for entry in others.get("speakers") or []:  # type: ignore[union-attr]
        _need(isinstance(entry, Mapping), "has a speaker that is not an object")
        name = str(entry.get("speaker"))
        _need(entry.get("sex") in SEXES, f"speaker {name} has no sex")
        f0 = entry.get("f0_median_hz")
        _need(isinstance(f0, int | float) and not isinstance(f0, bool), f"speaker {name} F0")
        files = tuple(_files(entry, str(others["folder"]), f"speaker {name}"))
        speakers.append(Speaker(name, str(entry["sex"]), float(f0), files))  # type: ignore[arg-type]
    _need(bool(speakers), "lists no other speakers")
    _need(len({s.speaker for s in speakers}) == len(speakers), "repeats a speaker")
    musan = []
    for name in ("music", "noise"):
        group = _group(spec, name)  # type: ignore[arg-type]
        files: list[tuple[str, str]] = []
        for collection in group.get("collections") or []:  # type: ignore[union-attr]
            _need(isinstance(collection, Mapping), f"{name} has a bad collection")
            licence = collection.get("licence_file")
            _need(isinstance(licence, str) and bool(licence), f"{name} collection has no licence")
            files += _files(collection, str(group["folder"]), f"{name} collection")
        _need(bool(files), f"{name} lists no files")
        musan.append(tuple(files))
    sources = Sources(
        target_speaker=str(target.get("speaker")),
        target_sex=str(sex),
        states={str(k): str(v) for k, v in states.items()},  # type: ignore[union-attr]
        target=target_files,
        others=tuple(speakers),
        clear_f0_min_hz=float(clear),  # type: ignore[arg-type]
        music=musan[0],
        noise=musan[1],
        raw=dict(spec),  # type: ignore[arg-type]
    )
    every = [p for files in target_files.values() for p, _ in files]
    every += [p for s in speakers for p, _ in s.files]
    every += [p for p, _ in musan[0] + musan[1]]
    _need(len(set(every)) == len(every), "names one file twice")
    _need(
        len(sources.clear_same_sex()) >= MIN_CLEAR_SAME_SEX,
        f"has fewer than {MIN_CLEAR_SAME_SEX} others clearly of the target's sex",
    )
    return sources


def load_sources(path: Path = SOURCES_PATH) -> Sources:
    """Read and check `sources.json`.

    Raises:
        InputError: The file is missing, not JSON, or fails `parse_sources`.
    """
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError("refused: sources.json is missing or not JSON") from exc
    return parse_sources(spec)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TruthLine:
    """One line of `rehearsal/truth.jsonl`: one speaker's turn in one episode.

    `speaker` is `target`, `impostor` or `other`. Target lines carry their state (from the
    Expresso style) and their channel geometry: `centre`, `off-centre` (front left and right
    only) or `fallback` (dry in front left and right, a reverberant copy under a burst in the
    centre). `overlap` marks a line that overlaps another speaker's.
    """

    episode: str
    episode_sha256: str
    line: int
    speaker: str
    speaker_id: str
    start_s: float
    end_s: float
    state: str | None
    style: str | None
    geometry: str | None
    clip: str
    overlap: bool

    def to_json(self) -> dict[str, object]:
        """Return the line as its JSONL object."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, data: object) -> TruthLine:
        """Build a line from its JSONL object.

        Raises:
            InputError: Unknown or missing keys, or a speaker, state or geometry out of range.
        """
        names = [f.name for f in fields(cls)]
        if not isinstance(data, Mapping) or sorted(data) != sorted(names):
            raise InputError("refused: a ground-truth line has unknown or missing keys")
        line = cls(**{name: data[name] for name in names})
        if (
            line.speaker not in SPEAKERS
            or (line.speaker == "target") != (line.state in SPEAKING_STATES)
            or (line.speaker == "target") != (line.geometry in GEOMETRIES)
        ):
            raise InputError("refused: a ground-truth line has a bad speaker, state or geometry")
        return line


def read_truth(path: Path) -> list[TruthLine]:
    """Read `rehearsal/truth.jsonl`.

    Raises:
        InputError: The file is missing or a line is malformed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"refused: {TRUTH_REL} is missing; run rehearsal build first") from exc
    lines = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            lines.append(TruthLine.from_json(json.loads(raw)))
        except json.JSONDecodeError as exc:
            raise InputError(f"refused: {TRUTH_REL} line {number} is not JSON") from exc
    return lines


def overlap_s(start: float, end: float, lines: Sequence[TruthLine]) -> float:
    """Return how many seconds of `[start, end]` the lines cover, counting each line separately."""
    return sum(max(0.0, min(end, ln.end_s) - max(start, ln.start_s)) for ln in lines)


def majority(start: float, end: float, lines: Sequence[TruthLine]) -> TruthLine | None:
    """Return the line covering most of `[start, end]` when it covers at least half of it."""
    span = end - start
    if span <= 0 or not lines:
        return None
    best = max(lines, key=lambda ln: overlap_s(start, end, [ln]))
    return best if overlap_s(start, end, [best]) >= 0.5 * span else None
