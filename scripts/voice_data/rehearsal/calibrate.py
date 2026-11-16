"""Gate calibration on public speakers: pools, trials and the threshold rule (A22).

Runs as `/opt/venv-score/bin/python -I -m scripts.voice_data.rehearsal.calibrate --data /data
--plan rehearsal/calibration/<C>/calibration-plan.json` in the `score` image, on the CPU, with
`public/` mounted read-only. It sets the six per-embedder values `anchors.sim_min.<e>`,
`retrieval.anchor_floor.<e>` and `retrieval.margin.<e>` on public speakers that no rehearsal
episode uses, so the rehearsal (H6) tests the values instead of fitting them. It reads the plan
and the public files the plan names, nothing else: never `build-plan.json`, `build.json` or the
gates. `calibrate run` (T34) writes the plan and chooses the speakers.

The plan (`calibration-plan.json`, schema 1). Every key below is required unless marked
optional; the job refuses (exit 1) a plan that breaks any rule, naming ids and counts only::

    {
      "schema": 1,
      "kind": "calibration-plan",
      "id": "<C>",                  # [A-Za-z0-9][A-Za-z0-9_.-]{0,63}
      "alpha": 0.01,                # on the 0.0001 grid, in (0, 0.5]
      "seed": 20261116,             # int >= 0: shuffles, cluster draws, rivals, the bootstrap
      "embedders": {                # each loaded embedder's name, exactly; revision from
        "wavlm-sv": {"revision": "<sha>"},   # families.json, copied to calibration.json
        "ecapa": {"revision": "<sha>"}
      },
      "trials": {                   # exactly these keys
        "piece_s": [3.0, 1.5, 6.0], # turn-piece lengths, cycled
        "max_pieces": 100,          # >= 40
        "model_pieces": 30,         # pieces 1-30 build the target centroid; < 40
        "rival_pieces": 20,         # pieces 1-20 build the rival centroid; <= model_pieces
        "cluster_n": [10, 20, 30],  # anchor-floor cluster sizes that set values; each <= 40
        "reported_n": [5],          # cluster sizes reported, never used; each <= 40
        "draws": 10,                # clusters drawn per pair and size
        "anchor_k": [1, 2, 4],      # anchor subsets holding the solo clip; each 1-5
        "bootstrap": 1000,          # speaker-level resamples
        "anchor_sets": 10           # anchor sets drawn per speaker, draw 1 the pool's; >= 1
      },
      "matched": {                  # exactly the three gates; each a list of unique target
        "anchors.sim_min": ["sets"],          # forms of that gate, whose miss the rule
        "retrieval.anchor_floor": ["centroid/n10", "centroid/n20", "centroid/n30"],
        "retrieval.margin": ["rival"]         # minimises
      },
      "datasets": [                # 1 or 2, names unique; never pooled
        {
          "name": "librittsr",
          "licence": "<text>", "subset": "<text>",
          "speakers": [             # >= 1; ids unique
            {"speaker": "<id>",     # [A-Za-z0-9_-]+
             "sex": "female",       # female or male
             "files": [["public/<path>.wav", "<sha256>"], ...]}   # >= 1, sorted by path
          ],
          "excluded": [             # optional: speakers left out by `calibrate run`
            {"speaker": "<id>", "sex": "male", "reason": "build"}
          ]
        },
        {
          "name": "ravdess",
          "licence": "<text>", "subset": "<text>",
          "archive": ["public/ravdess/Audio_Speech_Actors_01-24.zip", "<sha256>"],
          "excluded": []            # optional, as above
        }
      ]
    }

Every path is root-relative under `public/`, with no `.` or `..` part, and no path repeats.
Every file's sha256, and the archive's, is checked before any of them is read. The RAVDESS
archive is read in place with `zipfile`, never extracted. Its entries must all fit
`[Actor_NN/]03-01-EE-II-SS-RR-NN.wav` (audio-only speech; emotion 01-08, intensity, statement
and repetition 01-02, actor 01-24, the folder's actor equal to the name's); any other name is
refused rather than guessed, because the naming is from memory and unverified. The actor's sex
is male when odd; emotions map neutral, calm, happy, sad and surprised to neutral, calm, awake,
sleepy and excited, and angry, fearful and disgust to `other`.

Pools (Design, Pool per speaker). A speaker's files are read in path order, trimmed
(`mix.trim`), resampled to 16 kHz and shuffled with the seed. The solo clip joins clips with
0.25 s gaps until 12-20 s as `mix._anchor` does (RAVDESS: neutral clips first, then calm). Four
timestamp anchors of 3-30 s follow: for LibriTTS-R the next four files of that length, for
RAVDESS one per register (calm, awake, sleepy, excited) joined as the solo clip until 3 s. The
remaining files give at most 100 turn pieces: a LibriTTS-R file is cut back to back with the
lengths cycling 3, 1.5 and 6 s, the cycle running on from one file to the next, and a
remainder of 0.5 s or more is kept; a RAVDESS clip is one piece. A speaker with fewer than 40
pieces, or without the solo clip or the four anchors, is ineligible, and the job refuses when a
dataset has fewer than three eligible speakers of either sex.

Anchor sets (Design, Anchor-set draws). The pool's solo clip and four anchors are draw 1. Each
eligible speaker's later draws, up to `anchor_sets`, reshuffle the speaker's clips in path order
with a stream seeded by the seed, the dataset, the speaker and the draw number, and build the
solo clip and four anchors from them as the pool does; a later draw may reuse a file draw 1 cut
into pieces. A draw that cannot form a set is skipped and counted. Only the anchor check's target
trials read the later draws: impostor trials, the floor and the margin read draw 1.

Trials pair two speakers of one sex within one dataset, in both orders, and score them with
extract's own `retrieve.centroids` and `retrieve.sims` (`score.cosine` underneath). Each target
trial records its owner's speaker index, and a margin target trial its rivals' too.

The threshold rule (A22), per gate, with e1 WavLM-SV and e2 ECAPA in `EMBEDDERS` order:

1. A pair (v1, v2) accepts a trial when each embedder's score is at or above its value less
   `EPS`, extract's rule. A form's FAR is the share of its impostor trials accepted; its miss the
   share of its target trials not accepted. A weight counts a trial that many times in both.
2. v1 runs over the 0.0001 grid from the largest value accepting every trial of the gate on e1
   (target and impostor, every form, both datasets) to the smallest accepting none. v2(v1) is the
   least grid value, from the same lower end on e2, at which every impostor form of each dataset
   holds FAR at or below alpha; the n-of-5 forms stay reported. At v1's lower end the rule is
   ECAPA's alone, and where v2(v1) is e2's lower end it is WavLM-SV's alone.
3. A larger v2 also holds alpha but misses at least as many target trials, so the candidates
   are (v1, v2(v1)), and v2 is never raised above v2(v1).
4. The pair is the candidate with the smallest largest miss over the plan's `matched` forms of
   both datasets, then the smallest mean of those misses, then the smallest largest FAR over the
   impostor forms, then the larger v1.
5. v5's pair, alpha split equally between the embedders, holds alpha, so the candidate at its v1
   has a v2 no higher than its own and the chosen pair never misses more in sample. It is
   reported beside as `equal_beta`.

v2(v1) comes from one sweep per form: as v1 falls, the trials it admits are a growing prefix in
e1's order, and a heap gives each prefix's bound on v2. The bootstrap re-solves the rule on
`bootstrap` speaker resamples: an impostor trial counts the copies of A times those of B, and a
target trial its owner's copies.

Output: `calibration.json` and `outputs.json` beside the plan. Printed lines and messages name
ids and counts, never a file.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import itertools
import json
import math
import os
import re
import sys
import zipfile
import zlib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PurePosixPath

import numpy as np

from ..stages import retrieve
from ..stages.retrieve import EPS, Embedding
from . import OUTPUTS_NAME, SCHEMA, SEXES, mix
from .mix import (
    ANCHOR_GAP_S,
    ANCHOR_MAX_S,
    ANCHOR_MIN_S,
    StageError,
    read_audio,
    resample,
    sha256_file,
    trim,
)

RATE = 16000
"""Both embedders take 16 kHz (`bindings/embedders.py`), so clips are resampled once, here."""
GRID = 10_000
BETA_MAX = 5_000
MIN_EMBED_S = 0.5
"""extract's shortest embedded turn (`stages/extract.py` `MIN_EMBED_S`)."""
ANCHOR_ROWS = 4
ROW_MIN_S = 3.0
ROW_MAX_S = 30.0
"""`anchors.min_s` and `anchors.max_s`; the job reads no gates, so they are restated here."""
MIN_PIECES = 40
MIN_ELIGIBLE = 3
TARGET_RIVALS = 3

SIM_MIN = "anchors.sim_min"
FLOOR = "retrieval.anchor_floor"
MARGIN = "retrieval.margin"
GATES = (SIM_MIN, FLOOR, MARGIN)
PLAN_KIND = "calibration-plan"
CALIBRATION_NAME = "calibration.json"
LIBRITTSR = "librittsr"
RAVDESS = "ravdess"
DATASETS = (LIBRITTSR, RAVDESS)

NEUTRAL = "neutral"
CALM = "calm"
SOLO = "solo"
CENTROID = "centroid"
RIVAL = "rival"
MIXED = "mixed"
ALL = "all"
OTHERS = "others"
SETS = "sets"
BOTH = "both"
RAVDESS_REGISTERS = {
    "01": NEUTRAL,
    "02": CALM,
    "03": "awake",
    "04": "sleepy",
    "05": "other",
    "06": "other",
    "07": "other",
    "08": "excited",
}
RAVDESS_ANCHORS = (CALM, "awake", "sleepy", "excited")
RAVDESS_ACTORS = 24
_RAVDESS_NAME = re.compile(
    r"(?:Actor_(\d{2})/)?03-01-(0[1-8])-(0[12])-(0[12])-(0[12])-(\d{2})\.wav"
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_SPEAKER = re.compile(r"[A-Za-z0-9_-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# Random streams, each seeded from the plan's seed and its own keys, so one speaker's shuffle or
# one pair's draws do not move when another speaker joins the plan.
_SHUFFLE, _DRAWS, _TARGET, _RIVALS, _BOOT, _SETS = range(6)


def _key(text: str) -> int:
    return zlib.crc32(text.encode("utf-8"))


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng([seed, *keys])


def load_embedders() -> list:
    """Return both embedders on the CPU, as `rehearsal build` loads them; tests replace this."""
    return mix.load_embedders()


def read_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Return an in-memory WAV as float64 mono and its rate."""
    import soundfile

    audio, rate = soundfile.read(io.BytesIO(data), dtype="float64", always_2d=True)
    return audio.mean(axis=1), int(rate)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrialParams:
    """The trial parameters; the defaults are the Design's."""

    piece_s: tuple[float, ...] = (3.0, 1.5, 6.0)
    max_pieces: int = 100
    model_pieces: int = 30
    rival_pieces: int = 20
    cluster_n: tuple[int, ...] = (10, 20, 30)
    reported_n: tuple[int, ...] = (5,)
    draws: int = 10
    anchor_k: tuple[int, ...] = (1, 2, 4)
    bootstrap: int = 1000
    anchor_sets: int = 10


@dataclass(frozen=True, slots=True)
class PlanSpeaker:
    """A LibriTTS-R speaker of the plan and its files, root-relative path and sha256."""

    speaker: str
    sex: str
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """One dataset of the plan: listed files (LibriTTS-R) or one archive (RAVDESS)."""

    name: str
    licence: str
    subset: str
    speakers: tuple[PlanSpeaker, ...]
    archive: tuple[str, str] | None
    excluded: tuple[tuple[str, str, str], ...]

    def files(self) -> dict[str, str]:
        """Return every file this dataset reads, root-relative path to sha256."""
        if self.archive is not None:
            return {self.archive[0]: self.archive[1]}
        return {rel: sha for s in self.speakers for rel, sha in s.files}


@dataclass(frozen=True, slots=True)
class Plan:
    """A checked calibration plan."""

    id: str
    alpha: float
    seed: int
    embedders: dict[str, str]
    trials: TrialParams
    matched: dict[str, tuple[str, ...]]
    datasets: tuple[DatasetPlan, ...]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _safe_rel(rel: object) -> bool:
    if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel or rel.startswith("/"):
        return False
    parts = rel.split("/")
    return parts[0] == "public" and len(parts) > 1 and all(p not in ("", ".", "..") for p in parts)


def _alpha_units(alpha: float) -> int:
    units = round(alpha * GRID)
    if abs(units - alpha * GRID) > 1e-6 or not 0 < units <= BETA_MAX:
        raise StageError("alpha is not on the 0.0001 grid in (0, 0.5]")
    return units


def _pair(value: object) -> tuple[str, str] | None:
    if (
        isinstance(value, list)
        and len(value) == 2
        and _safe_rel(value[0])
        and isinstance(value[1], str)
        and _SHA256.fullmatch(value[1])
    ):
        return value[0], value[1]
    return None


def _trials(raw: object, problems: list[str]) -> TrialParams:
    names = [f.name for f in fields(TrialParams)]
    if not isinstance(raw, dict):
        problems.append("trials is not an object")
        return TrialParams()
    missing = [n for n in names if n not in raw]
    extra = sorted(set(raw) - set(names))
    if missing or extra:
        problems.append(
            f"trials lacks {len(missing)} keys ({', '.join(missing) or 'none'}) and has "
            f"{len(extra)} unknown ones"
        )
        return TrialParams()
    ok = (
        isinstance(raw["piece_s"], list)
        and bool(raw["piece_s"])
        and all(_is_number(v) and v > 0 for v in raw["piece_s"])
    )
    for name in ("max_pieces", "model_pieces", "rival_pieces", "draws", "bootstrap", "anchor_sets"):
        ok = ok and _is_int(raw[name]) and raw[name] > 0
    for name in ("cluster_n", "reported_n", "anchor_k"):
        ok = ok and isinstance(raw[name], list) and all(_is_int(v) and v > 0 for v in raw[name])
    if not ok:
        problems.append("trials holds a value of the wrong type or sign")
        return TrialParams()
    params = TrialParams(
        piece_s=tuple(float(v) for v in raw["piece_s"]),
        max_pieces=raw["max_pieces"],
        model_pieces=raw["model_pieces"],
        rival_pieces=raw["rival_pieces"],
        cluster_n=tuple(raw["cluster_n"]),
        reported_n=tuple(raw["reported_n"]),
        draws=raw["draws"],
        anchor_k=tuple(raw["anchor_k"]),
        bootstrap=raw["bootstrap"],
        anchor_sets=raw["anchor_sets"],
    )
    sizes = params.cluster_n + params.reported_n
    if not (
        params.cluster_n
        and params.anchor_k
        and params.rival_pieces <= params.model_pieces < MIN_PIECES <= params.max_pieces
        and max(sizes) <= MIN_PIECES
        and len(set(sizes)) == len(sizes)
        and all(k <= ANCHOR_ROWS + 1 for k in params.anchor_k)
        and len(set(params.anchor_k)) == len(params.anchor_k)
    ):
        problems.append(
            "trials breaks a bound: rival <= model < 40 <= max pieces, cluster sizes unique "
            "and at most 40, anchor k unique and 1-5"
        )
    return params


def target_forms(params: TrialParams) -> dict[str, frozenset[str]]:
    """Return each gate's target forms, the names `dataset_trials` gives them."""
    sizes = (*params.cluster_n, *params.reported_n)
    return {
        SIM_MIN: frozenset({OTHERS, SETS}),
        FLOOR: frozenset(f"{anchor}/n{n}" for anchor in (SOLO, CENTROID) for n in sizes),
        MARGIN: frozenset({RIVAL}),
    }


def _matched(raw: object, params: TrialParams, problems: list[str]) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict):
        problems.append("matched is missing or not an object")
        return {}
    if set(raw) != set(GATES):
        problems.append(f"matched does not name exactly the gates {', '.join(GATES)}")
        return {}
    known = target_forms(params)
    found: dict[str, tuple[str, ...]] = {}
    for gate in GATES:
        forms = raw[gate]
        if (
            not isinstance(forms, list)
            or not forms
            or not all(isinstance(f, str) for f in forms)
            or len(set(forms)) != len(forms)
        ):
            problems.append(f"matched {gate} is not a list of unique form names")
            continue
        unknown = [f for f in forms if f not in known[gate]]
        if unknown:
            problems.append(
                f"matched {gate} names {', '.join(unknown)}, not a target form of the gate"
            )
        found[gate] = tuple(forms)
    return found


def _excluded(raw: object, name: str, problems: list[str]) -> tuple[tuple[str, str, str], ...]:
    if raw is None:
        return ()
    rows = []
    if isinstance(raw, list):
        for row in raw:
            if (
                isinstance(row, dict)
                and isinstance(row.get("speaker"), str)
                and _SPEAKER.fullmatch(row["speaker"])
                and row.get("sex") in SEXES
                and isinstance(row.get("reason"), str)
                and row["reason"]
            ):
                rows.append((row["speaker"], row["sex"], row["reason"]))
    if not isinstance(raw, list) or len(rows) != len(raw):
        problems.append(f"{name}'s excluded list holds a malformed row")
    return tuple(rows)


def _speakers(raw: object, name: str, problems: list[str]) -> tuple[PlanSpeaker, ...]:
    if not isinstance(raw, list) or not raw:
        problems.append(f"{name} lists no speakers")
        return ()
    found: list[PlanSpeaker] = []
    for index, entry in enumerate(raw):
        speaker = entry.get("speaker") if isinstance(entry, dict) else None
        if not isinstance(speaker, str) or not _SPEAKER.fullmatch(speaker):
            problems.append(f"{name} speaker #{index + 1} has no valid id")
            continue
        if entry.get("sex") not in SEXES:
            problems.append(f"{name} speaker {speaker} has no sex of {'/'.join(SEXES)}")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            problems.append(f"{name} speaker {speaker} lists no files")
            continue
        pairs = [_pair(f) for f in files]
        bad = sum(p is None for p in pairs)
        if bad:
            problems.append(f"{name} speaker {speaker} has {bad} bad paths or sha256s")
            continue
        good = [p for p in pairs if p is not None]
        if good != sorted(good):
            problems.append(f"{name} speaker {speaker}'s files are not in path order")
        found.append(PlanSpeaker(speaker, str(entry.get("sex")), tuple(good)))
    ids = [s.speaker for s in found]
    if len(set(ids)) != len(ids):
        problems.append(f"{name} names {len(ids) - len(set(ids))} speakers twice")
    return tuple(found)


def _dataset(raw: object, index: int, problems: list[str]) -> DatasetPlan | None:
    if not isinstance(raw, dict) or raw.get("name") not in DATASETS:
        problems.append(f"dataset #{index + 1} is not one of {', '.join(DATASETS)}")
        return None
    name = raw["name"]
    for key in ("licence", "subset"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            problems.append(f"{name} has no {key}")
    excluded = _excluded(raw.get("excluded"), name, problems)
    speakers: tuple[PlanSpeaker, ...] = ()
    archive = None
    if name == LIBRITTSR:
        speakers = _speakers(raw.get("speakers"), name, problems)
        overlap = {s.speaker for s in speakers} & {row[0] for row in excluded}
        if overlap:
            problems.append(f"{name} lists {len(overlap)} speakers as both used and excluded")
    else:
        archive = _pair(raw.get("archive"))
        if archive is None:
            problems.append(f"{name} has no valid archive path and sha256")
    return DatasetPlan(
        name=name,
        licence=str(raw.get("licence")),
        subset=str(raw.get("subset")),
        speakers=speakers,
        archive=archive,
        excluded=excluded,
    )


def parse_plan(raw: object) -> Plan:
    """Check a parsed plan and return it.

    Raises:
        StageError: Naming every problem found, by id and count; never a file.
    """
    if not isinstance(raw, dict):
        raise StageError("the plan is refused: it is not an object")
    problems: list[str] = []
    if raw.get("schema") != SCHEMA:
        problems.append(f"schema is not {SCHEMA}")
    if raw.get("kind") != PLAN_KIND:
        problems.append(f"kind is not {PLAN_KIND}")
    plan_id = raw.get("id")
    if not isinstance(plan_id, str) or not _ID.fullmatch(plan_id):
        problems.append("id is missing or malformed")
    alpha = raw.get("alpha")
    try:
        if not _is_number(alpha):
            raise StageError("alpha")
        _alpha_units(float(alpha))  # type: ignore[arg-type]
    except StageError:
        problems.append("alpha is not on the 0.0001 grid in (0, 0.5]")
    seed = raw.get("seed")
    if not _is_int(seed) or seed < 0:  # type: ignore[operator]
        problems.append("seed is not an integer of 0 or more")
    embedders: dict[str, str] = {}
    raw_embedders = raw.get("embedders")
    if isinstance(raw_embedders, dict):
        for name, entry in raw_embedders.items():
            revision = entry.get("revision") if isinstance(entry, dict) else None
            if isinstance(revision, str) and revision:
                embedders[name] = revision
            else:
                problems.append(f"embedder {name} has no revision")
    if not embedders:
        problems.append("the plan names no embedder")
    trials = _trials(raw.get("trials"), problems)
    matched = _matched(raw.get("matched"), trials, problems)
    datasets: list[DatasetPlan] = []
    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not 1 <= len(raw_datasets) <= len(DATASETS):
        problems.append("datasets is not a list of 1 or 2")
    else:
        for index, entry in enumerate(raw_datasets):
            found = _dataset(entry, index, problems)
            if found is not None:
                datasets.append(found)
        names = [d.name for d in datasets]
        if len(set(names)) != len(names):
            problems.append("a dataset is named twice")
        paths = [rel for d in datasets for rel in d.files()]
        if len(set(paths)) != len(paths):
            problems.append(f"{len(paths) - len(set(paths))} paths repeat")
    if problems:
        raise StageError("the plan is refused: " + "; ".join(problems))
    return Plan(
        id=str(plan_id),
        alpha=float(alpha),  # type: ignore[arg-type]
        seed=int(seed),  # type: ignore[arg-type]
        embedders=embedders,
        trials=trials,
        matched=matched,
        datasets=tuple(datasets),
    )


def read_plan(path: Path) -> Plan:
    """Read and check the plan.

    Raises:
        StageError: The plan is missing, not JSON, or breaks a rule of the module docstring.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the plan is missing or not JSON") from exc
    return parse_plan(raw)


def verify(data: Path, plan: Plan) -> int:
    """Check every file's sha256, and the archive's, before any is read; return the count.

    Raises:
        StageError: A file is missing or its sha256 differs; the message names the dataset and
            speaker, never the file.
    """
    count = 0
    for dataset in plan.datasets:
        owners = [(s.speaker, s.files) for s in dataset.speakers]
        if dataset.archive is not None:
            owners = [("archive", (dataset.archive,))]
        for owner, files in owners:
            what = f"{dataset.name} {'the archive' if owner == 'archive' else 'speaker ' + owner}"
            for rel, digest in files:
                path = data / rel
                if not path.is_file():
                    raise StageError(f"{what}: a file the plan lists is missing; the job stops")
                if sha256_file(path) != digest:
                    raise StageError(
                        f"{what}: a file's sha256 differs from the plan; the job stops"
                    )
                count += 1
    return count


# ---------------------------------------------------------------------------
# RAVDESS names and the archive
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RavdessName:
    """What a RAVDESS speech file's name says: the actor, their sex and the register."""

    actor: str
    sex: str
    register: str


def parse_ravdess(name: str) -> RavdessName:
    """Parse one archive entry's name.

    Raises:
        StageError: The name does not fit the pattern of the module docstring.
    """
    match = _RAVDESS_NAME.fullmatch(name)
    if match is not None:
        folder, emotion, _, _, _, actor = match.groups()
        number = int(actor)
        if 1 <= number <= RAVDESS_ACTORS and folder in (None, actor):
            sex = "male" if number % 2 else "female"
            return RavdessName(actor=actor, sex=sex, register=RAVDESS_REGISTERS[emotion])
    raise StageError(
        "a RAVDESS entry is off the pattern 03-01-<emotion>-<intensity>-<statement>-"
        "<repetition>-<actor>.wav; the job refuses rather than guess"
    )


def _entries(names: Sequence[str]) -> dict[str, list[tuple[str, RavdessName]]]:
    files = sorted(n for n in names if not n.endswith("/"))
    off = 0
    grouped: dict[str, list[tuple[str, RavdessName]]] = {}
    for name in files:
        try:
            parsed = parse_ravdess(name)
        except StageError:
            off += 1
            continue
        grouped.setdefault(parsed.actor, []).append((name, parsed))
    if off:
        raise StageError(
            f"the RAVDESS archive holds {off} of {len(files)} entries off the pattern "
            "03-01-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav; the job refuses "
            "rather than guess"
        )
    return dict(sorted(grouped.items()))


def ravdess_actors(path: Path) -> dict[str, str]:
    """Return each actor in the archive and their sex, after checking every entry's name.

    Raises:
        StageError: An entry is off the pattern.
    """
    with zipfile.ZipFile(path) as archive:
        entries = _entries(archive.namelist())
    return {actor: items[0][1].sex for actor, items in entries.items()}


def read_ravdess(
    path: Path, rate: int, actors: Collection[str] | None = None
) -> dict[str, list[Clip]]:
    """Read actors' clips from the archive in place, in entry path order, trimmed and resampled.

    Every entry's name is checked before any audio is read.

    Raises:
        StageError: An entry is off the pattern.
    """
    found: dict[str, list[Clip]] = {}
    with zipfile.ZipFile(path) as archive:
        entries = _entries(archive.namelist())
        for actor, items in entries.items():
            if actors is not None and actor not in actors:
                continue
            clips = []
            for name, parsed in items:
                audio, native = read_audio_bytes(archive.read(name))
                clips.append(Clip(resample(trim(audio, native), native, rate), parsed.register))
            found[actor] = clips
    return found


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Clip:
    """One trimmed clip and its register."""

    audio: np.ndarray
    register: str


@dataclass(frozen=True, slots=True)
class Pool:
    """One speaker's solo clip, four timestamp anchors and turn pieces, as audio."""

    solo: np.ndarray
    solo_registers: tuple[str, ...]
    anchors: tuple[Clip, ...]
    pieces: tuple[Clip, ...]


def join(
    clips: Sequence[Clip], min_s: float, max_s: float, rate: int
) -> tuple[list[int], np.ndarray] | None:
    """Join clips in order with 0.25 s gaps until `min_s`, as `mix._anchor` does.

    A clip that would carry the length past `max_s` is skipped; the length counts each gap
    taken, the last one included, and the joined audio drops that last gap.

    Returns:
        The indices taken and the joined audio, or `None` when the clips cannot reach `min_s`.
    """
    chosen: list[int] = []
    parts: list[np.ndarray] = []
    seconds = 0.0
    gap = np.zeros(int(ANCHOR_GAP_S * rate))
    for index, clip in enumerate(clips):
        length = len(clip.audio) / rate
        if seconds + length > max_s:
            continue
        chosen.append(index)
        parts += [clip.audio, gap]
        seconds += length + ANCHOR_GAP_S
        if seconds >= min_s:
            return chosen, np.concatenate(parts[:-1])
    return None


def cut(
    audio: np.ndarray, rate: int, lengths: Sequence[float], start: int = 0
) -> tuple[list[np.ndarray], int]:
    """Cut a file back to back into pieces whose lengths cycle through `lengths`.

    The cycle starts at position `start`, and a remainder shorter than the next length is a
    piece when it holds 0.5 s or more. A kept piece moves the cycle on.

    Returns:
        The pieces, and the cycle position the next file starts at.
    """
    pieces: list[np.ndarray] = []
    position = 0
    index = start
    shortest = int(round(MIN_EMBED_S * rate))
    while position < len(audio):
        size = int(round(lengths[index % len(lengths)] * rate))
        piece = audio[position : position + size]
        if len(piece) < size:
            if len(piece) >= shortest:
                pieces.append(piece)
                index += 1
            break
        pieces.append(piece)
        index += 1
        position += size
    return pieces, index


def _short(count: int) -> str:
    return f"{count} pieces, under {MIN_PIECES}"


@dataclass(frozen=True, slots=True)
class AnchorAudio:
    """One anchor set in R2's shape, as audio: the solo clip and four timestamp anchors."""

    solo: np.ndarray
    solo_registers: tuple[str, ...]
    rows: tuple[Clip, ...]


def librittsr_set(items: Sequence[Clip], rate: int) -> tuple[AnchorAudio, set[int]] | str:
    """Build a LibriTTS-R anchor set from shuffled clips.

    The solo clip joins clips to 12-20 s, and the rows are the next four files of 3-30 s.

    Returns:
        The set and the indices of the clips it took, or the reason no set forms.
    """
    solo = join(items, ANCHOR_MIN_S, ANCHOR_MAX_S, rate)
    if solo is None:
        return "no 12-20 s solo clip"
    used = set(solo[0])
    rows = [
        i
        for i in range(len(items))
        if i not in used and ROW_MIN_S <= len(items[i].audio) / rate <= ROW_MAX_S
    ][:ANCHOR_ROWS]
    if len(rows) < ANCHOR_ROWS:
        return f"{len(rows)} anchors of 3-30 s, under {ANCHOR_ROWS}"
    found = AnchorAudio(
        solo=solo[1],
        solo_registers=(NEUTRAL,) * len(solo[0]),
        rows=tuple(items[i] for i in rows),
    )
    return found, used | set(rows)


def ravdess_set(clips: Sequence[Clip], rate: int) -> tuple[AnchorAudio, set[int]] | str:
    """Build a RAVDESS anchor set from shuffled clips.

    The solo clip joins neutral and then calm clips to 12-20 s; each row joins the clips of one
    register of `RAVDESS_ANCHORS` that the solo clip left, to 3-30 s.

    Returns:
        The set and the indices of the clips it took, or the reason no set forms.
    """
    order = [i for i, c in enumerate(clips) if c.register == NEUTRAL]
    order += [i for i, c in enumerate(clips) if c.register == CALM]
    solo = join([clips[i] for i in order], ANCHOR_MIN_S, ANCHOR_MAX_S, rate)
    if solo is None:
        return "no 12-20 s solo clip of neutral and calm clips"
    used = {order[j] for j in solo[0]}
    rows = []
    for register in RAVDESS_ANCHORS:
        free = [i for i, c in enumerate(clips) if c.register == register and i not in used]
        got = join([clips[i] for i in free], ROW_MIN_S, ROW_MAX_S, rate)
        if got is None:
            return f"no 3-30 s {register} anchor"
        used |= {free[j] for j in got[0]}
        rows.append(Clip(got[1], register))
    found = AnchorAudio(
        solo=solo[1],
        solo_registers=tuple(clips[order[j]].register for j in solo[0]),
        rows=tuple(rows),
    )
    return found, used


def anchor_set(
    dataset: str, clips: Sequence[Clip], rate: int
) -> tuple[AnchorAudio, set[int]] | str:
    """Build one anchor set from shuffled clips, the way `dataset`'s pool builds draw 1."""
    return ravdess_set(clips, rate) if dataset == RAVDESS else librittsr_set(clips, rate)


def librittsr_pool(clips: Sequence[np.ndarray], rate: int, params: TrialParams) -> Pool | str:
    """Build a LibriTTS-R speaker's pool from their shuffled clips.

    Returns:
        The pool, or the reason the speaker is ineligible.
    """
    items = [Clip(np.asarray(c), NEUTRAL) for c in clips]
    got = librittsr_set(items, rate)
    if isinstance(got, str):
        return got
    anchors, used = got
    pieces: list[Clip] = []
    position = 0
    for index in range(len(items)):
        if index in used:
            continue
        cuts, position = cut(items[index].audio, rate, params.piece_s, position)
        pieces += [Clip(p, NEUTRAL) for p in cuts]
        if len(pieces) >= params.max_pieces:
            break
    pieces = pieces[: params.max_pieces]
    if len(pieces) < MIN_PIECES:
        return _short(len(pieces))
    return Pool(
        solo=anchors.solo,
        solo_registers=anchors.solo_registers,
        anchors=anchors.rows,
        pieces=tuple(pieces),
    )


def ravdess_pool(clips: Sequence[Clip], rate: int, params: TrialParams) -> Pool | str:
    """Build a RAVDESS actor's pool from their shuffled clips.

    Returns:
        The pool, or the reason the actor is ineligible.
    """
    got = ravdess_set(clips, rate)
    if isinstance(got, str):
        return got
    anchors, used = got
    shortest = int(round(MIN_EMBED_S * rate))
    pieces = [c for i, c in enumerate(clips) if i not in used and len(c.audio) >= shortest]
    pieces = pieces[: params.max_pieces]
    if len(pieces) < MIN_PIECES:
        return _short(len(pieces))
    return Pool(
        solo=anchors.solo,
        solo_registers=anchors.solo_registers,
        anchors=anchors.rows,
        pieces=tuple(pieces),
    )


def anchor_draws(
    clips: Sequence[Clip], dataset: str, speaker: str, seed: int, count: int, rate: int
) -> tuple[list[AnchorAudio], int]:
    """Draw a speaker's anchor sets 2 to `count` (Design, Anchor-set draws).

    Each draw reshuffles `clips`, the speaker's clips in path order, with its own stream, so a
    draw does not move when `count` or another speaker changes. Draw 1 is the pool's and is not
    drawn here.

    Returns:
        The sets that formed, in draw order, and how many draws could not form one.
    """
    found: list[AnchorAudio] = []
    skipped = 0
    for draw in range(2, count + 1):
        order = _rng(seed, _SETS, _key(dataset), _key(speaker), draw).permutation(len(clips))
        got = anchor_set(dataset, [clips[int(i)] for i in order], rate)
        if isinstance(got, str):
            skipped += 1
        else:
            found.append(got[0])
    return found, skipped


def require_speakers(dataset: str, eligible: Mapping[str, int]) -> None:
    """Refuse a dataset with fewer than three eligible speakers of either sex.

    Raises:
        StageError: Naming the dataset, the sex and the count.
    """
    short = [
        f"{sex} has {eligible.get(sex, 0)} eligible speakers"
        for sex in SEXES
        if eligible.get(sex, 0) < MIN_ELIGIBLE
    ]
    if short:
        raise StageError(f"{dataset}: {'; '.join(short)}, under {MIN_ELIGIBLE}; the job stops")


# ---------------------------------------------------------------------------
# Embedded speakers and trials
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AnchorSet:
    """One embedded anchor set: the solo clip first among five anchors, each one's register."""

    anchors: tuple[Embedding, ...]
    registers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Speaker:
    """An eligible speaker's embeddings: the solo clip first among the five anchors.

    `anchors` is draw 1, which every impostor trial, the floor and the margin read; `draws`
    holds the later anchor sets that formed, and `skipped` counts the later draws that did not.
    """

    dataset: str
    speaker: str
    sex: str
    anchors: tuple[Embedding, ...]
    anchor_registers: tuple[str, ...]
    pieces: tuple[Embedding, ...]
    registers: tuple[str, ...]
    draws: tuple[AnchorSet, ...] = ()
    skipped: int = 0

    def anchor_sets(self) -> tuple[AnchorSet, ...]:
        """Return every drawn anchor set, draw 1 first."""
        return (AnchorSet(self.anchors, self.anchor_registers), *self.draws)


def embed(embedders: Sequence[object], audio: np.ndarray, rate: int) -> Embedding:
    """Return one clip's embedding on each embedder."""
    return {
        e.name: np.asarray(e.embed(audio, rate), dtype=np.float64)  # type: ignore[attr-defined]
        for e in embedders
    }


def embed_set(embedders: Sequence[object], found: AnchorAudio, rate: int) -> AnchorSet:
    """Embed one anchor set: the solo clip, then its four rows."""
    return AnchorSet(
        anchors=tuple(
            embed(embedders, a, rate) for a in [found.solo, *(c.audio for c in found.rows)]
        ),
        registers=(SOLO, *(c.register for c in found.rows)),
    )


def embed_pool(
    embedders: Sequence[object],
    pool: Pool,
    dataset: str,
    speaker: str,
    sex: str,
    rate: int,
    draws: Sequence[AnchorAudio] = (),
    skipped: int = 0,
) -> Speaker:
    """Embed a pool once, the solo clip, the four anchors and every piece, and the later draws."""
    first = embed_set(embedders, AnchorAudio(pool.solo, pool.solo_registers, pool.anchors), rate)
    return Speaker(
        dataset=dataset,
        speaker=speaker,
        sex=sex,
        anchors=first.anchors,
        anchor_registers=first.registers,
        pieces=tuple(embed(embedders, c.audio, rate) for c in pool.pieces),
        registers=tuple(c.register for c in pool.pieces),
        draws=tuple(embed_set(embedders, d, rate) for d in draws),
        skipped=skipped,
    )


@dataclass(frozen=True, slots=True)
class FormTrials:
    """One form's impostor trials in one dataset: scores per embedder and each trial's pair."""

    scores: np.ndarray
    a: np.ndarray
    b: np.ndarray
    pairs: int


@dataclass(frozen=True, slots=True)
class TargetTrials:
    """One form's target trials in one dataset, with each trial's register and speakers.

    `owners` holds each trial's speaker index (into `DatasetTrials.sexes`), and `rivals` the
    indices of the speakers whose rival centroids a margin trial was measured against (empty for
    other forms), so a resample or a left-out speaker can find every trial a speaker is part of.
    """

    scores: np.ndarray
    registers: tuple[str, ...]
    owners: np.ndarray
    rivals: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class DatasetTrials:
    """Every trial of one dataset; `a` and `b` index `sexes`, one entry per speaker."""

    name: str
    sexes: tuple[str, ...]
    impostor: dict[str, dict[str, FormTrials]]
    reported: dict[str, dict[str, FormTrials]]
    target: dict[str, dict[str, TargetTrials]]


def _row(values: Mapping[str, float], names: Sequence[str]) -> list[float]:
    return [values[n] for n in names]


def _piece_scores(
    pieces: Sequence[Embedding], centre: Mapping[str, np.ndarray], names: Sequence[str]
) -> np.ndarray:
    return np.array([_row(retrieve.sims(p, centre), names) for p in pieces])


def _register(registers: Sequence[str]) -> str:
    return registers[0] if len(set(registers)) == 1 else MIXED


def dataset_trials(
    name: str,
    speakers: Sequence[Speaker],
    params: TrialParams,
    seed: int,
    names: Sequence[str],
) -> DatasetTrials:
    """Build every impostor and target trial of one dataset (Design, Trials).

    Impostor trials pair A, standing for the target, with another speaker B of A's sex:

    - `retrieval.anchor_floor`: the median SIM of a cluster of n of B's pieces to A's solo clip
      (`solo/n<n>`) or to the centroid of A's five anchors (`centroid/n<n>`);
    - `retrieval.margin` (`rival`): each of B's held-out pieces, its SIM to the centroid of A's
      model pieces less its SIM to the centroid of B's rival pieces;
    - `anchors.sim_min` (`k<k>`): each of B's five anchors against the centroid of every k of
      A's anchors that holds A's solo clip.

    Target trials: clusters of A's own pieces; A's held-out pieces against up to three other
    same-sex speakers' rival centroids; for every anchor set A drew, each anchor against the
    centroid of the other four (`others`, the anchor's register) and the set as a whole (`sets`):
    per embedder, the minimum over its anchors of `retrieve.anchor_check`'s SIMs, which reaches a
    value exactly when every anchor does, so the set passes at a pair exactly when the check
    does. A set's register is `mixed` for RAVDESS and `neutral` otherwise.
    """
    names = tuple(names)
    dkey = _key(name)
    sizes = list(params.cluster_n) + list(params.reported_n)
    centres = {
        SOLO: [retrieve.centroids([s.anchors[0]]) for s in speakers],
        CENTROID: [retrieve.centroids(list(s.anchors)) for s in speakers],
    }
    model = [retrieve.centroids(list(s.pieces[: params.model_pieces])) for s in speakers]
    rival = [retrieve.centroids(list(s.pieces[: params.rival_pieces])) for s in speakers]
    subsets = {
        k: [(0, *rest) for rest in itertools.combinations(range(1, ANCHOR_ROWS + 1), k - 1)]
        for k in params.anchor_k
    }
    subset_centres = [
        {
            k: [retrieve.centroids([s.anchors[i] for i in sub]) for sub in subs]
            for k, subs in subsets.items()
        }
        for s in speakers
    ]
    set_register = MIXED if name == RAVDESS else NEUTRAL
    unset = {n: 0.0 for n in names}  # `anchor_check` needs values; only its SIMs are read
    rows: dict[tuple[str, str], tuple[list, list, list]] = {}
    targets: dict[tuple[str, str], tuple[list, list, list, list]] = {}

    def add(gate: str, form: str, a: int, b: int, row: Sequence[float]) -> None:
        scores, left, right = rows.setdefault((gate, form), ([], [], []))
        scores.append(list(row))
        left.append(a)
        right.append(b)

    def add_target(
        gate: str,
        form: str,
        row: Sequence[float],
        register: str,
        owner: int,
        rivals: Sequence[int] = (),
    ) -> None:
        scores, registers, owners, rival_rows = targets.setdefault((gate, form), ([], [], [], []))
        scores.append(list(row))
        registers.append(register)
        owners.append(owner)
        rival_rows.append(tuple(rivals))

    for ia, a in enumerate(speakers):
        for ib, b in enumerate(speakers):
            if ia == ib or a.sex != b.sex:
                continue
            per_anchor = {
                anchor: _piece_scores(b.pieces, centre[ia], names)
                for anchor, centre in centres.items()
            }
            for n in sizes:
                rng = _rng(seed, _DRAWS, dkey, _key(a.speaker), _key(b.speaker), n)
                for _ in range(params.draws):
                    idx = rng.choice(len(b.pieces), size=n, replace=False)
                    for anchor, scores in per_anchor.items():
                        add(FLOOR, f"{anchor}/n{n}", ia, ib, np.median(scores[idx], axis=0))
            for held in b.pieces[params.model_pieces :]:
                to_a = retrieve.sims(held, model[ia])
                to_b = retrieve.sims(held, rival[ib])
                add(MARGIN, RIVAL, ia, ib, [to_a[n] - to_b[n] for n in names])
            for k in params.anchor_k:
                for centre in subset_centres[ia][k]:
                    for anchor in b.anchors:
                        add(SIM_MIN, f"k{k}", ia, ib, _row(retrieve.sims(anchor, centre), names))
        own = {
            anchor: _piece_scores(a.pieces, centre[ia], names) for anchor, centre in centres.items()
        }
        for n in sizes:
            rng = _rng(seed, _TARGET, dkey, _key(a.speaker), n)
            for _ in range(params.draws):
                idx = rng.choice(len(a.pieces), size=n, replace=False)
                register = _register([a.registers[i] for i in idx])
                for anchor, scores in own.items():
                    median = np.median(scores[idx], axis=0)
                    add_target(FLOOR, f"{anchor}/n{n}", median, register, ia)
        others = [i for i, s in enumerate(speakers) if i != ia and s.sex == a.sex]
        order = _rng(seed, _RIVALS, dkey, _key(a.speaker)).permutation(len(others))
        chosen = [others[int(i)] for i in order[:TARGET_RIVALS]]
        if chosen:
            for held, register in zip(
                a.pieces[params.model_pieces :], a.registers[params.model_pieces :]
            ):
                to_a = retrieve.sims(held, model[ia])
                nearest = [retrieve.sims(held, rival[c]) for c in chosen]
                gap = [to_a[n] - max(s[n] for s in nearest) for n in names]
                add_target(MARGIN, RIVAL, gap, register, ia, chosen)
        for drawn in a.anchor_sets():
            _, found = retrieve.anchor_check(list(drawn.anchors), unset)
            for sims, register in zip(found, drawn.registers):
                add_target(SIM_MIN, OTHERS, _row(sims, names), register, ia)
            least = [min(sims[n] for sims in found) for n in names]
            add_target(SIM_MIN, SETS, least, set_register, ia)

    sex_count: dict[str, int] = {}
    for s in speakers:
        sex_count[s.sex] = sex_count.get(s.sex, 0) + 1
    pairs = sum(n * (n - 1) for n in sex_count.values())
    reported_forms = {f"{anchor}/n{n}" for anchor in centres for n in params.reported_n}
    impostor: dict[str, dict[str, FormTrials]] = {gate: {} for gate in GATES}
    reported: dict[str, dict[str, FormTrials]] = {gate: {} for gate in GATES}
    for (gate, form), (scores, left, right) in rows.items():
        trials = FormTrials(
            scores=np.asarray(scores, dtype=np.float64).reshape(-1, len(names)),
            a=np.asarray(left, dtype=np.int64),
            b=np.asarray(right, dtype=np.int64),
            pairs=pairs,
        )
        (reported if form in reported_forms else impostor)[gate][form] = trials
    target: dict[str, dict[str, TargetTrials]] = {gate: {} for gate in GATES}
    for (gate, form), (scores, registers, owners, rival_rows) in targets.items():
        target[gate][form] = TargetTrials(
            scores=np.asarray(scores, dtype=np.float64).reshape(-1, len(names)),
            registers=tuple(registers),
            owners=np.asarray(owners, dtype=np.int64),
            rivals=tuple(rival_rows),
        )
    return DatasetTrials(
        name=name,
        sexes=tuple(s.sex for s in speakers),
        impostor=impostor,
        reported=reported,
        target=target,
    )


# ---------------------------------------------------------------------------
# v5's equal split (A19), kept only as the comparator `equal_beta`
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FormRule:
    """v5's steps 1-3 on one form: beta*, each embedder's `t_e(beta*)` and the joint FAR there."""

    beta: float
    values: tuple[float, ...]
    joint_far: float
    trials: int
    units: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GateValues:
    """v5's step 4: each embedder's value, the maximum over forms and datasets, and what set it."""

    values: tuple[float, ...]
    set_by: tuple[tuple[str, str], ...]
    rules: dict[tuple[str, str], FormRule]


class _Prepared:
    """One form's scores with each embedder's descending order, kept across resamples."""

    __slots__ = ("scores", "orders", "ordered")

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = np.asarray(scores, dtype=np.float64)
        self.orders = [np.argsort(-col, kind="stable") for col in self.scores.T]
        self.ordered = [self.scores[o, e] for e, o in enumerate(self.orders)]


def _above(score: float) -> int:
    """Return the smallest grid unit k with `k / GRID - EPS > score`: k rejects the score."""
    k = math.floor((score + EPS) * GRID) + 1
    while (k - 1) / GRID - EPS > score:
        k -= 1
    while not k / GRID - EPS > score:
        k += 1
    return k


def _accepted(scores: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Return which trials every embedder accepts: `score >= value - EPS`, extract's rule."""
    return (scores >= np.asarray(values, dtype=np.float64) - EPS).all(axis=1)


def _rule(prep: _Prepared, weights: np.ndarray, alpha_units: int) -> FormRule | None:
    total = int(weights.sum())
    if total == 0:
        return None
    cums = [np.cumsum(weights[o]) for o in prep.orders]

    def units(allowed: int) -> tuple[int, ...]:
        # The first trial in descending order past `allowed` accepted ones must be rejected.
        found = []
        for cum, ordered in zip(cums, prep.ordered):
            index = int(np.searchsorted(cum, allowed, side="right"))
            found.append(_above(float(ordered[index])))
        return tuple(found)

    def joint(allowed: int) -> int:
        values = [u / GRID for u in units(allowed)]
        return int(weights[_accepted(prep.scores, values)].sum())

    # The marginal FAR is at most beta when accepts * GRID <= beta_units * total, so t_e(beta)
    # depends on beta only through allowed = beta_units * total // GRID. The joint FAR never
    # falls as allowed rises, and allowed at alpha always qualifies.
    low, high = alpha_units * total // GRID, BETA_MAX * total // GRID
    limit = alpha_units * total
    while low < high:
        middle = (low + high + 1) // 2
        if joint(middle) * GRID <= limit:
            low = middle
        else:
            high = middle - 1
    best = units(low)
    beta = min(BETA_MAX, ((low + 1) * GRID - 1) // total)
    return FormRule(
        beta=round(beta / GRID, 4),
        values=tuple(round(u / GRID, 4) for u in best),
        joint_far=joint(low) / total,
        trials=total,
        units=best,
    )


def _ones(scores: np.ndarray) -> np.ndarray:
    return np.ones(len(scores), dtype=np.int64)


def threshold(scores: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> FormRule:
    """Run v5's equal-split steps 1-3 on one form's impostor trials.

    Args:
        scores: One row per trial, one column per embedder.
        alpha: The joint false-accept rate to hold.
        weights: How many times each trial counts (a bootstrap resample); one each when `None`.

    Raises:
        StageError: The form has no trial, or alpha is off the grid.
    """
    scores = np.asarray(scores, dtype=np.float64)
    rule = _rule(
        _Prepared(scores), _ones(scores) if weights is None else weights, _alpha_units(alpha)
    )
    if rule is None:
        raise StageError("a form has no trial")
    return rule


def joint_far(
    scores: np.ndarray, values: Sequence[float], weights: np.ndarray | None = None
) -> float:
    """Return the share of trials every embedder accepts at `values`."""
    scores = np.asarray(scores, dtype=np.float64)
    weights = _ones(scores) if weights is None else weights
    total = int(weights.sum())
    return int(weights[_accepted(scores, values)].sum()) / total if total else 0.0


def _gate(
    prepared: Mapping[tuple[str, str], _Prepared],
    weights: Mapping[tuple[str, str], np.ndarray],
    alpha_units: int,
) -> GateValues | None:
    rules: dict[tuple[str, str], FormRule] = {}
    for key, prep in prepared.items():
        rule = _rule(prep, weights[key], alpha_units)
        if rule is not None:
            rules[key] = rule
    if not rules:
        return None
    count = len(next(iter(rules.values())).units)
    best: list[int] = []
    set_by: list[tuple[str, str]] = []
    for e in range(count):
        key = max(rules, key=lambda k: rules[k].units[e])  # the first form wins a tie
        best.append(rules[key].units[e])
        set_by.append(key)
    return GateValues(
        values=tuple(round(u / GRID, 4) for u in best), set_by=tuple(set_by), rules=rules
    )


def gate_values(forms: Mapping[tuple[str, str], np.ndarray], alpha: float) -> GateValues:
    """Run v5's equal-split step 4 on one gate: forms keyed by (dataset, form), never pooled.

    Raises:
        StageError: No form has a trial, or alpha is off the grid.
    """
    prepared = {key: _Prepared(scores) for key, scores in forms.items()}
    found = _gate(prepared, {k: _ones(p.scores) for k, p in prepared.items()}, _alpha_units(alpha))
    if found is None:
        raise StageError("a gate has no trial")
    return found


# ---------------------------------------------------------------------------
# The threshold rule (A22): the least matched target miss within alpha
# ---------------------------------------------------------------------------

Key = tuple[str, str]
"""A form of one dataset, (dataset, form); forms of two datasets are never pooled."""
_UNSET = -(1 << 62)
"""`_bounds`' value for a prefix that does not bound the second embedder's value."""
TIE = 1e-12
"""Rates closer than this tie in step 4, so a mean's float rounding cannot order two pairs."""


def _accept_units(scores: object) -> np.ndarray:
    """Return per score the largest grid unit u that accepts it: `score >= u / GRID - EPS`.

    A value of u units accepts a trial exactly when u is at most the trial's unit, so the rule
    compares integers and still applies extract's own test (`retrieve.EPS`).
    """
    scores = np.asarray(scores, dtype=np.float64)
    units = np.floor((scores + EPS) * GRID).astype(np.int64)
    while (high := units / GRID - EPS > scores).any():
        units = units - high
    while (low := (units + 1) / GRID - EPS <= scores).any():
        units = units + low
    return units


class _Units:
    """One form's trials as accept units per embedder, with the first embedder's order kept."""

    __slots__ = ("units", "order")

    def __init__(self, scores: np.ndarray) -> None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != 2:
            raise StageError("the threshold rule takes trials scored on exactly two embedders")
        self.units = _accept_units(scores)
        self.order = np.argsort(-self.units[:, 0], kind="stable")

    def accepted(self, pair: Sequence[int], weights: np.ndarray) -> int:
        """Return the weight the pair accepts, both embedders at or above their values."""
        hit = (self.units[:, 0] >= pair[0]) & (self.units[:, 1] >= pair[1])
        return int(weights[hit].sum())


@dataclass(frozen=True, slots=True)
class GateTrials:
    """One gate's trials in grid units, prepared once for every solve of the gate.

    `impostor` holds the constraint forms, `matched` the matched target forms. `low` and `high`
    are step 2's range per embedder over every trial of the gate: the largest grid unit that
    accepts every trial, and the smallest that accepts none.
    """

    impostor: dict[Key, _Units]
    matched: dict[Key, _Units]
    low: tuple[int, int]
    high: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Solution:
    """The chosen pair and what it costs.

    `miss` and `mean_miss` are step 4's keys 1 and 2 over the matched forms, `far` its key 3
    over the impostor forms, and `binding` the impostor form with the largest FAR (the first in
    order on a tie). `misses` and `fars` hold each form's weighted rate at the pair, for the
    forms that carry weight.
    """

    values: tuple[float, float]
    units: tuple[int, int]
    miss: float
    mean_miss: float
    far: float
    binding: Key
    misses: dict[Key, float]
    fars: dict[Key, float]


def prepare_gate(
    impostor: Mapping[Key, np.ndarray],
    matched: Mapping[Key, np.ndarray],
    others: Sequence[np.ndarray] = (),
) -> GateTrials:
    """Return a gate's trials for `solve`; `others` are its remaining trials, read for the range.

    Raises:
        StageError: The gate has no trial, or a form is not scored on two embedders.
    """
    every = [
        np.asarray(s, dtype=np.float64).reshape(-1, 2)
        for s in (*impostor.values(), *matched.values(), *others)
    ]
    every = [s for s in every if len(s)]
    if not every:
        raise StageError("a gate has no trial")
    stacked = np.vstack(every)
    low = _accept_units(stacked.min(axis=0))
    high = _accept_units(stacked.max(axis=0)) + 1
    return GateTrials(
        impostor={key: _Units(scores) for key, scores in impostor.items()},
        matched={key: _Units(scores) for key, scores in matched.items()},
        low=(int(low[0]), int(low[1])),
        high=(int(high[0]), int(high[1])),
    )


def _weights(
    forms: Mapping[Key, _Units], given: Mapping[Key, np.ndarray] | None
) -> dict[Key, np.ndarray]:
    if given is None:
        return {key: np.ones(len(form.units), dtype=np.int64) for key, form in forms.items()}
    found = {key: np.asarray(given[key], dtype=np.int64) for key in forms}
    if any(len(found[key]) != len(form.units) for key, form in forms.items()):
        raise StageError("a form's weights do not match its trials")
    return found


def _bounds(form: _Units, weights: np.ndarray, allowed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the form's first-embedder units, descending, and the least v2 of each prefix.

    As v1 falls, the trials it admits are a growing prefix in that order. `bound[i]` is the least
    second-embedder unit at which the first i trials accept at most `allowed` weight: one above
    their (allowed + 1)-th largest second unit, counted by weight, or `_UNSET` while they weigh
    `allowed` or less. A min-heap keeps the fewest largest units that weigh more than `allowed`,
    so its least entry is that unit (the Design's prefix sweep).
    """
    order = form.order[weights[form.order] > 0]
    bound = [_UNSET] * (len(order) + 1)
    heap: list[tuple[int, int]] = []
    kept = 0
    seconds = form.units[order, 1].tolist()
    for i, (unit, count) in enumerate(zip(seconds, weights[order].tolist())):
        heapq.heappush(heap, (unit, count))
        kept += count
        while kept - heap[0][1] > allowed:
            kept -= heapq.heappop(heap)[1]
        if kept > allowed:
            bound[i + 1] = heap[0][0] + 1
    return form.units[order, 0], np.asarray(bound, dtype=np.int64)


def _accepted_at(
    form: _Units, weights: np.ndarray, firsts: np.ndarray, seconds: np.ndarray
) -> np.ndarray:
    """Return the weight the form accepts at each candidate (`firsts[i]`, `seconds[i]`).

    `firsts` rises and `seconds` never does, so a trial is accepted over one run of candidates:
    from the first whose v2 it reaches to the last whose v1 it reaches.
    """
    keep = weights > 0
    units, counts = form.units[keep], weights[keep]
    right = np.searchsorted(firsts, units[:, 0], side="right")
    left = np.searchsorted(-seconds, -units[:, 1], side="left")
    run = left < right
    size = len(firsts) + 1
    edges = np.bincount(left[run], weights=counts[run], minlength=size) - np.bincount(
        right[run], weights=counts[run], minlength=size
    )
    return np.rint(np.cumsum(edges)[:-1]).astype(np.int64)


def solve(
    gate: GateTrials,
    alpha_units: int,
    impostor_weights: Mapping[Key, np.ndarray] | None = None,
    target_weights: Mapping[Key, np.ndarray] | None = None,
) -> Solution | None:
    """Choose a gate's pair by the threshold rule, steps 2-6; `None` when no impostor weighs.

    Weights count a trial that many times, in FAR and miss alike (a bootstrap resample, or a
    left-out speaker at 0); every form of `gate` needs one, one each when `None`. The candidates
    are v1 at each unit some weighted trial reaches, and at `high`: between two of them every
    key is the same, so key 4 takes the upper one, and the search equals one over every v1 of
    the grid. Each v1's v2 is the least unit at or above `low` holding alpha in every impostor
    form, and is never raised further.

    Raises:
        StageError: A form's weights do not match its trials.
    """
    impostor_w = _weights(gate.impostor, impostor_weights)
    target_w = _weights(gate.matched, target_weights)
    totals = {key: int(w.sum()) for key, w in impostor_w.items()}
    live = [key for key in gate.impostor if totals[key] > 0]
    if not live:
        return None
    targets = {key: int(w.sum()) for key, w in target_w.items()}
    aimed = [key for key in gate.matched if targets[key] > 0]
    bounds = {
        key: _bounds(gate.impostor[key], impostor_w[key], alpha_units * totals[key] // GRID)
        for key in live
    }
    firsts = np.unique(
        np.concatenate(
            [
                *(bounds[key][0] for key in live),
                *(gate.matched[key].units[target_w[key] > 0, 0] for key in aimed),
                np.array([gate.high[0]], dtype=np.int64),
            ]
        )
    )
    seconds = np.full(len(firsts), gate.low[1], dtype=np.int64)
    for key in live:
        units, bound = bounds[key]
        seconds = np.maximum(seconds, bound[np.searchsorted(-units, -firsts, side="right")])
    fars = {
        key: _accepted_at(gate.impostor[key], impostor_w[key], firsts, seconds) / totals[key]
        for key in live
    }
    misses = {
        key: (targets[key] - _accepted_at(gate.matched[key], target_w[key], firsts, seconds))
        / targets[key]
        for key in aimed
    }
    far = np.vstack(list(fars.values())).max(axis=0)
    if misses:
        stacked = np.vstack(list(misses.values()))
        worst, mean = stacked.max(axis=0), stacked.mean(axis=0)
    else:
        worst = mean = np.zeros(len(firsts))
    keep = np.ones(len(firsts), dtype=bool)
    for key in (worst, mean, far):
        keep &= key <= key[keep].min() + TIE
    i = int(np.flatnonzero(keep)[-1])
    rates = {key: float(rate[i]) for key, rate in fars.items()}
    pair = (int(firsts[i]), int(seconds[i]))
    return Solution(
        values=(round(pair[0] / GRID, 4), round(pair[1] / GRID, 4)),
        units=pair,
        miss=float(worst[i]),
        mean_miss=float(mean[i]),
        far=float(far[i]),
        binding=max(live, key=lambda key: rates[key]),
        misses={key: float(rate[i]) for key, rate in misses.items()},
        fars=rates,
    )


def least_miss(
    impostor: Mapping[Key, np.ndarray],
    matched: Mapping[Key, np.ndarray],
    alpha: float,
    impostor_weights: Mapping[Key, np.ndarray] | None = None,
    target_weights: Mapping[Key, np.ndarray] | None = None,
    others: Sequence[np.ndarray] = (),
) -> Solution:
    """Run the threshold rule on one gate's trials, scores in rows, `EMBEDDERS` order in columns.

    Raises:
        StageError: No impostor trial weighs, alpha is off the grid, or a form is malformed.
    """
    found = solve(
        prepare_gate(impostor, matched, others),
        _alpha_units(alpha),
        impostor_weights,
        target_weights,
    )
    if found is None:
        raise StageError("a gate has no impostor trial")
    return found


def gate_trials(datasets: Sequence[DatasetTrials], gate: str, matched: Sequence[str]) -> GateTrials:
    """Return one gate's trials for `solve`: its impostor forms and its matched target forms."""
    impostor = {(d.name, f): t.scores for d in datasets for f, t in d.impostor[gate].items()}
    targets = {
        (d.name, f): d.target[gate][f].scores
        for d in datasets
        for f in matched
        if f in d.target[gate]
    }
    others = [
        t.scores for d in datasets for group in (d.reported, d.target) for t in group[gate].values()
    ]
    return prepare_gate(impostor, targets, others)


def gate_weights(
    datasets: Sequence[DatasetTrials],
    gate: str,
    matched: Sequence[str],
    copies: Mapping[str, np.ndarray],
) -> tuple[dict[Key, np.ndarray], dict[Key, np.ndarray]]:
    """Return each trial's weight when speaker i of a dataset counts `copies[dataset][i]` times.

    An impostor trial (A, B) weighs the copies of A times the copies of B, so a speaker never
    meets a copy of itself; a target trial weighs its owner's copies.
    """
    impostor = {
        (d.name, f): copies[d.name][t.a] * copies[d.name][t.b]
        for d in datasets
        for f, t in d.impostor[gate].items()
    }
    target = {
        (d.name, f): copies[d.name][d.target[gate][f].owners]
        for d in datasets
        for f in matched
        if f in d.target[gate]
    }
    return impostor, target


@dataclass(frozen=True, slots=True)
class EqualSplit:
    """v5's equal-split pair on the same trials, and its matched misses (`equal_beta`)."""

    values: tuple[float, ...]
    miss: float
    mean_miss: float


@dataclass(frozen=True, slots=True)
class Summary:
    """The values per gate, and the bootstrap's 5th-95th percentile intervals.

    `intervals` holds each embedder's value interval per gate; `far_intervals` the joint FAR
    interval at the chosen values per gate and (dataset, form), reported forms included;
    `miss_intervals` each target form's miss interval at the chosen values per register, `all`
    included. `resamples` counts, per gate, the resamples that gave values (one whose speakers
    are all copies of one per sex has no pair).
    """

    values: dict[str, tuple[float, ...]]
    solutions: dict[str, Solution]
    equal: dict[str, EqualSplit]
    intervals: dict[str, tuple[tuple[float, float] | None, ...]]
    far_intervals: dict[str, dict[Key, tuple[float, float] | None]]
    miss_intervals: dict[str, dict[Key, dict[str, tuple[float, float] | None]]]
    resamples: dict[str, int]


def _interval(samples: Sequence[float], places: int) -> tuple[float, float] | None:
    if not samples:
        return None
    low, high = np.percentile(np.asarray(samples, dtype=np.float64), [5, 95])
    return round(float(low), places), round(float(high), places)


def _equal_split(
    datasets: Sequence[DatasetTrials], gate: str, prepared: GateTrials, alpha_units: int
) -> EqualSplit:
    """Return v5's equal-split pair on the gate's impostor forms, and its matched misses."""
    forms = {
        (d.name, f): _Prepared(t.scores) for d in datasets for f, t in d.impostor[gate].items()
    }
    found = _gate(forms, {k: _ones(p.scores) for k, p in forms.items()}, alpha_units)
    if found is None:
        raise StageError(f"{gate} has no trial")
    pair = [int(round(v * GRID)) for v in found.values]
    rates = []
    for form in prepared.matched.values():
        ones = np.ones(len(form.units), dtype=np.int64)
        if len(ones):
            rates.append((len(ones) - form.accepted(pair, ones)) / len(ones))
    return EqualSplit(
        values=found.values,
        miss=max(rates, default=0.0),
        mean_miss=float(np.mean(rates)) if rates else 0.0,
    )


def _registers(target: TargetTrials) -> tuple[list[str], np.ndarray]:
    """Return a form's registers, `all` first, and each trial's register index."""
    registers = [ALL, *sorted(set(target.registers))]
    index = {register: i for i, register in enumerate(registers)}
    return registers, np.array([index[r] for r in target.registers], dtype=np.int64)


def summarise(
    datasets: Sequence[DatasetTrials],
    alpha: float,
    seed: int,
    bootstrap: int,
    names: Sequence[str],
    matched: Mapping[str, Sequence[str]],
) -> Summary:
    """Set each gate's values by the threshold rule and bootstrap their intervals.

    The bootstrap resamples speakers with replacement within each sex and dataset and solves the
    rule again on each resample. A pair of copies of A and B carries the original (A, B) trials,
    so an impostor trial weighs the copies of A times the copies of B, and a speaker never meets
    a copy of itself; a target trial weighs its owner's copies.

    Raises:
        StageError: A gate has no trial, alpha is off the grid, or `names` is not two embedders.
    """
    if len(names) != 2:
        raise StageError("the threshold rule takes exactly two embedders")
    alpha_units = _alpha_units(alpha)
    prepared = {gate: gate_trials(datasets, gate, matched[gate]) for gate in GATES}
    solutions: dict[str, Solution] = {}
    for gate in GATES:
        found = solve(prepared[gate], alpha_units)
        if found is None:
            raise StageError(f"{gate} has no trial")
        solutions[gate] = found
    values = {gate: solutions[gate].values for gate in GATES}
    equal = {gate: _equal_split(datasets, gate, prepared[gate], alpha_units) for gate in GATES}
    every = {
        gate: {
            (d.name, f): t
            for d in datasets
            for f, t in [*d.impostor[gate].items(), *d.reported[gate].items()]
        }
        for gate in GATES
    }
    targets = {
        gate: {
            (d.name, f): (t, *_registers(t), ~_accepted(t.scores, values[gate]))
            for d in datasets
            for f, t in d.target[gate].items()
        }
        for gate in GATES
    }
    value_samples: dict[str, list[tuple[float, ...]]] = {gate: [] for gate in GATES}
    far_samples: dict[str, dict[Key, list[float]]] = {
        gate: {key: [] for key in every[gate]} for gate in GATES
    }
    miss_samples: dict[str, dict[Key, list[list[float]]]] = {
        gate: {key: [[] for _ in found[1]] for key, found in targets[gate].items()}
        for gate in GATES
    }
    groups = {
        d.name: [np.flatnonzero(np.asarray(d.sexes) == sex) for sex in sorted(set(d.sexes))]
        for d in datasets
    }
    rng = _rng(seed, _BOOT)
    for _ in range(bootstrap):
        counts = {}
        for d in datasets:
            copies = np.zeros(len(d.sexes), dtype=np.int64)
            for group in groups[d.name]:
                np.add.at(copies, group[rng.integers(0, len(group), len(group))], 1)
            counts[d.name] = copies
        for gate in GATES:
            impostor_w, target_w = gate_weights(datasets, gate, matched[gate], counts)
            found = solve(prepared[gate], alpha_units, impostor_w, target_w)
            if found is not None:
                value_samples[gate].append(found.values)
            for key, trials in every[gate].items():
                weights = counts[key[0]][trials.a] * counts[key[0]][trials.b]
                if int(weights.sum()):
                    far_samples[gate][key].append(joint_far(trials.scores, values[gate], weights))
            for key, (target, registers, codes, missed) in targets[gate].items():
                weights = counts[key[0]][target.owners].astype(np.float64)
                size = len(registers)
                total = np.bincount(codes, weights=weights, minlength=size)
                lost = np.bincount(codes, weights=weights * missed, minlength=size)
                total[0], lost[0] = weights.sum(), (weights * missed).sum()
                for i in np.flatnonzero(total > 0):
                    miss_samples[gate][key][i].append(float(lost[i] / total[i]))
    intervals = {
        gate: tuple(_interval([v[e] for v in value_samples[gate]], 4) for e in range(len(names)))
        for gate in GATES
    }
    far_intervals = {
        gate: {key: _interval(samples, 6) for key, samples in far_samples[gate].items()}
        for gate in GATES
    }
    miss_intervals = {
        gate: {
            key: {
                register: _interval(miss_samples[gate][key][i], 6)
                for i, register in enumerate(found[1])
            }
            for key, found in targets[gate].items()
        }
        for gate in GATES
    }
    return Summary(
        values=values,
        solutions=solutions,
        equal=equal,
        intervals=intervals,
        far_intervals=far_intervals,
        miss_intervals=miss_intervals,
        resamples={gate: len(value_samples[gate]) for gate in GATES},
    )


def misses(
    trials: DatasetTrials, values: Mapping[str, Sequence[float]], names: Sequence[str]
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Return the target miss rates per gate, form and register (`all` for every register).

    `by` splits each miss by the embedders that rejected it: each embedder alone, or `both`
    when more than one did, so the log shows which embedder's value a register runs into.
    """
    found: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for gate in GATES:
        found[gate] = {}
        for form, target in trials.target[gate].items():
            scores = np.asarray(target.scores, dtype=np.float64).reshape(-1, len(names))
            rejects = scores < np.asarray(values[gate], dtype=np.float64) - EPS
            missed_on = rejects.sum(axis=1)
            registers = np.asarray(target.registers)
            rows: dict[str, dict[str, object]] = {}
            for register in [ALL, *sorted(set(target.registers))]:
                mask = np.ones(len(registers), bool) if register == ALL else registers == register
                count = int(mask.sum())
                missed = int((mask & (missed_on > 0)).sum())
                alone = mask & (missed_on == 1)
                by = {n: int((alone & rejects[:, e]).sum()) for e, n in enumerate(names)}
                by[BOTH] = int((mask & (missed_on > 1)).sum())
                rows[register] = {
                    "trials": count,
                    "missed": missed,
                    "rate": round(missed / count, 6) if count else 0.0,
                    "by": by,
                }
            found[gate][form] = rows
    return found


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Loaded:
    """One dataset's eligible speakers and exclusions, as embedded."""

    plan: DatasetPlan
    speakers: list[Speaker]
    excluded: list[tuple[str, str, str]]
    digest: str


def _shuffled(items: Sequence, seed: int, dataset: str, speaker: str) -> list:
    order = _rng(seed, _SHUFFLE, _key(dataset), _key(speaker)).permutation(len(items))
    return [items[int(i)] for i in order]


def _admit(
    loaded: list[Speaker],
    excluded: list[tuple[str, str, str]],
    embedders: Sequence[object],
    clips: Sequence[Clip],
    plan: Plan,
    dataset: str,
    speaker: str,
    sex: str,
) -> None:
    """Pool a speaker's clips, given in path order, and embed the pool and its later draws."""
    shuffled = _shuffled(clips, plan.seed, dataset, speaker)
    if dataset == RAVDESS:
        pool = ravdess_pool(shuffled, RATE, plan.trials)
    else:
        pool = librittsr_pool([c.audio for c in shuffled], RATE, plan.trials)
    if isinstance(pool, str):
        excluded.append((speaker, sex, pool))
        print(f"calibrate: {dataset} speaker {speaker} ({sex}) excluded: {pool}")
        return
    draws, skipped = anchor_draws(clips, dataset, speaker, plan.seed, plan.trials.anchor_sets, RATE)
    loaded.append(embed_pool(embedders, pool, dataset, speaker, sex, RATE, draws, skipped))
    print(
        f"calibrate: {dataset} speaker {speaker} ({sex}): {len(pool.pieces)} pieces and "
        f"{1 + len(draws)} anchor sets embedded, {skipped} draws skipped"
    )


def load_dataset(
    data: Path, dataset: DatasetPlan, plan: Plan, embedders: Sequence[object]
) -> Loaded:
    """Read, pool and embed one dataset's speakers, one speaker's audio at a time."""
    speakers: list[Speaker] = []
    excluded: list[tuple[str, str, str]] = list(dataset.excluded)
    if dataset.archive is not None:
        path = data / dataset.archive[0]
        for actor, sex in ravdess_actors(path).items():
            clips = read_ravdess(path, RATE, actors=(actor,))[actor]
            _admit(speakers, excluded, embedders, clips, plan, dataset.name, actor, sex)
        digest = dataset.archive[1]
    else:
        for entry in dataset.speakers:
            clips = []
            for rel, _ in entry.files:
                audio, native = read_audio(data / rel)
                clips.append(Clip(resample(trim(audio, native), native, RATE), NEUTRAL))
            _admit(
                speakers, excluded, embedders, clips, plan, dataset.name, entry.speaker, entry.sex
            )
        manifest = json.dumps(sorted(dataset.files().items())).encode("utf-8")
        digest = hashlib.sha256(manifest).hexdigest()
    return Loaded(plan=dataset, speakers=speakers, excluded=excluded, digest=digest)


def _counts(loaded: Loaded) -> dict[str, dict[str, int]]:
    return {
        sex: {
            "eligible": sum(s.sex == sex for s in loaded.speakers),
            "excluded": sum(row[1] == sex for row in loaded.excluded),
        }
        for sex in SEXES
    }


def report(
    plan: Plan,
    loaded: Sequence[Loaded],
    trials: Sequence[DatasetTrials],
    summary: Summary,
    names: Sequence[str],
    bootstrap: int,
) -> dict:
    """Return `calibration.json`'s content (Interfaces)."""
    gates: dict[str, dict] = {}
    missed = {d.name: misses(d, summary.values, names) for d in trials}
    for gate in GATES:
        solution = summary.solutions[gate]
        chosen = summary.values[gate]
        forms: dict[str, dict] = {}
        reported: dict[str, dict] = {}
        miss: dict[str, dict] = {}
        for d in trials:
            forms[d.name] = {
                form: {
                    "joint_far": round(joint_far(t.scores, chosen), 6),
                    "far_interval": summary.far_intervals[gate][(d.name, form)],
                    "trials": len(t.scores),
                    "pairs": t.pairs,
                }
                for form, t in d.impostor[gate].items()
            }
            if d.reported[gate]:
                reported[d.name] = {
                    form: {
                        "joint_far": round(joint_far(t.scores, chosen), 6),
                        "far_interval": summary.far_intervals[gate][(d.name, form)],
                        "trials": len(t.scores),
                        "pairs": t.pairs,
                    }
                    for form, t in d.reported[gate].items()
                }
            miss[d.name] = {
                form: {
                    register: {
                        "trials": row["trials"],
                        "missed": row["missed"],
                        "rate": row["rate"],
                        "rate_interval": summary.miss_intervals[gate][(d.name, form)][register],
                        "by": row["by"],
                    }
                    for register, row in rows.items()
                }
                for form, rows in missed[d.name][gate].items()
            }
        binding = solution.binding
        equal = summary.equal[gate]
        gates[gate] = {
            "values": dict(zip(names, chosen)),
            "intervals": dict(zip(names, summary.intervals[gate])),
            "objective": {
                "miss": round(solution.miss, 6),
                "mean_miss": round(solution.mean_miss, 6),
            },
            "binding": {
                "dataset": binding[0],
                "form": binding[1],
                "joint_far": round(solution.fars[binding], 6),
            },
            "forms": forms,
            "reported": reported,
            "misses": miss,
            "equal_beta": {
                "values": dict(zip(names, equal.values)),
                "miss": round(equal.miss, 6),
                "mean_miss": round(equal.mean_miss, 6),
            },
            "bootstrap": {"resamples": bootstrap, "used": summary.resamples[gate]},
        }
    params = asdict(plan.trials)
    return {
        "schema": SCHEMA,
        "id": plan.id,
        "alpha": plan.alpha,
        "seed": plan.seed,
        "datasets": [
            {
                "name": d.plan.name,
                "licence": d.plan.licence,
                "subset": d.plan.subset,
                "digest": d.digest,
                "speakers": _counts(d),
                "excluded": [
                    {"speaker": s, "sex": sex, "reason": reason} for s, sex, reason in d.excluded
                ],
            }
            for d in loaded
        ],
        "anchor_sets": {
            d.plan.name: {
                "drawn": sum(len(s.anchor_sets()) for s in d.speakers),
                "skipped": sum(s.skipped for s in d.speakers),
            }
            for d in loaded
        },
        "embedders": {n: {"revision": r} for n, r in plan.embedders.items()},
        "matched": {gate: list(plan.matched[gate]) for gate in GATES},
        "trials": {
            **{k: list(v) if isinstance(v, tuple) else v for k, v in params.items()},
            "min_pieces": MIN_PIECES,
            "min_eligible_per_sex": MIN_ELIGIBLE,
            "anchor_rows": ANCHOR_ROWS,
            "anchor_row_s": [ROW_MIN_S, ROW_MAX_S],
            "solo_s": [ANCHOR_MIN_S, ANCHOR_MAX_S],
            "gap_s": ANCHOR_GAP_S,
            "min_piece_s": MIN_EMBED_S,
            "target_rivals": TARGET_RIVALS,
            "rate": RATE,
            "grid": 1 / GRID,
            "beta_max": BETA_MAX / GRID,
        },
        "gates": gates,
        "keys": {f"{gate}.{n}": v for gate in GATES for n, v in zip(names, summary.values[gate])},
    }


def _span(interval: object) -> str:
    if interval is None:
        return "none"
    low, high = interval  # type: ignore[misc]
    return f"[{low}, {high}]"


def summary_lines(result: Mapping[str, object]) -> list[str]:
    """Return the printed summary, naming no file.

    Per gate: the values and intervals, the binding impostor form, the objective (the largest
    matched miss and the form that sets it, and the mean), v5's equal-split pair beside it, each
    form's FAR and counts, and every target miss rate per dataset, form and register with its
    interval and the embedders that missed it, so a run can be judged from its log alone.
    """
    lines = []
    for dataset in result["datasets"]:  # type: ignore[attr-defined]
        for sex, row in dataset["speakers"].items():
            lines.append(
                f"calibrate: {dataset['name']} {sex}: {row['eligible']} eligible, "
                f"{row['excluded']} excluded"
            )
    for dataset, row in result["anchor_sets"].items():  # type: ignore[attr-defined]
        lines.append(
            f"calibrate: {dataset} anchor sets: {row['drawn']} drawn, {row['skipped']} skipped"
        )
    for gate, entry in result["gates"].items():  # type: ignore[attr-defined]
        for name, value in entry["values"].items():
            lines.append(
                f"calibrate: {gate}.{name} = {value}, interval {_span(entry['intervals'][name])}"
            )
        binding = entry["binding"]
        lines.append(
            f"calibrate: {gate} binding: {binding['dataset']} {binding['form']}, joint FAR "
            f"{binding['joint_far']}"
        )
        rates = {
            (dataset, form): forms[form][ALL]["rate"]
            for dataset, forms in entry["misses"].items()
            for form in result["matched"][gate]  # type: ignore[index]
            if form in forms
        }
        where = ""
        if rates:
            dataset, form = max(rates, key=rates.__getitem__)
            where = f" ({dataset} {form})"
        objective = entry["objective"]
        lines.append(
            f"calibrate: {gate} objective: largest matched miss {objective['miss']}{where}, "
            f"mean matched miss {objective['mean_miss']}"
        )
        equal = entry["equal_beta"]
        pair = ", ".join(f"{name} {value}" for name, value in equal["values"].items())
        lines.append(
            f"calibrate: {gate} equal split: {pair}; largest matched miss {equal['miss']}, "
            f"mean matched miss {equal['mean_miss']}"
        )
        for dataset, forms in entry["forms"].items():
            for form, row in forms.items():
                lines.append(
                    f"calibrate: {gate} {dataset} {form}: joint FAR {row['joint_far']} at the "
                    f"values {_span(row['far_interval'])}; {row['trials']} trials, "
                    f"{row['pairs']} pairs"
                )
        for dataset, forms in entry["reported"].items():
            for form, row in forms.items():
                lines.append(
                    f"calibrate: {gate} {dataset} {form} (reported): joint FAR "
                    f"{row['joint_far']} at the values {_span(row['far_interval'])}; "
                    f"{row['trials']} trials, {row['pairs']} pairs"
                )
        for dataset, forms in entry["misses"].items():
            for form, rows in forms.items():
                for register, row in rows.items():
                    by = ", ".join(
                        f"{name} only {count}" if name != BOTH else f"{BOTH} {count}"
                        for name, count in row["by"].items()
                    )
                    lines.append(
                        f"calibrate: {gate} {dataset} {form} {register}: missed {row['missed']} "
                        f"of {row['trials']} trials, rate {row['rate']}, interval "
                        f"{_span(row['rate_interval'])}; by {by}"
                    )
    return lines


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


def run(data: Path, plan_rel: str) -> list[str]:
    """Run the calibration; return the root-relative files written.

    Raises:
        StageError: The plan is refused, a file is missing or altered, the embedders differ from
            the plan's, or a dataset has too few eligible speakers.
    """
    parts = PurePosixPath(plan_rel).parts
    if not parts or plan_rel.startswith("/") or any(p in ("..", ".") for p in parts):
        raise StageError("--plan is not a relative path under --data")
    plan = read_plan(data / plan_rel)
    checked = verify(data, plan)
    print(f"calibrate: {checked} files checked against the plan's sha256")
    embedders = load_embedders()
    names = tuple(e.name for e in embedders)
    if sorted(names) != sorted(plan.embedders):
        raise StageError(
            f"the job loaded {len(names)} embedders that differ from the plan's "
            f"{len(plan.embedders)}"
        )
    loaded: list[Loaded] = []
    trials: list[DatasetTrials] = []
    for dataset in plan.datasets:
        found = load_dataset(data, dataset, plan, embedders)
        require_speakers(dataset.name, {s: c["eligible"] for s, c in _counts(found).items()})
        loaded.append(found)
        trials.append(dataset_trials(dataset.name, found.speakers, plan.trials, plan.seed, names))
    summary = summarise(trials, plan.alpha, plan.seed, plan.trials.bootstrap, names, plan.matched)
    result = report(plan, loaded, trials, summary, names, plan.trials.bootstrap)
    folder = PurePosixPath(plan_rel).parent
    rel = (folder / CALIBRATION_NAME).as_posix()
    _write_json(data / rel, result)
    for line in summary_lines(result):
        print(line)
    for key, value in result["keys"].items():
        print(f"calibrate: key {key} = {value}")
    return [rel]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job; write `outputs.json` beside the plan for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.rehearsal.calibrate")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    try:
        written = run(args.data, args.plan)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    folder = PurePosixPath(args.plan).parent
    files = [{"rel": rel, "sha256": sha256_file(args.data / rel)} for rel in written]
    _write_json(args.data / folder / OUTPUTS_NAME, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
