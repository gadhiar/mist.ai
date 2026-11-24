"""Dialogue extraction and target-speaker retrieval (Design 4), run as the `extract` family's job.

Runs as `/opt/venv-extract/bin/python -I -u -m scripts.voice_data.stages.extract --data /data
--run R --device D [--vram-gb N]`, with each source's copied track and `seed/reference/`
mounted read-only and `seed/runs/<run>/` writable. On cuda, `--vram-gb` is required: the
allocator is capped under it before any model loads, and an out-of-memory error ends the run
with a result naming where it happened (`_job`). The host's plan names sources by id and gives
the timestamp anchors as source ids and times, so no file name reaches this job. The job log
carries a progress line per source, per step and channel, and per minute of TIGER-DnR windows
(`Tracker`), with ids, counts, durations and MiB only.

The order is Design 4's:

1. Every anchor is 3-30 s. The solo clips are read as they are.
2. Per source, the centre channel (or the downmix when the layout has none) goes through every
   separator candidate, `none` included. Each candidate's reference is the solo clips, else its
   own timestamp anchors cut from its own output. The median SIM of its 50 best 3 s VAD speech
   windows to that reference gives a provisional choice, whose output is diarized and embedded.
   The provisional target cluster is the one nearest its reference, with no floor; its embedded
   turns that touch no overlap are the spans. Each candidate is scored again by the median SIM
   of its own cuts of every span, and the best is kept; with no span, or no reference for the
   provisional candidate, the window scores decide. The chosen output is diarized again only
   when it is not the provisional one, and the downmix goes through it too.
3. The anchors are checked for agreement, timestamp anchors cut from their source's chosen
   output. The check reports and never refuses (Raj, 16 Nov): its result is `anchors_agree` in
   `extract-result.json` and `anchors.agree` in `report.json`, and the run goes on with the
   anchors as they are, so the host flags a set that disagrees and still exits 0. When
   `seed/sdh/` is mounted, the SDH spans `stages/sdh.py` holds as anchors are cut the same way
   and checked against the centroid of these anchors: one below that embedder's Anchors value on
   either embedder (each embedder has its own, since their cosines sit on different scales) is
   dropped and counted, and the rest join the anchor centroid that retrieval uses. They never
   enter the agreement check, so they can neither cause nor mask a disagreement.
4. Diarization on the separated centre channel; the first retrieval pass names each episode's
   primary and flagged clusters; the target model and the 1.5 s window model are built from
   primary turns that touch no overlap.
5. The second pass keeps a turn by the margin over non-admitted clusters and the floor. A turn
   it would drop on its centre-channel cut is judged again on the separated downmix cut and kept
   from there as `fc-failed:sim` when that passes. Downmix speech with no centre-channel speech
   at the same time becomes off-centre candidates, kept only as flagged.
6. Turns are cut into segments at VAD pauses and gated on overlap, SNR, C50 and the window
   check; a centre-channel cut failing SNR, C50 or the window check is retried from the downmix
   (`fc-failed:<gate>`). Every drop is a record with its reasons.
7. States are proposed for kept segments, and `report.json` is written with the keys the seed
   gate reads. `turns.jsonl` records every diarized turn, whatever its cluster's role, with the
   SIMs and per-embedder gaps retrieval measured on it, so a scorer can tell which cluster held
   which speaker.

Models come from `load_models`, which T11's bindings serve; tests pass fakes through it.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..bindings.protocols import (
    ArousalMonitor,
    Diarizer,
    QualityEstimator,
    QualityTrack,
    Separator,
    Span,
    SpeakerEmbedder,
    Turn,
    Vad,
)
from ..commands import extract as contract
from ..errors import JobError
from ..records import Prosody, SegmentRecord
from ..states import SPEAKING_STATES
from . import propose_states, retrieve, segments, vram
from . import sdh as sdh_stage
from .ingest import decode_argv
from .prosody import prosody as measure_prosody
from .vram import CAP_MARGIN_GIB, CUDA

SCHEMA = 1
NONE = "none"
FC = "FC"
DOWNMIX = "downmix"
CHOICE_WINDOW_S = 3.0
CHOICE_BEST_N = 50
# How a source's separator was chosen: on the provisional target turns, or on the best windows.
BASIS_TURNS = "turns"
BASIS_WINDOWS = "windows"
# A turn shorter than this is too short to embed; it is recorded as a length drop.
MIN_EMBED_S = 0.5
AUDIO_SUBTYPE = "PCM_24"
# The steps an out-of-memory result can name, besides a separator's id.
STEP_LOAD = "load"
STEP_DIARIZE = "diarize"
STEP_EMBED = "embed"
STEP_VAD = "vad"
STEP_QUALITY = "quality"
STEP_RETRIEVE = "retrieve"
# A separator that reports its windows gets a progress line at least this often.
WINDOW_REPORT_S = 60.0
# The clock progress lines are timed on; `main` hands it to the `Tracker`.
clock: Callable[[], float] = time.monotonic


class StageError(RuntimeError):
    """The run cannot proceed; the message names ids and counts only."""


class Unseparated:
    """The `none` candidate: the channel as it is."""

    name = NONE

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return the input unchanged."""
        return audio


@dataclass(frozen=True, slots=True)
class Models:
    """Every model the stage calls, as the protocols of `bindings.protocols`."""

    separators: tuple[Separator, ...]
    embedders: tuple[SpeakerEmbedder, ...]
    diarizer: Diarizer
    vad: Vad
    quality: QualityEstimator
    arousal: ArousalMonitor
    prosody: Callable[[np.ndarray, int], Prosody] = measure_prosody


def load_models(device: str, progress: Callable[[int, int], None] | None = None) -> Models:
    """Load the extract family's models from T11's bindings, on `device`.

    Each binding module exposes `load(device)`, imported here and not at the top so that a
    module walk imports no model package. Separators and embedders return a sequence.
    `progress` goes to the separators, whose TIGER-DnR calls it after each window.
    """
    from ..bindings import (  # type: ignore[attr-defined]
        arousal,
        diarizers,
        embedders,
        quality,
        separators,
        vad,
    )

    return Models(
        separators=tuple(separators.load(device, progress)),
        embedders=tuple(embedders.load(device)),
        diarizer=diarizers.load(device),
        vad=vad.load(device),
        quality=quality.load(device),
        arousal=arousal.load(device),
    )


# Where the run is, for an out-of-memory result.


@dataclass(frozen=True, slots=True)
class Where:
    """The place an out-of-memory result names.

    `step` is a separator's id, or `load`, `diarize`, `embed`, `vad`, `quality` or `retrieve`;
    `clip_s` is the clip's length for a call on a cut, and `None` for a whole channel.
    """

    source: str | None = None
    channel: str | None = None
    step: str | None = None
    clip_s: float | None = None


class Tracker:
    """Follows the run through its sources, channels and steps, reads the card, and logs progress.

    A block entered with `at` leaves `where` as it was when the block raised, so the handler of
    an out-of-memory error reads the place it happened. `vram_module` is `stages.vram`, or a fake
    of it in tests; on the CPU its readings are `None` and the lines leave the card out.

    The progress lines carry ids, counts, durations and MiB only. A whole-channel step (a
    separator, the diarizer, VAD, quality) logs a line when it returns; embeddings are many short
    calls, so their count and time are summed per source and channel and logged when the source
    changes, before the next step's line, or at `flush`.
    """

    def __init__(
        self,
        device: str = "cpu",
        vram_module: object = None,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        self.device = device
        self.vram = vram if vram_module is None else vram_module
        self.clock = time.monotonic if clock_fn is None else clock_fn
        self.where = Where()
        self._embeds: dict[tuple[str | None, str | None], list[float]] = {}
        self._windows_begun = self._windows_logged = 0.0

    @contextlib.contextmanager
    def at(self, **changes: object) -> Iterator[None]:
        """Set fields of `where` for the block, and restore them when it ends without an error."""
        saved = self.where
        moves = "source" in changes and changes["source"] != saved.source
        if moves:
            self.flush()
        self.where = replace(saved, **changes)  # type: ignore[arg-type]
        yield
        if moves:
            self.flush()
        self.where = saved

    @contextlib.contextmanager
    def step(self, label: str, **changes: object) -> Iterator[None]:
        """Run one whole-channel call under `changes`, then log its time and the card."""
        with self.at(**changes):
            self.flush()
            begun = self._windows_begun = self._windows_logged = self.clock()
            yield
            elapsed = self.clock() - begun
            print(f"extract: {self._place()} {label}: {elapsed:.1f} s{self._card()}")

    @contextlib.contextmanager
    def embed_call(self, clip_s: float) -> Iterator[None]:
        """Run one embedding call and add it to its source and channel's totals."""
        with self.at(step=STEP_EMBED, clip_s=clip_s):
            begun = self.clock()
            yield
            key = (self.where.source, self.where.channel)
            totals = self._embeds.setdefault(key, [0, 0.0])
            totals[0] += 1
            totals[1] += self.clock() - begun

    def flush(self) -> None:
        """Log the embedding totals not yet logged."""
        for (source, channel), (calls, seconds) in self._embeds.items():
            place = _place(source, channel)
            print(f"extract: {place} embed: {calls:.0f} calls, {seconds:.1f} s{self._card()}")
        self._embeds.clear()

    def windows(self, done: int, total: int) -> None:
        """A separator's progress callback: log the windows done at least once a minute."""
        now = self.clock()
        if now - self._windows_logged < WINDOW_REPORT_S:
            return
        self._windows_logged = now
        rate = (now - self._windows_begun) / done
        free = self.vram.free_mib(self.device)  # type: ignore[attr-defined]
        card = "" if free is None else f", free {free} MiB"
        print(
            f"extract: {self._place()} {self.where.step} windows {done} of {total}, "
            f"{rate:.2f} s per window{card}"
        )

    def loaded(self, begun: float) -> None:
        """Log that the models are loaded, with the time since `begun` and the card."""
        print(f"extract: models loaded in {self.clock() - begun:.1f} s{self._card()}")

    def source_start(self, index: int, count: int, source: str, channel: str, s: float) -> None:
        """Log the start of a source's first loop, with its length."""
        self.flush()
        print(f"extract: source {index} of {count}: {source} {channel}, {s / 60:.1f} min")

    def reset_peak(self) -> None:
        """Start the card's peak readings again."""
        self.vram.reset_peak(self.device)  # type: ignore[attr-defined]

    def peak_mib(self) -> dict[str, int] | None:
        """The reserved and allocated peaks since the last reset; `None` on the CPU."""
        return self.vram.peak_mib(self.device)  # type: ignore[attr-defined,no-any-return]

    def _place(self) -> str:
        return _place(self.where.source, self.where.channel)

    def _card(self) -> str:
        peak = self.peak_mib()
        free = self.vram.free_mib(self.device)  # type: ignore[attr-defined]
        if peak is None or free is None:
            return ""
        return f", peak reserved {peak['reserved']} MiB, free {free} MiB"


def _place(source: str | None, channel: str | None) -> str:
    # Before the first source, the only calls are the reference clips' embeddings.
    return " ".join(part for part in (source, channel) if part) or "reference"


def _clip_s(audio: np.ndarray, sample_rate: int) -> float:
    return round(len(audio) / sample_rate, 3)


class _TrackedSeparator:
    def __init__(self, inner: Separator, track: Tracker) -> None:
        self.name = inner.name
        self._inner = inner
        self._track = track

    def dialogue(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        with self._track.step(self.name, step=self.name, clip_s=None):
            return self._inner.dialogue(audio, sample_rate)


class _TrackedEmbedder:
    def __init__(self, inner: SpeakerEmbedder, track: Tracker) -> None:
        self.name = inner.name
        self._inner = inner
        self._track = track

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        with self._track.embed_call(_clip_s(audio, sample_rate)):
            return self._inner.embed(audio, sample_rate)


class _TrackedDiarizer:
    def __init__(self, inner: Diarizer, track: Tracker) -> None:
        self._inner = inner
        self._track = track

    def turns(self, audio: np.ndarray, sample_rate: int) -> list[Turn]:
        with self._track.step(STEP_DIARIZE, step=STEP_DIARIZE, clip_s=None):
            return self._inner.turns(audio, sample_rate)

    def overlaps(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        with self._track.step("overlaps", step=STEP_DIARIZE, clip_s=None):
            return self._inner.overlaps(audio, sample_rate)


class _TrackedVad:
    def __init__(self, inner: Vad, track: Tracker) -> None:
        self._inner = inner
        self._track = track

    def speech(self, audio: np.ndarray, sample_rate: int) -> list[Span]:
        with self._track.step(STEP_VAD, step=STEP_VAD, clip_s=None):
            return self._inner.speech(audio, sample_rate)


class _TrackedQuality:
    def __init__(self, inner: QualityEstimator, track: Tracker) -> None:
        self._inner = inner
        self._track = track

    def frames(self, audio: np.ndarray, sample_rate: int) -> QualityTrack:
        with self._track.step(STEP_QUALITY, step=STEP_QUALITY, clip_s=None):
            return self._inner.frames(audio, sample_rate)


class _TrackedArousal:
    # Arousal runs on kept cuts during the retrieval step, which stays the step named.
    def __init__(self, inner: ArousalMonitor, track: Tracker) -> None:
        self._inner = inner
        self._track = track

    def arousal(self, audio: np.ndarray, sample_rate: int) -> float:
        with self._track.at(clip_s=_clip_s(audio, sample_rate)):
            return self._inner.arousal(audio, sample_rate)


def tracked(models: Models, track: Tracker) -> Models:
    """Return `models` with every model call recorded in `track` as its step."""
    return Models(
        separators=tuple(_TrackedSeparator(s, track) for s in models.separators),
        embedders=tuple(_TrackedEmbedder(e, track) for e in models.embedders),
        diarizer=_TrackedDiarizer(models.diarizer, track),
        vad=_TrackedVad(models.vad, track),
        quality=_TrackedQuality(models.quality, track),
        arousal=_TrackedArousal(models.arousal, track),
        prosody=models.prosody,
    )


@dataclass(frozen=True, slots=True)
class Limits:
    """The gate values the run uses, from the plan's effective thresholds.

    The anchor check, the anchor floor and the margin hold one value per embedder name, since
    the embedders' cosines sit on different scales.
    """

    anchor_min_s: float
    anchor_max_s: float
    anchor_sim_min: dict[str, float]
    anchor_floor: dict[str, float]
    margin: dict[str, float]
    loo_percentile: float
    min_s: float
    max_s: float
    long_min_s: float
    long_max_s: float
    overlap_max_ms: float
    snr_min_db: float
    c50_min_db: float
    window_s: float
    window_percentile: float
    purity_percentile: float

    @classmethod
    def from_thresholds(cls, values: Mapping[str, float], embedders: Sequence[str]) -> Limits:
        """Read the gate keys the stage uses; per-embedder gates as `<key>.<embedder name>`.

        Args:
            values: The plan's effective thresholds.
            embedders: The names of the run's embedders.

        Raises:
            StageError: A key is missing; the message names every missing key.
        """
        # Each prefix takes an embedder name: `retrieval.margin.ecapa`.
        per_embedder = {
            "anchor_sim_min": "anchors.sim_min.",
            "anchor_floor": "retrieval.anchor_floor.",
            "margin": "retrieval.margin.",
        }
        keys = {
            "anchor_min_s": "anchors.min_s",
            "anchor_max_s": "anchors.max_s",
            "loo_percentile": "retrieval.loo_percentile",
            "min_s": "segment.min_s",
            "max_s": "segment.max_s",
            "long_min_s": "segment.long_min_s",
            "long_max_s": "segment.long_max_s",
            "overlap_max_ms": "segment.overlap_max_ms",
            "snr_min_db": "segment.snr_min_db",
            "c50_min_db": "segment.c50_min_db",
            "window_s": "segment.window_s",
            "window_percentile": "segment.window_percentile",
            "purity_percentile": "seed.purity_percentile",
        }
        wanted = list(keys.values())
        wanted += [f"{prefix}{e}" for prefix in per_embedder.values() for e in embedders]
        missing = [gate for gate in wanted if gate not in values]
        if missing:
            raise StageError(f"the plan's thresholds lack {', '.join(missing)}")
        scalars = {name: float(values[gate]) for name, gate in keys.items()}
        per = {
            name: {e: float(values[f"{prefix}{e}"]) for e in embedders}
            for name, prefix in per_embedder.items()
        }
        return cls(**scalars, **per)


@dataclass(frozen=True, slots=True)
class Source:
    """One source of the plan: its id, track, decode facts and the facts the report copies."""

    source_id: str
    track: str
    codec: str
    primary_channel: str
    sample_rate: int
    facts: dict[str, object]


@dataclass
class SourceWork:
    """What the first loop learns about one source, kept for the retrieval passes."""

    source: Source
    separator: str
    # The scores that decided, and the 50-best-window scores (the same when `basis` is windows).
    scores: dict[str, float | None]
    window_scores: dict[str, float | None]
    provisional: str
    basis: str
    spans: int
    rediarized: bool
    scratch: dict[str, Path]
    turns: list[Turn]
    turn_embs: list[retrieve.Embedding | None]
    overlaps: list[Span]
    speech: dict[str, list[Span]]
    quality: dict[str, QualityTrack]
    offcentre: list[Span]
    offcentre_embs: list[retrieve.Embedding]
    roles: dict[str, retrieve.ClusterRole] = field(default_factory=dict)
    rivals: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    # One per SDH anchor span of the source, in order; `None` when the cut is too short to embed.
    sdh_embs: list[retrieve.Embedding | None] = field(default_factory=list)
    # The card's reserved and allocated peaks over the source's first loop; `None` on the CPU.
    vram_peak_mib: dict[str, int] | None = None

    @property
    def channels(self) -> tuple[str, ...]:
        """The channels decoded: the primary one, and the downmix too when the primary is FC."""
        return tuple(self.scratch)

    def labelled(self) -> list[tuple[str, retrieve.Embedding]]:
        """Each embedded turn's diarizer label and embedding."""
        return [
            (t.speaker, e) for t, e in zip(self.turns, self.turn_embs, strict=True) if e is not None
        ]


@dataclass
class Draft:
    """A segment record before its id: every field but `segment_id` and the state."""

    source_id: str
    span: Span
    channel: str
    channel_reason: str | None
    separator: str
    diar_cluster: str | None
    cluster: str
    flagged_turn: bool
    sim: dict[str, float]
    overlap_ms: int
    snr_db: float
    c50_db: float
    set: str
    kept: bool
    reasons: tuple[str, ...]
    audio: np.ndarray | None = None
    sample_rate: int = 0
    margin: float | None = None
    prosody: Prosody | None = None
    arousal: float = 0.0
    failing_retrieval: bool = False


# Audio in and out.


def decode(data: Path, source: Source, channel: str) -> np.ndarray:
    """Decode a copied track to mono float32 at its own rate, DRC off (`ingest.decode_argv`).

    ffmpeg's stderr is discarded, as in ingest, and a failure names the source id only.
    """
    argv = decode_argv(str(data / source.track), source.codec, channel)
    proc = subprocess.run(argv, capture_output=True, check=False)
    if proc.returncode != 0:
        raise StageError(f"{source.source_id}: decode of the {channel} channel failed")
    audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if audio.size == 0:
        raise StageError(f"{source.source_id}: the {channel} channel decoded to nothing")
    return audio


def cut(audio: np.ndarray, sample_rate: int, span: Span) -> np.ndarray:
    """Return the samples of `span`."""
    start = max(0, int(round(span.start_s * sample_rate)))
    end = min(len(audio), int(round(span.end_s * sample_rate)))
    return audio[start:end]


def embed(embedders: Sequence[SpeakerEmbedder], audio: np.ndarray, rate: int) -> retrieve.Embedding:
    """Return the clip's embedding on every embedder."""
    return {e.name: np.asarray(e.embed(audio, rate), dtype=np.float64) for e in embedders}


def _write_audio(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile

    part = path.with_name(path.name + ".part")
    soundfile.write(str(part), audio, rate, format="FLAC", subtype=AUDIO_SUBTYPE)
    os.replace(part, path)


def _read_clip(path: Path) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _write_text(path: Path, text: str) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


# The plan and the anchors.


def _read_plan(data: Path, run_id: str) -> dict[str, object]:
    path = data / contract.run_rel(run_id) / contract.PLAN_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the extract plan is missing or not JSON") from exc
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise StageError("the extract plan is not a schema 1 plan")
    for key in ("sources", "anchors"):
        if not isinstance(plan.get(key), list):
            raise StageError(f"the extract plan has no {key} list")
    if not isinstance(plan.get("thresholds"), dict):
        raise StageError("the extract plan has no thresholds")
    return plan


def _sources(plan: Mapping[str, object]) -> list[Source]:
    found = []
    for entry in plan["sources"]:  # type: ignore[attr-defined]
        try:
            found.append(
                Source(
                    source_id=str(entry["source_id"]),
                    track=str(entry["track"]),
                    codec=str(entry["codec"]),
                    primary_channel=str(entry["primary_channel"]),
                    sample_rate=int(entry["sample_rate"]),
                    facts=dict(entry.get("facts") or {}),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError("the extract plan has an incomplete source") from exc
    if not found:
        raise StageError("the extract plan names no source")
    return found


def reference_clips(data: Path) -> list[Path]:
    """Return the solo clips in `seed/reference/`, sorted; they are named `clip-<n>` in output."""
    folder = data / contract.REFERENCE_REL
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in contract.CLIP_SUFFIXES
    )


def _timestamp_anchors(plan: Mapping[str, object], known: set[str]) -> list[tuple[str, Span]]:
    anchors = []
    for entry in plan["anchors"]:  # type: ignore[attr-defined]
        try:
            sid, start, end = (
                str(entry["source_id"]),
                float(entry["start_s"]),
                float(entry["end_s"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StageError("the extract plan has an incomplete anchor") from exc
        if sid not in known:
            raise StageError(f"an anchor names {sid}, which the plan does not list")
        anchors.append((sid, Span(start, end)))
    return anchors


# Separator choice.


def window_embeddings(
    audio: np.ndarray,
    rate: int,
    speech: Sequence[Span],
    embedders: Sequence[SpeakerEmbedder],
    window_s: float = CHOICE_WINDOW_S,
) -> list[retrieve.Embedding]:
    """Embed every whole `window_s` window tiled over the VAD speech spans."""
    return [
        embed(embedders, cut(audio, rate, tile), rate)
        for span in speech
        for tile in segments.tiles(span, window_s)
    ]


def separate(
    audio: np.ndarray, rate: int, separators: Sequence[Separator]
) -> dict[str, np.ndarray]:
    """Return every candidate's output, `none` first."""
    outputs = {NONE: audio}
    for separator in separators:
        if separator.name == NONE:
            raise StageError("a separator binding may not be named none")
        outputs[separator.name] = np.asarray(separator.dialogue(audio, rate), dtype=np.float32)
    return outputs


def candidate_scores(
    outputs: Mapping[str, np.ndarray],
    rate: int,
    models: Models,
    references: Mapping[str, Mapping[str, np.ndarray] | None],
) -> dict[str, float | None]:
    """Score each candidate's output against its reference: the solo clips or its own anchors.

    A candidate with no reference (no solo clip and no anchor of its own) scores `None`.
    """
    scores: dict[str, float | None] = {}
    for name, output in outputs.items():
        reference = references.get(name)
        if reference is None:
            scores[name] = None
            continue
        speech = models.vad.speech(output, rate)
        wins = window_embeddings(output, rate, speech, models.embedders)
        scores[name] = retrieve.candidate_score(wins, reference, CHOICE_BEST_N)
    return scores


def span_scores(
    outputs: Mapping[str, np.ndarray],
    rate: int,
    models: Models,
    references: Mapping[str, Mapping[str, np.ndarray] | None],
    spans: Sequence[Span],
) -> dict[str, float | None]:
    """Score each candidate on its own cuts of `spans`: the median SIM over every span.

    `best_n` is the span count, so no span is left out. A candidate with no reference, or no
    span to cut, scores `None`.
    """
    scores: dict[str, float | None] = {}
    for name, output in outputs.items():
        reference = references.get(name)
        if reference is None:
            scores[name] = None
            continue
        embs = [embed(models.embedders, cut(output, rate, span), rate) for span in spans]
        scores[name] = retrieve.candidate_score(embs, reference, len(spans))
    return scores


@dataclass(frozen=True, slots=True)
class Diarized:
    """One output's diarization: its turns by start, its overlaps, and each turn's embedding.

    A turn shorter than `MIN_EMBED_S` has no embedding.
    """

    turns: list[Turn]
    overlaps: list[Span]
    turn_embs: list[retrieve.Embedding | None]

    def labelled(self) -> list[tuple[str, retrieve.Embedding]]:
        """Each embedded turn's diarizer label and embedding."""
        return [
            (t.speaker, e) for t, e in zip(self.turns, self.turn_embs, strict=True) if e is not None
        ]


def diarize(audio: np.ndarray, rate: int, models: Models) -> Diarized:
    """Diarize one output and embed each turn long enough to embed."""
    turns = sorted(models.diarizer.turns(audio, rate), key=lambda t: t.start_s)
    overlaps = list(models.diarizer.overlaps(audio, rate))
    turn_embs = [
        (
            embed(models.embedders, cut(audio, rate, Span(t.start_s, t.end_s)), rate)
            if t.end_s - t.start_s >= MIN_EMBED_S
            else None
        )
        for t in turns
    ]
    return Diarized(turns=turns, overlaps=overlaps, turn_embs=turn_embs)


@dataclass(frozen=True, slots=True)
class SeparatorChoice:
    """The separator a source keeps, how it was chosen, and the chosen output's diarization.

    `basis` is `turns` when the candidates were scored on the provisional target cluster's
    spans and `windows` when the window scores decided; `scores` are the scores that decided.
    """

    separator: str
    provisional: str
    basis: str
    scores: dict[str, float | None]
    window_scores: dict[str, float | None]
    spans: int
    rediarized: bool
    diarized: Diarized


def separator_choice(
    outputs: Mapping[str, np.ndarray],
    rate: int,
    models: Models,
    references: Mapping[str, Mapping[str, np.ndarray] | None],
) -> SeparatorChoice:
    """Choose a source's separator on the provisional target cluster's turns.

    The window scores give a provisional choice, whose output is diarized. Its target cluster,
    ranked against its reference with no floor (`retrieve.best_cluster`), gives the spans: that
    cluster's embedded turns that touch no overlap. The best 50 windows can be another speaker's
    when the target speaks little, while these spans are the target's wherever the provisional
    cluster is. Each candidate is scored on its own cuts of the spans; the provisional
    candidate's cuts are its turn embeddings already. With no span, or no reference for the
    provisional candidate, the window scores decide. The chosen output is diarized again only
    when the choice differs from the provisional one.
    """
    window_scores = candidate_scores(outputs, rate, models, references)
    provisional = retrieve.choose_separator(window_scores)
    diarized = diarize(outputs[provisional], rate, models)
    reference = references.get(provisional)
    picked: list[tuple[Span, retrieve.Embedding]] = []
    if reference is not None:
        label = retrieve.best_cluster(diarized.labelled(), reference)
        for turn, emb in zip(diarized.turns, diarized.turn_embs, strict=True):
            span = Span(turn.start_s, turn.end_s)
            if (
                emb is not None
                and turn.speaker == label
                and not segments.intersects(span, diarized.overlaps)
            ):
                picked.append((span, emb))
    if reference is None or not picked:
        return SeparatorChoice(
            separator=provisional,
            provisional=provisional,
            basis=BASIS_WINDOWS,
            scores=window_scores,
            window_scores=window_scores,
            spans=0,
            rediarized=False,
            diarized=diarized,
        )
    spans = [span for span, _ in picked]
    others = {name: output for name, output in outputs.items() if name != provisional}
    found = span_scores(others, rate, models, references, spans)
    own = retrieve.candidate_score([emb for _, emb in picked], reference, len(spans))
    scores = {name: own if name == provisional else found[name] for name in outputs}
    chosen = retrieve.choose_separator(scores)
    rediarized = chosen != provisional
    return SeparatorChoice(
        separator=chosen,
        provisional=provisional,
        basis=BASIS_TURNS,
        scores=scores,
        window_scores=window_scores,
        spans=len(spans),
        rediarized=rediarized,
        diarized=diarize(outputs[chosen], rate, models) if rediarized else diarized,
    )


# The stage.


class Extraction:
    """One run of the stage over the plan's sources.

    `track` follows the run for an out-of-memory result; every model call goes through it.
    `started` is the card at the job's start, `None` on the CPU.
    """

    def __init__(
        self,
        data: Path,
        run_id: str,
        models: Models,
        track: Tracker | None = None,
        started: vram.VramStart | None = None,
    ) -> None:
        if not contract.RUN_ID.fullmatch(run_id) or ".." in run_id:
            raise StageError("a run id is one folder name under seed/runs")
        self.data = data
        self.run_id = run_id
        self.run_dir = data / contract.run_rel(run_id)
        self.track = Tracker() if track is None else track
        self.started = started
        models = tracked(models, self.track)
        self.models = models
        self.plan = _read_plan(data, run_id)
        self.thresholds = {str(k): float(v) for k, v in self.plan["thresholds"].items()}  # type: ignore[attr-defined]
        self.limits = Limits.from_thresholds(self.thresholds, [e.name for e in models.embedders])
        self.sources = _sources(self.plan)
        self.by_id = {s.source_id: s for s in self.sources}
        self.anchors = _timestamp_anchors(self.plan, set(self.by_id))
        try:
            self.sdh = sdh_stage.read_spans(
                data, self.by_id, self.limits.anchor_min_s, self.limits.anchor_max_s
            )
        except sdh_stage.StageError as exc:
            raise StageError(str(exc)) from exc
        self.clips = reference_clips(data)
        self.separators = {s.name: s for s in models.separators}
        self.rules = propose_states.load_rules()

    def _separator(self, name: str) -> Separator:
        return Unseparated() if name == NONE else self.separators[name]

    def _embed(self, audio: np.ndarray, rate: int) -> retrieve.Embedding:
        return embed(self.models.embedders, audio, rate)

    def _check_lengths(self, clips: Sequence[tuple[np.ndarray, int]]) -> list[str]:
        low, high = self.limits.anchor_min_s, self.limits.anchor_max_s
        bad = [f"clip-{i}" for i, (a, r) in enumerate(clips, 1) if not low <= len(a) / r <= high]
        for i, (_sid, span) in enumerate(self.anchors, 1):
            if not low <= span.end_s - span.start_s <= high:
                bad.append(f"time-{i}")
        return bad

    def _anchor_cuts(self, sid: str, audio: np.ndarray, rate: int) -> dict[int, retrieve.Embedding]:
        return {
            i: self._embed(cut(audio, rate, span), rate)
            for i, (anchor_sid, span) in enumerate(self.anchors)
            if anchor_sid == sid
        }

    def _candidate_references(
        self, clip_centroid: dict[str, np.ndarray] | None
    ) -> dict[str, dict[str, np.ndarray] | None]:
        names = [NONE, *self.separators]
        if clip_centroid is not None:
            return {name: clip_centroid for name in names}
        # No solo clip: each candidate is scored against its own timestamp anchors, cut from its
        # own output, so every anchor's source is separated by every candidate first.
        own: dict[str, list[retrieve.Embedding]] = {name: [] for name in names}
        for sid in sorted({sid for sid, _ in self.anchors}):
            source = self.by_id[sid]
            with self.track.at(source=sid, channel=source.primary_channel):
                primary = decode(self.data, source, source.primary_channel)
                for name, output in separate(
                    primary, source.sample_rate, self.models.separators
                ).items():
                    own[name].extend(self._anchor_cuts(sid, output, source.sample_rate).values())
        return {name: retrieve.centroids(embs) if embs else None for name, embs in own.items()}

    def _first_loop(
        self,
        source: Source,
        references: Mapping[str, Mapping[str, np.ndarray] | None],
        scratch: Path,
        position: tuple[int, int] = (1, 1),
    ) -> tuple[SourceWork, dict[int, retrieve.Embedding]]:
        rate = source.sample_rate
        primary = decode(self.data, source, source.primary_channel)
        self.track.source_start(
            *position, source.source_id, source.primary_channel, len(primary) / rate
        )
        outputs = separate(primary, rate, self.models.separators)
        choice = separator_choice(outputs, rate, self.models, references)
        chosen = choice.separator
        channels = {source.primary_channel: outputs[chosen]}
        del outputs, primary
        if source.primary_channel == FC:
            with self.track.at(channel=DOWNMIX):
                downmix = decode(self.data, source, DOWNMIX)
                channels[DOWNMIX] = np.asarray(
                    self._separator(chosen).dialogue(downmix, rate), dtype=np.float32
                )
        main = channels[source.primary_channel]
        anchors = self._anchor_cuts(source.source_id, main, rate)
        sdh_embs = self._sdh_cuts(source.source_id, main, rate)
        turns, overlaps = choice.diarized.turns, choice.diarized.overlaps
        turn_embs = choice.diarized.turn_embs
        speech: dict[str, list[Span]] = {}
        quality: dict[str, QualityTrack] = {}
        for name, audio in channels.items():
            with self.track.at(channel=name):
                speech[name] = list(self.models.vad.speech(audio, rate))
        for name, audio in channels.items():
            with self.track.at(channel=name):
                quality[name] = self.models.quality.frames(audio, rate)
        offcentre: list[Span] = []
        if DOWNMIX in channels and source.primary_channel == FC:
            offcentre = [
                s
                for s in segments.subtract(speech[DOWNMIX], speech[FC])
                if s.end_s - s.start_s >= self.limits.min_s
            ]
        with self.track.at(channel=DOWNMIX):
            offcentre_embs = [self._embed(cut(channels[DOWNMIX], rate, s), rate) for s in offcentre]
        paths = {}
        for name, audio in channels.items():
            paths[name] = scratch / f"{source.source_id}-{name}.npy"
            np.save(paths[name], audio)
        again = "diarized again" if choice.rediarized else "diarized once"
        print(
            f"extract: {source.source_id} separator {chosen} (provisional {choice.provisional}, "
            f"basis {choice.basis}, {choice.spans} turns scored, {again}), {len(turns)} turns, "
            f"{len(overlaps)} overlaps, {len(offcentre)} off-centre spans"
        )
        work = SourceWork(
            source=source,
            separator=chosen,
            scores=choice.scores,
            window_scores=choice.window_scores,
            provisional=choice.provisional,
            basis=choice.basis,
            spans=choice.spans,
            rediarized=choice.rediarized,
            scratch=paths,
            turns=turns,
            turn_embs=turn_embs,
            overlaps=overlaps,
            speech=speech,
            quality=quality,
            offcentre=offcentre,
            offcentre_embs=offcentre_embs,
            sdh_embs=sdh_embs,
        )
        return work, anchors

    def _sdh_cuts(self, sid: str, audio: np.ndarray, rate: int) -> list[retrieve.Embedding | None]:
        # Cut from the chosen separator's output, as Raj's timestamp anchors are. A span that
        # runs past the track's end yields a short cut, which is dropped rather than embedded.
        spans = self.sdh.get(sid)
        if spans is None:
            return []
        found: list[retrieve.Embedding | None] = []
        for span in spans.anchors:
            clip = cut(audio, rate, Span(span.start_s, span.end_s))
            long_enough = len(clip) >= self.limits.anchor_min_s * rate - 1
            found.append(self._embed(clip, rate) if long_enough else None)
        return found

    def _sdh_anchors(
        self, works: Sequence[SourceWork], anchors: Sequence[retrieve.Embedding]
    ) -> tuple[dict[str, np.ndarray], dict[str, int] | None]:
        """Return the anchor centroid retrieval uses, and the SDH counts for the report.

        Each SDH anchor's SIM is taken to the centroid of Raj's anchors alone, his ground truth;
        one below that embedder's Anchors value on either embedder is dropped.
        """
        own = retrieve.centroids(anchors)
        if not self.sdh:
            return own, None
        sim_min = self.limits.anchor_sim_min

        def reaches(emb: retrieve.Embedding) -> bool:
            found = retrieve.sims(emb, own)
            return all(v >= sim_min[name] - retrieve.EPS for name, v in found.items())

        joined: list[retrieve.Embedding] = []
        dropped = dropped_offscreen = 0
        for work in works:
            spans = self.sdh.get(work.source.source_id)
            if spans is None:
                continue
            for span, emb in zip(spans.anchors, work.sdh_embs, strict=True):
                if emb is not None and reaches(emb):
                    joined.append(emb)
                else:
                    dropped += 1
                    dropped_offscreen += span.offscreen
        every = list(self.sdh.values())

        def total(name: str) -> int:
            return sum(len(getattr(s, name)) for s in every)

        def offscreen(name: str) -> int:
            return sum(span.offscreen for s in every for span in getattr(s, name))

        counts = {
            "sources": len(every),
            "found": sum(s.found for s in every),
            "found_offscreen": sum(s.found_offscreen for s in every),
            "multi": sum(s.multi for s in every),
            "merged": total("merged"),
            "merged_offscreen": offscreen("merged"),
            "kept": total("kept"),
            "kept_offscreen": offscreen("kept"),
            "held_out": total("held_out"),
            "held_out_offscreen": offscreen("held_out"),
            "anchors": total("anchors"),
            "dropped": dropped,
            "dropped_offscreen": dropped_offscreen,
            "joined": len(joined),
        }
        print(f"extract: sdh anchors {len(joined)} joined, {dropped} dropped")
        return retrieve.centroids([*anchors, *joined]), counts

    def _load(self, work: SourceWork) -> dict[str, np.ndarray]:
        return {name: np.load(path, mmap_mode="r") for name, path in work.scratch.items()}

    def _eligible(self, work: SourceWork) -> list[tuple[Turn, retrieve.Embedding]]:
        # Primary turns that touch no overlap define the target: an overlapped turn's embedding
        # holds two voices.
        return [
            (t, e)
            for t, e in zip(work.turns, work.turn_embs, strict=True)
            if e is not None
            and work.roles[t.speaker].role == retrieve.PRIMARY
            and not segments.intersects(Span(t.start_s, t.end_s), work.overlaps)
        ]

    def _window_model(self, works: Sequence[SourceWork]) -> retrieve.TargetModel | None:
        embs: list[retrieve.Embedding] = []
        sources: list[str] = []
        for work in works:
            rate = work.source.sample_rate
            main = self._load(work)[work.source.primary_channel]
            with self.track.at(source=work.source.source_id, channel=work.source.primary_channel):
                for turn, _ in self._eligible(work):
                    span = Span(turn.start_s, turn.end_s)
                    for tile in segments.tiles(span, self.limits.window_s):
                        embs.append(self._embed(np.asarray(cut(main, rate, tile)), rate))
                        sources.append(work.source.source_id)
        return retrieve.target_model(embs, sources, self.limits.window_percentile)

    def run(self) -> list[str]:
        """Run the stage; return the root-relative files written."""
        written: list[str] = []
        clips = [_read_clip(p) for p in self.clips]
        if not clips and not self.anchors:
            return self._result("no-anchor", {}, written)
        bad = self._check_lengths(clips)
        if bad:
            return self._result("anchor-length", {"anchors": bad}, written)
        clip_embs = [self._embed(a, r) for a, r in clips]
        references = self._candidate_references(
            retrieve.centroids(clip_embs) if clip_embs else None
        )
        with tempfile.TemporaryDirectory(dir=self.run_dir, prefix=".work-") as tmp:
            works = []
            anchor_embs: dict[int, retrieve.Embedding] = {}
            for index, source in enumerate(self.sources, 1):
                self.track.reset_peak()
                position = (index, len(self.sources))
                with self.track.at(source=source.source_id, channel=source.primary_channel):
                    work, cuts = self._first_loop(source, references, Path(tmp), position)
                work.vram_peak_mib = self.track.peak_mib()
                works.append(work)
                anchor_embs.update(cuts)
            anchors = clip_embs + [anchor_embs[i] for i in sorted(anchor_embs)]
            agree, anchor_sims = retrieve.anchor_check(anchors, self.limits.anchor_sim_min)
            detail: dict[str, object] = {
                "anchors": self._anchor_rows(anchor_sims),
                "anchors_agree": agree,
            }
            centroid, sdh_counts = self._sdh_anchors(works, anchors)
            if sdh_counts is not None:
                detail["sdh"] = sdh_counts
            return self._retrieve(works, centroid, detail, written)

    def _anchor_rows(self, sims: Sequence[Mapping[str, float]]) -> list[dict[str, object]]:
        names = [f"clip-{i}" for i in range(1, len(self.clips) + 1)]
        names += [f"time-{i}" for i in range(1, len(self.anchors) + 1)]
        return [
            {"anchor": name, "sim": {k: round(v, 4) for k, v in row.items()}}
            for name, row in zip(names, sims, strict=True)
        ]

    def _result(self, status: str, detail: Mapping[str, object], written: list[str]) -> list[str]:
        return [*written, write_result(self.data, self.run_id, status, detail)]

    def _retrieve(
        self,
        works: list[SourceWork],
        anchor_centroid: dict[str, np.ndarray],
        detail: dict[str, object],
        written: list[str],
    ) -> list[str]:
        for work in works:
            work.roles = retrieve.cluster_roles(
                work.labelled(), anchor_centroid, self.limits.anchor_floor
            )
            work.rivals = retrieve.rival_centroids(work.labelled(), work.roles)
        eligible = [(work.source.source_id, e) for work in works for _, e in self._eligible(work)]
        model = retrieve.target_model(
            [e for _, e in eligible], [sid for sid, _ in eligible], self.limits.loo_percentile
        )
        audio_dir = self.run_dir / contract.AUDIO_DIR
        audio_dir.mkdir(exist_ok=True)
        drafts: list[Draft] = []
        with self.track.at(step=STEP_RETRIEVE):
            windows = self._window_model(works) if model is not None else None
            for work in works:
                with self.track.at(
                    source=work.source.source_id, channel=work.source.primary_channel
                ):
                    source_drafts = self._source_drafts(work, model, windows)
                drafts.extend(sorted(source_drafts, key=lambda d: (d.span.start_s, d.span.end_s)))
        records, proposals = self._records(drafts, audio_dir, written)
        report = self._report(works, records, drafts, model, windows, detail)
        turns = turn_records(works, anchor_centroid, model, self.limits.margin)
        run_rel = contract.run_rel(self.run_id)
        seg_rel = f"{run_rel}/{contract.SEGMENTS_NAME}"
        _write_text(self.data / seg_rel, "".join(json.dumps(r.to_json()) + "\n" for r in records))
        turns_rel = f"{run_rel}/{contract.TURNS_NAME}"
        _write_text(self.data / turns_rel, "".join(json.dumps(t.to_json()) + "\n" for t in turns))
        prop_rel = f"{run_rel}/{contract.PROPOSALS_NAME}"
        _write_text(self.data / prop_rel, "".join(json.dumps(p) + "\n" for p in proposals))
        report_rel = f"{run_rel}/{contract.REPORT_NAME}"
        _write_json(self.data / report_rel, report)
        kept = sum(r.kept for r in records)
        self.track.flush()
        print(f"extract: {len(records)} segments, {kept} kept, {len(turns)} turns recorded")
        card = None if self.started is None else self.started.as_json()
        files = [*written, seg_rel, turns_rel, prop_rel, report_rel]
        return self._result("ok", {**detail, "vram": card}, files)

    # The second pass and the segment gates.

    def _window_ok(
        self, audio: np.ndarray, rate: int, span: Span, windows: retrieve.TargetModel
    ) -> bool:
        hop = self.limits.window_s / 2.0
        for win in segments.windows(span, self.limits.window_s, hop):
            sim = retrieve.sims(
                self._embed(np.asarray(cut(audio, rate, win)), rate), windows.centroid
            )
            if any(sim[name] < windows.floor[name] - retrieve.EPS for name in sim):
                return False
        return True

    def _gates(
        self,
        work: SourceWork,
        audio: Mapping[str, np.ndarray],
        channel: str,
        span: Span,
        windows: retrieve.TargetModel,
    ) -> tuple[list[str], float, float]:
        snr, c50 = segments.span_quality(work.quality[channel], span)
        with self.track.at(channel=channel):
            window_ok = self._window_ok(audio[channel], work.source.sample_rate, span, windows)
        reasons = segments.quality_reasons(
            snr, c50, window_ok, snr_min=self.limits.snr_min_db, c50_min=self.limits.c50_min_db
        )
        return reasons, snr, c50

    def _source_drafts(
        self,
        work: SourceWork,
        model: retrieve.TargetModel | None,
        windows: retrieve.TargetModel | None,
    ) -> list[Draft]:
        audio = self._load(work)
        drafts: list[Draft] = []
        primary = work.source.primary_channel
        for turn, emb in zip(work.turns, work.turn_embs, strict=True):
            # Turns of clusters not admitted are other speakers, not candidates, so they leave no
            # segment record (`turn_records` holds them); a turn too short to embed is a
            # candidate when its cluster is admitted.
            role = work.roles.get(turn.speaker)
            if role is None or role.role == retrieve.OTHER:
                continue
            cluster = role.role
            base = {
                "diar_cluster": turn.speaker,
                "cluster": cluster,
                "flagged_turn": cluster == retrieve.FLAGGED,
            }
            span = Span(turn.start_s, turn.end_s)
            drafts.extend(
                self._candidate(work, audio, span, emb, primary, None, base, model, windows)
            )
        for span, emb in zip(work.offcentre, work.offcentre_embs, strict=True):
            base = {"diar_cluster": None, "cluster": retrieve.FLAGGED, "flagged_turn": False}
            drafts.extend(
                self._candidate(work, audio, span, emb, DOWNMIX, "off-centre", base, model, windows)
            )
        return drafts

    def _draft(
        self,
        work: SourceWork,
        span: Span,
        channel: str,
        channel_reason: str | None,
        base: Mapping[str, object],
        **values: object,
    ) -> Draft:
        snr, c50 = segments.span_quality(work.quality[channel], span)
        fields_ = {
            "source_id": work.source.source_id,
            "span": span,
            "channel": channel,
            "channel_reason": channel_reason,
            "separator": work.separator,
            "overlap_ms": segments.overlap_ms(span, work.overlaps),
            "snr_db": snr,
            "c50_db": c50,
            "set": (
                segments.LONG if span.end_s - span.start_s > self.limits.max_s else segments.SHORT
            ),
            "kept": False,
            "sim": {},
            "reasons": (),
            **base,
            **values,
        }
        return Draft(**fields_)  # type: ignore[arg-type]

    def _candidate(
        self,
        work: SourceWork,
        audio: Mapping[str, np.ndarray],
        span: Span,
        emb: retrieve.Embedding | None,
        channel: str,
        channel_reason: str | None,
        base: Mapping[str, object],
        model: retrieve.TargetModel | None,
        windows: retrieve.TargetModel | None,
    ) -> list[Draft]:
        rate = work.source.sample_rate
        if emb is None:
            return [self._draft(work, span, channel, channel_reason, base, reasons=("length",))]
        if model is None or windows is None:
            return [self._draft(work, span, channel, channel_reason, base, reasons=("no-target",))]
        verdict = retrieve.judge(emb, model, work.rivals, self.limits.margin)
        if not verdict.passed and channel == FC and DOWNMIX in audio:
            with self.track.at(channel=DOWNMIX):
                downmix_emb = self._embed(np.asarray(cut(audio[DOWNMIX], rate, span)), rate)
            retry = retrieve.judge(downmix_emb, model, work.rivals, self.limits.margin)
            if retry.passed:
                verdict, channel, channel_reason = retry, DOWNMIX, "fc-failed:sim"
            else:
                reasons = verdict.reasons + tuple(f"downmix:{r}" for r in retry.reasons)
                return [
                    self._draft(
                        work, span, channel, channel_reason, base, sim=verdict.sim, reasons=reasons
                    )
                ]
        if not verdict.passed:
            return [
                self._draft(
                    work,
                    span,
                    channel,
                    channel_reason,
                    base,
                    sim=verdict.sim,
                    reasons=verdict.reasons,
                )
            ]
        drafts = []
        limits = self.limits
        cuts = segments.cut_line(
            span,
            work.speech[channel],
            min_s=limits.min_s,
            max_s=limits.max_s,
            long_min_s=limits.long_min_s,
            long_max_s=limits.long_max_s,
        )
        for piece in cuts:
            drafts.append(
                self._segment(
                    work, audio, piece, channel, channel_reason, base, verdict, model, windows
                )
            )
        return drafts

    def _segment(
        self,
        work: SourceWork,
        audio: Mapping[str, np.ndarray],
        piece: segments.Cut,
        channel: str,
        channel_reason: str | None,
        base: Mapping[str, object],
        verdict: retrieve.Verdict,
        model: retrieve.TargetModel,
        windows: retrieve.TargetModel,
    ) -> Draft:
        span = piece.span
        if piece.set is None:
            return self._draft(
                work, span, channel, channel_reason, base, sim=verdict.sim, reasons=piece.reasons
            )
        offcentre = channel_reason == "off-centre"
        if not offcentre and segments.overlap_ms(span, work.overlaps) > self.limits.overlap_max_ms:
            return self._draft(
                work, span, channel, channel_reason, base, sim=verdict.sim, reasons=("overlap",)
            )
        reasons, _, _ = self._gates(work, audio, channel, span, windows)
        if reasons and channel == FC and DOWNMIX in audio:
            retry, _, _ = self._gates(work, audio, DOWNMIX, span, windows)
            if not retry:
                channel, channel_reason, reasons = DOWNMIX, f"fc-failed:{reasons[0]}", []
            else:
                reasons = reasons + [f"downmix:{r}" for r in retry]
        if reasons:
            return self._draft(
                work, span, channel, channel_reason, base, sim=verdict.sim, reasons=tuple(reasons)
            )
        rate = work.source.sample_rate
        clip = np.array(cut(audio[channel], rate, span), dtype=np.float32)
        with self.track.at(channel=channel):
            own = retrieve.judge(self._embed(clip, rate), model, work.rivals, self.limits.margin)
            prosody = self.models.prosody(clip, rate)
            arousal = float(self.models.arousal.arousal(clip, rate))
        return self._draft(
            work,
            span,
            channel,
            channel_reason,
            base,
            sim=own.sim,
            kept=True,
            set=piece.set,
            audio=clip,
            sample_rate=rate,
            margin=own.margin,
            prosody=prosody,
            arousal=arousal,
            failing_retrieval=not own.passed,
        )

    # Records and the report.

    def _records(
        self, drafts: Sequence[Draft], audio_dir: Path, written: list[str]
    ) -> tuple[list[SegmentRecord], list[dict[str, object]]]:
        kept = [d for d in drafts if d.kept]
        proposals = propose_states.propose_all(
            [d.prosody for d in kept],  # type: ignore[misc]
            [d.arousal for d in kept],
            self.rules,
        )
        proposed = dict(zip((id(d) for d in kept), proposals, strict=True))
        records: list[SegmentRecord] = []
        rows: list[dict[str, object]] = []
        run_rel = contract.run_rel(self.run_id)
        for number, draft in enumerate(drafts, start=1):
            seg_id = f"seg-{number:06d}"
            audio_rel = ""
            state = ""
            if draft.kept and draft.audio is not None:
                audio_rel = f"{run_rel}/{contract.AUDIO_DIR}/{seg_id}.flac"
                _write_audio(self.data / audio_rel, draft.audio, draft.sample_rate)
                written.append(audio_rel)
                proposal = proposed[id(draft)]
                state = proposal.state
                rows.append(
                    {
                        "segment_id": seg_id,
                        "state_proposed": proposal.state,
                        "margin": round(proposal.margin, 4),
                        "features": {k: round(v, 4) for k, v in proposal.features.items()},
                    }
                )
            if draft.cluster == retrieve.FLAGGED:
                group = f"{draft.source_id}/{draft.diar_cluster}" if draft.flagged_turn else seg_id
            else:
                group = None
            records.append(
                SegmentRecord(
                    segment_id=seg_id,
                    source_id=draft.source_id,
                    start_s=round(draft.span.start_s, 3),
                    end_s=round(draft.span.end_s, 3),
                    channel=draft.channel,
                    channel_reason=draft.channel_reason,
                    separator=draft.separator,
                    audio=audio_rel,
                    diar_cluster=draft.diar_cluster,
                    cluster=draft.cluster,
                    flag_group=group,
                    sim={k: round(v, 4) for k, v in draft.sim.items()},
                    overlap_ms=draft.overlap_ms,
                    snr_db=round(draft.snr_db, 2),
                    c50_db=round(draft.c50_db, 2),
                    transcripts={},
                    asr_agreement_cer=0.0,
                    prosody=draft.prosody or Prosody(0.0, 0.0, 0.0, 0.0),
                    arousal=round(draft.arousal, 4),
                    state_proposed=state,
                    state=None,
                    set=draft.set,
                    kept=draft.kept,
                    reasons=draft.reasons,
                )
            )
        return records, rows

    def _report(
        self,
        works: Sequence[SourceWork],
        records: Sequence[SegmentRecord],
        drafts: Sequence[Draft],
        model: retrieve.TargetModel | None,
        windows: retrieve.TargetModel | None,
        detail: Mapping[str, object],
    ) -> dict[str, object]:
        def minutes(rows: Sequence[SegmentRecord]) -> float:
            return round(sum(r.end_s - r.start_s for r in rows) / 60.0, 4)

        seed = [r for r in records if r.kept and r.cluster == retrieve.PRIMARY]
        flagged = [r for r in records if r.kept and r.cluster == retrieve.FLAGGED]
        kept_drafts = [d for d in drafts if d.kept]
        margins = [d.margin for d in kept_drafts if d.margin is not None]
        drops: dict[str, int] = {}
        for record in records:
            for reason in record.reasons if not record.kept else ():
                drops[reason] = drops.get(reason, 0) + 1
        sources = []
        for work in works:
            sid = work.source.source_id
            own = [r for r in records if r.source_id == sid]
            seconds = cluster_seconds(work.turns)
            primary = [r.label for r in work.roles.values() if r.role == retrieve.PRIMARY]
            sources.append(
                {
                    "source_id": sid,
                    **work.source.facts,
                    "separator": work.separator,
                    "separator_scores": _scores(work.scores),
                    "separator_basis": work.basis,
                    "separator_window_scores": _scores(work.window_scores),
                    "separator_provisional": work.provisional,
                    "separator_turns": work.spans,
                    "separator_rediarized": work.rediarized,
                    "primary_s": seconds[primary[0]] if primary else 0.0,
                    "clusters": [
                        {
                            "diar_cluster": role.label,
                            "role": role.role,
                            "turns": role.turns,
                            "seconds": seconds[role.label],
                            "median_sim": {k: round(v, 4) for k, v in role.median_sim.items()},
                        }
                        for role in work.roles.values()
                    ],
                    "off_centre_spans": len(work.offcentre),
                    "kept": sum(r.kept and r.cluster == retrieve.PRIMARY for r in own),
                    "flagged": sum(r.kept and r.cluster == retrieve.FLAGGED for r in own),
                    "dropped": sum(not r.kept for r in own),
                    "vram_peak_mib": work.vram_peak_mib,
                }
            )
        return {
            "schema": SCHEMA,
            "run": self.run_id,
            "clean_minutes": minutes(seed),
            "minutes_per_state": {
                s: minutes([r for r in seed if r.state_proposed == s]) for s in SPEAKING_STATES
            },
            "minutes_per_channel": {
                c: minutes([r for r in seed if r.channel == c]) for c in (FC, DOWNMIX)
            },
            "minutes_per_set": {
                s: minutes([r for r in seed if r.set == s]) for s in (segments.SHORT, segments.LONG)
            },
            "flagged": {
                "segments": len(flagged),
                "groups": len({r.flag_group for r in flagged}),
                "minutes": minutes(flagged),
                "minutes_per_channel": {
                    c: minutes([r for r in flagged if r.channel == c]) for c in (FC, DOWNMIX)
                },
            },
            "channel_reasons": {
                reason: sum(r.kept and r.channel_reason == reason for r in records)
                for reason in sorted(
                    {str(r.channel_reason) for r in records if r.kept and r.channel_reason}
                )
            },
            "sources": sources,
            "one_speaker": {
                "kept": len(seed),
                "kept_failing_retrieval": sum(
                    d.failing_retrieval for d in kept_drafts if d.cluster == retrieve.PRIMARY
                ),
                "kept_failing_window": sum(
                    any(x.endswith("window") for x in r.reasons) for r in seed
                ),
                # Flagged segments stay out of the seed until Raj confirms them (T9), so none
                # in this run's seed is unconfirmed.
                "flagged_unconfirmed": 0,
            },
            "purity_p10": (
                round(float(np.percentile(margins, self.limits.purity_percentile)), 4)
                if margins
                else None
            ),
            "distribution": _distribution(model),
            "windows": _distribution(windows),
            "drops": dict(sorted(drops.items())),
            "anchors": {
                "clips": len(self.clips),
                "timestamps": len(self.anchors),
                "sim": detail.get("anchors", []),
                "agree": bool(detail["anchors_agree"]),
            },
            "sdh": detail.get("sdh"),
            "rehearsal": None,
            "thresholds": dict(sorted(self.thresholds.items())),
        }


def cluster_seconds(turns: Sequence[Turn]) -> dict[str, float]:
    """Return each diarizer label's seconds over all its turns, those too short to embed too."""
    seconds: dict[str, float] = {}
    for turn in turns:
        seconds[turn.speaker] = seconds.get(turn.speaker, 0.0) + (turn.end_s - turn.start_s)
    return {label: round(value, 3) for label, value in seconds.items()}


def _rounded(values: Mapping[str, float]) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in values.items()}


def _scores(values: Mapping[str, float | None]) -> dict[str, float | None]:
    return {name: None if v is None else round(v, 4) for name, v in values.items()}


def turn_records(
    works: Sequence[SourceWork],
    anchor_centroid: Mapping[str, np.ndarray],
    model: retrieve.TargetModel | None,
    margin: Mapping[str, float],
) -> list[contract.TurnRecord]:
    """Return one record per diarized turn of every source, sorted by source and start.

    It writes down what retrieval already computed and embeds nothing: each turn's centre-channel
    embedding from the first loop, the anchor centroid, the target model and the source's rival
    centroids. `retrieve.judge` is run on every embedded turn, whatever its role, so a turn of a
    non-admitted cluster carries the gaps a scorer needs to say how far another speaker sat from
    the target. A label with no embedded turn has no role from `retrieve.cluster_roles`; its
    turns are recorded as `other`, as the second pass treats them.
    """
    rows: list[contract.TurnRecord] = []
    for work in works:
        for index, (turn, emb) in enumerate(zip(work.turns, work.turn_embs, strict=True)):
            role = work.roles.get(turn.speaker)
            sim_anchor = sim_target = gaps = None
            passed: bool | None = None
            reasons: tuple[str, ...] = ()
            if emb is not None:
                sim_anchor = _rounded(retrieve.sims(emb, anchor_centroid))
                if model is not None:
                    verdict = retrieve.judge(emb, model, work.rivals, margin)
                    sim_target = _rounded(verdict.sim)
                    gaps = None if verdict.gaps is None else _rounded(verdict.gaps)
                    passed, reasons = verdict.passed, verdict.reasons
            span = Span(turn.start_s, turn.end_s)
            rows.append(
                contract.TurnRecord(
                    source_id=work.source.source_id,
                    turn=index,
                    start_s=round(turn.start_s, 3),
                    end_s=round(turn.end_s, 3),
                    diar_cluster=turn.speaker,
                    role=retrieve.OTHER if role is None else role.role,
                    embedded=emb is not None,
                    touches_overlap=segments.intersects(span, work.overlaps),
                    sim_anchor=sim_anchor,
                    sim_target=sim_target,
                    gaps=gaps,
                    passed=passed,
                    reasons=reasons,
                )
            )
    return sorted(rows, key=lambda r: (r.source_id, r.start_s, r.turn))


def _distribution(model: retrieve.TargetModel | None) -> dict[str, object] | None:
    if model is None:
        return None
    return {
        name: {
            "n": dist.n,
            "median": round(dist.median, 4),
            "p02": round(dist.p02, 4),
            "p10": round(dist.p10, 4),
            "floor": round(model.floor[name], 4),
        }
        for name, dist in model.distribution.items()
    }


def write_result(data: Path, run_id: str, status: str, detail: Mapping[str, object]) -> str:
    """Write the run's `extract-result.json`; return its root-relative path."""
    rel = f"{contract.run_rel(run_id)}/{contract.RESULT_NAME}"
    _write_json(data / rel, {"schema": SCHEMA, "status": status, **detail})
    print(f"extract: {status}")
    return rel


def run(
    data: Path,
    run_id: str,
    models: Models,
    track: Tracker | None = None,
    started: vram.VramStart | None = None,
) -> list[str]:
    """Run the stage over the plan of `run_id`; return the root-relative files written."""
    return Extraction(data, run_id, models, track, started).run()


def _budget(text: str) -> float:
    value = float(text)
    if not value > CAP_MARGIN_GIB:
        raise argparse.ArgumentTypeError(f"must be above {CAP_MARGIN_GIB:g} GiB")
    return value


def _job(
    data: Path, run_id: str, device: str, budget_gib: float | None, track: Tracker
) -> list[str]:
    """Start the card, load the models and run; stop on the card's memory with a named result.

    `vram.start` runs before `load_models`, so the weights count against the allocator's cap.
    A CUDA out-of-memory error, however a binding wrapped it, ends the run with an
    `out-of-memory` result naming where it happened (`track.where`), and a start with too little
    free VRAM with a `vram-short` result before any model loads. Both exit 0 so the executor
    verifies the result and the host reports it.
    """
    started = vram.start(device, budget_gib) if budget_gib is not None else None
    if started is not None and started.short:
        detail = {
            "free_gib": started.free_gib,
            "needed_gib": started.needed_gib,
            "budget_gib": started.budget_gib,
        }
        return [write_result(data, run_id, contract.VRAM_SHORT, detail)]
    try:
        begun = track.clock()
        with track.at(step=STEP_LOAD):
            models = load_models(device, track.windows)
        track.loaded(begun)
        return run(data, run_id, models, track, started)
    except StageError:
        raise
    except (JobError, RuntimeError) as exc:
        if not vram.out_of_memory(exc):
            raise
        where = track.where
        peak = vram.peak_mib(device)
        detail = {
            "source": where.source,
            "channel": where.channel,
            "step": where.step,
            "clip_s": where.clip_s,
            "budget_gib": budget_gib if started is None else started.budget_gib,
            "cap_gib": None if started is None else started.cap_gib,
            "peak_reserved_mib": None if peak is None else peak["reserved"],
        }
        return [write_result(data, run_id, contract.OUT_OF_MEMORY, detail)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.extract")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cpu", choices=contract.DEVICES)
    parser.add_argument("--vram-gb", type=_budget, default=None, help="the job's VRAM budget")
    args = parser.parse_args(argv)
    if args.device == CUDA and args.vram_gb is None:
        parser.error("--vram-gb is required with --device cuda")
    budget = args.vram_gb if args.device == CUDA else None
    track = Tracker(args.device, vram, clock)
    try:
        written = _job(args.data, args.run, args.device, budget, track)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.run_rel(args.run) / "outputs.json"
    _write_json(outputs, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
