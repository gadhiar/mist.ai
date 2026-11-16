"""Dialogue extraction and target-speaker retrieval (Design 4), run as the `extract` family's job.

Runs as `/opt/venv-extract/bin/python -I -m scripts.voice_data.stages.extract --data /data
--run R --device D`, with each source's copied track and `seed/reference/` mounted read-only and
`seed/runs/<run>/` writable. The host's plan names sources by id and gives the timestamp anchors
as source ids and times, so no file name reaches this job.

The order is Design 4's:

1. Every anchor is 3-30 s. The solo clips are read as they are.
2. Per source, the centre channel (or the downmix when the layout has none) goes through every
   separator candidate, `none` included. Each candidate is scored by the median SIM of its 50
   best 3 s VAD speech windows to the solo clips, else to its own timestamp anchors, cut from its
   own output; the best is kept and the downmix goes through it too.
3. The anchors must agree, timestamp anchors cut from their source's chosen output; else the run
   stops with the result `anchors-disagree` (the host exits 3). When `seed/sdh/` is mounted, the
   SDH spans `stages/sdh.py` holds as anchors are cut the same way and checked against the
   centroid of these anchors: one below the Anchors gate on either embedder is dropped and
   counted, and the rest join the anchor centroid that retrieval uses. They never enter the
   agreement check, so they can neither cause nor mask `anchors-disagree`.
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
   gate reads.

Models come from `load_models`, which T11's bindings serve; tests pass fakes through it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
from ..records import Prosody, SegmentRecord
from ..states import SPEAKING_STATES
from . import propose_states, retrieve, segments
from . import sdh as sdh_stage
from .ingest import decode_argv
from .prosody import prosody as measure_prosody

SCHEMA = 1
NONE = "none"
FC = "FC"
DOWNMIX = "downmix"
CHOICE_WINDOW_S = 3.0
CHOICE_BEST_N = 50
# A turn shorter than this is too short to embed; it is recorded as a length drop.
MIN_EMBED_S = 0.5
AUDIO_SUBTYPE = "PCM_24"


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


def load_models(device: str) -> Models:
    """Load the extract family's models from T11's bindings, on `device`.

    Each binding module exposes `load(device)`, imported here and not at the top so that a
    module walk imports no model package. Separators and embedders return a sequence.
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
        separators=tuple(separators.load(device)),
        embedders=tuple(embedders.load(device)),
        diarizer=diarizers.load(device),
        vad=vad.load(device),
        quality=quality.load(device),
        arousal=arousal.load(device),
    )


@dataclass(frozen=True, slots=True)
class Limits:
    """The gate values the run uses, from the plan's effective thresholds."""

    anchor_min_s: float
    anchor_max_s: float
    anchor_sim_min: float
    anchor_floor: float
    margin: float
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
    def from_thresholds(cls, values: Mapping[str, float]) -> Limits:
        """Read the gate keys the stage uses.

        Raises:
            StageError: A key is missing.
        """
        keys = {
            "anchor_min_s": "anchors.min_s",
            "anchor_max_s": "anchors.max_s",
            "anchor_sim_min": "anchors.sim_min",
            "anchor_floor": "retrieval.anchor_floor",
            "margin": "retrieval.margin",
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
        missing = [gate for gate in keys.values() if gate not in values]
        if missing:
            raise StageError(f"the plan's thresholds lack {', '.join(missing)}")
        return cls(**{name: float(values[gate]) for name, gate in keys.items()})


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
    scores: dict[str, float | None]
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


# The stage.


class Extraction:
    """One run of the stage over the plan's sources."""

    def __init__(self, data: Path, run_id: str, models: Models) -> None:
        if not contract.RUN_ID.fullmatch(run_id) or ".." in run_id:
            raise StageError("a run id is one folder name under seed/runs")
        self.data = data
        self.run_id = run_id
        self.run_dir = data / contract.run_rel(run_id)
        self.models = models
        self.plan = _read_plan(data, run_id)
        self.thresholds = {str(k): float(v) for k, v in self.plan["thresholds"].items()}  # type: ignore[attr-defined]
        self.limits = Limits.from_thresholds(self.thresholds)
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
    ) -> tuple[SourceWork, dict[int, retrieve.Embedding]]:
        rate = source.sample_rate
        primary = decode(self.data, source, source.primary_channel)
        outputs = separate(primary, rate, self.models.separators)
        scores = candidate_scores(outputs, rate, self.models, references)
        chosen = retrieve.choose_separator(scores)
        channels = {source.primary_channel: outputs[chosen]}
        del outputs, primary
        if source.primary_channel == FC:
            downmix = decode(self.data, source, DOWNMIX)
            channels[DOWNMIX] = np.asarray(
                self._separator(chosen).dialogue(downmix, rate), dtype=np.float32
            )
        main = channels[source.primary_channel]
        anchors = self._anchor_cuts(source.source_id, main, rate)
        sdh_embs = self._sdh_cuts(source.source_id, main, rate)
        turns = sorted(self.models.diarizer.turns(main, rate), key=lambda t: t.start_s)
        overlaps = list(self.models.diarizer.overlaps(main, rate))
        speech = {name: list(self.models.vad.speech(a, rate)) for name, a in channels.items()}
        quality = {name: self.models.quality.frames(a, rate) for name, a in channels.items()}
        turn_embs = [
            (
                self._embed(cut(main, rate, Span(t.start_s, t.end_s)), rate)
                if t.end_s - t.start_s >= MIN_EMBED_S
                else None
            )
            for t in turns
        ]
        offcentre: list[Span] = []
        if DOWNMIX in channels and source.primary_channel == FC:
            offcentre = [
                s
                for s in segments.subtract(speech[DOWNMIX], speech[FC])
                if s.end_s - s.start_s >= self.limits.min_s
            ]
        offcentre_embs = [self._embed(cut(channels[DOWNMIX], rate, s), rate) for s in offcentre]
        paths = {}
        for name, audio in channels.items():
            paths[name] = scratch / f"{source.source_id}-{name}.npy"
            np.save(paths[name], audio)
        print(
            f"extract: {source.source_id} separator {chosen}, {len(turns)} turns, "
            f"{len(overlaps)} overlaps, {len(offcentre)} off-centre spans"
        )
        work = SourceWork(
            source=source,
            separator=chosen,
            scores=scores,
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
        one below the Anchors gate on either embedder is dropped.
        """
        own = retrieve.centroids(anchors)
        if not self.sdh:
            return own, None
        floor = self.limits.anchor_sim_min - retrieve.EPS
        joined: list[retrieve.Embedding] = []
        dropped = dropped_offscreen = 0
        for work in works:
            spans = self.sdh.get(work.source.source_id)
            if spans is None:
                continue
            for span, emb in zip(spans.anchors, work.sdh_embs, strict=True):
                if emb is not None and all(v >= floor for v in retrieve.sims(emb, own).values()):
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
            for turn, _ in self._eligible(work):
                for tile in segments.tiles(Span(turn.start_s, turn.end_s), self.limits.window_s):
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
            for source in self.sources:
                work, cuts = self._first_loop(source, references, Path(tmp))
                works.append(work)
                anchor_embs.update(cuts)
            anchors = clip_embs + [anchor_embs[i] for i in sorted(anchor_embs)]
            agree, anchor_sims = retrieve.anchor_check(anchors, self.limits.anchor_sim_min)
            detail = {"anchors": self._anchor_rows(anchor_sims)}
            if not agree:
                return self._result("anchors-disagree", detail, written)
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
        rel = f"{contract.run_rel(self.run_id)}/{contract.RESULT_NAME}"
        _write_json(self.data / rel, {"schema": SCHEMA, "status": status, **detail})
        print(f"extract: {status}")
        return [*written, rel]

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
        windows = self._window_model(works) if model is not None else None
        audio_dir = self.run_dir / contract.AUDIO_DIR
        audio_dir.mkdir(exist_ok=True)
        drafts: list[Draft] = []
        for work in works:
            source_drafts = self._source_drafts(work, model, windows)
            drafts.extend(sorted(source_drafts, key=lambda d: (d.span.start_s, d.span.end_s)))
        records, proposals = self._records(drafts, audio_dir, written)
        report = self._report(works, records, drafts, model, windows, detail)
        run_rel = contract.run_rel(self.run_id)
        seg_rel = f"{run_rel}/{contract.SEGMENTS_NAME}"
        _write_text(self.data / seg_rel, "".join(json.dumps(r.to_json()) + "\n" for r in records))
        prop_rel = f"{run_rel}/{contract.PROPOSALS_NAME}"
        _write_text(self.data / prop_rel, "".join(json.dumps(p) + "\n" for p in proposals))
        report_rel = f"{run_rel}/{contract.REPORT_NAME}"
        _write_json(self.data / report_rel, report)
        kept = sum(r.kept for r in records)
        print(f"extract: {len(records)} segments, {kept} kept")
        return self._result("ok", detail, [*written, seg_rel, prop_rel, report_rel])

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
            # record; a turn too short to embed is a candidate when its cluster is admitted.
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
            retry = retrieve.judge(
                self._embed(np.asarray(cut(audio[DOWNMIX], rate, span)), rate),
                model,
                work.rivals,
                self.limits.margin,
            )
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
        own = retrieve.judge(self._embed(clip, rate), model, work.rivals, self.limits.margin)
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
            prosody=self.models.prosody(clip, rate),
            arousal=float(self.models.arousal.arousal(clip, rate)),
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
            sources.append(
                {
                    "source_id": sid,
                    **work.source.facts,
                    "separator": work.separator,
                    "separator_scores": {
                        k: None if v is None else round(v, 4) for k, v in work.scores.items()
                    },
                    "clusters": [
                        {
                            "diar_cluster": role.label,
                            "role": role.role,
                            "turns": role.turns,
                            "median_sim": {k: round(v, 4) for k, v in role.median_sim.items()},
                        }
                        for role in work.roles.values()
                    ],
                    "off_centre_spans": len(work.offcentre),
                    "kept": sum(r.kept and r.cluster == retrieve.PRIMARY for r in own),
                    "flagged": sum(r.kept and r.cluster == retrieve.FLAGGED for r in own),
                    "dropped": sum(not r.kept for r in own),
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
            },
            "sdh": detail.get("sdh"),
            "rehearsal": None,
            "thresholds": dict(sorted(self.thresholds.items())),
        }


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


def run(data: Path, run_id: str, models: Models) -> list[str]:
    """Run the stage over the plan of `run_id`; return the root-relative files written."""
    return Extraction(data, run_id, models).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.extract")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cpu", choices=contract.DEVICES)
    args = parser.parse_args(argv)
    try:
        written = run(args.data, args.run, load_models(args.device))
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.run_rel(args.run) / "outputs.json"
    _write_json(outputs, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
