"""SIM-gated cleaning, transcripts and references (Design 5), run as the `extract` family's job.

Runs as `/opt/venv-extract/bin/python -I -m scripts.voice_data.stages.clean --data /data --run R
--device D`, with the extract run the plan names mounted read-only and `seed/runs/<R>/`
writable. The host's plan names the extract run, the target id, the thresholds and Raj's labels
(segment ids and states, no text), so nothing private reaches the argv.

The order is Design 5's:

1. Each kept segment is read from the extract run and resampled to 24 kHz mono, where every
   later step works. With labels, a kept segment they do not list is dropped as `not-her`, and
   a listed one takes its labelled state.
2. The cleaning gate's distribution is the leave-one-out SIM of the confirmed segments
   (labelled ones, else primary-cluster ones), and its threshold per embedder the
   `cleaning.sim_percentile` of it (the median). Each `Enhancer` step runs on every segment in
   turn, and is kept only when the median SIM of its output to its input reaches the threshold
   on every embedder and the median spectrum distance (`score.spectrum_distance_db`) is at most
   `cleaning.spectrum_max_db`. A kept step's output is the next step's input; a dropped step
   changes nothing. `gates.json` records each decision with both measures.
3. One loudness gain per source brings the source's median active energy to the target level,
   lowered when it would lift any of its segments' peaks above the ceiling, so the energy
   differences between a source's lines (sleepy against excited) survive.
4. Both ASRs transcribe the cleaned clip (`transcribe`), the prosody is measured again with the
   timing ASR's word times, and the clip is written as 24 kHz mono FLAC under `audio/`.
5. References per state are chosen from the confirmed segments (`references`) and written as
   WAV PCM16 under `refs/<state>/`; the host copies them to `targets/<T>/refs/<state>/`.

The reference SIM reuses the embeddings of the cleaned audio before its gain: a scalar gain
does not change a speaker embedding, and embedding every clip again would double the cost of the
slowest models here.

Models come from `load_models`, which T11's bindings serve; tests pass fakes through it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..bindings.protocols import Asr, Enhancer, SpeakerEmbedder
from ..commands import clean as contract
from ..records import SegmentRecord
from . import references, score, transcribe
from .prosody import active_energy_db
from .prosody import prosody as measure_prosody

SCHEMA = 1
RATE = references.RATE
AUDIO_SUBTYPE = "PCM_24"
LOUDNESS_TARGET_DB = -23.0
PEAK_CEILING_DBFS = -1.0

Embedding = dict[str, np.ndarray]


class StageError(RuntimeError):
    """The run cannot proceed; the message names ids and counts only."""


@dataclass(frozen=True, slots=True)
class Models:
    """Every model the stage calls, as the protocols of `bindings.protocols`.

    `enhancers` are the cleaning steps in the order they run; `timing` gives word times and
    `verbatim` the training text.
    """

    embedders: tuple[SpeakerEmbedder, ...]
    enhancers: tuple[Enhancer, ...]
    timing: Asr
    verbatim: Asr
    prosody: transcribe.ProsodyMeasure = measure_prosody


def load_models(device: str) -> Models:
    """Load the clean stage's models from T11's bindings, on `device`.

    Each binding module exposes `load(device)`, imported here and not at the top so that a
    module walk imports no model package. Embedders and enhancers return a sequence; the ASR
    binding returns the timing ASR, then the verbatim one.
    """
    from ..bindings import asr, embedders, enhancers  # type: ignore[attr-defined]

    timing, verbatim = asr.load(device)
    return Models(
        embedders=tuple(embedders.load(device)),
        enhancers=tuple(enhancers.load(device)),
        timing=timing,
        verbatim=verbatim,
    )


@dataclass(frozen=True, slots=True)
class StepDecision:
    """One cleaning step's gate decision and both of its measures."""

    step: str
    sim: dict[str, float]
    threshold: dict[str, float]
    spectrum_db: float
    spectrum_max_db: float
    kept: bool
    reasons: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        """Return the decision as `gates.json` records it; a non-finite measure is `null`."""
        return {
            "step": self.step,
            "sim": {name: _finite(value) for name, value in self.sim.items()},
            "threshold": {name: _finite(value) for name, value in self.threshold.items()},
            "spectrum_db": _finite(self.spectrum_db),
            "spectrum_max_db": self.spectrum_max_db,
            "kept": self.kept,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    """A step's decision, and the audio and embeddings the next step starts from."""

    decision: StepDecision
    audios: list[np.ndarray]
    embeddings: list[Embedding]


def _finite(value: float, digits: int = 6) -> float | None:
    return round(float(value), digits) if math.isfinite(value) else None


# Audio.


def resample(audio: np.ndarray, rate_in: int, rate_out: int = RATE) -> np.ndarray:
    """Return the clip at `rate_out` by polyphase resampling; the same rate returns the input."""
    if rate_in == rate_out:
        return audio
    from scipy.signal import resample_poly

    common = math.gcd(rate_in, rate_out)
    out = resample_poly(np.asarray(audio, dtype=np.float64), rate_out // common, rate_in // common)
    return out.astype(np.float32)


def _read_audio(path: Path) -> np.ndarray:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), int(rate))


def _write_audio(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    soundfile.write(str(part), audio, rate, format="FLAC", subtype=AUDIO_SUBTYPE)
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return 20.0 * math.log10(peak) if peak > 0.0 else float("-inf")


# The cleaning gate.


def embed(embedders: Sequence[SpeakerEmbedder], audio: np.ndarray, rate: int) -> Embedding:
    """Return the clip's embedding on every embedder."""
    return {e.name: np.asarray(e.embed(audio, rate), dtype=np.float64) for e in embedders}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # A silenced clip can embed as a zero vector; its cosine is NaN, which fails every gate.
    with np.errstate(divide="ignore", invalid="ignore"):
        return score.cosine(a, b)


def sim_thresholds(
    embeddings: Sequence[Mapping[str, np.ndarray]], sources: Sequence[str], percentile: float
) -> dict[str, float]:
    """Return the `percentile` of the leave-one-out SIM distribution, per embedder.

    Raises:
        ValueError: Fewer than two clips, or `sources` is not one per clip.
    """
    return {
        name: float(
            np.percentile(
                score.leave_one_out_scores([e[name] for e in embeddings], sources), percentile
            )
        )
        for name in embeddings[0]
    }


def gate_step(
    step: Enhancer,
    audios: Sequence[np.ndarray],
    embeddings: Sequence[Embedding],
    rate: int,
    embedders: Sequence[SpeakerEmbedder],
    thresholds: Mapping[str, float],
    spectrum_max_db: float,
) -> StepResult:
    """Run one cleaning step on every clip and keep it only if the cleaning gate passes.

    The measures are the median over clips of the SIM of each output to its input, per
    embedder, and the median spectrum distance. A failed check adds a reason: `sim:<embedder>`
    in the order of `thresholds`, then `spectrum`.
    """
    outs = [np.asarray(step.apply(audio, rate), dtype=np.float32) for audio in audios]
    out_embs = [embed(embedders, out, rate) for out in outs]
    sim = {
        name: float(np.median([_cosine(o[name], i[name]) for o, i in zip(out_embs, embeddings)]))
        for name in thresholds
    }
    spectrum = float(
        np.median([score.spectrum_distance_db(o, a, rate) for o, a in zip(outs, audios)])
    )
    reasons = [f"sim:{name}" for name, floor in thresholds.items() if not sim[name] >= floor]
    if not spectrum <= spectrum_max_db:
        reasons.append("spectrum")
    decision = StepDecision(
        step=step.name,
        sim=sim,
        threshold=dict(thresholds),
        spectrum_db=spectrum,
        spectrum_max_db=spectrum_max_db,
        kept=not reasons,
        reasons=tuple(reasons),
    )
    if decision.kept:
        return StepResult(decision, outs, out_embs)
    return StepResult(decision, list(audios), list(embeddings))


def source_gains(
    sources: Sequence[str],
    energies_db: Sequence[float],
    peaks_dbfs: Sequence[float],
    *,
    target_db: float = LOUDNESS_TARGET_DB,
    ceiling_dbfs: float = PEAK_CEILING_DBFS,
) -> dict[str, float]:
    """Return one gain in dB per source, so the energy differences within a source survive.

    The gain brings the source's median active energy to `target_db`, lowered where it would
    lift the source's loudest peak above `ceiling_dbfs`. A source with no measurable energy
    gets 0 dB.
    """
    gains: dict[str, float] = {}
    for source in dict.fromkeys(sources):
        mine = [i for i, s in enumerate(sources) if s == source]
        finite = [energies_db[i] for i in mine if math.isfinite(energies_db[i])]
        if not finite:
            gains[source] = 0.0
            continue
        gain = target_db - float(np.median(finite))
        peak = max(peaks_dbfs[i] for i in mine)
        if math.isfinite(peak):
            gain = min(gain, ceiling_dbfs - peak)
        gains[source] = float(gain)
    return gains


# The run.


def _read_plan(data: Path, run_id: str) -> dict[str, object]:
    path = data / contract.run_rel(run_id) / contract.PLAN_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the clean plan is missing or not JSON") from exc
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise StageError("the clean plan is not a schema 1 plan")
    if not isinstance(plan.get("from"), str) or not isinstance(plan.get("target"), str):
        raise StageError("the clean plan names no extract run or target")
    if not isinstance(plan.get("thresholds"), dict):
        raise StageError("the clean plan has no thresholds")
    if plan.get("labels") is not None and not isinstance(plan.get("labels"), dict):
        raise StageError("the clean plan's labels are not a map")
    return plan


class Cleaning:
    """One clean run over the extract run its plan names."""

    def __init__(self, data: Path, run_id: str, models: Models) -> None:
        plan = _read_plan(data, run_id)
        self.data = data
        self.run_id = run_id
        self.models = models
        self.from_run = str(plan["from"])
        self.target = str(plan["target"])
        self.labels: dict[str, str] | None = plan["labels"]  # type: ignore[assignment]
        thresholds: Mapping[str, float] = plan["thresholds"]  # type: ignore[assignment]
        self.sim_percentile = float(thresholds["cleaning.sim_percentile"])
        self.spectrum_max_db = float(thresholds["cleaning.spectrum_max_db"])
        self.run_rel = contract.run_rel(run_id)
        self.run_dir = data / self.run_rel

    def _records(self) -> list[SegmentRecord]:
        from_rel = contract.run_rel(self.from_run)
        path = self.data / from_rel / contract.SEGMENTS_NAME
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise StageError(f"run {self.from_run}'s segments are missing") from exc
        records = [SegmentRecord.from_json(json.loads(line)) for line in lines if line.strip()]
        for record in records:
            if record.kept and not record.audio.startswith(f"{from_rel}/"):
                raise StageError(f"{record.segment_id}'s audio is outside run {self.from_run}")
        return records

    def run(self) -> list[str]:
        """Run the stage; return the root-relative files written."""
        records = self._records()
        rows: list[SegmentRecord] = []
        work: list[int] = []
        audios: list[np.ndarray] = []
        for record in records:
            if not record.kept:
                rows.append(record)
            elif self.labels is not None and record.segment_id not in self.labels:
                reasons = (*record.reasons, contract.NOT_HER)
                rows.append(replace(record, kept=False, audio="", reasons=reasons))
            else:
                state = self.labels[record.segment_id] if self.labels is not None else record.state
                rows.append(replace(record, state=state))
                work.append(len(rows) - 1)
                audios.append(_read_audio(self.data / record.audio))
        ids = set(contract.confirmed_ids([rows[i] for i in work], self.labels))
        confirmed = [k for k, i in enumerate(work) if rows[i].segment_id in ids]
        if len(confirmed) < 2:
            raise StageError("fewer than two confirmed segments to clean")
        sources = [rows[i].source_id for i in work]
        conf_sources = [sources[k] for k in confirmed]

        embedders = self.models.embedders
        embs = [embed(embedders, audio, RATE) for audio in audios]
        conf_embs = [embs[k] for k in confirmed]
        thresholds = sim_thresholds(conf_embs, conf_sources, self.sim_percentile)
        distribution = self._distribution(conf_embs, conf_sources, thresholds)
        decisions = []
        for step in self.models.enhancers:
            result = gate_step(
                step, audios, embs, RATE, embedders, thresholds, self.spectrum_max_db
            )
            decisions.append(result.decision)
            audios, embs = result.audios, result.embeddings

        energies = [active_energy_db(audio, RATE) for audio in audios]
        gains = source_gains(sources, energies, [_peak_dbfs(a) for a in audios])
        audios = [
            (audio * 10.0 ** (gains[source] / 20.0)).astype(np.float32)
            for audio, source in zip(audios, sources, strict=True)
        ]

        written: list[str] = []
        for k, i in enumerate(work):
            record = rows[i]
            result = transcribe.transcribe(
                audios[k], RATE, self.models.timing, self.models.verbatim
            )
            rel = f"{self.run_rel}/{contract.AUDIO_DIR}/{record.segment_id}.flac"
            _write_audio(self.data / rel, audios[k], RATE)
            written.append(rel)
            annotated = transcribe.annotate(record, audios[k], RATE, result, self.models.prosody)
            rows[i] = replace(annotated, audio=rel)

        choices, medians = self._choose(rows, work, confirmed, audios, embs)
        refs_doc, ref_files = self._write_references(choices, medians, rows, work, audios)
        written += ref_files

        segments_rel = f"{self.run_rel}/{contract.SEGMENTS_NAME}"
        part = (self.data / segments_rel).with_name(contract.SEGMENTS_NAME + ".part")
        part.write_text("".join(json.dumps(r.to_json()) + "\n" for r in rows), encoding="utf-8")
        os.replace(part, self.data / segments_rel)
        disagree = sum(transcribe.ASR_DISAGREE in rows[i].reasons for i in work)
        gates_doc = {
            "schema": SCHEMA,
            "run": self.run_id,
            "from": self.from_run,
            "distribution": distribution,
            "steps": [d.to_json() for d in decisions],
            "loudness": {
                "target_db": LOUDNESS_TARGET_DB,
                "ceiling_dbfs": PEAK_CEILING_DBFS,
                "gains_db": {source: round(gain, 4) for source, gain in gains.items()},
            },
            "asr": {"cer_max": transcribe.AGREEMENT_CER_MAX, "disagree": disagree},
        }
        report = self._report(records, rows, work, decisions, choices, disagree)
        for name, payload in (
            (contract.GATES_NAME, gates_doc),
            (contract.REFS_NAME, refs_doc),
            (contract.REPORT_NAME, report),
        ):
            _write_json(self.run_dir / name, payload)
            written.append(f"{self.run_rel}/{name}")
        written.append(segments_rel)
        print(f"clean: ok, {len(work)} segments")
        return written

    def _distribution(
        self,
        embs: Sequence[Embedding],
        sources: Sequence[str],
        thresholds: Mapping[str, float],
    ) -> dict[str, object]:
        out: dict[str, object] = {}
        for name, threshold in thresholds.items():
            dist = score.leave_one_out([e[name] for e in embs], sources)
            out[name] = {
                "n": dist.n,
                "median": round(dist.median, 6),
                "p02": round(dist.p02, 6),
                "p10": round(dist.p10, 6),
                "percentile": self.sim_percentile,
                "threshold": round(threshold, 6),
            }
        return out

    def _choose(
        self,
        rows: Sequence[SegmentRecord],
        work: Sequence[int],
        confirmed: Sequence[int],
        audios: Sequence[np.ndarray],
        embs: Sequence[Embedding],
    ) -> tuple[dict[str, references.StateChoice], dict[str, float]]:
        sources = [rows[work[k]].source_id for k in confirmed]
        loo = {
            e.name: score.leave_one_out_scores([embs[k][e.name] for k in confirmed], sources)
            for e in self.models.embedders
        }
        medians = {name: float(np.median(values)) for name, values in loo.items()}
        candidates = []
        for j, k in enumerate(confirmed):
            record = rows[work[k]]
            candidates.append(
                references.Candidate(
                    segment_id=record.segment_id,
                    source_id=record.source_id,
                    state=record.state or record.state_proposed,
                    duration_s=len(audios[k]) / RATE,
                    snr_db=record.snr_db,
                    prosody=record.prosody,
                    sim={name: float(values[j]) for name, values in loo.items()},
                )
            )
        return references.choose(candidates, medians), medians

    def _write_references(
        self,
        choices: Mapping[str, references.StateChoice],
        medians: Mapping[str, float],
        rows: Sequence[SegmentRecord],
        work: Sequence[int],
        audios: Sequence[np.ndarray],
    ) -> tuple[dict[str, object], list[str]]:
        by_id = {rows[i].segment_id: audios[k] for k, i in enumerate(work)}
        files: list[str] = []
        states: dict[str, object] = {}
        for state, choice in choices.items():
            refs = []
            for number, chosen in enumerate(choice.chosen, start=1):
                rel = references.reference_rel(state, number)
                references.write_wav(self.run_dir / rel, by_id[chosen.segment_id], RATE)
                files.append(f"{self.run_rel}/{rel}")
                refs.append(
                    {
                        "audio": rel,
                        "segment_id": chosen.segment_id,
                        "source_id": chosen.source_id,
                        "duration_s": round(chosen.duration_s, 3),
                        "sim": {name: round(v, 6) for name, v in chosen.sim.items()},
                    }
                )
            states[state] = {
                "members": choice.members,
                "eligible": choice.eligible,
                "short": choice.short,
                "refs": refs,
            }
        doc = {
            "schema": SCHEMA,
            "run": self.run_id,
            "target": self.target,
            "medians": {name: round(m, 6) for name, m in medians.items()},
            "states": states,
        }
        return doc, files

    def _report(
        self,
        records: Sequence[SegmentRecord],
        rows: Sequence[SegmentRecord],
        work: Sequence[int],
        decisions: Sequence[StepDecision],
        choices: Mapping[str, references.StateChoice],
        disagree: int,
    ) -> dict[str, object]:
        minutes: dict[str, float] = {}
        for i in work:
            row = rows[i]
            state = row.state or row.state_proposed
            minutes[state] = minutes.get(state, 0.0) + (row.end_s - row.start_s) / 60.0
        return {
            "schema": SCHEMA,
            "run": self.run_id,
            "from": self.from_run,
            "target": self.target,
            "segments": {
                "in": len(records),
                "kept": len(work),
                "not_her": sum(contract.NOT_HER in r.reasons for r in rows),
                "asr_disagree": disagree,
            },
            "steps": [d.to_json() for d in decisions],
            "minutes_per_state": {s: round(m, 3) for s, m in sorted(minutes.items())},
            "references": {state: len(c.chosen) for state, c in choices.items()},
            "short_states": [state for state, c in choices.items() if c.short],
        }


def run(data: Path, run_id: str, models: Models) -> list[str]:
    """Run the stage over the plan of `run_id`; return the root-relative files written."""
    return Cleaning(data, run_id, models).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.clean")
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
