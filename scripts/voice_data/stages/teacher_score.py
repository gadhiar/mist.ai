"""The take gate over a teacher run: CER, SIM, prosody and arousal, keep and select (`score` job).

Runs as `/opt/venv-score/bin/python -I -m scripts.voice_data.stages.teacher_score --data /data
--run R --device D`, with the text version's `lines.jsonl`, the anchor's neutral clip and (for
a seed floor) the target's centroid files mounted read-only and `teachers/omnivoice/<R>/`
writable. The host's `score-plan.json` gives the floor's basis, `take.cer_max` and, from a seed
target, the neutral profile and spread.

Each unscored take is measured once: its CER against its line (`cer.character_error_rate` on
Whisper large-v3-turbo's text), its SIM on both embedders, its prosody and its arousal. With a
seed floor the SIM is to the target's centroid, the quantity the seed's floors were taken on;
under the seed fallback it is to the anchor clip, and each embedder's floor is the plan's
percentile of the round-1 takes' SIMs. Every take is then gated again with `score.keep` (a
floor from round-1 takes moves while round 1 is still being generated), and each line's kept
takes go through `score.select`. The profile is the seed's neutral profile and spread; without
one, the anchor clip's prosody, with the spread of the takes' own prosody.

Nothing this stage prints carries a line's text, a transcript or a clip's path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..bindings.protocols import ArousalMonitor, Asr, SpeakerEmbedder
from ..commands import teacher as contract
from ..errors import InputError
from ..records import Prosody, TakeRecord, read_jsonl
from . import score
from .cer import character_error_rate
from .prosody import prosody as measure_prosody

SCHEMA = 1
DIGITS = 6
_FIELDS = ("f0_mean_hz", "f0_std_hz", "rate_sps", "energy_db")


class StageError(RuntimeError):
    """The scoring cannot proceed; the message names ids and counts only."""


@dataclass(frozen=True, slots=True)
class Models:
    """The embedders, the CER's ASR, the arousal monitor and the prosody measure."""

    embedders: tuple[SpeakerEmbedder, ...]
    asr: Asr
    arousal: ArousalMonitor
    prosody: Callable[[np.ndarray, int], Prosody] = measure_prosody


def load_models(device: str) -> Models:
    """Load the score family's models from T11's bindings, on `device`.

    Imported here and not at the top, so a module walk imports no model package.
    """
    from ..bindings import arousal, asr, embedders  # type: ignore[attr-defined]

    return Models(
        embedders=tuple(embedders.load(device)),
        asr=asr.load_turbo(device),
        arousal=arousal.load(device),
    )


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _write_bytes(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_plan(data: Path, run_id: str) -> dict[str, object]:
    path = data / contract.run_rel(run_id) / contract.SCORE_PLAN_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the score plan is missing or not JSON") from exc
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("run") != run_id:
        raise StageError("the score plan is not this run's schema 1 plan")
    floor = plan.get("floor")
    if not isinstance(floor, dict) or floor.get("from") not in (
        contract.FLOOR_SEED,
        contract.FLOOR_TAKES,
    ):
        raise StageError("the score plan has no floor basis")
    return plan


def _texts(data: Path, lines_file: str) -> dict[str, str]:
    try:
        rows = (data / lines_file).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(row) for row in rows if row.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the lines file is missing or not JSONL") from exc
    return {
        str(row["line_id"]): str(row["text"])
        for row in parsed
        if isinstance(row, dict) and "line_id" in row and "text" in row
    }


def _round(value: float) -> float:
    return round(float(value), DIGITS)


def _prosody(values: Prosody) -> Prosody:
    return Prosody(**{name: _round(getattr(values, name)) for name in _FIELDS})


def _profile(value: object) -> Prosody | None:
    if not isinstance(value, Mapping) or not all(name in value for name in _FIELDS):
        return None
    return Prosody(**{name: float(value[name]) for name in _FIELDS})


@dataclass(frozen=True, slots=True)
class Scoring:
    """What a take is scored against: per-embedder SIM targets, the floor basis and the gate."""

    targets: dict[str, np.ndarray]
    floor_from: str
    floor: dict[str, float] | None
    percentile: float
    cer_max: float


class Scorer:
    """One scoring pass over a run's takes."""

    def __init__(self, data: Path, run_id: str, models: Models) -> None:
        self.data = data
        self.run_id = run_id
        self.models = models
        self.plan = _read_plan(data, run_id)
        self.names = tuple(e.name for e in models.embedders)
        if set(self.names) != set(self.plan.get("embedders", ())):  # type: ignore[arg-type]
            raise StageError("the embedders loaded are not the plan's")
        anchor: Mapping[str, object] = self.plan["anchor"]  # type: ignore[assignment]
        path = data / str(anchor["audio"])
        if not path.is_file() or _sha256(path) != anchor["audio_sha256"]:
            raise StageError("the anchor's audio is missing or does not match the plan")
        self.anchor_audio, self.anchor_rate = _read_audio(path)

    def _embed(self, audio: np.ndarray, rate: int) -> dict[str, np.ndarray]:
        return {e.name: np.asarray(e.embed(audio, rate), np.float64) for e in self.models.embedders}

    def scoring(self) -> Scoring:
        """Return the SIM targets and the gate the plan's floor basis calls for."""
        floor_doc: Mapping[str, object] = self.plan["floor"]  # type: ignore[assignment]
        cer_max = float(self.plan["cer_max"])  # type: ignore[arg-type]
        if floor_doc["from"] == contract.FLOOR_SEED:
            centroids: Mapping[str, str] = self.plan.get("centroids") or {}  # type: ignore[assignment]
            targets = {}
            for name in self.names:
                try:
                    targets[name] = np.load(self.data / centroids[name], allow_pickle=False)
                except (KeyError, OSError, ValueError) as exc:
                    raise StageError(f"the target's {name} centroid is missing") from exc
            values: Mapping[str, float] = floor_doc["values"]  # type: ignore[assignment]
            floor = {name: float(values[name]) for name in self.names}
            return Scoring(targets, contract.FLOOR_SEED, floor, 0.0, cer_max)
        targets = self._embed(self.anchor_audio, self.anchor_rate)
        percentile = float(floor_doc["percentile"])  # type: ignore[arg-type]
        return Scoring(targets, contract.FLOOR_TAKES, None, percentile, cer_max)

    def measure(self, take: TakeRecord, text: str, scoring: Scoring) -> TakeRecord:
        """Return the take with its SIM, CER, prosody and arousal; gating comes after."""
        audio, rate = _read_audio(self.data / take.audio)
        embedding = self._embed(audio, rate)
        sim = {n: _round(score.cosine(embedding[n], scoring.targets[n])) for n in self.names}
        heard = self.models.asr.transcribe(audio, rate).text
        return replace(
            take,
            sim=sim,
            cer=_round(character_error_rate(text, heard)),
            prosody=_prosody(self.models.prosody(audio, rate)),
            arousal=_round(self.models.arousal.arousal(audio, rate)),
            kept=False,
            selected=False,
            reasons=(),
        )

    def floors(self, takes: Sequence[TakeRecord], scoring: Scoring) -> tuple[dict[str, float], int]:
        """Return each embedder's floor and how many takes it was taken on (0 for the seed's)."""
        if scoring.floor is not None:
            return dict(scoring.floor), 0
        first = [t for t in takes if t.round == 1]
        if not first:
            raise StageError("the run has no round-1 take to take a floor on")
        return {
            n: _round(np.percentile([t.sim[n] for t in first], scoring.percentile))
            for n in self.names
        }, len(first)

    def profile(self, takes: Sequence[TakeRecord]) -> tuple[Prosody, Prosody]:
        """Return the neutral profile and spread `select` measures distance against."""
        profile, spread = _profile(self.plan.get("profile")), _profile(self.plan.get("spread"))
        if profile is not None and spread is not None:
            return profile, spread
        own = _prosody(self.models.prosody(self.anchor_audio, self.anchor_rate))
        values = np.asarray([[getattr(t.prosody, f) for f in _FIELDS] for t in takes])
        spreads = values.std(axis=0) if len(values) else np.zeros(len(_FIELDS))
        return own, Prosody(**{f: _round(s) for f, s in zip(_FIELDS, spreads, strict=True)})

    def run(self) -> tuple[list[str], dict[str, object]]:
        """Score, gate and select; return the files written and the summary."""
        run_dir = self.data / contract.run_rel(self.run_id)
        takes_path = run_dir / contract.TAKES_NAME
        try:
            takes = read_jsonl(takes_path, TakeRecord)
        except InputError as exc:
            raise StageError("takes.jsonl is missing or malformed") from exc
        texts = _texts(self.data, str(self.plan["lines_file"]))
        missing = {t.line_id for t in takes} - set(texts)
        if missing:
            raise StageError(f"{len(missing)} lines of takes.jsonl are not in the lines file")
        scoring = self.scoring()
        measured = [
            t if contract.is_scored(t) else self.measure(t, texts[t.line_id], scoring)
            for t in takes
        ]
        floors, floor_n = self.floors(measured, scoring)
        profile, spread = self.profile(measured)
        gated = []
        for take in measured:
            kept, reasons = score.keep(
                score.TakeScores(take.take_id, take.sim, take.cer, take.prosody),
                floors,
                cer_max=scoring.cer_max,
            )
            gated.append(replace(take, kept=kept, selected=False, reasons=reasons))
        chosen: set[str] = set()
        for line_id in dict.fromkeys(t.line_id for t in gated):
            kept_takes = [
                score.TakeScores(t.take_id, t.sim, t.cer, t.prosody)
                for t in gated
                if t.line_id == line_id and t.kept
            ]
            if kept_takes:
                chosen.add(score.select(kept_takes, profile, spread).take_id)
        final = [replace(t, selected=t.take_id in chosen) for t in gated]
        _write_bytes(
            takes_path, "".join(json.dumps(t.to_json()) + "\n" for t in final).encode("utf-8")
        )
        summary = self._summary(final, scoring, floors, floor_n, run_dir)
        _write_json(run_dir / contract.SUMMARY_NAME, summary)
        rel = contract.run_rel(self.run_id)
        written = [f"{rel}/{contract.TAKES_NAME}", f"{rel}/{contract.SUMMARY_NAME}"]
        print(f"teacher_score: {summary['kept']} of {summary['takes']} takes kept")
        return written, summary

    def _summary(
        self,
        takes: Sequence[TakeRecord],
        scoring: Scoring,
        floors: Mapping[str, float],
        floor_n: int,
        run_dir: Path,
    ) -> dict[str, object]:
        import soundfile

        def hours(rows: Sequence[TakeRecord]) -> float:
            seconds = 0.0
            for take in rows:
                info = soundfile.info(str(self.data / take.audio))
                seconds += info.frames / info.samplerate
            return round(seconds / 3600.0, 4)

        previous: object = None
        try:
            previous = json.loads((run_dir / contract.SUMMARY_NAME).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            previous = None
        status = previous.get("status") if isinstance(previous, dict) else None
        kept = [t for t in takes if t.kept]
        selected = [t for t in takes if t.selected]
        floor_doc: Mapping[str, object] = self.plan["floor"]  # type: ignore[assignment]
        return {
            "schema": SCHEMA,
            "run": self.run_id,
            "step": "score",
            "status": status,
            "lines": len({t.line_id for t in takes}),
            "lines_kept": len({t.line_id for t in kept}),
            "takes": len(takes),
            "kept": len(kept),
            "selected": len(selected),
            "kept_hours": hours(kept),
            "selected_hours": hours(selected),
            "floors": {
                "from": scoring.floor_from,
                "target": floor_doc.get("target"),
                "values": dict(floors),
                "n": floor_n,
            },
            "cer_max": scoring.cer_max,
        }


def run(data: Path, run_id: str, models: Models) -> tuple[list[str], dict[str, object]]:
    """Score the run's takes; return the root-relative files written and the summary."""
    return Scorer(data, run_id, models).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.teacher_score")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cpu", choices=contract.DEVICES)
    args = parser.parse_args(argv)
    try:
        written, _summary = run(args.data, args.run, load_models(args.device))
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.run_rel(args.run) / contract.OUTPUTS_NAME
    _write_json(outputs, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
