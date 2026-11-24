"""The numbers of a target (Design 6), run as the `score` family's job.

Runs as `/opt/venv-score/bin/python -I -u -m scripts.voice_data.stages.target_build --data /data
--target T --device D`, with the clean run and the identity mounted read-only and
`targets/<T>/` writable. The host's plan (`target-plan.json`) names the confirmed segments
(ids, sources, states, audio paths and the prosody `clean` measured), the identity's clips
(states and audio paths) and the thresholds. No transcript reaches this job: the host places
references and hashes their transcripts after it.

With a seed, each confirmed segment is embedded whole. The centroid is the mean of the
segments, the distribution each segment's SIM to the centroid of the others (other sources' where
any exist, `score.leave_one_out_scores`), and each state's floor T6's `score.state_floors` over
its own segments' SIMs. The window distribution tiles every segment into `segment.window_s`
windows, scored the same way.

With an identity, each clip is embedded whole and scored against the seed centroid; with every
clip at its state's floor on both embedders the mode is `seed+identity`, otherwise `identity`
(`target.decide_mode`). In `identity` mode, or with no seed, the seed is not used: the identity's
clips are cut into 3 s windows and each window scored against the centroid of the other clips'
windows. With `state_floor.identity_windows_min` windows or more, their 10th percentile is one
pooled floor for every state; with fewer, the rehearsal-calibrated
`state_floor.rehearsal_identity_window_floor` is, and `distribution.kind` records which.

Profiles (medians) and spreads (standard deviations) per state come from the confirmed
segments' prosody, or, for a state only the identity voices, from its clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import target as contract
from ..bindings.protocols import Span, SpeakerEmbedder
from ..records import Prosody
from ..states import SPEAKING_STATES
from . import score, segments, transcribe
from .prosody import prosody as measure_prosody

SCHEMA = 1
RATE = 24000
DIGITS = 6
_FIELDS = ("f0_mean_hz", "f0_std_hz", "rate_sps", "energy_db")

Embedding = dict[str, np.ndarray]
Floors = Mapping[str, Mapping[str, score.StateFloor]]
"""Each embedder's floor per state, as `score.state_floors` returns them."""


class StageError(RuntimeError):
    """The build cannot proceed; the message names ids and counts only."""


@dataclass(frozen=True, slots=True)
class Models:
    """The embedders, and the prosody measure an identity clip's profile is taken with."""

    embedders: tuple[SpeakerEmbedder, ...]
    prosody: transcribe.ProsodyMeasure = measure_prosody


def load_models(device: str) -> Models:
    """Load the embedders from T11's bindings, on `device`.

    Imported here and not at the top, so a module walk imports no model package.
    """
    from ..bindings import embedders  # type: ignore[attr-defined]

    return Models(embedders=tuple(embedders.load(device)))


def _read_audio(path: Path) -> np.ndarray:
    import soundfile

    from .clean import resample

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return resample(audio.mean(axis=1), int(rate), RATE)


def _r(value: float) -> float:
    return round(float(value), DIGITS)


def _stats(scores: Sequence[float], keys: Sequence[str]) -> dict[str, object]:
    if len(scores) == 0:
        return {"n": 0, **{key: None for key in keys}}
    dist = score.distribution(scores)
    return {"n": dist.n, **{key: _r(getattr(dist, key)) for key in keys}}


def _loo(embs: Sequence[np.ndarray], groups: Sequence[str]) -> list[float]:
    return score.leave_one_out_scores(embs, groups).tolist() if len(embs) >= 2 else []


def _tiles(audio: np.ndarray, window_s: float) -> list[np.ndarray]:
    span = Span(0.0, len(audio) / RATE)
    return [
        audio[int(round(t.start_s * RATE)) : int(round(t.end_s * RATE))]
        for t in segments.tiles(span, window_s)
    ]


def _profile(values: Sequence[Prosody]) -> tuple[dict[str, float], dict[str, float]]:
    table = np.array([[getattr(p, f) for f in _FIELDS] for p in values], dtype=np.float64)
    return (
        {f: _r(v) for f, v in zip(_FIELDS, np.median(table, axis=0), strict=True)},
        {f: _r(v) for f, v in zip(_FIELDS, np.std(table, axis=0), strict=True)},
    )


class Build:
    """One target build over the plan in `targets/<T>/target-plan.json`."""

    def __init__(self, data: Path, target_id: str, models: Models) -> None:
        self.data = data
        self.target_id = target_id
        self.models = models
        self.rel = contract.target_rel(target_id)
        self.plan = self._read_plan()
        self.names = [e.name for e in models.embedders]
        if self.names != list(self.plan["embedders"]):  # type: ignore[call-overload]
            raise StageError("the loaded embedders are not the ones the plan names")
        self.th: Mapping[str, float] = self.plan["thresholds"]  # type: ignore[assignment]

    def _read_plan(self) -> dict[str, object]:
        path = self.data / self.rel / contract.PLAN_NAME
        try:
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StageError("the target plan is missing or not JSON") from exc
        if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
            raise StageError("the target plan is not a schema 1 plan")
        if plan.get("target") != self.target_id:
            raise StageError("the target plan is for another target")
        for key in ("segments", "clips", "embedders"):
            if not isinstance(plan.get(key), list):
                raise StageError(f"the target plan has no {key} list")
        if not plan["segments"] and not plan["clips"]:
            raise StageError("the target plan names neither segments nor identity clips")
        if not isinstance(plan.get("thresholds"), dict):
            raise StageError("the target plan has no thresholds")
        return plan

    def _embed(self, audio: np.ndarray) -> Embedding:
        return {
            e.name: np.asarray(e.embed(audio, RATE), dtype=np.float64)
            for e in self.models.embedders
        }

    def _measure(self, audio: np.ndarray) -> Prosody:
        return self.models.prosody(audio, RATE, ())

    # The seed.

    def _seed(self) -> dict[str, object]:
        rows: list[dict[str, object]] = self.plan["segments"]  # type: ignore[assignment]
        audios = [_read_audio(self.data / str(row["audio"])) for row in rows]
        embs = [self._embed(audio) for audio in audios]
        sources = [str(row["source_id"]) for row in rows]
        states = [str(row["state"]) for row in rows]
        window_embs: list[Embedding] = []
        window_sources: list[str] = []
        for audio, source in zip(audios, sources, strict=True):
            for tile in _tiles(audio, float(self.th["segment.window_s"])):
                window_embs.append(self._embed(tile))
                window_sources.append(source)
        offsets = {
            s: float(self.th[f"state_floor.offset.{s}"])
            for s in SPEAKING_STATES
            if f"state_floor.offset.{s}" in self.th
        }
        centroids, distribution, windows, floors = {}, {}, {}, {}
        for name in self.names:
            vectors = [e[name] for e in embs]
            centroids[name] = score.centroid(vectors)
            scores = score.leave_one_out_scores(vectors, sources)
            pooled = score.distribution(scores.tolist())
            distribution[name] = _stats(scores.tolist(), ("median", "p02", "p10"))
            by_state = {
                s: [float(v) for v, t in zip(scores, states, strict=True) if t == s]
                for s in SPEAKING_STATES
            }
            floors[name] = score.state_floors(
                by_state,
                pooled,
                offsets,
                min_n=int(self.th["state_floor.own_n_min"]),
                thin_n=int(self.th["state_floor.thin_n_min"]),
            )
            windows[name] = _stats(
                _loo([w[name] for w in window_embs], window_sources), ("median", "p10")
            )
        measured = [Prosody.from_json(r["prosody"]) for r in rows]  # type: ignore[arg-type]
        prosody = {
            s: [p for p, t in zip(measured, states, strict=True) if t == s] for s in SPEAKING_STATES
        }
        return {
            "centroids": centroids,
            "distribution": {"kind": contract.LOO_SEGMENTS, **distribution},
            "windows": windows,
            "floors": floors,
            "prosody": prosody,
        }

    def _seed_state(self, seed: Mapping[str, object], state: str) -> dict[str, object]:
        floors: Floors = seed["floors"]  # type: ignore[assignment]
        first = floors[self.names[0]][state]
        members: Sequence[Prosody] = seed["prosody"][state]  # type: ignore[index]
        profile, spread = _profile(members) if members else (None, None)
        return {
            "floor": {name: _r(floors[name][state].floor) for name in self.names},
            "floor_from": first.floor_from,
            "n": first.n,
            "profile": profile,
            "spread": spread,
        }

    # The identity.

    def _identity(
        self, clips: Sequence[Mapping[str, object]], audios: Sequence[np.ndarray]
    ) -> dict[str, object]:
        window_s = float(self.th["segment.window_s"])
        tiles: list[Embedding] = []
        groups: list[str] = []
        short: list[Embedding] = []
        short_groups: list[str] = []
        per_clip: dict[str, int] = {}
        for clip, audio in zip(clips, audios, strict=True):
            state = str(clip["state"])
            cut = _tiles(audio, contract.IDENTITY_WINDOW_S)
            per_clip[state] = len(cut)
            for tile in cut:
                tiles.append(self._embed(tile))
                groups.append(state)
            for tile in _tiles(audio, window_s):
                short.append(self._embed(tile))
                short_groups.append(state)
        if not tiles:
            raise StageError("the identity's clips hold no whole 3 s window")
        enough = len(tiles) >= max(2, int(self.th["state_floor.identity_windows_min"]))
        kind = contract.LOO_IDENTITY_WINDOWS if enough else contract.REHEARSAL_FLOOR
        rehearsal = float(self.th["state_floor.rehearsal_identity_window_floor"])
        wholes = [self._embed(audio) for audio in audios]
        centroids, distribution, windows, floor = {}, {}, {}, {}
        sims: dict[str, dict[str, float]] = {}
        for name in self.names:
            vectors = [t[name] for t in tiles]
            centroids[name] = score.centroid(vectors)
            scores = _loo(vectors, groups)
            distribution[name] = _stats(scores, ("median", "p02", "p10"))
            floor[name] = _r(score.distribution(scores).p10) if enough else _r(rehearsal)
            windows[name] = _stats(_loo([w[name] for w in short], short_groups), ("median", "p10"))
            for clip, whole in zip(clips, wholes, strict=True):
                state = str(clip["state"])
                others = [v for v, g in zip(vectors, groups, strict=True) if g != state]
                centre = score.centroid(others) if others else centroids[name]
                sims.setdefault(state, {})[name] = _r(score.cosine(whole[name], centre))
        states = {}
        for clip, audio in zip(clips, audios, strict=True):
            state = str(clip["state"])
            profile, spread = _profile([self._measure(audio)])
            states[state] = {
                "floor": dict(floor),
                "floor_from": kind,
                "n": per_clip[state],
                "profile": profile,
                "spread": spread,
            }
        return {
            "centroids": centroids,
            "distribution": {"kind": kind, **distribution},
            "windows": windows,
            "states": states,
            "identity_sims": sims,
        }

    # The run.

    def run(self) -> list[str]:
        """Compute the target's numbers and write the centroids and `build.json`."""
        clips: list[dict[str, object]] = self.plan["clips"]  # type: ignore[assignment]
        clip_audios = [_read_audio(self.data / str(c["audio"])) for c in clips]
        seed = self._seed() if self.plan["segments"] else None
        checks: list[dict[str, object]] = []
        if seed is not None:
            centroids: Mapping[str, np.ndarray] = seed["centroids"]  # type: ignore[assignment]
            for clip, audio in zip(clips, clip_audios, strict=True):
                state = str(clip["state"])
                emb = self._embed(audio)
                check: dict[str, object] = {
                    "state": state,
                    "sim": {n: _r(score.cosine(emb[n], centroids[n])) for n in self.names},
                    "floor": self._seed_state(seed, state)["floor"],
                }
                check["pass"] = contract.clip_passes(check)
                checks.append(check)
        mode = contract.decide_mode(seed=seed is not None, checks=checks)
        if mode == contract.IDENTITY:
            numbers = self._identity(clips, clip_audios)
        else:
            numbers = self._seed_numbers(seed, clips, clip_audios, checks)  # type: ignore[arg-type]
        return self._write(mode, checks, numbers)

    def _seed_numbers(
        self,
        seed: Mapping[str, object],
        clips: Sequence[Mapping[str, object]],
        audios: Sequence[np.ndarray],
        checks: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        floors: Floors = seed["floors"]  # type: ignore[assignment]
        clip_audio = {str(c["state"]): a for c, a in zip(clips, audios, strict=True)}
        states: dict[str, object] = {}
        for state in SPEAKING_STATES:
            if floors[self.names[0]][state].n == 0 and state not in clip_audio:
                continue
            entry = self._seed_state(seed, state)
            if entry["profile"] is None:
                entry["profile"], entry["spread"] = _profile([self._measure(clip_audio[state])])
            states[state] = entry
        return {
            "centroids": seed["centroids"],
            "distribution": seed["distribution"],
            "windows": seed["windows"],
            "states": states,
            "identity_sims": {str(c["state"]): c["sim"] for c in checks},
        }

    def _write(
        self, mode: str, checks: Sequence[Mapping[str, object]], numbers: Mapping[str, object]
    ) -> list[str]:
        written: list[str] = []
        centroid_files: dict[str, str] = {}
        centroids: Mapping[str, np.ndarray] = numbers["centroids"]  # type: ignore[assignment]
        for name in self.names:
            file_name = f"centroid-{name}.npy"
            path = self.data / self.rel / file_name
            part = path.with_name(file_name + ".part")
            with part.open("wb") as handle:
                np.save(handle, np.asarray(centroids[name], dtype=np.float64))
            os.replace(part, path)
            centroid_files[name] = file_name
            written.append(f"{self.rel}/{file_name}")
        doc = {
            "schema": SCHEMA,
            "target": self.target_id,
            "mode": mode,
            "centroids": centroid_files,
            "distribution": numbers["distribution"],
            "windows": numbers["windows"],
            "mode_check": list(checks),
            "states": numbers["states"],
            "identity_sims": numbers["identity_sims"],
        }
        path = self.data / self.rel / contract.BUILD_NAME
        part = path.with_name(contract.BUILD_NAME + ".part")
        part.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        os.replace(part, path)
        written.append(f"{self.rel}/{contract.BUILD_NAME}")
        print(f"target_build: ok, mode {mode}")
        return written


def run(data: Path, target_id: str, models: Models) -> list[str]:
    """Build the target's numbers; return the root-relative files written."""
    return Build(data, target_id, models).run()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.target_build")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    try:
        written = run(args.data, args.target, load_models(args.device))
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.target_rel(args.target) / "outputs.json"
    part = outputs.with_name("outputs.json.part")
    part.write_text(json.dumps({"schema": SCHEMA, "files": files}, indent=2) + "\n", "utf-8")
    os.replace(part, outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
