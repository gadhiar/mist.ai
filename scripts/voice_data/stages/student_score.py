"""Render and score a student checkpoint: the two jobs of `student score`.

Runs as `<venv>/bin/python -I -u -m scripts.voice_data.stages.student_score --data /data render
--run R --checkpoint C --device D` in the `train-pocket` family (`/opt/venv-train-pocket`), then
with `score` in place of `render` in the `score` family (`/opt/venv-score`). Both read the host's
`students/<R>/scores/<C>/score-plan.json` and write into that folder, the jobs' writable mount:
the render job has the checkpoint's runtime-layout weights, the corpus's validation split and the
anchor's neutral clip mounted read-only, and the weight cache at `/cache`; the scoring job has
the validation split and the anchor's clip.

`render` loads the installed pocket-tts 4.0.1 wheel, the release the realtime adapter runs, not
the training clone: a copy of its `english.yaml` whose `weights_path` is the checkpoint's weights,
whose fallback weights are unset (4.0.1 reads them only when loading `weights_path` raises, which
a local file does not), and whose tokenizer is the cached one with the sha256 the student was
trained with. The anchor's neutral clip becomes the voice state, as the realtime adapter's
`get_state_for_audio_prompt` does, and every validation line and bench line is rendered once,
torch seeded with the line's `render_seed`, to a 16-bit WAV under `renders/`, listed in
`renders.json`.

`score` measures each render with the score family's models (`teacher_score.load_models`):
Whisper large-v3-turbo's CER against its line (`cer.character_error_rate`), SIM to the anchor clip
on both embedders, and the F0 standard deviation (`prosody`). The references are the seed
target's numbers, which the host copied into the plan (`commands.student.seed_reference`).
`score.json` holds the highest line's CER (the gated one), the mean CER, the SIM ratio (each embedder's median
render SIM over the seed's leave-one-out SIM median, the lower of the two), the F0 SD ratio (the
renders' median over the seed's neutral F0 SD) and the gates' verdict
(`commands.student.gate_failures`); `lines.jsonl` holds each line's measures by id.

Nothing this stage prints carries a transcript, a line's text or a clip's path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ..commands import student as contract
from ..errors import InputError, JobError
from . import fetch, teacher_score
from .cer import character_error_rate
from .score import cosine

SCHEMA = 1
DIGITS = 6
CONFIG_NAME = "english.yaml"
LIBRARY = "pocket-tts"
_REMOTE_PREFIXES = ("hf://", "http://", "https://")


class StageError(RuntimeError):
    """The render or the scoring cannot proceed; the message names ids and counts only."""


class Synth(Protocol):
    """A loaded student: a voice state from a clip, and a line rendered in it."""

    sample_rate: int

    def voice(self, clip: Path) -> object:
        """Return the voice state of a reference clip."""
        ...

    def render(self, voice: object, text: str, seed: int) -> np.ndarray:
        """Return `text` spoken in `voice` as mono float audio at `sample_rate`."""
        ...


# Files.


def _sha256(path: Path) -> str:
    return fetch.sha256_file(path)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _read_json(path: Path, what: str) -> dict[str, object]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError(f"{what} is missing or not JSON") from exc
    if not isinstance(doc, dict):
        raise StageError(f"{what} is not a JSON object")
    return doc


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    soundfile.write(str(part), clipped, rate, format="WAV", subtype="PCM_16")
    os.replace(part, path)


# The plan and its lines.


def read_plan(data: Path, run_id: str, checkpoint: str) -> dict[str, object]:
    """Return the host's score plan for this run and checkpoint.

    Raises:
        StageError: It is missing, or not this run's and checkpoint's schema 1 plan.
    """
    path = data / contract.scores_rel(run_id, checkpoint) / contract.SCORE_PLAN_NAME
    plan = _read_json(path, "the score plan")
    if (plan.get("schema"), plan.get("run"), plan.get("checkpoint")) != (
        SCHEMA,
        run_id,
        checkpoint,
    ):
        raise StageError("the score plan is not this checkpoint's schema 1 plan")
    if not isinstance(plan.get("lines"), list) or not plan["lines"]:
        raise StageError("the score plan lists no line")
    return plan


def _checked(path: Path, sha256: object, what: str) -> Path:
    if not path.is_file() or _sha256(path) != sha256:
        raise StageError(f"{what} is missing or does not match the plan")
    return path


def valid_rows(data: Path, plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return the validation split's rows, checked against the plan's sha256."""
    path = _checked(data / str(plan["valid"]), plan.get("valid_sha256"), "the validation split")
    try:
        return contract.read_valid_rows(path)
    except InputError as exc:
        raise StageError(str(exc)) from exc


def line_texts(data: Path, plan: Mapping[str, object]) -> dict[str, str]:
    """Return each planned line's text by id: the validation transcripts, then the bench lines.

    Raises:
        StageError: A source does not match the plan's sha256, or the plan's ids are not
            exactly the sources' ids.
    """
    texts = {
        contract.valid_id(n): str(row["transcript"])
        for n, row in enumerate(valid_rows(data, plan), start=1)
    }
    bench = _checked(contract.bench_lines_path(), plan.get("bench_sha256"), "the bench lines")
    try:
        texts.update({contract.bench_id(i): t for i, t in contract.read_bench_lines(bench)})
    except InputError as exc:
        raise StageError(str(exc)) from exc
    planned = [line.get("id") for line in plan["lines"] if isinstance(line, dict)]  # type: ignore[union-attr]
    if len(planned) != len(plan["lines"]) or sorted(planned) != sorted(texts):  # type: ignore[arg-type,type-var]
        raise StageError("the plan's lines are not the validation split's and the bench's")
    return texts


def _anchor(data: Path, plan: Mapping[str, object]) -> Path:
    anchor = plan.get("anchor")
    if not isinstance(anchor, dict):
        raise StageError("the score plan names no anchor")
    return _checked(data / str(anchor["audio"]), anchor.get("audio_sha256"), "the anchor's audio")


# `render`.


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def runtime_config(installed: object, weights: Path, tokenizer: Path) -> dict[str, object]:
    """Return a copy of pocket-tts's installed `english.yaml` that loads the student offline.

    Raises:
        StageError: The config lacks a key 4.0.1's has, or the copy still names a remote file.
    """
    if not isinstance(installed, dict):
        raise StageError(f"the installed {CONFIG_NAME} is not a mapping")
    flow_lm = installed.get("flow_lm")
    lookup = flow_lm.get("lookup_table") if isinstance(flow_lm, dict) else None
    if not isinstance(lookup, dict) or "tokenizer_path" not in lookup:
        raise StageError(f"the installed {CONFIG_NAME} has no flow_lm.lookup_table.tokenizer_path")
    for key in ("weights_path", "weights_path_without_voice_cloning"):
        if key not in installed:
            raise StageError(f"the installed {CONFIG_NAME} has no {key}")
    config = copy.deepcopy(installed)
    config["weights_path"] = str(weights)
    config["weights_path_without_voice_cloning"] = None
    config["flow_lm"]["lookup_table"]["tokenizer_path"] = str(tokenizer)
    remote = {s for s in _strings(config) if s.startswith(_REMOTE_PREFIXES)}
    if remote:
        raise StageError(f"the student's config still names {len(remote)} remote files")
    return config


def _mono(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape(-1, array.shape[-1]).mean(axis=0)
    return array


class PocketSynth:
    """A pocket-tts `TTSModel` holding the student's weights, rendering seeded lines."""

    def __init__(self, model: object, torch: object) -> None:
        self._model = model
        self._torch = torch
        self.sample_rate = int(model.sample_rate)  # type: ignore[attr-defined]

    def voice(self, clip: Path) -> object:
        """Return the voice state of a reference clip (a path not ending `.safetensors`)."""
        return self._model.get_state_for_audio_prompt(clip)  # type: ignore[attr-defined]

    def render(self, voice: object, text: str, seed: int) -> np.ndarray:
        """Render `text` with torch seeded first, so a rerun renders the same audio."""
        torch = self._torch
        torch.manual_seed(seed)  # type: ignore[attr-defined]
        with torch.no_grad():  # type: ignore[attr-defined]
            audio = self._model.generate_audio(voice, text)  # type: ignore[attr-defined]
        return _mono(audio.detach().float().cpu().numpy())


def load_synth(weights: Path, tokenizer: Path, device: str) -> Synth:
    """Load the student with the installed pocket-tts; tests replace this function.

    Raises:
        StageError: The model loaded without voice cloning.
        JobError: pocket-tts or torch failed (`fetch.library_errors`).
    """
    # `library_errors` turns a RuntimeError into JobError, so this module's own StageErrors are
    # raised outside it.
    with fetch.library_errors(LIBRARY):
        import pocket_tts
        import torch
        import yaml
        from pocket_tts.utils import config as pocket_config

        source = Path(pocket_config.CONFIGS_DIR) / CONFIG_NAME
        installed = yaml.safe_load(source.read_text(encoding="utf-8"))
    config = runtime_config(installed, weights, tokenizer)
    with fetch.library_errors(LIBRARY), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / CONFIG_NAME
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        # The config is read during this call; the model keeps no handle on the file.
        model = pocket_tts.TTSModel.load_model(config=path)
    if getattr(model, "has_voice_cloning", None) is not True:
        raise StageError("the student loaded without voice cloning")
    if device != "cpu":
        with fetch.library_errors(LIBRARY):
            model = model.to(device)
    return PocketSynth(model, torch)


def _tokenizer(plan: Mapping[str, object]) -> Path:
    entry = plan.get("tokenizer")
    if not isinstance(entry, dict):
        raise StageError("the score plan names no tokenizer")
    try:
        path = fetch.model_file(contract.FAMILY, str(entry["model"]), str(entry["file"]))
    except JobError as exc:
        raise StageError("the student's tokenizer is not in the cache; run fetch") from exc
    return _checked(path, entry.get("sha256"), "the student's tokenizer")


def run_render(
    data: Path, run_id: str, checkpoint: str, device: str
) -> tuple[list[str], dict[str, object]]:
    """Render every planned line with the checkpoint; return the files written and `renders.json`."""
    rel = contract.scores_rel(run_id, checkpoint)
    plan = read_plan(data, run_id, checkpoint)
    weights = _checked(data / str(plan["weights"]), plan.get("weights_sha256"), "the weights")
    texts = line_texts(data, plan)
    anchor = _anchor(data, plan)
    synth = load_synth(weights, _tokenizer(plan), device)
    with fetch.library_errors(LIBRARY):
        voice = synth.voice(anchor)
    rows: list[dict[str, object]] = []
    for line in plan["lines"]:  # type: ignore[union-attr]
        line_id = str(line["id"])
        with fetch.library_errors(LIBRARY):
            audio = synth.render(voice, texts[line_id], contract.render_seed(line_id))
        audio_rel = f"{rel}/{contract.RENDERS_DIR}/{line_id}.wav"
        _write_wav(data / audio_rel, audio, synth.sample_rate)
        rows.append(
            {
                "id": line_id,
                "set": line["set"],
                "audio": audio_rel,
                "sha256": _sha256(data / audio_rel),
                "seconds": round(len(audio) / synth.sample_rate, 3),
            }
        )
    doc: dict[str, object] = {
        "schema": SCHEMA,
        "run": run_id,
        "checkpoint": checkpoint,
        "sample_rate": synth.sample_rate,
        "lines": rows,
    }
    _write_json(data / rel / contract.RENDERS_NAME, doc)
    print(f"student_score: rendered {len(rows)} lines")
    return [*(str(r["audio"]) for r in rows), f"{rel}/{contract.RENDERS_NAME}"], doc


# `score`.


def load_models(device: str) -> teacher_score.Models:
    """Load the score family's models, as `teacher score` does; tests replace this function."""
    return teacher_score.load_models(device)


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or not denominator > 0.0:
        return None
    return round(numerator / denominator, DIGITS)


def _seed_number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0.0:
        raise StageError(f"the score plan's seed has no {what}")
    return float(value)


def summarise(
    rows: Sequence[Mapping[str, object]],
    seed: Mapping[str, object],
    names: Sequence[str],
    gates: Mapping[str, float],
) -> dict[str, object]:
    """Return the checkpoint's measures, their gates and the verdict.

    `rows` are the renders' measures by line id; `seed` the plan's seed-target numbers
    (`commands.student.seed_reference`): each embedder's SIM median and the neutral F0 SD.

    Raises:
        StageError: The seed lacks a SIM median on an embedder or its F0 SD.
    """
    medians = seed.get("sim_median")
    medians = medians if isinstance(medians, Mapping) else {}
    cers = [float(r["cer"]) for r in rows]  # type: ignore[arg-type]
    worst = max(range(len(rows)), key=lambda k: cers[k])
    sim: dict[str, dict[str, float | None]] = {}
    for name in names:
        seed_median = _seed_number(medians.get(name), f"{name} SIM median")
        rendered = _median([float(r["sim"][name]) for r in rows])  # type: ignore[index]
        sim[name] = {
            "render_median": None if rendered is None else round(rendered, DIGITS),
            "seed_median": round(seed_median, DIGITS),
            "ratio": _ratio(rendered, seed_median),
        }
    ratios = [entry["ratio"] for entry in sim.values()]
    sim_ratio = None if any(r is None for r in ratios) else min(ratios)  # type: ignore[type-var]
    seed_sd = _seed_number(seed.get("f0_std_hz"), "F0 SD")
    render_sd = _median([float(r["f0_std_hz"]) for r in rows])  # type: ignore[arg-type]
    # `bench.cer_max` bounds the highest CER (its note in `gates_default.json`), not the mean.
    metrics: dict[str, object] = {
        contract.CER: round(cers[worst], DIGITS),
        contract.SIM_RATIO: sim_ratio,
        contract.F0_SD_RATIO: _ratio(render_sd, seed_sd),
    }
    failed = contract.gate_failures(metrics, gates)
    return {
        **metrics,
        "pass": not failed,
        "failed": failed,
        "gates": dict(gates),
        "lines": {
            contract.VALID: sum(r["set"] == contract.VALID for r in rows),
            contract.BENCH: sum(r["set"] == contract.BENCH for r in rows),
        },
        "cer_mean": round(statistics.fmean(cers), DIGITS),
        "cer_line_max_id": rows[worst]["id"],
        "seed": {"target": seed.get("target"), "state": seed.get("state")},
        "sim": sim,
        "f0": {
            "seed_sd_hz": round(seed_sd, DIGITS),
            "seed_sd_spread_hz": seed.get("f0_std_spread_hz"),
            "render_sd_median_hz": None if render_sd is None else round(render_sd, DIGITS),
        },
    }


class Scorer:
    """One scoring pass over a checkpoint's renders."""

    def __init__(self, data: Path, run_id: str, checkpoint: str, models: teacher_score.Models):
        self.data = data
        self.run_id = run_id
        self.checkpoint = checkpoint
        self.models = models
        self.plan = read_plan(data, run_id, checkpoint)
        self.names = tuple(e.name for e in models.embedders)
        if set(self.names) != set(self.plan.get("embedders", ())):  # type: ignore[arg-type]
            raise StageError("the embedders loaded are not the plan's")

    def _embed(self, audio: np.ndarray, rate: int) -> dict[str, np.ndarray]:
        return {e.name: np.asarray(e.embed(audio, rate), np.float64) for e in self.models.embedders}

    def _sims(self, audio: np.ndarray, rate: int, anchor: Mapping[str, np.ndarray]) -> dict:
        embedding = self._embed(audio, rate)
        return {n: round(cosine(embedding[n], anchor[n]), DIGITS) for n in self.names}

    def renders(self) -> list[Mapping[str, object]]:
        """Return `renders.json`'s rows, each checked against its sha256 and the plan's ids."""
        folder = self.data / contract.scores_rel(self.run_id, self.checkpoint)
        doc = _read_json(folder / contract.RENDERS_NAME, "renders.json")
        rows = doc.get("lines")
        planned = [line["id"] for line in self.plan["lines"]]  # type: ignore[union-attr,index]
        if not isinstance(rows, list) or [r.get("id") for r in rows] != planned:
            raise StageError("renders.json does not list the plan's lines")
        for row in rows:
            _checked(self.data / str(row["audio"]), row.get("sha256"), "a render")
        return rows

    def run(self) -> tuple[list[str], dict[str, object]]:
        """Measure the renders, gate them against the seed, and write the score; return files."""
        rel = contract.scores_rel(self.run_id, self.checkpoint)
        seed = self.plan.get("seed")
        if not isinstance(seed, dict):
            raise StageError("the score plan names no seed target")
        renders = self.renders()
        texts = line_texts(self.data, self.plan)
        anchor = self._embed(*_read_audio(_anchor(self.data, self.plan)))
        rows: list[dict[str, object]] = []
        for render in renders:
            audio, rate = _read_audio(self.data / str(render["audio"]))
            heard = self.models.asr.transcribe(audio, rate).text
            rows.append(
                {
                    "id": render["id"],
                    "set": render["set"],
                    "cer": round(character_error_rate(texts[str(render["id"])], heard), DIGITS),
                    "sim": self._sims(audio, rate, anchor),
                    "f0_std_hz": round(float(self.models.prosody(audio, rate).f0_std_hz), DIGITS),
                }
            )
        gates = {k: float(v) for k, v in dict(self.plan["gates"]).items()}  # type: ignore[call-overload]
        summary = summarise(rows, seed, self.names, gates)
        doc = {
            "schema": SCHEMA,
            "run": self.run_id,
            "checkpoint": self.checkpoint,
            "weights_sha256": self.plan.get("weights_sha256"),
            **summary,
        }
        folder = self.data / rel
        _write_bytes(
            folder / contract.LINE_SCORES_NAME,
            "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"),
        )
        _write_json(folder / contract.SCORE_NAME, doc)
        verdict = "pass" if doc["pass"] else "fail"
        print(f"student_score: scored {len(rows)} lines, {verdict}")
        return [f"{rel}/{contract.LINE_SCORES_NAME}", f"{rel}/{contract.SCORE_NAME}"], doc


def run_score(
    data: Path, run_id: str, checkpoint: str, models: teacher_score.Models
) -> tuple[list[str], dict[str, object]]:
    """Score the checkpoint's renders; return the root-relative files written and `score.json`."""
    return Scorer(data, run_id, checkpoint, models).run()


# The command line.


def main(argv: Sequence[str] | None = None) -> int:
    """Run the render or the scoring; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.student_score")
    parser.add_argument("--data", required=True, type=Path)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in (contract.RENDER, contract.SCORE):
        sub = modes.add_parser(mode)
        sub.add_argument("--run", required=True)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--device", default="cuda", choices=fetch.DEVICES)
    args = parser.parse_args(argv)
    try:
        fetch.prepare(args.device)
        if args.mode == contract.RENDER:
            written, _doc = run_render(args.data, args.run, args.checkpoint, args.device)
        else:
            models = load_models(args.device)
            written, _doc = run_score(args.data, args.run, args.checkpoint, models)
    except (StageError, JobError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.scores_rel(args.run, args.checkpoint) / contract.OUTPUTS_NAME
    _write_json(outputs, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
