"""The student corpus's alignment job: measure, align, split and cap (`train-pocket` job).

Runs as `/opt/venv-train-pocket/bin/python -I -u -m scripts.voice_data.stages.corpus_build
--data /data --version C --device cuda --batch-size 8`, with the text version's `lines.jsonl`,
the teacher run and the seed's clean run mounted read-only, `corpus/<C>/` writable, and the
weight cache at `/cache`. The host's `corpus-plan.json` lists the selected takes, the seed's
confirmed segments and the corpus gates.

Every take is a candidate with its line's text; a seed segment is one when its two ASR
transcripts agree (`transcribe.training_text`) and it speaks no ear-test line. Each candidate's
duration is measured, and all of them go through upstream's CTC aligner once
(`training/scripts/align_data.py`, run in this process with the clone first on `sys.path`,
since `-I` keeps it off). Its `--model` is the pinned wav2vec2 snapshot folder from the offline
cache (`fetch.model_dir`, through `fetch.snapshot_dir`): a snapshot fetched at a pinned commit
writes no `refs/main`, so the repository id alone does not resolve offline. A row the aligner
leaves out, or returns without words, is dropped and counted.

The aligned teacher rows are split by line (`commands.corpus.holdout_lines`), the aligned seed
segments by whole source (`commands.corpus.holdout_sources`, each plan segment naming its
`source_id`), the training seed is capped on the training split's teacher hours
(`commands.corpus.cap_seed`) and repeated `oversample` times, all in the order of the plan's
`random_seed`. The held-out seed rows open the validation split, once each. The splits are
written as upstream's rows: `path` (the clip's path
under `--data`), `duration`, `transcript` and `words`, nothing else.

Nothing this stage prints carries a transcript, a line's text or a clip's path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..commands import clean as clean_contract
from ..commands import corpus as contract
from ..commands import teacher as teacher_contract
from ..errors import InputError, JobError
from ..records import SegmentRecord, TakeRecord, read_jsonl
from ..text_corpus import HoldOut, load_ear_test
from . import fetch
from .transcribe import training_text

SCHEMA = 1
DIGITS = 6
UPSTREAM = fetch.UPSTREAM / "pocket-tts"
ALIGN_MODULE = "training.scripts.align_data"


class StageError(RuntimeError):
    """The corpus cannot be built; the message names ids and counts only."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """An utterance offered to the aligner: its source, group and upstream row fields."""

    kind: str
    key: str  # the take or segment id
    group: str  # the line id for a take, the segment id for a seed segment
    path: str
    duration: float
    transcript: str
    source: str | None = None  # a seed segment's source, held out whole; None for a take

    def row(self) -> dict[str, object]:
        """Return the aligner's input row."""
        return {"path": self.path, "duration": self.duration, "transcript": self.transcript}


def align(source: Path, dest: Path, options: Sequence[str]) -> None:
    """Run upstream's `align_data` over `source` into `dest`, in this process.

    Tests replace this function. The script is a typer CLI, so it is run as `__main__` with its
    argv, and its `SystemExit` is read as an exit code.

    Raises:
        StageError: The aligner exited non-zero.
    """
    at = list(options).index("--device") if "--device" in options else -1
    fetch.prepare(options[at + 1] if at >= 0 else "cpu", "transformers")
    fetch.on_path(UPSTREAM)
    saved = sys.argv
    sys.argv = [ALIGN_MODULE, str(source), str(dest), *options]
    code: object = 0
    try:
        runpy.run_module(ALIGN_MODULE, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = exc.code
    finally:
        sys.argv = saved
    if code not in (0, None):
        raise StageError(f"the aligner exited {code}")


def aligner_model() -> Path:
    """Return the aligner's snapshot folder at its pinned revision, from the offline cache."""
    return fetch.model_dir(contract.FAMILY, contract.ALIGNER_MODEL)


def _read_plan(data: Path, version: str) -> dict[str, object]:
    path = data / contract.corpus_rel(version) / contract.PLAN_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the corpus plan is missing or not JSON") from exc
    if not isinstance(plan, dict) or (plan.get("schema"), plan.get("version")) != (
        SCHEMA,
        version,
    ):
        raise StageError("the corpus plan is not this version's schema 1 plan")
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


def _seconds(path: Path) -> float:
    import soundfile

    try:
        info = soundfile.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise StageError("a clip is unreadable") from exc
    return round(info.frames / info.samplerate, DIGITS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes(path, "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))


def _hours(seconds: float) -> float:
    return round(seconds / 3600.0, DIGITS)


class Builder:
    """One corpus build: candidates, alignment, split, cap and the written splits."""

    def __init__(self, data: Path, version: str, device: str, batch_size: int) -> None:
        self.data = data
        self.version = version
        self.device = device
        self.batch_size = batch_size
        self.plan = _read_plan(data, version)
        random_seed = self.plan.get("random_seed")
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise StageError("the corpus plan records no integer random_seed")
        self.random_seed = random_seed
        share = self.plan.get("seed_val_share")
        if isinstance(share, bool) or not isinstance(share, int | float):
            raise StageError("the corpus plan records no numeric seed_val_share")
        self.seed_val_share = float(share)
        self.rel = contract.corpus_rel(version)
        self.hold_out = HoldOut(load_ear_test())
        self.left_out = {"untrusted_text": 0, "ear_test": 0, "share_cap": 0}

    def _path(self, rel: str) -> str:
        return str(self.data / rel)

    def teacher(self) -> list[Candidate]:
        """Return a candidate per planned take, checked against the run's `takes.jsonl`."""
        takes_path = (
            self.data / teacher_contract.run_rel(str(self.plan["takes_run"]))
            / teacher_contract.TAKES_NAME
        )  # fmt: skip
        try:
            records = {t.take_id: t for t in read_jsonl(takes_path, TakeRecord)}
        except InputError as exc:
            raise StageError("takes.jsonl is missing or malformed") from exc
        texts = _texts(self.data, str(self.plan["lines_file"]))
        found = []
        for entry in self.plan["takes"]:  # type: ignore[attr-defined]
            take = records.get(entry["take_id"])
            if take is None or not take.selected or take.audio != entry["audio"]:
                raise StageError(f"take {entry['take_id']} is not a selected take of the run")
            text = texts.get(take.line_id)
            if text is None:
                raise StageError(f"line {take.line_id} is not in the lines file")
            if self.hold_out.holds(text):
                raise StageError(f"take {take.take_id} speaks an ear-test line")
            found.append(
                Candidate(contract.TEACHER, take.take_id, take.line_id, self._path(take.audio),
                          _seconds(self.data / take.audio), text)
            )  # fmt: skip
        return found

    def seed(self) -> list[Candidate]:
        """Return a candidate per planned segment with trusted text and no ear-test line."""
        seed_run = self.plan.get("seed_run")
        if seed_run is None:
            return []
        path = self.data / clean_contract.run_rel(str(seed_run)) / clean_contract.SEGMENTS_NAME
        try:
            records = {s.segment_id: s for s in read_jsonl(path, SegmentRecord)}
        except InputError as exc:
            raise StageError("the seed's segments.jsonl is missing or malformed") from exc
        found = []
        for entry in self.plan["segments"]:  # type: ignore[attr-defined]
            source = entry.get("source_id")
            if not isinstance(source, str) or not source:
                raise StageError(f"segment {entry['segment_id']} has no source_id in the plan")
            record = records.get(entry["segment_id"])
            if record is None or record.audio != entry["audio"] or record.source_id != source:
                raise StageError(f"segment {entry['segment_id']} is not in the seed run")
            text = training_text(record)
            if not text:
                self.left_out["untrusted_text"] += 1
                continue
            if self.hold_out.holds(text):
                self.left_out["ear_test"] += 1
                continue
            found.append(
                Candidate(contract.SEED, record.segment_id, record.segment_id,
                          self._path(record.audio), _seconds(self.data / record.audio), text,
                          record.source_id)
            )  # fmt: skip
        return found

    def align_all(self, candidates: Sequence[Candidate]) -> dict[str, list[dict[str, object]]]:
        """Return each aligned candidate's words by path; its other fields stay the plan's."""
        folder = self.data / self.rel / contract.ALIGN_DIR
        source, dest = folder / contract.ALIGN_IN, folder / contract.ALIGN_OUT
        _write_jsonl(source, [c.row() for c in candidates])
        if dest.exists():
            dest.unlink()
        options = [
            "--model", str(aligner_model()),
            "--device", self.device,
            "--batch-size", str(self.batch_size),
        ]  # fmt: skip
        align(source, dest, options)
        try:
            rows = [json.loads(r) for r in dest.read_text(encoding="utf-8").splitlines() if r]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StageError("the aligner wrote no readable output") from exc
        words: dict[str, list[dict[str, object]]] = {}
        for row in rows:
            timed = row.get("words") if isinstance(row, dict) else None
            if isinstance(timed, list) and timed and all(isinstance(w, dict) for w in timed):
                words[str(row["path"])] = timed
        return words

    def run(self) -> tuple[list[str], dict[str, object]]:
        """Build the corpus; return the root-relative files written and the summary."""
        teacher, seed = self.teacher(), self.seed()
        words = self.align_all([*teacher, *seed])
        teacher_ok = [c for c in teacher if c.path in words]
        seed_ok = [c for c in seed if c.path in words]
        random_seed = self.random_seed
        fraction = float(self.plan["val_fraction"])  # type: ignore[arg-type]
        held = contract.holdout_lines((c.group for c in teacher_ok), fraction, seed=random_seed)
        train = [c for c in teacher_ok if c.group not in held]
        valid = [c for c in teacher_ok if c.group in held]
        train_s = sum(c.duration for c in train)
        # The seed's hold-out counts only segments that passed the text rules and aligned, so a
        # dropped segment never moves it; the share cap then applies to the training seed alone.
        by_source: dict[str, float] = {}
        for c in seed_ok:
            by_source[str(c.source)] = by_source.get(str(c.source), 0.0) + c.duration
        seed_hold = contract.holdout_sources(by_source, self.seed_val_share, seed=random_seed)
        seed_valid = [c for c in seed_ok if c.source in seed_hold.sources]
        seed_train = [c for c in seed_ok if c.source not in seed_hold.sources]
        oversample = int(self.plan["oversample"])  # type: ignore[arg-type]
        share_max = float(self.plan["seed_share_max"])  # type: ignore[arg-type]
        offered = [(c.key, c.duration) for c in seed_train]
        fits = set(contract.cap_seed(offered, train_s, share_max, oversample, seed=random_seed))
        seed_kept = [c for c in seed_train if c.key in fits]
        self.left_out["share_cap"] = len(seed_train) - len(seed_kept)

        def row(c: Candidate) -> dict[str, object]:
            return {"path": c.path, "duration": c.duration, "transcript": c.transcript,
                    "words": words[c.path]}  # fmt: skip

        train_rows = [row(c) for c in train]
        train_rows += [row(c) for _ in range(oversample) for c in seed_kept]
        # Held-out seed rows first and once: a validation pass over whole batches in file order
        # drops a short tail, which is then teacher rows, never her real voice.
        valid_rows = [row(c) for c in seed_valid] + [row(c) for c in valid]
        ear = sum(self.hold_out.holds(str(r["transcript"])) for r in train_rows + valid_rows)
        if ear:
            raise StageError(f"{ear} corpus rows speak an ear-test line")
        train_rel = f"{self.rel}/{contract.TRAIN_NAME}"
        valid_rel = f"{self.rel}/{contract.VALID_NAME}"
        _write_jsonl(self.data / train_rel, train_rows)
        _write_jsonl(self.data / valid_rel, valid_rows)
        seed_s = oversample * sum(c.duration for c in seed_kept)
        teacher_s = sum(c.duration for c in teacher_ok)
        summary: dict[str, object] = {
            "schema": SCHEMA,
            "version": self.version,
            "random_seed": random_seed,
            "teacher_hours": _hours(teacher_s),
            "teacher_train_hours": _hours(train_s),
            "valid_hours": _hours(teacher_s - train_s),
            "seed_hours": _hours(seed_s),
            "seed_share": round(seed_s / (train_s + seed_s), DIGITS) if train_s + seed_s else 0.0,
            "seed_share_max": share_max,
            "oversample": oversample,
            "val_fraction": fraction,
            "teacher_takes": len(teacher_ok),
            "teacher_lines": len({c.group for c in teacher_ok}),
            "valid_lines": len({c.group for c in valid}),
            "seed_segments": len(seed_kept),
            "rows": {"train": len(train_rows), "valid": len(valid_rows)},
            "unaligned": {
                contract.TEACHER: len(teacher) - len(teacher_ok),
                contract.SEED: len(seed) - len(seed_ok),
            },
            "seed_left_out": dict(self.left_out),
            "ear_test_lines": ear,
            "seed_val_share": self.seed_val_share,
            "seed_val_sources": len(seed_hold.sources),
            "seed_val_segments": len(seed_valid),
            "seed_val_hours": _hours(sum(c.duration for c in seed_valid)),
            "seed_val_reason": seed_hold.reason,
        }
        summary_rel = f"{self.rel}/{contract.SUMMARY_NAME}"
        payload = json.dumps(summary, indent=2) + "\n"
        _write_bytes(self.data / summary_rel, payload.encode("utf-8"))
        print(
            f"corpus_build: {len(teacher_ok)} of {len(teacher)} takes and {len(seed_ok)} of "
            f"{len(seed)} seed segments aligned"
        )
        return [train_rel, valid_rel, summary_rel], summary


def run(
    data: Path, version: str, device: str, batch_size: int
) -> tuple[list[str], dict[str, object]]:
    """Build corpus `version`; return the root-relative files written and the summary."""
    return Builder(data, version, device, batch_size).run()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.corpus_build")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--device", default=contract.DEVICE, choices=fetch.DEVICES)
    parser.add_argument("--batch-size", type=int, default=contract.BATCH_SIZE)
    args = parser.parse_args(argv)
    try:
        written, _summary = run(args.data, args.version, args.device, args.batch_size)
    except (StageError, JobError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.corpus_rel(args.version) / contract.OUTPUTS_NAME
    payload = json.dumps({"schema": SCHEMA, "files": files}, indent=2) + "\n"
    _write_bytes(outputs, payload.encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
