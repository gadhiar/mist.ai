"""OmniVoice takes of the neutral lines in the anchor's voice, run as the `teacher-omnivoice` job.

Runs as `/opt/venv-teacher-omnivoice/bin/python -I -m scripts.voice_data.stages.teacher_omnivoice
--data /data --run R [--deadline-utc YYYY-MM-DDTHH:MM:SSZ]`, with the text version's
`lines.jsonl` and the anchor's neutral clip and transcript mounted read-only and
`teachers/omnivoice/<R>/` writable. The host's `generate-plan.json` names the lines by id, the
anchor's files with their sha256s, `take_n` and `rounds_max`.

OmniVoice 0.2.1 (pinned in `requirements-omnivoice.txt`) is called as its package README and
source give it: `OmniVoice.from_pretrained(<snapshot folder>, device_map="cuda:0",
dtype=torch.float16)`, one `create_voice_clone_prompt(ref_audio=<path>, ref_text=<transcript>)`
per job (with `ref_text` given, it runs no ASR), then per take
`generate(text=..., voice_clone_prompt=...)`, which returns a list of 1-D arrays at
`model.sampling_rate`. Neutral needs no `instruct` (the family's neutral control is an empty
instruction). The package takes no seed: sampling draws from torch's global generator, so each
take seeds it with `torch.manual_seed` and `torch.cuda.manual_seed_all` immediately before its
call. The weights resolve offline to the pinned snapshot folder (`fetch.model_dir`), never a
repository id.

Lines run in plan order; `commands/teacher.line_status` says which need takes. The deadline and
SIGTERM are checked before each line: the line running finishes, `takes.jsonl` is rewritten
after every line, and the job writes `summary.json` with `status: paused` and exits 0. Nothing
this stage prints carries a line's text or a transcript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Protocol

import numpy as np

from .. import deadline as deadline_mod
from ..commands import teacher as contract
from ..errors import InputError
from ..records import Prosody, TakeRecord, read_jsonl
from . import fetch

SCHEMA = 1
SUBTYPE = "PCM_16"
DEADLINE, SIGNAL = "deadline", "signal"
# An unscored take's placeholders: `reasons` says it is unscored; the numbers mean nothing yet.
UNSCORED_PROSODY = Prosody(f0_mean_hz=0.0, f0_std_hz=0.0, rate_sps=0.0, energy_db=0.0)


class StageError(RuntimeError):
    """The run cannot proceed; the message names ids and counts only."""


class Teacher(Protocol):
    """What the stage needs of a zero-shot teacher; `OmniVoiceTeacher` is the real one."""

    @property
    def sample_rate(self) -> int:
        """The rate of the audio `take` returns."""
        ...

    def prompt(self, audio: Path, transcript: str) -> object:
        """Return the voice prompt built from the reference clip and its transcript."""
        ...

    def take(self, text: str, prompt: object, seed: int) -> np.ndarray:
        """Return one take of `text` in the prompt's voice, sampled from `seed`."""
        ...


class OmniVoiceTeacher:
    """OmniVoice behind the `Teacher` protocol; `torch` is the module the model runs on."""

    def __init__(self, model: object, torch: object) -> None:
        self._model = model
        self._torch = torch

    @property
    def sample_rate(self) -> int:
        """The model's `sampling_rate` (24000 for 0.2.1)."""
        return int(self._model.sampling_rate)  # type: ignore[attr-defined]

    def prompt(self, audio: Path, transcript: str) -> object:
        """Build the voice-clone prompt once; the transcript is passed so no ASR runs."""
        with fetch.library_errors(contract.MODEL_ID):
            return self._model.create_voice_clone_prompt(  # type: ignore[attr-defined]
                ref_audio=str(audio), ref_text=transcript
            )

    def take(self, text: str, prompt: object, seed: int) -> np.ndarray:
        """Seed torch's generators, then generate one take of `text`."""
        self._torch.manual_seed(seed)  # type: ignore[attr-defined]
        self._torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        with fetch.library_errors(contract.MODEL_ID):
            out = self._model.generate(  # type: ignore[attr-defined]
                text=text, voice_clone_prompt=prompt
            )
        if not out:
            raise StageError("the teacher returned no audio")
        return np.asarray(out[0], dtype=np.float32).reshape(-1)


def load_teacher() -> OmniVoiceTeacher:
    """Load OmniVoice from the pinned snapshot, offline, in float16 on the first GPU.

    Imported here and not at the top, so a module walk imports no model package.
    """
    fetch.prepare("cuda", "omnivoice", "transformers")
    folder = fetch.model_dir(contract.FAMILY, contract.MODEL_ID)
    with fetch.library_errors(contract.MODEL_ID):
        import torch
        from omnivoice import OmniVoice

        model = OmniVoice.from_pretrained(str(folder), device_map="cuda:0", dtype=torch.float16)
    return OmniVoiceTeacher(model, torch)


def utc_now() -> datetime:
    """Return the current time in UTC; tests replace this function."""
    return datetime.now(UTC)


class Stop:
    """When to stop between lines: the deadline has come, or SIGTERM arrived."""

    def __init__(self, deadline: datetime | None, clock: Callable[[], datetime]) -> None:
        self._deadline = deadline
        self._clock = clock
        self.signalled = False

    def on_signal(self, signum: int, frame: FrameType | None) -> None:
        """The SIGTERM handler: record it; the line running finishes first."""
        self.signalled = True

    def reason(self) -> str | None:
        """Return `signal` or `deadline` when the job should stop now, else `None`."""
        if self.signalled:
            return SIGNAL
        if self._deadline is not None and self._clock() >= self._deadline:
            return DEADLINE
        return None


# Files.


def _write_bytes(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _write_audio(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile

    part = path.with_name(path.name + ".part")
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    soundfile.write(str(part), clipped, rate, format="WAV", subtype=SUBTYPE)
    os.replace(part, path)


def _seconds(path: Path) -> float:
    import soundfile

    info = soundfile.info(str(path))
    return info.frames / info.samplerate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_plan(data: Path, run_id: str) -> dict[str, object]:
    path = data / contract.run_rel(run_id) / contract.PLAN_NAME
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the generate plan is missing or not JSON") from exc
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA or plan.get("run") != run_id:
        raise StageError("the generate plan is not this run's schema 1 plan")
    for key in ("line_ids", "anchor", "lines_file", "take_n", "rounds_max"):
        if key not in plan:
            raise StageError(f"the generate plan has no {key}")
    return plan


def read_lines(data: Path, lines_file: str, line_ids: Sequence[str]) -> dict[str, str]:
    """Return the text of each planned line, by id.

    Raises:
        StageError: The file is unreadable, or a planned line is missing or not neutral.
    """
    try:
        rows = (data / lines_file).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(row) for row in rows if row.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the lines file is missing or not JSONL") from exc
    texts = {
        str(row["line_id"]): str(row["text"])
        for row in parsed
        if isinstance(row, dict) and row.get("state") == contract.STATE and "text" in row
    }
    missing = [line_id for line_id in line_ids if line_id not in texts]
    if missing:
        raise StageError(f"{len(missing)} planned lines are not neutral lines of the lines file")
    return {line_id: texts[line_id] for line_id in line_ids}


def read_anchor(data: Path, anchor: Mapping[str, object]) -> tuple[Path, str]:
    """Return the anchor clip's path and its transcript, each checked against its sha256.

    Raises:
        StageError: A file is missing or no longer matches the plan.
    """
    audio = data / str(anchor["audio"])
    text = data / str(anchor["transcript"])
    if not audio.is_file() or _sha256(audio) != anchor["audio_sha256"]:
        raise StageError("the anchor's audio is missing or does not match the plan")
    if not text.is_file() or _sha256(text) != anchor["transcript_sha256"]:
        raise StageError("the anchor's transcript is missing or does not match the plan")
    return audio, text.read_text(encoding="utf-8").strip()


def read_takes(path: Path) -> list[TakeRecord]:
    """Return the run's takes so far; none before the first line is recorded.

    Raises:
        StageError: `takes.jsonl` exists but is malformed.
    """
    if not path.is_file():
        return []
    try:
        return read_jsonl(path, TakeRecord)
    except InputError as exc:
        raise StageError("takes.jsonl is malformed") from exc


def unscored_take(run_id: str, line_id: str, round_number: int, seed: int) -> TakeRecord:
    """Return a new take's record, unscored until `teacher score` measures it."""
    take = contract.take_id(line_id, seed)
    return TakeRecord(
        take_id=take,
        line_id=line_id,
        state=contract.STATE,
        teacher=contract.TEACHER,
        seed=seed,
        round=round_number,
        audio=contract.take_audio_rel(run_id, take),
        sim={},
        cer=1.0,
        prosody=UNSCORED_PROSODY,
        arousal=0.0,
        kept=False,
        selected=False,
        reasons=(contract.UNSCORED,),
    )


def run(
    data: Path, run_id: str, load: Callable[[], Teacher], stop: Stop
) -> tuple[list[str], dict[str, object]]:
    """Generate the takes the plan's lines need; return the files written and the summary.

    The teacher is loaded, and the prompt built, only when a line needs a take.
    """
    plan = _read_plan(data, run_id)
    line_ids = [str(x) for x in plan["line_ids"]]  # type: ignore[attr-defined]
    take_n, rounds_max = int(plan["take_n"]), int(plan["rounds_max"])  # type: ignore[arg-type]
    texts = read_lines(data, str(plan["lines_file"]), line_ids)
    anchor_audio, transcript = read_anchor(data, plan["anchor"])  # type: ignore[arg-type]
    run_dir = data / contract.run_rel(run_id)
    takes_path = run_dir / contract.TAKES_NAME
    takes = read_takes(takes_path)
    unknown = {t.line_id for t in takes} - set(line_ids)
    if unknown:
        raise StageError(f"takes.jsonl holds {len(unknown)} lines the plan does not")
    (run_dir / contract.TAKES_DIR).mkdir(parents=True, exist_ok=True)
    by_line: dict[str, list[TakeRecord]] = {line_id: [] for line_id in line_ids}
    for take in takes:
        by_line[take.line_id].append(take)
    # Each record is serialised once; `takes.jsonl` is rewritten whole after every line, so a
    # reader or a killed job never meets a half-written last line.
    rows = [json.dumps(t.to_json()) + "\n" for t in takes]
    teacher: Teacher | None = None
    prompt: object = None
    new = 0
    stopped_by: str | None = None
    for line_id in line_ids:
        status = contract.line_status(line_id, by_line[line_id], take_n, rounds_max)
        if status.status != contract.GENERATE:
            continue
        stopped_by = stop.reason()
        if stopped_by is not None:
            break
        if teacher is None:
            teacher = load()
            prompt = teacher.prompt(anchor_audio, transcript)
        for index in status.indices:
            seed = contract.take_seed(line_id, status.round, index)
            record = unscored_take(run_id, line_id, status.round, seed)
            audio = teacher.take(texts[line_id], prompt, seed)
            _write_audio(data / record.audio, audio, teacher.sample_rate)
            takes.append(record)
            by_line[line_id].append(record)
            rows.append(json.dumps(record.to_json()) + "\n")
            new += 1
        _write_bytes(takes_path, "".join(rows).encode("utf-8"))
    if not takes_path.exists():
        _write_bytes(takes_path, b"")
    counts = {key: 0 for key in (contract.KEPT, contract.TO_SCORE, contract.EXHAUSTED)}
    counts[contract.GENERATE] = 0
    for line_id in line_ids:
        counts[contract.line_status(line_id, by_line[line_id], take_n, rounds_max).status] += 1
    seconds = sum(_seconds(data / t.audio) for t in takes)
    summary: dict[str, object] = {
        "schema": SCHEMA,
        "run": run_id,
        "step": "generate",
        "status": contract.PAUSED if stopped_by else contract.COMPLETE,
        "stopped_by": stopped_by,
        "lines": len(line_ids),
        "line_counts": counts,
        "takes": len(takes),
        "takes_new": new,
        "takes_hours": round(seconds / 3600.0, 4),
    }
    _write_json(run_dir / contract.SUMMARY_NAME, summary)
    rel = contract.run_rel(run_id)
    written = [f"{rel}/{contract.TAKES_NAME}", f"{rel}/{contract.SUMMARY_NAME}"]
    written += [t.audio for t in takes]
    print(f"teacher_omnivoice: {summary['status']}, {new} new takes")
    return written, summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage under a SIGTERM handler; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.teacher_omnivoice")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--deadline-utc", type=deadline_mod.parse_utc)
    args = parser.parse_args(argv)
    stop = Stop(args.deadline_utc, utc_now)
    previous = signal.signal(signal.SIGTERM, stop.on_signal)
    try:
        written, _summary = run(args.data, args.run, load_teacher, stop)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / contract.run_rel(args.run) / contract.OUTPUTS_NAME
    _write_json(outputs, {"schema": SCHEMA, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
