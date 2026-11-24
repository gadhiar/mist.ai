"""`teacher generate` and `teacher score`: OmniVoice takes in the anchor voice, gated by CER and SIM.

`teacher generate --voice V --text T --run R [--rounds-max N] [--until HH:MM] [--resume]` runs one
GPU job in the `teacher-omnivoice` family. It makes `take.n` takes of every neutral line of
`text/<T>/lines.jsonl`, conditioned on the neutral clip and transcript of the identity
`voices/<V>/` (the anchor), into `teachers/omnivoice/<R>/takes/` and `takes.jsonl`. Stage 1
generates neutral only. A take is written unscored (its one reason is `unscored`) until
`teacher score` measures it.

Rounds alternate between the two commands. `teacher generate --resume` skips lines that are
finished: a line with a kept take, a line whose takes wait for `teacher score`, and a line that
has used `--rounds-max` rounds (default `take.rounds_max`) without a kept take. Any other line
gets its next round, or the missing takes of an interrupted one. The job checks its deadline
and SIGTERM between lines: it finishes the current line, records it, and exits 0 with
`status: paused` in `summary.json`. `--until HH:MM` is host local time; the job receives it as
`--deadline-utc`, since a container reads its clock as UTC (`deadline.py`).

Nothing printed carries a line's text, a transcript or a clip's path: only ids, counts, hours
and floors. The stages import the names and helpers below, so host and job share one definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .. import deadline as deadline_mod
from .. import gates as gates_mod
from .. import identity as identity_mod
from .. import jobs
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import JobInput, TakeRecord, read_json, sha256_file, write_json
from ..states import NEUTRAL_STATE
from . import extract as extract_cmd

SCHEMA = 1
TEACHER = "omnivoice"
FAMILY = "teacher-omnivoice"
MODEL_ID = "omnivoice"
GENERATE_MODULE = "scripts.voice_data.stages.teacher_omnivoice"
STATE = NEUTRAL_STATE
PLAN_NAME = "generate-plan.json"
TAKES_NAME = "takes.jsonl"
SUMMARY_NAME = "summary.json"
OUTPUTS_NAME = "outputs.json"
TAKES_DIR = "takes"
LINES_NAME = "lines.jsonl"
UNSCORED = "unscored"
PAUSED, COMPLETE = "paused", "complete"
# What `line_status` returns: a line to generate, or why it is finished for now.
GENERATE, KEPT, TO_SCORE, EXHAUSTED = "generate", "kept", "to-score", "exhausted"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def run_rel(run_id: str) -> str:
    """Return a teacher run's folder, relative to the root."""
    return f"teachers/{TEACHER}/{run_id}"


def lines_rel(version: str) -> str:
    """Return a text version's lines file, relative to the root."""
    return f"text/{version}/{LINES_NAME}"


def take_seed(line_id: str, round_number: int, index: int) -> int:
    """Return the seed of take `index` of a line's round: 31 bits of a sha256 of the three.

    A rerun recomputes the same seed, and through `root.take_id` the same take id, so a take
    already made is recognised and an interrupted round makes only its missing takes.
    """
    key = json.dumps([line_id, round_number, index]).encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:8], 16) & 0x7FFFFFFF


def take_id(line_id: str, seed: int) -> str:
    """Return a neutral OmniVoice take's id."""
    return root_mod.take_id(line_id, STATE, TEACHER, seed)


def take_audio_rel(run_id: str, take: str) -> str:
    """Return a take's WAV, relative to the root."""
    return f"{run_rel(run_id)}/{TAKES_DIR}/{take}.wav"


def is_scored(take: TakeRecord) -> bool:
    """Whether `teacher score` has measured the take."""
    return UNSCORED not in take.reasons


@dataclass(frozen=True, slots=True)
class LineStatus:
    """What a line needs: `generate` with the round and the take indices to make, or a finish."""

    status: str
    round: int = 0
    indices: tuple[int, ...] = ()


def line_status(
    line_id: str, takes: Sequence[TakeRecord], take_n: int, rounds_max: int
) -> LineStatus:
    """Return what a line needs next, from its takes so far.

    A kept take finishes the line. A round missing takes (interrupted before it was recorded in
    full) gets them first; a complete round waits for scoring; a scored round with nothing kept
    is followed by the next round, up to `rounds_max`.
    """
    if any(t.kept for t in takes):
        return LineStatus(KEPT)
    latest = max((t.round for t in takes), default=0)
    if latest:
        seeds = {t.seed for t in takes if t.round == latest}
        missing = tuple(k for k in range(take_n) if take_seed(line_id, latest, k) not in seeds)
        if missing:
            return LineStatus(GENERATE, latest, missing)
        if any(not is_scored(t) for t in takes if t.round == latest):
            return LineStatus(TO_SCORE)
    if latest >= rounds_max:
        return LineStatus(EXHAUSTED)
    return LineStatus(GENERATE, latest + 1, tuple(range(take_n)))


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `teacher` command with its `generate` step."""
    parser = subparsers.add_parser(
        "teacher",
        help="teacher takes in the anchor voice",
        description="Generate OmniVoice takes of the neutral lines in the anchor identity's voice.",
    )
    steps = parser.add_subparsers(dest="step", metavar="STEP", required=True)
    gen = steps.add_parser("generate", help="make take.n takes per line (a GPU job)")
    gen.add_argument("--voice", required=True, help="the anchor identity under voices/")
    gen.add_argument("--text", required=True, help="the text version under text/")
    gen.add_argument("--run", required=True, help="the run id, a folder under teachers/omnivoice")
    gen.add_argument("--rounds-max", type=int, help="rounds per line (default take.rounds_max)")
    gen.add_argument("--until", metavar="HH:MM", help="stop after the line running then (local)")
    gen.add_argument("--resume", action="store_true", help="continue an existing run")
    gen.set_defaults(func=run_generate)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def utc_now() -> datetime:
    """Return the current time in UTC; tests replace this function."""
    return datetime.now(UTC)


def _check_id(value: str, what: str) -> None:
    if not _ID.fullmatch(value) or ".." in value:
        raise InputError(f"refused: a {what} is one folder name")


def _job_id(root: root_mod.Root) -> str:
    now = utc_now()
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


# The anchor.


@dataclass(frozen=True, slots=True)
class Anchor:
    """The anchor identity's neutral clip and transcript, as root-relative paths and sha256s."""

    voice_id: str
    audio: str
    transcript: str
    audio_sha256: str
    transcript_sha256: str

    def to_json(self) -> dict[str, str]:
        """Return the anchor as a plan entry."""
        return {
            "voice": self.voice_id,
            "audio": self.audio,
            "audio_sha256": self.audio_sha256,
            "transcript": self.transcript,
            "transcript_sha256": self.transcript_sha256,
        }


def _inside(directory: Path, rel: object, where: str) -> Path:
    if not isinstance(rel, str) or not rel or "\\" in rel or re.match(r"[A-Za-z]:", rel):
        raise InputError(f"refused: the anchor's {where} is not a relative path")
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts:
        raise InputError(f"refused: the anchor's {where} is not inside the identity")
    path = (directory / rel).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise InputError(f"refused: the anchor's {where} is missing")
    return path


def read_anchor(root: root_mod.Root, voice_id: str) -> Anchor:
    """Read the neutral clip and transcript of identity `voices/<voice_id>/`.

    `voice.json` schema 1 and schema 2 (which adds trained weights) list clips alike, and only
    the neutral clip is read, so this checks that clip by v6's rules (`identity.py`) without
    the schema-1-only `identity.check`. The transcript is hashed and never shown.

    Raises:
        InputError: The id is malformed, the manifest is missing or not schema 1 or 2, it has
            no neutral clip, or a file is missing, outside the identity, or not its sha256.
    """
    identity_mod.check_voice_id(voice_id)
    directory = root.path / identity_mod.voice_rel(voice_id)
    if not (directory / identity_mod.MANIFEST).is_file():
        raise InputError(f"refused: identity {voice_id} has no {identity_mod.MANIFEST}")
    data = read_json(directory / identity_mod.MANIFEST)
    if not isinstance(data, dict) or data.get("schema") not in (1, 2):
        raise InputError(f"refused: identity {voice_id}'s manifest is not schema 1 or 2")
    if isinstance(data.get("schema"), bool) or data.get("voice_id") != voice_id:
        raise InputError(f"refused: identity {voice_id}'s manifest names another voice")
    clips = data.get("clips")
    neutral = [
        c for c in (clips if isinstance(clips, list) else []) if isinstance(c, dict)
        and c.get("state") == STATE
    ]  # fmt: skip
    entry = neutral[0] if len(neutral) == 1 else None
    if entry is None:
        raise InputError(f"refused: identity {voice_id} needs exactly one {STATE} clip")
    audio = _inside(directory, entry.get("audio"), "audio")
    text = _inside(directory, entry.get("transcript"), "transcript")
    audio_sha, text_bytes = sha256_file(audio), text.read_bytes()
    if audio_sha != entry.get("audio_sha256"):
        raise InputError(f"refused: identity {voice_id}'s {STATE} audio does not match its sha256")
    text_sha = identity_mod.sha256_bytes(text_bytes)
    if text_sha != entry.get("transcript_sha256"):
        raise InputError(
            f"refused: identity {voice_id}'s {STATE} transcript does not match its sha256"
        )
    identity_mod.check_transcript(text_bytes, f"the anchor's {STATE} transcript")
    identity_mod.check_wav(audio, f"the anchor's {STATE} audio")
    return Anchor(
        voice_id=voice_id,
        audio=root.rel(audio),
        transcript=root.rel(text),
        audio_sha256=audio_sha,
        transcript_sha256=text_sha,
    )


def read_line_ids(root: root_mod.Root, version: str) -> list[str]:
    """Return the ids of the neutral lines of `text/<version>/lines.jsonl`, in file order.

    Raises:
        InputError: The file is missing, a row is not an object with a string `line_id`,
            `state` and `text`, a line id repeats, or no line is neutral. Errors name row
            numbers, never text.
    """
    rel = lines_rel(version)
    path = root.path / rel
    try:
        rows = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"refused: {rel} is missing or unreadable") from exc
    seen: set[str] = set()
    ids: list[str] = []
    for number, row in enumerate(rows, start=1):
        if not row.strip():
            continue
        try:
            data = json.loads(row)
        except json.JSONDecodeError as exc:
            raise InputError(f"refused: {rel} row {number} is not JSON") from exc
        if not isinstance(data, dict) or not all(
            isinstance(data.get(key), str) for key in ("line_id", "state", "text")
        ):
            raise InputError(f"refused: {rel} row {number} lacks a line_id, state or text")
        if data["line_id"] in seen:
            raise InputError(f"refused: {rel} row {number} repeats a line_id")
        seen.add(data["line_id"])
        if data["state"] == STATE:
            ids.append(data["line_id"])
    if not ids:
        raise InputError(f"refused: text version {version} has no {STATE} line")
    return ids


def _take_n(gates: gates_mod.Gates) -> int:
    key = f"take.n.{STATE}"
    return int(gates.value(key if key in gates.defaults else "take.n"))


def _run(root: root_mod.Root, job: jobs.JobSpec) -> Mapping[str, object]:
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    rel = f"{job.outputs}/{SUMMARY_NAME}"
    try:
        summary = read_json(root.path / rel)
    except InputError as exc:
        raise JobError(f"job {job.job_id}: {rel} is missing") from exc
    if not isinstance(summary, dict):
        raise JobError(f"job {job.job_id}: {rel} is not a JSON object")
    return summary


# `teacher generate`.


def run_generate(args: argparse.Namespace) -> int:
    """Plan the run, run the generation job, and print its summary."""
    root = root_mod.open_root(args.root)
    _check_id(args.run, "run id")
    _check_id(args.text, "text version")
    if args.rounds_max is not None and args.rounds_max < 1:
        raise InputError("refused: --rounds-max is at least 1")
    deadline = deadline_mod.deadline_utc(args.until, now=utc_now()) if args.until else None
    gates = gates_mod.load(root.path)
    anchor = read_anchor(root, args.voice)
    line_ids = read_line_ids(root, args.text)
    run_dir = run_rel(args.run)
    plan_path = root.path / run_dir / PLAN_NAME
    if plan_path.exists() and not args.resume:
        raise InputError(f"refused: run {args.run} exists; pass --resume to continue it")
    if args.resume and not plan_path.exists():
        raise InputError(f"refused: run {args.run} has no plan to resume")
    rounds_max = args.rounds_max or int(gates.value("take.rounds_max"))
    take_n = _take_n(gates)
    if args.resume:
        old = read_json(plan_path)
        if not isinstance(old, dict) or old.get("schema") != SCHEMA:
            raise InputError(f"refused: run {args.run}'s plan is not a schema {SCHEMA} plan")
        if old.get("text") != args.text or old.get("anchor") != anchor.to_json():
            raise InputError(
                f"refused: run {args.run} was planned with another text version or anchor"
            )
        take_n = int(old["take_n"])
        rounds_max = args.rounds_max or int(old["rounds_max"])
    plan = {
        "schema": SCHEMA,
        "run": args.run,
        "teacher": TEACHER,
        "state": STATE,
        "text": args.text,
        "lines_file": lines_rel(args.text),
        "line_ids": line_ids,
        "anchor": anchor.to_json(),
        "take_n": take_n,
        "rounds_max": rounds_max,
        "thresholds": gates.effective(),
    }
    write_json(root, f"{run_dir}/{PLAN_NAME}", plan)
    inputs = (
        JobInput(rel=lines_rel(args.text), sha256=sha256_file(root.path / lines_rel(args.text))),
        JobInput(rel=anchor.audio, sha256=anchor.audio_sha256),
        JobInput(rel=anchor.transcript, sha256=anchor.transcript_sha256),
    )
    extra = ("--deadline-utc", deadline_mod.format_utc(deadline)) if deadline else ()
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=GENERATE_MODULE,
        args=("--run", args.run, *extra),
        inputs=inputs,
        outputs=run_dir,
        gpu=True,
        network=False,
    )
    summary = _run(root, job)
    for line in generate_lines(summary):
        print(line)
    return 0


def _num(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "none" if value is None else str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "?"


def generate_lines(summary: Mapping[str, object]) -> list[str]:
    """Return what `teacher generate` prints: the status, counts and hours."""
    status = summary.get("status") if summary.get("status") in (PAUSED, COMPLETE) else "?"
    by = summary.get("stopped_by")
    reason = f" ({by})" if by in ("deadline", "signal") else ""
    lines = [f"teacher generate run {summary.get('run')}: {status}{reason}"]
    counts = summary.get("line_counts")
    table = counts if isinstance(counts, Mapping) else {}
    lines.append(
        f"  lines {_num(summary.get('lines'))}: kept {_num(table.get(KEPT))}, to score "
        f"{_num(table.get(TO_SCORE))}, out of rounds {_num(table.get(EXHAUSTED))}, to generate "
        f"{_num(table.get(GENERATE))}"
    )
    lines.append(
        f"  takes {_num(summary.get('takes'))} ({_num(summary.get('takes_new'))} new), "
        f"{_num(summary.get('takes_hours'))} h"
    )
    return lines
