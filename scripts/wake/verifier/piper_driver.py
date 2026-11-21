"""Speak every text with piper-sample-generator, the same number of times each.

    python3 scripts/wake/verifier/piper_driver.py --texts <texts.json> --generator <dir> \
        --python <exe> --per-text <n> --out <dir>

This runs in pe-v2 H6's WSL environment, beside that piper-sample-generator checkout (MIT, with
its LibriTTS-R voice, CC BY 4.0), so it imports the standard library only and runs as a file,
not as a module of this repository. It first runs `generate_samples.py --help` and refuses
(exit 2) unless the help names both `--max-samples` and `--output-dir`, the only two flags it
passes; then, for each text, `<python> generate_samples.py <text> --max-samples <n>
--output-dir <out>/<text id>`, from the generator's directory.

Every text is spoken `--per-text` times, and `synth.py` writes as many positive texts as
negative ones, so both classes get the same number of clips; a text with any other count of
clips fails the run (exit 1). `clips.jsonl` lists each clip with its text id, label and the
format its header states: the generator's rate is read from its output, never assumed, and
`mine.py` resamples to 16 kHz before mixing. `meta.json` records the generator and the count.
`--out` is written new, never over a directory. Exit 0 written, 1 a generation failed, 2
refused.
"""

import argparse
import hashlib
import json
import subprocess  # nosec B404 - runs the generator named on the command line
import sys
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

SCRIPT = "generate_samples.py"
REQUIRED_FLAGS = ("--max-samples", "--output-dir")
CLIPS = "clips.jsonl"
META = "meta.json"

Runner = Callable[..., subprocess.CompletedProcess]


class DriverError(Exception):
    """A failure the run reports; `code` is its exit code."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def _texts(path: Path) -> list[dict]:
    try:
        texts = json.loads(path.read_text(encoding="utf-8"))["texts"]
    except (OSError, ValueError, KeyError) as exc:
        raise DriverError(f"{path.name} is not a texts file: {exc}", 2) from exc
    labels = [item.get("label") for item in texts]
    if set(labels) != {"positive", "negative"} or labels.count("positive") * 2 != len(labels):
        raise DriverError(f"{path.name} must hold as many positive texts as negative ones", 2)
    return texts


def _check_help(run: Runner, python: str, generator: Path) -> None:
    result = run(
        [python, str(generator / SCRIPT), "--help"],
        cwd=str(generator),
        capture_output=True,
        text=True,
        check=False,
    )
    shown = f"{result.stdout or ''}\n{result.stderr or ''}"
    missing = [flag for flag in REQUIRED_FLAGS if flag not in shown]
    if result.returncode != 0 or missing:
        raise DriverError(
            f"{SCRIPT} --help exited {result.returncode} and does not name {missing}; "
            "this driver passes only --max-samples and --output-dir",
            2,
        )


def _clips(directory: Path, text: dict, out: Path) -> list[dict]:
    rows = []
    for path in sorted(directory.glob("*.wav")):
        try:
            with wave.open(str(path), "rb") as handle:
                rate, channels = handle.getframerate(), handle.getnchannels()
                width, frames = handle.getsampwidth(), handle.getnframes()
        except (wave.Error, EOFError, OSError) as exc:
            raise DriverError(f"{path.name} is not a PCM WAV file: {exc}", 1) from exc
        rows.append(
            {
                "file": path.relative_to(out).as_posix(),
                "text_id": text["id"],
                "label": text["label"],
                "speaker": text["id"],
                "rate": rate,
                "channels": channels,
                "width": width,
                "frames": frames,
            }
        )
    return rows


def generate(args: argparse.Namespace, run: Runner) -> int:
    """Check the generator's flags, speak every text and write the listing."""
    if args.out.exists():
        raise DriverError("--out already exists; the clips are written new", 2)
    if args.per_text < 1:
        raise DriverError("--per-text must be at least 1", 2)
    texts = _texts(args.texts)
    _check_help(run, args.python, args.generator)
    args.out.mkdir(parents=True)
    rows: list[dict] = []
    for text in texts:
        directory = args.out / text["id"]
        result = run(
            [
                args.python,
                str(args.generator / SCRIPT),
                text["text"],
                "--max-samples",
                str(args.per_text),
                "--output-dir",
                str(directory),
            ],
            cwd=str(args.generator),
            check=False,
        )
        if result.returncode != 0:
            raise DriverError(f"{SCRIPT} exited {result.returncode} on text {text['id']}", 1)
        clips = _clips(directory, text, args.out)
        if len(clips) != args.per_text:
            raise DriverError(
                f"text {text['id']} gave {len(clips)} clips, not {args.per_text}; the classes "
                "would differ in size",
                1,
            )
        rows.extend(clips)
    (args.out / CLIPS).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    meta = {
        "version": 1,
        "synthesiser": "piper",
        "generator": args.generator.name,
        "per_text": args.per_text,
        "texts_sha256": hashlib.sha256(args.texts.read_bytes()).hexdigest(),
        "clips": len(rows),
    }
    (args.out / META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    labels = [row["label"] for row in rows]
    print(f"piper: {labels.count('positive')} positive, {labels.count('negative')} negative")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piper_driver.py", description=__doc__)
    parser.add_argument("--texts", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--python", required=True)
    parser.add_argument("--per-text", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, run: Runner = subprocess.run) -> int:
    """Run the driver; see the module docstring for the exit codes."""
    args = _parser().parse_args(argv)
    try:
        return generate(args, run)
    except DriverError as exc:
        print(f"piper_driver: {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
