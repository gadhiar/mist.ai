"""The voice design probe: render candidate voices from text descriptions, in the qwen3 image.

Invocation, from `/work` (the repository, mounted read-only):

    python -m scripts.model_bench.speech.probes.design_probe --model /models/voice-design
        --model-pin <repo@revision> --descriptions /inputs/descriptions.json
        --lines /inputs/lines.json --seeds N --out /out

Both inputs are JSON lists of `{"id", "text"}` with unique ids matching `[a-z0-9][a-z0-9-]*`.
For each description, in file order, and each seed from 1 to N, the probe seeds torch and calls
`qwen_tts.Qwen3TTSModel.generate_voice_design` once, with the description as `instruct` and every
line's text, joined in file order, as `text`. One call per candidate keeps each candidate one
voice: a voice design call designs its voice afresh, so calls made line by line could drift. The
candidate is `<out>/cand-<description id>-s<seed>.wav`, mono 16-bit PCM at the model's own rate,
and one `design-log.jsonl` row records the candidate, description id, seed, model pin, package
versions, language, line ids, sample rate, duration and peak.

Neither a description's text nor a line's text is ever printed or logged: stdout carries
`CANDIDATE <file name>` per candidate and `DESIGN_DONE <count>`, and every refusal and failure
names ids, indexes and paths only.

The entry point's signature was read from the qwen3 bench image (host step H13):
`from_pretrained(pretrained_model_name_or_path, **kwargs)` and `generate_voice_design(text,
instruct, language=None, non_streaming_mode=True, **kwargs) -> (list of arrays, rate)`. The
loading keyword arguments, `device_map` and `dtype`, are not part of that signature; they are
the ones the Qwen3-TTS model card shows, unverified against the installed package until the
first design session runs.

Exit codes: 0 done, 2 refused before the model loads (a malformed input, an output that already
exists), 1 failed. qwen_tts, torch and numpy are imported only when the model loads, so the host
can import this module to share its input checks.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import sys
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from scripts.model_bench.speech import core
else:
    from .. import core

Importer = Callable[[str], object]

ENTRY_MODULE = "qwen_tts"
ENTRY_CLASS = "Qwen3TTSModel"
LANGUAGE = "English"
DEVICE_MAP = "cuda:0"
DTYPE = "bfloat16"
LOG_FILE = "design-log.jsonl"
CANDIDATE_GLOB = "cand-*.wav"
CANDIDATE_PREFIX = "CANDIDATE "
DONE_PREFIX = "DESIGN_DONE"
PACKAGES = ("qwen-tts", "torch", "transformers", "numpy")

_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")


class RefusedError(core.SpeechBenchError):
    """A request the probe, or the host before any container, refuses."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Item:
    """One description or line. The text stays out of `repr`, so no error message can show it."""

    id: str
    text: str = field(repr=False)


def load_items(path: Path, what: str) -> list[Item]:
    """`path` as a JSON list of `{"id", "text"}`, refusing anything else.

    Messages name the file, an index and the rule broken, never a value read from the file: an
    id that breaks the rule may itself be text.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RefusedError(f"{what} {path} cannot be read ({type(exc).__name__})") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RefusedError(f"{what} {path} is not JSON") from None
    if not isinstance(doc, list) or not doc:
        raise RefusedError(f'{what} {path} is not a non-empty JSON list of {{"id", "text"}}')
    items: list[Item] = []
    seen: set[str] = set()
    for index, raw in enumerate(doc):
        where = f"{what} {path}: item {index}"
        if not isinstance(raw, dict) or set(raw) != {"id", "text"}:
            raise RefusedError(f'{where} is not an object with exactly "id" and "text"')
        item_id, text = raw["id"], raw["text"]
        if not isinstance(item_id, str) or not _ID_RE.fullmatch(item_id):
            raise RefusedError(f"{where}: the id does not match {_ID_RE.pattern}")
        if not isinstance(text, str) or not text.strip():
            raise RefusedError(f"{where} ({item_id}): the text is not a non-empty string")
        if item_id in seen:
            raise RefusedError(f"{where}: the id {item_id} repeats")
        seen.add(item_id)
        items.append(Item(item_id, text))
    return items


def candidate_name(description_id: str, seed: int) -> str:
    return f"cand-{description_id}-s{seed}.wav"


def seeds_for(count: int) -> range:
    """The seeds of `--seeds N`: 1 to N."""
    if count < 1:
        raise RefusedError(f"--seeds must be at least 1, got {count}")
    return range(1, count + 1)


def existing_outputs(out: Path) -> list[str]:
    """The names in `out` a run would write: any candidate, and the design log."""
    if not out.is_dir():
        return []
    names = sorted(p.name for p in out.glob(CANDIDATE_GLOB))
    if (out / LOG_FILE).exists():
        names.append(LOG_FILE)
    return names


def spoken_text(lines: Sequence[Item]) -> str:
    return " ".join(line.text.strip() for line in lines)


def package_versions(packages: Sequence[str] = PACKAGES) -> dict[str, str | None]:
    """Installed version of each package, `None` for one that is not installed."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


class Designer:
    """The VoiceDesign model behind `qwen_tts`, loaded once and seeded before every call."""

    def __init__(self, model_dir: Path, importer: Importer):
        self._import = importer
        self._torch = importer("torch")
        self._np = importer("numpy")
        package = importer(ENTRY_MODULE)
        self._model = getattr(package, ENTRY_CLASS).from_pretrained(
            str(model_dir), device_map=DEVICE_MAP, dtype=getattr(self._torch, DTYPE)
        )

    def design(self, *, description: str, text: str, seed: int) -> tuple[object, int]:
        """One candidate: a mono float32 array and its sample rate."""
        # Sampling draws from torch's generator, which `manual_seed` sets on every device.
        self._torch.manual_seed(seed)
        wavs, rate = self._model.generate_voice_design(
            text=text, instruct=description, language=LANGUAGE
        )
        if not wavs:
            raise core.SpeechBenchError("the model returned no audio")
        audio = self._np.asarray(wavs[0], dtype=self._np.float32).reshape(-1)
        return audio, int(rate)

    def peak(self, audio: object) -> float:
        return float(self._np.max(self._np.abs(audio))) if len(audio) else 0.0

    def write_wav(self, path: Path, audio: object, rate: int) -> None:
        """Mono 16-bit PCM, refusing to overwrite an existing file."""
        np = self._np
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
        try:
            handle = path.open("xb")
        except FileExistsError as exc:
            raise core.OutputExistsError(f"{path} already exists; refusing to overwrite it") from exc
        with handle, wave.open(handle, "wb") as clip:
            clip.setnchannels(1)
            clip.setsampwidth(2)
            clip.setframerate(rate)
            clip.writeframes(pcm.tobytes())


def run_design(
    designer: Designer,
    *,
    descriptions: Sequence[Item],
    lines: Sequence[Item],
    seeds: range,
    out: Path,
    model_pin: str,
    versions: Mapping[str, str | None],
    emit: Callable[[str], None],
) -> int:
    """Every candidate, description by description; returns the count written."""
    text = spoken_text(lines)
    count = 0
    with core.JsonlWriter(out / LOG_FILE, stamp={}) as log:
        for description in descriptions:
            for seed in seeds:
                name = candidate_name(description.id, seed)
                audio, rate = designer.design(description=description.text, text=text, seed=seed)
                if len(audio) == 0:
                    raise core.SpeechBenchError(f"{name}: the model returned an empty clip")
                designer.write_wav(out / name, audio, rate)
                log.write(
                    {
                        "candidate": name,
                        "description_id": description.id,
                        "seed": seed,
                        "model": model_pin,
                        "packages": dict(versions),
                        "language": LANGUAGE,
                        "line_ids": [line.id for line in lines],
                        "sample_rate": rate,
                        "duration_s": len(audio) / rate,
                        "peak": designer.peak(audio),
                    }
                )
                count += 1
                emit(f"{CANDIDATE_PREFIX}{name}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.model_bench.speech.probes.design_probe"
    )
    parser.add_argument("--model", type=Path, required=True, help="the VoiceDesign weights")
    parser.add_argument("--model-pin", required=True, help="<repository>@<revision>, logged")
    parser.add_argument("--descriptions", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--seeds", type=int, required=True, help="seeds 1 to N per description")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _print_flush(line: str) -> None:
    print(line, flush=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    importer: Importer = importlib.import_module,
    versions: Callable[[], Mapping[str, str | None]] = package_versions,
    emit: Callable[[str], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    say = emit or _print_flush
    try:
        descriptions = load_items(args.descriptions, "--descriptions")
        lines = load_items(args.lines, "--lines")
        seeds = seeds_for(args.seeds)
        if not args.out.is_dir():
            raise RefusedError(f"--out {args.out} is not a directory")
        existing = existing_outputs(args.out)
        if existing:
            raise RefusedError(f"--out {args.out} already holds {existing}")
    except RefusedError as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr, flush=True)
        return 2
    try:
        designer = Designer(args.model, importer)
        count = run_design(
            designer,
            descriptions=descriptions,
            lines=lines,
            seeds=seeds,
            out=args.out,
            model_pin=args.model_pin,
            versions=versions(),
            emit=say,
        )
    except core.SpeechBenchError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr, flush=True)
        return 1
    say(f"{DONE_PREFIX} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
