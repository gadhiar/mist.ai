"""Speak the texts with Chatterbox, cloning LibriSpeech train-clean-100 speakers.

    python -m scripts.wake.verifier.chatterbox_driver --texts <texts.json> \
        --references <train-clean-100 dir> --snapshot <hf snapshot dir> --count <n> \
        --seed <n> --out <dir> [--device cuda|cpu]

The chatterbox-tts 0.1.7 API this calls, read from the installed package in the worker image
(`chatterbox/tts.py`) on 2026-11-21:

- `ChatterboxTTS.from_local(ckpt_dir, device)` (`tts.py:129`) loads `ve.safetensors`,
  `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json` and, when present, `conds.pt`
  from one directory. `from_pretrained` (`tts.py:168-178`) fetches the same files with
  `hf_hub_download` and no revision, so it is never called: this loads the H34 snapshot
  directory alone, and the revision recorded is that directory's name, the commit it holds.
- `ChatterboxTTS.sr` (`tts.py:119`) is `S3GEN_SR`, 24000 (`models/s3gen/const.py:1`).
- `generate(text, repetition_penalty=1.2, min_p=0.05, top_p=1.0, audio_prompt_path=None,
  exaggeration=0.5, cfg_weight=0.5, temperature=0.8)` (`tts.py:208`) conditions on the
  reference WAV at `audio_prompt_path` and returns a float tensor of shape `[1, n]` at `sr`,
  with the Perth watermark applied (`tts.py:271`). Only `text` and `audio_prompt_path` are
  passed; the rest keep their defaults.

Clip i is the positive class when i is even and the negative class when odd, so `--count`
must be even and the classes are the same size, which keeps the watermark from carrying the
label. Its text, speaker and reference utterance are drawn by `random.Random(seed * 1000003 +
i)`, and torch is seeded with the same number before it is spoken, so a rerun on the same
device speaks the same clips. Each is written at the model's 24 kHz as PCM16; `mine.py`
resamples to 16 kHz before mixing. chatterbox and torch are imported inside the functions that
use them, so the tests run without either. `--out` is written new. Exit 0 written, 2 refused.
"""

import argparse
import importlib.metadata
import json
import random
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .dataset import CLIPS_FILE, SYNTH_META, to_pcm16, write_wav
from .synth import SynthError, load_texts

SNAPSHOT_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
)
"""The five files `from_local` reads, as H34 fetches them."""
PACKAGE = "chatterbox-tts"
SEED_STRIDE = 1000003


class ChatterboxError(Exception):
    """A refusal; the command exits 2 with the message."""


def load_chatterbox(snapshot: Path, device: str) -> object:
    """`ChatterboxTTS.from_local(snapshot, device)`, the only load this driver makes."""
    from chatterbox.tts import ChatterboxTTS  # noqa: PLC0415 - imported only to synthesise

    return ChatterboxTTS.from_local(snapshot, device)


def seed_torch(seed: int) -> None:
    """Seed torch's generators, CPU and CUDA, before a clip is spoken."""
    import torch  # noqa: PLC0415 - imported only to synthesise

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _default_device() -> str:
    import torch  # noqa: PLC0415 - imported only to synthesise

    return "cuda" if torch.cuda.is_available() else "cpu"


def _references(root: Path) -> dict[str, list[Path]]:
    """Each LibriSpeech speaker's utterances, `<speaker>/<chapter>/<utterance>.flac`."""
    speakers: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.flac")):
        speakers.setdefault(path.parent.parent.name, []).append(path)
    if not speakers:
        raise ChatterboxError(f"--references {root} holds no .flac utterance")
    return speakers


def _samples(wav: object) -> np.ndarray:
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).reshape(-1)


def synthesise(
    args: argparse.Namespace,
    load: Callable[[Path, str], object],
    seed_all: Callable[[int], None],
) -> int:
    """Check the inputs, load the snapshot and speak `--count` clips."""
    if args.out.exists():
        raise ChatterboxError("--out already exists; the clips are written new")
    if args.count < 2 or args.count % 2:
        raise ChatterboxError("--count must be even, half of each class")
    missing = [name for name in SNAPSHOT_FILES if not (args.snapshot / name).is_file()]
    if missing:
        raise ChatterboxError(f"--snapshot lacks {missing}; stage the snapshot as H34 does")
    try:
        texts = load_texts(args.texts)
    except SynthError as exc:
        raise ChatterboxError(str(exc)) from exc
    by_label = {
        label: [item for item in texts if item.label == label] for label in ("positive", "negative")
    }
    speakers = _references(args.references)
    names = sorted(speakers)
    device = args.device or _default_device()
    model = load(args.snapshot, device)
    rate = int(model.sr)
    args.out.mkdir(parents=True)
    rows = []
    for index in range(args.count):
        label = "positive" if index % 2 == 0 else "negative"
        clip_seed = args.seed * SEED_STRIDE + index
        # Not a security use: the seed only makes the draw reproducible.
        rng = random.Random(clip_seed)  # nosec B311
        text = rng.choice(by_label[label])
        speaker = rng.choice(names)
        reference = rng.choice(speakers[speaker])
        seed_all(clip_seed)
        wav = model.generate(text.text, audio_prompt_path=str(reference))
        file = f"{label}/{index:06d}.wav"
        write_wav(args.out / file, to_pcm16(_samples(wav)), rate)
        rows.append(
            {
                "file": file,
                "text_id": text.id,
                "label": label,
                "speaker": speaker,
                "reference": reference.relative_to(args.references).as_posix(),
                "rate": rate,
                "seed": clip_seed,
            }
        )
    (args.out / CLIPS_FILE).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    try:
        version = importlib.metadata.version(PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        version = "not installed"
    meta = {
        "version": 1,
        "synthesiser": "chatterbox",
        "package_version": version,
        "revision": args.snapshot.name,
        "seed": args.seed,
        "count": args.count,
        "device": device,
    }
    (args.out / SYNTH_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"chatterbox: {args.count // 2} positive, {args.count // 2} negative at {rate} Hz")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.chatterbox_driver", description=__doc__
    )
    parser.add_argument("--texts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[Path, str], object] = load_chatterbox,
    seed_all: Callable[[int], None] = seed_torch,
) -> int:
    """Run the driver; see the module docstring for the exit codes."""
    args = _parser().parse_args(argv)
    try:
        return synthesise(args, load, seed_all)
    except ChatterboxError as exc:
        print(f"chatterbox_driver: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
