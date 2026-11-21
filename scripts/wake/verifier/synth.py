"""The synthetic speech's texts, and the checks and mixing every synthetic clip goes through.

    python -m scripts.wake.verifier.synth texts --out <texts.json> --seed <n> [--per-class N]

Positives are the name alone and in vocative positions: first, last, after a greeting and
mid-sentence. Negatives are its sound-alikes ("missed", "mister", "midst", "misty"), the word
in another sense ("mist" as fog), and vocatives to other names. The two classes are the same
size, and each synthesiser speaks every text the same number of times, so neither a
synthesiser's watermark nor its timbre carries the label. No text is a sentence of `record.py`'s
`MISSED`, `FOG` or `WAKE_PHRASES`, compared as `manifest.phrase_key` compares them, so no
recorded sentence is heard in training by another voice; only the `NAME_ALONE` forms, the name
itself, repeat. `--out` is written new, never over a file.

A clip is read at the rate its file states, never an assumed one (Chatterbox writes 24 kHz; the
piper generator's rate is read from its output), and resampled to 16 kHz mono PCM16 before
mixing (`load_clip`). `plan_mix` draws a mix from a seeded generator and `mix` applies it, so
the same seed gives the same file: a background segment (MUSAN noise or music, or the AMI
mining share) at 0 to 20 dB SNR, `LEAD_*` of it before the clip and `TAIL_SAMPLES` after, the
clip convolved with an OpenSLR 28 room response half the time.
"""

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from ..manifest import phrase_key
from ..record import FOG, MISSED, NAME_ALONE, WAKE_PHRASES
from .dataset import SAMPLE_RATE, DatasetError, read_audio, to_16k, to_pcm16

NAME = "mist"
DEFAULT_PER_CLASS = 500
LEAD_MIN_SAMPLES = SAMPLE_RATE
LEAD_MAX_SAMPLES = SAMPLE_RATE * 5 // 2
TAIL_SAMPLES = SAMPLE_RATE // 2
SNR_DB = (0.0, 20.0)
RIR_SHARE = 0.5
_PEAK = 0.99

REQUESTS = (
    "what's the weather like",
    "set a timer for ten minutes",
    "remind me to call the bank",
    "what's on my calendar today",
    "play something quiet",
    "how long until dinner",
    "add milk to the list",
    "can you read that again",
    "what did I ask you yesterday",
    "turn the music down",
    "how far is the station",
    "what's the date tomorrow",
    "start a note for me",
    "what should I cook tonight",
    "check my messages",
    "how many steps did I take",
    "what's the news",
    "wake me at seven",
    "pause that for a second",
    "tell me a fact",
    "where did I leave my keys",
    "how's the traffic on the bridge",
)
GREETINGS = ("hey", "hi", "okay", "hello", "morning", "excuse me", "right", "yo")
LEADS = ("so", "well", "actually", "okay then", "listen", "quick question", "by the way", "um")
CLOSERS = (
    "are you listening",
    "can you hear me",
    "you there",
    "help me out",
    "one more thing",
    "are you awake",
    "got a minute",
    "what do you reckon",
)
SOUND_ALIKES = (
    "I {missed} the {thing} {when}",
    "the {mister} at the door left {when}",
    "we stood in the midst of the {thing} {when}",
    "it was a misty {thing} {when}",
)
SOUND_ALIKE_FILLS = {
    "missed": ("almost missed", "nearly missed", "completely missed", "just missed"),
    "mister": ("mister from next door", "mister with the parcel", "mister in the grey coat"),
    "thing": ("crossing", "parade", "concert", "storm", "rush hour", "harvest", "market"),
    "when": ("last night", "on sunday", "this afternoon", "after work", "before lunch"),
}
FOG_TEMPLATES = (
    "the {adjective} mist {verb} {where}",
    "a {adjective} mist {verb} {where}",
)
FOG_FILLS = {
    "adjective": ("grey", "low", "damp", "early", "soft", "drifting", "pale"),
    "verb": ("settled", "lingered", "crept in", "spread", "thickened", "cleared"),
    "where": ("over the car park", "across the garden", "along the canal", "by the old mill"),
}
OTHER_NAMES = ("max", "mike", "liz", "chris", "sam", "kristen", "misha", "mason")


class SynthError(Exception):
    """A refusal; the command exits 2 with the message."""


@dataclass(frozen=True, slots=True)
class Text:
    """One text to synthesise: its id, words, label and the form it was built in."""

    id: str
    text: str
    label: str
    form: str


def _recorded_keys() -> frozenset[str]:
    """The sentences recorded for the manifest, which training must not hear in another voice."""
    phrases = (*MISSED, *FOG, *(phrase for pair in WAKE_PHRASES for phrase in pair))
    return frozenset(phrase_key(phrase) for phrase in phrases)


def _fill(templates: Sequence[str], fills: dict[str, tuple[str, ...]]) -> list[str]:
    texts = []
    for template in templates:
        keys = [key for key in fills if "{" + key + "}" in template]
        for values in product(*(fills[key] for key in keys)):
            texts.append(template.format(**dict(zip(keys, values))))
    return texts


def _candidates() -> dict[str, list[tuple[str, str]]]:
    """Every (form, text) of each label, before the disjointness filter and the draw."""
    positives = [("alone", name) for name in NAME_ALONE]
    positives += [("first", f"{NAME}, {request}") for request in REQUESTS]
    positives += [("last", f"{request}, {NAME}") for request in REQUESTS + CLOSERS]
    positives += [
        ("after_greeting", f"{greeting} {NAME}, {request}")
        for greeting, request in product(GREETINGS, REQUESTS)
    ]
    positives += [
        ("after_greeting", f"{greeting} {NAME}, {closer}")
        for greeting, closer in product(GREETINGS, CLOSERS)
    ]
    positives += [
        ("mid", f"{lead}, {NAME}, {request}")
        for lead, request in product(LEADS, REQUESTS + CLOSERS)
    ]
    negatives = [("sound_alike", text) for text in _fill(SOUND_ALIKES, SOUND_ALIKE_FILLS)]
    negatives += [("fog", text) for text in _fill(FOG_TEMPLATES, FOG_FILLS)]
    negatives += [
        ("other_name", f"{name}, {request}") for name, request in product(OTHER_NAMES, REQUESTS)
    ]
    negatives += [
        ("other_name", f"{greeting} {name}, {closer}")
        for greeting, name, closer in product(GREETINGS, OTHER_NAMES, CLOSERS)
    ]
    return {"positive": positives, "negative": negatives}


def make_texts(seed: int, per_class: int = DEFAULT_PER_CLASS) -> list[Text]:
    """`per_class` texts of each label, or as many as both labels have, drawn by `seed`.

    The name-alone forms are always kept; every other text whose key is a recorded sentence is
    dropped, and a key appears once.
    """
    recorded = _recorded_keys()
    names = {phrase_key(name) for name in NAME_ALONE}
    pools: dict[str, list[tuple[str, str]]] = {}
    seen: set[str] = set()
    for label, candidates in _candidates().items():
        pool = []
        for form, text in candidates:
            key = phrase_key(text)
            if key in seen or (key in recorded and key not in names):
                continue
            seen.add(key)
            pool.append((form, text))
        pools[label] = pool
    count = min(per_class, *(len(pool) for pool in pools.values()))
    # Not a security use: the seed only makes the draw reproducible.
    rng = random.Random(seed)  # nosec B311
    drawn: list[tuple[str, str, str]] = []
    for label in ("positive", "negative"):
        pool = pools[label]
        alone = [item for item in pool if item[0] == "alone"]
        rest = [item for item in pool if item[0] != "alone"]
        rng.shuffle(rest)
        drawn += [(label, form, text) for form, text in (alone + rest)[:count]]
    rng.shuffle(drawn)
    return [
        Text(id=f"t{number:05d}", text=text, label=label, form=form)
        for number, (label, form, text) in enumerate(drawn)
    ]


def load_texts(path: Path) -> list[Text]:
    """Read `texts.json`, checking each text's fields and the classes' balance.

    Raises:
        SynthError: The file is malformed or its classes differ in size.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        texts = [Text(**item) for item in raw["texts"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SynthError(f"{path.name} is not a texts file: {exc}") from exc
    labels = [item.label for item in texts]
    if set(labels) != {"positive", "negative"} or labels.count("positive") * 2 != len(labels):
        raise SynthError(f"{path.name} must hold as many positive texts as negative ones")
    return texts


def load_clip(path: Path) -> np.ndarray:
    """A synthetic clip as 16 kHz mono PCM16, resampled from the rate its file states.

    Raises:
        DatasetError: The file cannot be read as audio or holds none.
    """
    samples, rate = read_audio(path)
    if samples.size == 0:
        raise DatasetError(f"{path.name} holds no audio")
    return to_pcm16(to_16k(samples, rate))


@dataclass(frozen=True, slots=True)
class MixPlan:
    """One mix: which background and from where, the SNR, the room response, the lead-in."""

    background: int
    offset: int
    snr_db: float
    rir: int | None
    lead: int

    def total(self, clip_samples: int) -> int:
        """The mixed file's length in samples."""
        return self.lead + clip_samples + TAIL_SAMPLES


def plan_mix(
    rng: np.random.Generator, clip_samples: int, backgrounds: Sequence[int], rirs: int
) -> MixPlan:
    """Draw a mix for a clip from `rng`, over backgrounds of the given lengths in samples."""
    background = int(rng.integers(len(backgrounds)))
    lead = int(rng.integers(LEAD_MIN_SAMPLES, LEAD_MAX_SAMPLES + 1))
    total = lead + clip_samples + TAIL_SAMPLES
    offset = int(rng.integers(max(1, backgrounds[background] - total + 1)))
    snr_db = float(rng.uniform(*SNR_DB))
    use_rir = bool(rng.random() < RIR_SHARE) and rirs > 0
    rir = int(rng.integers(rirs)) if use_rir else None
    return MixPlan(background=background, offset=offset, snr_db=snr_db, rir=rir, lead=lead)


def mix(
    clip: np.ndarray, background: np.ndarray, rir: np.ndarray | None, plan: MixPlan
) -> np.ndarray:
    """Apply `plan`: the clip, reverberated when `rir` is given, over the background at its SNR.

    `background` is the segment from `plan.offset`, repeated when shorter than the mix. The
    reverberated clip keeps the clip's length; the sum is scaled down only if it would clip.
    """
    speech = np.asarray(clip, dtype=np.float64) / 32768.0
    if rir is not None:
        from scipy.signal import fftconvolve  # noqa: PLC0415 - only reverberation needs scipy

        response = np.asarray(rir, dtype=np.float64)
        peak = np.max(np.abs(response))
        if peak > 0:
            speech = fftconvolve(speech, response / peak)[: len(speech)]
    total = plan.total(len(speech))
    noise = np.asarray(background, dtype=np.float64) / 32768.0
    noise = np.resize(noise, total) if noise.size else np.zeros(total)
    placed = np.zeros(total)
    placed[plan.lead : plan.lead + len(speech)] = speech
    speech_power = float(np.mean(speech**2)) if speech.size else 0.0
    noise_power = float(np.mean(noise[plan.lead : plan.lead + len(speech)] ** 2))
    if speech_power > 0 and noise_power > 0:
        noise = noise * np.sqrt(speech_power / (noise_power * 10 ** (plan.snr_db / 10)))
    mixed = placed + noise
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > _PEAK:
        mixed = mixed * (_PEAK / peak)
    return to_pcm16(mixed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.synth", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    texts = commands.add_parser("texts", help="write the texts both synthesisers speak")
    texts.add_argument("--out", required=True, type=Path)
    texts.add_argument("--seed", required=True, type=int)
    texts.add_argument("--per-class", type=int, default=DEFAULT_PER_CLASS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run `texts`; 0 written, 2 refused (an existing `--out`)."""
    args = _parser().parse_args(argv)
    if args.out.exists():
        print("synth: --out already exists; the texts are written new", file=sys.stderr)
        return 2
    texts = make_texts(args.seed, args.per_class)
    payload = {"version": 1, "seed": args.seed, "texts": [asdict(item) for item in texts]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    labels = [item.label for item in texts]
    print(f"texts: {labels.count('positive')} positive, {labels.count('negative')} negative")
    return 0


if __name__ == "__main__":
    sys.exit(main())
