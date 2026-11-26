"""Text and audio rules every TTS adapter shares: what reaches the engine, what leaves it.

Text: `speakable_text` removes only what is not speech (Markdown emphasis and code markers,
heading markers, list bullets at a line start, link syntax) and, for an engine without tag
support, bracketed tags such as `[laugh]`. Case, digits and every other punctuation mark pass
unchanged, because each engine normalises its own punctuation and some read case.

Audio: engine output is resampled to the 24 kHz the realtime path plays, converted to PCM16
with clipping, and cut into chunks of at most 200 ms. Loudness is measured as integrated
loudness (ITU-R BS.1770 gating), and `target_gain_db` is the gain that brings a measured level
to -20 LUFS, capped at +20 dB so near-silence is not raised into noise. An adapter decides what
it measures: Pocket measures each voice state once, when its artefacts are built, so a segment
can leave piece by piece at a fixed gain.

Only numpy and scipy are imported here; no engine library.
"""

import math
import re
from collections.abc import Iterator

import numpy as np
from scipy.signal import lfilter, resample_poly

OUTPUT_RATE = 24000
TARGET_LUFS = -20.0
MAX_GAIN_DB = 20.0
MAX_CHUNK_MS = 200

_LINK = re.compile(r"\[([^\[\]\n]*)\]\([^()\s]*\)")
_CODE = re.compile(r"`+")
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)
_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
# Paired emphasis whose markers sit at word edges, so `snake_case` keeps its underscore.
_EMPHASIS = re.compile(r"(?<![\w*])(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1(?![\w*])")
# What pairing left: runs of asterisks, and a lone asterisk at a word edge (not `2 * 3`).
_STRAY_ASTERISKS = re.compile(r"\*{2,}|(?<=\w)\*(?!\w)|(?<!\w)\*(?=\w)")
_TAG = re.compile(r"\[[^\[\]\n]*\]")
_SPACES = re.compile(r"\s+")


def speakable_text(text: str, *, keep_tags: bool) -> str:
    """The text with what is not speech removed and whitespace collapsed.

    `keep_tags` keeps bracketed tags for an engine that voices them; links lose their syntax
    either way (`[the docs](http://x)` becomes `the docs`), before tags are judged.
    """
    text = _LINK.sub(r"\1", text)
    text = _CODE.sub("", text)
    text = _HEADING.sub("", text)
    text = _BULLET.sub("", text)
    previous = None
    while previous != text:
        previous, text = text, _EMPHASIS.sub(r"\2", text)
    text = _STRAY_ASTERISKS.sub("", text)
    if not keep_tags:
        text = _TAG.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def to_mono_float(audio: object) -> np.ndarray:
    """A 1-D float32 array from an engine's numpy array or torch tensor, without importing torch."""
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _k_weighting(rate: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """BS.1770's two K-weighting stages, the high shelf then the high pass, derived for `rate`.

    The standard tabulates the coefficients at 48 kHz only; these analogue prototypes reproduce
    that table, so audio at any rate is weighted the same way.
    """
    gain_db, q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    k = math.tan(math.pi * fc / rate)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = (
        np.array([vh + vb * k / q + k * k, 2 * (k * k - vh), vh - vb * k / q + k * k]) / a0,
        np.array([a0, 2 * (k * k - 1), 1 - k / q + k * k]) / a0,
    )
    q, fc = 0.5003270373238773, 38.13547087602444
    k = math.tan(math.pi * fc / rate)
    a0 = 1 + k / q + k * k
    highpass = (
        np.array([1.0, -2.0, 1.0]),
        np.array([a0, 2 * (k * k - 1), 1 - k / q + k * k]) / a0,
    )
    return shelf, highpass


def _lufs(power: np.ndarray | float) -> np.ndarray | float:
    with np.errstate(divide="ignore"):
        return -0.691 + 10 * np.log10(power)


def integrated_loudness(samples: np.ndarray, rate: int) -> float:
    """Gated integrated loudness of mono audio in LUFS, `-inf` for silence (ITU-R BS.1770).

    400 ms blocks overlapping by 75%, the -70 LUFS absolute gate, then the gate 10 LU below the
    absolutely gated level. Audio shorter than one block is measured as a single block.
    """
    weighted = np.asarray(samples, dtype=np.float64)
    if weighted.size == 0:
        return float("-inf")
    for b, a in _k_weighting(rate):
        weighted = lfilter(b, a, weighted)
    cumulative = np.concatenate(([0.0], np.cumsum(weighted * weighted)))
    block, step = round(0.4 * rate), round(0.1 * rate)
    if weighted.size < block:
        powers = np.array([cumulative[-1] / weighted.size])
    else:
        starts = np.arange(0, weighted.size - block + 1, step)
        powers = (cumulative[starts + block] - cumulative[starts]) / block
    gated = powers[_lufs(powers) > -70.0]
    if gated.size == 0:
        return float("-inf")
    relative = _lufs(gated.mean()) - 10.0
    gated = gated[_lufs(gated) > relative]
    return float(_lufs(gated.mean()))


def target_gain_db(
    loudness: float, target_lufs: float = TARGET_LUFS, max_gain_db: float = MAX_GAIN_DB
) -> float:
    """The gain in dB that brings `loudness` (LUFS) to `target_lufs`, at most `max_gain_db`.

    Silence (`-inf`) gets the cap, as anything quieter than the cap allows does.
    """
    return float(min(target_lufs - loudness, max_gain_db))


def resample(samples: np.ndarray, rate: int, target: int = OUTPUT_RATE) -> np.ndarray:
    """`samples` at `target` Hz by polyphase filtering; the same array when the rates match."""
    if rate == target:
        return samples
    divisor = math.gcd(rate, target)
    return resample_poly(samples, target // divisor, rate // divisor).astype(np.float32)


def to_pcm16(samples: np.ndarray) -> bytes:
    """Little-endian PCM16, clipped to [-1, 1] first, the sample format of a binary audio frame."""
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def pcm16_chunks(pcm: bytes, rate: int, max_ms: int = MAX_CHUNK_MS) -> Iterator[bytes]:
    """`pcm` cut into chunks of at most `max_ms`, in order, the last one shorter."""
    step = 2 * (rate * max_ms // 1000)
    for start in range(0, len(pcm), step):
        yield pcm[start : start + step]
