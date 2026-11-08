"""Run 1's speech synthesis probe, run inside an engine's bench container.

Invocation, from `/work` (the repository, mounted read-only):

    python -m scripts.model_bench.speech.probes.tts_probe --engine <id> --mode timed
        --condition idle|decoding --out /out [--seed N] [--repeats N] [--lines <id> ...]
    python -m scripts.model_bench.speech.probes.tts_probe --engine <id> --mode render
        --out /out [--voice <name> ...] [--expressive] [--lines <id> ...]
    python -m scripts.model_bench.speech.probes.tts_probe --engine <id> --mode loop
        --line-id <id> --seconds <s>

`timed` loads the engine in its timing voice (timed), synthesizes three warm-up lines it
discards, prints `TIMED_START`, times every line and acknowledgement `--repeats` times in a
seeded shuffled order, then prints `TIMED_END`; the host marks its decode-rate window with those
two lines. `render` writes untimed clips, repeat 1 of each line, in each `--voice` and, with
`--expressive`, `original`'s emotive lines in its `expressive` setting. `loop` is run 5's memory
fit: it warms up, prints `READY`, synthesizes one line over and over for `--seconds`, prints
`DONE` and exits 0.

Timing: a streaming engine's first audio runs from the call that submits the text to the first
non-empty chunk. A whole-segment engine synthesizes the line's first segment (cut at the first
clause boundary ending at or after character 40, as MIST's realtime design cuts it) and then the
whole line, and both times are recorded. CUDA work is synchronized before every timestamp, and
every timestamp is `time.perf_counter` inside this container. Compute per second of speech is the
whole line's synthesis time over its audio seconds.

Exit codes: 0 done, 2 refused before any write (an unknown line, a voice the engine does not
list, a setting it does not have), 1 failed. numpy and torch are imported only when used, so the
host's selftest can import this module.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
import time
import wave
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from scripts.model_bench.speech import core
    from scripts.model_bench.speech.probes import tts_engines
else:
    from .. import core
    from . import tts_engines

TIMED_START = "TIMED_START"
TIMED_END = "TIMED_END"
READY = "READY"
DONE = "DONE"

WARMUPS = 3
DEFAULT_REPEATS = 3
MIN_SEGMENT_CHARS = 40
LISTENING_FILE = "listening.jsonl"
EXPRESSIVE = "expressive"
MODES = ("timed", "render", "loop")
CONDITIONS = ("idle", "decoding")

# A clause boundary: one of , ; : . ! ? followed by whitespace.
_BOUNDARY_RE = re.compile(r"[,;:.!?](?=\s)")

Clock = Callable[[], float]


class RefusedError(core.SpeechBenchError):
    """A request the probe refuses before writing anything."""


# ---------------------------------------------------------------------------
# Pure pieces: the first segment, the order, the text an engine receives
# ---------------------------------------------------------------------------


def first_segment(text: str) -> str:
    """`text` up to the first clause boundary that ends at or after character 40, or all of it."""
    for match in _BOUNDARY_RE.finditer(text):
        if match.end() >= MIN_SEGMENT_CHARS:
            return text[: match.end()]
    return text


def timed_order(item_ids: Sequence[str], *, repeats: int, seed: int) -> list[tuple[str, int]]:
    """Every item times every repeat, shuffled by `seed`: the same seed gives the same order."""
    order = [(item, rep) for rep in range(1, repeats + 1) for item in item_ids]
    random.Random(seed).shuffle(order)
    return order


def text_for(item: Mapping[str, object], cfg: Mapping[str, object]) -> str:
    """The text an engine receives: its `text_field` (Turbo's tagged text) where the line has one."""
    field = str(cfg.get("text_field", "text"))
    return str(item.get(field) or item["text"])


def timed_wav_name(condition: str, item_id: str, rep: int) -> str:
    return f"{condition}_{item_id}_r{rep}.wav"


def probe_file(condition: str) -> str:
    """The probe's row file for a timed condition, inside the engine's output directory."""
    return f"probe-{condition}.jsonl"


# ---------------------------------------------------------------------------
# Devices: CUDA synchronization and peak memory
# ---------------------------------------------------------------------------


class NoDevice:
    """A CPU engine: nothing to synchronize and no GPU memory to read."""

    def sync(self) -> None:
        """Nothing runs asynchronously on the CPU path."""

    def reset_peak(self) -> None:
        """No GPU memory statistics on the CPU path."""

    def peak_reserved_mib(self) -> float | None:
        """No GPU memory to report, so the row carries null."""
        return None


class TorchCuda:
    """`torch.cuda` synchronization and the caching allocator's peak reserved memory."""

    def __init__(self, torch: object):
        self._cuda = torch.cuda

    def sync(self) -> None:
        """Wait for queued CUDA work, so a timestamp marks finished work."""
        self._cuda.synchronize()

    def reset_peak(self) -> None:
        """Start a new peak-memory window."""
        self._cuda.reset_peak_memory_stats()

    def peak_reserved_mib(self) -> float | None:
        """Reserved rather than allocated memory: what the process holds from other users."""
        return self._cuda.max_memory_reserved() / 2**20


def default_device(engine: object) -> NoDevice | TorchCuda:
    if getattr(engine, "device", "cpu") == "cuda":
        return TorchCuda(importlib.import_module("torch"))
    return NoDevice()


# ---------------------------------------------------------------------------
# Timing one item
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ItemTiming:
    """One timed synthesis and the audio it produced."""

    first_audio_ms: float
    first_segment_ms: float | None
    first_segment_chars: int | None
    total_ms: float
    audio_s: float
    compute_per_speech_s: float
    torch_peak_reserved_mib: float | None
    t_start_unix: float
    t_end_unix: float
    chunks: tuple


def _samples(chunks: Iterable[object]) -> int:
    return sum(len(chunk) for chunk in chunks)


def time_item(
    engine: object, text: str, *, clock: Clock, wall: Clock, device: object
) -> ItemTiming:
    """Synthesize `text` once and time it as the engine's kind requires (see the module doc)."""
    device.reset_peak()
    segment_ms: float | None = None
    segment_chars: int | None = None
    if engine.streaming:
        device.sync()
        t_start_unix = wall()
        t0 = clock()
        chunks = []
        t_first: float | None = None
        for chunk in engine.stream(text):
            device.sync()
            now = clock()
            if t_first is None and len(chunk) > 0:
                t_first = now
            chunks.append(chunk)
        device.sync()
        t_end = clock()
        total_s = t_end - t0
        first_s = None if t_first is None else t_first - t0
    else:
        segment = first_segment(text)
        device.sync()
        t_start_unix = wall()
        t0 = clock()
        t_whole = t0
        if segment != text:
            for _ in engine.stream(segment):
                pass
            device.sync()
            t_whole = clock()
            segment_ms = (t_whole - t0) * 1000.0
        chunks = list(engine.stream(text))
        device.sync()
        t_end = clock()
        total_s = t_end - t_whole
        if segment_ms is None:
            segment_ms = total_s * 1000.0
        segment_chars = len(segment)
        first_s = segment_ms / 1000.0
    t_end_unix = wall()

    n_samples = _samples(chunks)
    if n_samples == 0 or first_s is None:
        raise core.SpeechBenchError(f"{engine.name} returned no audio for {text!r}")
    audio_s = n_samples / engine.sample_rate
    return ItemTiming(
        first_audio_ms=first_s * 1000.0,
        first_segment_ms=segment_ms,
        first_segment_chars=segment_chars,
        total_ms=total_s * 1000.0,
        audio_s=audio_s,
        compute_per_speech_s=total_s / audio_s,
        torch_peak_reserved_mib=device.peak_reserved_mib(),
        t_start_unix=t_start_unix,
        t_end_unix=t_end_unix,
        chunks=tuple(chunks),
    )


def write_wav(path: Path, chunks: Sequence[object], sample_rate: int) -> None:
    """Write mono 16-bit PCM, refusing to overwrite an existing clip."""
    np = importlib.import_module("numpy")
    audio = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1) for c in chunks])
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("xb")
    except FileExistsError as exc:
        raise core.OutputExistsError(f"{path} already exists; refusing to overwrite it") from exc
    with handle, wave.open(handle, "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(int(sample_rate))
        clip.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProbeEnv:
    """What every mode needs, injected so tests run the modes with fakes."""

    engine_id: str
    cfg: Mapping[str, object]
    items: Mapping[str, Mapping[str, object]]
    acks: frozenset[str]
    clock: Clock
    wall: Clock
    emit: Callable[[str], None]


def _rel(engine_id: str, *parts: str) -> str:
    return "/".join(("tts", engine_id, *parts))


def _warm_up(engine: object, texts: Sequence[str]) -> None:
    """Synthesize `WARMUPS` lines and discard them, cycling `texts` when it is shorter."""
    for index in range(WARMUPS):
        for _ in engine.stream(texts[index % len(texts)]):
            pass


def run_timed(engine, device, env: ProbeEnv, *, condition, item_ids, repeats, seed, out) -> None:
    voice = str(env.cfg["timing_voice"])
    rows = core.JsonlWriter(out / probe_file(condition), stamp={})
    with rows:
        device.sync()
        t0 = env.clock()
        engine.load(voice)
        device.sync()
        load_s = env.clock() - t0
        rows.write(
            {
                "row_type": "load",
                "engine": env.engine_id,
                "voice": voice,
                "condition": condition,
                "device": engine.device,
                "streaming": bool(engine.streaming),
                "sample_rate": engine.sample_rate,
                "load_s": load_s,
            }
        )
        warm = [text_for(env.items[i], env.cfg) for i in item_ids if i not in env.acks]
        _warm_up(engine, warm or [text_for(env.items[i], env.cfg) for i in item_ids])

        env.emit(TIMED_START)
        for position, (item_id, rep) in enumerate(
            timed_order(item_ids, repeats=repeats, seed=seed)
        ):
            item = env.items[item_id]
            timing = time_item(
                engine, text_for(item, env.cfg), clock=env.clock, wall=env.wall, device=device
            )
            name = timed_wav_name(condition, item_id, rep)
            write_wav(out / "audio" / name, timing.chunks, engine.sample_rate)
            plain = str(item["text"])
            rows.write(
                {
                    "row_type": "sample",
                    "engine": env.engine_id,
                    "voice": voice,
                    "setting": engine.setting,
                    "condition": condition,
                    "kind": "ack" if item_id in env.acks else "line",
                    "line_id": item_id,
                    "rep": rep,
                    "position": position,
                    "streaming": bool(engine.streaming),
                    "words": len(plain.split()),
                    "chars": len(plain),
                    "first_audio_ms": timing.first_audio_ms,
                    "first_segment_ms": timing.first_segment_ms,
                    "first_segment_chars": timing.first_segment_chars,
                    "total_ms": timing.total_ms,
                    "audio_s": timing.audio_s,
                    "compute_per_speech_s": timing.compute_per_speech_s,
                    "torch_peak_reserved_mib": timing.torch_peak_reserved_mib,
                    "t_start_unix": timing.t_start_unix,
                    "t_end_unix": timing.t_end_unix,
                    "wav": _rel(env.engine_id, "audio", name),
                }
            )
        env.emit(TIMED_END)


def _render(engine, env: ProbeEnv, rows, *, voice, label, line_ids, out) -> None:
    engine.load(voice)
    for line_id in line_ids:
        chunks = list(engine.stream(text_for(env.items[line_id], env.cfg)))
        name = f"{line_id}.wav"
        write_wav(out / "listening" / label / name, chunks, engine.sample_rate)
        rows.write(
            {
                "engine": env.engine_id,
                "voice": voice,
                "setting": engine.setting,
                "line_id": line_id,
                "rep": 1,
                "audio_s": sum(len(c) for c in chunks) / engine.sample_rate,
                "wav": _rel(env.engine_id, "listening", label, name),
            }
        )


def run_render(engine, env: ProbeEnv, *, voices, expressive, line_ids, out) -> None:
    with core.JsonlWriter(out / LISTENING_FILE, stamp={}) as rows:
        for voice in voices:
            _render(engine, env, rows, voice=voice, label=voice, line_ids=line_ids, out=out)
        if expressive:
            emotive = [i for i in line_ids if env.items[i].get("emotive")]
            engine.select_setting(EXPRESSIVE)
            timing_voice = str(env.cfg["timing_voice"])
            _render(
                engine, env, rows, voice=timing_voice, label=EXPRESSIVE, line_ids=emotive, out=out
            )


def run_loop(engine, env: ProbeEnv, *, line_id, seconds) -> int:
    engine.load(str(env.cfg["timing_voice"]))
    text = text_for(env.items[line_id], env.cfg)
    _warm_up(engine, [text])
    env.emit(READY)
    t0 = env.clock()
    count = 0
    while True:
        for _ in engine.stream(text):
            pass
        count += 1
        if env.clock() - t0 >= seconds:
            break
    env.emit(f"loop: {count} syntheses of {line_id}")
    env.emit(DONE)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m scripts.model_bench.speech.probes.tts_probe")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--models-dir", type=Path, default=Path(tts_engines.CONTAINER_MODELS_DIR))
    parser.add_argument("--out", type=Path, help="the engine's output directory (timed, render)")
    parser.add_argument("--condition", choices=CONDITIONS, help="timed: idle or decoding")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--lines", action="append", help="restrict to these line ids")
    parser.add_argument("--voice", action="append", default=[], help="render: a stock voice")
    parser.add_argument("--expressive", action="store_true", help="render: the expressive setting")
    parser.add_argument("--line-id", help="loop: the line to synthesize")
    parser.add_argument("--seconds", type=float, help="loop: how long to keep synthesizing")
    return parser


def _print_flush(line: str) -> None:
    print(line, flush=True)


def _check_request(args, cfg, items) -> list[str]:
    """Everything the probe refuses, checked before the engine is built or a file is written."""
    if args.repeats < 1:
        raise RefusedError(f"--repeats must be at least 1, got {args.repeats}")
    # `items` keeps the file's order (lines, then acknowledgements); `acks` is a set, whose order
    # would change with string hashing and break the seeded order.
    selected = list(args.lines) if args.lines else list(items)
    unknown = [i for i in selected if i not in items]
    if unknown:
        raise RefusedError(f"unknown line ids {unknown}")
    stock = [str(v) for v in cfg["voices"]]
    if args.mode == "timed":
        if args.condition is None or args.out is None:
            raise RefusedError("timed needs --condition and --out")
    elif args.mode == "render":
        if args.out is None:
            raise RefusedError("render needs --out")
        refused = [v for v in args.voice if v not in stock]
        if refused:
            raise RefusedError(f"{args.engine} has no stock voices {refused}; it lists {stock}")
        if args.expressive and EXPRESSIVE not in (cfg.get("settings") or {}):
            raise RefusedError(f"{args.engine} has no {EXPRESSIVE!r} setting")
    else:
        if args.line_id not in items or args.seconds is None or args.seconds <= 0:
            raise RefusedError("loop needs a known --line-id and a positive --seconds")
    if args.out is not None and not args.out.is_dir():
        raise RefusedError(f"--out {args.out} is not a directory")
    return selected


def main(
    argv: Sequence[str] | None = None,
    *,
    engine_factory: Callable[[str, Path], object] | None = None,
    clock: Clock = time.perf_counter,
    wall: Clock = time.time,
    device_factory: Callable[[object], object] | None = None,
    emit: Callable[[str], None] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    say = emit or _print_flush
    try:
        tts = core.load_json(core.SPEECH_ARMS_PATH)["tts"]
        if args.engine not in tts:
            raise RefusedError(f"unknown speech engine {args.engine!r}; known: {list(tts)}")
        cfg = tts[args.engine]
        lines_doc = core.load_json(core.LINES_PATH)
        items = {str(i["id"]): i for i in [*lines_doc["lines"], *lines_doc["acknowledgements"]]}
        acks = frozenset(str(a["id"]) for a in lines_doc["acknowledgements"])
        selected = _check_request(args, cfg, items)
    except RefusedError as exc:
        print(f"[REFUSED] {exc}", file=sys.stderr, flush=True)
        return 2

    env = ProbeEnv(args.engine, cfg, items, acks, clock, wall, say)
    try:
        if engine_factory is None:
            engine = tts_engines.make_engine(args.engine, models_dir=args.models_dir)
        else:
            engine = engine_factory(args.engine, args.models_dir)
        device = (device_factory or default_device)(engine)
        if args.mode == "timed":
            run_timed(
                engine,
                device,
                env,
                condition=args.condition,
                item_ids=selected,
                repeats=args.repeats,
                seed=args.seed,
                out=args.out,
            )
        elif args.mode == "render":
            line_ids = [i for i in selected if i not in acks]
            run_render(
                engine,
                env,
                voices=args.voice,
                expressive=args.expressive,
                line_ids=line_ids,
                out=args.out,
            )
        else:
            run_loop(engine, env, line_id=args.line_id, seconds=args.seconds)
    except core.SpeechBenchError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps({"probe": "tts", "engine": args.engine, "mode": args.mode, "ok": True}),
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
