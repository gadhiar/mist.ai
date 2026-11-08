"""Run 3's container probe: final-text latency per engine, and run 5's transcription loop.

`run` mode loads one engine, warms it up 3 times, then transcribes each utterance 3 times. A
batch engine gets the whole utterance in one call, timed from that call to the final text. The
streaming engine is fed 160 ms chunks paced to the wall clock, each handed over when a live
microphone would have delivered it, and is timed from the moment the last chunk is fed to the
final text. On CUDA, `torch.cuda.synchronize()` runs before each timestamp; time is
`time.perf_counter` inside the container. Rows go to `/out/raw.jsonl`; the host adds the word
error rate.

`loop` mode, for run 5's memory-fit suite, loads one engine, warms it up, prints `READY` on its
own flushed line, transcribes one utterance repeatedly for `--seconds`, prints `DONE` and exits 0.

An engine that fails to load is written to `/out/load.json` as not loaded, with its error, and
the probe exits 3, so the host reports it and goes on to the next engine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .. import core
from . import prepare_material, stt_engines

SAMPLE_RATE = 16000
CHUNK_MS = 160
NOT_LOADED_EXIT = 3

# What a failed load raises across whisper, NeMo and onnx-asr: a missing package or weights file,
# a CUDA or model error, or a config the installed version does not know.
LOAD_ERRORS = (ImportError, OSError, RuntimeError, ValueError, KeyError, AttributeError, TypeError)

Clock = Callable[[], float]


class _Out(Protocol):
    def write(self, text: str) -> object: ...

    def flush(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Timing:
    """One transcription's final text and its latency."""

    text: str
    final_ms: float
    feed_lag_ms: float | None = None
    feed_times: tuple[float, ...] = ()


def _sync(sync: Callable[[], None] | None) -> None:
    if sync is not None:
        sync()


def batch_timed(
    engine, audio: np.ndarray, *, sample_rate: int, clock: Clock, sync: Callable[[], None] | None
) -> Timing:
    """Time one whole-utterance call, from the call to the returned text."""
    _sync(sync)
    t0 = clock()
    text = engine.transcribe(audio, sample_rate)
    _sync(sync)
    return Timing(text, (clock() - t0) * 1000.0)


def stream_timed(
    engine,
    audio: np.ndarray,
    *,
    sample_rate: int,
    chunk_ms: int,
    clock: Clock,
    sleep: Callable[[float], None],
    sync: Callable[[], None] | None,
) -> Timing:
    """Feed `chunk_ms` chunks paced to the wall clock, then time the last chunk to the final text.

    Chunk i is due when a microphone would have delivered it, `(i + 1) * chunk_ms` after the start
    (the last, shorter chunk when its final sample would arrive). An engine slower than real time
    gets its chunks late; `feed_lag_ms` records how late the last one was handed over.
    """
    size = sample_rate * chunk_ms // 1000
    bounds = list(range(size, len(audio), size)) + [len(audio)]
    _sync(sync)
    t_start = clock()
    fed = []
    t_last = due = t_start
    begin = 0
    for index, end in enumerate(bounds):
        due = t_start + end / sample_rate
        wait = due - clock()
        if wait > 0:
            sleep(wait)
        if index == len(bounds) - 1:
            _sync(sync)
            t_last = clock()
        fed.append(clock())
        engine.feed(audio[begin:end])
        begin = end
    text = engine.finish()
    _sync(sync)
    t_end = clock()
    return Timing(
        text,
        (t_end - t_last) * 1000.0,
        feed_lag_ms=(t_last - due) * 1000.0,
        feed_times=tuple(fed),
    )


def run_loop(
    engine,
    audio: np.ndarray,
    *,
    sample_rate: int,
    seconds: float,
    warmups: int,
    clock: Clock = time.perf_counter,
    out: _Out = sys.stdout,
) -> int:
    """Warm up, print `READY`, transcribe `audio` until `seconds` pass, print `DONE`.

    Returns the number of transcriptions after `READY`; at least one runs.
    """
    for _ in range(warmups):
        engine.transcribe(audio, sample_rate)
    out.write("READY\n")
    out.flush()
    t0 = clock()
    count = 0
    while True:
        engine.transcribe(audio, sample_rate)
        count += 1
        if clock() - t0 >= seconds:
            break
    out.write("DONE\n")
    out.flush()
    return count


# ---------------------------------------------------------------------------
# Device helpers, imported lazily so the module loads no torch
# ---------------------------------------------------------------------------


def cuda_sync(device: str) -> Callable[[], None] | None:
    if device != "cuda":
        return None
    import torch

    return torch.cuda.synchronize


def peak_reserved_mib(device: str) -> float | None:
    if device != "cuda":
        return None
    import torch

    return torch.cuda.max_memory_reserved() / (1 << 20)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def _items(material_dir: Path, utt_ids: Sequence[str]) -> list[core.MaterialItem]:
    manifest = core.verify_material(material_dir / "manifest.json")
    items = [i for i in manifest.items if not utt_ids or i.id in utt_ids]
    missing = sorted(set(utt_ids) - {i.id for i in items})
    if missing or not items:
        raise core.ManifestError(f"the manifest lists no utterance {missing or list(utt_ids)}")
    return items


def _audio(material_dir: Path, item: core.MaterialItem) -> np.ndarray:
    audio, rate = prepare_material.read_wav(material_dir / item.file)
    if rate != SAMPLE_RATE:
        raise core.ManifestError(f"{item.file} is {rate} Hz, not {SAMPLE_RATE}; rerun prepare")
    return audio


def _load(args: argparse.Namespace, out_dir: Path | None):
    t0 = time.perf_counter()
    try:
        engine = stt_engines.make_transcriber(
            args.engine, weights_dir=args.weights, threads=args.threads
        )
    except LOAD_ERRORS as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[not loaded] {args.engine}: {error}", flush=True)
        if out_dir is not None:
            core.write_json_exclusive(
                out_dir / "load.json", {"engine": args.engine, "loaded": False, "error": error}
            )
        return None
    load_s = time.perf_counter() - t0
    if out_dir is not None:
        core.write_json_exclusive(
            out_dir / "load.json", {"engine": args.engine, "loaded": True, "load_s": load_s}
        )
    return engine


def run_mode(args: argparse.Namespace) -> int:
    items = _items(args.material, args.utt)
    device = stt_engines.engine_config(args.engine)["device"]
    chunk_ms = int(stt_engines.engine_config(args.engine).get("chunk_ms", CHUNK_MS))
    engine = _load(args, args.out)
    if engine is None:
        return NOT_LOADED_EXIT
    sync = cuda_sync(device)
    streaming = stt_engines.is_streaming(engine)
    first = _audio(args.material, items[0])
    for _ in range(args.warmups):
        engine.transcribe(first, SAMPLE_RATE)

    with (args.out / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for item in items:
            audio = _audio(args.material, item)
            for rep in range(1, args.reps + 1):
                if streaming:
                    timing = stream_timed(
                        engine,
                        audio,
                        sample_rate=SAMPLE_RATE,
                        chunk_ms=chunk_ms,
                        clock=time.perf_counter,
                        sleep=time.sleep,
                        sync=sync,
                    )
                else:
                    timing = batch_timed(
                        engine, audio, sample_rate=SAMPLE_RATE, clock=time.perf_counter, sync=sync
                    )
                row = {
                    "utt_id": item.id,
                    "rep": rep,
                    "audio_s": len(audio) / SAMPLE_RATE,
                    "final_ms": timing.final_ms,
                    "text": timing.text,
                    "torch_peak_reserved_mib": peak_reserved_mib(device),
                }
                if streaming:
                    row["feed_lag_ms"] = timing.feed_lag_ms
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
    print(f"[ok] {args.engine}: {len(items)} utterances x {args.reps} reps", flush=True)
    return 0


def loop_mode(args: argparse.Namespace) -> int:
    items = _items(args.material, [args.utt])
    audio = _audio(args.material, items[0])
    engine = _load(args, None)
    if engine is None:
        return NOT_LOADED_EXIT
    run_loop(engine, audio, sample_rate=SAMPLE_RATE, seconds=args.seconds, warmups=args.warmups)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.model_bench.speech.probes.stt_probe",
        description="Speech-to-text timing (run 3) and transcription loop (run 5).",
    )
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("run", "loop"):
        mode = modes.add_parser(name)
        mode.add_argument("--engine", required=True, choices=list(stt_engines.WEIGHT_PATHS))
        mode.add_argument("--material", type=Path, default=Path("/material"))
        mode.add_argument("--weights", type=Path, default=Path("/weights"))
        mode.add_argument("--warmups", type=int, default=3)
        mode.add_argument("--threads", type=int, default=stt_engines.MAX_CPU_THREADS)
    run = modes.choices["run"]
    run.add_argument("--out", type=Path, default=Path("/out"))
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--utt", action="append", default=[], help="limit to these utterances")
    run.set_defaults(func=run_mode)
    loop = modes.choices["loop"]
    loop.add_argument("--utt", required=True, help="the utterance transcribed in the loop")
    loop.add_argument("--seconds", type=float, required=True)
    loop.set_defaults(func=loop_mode)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
