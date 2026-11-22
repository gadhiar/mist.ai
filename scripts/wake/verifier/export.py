"""Export a trained verifier run to ONNX, check it against torch, and time it in ONNX Runtime.

    python -m scripts.wake.verifier.export --run <run dir> --dataset <dir> \
        --manifest <train-only manifest> --root-map <from>=<to> --out <model.onnx> \
        --bench-threads 1,2,4 [--bench-contention <threads>]

The graph is `torch.onnx.export(..., dynamo=False)` at opset 17 of `model.ScoreGraph`: input
`waveform` float32 `[1, n]` with `n` dynamic, output `score` float32 `[1]`. torch 2.6's
TorchScript exporter needs the `onnx` package (`torch/onnx/utils.py` calls
`_add_onnxscript_fn`), which reaches the image only through chatterbox-tts, so the command
exits 2 naming `onnx` when it cannot import it, before anything else.

Parity: 256 windows of the index, read through the train-only manifest (refused, exit 2,
unless its sha256 is the index's `training` and it lists train entries alone), each cut to 800,
1600 and 2320 ms around its hit where it is that long, are scored by ONNX Runtime on the CPU and
by torch; the largest absolute difference must be at most `PARITY_BOUND`, else exit 1.
Latency: ONNX Runtime's p50 and p90 on one 2320 ms window at each `--bench-threads` count, and
with `--bench-contention` at 2 threads while another process runs the same model on that many
threads in a loop. The figures go to `export.json` beside the model and to stdout, never a file
name. onnxruntime is imported only at run time. Exit 0 written and in parity, 1 out of parity or
an engine failed, 2 refused.
"""

import argparse
import json
import multiprocessing
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from backend.realtime.engines.silero_vad import CPU_PROVIDER, SessionSpec, open_onnx_session
from backend.realtime.errors import SpeechEngineError

from ..manifest import ManifestError
from .dataset import AFTER_FRAMES, FRAME_SAMPLES, DatasetError, Window, parse_root_map
from .model import ModelError, ScoreGraph, VerifierModel, build_verifier
from .train import (
    RUN_CONFIG,
    RUN_MODEL,
    TrainError,
    WindowReader,
    hit_offset,
    load_index,
    train_manifest,
)

OPSET = 17
PARITY_BOUND = 1e-3
PARITY_WINDOWS = 256
PARITY_FRAMES = (40, 80, 116)
"""800, 1600 and 2320 ms."""
BENCH_FRAMES = 116
BENCH_RUNS = 50
BENCH_WARMUP = 5
REPORT_FILE = "export.json"
ONNX_MISSING = (
    "the onnx package cannot be imported; torch.onnx.export's TorchScript exporter needs it "
    "(Design 2), and it reaches the image through chatterbox-tts"
)


class ExportError(Exception):
    """A refusal; the command exits 2 with the message."""


def export_onnx(model: VerifierModel, path: Path, *, probe_samples: int = 37120) -> None:
    """Write `model` as the serving graph: `waveform` `[1, n]` to `score` `[1]`, opset 17."""
    graph = ScoreGraph(model).eval()
    probe = torch.zeros(1, probe_samples)
    with torch.no_grad():
        torch.onnx.export(
            graph,
            (probe,),
            str(path),
            input_names=["waveform"],
            output_names=["score"],
            dynamic_axes={"waveform": {1: "samples"}},
            opset_version=OPSET,
            dynamo=False,
        )


def load_run(run: Path) -> VerifierModel:
    """The run's model, rebuilt from its `config.json` params and loaded strictly from `model.pt`.

    Raises:
        ExportError: The run's config or state dict is missing or does not fit.
    """
    try:
        config = json.loads((run / RUN_CONFIG).read_text(encoding="utf-8"))
        params = config["params"]
        first = config["first_trainable_layer"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ExportError(f"the run's {RUN_CONFIG} could not be read: {exc}") from exc
    model = build_verifier(params, first_trainable=first)
    try:
        state = torch.load(run / RUN_MODEL, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    except (OSError, RuntimeError) as exc:
        raise ExportError(f"the run's {RUN_MODEL} does not load: {exc}") from exc
    return model.eval()


def crop_around_hit(window: Window, frames: int) -> tuple[int, int] | None:
    """Sample offsets within the window of `frames` frames ending `AFTER_FRAMES` past the hit."""
    total = (window.end - window.start) // FRAME_SAMPLES
    if total < frames:
        return None
    first = min(max(hit_offset(window) + AFTER_FRAMES - frames + 1, 0), total - frames)
    return first * FRAME_SAMPLES, (first + frames) * FRAME_SAMPLES


Runner = Callable[[np.ndarray], float]


def parity(
    model: VerifierModel,
    onnx_score: Runner,
    windows: Sequence[Window],
    reader: WindowReader,
    *,
    seed: int = 17,
) -> tuple[float, int]:
    """The largest |ONNX - torch| over up to `PARITY_WINDOWS` windows at each parity length."""
    rng = random.Random(seed)  # nosec B311 - a reproducible draw only
    chosen = rng.sample(list(windows), min(PARITY_WINDOWS, len(windows)))
    worst, count = 0.0, 0
    graph = ScoreGraph(model).eval()
    for window in chosen:
        samples = reader.samples(window).astype(np.float32) / np.float32(32768.0)
        for frames in PARITY_FRAMES:
            bounds = crop_around_hit(window, frames)
            if bounds is None:
                continue
            waveform = samples[bounds[0] : bounds[1]].reshape(1, -1)
            with torch.no_grad():
                expected = float(graph(torch.from_numpy(waveform))[0])
            worst = max(worst, abs(onnx_score(waveform) - expected))
            count += 1
    return worst, count


def _session(path: str, threads: int):  # noqa: ANN202 - an onnxruntime session
    return open_onnx_session(
        SessionSpec(
            path=path,
            providers=(CPU_PROVIDER,),
            device_id=0,
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
        )
    )


def _latency(path: str, threads: int) -> dict[str, float]:
    session = _session(path, threads)
    waveform = np.zeros((1, BENCH_FRAMES * FRAME_SAMPLES), dtype=np.float32)
    waveform += np.random.default_rng(0).normal(0, 0.05, waveform.shape).astype(np.float32)
    for _ in range(BENCH_WARMUP):
        session.run(None, {"waveform": waveform})
    times = []
    for _ in range(BENCH_RUNS):
        began = time.perf_counter()
        session.run(None, {"waveform": waveform})
        times.append(1000 * (time.perf_counter() - began))
    cuts = statistics.quantiles(times, n=10)
    return {"p50": statistics.median(times), "p90": cuts[8]}


def _contend(path: str, threads: int, stop: object) -> None:
    session = _session(path, threads)
    waveform = np.zeros((1, BENCH_FRAMES * FRAME_SAMPLES), dtype=np.float32)
    while not stop.is_set():  # type: ignore[attr-defined]
        session.run(None, {"waveform": waveform})


def _threads(value: str) -> list[int]:
    try:
        counts = [int(part) for part in value.split(",") if part]
    except ValueError as exc:
        raise ExportError(f"--bench-threads must be integers, got {value!r}") from exc
    if not counts or min(counts) < 1:
        raise ExportError("--bench-threads must list thread counts of at least 1")
    return counts


def export(args: argparse.Namespace) -> int:
    """Check the inputs, export, check parity, time it and write the report."""
    try:
        import onnx  # noqa: PLC0415 - checked first, so its absence is the message
    except ImportError as exc:
        raise ExportError(ONNX_MISSING) from exc
    threads = _threads(args.bench_threads)
    indexed = load_index(args.dataset)
    manifest = train_manifest(args.manifest, parse_root_map(args.root_map), indexed.training)
    reader = WindowReader(manifest, indexed.roots)
    model = load_run(args.run)
    if args.out.exists():
        raise ExportError("--out already exists; the model is written new")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, args.out)
    onnx.checker.check_model(str(args.out))
    session = _session(str(args.out), 2)

    def onnx_score(waveform: np.ndarray) -> float:
        return float(session.run(None, {"waveform": waveform})[0].reshape(-1)[0])

    worst, compared = parity(model, onnx_score, indexed.windows, reader)
    latency = {str(count): _latency(str(args.out), count) for count in threads}
    contention = None
    if args.bench_contention:
        context = multiprocessing.get_context("spawn")
        stop = context.Event()
        worker = context.Process(target=_contend, args=(str(args.out), args.bench_contention, stop))
        worker.start()
        try:
            time.sleep(2.0)
            contention = {"threads": 2, "beside": args.bench_contention}
            contention.update(_latency(str(args.out), 2))
        finally:
            stop.set()
            worker.join()
    report = {
        "version": 1,
        "opset": OPSET,
        "parity": {"compared": compared, "max_abs_diff": worst, "bound": PARITY_BOUND},
        "latency_ms": latency,
        "contention_ms": contention,
        "training": indexed.training,
    }
    report_path = args.out.parent / REPORT_FILE
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"parity: max |onnx - torch| {worst:.2e} over {compared} windows")
    for count, figures in latency.items():
        print(f"{count} threads: p50 {figures['p50']:.1f} ms, p90 {figures['p90']:.1f} ms")
    if contention is not None:
        print(
            f"2 threads beside {args.bench_contention}: p50 {contention['p50']:.1f} ms, "
            f"p90 {contention['p90']:.1f} ms"
        )
    return 0 if worst <= PARITY_BOUND else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.export", description=__doc__
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root-map", action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bench-threads", default="1,2,4")
    parser.add_argument("--bench-contention", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export; see the module docstring for the exit codes."""
    args = _parser().parse_args(argv)
    try:
        return export(args)
    except (ExportError, TrainError, ManifestError, DatasetError, ModelError) as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 2
    except SpeechEngineError as exc:
        print(f"export: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
