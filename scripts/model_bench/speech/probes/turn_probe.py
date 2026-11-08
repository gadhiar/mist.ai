"""Run 4's container probe: Smart Turn v3.2 at every decision point of every utterance.

A decision point is offered 200 ms into each silence after the first word, as the realtime
design's detector would offer it, and Smart Turn sees the audio up to that moment. The truth is
`not_finished` for every point before the utterance's last word and `finished` for the point in
the trailing silence. Each point is inferred 5 times after 3 warm-ups of the loaded model, and
the median is the point's time.

Smart Turn v3.2 (`smart-turn-v3.2-cpu.onnx`, int8) runs on onnxruntime's CPU provider with 4
threads. Its input, per pipecat-ai/smart-turn, is a Whisper-style log-mel extractor over the last
8 s at 16 kHz, zero-padded at the front when shorter; host step H2 confirms that feature path
against the installed packages. onnxruntime and transformers are imported only inside
`load_smart_turn`, so importing this module loads no engine package.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import core
from . import prepare_material

SAMPLE_RATE = 16000
WINDOW_S = 8
WINDOW_SAMPLES = WINDOW_S * SAMPLE_RATE
FINISHED_AT = 0.5
MAX_THREADS = 4
ENGINE_ID = "smart-turn-v3.2"
# Relative to the mounted speech weights directory: H2's `<org>/<repo>/` download layout.
SMART_TURN_WEIGHTS = "pipecat-ai/smart-turn-v3/smart-turn-v3.2-cpu.onnx"


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """Where a detector would ask "is the turn over?", and the right answer."""

    cut_ms: int
    silence_start_ms: int
    silence_end_ms: int
    truth: str


def decision_points(audio: np.ndarray, sample_rate: int) -> list[DecisionPoint]:
    """One point 200 ms into every silence of 200 ms or more after the first word."""
    return [
        DecisionPoint(
            cut_ms=s.start_ms + prepare_material.MIN_SILENCE_MS,
            silence_start_ms=s.start_ms,
            silence_end_ms=s.end_ms,
            truth="finished" if s.trailing else "not_finished",
        )
        for s in prepare_material.silences(audio, sample_rate)
    ]


def smart_turn_window(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """The last 8 s of `audio` as float32, zero-padded at the front when shorter."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Smart Turn takes {SAMPLE_RATE} Hz audio, not {sample_rate}")
    tail = np.asarray(audio, dtype=np.float32)[-WINDOW_SAMPLES:]
    if len(tail) == WINDOW_SAMPLES:
        return tail
    return np.concatenate([np.zeros(WINDOW_SAMPLES - len(tail), dtype=np.float32), tail])


def called_finished(p_finished: float) -> bool:
    """Smart Turn's call: finished at a probability of 0.5 or more."""
    return p_finished >= FINISHED_AT


class SmartTurnDetector:
    """`TurnDetector` over an onnxruntime session and a Whisper feature extractor."""

    def __init__(self, session, feature_extractor):
        self._session = session
        self._features = feature_extractor

    def predict(self, audio: np.ndarray) -> float:
        """The probability that the turn is finished, for audio ending at the decision point."""
        window = smart_turn_window(audio)
        features = self._features(
            window,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        inputs = np.asarray(features.input_features, dtype=np.float32)
        outputs = self._session.run(None, {"input_features": inputs})
        return float(np.asarray(outputs[0]).reshape(-1)[0])


def load_smart_turn(model_path: Path, *, threads: int = MAX_THREADS) -> SmartTurnDetector:
    """Load the int8 ONNX model on the CPU provider with `threads` intra-op threads."""
    if not 1 <= threads <= MAX_THREADS:
        raise ValueError(f"Smart Turn runs on 1 to {MAX_THREADS} threads, not {threads}")
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return SmartTurnDetector(session, WhisperFeatureExtractor(chunk_length=WINDOW_S))


def measure_point(
    detector, audio: np.ndarray, *, reps: int, clock: Callable[[], float] = time.perf_counter
) -> tuple[float, list[float]]:
    """Infer one point `reps` times; return its probability and each inference in ms."""
    times = []
    p = 0.0
    for _ in range(reps):
        t0 = clock()
        p = detector.predict(audio)
        times.append((clock() - t0) * 1000.0)
    return p, times


def median_ms(times: Sequence[float]) -> float:
    return float(statistics.median(times))


def run(
    detector,
    material_dir: Path,
    out: Path,
    *,
    reps: int,
    warmups: int,
    utt_ids: Sequence[str] = (),
    clock: Callable[[], float] = time.perf_counter,
) -> int:
    """Write one raw row per decision point into `out/raw.jsonl`; return the row count."""
    manifest = core.verify_material(material_dir / "manifest.json")
    items = [i for i in manifest.items if not utt_ids or i.id in utt_ids]
    if not items:
        raise core.ManifestError(f"no manifest item matches {list(utt_ids)}")
    first, _ = prepare_material.read_wav(material_dir / items[0].file)
    for _ in range(warmups):
        detector.predict(first)

    count = 0
    with (out / "raw.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for item in items:
            audio, rate = prepare_material.read_wav(material_dir / item.file)
            for point in decision_points(audio, rate):
                cut = audio[: point.cut_ms * rate // 1000]
                p, times = measure_point(detector, cut, reps=reps, clock=clock)
                row = {
                    "utt_id": item.id,
                    "kind": item.kind,
                    "cut_ms": point.cut_ms,
                    "silence_start_ms": point.silence_start_ms,
                    "silence_end_ms": point.silence_end_ms,
                    "truth": point.truth,
                    "p_finished": p,
                    "called_finished": called_finished(p),
                    "inference_ms": median_ms(times),
                    "inference_ms_reps": times,
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.model_bench.speech.probes.turn_probe",
        description="Smart Turn at every decision point of the material.",
    )
    parser.add_argument("--material", type=Path, default=Path("/material"))
    parser.add_argument("--weights", type=Path, default=Path("/weights"))
    parser.add_argument("--out", type=Path, default=Path("/out"))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--threads", type=int, default=MAX_THREADS)
    parser.add_argument("--utt", action="append", default=[], help="limit to these utterances")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.perf_counter()
    detector = load_smart_turn(args.weights / SMART_TURN_WEIGHTS, threads=args.threads)
    load_s = time.perf_counter() - t0
    core.write_json_exclusive(
        args.out / "load.json", {"engine": ENGINE_ID, "loaded": True, "load_s": load_s}
    )
    count = run(
        detector,
        args.material,
        args.out,
        reps=args.reps,
        warmups=args.warmups,
        utt_ids=args.utt,
    )
    print(f"[ok] {count} decision points written to {args.out / 'raw.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
