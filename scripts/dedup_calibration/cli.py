"""Command line entry point: `python -m scripts.dedup_calibration`.

Exit codes:
    0   report produced
    1   dataset unreadable or invalid, or the output file cannot be written
    2   the embedding model cannot load or run (the tool refuses to guess)
    64  command line usage error

The tool reads only the dataset and the embedding model. It opens no Neo4j
connection and makes no network call of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from backend.errors import EmbeddingError
from backend.interfaces import EmbeddingProvider

from .dataset import DEFAULT_DATASET_PATH, DatasetError, load_dataset
from .embedder import EmbedderUnavailableError, build_real_embedder
from .report import build_report, format_table
from .scoring import score_pairs

EXIT_OK = 0
EXIT_BAD_INPUT = 1
EXIT_MODEL_UNAVAILABLE = 2
EXIT_USAGE = 64

EmbedderFactory = Callable[[], tuple[EmbeddingProvider, str]]


class _Parser(argparse.ArgumentParser):
    """argparse exits 2 on a usage error, which here means "model unavailable"."""

    def error(self, message: str):  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _current_threshold() -> float:
    from backend.knowledge.curation.deduplication import SIMILARITY_THRESHOLD

    return SIMILARITY_THRESHOLD


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m scripts.dedup_calibration",
        description=(
            "Measure where labelled duplicate and distinct entity pairs fall under the "
            "dedup embedding score, and what threshold each policy implies."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="labelled pair dataset (default: the committed pairs.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the full JSON report (with every scored pair) to this path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Neo4j-score threshold to evaluate (default: SIMILARITY_THRESHOLD from "
        "backend/knowledge/curation/deduplication.py)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    embedder_factory: EmbedderFactory = build_real_embedder,
) -> int:
    """Run the calibration and return the process exit code.

    Args:
        argv: Command line arguments (defaults to `sys.argv[1:]`).
        embedder_factory: Builds the embedding provider and its model name.
            Tests inject a fake here.
    """
    args = _build_parser().parse_args(argv)

    try:
        pairs = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"dedup_calibration: invalid dataset {args.dataset}:", file=sys.stderr)
        for problem in exc.problems:
            print(f"  - {problem}", file=sys.stderr)
        return EXIT_BAD_INPUT

    threshold = args.threshold if args.threshold is not None else _current_threshold()
    if not 0.0 <= threshold <= 1.0:
        print(
            f"dedup_calibration: --threshold {threshold} is outside [0, 1] "
            "(it is a Neo4j score, (1 + cosine) / 2)",
            file=sys.stderr,
        )
        return EXIT_BAD_INPUT

    try:
        embedder, model_name = embedder_factory()
        scored = score_pairs(pairs, embedder)
    except (EmbedderUnavailableError, EmbeddingError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"dedup_calibration: refusing to run, the embedding model is unavailable: {exc}",
            file=sys.stderr,
        )
        return EXIT_MODEL_UNAVAILABLE

    report = build_report(
        scored,
        current_threshold=threshold,
        model_name=model_name,
        dataset_path=str(args.dataset),
    )
    print(format_table(report), end="")

    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"dedup_calibration: cannot write {args.output}: {exc}", file=sys.stderr)
            return EXIT_BAD_INPUT
        print(f"JSON report written to {args.output}")
    return EXIT_OK
