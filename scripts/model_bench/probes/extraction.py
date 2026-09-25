"""T6 universal `extraction` suite probe.

Drives MIST's PRODUCTION extraction path -- the same code
`scripts/mist_admin.py replay --extraction-only` uses -- against a running
llama-server, and scores the result with `scripts/eval_harness/score_extraction_run.py`
UNCHANGED. This module is invoked INSIDE a fresh container of the mist-backend
image (never the production `mist-backend` container: see
`scripts.model_bench.bench_host.build_extraction_container_argv`), sharing the
bench server's network namespace, so `--base-url http://127.0.0.1:8080` reaches
`mist-bench-llm`.

Reuse, not reimplementation:
- `scripts.mist_admin.run_extraction_only_replay` runs each gold utterance
  through the handler's own extraction entry point (subject-scope classifier
  -> ontology extraction -> validation -> curation), exactly as
  `replay --extraction-only` does. This module never builds an `LLMRequest`
  itself.
- `backend.factories.build_conversation_handler` wires the handler, given an
  injected `graph_store` / `vector_store` (the unit-tier fakes -- no Neo4j) and
  the real `LlamaServerProvider` pointed at `--base-url`.
- `scripts.eval_harness.score_extraction_run` (`iter_gold_probes`,
  `iter_debug_records`, `build_produced_index`, `score_run`) is imported and
  called unchanged; this module only adds Wilson + cluster-bootstrap CIs on
  top of the `Report` it returns.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import uuid
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GOLD_CORPUS = "data/ingest/extraction-gold-2026-06-14.jsonl"

# decision_rules.json's statistics block (read-only; never edited by this
# module -- its sha256 must stay pinned, see the task's write-zone rules).
_DECISION_RULES_PATH = Path(__file__).resolve().parents[1] / "decision_rules.json"


def load_bootstrap_params(path: Path = _DECISION_RULES_PATH) -> tuple[int, int, float, float]:
    """Return (seed, B, confidence, wilson_z) from decision_rules.json's statistics block.

    Read-only: this module never writes decision_rules.json.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    stats = doc["statistics"]
    seed = int(stats["bootstrap"]["seed"])
    b = int(stats["bootstrap"]["B"])
    confidence = float(stats["bootstrap"]["confidence"])
    z = float(stats["wilson"]["z"])
    return seed, b, confidence, z


def wilson_interval(k: int, n: int, z: float) -> tuple[float, float] | None:
    """Wilson score interval for a simple (unclustered) proportion k/n.

    Byte-identical formula to `scripts.model_bench.analyse.wilson_interval`
    (duplicated rather than imported so this module has no dependency on
    `analyse.py`'s heavier import surface when run standalone inside the
    probe container; the two are covered by a parity test).
    """
    if n <= 0:
        return None
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return (lo, hi)


def _nearest_rank_percentile(values: list[float], p: float) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    rank = max(1, min(n, math.ceil((p / 100.0) * n)))
    return sorted_vals[rank - 1]


def cluster_bootstrap_ratio_ci(
    per_probe_counts: dict[str, tuple[int, int, int]],
    *,
    B: int,
    seed: int,
    confidence: float,
    kind: str,
) -> tuple[float, float] | None:
    """Cluster bootstrap CI for a precision/recall ratio metric, resampled by probe tag.

    `per_probe_counts` maps probe tag -> (tp, fp, fn) for the metric family
    (entity or relationship). Each replicate resamples probe tags with
    replacement (same count as the original corpus), pools tp/fp/fn across the
    resampled tags, and recomputes the ratio from the POOLED counts -- not a
    mean of per-probe ratios, since precision/recall are ratios of sums, not
    an average of a per-row value (that is what
    `scripts.model_bench.analyse.cluster_bootstrap_ci` computes, and why this
    module does not reuse it for these metrics).

    `kind` is "precision" (tp/(tp+fp)) or "recall" (tp/(tp+fn)).
    """
    tags = sorted(per_probe_counts)
    if not tags:
        return None
    rng = random.Random(seed)
    n_tags = len(tags)
    replicate_values: list[float] = []
    for _ in range(B):
        tp = fp = fn = 0
        for _ in range(n_tags):
            tag = tags[rng.randrange(n_tags)]
            t, f_p, f_n = per_probe_counts[tag]
            tp += t
            fp += f_p
            fn += f_n
        if kind == "precision":
            replicate_values.append(tp / (tp + fp) if (tp + fp) else 1.0)
        else:
            replicate_values.append(tp / (tp + fn) if (tp + fn) else 1.0)
    alpha = 1.0 - confidence
    lo = _nearest_rank_percentile(replicate_values, 100.0 * (alpha / 2.0))
    hi = _nearest_rank_percentile(replicate_values, 100.0 * (1.0 - alpha / 2.0))
    return (lo, hi)


def per_probe_rel_counts(per_probe: list[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    """Derive (tp, fp, fn) per probe tag for relationships from `Report.per_probe`.

    `gold_relationships` is TP+FN by construction (score_extraction_run's
    per-probe entry records every gold relationship key once); `rel_fns` is
    the unmatched subset, so tp = gold_relationships - len(rel_fns).
    """
    out: dict[str, tuple[int, int, int]] = {}
    for p in per_probe:
        fp = len(p["rel_fps"])
        fn = len(p["rel_fns"])
        tp = p["gold_relationships"] - fn
        out[p["tag"]] = (tp, fp, fn)
    return out


def per_probe_entity_counts(per_probe: list[dict[str, Any]]) -> dict[str, tuple[int, int, int]]:
    """Derive (tp, fp, fn) per probe tag for entities from `Report.per_probe`."""
    out: dict[str, tuple[int, int, int]] = {}
    for p in per_probe:
        fp = len(p["entity_fps"])
        fn = len(p["entity_fns"])
        tp = p["gold_entities"] - fn
        out[p["tag"]] = (tp, fp, fn)
    return out


def build_summary(
    report: Any,
    *,
    gold_path: Path,
    ontology_version: str,
    bootstrap_seed: int,
    bootstrap_b: int,
    bootstrap_confidence: float,
    wilson_z: float,
) -> dict[str, Any]:
    """The aggregate `extraction_summary.json` document: the scorer's own metrics
    plus Wilson intervals and a probe-id cluster bootstrap CI, unchanged from
    what `score_extraction_run.Report` computes.
    """
    entity_counts = per_probe_entity_counts(report.per_probe)
    rel_counts = per_probe_rel_counts(report.per_probe)

    rel_p, rel_r = report.rel_precision, report.rel_recall
    rel_f1 = (2 * rel_p * rel_r / (rel_p + rel_r)) if (rel_p + rel_r) else 0.0

    return {
        "schema": 1,
        "gold_corpus": str(gold_path),
        "gold_corpus_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "ontology_version": ontology_version,
        "total_probes": report.total_probes,
        "matched_probes": report.matched_probes,
        "entity_precision": report.entity_precision,
        "entity_recall": report.entity_recall,
        "rel_precision": rel_p,
        "rel_recall": rel_r,
        "rel_f1": rel_f1,
        "typing_accuracy": report.typing_accuracy,
        "related_to_rate": report.related_to_rate,
        "valid_time_accuracy": report.valid_time_accuracy,
        "negative_violations": report.negative_violations,
        "wilson": {
            "entity_precision": wilson_interval(
                report.entity_tp, report.entity_precision_denominator, wilson_z
            ),
            "entity_recall": wilson_interval(
                report.entity_tp, report.entity_recall_denominator, wilson_z
            ),
            "rel_precision": wilson_interval(
                report.rel_tp, report.rel_precision_denominator, wilson_z
            ),
            "rel_recall": wilson_interval(report.rel_tp, report.rel_recall_denominator, wilson_z),
            "typing_accuracy": wilson_interval(report.typing_ok, report.typing_total, wilson_z),
        },
        "bootstrap": {
            "method": "cluster-by-probe-id",
            "seed": bootstrap_seed,
            "B": bootstrap_b,
            "confidence": bootstrap_confidence,
            "entity_precision": cluster_bootstrap_ratio_ci(
                entity_counts, B=bootstrap_b, seed=bootstrap_seed,
                confidence=bootstrap_confidence, kind="precision",
            ),
            "entity_recall": cluster_bootstrap_ratio_ci(
                entity_counts, B=bootstrap_b, seed=bootstrap_seed,
                confidence=bootstrap_confidence, kind="recall",
            ),
            "rel_precision": cluster_bootstrap_ratio_ci(
                rel_counts, B=bootstrap_b, seed=bootstrap_seed,
                confidence=bootstrap_confidence, kind="precision",
            ),
            "rel_recall": cluster_bootstrap_ratio_ci(
                rel_counts, B=bootstrap_b, seed=bootstrap_seed,
                confidence=bootstrap_confidence, kind="recall",
            ),
        },
    }


def build_per_item_rows(report: Any) -> list[dict[str, Any]]:
    """`extraction.jsonl` rows: one per gold item, id + per-item scores + errored flag."""
    rows: list[dict[str, Any]] = []
    for p in report.per_probe:
        rows.append(
            {
                "id": p["tag"],
                "matched": p["matched"],
                "errored": not p["matched"],
                "gold_entities": p["gold_entities"],
                "gold_relationships": p["gold_relationships"],
                "entity_fp": len(p["entity_fps"]),
                "entity_fn": len(p["entity_fns"]),
                "rel_fp": len(p["rel_fps"]),
                "rel_fn": len(p["rel_fns"]),
            }
        )
    return rows


def refuse_if_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")


def run_probe(
    *,
    base_url: str,
    gold_path: Path,
    out_dir: Path,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the extraction suite end to end: drive production extraction, score,
    write `extraction.jsonl` + `extraction_summary.json` under `out_dir`.

    Refuses to overwrite either output file (checked before either is
    written). Returns the summary dict.
    """
    extraction_jsonl_path = out_dir / "extraction.jsonl"
    summary_path = out_dir / "extraction_summary.json"
    refuse_if_exists(extraction_jsonl_path)
    refuse_if_exists(summary_path)

    from scripts.eval_harness.score_extraction_run import (
        build_produced_index,
        iter_debug_records,
        iter_gold_probes,
        score_run,
    )
    from scripts.mist_admin import run_extraction_only_replay

    session_id = session_id or f"model-bench-extraction-{uuid.uuid4().hex[:12]}"
    debug_path = out_dir / "extraction_debug.jsonl"
    refuse_if_exists(debug_path)

    # Isolation env: no Neo4j (graph_store/vector_store are the unit-tier
    # fakes, injected below), event store / vault sidecar / vault root
    # redirected to a writable location inside the (otherwise read-only
    # repo mount) container, mirroring
    # scripts/eval_harness/extraction_probe_set_design.md's env recipe minus
    # the Neo4j URI (there is no Neo4j on this path at all).
    os.environ.setdefault("EVENT_STORE_DB_PATH", str(out_dir / "event_store.db"))
    os.environ.setdefault("MIST_SIDECAR_DB_PATH", str(out_dir / "vault_sidecar.db"))
    os.environ.setdefault("MIST_VAULT_ROOT", str(out_dir / "vault"))
    os.environ.setdefault("LLM_SERVER_URL", base_url)
    os.environ.setdefault("LLM_TEMPERATURE", "0.0")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ["MIST_DEBUG_JSONL"] = str(debug_path)
    os.environ["MIST_DEBUG_LLM_JSONL"] = "1"
    os.environ.setdefault("MIST_SESSION_ORIGIN", "test")

    from backend.factories import build_conversation_handler
    from backend.knowledge.config import KnowledgeConfig
    from backend.knowledge.storage.graph_store import GraphStore
    from backend.knowledge.version_stamps import ONTOLOGY_VERSION
    from tests.mocks.neo4j import FakeNeo4jConnection
    from tests.unit.knowledge.conftest import FakeEmbeddingProvider, FakeVectorStore

    config = KnowledgeConfig.from_env()
    graph_store = GraphStore(
        connection=FakeNeo4jConnection(), embedding_generator=FakeEmbeddingProvider()
    )
    handler = build_conversation_handler(
        config, graph_store=graph_store, vector_store=FakeVectorStore()
    )

    gold_probes = iter_gold_probes(gold_path)
    inputs = [{"utterance": p.utterance, "tag": p.tag} for p in gold_probes]
    asyncio.run(run_extraction_only_replay(handler, inputs, session_id))

    debug_records = iter_debug_records(debug_path, session_id=session_id)
    report = score_run(gold_probes, build_produced_index(debug_records))

    seed, b, confidence, z = load_bootstrap_params()
    summary = build_summary(
        report,
        gold_path=gold_path,
        ontology_version=ONTOLOGY_VERSION,
        bootstrap_seed=seed,
        bootstrap_b=b,
        bootstrap_confidence=confidence,
        wilson_z=z,
    )

    with open(extraction_jsonl_path, "w", encoding="utf-8") as fh:
        for row in build_per_item_rows(report):
            fh.write(json.dumps(row) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="T6 universal extraction suite probe.")
    p.add_argument("--base-url", required=True, help="llama-server base URL, e.g. http://127.0.0.1:8080")
    p.add_argument("--gold", default=DEFAULT_GOLD_CORPUS, help="Gold corpus JSONL, relative to repo root.")
    p.add_argument("--out", required=True, help="Output directory (extraction.jsonl + extraction_summary.json).")
    p.add_argument("--session-id", default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gold_path = Path(args.gold)
    if not gold_path.is_absolute():
        gold_path = _REPO_ROOT / gold_path
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_probe(
            base_url=args.base_url, gold_path=gold_path, out_dir=out_dir, session_id=args.session_id
        )
    except FileExistsError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"rel_precision": summary["rel_precision"], "typing_accuracy": summary["typing_accuracy"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
