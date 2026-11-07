"""Recall floors of the fetcher on the fixture, with the weight-free fakes.

The run goes through the harness (scripts/eval_harness/memory_recall_run.py) over the fact,
note and noise probes, with a MemoryConfig whose `min_score`, `rerank_top_n` and block budget
are set here, so a change to the defaults cannot move the floors. Lexical probes are held to
the floors; paraphrase probes need a real embedder, so their metrics are printed only.

The type_scoped probes ask for a kind of thing ("Which devices and programs are part of my
kit?"), and every one is tagged paraphrase. The fake reranker scores the rendered fact
sentence, which holds no type words, so the fakes are not the measure of them. Their recall
over all of them, from the full report, is printed here; the 0.90 floor on it is checked with
the real models at the read path's H4 gate (plan v4, A25).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend.memory.config import MemoryConfig
from scripts.eval_harness.memory_recall import (
    MetricSet,
    RecallReport,
    parse_run_record,
    score_run,
)
from scripts.eval_harness.memory_recall_run import (
    Fixture,
    TokenEmbedder,
    TokenReranker,
    run_fixture,
)

_KINDS = ("fact", "note", "noise")
_CONFIG = MemoryConfig(
    min_score=0.4,
    rerank_top_n=20,
    block_tokens=500,
    list_max_items=16,  # the fixture's lists have 16 siblings
    deadline_ms=10_000,  # the floors measure ranking, not the time a shared runner takes
)


@dataclass(frozen=True, slots=True)
class _Reports:
    """The run scored over every probe it ran, and over its lexical probes alone."""

    full: RecallReport
    lexical: RecallReport


def _print(label: str, metrics: MetricSet) -> None:
    print(f"{label}:", json.dumps(metrics.to_json(), sort_keys=True))


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> _Reports:
    fixture = Fixture.load()
    probes = [probe for probe in fixture.probes if probe.kind in _KINDS]
    workdir: Path = tmp_path_factory.mktemp("recall-floors")
    records = asyncio.run(
        run_fixture(
            fixture,
            config=_CONFIG,
            embedder=TokenEmbedder(),
            reranker=TokenReranker(),
            workdir=workdir,
            probes=probes,
        )
    )
    run = [parse_run_record(record) for record in records]
    full = score_run(probes, run)
    _print("paraphrase", full.by_lexical["paraphrase"])
    lexical = [probe for probe in probes if probe.lexical]
    ids = {probe.probe_id for probe in lexical}
    return _Reports(
        full=full,
        lexical=score_run(lexical, [record for record in run if record.probe_id in ids]),
    )


def test_lexical_probes_reach_the_floors(reports: _Reports) -> None:
    overall = reports.lexical.overall
    _print("lexical", overall)
    assert overall.recall_at_block is not None and overall.recall_at_block >= 0.90
    assert overall.mrr is not None and overall.mrr >= 0.70
    assert overall.empty_on_noise is not None and overall.empty_on_noise >= 0.90
    assert overall.forgotten_leak in (None, 0.0)
    assert overall.budget_ok == 1.0


def test_lexical_list_item_probes_reach_recall(reports: _Reports) -> None:
    metrics = reports.lexical.by_shape["list_item"]
    _print("lexical list_item", metrics)
    assert metrics.recall_probes > 0
    assert metrics.recall_at_block is not None and metrics.recall_at_block >= 0.90


def test_type_scoped_recall_is_printed_for_the_real_model_gate(reports: _Reports) -> None:
    # Printed, not asserted: the 0.90 floor over every type_scoped probe is H4's (A25).
    _print("type_scoped", reports.full.by_shape["type_scoped"])
