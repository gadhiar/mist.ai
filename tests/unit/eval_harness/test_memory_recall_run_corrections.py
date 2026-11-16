"""The recall harness reads the fixture's corrections through the correction log (T21).

The harness no longer takes a corrected fact out of the index (`facts_known_at`); it records
each fixture correction through `CorrectionService` on an event store of its own, at the
correction's `recorded_at`, and the fetcher suppresses at read time what the log names. The
ten correction probes run here with the weight-free fakes: no forbidden item reaches a block,
and a forbidden note that reaches the shortlist leaves it as `dropped_suppressed`. With
`corrections_enabled` off the same walk leaks, so it is the log that keeps the block clean.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.event_store.store import EventStore
from backend.memory.config import MemoryConfig
from backend.memory.corrections import CorrectionService
from scripts.eval_harness.memory_recall import Probe, parse_run_record, score_run
from scripts.eval_harness.memory_recall_run import (
    Fixture,
    TokenEmbedder,
    TokenReranker,
    TraceSetup,
    facts_known_at,
    record_due_corrections,
    run_fixture,
    word_pair_tokens,
)

# The floors test's settings, so the shortlist these probes see is the one it scores.
_CONFIG = MemoryConfig(
    fact_view="F3",
    rerank_top_n=20,
    select_k=10,
    block_tokens=500,
    list_max_items=25,
    facets=True,
    deadline_ms=10_000,
    need_gate="off",  # every probe is served; the gate is the floors test's subject
)


@dataclass(frozen=True, slots=True)
class _Run:
    """One walk over the correction probes: its run records and its trace's probe records."""

    probes: tuple[Probe, ...]
    records: tuple[dict[str, object], ...]
    traces: tuple[dict[str, object], ...]


def _walk(fixture: Fixture, config: MemoryConfig, workdir: Path) -> _Run:
    probes = tuple(probe for probe in fixture.probes if probe.kind == "correction")
    trace = TraceSetup(
        path=workdir / "trace.jsonl", header={"record": "run"}, pair_tokens=word_pair_tokens
    )
    records = asyncio.run(
        run_fixture(
            fixture,
            config=config,
            embedder=TokenEmbedder(),
            reranker=TokenReranker(),
            workdir=workdir,
            probes=probes,
            trace=trace,
        )
    )
    lines = trace.path.read_text(encoding="utf-8").splitlines()
    traces = tuple(row for row in map(json.loads, lines) if row.get("record") == "probe")
    return _Run(probes=probes, records=tuple(records), traces=traces)


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return Fixture.load()


@pytest.fixture(scope="module")
def suppressed(fixture: Fixture, tmp_path_factory: pytest.TempPathFactory) -> _Run:
    return _walk(fixture, _CONFIG, tmp_path_factory.mktemp("corrections-on"))


def test_the_ten_correction_probes_leak_nothing(suppressed: _Run) -> None:
    assert len(suppressed.probes) == 10
    report = score_run(suppressed.probes, [parse_run_record(r) for r in suppressed.records])

    print("corrections:", json.dumps(report.overall.to_json(), sort_keys=True))
    assert report.overall.forgotten_probes == 10
    assert report.overall.forgotten_leak == 0.0


def test_a_forbidden_note_that_reaches_the_shortlist_leaves_it_suppressed(
    suppressed: _Run,
) -> None:
    reached = [
        target
        for trace in suppressed.traces
        for target in trace["targets"]
        if target["role"] == "forbidden"
        and target["kind"] == "note"
        and target["shortlist"] is not None
    ]

    assert reached, "no forbidden note reached the shortlist, so nothing here is tested"
    for target in reached:
        assert target["disposition"] == "dropped_suppressed", target["id"]
        assert target["in_block"] is False


def test_without_the_log_the_same_walk_leaks(fixture: Fixture, tmp_path: Path) -> None:
    # The control: the index still holds every corrected fact, so only the log suppresses them.
    run = _walk(fixture, replace(_CONFIG, corrections_enabled=False), tmp_path)

    report = score_run(run.probes, [parse_run_record(r) for r in run.records])

    assert report.overall.forgotten_leak is not None and report.overall.forgotten_leak > 0.0


def test_the_index_keeps_a_corrected_fact(fixture: Fixture) -> None:
    latest = max(probe.now for probe in fixture.probes)
    known = {row.fact_key for row in facts_known_at(fixture.facts, fixture.corrections, latest)}

    assert {correction.fact_key for correction in fixture.corrections} <= known


def test_corrections_are_recorded_once_each_at_their_recorded_at(fixture: Fixture) -> None:
    store = EventStore(db_path=":memory:")
    store.initialize()
    service = CorrectionService(store, now_fn=lambda: datetime(2001, 1, 1, tzinfo=UTC))
    ordered = sorted(fixture.corrections, key=lambda c: (c.recorded_at, c.correction_id))
    first = ordered[0]
    recorded: set[str] = set()

    before = record_due_corrections(
        service, fixture.corrections, first.recorded_at - timedelta(seconds=1), recorded
    )
    at_first = record_due_corrections(service, fixture.corrections, first.recorded_at, recorded)
    again = record_due_corrections(service, fixture.corrections, first.recorded_at, recorded)
    rest = record_due_corrections(service, fixture.corrections, ordered[-1].recorded_at, recorded)

    assert (before, again) == ([], [])
    assert [record.correction_id for record in at_first] == [first.correction_id]
    assert len(rest) == len(ordered) - 1
    rows = store.list_memory_corrections()
    assert [row["correction_id"] for row in rows] == [c.correction_id for c in ordered]
    for row, correction in zip(rows, ordered, strict=True):
        assert datetime.fromisoformat(row["recorded_at"]) == correction.recorded_at
        assert (row["action"], row["fact_key"], row["origin"]) == (
            correction.action,
            correction.fact_key,
            correction.origin,
        )
        assert row["replacement_object_name"] == correction.replacement_object_name
    store.close()
