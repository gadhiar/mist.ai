"""Recall floors of the fetcher on the fixture, with the weight-free fakes, through the need gate.

The run goes through the harness (scripts/eval_harness/memory_recall_run.py) over the fact,
note, noise and correction probes, with a MemoryConfig whose fact view, `rerank_top_n`,
`select_k`, list cap, facets and block budget are set here, so a change to the defaults cannot
move the floors. The recall figures are scored over the fact, note and noise probes alone; the
correction probes are held to `forgotten_leak` 0, which the harness's correction log (T21)
keeps. Lexical probes are held to the floors; paraphrase probes need a real embedder, so their
metrics are printed only.

The run is gated, and the gate checks the gating path, not the gate's quality. The real bank
(`load_bank`), `fit_form` (logistic, c 1), `NeedGate.decide` and the harness's gating path all
run; only the gate's embedder is an oracle. It keys on each row's fixture label alone: a bank
row by its `label`, a fact, note or correction probe as need and a noise probe as no-need, by
the probe's kind; it never sees text a run produces, and refuses any text it has no label for.
The thresholds are the Test strategy's, fit on the fact, note and noise probes alone: theta_low
is the larger of theta(0.98) over the recall probes and the float just above the top noise
probe's p, theta_high the larger of theta_low and theta(0.88), band_k 2, need_context off.
So the noise floor holds by construction and is no evidence that need and no-need turns
separate: the real gate's noise and recall are measured at
the read path's H4 and H18 gates. What the floors do show is that a gate on the path changes no
recall figure: the same walk runs ungated beside it, and the two must agree.

The type_scoped probes ask for a kind of thing ("Which devices and programs are part of my
kit?"), and every one is tagged paraphrase. The fake reranker scores the rendered fact
sentence, which holds no type words, so the fakes are not the measure of them. Their recall
over all of them, from the full report, is printed here; the 0.90 floor on it is checked with
the real models at the read path's H4 gate (plan v4, A25).
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from backend.memory.config import MemoryConfig
from backend.memory.need_gate import (
    FormSpec,
    NeedBank,
    NeedGate,
    NeedKey,
    NeedModel,
    fit_form,
    load_bank,
    need_model_key,
    theta,
)
from scripts.eval_harness.memory_recall import (
    MetricSet,
    Probe,
    RecallReport,
    parse_run_record,
    score_run,
)
from scripts.eval_harness.memory_recall_run import (
    Fixture,
    RunSetting,
    TokenEmbedder,
    TokenReranker,
    run_settings,
)

_KINDS = ("fact", "note", "noise")
_RECALL_KINDS = ("fact", "note")
_CORRECTION = "correction"
# Every setting whose default H18 may still choose is pinned, so no default moves a figure.
_CONFIG = MemoryConfig(
    fact_view="F3",
    rerank_top_n=20,
    select_k=10,
    block_tokens=500,
    list_max_items=25,  # the fixture's lists have 16 siblings
    facets=True,
    deadline_ms=10_000,  # the floors measure ranking, not the time a shared runner takes
    need_gate="on",
    need_context=False,  # the Test strategy's gate: each probe is scored on its own query
)
_UNGATED = replace(_CONFIG, need_gate="off")
_ORACLE_NAME = "oracle-fixture-label"
_ORACLE_REVISION = "bank-label-and-probe-kind-v1"
_BAND_K = 2
_FORM = FormSpec("logistic", 1.0, None)
_VECTORS = {"need": [1.0, 0.0], "no_need": [0.0, 1.0]}


class _LabelOracle:
    """The gate's embedder: each text's vector is its fixture label's, and nothing else."""

    def __init__(self, labels: Mapping[str, str]) -> None:
        self._labels = dict(labels)
        self.texts: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        label = self._labels.get(text)
        if label is None:
            raise AssertionError(f"the oracle has no fixture label for {text!r}")
        self.texts.append(text)
        return list(_VECTORS[label])

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(text) for text in texts]


def _labels(bank: NeedBank, probes: Sequence[Probe]) -> dict[str, str]:
    labels: dict[str, str] = {}
    rows = [(exemplar.text, exemplar.label) for exemplar in bank.exemplars]
    rows += [(p.query, "no_need" if p.kind == "noise" else "need") for p in probes]
    for text, label in rows:
        if labels.setdefault(text, label) != label:
            raise AssertionError(f"{text!r} carries both labels")
    return labels


def _gate(oracle: _LabelOracle, bank: NeedBank, probes: Sequence[Probe]) -> NeedGate:
    """The Test strategy's gate, fit on the real bank and thresholded on the probes' `p`."""
    vectors = oracle.generate_embeddings([exemplar.text for exemplar in bank.exemplars])
    params = fit_form(vectors, [exemplar.label for exemplar in bank.exemplars], _FORM)
    key = NeedKey(bank.sha256, _ORACLE_NAME, _ORACLE_REVISION, "")

    def gate(theta_low: float, theta_high: float) -> NeedGate:
        model = NeedModel(
            key=need_model_key(key),
            bank_sha256=key.bank_sha256,
            embedder_name=key.embedder_name,
            embedder_revision=key.embedder_revision,
            query_prefix=key.query_prefix,
            normalisation=key.normalisation,
            form=_FORM.form,
            params=params,
            theta_low=theta_low,
            theta_high=theta_high,
            band_k=_BAND_K,
            recall_target=0.98,
            source="the fixture floors test",
        )
        return NeedGate(
            oracle,
            bank,
            model,
            embedder_name=key.embedder_name,
            embedder_revision=key.embedder_revision,
            query_prefix=key.query_prefix,
        )

    scoring = gate(0.5, 0.5)
    p = {probe.probe_id: scoring.decide(scoring.embed_query(probe.query)).p for probe in probes}
    need = [p[probe.probe_id] for probe in probes if probe.kind in _RECALL_KINDS]
    noise = [p[probe.probe_id] for probe in probes if probe.kind == "noise"]
    theta_low = max(theta(need, 0.98), math.nextafter(max(noise), math.inf))
    theta_high = max(theta_low, theta(need, 0.88))
    figures = {
        "need_p": [min(need), max(need)],
        "noise_p": [min(noise), max(noise)],
        "theta_low": theta_low,
        "theta_high": theta_high,
    }
    print("gate:", json.dumps(figures, sort_keys=True))
    final = gate(theta_low, theta_high)
    oracle.texts.clear()  # what the run embeds starts here
    return final


@dataclass(frozen=True, slots=True)
class _Reports:
    """The gated run scored over every probe and over its lexical probes, and the ungated one.

    Attributes:
        full: The gated run over the fact, note and noise probes.
        corrections: The gated run over the correction probes.
        probes: Every probe run, the correction probes last.
        bands: Each probe's band from the gated run's `need` trace.
        gate_texts: Every text the gate embedded during the run, in order.
    """

    full: RecallReport
    lexical: RecallReport
    ungated_lexical: RecallReport
    corrections: RecallReport
    probes: tuple[Probe, ...]
    bands: Mapping[str, str]
    gate_texts: tuple[str, ...]


def _print(label: str, metrics: MetricSet) -> None:
    print(f"{label}:", json.dumps(metrics.to_json(), sort_keys=True))


def _lexical(probes: Sequence[Probe], records: Sequence[dict[str, object]]) -> RecallReport:
    run = [parse_run_record(record) for record in records]
    lexical = [probe for probe in probes if probe.lexical]
    ids = {probe.probe_id for probe in lexical}
    return score_run(lexical, [record for record in run if record.probe_id in ids])


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> _Reports:
    fixture = Fixture.load()
    probes = [probe for probe in fixture.probes if probe.kind in _KINDS]
    corrections = [probe for probe in fixture.probes if probe.kind == _CORRECTION]
    run = probes + corrections
    bank = load_bank()
    oracle = _LabelOracle(_labels(bank, run))
    workdir: Path = tmp_path_factory.mktemp("recall-floors")
    gated_all, ungated_all = asyncio.run(
        run_settings(
            fixture,
            settings=[
                RunSetting(_CONFIG, _gate(oracle, bank, probes)),
                RunSetting(_UNGATED, None),
            ],
            embedder=TokenEmbedder(),
            reranker=TokenReranker(),
            workdir=workdir,
            probes=run,
        )
    )
    # Records come back in `run`'s order, so the first `len(probes)` are the recall probes'.
    gated, ungated = gated_all[: len(probes)], ungated_all[: len(probes)]
    full = score_run(probes, [parse_run_record(record) for record in gated])
    corrected = score_run(
        corrections, [parse_run_record(record) for record in gated_all[len(probes) :]]
    )
    _print("paraphrase", full.by_lexical["paraphrase"])
    bands: dict[str, str] = {}
    for record in gated_all:
        need = record["need"]
        assert isinstance(need, dict)
        bands[str(record["probe_id"])] = str(need["band"])
    return _Reports(
        full=full,
        lexical=_lexical(probes, gated),
        ungated_lexical=_lexical(probes, ungated),
        corrections=corrected,
        probes=tuple(run),
        bands=bands,
        gate_texts=tuple(oracle.texts),
    )


def test_the_gating_path_decides_every_probe_from_its_query_alone(reports: _Reports) -> None:
    # The bank, fit_form, NeedGate.decide and the harness's gating ran for every probe, and the
    # gate saw each probe's own query once and nothing else. The bands follow from the oracle.
    assert sorted(reports.gate_texts) == sorted(probe.query for probe in reports.probes)
    for probe in reports.probes:
        expected = "closed" if probe.kind == "noise" else "open"
        assert reports.bands[probe.probe_id] == expected, probe.probe_id


def test_the_gate_changes_no_recall_figure(reports: _Reports) -> None:
    gated, ungated = reports.lexical, reports.ungated_lexical
    for metrics, reference in (
        (gated.overall, ungated.overall),
        (gated.by_shape["list_item"], ungated.by_shape["list_item"]),
    ):
        assert (metrics.recall_at_block, metrics.mrr, metrics.budget_ok) == (
            reference.recall_at_block,
            reference.mrr,
            reference.budget_ok,
        )


def test_lexical_probes_reach_the_floors(reports: _Reports) -> None:
    overall = reports.lexical.overall
    _print("lexical", overall)
    assert overall.recall_at_block is not None and overall.recall_at_block >= 0.90
    assert overall.mrr is not None and overall.mrr >= 0.70
    assert overall.forgotten_leak in (None, 0.0)
    assert overall.budget_ok == 1.0


def test_no_correction_probe_serves_a_forgotten_item(reports: _Reports) -> None:
    # The index keeps every corrected fact; the harness's correction log suppresses them.
    metrics = reports.corrections.overall
    _print("corrections", metrics)
    assert metrics.forgotten_probes == 10
    assert metrics.forgotten_leak == 0.0


def test_lexical_noise_probes_stay_empty(reports: _Reports) -> None:
    # Holds by construction of the oracle's thresholds; H4 and H18 measure the real gate's.
    overall = reports.lexical.overall
    assert overall.empty_on_noise is not None and overall.empty_on_noise >= 0.90


def test_lexical_list_item_probes_reach_recall(reports: _Reports) -> None:
    metrics = reports.lexical.by_shape["list_item"]
    _print("lexical list_item", metrics)
    assert metrics.recall_probes > 0
    assert metrics.recall_at_block is not None and metrics.recall_at_block >= 0.90


def test_type_scoped_recall_is_printed_for_the_real_model_gate(reports: _Reports) -> None:
    # Printed, not asserted: the 0.90 floor over every type_scoped probe is H4's (A25).
    _print("type_scoped", reports.full.by_shape["type_scoped"])
