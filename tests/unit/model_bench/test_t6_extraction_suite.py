"""T6: the universal `extraction` suite.

Covers (see the T6 task brief's "Red before green" list):
- `--suites extraction` is accepted for an arm that does not declare it
  (`UNIVERSAL_SUITES`).
- Every arm's resolved `arm_config` is unchanged by this task (T6 never
  touches `arms.json` or `resolve_arm`).
- The extraction container's `docker run` argv is safe: `docker run --rm`,
  no `exec`/`start`, never names the production `mist-backend` container.
- `probes/extraction.py` drives the production extraction path by delegating
  to `backend.knowledge.extraction.ontology_extractor.OntologyConstrainedExtractor`
  (via a fake LLM provider capturing the request) rather than building its
  own `LLMRequest`.
- `score_extraction_run.py`'s scoring functions are imported and called, not
  reimplemented, by `probes/extraction.py`.
- The summary math (Wilson interval, cluster-by-probe-id bootstrap CI,
  rel_f1) is correct on a small hand-computable fixture.
- `analyse.py`'s new "Extraction quality" section is report-only (no
  verdict) and the v1 `Rules` + `Exploratory` sections stay byte-identical.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402

from scripts.model_bench import analyse  # noqa: E402
from scripts.model_bench import bench_host  # noqa: E402
from scripts.model_bench.probes import extraction as extraction_probe  # noqa: E402

FIXTURE_RUN = Path(__file__).resolve().parent / "fixtures" / "analyse" / "run"


# ---------------------------------------------------------------------------
# validate_run_suites: extraction is universal
# ---------------------------------------------------------------------------


def test_extraction_universal_suite_constant():
    assert bench_host.UNIVERSAL_SUITES == ("extraction",)


def test_extraction_accepted_for_a_non_declaring_arm():
    arms_doc = bench_host.load_arms_doc()
    arm = bench_host.resolve_arm(arms_doc, "c1-512")
    assert "extraction" not in arm["suites"]
    assert bench_host.validate_run_suites(arm, ["extraction"], None) == ["extraction"]


def test_extraction_sorts_last_in_run_order():
    arms_doc = bench_host.load_arms_doc()
    arm = bench_host.resolve_arm(arms_doc, "c0")
    result = bench_host.validate_run_suites(arm, ["extraction", "correctness"], None)
    assert result == ["correctness", "extraction"]
    assert bench_host.RUN_SUITE_ORDER[-1] == "extraction"


def test_default_suites_unaffected_by_universal_extraction():
    """A `run` with no --suites still runs exactly the pre-T6 default suites."""
    arms_doc = bench_host.load_arms_doc()
    arm = bench_host.resolve_arm(arms_doc, "a1")
    assert bench_host.validate_run_suites(arm, None, None) == ["ttft", "correctness", "harness"]


def test_arm_still_declaring_extraction_would_also_work(monkeypatch):
    """UNIVERSAL_SUITES is an allowance, not exclusive: an arm that DID declare
    extraction (hypothetically) would still resolve it via the normal branch."""
    arms_doc = bench_host.load_arms_doc()
    arm = dict(bench_host.resolve_arm(arms_doc, "c1-512"))
    arm["suites"] = [*arm["suites"], "extraction"]
    assert bench_host.validate_run_suites(arm, ["extraction"], None) == ["extraction"]


# ---------------------------------------------------------------------------
# arm_config unchanged: T6 never touches resolve_arm / arms.json
# ---------------------------------------------------------------------------


def test_every_arm_config_unchanged_by_t6():
    arms_doc = bench_host.load_arms_doc()
    resolved = bench_host.resolve_all_arms(arms_doc)
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", "HEAD:scripts/model_bench/arms.json"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot read HEAD arms.json via git show: {proc.stderr.strip()}")
    head_doc = json.loads(proc.stdout)
    head_resolved = bench_host.resolve_all_arms(head_doc)
    assert resolved == head_resolved


# ---------------------------------------------------------------------------
# docker run argv safety
# ---------------------------------------------------------------------------


def _argv():
    return bench_host.build_extraction_container_argv(
        backend_image="sha256:deadbeefcafe",
        repo_root=Path("/repo"),
        out_dir=Path("/results/run1/c0"),
        base_url="http://127.0.0.1:8080",
    )


def test_argv_starts_with_docker_run_rm():
    argv = _argv()
    assert argv[0:3] == ["docker", "run", "--rm"]


def test_argv_never_execs_or_starts():
    argv = _argv()
    assert "exec" not in argv
    assert "start" not in argv


def test_argv_never_names_the_production_backend_container():
    argv = _argv()
    assert "mist-backend" not in argv
    # The network target is the bench llama-server container, never the
    # production backend.
    assert f"container:{bench_host.BENCH_LLM_CONTAINER}" in argv
    assert "container:mist-backend" not in argv


def test_argv_mounts_repo_read_only_and_out_writable():
    argv = _argv()
    assert "/repo:/work:ro" in argv
    assert "/results/run1/c0:/out" in argv


def test_resolve_backend_image_ref_reads_snapshot(tmp_path):
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps({"mist-backend": {"Image": "sha256:abc123"}}), encoding="utf-8")
    assert bench_host.resolve_backend_image_ref(snap) == "sha256:abc123"


def test_resolve_backend_image_ref_missing_raises(tmp_path):
    snap = tmp_path / "snapshot.json"
    snap.write_text(json.dumps({"mist-llm": {"Image": "sha256:xyz"}}), encoding="utf-8")
    with pytest.raises(bench_host.ImageRefError):
        bench_host.resolve_backend_image_ref(snap)


def test_resolve_backend_image_ref_missing_snapshot_file_raises_cleanly(tmp_path):
    """T6 reviewer finding 4: a missing snapshot file must not surface as an
    uncaught FileNotFoundError -- cmd_run's preflight needs an ImageRefError
    it can catch and turn into a clean [FAIL]."""
    with pytest.raises(bench_host.ImageRefError):
        bench_host.resolve_backend_image_ref(tmp_path / "does-not-exist.json")


# ---------------------------------------------------------------------------
# T6 reviewer finding 3: PYTHONHASHSEED / MIST_FIXED_CLOCK are docker run -e
# flags, not in-process env (setting them after interpreter start has no
# effect on PYTHONHASHSEED).
# ---------------------------------------------------------------------------


def test_argv_sets_pythonhashseed_via_docker_run_e():
    argv = _argv()
    idx = argv.index("-e")
    assert argv[idx + 1] == "PYTHONHASHSEED=0"


def test_argv_sets_mist_fixed_clock_via_docker_run_e():
    argv = _argv()
    assert "-e" in argv
    assert "MIST_FIXED_CLOCK=2026-06-13T00:00:00+00:00" in argv
    # Both -e flags precede the image ref (docker run positional argument
    # ordering: options first, then image, then command).
    image_idx = argv.index("sha256:deadbeefcafe")
    e_positions = [i for i, tok in enumerate(argv) if tok == "-e"]
    assert e_positions and all(i < image_idx for i in e_positions)


# ---------------------------------------------------------------------------
# suite_output_paths: refuses to overwrite either output file
# ---------------------------------------------------------------------------


def test_suite_output_paths_covers_both_extraction_files(tmp_path):
    paths = bench_host.suite_output_paths(tmp_path, ["extraction"], rep=1, layout_pass="screen")
    assert paths["extraction"] == tmp_path / "extraction.jsonl"
    assert paths["extraction_summary"] == tmp_path / "extraction_summary.json"


def test_refuse_if_exists_blocks_an_existing_extraction_jsonl(tmp_path):
    (tmp_path / "extraction.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(bench_host.SuiteOutputExistsError):
        bench_host.refuse_if_exists(tmp_path / "extraction.jsonl", "extraction.jsonl")


def test_refuse_if_exists_blocks_an_existing_extraction_summary(tmp_path):
    (tmp_path / "extraction_summary.json").write_text("x", encoding="utf-8")
    with pytest.raises(bench_host.SuiteOutputExistsError):
        bench_host.refuse_if_exists(tmp_path / "extraction_summary.json", "extraction_summary.json")


# ---------------------------------------------------------------------------
# probes/extraction.py: reuse, not reimplementation
# ---------------------------------------------------------------------------


def test_probe_module_does_not_define_its_own_scorer():
    """The scorer is imported and called, never copied: `score_run` /
    `iter_gold_probes` / `build_produced_index` are not redefined at module
    scope in probes/extraction.py (they are local-imported inside
    `run_probe`, from `scripts.eval_harness.score_extraction_run`)."""
    for name in ("score_run", "iter_gold_probes", "build_produced_index", "iter_debug_records"):
        assert not hasattr(extraction_probe, name), f"{name} must not be redefined in probes/extraction.py"


def test_probe_module_does_not_build_its_own_llm_request():
    """probes/extraction.py never constructs `backend.llm.models.LLMRequest`
    itself -- it delegates entirely to `run_extraction_only_replay`."""
    source = inspect.getsource(extraction_probe)
    assert "LLMRequest(" not in source
    assert "run_extraction_only_replay" in source
    assert "build_conversation_handler" in source


def test_default_gold_corpus_matches_the_v1_4_0_adjudicated_corpus():
    assert extraction_probe.DEFAULT_GOLD_CORPUS == "data/ingest/extraction-gold-2026-06-14.jsonl"
    assert (_REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS).is_file()


# ---------------------------------------------------------------------------
# probes/extraction.py: the production extractor request shape
# ---------------------------------------------------------------------------


class _CapturingLLMProvider:
    """Captures every `LLMRequest` the ontology extractor builds. Satisfies
    just enough of `LLMProvider` for `OntologyConstrainedExtractor.invoke`."""

    def __init__(self, response_content: str) -> None:
        self.calls = []
        self._response_content = response_content
        # ConversationHandler.__init__ logs llm_provider.model -- only needed
        # by the (T6 finding 2) full-handler ext-11 gate tests below, which
        # go through build_conversation_handler rather than calling
        # OntologyConstrainedExtractor directly.
        self.model = "fake-model"

    async def invoke(self, request):
        from backend.llm.models import LLMResponse

        self.calls.append(request)
        return LLMResponse(content=self._response_content, partial=False)


def test_ontology_extractor_request_matches_production_shape():
    """The production extractor (`OntologyConstrainedExtractor.extract`, the
    same code `run_extraction_only` reaches via `_extract_knowledge_async`)
    builds a `json_mode=True`, `max_tokens=2048`,
    `temperature=config.llm.temperature` request. probes/extraction.py never
    builds this request itself (see the two tests above) -- it is produced
    by this SAME production code, so asserting its shape here is what
    "the probe's request build equals the production extractor's own" means.
    """
    from backend.knowledge.config import KnowledgeConfig
    from backend.knowledge.extraction.ontology_extractor import OntologyConstrainedExtractor
    from backend.knowledge.extraction.preprocessor import PreProcessedInput
    from datetime import UTC, datetime

    config = KnowledgeConfig.from_env()
    provider = _CapturingLLMProvider('{"entities": [], "relationships": []}')
    extractor = OntologyConstrainedExtractor(config, provider)
    pre_processed = PreProcessedInput(
        original_text="I use Rust",
        resolved_text="I use Rust",
        conversation_context=[],
        reference_date=datetime(2026, 6, 13, tzinfo=UTC),
        turn_index=0,
        metadata={"subject_scope": "user"},
    )

    asyncio.run(extractor.extract(pre_processed))

    assert len(provider.calls) == 1
    request = provider.calls[0]
    assert request.json_mode is True
    assert request.max_tokens == 2048
    assert request.temperature == config.llm.temperature
    assert "I use Rust" in request.messages[-1]["content"]


# ---------------------------------------------------------------------------
# T6 reviewer finding 2: the significance gate must not spuriously skip
# ext-11-smalltalk-negative because of the fake embedding provider's
# collisions. Runs the REAL ExtractionPipeline (via build_conversation_handler
# + run_extraction_only_replay, the same production path probes/extraction.py
# drives) over the gold corpus's first 11 probes, in order, so the dedup
# cache accumulates exactly as it would in a full 60-probe replay.
# ---------------------------------------------------------------------------


def _gold_inputs_through_ext11() -> list[dict[str, str]]:
    inputs: list[dict[str, str]] = []
    gold_path = _REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        inputs.append({"utterance": rec["utterance"], "tag": rec["tag"]})
        if rec["tag"] == "ext-11-smalltalk-negative":
            break
    return inputs


def _ext11_llm_call_count(embedding_provider) -> int:
    from backend.factories import build_conversation_handler
    from backend.knowledge.config import KnowledgeConfig
    from backend.knowledge.storage.graph_store import GraphStore
    from tests.mocks.neo4j import FakeNeo4jConnection
    from tests.unit.knowledge.conftest import FakeVectorStore

    provider = _CapturingLLMProvider('{"entities": [], "relationships": []}')
    config = KnowledgeConfig.from_env()
    graph_store = GraphStore(connection=FakeNeo4jConnection(), embedding_generator=embedding_provider)
    handler = build_conversation_handler(
        config, llm_provider=provider, graph_store=graph_store, vector_store=FakeVectorStore()
    )
    inputs = _gold_inputs_through_ext11()
    asyncio.run(_run_replay(handler, inputs))
    ext11_utterance = next(i["utterance"] for i in inputs if i["tag"] == "ext-11-smalltalk-negative")
    return sum(
        1
        for call in provider.calls
        if any(ext11_utterance in (m.get("content") or "") for m in call.messages)
    )


async def _run_replay(handler, inputs):
    from scripts.mist_admin import run_extraction_only_replay

    await run_extraction_only_replay(handler, inputs, "test-ext11-gate")


def test_ext11_is_gated_out_by_the_old_conftest_fake_embedding_provider():
    """Documents the bug: tests.unit.knowledge.conftest.FakeEmbeddingProvider's
    tiled, all-positive vectors spuriously collide, so the significance gate
    (pipeline.py:673-674) skips ext-11-smalltalk-negative before it ever
    reaches the LLM."""
    from tests.unit.knowledge.conftest import FakeEmbeddingProvider as OldFakeEmbeddingProvider

    assert _ext11_llm_call_count(OldFakeEmbeddingProvider()) == 0


def test_ext11_reaches_the_llm_with_the_near_orthogonal_fake_embedding_provider():
    """The fix: probes/extraction.py's HashSeededUnitEmbeddingProvider does
    not spuriously collide, so ext-11-smalltalk-negative reaches the LLM."""
    assert _ext11_llm_call_count(extraction_probe.HashSeededUnitEmbeddingProvider()) >= 1


# ---------------------------------------------------------------------------
# probes/extraction.py: summary math
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_analyse_py_formula():
    """Duplicated formula (see the module docstring) -- pinned against
    `analyse.py`'s own `wilson_interval` for a spread of k/n."""
    z = 1.9599639845400545
    for k, n in [(9, 10), (1, 1), (0, 5), (3, 3), (27, 60)]:
        assert extraction_probe.wilson_interval(k, n, z) == analyse.wilson_interval(k, n, z)


def test_wilson_interval_none_on_zero_denominator():
    assert extraction_probe.wilson_interval(0, 0, 1.96) is None


def test_cluster_bootstrap_ratio_ci_precision_is_between_0_and_1():
    counts = {"p1": (2, 0, 0), "p2": (1, 1, 0), "p3": (0, 0, 1)}
    ci = extraction_probe.cluster_bootstrap_ratio_ci(
        counts, B=200, seed=1, confidence=0.95, kind="precision"
    )
    assert ci is not None
    lo, hi = ci
    assert 0.0 <= lo <= hi <= 1.0


def test_cluster_bootstrap_ratio_ci_deterministic_for_fixed_seed():
    counts = {"p1": (2, 0, 0), "p2": (1, 1, 0), "p3": (0, 0, 1)}
    ci_a = extraction_probe.cluster_bootstrap_ratio_ci(
        counts, B=500, seed=42, confidence=0.95, kind="recall"
    )
    ci_b = extraction_probe.cluster_bootstrap_ratio_ci(
        counts, B=500, seed=42, confidence=0.95, kind="recall"
    )
    assert ci_a == ci_b


def test_cluster_bootstrap_ratio_ci_empty_returns_none():
    assert (
        extraction_probe.cluster_bootstrap_ratio_ci({}, B=10, seed=1, confidence=0.95, kind="precision")
        is None
    )


def test_per_probe_rel_counts_derives_tp_from_gold_minus_fn():
    per_probe = [
        {"tag": "p1", "gold_relationships": 3, "rel_fps": [["a", "USES", "b"]], "rel_fns": [["c", "USES", "d"]]}
    ]
    counts = extraction_probe.per_probe_rel_counts(per_probe)
    # tp = gold_relationships - fn = 3 - 1 = 2; fp = 1; fn = 1
    assert counts["p1"] == (2, 1, 1)


def test_per_probe_entity_counts_derives_tp_from_gold_minus_fn():
    per_probe = [{"tag": "p1", "gold_entities": 2, "entity_fps": [], "entity_fns": [["a", "T"]]}]
    counts = extraction_probe.per_probe_entity_counts(per_probe)
    assert counts["p1"] == (1, 0, 1)


class _FakeReport:
    """Minimal stand-in for score_extraction_run.Report's public surface
    that build_summary/build_per_item_rows reads."""

    def __init__(self):
        self.per_probe = [
            {
                "tag": "p1",
                "matched": True,
                "gold_entities": 2,
                "gold_relationships": 2,
                "entity_fps": [],
                "entity_fns": [],
                "rel_fps": [],
                "rel_fns": [],
            },
            {
                "tag": "p2",
                "matched": False,
                "gold_entities": 1,
                "gold_relationships": 1,
                "entity_fps": [],
                "entity_fns": [["x", "T"]],
                "rel_fps": [],
                "rel_fns": [["x", "USES", "y"]],
            },
        ]
        self.total_probes = 2
        self.matched_probes = 1
        self.entity_tp = 2
        self.entity_precision_denominator = 2
        self.entity_recall_denominator = 3
        self.rel_tp = 2
        self.rel_precision_denominator = 2
        self.rel_recall_denominator = 3
        self.typing_ok = 2
        self.typing_total = 2
        self.related_to_count = 0
        self.produced_rel_total = 2
        self.valid_time_ok = 1
        self.valid_time_total = 1
        self.negative_violations = 0

    @property
    def entity_precision(self):
        return self.entity_tp / self.entity_precision_denominator

    @property
    def entity_recall(self):
        return self.entity_tp / self.entity_recall_denominator

    @property
    def rel_precision(self):
        return self.rel_tp / self.rel_precision_denominator

    @property
    def rel_recall(self):
        return self.rel_tp / self.rel_recall_denominator

    @property
    def typing_accuracy(self):
        return self.typing_ok / self.typing_total

    @property
    def related_to_rate(self):
        return self.related_to_count / self.produced_rel_total

    @property
    def valid_time_accuracy(self):
        return self.valid_time_ok / self.valid_time_total


def test_build_per_item_rows_flags_unmatched_as_errored():
    rows = extraction_probe.build_per_item_rows(_FakeReport())
    by_id = {r["id"]: r for r in rows}
    assert by_id["p1"]["errored"] is False
    assert by_id["p2"]["errored"] is True


def test_build_summary_rel_f1_is_harmonic_mean():
    report = _FakeReport()
    summary = extraction_probe.build_summary(
        report,
        gold_path=Path(_REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS),
        ontology_version="1.4.0",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    p, r = report.rel_precision, report.rel_recall
    expected_f1 = 2 * p * r / (p + r)
    assert summary["rel_f1"] == pytest.approx(expected_f1)
    assert summary["ontology_version"] == "1.4.0"
    assert summary["wilson"]["rel_precision"] is not None
    assert summary["bootstrap"]["rel_precision"] is not None


def test_build_summary_complete_true_when_matched_equals_total():
    """T6 reviewer finding 1b: `complete` mirrors the SAME invariant
    score_extraction_run.py:812 checks (`matched_probes == total_probes`),
    computed from `report.per_probe`'s own `matched` flags."""
    report = _FakeReport()
    report.per_probe[1]["matched"] = True  # make both probes matched
    report.matched_probes = 2
    summary = extraction_probe.build_summary(
        report,
        gold_path=Path(_REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS),
        ontology_version="1.4.0",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    assert summary["complete"] is True
    assert summary["unmatched_probe_ids"] == []


def test_build_summary_incomplete_when_a_probe_is_unmatched():
    """The `_FakeReport` fixture's p2 is unmatched by construction -- this is
    the "fewer probes matched than exist" scenario the fail-closed check
    (probes/extraction.py's `main`) must catch."""
    report = _FakeReport()
    summary = extraction_probe.build_summary(
        report,
        gold_path=Path(_REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS),
        ontology_version="1.4.0",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    assert summary["complete"] is False
    assert summary["unmatched_probe_ids"] == ["p2"]


def test_build_summary_empty_corpus_never_complete():
    """Mirrors score_extraction_run.py:804-810's guard: on an empty corpus,
    matched_probes == total_probes is a vacuous 0 == 0 -- must not read as
    complete."""
    report = _FakeReport()
    report.per_probe = []
    report.total_probes = 0
    report.matched_probes = 0
    summary = extraction_probe.build_summary(
        report,
        gold_path=Path(_REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS),
        ontology_version="1.4.0",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
    )
    assert summary["complete"] is False


def test_build_summary_records_env_values():
    report = _FakeReport()
    summary = extraction_probe.build_summary(
        report,
        gold_path=Path(_REPO_ROOT / extraction_probe.DEFAULT_GOLD_CORPUS),
        ontology_version="1.4.0",
        bootstrap_seed=1,
        bootstrap_b=50,
        bootstrap_confidence=0.95,
        wilson_z=1.96,
        rate_limit_max_per_minute=61,
        pythonhashseed="0",
        mist_fixed_clock="2026-06-13T00:00:00+00:00",
    )
    assert summary["env"] == {
        "rate_limit_max_per_minute": 61,
        "pythonhashseed": "0",
        "mist_fixed_clock": "2026-06-13T00:00:00+00:00",
    }


def test_refuse_if_exists_raises_file_exists_error(tmp_path):
    p = tmp_path / "extraction.jsonl"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        extraction_probe.refuse_if_exists(p)


# ---------------------------------------------------------------------------
# analyse.py: report-only "Extraction quality" section
# ---------------------------------------------------------------------------


def _section(report_text: str, heading: str, next_heading: str) -> str:
    start = report_text.index(heading)
    end = report_text.index(next_heading, start)
    return report_text[start:end]


def test_v1_rules_and_exploratory_sections_unchanged_by_t6():
    """Same v1-region byte-identity guarantee test_decision_rules_v1_unchanged.py
    makes for decision_rules.json's content, applied to render_report's output:
    everything from '## Rules' up to (not including) the new 'Extraction
    quality' section must be byte-identical to the pre-T6 committed fixture
    (agent/mist-model-bench/integration @ 1fbf995, the merge base for this task)."""
    proc = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", "1fbf995:tests/unit/model_bench/fixtures/analyse/expected/REPORT.md"],
        capture_output=True,
        text=True,
        shell=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"cannot read pre-T6 REPORT.md via git show: {proc.stderr.strip()}")
    pre_t6 = proc.stdout

    # Regenerate from the CURRENT fixture run/ (which now has the two hand-built
    # extraction_summary.json files) -- but the v1/exploratory region must not
    # have moved a single byte relative to the pre-T6 committed report.
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    current = outputs.files["REPORT.md"]

    pre_section = _section(pre_t6, "## Rules", "## Finalist candidates")
    current_section = _section(current, "## Rules", "## Extraction quality")
    assert current_section == pre_section


def test_extraction_quality_section_renders_no_verdict():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    report = outputs.files["REPORT.md"]
    section = _section(report, "## Extraction quality", "## Finalist candidates")
    assert "report-only" in section
    # The v1/exploratory rule pattern ("verdict: **pass**" etc.) never appears
    # here -- this section computes no verdict of its own.
    assert "verdict: **" not in section


def test_extraction_quality_section_shows_arm_with_no_extraction_data():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    report = outputs.files["REPORT.md"]
    section = _section(report, "## Extraction quality", "## Finalist candidates")
    # a3 has no extraction_summary.json fixture.
    a3_idx = section.index("### a3")
    next_idx = section.index("###", a3_idx + 1) if "###" in section[a3_idx + 1 :] else len(section)
    assert "no extraction_summary.json" in section[a3_idx:next_idx]


def test_extraction_quality_section_c0_and_c1_512_present():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    report = outputs.files["REPORT.md"]
    section = _section(report, "## Extraction quality", "## Finalist candidates")
    assert "### c0" in section
    assert "### c1-512" in section
    assert "rel_precision" in section


def test_extraction_quality_section_renders_incomplete_arm_with_no_metrics_row(monkeypatch):
    """T6 reviewer finding 1b: an arm whose extraction_summary.json has
    `complete: false` (a partial match) must render as incomplete, with no
    metrics table row -- not silently scored on a partial run."""
    real_load = analyse.load_extraction_summary

    def fake_load(results_dir, arm_id):
        if arm_id == "c1-512":
            return {
                "schema": 1,
                "total_probes": 60,
                "matched_probes": 30,
                "complete": False,
                "unmatched_probe_ids": ["ext-31-x", "ext-32-x"],
            }
        return real_load(results_dir, arm_id)

    monkeypatch.setattr(analyse, "load_extraction_summary", fake_load)
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    report = outputs.files["REPORT.md"]
    section = _section(report, "## Extraction quality", "## Finalist candidates")
    c1_idx = section.index("### c1-512")
    next_idx = section.index("###", c1_idx + 1) if "###" in section[c1_idx + 1 :] else len(section)
    c1_block = section[c1_idx:next_idx]
    assert "incomplete" in c1_block
    assert "30/60" in c1_block
    assert "ext-31-x" in c1_block
    assert "| metric | value |" not in c1_block


def test_summary_json_extraction_quality_key_present():
    outputs = analyse.generate_outputs(FIXTURE_RUN)
    summary = json.loads(outputs.files["summary.json"])
    assert "extraction_quality" in summary
    assert summary["extraction_quality"]["c0"]["rel_precision"] == 0.85
    assert summary["extraction_quality"]["a3"] is None
