"""The extraction retrieval runner, end to end against fakes.

`build` is run on the real dev sets (the 60 probes and `dev-lists`, never the held-out set)
through the real `materialize_isolated`, the real `LogRegenerator` and the real Stage 3-6
components, into an isolated store under `tmp_path`. Only the graph is faked: a connection
whose fact read answers from what the recording curation pipeline was handed, so a turn that
reaches curation puts its facts in the "graph" and a turn that does not leaves them out.
`query` is run against a fake retriever. No Neo4j, no network, no model download.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.extraction_contract.models import (  # noqa: E402
    CONTRACT_VERSION,
    ExtractionPayload,
    ExtractResponse,
    ResultStamps,
    ScopeOut,
    TimingsMs,
)
from backend.knowledge.curation.deduplication import DeduplicationResult  # noqa: E402
from backend.knowledge.curation.graph_writer import WriteResult  # noqa: E402
from backend.knowledge.curation.pipeline import CurationResult  # noqa: E402
from backend.knowledge.curation.reconciliation import ReconcileTurnResult  # noqa: E402
from backend.knowledge.extraction.validator import ValidationResult  # noqa: E402
from backend.knowledge.extraction_cache import (  # noqa: E402
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_EXTRACTION_FAILED,
    ExtractionCache,
)
from backend.knowledge.models import RetrievalResult, RetrievedFact  # noqa: E402
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever  # noqa: E402
from backend.knowledge.storage.graph_store import GraphStore  # noqa: E402
from scripts.eval_harness import extraction_retrieval as runner  # noqa: E402
from scripts.eval_harness import extraction_retrieval_score as scorer  # noqa: E402
from scripts.golden_log.translate import load_gold_corpus, translate_gold_record  # noqa: E402
from tests.mocks.config import build_test_config  # noqa: E402
from tests.mocks.embeddings import FakeEmbeddingGenerator  # noqa: E402
from tests.mocks.neo4j import FakeNeo4jConnection  # noqa: E402
from tests.mocks.seeder import FakeStagingSeeder  # noqa: E402

EVAL_TARGET = "bolt://mist-neo4j-eval:7687"
GOLD_60 = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"
DEV_LISTS = _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl"
QUERIES = _REPO_ROOT / "data" / "gold" / "retrieval" / "queries-2026-11.jsonl"


@pytest.fixture(autouse=True)
def _rebuild_allowlist_admits_the_eval_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator setting H7 needs: the rebuild guard admits staging only by default."""
    monkeypatch.setenv("MIST_REBUILD_NEO4J_HOSTS", "mist-neo4j-eval:7687")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CuratedTurn:
    event_id: str
    session_id: str
    recorded_at: str
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


@dataclass(slots=True)
class RecordingCurationPipeline:
    """Records what the replay hands to curation and answers with a real `CurationResult`."""

    calls: list[CuratedTurn] = field(default_factory=list)

    async def curate_and_store(
        self,
        validation_result: ValidationResult,
        event_id: str,
        session_id: str,
        source_metadata: Any = None,
        recorded_at: str | None = None,
    ) -> CurationResult:
        self.calls.append(
            CuratedTurn(
                event_id=event_id,
                session_id=session_id,
                recorded_at=recorded_at or "",
                entities=list(validation_result.entities),
                relationships=list(validation_result.relationships),
            )
        )
        return CurationResult(
            write_result=WriteResult(),
            dedup_result=DeduplicationResult(entities=[], merge_actions=[], entities_merged=0),
            reconcile_result=ReconcileTurnResult(),
            curation_time_ms=0.0,
            validated_entities=validation_result.entities,
            validated_relationships=validation_result.relationships,
        )


def _graph_rows(pipeline: RecordingCurationPipeline, retired: set[tuple[str, str, str]]) -> list:
    """The facts query's answer: every curated relationship, current unless retired."""
    rows = []
    for call in pipeline.calls:
        types = {str(e.get("id")): str(e.get("type", "")) for e in call.entities}
        for rel in call.relationships:
            key = (str(rel["source"]), str(rel["type"]), str(rel["target"]))
            rows.append(
                {
                    "source": key[0],
                    "source_type": types.get(key[0], ""),
                    "predicate": key[1],
                    "target": key[2],
                    "target_type": types.get(key[2], ""),
                    "current": key not in retired,
                }
            )
    return rows


class FakeRetriever:
    """Answers each query text with the facts given for it, recording every call."""

    def __init__(self, answers: dict[str, list[RetrievedFact]] | None = None) -> None:
        self._answers = answers or {}
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, *, limit: int) -> RetrievalResult:
        self.calls.append((query, limit))
        facts = self._answers.get(query, [])
        return RetrievalResult(
            query=query,
            user_id="user",
            facts=facts,
            entities_found=len(facts),
            total_facts=len(facts),
            formatted_context="",
            retrieval_time_ms=0.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
        )


class FakeWiring:
    """The runner's graph wiring with every external system replaced by a fake."""

    def __init__(
        self,
        *,
        retriever: FakeRetriever | None = None,
        retired: set[tuple[str, str, str]] | None = None,
    ) -> None:
        self.config = build_test_config()
        self.pipeline = RecordingCurationPipeline()
        self.seeder = FakeStagingSeeder()
        self.retriever_obj = retriever or FakeRetriever()
        self.retired = retired or set()
        self.connected: list[str] = []
        self.schema_initialized = 0
        self.connection = FakeNeo4jConnection(query_router=self._route)

    def _route(self, query: str, params: dict | None) -> list | None:
        if query == runner.GRAPH_FACTS_QUERY:
            return _graph_rows(self.pipeline, self.retired)
        return None

    def connect(self, target: str) -> FakeNeo4jConnection:
        self.connected.append(target)
        return self.connection

    def initialize_schema(self, connection: Any) -> None:
        self.schema_initialized += 1

    def curation_pipeline(self, connection: Any) -> RecordingCurationPipeline:
        return self.pipeline

    def seed_step(self, connection: Any) -> tuple[FakeStagingSeeder, int]:
        return self.seeder, 1

    def retriever(self, connection: Any) -> FakeRetriever:
        return self.retriever_obj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dev_records() -> dict[str, dict[str, Any]]:
    records = dict(load_gold_corpus(GOLD_60))
    records.update(load_gold_corpus(DEV_LISTS))
    return records


def _response(tag: str, record: dict[str, Any]) -> dict[str, Any]:
    """A 200 body the way the service returns it, carrying the gold payload as its output."""
    entities, relationships = translate_gold_record(record)
    return ExtractResponse(
        contract_version=CONTRACT_VERSION,
        job_id=f"job-{tag}",
        outcome="extracted",
        scope=ScopeOut(label="user-scope", confidence=0.9),
        payload=ExtractionPayload(entities=entities, relationships=relationships),
        derivation=None,
        stamps=ResultStamps(
            extraction_version="test-version",
            model_hash="test-model-hash",
            prompt_sha256="test-prompt",
            llama_cpp_build="b0",
            adapter="test-adapter",
            run_identity="ri1-test",
        ),
        timings_ms=TimingsMs(scope=1.0, extract=2.0, derive=None, total=3.0),
        attempts=1,
        stage2_calls=2,
        warnings=[],
    ).model_dump(mode="json")


def _write_raw(path: Path, source: Path, *, skip: set[str] = frozenset()) -> Path:
    """An `extraction_service_raw.jsonl` for every probe of `source` except `skip`."""
    lines = []
    for tag, record in load_gold_corpus(source).items():
        if tag in skip:
            continue
        row = {
            "id": tag,
            "job_id": f"job-{tag}",
            "utterance": record["utterance"],
            "response": _response(tag, record),
        }
        lines.append(json.dumps(row) + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _relevant_fact_in_one_probe() -> tuple[tuple[str, str, str], str]:
    """A grade-2 relevant fact that exactly one dev probe asserts, and that probe's tag."""
    probes_by_fact: dict[tuple[str, str, str], set[str]] = {}
    for tag, record in _dev_records().items():
        for rel in record.get("expected_relationships", []):
            key = (
                scorer.canonical_id(rel["source"]),
                rel["predicate"].upper(),
                scorer.canonical_id(rel["target"]),
            )
            probes_by_fact.setdefault(key, set()).add(tag)
    for query in scorer.load_queries(QUERIES):
        for fact in query.relevant:
            tags = probes_by_fact.get(fact.key, set())
            if fact.grade == scorer.ANSWER_GRADE and len(tags) == 1:
                return fact.key, next(iter(tags))
    raise AssertionError("no relevant fact is asserted by exactly one dev probe")


def _build(tmp_path: Path, wiring: FakeWiring, *payloads: str, out: str = "out") -> int:
    return runner.main(
        ["build", "--payloads", *payloads, "--target", EVAL_TARGET, "--out", str(tmp_path / out)],
        wiring=wiring,
    )


def _report(tmp_path: Path, out: str = "out") -> dict[str, Any]:
    return json.loads((tmp_path / out / runner.BUILD_REPORT).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Decision 3: build materialises the dev turns and rebuilds them
# ---------------------------------------------------------------------------


class TestGoldBuild:
    def test_replays_every_dev_turn_through_the_log_regenerator_in_file_order(self, tmp_path):
        # Arrange
        wiring = FakeWiring()
        expected = [f"golden-{tag}" for tag in _dev_records()]

        # Act
        code = _build(tmp_path, wiring, "gold")

        # Assert
        assert code == 0
        assert [call.event_id for call in wiring.pipeline.calls] == expected
        assert len(expected) == 100
        assert wiring.seeder.calls == 1
        assert _report(tmp_path)["rebuild"]["turns_processed"] == 100

    def test_reads_only_the_two_dev_sets(self):
        # Assert: the held-out set is never a build input.
        assert runner.DEV_GOLD_PATHS == (GOLD_60, DEV_LISTS)

    def test_lists_no_missing_fact_when_every_relevant_fact_reached_the_graph(self, tmp_path):
        # Act
        _build(tmp_path, FakeWiring(), "gold")

        # Assert
        report = _report(tmp_path)
        assert report["payloads"] == "gold"
        assert report["relevant_facts"] > 0
        assert report["missing"] == []

    def test_lists_a_relevant_fact_whose_edge_is_no_longer_current(self, tmp_path):
        # Arrange: the edge is in the graph, but a cessation has closed it.
        fact, _ = _relevant_fact_in_one_probe()
        wiring = FakeWiring(retired={fact})

        # Act
        _build(tmp_path, wiring, "gold")

        # Assert
        missing = _report(tmp_path)["missing"]
        assert [(m["source"], m["predicate"], m["target"], m["reason"]) for m in missing] == [
            (*fact, "not_current")
        ]

    def test_turn_schedule_is_identical_across_two_builds(self, tmp_path):
        # Arrange
        first, second = FakeWiring(), FakeWiring()

        # Act
        _build(tmp_path, first, "gold", out="a")
        _build(tmp_path, second, "gold", out="b")

        # Assert
        def schedule(wiring: FakeWiring) -> list[tuple[str, str, str]]:
            return [(c.event_id, c.session_id, c.recorded_at) for c in wiring.pipeline.calls]

        assert schedule(first) == schedule(second)
        stamps = [c.recorded_at for c in first.pipeline.calls]
        assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)

    def test_wipes_the_target_and_initialises_its_schema_before_the_rebuild(self, tmp_path):
        # Arrange
        wiring = FakeWiring()

        # Act
        _build(tmp_path, wiring, "gold")

        # Assert
        assert wiring.connected == [EVAL_TARGET]
        assert wiring.connection.writes[0][0] == "MATCH (n) DETACH DELETE n"
        assert wiring.schema_initialized == 1

    def test_refuses_an_out_directory_that_already_holds_a_build(self, tmp_path):
        # Arrange
        _build(tmp_path, FakeWiring(), "gold")
        again = FakeWiring()

        # Act
        code = _build(tmp_path, again, "gold")

        # Assert
        assert code == 2
        assert again.connected == []


# ---------------------------------------------------------------------------
# Decisions 3 and 6: build from the service's own payloads
# ---------------------------------------------------------------------------


class TestServiceBuild:
    def test_lists_the_relevant_fact_a_failed_turn_never_delivered(self, tmp_path):
        # Arrange: the only probe asserting `fact` has no raw row, as a failed case has none.
        fact, tag = _relevant_fact_in_one_probe()
        raw_60 = _write_raw(tmp_path / "raw-60.jsonl", GOLD_60, skip={tag})
        raw_lists = _write_raw(tmp_path / "raw-lists.jsonl", DEV_LISTS, skip={tag})

        # Act
        code = _build(tmp_path, FakeWiring(), str(raw_60), str(raw_lists))

        # Assert
        report = _report(tmp_path)
        assert code == 0
        assert report["payloads"] == "service"
        assert report["turns_without_payload"] == [tag]
        assert (fact[0], fact[1], fact[2], "not_in_graph") in {
            (m["source"], m["predicate"], m["target"], m["reason"]) for m in report["missing"]
        }
        listed = next(
            m for m in report["missing"] if (m["source"], m["predicate"], m["target"]) == fact
        )
        assert listed["query_ids"]

    def test_writes_the_dispatcher_cache_row_for_each_response(self, tmp_path):
        # Arrange
        raw_60 = _write_raw(tmp_path / "raw-60.jsonl", GOLD_60)
        raw_lists = _write_raw(tmp_path / "raw-lists.jsonl", DEV_LISTS)
        tag = next(iter(load_gold_corpus(DEV_LISTS)))

        # Act
        _build(tmp_path, FakeWiring(), str(raw_60), str(raw_lists))

        # Assert
        epoch = _report(tmp_path)["epoch"]
        cache = ExtractionCache(str(tmp_path / "out" / "store" / "extraction-cache.db"))
        row = cache.get(f"golden-{tag}", epoch["extraction_version"], epoch["model_hash"])
        assert row["outcome"] == OUTCOME_EXTRACTED
        assert row["scope"] == "user-scope"
        assert row["service_stamps"]["stamps"]["run_identity"] == "ri1-test"
        assert row["service_stamps"]["stage2_calls"] == 2
        assert row["service_stamps"]["job_id"] == f"job-{tag}"

    def test_records_a_turn_without_a_response_as_a_failed_extraction(self, tmp_path):
        # Arrange
        tag = next(iter(load_gold_corpus(GOLD_60)))
        raw_60 = _write_raw(tmp_path / "raw-60.jsonl", GOLD_60, skip={tag})
        raw_lists = _write_raw(tmp_path / "raw-lists.jsonl", DEV_LISTS)
        wiring = FakeWiring()

        # Act
        _build(tmp_path, wiring, str(raw_60), str(raw_lists))

        # Assert
        epoch = _report(tmp_path)["epoch"]
        cache = ExtractionCache(str(tmp_path / "out" / "store" / "extraction-cache.db"))
        row = cache.get(f"golden-{tag}", epoch["extraction_version"], epoch["model_hash"])
        assert row["outcome"] == OUTCOME_SKIPPED
        assert row["skip_reason"] == SKIP_EXTRACTION_FAILED
        assert f"golden-{tag}" not in {c.event_id for c in wiring.pipeline.calls}

    def test_refuses_a_row_whose_id_is_not_a_dev_probe(self, tmp_path):
        # Arrange
        raw = _write_raw(tmp_path / "raw-60.jsonl", GOLD_60)
        foreign = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
        foreign["id"] = "not-a-dev-probe"
        with raw.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(foreign) + "\n")
        wiring = FakeWiring()

        # Act
        code = _build(tmp_path, wiring, str(raw))

        # Assert
        assert code == 2
        assert wiring.connected == []
        assert not (tmp_path / "out" / "store").exists()

    def test_refuses_a_row_whose_utterance_differs_from_the_dev_probe(self, tmp_path):
        # Arrange
        raw = _write_raw(tmp_path / "raw-60.jsonl", GOLD_60)
        rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
        rows[0]["utterance"] = rows[0]["utterance"] + " (edited)"
        raw.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        wiring = FakeWiring()

        # Act
        code = _build(tmp_path, wiring, str(raw))

        # Assert
        assert code == 2
        assert wiring.connected == []

    def test_refuses_the_same_probe_in_two_raw_files(self, tmp_path):
        # Arrange
        raw_a = _write_raw(tmp_path / "raw-a.jsonl", GOLD_60)
        raw_b = _write_raw(tmp_path / "raw-b.jsonl", GOLD_60)
        wiring = FakeWiring()

        # Act
        code = _build(tmp_path, wiring, str(raw_a), str(raw_b))

        # Assert
        assert code == 2
        assert wiring.connected == []


# ---------------------------------------------------------------------------
# Decision 4: the target must be an eval endpoint
# ---------------------------------------------------------------------------


class TestTargetRefusal:
    @pytest.mark.parametrize(
        "target",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-port"),
            pytest.param("bolt://mist-neo4j-staging:7687", id="staging-not-eval"),
            pytest.param("bolt://mist-neo4j-eval", id="no-port"),
        ],
    )
    def test_build_refuses_a_target_that_is_not_an_eval_endpoint(self, tmp_path, target):
        # Arrange
        wiring = FakeWiring()

        # Act
        code = runner.main(
            ["build", "--payloads", "gold", "--target", target, "--out", str(tmp_path / "out")],
            wiring=wiring,
        )

        # Assert
        assert code == 2
        assert wiring.connected == []
        assert not (tmp_path / "out").exists()

    def test_build_refuses_when_eval_isolation_is_off(self, tmp_path, monkeypatch):
        # Arrange
        monkeypatch.delenv("MIST_EVAL_ISOLATION", raising=False)
        wiring = FakeWiring()

        # Act
        code = _build(tmp_path, wiring, "gold")

        # Assert
        assert code == 2
        assert wiring.connected == []

    def test_build_refuses_an_eval_target_the_rebuild_guard_does_not_admit(
        self, tmp_path, monkeypatch
    ):
        # Arrange: the default rebuild allowlist names staging only.
        monkeypatch.delenv("MIST_REBUILD_NEO4J_HOSTS", raising=False)
        wiring = FakeWiring()

        # Act
        code = _build(tmp_path, wiring, "gold")

        # Assert: refused before the store is written or the target wiped.
        assert code == 2
        assert wiring.connected == []
        assert not (tmp_path / "out").exists()

    def test_query_refuses_the_live_graph(self, tmp_path):
        # Arrange
        wiring = FakeWiring()

        # Act
        code = runner.main(
            ["query", "--target", "bolt://localhost:7687", "--out", str(tmp_path / "q")],
            wiring=wiring,
        )

        # Assert
        assert code == 2
        assert wiring.connected == []
        assert wiring.retriever_obj.calls == []


# ---------------------------------------------------------------------------
# Decision 5: query writes a run file and scores it with the T10 scorer
# ---------------------------------------------------------------------------


def _fact(source: str, predicate: str, target: str) -> RetrievedFact:
    return RetrievedFact(
        subject=source,
        subject_type="",
        predicate=predicate,
        object=target,
        object_type="",
        properties={},
        similarity_score=0.5,
        graph_distance=1,
    )


def _answers_for_every_other_query() -> dict[str, list[RetrievedFact]]:
    answers: dict[str, list[RetrievedFact]] = {}
    for index, query in enumerate(scorer.load_queries(QUERIES)):
        if index % 2 == 0 and query.relevant:
            answers[query.query] = [_fact(*f.key) for f in query.relevant]
    return answers


class TestQuery:
    def test_runs_every_query_through_the_retriever_at_limit_five(self, tmp_path):
        # Arrange
        retriever = FakeRetriever()
        wiring = FakeWiring(retriever=retriever)
        queries = scorer.load_queries(QUERIES)

        # Act
        code = runner.main(
            ["query", "--target", EVAL_TARGET, "--out", str(tmp_path)], wiring=wiring
        )

        # Assert
        assert code == 0
        assert retriever.calls == [(q.query, 5) for q in queries]

    def test_writes_a_run_file_the_scorer_loads(self, tmp_path):
        # Arrange
        answers = _answers_for_every_other_query()
        wiring = FakeWiring(retriever=FakeRetriever(answers))
        queries = scorer.load_queries(QUERIES)

        # Act
        runner.main(["query", "--target", EVAL_TARGET, "--out", str(tmp_path)], wiring=wiring)

        # Assert
        run = scorer.load_run(tmp_path / runner.RUN_FILE)
        assert list(run) == [q.id for q in queries]
        for query in queries:
            expected = tuple(f.key for f in query.relevant) if query.query in answers else ()
            assert run[query.id] == expected

    def test_writes_the_summary_and_markdown_the_scorer_renders(self, tmp_path):
        # Arrange
        wiring = FakeWiring(retriever=FakeRetriever(_answers_for_every_other_query()))

        # Act
        runner.main(["query", "--target", EVAL_TARGET, "--out", str(tmp_path)], wiring=wiring)

        # Assert
        summary = scorer.score_run(
            scorer.load_queries(QUERIES), scorer.load_run(tmp_path / runner.RUN_FILE)
        )
        written = json.loads((tmp_path / runner.SUMMARY_JSON).read_text(encoding="utf-8"))
        assert written == scorer.summary_to_dict(summary)
        assert (tmp_path / runner.SUMMARY_MD).read_text(encoding="utf-8") == (
            scorer.render_markdown(summary)
        )
        assert 0.0 < written["answerable"]["hit_at_5"] < 1.0

    def test_scores_through_the_scorer_functions_not_a_copy(self):
        # Assert
        assert runner.score_run is scorer.score_run
        assert runner.summary_to_dict is scorer.summary_to_dict
        assert runner.render_markdown is scorer.render_markdown

    @pytest.mark.asyncio
    async def test_the_default_retriever_is_todays_knowledge_retriever(self):
        # Arrange: a graph store over a fake connection, so no graph and no model is reached.
        graph_store = GraphStore(FakeNeo4jConnection(), FakeEmbeddingGenerator())
        retriever = runner.build_default_retriever(build_test_config(), graph_store)

        # Act
        rows = await runner.run_queries(retriever, scorer.load_queries(QUERIES)[:1])

        # Assert
        assert isinstance(retriever, KnowledgeRetriever)
        assert rows == [{"id": scorer.load_queries(QUERIES)[0].id, "retrieved": []}]
