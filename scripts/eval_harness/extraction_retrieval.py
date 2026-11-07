"""Build a graph from the dev turns, then run the retrieval queries against it.

The retrieval test asks whether what extraction stored comes back when the retriever is
asked for it. `extraction_retrieval_score` scores a run file; this module produces one, in two
steps, so the same graph can be queried by any retriever:

    python -m scripts.eval_harness.extraction_retrieval build
        --payloads gold | <extraction_service_raw.jsonl> [<raw.jsonl> ...]
        --target <eval bolt uri> --out <dir> [--queries <queries.jsonl>]
    python -m scripts.eval_harness.extraction_retrieval query
        --target <eval bolt uri> --out <dir> [--queries <queries.jsonl>]

`build` materialises the dev turns, the 60-probe set and `dev-lists` (`DEV_GOLD_PATHS`, never
the held-out set), into a fresh event store and extraction cache under `<out>/store`, through
the golden log's own `materialize_isolated`. Each turn's payload is either the gold record,
translated by `translate_gold_record`, or the extraction service's own response from a
gauntlet's raw rows, recorded with the backlog store's `put_extracted` and the same fields the
dispatcher stores. A dev turn with no raw row (the gauntlet writes none for a failed case) is
recorded as a failed extraction, as a dead-lettered turn is in production. The turns are then
replayed into the target by `LogRegenerator.rebuild`, the existing rebuild path, after the
target is wiped. `<out>/build_report.json` lists every relevant fact of the query file that is
not a current edge of the built graph, so a miss the build caused is not read as a retrieval
miss.

`query` runs each query through a `Retriever` (by default today's `KnowledgeRetriever`, at
`limit=5`), writes `<out>/retrieval_run.jsonl` in the scorer's run-file schema, and writes
`retrieval_summary.json` and `retrieval.md` with the scorer's own functions.

The target is refused unless it is a disposable eval endpoint: never live
(`assert_neo4j_uri_not_live`), eval isolation on (`MIST_EVAL_ISOLATION=1`) and the endpoint on
the eval allowlist (`assert_neo4j_isolated`). `build` also needs the rebuild guard to admit it
(`assert_rebuild_target_not_live`, which `LogRegenerator.rebuild` calls itself); that guard's
default allowlist names the staging instance only, so a build on the eval instance sets
`MIST_REBUILD_NEO4J_HOSTS` to the eval endpoint. Every check runs before any file or graph
write.

Exit 0, or 2 when a target is refused or an input is missing or malformed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.errors import MistError  # noqa: E402
from backend.extraction_backlog.store import BacklogStore, Epoch  # noqa: E402
from backend.extraction_contract.models import ExtractResponse  # noqa: E402
from backend.interfaces import GraphConnection  # noqa: E402
from backend.knowledge.config import KnowledgeConfig, Neo4jConfig  # noqa: E402
from backend.knowledge.eval_isolation import (  # noqa: E402
    EvalIsolationError,
    assert_neo4j_isolated,
    assert_neo4j_uri_not_live,
    assert_rebuild_target_not_live,
    is_eval_isolation_active,
)
from backend.knowledge.extraction.confidence import ConfidenceScorer  # noqa: E402
from backend.knowledge.extraction.normalizer import EntityNormalizer  # noqa: E402
from backend.knowledge.extraction.temporal import TemporalResolver  # noqa: E402
from backend.knowledge.extraction.validator import ExtractionValidator  # noqa: E402
from backend.knowledge.extraction_cache import SKIP_EXTRACTION_FAILED  # noqa: E402
from backend.knowledge.models import RetrievalResult, RetrievedFact  # noqa: E402
from backend.knowledge.regeneration.log_regenerator import (  # noqa: E402
    LogRegenerator,
    RebuildReport,
)
from backend.knowledge.regeneration.rebuild_journal import EventStoreRebuildJournal  # noqa: E402
from scripts.eval_harness.extraction_retrieval_score import (  # noqa: E402
    FactKey,
    K,
    RetrievalQuery,
    load_queries,
    load_run,
    normalise_fact,
    render_markdown,
    score_run,
    summary_to_dict,
)
from scripts.golden_log.generate import (  # noqa: E402
    SESSION_ORIGIN,
    GoldenTurn,
    MaterializedGoldenLog,
    event_id_for,
    materialize_isolated,
    session_id_for,
)
from scripts.golden_log.translate import (  # noqa: E402
    GOLD_UTTERANCE_FIELD,
    load_gold_corpus,
    translate_gold_record,
)

# The dev turns, in replay order. The held-out set is deliberately absent: the retrieval
# test must never be a way to tune against it.
DEV_GOLD_PATHS: tuple[Path, ...] = (
    _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl",
    _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl",
)
QUERIES_PATH = _REPO_ROOT / "data" / "gold" / "retrieval" / "queries-2026-11.jsonl"

GOLD_PAYLOADS = "gold"

# The turn schedule: one turn an hour from a fixed instant, in `DEV_GOLD_PATHS` file order.
# Fixed rather than read from the clock so two builds replay the same log; in the past so
# a fact stamped from its turn's time is current when the graph is read.
SCHEDULE_START = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
TURN_SPACING = timedelta(hours=1)

# The rebuild guard's second arm compares the target with the live graph's URI. The service
# spelling is given here; the guard's hardcoded denylist refuses every other spelling.
LIVE_NEO4J_URI = "bolt://mist-neo4j:7687"

STORE_DIR = "store"
BUILD_REPORT = "build_report.json"
RUN_FILE = "retrieval_run.jsonl"
SUMMARY_JSON = "retrieval_summary.json"
SUMMARY_MD = "retrieval.md"

WIPE_QUERY = "MATCH (n) DETACH DELETE n"

# Every entity-to-entity edge of the built graph, with the currency test the retriever's
# reads apply (`GraphStore.get_entity_neighborhood`, `get_user_relationships_to_entities`):
# not orphaned, the latest belief, and valid now. An edge a later cessation closed is in the
# graph but not current, so the retriever cannot return it.
GRAPH_FACTS_QUERY = """
MATCH (s:__Entity__)-[r]->(t:__Entity__)
RETURN s.id AS source, s.entity_type AS source_type, type(r) AS predicate,
       t.id AS target, t.entity_type AS target_type,
       ((r.status IS NULL OR r.status <> 'orphaned')
        AND coalesce(r.is_latest_belief, true)
        AND (r.valid_to IS NULL OR r.valid_to > $now)
        AND (r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now)) AS current
"""

REASON_NOT_IN_GRAPH = "not_in_graph"
REASON_NOT_CURRENT = "not_current"


class RetrievalRunnerError(MistError):
    """Raised when the runner's inputs cannot be built into an eval graph."""


class Retriever(Protocol):
    """Anything that answers a query with ranked facts; `KnowledgeRetriever` is one."""

    async def retrieve(self, query: str, *, limit: int) -> RetrievalResult:
        """The facts for `query`, best first, at most `limit` of them."""
        ...


class GraphWiring(Protocol):
    """The external systems a build or a query needs, injected so tests can fake them."""

    config: KnowledgeConfig

    def connect(self, target: str) -> GraphConnection:
        """An open connection to `target`, which the caller has already checked."""
        ...

    def initialize_schema(self, connection: GraphConnection) -> None:
        """Create the constraints and indexes the writers and the retriever expect."""
        ...

    def curation_pipeline(self, connection: GraphConnection) -> Any:
        """The curation pipeline the rebuild writes each replayed turn through."""
        ...

    def seed_step(self, connection: GraphConnection) -> tuple[Any, int]:
        """The rebuild's seeder and the node floor its apply must clear."""
        ...

    def retriever(self, connection: GraphConnection) -> Retriever:
        """The retriever the queries run through."""
        ...


def build_default_retriever(config: KnowledgeConfig, graph_store: Any) -> Retriever:
    """Today's retriever over `graph_store`, with no vector store and no vault sidecar.

    The graph is the only thing the dev turns built, so the other two legs would add
    nothing the build put there.
    """
    from backend.factories import build_knowledge_retriever

    return build_knowledge_retriever(config, graph_store=graph_store)


class Neo4jEvalWiring:
    """The real wiring, as `mist_admin`'s `_build_log_regenerator` builds it for a rebuild.

    A composition root for this one command: one embedding model, loaded on first use and
    shared by the schema, the curation pipeline, the seed and the retriever, as the rebuild
    command shares it.
    """

    def __init__(self, config: KnowledgeConfig) -> None:
        self.config = config
        self._embedding_provider: Any = None

    def _embeddings(self) -> Any:
        if self._embedding_provider is None:
            from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator

            self._embedding_provider = EmbeddingGenerator(
                self.config.embedding.model_name, device=self.config.embedding.resolved_device()
            )
        return self._embedding_provider

    def connect(self, target: str) -> GraphConnection:
        """Connect to `target` with the configured credentials.

        `Neo4jConnection.connect` runs the eval guard again on the target's own config.
        """
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        connection = Neo4jConnection(
            Neo4jConfig(
                uri=target,
                username=self.config.neo4j.username,
                password=self.config.neo4j.password,
            )
        )
        connection.connect()
        return connection

    def initialize_schema(self, connection: GraphConnection) -> None:
        """Constraints and the vector index the retriever's entity search reads."""
        from backend.knowledge.storage.graph_store import GraphStore

        GraphStore(
            connection=connection, embedding_generator=self._embeddings()
        ).initialize_schema()

    def curation_pipeline(self, connection: GraphConnection) -> Any:
        """The production curation pipeline, writing to `connection` only."""
        from backend.factories import build_curation_pipeline
        from backend.knowledge.storage.graph_executor import GraphExecutor

        return build_curation_pipeline(
            self.config, GraphExecutor(connection), embedding_provider=self._embeddings()
        )

    def seed_step(self, connection: GraphConnection) -> tuple[Any, int]:
        """The seeder for the configured seed corpus, and the node floor it must clear.

        The floor is the number of distinct node ids the corpus's facts reference, the ids
        the applier writes, so a seed that wrote fewer stops the rebuild before any turn.
        """
        from backend.knowledge.regeneration.staging_seeder import StagingSeeder
        from backend.knowledge.seed.loader import load_seed_documents

        documents = load_seed_documents(Path(self.config.vault.root) / "seed")
        seeder = StagingSeeder(
            connection=connection,
            documents=documents,
            seed_version=documents[0].seed_version,
            embedding_generator=self._embeddings(),
            expected_dimension=self.config.embedding.dimension,
        )
        node_ids = {
            node_id
            for document in documents
            for fact in document.facts
            for node_id in (fact.subject, fact.object)
        }
        return seeder, max(1, len(node_ids))

    def retriever(self, connection: GraphConnection) -> Retriever:
        """`build_default_retriever` over a graph store on `connection`."""
        from backend.knowledge.storage.graph_store import GraphStore

        return build_default_retriever(
            self.config, GraphStore(connection=connection, embedding_generator=self._embeddings())
        )


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------


def assert_eval_target(target: str, *, rebuild: bool) -> None:
    """Refuse a target that is not a disposable eval endpoint.

    Raises:
        EvalIsolationError: When the target names the live graph, has no port, eval
            isolation is off, or the endpoint is not on the eval allowlist.
        RebuildTargetError: With `rebuild`, when the rebuild guard does not admit it.
    """
    assert_neo4j_uri_not_live(target, action="the extraction retrieval runner")
    if not is_eval_isolation_active():
        raise EvalIsolationError(
            f"eval isolation is off, so {target!r} cannot be checked against the eval "
            "allowlist. Set MIST_EVAL_ISOLATION=1 for this run."
        )
    assert_neo4j_isolated(Neo4jConfig(uri=target))
    if rebuild:
        assert_rebuild_target_not_live(target, LIVE_NEO4J_URI)


# ---------------------------------------------------------------------------
# The dev turns and their payloads
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServiceResult:
    """One dev turn's 200 response from a gauntlet's raw rows."""

    tag: str
    job_id: str
    response: ExtractResponse


def load_dev_records(paths: Sequence[Path] = DEV_GOLD_PATHS) -> dict[str, dict[str, Any]]:
    """Every dev gold record by tag, in file order across `paths`.

    Raises:
        RetrievalRunnerError: When a tag appears in two files, which would make the derived
            event id ambiguous.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        for tag, record in load_gold_corpus(path).items():
            if tag in records:
                raise RetrievalRunnerError(f"{path}: tag {tag!r} is already in an earlier dev set")
            records[tag] = record
    return records


def load_service_results(
    paths: Sequence[Path], records: dict[str, dict[str, Any]]
) -> dict[str, ServiceResult]:
    """Parse the raw rows of one or more gauntlet runs over the dev sets.

    A row's `id` is the probe tag and its `response` the service's 200 body
    (`scripts/model_bench/probes/extraction_service.py`, `raw_row`).

    Raises:
        RetrievalRunnerError: When a row names a tag outside the dev sets, carries an
            utterance other than the dev probe's, or repeats a tag. Foreign tags are counted,
            never named: a held-out run's tags name held-out probes.
    """
    results: dict[str, ServiceResult] = {}
    foreign = 0
    mismatched = 0
    for path in paths:
        for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            tag = str(row["id"])
            if tag not in records:
                foreign += 1
                continue
            if tag in results:
                raise RetrievalRunnerError(f"{path}:{line_no}: probe {tag!r} has a second row")
            if row.get("utterance") != records[tag][GOLD_UTTERANCE_FIELD]:
                mismatched += 1
                continue
            response = ExtractResponse.model_validate(row["response"])
            results[tag] = ServiceResult(
                tag=tag, job_id=str(row.get("job_id") or response.job_id), response=response
            )
    if foreign:
        raise RetrievalRunnerError(
            f"{foreign} raw row(s) name no probe of the dev sets; the build takes dev runs "
            "only (the 60 probes and dev-lists)"
        )
    if mismatched:
        raise RetrievalRunnerError(
            f"{mismatched} raw row(s) carry an utterance other than their dev probe's; the "
            "run was made on a different gold file"
        )
    return results


def schedule_turns(
    records: dict[str, dict[str, Any]],
    payloads: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> list[GoldenTurn]:
    """The dev turns in record order, one `TURN_SPACING` apart from `SCHEDULE_START`.

    Event and session ids are derived the golden log's way (`event_id_for`,
    `session_id_for`). A tag with no payload gets an empty one.
    """
    turns: list[GoldenTurn] = []
    turn_index_by_session: dict[str, int] = {}
    for position, (tag, record) in enumerate(records.items()):
        timestamp = (SCHEDULE_START + position * TURN_SPACING).isoformat()
        session_id = session_id_for(timestamp)
        turn_index = turn_index_by_session.get(session_id, 0)
        turn_index_by_session[session_id] = turn_index + 1
        entities, relationships = payloads.get(tag, ([], []))
        turns.append(
            GoldenTurn(
                event_id=event_id_for(tag),
                session_id=session_id,
                timestamp=timestamp,
                turn_index=turn_index,
                utterance=record[GOLD_UTTERANCE_FIELD],
                entities=list(entities),
                relationships=list(relationships),
            )
        )
    return turns


def _service_stamps(result: ServiceResult) -> dict[str, Any]:
    """The dispatcher's `service_stamps` for a response (`dispatcher.py`, `put_extracted`).

    `request_id` and `location_label` come from the dispatcher's own request and the
    service's `/v1/info`; a raw row carries neither, so both are None.
    """
    response = result.response
    return {
        "stamps": response.stamps.model_dump(mode="json"),
        "timings_ms": response.timings_ms.model_dump(mode="json"),
        "attempts": response.attempts,
        "stage2_calls": response.stage2_calls,
        "warnings": list(response.warnings),
        "contract_version": response.contract_version,
        "job_id": result.job_id,
        "request_id": None,
        "location_label": None,
    }


def record_service_results(
    materialized: MaterializedGoldenLog,
    tags: Sequence[str],
    turns: Sequence[GoldenTurn],
    results: dict[str, ServiceResult],
) -> list[str]:
    """Record each turn's service outcome in the cache the way the dispatcher does.

    A response is stored with `BacklogStore.put_extracted` and the fields the dispatcher
    passes; a turn without one with `put_skip(SKIP_EXTRACTION_FAILED)`, the dead-letter
    record. Both replace the row `materialize_isolated` wrote under the same key.

    Returns:
        The tags with no response, in turn order.
    """
    store = BacklogStore(materialized.event_store, materialized.extraction_cache)
    epoch = Epoch.from_row(materialized.epoch)
    without: list[str] = []
    for tag, turn in zip(tags, turns, strict=True):
        result = results.get(tag)
        if result is None:
            store.put_skip(
                turn.event_id, epoch, skip_reason=SKIP_EXTRACTION_FAILED, created_at=turn.timestamp
            )
            without.append(tag)
            continue
        response = result.response
        store.put_extracted(
            turn.event_id,
            epoch,
            created_at=turn.timestamp,
            entities=response.payload.entities,
            relationships=response.payload.relationships,
            scope=response.scope.label,
            scope_confidence=response.scope.confidence,
            derivation=(
                None
                if response.derivation is None
                else {"operations": response.derivation.operations}
            ),
            service_stamps=_service_stamps(result),
        )
    return without


# ---------------------------------------------------------------------------
# The built graph against the query file
# ---------------------------------------------------------------------------


def read_graph_facts(connection: GraphConnection, *, now: datetime) -> dict[FactKey, bool]:
    """Every edge of the graph by its scorer key, True when any copy of it is current."""
    facts: dict[FactKey, bool] = {}
    for row in connection.execute_query(GRAPH_FACTS_QUERY, {"now": now.isoformat()}):
        key = normalise_fact(dict(row))
        facts[key] = facts.get(key, False) or bool(row["current"])
    return facts


def relevant_facts(queries: Sequence[RetrievalQuery]) -> dict[FactKey, list[str]]:
    """Each distinct relevant fact of the query set, with the ids of the queries naming it."""
    by_fact: dict[FactKey, list[str]] = {}
    for query in queries:
        for fact in query.relevant:
            ids = by_fact.setdefault(fact.key, [])
            if query.id not in ids:
                ids.append(query.id)
    return by_fact


def missing_relevant_facts(
    queries: Sequence[RetrievalQuery], graph: dict[FactKey, bool]
) -> list[dict[str, Any]]:
    """The relevant facts the retriever could not return from `graph`, in query-file order."""
    missing = []
    for key, query_ids in relevant_facts(queries).items():
        current = graph.get(key)
        if current:
            continue
        missing.append(
            {
                "source": key[0],
                "predicate": key[1],
                "target": key[2],
                "reason": REASON_NOT_IN_GRAPH if current is None else REASON_NOT_CURRENT,
                "query_ids": query_ids,
            }
        )
    return missing


def _stage_components(config: KnowledgeConfig) -> dict[str, Any]:
    """Stages 3-6 as the rebuild command wires them (`mist_admin._build_log_regenerator`)."""
    return {
        "confidence_scorer": ConfidenceScorer(),
        "temporal_resolver": TemporalResolver(),
        "normalizer": EntityNormalizer(embedding_generator=None, executor=None),
        "validator": ExtractionValidator(
            min_confidence=config.extraction.min_extraction_confidence
        ),
    }


def _rebuild_dict(report: RebuildReport) -> dict[str, Any]:
    return {
        "job_id": report.job_id,
        "turns_processed": report.turns_processed,
        "turns_failed": report.turns_failed,
        "total_logged": report.total_logged,
        "seed_nodes_written": report.seed_nodes_written,
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def build(
    *,
    payloads: Sequence[str],
    target: str,
    out: Path,
    wiring: GraphWiring,
    queries_path: Path = QUERIES_PATH,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Build the eval graph from the dev turns and write `build_report.json`.

    Args:
        payloads: `["gold"]`, or the paths of one or more raw-row files.
        target: The eval Neo4j to wipe and rebuild into.
        out: Output directory; `<out>/store` must not exist yet.
        wiring: The graph-side dependencies.
        queries_path: The query file whose relevant facts the report checks.
        clock: The instant the graph's currency is read at.

    Returns:
        The build report.

    Raises:
        EvalIsolationError, RebuildTargetError: When the target is refused.
        RetrievalRunnerError: When an input is unusable or `<out>/store` exists.
    """
    assert_eval_target(target, rebuild=True)
    records = load_dev_records()
    queries = load_queries(queries_path)

    results: dict[str, ServiceResult] = {}
    if list(payloads) == [GOLD_PAYLOADS]:
        source = "gold"
        per_tag = {tag: translate_gold_record(record) for tag, record in records.items()}
    elif GOLD_PAYLOADS in payloads:
        raise RetrievalRunnerError("--payloads takes 'gold' alone, or raw-row files only")
    else:
        source = "service"
        results = load_service_results([Path(p) for p in payloads], records)
        per_tag = {
            tag: (r.response.payload.entities, r.response.payload.relationships)
            for tag, r in results.items()
        }

    root = out / STORE_DIR
    if root.exists():
        raise RetrievalRunnerError(f"{root} already exists; give each build a fresh --out")

    tags = list(records)
    turns = schedule_turns(records, per_tag)
    materialized = materialize_isolated(turns, root=root)
    without_payload: list[str] = []
    if source == "service":
        without_payload = record_service_results(materialized, tags, turns, results)

    connection = wiring.connect(target)
    try:
        connection.execute_write(WIPE_QUERY, {})
        wiring.initialize_schema(connection)
        seeder, min_seed_nodes = wiring.seed_step(connection)
        regenerator = LogRegenerator(
            event_store=materialized.event_store,
            extraction_cache=materialized.extraction_cache,
            staging_curation_pipeline=wiring.curation_pipeline(connection),
            journal=EventStoreRebuildJournal(materialized.event_store),
            staging_seeder=seeder,
            **_stage_components(wiring.config),
        )
        report = asyncio.run(
            regenerator.rebuild(
                staging_uri=target,
                live_uri=LIVE_NEO4J_URI,
                epoch=materialized.epoch,
                min_seed_nodes=min_seed_nodes,
                origins=(SESSION_ORIGIN,),
            )
        )
        checked_at = clock()
        graph = read_graph_facts(connection, now=checked_at)
    finally:
        connection.disconnect()

    epoch = materialized.epoch
    build_report = {
        "payloads": source,
        "payload_files": [str(p) for p in payloads] if source == "service" else [],
        "run_identities": sorted(
            {r.response.stamps.run_identity or "unrecorded" for r in results.values()}
        ),
        "target": target,
        "dev_sets": [_repo_relative(p) for p in DEV_GOLD_PATHS],
        "queries": _repo_relative(queries_path),
        "turns": len(turns),
        "turns_without_payload": without_payload,
        "epoch": {
            "ontology_version": epoch["ontology_version"],
            "extraction_version": epoch["extraction_version"],
            "model_hash": epoch["model_hash"],
            "activated_at": epoch["activated_at"],
        },
        "rebuild": _rebuild_dict(report),
        "checked_at": checked_at.isoformat(),
        "relevant_facts": len(relevant_facts(queries)),
        "missing": missing_relevant_facts(queries, graph),
    }
    (out / BUILD_REPORT).write_text(json.dumps(build_report, indent=2) + "\n", encoding="utf-8")
    return build_report


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------


def fact_row(fact: RetrievedFact) -> dict[str, Any]:
    """One retrieved fact in the run-file schema; the extra keys are ignored by the scorer."""
    return {
        "source": fact.subject,
        "predicate": fact.predicate,
        "target": fact.object,
        "source_type": fact.subject_type,
        "target_type": fact.object_type,
        "similarity_score": fact.similarity_score,
        "graph_distance": fact.graph_distance,
    }


async def run_queries(
    retriever: Retriever, queries: Sequence[RetrievalQuery]
) -> list[dict[str, Any]]:
    """Run-file rows: each query's facts at `limit=K`, in the order the retriever ranked them."""
    rows = []
    for query in queries:
        result = await retriever.retrieve(query.query, limit=K)
        rows.append({"id": query.id, "retrieved": [fact_row(f) for f in result.facts]})
    return rows


def query(
    *,
    target: str,
    out: Path,
    wiring: GraphWiring,
    queries_path: Path = QUERIES_PATH,
) -> dict[str, Any]:
    """Query the built graph, write the run file, and score it with the T10 scorer.

    Returns:
        The summary dict written to `retrieval_summary.json`.

    Raises:
        EvalIsolationError: When the target is refused.
    """
    assert_eval_target(target, rebuild=False)
    queries = load_queries(queries_path)
    connection = wiring.connect(target)
    try:
        rows = asyncio.run(run_queries(wiring.retriever(connection), queries))
    finally:
        connection.disconnect()

    out.mkdir(parents=True, exist_ok=True)
    run_path = out / RUN_FILE
    run_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = score_run(queries, load_run(run_path))
    summary_dict = summary_to_dict(summary)
    (out / SUMMARY_JSON).write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
    (out / SUMMARY_MD).write_text(render_markdown(summary), encoding="utf-8")
    return summary_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the retrieval eval graph, or query it.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Materialise the dev turns and rebuild them into --target.")
    b.add_argument(
        "--payloads",
        required=True,
        nargs="+",
        help="'gold', or one or more extraction_service_raw.jsonl files over the dev sets.",
    )
    b.add_argument("--target", required=True, help="Eval Neo4j bolt URI (wiped first).")
    b.add_argument("--out", required=True, type=Path, help="Output directory.")
    b.add_argument("--queries", default=QUERIES_PATH, type=Path, help="Retrieval query set.")

    q = sub.add_parser("query", help="Run the retrieval queries against --target and score.")
    q.add_argument("--target", required=True, help="Eval Neo4j bolt URI.")
    q.add_argument("--out", required=True, type=Path, help="Output directory.")
    q.add_argument("--queries", default=QUERIES_PATH, type=Path, help="Retrieval query set.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, *, wiring: GraphWiring | None = None) -> int:
    args = parse_args(argv)
    try:
        active = wiring if wiring is not None else Neo4jEvalWiring(KnowledgeConfig.from_env())
        if args.command == "build":
            report = build(
                payloads=args.payloads,
                target=args.target,
                out=args.out,
                wiring=active,
                queries_path=args.queries,
            )
            print(
                f"[extraction-retrieval] built {report['turns']} turns "
                f"({report['payloads']} payloads, {len(report['turns_without_payload'])} "
                f"without one); {len(report['missing'])} of {report['relevant_facts']} "
                f"relevant facts missing; report: {args.out / BUILD_REPORT}"
            )
        else:
            summary = query(
                target=args.target, out=args.out, wiring=active, queries_path=args.queries
            )
            print(
                f"[extraction-retrieval] ran {summary['ran']} of {summary['queries']} queries; "
                f"hit at 5: {summary['answerable']['hit_at_5']}; "
                f"summary: {args.out / SUMMARY_JSON}"
            )
    except (EvalIsolationError, MistError, OSError, ValueError, KeyError) as exc:
        print(f"[extraction-retrieval] error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
