"""`mist_admin memory-index-rebuild --facts`: drop the fact tables and refill them from the graph.

The graph is a FakeGraph whose connection answers `current_facts_query` pages the way Neo4j
would, and each index is a real `memory_index.db` on a tmp file with FakeEmbedder. The central
test builds one index incrementally, refresh by refresh while the graph changes, and rebuilds
another from a stale state, then compares every stored row, mirror and vector.
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import pytest
import sqlite_vec

# scripts/ is not a package; insert it so mist_admin is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion

from backend.errors import Neo4jQueryError  # noqa: E402
from backend.knowledge.config import (  # noqa: E402
    EmbeddingConfig,
    EventStoreConfig,
    ExtractionConfig,
    KnowledgeConfig,
    LLMConfig,
    Neo4jConfig,
)
from backend.memory.config import MemoryConfig  # noqa: E402
from backend.memory.fact_index import FactIndex  # noqa: E402
from backend.memory.fact_source import FactIndexRefresher, GraphFactSource  # noqa: E402
from backend.memory.index_db import (  # noqa: E402
    FACTS_DDL,
    META_INDEX_STATE,
    degraded_reason,
    open_index,
    open_reader,
)
from backend.memory.models import FactRow, fact_key  # noqa: E402
from backend.memory.retrieval_view import (  # noqa: E402
    DEFAULT_FACT_VIEW,
    FACT_VIEWS,
    ViewSpec,
    render_fact_view,
)
from tests.fakes.memory_fakes import FakeEmbedder  # noqa: E402

_DEFAULT_VIEW = FACT_VIEWS[DEFAULT_FACT_VIEW]
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection  # noqa: E402

_MODEL = "all-MiniLM-L6-v2"


# -- the fake graph -----------------------------------------------------------------------------


def _record(page_key: str, object_id: str, **overrides: object) -> dict[str, object]:
    """One row as `current_facts_query` returns it: the user USES `object_id`."""
    record: dict[str, object] = {
        "page_key": page_key,
        "subject_id": "user",
        "subject_name": "User",
        "subject_type": "User",
        "rel_type": "USES",
        "object_id": object_id,
        "object_name": object_id.title(),
        "object_type": "Technology",
        "version_key": f"{object_id}|open|open",
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_to": None,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "confidence": 0.9,
        "source_utterance_id": "utt-1",
    }
    record.update(overrides)
    return record


class FakeGraph:
    """Answers `current_facts_query` pages from a list of records, as Neo4j would."""

    def __init__(self, records: Sequence[dict[str, object]] = ()) -> None:
        self.records = list(records)
        self.error: BaseException | None = None
        self.connection = FakeNeo4jConnection(query_router=self._route)
        self.executor = FakeGraphExecutor(self.connection)

    def _route(self, query: str, params: dict[str, object] | None) -> list[dict[str, object]]:
        assert params is not None
        if self.error is not None:
            raise self.error
        after = params["after_key"]
        rows = sorted(
            (r for r in self.records if after is None or r["page_key"] > after),
            key=lambda r: r["page_key"],
        )
        return rows[: params["limit"]]


# The graph's history: three facts appear, two more arrive and one changes, then one goes.
_STEP_1 = [_record("k01", "python"), _record("k02", "neo4j"), _record("k03", "sqlite")]
_STEP_2 = [
    _record("k01", "python", confidence=0.95),
    _record("k02", "neo4j"),
    _record("k03", "sqlite"),
    _record("k04", "docker", valid_from="2026-03-01T00:00:00+00:00"),
    _record("k05", "rust", valid_to="2026-06-01T00:00:00+00:00"),
]
_FINAL = [record for record in _STEP_2 if record["object_id"] != "neo4j"]


def _stray_row() -> FactRow:
    """A row the graph never had, standing in for whatever a damaged index holds."""
    key = fact_key("user", "USES", "cobol")
    return FactRow(
        row_key=f"{key}#stray",
        fact_key=key,
        subject_id="user",
        subject_name="User",
        subject_type="User",
        rel_type="USES",
        object_id="cobol",
        object_name="Cobol",
        object_type="Technology",
        valid_from=None,
        valid_to=None,
        recorded_at=None,
        confidence=0.5,
        source_utterance_id=None,
    )


# -- index helpers ------------------------------------------------------------------------------


def _refresher(index: FactIndex, graph: FakeGraph, pool: ThreadPoolExecutor) -> FactIndexRefresher:
    return FactIndexRefresher(
        GraphFactSource(graph.executor),
        index,
        pool,
        clock=time,
        debounce_s=0.0,
        rescan_interval_s=300.0,
        periodic=False,
        rebuild=None,
    )


def _build_incrementally(path: Path, steps: Sequence[Sequence[dict[str, object]]]) -> None:
    """Refresh one index after each change to the graph, as the live refresher does."""
    writer = open_index(path, embedding_model_id=_MODEL)
    reader = open_reader(path, embedding_model_id=_MODEL)
    index = FactIndex(writer, FakeEmbedder(), tz=UTC, reader=reader)
    graph = FakeGraph()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            refresher = _refresher(index, graph, pool)
            for step in steps:
                graph.records = list(step)
                asyncio.run(refresher.refresh())
    finally:
        reader.close()
        writer.close()


def _seed_stale(path: Path) -> None:
    """An index refreshed from an old graph that also holds a row no graph ever had."""
    _build_incrementally(path, [_STEP_1])
    writer = open_index(path, embedding_model_id=_MODEL)
    reader = open_reader(path, embedding_model_id=_MODEL)
    try:
        FactIndex(writer, FakeEmbedder(), tz=UTC, reader=reader).upsert([_stray_row()])
    finally:
        reader.close()
        writer.close()


def _dump(path: Path) -> dict[str, list[tuple]]:
    """Every stored row, keyed by `row_key` rather than by rowid, which differs between builds."""
    conn = open_reader(path, embedding_model_id=_MODEL)
    try:
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall() if row[1] != "id"
        ]
        return {
            "facts": conn.execute(
                f"SELECT {', '.join(columns)} FROM facts ORDER BY row_key"
            ).fetchall(),
            "facts_fts": conn.execute(
                "SELECT f.row_key, t.text, t.subject_name, t.object_name, t.type_terms "
                "FROM facts_fts t JOIN facts f ON f.id = t.rowid ORDER BY f.row_key"
            ).fetchall(),
            "facts_vec": conn.execute(
                "SELECT f.row_key, v.embedding FROM facts_vec v "
                "JOIN facts f ON f.id = v.rowid ORDER BY f.row_key"
            ).fetchall(),
            "facets": conn.execute(
                "SELECT facet_key, subject_id, subject_name, subject_type, rel_type, object_type, "
                "view_text, content_hash FROM facets ORDER BY facet_key"
            ).fetchall(),
            "facets_fts": conn.execute(
                "SELECT f.facet_key, t.view_text, t.subject_name, t.type_terms "
                "FROM facets_fts t JOIN facets f ON f.id = t.rowid ORDER BY f.facet_key"
            ).fetchall(),
            "facets_vec": conn.execute(
                "SELECT f.facet_key, v.embedding FROM facets_vec v "
                "JOIN facets f ON f.id = v.rowid ORDER BY f.facet_key"
            ).fetchall(),
            "orphans": conn.execute(
                "SELECT 'fts', rowid FROM facts_fts WHERE rowid NOT IN (SELECT id FROM facts) "
                "UNION ALL "
                "SELECT 'vec', rowid FROM facts_vec WHERE rowid NOT IN (SELECT id FROM facts) "
                "UNION ALL "
                "SELECT 'facet fts', rowid FROM facets_fts "
                "WHERE rowid NOT IN (SELECT id FROM facets) "
                "UNION ALL "
                "SELECT 'facet vec', rowid FROM facets_vec "
                "WHERE rowid NOT IN (SELECT id FROM facets)"
            ).fetchall(),
            "index_meta": conn.execute("SELECT key, value FROM index_meta ORDER BY key").fetchall(),
        }
    finally:
        conn.close()


def _view_texts(path: Path) -> list[str]:
    """Every stored row's embedded view text, sorted."""
    conn = open_reader(path, embedding_model_id=_MODEL)
    try:
        return sorted(text for (text,) in conn.execute("SELECT view_text FROM facts"))
    finally:
        conn.close()


def _expected_view_texts(view: ViewSpec) -> list[str]:
    """The view text of each `_FINAL` record under `view`, sorted."""
    fields = (
        "subject_id",
        "subject_name",
        "subject_type",
        "rel_type",
        "object_id",
        "object_name",
        "object_type",
        "valid_from",
        "valid_to",
        "recorded_at",
        "confidence",
        "source_utterance_id",
    )
    rows = [
        FactRow(row_key="-", fact_key="-", **{name: record[name] for name in fields})
        for record in _FINAL
    ]
    return sorted(render_fact_view(row, view, UTC) for row in rows)


def _config(
    tmp_path: Path, *, index_path: str | None = None, fact_view: str = "F3"
) -> KnowledgeConfig:
    return KnowledgeConfig(
        neo4j=Neo4jConfig(),
        llm=LLMConfig(),
        embedding=EmbeddingConfig(model_name=_MODEL, device="cpu"),
        extraction=ExtractionConfig(),
        event_store=EventStoreConfig(db_path=str(tmp_path / "store" / "event_store.db")),
        memory=MemoryConfig(index_path=index_path, fact_view=fact_view),  # type: ignore[arg-type]
    )


class _RecordingFactIndex(FactIndex):
    """A FactIndex that records the view each construction is given."""

    views: list[ViewSpec] = []

    def __init__(self, *args: object, view: ViewSpec, **kwargs: object) -> None:
        type(self).views.append(view)
        super().__init__(*args, view=view, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def recorded_views(monkeypatch: pytest.MonkeyPatch) -> list[ViewSpec]:
    """Swap in `_RecordingFactIndex` where the rebuild imports FactIndex; the views it saw."""
    import backend.memory.fact_index as fact_index_module

    views: list[ViewSpec] = []
    monkeypatch.setattr(_RecordingFactIndex, "views", views)
    monkeypatch.setattr(fact_index_module, "FactIndex", _RecordingFactIndex)
    return views


@dataclass
class _Cli:
    """What the `cli` fixture wires in: the graph, the config `get_config` returns, the closes."""

    graph: FakeGraph
    config: KnowledgeConfig
    closed: list[str] = field(default_factory=list)


@pytest.fixture
def cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Cli:
    """Point the command at a fake graph and a tmp config, as `_load_backend` would give."""
    wired = _Cli(graph=FakeGraph(_FINAL), config=_config(tmp_path))

    class _FakeBackend:
        @staticmethod
        def get_config() -> KnowledgeConfig:
            return wired.config

    def _fake_graph(config: KnowledgeConfig) -> mist_admin.FactRebuildGraph:
        assert config is wired.config
        return mist_admin.FactRebuildGraph(
            executor=wired.graph.executor,
            embedder=FakeEmbedder(),
            close=lambda: wired.closed.append("graph"),
        )

    monkeypatch.setattr(mist_admin, "_load_backend", lambda: _FakeBackend())
    monkeypatch.setattr(mist_admin, "_build_fact_rebuild_graph", _fake_graph)
    return wired


# -- the rebuild --------------------------------------------------------------------------------


class TestRebuildEqualsIncremental:
    def test_rebuilt_rows_equal_rows_built_incrementally(self, tmp_path: Path) -> None:
        incremental = tmp_path / "incremental" / "memory_index.db"
        rebuilt = tmp_path / "rebuilt" / "memory_index.db"
        _build_incrementally(incremental, [_STEP_1, _STEP_2, _FINAL])
        _seed_stale(rebuilt)
        assert _dump(rebuilt) != _dump(incremental)

        result = asyncio.run(
            mist_admin.rebuild_fact_index(
                executor=FakeGraph(_FINAL).executor,
                embedder=FakeEmbedder(),
                index_path=rebuilt,
                embedding_model_id=_MODEL,
                tz=UTC,
                view=_DEFAULT_VIEW,
                debounce_s=0.0,
                rescan_interval_s=300.0,
            )
        )

        assert _dump(rebuilt) == _dump(incremental)
        # Every `_FINAL` fact is the user using a technology: one facet, which the incremental
        # index already held and the rebuild wrote anew, since the drop took the facet tables.
        assert len(_dump(rebuilt)["facets"]) == 1
        assert (result.report.facets_written, result.report.facets_deleted) == (1, 0)
        assert result.dropped_rows == len(_STEP_1) + 1
        assert result.rows == len(_FINAL)
        assert result.report.inserted == len(_FINAL)
        assert result.report.embedded == len(_FINAL)
        assert (result.report.updated, result.report.deleted, result.report.unchanged) == (0, 0, 0)

    def test_the_rebuild_completes_and_leaves_no_rebuild_state(self, tmp_path: Path) -> None:
        path = tmp_path / "memory_index.db"
        _seed_stale(path)

        asyncio.run(
            mist_admin.rebuild_fact_index(
                executor=FakeGraph(_FINAL).executor,
                embedder=FakeEmbedder(),
                index_path=path,
                embedding_model_id=_MODEL,
                tz=UTC,
                view=_DEFAULT_VIEW,
                debounce_s=0.0,
                rescan_interval_s=300.0,
            )
        )

        reader = open_reader(path, embedding_model_id=_MODEL)
        try:
            assert degraded_reason(reader) is None
            keys = {key for (key,) in reader.execute("SELECT key FROM index_meta")}
        finally:
            reader.close()
        assert META_INDEX_STATE not in keys

    def test_the_rebuild_indexes_under_the_view_it_is_given(
        self, tmp_path: Path, recorded_views: list[ViewSpec]
    ) -> None:
        path = tmp_path / "memory_index.db"

        asyncio.run(
            mist_admin.rebuild_fact_index(
                executor=FakeGraph(_FINAL).executor,
                embedder=FakeEmbedder(),
                index_path=path,
                embedding_model_id=_MODEL,
                tz=UTC,
                view=FACT_VIEWS["F1"],
                debounce_s=0.0,
                rescan_interval_s=300.0,
            )
        )

        assert recorded_views == [FACT_VIEWS["F1"]]
        assert _view_texts(path) == _expected_view_texts(FACT_VIEWS["F1"])

    def test_the_dropped_tables_are_every_table_the_fact_ddl_creates(self) -> None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            for statement in FACTS_DDL:
                conn.execute(statement)
            tables = conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        finally:
            conn.close()
        virtual = {name for name, sql in tables if sql.upper().startswith("CREATE VIRTUAL")}
        # A virtual table's shadow tables are named after it and go when it is dropped.
        owned = {
            name for name, _sql in tables if not any(name.startswith(f"{v}_") for v in virtual)
        }

        assert owned == set(mist_admin.FACT_INDEX_TABLES)


# -- the command --------------------------------------------------------------------------------


class TestCommand:
    def test_rebuilds_prints_counts_and_duration_and_exits_zero(
        self, cli: _Cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "memory_index.db"
        _seed_stale(path)

        rc = mist_admin.main(["memory-index-rebuild", "--facts", "--index-path", str(path)])

        out = capsys.readouterr().out
        assert rc == 0
        assert f"index: {path}" in out
        assert f"dropped {len(_STEP_1) + 1} rows" in out
        assert f"rows={len(_FINAL)} inserted={len(_FINAL)}" in out
        assert f"embedded={len(_FINAL)}" in out
        assert "duration:" in out
        assert cli.closed == ["graph"]
        reference = tmp_path / "reference" / "memory_index.db"
        _build_incrementally(reference, [_FINAL])
        assert _dump(path) == _dump(reference)

    @pytest.mark.parametrize("view", ["F0", "F2", "F3"])
    def test_the_rebuild_passes_the_configured_fact_view_to_the_index(
        self, cli: _Cli, tmp_path: Path, recorded_views: list[ViewSpec], view: str
    ) -> None:
        path = tmp_path / "memory_index.db"
        cli.config = _config(tmp_path, fact_view=view)

        rc = mist_admin.main(["memory-index-rebuild", "--facts", "--index-path", str(path)])

        assert rc == 0
        assert recorded_views == [FACT_VIEWS[view]]
        assert _view_texts(path) == _expected_view_texts(FACT_VIEWS[view])

    def test_the_index_defaults_to_beside_the_event_store(
        self, cli: _Cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = mist_admin.main(["memory-index-rebuild", "--facts"])

        expected = tmp_path / "store" / "memory_index.db"
        assert rc == 0
        assert f"index: {expected}" in capsys.readouterr().out
        assert expected.exists()

    def test_the_configured_index_path_wins_over_the_event_store(
        self, cli: _Cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configured = tmp_path / "configured" / "memory_index.db"
        cli.config = _config(tmp_path, index_path=str(configured))

        rc = mist_admin.main(["memory-index-rebuild", "--facts"])

        assert rc == 0
        assert f"index: {configured}" in capsys.readouterr().out

    def test_a_missing_scope_is_a_usage_error(self, cli: _Cli) -> None:
        with pytest.raises(SystemExit) as exc_info:
            mist_admin.main(["memory-index-rebuild"])

        assert exc_info.value.code == 2

    def test_an_unknown_option_is_a_usage_error(self, cli: _Cli) -> None:
        with pytest.raises(SystemExit) as exc_info:
            mist_admin.main(["memory-index-rebuild", "--facts", "--everything"])

        assert exc_info.value.code == 2

    def test_an_in_memory_index_is_refused_with_exit_two(
        self, cli: _Cli, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = mist_admin.main(["memory-index-rebuild", "--facts", "--index-path", ":memory:"])

        assert rc == 2
        assert "in-memory" in capsys.readouterr().err
        assert cli.closed == []

    def test_a_failed_scan_exits_one_and_leaves_the_index_marked_rebuilding(
        self, cli: _Cli, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.graph.error = Neo4jQueryError("graph went away")
        path = tmp_path / "memory_index.db"
        _seed_stale(path)

        rc = mist_admin.main(["memory-index-rebuild", "--facts", "--index-path", str(path)])

        err = capsys.readouterr().err
        assert rc == 1
        assert "graph went away" in err
        assert "rebuilding" in err
        assert cli.closed == ["graph"]
        reader = open_reader(path, embedding_model_id=_MODEL)
        try:
            assert degraded_reason(reader) == "index_rebuilding"
            assert reader.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
        finally:
            reader.close()


class TestGraphWiring:
    def test_the_embedder_is_the_graph_stores_and_the_connection_is_shared(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import backend.factories as factories

        connection = FakeNeo4jConnection()
        built: list[tuple[str, object]] = []
        embedder = FakeEmbedder()

        class _Store:
            embedding_generator = embedder

        def _store(config: KnowledgeConfig, connection: object = None, **kwargs: object):
            built.append(("store", connection))
            assert kwargs == {}
            return _Store()

        monkeypatch.setattr(factories, "build_neo4j_connection", lambda config: connection)
        monkeypatch.setattr(factories, "build_graph_store", _store)

        graph = mist_admin._build_fact_rebuild_graph(_config(tmp_path))

        assert graph.embedder is embedder
        assert built == [("store", connection)]
        assert graph.executor.sync is connection
        assert connection.is_connected()
        graph.close()
        assert not connection.is_connected()

    def test_the_command_is_registered_with_its_scope_flag(self) -> None:
        args = mist_admin.build_parser().parse_args(["memory-index-rebuild", "--facts"])

        assert isinstance(args, argparse.Namespace)
        assert args.facts is True
        assert args.index_path is None
        assert args.func is mist_admin.cmd_memory_index_rebuild
