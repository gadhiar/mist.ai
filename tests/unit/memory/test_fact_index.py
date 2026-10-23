"""Tests for backend.memory.fact_index: every latest-belief fact version, searchable three ways."""

import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.errors import EmbeddingError, MemoryIndexError
from backend.memory.fact_index import FactIndex, content_hash, fact_row_key, structural_row_key
from backend.memory.fact_render import render_fact
from backend.memory.index_db import open_index, open_reader, write_transaction
from backend.memory.models import FactRow, fact_key
from tests.fakes.memory_fakes import FakeEmbedder

_MODEL = "all-MiniLM-L6-v2"
_WAIT_S = 5.0


def _fact(
    row_key: str,
    object_name: str,
    *,
    rel_type: str = "WORKS_AT",
    subject_name: str = "User",
    valid_from: str | None = None,
    valid_to: str | None = None,
    recorded_at: str | None = "2026-01-01T00:00:00+00:00",
    confidence: float = 0.9,
    source_utterance_id: str | None = "utt-1",
) -> FactRow:
    subject_id = subject_name.lower()
    object_id = object_name.lower()
    return FactRow(
        row_key=row_key,
        fact_key=fact_key(subject_id, rel_type, object_id),
        subject_id=subject_id,
        subject_name=subject_name,
        rel_type=rel_type,
        object_id=object_id,
        object_name=object_name,
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at=recorded_at,
        confidence=confidence,
        source_utterance_id=source_utterance_id,
    )


def _edge(subject_name: str, rel_type: str, object_name: str, version_key: str | None) -> FactRow:
    """A fact version keyed the way the graph source keys it, from its edge's `version_key`."""
    key = fact_key(subject_name.lower(), rel_type, object_name.lower())
    return replace(
        _fact("placeholder", object_name, rel_type=rel_type, subject_name=subject_name),
        row_key=fact_row_key(key, version_key),
    )


def _at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Path of the tmp `memory_index.db`."""
    return tmp_path / "memory_index.db"


@pytest.fixture
def conn(db_path: Path) -> Iterator[sqlite3.Connection]:
    """The writer connection to the tmp `memory_index.db`."""
    connection = open_index(db_path, embedding_model_id=_MODEL)
    yield connection
    connection.close()


@pytest.fixture
def reader(conn: sqlite3.Connection, db_path: Path) -> Iterator[sqlite3.Connection]:
    """A read-only second connection to the same file, opened after the writer."""
    connection = open_reader(db_path, embedding_model_id=_MODEL)
    yield connection
    connection.close()


@pytest.fixture
def index(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> FactIndex:
    """A FactIndex writing on `conn` and searching on `reader`, rendering in UTC."""
    return FactIndex(conn, fake_embedder, tz=UTC, reader=reader)


# -- upsert and delete -------------------------------------------------------------------------


def test_upsert_stores_the_row_under_its_row_key(index: FactIndex) -> None:
    row = _fact("rk-acme", "Acme", source_utterance_id="utt-7")

    index.upsert([row])

    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.fact_key == row.fact_key
    assert stored.source_utterance_id == "utt-7"
    assert stored.text == "User works at Acme."
    assert stored.content_hash == content_hash(row, "User works at Acme.")


def test_upsert_embeds_the_rendered_text(index: FactIndex, fake_embedder: FakeEmbedder) -> None:
    index.upsert([_fact("rk-acme", "Acme")])

    assert fake_embedder.calls == ["User works at Acme."]


def test_upsert_skips_the_embed_when_the_content_hash_is_unchanged(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-acme", "Acme")
    index.upsert([row])

    report = index.upsert([row])

    assert fake_embedder.calls == ["User works at Acme."]
    assert (report.inserted, report.updated, report.unchanged, report.embedded) == (0, 0, 1, 0)


def test_upsert_re_embeds_when_the_rendering_changes(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-acme", "Acme")
    index.upsert([row])

    report = index.upsert([replace(row, valid_to="2026-03-01")])

    assert fake_embedder.calls[-1] == "User works at Acme until 1 March 2026."
    assert (report.updated, report.embedded) == (1, 1)
    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.text == "User works at Acme until 1 March 2026."


def test_served_field_change_updates_the_row_without_re_embedding(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-acme", "Acme", confidence=0.6)
    index.upsert([row])

    report = index.upsert([replace(row, confidence=0.95, source_utterance_id="utt-9")])

    assert (report.updated, report.embedded) == (1, 0)
    assert len(fake_embedder.calls) == 1
    stored = index.get("rk-acme")
    assert stored is not None
    assert (stored.confidence, stored.source_utterance_id) == (0.95, "utt-9")


def test_a_different_zone_changes_the_hash_and_re_embeds(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-acme", "Acme", valid_from="2026-03-01T23:30:00+00:00")
    FactIndex(conn, fake_embedder, tz=UTC, reader=reader).upsert([row])
    east = timezone(timedelta(hours=5))

    report = FactIndex(conn, fake_embedder, tz=east, reader=reader).upsert([row])

    assert render_fact(row, UTC) != render_fact(row, east)
    assert (report.updated, report.embedded) == (1, 1)
    assert fake_embedder.calls[-1] == "User works at Acme since 2 March 2026."


def test_delete_removes_the_row_from_every_search(index: FactIndex) -> None:
    row = _fact("rk-acme", "Acme")
    index.upsert([row, _fact("rk-kestrel", "Kestrel")])
    query = FakeEmbedder().generate_embedding(render_fact(row))

    deleted = index.delete(["rk-acme", "rk-missing"])

    assert deleted == 1
    assert index.get("rk-acme") is None
    assert index.search_fts("acme") == []
    assert index.search_names(["tell me about Acme"]) == []
    assert [group.fact_key for group in index.search_vector(query)] == [
        _fact("rk-kestrel", "Kestrel").fact_key
    ]


def test_upsert_rejects_a_repeated_row_key(index: FactIndex) -> None:
    with pytest.raises(ValueError, match="rk-acme"):
        index.upsert([_fact("rk-acme", "Acme"), _fact("rk-acme", "Acme", confidence=0.1)])


def test_an_embedder_of_another_dimension_raises_memory_index_error(
    conn: sqlite3.Connection, reader: sqlite3.Connection
) -> None:
    index = FactIndex(conn, FakeEmbedder(dimension=8), tz=UTC, reader=reader)

    with pytest.raises(MemoryIndexError, match="384"):
        index.upsert([_fact("rk-acme", "Acme")])

    assert index.get("rk-acme") is None


class _FailingEmbedder:
    """An embedder whose `generate_embeddings` raises `error`, as a model that fails to load does."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def generate_embedding(self, text: str) -> list[float]:
        raise self.error

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        raise self.error


def _snapshot_of(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], int, int]:
    rows = conn.execute("SELECT row_key, content_hash, text FROM facts ORDER BY row_key").fetchall()
    fts = conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0]
    vec = conn.execute("SELECT count(*) FROM facts_vec").fetchone()[0]
    return rows, fts, vec


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, ValueError])
@pytest.mark.parametrize("method", ["upsert", "diff_apply"])
def test_an_embedder_failure_raises_memory_index_error_and_writes_nothing(
    index: FactIndex,
    conn: sqlite3.Connection,
    reader: sqlite3.Connection,
    error_type: type[Exception],
    method: str,
) -> None:
    index.upsert([_fact("rk-a", "Acme"), _fact("rk-b", "Kestrel")])
    before = _snapshot_of(conn)
    original = error_type("cannot embed 'User works at Northwind.'")
    failing = FactIndex(conn, _FailingEmbedder(original), tz=UTC, reader=reader)
    # rk-a changes its text, rk-c is new and rk-b is absent, so diff_apply would also delete.
    rows = [_fact("rk-a", "Acme", valid_to="2026-03-01"), _fact("rk-c", "Northwind")]

    with pytest.raises(MemoryIndexError) as caught:
        getattr(failing, method)(rows)

    assert caught.value.__cause__ is original
    message = str(caught.value)
    assert error_type.__name__ in message
    assert "2 texts" in message
    assert "Northwind" not in message
    assert _snapshot_of(conn) == before


def test_an_embedding_error_propagates_unchanged(
    conn: sqlite3.Connection, reader: sqlite3.Connection
) -> None:
    original = EmbeddingError("the embedding model is unavailable")
    failing = FactIndex(conn, _FailingEmbedder(original), tz=UTC, reader=reader)

    with pytest.raises(EmbeddingError) as caught:
        failing.upsert([_fact("rk-a", "Acme")])

    assert caught.value is original
    assert _snapshot_of(conn) == ([], 0, 0)


def test_a_database_failure_raises_memory_index_error(
    index: FactIndex, reader: sqlite3.Connection
) -> None:
    reader.close()

    with pytest.raises(MemoryIndexError):
        index.search_fts("acme")


# -- the reader connection ---------------------------------------------------------------------


def test_search_returns_while_the_writer_holds_a_write_transaction(
    index: FactIndex, conn: sqlite3.Connection
) -> None:
    index.upsert([_fact("rk-acme", "Acme")])

    with write_transaction(conn), ThreadPoolExecutor(max_workers=1) as pool:
        conn.execute("DELETE FROM facts_fts")
        groups = pool.submit(index.search_fts, "acme").result(timeout=_WAIT_S)

    assert [group.versions[0].row_key for group in groups] == ["rk-acme"]


def test_search_returns_while_a_fact_index_write_is_inside_its_transaction(
    index: FactIndex, conn: sqlite3.Connection
) -> None:
    index.upsert([_fact("rk-acme", "Acme")])
    inside = threading.Event()
    release = threading.Event()

    def hold_inside_the_transaction() -> int:
        if conn.in_transaction:
            inside.set()
            release.wait(_WAIT_S)
        return 0

    conn.set_progress_handler(hold_inside_the_transaction, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        deleting = pool.submit(index.delete, ["rk-acme"])
        try:
            assert inside.wait(_WAIT_S)
            groups = pool.submit(index.search_fts, "acme").result(timeout=_WAIT_S)
        finally:
            release.set()
        assert deleting.result(timeout=_WAIT_S) == 1
    conn.set_progress_handler(None, 0)

    assert [group.versions[0].row_key for group in groups] == ["rk-acme"]


def test_a_memory_index_serves_reads_on_its_single_connection(
    fake_embedder: FakeEmbedder,
) -> None:
    connection = open_index(":memory:", embedding_model_id=_MODEL)
    index = FactIndex(connection, fake_embedder, tz=UTC, reader=connection)

    index.upsert([_fact("rk-acme", "Acme")])

    assert [group.versions[0].row_key for group in index.search_fts("acme")] == ["rk-acme"]
    connection.close()


def test_the_reader_is_a_required_argument(
    conn: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    with pytest.raises(TypeError, match="reader"):
        FactIndex(conn, fake_embedder, tz=UTC)  # type: ignore[call-arg]


def test_a_file_backed_index_refuses_its_writer_as_its_reader(
    conn: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    with pytest.raises(ValueError, match="open_reader"):
        FactIndex(conn, fake_embedder, tz=UTC, reader=conn)


# -- row keys ----------------------------------------------------------------------------------


def _stored_row_keys(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT row_key FROM facts")]


def _prior(snapshot: list[FactRow], way: str) -> list[FactRow]:
    """The index content before the diff: nothing, or a changed and partly stale snapshot."""
    if way == "into-empty":
        return []
    stale = _edge("User", "KNOWS", "Someone Gone", "seed")
    return [replace(snapshot[0], confidence=0.1), *snapshot[2:], stale]


_WAYS = [pytest.param("into-empty", id="into-empty"), pytest.param("changed", id="changed")]


@pytest.mark.parametrize("way", _WAYS)
def test_seed_edges_sharing_one_version_key_get_distinct_row_keys(
    index: FactIndex, conn: sqlite3.Connection, way: str
) -> None:
    snapshot = [_edge("User", "KNOWS", f"Person {n:02d}", "seed") for n in range(12)]
    index.diff_apply(_prior(snapshot, way))

    index.diff_apply(snapshot)

    stored = _stored_row_keys(conn)
    assert len(set(stored)) == len(stored) == 12
    assert set(stored) == {row.row_key for row in snapshot}


@pytest.mark.parametrize("way", _WAYS)
def test_open_ended_facts_from_one_turn_get_distinct_row_keys(
    index: FactIndex, conn: sqlite3.Connection, way: str
) -> None:
    snapshot = [
        _edge("User", "WORKS_AT", "Acme", "evt-1|open|open"),
        _edge("User", "USES", "Kestrel", "evt-1|open|open"),
        _edge("User", "LEARNING", "Rust", "evt-1|open|open"),
    ]
    index.diff_apply(_prior(snapshot, way))

    index.diff_apply(snapshot)

    stored = _stored_row_keys(conn)
    assert len(set(stored)) == len(stored) == 3
    assert set(stored) == {row.row_key for row in snapshot}


@pytest.mark.parametrize("way", _WAYS)
def test_the_same_edge_keeps_its_row_key_across_runs(index: FactIndex, way: str) -> None:
    def rescan() -> list[FactRow]:
        return [
            _edge("User", "WORKS_AT", "Acme", "evt-1|2024-01-01|open"),
            _edge("User", "KNOWS", "Sam", "seed"),
            _edge("User", "IS_A", "Person", None),
        ]

    index.diff_apply(_prior(rescan(), way))
    index.diff_apply(rescan())

    report = index.diff_apply(rescan())

    assert (report.inserted, report.deleted, report.unchanged) == (0, 0, 3)


def test_row_key_encoding_cannot_collide_across_the_separator() -> None:
    assert fact_row_key("a|b", "c") != fact_row_key("a", "b|c")
    assert fact_row_key("ab", "c") != fact_row_key("a", "bc")


def test_a_version_named_structural_is_not_the_structural_row() -> None:
    key = fact_key("user", "IS_A", "person")

    assert fact_row_key(key, "structural") != structural_row_key(key)
    assert structural_row_key(key) == fact_row_key(key, None)


# -- FTS search --------------------------------------------------------------------------------


def test_fts_orders_by_bm25(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-tea", "green tea", rel_type="PREFERS"),
            _fact("rk-coffee", "black coffee", rel_type="PREFERS"),
            _fact("rk-both", "green tea with black coffee", rel_type="DISLIKES"),
        ]
    )

    groups = index.search_fts("green tea")

    assert [group.versions[0].row_key for group in groups] == ["rk-tea", "rk-both"]
    assert groups[0].score >= groups[1].score


def test_fts_groups_versions_by_fact_key(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-acme-1", "Acme", valid_from="2019", valid_to="2020"),
            _fact("rk-acme-2", "Acme", valid_from="2022"),
        ]
    )

    groups = index.search_fts("where did I work, Acme?")

    assert len(groups) == 1
    assert {version.row_key for version in groups[0].versions} == {"rk-acme-1", "rk-acme-2"}
    assert set(groups[0].matched) == {"rk-acme-1", "rk-acme-2"}


def test_fts_group_carries_unmatched_versions_of_the_fact(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-old", "Acme", valid_from="2019", valid_to="2020"),
            _fact("rk-new", "Acme", valid_from="2022"),
        ]
    )

    groups = index.search_fts("2019")

    assert groups[0].matched == ("rk-old",)
    current = groups[0].current_at(_at("2024-06-01T00:00:00"))
    assert current is not None
    assert current.row_key == "rk-new"


def test_fts_limit_counts_facts_not_versions(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-a1", "Acme", valid_from="2019", valid_to="2020"),
            _fact("rk-a2", "Acme", valid_from="2022"),
            _fact("rk-b", "Acme Labs"),
        ]
    )

    groups = index.search_fts("acme", limit=1)

    assert len(groups) == 1


def test_fts_query_without_tokens_returns_nothing(index: FactIndex) -> None:
    index.upsert([_fact("rk-acme", "Acme")])

    assert index.search_fts("what is the?") == []


def test_fts_survives_operator_text(index: FactIndex) -> None:
    index.upsert([_fact("rk-acme", "Acme")])

    groups = index.search_fts('NEAR(" acme AND NOT *')

    assert [group.versions[0].row_key for group in groups] == ["rk-acme"]


def test_fts_fetches_a_bounded_number_of_rows(index: FactIndex, reader: sqlite3.Connection) -> None:
    index.upsert([_fact(f"rk-{n:02d}", f"Project {n:02d}", rel_type="WORKS_ON") for n in range(40)])
    statements: list[str] = []
    reader.set_trace_callback(statements.append)

    groups = index.search_fts("user", limit=3)

    reader.set_trace_callback(None)
    assert len(groups) == 3
    assert any("MATCH" in sql and "LIMIT" in sql for sql in statements)


def test_fts_widens_its_row_bound_until_enough_facts_are_found(index: FactIndex) -> None:
    acme_versions = [
        _fact(f"rk-acme-{year}", "Acme", valid_from=str(year), valid_to=str(year + 1))
        for year in range(2000, 2010)
    ]
    depot = _fact("rk-depot", "Acme harbour logistics depot north wing")
    index.upsert([*acme_versions, depot])

    groups = index.search_fts("acme", limit=2)

    assert [len(group.versions) for group in groups] == [10, 1]
    assert groups[1].versions[0].row_key == "rk-depot"


def test_a_fullwidth_stored_name_is_found_by_its_ascii_form(index: FactIndex) -> None:
    index.upsert([_fact("rk-kestrel", "Ｋｅｓｔｒｅｌ")])

    by_text = index.search_fts("kestrel")
    by_name = index.search_names(["is Kestrel hiring"])

    assert [group.versions[0].row_key for group in by_text] == ["rk-kestrel"]
    assert [group.versions[0].row_key for group in by_name] == ["rk-kestrel"]


@pytest.mark.parametrize(
    ("stored", "query"),
    [
        pytest.param("İstanbul", "istanbul", id="stored-dotted-capital"),
        pytest.param("Istanbul", "İstanbul", id="query-dotted-capital"),
        pytest.param("Istanbul", "İSTANBUL", id="query-dotted-upper"),
    ],
)
def test_a_dotted_capital_i_matches_a_plain_i(index: FactIndex, stored: str, query: str) -> None:
    index.upsert([_fact("rk-city", stored, rel_type="INTERESTED_IN")])

    by_text = index.search_fts(query)
    by_name = index.search_names([f"a trip to {query}"])

    assert [group.versions[0].row_key for group in by_text] == ["rk-city"]
    assert [group.versions[0].row_key for group in by_name] == ["rk-city"]


# -- vector search -----------------------------------------------------------------------------


def test_vector_search_returns_the_nearest_fact_first(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-tea", "green tea", rel_type="PREFERS"),
            _fact("rk-acme", "Acme"),
            _fact("rk-rust", "Rust", rel_type="LEARNING"),
        ]
    )
    query = FakeEmbedder().generate_embedding("User prefers green tea.")

    groups = index.search_vector(query, limit=2)

    assert len(groups) == 2
    assert groups[0].versions[0].row_key == "rk-tea"
    assert groups[0].score > groups[1].score


def test_vector_search_groups_versions_by_fact_key(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-acme-1", "Acme", valid_from="2019", valid_to="2020"),
            _fact("rk-acme-2", "Acme", valid_from="2022"),
            _fact("rk-rust", "Rust", rel_type="LEARNING"),
        ]
    )
    query = FakeEmbedder().generate_embedding("User works at Acme.")

    groups = index.search_vector(query, limit=2)

    assert [len(group.versions) for group in groups] == [2, 1]
    assert {version.row_key for version in groups[0].versions} == {"rk-acme-1", "rk-acme-2"}


def test_vector_search_widens_k_until_enough_facts_are_found(index: FactIndex) -> None:
    acme_versions = [
        _fact(f"rk-acme-{year}", "Acme", valid_from=str(year), valid_to=str(year + 1))
        for year in range(2010, 2016)
    ]
    index.upsert([*acme_versions, _fact("rk-rust", "Rust", rel_type="LEARNING")])
    query = FakeEmbedder().generate_embedding("User works at Acme.")

    groups = index.search_vector(query, limit=2)

    assert [len(group.versions) for group in groups] == [6, 1]


def test_vector_search_rejects_a_wrong_dimension(index: FactIndex) -> None:
    with pytest.raises(ValueError, match="384"):
        index.search_vector([0.1, 0.2])


def test_vector_search_on_an_empty_index_returns_nothing(index: FactIndex) -> None:
    assert index.search_vector(FakeEmbedder().generate_embedding("anything")) == []


# -- name search -------------------------------------------------------------------------------


def test_name_search_anchors_facts_whose_entity_is_named(index: FactIndex) -> None:
    index.upsert(
        [
            _fact("rk-corp", "Acme Corp"),
            _fact("rk-labs", "Acme Labs", rel_type="KNOWS"),
            _fact("rk-rust", "Rust", rel_type="LEARNING"),
        ]
    )

    groups = index.search_names(["How is acme corp doing?"])

    assert [group.versions[0].row_key for group in groups] == ["rk-corp"]


def test_name_search_reads_every_text_given(index: FactIndex) -> None:
    index.upsert([_fact("rk-rust", "Rust", rel_type="LEARNING"), _fact("rk-acme", "Acme")])

    groups = index.search_names(["what about it?", "I started on Rust last week"])

    assert [group.versions[0].row_key for group in groups] == ["rk-rust"]


def test_name_search_matches_a_subject_name(index: FactIndex) -> None:
    index.upsert([_fact("rk-sam", "Kestrel", subject_name="Sam", rel_type="WORKS_ON")])

    groups = index.search_names(["what is Sam building"])

    assert [group.versions[0].row_key for group in groups] == ["rk-sam"]


def test_name_search_fetches_a_bounded_number_of_rows(
    index: FactIndex, reader: sqlite3.Connection
) -> None:
    index.upsert(
        [
            _fact(f"rk-{n:02d}", f"Project {n:02d}", subject_name="Sam", rel_type="WORKS_ON")
            for n in range(40)
        ]
    )
    statements: list[str] = []
    reader.set_trace_callback(statements.append)

    groups = index.search_names(["what is Sam building"], limit=3)

    reader.set_trace_callback(None)
    assert len(groups) == 3
    assert any("MATCH" in sql and "LIMIT" in sql for sql in statements)


# -- diff_apply --------------------------------------------------------------------------------


def _table(index: FactIndex, conn: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    rows = conn.execute("SELECT row_key FROM facts").fetchall()
    out: dict[str, tuple[str, str]] = {}
    for (row_key,) in rows:
        stored = index.get(row_key)
        assert stored is not None
        out[row_key] = (stored.text, stored.content_hash)
    return out


def test_diff_apply_leaves_the_table_equal_to_the_snapshot(
    index: FactIndex, conn: sqlite3.Connection
) -> None:
    index.upsert([_fact("rk-a", "Acme"), _fact("rk-b", "Kestrel")])
    snapshot = [
        _fact("rk-b", "Kestrel", valid_to="2026-03-01"),
        _fact("rk-c", "Rust", rel_type="LEARNING"),
    ]

    report = index.diff_apply(snapshot)

    assert (report.inserted, report.updated, report.deleted, report.unchanged) == (1, 1, 1, 0)
    assert _table(index, conn) == {
        row.row_key: (render_fact(row), content_hash(row, render_fact(row))) for row in snapshot
    }
    assert conn.execute("SELECT count(*) FROM facts_fts").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM facts_vec").fetchone()[0] == 2


def test_diff_apply_of_the_same_snapshot_embeds_nothing(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    snapshot = [_fact("rk-a", "Acme"), _fact("rk-b", "Kestrel")]
    index.diff_apply(snapshot)
    calls_before = len(fake_embedder.calls)

    report = index.diff_apply(snapshot)

    assert len(fake_embedder.calls) == calls_before
    assert (report.unchanged, report.embedded) == (2, 0)


def test_diff_apply_of_an_empty_snapshot_empties_the_index(
    index: FactIndex, conn: sqlite3.Connection
) -> None:
    index.upsert([_fact("rk-a", "Acme")])

    index.diff_apply([])

    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM facts_vec").fetchone()[0] == 0


# -- versions and read-time currency -----------------------------------------------------------


def test_two_versions_with_separate_intervals_are_both_kept(index: FactIndex) -> None:
    first = _fact("rk-acme-1", "Acme", valid_from="2019-01-01", valid_to="2020-01-01")
    second = _fact("rk-acme-2", "Acme", valid_from="2022-01-01")

    index.diff_apply([first, second])

    versions = index.versions(first.fact_key)
    assert [version.row_key for version in versions] == ["rk-acme-1", "rk-acme-2"]


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        pytest.param("2019-06-01T00:00:00", "rk-acme-1", id="first-period"),
        pytest.param("2021-06-01T00:00:00", None, id="gap"),
        pytest.param("2024-06-01T00:00:00", "rk-acme-2", id="second-period"),
    ],
)
def test_the_version_current_at_now_is_chosen(
    index: FactIndex, now: str, expected: str | None
) -> None:
    index.diff_apply(
        [
            _fact("rk-acme-1", "Acme", valid_from="2019-01-01", valid_to="2020-01-01"),
            _fact("rk-acme-2", "Acme", valid_from="2022-01-01"),
        ]
    )

    group = index.search_fts("acme")[0]
    current = group.current_at(_at(now))

    assert (current.row_key if current else None) == expected


@pytest.mark.parametrize(
    "now",
    [
        pytest.param("2025-05-04T11:59:59", id="before"),
        pytest.param("2025-05-04T12:00:00", id="at"),
        pytest.param("2025-05-04T12:00:01", id="after"),
    ],
)
def test_a_tombstone_is_kept_but_never_current(index: FactIndex, now: str) -> None:
    instant = "2025-05-04T12:00:00+00:00"
    index.upsert([_fact("rk-tomb", "Acme", valid_from=instant, valid_to=instant)])

    stored = index.get("rk-tomb")

    assert stored is not None
    assert not stored.is_current_at(_at(now))
    assert index.versions(stored.fact_key)[0].row_key == "rk-tomb"


def test_a_clamped_copy_and_its_successor_each_hold_their_own_interval(
    index: FactIndex,
) -> None:
    clamped = _fact(
        "rk-clamped",
        "Acme",
        valid_from="-inf",
        valid_to="2023-04-01T00:00:00+00:00",
        recorded_at="2023-04-01T00:00:00+00:00",
    )
    successor = _fact(
        "rk-open",
        "Acme",
        valid_from="2023-04-01T00:00:00+00:00",
        recorded_at="2023-04-01T00:00:00+00:00",
    )
    index.diff_apply([clamped, successor])

    group = index.search_fts("acme")[0]

    assert {version.row_key for version in group.versions} == {"rk-clamped", "rk-open"}
    earlier = group.current_at(_at("2022-01-01T00:00:00"))
    later = group.current_at(_at("2025-01-01T00:00:00"))
    assert earlier is not None and earlier.row_key == "rk-clamped"
    assert later is not None and later.row_key == "rk-open"


def test_a_future_dated_version_is_not_yet_current(index: FactIndex) -> None:
    index.upsert([_fact("rk-future", "Acme", valid_from="2030-01-01")])

    stored = index.get("rk-future")

    assert stored is not None
    assert not stored.is_current_at(_at("2026-10-22T18:00:00"))
    assert stored.is_current_at(_at("2030-01-02T00:00:00"))


def test_currency_compares_now_in_utc(index: FactIndex) -> None:
    index.upsert([_fact("rk-acme", "Acme", valid_to="2026-10-22T18:00:00+00:00")])
    stored = index.get("rk-acme")
    east = timezone(timedelta(hours=5))

    assert stored is not None
    assert stored.is_current_at(datetime(2026, 10, 22, 22, 59, tzinfo=east))
    assert not stored.is_current_at(datetime(2026, 10, 22, 23, 0, tzinfo=east))


def test_currency_rejects_a_naive_now(index: FactIndex) -> None:
    index.upsert([_fact("rk-acme", "Acme")])
    stored = index.get("rk-acme")

    assert stored is not None
    with pytest.raises(ValueError, match="aware"):
        stored.is_current_at(datetime(2026, 10, 22, 18, 0))
