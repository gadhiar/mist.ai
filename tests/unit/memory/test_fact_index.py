"""Tests for backend.memory.fact_index: every latest-belief fact version, searchable three ways."""

import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest

from backend.errors import EmbeddingError, MemoryIndexError
from backend.memory import retrieval_view
from backend.memory.fact_index import FactIndex, content_hash, fact_row_key, structural_row_key
from backend.memory.fact_render import render_fact
from backend.memory.index_db import open_index, open_reader, write_transaction
from backend.memory.models import FactRow, fact_key
from backend.memory.retrieval_view import FACT_VIEWS, render_fact_view
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
    subject_type: str | None = "User",
    object_type: str | None = "Organization",
) -> FactRow:
    subject_id = subject_name.lower()
    object_id = object_name.lower()
    return FactRow(
        row_key=row_key,
        fact_key=fact_key(subject_id, rel_type, object_id),
        subject_id=subject_id,
        subject_name=subject_name,
        subject_type=subject_type,
        rel_type=rel_type,
        object_id=object_id,
        object_name=object_name,
        object_type=object_type,
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
    """A FactIndex writing on `conn` and searching on `reader`, rendering in UTC, default view."""
    return FactIndex(conn, fake_embedder, tz=UTC, reader=reader)


def _viewed(
    conn: sqlite3.Connection,
    reader: sqlite3.Connection,
    embedder: FakeEmbedder,
    view: str,
    *,
    tz: tzinfo = UTC,
) -> FactIndex:
    """A FactIndex on the same connections that embeds the fact view `view`."""
    return FactIndex(conn, embedder, tz=tz, reader=reader, view=FACT_VIEWS[view])


# -- upsert and delete -------------------------------------------------------------------------


def test_upsert_stores_the_row_under_its_row_key(index: FactIndex) -> None:
    row = _fact("rk-acme", "Acme", source_utterance_id="utt-7")

    index.upsert([row])

    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.fact_key == row.fact_key
    assert stored.source_utterance_id == "utt-7"
    assert stored.text == "User works at Acme."
    assert stored.view_text == "You work at Acme (organization)."
    assert stored.content_hash == content_hash(
        row, "User works at Acme.", "You work at Acme (organization)."
    )


def test_upsert_embeds_the_default_view_f3_not_the_served_text(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    index.upsert([_fact("rk-acme", "Acme", valid_from="2024-03-05")])

    # F3: second person, the object's type gloss, no temporal qualifier.
    assert fake_embedder.calls == ["You work at Acme (organization)."]
    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.text == "User works at Acme since 5 March 2024."


def test_every_read_carries_the_view_text(index: FactIndex) -> None:
    row = _fact("rk-acme", "Acme", source_utterance_id="utt-7")
    index.upsert([row])
    view = "You work at Acme (organization)."
    now = _at("2026-06-01T00:00:00")

    got = index.get("rk-acme")
    (version,) = index.versions(row.fact_key)
    (sibling,) = index.siblings("user", "WORKS_AT", "utt-7", now=now)
    (group,) = index.search_fts("acme")

    assert got is not None
    assert {got.view_text, version.view_text, sibling.view_text} == {view}
    assert group.versions[0].view_text == view


@pytest.mark.parametrize("view", list(FACT_VIEWS))
def test_upsert_embeds_the_view_it_is_given(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder, view: str
) -> None:
    row = _fact("rk-acme", "Acme", valid_from="2024-03-05")

    _viewed(conn, reader, fake_embedder, view).upsert([row])

    expected = render_fact_view(row, FACT_VIEWS[view], UTC)
    assert fake_embedder.calls == [expected]
    stored = reader.execute("SELECT view_text FROM facts WHERE row_key = 'rk-acme'").fetchone()
    assert stored == (expected,)


def test_upsert_skips_the_embed_when_the_content_hash_is_unchanged(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-acme", "Acme")
    index.upsert([row])

    report = index.upsert([row])

    assert fake_embedder.calls == ["You work at Acme (organization)."]
    assert (report.inserted, report.updated, report.unchanged, report.embedded) == (0, 0, 1, 0)


def test_upsert_re_embeds_when_the_view_changes(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    # F2 keeps the qualifier, so a new end date changes the view as well as the served text.
    index = _viewed(conn, reader, fake_embedder, "F2")
    row = _fact("rk-acme", "Acme")
    index.upsert([row])

    report = index.upsert([replace(row, valid_to="2026-03-01")])

    assert fake_embedder.calls[-1] == "You work at Acme (organization) until 1 March 2026."
    assert (report.updated, report.embedded) == (1, 1)
    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.text == "User works at Acme until 1 March 2026."


def test_a_served_text_change_the_view_drops_rewrites_without_re_embedding(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    # F3 drops the qualifier: the served text changes, the embedded view does not.
    row = _fact("rk-acme", "Acme")
    index.upsert([row])

    report = index.upsert([replace(row, valid_to="2026-03-01")])

    assert (report.updated, report.embedded) == (1, 0)
    assert len(fake_embedder.calls) == 1
    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.text == "User works at Acme until 1 March 2026."
    assert stored.view_text == "You work at Acme (organization)."


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
    # Under a view that keeps the qualifier, whose date the zone moves.
    row = _fact("rk-acme", "Acme", valid_from="2026-03-01T23:30:00+00:00")
    _viewed(conn, reader, fake_embedder, "F2").upsert([row])
    east = timezone(timedelta(hours=5))

    report = _viewed(conn, reader, fake_embedder, "F2", tz=east).upsert([row])

    assert render_fact(row, UTC) != render_fact(row, east)
    assert (report.updated, report.embedded) == (1, 1)
    assert fake_embedder.calls[-1] == "You work at Acme (organization) since 2 March 2026."


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
    # rk-a changes its view (the type gloss), rk-c is new and rk-b is absent, so diff_apply
    # would also delete.
    rows = [_fact("rk-a", "Acme", object_type="Project"), _fact("rk-c", "Northwind")]

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


def _table(index: FactIndex, conn: sqlite3.Connection) -> dict[str, tuple[str, str | None, str]]:
    rows = conn.execute("SELECT row_key FROM facts").fetchall()
    out: dict[str, tuple[str, str | None, str]] = {}
    for (row_key,) in rows:
        stored = index.get(row_key)
        assert stored is not None
        out[row_key] = (stored.text, stored.view_text, stored.content_hash)
    return out


def _expected_row(row: FactRow, view: str = "F3") -> tuple[str, str, str]:
    text = render_fact(row)
    view_text = render_fact_view(row, FACT_VIEWS[view])
    return text, view_text, content_hash(row, text, view_text)


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
    assert _table(index, conn) == {row.row_key: _expected_row(row) for row in snapshot}
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


# -- the embedded view -------------------------------------------------------------------------


def _vectors(conn: sqlite3.Connection) -> dict[str, bytes]:
    return dict(
        conn.execute(
            "SELECT f.row_key, v.embedding FROM facts_vec v JOIN facts f ON f.id = v.rowid"
        ).fetchall()
    )


def _dated_snapshot() -> list[FactRow]:
    """Facts that each carry a qualifier, so F2's view and F3's differ for every one."""
    return [
        _fact("rk-acme", "Acme", valid_from="2024-03-05"),
        _fact("rk-rust", "Rust", rel_type="LEARNING", object_type="Skill", valid_from="2025-01-02"),
        _fact(
            "rk-ana",
            "Kestrel",
            subject_name="Ana",
            subject_type="Person",
            rel_type="WORKS_ON",
            object_type="Project",
            valid_to="2026-02-01",
        ),
    ]


def test_a_diff_apply_under_another_view_rewrites_and_re_embeds_every_row(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    snapshot = _dated_snapshot()
    _viewed(conn, reader, fake_embedder, "F2").diff_apply(snapshot)
    before = _vectors(conn)

    report = _viewed(conn, reader, fake_embedder, "F3").diff_apply(snapshot)

    assert (report.inserted, report.updated, report.deleted, report.unchanged) == (0, 3, 0, 0)
    assert report.embedded == len(snapshot)
    assert fake_embedder.calls[-3:] == [render_fact_view(row, FACT_VIEWS["F3"]) for row in snapshot]
    after = _vectors(conn)
    assert after.keys() == before.keys()
    assert all(after[key] != before[key] for key in after)
    index = _viewed(conn, reader, fake_embedder, "F3")
    assert _table(index, conn) == {row.row_key: _expected_row(row, "F3") for row in snapshot}


def test_a_diff_apply_under_the_same_view_embeds_nothing(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    snapshot = _dated_snapshot()
    _viewed(conn, reader, fake_embedder, "F2").diff_apply(snapshot)

    report = _viewed(conn, reader, fake_embedder, "F2").diff_apply(snapshot)

    assert (report.updated, report.unchanged, report.embedded) == (0, 3, 0)


def test_a_subject_type_changed_to_user_keeps_the_text_and_re_embeds(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-acme", "Acme", subject_type="Person")
    index.upsert([row])

    report = index.upsert([replace(row, subject_type="User")])

    assert (report.updated, report.embedded) == (1, 1)
    assert fake_embedder.calls == [
        "User works at Acme (organization).",
        "You work at Acme (organization).",
    ]
    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.text == "User works at Acme."


def _snapshot_of_two_relations() -> list[FactRow]:
    return [
        _fact("rk-acme", "Acme"),
        _fact("rk-brightgrid", "Brightgrid"),
        _fact("rk-vim", "Vim", rel_type="USES", object_type="Technology"),
        _fact("rk-ana", "Ana", rel_type="KNOWS", object_type="Person"),
    ]


def test_a_patched_second_person_template_re_embeds_exactly_its_rows(
    index: FactIndex, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot_of_two_relations()
    index.diff_apply(snapshot)
    calls_before = len(fake_embedder.calls)
    patched = dict(retrieval_view.SECOND_PERSON_TEMPLATES, WORKS_AT="You are employed at {o}")
    monkeypatch.setattr(retrieval_view, "SECOND_PERSON_TEMPLATES", patched)

    report = index.diff_apply(snapshot)

    assert (report.updated, report.unchanged, report.embedded) == (2, 2, 2)
    assert fake_embedder.calls[calls_before:] == [
        "You are employed at Acme (organization).",
        "You are employed at Brightgrid (organization).",
    ]
    stored = index.get("rk-acme")
    assert stored is not None
    assert stored.text == "User works at Acme."


def test_an_unpatched_snapshot_re_embeds_nothing(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    snapshot = _snapshot_of_two_relations()
    index.diff_apply(snapshot)

    report = index.diff_apply(snapshot)

    assert (report.updated, report.unchanged, report.embedded) == (0, 4, 0)


def test_fts_indexes_the_served_text_and_vector_search_the_view(
    index: FactIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The gloss word is also a type term, so a patched template supplies a word that only the
    # view holds.
    patched = dict(retrieval_view.SECOND_PERSON_TEMPLATES, WORKS_AT="You moonlight at {o}")
    monkeypatch.setattr(retrieval_view, "SECOND_PERSON_TEMPLATES", patched)
    index.upsert([_fact("rk-acme", "Acme"), _fact("rk-rust", "Rust", rel_type="LEARNING")])
    query = FakeEmbedder().generate_embedding("moonlight")

    assert index.search_fts("moonlight") == []
    assert index.search_vector(query, limit=1)[0].versions[0].row_key == "rk-acme"
    assert [group.versions[0].row_key for group in index.search_fts("works acme")] == ["rk-acme"]


def test_the_content_hash_covers_the_view_text() -> None:
    row = _fact("rk-acme", "Acme")
    text = render_fact(row)

    assert content_hash(row, text, "You work at Acme.") != content_hash(
        row, text, "You work at Acme (organization)."
    )


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


# -- entity types and type terms ---------------------------------------------------------------


def test_upsert_stores_both_entity_types(index: FactIndex) -> None:
    index.upsert([_fact("rk-kestrel", "Kestrel", rel_type="USES", object_type="Technology")])

    stored = index.get("rk-kestrel")

    assert stored is not None
    assert (stored.subject_type, stored.object_type) == ("User", "Technology")
    assert index.versions(stored.fact_key)[0].object_type == "Technology"


def test_an_untyped_row_stores_no_type(index: FactIndex) -> None:
    index.upsert([_fact("rk-acme", "Acme", subject_type=None, object_type=None)])

    stored = index.get("rk-acme")

    assert stored is not None
    assert (stored.subject_type, stored.object_type) == (None, None)


def test_the_content_hash_covers_both_entity_types() -> None:
    row = _fact("rk-acme", "Acme")
    text = render_fact(row)
    view = "You work at Acme."

    assert content_hash(row, text, view) != content_hash(
        replace(row, object_type="Project"), text, view
    )
    assert content_hash(row, text, view) != content_hash(
        replace(row, subject_type="Person"), text, view
    )


def test_a_type_scoped_query_matches_a_fact_by_its_type_with_no_name_in_it(
    index: FactIndex,
) -> None:
    index.upsert(
        [
            _fact("rk-kestrel", "Kestrel", rel_type="USES", object_type="Technology"),
            _fact("rk-ana", "Ana", rel_type="KNOWS", object_type="Person"),
        ]
    )

    # Neither "Kestrel" nor "uses" is in the query; only the object's type is.
    groups = index.search_fts("which technologies?")

    assert [group.versions[0].row_key for group in groups] == ["rk-kestrel"]
    assert [group.versions[0].row_key for group in index.search_fts("what people")] == ["rk-ana"]


def test_a_type_scoped_query_matches_the_subject_type_too(index: FactIndex) -> None:
    index.upsert(
        [
            _fact(
                "rk-ana-acme",
                "Acme",
                subject_name="Ana",
                subject_type="Person",
                object_type="Organization",
            )
        ]
    )

    assert [group.fact_key for group in index.search_fts("person")] == [
        fact_key("ana", "WORKS_AT", "acme")
    ]


def test_an_untyped_fact_is_not_matched_by_a_type_term(index: FactIndex) -> None:
    index.upsert([_fact("rk-kestrel", "Kestrel", rel_type="USES", object_type=None)])

    assert index.search_fts("technologies") == []


def test_a_type_change_rewrites_the_type_terms_without_re_embedding(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    # F1 has no type gloss, so its view, and so the embedding, do not depend on the type.
    index = _viewed(conn, reader, fake_embedder, "F1")
    row = _fact("rk-kestrel", "Kestrel", rel_type="USES", object_type="Concept")
    index.upsert([row])

    report = index.upsert([replace(row, object_type="Technology")])

    assert (report.updated, report.embedded) == (1, 0)
    assert len(fake_embedder.calls) == 1
    assert [group.fact_key for group in index.search_fts("technology")] == [row.fact_key]
    assert index.search_fts("concept") == []


def test_an_object_type_change_re_embeds_under_a_glossed_view(
    index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    row = _fact("rk-kestrel", "Kestrel", rel_type="USES", object_type="Concept")
    index.upsert([row])

    report = index.upsert([replace(row, object_type="Technology")])

    assert (report.updated, report.embedded) == (1, 1)
    assert fake_embedder.calls == ["You use Kestrel (concept).", "You use Kestrel (technology)."]


def test_name_search_ignores_the_type_terms(index: FactIndex) -> None:
    # "Technology" is the object's type, not a name, so it anchors nothing.
    index.upsert([_fact("rk-kestrel", "Kestrel", rel_type="USES", object_type="Technology")])

    assert index.search_names(["tell me about technology"]) == []


# -- siblings ----------------------------------------------------------------------------------

_NOW = _at("2026-06-01T00:00:00")


def _uses(
    object_name: str,
    *,
    utterance: str = "utt-list",
    recorded_at: str = "2026-01-01T00:00:00+00:00",
    **overrides: object,
) -> FactRow:
    return _fact(
        f"rk-{object_name.lower()}-{utterance}-{recorded_at}",
        object_name,
        rel_type="USES",
        object_type="Technology",
        source_utterance_id=utterance,
        recorded_at=recorded_at,
        **overrides,  # type: ignore[arg-type]
    )


def test_siblings_are_the_current_facts_of_one_subject_relation_and_utterance(
    index: FactIndex,
) -> None:
    index.upsert(
        [
            _uses("Vim"),
            _uses("Rust"),
            _uses("Zig"),
            _uses("Go", utterance="utt-other"),
            _fact("rk-acme", "Acme", source_utterance_id="utt-list"),
            _uses("Kestrel", subject_name="Ana"),
        ]
    )

    siblings = index.siblings("user", "USES", "utt-list", now=_NOW)

    assert [fact.object_name for fact in siblings] == ["Rust", "Vim", "Zig"]


def test_siblings_leave_out_versions_not_current_at_now(index: FactIndex) -> None:
    index.upsert(
        [
            _uses("Vim"),
            _uses("Rust", valid_to="2026-03-01T00:00:00+00:00"),
            _uses("Zig", valid_from="2027-01-01"),
            _uses(
                "Go", valid_from="2026-02-01T00:00:00+00:00", valid_to="2026-02-01T00:00:00+00:00"
            ),
        ]
    )

    siblings = index.siblings("user", "USES", "utt-list", now=_NOW)

    assert [fact.object_name for fact in siblings] == ["Vim"]


def test_siblings_keep_one_version_per_fact(index: FactIndex) -> None:
    # Two current versions of one fact from the utterance: the one `current_at` picks is kept.
    early = _uses("Vim", recorded_at="2026-01-01T00:00:00+00:00")
    late = _uses("Vim", recorded_at="2026-01-02T00:00:00+00:00")
    index.upsert([early, late, _uses("Rust")])

    siblings = index.siblings("user", "USES", "utt-list", now=_NOW)

    assert [fact.row_key for fact in siblings] == [_uses("Rust").row_key, late.row_key]
    group = index.search_fts("vim")[0]
    current = group.current_at(_NOW)
    assert current is not None and current.row_key == late.row_key


def test_siblings_order_is_stable_whatever_the_write_order(
    conn: sqlite3.Connection, reader: sqlite3.Connection, fake_embedder: FakeEmbedder
) -> None:
    rows = [
        _uses("zig", recorded_at="2026-01-01T00:00:01+00:00"),
        _uses("Vim"),
        _uses("rust"),
        _uses("Ada"),
    ]
    index = FactIndex(conn, fake_embedder, tz=UTC, reader=reader)
    index.upsert(rows)
    forward = index.siblings("user", "USES", "utt-list", now=_NOW)
    index.diff_apply([])
    index.upsert(list(reversed(rows)))

    backward = index.siblings("user", "USES", "utt-list", now=_NOW)

    # Recorded order first, then the name case-folded, then the row key.
    assert [fact.object_name for fact in forward] == ["Ada", "rust", "Vim", "zig"]
    assert [fact.row_key for fact in backward] == [fact.row_key for fact in forward]


def test_siblings_of_an_unknown_utterance_are_empty(index: FactIndex) -> None:
    index.upsert([_uses("Vim")])

    assert index.siblings("user", "USES", "utt-missing", now=_NOW) == ()


def test_siblings_reject_a_naive_now(index: FactIndex) -> None:
    index.upsert([_uses("Vim")])

    with pytest.raises(ValueError, match="aware"):
        index.siblings("user", "USES", "utt-list", now=datetime(2026, 6, 1))
