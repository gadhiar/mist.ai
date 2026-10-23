"""Every latest-belief fact version in `memory_index.db`, searchable by FTS, vector and name.

One row per version, keyed by `row_key`, so a clamped history copy, a tombstone and a fact that
held in two separate periods each keep their own interval. Every search groups its hits by
`fact_key` and returns all indexed versions of each fact, matched or not, so the fetcher can
pick the version current at the turn's `now` even when only a past version matched the query.
Currency is decided at read time (`IndexedFact.is_current_at`), so the passage of time never
needs a refresh.

A write skips rows whose content hash is unchanged and re-embeds only rows whose rendered text
changed. The embedder runs holding neither a connection lock nor a transaction (the
resource-lifetime rule): the stored hashes are read, the new texts embedded, and then one write
transaction applies every change. Searches run on a separate read-only connection, so under WAL
they read the last committed snapshot instead of waiting for a write.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo

from backend.errors import MemoryIndexError
from backend.interfaces import EmbeddingProvider

from .fact_render import render_fact
from .fts import build_or_query, fold, words
from .index_db import EMBEDDING_DIMENSION, read_transaction, write_transaction
from .models import FactRow

# The "always / since forever" lower bound, `ALWAYS` in backend/knowledge/curation/intervals.py,
# copied for the reason fact_render.py gives (importing it loads the curation pipeline).
_ALWAYS = "-inf"

# Bumped when the hashed fields change, so every row re-hashes (and re-embeds if its text
# changed) after an upgrade instead of keeping a hash computed another way.
_HASH_VERSION = 1

# Keys per `IN (...)` list: below SQLite's historical 999-variable limit.
_IN_CHUNK = 500

# sqlite-vec refuses a KNN `k` above 4096 ("k value in knn query too large").
_VEC_MAX_K = 4096

_PACK = struct.Struct(f"<{EMBEDDING_DIMENSION}f")

_COLUMNS = (
    "row_key, fact_key, subject_id, subject_name, rel_type, object_id, object_name, text, "
    "valid_from, valid_to, recorded_at, confidence, source_utterance_id, content_hash"
)

# First row bound of a MATCH query, per fact wanted; most facts have one or two versions.
_ROWS_PER_FACT = 4

# bm25-ordered MATCH queries; parameters are the expression and the row bound.
_FTS_SQL = (
    "SELECT f.row_key, f.fact_key, bm25(facts_fts) AS rank "
    "FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
    "WHERE facts_fts MATCH ? ORDER BY rank, f.row_key LIMIT ?"
)
_NAME_SQL = (
    "SELECT f.row_key, f.fact_key, f.subject_name, f.object_name, bm25(facts_fts) AS rank "
    "FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
    "WHERE facts_fts MATCH ? ORDER BY rank, f.row_key LIMIT ?"
)


def fact_row_key(fact_key: str, version_key: str | None) -> str:
    """The `row_key` of one fact version; every producer of FactRows gets its keys here.

    `version_key` is unique only within one (subject, type, object) triple: the versioned
    MERGE is scoped to the triple (reconciliation.py:786), every seed edge carries `'seed'`
    (admin.py:608, :2095), and two open-ended facts stated in one turn share
    `<event>|open|open`. Used raw, those keys would collide across facts, so the key hashes the
    version with its `fact_key` (lead ruling, 2026-10-23). A structural edge has no
    `version_key`; it is merged once per triple, so the fact key alone names its one version.

    The parts are length-prefixed before hashing, so no choice of `fact_key` and `version_key`
    can collide with another (a `|` join would make `("a|b", "c")` and `("a", "b|c")` equal),
    and the structural form hashes a different number of parts, so a version literally named
    "structural" is not the structural row.

    Args:
        fact_key: The fact's identity, from `models.fact_key`.
        version_key: The edge's `version_key`, or None for a structural edge.

    Returns:
        A hex SHA-256 digest, stable across rebuilds that replay the same graph.
    """
    parts = (fact_key, "structural") if version_key is None else (fact_key, "version", version_key)
    encoded = "".join(f"{len(part)}:{part}" for part in parts)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def structural_row_key(fact_key: str) -> str:
    """`row_key` of a structural edge, which has no `version_key`: `fact_row_key(fact_key, None)`."""
    return fact_row_key(fact_key, None)


def content_hash(row: FactRow, text: str) -> str:
    """Hash of everything a row embeds or serves, so any change to it rewrites the row.

    It covers the rendered `text`, which depends on the zone the index renders in, so a zone
    change re-embeds, and every stored field the fetcher can serve. `row_key` is left out: it is
    the identity the hash is stored under, not content.

    Args:
        row: The fact version.
        text: Its rendered sentence, as `render_fact` produced it for the index's zone.

    Returns:
        A hex SHA-256 digest.
    """
    payload = [
        _HASH_VERSION,
        text,
        row.fact_key,
        row.subject_id,
        row.subject_name,
        row.rel_type,
        row.object_id,
        row.object_name,
        row.valid_from,
        row.valid_to,
        row.recorded_at,
        row.confidence,
        row.source_utterance_id,
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexedFact:
    """One fact version as stored in the index: the FactRow fields plus its text and hash."""

    row_key: str
    fact_key: str
    subject_id: str
    subject_name: str
    rel_type: str
    object_id: str
    object_name: str
    text: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str | None
    confidence: float
    source_utterance_id: str | None
    content_hash: str

    def is_current_at(self, now: datetime) -> bool:
        """Whether this version holds at `now`, by graph_store's C1 currency predicate.

        The same string comparison as graph_store.py:684-687 against `now` in UTC ISO form:
        `valid_to` NULL or after `now`, and `valid_from` NULL, `-inf` or not after `now`. A
        tombstone `[t, t)` is therefore never current, and a future-dated version becomes
        current when `now` reaches it.

        Raises:
            ValueError: `now` is naive, so it names no instant to compare with.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        at = now.astimezone(UTC).isoformat()
        if self.valid_to is not None and not self.valid_to > at:
            return False
        return self.valid_from is None or self.valid_from == _ALWAYS or self.valid_from <= at


@dataclass(frozen=True, slots=True)
class FactGroup:
    """One fact in a search result, with every indexed version of it.

    Attributes:
        fact_key: The fact's identity across versions.
        versions: Every indexed version, ordered by `recorded_at` then `row_key`.
        matched: Row keys of the versions the search matched, best first.
        score: Score of the best matched version, higher is better: `-bm25` for text and
            name search, cosine similarity for vector search.
    """

    fact_key: str
    versions: tuple[IndexedFact, ...]
    matched: tuple[str, ...]
    score: float

    def current_at(self, now: datetime) -> IndexedFact | None:
        """The version current at `now`, or None when none holds then.

        Two versions can both hold when the graph has overlapping open intervals; the one
        recorded last wins, then the larger `row_key`, so the choice is stable.
        """
        current = [version for version in self.versions if version.is_current_at(now)]
        if not current:
            return None
        return max(current, key=lambda version: (version.recorded_at or "", version.row_key))


@dataclass(frozen=True, slots=True)
class DiffReport:
    """What one write changed, by row: counts of inserted, updated, deleted and unchanged rows,
    and how many texts were embedded.
    """

    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    embedded: int = 0


@dataclass(frozen=True, slots=True)
class _Prepared:
    row: FactRow
    text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class _Stored:
    content_hash: str
    text: str


class FactIndex:
    """The `facts` table of `memory_index.db` and its FTS5 and vec0 mirrors.

    Writes go through the writer connection; searches and lookups go through the reader, a
    second WAL connection to the same file, so a search reads the last committed snapshot and
    never waits for a write transaction or for `diff_apply`'s scan of the stored hashes. Each
    connection has its own lock, because the `memory-index` executor uses them from two
    threads. Writes through this instance also hold a write lock for the whole operation,
    embedding included, so the hashes a write read are still the stored ones when it commits.
    A writer on another connection (an admin rebuild) is not covered by that lock; the row
    upsert and `BEGIN IMMEDIATE` keep its writes and these from corrupting the table, not from
    racing.

    Args:
        writer: Connection from `index_db.open_index`.
        embedder: Embeds rendered fact text (all-MiniLM-L6-v2 in production).
        tz: Zone timestamp bounds are rendered in, `MemoryConfig.tz`. It is required because
            the rendered text, and so the hash and the embedding, depends on it.
        reader: Connection from `index_db.open_reader` on the same file. None reads on the
            writer under its lock, so a search waits for a running write; that is the only
            option for a `:memory:` index, which has no second connection.
    """

    def __init__(
        self,
        writer: sqlite3.Connection,
        embedder: EmbeddingProvider,
        *,
        tz: tzinfo,
        reader: sqlite3.Connection | None = None,
    ):
        self._writer = writer
        self._embedder = embedder
        self._tz = tz
        self._writer_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader = writer if reader is None else reader
        self._reader_lock = self._writer_lock if reader is None else threading.Lock()

    # -- writes ---------------------------------------------------------------------------------

    def upsert(self, rows: Iterable[FactRow]) -> DiffReport:
        """Insert new rows and rewrite changed ones; rows whose content hash matches are skipped.

        Args:
            rows: Fact versions, each `row_key` at most once.

        Returns:
            The counts; `deleted` is always 0.

        Raises:
            ValueError: A `row_key` repeats.
            MemoryIndexError: The database fails, or the embedder returns the wrong number
                or size of vectors. An exception the embedder raises propagates unchanged.
        """
        prepared = self._prepare(rows)
        with self._write_lock:
            existing = self._read_stored([item.row.row_key for item in prepared])
            return self._apply(prepared, existing, delete_keys=())

    def diff_apply(self, snapshot: Iterable[FactRow]) -> DiffReport:
        """Make the table equal to `snapshot`: delete absent rows, upsert the rest.

        The refresher passes a full rescan of the graph's latest-belief versions, so a row the
        graph no longer has is dropped here, and an unchanged row costs no embedding.

        Args:
            snapshot: Every fact version that should be indexed, each `row_key` at most once.

        Returns:
            The counts.

        Raises:
            ValueError: A `row_key` repeats.
            MemoryIndexError: The database fails, or the embedder returns the wrong number
                or size of vectors. An exception the embedder raises propagates unchanged.
        """
        prepared = self._prepare(snapshot)
        wanted = {item.row.row_key for item in prepared}
        with self._write_lock:
            existing = self._read_stored(None)
            delete_keys = tuple(sorted(key for key in existing if key not in wanted))
            return self._apply(prepared, existing, delete_keys=delete_keys)

    def delete(self, row_keys: Iterable[str]) -> int:
        """Remove the rows with these keys from the table and both mirrors.

        Returns:
            How many rows existed and were removed.

        Raises:
            MemoryIndexError: The database fails.
        """
        keys = list(dict.fromkeys(row_keys))
        with (
            self._write_lock,
            self._writer_lock,
            _index_errors("delete"),
            write_transaction(self._writer),
        ):
            return sum(self._delete_row(key) for key in keys)

    # -- reads, all on the reader ---------------------------------------------------------------

    def get(self, row_key: str) -> IndexedFact | None:
        """The stored version with this `row_key`, or None."""
        with self._reader_lock, _index_errors("get"):
            row = self._reader.execute(
                # `_COLUMNS` is a module constant
                f"SELECT {_COLUMNS} FROM facts WHERE row_key = ?",  # nosec B608
                (row_key,),
            ).fetchone()
        return None if row is None else IndexedFact(*row)

    def versions(self, fact_key: str) -> tuple[IndexedFact, ...]:
        """Every stored version of the fact, ordered by `recorded_at` then `row_key`."""
        with self._reader_lock, _index_errors("versions"):
            return self._load_versions([fact_key]).get(fact_key, ())

    def count(self) -> int:
        """Number of stored fact versions."""
        with self._reader_lock, _index_errors("count"):
            return self._reader.execute("SELECT count(*) FROM facts").fetchone()[0]

    def search_fts(self, query_text: str, *, limit: int = 20) -> list[FactGroup]:
        """Facts whose text or names share a content word with `query_text`, best bm25 first.

        The query is the OR-mode expression of `fts.build_or_query`, so a fact needs only one
        of the query's content words to be a candidate.

        Args:
            query_text: The user's text, unescaped.
            limit: Most facts (not versions) to return.

        Raises:
            MemoryIndexError: The database fails.
        """
        _check_limit(limit)
        expression = build_or_query(query_text)
        if expression is None:
            return []
        with self._reader_lock, _index_errors("FTS search"), read_transaction(self._reader):
            ranked = self._bounded_ranking(
                _FTS_SQL, expression, limit, lambda key, fact, rank: (key, fact, -rank)
            )
            return self._groups(ranked)

    def search_vector(self, embedding: Sequence[float], *, limit: int = 20) -> list[FactGroup]:
        """Facts nearest to `embedding` by cosine distance, nearest first.

        KNN returns versions, and several can belong to one fact, so `k` grows until `limit`
        facts are found, the index is exhausted, or sqlite-vec's ceiling on `k` is reached.

        Args:
            embedding: The query's 384-d embedding.
            limit: Most facts (not versions) to return.

        Raises:
            ValueError: The embedding is not 384-d.
            MemoryIndexError: The database fails.
        """
        _check_limit(limit)
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"query embedding must have {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(embedding)}"
            )
        packed = _PACK.pack(*embedding)
        k = min(limit * 2, _VEC_MAX_K)
        with self._reader_lock, _index_errors("vector search"), read_transaction(self._reader):
            while True:
                hits = self._reader.execute(
                    "SELECT f.row_key, f.fact_key, v.distance "
                    "FROM facts_vec v JOIN facts f ON f.id = v.rowid "
                    "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance, f.row_key",
                    (packed, k),
                ).fetchall()
                ranked = _rank_by_fact((key, fact, 1.0 - distance) for key, fact, distance in hits)
                if len(ranked) >= limit or len(hits) < k or k >= _VEC_MAX_K:
                    break
                k = min(k * 2, _VEC_MAX_K)
            return self._groups(ranked[:limit])

    def search_names(self, texts: Sequence[str], *, limit: int = 20) -> list[FactGroup]:
        """Facts whose subject or object name appears in any of `texts`, for entity anchoring.

        FTS over the two name columns finds candidates; a candidate is kept only when one of
        its names occurs whole in a text, as a run of words. So "Acme Labs" is not anchored by
        a query that names only "Acme Corp", although both share "acme".

        Args:
            texts: The query and the recent user turns.
            limit: Most facts (not versions) to return.

        Raises:
            MemoryIndexError: The database fails.
        """
        _check_limit(limit)
        expression = build_or_query("\n".join(texts))
        if expression is None:
            return []
        haystacks = [f" {' '.join(words(text))} " for text in texts]

        def anchored(
            key: str, fact: str, subject_name: str, object_name: str, rank: float
        ) -> tuple[str, str, float] | None:
            if _named_in(subject_name, haystacks) or _named_in(object_name, haystacks):
                return key, fact, -rank
            return None

        with self._reader_lock, _index_errors("name search"), read_transaction(self._reader):
            ranked = self._bounded_ranking(
                _NAME_SQL, f"{{subject_name object_name}} : ({expression})", limit, anchored
            )
            return self._groups(ranked)

    # -- internals ------------------------------------------------------------------------------

    def _bounded_ranking(
        self,
        sql: str,
        expression: str,
        limit: int,
        to_hit: Callable[..., tuple[str, str, float] | None],
    ) -> list[tuple[str, list[str], float]]:
        """Rank the facts a bm25-ordered MATCH query returns, fetching a bounded row count.

        A query word such as "user" can match every stored row, so rows are fetched under a
        LIMIT that starts at a few rows per wanted fact and doubles only while the rows fetched
        hold fewer than `limit` facts and more rows remain. `to_hit` maps a row to
        `(row_key, fact_key, score)`, or to None to drop it.
        """
        bound = limit * _ROWS_PER_FACT
        while True:
            rows = self._reader.execute(sql, (expression, bound)).fetchall()
            ranked = _rank_by_fact(hit for hit in (to_hit(*row) for row in rows) if hit)
            if len(ranked) >= limit or len(rows) < bound:
                return ranked[:limit]
            bound *= 2

    def _prepare(self, rows: Iterable[FactRow]) -> list[_Prepared]:
        prepared: list[_Prepared] = []
        seen: set[str] = set()
        for row in rows:
            if row.row_key in seen:
                raise ValueError(f"row_key {row.row_key!r} appears more than once")
            seen.add(row.row_key)
            text = render_fact(row, self._tz)
            prepared.append(_Prepared(row, text, content_hash(row, text)))
        return prepared

    def _read_stored(self, row_keys: list[str] | None) -> dict[str, _Stored]:
        """Stored hash and text by `row_key`: of these keys, or of every row when None.

        Read on the writer, as part of the write: a search on the reader does not wait for it.
        """
        stored: dict[str, _Stored] = {}
        with self._writer_lock, _index_errors("read"), read_transaction(self._writer):
            if row_keys is None:
                cursors = [self._writer.execute("SELECT row_key, content_hash, text FROM facts")]
            else:
                cursors = [
                    self._writer.execute(
                        # the IN list holds only `?` placeholders
                        "SELECT row_key, content_hash, text FROM facts "  # nosec B608
                        f"WHERE row_key IN ({_placeholders(len(chunk))})",
                        chunk,
                    )
                    for chunk in _chunks(row_keys)
                ]
            for cursor in cursors:
                for row_key, hashed, text in cursor:
                    stored[row_key] = _Stored(hashed, text)
        return stored

    def _apply(
        self,
        prepared: list[_Prepared],
        existing: dict[str, _Stored],
        *,
        delete_keys: tuple[str, ...],
    ) -> DiffReport:
        writes: list[_Prepared] = []
        to_embed: list[_Prepared] = []
        inserted = updated = unchanged = 0
        for item in prepared:
            stored = existing.get(item.row.row_key)
            if stored is None:
                inserted += 1
            elif stored.content_hash == item.content_hash:
                unchanged += 1
                continue
            else:
                updated += 1
            writes.append(item)
            if stored is None or stored.text != item.text:
                to_embed.append(item)

        vectors = self._embed([item.text for item in to_embed])
        embedding_by_key = {item.row.row_key: vector for item, vector in zip(to_embed, vectors)}

        if writes or delete_keys:
            with self._writer_lock, _index_errors("write"), write_transaction(self._writer):
                deleted = sum(self._delete_row(key) for key in delete_keys)
                for item in writes:
                    self._write_row(item, embedding_by_key.get(item.row.row_key))
        else:
            deleted = 0
        return DiffReport(
            inserted=inserted,
            updated=updated,
            deleted=deleted,
            unchanged=unchanged,
            embedded=len(to_embed),
        )

    def _embed(self, texts: list[str]) -> list[bytes]:
        if not texts:
            return []
        vectors = self._embedder.generate_embeddings(texts)
        if len(vectors) != len(texts):
            raise MemoryIndexError(
                f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
            )
        packed: list[bytes] = []
        for vector in vectors:
            if len(vector) != EMBEDDING_DIMENSION:
                raise MemoryIndexError(
                    f"embedder returned {len(vector)} dimensions, the index stores "
                    f"{EMBEDDING_DIMENSION}"
                )
            packed.append(_PACK.pack(*vector))
        return packed

    def _write_row(self, item: _Prepared, embedding: bytes | None) -> None:
        """Write one row and its FTS mirror, and its vector when it was re-embedded.

        The row keeps its `id` on update, so the mirrors stay joined to it.
        """
        row = item.row
        self._writer.execute(
            "INSERT INTO facts (row_key, fact_key, subject_id, subject_name, rel_type, "
            "object_id, object_name, text, valid_from, valid_to, recorded_at, confidence, "
            "source_utterance_id, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(row_key) DO UPDATE SET "
            "fact_key = excluded.fact_key, subject_id = excluded.subject_id, "
            "subject_name = excluded.subject_name, rel_type = excluded.rel_type, "
            "object_id = excluded.object_id, object_name = excluded.object_name, "
            "text = excluded.text, valid_from = excluded.valid_from, "
            "valid_to = excluded.valid_to, recorded_at = excluded.recorded_at, "
            "confidence = excluded.confidence, "
            "source_utterance_id = excluded.source_utterance_id, "
            "content_hash = excluded.content_hash",
            (
                row.row_key,
                row.fact_key,
                row.subject_id,
                row.subject_name,
                row.rel_type,
                row.object_id,
                row.object_name,
                item.text,
                row.valid_from,
                row.valid_to,
                row.recorded_at,
                row.confidence,
                row.source_utterance_id,
                item.content_hash,
            ),
        )
        (row_id,) = self._writer.execute(
            "SELECT id FROM facts WHERE row_key = ?", (row.row_key,)
        ).fetchone()
        self._writer.execute("DELETE FROM facts_fts WHERE rowid = ?", (row_id,))
        # The mirror holds the folded form, the same form every query is built in.
        self._writer.execute(
            "INSERT INTO facts_fts (rowid, text, subject_name, object_name) VALUES (?, ?, ?, ?)",
            (row_id, fold(item.text), fold(row.subject_name), fold(row.object_name)),
        )
        if embedding is not None:
            self._writer.execute("DELETE FROM facts_vec WHERE rowid = ?", (row_id,))
            self._writer.execute(
                "INSERT INTO facts_vec (rowid, embedding) VALUES (?, ?)", (row_id, embedding)
            )

    def _delete_row(self, row_key: str) -> int:
        found = self._writer.execute(
            "SELECT id FROM facts WHERE row_key = ?", (row_key,)
        ).fetchone()
        if found is None:
            return 0
        (row_id,) = found
        self._writer.execute("DELETE FROM facts_vec WHERE rowid = ?", (row_id,))
        self._writer.execute("DELETE FROM facts_fts WHERE rowid = ?", (row_id,))
        self._writer.execute("DELETE FROM facts WHERE id = ?", (row_id,))
        return 1

    def _groups(self, ranked: list[tuple[str, list[str], float]]) -> list[FactGroup]:
        versions = self._load_versions([fact for fact, _, _ in ranked])
        return [
            FactGroup(
                fact_key=fact,
                versions=versions.get(fact, ()),
                matched=tuple(matched),
                score=score,
            )
            for fact, matched, score in ranked
        ]

    def _load_versions(self, fact_keys: list[str]) -> dict[str, tuple[IndexedFact, ...]]:
        """Every stored version of these facts, read on the reader; hold the reader lock."""
        found: dict[str, list[IndexedFact]] = {}
        for chunk in _chunks(fact_keys):
            rows = self._reader.execute(
                # constant columns, and the IN list holds only `?` placeholders
                f"SELECT {_COLUMNS} FROM facts "  # nosec B608
                f"WHERE fact_key IN ({_placeholders(len(chunk))})",
                chunk,
            )
            for row in rows:
                fact = IndexedFact(*row)
                found.setdefault(fact.fact_key, []).append(fact)
        return {
            key: tuple(sorted(group, key=lambda f: (f.recorded_at or "", f.row_key)))
            for key, group in found.items()
        }


def _rank_by_fact(
    hits: Iterable[tuple[str, str, float]],
) -> list[tuple[str, list[str], float]]:
    """Group best-first `(row_key, fact_key, score)` hits by fact, in order of first hit.

    Each entry is `(fact_key, matched row keys best first, best score)`.
    """
    order: dict[str, tuple[list[str], float]] = {}
    for row_key, fact, score in hits:
        entry = order.get(fact)
        if entry is None:
            order[fact] = ([row_key], score)
        else:
            entry[0].append(row_key)
    return [(fact, matched, score) for fact, (matched, score) in order.items()]


def _named_in(name: str, haystacks: list[str]) -> bool:
    """Whether `name` occurs as a whole run of words in one of the space-padded haystacks."""
    form = " ".join(words(name))
    if len(form) < 2:
        return False
    needle = f" {form} "
    return any(needle in haystack for haystack in haystacks)


def _placeholders(count: int) -> str:
    """`count` comma-separated `?` placeholders for an `IN (...)` list."""
    return ", ".join("?" * count)


def _chunks(keys: list[str]) -> Iterator[list[str]]:
    for start in range(0, len(keys), _IN_CHUNK):
        yield keys[start : start + _IN_CHUNK]


def _check_limit(limit: int) -> None:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")


@contextmanager
def _index_errors(action: str) -> Iterator[None]:
    """Turn an SQLite failure into MemoryIndexError, which the fetcher degrades on."""
    try:
        yield
    except sqlite3.Error as exc:
        raise MemoryIndexError(f"memory index {action} failed: {exc}") from exc
