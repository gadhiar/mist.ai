"""Every latest-belief fact version in `memory_index.db`, searchable by FTS, vector and name.

The FTS mirror also holds each fact's type terms (`type_terms.TYPE_TERMS`), so a question that
names a type ("which technologies do I use?") finds facts with no name in common with it.

One row per version, keyed by `row_key`, so a clamped history copy, a tombstone and a fact that
held in two separate periods each keep their own interval. Every search groups its hits by
`fact_key` and returns all indexed versions of each fact, matched or not, so the fetcher can
pick the version current at the turn's `now` even when only a past version matched the query.
Currency is decided at read time (`IndexedFact.is_current_at`), so the passage of time never
needs a refresh.

Each row holds two texts. `text` is the sentence the block serves (`render_fact`), which the FTS
mirror indexes with the names and type terms. `view_text` is the fact view the index was built
with (`retrieval_view.render_fact_view` under the configured `ViewSpec`), which the vector mirror
embeds, so the vector pool ranks by the view while what is served stays the same.

A write skips rows whose content hash is unchanged and re-embeds only rows whose view text
changed. The hash covers both texts, so a changed view setting rewrites, and re-embeds, every
row whose view it changes at the next diff. The embedder runs holding neither a connection lock
nor a transaction (the resource-lifetime rule): the stored hashes are read, the new texts
embedded, and then one write transaction applies every change. Searches run on a separate read-only connection, so under WAL
they read the last committed snapshot instead of waiting for a write.

Beside the facts the index keeps their facets (backend/memory/facets.py): each subject, relation
and object type held by two or more stored fact keys, with the facet's own view
(`render_facet_view`). Every write keeps the facet tables equal to `facets_of` over the stored
rows, in the same transaction as the fact rows, and embeds new or changed facet views in the
same embedder call as the fact views, so a rescan that fails leaves facts and facets as they
were. A facet stores no members: `facet_members_many` reads them from the fact rows when the
facet is fetched, and keeps those current at the turn's `now`.
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

from .facets import FacetMemberRow, FacetSpec, facet_key, facets_of
from .fact_render import render_fact
from .fts import build_or_query, fold, words
from .index_db import EMBEDDING_DIMENSION, read_transaction, write_transaction
from .models import FactRow
from .retrieval_view import (
    DEFAULT_FACT_VIEW,
    FACT_VIEWS,
    ViewSpec,
    render_facet_view,
    render_fact_view,
)
from .type_terms import type_terms_text

# The "always / since forever" lower bound, `ALWAYS` in backend/knowledge/curation/intervals.py,
# copied for the reason fact_render.py gives (importing it loads the curation pipeline).
_ALWAYS = "-inf"

# Bumped when the hashed fields change, so every row re-hashes (and re-embeds if its view text
# changed) after an upgrade instead of keeping a hash computed another way. 2 added the
# entity types, 3 the view text.
_HASH_VERSION = 3

# Keys per `IN (...)` list: below SQLite's historical 999-variable limit.
_IN_CHUNK = 500

# sqlite-vec refuses a KNN `k` above 4096 ("k value in knn query too large").
_VEC_MAX_K = 4096

_PACK = struct.Struct(f"<{EMBEDDING_DIMENSION}f")

# In `IndexedFact` field order, which builds from these rows positionally.
_COLUMNS = (
    "row_key, fact_key, subject_id, subject_name, subject_type, rel_type, object_id, "
    "object_name, object_type, text, valid_from, valid_to, recorded_at, confidence, "
    "source_utterance_id, content_hash, view_text"
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

# Bumped when the hashed facet fields change, as `_HASH_VERSION` is for the fact rows.
_FACET_HASH_VERSION = 1

# In `IndexedFacet` field order.
_FACET_COLUMNS = (
    "facet_key, subject_id, subject_name, subject_type, rel_type, object_type, view_text"
)

# The fact columns `facets_of` groups by, in `_MemberRow` field order.
_MEMBER_COLUMNS = (
    "row_key, fact_key, subject_id, subject_name, subject_type, rel_type, object_type, "
    "recorded_at"
)

# bm25-ordered MATCH queries over the facet mirror, one row per facet.
_FACET_FTS_SQL = (
    "SELECT f.facet_key, f.subject_name, bm25(facets_fts) AS rank "
    "FROM facets_fts JOIN facets f ON f.id = facets_fts.rowid "
    "WHERE facets_fts MATCH ? ORDER BY rank, f.facet_key LIMIT ?"
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
    """`row_key` of a structural edge, which has no `version_key`.

    Equal to `fact_row_key(fact_key, None)`.
    """
    return fact_row_key(fact_key, None)


def content_hash(row: FactRow, text: str, view_text: str) -> str:
    """Hash of everything a row embeds or serves, so any change to it rewrites the row.

    It covers the served `text` and the embedded `view_text`, both of which can depend on the
    zone the index renders in, and every stored field the fetcher can serve. Covering the view
    text is what makes a changed view setting rewrite the rows whose view it changes. `row_key`
    is left out: it is the identity the hash is stored under, not content.

    Args:
        row: The fact version.
        text: Its served sentence, as `render_fact` produced it for the index's zone.
        view_text: Its fact view, as `render_fact_view` produced it for the index's view and
            zone.

    Returns:
        A hex SHA-256 digest.
    """
    payload = [
        _HASH_VERSION,
        text,
        view_text,
        row.fact_key,
        row.subject_id,
        row.subject_name,
        row.subject_type,
        row.rel_type,
        row.object_id,
        row.object_name,
        row.object_type,
        row.valid_from,
        row.valid_to,
        row.recorded_at,
        row.confidence,
        row.source_utterance_id,
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _facet_hash(facet: FacetSpec, view_text: str) -> str:
    """Hash of everything a facet row stores, so any change to it rewrites the row.

    The members are left out: the row holds none, so a new member leaves it as it is.
    """
    payload = [
        _FACET_HASH_VERSION,
        view_text,
        facet.facet_key,
        facet.subject_id,
        facet.subject_name,
        facet.subject_type,
        facet.rel_type,
        facet.object_type,
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class IndexedFacet:
    """One facet as stored in the index: its group and its embedded view.

    Its members are the fact rows of the same subject, relation and object type;
    `FactIndex.facet_members_many` reads those current at a given time.
    """

    facet_key: str
    subject_id: str
    subject_name: str
    subject_type: str | None
    rel_type: str
    object_type: str
    view_text: str


@dataclass(frozen=True, slots=True)
class IndexedFact:
    """One fact version as stored in the index: the FactRow fields, its texts and its hash.

    `text` is the served sentence and `view_text` the fact view the index embedded. Every read
    of the index fills `view_text`; its None default serves only a fact built outside the index,
    such as a test's, which has no embedding either.
    """

    row_key: str
    fact_key: str
    subject_id: str
    subject_name: str
    subject_type: str | None
    rel_type: str
    object_id: str
    object_name: str
    object_type: str | None
    text: str
    valid_from: str | None
    valid_to: str | None
    recorded_at: str | None
    confidence: float
    source_utterance_id: str | None
    content_hash: str
    view_text: str | None = None

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
        matched: Row keys of the matched versions among the rows the search fetched, best
            first; never empty, and each key is in `versions`. The fetch is cut at a row count
            (the bounded MATCH fetch, or the KNN's `k`), and the cut can fall inside one fact,
            so a version that also matches but ranks past the cut is absent from `matched`
            although it is in `versions`. `matched` is therefore not every matching version.
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
        return max(current, key=_recorded_order)


def _recorded_order(version: IndexedFact) -> tuple[str, str]:
    """The order `current_at` picks among current versions by: recorded last, then `row_key`."""
    return (version.recorded_at or "", version.row_key)


@dataclass(frozen=True, slots=True)
class DiffReport:
    """What one write changed.

    Attributes:
        inserted: Fact rows added.
        updated: Fact rows rewritten.
        deleted: Fact rows removed.
        unchanged: Fact rows skipped, their content hash unchanged.
        embedded: Fact view texts embedded. The facet views embedded in the same call are not
            counted here; each is a facet in `facets_written`.
        facets_written: Facets added or rewritten.
        facets_deleted: Facets removed, having fewer than two fact keys left.
    """

    inserted: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    embedded: int = 0
    facets_written: int = 0
    facets_deleted: int = 0


@dataclass(frozen=True, slots=True)
class _Prepared:
    row: FactRow
    text: str
    view_text: str
    content_hash: str


# A facet's group: subject id, relation type and object type.
_Group = tuple[str, str, str | None]


@dataclass(frozen=True, slots=True)
class _Stored:
    content_hash: str
    view_text: str
    group: _Group


@dataclass(frozen=True, slots=True)
class _StoredFacet:
    content_hash: str
    view_text: str


@dataclass(frozen=True, slots=True)
class _MemberRow:
    """The stored fields of one fact row that `facets_of` reads (`FacetMemberRow`)."""

    row_key: str
    fact_key: str
    subject_id: str
    subject_name: str
    subject_type: str | None
    rel_type: str
    object_type: str | None
    recorded_at: str | None


@dataclass(frozen=True, slots=True)
class _PreparedFacet:
    facet: FacetSpec
    view_text: str
    content_hash: str


@dataclass(frozen=True, slots=True)
class _FacetChanges:
    """The facet writes one apply makes: rows to write, which of them to embed, keys to drop."""

    writes: tuple[_PreparedFacet, ...] = ()
    to_embed: tuple[_PreparedFacet, ...] = ()
    delete_keys: tuple[str, ...] = ()


class FactIndex:
    """The `facts` table of `memory_index.db`, the `facets` grouped from it, and their mirrors.

    Writes go through the writer connection; searches and lookups go through the reader, a
    second WAL connection to the same file, so a search reads the last committed snapshot and
    never waits for a write transaction or for `diff_apply`'s scan of the stored hashes. Each
    connection has its own lock, because the `memory-index` executor uses them from two
    threads. Writes through this instance also hold a write lock for the whole operation,
    embedding included, so the hashes a write read are still the stored ones when it commits.
    A writer on another connection (an admin rebuild) is not covered by that lock; the row
    upsert and `BEGIN IMMEDIATE` keep its writes and these from corrupting the table, not from
    racing.

    A write embeds before it opens its transaction, so an embedder failure leaves the table
    untouched. RuntimeError, OSError and ValueError from the embedder become MemoryIndexError,
    so a caller that handles MistError sees the failure; EmbeddingError, already a MistError,
    propagates unchanged.

    Args:
        writer: Connection from `index_db.open_index`.
        embedder: Embeds each fact's view text (all-MiniLM-L6-v2 in production).
        tz: Zone timestamp bounds are rendered in, `MemoryConfig.tz`. It is required because
            the rendered texts, and so the hash and the embedding, depend on it.
        reader: Connection from `index_db.open_reader` on the same file. It is required, so
            the caller chooses the connection searches use. A `:memory:` index has no second
            connection and passes its writer here; searches then run on the writer under its
            lock and wait for a running write. A file-backed index given its writer raises.
        view: The fact view to embed, `FACT_VIEWS[MemoryConfig.fact_view]` in production.
            Rows written under another view are rewritten and re-embedded at the next write
            that includes them, wherever their view text differs.

    Raises:
        ValueError: `reader` is `writer` and the writer's database is a file, which would
            make every search wait for the writes the reader exists to avoid.
    """

    def __init__(
        self,
        writer: sqlite3.Connection,
        embedder: EmbeddingProvider,
        *,
        tz: tzinfo,
        reader: sqlite3.Connection,
        view: ViewSpec = FACT_VIEWS[DEFAULT_FACT_VIEW],
    ):
        shared = reader is writer
        if shared and _main_database_file(writer):
            raise ValueError(
                "a file-backed index needs its own reader from index_db.open_reader, "
                "not the writer"
            )
        self._writer = writer
        self._embedder = embedder
        self._tz = tz
        self._view = view
        self._writer_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._reader = reader
        self._reader_lock = self._writer_lock if shared else threading.Lock()

    # -- writes ---------------------------------------------------------------------------------

    def upsert(self, rows: Iterable[FactRow]) -> DiffReport:
        """Insert new rows and rewrite changed ones; rows whose content hash matches are skipped.

        Args:
            rows: Fact versions, each `row_key` at most once.

        Returns:
            The counts; `deleted` is always 0.

        Raises:
            ValueError: A `row_key` repeats.
            MemoryIndexError: The database fails; the embedder raises RuntimeError, OSError
                or ValueError; or it returns the wrong number or size of vectors. Nothing is
                written.
            EmbeddingError: The embedder raised it; it propagates unchanged.
        """
        prepared = self._prepare(rows)
        with self._write_lock:
            with self._writer_lock, _index_errors("read"), read_transaction(self._writer):
                existing = self._read_stored([item.row.row_key for item in prepared])
                # A row can leave one group and join another, so both groups are read.
                groups = {_group(item.row) for item in prepared}
                groups.update(stored.group for stored in existing.values())
                members = self._read_members(groups)
                stored_facets = self._read_stored_facets(_facet_keys(groups))
            members.update((item.row.row_key, item.row) for item in prepared)
            facets = self._facet_changes(members.values(), stored_facets)
            return self._apply(prepared, existing, delete_keys=(), facets=facets)

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
            MemoryIndexError: The database fails; the embedder raises RuntimeError, OSError
                or ValueError; or it returns the wrong number or size of vectors. Nothing is
                written or deleted.
            EmbeddingError: The embedder raised it; it propagates unchanged.
        """
        prepared = self._prepare(snapshot)
        wanted = {item.row.row_key for item in prepared}
        with self._write_lock:
            with self._writer_lock, _index_errors("read"), read_transaction(self._writer):
                existing = self._read_stored(None)
                stored_facets = self._read_stored_facets(None)
            delete_keys = tuple(sorted(key for key in existing if key not in wanted))
            # The snapshot is every row the table holds after the apply.
            facets = self._facet_changes([item.row for item in prepared], stored_facets)
            return self._apply(prepared, existing, delete_keys=delete_keys, facets=facets)

    def delete(self, row_keys: Iterable[str]) -> int:
        """Remove the rows with these keys from the table and both mirrors.

        A facet left with fewer than two fact keys goes too. One whose view the removal changes
        (its newest member named the subject differently) is rewritten and re-embedded, the
        embedding computed before the write transaction opens.

        Returns:
            How many rows existed and were removed.

        Raises:
            MemoryIndexError: The database fails, or the embedder fails as for `upsert`;
                nothing is removed.
            EmbeddingError: The embedder raised it; it propagates unchanged.
        """
        keys = list(dict.fromkeys(row_keys))
        with self._write_lock:
            with self._writer_lock, _index_errors("read"), read_transaction(self._writer):
                existing = self._read_stored(keys)
                groups = {stored.group for stored in existing.values()}
                members = self._read_members(groups)
                stored_facets = self._read_stored_facets(_facet_keys(groups))
            remaining = [row for key, row in members.items() if key not in existing]
            facets = self._facet_changes(remaining, stored_facets)
            report = self._apply([], existing, delete_keys=tuple(keys), facets=facets)
            return report.deleted

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

    def siblings(
        self,
        subject_id: str,
        rel_type: str,
        source_utterance_id: str,
        *,
        now: datetime,
    ) -> tuple[IndexedFact, ...]:
        """The current facts one utterance stated with one subject and relation, as a list.

        "I use Vim, Rust and Zig" is three facts that share `subject_id`, `rel_type` and
        `source_utterance_id`; the fetcher serves them together as one list item. Only the
        versions current at `now` (`IndexedFact.is_current_at`) are siblings, so a list item
        never mixes in a closed, tombstoned or future-dated version. A fact with two current
        versions from the utterance keeps the one `FactGroup.current_at` would pick.

        `now` is keyword-only and beyond the plan's three-key signature because currency is
        decided at read time; the index holds versions that are not current.

        Args:
            subject_id: The shared subject.
            rel_type: The shared relation type.
            source_utterance_id: The utterance the facts were extracted from.
            now: The turn's instant, timezone-aware.

        Returns:
            One version per fact, ordered by `recorded_at`, then the folded object name, then
            `row_key`: a total order, so it does not depend on the order the rows were
            written to the index in.

        Raises:
            ValueError: `now` is naive.
            MemoryIndexError: The database fails.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._reader_lock, _index_errors("sibling lookup"):
            rows = self._reader.execute(
                # `_COLUMNS` is a module constant
                f"SELECT {_COLUMNS} FROM facts "  # nosec B608
                "WHERE subject_id = ? AND rel_type = ? AND source_utterance_id = ?",
                (subject_id, rel_type, source_utterance_id),
            ).fetchall()
        chosen: dict[str, IndexedFact] = {}
        for fact in (IndexedFact(*row) for row in rows):
            if not fact.is_current_at(now):
                continue
            kept = chosen.get(fact.fact_key)
            if kept is None or _recorded_order(fact) > _recorded_order(kept):
                chosen[fact.fact_key] = fact
        return tuple(
            sorted(
                chosen.values(),
                key=lambda fact: (fact.recorded_at or "", fold(fact.object_name), fact.row_key),
            )
        )

    def count(self) -> int:
        """Number of stored fact versions."""
        with self._reader_lock, _index_errors("count"):
            return self._reader.execute("SELECT count(*) FROM facts").fetchone()[0]

    def search_fts(self, query_text: str, *, limit: int = 20) -> list[FactGroup]:
        """Facts whose text, names or type terms share a content word with `query_text`.

        Best bm25 first.

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

    # -- facets, all on the reader ----------------------------------------------------------------

    def search_facets_fts(self, query_text: str, *, limit: int = 20) -> list[IndexedFacet]:
        """Facets whose view, subject name or type terms share a content word with the query.

        Best bm25 first, with the same OR-mode expression `search_fts` uses, so "which
        organisations am I a member of?" finds the facet worded "You are a member of an
        organization, organisation or company." through its type's terms.

        Args:
            query_text: The user's text, unescaped.
            limit: Most facets to return.

        Raises:
            MemoryIndexError: The database fails.
        """
        _check_limit(limit)
        expression = build_or_query(query_text)
        if expression is None:
            return []
        with self._reader_lock, _index_errors("facet FTS search"), read_transaction(self._reader):
            rows = self._reader.execute(_FACET_FTS_SQL, (expression, limit)).fetchall()
            return self._load_facets([key for key, _name, _rank in rows])

    def search_facets_vector(
        self, embedding: Sequence[float], *, limit: int = 20
    ) -> list[IndexedFacet]:
        """Facets whose view is nearest to `embedding` by cosine distance, nearest first.

        Args:
            embedding: The query's 384-d embedding.
            limit: Most facets to return; sqlite-vec's ceiling on `k` caps it.

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
        with (
            self._reader_lock,
            _index_errors("facet vector search"),
            read_transaction(self._reader),
        ):
            rows = self._reader.execute(
                "SELECT f.facet_key FROM facets_vec v JOIN facets f ON f.id = v.rowid "
                "WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance, f.facet_key",
                (packed, min(limit, _VEC_MAX_K)),
            ).fetchall()
            return self._load_facets([key for (key,) in rows])

    def search_facet_names(self, texts: Sequence[str], *, limit: int = 20) -> list[IndexedFacet]:
        """Facets whose subject's name occurs whole in one of `texts`, for entity anchoring.

        The rule `search_names` applies: FTS over the subject name finds candidates, and one is
        kept only when its name occurs in a text as a whole run of words, so "Ana Lima" is not
        anchored by a text that names only "Ana" or only "Ana Costa".

        Args:
            texts: The query and the recent user turns.
            limit: Most facets to return.

        Raises:
            MemoryIndexError: The database fails.
        """
        _check_limit(limit)
        expression = build_or_query("\n".join(texts))
        if expression is None:
            return []
        haystacks = [f" {' '.join(words(text))} " for text in texts]
        bound = limit * _ROWS_PER_FACT
        with self._reader_lock, _index_errors("facet name search"), read_transaction(self._reader):
            # A word such as "user" can match many facets the whole-name rule then drops, so
            # the row bound doubles while too few facets are kept and more rows remain.
            while True:
                rows = self._reader.execute(
                    _FACET_FTS_SQL, (f"{{subject_name}} : ({expression})", bound)
                ).fetchall()
                keys = [key for key, name, _rank in rows if _named_in(name, haystacks)]
                if len(keys) >= limit or len(rows) < bound:
                    break
                bound *= 2
            return self._load_facets(keys[:limit])

    def facet_members_many(
        self, facet_keys: Iterable[str], *, now: datetime
    ) -> dict[str, tuple[IndexedFact, ...]]:
        """The members of each facet current at `now`, every facet in one read.

        A member is a fact key of the facet's subject, relation and object type. Facets are
        grouped over every stored version, so a member with no version current at `now`
        (`IndexedFact.is_current_at`) is left out, and a facet may have fewer than two members
        left; the caller decides what that means.

        Args:
            facet_keys: The facets to look up; a repeated key is read once.
            now: The turn's instant, timezone-aware.

        Returns:
            Each requested key mapped to its members, one version per fact key (the one
            `FactGroup.current_at` would pick), newest recorded first and then by fact key. An
            unknown key maps to an empty tuple.

        Raises:
            ValueError: `now` is naive.
            MemoryIndexError: The database fails.
        """
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        keys = list(dict.fromkeys(facet_keys))
        rows_by_key: dict[str, list[tuple[object, ...]]] = {}
        with (
            self._reader_lock,
            _index_errors("facet member lookup"),
            read_transaction(self._reader),
        ):
            for facet in self._load_facets(keys):
                rows_by_key[facet.facet_key] = self._reader.execute(
                    # `_COLUMNS` is a module constant
                    f"SELECT {_COLUMNS} FROM facts "  # nosec B608
                    "WHERE subject_id = ? AND rel_type = ? AND object_type = ?",
                    (facet.subject_id, facet.rel_type, facet.object_type),
                ).fetchall()
        members: dict[str, tuple[IndexedFact, ...]] = {}
        for key in keys:
            chosen: dict[str, IndexedFact] = {}
            for fact in (IndexedFact(*row) for row in rows_by_key.get(key, ())):
                if not fact.is_current_at(now):
                    continue
                kept = chosen.get(fact.fact_key)
                if kept is None or _recorded_order(fact) > _recorded_order(kept):
                    chosen[fact.fact_key] = fact
            # Two stable passes: fact key ascending, then recorded descending.
            ordered = sorted(chosen.values(), key=lambda fact: fact.fact_key)
            ordered.sort(key=lambda fact: fact.recorded_at or "", reverse=True)
            members[key] = tuple(ordered)
        return members

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
            view_text = render_fact_view(row, self._view, self._tz)
            prepared.append(_Prepared(row, text, view_text, content_hash(row, text, view_text)))
        return prepared

    def _read_stored(self, row_keys: list[str] | None) -> dict[str, _Stored]:
        """Stored hash, view text and group by `row_key`: of these keys, or of every row.

        Read on the writer, as part of the write, so a search on the reader does not wait for
        it; the caller holds the writer lock and a read transaction.
        """
        select = "SELECT row_key, content_hash, view_text, subject_id, rel_type, object_type"
        if row_keys is None:
            cursors = [self._writer.execute(f"{select} FROM facts")]
        else:
            cursors = [
                self._writer.execute(
                    # constant columns, and the IN list holds only `?` placeholders
                    f"{select} FROM facts "  # nosec B608
                    f"WHERE row_key IN ({_placeholders(len(chunk))})",
                    chunk,
                )
                for chunk in _chunks(row_keys)
            ]
        stored: dict[str, _Stored] = {}
        for cursor in cursors:
            for row_key, hashed, view_text, subject_id, rel_type, object_type in cursor:
                stored[row_key] = _Stored(hashed, view_text, (subject_id, rel_type, object_type))
        return stored

    def _read_members(self, groups: Iterable[_Group]) -> dict[str, FacetMemberRow]:
        """Every stored row of these groups by `row_key`, with the fields facets group by.

        On the writer, under the caller's lock and read transaction. A group with no object
        type can hold no facet, so it is not read. Each read is served by
        `idx_facts_siblings`, whose leading columns are the subject and the relation.
        """
        members: dict[str, FacetMemberRow] = {}
        for subject_id, rel_type, object_type in sorted(_typed(groups)):
            rows = self._writer.execute(
                # `_MEMBER_COLUMNS` is a module constant
                f"SELECT {_MEMBER_COLUMNS} FROM facts "  # nosec B608
                "WHERE subject_id = ? AND rel_type = ? AND object_type = ?",
                (subject_id, rel_type, object_type),
            )
            for row in rows:
                member = _MemberRow(*row)
                members[member.row_key] = member
        return members

    def _read_stored_facets(self, facet_keys: list[str] | None) -> dict[str, _StoredFacet]:
        """Stored hash and view text by `facet_key`: of these keys, or of every facet.

        On the writer, under the caller's lock and read transaction.
        """
        select = "SELECT facet_key, content_hash, view_text FROM facets"
        if facet_keys is None:
            cursors = [self._writer.execute(select)]
        else:
            cursors = [
                self._writer.execute(
                    # the IN list holds only `?` placeholders
                    f"{select} WHERE facet_key IN ({_placeholders(len(chunk))})",  # nosec B608
                    chunk,
                )
                for chunk in _chunks(facet_keys)
            ]
        return {
            key: _StoredFacet(hashed, view_text)
            for cursor in cursors
            for key, hashed, view_text in cursor
        }

    def _facet_changes(
        self, rows_after: Iterable[FacetMemberRow], stored: dict[str, _StoredFacet]
    ) -> _FacetChanges:
        """The facet writes that make the stored facets equal to `facets_of(rows_after)`.

        `rows_after` is every row the write leaves in the groups `stored` was read for. A
        facet whose relation has no template has no view, so it is not stored. A facet is
        rewritten when its hash changes and re-embedded only when its view text does.
        """
        wanted: dict[str, _PreparedFacet] = {}
        for facet in facets_of(rows_after).values():
            view_text = render_facet_view(facet, self._view)
            if view_text is not None:
                wanted[facet.facet_key] = _PreparedFacet(
                    facet, view_text, _facet_hash(facet, view_text)
                )
        writes: list[_PreparedFacet] = []
        to_embed: list[_PreparedFacet] = []
        for key, item in wanted.items():
            kept = stored.get(key)
            if kept is not None and kept.content_hash == item.content_hash:
                continue
            writes.append(item)
            if kept is None or kept.view_text != item.view_text:
                to_embed.append(item)
        return _FacetChanges(
            writes=tuple(writes),
            to_embed=tuple(to_embed),
            delete_keys=tuple(sorted(key for key in stored if key not in wanted)),
        )

    def _apply(
        self,
        prepared: list[_Prepared],
        existing: dict[str, _Stored],
        *,
        delete_keys: tuple[str, ...],
        facets: _FacetChanges,
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
            # The vector mirror holds the view's embedding, so only a new view needs one.
            if stored is None or stored.view_text != item.view_text:
                to_embed.append(item)

        # One embedder call for the fact views and the facet views together.
        vectors = self._embed(
            [item.view_text for item in to_embed] + [item.view_text for item in facets.to_embed]
        )
        embedding_by_key = {item.row.row_key: vector for item, vector in zip(to_embed, vectors)}
        facet_embedding_by_key = {
            item.facet.facet_key: vector
            for item, vector in zip(facets.to_embed, vectors[len(to_embed) :])
        }

        deleted = facets_deleted = 0
        if writes or delete_keys or facets.writes or facets.delete_keys:
            with self._writer_lock, _index_errors("write"), write_transaction(self._writer):
                deleted = sum(self._delete_row(key) for key in delete_keys)
                for item in writes:
                    self._write_row(item, embedding_by_key.get(item.row.row_key))
                facets_deleted = sum(self._delete_facet(key) for key in facets.delete_keys)
                for facet in facets.writes:
                    self._write_facet(facet, facet_embedding_by_key.get(facet.facet.facet_key))
        return DiffReport(
            inserted=inserted,
            updated=updated,
            deleted=deleted,
            unchanged=unchanged,
            embedded=len(to_embed),
            facets_written=len(facets.writes),
            facets_deleted=facets_deleted,
        )

    def _embed(self, texts: list[str]) -> list[bytes]:
        if not texts:
            return []
        try:
            vectors = self._embedder.generate_embeddings(texts)
        except (RuntimeError, OSError, ValueError) as exc:
            # The production embedder loads its model on first use, so a missing model or CUDA
            # out of memory arrives here as a raw RuntimeError or OSError. The message names
            # the type and the count only: the texts are the user's facts.
            raise MemoryIndexError(
                f"embedder failed with {type(exc).__name__} on {len(texts)} texts"
            ) from exc
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
        """Write one row and its FTS mirror, and its vector when its view was re-embedded.

        The row keeps its `id` on update, so the mirrors stay joined to it. The FTS mirror takes
        the served text, as before views existed; only the vector mirror holds the view.
        """
        row = item.row
        self._writer.execute(
            "INSERT INTO facts (row_key, fact_key, subject_id, subject_name, subject_type, "
            "rel_type, object_id, object_name, object_type, text, valid_from, valid_to, "
            "recorded_at, confidence, source_utterance_id, content_hash, view_text) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(row_key) DO UPDATE SET "
            "fact_key = excluded.fact_key, subject_id = excluded.subject_id, "
            "subject_name = excluded.subject_name, subject_type = excluded.subject_type, "
            "rel_type = excluded.rel_type, object_id = excluded.object_id, "
            "object_name = excluded.object_name, object_type = excluded.object_type, "
            "text = excluded.text, valid_from = excluded.valid_from, "
            "valid_to = excluded.valid_to, recorded_at = excluded.recorded_at, "
            "confidence = excluded.confidence, "
            "source_utterance_id = excluded.source_utterance_id, "
            "content_hash = excluded.content_hash, view_text = excluded.view_text",
            (
                row.row_key,
                row.fact_key,
                row.subject_id,
                row.subject_name,
                row.subject_type,
                row.rel_type,
                row.object_id,
                row.object_name,
                row.object_type,
                item.text,
                row.valid_from,
                row.valid_to,
                row.recorded_at,
                row.confidence,
                row.source_utterance_id,
                item.content_hash,
                item.view_text,
            ),
        )
        (row_id,) = self._writer.execute(
            "SELECT id FROM facts WHERE row_key = ?", (row.row_key,)
        ).fetchone()
        self._writer.execute("DELETE FROM facts_fts WHERE rowid = ?", (row_id,))
        # The mirror holds the folded form, the same form every query is built in.
        self._writer.execute(
            "INSERT INTO facts_fts (rowid, text, subject_name, object_name, type_terms) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                row_id,
                fold(item.text),
                fold(row.subject_name),
                fold(row.object_name),
                fold(type_terms_text(row.subject_type, row.object_type)),
            ),
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

    def _write_facet(self, item: _PreparedFacet, embedding: bytes | None) -> None:
        """Write one facet row and its FTS mirror, and its vector when its view was embedded.

        The row keeps its `id` on update, so the mirrors stay joined to it.
        """
        facet = item.facet
        self._writer.execute(
            "INSERT INTO facets (facet_key, subject_id, subject_name, subject_type, rel_type, "
            "object_type, view_text, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(facet_key) DO UPDATE SET "
            "subject_id = excluded.subject_id, subject_name = excluded.subject_name, "
            "subject_type = excluded.subject_type, rel_type = excluded.rel_type, "
            "object_type = excluded.object_type, view_text = excluded.view_text, "
            "content_hash = excluded.content_hash",
            (
                facet.facet_key,
                facet.subject_id,
                facet.subject_name,
                facet.subject_type,
                facet.rel_type,
                facet.object_type,
                item.view_text,
                item.content_hash,
            ),
        )
        (facet_id,) = self._writer.execute(
            "SELECT id FROM facets WHERE facet_key = ?", (facet.facet_key,)
        ).fetchone()
        self._writer.execute("DELETE FROM facets_fts WHERE rowid = ?", (facet_id,))
        self._writer.execute(
            "INSERT INTO facets_fts (rowid, view_text, subject_name, type_terms) "
            "VALUES (?, ?, ?, ?)",
            (
                facet_id,
                fold(item.view_text),
                fold(facet.subject_name),
                fold(type_terms_text(facet.subject_type, facet.object_type)),
            ),
        )
        if embedding is not None:
            self._writer.execute("DELETE FROM facets_vec WHERE rowid = ?", (facet_id,))
            self._writer.execute(
                "INSERT INTO facets_vec (rowid, embedding) VALUES (?, ?)", (facet_id, embedding)
            )

    def _delete_facet(self, key: str) -> int:
        found = self._writer.execute("SELECT id FROM facets WHERE facet_key = ?", (key,)).fetchone()
        if found is None:
            return 0
        (facet_id,) = found
        self._writer.execute("DELETE FROM facets_vec WHERE rowid = ?", (facet_id,))
        self._writer.execute("DELETE FROM facets_fts WHERE rowid = ?", (facet_id,))
        self._writer.execute("DELETE FROM facets WHERE id = ?", (facet_id,))
        return 1

    def _load_facets(self, facet_keys: list[str]) -> list[IndexedFacet]:
        """The stored facets with these keys, in the order given; read on the reader, under
        the reader lock. An unknown key is left out.
        """
        found: dict[str, IndexedFacet] = {}
        for chunk in _chunks(facet_keys):
            rows = self._reader.execute(
                # constant columns, and the IN list holds only `?` placeholders
                f"SELECT {_FACET_COLUMNS} FROM facets "  # nosec B608
                f"WHERE facet_key IN ({_placeholders(len(chunk))})",
                chunk,
            )
            for row in rows:
                facet = IndexedFacet(*row)
                found[facet.facet_key] = facet
        return [found[key] for key in facet_keys if key in found]

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


def _group(row: FacetMemberRow) -> _Group:
    return (row.subject_id, row.rel_type, row.object_type)


def _typed(groups: Iterable[_Group]) -> set[tuple[str, str, str]]:
    """The groups with an object type, the only ones that can form a facet."""
    return {
        (subject_id, rel_type, object_type)
        for subject_id, rel_type, object_type in groups
        if object_type
    }


def _facet_keys(groups: Iterable[_Group]) -> list[str]:
    """The facet key of each group that has an object type, sorted."""
    return sorted(facet_key(*group) for group in _typed(groups))


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


def _main_database_file(conn: sqlite3.Connection) -> str:
    """The file path of `conn`'s main database; empty for `:memory:`, which has no file."""
    for _seq, name, path in conn.execute("PRAGMA database_list"):
        if name == "main":
            return str(path or "")
    return ""


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
