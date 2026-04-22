"""Vault sidecar index for semantic and lexical retrieval over vault notes.

Implements `SidecarIndexProtocol` from `backend.interfaces`. Backed by a
SQLite database with two virtual tables:

- `vault_chunks_fts`: FTS5 full-text index over note content and
  heading context (Crosley pattern -- frontmatter fields concatenated
  into the heading_context column with a lower BM25 weight).
- `vault_chunks_vec`: sqlite-vec `vec0` table for embedding-based
  nearest-neighbour retrieval.

Two-tier chunking per file:
- File-level chunk (heading=NULL): the full note body, truncated to
  `config.chunk_max_chars`. Provides coarse recall.
- Heading-block chunks (one per `##` heading): the block of content
  from the heading line to the next sibling-or-shallower heading or EOF.
  Provides pinpoint retrieval. The `## Provenance` section is skipped
  (boilerplate per ADR-010).

Retrieval methods:
- `query_vector`: embedding nearest-neighbour via vec0.
- `query_fts`: BM25 lexical search via FTS5.
- `query_hybrid`: Reciprocal Rank Fusion (RRF) merge of both.

This module is sync. Async callers (filewatcher, ConversationHandler)
wrap calls via `asyncio.get_event_loop().run_in_executor`.
"""

import json
import logging
import re
import sqlite3
import struct
import time
from pathlib import Path

import sqlite_vec

from backend.errors import SidecarIndexError
from backend.interfaces import EmbeddingProvider
from backend.knowledge.config import SidecarIndexConfig

logger = logging.getLogger(__name__)

# FTS5 special characters that require quoting the query to avoid parse errors.
_FTS5_SPECIAL = re.compile(r'["\*(:\-]')

# Heading regex: one or more '#' followed by a space and heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# The Provenance section is boilerplate; skip it during chunking.
_PROVENANCE_HEADING = "Provenance"


def _quote_fts5(text: str) -> str:
    """Wrap text in double-quotes for literal FTS5 matching.

    Applied when the query contains FTS5 special characters. SQLite FTS5
    treats `"..."` as a phrase query, which is safe for arbitrary text.

    Args:
        text: Raw user-supplied query text.

    Returns:
        Query string safe for use in a MATCH expression.
    """
    if _FTS5_SPECIAL.search(text):
        escaped = text.replace('"', '""')
        return f'"{escaped}"'
    return text


def _extract_heading_blocks(content: str) -> list[tuple[str, int, str]]:
    """Split note body into heading-block chunks.

    Each block spans from the heading line to the next heading of the same
    or shallower level (lower `#` count) or EOF.  The `## Provenance`
    section is skipped.

    Args:
        content: Full note body text.

    Returns:
        List of `(heading_text, heading_level, block_content)` tuples.
        `block_content` does NOT include the heading line itself.
    """
    matches = list(_HEADING_RE.finditer(content))
    if not matches:
        return []

    blocks: list[tuple[str, int, str]] = []
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        heading_text = match.group(2).strip()

        if heading_text == _PROVENANCE_HEADING:
            continue

        # Block ends at the next heading of same or shallower depth, or EOF.
        block_start = match.end()
        block_end = len(content)
        for next_match in matches[idx + 1 :]:
            next_level = len(next_match.group(1))
            if next_level <= level:
                block_end = next_match.start()
                break

        block_text = content[block_start:block_end].strip()
        blocks.append((heading_text, level, block_text))

    return blocks


def _heading_context(
    heading: str | None,
    note_type: str | None,
    frontmatter: dict | None,
) -> str:
    """Build the heading_context string for BM25 scoring.

    Concatenates heading text, note_type, and top-3 frontmatter tags so that
    structured-field queries (e.g. "mist-session memory") score correctly
    on the heading_context column (weight 0.3) without polluting the main
    content column (weight 1.0).

    Args:
        heading: Heading text, or None for file-level chunks.
        note_type: Value of frontmatter `type` field, or None.
        frontmatter: Raw frontmatter dict, may be None.

    Returns:
        Space-joined context string. Empty string if all inputs are absent.
    """
    parts: list[str] = []
    if heading:
        parts.append(heading)
    if note_type:
        parts.append(note_type)
    if frontmatter:
        tags = frontmatter.get("tags", [])
        if isinstance(tags, list):
            parts.extend(str(t) for t in tags[:3])
    return " ".join(parts)


class VaultSidecarIndex:
    """SQLite-backed two-tier sidecar index for vault note retrieval.

    Lifecycle::

        idx = VaultSidecarIndex(config, embedding_provider)
        idx.initialize()          # creates / opens DB, emits DDL
        idx.upsert_file(...)      # index a note
        results = idx.query_hybrid(embedding, text, k=5)
        idx.close()               # release connection

    All public methods are synchronous. Call from async code via
    `loop.run_in_executor(None, idx.method, *args)`.

    Implements `SidecarIndexProtocol` from `backend.interfaces`.
    """

    def __init__(
        self,
        config: SidecarIndexConfig,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """Store config and provider; do NOT open DB here.

        Args:
            config: Sidecar index configuration (db_path, dim, weights, etc.).
            embedding_provider: Satisfies `EmbeddingProvider` protocol.
        """
        self.config = config
        self._embeddings = embedding_provider
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open or create the SQLite database and emit schema DDL.

        Creates the parent directory if missing. Loads the sqlite-vec
        extension. If `config.rebuild_on_startup` is True, all four
        schema objects are dropped before re-creation.

        Idempotent: safe to call multiple times.

        Raises:
            SidecarIndexError: If the database cannot be opened or the
                schema cannot be created.
        """
        try:
            db_path = Path(self.config.db_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)

            if self.config.rebuild_on_startup:
                logger.info("rebuild_on_startup=True -- dropping sidecar schema before recreate")
                self._drop_schema(conn)

            self._create_schema(conn)
            self._conn = conn
            logger.info("VaultSidecarIndex initialized at %s", db_path)
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError) as exc:
            raise SidecarIndexError(f"Failed to initialize sidecar index: {exc}") from exc

    def close(self) -> None:
        """Close the database connection. Idempotent."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.OperationalError:
                pass
            finally:
                self._conn = None

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert_file(
        self,
        path: str,
        content: str,
        mtime: int,
        frontmatter: dict | None = None,
    ) -> int:
        """Index or re-index all chunks for a vault note.

        Produces one file-level chunk (heading=NULL) and one chunk per
        `##...` heading block. The `## Provenance` section is skipped.
        Each chunk is embedded, upserted into `vault_chunks`, then
        mirrored to `vault_chunks_vec` and `vault_chunks_fts`.

        The entire operation runs inside a single transaction; failure
        rolls back all writes for this file.

        Args:
            path: Vault-relative or absolute path to the note file.
                  Used as the join key across all three tables.
            content: Full note body (after frontmatter has been stripped).
            mtime: Unix timestamp (integer seconds) of the file's mtime.
            frontmatter: Parsed frontmatter dict, or None.

        Returns:
            Number of chunks written (file-level + heading-block chunks).

        Raises:
            SidecarIndexError: On any database or embedding error.
        """
        self._require_connection()
        note_type = frontmatter.get("type") if frontmatter else None
        # PyYAML auto-converts ISO dates and datetimes to Python date/datetime
        # objects; default=str stringifies them to ISO-8601 so json.dumps can
        # serialize the frontmatter without round-trip loss. Other non-JSON
        # types fall back to repr() via str() too.
        frontmatter_json = json.dumps(frontmatter, default=str) if frontmatter else None
        dim = self.config.embedding_dimension
        now = int(time.time())

        # Build chunk list: (heading, heading_level, chunk_content)
        chunks: list[tuple[str | None, int | None, str]] = []

        # File-level chunk
        file_content = content[: self.config.chunk_max_chars]
        chunks.append((None, None, file_content))

        # Heading-block chunks
        for heading_text, heading_level, block_content in _extract_heading_blocks(content):
            block_content = block_content[: self.config.chunk_max_chars]
            chunks.append((heading_text, heading_level, block_content))

        try:
            with self._conn:  # type: ignore[union-attr]
                written = 0
                for heading, heading_level, chunk_content in chunks:
                    ctx = _heading_context(heading, note_type, frontmatter)
                    embedding = self._embeddings.generate_embedding(chunk_content)
                    packed = struct.pack(f"<{dim}f", *embedding)

                    chunk_id = self._upsert_chunk_row(
                        path=path,
                        heading=heading,
                        heading_level=heading_level,
                        content=chunk_content,
                        mtime=mtime,
                        frontmatter_json=frontmatter_json,
                        note_type=note_type,
                        created_at=now,
                    )

                    # Mirror to vec table: delete old entry, insert new.
                    self._conn.execute(  # type: ignore[union-attr]
                        "DELETE FROM vault_chunks_vec WHERE rowid = ?", (chunk_id,)
                    )
                    self._conn.execute(  # type: ignore[union-attr]
                        "INSERT INTO vault_chunks_vec(rowid, embedding) VALUES (?, ?)",
                        (chunk_id, packed),
                    )

                    # Mirror to FTS table: delete old, insert new.
                    self._conn.execute(  # type: ignore[union-attr]
                        "DELETE FROM vault_chunks_fts WHERE rowid = ?", (chunk_id,)
                    )
                    self._conn.execute(  # type: ignore[union-attr]
                        "INSERT INTO vault_chunks_fts(rowid, content, heading_context) VALUES (?, ?, ?)",
                        (chunk_id, chunk_content, ctx),
                    )
                    written += 1

        except (
            sqlite3.OperationalError,
            sqlite3.IntegrityError,
            struct.error,
            ValueError,
        ) as exc:
            raise SidecarIndexError(f"upsert_file failed for '{path}': {exc}") from exc

        return written

    def delete_path(self, path: str) -> int:
        """Remove all chunks for a vault note from all three tables.

        Args:
            path: Exact path value used when the chunks were indexed.

        Returns:
            Number of chunks deleted.

        Raises:
            SidecarIndexError: On database error.
        """
        self._require_connection()
        try:
            with self._conn:  # type: ignore[union-attr]
                rows = self._conn.execute(  # type: ignore[union-attr]
                    "SELECT chunk_id FROM vault_chunks WHERE path = ?", (path,)
                ).fetchall()
                count = len(rows)
                for row in rows:
                    cid = row[0]
                    self._conn.execute(  # type: ignore[union-attr]
                        "DELETE FROM vault_chunks_vec WHERE rowid = ?", (cid,)
                    )
                    self._conn.execute(  # type: ignore[union-attr]
                        "DELETE FROM vault_chunks_fts WHERE rowid = ?", (cid,)
                    )
                self._conn.execute(  # type: ignore[union-attr]
                    "DELETE FROM vault_chunks WHERE path = ?", (path,)
                )
            return count
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as exc:
            raise SidecarIndexError(f"delete_path failed for '{path}': {exc}") from exc

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query_vector(self, embedding: list[float], k: int = 10) -> list[dict]:
        """Return top-k chunks by embedding similarity.

        Args:
            embedding: Query embedding (must match `config.embedding_dimension`).
            k: Maximum results to return.

        Returns:
            List of result dicts, each containing:
            ``{path, heading, content, frontmatter, note_type, score, distance, source}``.
            ``score`` is ``1.0 / (1.0 + distance)``. Empty list on empty index.

        Raises:
            SidecarIndexError: On database error.
        """
        self._require_connection()
        dim = self.config.embedding_dimension
        try:
            packed = struct.pack(f"<{dim}f", *embedding)
        except struct.error as exc:
            raise SidecarIndexError(f"Failed to pack embedding: {exc}") from exc

        try:
            rows = self._conn.execute(  # type: ignore[union-attr]
                """
                SELECT vc.path, vc.heading, vc.content, vc.frontmatter_json,
                       vc.note_type, distance
                FROM vault_chunks_vec
                JOIN vault_chunks vc ON vc.chunk_id = vault_chunks_vec.rowid
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (packed, k),
            ).fetchall()
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as exc:
            raise SidecarIndexError(f"query_vector failed: {exc}") from exc

        return [
            {
                "path": row[0],
                "heading": row[1],
                "content": row[2],
                "frontmatter": row[3],
                "note_type": row[4],
                "score": 1.0 / (1.0 + row[5]),
                "distance": row[5],
                "source": "vector",
            }
            for row in rows
        ]

    def query_fts(self, text: str, k: int = 10) -> list[dict]:
        """Return top-k chunks by BM25 lexical relevance.

        FTS5 returns smaller (more-negative) BM25 scores for better matches.
        Results are returned with ``score = -bm25(...)`` so that higher is
        better, consistent with `query_vector`.

        Special FTS5 characters in `text` are phrase-quoted to avoid parse
        errors. If the query produces an FTS5 error despite quoting, an empty
        list is returned and a warning is logged.

        Args:
            text: User-supplied search query.
            k: Maximum results to return.

        Returns:
            List of result dicts with the same shape as `query_vector`,
            with ``source="fts"`` and ``distance=None``.

        Raises:
            SidecarIndexError: On unexpected database errors.
        """
        self._require_connection()
        safe_text = _quote_fts5(text)
        weight = self.config.heading_context_weight

        try:
            rows = self._conn.execute(  # type: ignore[union-attr]
                """
                SELECT vc.path, vc.heading, vc.content, vc.frontmatter_json,
                       vc.note_type,
                       -bm25(vault_chunks_fts, 1.0, ?) as score
                FROM vault_chunks_fts
                JOIN vault_chunks vc ON vc.chunk_id = vault_chunks_fts.rowid
                WHERE vault_chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
                """,
                (weight, safe_text, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            # FTS5 parse errors surface as OperationalError. Return empty list
            # so retrieval degrades gracefully when a query is malformed.
            logger.warning("query_fts: FTS5 error for query %r: %s", text, exc)
            return []
        except sqlite3.IntegrityError as exc:
            raise SidecarIndexError(f"query_fts failed: {exc}") from exc

        return [
            {
                "path": row[0],
                "heading": row[1],
                "content": row[2],
                "frontmatter": row[3],
                "note_type": row[4],
                "score": row[5],
                "distance": None,
                "source": "fts",
            }
            for row in rows
        ]

    def query_hybrid(
        self,
        embedding: list[float],
        text: str,
        k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Return top-k chunks via Reciprocal Rank Fusion of vector + FTS.

        Runs both retrievers with ``k * 2`` candidates (overshoot for RRF
        quality), merges by summing per-result RRF contributions, and
        returns the top-k by combined RRF score.

        RRF score per rank: ``1.0 / (rrf_k + rank)`` where rank is 1-based.

        Args:
            embedding: Query embedding for the vector retriever.
            text: Query text for the FTS retriever.
            k: Final top-k to return.
            rrf_k: RRF constant (larger = flatter ranking curve).

        Returns:
            List of result dicts with ``score=rrf_combined``,
            ``vector_rank``, ``fts_rank`` (None if absent from that list),
            and ``sources`` list. Empty if both retrievers return nothing.
        """
        overshoot = k * 2
        vector_results = self.query_vector(embedding, k=overshoot)
        fts_results = self.query_fts(text, k=overshoot)

        # Keyed by (path, heading) -- heading may be None for file-level chunks.
        merged: dict[tuple[str, str | None], dict] = {}

        for rank, result in enumerate(vector_results, start=1):
            key = (result["path"], result["heading"])
            rrf = 1.0 / (rrf_k + rank)
            if key not in merged:
                merged[key] = {
                    "path": result["path"],
                    "heading": result["heading"],
                    "content": result["content"],
                    "frontmatter": result["frontmatter"],
                    "note_type": result["note_type"],
                    "score": 0.0,
                    "vector_rank": rank,
                    "fts_rank": None,
                    "sources": [],
                }
            merged[key]["score"] += rrf
            if "vector" not in merged[key]["sources"]:
                merged[key]["sources"].append("vector")

        for rank, result in enumerate(fts_results, start=1):
            key = (result["path"], result["heading"])
            rrf = 1.0 / (rrf_k + rank)
            if key not in merged:
                merged[key] = {
                    "path": result["path"],
                    "heading": result["heading"],
                    "content": result["content"],
                    "frontmatter": result["frontmatter"],
                    "note_type": result["note_type"],
                    "score": 0.0,
                    "vector_rank": None,
                    "fts_rank": rank,
                    "sources": [],
                }
            else:
                merged[key]["fts_rank"] = rank
            merged[key]["score"] += rrf
            if "fts" not in merged[key]["sources"]:
                merged[key]["sources"].append("fts")

        sorted_results = sorted(merged.values(), key=lambda r: r["score"], reverse=True)
        return sorted_results[:k]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def chunk_count(self) -> int:
        """Return the total number of indexed chunks.

        Returns:
            Row count of `vault_chunks`.

        Raises:
            SidecarIndexError: On database error.
        """
        self._require_connection()
        try:
            row = self._conn.execute(  # type: ignore[union-attr]
                "SELECT COUNT(*) FROM vault_chunks"
            ).fetchone()
            return row[0]
        except (sqlite3.OperationalError, sqlite3.IntegrityError) as exc:
            raise SidecarIndexError(f"chunk_count failed: {exc}") from exc

    def health_check(self) -> bool:
        """Return True if the connection is open and the schema is intact.

        Returns:
            True on healthy state, False on any error or closed connection.
        """
        if self._conn is None:
            return False
        try:
            tables = {
                row[0]
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
                ).fetchall()
            }
            required = {
                "vault_chunks",
                "vault_chunks_fts",
                "vault_chunks_vec",
                "idx_vault_chunks_path",
            }
            return required.issubset(tables)
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_connection(self) -> None:
        """Raise SidecarIndexError if the connection is not open."""
        if self._conn is None:
            raise SidecarIndexError(
                "VaultSidecarIndex is not initialized. Call initialize() first."
            )

    def _drop_schema(self, conn: sqlite3.Connection) -> None:
        """Drop all four schema objects if they exist."""
        conn.execute("DROP TABLE IF EXISTS vault_chunks_fts")
        conn.execute("DROP TABLE IF EXISTS vault_chunks_vec")
        conn.execute("DROP INDEX IF EXISTS idx_vault_chunks_path")
        conn.execute("DROP TABLE IF EXISTS vault_chunks")

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Emit all four DDL statements. Idempotent via IF NOT EXISTS."""
        dim = self.config.embedding_dimension

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vault_chunks (
                chunk_id    INTEGER PRIMARY KEY,
                path        TEXT    NOT NULL,
                heading     TEXT,
                heading_level INTEGER,
                content     TEXT    NOT NULL,
                mtime       INTEGER NOT NULL,
                frontmatter_json TEXT,
                note_type   TEXT,
                created_at  INTEGER NOT NULL
            )
            """
        )
        # Partial unique indexes handle SQLite's NULL != NULL UNIQUE behavior:
        # - One file-level chunk per path (heading IS NULL).
        # - One heading chunk per (path, heading) pair (heading IS NOT NULL).
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_vault_chunks_path_null_heading
            ON vault_chunks(path)
            WHERE heading IS NULL
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_vault_chunks_path_heading
            ON vault_chunks(path, heading)
            WHERE heading IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS vault_chunks_fts USING fts5(
                content,
                heading_context,
                tokenize='porter unicode61'
            )
            """
        )
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vault_chunks_vec USING vec0(
                embedding float[{dim}]
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_vault_chunks_path
            ON vault_chunks(path)
            """
        )

    def _upsert_chunk_row(
        self,
        *,
        path: str,
        heading: str | None,
        heading_level: int | None,
        content: str,
        mtime: int,
        frontmatter_json: str | None,
        note_type: str | None,
        created_at: int,
    ) -> int:
        """Insert or update a row in `vault_chunks` and return its chunk_id.

        Uses conditional ON CONFLICT clauses to handle the two distinct
        uniqueness constraints (NULL heading vs. non-NULL heading).

        Args:
            path: Note path.
            heading: Heading text, or None for file-level chunks.
            heading_level: Heading depth (1-6), or None.
            content: Chunk text.
            mtime: File modification timestamp.
            frontmatter_json: Serialised frontmatter, or None.
            note_type: Value of frontmatter `type`, or None.
            created_at: Insertion timestamp.

        Returns:
            The `chunk_id` of the inserted or updated row.
        """
        conn = self._conn  # type: ignore[union-attr]

        if heading is None:
            # File-level chunk: conflict on the partial index (path WHERE heading IS NULL).
            conn.execute(
                """
                INSERT INTO vault_chunks
                    (path, heading, heading_level, content, mtime,
                     frontmatter_json, note_type, created_at)
                VALUES (?, NULL, NULL, ?, ?, ?, ?, ?)
                ON CONFLICT(path) WHERE heading IS NULL DO UPDATE SET
                    content          = excluded.content,
                    mtime            = excluded.mtime,
                    frontmatter_json = excluded.frontmatter_json,
                    note_type        = excluded.note_type
                """,
                (path, content, mtime, frontmatter_json, note_type, created_at),
            )
            row = conn.execute(
                "SELECT chunk_id FROM vault_chunks WHERE path = ? AND heading IS NULL",
                (path,),
            ).fetchone()
        else:
            # Heading-block chunk: conflict on partial index (path, heading WHERE heading IS NOT NULL).
            conn.execute(
                """
                INSERT INTO vault_chunks
                    (path, heading, heading_level, content, mtime,
                     frontmatter_json, note_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path, heading) WHERE heading IS NOT NULL DO UPDATE SET
                    heading_level    = excluded.heading_level,
                    content          = excluded.content,
                    mtime            = excluded.mtime,
                    frontmatter_json = excluded.frontmatter_json,
                    note_type        = excluded.note_type
                """,
                (
                    path,
                    heading,
                    heading_level,
                    content,
                    mtime,
                    frontmatter_json,
                    note_type,
                    created_at,
                ),
            )
            row = conn.execute(
                "SELECT chunk_id FROM vault_chunks WHERE path = ? AND heading = ?",
                (path, heading),
            ).fetchone()

        return row[0]
