"""The candidate pools of the memory read path: three fact pools and the vault note pool.

The fetcher (T9) fuses and ranks what these return; nothing here decides currency, fusion or
order beyond each pool's own rank. Pools are produced here, in the read path, never inside a
store (ADR-010 Invariant 4, last clause).

- FactCandidates searches the derived fact index three ways (FTS, vector, entity anchors) and
  returns each fact once with every indexed version, because which version is offered depends
  on the turn's `now`, which T9 applies.
- VaultCandidates queries the vault sidecar on its read-only connection in match-any mode and
  returns note Candidates, minus the notes the block must never show twice or at all.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import PurePosixPath

from backend.interfaces import SidecarIndexProtocol
from backend.vault.models import AuthoredBy

from .fact_index import FactGroup, FactIndex
from .fts import query_tokens
from .models import Candidate, NoteSource

logger = logging.getLogger(__name__)

FTS_POOL = "fts"
VECTOR_POOL = "vector"
ENTITY_POOL = "entity"
FACT_POOLS: tuple[str, ...] = (FTS_POOL, VECTOR_POOL, ENTITY_POOL)

NOTE_VECTOR_POOL = "note_vector"
NOTE_FTS_POOL = "note_fts"

DEFAULT_FACT_POOL_SIZE = 20
DEFAULT_NOTE_K = 10

# Entity anchors read the query and this many of the most recent user turns (Design point 4:
# "the last two user utterances"), so a name the user just mentioned anchors a follow-up such
# as "where is it based?".
ANCHOR_TURNS = 2

# The heading part of a file-level chunk's item id. knowledge_retriever.py stamps this value
# for a heading-less chunk and conversation_handler.py maps it back to None, so a note id built
# here names the same chunk the legacy vault path names.
FILE_HEADING = "(file)"

# The note pool asks the sidecar for this many times `k` rows, so the notes it then drops
# (excluded paths, rejected notes) rarely leave it short; it asks again with a doubled count,
# up to `_MAX_FETCH_FACTOR` times `k`, only while it is short and the sidecar had more.
_FETCH_FACTOR = 2
_MAX_FETCH_FACTOR = 8


def note_item_id(path: str, heading: str | None) -> str:
    """The `item_id` of a vault chunk: its path and heading, `(file)` for a file-level chunk.

    The id must name one chunk across turns and across the legacy vault path, which keys a
    chunk by the same path and the same heading value (`FILE_HEADING` when there is none).
    """
    return f"{path}#{heading if heading is not None else FILE_HEADING}"


@dataclass(frozen=True, slots=True)
class FactHit:
    """One fact offered by the fact pools, before T9 picks the version current at `now`.

    Attributes:
        group: The fact with every indexed version (`FactGroup.versions`), as the first pool
            that returned it read it, in pool order fts, vector, entity. Its `matched` and
            `score` are that pool's; the versions are the same in every pool unless the
            refresher committed between two searches.
        ranks: The fact's 1-based rank in each pool that returned it, keyed by pool name
            (`fts`, `vector`, `entity`); a pool that did not return it is absent.
    """

    group: FactGroup
    ranks: Mapping[str, int]

    @property
    def fact_key(self) -> str:
        """The fact's identity, which T9 uses as the Candidate's `item_id`."""
        return self.group.fact_key


class FactCandidates:
    """The three fact pools of Design point 4, searched on the fact index's reader.

    `gather` returns a mapping from `fact_key` to FactHit, in order of first appearance across
    the pools (fts, then vector, then entity). Each FactHit carries the FactGroup with every
    version and the fact's rank in each pool. A fact appears once however many of its versions
    matched, and with all its versions, because currency is T9's: it filters the versions
    against the turn's `now` and offers the fact once, by the version current then, so a fact
    whose only matching version is past still surfaces its current version.

    The searches are synchronous SQLite reads; the fetcher runs `gather` on the
    `memory-index` executor. A database failure raises MemoryIndexError, which the fetcher
    degrades on.

    Args:
        index: The fact index; searches run on its read-only connection.
        fts_limit: Facts in the FTS pool (OR-mode query over text and names).
        vector_limit: Facts in the vector pool (KNN on the query embedding).
        entity_limit: Facts in the entity-anchored pool (a subject or object name occurs in
            the query or the recent user turns).

    Raises:
        ValueError: A pool size is not positive.
    """

    def __init__(
        self,
        index: FactIndex,
        *,
        fts_limit: int = DEFAULT_FACT_POOL_SIZE,
        vector_limit: int = DEFAULT_FACT_POOL_SIZE,
        entity_limit: int = DEFAULT_FACT_POOL_SIZE,
    ) -> None:
        for name, value in (
            ("fts_limit", fts_limit),
            ("vector_limit", vector_limit),
            ("entity_limit", entity_limit),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        self._index = index
        self._fts_limit = fts_limit
        self._vector_limit = vector_limit
        self._entity_limit = entity_limit

    def gather(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        recent_user_turns: Sequence[str] = (),
    ) -> dict[str, FactHit]:
        """Search the three pools and merge them by `fact_key`.

        Args:
            query_text: The user's text for this turn.
            query_embedding: Its embedding, computed once by the fetcher and shared with the
                vault pool.
            recent_user_turns: Earlier user turns, oldest first; the last `ANCHOR_TURNS` of
                them join the query as entity-anchor texts.

        Returns:
            `fact_key` to FactHit, in order of first appearance across fts, vector, entity.

        Raises:
            MemoryIndexError: The index's database fails.
            ValueError: The embedding has the wrong dimension.
        """
        anchor_texts = [
            text
            for text in (query_text, *recent_user_turns[-ANCHOR_TURNS:])
            if text and text.strip()
        ]
        pools: tuple[tuple[str, list[FactGroup]], ...] = (
            (FTS_POOL, self._index.search_fts(query_text, limit=self._fts_limit)),
            (VECTOR_POOL, self._index.search_vector(query_embedding, limit=self._vector_limit)),
            (
                ENTITY_POOL,
                (
                    self._index.search_names(anchor_texts, limit=self._entity_limit)
                    if anchor_texts
                    else []
                ),
            ),
        )
        groups: dict[str, FactGroup] = {}
        ranks: dict[str, dict[str, int]] = {}
        for pool, found in pools:
            for rank, group in enumerate(found, start=1):
                groups.setdefault(group.fact_key, group)
                ranks.setdefault(group.fact_key, {})[pool] = rank
        return {key: FactHit(group=groups[key], ranks=ranks[key]) for key in groups}


class VaultCandidates:
    """The vault note pool: the sidecar's hybrid query, read-only and in match-any mode.

    The query runs on the sidecar's read-only WAL connection (`query_hybrid_readonly`), so a
    turn never waits behind the file watcher's reindex, and in `match="any"` so one query word
    absent from every note does not empty the lexical leg.

    Each kept chunk becomes a `note` Candidate whose ranks are the sidecar's own per-leg ranks,
    as two pools: `note_vector` (`vector_rank`) and `note_fts` (`fts_rank`), with a leg that
    missed the chunk omitted. The sidecar's single fused RRF rank is not used: ranking.py fuses
    every pool with RRF, so a note carrying one fused rank would get half the pool count a fact
    gets and could never collect the agreement bonus of two legs. The ranks are the sidecar's
    as returned, so a dropped note leaves a gap in them rather than promoting the next one.

    The lexical leg gets the query's content words only (`fts.query_tokens`, the tokens the
    fact pool's FTS query uses): under OR, a stopword such as "the" matches nearly every note
    and would hand an unrelated note the best FTS rank. The vector leg still embeds the full
    query, which the fetcher passes in.

    `item_id` is `note_item_id(path, heading)`, `(file)` standing for the heading of a
    file-level chunk, while `NoteSource.heading` stays None for such a chunk. `authored_by` is
    read from the frontmatter the sidecar stores with each chunk.

    `when`, the instant recency is measured from, is the frontmatter `last_updated`. MIST
    writes it as a date (writer.py, `now_iso[:10]`), read as midnight in the configured zone;
    a naive datetime is read in that zone and an aware one as given. Without a readable
    `last_updated`, `when` is the file mtime the sidecar stored at indexing, in UTC.

    Dropped before the pool is cut to `k`:

    - a note whose `authored_by` is `user-rejected`, the terminal state of a note the user
      deleted or marked invalid (ADR-010): it must never surface;
    - the user profile, which the handler always injects already;
    - rendered episode notes and legacy session notes whose conversation has episodes, passed
      in as path sets (Phase B fills them), because the episode itself is a candidate.

    Paths are compared exactly, after both sides are made vault-relative POSIX paths: the
    sidecar stores whatever path the file watcher or the rebuild built under the vault root,
    while a caller may pass a path relative to the root or under it. A path under neither the
    root as configured nor its absolute form is taken as already vault-relative.

    Args:
        sidecar: The vault sidecar index, initialized.
        vault_root: The vault root the sidecar's paths were built under (`VaultConfig.root`).
        tz: Zone a date-only or naive `last_updated` is read in (`MemoryConfig.tz`).
        k: Notes in the pool.

    Raises:
        ValueError: `k` is not positive.
    """

    def __init__(
        self,
        sidecar: SidecarIndexProtocol,
        *,
        vault_root: str | os.PathLike[str],
        tz: tzinfo,
        k: int = DEFAULT_NOTE_K,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self._sidecar = sidecar
        self._paths = _VaultPaths.of(vault_root)
        self._tz = tz
        self._k = k

    async def gather(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        *,
        profile_path: str | None = None,
        rendered_episode_paths: Collection[str] = frozenset(),
        legacy_note_paths: Collection[str] = frozenset(),
    ) -> list[Candidate]:
        """Query the sidecar and return up to `k` note Candidates in the sidecar's order.

        Args:
            query_text: The user's text for this turn.
            query_embedding: Its embedding, computed once by the fetcher.
            profile_path: Path of the always-injected user profile, or None when there is none.
            rendered_episode_paths: Vault paths of rendered episode notes.
            legacy_note_paths: Vault paths of legacy session notes superseded by episodes.

        Returns:
            Note Candidates, best sidecar RRF first, at most `k`.

        Raises:
            SidecarIndexError: The sidecar is not initialized or its database fails.
        """
        named = (profile_path, *rendered_episode_paths, *legacy_note_paths)
        excluded = frozenset(self._paths.relative(path) for path in named if path)
        embedding = list(query_embedding)
        lexical_text = " ".join(query_tokens(query_text))
        fetch = self._k * _FETCH_FACTOR
        while True:
            rows = await self._sidecar.query_hybrid_readonly(
                embedding, lexical_text, k=fetch, match="any"
            )
            kept = [c for c in (self._candidate(row, excluded) for row in rows) if c is not None]
            if len(kept) >= self._k or len(rows) < fetch or fetch >= self._k * _MAX_FETCH_FACTOR:
                return kept[: self._k]
            fetch *= 2

    def _candidate(self, row: Mapping[str, object], excluded: frozenset[str]) -> Candidate | None:
        """The Candidate of one sidecar row, or None when the row is dropped."""
        path = str(row.get("path") or "")
        content = str(row.get("content") or "")
        if not path or not content.strip() or self._paths.relative(path) in excluded:
            return None
        frontmatter = _frontmatter(row.get("frontmatter"))
        authored_by = _authored_by(frontmatter)
        if authored_by == AuthoredBy.USER_REJECTED.value:
            return None
        ranks: dict[str, int] = {}
        vector_rank = row.get("vector_rank")
        fts_rank = row.get("fts_rank")
        if isinstance(vector_rank, int):
            ranks[NOTE_VECTOR_POOL] = vector_rank
        if isinstance(fts_rank, int):
            ranks[NOTE_FTS_POOL] = fts_rank
        if not ranks:
            return None
        raw_heading = row.get("heading")
        heading = str(raw_heading) if raw_heading is not None else None
        return Candidate(
            kind="note",
            item_id=note_item_id(path, heading),
            text=content,
            when=_note_when(frontmatter, row.get("mtime"), self._tz),
            source=NoteSource(path=path, heading=heading, authored_by=authored_by),
            ranks=ranks,
        )


@dataclass(frozen=True, slots=True)
class _VaultPaths:
    """Turns a note path into its vault-relative POSIX form, for exact comparison.

    The configured root may be relative to the working directory (the file watcher and the
    rebuild join it as given), so a path is tried against the root as configured and against
    its absolute form.
    """

    roots: tuple[PurePosixPath, ...]

    @classmethod
    def of(cls, vault_root: str | os.PathLike[str]) -> _VaultPaths:
        configured = _posix(os.fspath(vault_root))
        absolute = _posix(os.path.abspath(os.fspath(vault_root)))
        return cls(tuple(dict.fromkeys((configured, absolute))))

    def relative(self, path: str) -> str:
        """`path` relative to the vault root, or `path` itself, normalized, when not under it."""
        posix = _posix(path)
        for root in self.roots:
            try:
                return posix.relative_to(root).as_posix()
            except ValueError:
                continue
        return posix.as_posix()


def _posix(path: str) -> PurePosixPath:
    """`path` with Windows separators turned to `/` and `.` and `..` segments resolved lexically."""
    return PurePosixPath(posixpath.normpath(path.replace("\\", "/")))


def _frontmatter(raw: object) -> Mapping[str, object] | None:
    """A chunk's stored frontmatter as a mapping, or None when absent or unreadable.

    The sidecar stores it as a JSON string (`frontmatter_json`). Malformed JSON or a non-object
    document reads as no frontmatter: the note is still a candidate, with no author and its
    mtime as its time.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("vault candidate frontmatter is not valid JSON; ignoring it")
            return None
    return raw if isinstance(raw, Mapping) else None


def _authored_by(frontmatter: Mapping[str, object] | None) -> str | None:
    """The frontmatter's `authored_by` string, or None when absent or not a string.

    A note without one is ranked with the default confidence of a MIST-written note.
    """
    if frontmatter is None:
        return None
    value = frontmatter.get("authored_by")
    return value if isinstance(value, str) else None


def _note_when(
    frontmatter: Mapping[str, object] | None, mtime: object, tz: tzinfo
) -> datetime | None:
    """The instant a note's recency is measured from.

    The frontmatter `last_updated` first: a date or a naive datetime is read in `tz` (MIST
    writes a date, which names the user's day, not UTC's), an aware datetime as given. The
    sidecar serializes YAML dates and datetimes with `str`, so a datetime may arrive with a
    space before the time, which `fromisoformat` accepts. Without a readable `last_updated`,
    the stored mtime (integer seconds) as a UTC instant; None when that is absent too.
    """
    raw = frontmatter.get("last_updated") if frontmatter is not None else None
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.strip())
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=tz)
    if isinstance(mtime, int | float) and not isinstance(mtime, bool):
        return datetime.fromtimestamp(mtime, UTC)
    return None
