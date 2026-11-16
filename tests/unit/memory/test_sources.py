"""Tests for backend.memory.sources: the fact pools and the vault note pool the fetcher fuses.

Both sources run against real stores on tmp files (a FactIndex over `memory_index.db` and a
VaultSidecarIndex), embedded with the deterministic FakeEmbedder.
"""

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.knowledge.config import SidecarIndexConfig
from backend.memory.fact_index import FactIndex
from backend.memory.index_db import open_index, open_reader
from backend.memory.models import FactRow, NoteSource, fact_key
from backend.memory.sources import (
    DEFAULT_FACT_POOL_SIZE,
    ENTITY_POOL,
    FACT_POOLS,
    FTS_POOL,
    NOTE_FTS_POOL,
    NOTE_VECTOR_POOL,
    VECTOR_POOL,
    FactCandidates,
    FactHit,
    VaultCandidates,
    _authored_by,
    _frontmatter,
    _note_when,
    _VaultPaths,
    note_item_id,
)
from backend.vault.models import AuthoredBy
from backend.vault.sidecar_index import VaultSidecarIndex
from tests.fakes.memory_fakes import FakeEmbedder

_MODEL = "all-MiniLM-L6-v2"


def _fact(
    subject_name: str,
    rel_type: str,
    object_name: str,
    *,
    row_key: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    recorded_at: str = "2026-01-01T00:00:00+00:00",
) -> FactRow:
    subject_id = subject_name.lower()
    object_id = object_name.lower()
    key = fact_key(subject_id, rel_type, object_id)
    return FactRow(
        row_key=row_key or f"rk-{subject_id}-{rel_type}-{object_id}",
        fact_key=key,
        subject_id=subject_id,
        subject_name=subject_name,
        subject_type="User" if subject_id == "user" else "Person",
        rel_type=rel_type,
        object_id=object_id,
        object_name=object_name,
        object_type="Organization",
        valid_from=valid_from,
        valid_to=valid_to,
        recorded_at=recorded_at,
        confidence=0.9,
        source_utterance_id="utt-1",
    )


@pytest.fixture
def fact_index(tmp_path: Path, fake_embedder: FakeEmbedder) -> Iterator[FactIndex]:
    """A FactIndex over a tmp `memory_index.db`, writer and reader both open."""
    db_path = tmp_path / "memory_index.db"
    writer = open_index(db_path, embedding_model_id=_MODEL)
    reader = open_reader(db_path, embedding_model_id=_MODEL)
    try:
        yield FactIndex(writer, fake_embedder, tz=UTC, reader=reader)
    finally:
        reader.close()
        writer.close()


def _pool(hits: dict[str, FactHit], pool: str) -> list[str]:
    """Fact keys of one pool, in rank order."""
    ranked = [(hit.ranks[pool], key) for key, hit in hits.items() if pool in hit.ranks]
    return [key for _, key in sorted(ranked)]


# -- FactCandidates ------------------------------------------------------------------------------


def test_the_three_pools_hold_twenty_facts_each_by_default(
    fact_index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    members = [f"Member {chr(97 + i) * 2}" for i in range(25)]
    fact_index.upsert([_fact(name, "WORKS_AT", "Kestrel") for name in members])
    query = "who works at kestrel"

    hits = FactCandidates(fact_index).gather(query, fake_embedder.generate_embedding(query))

    assert DEFAULT_FACT_POOL_SIZE == 20
    assert FACT_POOLS == (FTS_POOL, VECTOR_POOL, ENTITY_POOL) == ("fts", "vector", "entity")
    for pool in FACT_POOLS:
        ranked = [hit.ranks[pool] for hit in hits.values() if pool in hit.ranks]
        assert sorted(ranked) == list(range(1, 21)), pool


def test_each_fact_carries_its_rank_in_every_pool_that_returned_it(
    fact_index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    acme = _fact("User", "WORKS_AT", "Acme")
    lisbon = _fact("User", "LIVES_IN", "Lisbon")
    bob = _fact("User", "KNOWS", "Bob")
    fact_index.upsert([acme, lisbon, bob])
    query = "where do I work, acme?"
    turns = ("I chatted with Bob", "I moved to Lisbon last year", "it rained today")

    hits = FactCandidates(fact_index).gather(
        query, fake_embedder.generate_embedding(query), recent_user_turns=turns
    )

    assert hits[acme.fact_key].ranks[FTS_POOL] == 1
    assert ENTITY_POOL in hits[acme.fact_key].ranks
    assert FTS_POOL not in hits[lisbon.fact_key].ranks
    assert ENTITY_POOL in hits[lisbon.fact_key].ranks
    # Bob is named only three turns back: anchors use the query and the last two turns.
    assert ENTITY_POOL not in hits[bob.fact_key].ranks
    assert all(VECTOR_POOL in hit.ranks for hit in hits.values())
    assert set(_pool(hits, VECTOR_POOL)) == {acme.fact_key, lisbon.fact_key, bob.fact_key}


def test_a_hit_is_keyed_by_fact_key_and_carries_every_version(
    fact_index: FactIndex, fake_embedder: FakeEmbedder
) -> None:
    past = _fact(
        "User",
        "WORKS_AT",
        "Acme",
        row_key="rk-acme-past",
        valid_from="2020-01-01T00:00:00+00:00",
        valid_to="2023-01-01T00:00:00+00:00",
        recorded_at="2020-01-01T00:00:00+00:00",
    )
    now = _fact(
        "User",
        "WORKS_AT",
        "Acme",
        row_key="rk-acme-now",
        valid_from="2024-01-01T00:00:00+00:00",
        recorded_at="2024-01-01T00:00:00+00:00",
    )
    fact_index.upsert([past, now])
    query = "acme"

    hits = FactCandidates(fact_index).gather(query, fake_embedder.generate_embedding(query))

    assert list(hits) == [past.fact_key]
    hit = hits[past.fact_key]
    assert hit.fact_key == past.fact_key
    assert [v.row_key for v in hit.group.versions] == ["rk-acme-past", "rk-acme-now"]


def test_pool_sizes_are_configurable(fact_index: FactIndex, fake_embedder: FakeEmbedder) -> None:
    fact_index.upsert([_fact(f"Member {c * 2}", "WORKS_AT", "Kestrel") for c in "abcde"])
    query = "kestrel"

    hits = FactCandidates(fact_index, fts_limit=1, vector_limit=2, entity_limit=3).gather(
        query, fake_embedder.generate_embedding(query)
    )

    assert len(_pool(hits, FTS_POOL)) == 1
    assert len(_pool(hits, VECTOR_POOL)) == 2
    assert len(_pool(hits, ENTITY_POOL)) == 3


@pytest.mark.parametrize("field", ["fts_limit", "vector_limit", "entity_limit"])
def test_a_non_positive_pool_size_is_refused(fact_index: FactIndex, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        FactCandidates(fact_index, **{field: 0})


# -- VaultCandidates -----------------------------------------------------------------------------

_QUERY = "kestrel parser rewrite banana"
_VAULT_ROOT = "/vault"
_TZ = timezone(timedelta(hours=-7))
_GARDEN_MTIME = 1_760_000_000


@pytest.fixture
def sidecar(tmp_path: Path, fake_embedder: FakeEmbedder) -> Iterator[VaultSidecarIndex]:
    """A tmp VaultSidecarIndex seeded with one note per exclusion rule plus two keepers.

    Paths are stored vault-relative here; `test_absolute_stored_paths_are_excluded_too` covers
    paths stored under the root.
    """
    config = SidecarIndexConfig(db_path=str(tmp_path / "sidecar.db"))
    idx = VaultSidecarIndex(config, fake_embedder)
    idx.initialize()
    idx.upsert_file(
        "projects/kestrel.md",
        "Kestrel parser notes.\n\n## Stack\nThe kestrel parser rewrite uses rust.",
        1,
        {"type": "note", "authored_by": AuthoredBy.USER.value, "last_updated": date(2026, 10, 20)},
    )
    idx.upsert_file(
        "notes/garden.md", "Tomatoes and basil grow well.", _GARDEN_MTIME, {"authored_by": "mist"}
    )
    idx.upsert_file("users/user.md", "Profile: works on the kestrel parser.", 3)
    idx.upsert_file("episodes/2026-10-22-e1.md", "We discussed the kestrel parser.", 4)
    idx.upsert_file("sessions/2026-10-01-legacy.md", "Legacy kestrel parser session.", 5)
    yield idx
    idx.close()


def _rejected(idx: VaultSidecarIndex) -> None:
    idx.upsert_file(
        "notes/rejected.md",
        _QUERY,
        6,
        {"authored_by": AuthoredBy.USER_REJECTED.value},
    )


def _source(idx: VaultSidecarIndex, *, k: int = 10) -> VaultCandidates:
    return VaultCandidates(idx, vault_root=_VAULT_ROOT, tz=_TZ, k=k)


async def _gather(
    idx: VaultSidecarIndex,
    embedder: FakeEmbedder,
    *,
    k: int = 10,
    query: str = _QUERY,
    **kwargs: object,
) -> list:
    return await _source(idx, k=k).gather(query, embedder.generate_embedding(query), **kwargs)


def _by_path(candidates: list) -> dict[str, object]:
    """Candidates by note path; each path appears once, so the mapping loses nothing."""
    by_path = {c.source.path: c for c in candidates}
    assert len(by_path) == len(candidates)
    return by_path


@pytest.mark.asyncio
async def test_notes_become_note_candidates_with_path_heading_and_author(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    rows = await sidecar.query_hybrid_readonly(
        fake_embedder.generate_embedding(_QUERY), _QUERY, k=20, match="any"
    )
    first = next(r for r in rows if r["path"] == "projects/kestrel.md")

    candidates = await _gather(sidecar, fake_embedder)

    kestrel = _by_path(candidates)["projects/kestrel.md"]
    assert kestrel.kind == "note"
    # The note's first chunk in the sidecar's order stands for it.
    assert kestrel.item_id == note_item_id(first["path"], first["heading"])
    assert kestrel.source == NoteSource("projects/kestrel.md", first["heading"], "user")
    assert kestrel.text == first["content"]
    assert kestrel.view_text is None
    assert _by_path(candidates)["notes/garden.md"].source.authored_by == "mist"
    assert note_item_id("projects/kestrel.md", None) == "projects/kestrel.md#(file)"


@pytest.mark.asyncio
async def test_note_ranks_are_the_best_sidecar_leg_ranks_of_the_path_as_two_pools(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    rows = await sidecar.query_hybrid_readonly(
        fake_embedder.generate_embedding(_QUERY), _QUERY, k=20, match="any"
    )
    expected: dict[str, dict[str, int]] = {}
    for row in rows:
        best = expected.setdefault(row["path"], {})
        for pool, rank in (
            (NOTE_VECTOR_POOL, row["vector_rank"]),
            (NOTE_FTS_POOL, row["fts_rank"]),
        ):
            if rank is not None:
                best[pool] = min(best.get(pool, rank), rank)

    candidates = await _gather(sidecar, fake_embedder)

    assert candidates
    for candidate in candidates:
        assert dict(candidate.ranks) == expected[candidate.source.path]


@pytest.mark.asyncio
async def test_a_leg_that_missed_is_omitted_and_match_any_keeps_the_lexical_hit(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    candidates = await _gather(sidecar, fake_embedder)

    by_path = _by_path(candidates)
    # "banana" is in no note, yet the kestrel note keeps its FTS rank (match="any").
    assert NOTE_FTS_POOL in by_path["projects/kestrel.md"].ranks
    # The garden note shares no query word: only the vector leg found it.
    assert by_path["notes/garden.md"].ranks.keys() == {NOTE_VECTOR_POOL}


class _ScriptedSidecar:
    """Returns fixed rows, best fused rank first, cut at `k`, and records each `k` asked for."""

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.asked: list[int] = []

    async def query_hybrid_readonly(
        self, embedding: list[float], text: str, *, k: int, match: str
    ) -> list[dict[str, object]]:
        self.asked.append(k)
        return self.rows[:k]


def _chunk(
    path: str, heading: str | None, vector_rank: int | None, fts_rank: int | None
) -> dict[str, object]:
    return {
        "path": path,
        "heading": heading,
        "content": f"{path} {heading} content",
        "frontmatter": None,
        "mtime": 1,
        "vector_rank": vector_rank,
        "fts_rank": fts_rank,
    }


@pytest.mark.asyncio
async def test_a_note_returned_as_three_chunks_is_one_candidate_with_the_best_rank_per_pool(
    fake_embedder: FakeEmbedder,
) -> None:
    sidecar = _ScriptedSidecar(
        [
            _chunk("notes/bread.md", "Method", None, 4),
            _chunk("notes/other.md", None, 2, None),
            _chunk("notes/bread.md", "Flour", 5, 1),
            _chunk("notes/bread.md", None, 3, 6),
        ]
    )

    candidates = await VaultCandidates(sidecar, vault_root=_VAULT_ROOT, tz=_TZ, k=10).gather(
        _QUERY, fake_embedder.generate_embedding(_QUERY)
    )

    assert [c.source.path for c in candidates] == ["notes/bread.md", "notes/other.md"]
    bread = candidates[0]
    assert dict(bread.ranks) == {NOTE_VECTOR_POOL: 3, NOTE_FTS_POOL: 1}
    # Its text, id and heading are the first chunk's; the note view is N0, so no view text.
    assert bread.item_id == "notes/bread.md#Method"
    assert bread.source.heading == "Method"
    assert bread.text == "notes/bread.md Method content"
    assert bread.view_text is None


@pytest.mark.asyncio
async def test_the_fetch_counts_paths_toward_k(fake_embedder: FakeEmbedder) -> None:
    rows = [_chunk("notes/bread.md", f"Part {n}", n, None) for n in range(1, 5)]
    rows += [_chunk("notes/other.md", None, 5, None), _chunk("notes/third.md", None, 6, None)]
    sidecar = _ScriptedSidecar(rows)

    candidates = await VaultCandidates(sidecar, vault_root=_VAULT_ROOT, tz=_TZ, k=2).gather(
        _QUERY, fake_embedder.generate_embedding(_QUERY)
    )

    # Four chunks of one note fill the first fetch of 2k rows with one path, so it asks again.
    assert sidecar.asked == [4, 8]
    assert [c.source.path for c in candidates] == ["notes/bread.md", "notes/other.md"]


_EXCLUDED = {"users/user.md", "episodes/2026-10-22-e1.md", "sessions/2026-10-01-legacy.md"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "excluded",
    [
        pytest.param(
            {
                "profile_path": "users/user.md",
                "rendered_episode_paths": frozenset({"episodes/2026-10-22-e1.md"}),
                "legacy_note_paths": frozenset({"./sessions/2026-10-01-legacy.md"}),
            },
            id="relative",
        ),
        pytest.param(
            {
                "profile_path": "/vault/users/user.md",
                "rendered_episode_paths": ["/vault/episodes/2026-10-22-e1.md"],
                "legacy_note_paths": {"\\vault\\sessions\\2026-10-01-legacy.md"},
            },
            id="absolute",
        ),
    ],
)
async def test_the_profile_rendered_episode_and_legacy_notes_are_excluded(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder, excluded: dict
) -> None:
    before = await _gather(sidecar, fake_embedder)
    assert {c.source.path for c in before} >= _EXCLUDED

    after = await _gather(sidecar, fake_embedder, **excluded)

    assert {c.source.path for c in after} == {"projects/kestrel.md", "notes/garden.md"}


@pytest.mark.asyncio
async def test_absolute_stored_paths_are_excluded_too(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    sidecar.upsert_file("/vault/episodes/2026-10-23-e2.md", "Kestrel parser follow-up.", 7)

    after = await _gather(
        sidecar, fake_embedder, rendered_episode_paths={"episodes/2026-10-23-e2.md"}
    )

    assert "/vault/episodes/2026-10-23-e2.md" not in {c.source.path for c in after}


@pytest.mark.asyncio
async def test_only_the_named_note_is_excluded_not_one_sharing_its_tail(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    sidecar.upsert_file("/elsewhere/users/user.md", "Another kestrel parser user note.", 8)

    after = await _gather(
        sidecar, fake_embedder, profile_path="/vault/users/user.md", legacy_note_paths={"user.md"}
    )

    paths = {c.source.path for c in after}
    assert "users/user.md" not in paths
    assert "/elsewhere/users/user.md" in paths


def test_paths_under_a_relative_vault_root_compare_vault_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    paths = _VaultPaths.of("mist-memory")

    assert paths.relative("mist-memory/users/user.md") == "users/user.md"
    assert paths.relative(str(tmp_path / "mist-memory" / "users" / "user.md")) == "users/user.md"
    assert paths.relative("users/user.md") == "users/user.md"
    assert paths.relative("/elsewhere/users/user.md") == "/elsewhere/users/user.md"


@pytest.mark.asyncio
async def test_a_user_rejected_note_is_dropped_even_as_the_top_match(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    _rejected(sidecar)
    rows = await sidecar.query_hybrid_readonly(
        fake_embedder.generate_embedding(_QUERY), _QUERY, k=10, match="any"
    )
    assert rows[0]["path"] == "notes/rejected.md"

    candidates = await _gather(sidecar, fake_embedder)

    assert "notes/rejected.md" not in {c.source.path for c in candidates}
    assert candidates


@pytest.mark.asyncio
async def test_k_caps_the_pool_and_exclusions_are_refilled(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    _rejected(sidecar)
    excluded = {
        "profile_path": "users/user.md",
        "rendered_episode_paths": {"episodes/2026-10-22-e1.md"},
        "legacy_note_paths": {"sessions/2026-10-01-legacy.md"},
    }

    candidates = await _gather(sidecar, fake_embedder, k=2, **excluded)

    assert [c.source.path for c in candidates].count("projects/kestrel.md") == 1
    assert {c.source.path for c in candidates} == {"projects/kestrel.md", "notes/garden.md"}
    one = await _gather(sidecar, fake_embedder, k=1, **excluded)
    assert len(one) == 1


@pytest.mark.asyncio
async def test_the_pool_reads_through_the_readonly_query_in_match_any_mode(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    seen: list[tuple] = []
    readonly = sidecar.query_hybrid_readonly

    async def recording(*args, **kwargs):
        calls.append(kwargs)
        seen.append(args)
        return await readonly(*args, **kwargs)

    def refuse(*args, **kwargs):
        raise AssertionError("the read path must not queue on the writer")

    monkeypatch.setattr(sidecar, "query_hybrid_readonly", recording)
    monkeypatch.setattr(sidecar, "query_hybrid", refuse)
    query = "Where is the kestrel parser?"

    await _gather(sidecar, fake_embedder, query=query)

    assert calls and all(call["match"] == "any" for call in calls)
    # The vector leg embeds the full query; the lexical leg gets its content words only.
    assert all(args == (fake_embedder.generate_embedding(query), "kestrel parser") for args in seen)


@pytest.mark.asyncio
async def test_stopwords_do_not_give_an_unrelated_note_a_lexical_rank(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    sidecar.upsert_file("notes/weather.md", "The weather was the best of the year.", 9)

    candidates = await _gather(sidecar, fake_embedder, query="the kestrel !")

    by_path = _by_path(candidates)
    assert NOTE_FTS_POOL not in by_path["notes/weather.md"].ranks
    assert NOTE_FTS_POOL in by_path["projects/kestrel.md"].ranks


@pytest.mark.asyncio
async def test_a_note_dates_from_last_updated_and_falls_back_to_its_mtime(
    sidecar: VaultSidecarIndex, fake_embedder: FakeEmbedder
) -> None:
    sidecar.upsert_file(
        "notes/parser-log.md",
        "Kestrel parser log.",
        10,
        {"authored_by": "mist", "last_updated": datetime(2026, 10, 21, 9, 30)},
    )
    rows = sidecar.query_hybrid(fake_embedder.generate_embedding(_QUERY), _QUERY, k=20)
    assert {r["path"]: r["mtime"] for r in rows}["notes/garden.md"] == _GARDEN_MTIME

    candidates = await _gather(sidecar, fake_embedder)

    by_path = _by_path(candidates)
    # A date-only `last_updated` (what MIST writes) names a local day, kept as its midnight so
    # recency scoring is unchanged, and flagged so the block never invents a part of the day.
    dated = by_path["projects/kestrel.md"]
    assert dated.when == datetime(2026, 10, 20, tzinfo=_TZ)
    assert dated.when_is_day is True
    # A datetime `last_updated` and the mtime are instants.
    timed = by_path["notes/parser-log.md"]
    assert timed.when == datetime(2026, 10, 21, 9, 30, tzinfo=_TZ)
    assert timed.when_is_day is False
    from_mtime = by_path["notes/garden.md"]
    assert from_mtime.when == datetime.fromtimestamp(_GARDEN_MTIME, UTC)
    assert from_mtime.when_is_day is False


_MTIME_WHEN = datetime.fromtimestamp(1_700_000_000, UTC)


@pytest.mark.parametrize(
    ("last_updated", "expected", "is_day"),
    [
        pytest.param("2026-10-20", datetime(2026, 10, 20, tzinfo=_TZ), True, id="date-in-zone"),
        pytest.param(
            " 2026-10-20 ", datetime(2026, 10, 20, tzinfo=_TZ), True, id="date-with-spaces"
        ),
        pytest.param(
            "2026-10-20T00:00:00",
            datetime(2026, 10, 20, tzinfo=_TZ),
            False,
            id="midnight-datetime-is-an-instant",
        ),
        pytest.param(
            "2026-10-20T09:30:00",
            datetime(2026, 10, 20, 9, 30, tzinfo=_TZ),
            False,
            id="naive-in-zone",
        ),
        pytest.param(
            "2026-10-20 09:30:00+02:00",
            datetime(2026, 10, 20, 9, 30, tzinfo=timezone(timedelta(hours=2))),
            False,
            id="aware-as-given",
        ),
        pytest.param("last week", _MTIME_WHEN, False, id="not-a-date"),
        pytest.param(None, _MTIME_WHEN, False, id="absent"),
        pytest.param(20261020, _MTIME_WHEN, False, id="not-a-string"),
    ],
)
def test_note_when_reads_each_form_of_last_updated(
    last_updated: object, expected: datetime, is_day: bool
) -> None:
    frontmatter = {} if last_updated is None else {"last_updated": last_updated}

    when, when_is_day = _note_when(frontmatter, 1_700_000_000, _TZ)

    assert when == expected
    assert when is not None and when.utcoffset() == expected.utcoffset()
    assert when_is_day is is_day


def test_note_when_is_none_without_last_updated_or_mtime() -> None:
    assert _note_when(None, None, _TZ) == (None, False)


def test_a_non_positive_k_is_refused(sidecar: VaultSidecarIndex) -> None:
    with pytest.raises(ValueError, match="k"):
        _source(sidecar, k=0)


@pytest.mark.asyncio
async def test_the_note_pool_reads_whole_note_texts_in_one_sidecar_read(
    sidecar: VaultSidecarIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[tuple[str, ...]] = []
    read = sidecar.path_texts_readonly

    async def recording(paths):  # type: ignore[no-untyped-def]
        asked.append(tuple(paths))
        return await read(paths)

    monkeypatch.setattr(sidecar, "path_texts_readonly", recording)

    texts = await _source(sidecar).texts(["projects/kestrel.md", "notes/garden.md"])

    assert asked == [("projects/kestrel.md", "notes/garden.md")]
    assert set(texts) == {"projects/kestrel.md", "notes/garden.md"}
    # Every chunk of the path: the file-level chunk and the `## Stack` block.
    assert "uses rust" in texts["projects/kestrel.md"]
    assert texts["notes/garden.md"] == "Tomatoes and basil grow well."


def test_malformed_frontmatter_leaves_the_author_unknown() -> None:
    assert _authored_by(_frontmatter('{"authored_by": "user-edit"}')) == "user-edit"
    assert _frontmatter("{not json") is None
    assert _frontmatter('["a list"]') is None
    assert _authored_by(_frontmatter(None)) is None
    assert _authored_by(_frontmatter('{"authored_by": ["user"]}')) is None
