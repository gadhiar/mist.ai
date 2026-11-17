"""The correction log and read-time suppression (mist-read-path T21, Design point 12).

`CorrectionService` records a forget or a correct as an append-only `memory_corrections` row in
a real in-memory EventStore; `LoggedSuppression` reads the log into the suppression set the
fetcher applies before the cross-encoder. The fetch tests run a real MemoryFetcher over a real
FactIndex on a tmp database with the shared fakes; the note pool is a scripted double whose
whole-note texts stand in for the sidecar's `path_texts_readonly`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Collection, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.errors import MemoryCorrectionError, MistError, SidecarIndexError
from backend.event_store.store import EventStore
from backend.knowledge.models import graph_ref, vault_ref
from backend.memory.block import MemoryBlockBuilder
from backend.memory.config import MEMORY_ENV_VARS, MemoryConfig
from backend.memory.corrections import (
    CorrectedFact,
    CorrectionRecord,
    CorrectionService,
    LoggedSuppression,
    Suppression,
)
from backend.memory.fact_index import FactIndex, IndexedFact
from backend.memory.fetcher import FetcherExecutors, MemoryFetcher
from backend.memory.index_db import open_index, open_reader
from backend.memory.models import (
    Candidate,
    FactRow,
    MemoryCandidate,
    MemoryRequest,
    MemoryResult,
    NoteSource,
    fact_key,
)
from backend.memory.sources import NOTE_FTS_POOL, NOTE_VECTOR_POOL, note_item_id
from tests.fakes.memory_fakes import FakeClock, FakeEmbedder, FakeReranker, FakeTokenCounter

_MODEL = "all-MiniLM-L6-v2"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NOW = datetime(2026, 10, 22, 18, 0, tzinfo=UTC)  # FakeClock's default start
_CORRECTED_AT = datetime(2026, 3, 10, 21, 4, 59, tzinfo=UTC)
_BEFORE = "2026-01-01T00:00:00+00:00"
_AFTER = "2026-04-01T00:00:00+00:00"
_ROOMY_MS = 5000
_ID = re.compile(r"^corr-[0-9a-f]{32}$")

_KILN = CorrectedFact(
    subject_id="wren",
    subject_name="Wren",
    relation="MEMBER_OF",
    object_id="kiln-street-pottery",
    object_name="Kiln Street Pottery",
)
_KILN_KEY = fact_key("wren", "MEMBER_OF", "kiln-street-pottery")


# -- doubles and builders ---------------------------------------------------------------------


def _store() -> EventStore:
    store = EventStore(db_path=":memory:")
    store.initialize()
    return store


def _service(store: EventStore, now: datetime = _CORRECTED_AT) -> CorrectionService:
    return CorrectionService(store, now_fn=lambda: now)


def _fact(
    object_name: str,
    *,
    object_id: str | None = None,
    rel_type: str = "MEMBER_OF",
    recorded_at: str = _BEFORE,
    utterance: str | None = None,
) -> FactRow:
    object_id = object_id or object_name.lower().replace(" ", "-")
    return FactRow(
        row_key=f"rk-{rel_type}-{object_id}-{recorded_at}",
        fact_key=fact_key("wren", rel_type, object_id),
        subject_id="wren",
        subject_name="Wren",
        subject_type="User",
        rel_type=rel_type,
        object_id=object_id,
        object_name=object_name,
        object_type="Organization",
        valid_from=None,
        valid_to=None,
        recorded_at=recorded_at,
        confidence=0.9,
        source_utterance_id=utterance or f"utt-{object_id}",
    )


def _indexed(row: FactRow) -> IndexedFact:
    return IndexedFact(
        row_key=row.row_key,
        fact_key=row.fact_key,
        subject_id=row.subject_id,
        subject_name=row.subject_name,
        subject_type=row.subject_type,
        rel_type=row.rel_type,
        object_id=row.object_id,
        object_name=row.object_name,
        object_type=row.object_type,
        text=f"{row.subject_name} {row.rel_type} {row.object_name}",
        valid_from=row.valid_from,
        valid_to=row.valid_to,
        recorded_at=row.recorded_at,
        confidence=row.confidence,
        source_utterance_id=row.source_utterance_id,
        content_hash=hashlib.sha256(row.row_key.encode()).hexdigest(),
    )


def _record(
    *,
    recorded_at: datetime = _CORRECTED_AT,
    source_event_id: str | None = None,
    fact: CorrectedFact = _KILN,
    key: str = _KILN_KEY,
) -> CorrectionRecord:
    return CorrectionRecord(
        correction_id="corr-" + "0" * 32,
        recorded_at=recorded_at,
        action="forget",
        fact_key=key,
        subject_id=fact.subject_id,
        subject_name=fact.subject_name,
        relation=fact.relation,
        object_id=fact.object_id,
        object_name=fact.object_name,
        replacement_object_name=None,
        surface="tool",
        session_id="s1",
        conversation_id="c1",
        turn_index=3,
        source_event_id=source_event_id,
        origin="real",
    )


def _note(
    path: str, chunk: str, *, when: datetime = datetime(2026, 2, 26, tzinfo=UTC)
) -> Candidate:
    return Candidate(
        kind="note",
        item_id=note_item_id(path, "Thursdays"),
        text=chunk,
        when=when,
        source=NoteSource(path=path, heading="Thursdays", authored_by="user"),
        ranks={NOTE_VECTOR_POOL: 1, NOTE_FTS_POOL: 1},
    )


class ScriptedNotes:
    """A note pool returning fixed candidates, with whole-note texts by path, reads recorded."""

    def __init__(
        self,
        candidates: list[Candidate],
        texts: dict[str, str],
        *,
        on_texts: object | None = None,
    ) -> None:
        self.candidates = candidates
        self.whole = texts
        self.reads: list[tuple[str, ...]] = []
        self.on_texts = on_texts

    async def gather(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        *,
        profile_path: str | None = None,
    ) -> list[Candidate]:
        return list(self.candidates)

    async def texts(self, paths: Collection[str]) -> dict[str, str]:
        self.reads.append(tuple(paths))
        if callable(self.on_texts):
            self.on_texts()
        return {path: self.whole[path] for path in paths if path in self.whole}


class CountingSuppression:
    """A suppression source that counts its reads."""

    def __init__(self, inner: LoggedSuppression) -> None:
        self.inner = inner
        self.reads = 0

    def suppression(self) -> Suppression:
        self.reads += 1
        return self.inner.suppression()


class RecordingReranker(FakeReranker):
    """The fake reranker, keeping every text it scored; a text holding `favour` scores 1."""

    def __init__(self, favour: str | None = None) -> None:
        super().__init__()
        self.favour = favour

    def score(
        self, query: str, texts: Sequence[str], budget_ms: float | None = None
    ) -> list[float]:
        scores = super().score(query, texts, budget_ms)
        if self.favour is None:
            return scores
        return [1.0 if self.favour in text else min(s, 0.5) for s, text in zip(scores, texts)]

    @property
    def texts(self) -> list[str]:
        return [text for _, texts, _ in self.calls for text in texts]


@pytest.fixture
def fact_index(tmp_path: Path) -> Iterator[FactIndex]:
    db = tmp_path / "memory_index.db"
    writer = open_index(db, embedding_model_id=_MODEL)
    reader = open_reader(db, embedding_model_id=_MODEL)
    yield FactIndex(writer, FakeEmbedder(), tz=UTC, reader=reader)
    reader.close()
    writer.close()


@pytest.fixture
def executors() -> Iterator[FetcherExecutors]:
    pools = (
        ThreadPoolExecutor(2, thread_name_prefix="memory-index"),
        ThreadPoolExecutor(1, thread_name_prefix="memory-model"),
        ThreadPoolExecutor(1, thread_name_prefix="memory-gate"),
    )
    yield FetcherExecutors(index=pools[0], model=pools[1], gate=pools[2])
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


def _config(**overrides: object) -> MemoryConfig:
    values: dict[str, object] = {
        "rerank_top_n": 20,
        "select_k": 8,
        "block_tokens": 300,
        "deadline_ms": _ROOMY_MS,
        "facets": False,
    }
    values.update(overrides)
    return MemoryConfig(**values)  # type: ignore[arg-type]


def _fetcher(
    fact_index: FactIndex,
    executors: FetcherExecutors,
    suppression: object,
    *,
    config: MemoryConfig | None = None,
    notes: ScriptedNotes | None = None,
    reranker: FakeReranker | None = None,
    clock: FakeClock | None = None,
) -> MemoryFetcher:
    config = config or _config()
    return MemoryFetcher(
        index=fact_index,
        embedder=FakeEmbedder(),
        reranker=reranker or FakeReranker(),
        block_builder=MemoryBlockBuilder(
            counter=FakeTokenCounter(), config=config, placement="system_tail"
        ),
        config=config,
        executors=executors,
        need_gate=None,
        need_mode="on",
        need_context=False,
        clock=clock or FakeClock(),
        index_state=lambda: None,
        notes=notes,
        suppression=suppression,  # type: ignore[arg-type]
    )


def _request(text: str, *, turns: tuple[str, ...] = ()) -> MemoryRequest:
    return MemoryRequest(
        session_id="s1",
        conversation_id="c1",
        query_text=text,
        now=_NOW,
        recent_user_turns=turns,
        turn_index=1,
    )


def _candidate(result: MemoryResult, ref: str) -> MemoryCandidate:
    [found] = [c for c in result.stage.candidates if c.ref == ref]
    return found


def _ref(row: FactRow) -> str:
    return graph_ref(row.subject_id, row.rel_type, row.object_id)


def _served(result: MemoryResult) -> list[str]:
    return [item.item_id for item in result.items]


# -- recording --------------------------------------------------------------------------------


def test_a_forget_is_logged_with_the_injected_clock_and_returned_as_its_record() -> None:
    store = _store()
    stamped = datetime(2001, 2, 3, 4, 5, 6, 789, tzinfo=UTC)  # far from any wall clock

    record = _service(store, stamped).forget(
        _KILN_KEY, "ws", fact=_KILN, session_id="s1", conversation_id="c1", turn_index=4
    )

    assert _ID.match(record.correction_id)
    assert record.recorded_at == stamped
    assert (record.action, record.fact_key, record.replacement_object_name) == (
        "forget",
        _KILN_KEY,
        None,
    )
    assert (record.surface, record.session_id, record.conversation_id) == ("ws", "s1", "c1")
    assert (record.turn_index, record.source_event_id, record.origin) == (4, None, "real")
    [row] = store.list_memory_corrections()
    assert row["recorded_at"] == "2001-02-03T04:05:06.000789+00:00"
    assert CorrectionRecord.from_row(row) == record


def test_a_correct_logs_its_replacement() -> None:
    store = _store()

    record = _service(store).correct(_KILN_KEY, "Harbor Clay Studio", "cli", fact=_KILN)

    assert (record.action, record.replacement_object_name) == ("correct", "Harbor Clay Studio")
    [row] = store.list_memory_corrections()
    assert (row["action"], row["replacement_object_name"]) == ("correct", "Harbor Clay Studio")
    assert (row["object_name"], row["relation"], row["subject_name"]) == (
        "Kiln Street Pottery",
        "MEMBER_OF",
        "Wren",
    )


def test_an_explicit_recorded_at_is_stamped_in_utc_with_its_issuing_turn() -> None:
    store = _store()
    turn_time = datetime(2026, 3, 10, 22, 4, 59, tzinfo=timezone(timedelta(hours=1)))

    def wall_clock() -> datetime:
        raise AssertionError("an explicit recorded_at never reads the clock")

    record = CorrectionService(store, now_fn=wall_clock).forget(
        _KILN_KEY,
        "tool",
        fact=_KILN,
        recorded_at=turn_time,
        source_event_id="evt-0198",
        turn_index=3,
    )

    assert record.recorded_at == _CORRECTED_AT and record.recorded_at.tzinfo == UTC
    assert record.source_event_id == "evt-0198"
    [row] = store.list_memory_corrections()
    assert row["recorded_at"] == "2026-03-10T21:04:59.000000+00:00"
    assert row["source_event_id"] == "evt-0198"


def test_two_records_made_at_one_instant_list_in_id_order() -> None:
    store = _store()
    service = _service(store)

    first = service.forget(_KILN_KEY, "ws", fact=_KILN)
    second = service.forget(_KILN_KEY, "ws", fact=_KILN)

    assert [row["correction_id"] for row in store.list_memory_corrections()] == sorted(
        [first.correction_id, second.correction_id]
    )


def test_a_given_correction_id_is_kept_and_a_malformed_one_refused() -> None:
    store = _store()
    service = _service(store)

    record = service.forget(
        _KILN_KEY, "cli", fact=_KILN, correction_id="corr-e6e8012550785f874dd0e003736f0b48"
    )

    assert record.correction_id == "corr-e6e8012550785f874dd0e003736f0b48"
    with pytest.raises(MemoryCorrectionError, match="correction_id"):
        service.forget(_KILN_KEY, "cli", fact=_KILN, correction_id="corr-1")


@pytest.mark.parametrize(
    ("call", "fragment"),
    [
        (lambda s: s.forget(_KILN_KEY, "email", fact=_KILN), "surface"),
        (lambda s: s.forget("", "ws", fact=_KILN), "item_id"),
        (lambda s: s.correct(_KILN_KEY, "  ", "ws", fact=_KILN), "replacement"),
        (
            lambda s: s.forget(_KILN_KEY, "ws", fact=_KILN, recorded_at=datetime(2026, 3, 10)),
            "timezone",
        ),
    ],
)
def test_a_correction_that_cannot_be_recorded_is_refused_and_not_logged(
    call: object, fragment: str
) -> None:
    store = _store()

    with pytest.raises(MemoryCorrectionError, match=fragment):
        call(_service(store))  # type: ignore[operator]

    assert store.list_memory_corrections() == []


def test_a_store_failure_is_a_memory_correction_error() -> None:
    store = _store()
    store.close()  # the next connection opens a fresh in-memory database with no tables

    with pytest.raises(MemoryCorrectionError):
        _service(store).forget(_KILN_KEY, "ws", fact=_KILN)


# -- recorded listeners -----------------------------------------------------------------------


def test_listeners_get_each_record_after_its_row_is_appended() -> None:
    store = _store()
    service = _service(store)
    seen: list[tuple[str, list[str]]] = []

    def listener(record: CorrectionRecord) -> None:
        logged = [row["correction_id"] for row in store.list_memory_corrections()]
        seen.append((record.correction_id, logged))

    service.add_recorded_listener(listener)
    record = service.forget(_KILN_KEY, "ws", fact=_KILN)

    assert seen == [(record.correction_id, [record.correction_id])]


def test_a_listener_raising_a_mist_error_is_logged_and_the_others_still_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _service(_store())
    calls: list[str] = []

    def failing(record: CorrectionRecord) -> None:
        calls.append("failing")
        raise MistError("push queue full")

    service.add_recorded_listener(failing)
    service.add_recorded_listener(lambda record: calls.append("after"))

    with caplog.at_level(logging.ERROR, logger="backend.memory.corrections"):
        service.forget(_KILN_KEY, "ws", fact=_KILN)

    assert calls == ["failing", "after"]
    assert "push queue full" in caplog.text


def test_a_listener_raising_anything_else_propagates_after_the_row_is_logged() -> None:
    store = _store()
    service = _service(store)

    def broken(record: CorrectionRecord) -> None:
        raise TypeError("a listener of the wrong shape")

    service.add_recorded_listener(broken)

    with pytest.raises(TypeError):
        service.forget(_KILN_KEY, "ws", fact=_KILN)
    assert len(store.list_memory_corrections()) == 1


# -- the suppression set ----------------------------------------------------------------------


def test_versions_of_the_key_recorded_at_or_before_the_correction_are_suppressed() -> None:
    suppression = Suppression.of([_record()])

    assert suppression.suppresses_fact(_indexed(_fact("Kiln Street Pottery")))
    at = _fact("Kiln Street Pottery", recorded_at=_CORRECTED_AT.isoformat())
    assert suppression.suppresses_fact(_indexed(at))
    # Said again after the correction: a later version of the same fact is served.
    assert not suppression.suppresses_fact(
        _indexed(_fact("Kiln Street Pottery", recorded_at=_AFTER))
    )
    assert not suppression.suppresses_fact(_indexed(_fact("Harbor Rowing Club")))


def test_a_later_correction_of_the_same_key_moves_the_line() -> None:
    later = _record(recorded_at=datetime(2026, 5, 1, tzinfo=UTC))

    suppression = Suppression.of([_record(), later])

    assert suppression.suppresses_fact(_indexed(_fact("Kiln Street Pottery", recorded_at=_AFTER)))


def test_the_issuing_turn_cannot_reassert_the_same_fact_and_its_other_facts_stay() -> None:
    suppression = Suppression.of([_record(source_event_id="evt-9")])
    # The same triple under another key, from the turn that said "forget that", recorded later.
    same_triple = _fact(
        "kiln street pottery", object_id="kiln-st-pottery", recorded_at=_AFTER, utterance="evt-9"
    )
    other = _fact("Harbor Rowing Club", recorded_at=_AFTER, utterance="evt-9")
    elsewhere = _fact(
        "Kiln Street Pottery", object_id="kiln-st-pottery", recorded_at=_AFTER, utterance="evt-2"
    )

    assert suppression.suppresses_fact(_indexed(same_triple))
    assert not suppression.suppresses_fact(_indexed(other))
    assert not suppression.suppresses_fact(_indexed(elsewhere))


def test_a_note_is_suppressed_by_the_object_name_as_a_whole_word_when_recorded_earlier() -> None:
    suppression = Suppression.of([_record()])
    earlier = datetime(2026, 2, 26, tzinfo=UTC)

    assert suppression.narrates
    assert suppression.suppresses_note("Signed up at kiln street\npottery's wheel.", earlier)
    assert suppression.suppresses_note("Kiln Street Pottery, Thursdays.", _CORRECTED_AT)
    assert suppression.suppresses_note("Kiln Street Pottery, Thursdays.", None)
    assert not suppression.suppresses_note("Kiln Street Pottery, Thursdays.", _NOW)
    assert not suppression.suppresses_note("Kiln Street Potteryworks, Thursdays.", earlier)
    assert not suppression.suppresses_note("Wheel course on Thursdays.", earlier)
    assert not Suppression().narrates


def test_a_non_ascii_object_name_matches_its_note_in_any_case() -> None:
    # Case folding maps "ß" to "ss", so the name and the text must be folded the same way.
    market = _record(
        fact=CorrectedFact("wren", "Wren", "SHOPS_AT", "grossmarkt-sued", "Großmarkt Süd")
    )
    suppression = Suppression.of([market])
    earlier = datetime(2026, 2, 26, tzinfo=UTC)

    assert suppression.suppresses_note("Bought peppers at the Großmarkt Süd.", earlier)
    assert suppression.suppresses_note("GROSSMARKT SÜD, Saturdays.", earlier)
    assert not suppression.suppresses_note("Großmarkt Südwest, Saturdays.", earlier)


def test_the_log_is_read_into_the_suppression_set_with_the_origins_filter() -> None:
    store = _store()
    service = _service(store)
    service.forget(_KILN_KEY, "ws", fact=_KILN)
    service.forget(
        fact_key("wren", "MEMBER_OF", "harbor-rowing-club"),
        "ws",
        fact=CorrectedFact("wren", "Wren", "MEMBER_OF", "harbor-rowing-club", "Harbor Rowing Club"),
        origin="test",
    )
    rowing = _indexed(_fact("Harbor Rowing Club"))

    assert LoggedSuppression(store).suppression().suppresses_fact(rowing)
    real = LoggedSuppression(store, origins=("real",)).suppression()
    assert real.suppresses_fact(_indexed(_fact("Kiln Street Pottery")))
    assert not real.suppresses_fact(rowing)


# -- suppression in the fetch -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_forgotten_fact_leaves_before_the_cross_encoder_traced_with_its_ref(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    kiln, rowing = _fact("Kiln Street Pottery"), _fact("Harbor Rowing Club")
    fact_index.upsert([kiln, rowing])
    store = _store()
    _service(store).forget(_KILN_KEY, "ws", fact=_KILN)
    reranker = RecordingReranker()
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(store), reranker=reranker)

    result = await fetcher.fetch(_request("kiln street pottery harbor rowing club"))

    assert _served(result) == [rowing.fact_key]
    dropped = _candidate(result, _ref(kiln))
    assert (dropped.disposition, dropped.item, dropped.ce_rank) == (
        "dropped_suppressed",
        None,
        None,
    )
    assert (result.suppressed, result.narrative_suppressed) == (1, 0)
    assert result.stage.suppressed == 1
    assert not [text for text in reranker.texts if "Kiln" in text]


@pytest.mark.asyncio
async def test_another_fact_the_issuing_turn_asserted_is_still_served(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    said = _CORRECTED_AT.isoformat()
    kiln = _fact("Kiln Street Pottery", recorded_at=said, utterance="evt-9")
    restated = _fact(
        "Kiln Street Pottery", object_id="kiln-st-pottery", recorded_at=_AFTER, utterance="evt-9"
    )
    rowing = _fact("Harbor Rowing Club", recorded_at=said, utterance="evt-9")
    fact_index.upsert([kiln, restated, rowing])
    store = _store()
    _service(store).forget(_KILN_KEY, "tool", fact=_KILN, source_event_id="evt-9")
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(store))

    result = await fetcher.fetch(_request("kiln street pottery harbor rowing club"))

    assert _served(result) == [rowing.fact_key]
    assert _candidate(result, _ref(kiln)).disposition == "dropped_suppressed"
    assert _candidate(result, _ref(restated)).disposition == "dropped_suppressed"
    assert result.suppressed == 2


@pytest.mark.asyncio
async def test_a_note_is_judged_on_its_whole_text_and_leaves_before_the_cross_encoder(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    rowing = _fact("Harbor Rowing Club")
    fact_index.upsert([rowing])
    store = _store()
    _service(store).forget(_KILN_KEY, "tool", fact=_KILN)
    # The pool returned a chunk without the name; another chunk of the same note holds it.
    pottery = _note("hobbies/pottery-class.md", "Thursday evenings, six-week wheel course.")
    garden = _note("notes/garden.md", "Tomatoes and basil, Thursday evenings.")
    notes = ScriptedNotes(
        [pottery, garden],
        {
            "hobbies/pottery-class.md": "Thursday evenings, six-week wheel course.\n\n"
            "Kiln Street Pottery, made two lopsided bowls.",
            "notes/garden.md": "Tomatoes and basil, Thursday evenings.",
        },
    )
    # Unsuppressed, the pottery chunk would take the one `select_k` slot.
    reranker = RecordingReranker(favour="six-week")
    fetcher = _fetcher(
        fact_index,
        executors,
        LoggedSuppression(store),
        notes=notes,
        reranker=reranker,
        config=_config(select_k=1),
    )

    result = await fetcher.fetch(_request("thursday evenings harbor rowing club"))

    # One read of the whole texts for the fetch, every shortlisted note's path in it.
    assert [sorted(read) for read in notes.reads] == [
        ["hobbies/pottery-class.md", "notes/garden.md"]
    ]
    ref = vault_ref("hobbies/pottery-class.md", "Thursdays", pottery.text)
    dropped = _candidate(result, ref)
    assert (dropped.disposition, dropped.source, dropped.ce_rank) == (
        "dropped_suppressed",
        "vault",
        None,
    )
    assert (result.suppressed, result.narrative_suppressed) == (0, 1)
    assert result.stage.narrative_suppressed == 1
    # It took no rerank pair, and the one `select_k` slot went to a candidate still standing.
    assert pottery.text not in reranker.texts
    assert garden.text in reranker.texts
    assert len(result.items) == 1 and result.items[0].item_id != pottery.item_id
    # The control: with nothing corrected, the same fetch serves the pottery note first.
    control = _fetcher(
        fact_index,
        executors,
        LoggedSuppression(_store()),
        notes=notes,
        reranker=RecordingReranker(favour="six-week"),
        config=_config(select_k=1),
    )
    assert _served(await control.fetch(_request("thursday evenings harbor rowing club"))) == [
        pottery.item_id
    ]


@pytest.mark.asyncio
async def test_a_note_recorded_after_the_correction_is_served(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    store = _store()
    _service(store).forget(_KILN_KEY, "tool", fact=_KILN)
    later = _note("hobbies/pottery-class.md", "Kiln Street Pottery again.", when=_NOW)
    notes = ScriptedNotes([later], {"hobbies/pottery-class.md": "Kiln Street Pottery again."})
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(store), notes=notes)

    result = await fetcher.fetch(_request("kiln street pottery"))

    assert _served(result) == [later.item_id]
    assert result.narrative_suppressed == 0


@pytest.mark.asyncio
async def test_no_whole_text_read_without_a_correction(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    pottery = _note("hobbies/pottery-class.md", "Kiln Street Pottery.")
    notes = ScriptedNotes([pottery], {"hobbies/pottery-class.md": "Kiln Street Pottery."})
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(_store()), notes=notes)

    result = await fetcher.fetch(_request("kiln street pottery"))

    assert notes.reads == []
    assert _served(result) == [pottery.item_id]


@pytest.mark.asyncio
async def test_a_late_whole_text_read_drops_every_note_and_marks_the_fetch(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    rowing = _fact("Harbor Rowing Club")
    fact_index.upsert([rowing])
    store = _store()
    _service(store).forget(_KILN_KEY, "tool", fact=_KILN)
    clock = FakeClock()
    pottery = _note("hobbies/pottery-class.md", "Thursday evenings.")
    notes = ScriptedNotes(
        [pottery],
        {"hobbies/pottery-class.md": "Thursday evenings. Kiln Street Pottery."},
        on_texts=lambda: clock.advance(ms=_ROOMY_MS + 1),
    )
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(store), notes=notes, clock=clock)

    result = await fetcher.fetch(_request("thursday evenings harbor rowing club"))

    assert result.degraded_reason == "deadline"
    ref = vault_ref("hobbies/pottery-class.md", "Thursdays", pottery.text)
    assert _candidate(result, ref).disposition == "dropped_late"
    assert pottery.item_id not in _served(result)


@pytest.mark.asyncio
async def test_a_failed_whole_text_read_serves_no_note(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    store = _store()
    _service(store).forget(_KILN_KEY, "tool", fact=_KILN)
    pottery = _note("hobbies/pottery-class.md", "Thursday evenings.")

    def fail() -> None:
        raise SidecarIndexError("sidecar gone")

    notes = ScriptedNotes([pottery], {}, on_texts=fail)
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(store), notes=notes)

    result = await fetcher.fetch(_request("thursday evenings"))

    ref = vault_ref("hobbies/pottery-class.md", "Thursdays", pottery.text)
    assert _candidate(result, ref).disposition == "dropped_late"
    assert result.items == ()


@pytest.mark.asyncio
async def test_with_corrections_disabled_nothing_is_suppressed_or_read(
    fact_index: FactIndex, executors: FetcherExecutors
) -> None:
    kiln = _fact("Kiln Street Pottery")
    fact_index.upsert([kiln])
    store = _store()
    _service(store).forget(_KILN_KEY, "ws", fact=_KILN)
    source = CountingSuppression(LoggedSuppression(store))
    fetcher = _fetcher(fact_index, executors, source, config=_config(corrections_enabled=False))

    result = await fetcher.fetch(_request("kiln street pottery"))

    assert _served(result) == [kiln.fact_key]
    assert source.reads == 0 and result.suppressed == 0


# -- invalidation -----------------------------------------------------------------------------


async def _settle() -> None:
    names = {"memory-prefetch-s1", "memory-stage1-s1"}
    pending = [task for task in asyncio.all_tasks() if task.get_name() in names]
    await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize("adapter", [True, False], ids=["adapter", "bare-method"])
@pytest.mark.asyncio
async def test_the_recorded_listener_adapter_drops_a_stage1_cached_before_the_correction(
    fact_index: FactIndex, executors: FetcherExecutors, adapter: bool
) -> None:
    kiln = _fact("Kiln Street Pottery")
    fact_index.upsert([kiln])
    store = _store()
    service = _service(store)
    fetcher = _fetcher(fact_index, executors, LoggedSuppression(store))
    if adapter:
        service.add_recorded_listener(lambda record: fetcher.invalidate())
    else:
        # The negative control: the bare method takes the record as `session_id`.
        service.add_recorded_listener(fetcher.invalidate)  # type: ignore[arg-type]
    await fetcher.fetch(_request("hello there"))
    await fetcher.prefetch("s1")  # speech start: stage 1, with no correction yet, is cached
    await _settle()

    service.forget(_KILN_KEY, "ws", fact=_KILN)
    result = await fetcher.fetch(_request("kiln street pottery", turns=("hello there",)))

    if adapter:
        assert result.stage.prefetch_reuse == "none"
        assert _served(result) == []
        assert result.suppressed == 1
    else:
        # The stale stage 1 was reused, and its suppression set predates the correction.
        assert result.stage.prefetch_reuse == "stage1"
        assert _served(result) == [kiln.fact_key]


# -- the setting ------------------------------------------------------------------------------


def test_corrections_are_enabled_by_default_and_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in MEMORY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    assert MemoryConfig().corrections_enabled is True
    assert MemoryConfig.from_env().corrections_enabled is True
    monkeypatch.setenv("MIST_MEMORY_CORRECTIONS_ENABLED", "false")
    assert MemoryConfig.from_env().corrections_enabled is False
    assert "MIST_MEMORY_CORRECTIONS_ENABLED" in MEMORY_ENV_VARS


def test_the_setting_is_documented_beside_the_other_memory_settings() -> None:
    lines = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    settings = [line for line in lines if re.match(r"^# ?[A-Z][A-Z0-9_]+=", line)]

    at = settings.index("# MIST_MEMORY_CORRECTIONS_ENABLED=true")

    assert settings[at - 1].startswith("# MIST_MEMORY_")
