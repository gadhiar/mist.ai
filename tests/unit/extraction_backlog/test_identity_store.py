"""`BacklogStore` over bound and unbound targets (mist-extraction-epoch-identity T3).

A target (the active epoch or a cutover candidate) reads and writes the cache in
its binding's namespace. An unbound target uses the 3-part namespace, so with no
binding written nothing changes. The no-refill test opens a database built in
the shape it had before this change and checks the backlog reads the same.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EventStore
from backend.extraction_backlog.store import BacklogStore, Cutover, Epoch
from backend.knowledge.extraction_cache import (
    OUTCOME_EXTRACTED,
    OUTCOME_SKIPPED,
    SKIP_EXTRACTION_FAILED,
    SKIP_RATE_LIMITED,
    ExtractionCache,
    cache_key,
)

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
EV = "ev"
MH = "m1|emb:e"
NS = "si1-a+di1-a"
_AT = "2026-09-01T00:00:00+00:00"


def _binding(namespace: str | None, source: str = "adopted") -> dict:
    return {
        "serving_identity": "si1-a",
        "serving_components": {"adapter": "gptoss"},
        "dispatch_identity": "di1-a",
        "dispatch_components": {"history_messages": 10},
        "identity_namespace": namespace,
        "source": source,
        "evidence": {},
        "bound_at": "t-bind",
    }


def _world(
    event_path: str = ":memory:", cache_path: str = ":memory:"
) -> tuple[EventStore, ExtractionCache, BacklogStore]:
    events = EventStore(db_path=event_path)
    events.initialize()
    events.ensure_initial_epoch(
        now_iso=_AT, ontology_version="1.4.0", extraction_version=EV, model_hash=MH
    )
    cache = ExtractionCache(cache_path)
    cache.initialize()
    return events, cache, BacklogStore(events, cache)


def _log(events: EventStore, n: int) -> list[str]:
    events.start_session("s1", input_modality="text")
    return [
        events.append_turn(
            ConversationTurnEvent(
                session_id="s1",
                turn_index=i,
                timestamp=T0 + timedelta(minutes=i),
                user_utterance=f"turn {i}",
                system_response="ok",
                ontology_version="1.4.0",
            )
        )
        for i in range(n)
    ]


def _put(cache: ExtractionCache, event_id: str, *, model_hash: str = MH, **kwargs) -> None:
    cache.put(
        event_id, "1.4.0", EV, model_hash, outcome=OUTCOME_EXTRACTED, created_at=_AT, **kwargs
    )


def _begin(store: BacklogStore, *, ontology: str = "1.4.0", model_hash: str = MH) -> Cutover:
    cutover = store.begin_cutover(
        ontology_version=ontology,
        extraction_version=EV,
        model_hash=model_hash,
        bare_model_hash=model_hash.split("|")[0],
        source_epoch_id=1,
        requested_at=_AT,
    )
    assert cutover is not None
    return cutover


def _namespaces(cache: ExtractionCache) -> list[tuple]:
    rows = cache._get_connection().execute(
        "SELECT event_id, skip_reason, identity_namespace FROM extraction_cache ORDER BY rowid"
    )
    return [tuple(r) for r in rows]


class TestTheBindingOnTheTarget:
    def test_epoch_and_cutover_default_to_unbound(self):
        epoch = Epoch(epoch_id=1, ontology_version="o", extraction_version="e", model_hash="m")

        assert (
            epoch.serving_identity,
            epoch.dispatch_identity,
            epoch.identity_namespace,
            epoch.identity_source,
        ) == (None, None, None, None)

    def test_active_epoch_is_unbound_until_a_binding_is_written(self):
        events, _, store = _world()

        assert store.active_epoch().identity_namespace is None
        assert store.active_epoch().serving_identity is None

    @pytest.mark.parametrize("namespace", [None, NS], ids=["3-part", "keyed"])
    def test_active_epoch_joins_its_binding(self, namespace):
        events, _, store = _world()
        events.bind_epoch_identity(epoch_id=1, **_binding(namespace, source="first-contact"))

        epoch = store.active_epoch()

        assert (
            epoch.epoch_id,
            epoch.serving_identity,
            epoch.dispatch_identity,
            epoch.identity_namespace,
            epoch.identity_source,
        ) == (1, "si1-a", "di1-a", namespace, "first-contact")

    def test_get_epoch_reads_a_ledger_epoch_with_its_binding(self):
        events, _, store = _world()
        events.bind_epoch_identity(epoch_id=1, **_binding(NS))

        assert store.get_epoch(1).identity_namespace == NS
        assert store.get_epoch(42) is None

    def test_a_bound_cutover_carries_its_binding_to_target_and_epoch_dict(self):
        events, _, store = _world()
        cutover = _begin(store, model_hash="m2|emb:e")
        events.bind_cutover_identity(cutover.cutover_id, **_binding(NS, source="first-contact"))

        bound = store.get_cutover(cutover.cutover_id)

        assert (
            bound.serving_identity,
            bound.dispatch_identity,
            bound.identity_namespace,
            bound.identity_source,
        ) == ("si1-a", "di1-a", NS, "first-contact")
        target = bound.target
        assert (target.epoch_id, target.identity_namespace, target.serving_identity) == (
            -cutover.cutover_id,
            NS,
            "si1-a",
        )
        assert bound.epoch_dict()["identity_namespace"] == NS

    def test_an_unbound_cutover_reads_unbound_and_its_epoch_dict_names_the_3_part_namespace(
        self,
    ):
        _, _, store = _world()
        cutover = _begin(store, model_hash="m2|emb:e")

        assert (cutover.serving_identity, cutover.identity_source) == (None, None)
        assert cutover.target.identity_namespace is None
        assert cutover.epoch_dict() == {
            "epoch_id": -cutover.cutover_id,
            "ontology_version": "1.4.0",
            "extraction_version": EV,
            "model_hash": "m2|emb:e",
            "activated_at": _AT,
            "identity_namespace": None,
        }


class TestCacheCallsUseTheTargetsNamespace:
    def test_an_unbound_target_writes_and_reads_the_3_part_namespace(self):
        events, cache, store = _world()
        ids = _log(events, 2)
        epoch = store.active_epoch()

        store.put_extracted(
            ids[0],
            epoch,
            created_at=_AT,
            entities=[],
            relationships=[],
            scope=None,
            scope_confidence=None,
            derivation=None,
            service_stamps=None,
        )
        store.put_skip(ids[1], epoch, skip_reason=SKIP_RATE_LIMITED, created_at=_AT)

        assert _namespaces(cache) == [(ids[0], None, None), (ids[1], SKIP_RATE_LIMITED, None)]
        assert cache.get(ids[0], EV, MH) == store.get_cached(ids[0], epoch)

    def test_a_keyed_target_writes_and_reads_only_its_namespace(self):
        # Arrange: a 3-part row and a keyed epoch over the same triple.
        events, cache, store = _world()
        ids = _log(events, 3)
        _put(cache, ids[0], scope="legacy")
        events.bind_epoch_identity(epoch_id=1, **_binding(NS))
        epoch = store.active_epoch()

        # Act
        store.put_extracted(
            ids[1],
            epoch,
            created_at=_AT,
            entities=[],
            relationships=[],
            scope="keyed",
            scope_confidence=None,
            derivation=None,
            service_stamps=None,
        )
        store.put_skip(ids[2], epoch, skip_reason=SKIP_EXTRACTION_FAILED, created_at=_AT)

        # Assert
        assert _namespaces(cache) == [
            (ids[0], None, None),
            (ids[1], None, NS),
            (ids[2], SKIP_EXTRACTION_FAILED, NS),
        ]
        assert store.get_cached(ids[0], epoch) is None
        assert store.get_cached(ids[1], epoch)["scope"] == "keyed"
        scan = store.scan(epoch)
        assert [t.event_id for t in scan.pending] == [ids[0], ids[1], ids[2]]
        assert (scan.backlog_depth, scan.apply_pending, scan.dead_lettered) == (1, 2, 1)
        assert store.list_dead_letters(epoch) == [ids[2]]

    def test_retry_dead_letter_deletes_only_the_keyed_row(self):
        events, cache, store = _world()
        ids = _log(events, 1)
        cache.put(
            ids[0],
            "1.4.0",
            EV,
            MH,
            outcome=OUTCOME_SKIPPED,
            skip_reason=SKIP_EXTRACTION_FAILED,
            created_at=_AT,
        )
        events.bind_epoch_identity(epoch_id=1, **_binding(NS))
        epoch = store.active_epoch()
        store.put_skip(ids[0], epoch, skip_reason=SKIP_EXTRACTION_FAILED, created_at=_AT)

        assert store.retry_dead_letter(ids[0], epoch) is True

        assert _namespaces(cache) == [(ids[0], SKIP_EXTRACTION_FAILED, None)]

    def test_activation_counts_only_the_targets_namespace(self):
        events, cache, store = _world()
        ids = _log(events, 2)
        _put(cache, ids[0])
        _put(cache, ids[1], identity_namespace=NS)
        events.bind_epoch_identity(epoch_id=1, **_binding(NS))

        activation = store.ensure_activation(store.active_epoch(), now_iso=_AT)

        assert (activation.marked_applied, activation.legacy_unextracted) == (1, 1)
        assert events.get_extraction_applied(1) == {ids[1]: "applied"}


class TestFillScan:
    def test_an_unbound_settings_change_candidate_is_covered_0_with_the_first_turn_as_head(
        self,
    ):
        """Its triple is its source's, so the 3-part rows are the source's, not its own."""
        events, cache, store = _world()
        ids = _log(events, 3)
        for event_id in ids:
            _put(cache, event_id)
        cutover = _begin(store)

        fill = store.fill_scan(cutover)

        assert (fill.head.event_id, fill.covered, fill.total, fill.bound) == (
            ids[0],
            0,
            3,
            False,
        )

    def test_an_unbound_settings_change_candidate_over_an_empty_log_has_no_head(self):
        _, _, store = _world()

        fill = store.fill_scan(_begin(store))

        assert (fill.head, fill.covered, fill.total, fill.bound) == (None, 0, 0, False)

    @pytest.mark.parametrize(
        ("ontology", "model_hash"),
        [
            pytest.param("1.5.0", MH, id="ontology-only"),
            pytest.param("1.4.0", "m2|emb:e", id="model-change"),
        ],
    )
    def test_any_other_unbound_candidate_is_scanned_in_the_3_part_namespace(
        self, ontology, model_hash
    ):
        events, cache, store = _world()
        ids = _log(events, 3)
        _put(cache, ids[0], model_hash=model_hash)
        _put(cache, ids[1], model_hash=model_hash, identity_namespace=NS)
        cutover = _begin(store, ontology=ontology, model_hash=model_hash)

        fill = store.fill_scan(cutover)

        assert (fill.head.event_id, fill.covered, fill.total, fill.bound) == (
            ids[1],
            1,
            3,
            False,
        )

    def test_a_bound_keyed_candidate_is_scanned_in_its_namespace(self):
        events, cache, store = _world()
        ids = _log(events, 3)
        _put(cache, ids[0])
        _put(cache, ids[1])
        _put(cache, ids[0], identity_namespace=NS)
        cutover = _begin(store)
        events.bind_cutover_identity(cutover.cutover_id, **_binding(NS))

        fill = store.fill_scan(store.get_cutover(cutover.cutover_id))

        assert (fill.head.event_id, fill.covered, fill.bound) == (ids[1], 1, True)

    def test_a_bound_3_part_candidate_with_its_sources_triple_is_scanned_in_the_3_part_namespace(
        self,
    ):
        events, cache, store = _world()
        ids = _log(events, 2)
        _put(cache, ids[0])
        cutover = _begin(store, ontology="1.5.0")
        events.bind_cutover_identity(cutover.cutover_id, **_binding(None))

        fill = store.fill_scan(store.get_cutover(cutover.cutover_id))

        assert (fill.head.event_id, fill.covered, fill.bound) == (ids[1], 1, True)

    def test_a_candidate_whose_source_epoch_is_missing_is_scanned_as_today(self):
        events, cache, store = _world()
        ids = _log(events, 2)
        _put(cache, ids[0])
        cutover_id = events.begin_epoch_cutover(
            ontology_version="1.4.0",
            extraction_version=EV,
            model_hash=MH,
            bare_model_hash="m1",
            source_epoch_id=99,
            requested_at=_AT,
        )

        fill = store.fill_scan(store.get_cutover(cutover_id))

        assert (fill.head.event_id, fill.covered, fill.bound) == (ids[1], 1, False)


# `epoch_cutover` as `schema.sql` declared it at 5f047d1, before the binding columns.
_PRE_CHANGE_EPOCH_CUTOVER = """
CREATE TABLE epoch_cutover (
    cutover_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ontology_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    bare_model_hash TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    state TEXT NOT NULL,
    source_epoch_id INTEGER,
    rebuild_job_id TEXT,
    rebuilt_through_event_id TEXT,
    check_report TEXT,
    promoted_epoch_id INTEGER,
    updated_at TEXT NOT NULL
);
"""
# The cache table as `extraction_cache._DDL` declared it at 5f047d1.
_PRE_CHANGE_CACHE = """
CREATE TABLE extraction_cache (
    cache_key TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    ontology_version TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'extracted',
    skip_reason TEXT,
    scope TEXT,
    scope_confidence REAL,
    payload TEXT NOT NULL,
    created_at TEXT,
    derivation TEXT,
    service_stamps TEXT
);
"""


def _pre_change_world(tmp_path) -> tuple[str, str, list[str]]:
    """Files in the pre-change shape: legacy rows, markers, an activation and an open cutover.

    Turns 0-1 are applied, turn 2 is cached but not applied, turns 3-4 have no
    row, turn 5 is a dead letter (applied as a no-op). The open cutover is an
    ontology-only candidate with candidate rows written by old code: the same
    3-part rows, by spec D3.
    """
    event_path = str(tmp_path / "event_store.db")
    cache_path = str(tmp_path / "extraction_cache.db")
    events = EventStore(db_path=event_path)
    events.initialize()
    events.ensure_initial_epoch(
        now_iso=_AT, ontology_version="1.4.0", extraction_version=EV, model_hash=MH
    )
    events.record_extraction_activation(
        epoch_id=1,
        activated_at=_AT,
        turns_at_activation=0,
        applied_event_ids=[],
        legacy_event_ids=[],
    )
    ids = _log(events, 6)
    for event_id in (ids[0], ids[1], ids[5]):
        events.mark_extraction_stage(event_id=event_id, epoch_id=1, stage="applied", updated_at=_AT)
    events.close()
    conn = sqlite3.connect(event_path)
    conn.executescript(
        "DROP TABLE epoch_identity; DROP TABLE epoch_identity_acceptance; "
        "DROP TABLE epoch_cutover;" + _PRE_CHANGE_EPOCH_CUTOVER
    )
    conn.execute(
        "INSERT INTO epoch_cutover (ontology_version, extraction_version, model_hash, "
        "bare_model_hash, requested_at, state, source_epoch_id, updated_at) "
        "VALUES ('1.5.0', ?, ?, 'm1', ?, 'filling', 1, ?)",
        (EV, MH, _AT, _AT),
    )
    conn.commit()
    conn.close()

    conn = sqlite3.connect(cache_path)
    conn.executescript(_PRE_CHANGE_CACHE)
    for event_id, outcome, reason in (
        (ids[0], "extracted", None),
        (ids[1], "extracted", None),
        (ids[2], "extracted", None),
        (ids[5], "skipped", SKIP_EXTRACTION_FAILED),
    ):
        conn.execute(
            "INSERT INTO extraction_cache (cache_key, event_id, ontology_version, "
            "extraction_version, model_hash, outcome, skip_reason, payload, created_at) "
            "VALUES (?, ?, '1.4.0', ?, ?, ?, ?, '{}', ?)",
            (cache_key(event_id, EV, MH), event_id, EV, MH, outcome, reason, _AT),
        )
    conn.commit()
    conn.close()
    return event_path, cache_path, ids


def _cache_rows(cache_path: str) -> list[tuple]:
    conn = sqlite3.connect(cache_path)
    columns = (
        "cache_key, event_id, ontology_version, extraction_version, model_hash, outcome, "
        "skip_reason, scope, scope_confidence, payload, created_at, derivation, service_stamps"
    )
    rows = [
        tuple(r) for r in conn.execute(f"SELECT {columns} FROM extraction_cache ORDER BY rowid")
    ]
    conn.close()
    return rows


class TestNoRefill:
    def test_scan_and_fill_scan_read_the_same_after_the_migration(self, tmp_path):
        # Arrange
        event_path, cache_path, ids = _pre_change_world(tmp_path)
        rows_before = _cache_rows(cache_path)

        # Act
        events = EventStore(db_path=event_path)
        events.initialize()
        cache = ExtractionCache(cache_path)
        cache.initialize()
        store = BacklogStore(events, cache)
        epoch = store.active_epoch()
        scan = store.scan(epoch)
        fill = store.fill_scan(store.open_cutover())

        # Assert: what old code read from the same files (see `_pre_change_world`).
        assert epoch.identity_namespace is None
        assert [(t.event_id, t.cached) for t in scan.pending] == [
            (ids[2], True),
            (ids[3], False),
            (ids[4], False),
        ]
        assert (
            scan.backlog_depth,
            scan.apply_pending,
            scan.dead_lettered,
            scan.legacy_unextracted,
        ) == (2, 1, 1, 0)
        assert (fill.head.event_id, fill.covered, fill.total, fill.bound) == (ids[3], 4, 6, False)
        assert store.ensure_activation(epoch, now_iso="later").created_now is False
        assert _cache_rows(cache_path) == rows_before
        events.close()
        cache.close()
