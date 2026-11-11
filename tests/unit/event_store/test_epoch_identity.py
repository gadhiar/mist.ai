"""Epoch identity binding in the event store (mist-extraction-epoch-identity T3).

An epoch, or a cutover candidate, is bound once to a serving and a dispatch
identity and to a cache namespace (NULL: the 3-part key). The binding of an
epoch lives in `epoch_identity`; a candidate's lives in new nullable
`epoch_cutover` columns until promotion copies it, in the ledger insert's
transaction.

The migration tests build the pre-change shape on disk by dropping the new
tables and recreating `epoch_cutover` exactly as it was declared before this
change (`_PRE_CHANGE_EPOCH_CUTOVER`).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EpochCutoverStateError, EventStore

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
NS = "si1-a+di1-a"  # the keyed namespace of `_binding`'s default identities
PROBE = {"node_count": 1}

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
_PRE_CHANGE_CUTOVER_COLUMNS = (
    "cutover_id",
    "ontology_version",
    "extraction_version",
    "model_hash",
    "bare_model_hash",
    "requested_at",
    "state",
    "source_epoch_id",
    "rebuild_job_id",
    "rebuilt_through_event_id",
    "check_report",
    "promoted_epoch_id",
    "updated_at",
)
_BINDING_COLUMNS = (
    "serving_identity",
    "serving_components",
    "dispatch_identity",
    "dispatch_components",
    "identity_namespace",
    "binding_source",
    "binding_evidence",
    "bound_at",
)


class SimulatedCrashError(Exception):
    """The process dying at the injected point."""


def _store(path: str = ":memory:") -> EventStore:
    store = EventStore(db_path=path)
    store.initialize()
    store.ensure_initial_epoch(
        now_iso=T0.isoformat(), ontology_version="1.4.0", extraction_version="ev", model_hash="m1"
    )
    return store


def _log(store: EventStore, n: int) -> list[str]:
    store.start_session("s1", input_modality="text")
    return [
        store.append_turn(
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


def _begin(store: EventStore, *, model_hash: str = "m2|emb:e") -> int:
    cutover_id = store.begin_epoch_cutover(
        ontology_version="1.4.0",
        extraction_version="ev",
        model_hash=model_hash,
        bare_model_hash=model_hash.split("|")[0],
        source_epoch_id=int(store.get_current_epoch()["epoch_id"]),
        requested_at=T0.isoformat(),
    )
    assert cutover_id is not None
    return cutover_id


def _to(store: EventStore, cutover_id: int, state: str, through: str | None = None) -> None:
    """Walk a filling cutover forward to 'ready' or 'checked'."""
    assert store.transition_epoch_cutover(
        cutover_id, from_states=("filling",), to_state="ready", updated_at="t1"
    )
    if state == "checked":
        assert store.transition_epoch_cutover(
            cutover_id,
            from_states=("ready",),
            to_state="checked",
            updated_at="t2",
            fields={
                "rebuilt_through_event_id": through,
                "rebuild_job_id": "job",
                "check_report": '{"passed": true}',
            },
        )


def _binding(
    serving: str = "si1-a", dispatch: str = "di1-a", namespace: str | None = None, **overrides
) -> dict:
    values = {
        "serving_identity": serving,
        "serving_components": {"adapter": "gptoss", "n_ctx": 16384},
        "dispatch_identity": dispatch,
        "dispatch_components": {"history_messages": 10},
        "identity_namespace": namespace,
        "source": "adopted",
        "evidence": {"rule": "5c"},
        "bound_at": "t-bind",
    }
    values.update(overrides)
    return values


def _bind_epoch(store: EventStore, epoch_id: int, **kwargs) -> bool:
    return store.bind_epoch_identity(epoch_id=epoch_id, **_binding(**kwargs))


def _bind_cutover(store: EventStore, cutover_id: int, **kwargs) -> bool:
    return store.bind_cutover_identity(cutover_id, **_binding(**kwargs))


def _make_pre_change(path: str) -> None:
    """Rewrite a freshly initialized database into the shape it had before this change."""
    conn = sqlite3.connect(path)
    conn.executescript(
        "DROP TABLE epoch_identity; DROP TABLE epoch_identity_acceptance; "
        "DROP TABLE epoch_cutover;" + _PRE_CHANGE_EPOCH_CUTOVER
    )
    conn.close()


def _columns(conn: sqlite3.Connection, table: str) -> set[tuple]:
    return {(r[1], r[2], r[3], r[4]) for r in conn.execute(f"PRAGMA table_info({table})")}


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


class TestSchema:
    def test_initialize_creates_both_identity_tables_idempotently(self):
        store = _store()
        store.initialize()

        tables = set(_tables(store._get_connection()))

        assert {"epoch_identity", "epoch_identity_acceptance"} <= tables

    def test_a_fresh_and_a_migrated_epoch_cutover_have_the_same_columns(self, tmp_path):
        """The CREATE TABLE and the guarded ALTER declare the binding columns independently."""
        fresh = _store()
        path = str(tmp_path / "old.db")
        _store(path).close()
        _make_pre_change(path)

        migrated = EventStore(db_path=path)
        migrated.initialize()

        fresh_columns = _columns(fresh._get_connection(), "epoch_cutover")
        assert fresh_columns == _columns(migrated._get_connection(), "epoch_cutover")
        assert {c[0] for c in fresh_columns} >= set(_BINDING_COLUMNS)
        migrated.close()

    def test_the_oldest_epoch_cutover_migrates_to_the_fresh_column_order(self, tmp_path):
        """A table from before the binding gains its columns first, then the two pack hashes."""
        fresh = _store()
        path = str(tmp_path / "old.db")
        _store(path).close()
        _make_pre_change(path)

        migrated = EventStore(db_path=path)
        migrated.initialize()

        def names(store: EventStore) -> list[str]:
            return [
                row[1]
                for row in store._get_connection().execute("PRAGMA table_info(epoch_cutover)")
            ]

        assert names(migrated) == names(fresh)
        assert names(fresh) == [
            *_PRE_CHANGE_CUTOVER_COLUMNS,
            *_BINDING_COLUMNS,
            "pack_prompt_hash",
            "pack_rails_hash",
        ]
        migrated.close()

    def test_a_pre_change_database_opens_and_keeps_every_row_byte_identical(self, tmp_path):
        # Arrange: a pre-change database with a ledger, a log, markers and an open cutover.
        path = str(tmp_path / "old.db")
        store = _store(path)
        ids = _log(store, 2)
        store.mark_extraction_stage(event_id=ids[0], epoch_id=1, stage="applied", updated_at="t0")
        store.close()
        _make_pre_change(path)
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO epoch_cutover (ontology_version, extraction_version, model_hash, "
            "bare_model_hash, requested_at, state, source_epoch_id, check_report, updated_at) "
            "VALUES ('1.5.0', 'ev', 'm1', 'm1', 't0', 'ready', 1, '{\"x\": 1}', 't0')"
        )
        conn.commit()
        before = {
            table: [tuple(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in _tables(conn)
        }
        conn.close()

        # Act
        migrated = EventStore(db_path=path)
        migrated.initialize()
        migrated.initialize()

        # Assert: every old row is unchanged in every old column; the new columns are NULL.
        after_conn = migrated._get_connection()
        for table, rows in before.items():
            columns = ", ".join(_PRE_CHANGE_CUTOVER_COLUMNS) if table == "epoch_cutover" else "*"
            after = [
                tuple(r)
                for r in after_conn.execute(f"SELECT {columns} FROM {table} ORDER BY rowid")
            ]
            assert after == rows, table
        cutover = migrated.get_open_epoch_cutover()
        assert all(cutover[name] is None for name in _BINDING_COLUMNS)
        assert migrated.get_epoch_identity(1) is None
        migrated.close()


class TestBindEpoch:
    def test_the_first_bind_writes_the_row_and_reads_it_back_decoded(self):
        store = _store()

        assert _bind_epoch(store, 1) is True

        assert store.get_epoch_identity(1) == {
            "epoch_id": 1,
            "serving_identity": "si1-a",
            "serving_components": {"adapter": "gptoss", "n_ctx": 16384},
            "dispatch_identity": "di1-a",
            "dispatch_components": {"history_messages": 10},
            "identity_namespace": None,
            "source": "adopted",
            "evidence": {"rule": "5c"},
            "bound_at": "t-bind",
        }

    def test_a_second_bind_returns_false_and_leaves_the_first_row(self):
        store = _store()
        _bind_epoch(store, 1)
        first = store.get_epoch_identity(1)

        second = _bind_epoch(store, 1, serving="si1-b", namespace="si1-b+di1-a", source="operator")

        assert second is False
        assert store.get_epoch_identity(1) == first

    def test_an_unbound_epoch_reads_as_none(self):
        assert _store().get_epoch_identity(1) is None

    def test_an_unknown_source_is_refused(self):
        store = _store()

        with pytest.raises(ValueError, match="source"):
            _bind_epoch(store, 1, source="guessed")
        assert store.get_epoch_identity(1) is None

    @pytest.mark.parametrize(
        "namespace",
        [
            pytest.param("si1-a+di1-other", id="another-identity"),
            pytest.param("", id="empty"),
        ],
    )
    def test_a_namespace_other_than_the_keyed_form_is_refused(self, namespace):
        store = _store()

        with pytest.raises(ValueError, match="identity_namespace"):
            _bind_epoch(store, 1, namespace=namespace)
        assert store.get_epoch_identity(1) is None

    def test_an_epoch_missing_from_the_ledger_is_refused(self):
        store = _store()

        with pytest.raises(ValueError, match="epoch 7"):
            _bind_epoch(store, 7)
        assert store.get_epoch_identity(7) is None


class TestBindCutover:
    def test_the_first_bind_writes_the_columns_and_a_second_returns_false(self):
        store = _store()
        cutover_id = _begin(store)

        first = _bind_cutover(store, cutover_id, namespace=NS, source="first-contact")
        second = _bind_cutover(store, cutover_id, serving="si1-b", source="operator")

        row = store.get_epoch_cutover(cutover_id)
        assert (first, second) == (True, False)
        assert (
            row["serving_identity"],
            row["dispatch_identity"],
            row["identity_namespace"],
            row["binding_source"],
            row["bound_at"],
            row["state"],
        ) == ("si1-a", "di1-a", NS, "first-contact", "t-bind", "filling")
        assert store.get_cutover_identity(cutover_id)["serving_components"] == {
            "adapter": "gptoss",
            "n_ctx": 16384,
        }

    @pytest.mark.parametrize("closed", ["abandoned", "promoted"])
    def test_a_closed_cutover_is_not_bound(self, closed):
        store = _store()
        ids = _log(store, 1)
        cutover_id = _begin(store)
        if closed == "abandoned":
            assert store.transition_epoch_cutover(
                cutover_id, from_states=("filling",), to_state="abandoned", updated_at="t1"
            )
        else:
            _to(store, cutover_id, "checked", ids[0])
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")

        assert _bind_cutover(store, cutover_id) is False
        assert store.get_epoch_cutover(cutover_id)["serving_identity"] is None

    def test_an_unknown_cutover_is_not_bound(self):
        assert _bind_cutover(_store(), 99) is False

    def test_an_unknown_source_is_refused(self):
        store = _store()
        cutover_id = _begin(store)

        with pytest.raises(ValueError, match="source"):
            _bind_cutover(store, cutover_id, source="promoted")


class TestKeyedBindAfterACheck:
    """Plan review F13: a keyed bind voids coverage and a check made in the 3-part namespace."""

    @pytest.mark.parametrize("state", ["ready", "checked"])
    def test_a_keyed_bind_moves_the_cutover_back_to_filling_and_clears_the_check(self, state):
        store = _store()
        ids = _log(store, 2)
        cutover_id = _begin(store)
        _to(store, cutover_id, state, ids[1])

        assert _bind_cutover(store, cutover_id, namespace=NS) is True

        row = store.get_epoch_cutover(cutover_id)
        assert (
            row["state"],
            row["rebuild_job_id"],
            row["rebuilt_through_event_id"],
            row["check_report"],
            row["updated_at"],
        ) == ("filling", None, None, None, "t-bind")

    @pytest.mark.parametrize("state", ["ready", "checked"])
    def test_both_promotion_paths_then_refuse(self, state):
        store = _store()
        ids = _log(store, 1)
        cutover_id = _begin(store)
        _to(store, cutover_id, state, ids[0])
        _bind_cutover(store, cutover_id, namespace=NS)

        with pytest.raises(EpochCutoverStateError, match="'filling'"):
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")
        with pytest.raises(EpochCutoverStateError, match="'filling'"):
            store.promote_epoch_cutover_seed_only(
                cutover_id=cutover_id, activated_at="t3", probe_report=PROBE
            )
        assert len(store.list_epochs()) == 1

    def test_a_3_part_bind_leaves_the_state_and_the_check_unchanged(self):
        store = _store()
        ids = _log(store, 2)
        cutover_id = _begin(store)
        _to(store, cutover_id, "checked", ids[1])
        before = store.get_epoch_cutover(cutover_id)

        assert _bind_cutover(store, cutover_id, namespace=None) is True

        after = store.get_epoch_cutover(cutover_id)
        assert {k: after[k] for k in _PRE_CHANGE_CUTOVER_COLUMNS} == {
            k: before[k] for k in _PRE_CHANGE_CUTOVER_COLUMNS
        }
        assert after["serving_identity"] == "si1-a"

    def test_a_keyed_bind_of_a_filling_cutover_changes_only_the_binding(self):
        store = _store()
        cutover_id = _begin(store)
        before = store.get_epoch_cutover(cutover_id)

        _bind_cutover(store, cutover_id, namespace=NS)

        after = store.get_epoch_cutover(cutover_id)
        assert {k: after[k] for k in _PRE_CHANGE_CUTOVER_COLUMNS} == {
            k: before[k] for k in _PRE_CHANGE_CUTOVER_COLUMNS
        }


class TestPromotionWritesTheBinding:
    @pytest.mark.parametrize("namespace", [None, NS], ids=["3-part", "keyed"])
    def test_promote_copies_the_cutovers_binding_to_the_new_epoch(self, namespace):
        store = _store()
        ids = _log(store, 2)
        cutover_id = _begin(store)
        _bind_cutover(store, cutover_id, namespace=namespace, source="first-contact")
        _to(store, cutover_id, "checked", ids[1])

        epoch_id = store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")["epoch_id"]

        identity = store.get_epoch_identity(epoch_id)
        assert (
            identity["serving_identity"],
            identity["dispatch_identity"],
            identity["identity_namespace"],
            identity["source"],
            identity["bound_at"],
            identity["serving_components"],
        ) == ("si1-a", "di1-a", namespace, "promoted", "t3", {"adapter": "gptoss", "n_ctx": 16384})
        assert identity["evidence"] == {
            "cutover_id": cutover_id,
            "cutover_binding_source": "first-contact",
            "cutover_binding_evidence": {"rule": "5c"},
        }

    def test_seed_only_promotion_copies_the_binding_too(self):
        store = _store()
        cutover_id = _begin(store)
        _bind_cutover(store, cutover_id, namespace=NS)
        _to(store, cutover_id, "ready")

        epoch_id = store.promote_epoch_cutover_seed_only(
            cutover_id=cutover_id, activated_at="t3", probe_report=PROBE
        )["epoch_id"]

        identity = store.get_epoch_identity(epoch_id)
        assert (identity["source"], identity["identity_namespace"]) == ("promoted", NS)

    def test_an_unbound_cutover_promotes_with_no_binding_row(self):
        store = _store()
        ids = _log(store, 1)
        cutover_id = _begin(store)
        _to(store, cutover_id, "checked", ids[0])

        epoch_id = store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")["epoch_id"]

        assert store.get_epoch_identity(epoch_id) is None
        count = store._get_connection().execute("SELECT COUNT(*) FROM epoch_identity")
        assert count.fetchone()[0] == 0

    @pytest.mark.parametrize("path_name", ["graph-swapped", "seed-only"])
    def test_a_crash_after_the_ledger_insert_leaves_neither_row(
        self, tmp_path, monkeypatch, path_name
    ):
        # Arrange
        path = str(tmp_path / "event_store.db")
        store = _store(path)
        ids = _log(store, 1) if path_name == "graph-swapped" else []
        cutover_id = _begin(store)
        _bind_cutover(store, cutover_id, namespace=NS)
        _to(store, cutover_id, "checked", ids[0] if ids else None)

        def crash(step: str) -> None:
            raise SimulatedCrashError(step)

        monkeypatch.setattr(store, "_promotion_fault_point", crash)

        # Act
        with pytest.raises(SimulatedCrashError):
            if path_name == "graph-swapped":
                store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")
            else:
                store.promote_epoch_cutover_seed_only(
                    cutover_id=cutover_id, activated_at="t3", probe_report=PROBE
                )

        # Assert
        store.close()
        reopened = EventStore(db_path=path)
        assert len(reopened.list_epochs()) == 1
        assert reopened.get_epoch_identity(2) is None
        assert (
            reopened._get_connection().execute("SELECT COUNT(*) FROM epoch_identity").fetchone()[0]
            == 0
        )
        assert reopened.get_epoch_cutover(cutover_id)["state"] == "checked"
        reopened.close()


class TestAcceptance:
    def test_an_acceptance_is_written_and_read_back(self):
        store = _store()

        store.accept_identity(
            -3, serving_identity="si1-a", dispatch_identity="di1-a", note="ok", accepted_at="t1"
        )

        assert store.get_identity_acceptance(-3) == {
            "target_id": -3,
            "serving_identity": "si1-a",
            "dispatch_identity": "di1-a",
            "note": "ok",
            "accepted_at": "t1",
        }

    def test_a_later_acceptance_replaces_the_earlier_one(self):
        store = _store()
        store.accept_identity(
            1, serving_identity="si1-a", dispatch_identity="di1-a", note="a", accepted_at="t1"
        )

        store.accept_identity(
            1, serving_identity="si1-b", dispatch_identity="di1-b", note="b", accepted_at="t2"
        )

        assert store.get_identity_acceptance(1)["serving_identity"] == "si1-b"

    def test_no_acceptance_reads_as_none(self):
        assert _store().get_identity_acceptance(1) is None


class TestLegacyClaims:
    def test_claims_come_from_bound_epochs_and_cutovers_in_the_3_part_namespace(self):
        # Arrange: the pair (ev, m1) is the active epoch's.
        store = _store()
        _bind_epoch(store, 1, serving="si1-epoch", dispatch="di1-epoch")
        open_id = _begin(store, model_hash="m1")
        _bind_cutover(store, open_id, serving="si1-open", dispatch="di1-open")
        assert store.transition_epoch_cutover(
            open_id, from_states=("filling",), to_state="abandoned", updated_at="t1"
        )
        live_id = _begin(store, model_hash="m1")
        _bind_cutover(store, live_id, serving="si1-epoch", dispatch="di1-epoch")

        claims = store.legacy_claims("ev", "m1")

        assert claims == [("si1-epoch", "di1-epoch"), ("si1-open", "di1-open")]

    def test_keyed_bindings_unbound_cutovers_and_other_pairs_are_not_claims(self):
        store = _store()
        _bind_epoch(store, 1, serving="si1-keyed", namespace="si1-keyed+di1-a")
        keyed_cutover = _begin(store, model_hash="m1")
        _bind_cutover(store, keyed_cutover, serving="si1-keyed-c", namespace="si1-keyed-c+di1-a")
        store.transition_epoch_cutover(
            keyed_cutover, from_states=("filling",), to_state="abandoned", updated_at="t1"
        )
        _begin(store, model_hash="m1")  # unbound and open
        store.transition_epoch_cutover(
            store.get_open_epoch_cutover()["cutover_id"],
            from_states=("filling",),
            to_state="abandoned",
            updated_at="t1",
        )
        other = _begin(store, model_hash="m2|emb:e")
        _bind_cutover(store, other, serving="si1-other")

        assert store.legacy_claims("ev", "m1") == []
        assert store.legacy_claims("ev", "m2|emb:e") == [("si1-other", "di1-a")]

    def test_a_promoted_cutovers_epoch_and_its_cutover_give_one_claim(self):
        store = _store()
        ids = _log(store, 1)
        cutover_id = _begin(store)
        _bind_cutover(store, cutover_id)
        _to(store, cutover_id, "checked", ids[0])
        store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")

        assert store.legacy_claims("ev", "m2|emb:e") == [("si1-a", "di1-a")]
