"""The `epoch_cutover` table and its store methods (T2b).

The atomicity test uses an on-disk database so a SECOND connection can look at
the file while promotion's transaction is open, and again after the injected
crash: what a restarted process would see.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from backend.event_store.models import ConversationTurnEvent
from backend.event_store.store import EpochCutoverStateError, EventStore

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


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
    ids = []
    for i in range(n):
        ids.append(
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
        )
    return ids


def _begin(store: EventStore) -> int | None:
    return store.begin_epoch_cutover(
        ontology_version="1.4.0",
        extraction_version="ev",
        model_hash="m2|emb:e",
        bare_model_hash="m2",
        source_epoch_id=1,
        requested_at=T0.isoformat(),
    )


def _checked(store: EventStore, cutover_id: int, through: str) -> None:
    assert store.transition_epoch_cutover(
        cutover_id, from_states=("filling",), to_state="ready", updated_at="t1"
    )
    assert store.transition_epoch_cutover(
        cutover_id,
        from_states=("ready",),
        to_state="checked",
        updated_at="t2",
        fields={"rebuilt_through_event_id": through, "rebuild_job_id": "job"},
    )


class TestLifecycle:
    def test_initialize_creates_the_table_idempotently(self):
        store = _store()
        store.initialize()

        assert store.list_epoch_cutovers() == []

    def test_begin_opens_one_and_refuses_a_second(self):
        store = _store()

        first = _begin(store)
        second = _begin(store)

        assert first is not None and second is None
        row = store.get_open_epoch_cutover()
        assert (row["cutover_id"], row["state"], row["source_epoch_id"]) == (first, "filling", 1)
        assert len(store.list_epochs()) == 1

    def test_a_closed_cutover_does_not_block_the_next(self):
        store = _store()
        first = _begin(store)
        assert store.transition_epoch_cutover(
            first, from_states=("filling",), to_state="abandoned", updated_at="t1"
        )

        assert _begin(store) is not None

    def test_transition_is_compare_and_set(self):
        store = _store()
        cutover_id = _begin(store)

        assert not store.transition_epoch_cutover(
            cutover_id, from_states=("ready",), to_state="checked", updated_at="t1"
        )
        assert store.get_epoch_cutover(cutover_id)["state"] == "filling"

    @pytest.mark.parametrize("bad", ["promoted", "bogus"])
    def test_transition_refuses_promoted_and_unknown_states(self, bad):
        store = _store()
        cutover_id = _begin(store)

        with pytest.raises(ValueError):
            store.transition_epoch_cutover(
                cutover_id, from_states=("filling",), to_state=bad, updated_at="t1"
            )

    def test_transition_refuses_unknown_columns(self):
        store = _store()
        cutover_id = _begin(store)

        with pytest.raises(ValueError, match="not a writable"):
            store.transition_epoch_cutover(
                cutover_id,
                from_states=("filling",),
                to_state="ready",
                updated_at="t1",
                fields={"state": "promoted"},
            )


class TestPromote:
    def test_promote_appends_the_epoch_and_marks_through_the_rebuilt_turn(self):
        store = _store()
        ids = _log(store, 4)
        cutover_id = _begin(store)
        _checked(store, cutover_id, ids[1])

        result = store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")

        assert result == {"epoch_id": 2, "turns_at_activation": 4, "marked_applied": 2}
        epoch = store.get_current_epoch()
        assert (epoch["epoch_id"], epoch["model_hash"], epoch["prev_epoch_id"]) == (
            2,
            "m2|emb:e",
            1,
        )
        assert store.get_extraction_applied(2) == {ids[0]: "applied", ids[1]: "applied"}
        assert store.get_extraction_activation(2)["legacy_unextracted"] == 0
        row = store.get_epoch_cutover(cutover_id)
        assert (row["state"], row["promoted_epoch_id"]) == ("promoted", 2)
        assert store.get_open_epoch_cutover() is None

    def test_promote_refuses_an_unchecked_cutover(self):
        store = _store()
        _log(store, 1)
        cutover_id = _begin(store)

        with pytest.raises(EpochCutoverStateError, match="'filling'"):
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")
        assert len(store.list_epochs()) == 1

    def test_promote_refuses_when_the_ledger_moved_on(self):
        store = _store()
        ids = _log(store, 1)
        cutover_id = _begin(store)
        _checked(store, cutover_id, ids[0])
        store.append_epoch("1.4.0", "ev", "m9", activated_at="t2")

        with pytest.raises(EpochCutoverStateError, match="began from epoch 1"):
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")
        assert len(store.list_epochs()) == 2

    def test_promote_refuses_a_through_turn_missing_from_the_log(self):
        store = _store()
        _log(store, 1)
        cutover_id = _begin(store)
        _checked(store, cutover_id, "not-an-event")

        with pytest.raises(EpochCutoverStateError, match="not in the event log"):
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")
        assert store.get_epoch_cutover(cutover_id)["state"] == "checked"

    def test_a_crash_between_ledger_append_and_activation_writes_neither(
        self, tmp_path, monkeypatch
    ):
        # Arrange
        path = str(tmp_path / "event_store.db")
        store = _store(path)
        ids = _log(store, 3)
        cutover_id = _begin(store)
        _checked(store, cutover_id, ids[1])
        seen_mid_transaction: dict = {}

        def crash(step: str) -> None:
            # What another process (the dispatcher, or this one restarted)
            # sees at the instant between the two writes.
            other = EventStore(db_path=path)
            seen_mid_transaction["step"] = step
            seen_mid_transaction["epochs"] = len(other.list_epochs())
            seen_mid_transaction["activation"] = other.get_extraction_activation(2)
            other.close()
            raise SimulatedCrashError("process died after the ledger append")

        monkeypatch.setattr(store, "_promotion_fault_point", crash)

        # Act
        with pytest.raises(SimulatedCrashError):
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")

        # Assert: neither the ledger row nor the activation was ever visible,
        # and neither is on disk afterwards.
        assert seen_mid_transaction == {"step": "ledger_appended", "epochs": 1, "activation": None}
        store.close()
        reopened = EventStore(db_path=path)
        assert len(reopened.list_epochs()) == 1
        assert reopened.get_extraction_activation(2) is None
        assert reopened.get_extraction_applied(2) == {}
        assert reopened.get_epoch_cutover(cutover_id)["state"] == "checked"

        # And the promotion can simply be run again.
        result = reopened.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t4")
        assert (result["epoch_id"], result["marked_applied"]) == (2, 2)
        reopened.close()


PROBE = {
    "node_count": 32,
    "nodes_without_seed_version": 0,
    "nodes_with_extraction_stamp": 0,
    "relationship_count": 30,
    "relationships_without_seed_version": 0,
    "relationships_with_extraction_stamp": 0,
}


def _ready(store: EventStore, cutover_id: int) -> None:
    assert store.transition_epoch_cutover(
        cutover_id, from_states=("filling",), to_state="ready", updated_at="t1"
    )


def _snapshot(store: EventStore) -> dict[str, list[tuple]]:
    """Every row promotion could write, in a stable order."""
    conn = store._get_connection()
    queries = {
        "ledger": "SELECT * FROM epoch_ledger ORDER BY epoch_id",
        "activation": "SELECT * FROM extraction_activation ORDER BY epoch_id",
        "applied": "SELECT * FROM extraction_applied ORDER BY epoch_id, event_id",
        "legacy": "SELECT * FROM extraction_legacy_turns ORDER BY epoch_id, event_id",
        "cutover": "SELECT * FROM epoch_cutover ORDER BY cutover_id",
    }
    return {name: [tuple(row) for row in conn.execute(sql)] for name, sql in queries.items()}


def _seed_only(store: EventStore, cutover_id: int, activated_at: str = "t3") -> dict:
    return store.promote_epoch_cutover_seed_only(
        cutover_id=cutover_id, activated_at=activated_at, probe_report=PROBE
    )


class TestPromoteSeedOnly:
    def test_it_appends_the_epoch_and_an_activation_that_marks_nothing(self):
        store = _store()
        cutover_id = _begin(store)
        _ready(store, cutover_id)

        result = _seed_only(store, cutover_id)

        assert result == {"epoch_id": 2, "turns_at_activation": 0, "marked_applied": 0}
        epoch = store.get_current_epoch()
        assert (
            epoch["epoch_id"],
            epoch["ontology_version"],
            epoch["extraction_version"],
            epoch["model_hash"],
            epoch["prev_epoch_id"],
            epoch["provisional"],
            epoch["activated_at"],
        ) == (2, "1.4.0", "ev", "m2|emb:e", 1, 0, "t3")
        activation = store.get_extraction_activation(2)
        assert (
            activation["turns_at_activation"],
            activation["marked_applied"],
            activation["legacy_unextracted"],
        ) == (0, 0, 0)
        assert _snapshot(store)["applied"] == []
        assert _snapshot(store)["legacy"] == []
        row = store.get_epoch_cutover(cutover_id)
        assert (row["state"], row["promoted_epoch_id"], row["updated_at"]) == ("promoted", 2, "t3")
        assert json.loads(row["check_report"]) == {
            "promotion_mode": "seed_only_graph",
            "promoted_at": "t3",
            "graph_probe": PROBE,
            "prior_check_report": None,
        }
        assert store.get_open_epoch_cutover() is None

    def test_a_checked_cutovers_prior_report_is_kept_nested(self):
        # Over an empty log: the seed-only method does not read
        # `rebuilt_through_event_id`, so it need not name a logged turn.
        store = _store()
        cutover_id = _begin(store)
        _checked(store, cutover_id, "evt-none")
        assert store.transition_epoch_cutover(
            cutover_id,
            from_states=("checked",),
            to_state="checked",
            updated_at="t2",
            fields={"check_report": json.dumps({"passed": True, "exit_code": 0})},
        )

        _seed_only(store, cutover_id)

        report = json.loads(store.get_epoch_cutover(cutover_id)["check_report"])
        assert report["prior_check_report"] == {"passed": True, "exit_code": 0}
        assert report["promotion_mode"] == "seed_only_graph"

    def test_an_empty_log_promotes_with_zero_turns(self):
        store = _store()
        cutover_id = _begin(store)
        _ready(store, cutover_id)

        result = _seed_only(store, cutover_id)

        assert result == {"epoch_id": 2, "turns_at_activation": 0, "marked_applied": 0}

    def test_a_filling_cutover_is_refused_and_nothing_is_written(self):
        store = _store()
        _log(store, 1)
        cutover_id = _begin(store)
        before = _snapshot(store)

        with pytest.raises(EpochCutoverStateError, match="'filling'"):
            _seed_only(store, cutover_id)

        assert _snapshot(store) == before

    @pytest.mark.parametrize(
        ("epoch_id", "stage"),
        [(1, "applied"), (1, "curated"), (7, "applied")],
        ids=["applied-under-source", "curated-only", "applied-under-another-epoch"],
    )
    def test_any_apply_marker_under_any_epoch_is_refused_inside_the_transaction(
        self, epoch_id, stage
    ):
        store = _store()
        ids = _log(store, 2)
        cutover_id = _begin(store)
        _ready(store, cutover_id)
        store.mark_extraction_stage(
            event_id=ids[0], epoch_id=epoch_id, stage=stage, updated_at="t1"
        )
        before = _snapshot(store)

        with pytest.raises(EpochCutoverStateError, match=f"apply markers \\({stage}=1\\)"):
            _seed_only(store, cutover_id)

        assert _snapshot(store) == before

    def test_a_moved_ledger_is_refused_inside_the_transaction(self):
        store = _store()
        cutover_id = _begin(store)
        _ready(store, cutover_id)
        store.append_epoch("1.4.0", "ev", "m9", activated_at="t2")
        before = _snapshot(store)

        with pytest.raises(EpochCutoverStateError, match="began from epoch 1"):
            _seed_only(store, cutover_id)

        assert _snapshot(store) == before

    def test_a_crash_between_ledger_append_and_activation_writes_nothing(
        self, tmp_path, monkeypatch
    ):
        # Arrange
        path = str(tmp_path / "event_store.db")
        store = _store(path)
        cutover_id = _begin(store)
        _ready(store, cutover_id)
        before = _snapshot(store)
        seen: dict = {}

        def crash(step: str) -> None:
            other = EventStore(db_path=path)
            seen["step"] = step
            seen["epochs"] = len(other.list_epochs())
            seen["activation"] = other.get_extraction_activation(2)
            seen["state"] = other.get_epoch_cutover(cutover_id)["state"]
            other.close()
            raise SimulatedCrashError("process died after the ledger append")

        monkeypatch.setattr(store, "_promotion_fault_point", crash)

        # Act
        with pytest.raises(SimulatedCrashError):
            _seed_only(store, cutover_id)

        # Assert: nothing was visible mid-transaction and nothing is on disk.
        assert seen == {
            "step": "ledger_appended",
            "epochs": 1,
            "activation": None,
            "state": "ready",
        }
        store.close()
        reopened = EventStore(db_path=path)
        assert _snapshot(reopened) == before

        # And the promotion can simply be run again.
        result = _seed_only(reopened, cutover_id, activated_at="t4")
        assert result == {"epoch_id": 2, "turns_at_activation": 0, "marked_applied": 0}
        reopened.close()

    @pytest.mark.parametrize("origin", ["real", "test", "seed"])
    def test_a_logged_turn_from_any_origin_is_refused_inside_the_transaction(self, origin):
        store = _store()
        store.start_session("s-origin", input_modality="text", origin=origin)
        store.append_turn(
            ConversationTurnEvent(
                session_id="s-origin",
                turn_index=0,
                timestamp=T0,
                user_utterance="turn 0",
                system_response="ok",
                ontology_version="1.4.0",
            )
        )
        cutover_id = _begin(store)
        _ready(store, cutover_id)
        before = _snapshot(store)

        with pytest.raises(EpochCutoverStateError, match=r"holds 1 logged turn\(s\)"):
            _seed_only(store, cutover_id)

        assert _snapshot(store) == before
        assert store.get_epoch_cutover(cutover_id)["state"] == "ready"

    def test_a_turn_another_connection_commits_before_the_transaction_is_refused(self, tmp_path):
        # The backend's own EventStore (a second connection to the same file)
        # logs a turn after the caller's checks; BEGIN IMMEDIATE sees it.
        path = str(tmp_path / "event_store.db")
        store = _store(path)
        cutover_id = _begin(store)
        _ready(store, cutover_id)
        before = _snapshot(store)
        backend = EventStore(db_path=path)
        _log(backend, 2)
        backend.close()

        with pytest.raises(EpochCutoverStateError, match=r"holds 2 logged turn\(s\)"):
            _seed_only(store, cutover_id)

        after = _snapshot(store)
        assert after == before
        assert len(after["ledger"]) == 1
        store.close()

    def test_the_graph_swapped_promotion_is_unchanged_by_a_seed_only_sibling(self):
        # The graph-swapped method still refuses a 'ready' cutover: seed-only
        # promotion is a separate method, not a relaxation of the old one.
        store = _store()
        _log(store, 1)
        cutover_id = _begin(store)
        _ready(store, cutover_id)

        with pytest.raises(EpochCutoverStateError, match="only a 'checked' cutover"):
            store.promote_epoch_cutover(cutover_id=cutover_id, activated_at="t3")
