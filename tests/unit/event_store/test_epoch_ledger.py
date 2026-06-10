"""Append-only epoch ledger on EventStore (F3)."""

from backend.event_store.store import EventStore


def _store(tmp_path):
    store = EventStore(str(tmp_path / "es.db"))
    store.initialize()
    return store


class TestEpochLedger:
    def test_append_then_get_current(self, tmp_path):
        store = _store(tmp_path)
        eid = store.append_epoch(
            "1.1.0", "2026-05-06-r1", "gemma-x", activated_at="2026-06-10T00:00:00Z"
        )
        current = store.get_current_epoch()
        assert current is not None
        assert current["epoch_id"] == eid
        assert current["ontology_version"] == "1.1.0"
        assert current["prev_epoch_id"] is None
        store.close()

    def test_append_same_triple_is_idempotent(self, tmp_path):
        store = _store(tmp_path)
        id1 = store.append_epoch("1.1.0", "v1", "m1", activated_at="t1")
        id2 = store.append_epoch("1.1.0", "v1", "m1", activated_at="t2")
        assert id2 == id1
        assert len(store.list_epochs()) == 1
        store.close()

    def test_changed_triple_appends_with_prev_pointer(self, tmp_path):
        store = _store(tmp_path)
        id1 = store.append_epoch("1.1.0", "v1", "m1", activated_at="t1")
        id3 = store.append_epoch("1.2.0", "v1", "m1", activated_at="t2")
        assert id3 != id1
        epochs = store.list_epochs()
        assert len(epochs) == 2
        current = store.get_current_epoch()
        assert current["ontology_version"] == "1.2.0"
        assert current["prev_epoch_id"] == id1
        store.close()

    def test_get_current_returns_none_when_empty(self, tmp_path):
        store = _store(tmp_path)
        assert store.get_current_epoch() is None
        store.close()
