"""Reconciliation phase records on DebugJSONLLogger (C2, design 8.4)."""

import json

from backend.debug_jsonl_logger import DebugJSONLLogger


class TestReconciliationPhase:
    def test_disabled_without_gate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(tmp_path / "d.jsonl"))
        monkeypatch.delenv("MIST_DEBUG_RECONCILIATION_JSONL", raising=False)
        logger = DebugJSONLLogger.from_env()
        assert logger.reconciliation_enabled is False
        logger.record_reconciliation(
            event_id="e1",
            session_id="s1",
            predicate="USES",
            source="user",
            target="rust",
            action="append_version",
            reason="assert",
            edge_ref=None,
            valid_from=None,
            valid_to=None,
        )
        assert not (tmp_path / "d.jsonl").exists()

    def test_emits_record_when_gated_on(self, tmp_path, monkeypatch):
        path = tmp_path / "d.jsonl"
        monkeypatch.setenv("MIST_DEBUG_JSONL", str(path))
        monkeypatch.setenv("MIST_DEBUG_RECONCILIATION_JSONL", "1")
        logger = DebugJSONLLogger.from_env()
        assert logger.reconciliation_enabled is True
        logger.record_reconciliation(
            event_id="e1",
            session_id="s1",
            predicate="WORKS_AT",
            source="user",
            target="initech",
            action="close_transaction",
            reason="single_supersession",
            edge_ref="ref-1",
            valid_from=None,
            valid_to="2026-06-10T12:00:00+00:00",
        )
        rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert rec["phase"] == "reconciliation"
        assert rec["action"] == "close_transaction"
        assert rec["reason"] == "single_supersession"
        assert rec["event_id"] == "e1"
