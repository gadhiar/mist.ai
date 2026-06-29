"""Unit tests for re_extraction_jobs lifecycle on EventStore (R1.2)."""

from backend.event_store.store import EventStore


def _store() -> EventStore:
    store = EventStore(":memory:")
    store.initialize()
    return store


class TestReextractionJobs:
    def test_create_then_get_round_trips_fields(self):
        store = _store()
        store.create_reextraction_job(
            job_id="job-1",
            target_ontology_version="1.4.0",
            source_ontology_version="1.3.0",
            total_events=42,
            started_at="2026-06-29T12:00:00+00:00",
        )
        job = store.get_reextraction_job("job-1")
        assert job is not None
        assert job["job_id"] == "job-1"
        assert job["target_ontology_version"] == "1.4.0"
        assert job["source_ontology_version"] == "1.3.0"
        assert job["total_events"] == 42
        assert job["status"] == "running"
        assert job["processed"] == 0
        assert job["last_event_id"] is None

    def test_checkpoint_advances_cursor_and_processed(self):
        store = _store()
        store.create_reextraction_job("job-1", "1.4.0", None, 2, "2026-06-29T12:00:00+00:00")
        store.checkpoint_reextraction_job("job-1", "evt-7", 1, "2026-06-29T12:00:01+00:00")
        job = store.get_reextraction_job("job-1")
        assert job["last_event_id"] == "evt-7"
        assert job["processed"] == 1

    def test_get_missing_job_returns_none(self):
        store = _store()
        assert store.get_reextraction_job("nope") is None

    def test_finalize_sets_completed_status(self):
        store = _store()
        store.create_reextraction_job("job-1", "1.4.0", None, 5, "2026-06-29T12:00:00+00:00")
        store.finalize_reextraction_job(
            job_id="job-1",
            status="completed",
            failed=0,
            errors=None,
            updated_at="2026-06-29T12:01:00+00:00",
        )
        job = store.get_reextraction_job("job-1")
        assert job["status"] == "completed"
        assert job["failed"] == 0
        assert job["errors"] is None
        assert job["updated_at"] == "2026-06-29T12:01:00+00:00"

    def test_finalize_sets_failed_status_with_errors(self):
        store = _store()
        store.create_reextraction_job("job-1", "1.4.0", None, 5, "2026-06-29T12:00:00+00:00")
        errors_json = '["Dedup failed: connection error", "Graph write failed: timeout"]'
        store.finalize_reextraction_job(
            job_id="job-1",
            status="failed",
            failed=2,
            errors=errors_json,
            updated_at="2026-06-29T12:01:30+00:00",
        )
        job = store.get_reextraction_job("job-1")
        assert job["status"] == "failed"
        assert job["failed"] == 2
        assert job["errors"] == errors_json
        assert job["updated_at"] == "2026-06-29T12:01:30+00:00"
