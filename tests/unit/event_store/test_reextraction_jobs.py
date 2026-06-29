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
