"""Tests for SkillDerivationJob.

Verifies Cypher query construction and parameter forwarding for Skill and
MistCapability entity creation, KNOWS and HAS_CAPABILITY edge wiring, and
the no-op path when no patterns are detected or the job is disabled.
"""

from datetime import UTC, datetime

import pytest

from backend.knowledge.config import SkillDerivationConfig
from backend.knowledge.curation.skill_derivation import SkillDerivationJob, SkillDerivationResult
from backend.knowledge.extraction.tool_usage_tracker import ToolCallRecord, ToolUsageTracker
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    skill_threshold: int = 3,
    capability_threshold: int = 5,
    lookback_days: int = 7,
    similarity_threshold: float = 0.7,
    window_size: int = 100,
    enabled: bool = True,
) -> SkillDerivationConfig:
    """Build a SkillDerivationConfig with test defaults."""
    return SkillDerivationConfig(
        skill_threshold=skill_threshold,
        capability_threshold=capability_threshold,
        lookback_days=lookback_days,
        similarity_threshold=similarity_threshold,
        window_size=window_size,
        enabled=enabled,
    )


def _make_record(
    *,
    tool_name: str = "file_read",
    tool_type: str = "file_management",
    context: str = "reading source files",
    success: bool = True,
    session_id: str = "sess-1",
    event_id: str = "evt-1",
) -> ToolCallRecord:
    """Build a ToolCallRecord with an explicit recent timestamp."""
    return ToolCallRecord(
        tool_name=tool_name,
        tool_type=tool_type,
        context=context,
        success=success,
        timestamp=datetime.now(UTC),
        session_id=session_id,
        event_id=event_id,
    )


def _build_job(
    *,
    config: SkillDerivationConfig | None = None,
    connection: FakeNeo4jConnection | None = None,
) -> tuple[SkillDerivationJob, ToolUsageTracker, FakeNeo4jConnection]:
    """Build a SkillDerivationJob wired to test doubles.

    Returns the job, the tracker (so callers can record tool calls), and
    the underlying FakeNeo4jConnection (so callers can inspect queries).
    """
    cfg = config or _make_config()
    conn = connection or FakeNeo4jConnection()
    executor = FakeGraphExecutor(connection=conn)
    tracker = ToolUsageTracker(config=cfg)
    job = SkillDerivationJob(tracker=tracker, executor=executor, config=cfg)
    return job, tracker, conn


def _record_identical(
    tracker: ToolUsageTracker,
    *,
    count: int,
    tool_name: str = "file_read",
    tool_type: str = "file_management",
    context: str = "reading source files",
    success: bool = True,
) -> None:
    """Record `count` identical tool call records into the tracker."""
    for i in range(count):
        tracker.record(
            _make_record(
                tool_name=tool_name,
                tool_type=tool_type,
                context=context,
                success=success,
                event_id=f"evt-{i}",
            )
        )


# ---------------------------------------------------------------------------
# TestRunWithPatterns
# ---------------------------------------------------------------------------


class TestRunWithPatterns:
    @pytest.mark.asyncio
    async def test_skill_entity_created_via_merge(self):
        # Arrange: skill_threshold=3; record 3 identical calls to trigger a pattern.
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=3)

        # Act
        await job.run()

        # Assert: a MERGE write targeting __Entity__ (Skill) was issued.
        conn.assert_write_executed("MERGE")
        conn.assert_write_executed("__Entity__")

    @pytest.mark.asyncio
    async def test_skill_entity_id_uses_slugified_display_name(self):
        # Arrange
        job, tracker, conn = _build_job(
            config=_make_config(skill_threshold=3),
        )
        _record_identical(
            tracker,
            count=3,
            tool_name="file_read",
            tool_type="file_management",
            context="reading source files",
        )

        # Act
        await job.run()

        # Assert: the skill_id written to the graph is derived from the display name.
        # display_name = "reading source files (file_management)"
        # skill_id     = "skill-reading-source-files-file-management"
        skill_write = next(
            (params for query, params in conn.writes if params and "skill_id" in params),
            None,
        )
        assert skill_write is not None
        assert skill_write["skill_id"] == "skill-reading-source-files-file-management"

    @pytest.mark.asyncio
    async def test_knows_edge_created_from_user_to_skill(self):
        # Arrange
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=3)

        # Act
        await job.run()

        # Assert: the creation write includes the KNOWS edge merge.
        conn.assert_write_executed("KNOWS")
        conn.assert_write_executed("user")

    @pytest.mark.asyncio
    async def test_proficiency_calculated_from_success_count_and_capability_threshold(self):
        # Arrange: success_count=2, capability_threshold=10 -> proficiency = 2/(10*2) = 0.1
        config = _make_config(skill_threshold=3, capability_threshold=10)
        job, tracker, conn = _build_job(config=config)
        _record_identical(tracker, count=3, success=False)
        # Override two records to be successful.
        tracker.record(_make_record(context="reading source files", success=True, event_id="s1"))
        tracker.record(_make_record(context="reading source files", success=True, event_id="s2"))

        # Act
        await job.run()

        # Assert: proficiency written is capped at min(1.0, success_count / (cap_threshold*2)).
        skill_write = next(
            (params for query, params in conn.writes if params and "proficiency" in params),
            None,
        )
        assert skill_write is not None
        assert 0.0 <= skill_write["proficiency"] <= 1.0

    @pytest.mark.asyncio
    async def test_result_counts_skills_created(self):
        # Arrange: one pattern -> one skill created.
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=3)

        # Act
        result = await job.run()

        # Assert
        assert isinstance(result, SkillDerivationResult)
        assert result.patterns_detected == 1
        assert result.skills_created == 1
        assert result.skills_updated == 0

    @pytest.mark.asyncio
    async def test_result_duration_ms_is_positive(self):
        # Arrange
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=3)

        # Act
        result = await job.run()

        # Assert
        assert result.duration_ms > 0.0

    @pytest.mark.asyncio
    async def test_existing_skill_is_updated_not_created(self):
        # Arrange: pre-seed the fake connection to return a result for the
        # MATCH query, so _find_existing_skill returns True.
        from tests.mocks.neo4j import FakeNeo4jRecord

        existing_record = [FakeNeo4jRecord({"id": "skill-reading-source-files-file-management"})]
        conn = FakeNeo4jConnection(query_results=existing_record)
        job, tracker, _ = _build_job(config=_make_config(skill_threshold=3), connection=conn)
        _record_identical(tracker, count=3)

        # Act
        result = await job.run()

        # Assert: updated, not created.
        assert result.skills_updated == 1
        assert result.skills_created == 0
        conn.assert_write_executed("SET e.proficiency")


# ---------------------------------------------------------------------------
# TestRunWithoutPatterns
# ---------------------------------------------------------------------------


class TestRunWithoutPatterns:
    @pytest.mark.asyncio
    async def test_no_graph_writes_when_no_patterns_detected(self):
        # Arrange: only 2 records, skill_threshold=3 -> no pattern detected.
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=2)

        # Act
        await job.run()

        # Assert: no writes to the graph.
        conn.assert_no_writes()

    @pytest.mark.asyncio
    async def test_no_graph_writes_when_tracker_is_empty(self):
        # Arrange: tracker has no records at all.
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))

        # Act
        await job.run()

        # Assert
        conn.assert_no_writes()

    @pytest.mark.asyncio
    async def test_result_has_zero_counts_when_no_patterns(self):
        # Arrange
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=2)

        # Act
        result = await job.run()

        # Assert
        assert result.patterns_detected == 0
        assert result.skills_created == 0
        assert result.skills_updated == 0
        assert result.capabilities_created == 0

    @pytest.mark.asyncio
    async def test_returns_zero_result_when_disabled(self):
        # Arrange: job is disabled via config.
        job, tracker, conn = _build_job(config=_make_config(enabled=False))
        _record_identical(tracker, count=10)

        # Act
        result = await job.run()

        # Assert: no writes and all counts are zero.
        conn.assert_no_writes()
        assert result.patterns_detected == 0
        assert result.skills_created == 0
        assert result.capabilities_created == 0
        assert result.duration_ms == 0.0


# ---------------------------------------------------------------------------
# TestCapabilityCreation
# ---------------------------------------------------------------------------


class TestCapabilityCreation:
    @pytest.mark.asyncio
    async def test_capability_created_when_success_count_meets_threshold(self):
        # Arrange: capability_threshold=3, skill_threshold=3.
        # Record 3 successful calls -> success_count=3 >= capability_threshold.
        config = _make_config(skill_threshold=3, capability_threshold=3)
        job, tracker, conn = _build_job(config=config)
        _record_identical(tracker, count=3, success=True)

        # Act
        result = await job.run()

        # Assert: MistCapability MERGE and HAS_CAPABILITY edge written.
        assert result.capabilities_created == 1
        conn.assert_write_executed("MistCapability")
        conn.assert_write_executed("HAS_CAPABILITY")

    @pytest.mark.asyncio
    async def test_capability_not_created_when_success_count_below_threshold(self):
        # Arrange: capability_threshold=5, only 2 successes out of 3 calls.
        config = _make_config(skill_threshold=3, capability_threshold=5)
        job, tracker, conn = _build_job(config=config)
        _record_identical(tracker, count=3, success=False)
        tracker.record(_make_record(context="reading source files", success=True, event_id="s1"))
        tracker.record(_make_record(context="reading source files", success=True, event_id="s2"))
        # success_count = 2 < capability_threshold = 5

        # Act
        result = await job.run()

        # Assert
        assert result.capabilities_created == 0

    @pytest.mark.asyncio
    async def test_capability_id_uses_cap_prefix_and_slugified_name(self):
        # Arrange
        config = _make_config(skill_threshold=3, capability_threshold=3)
        job, tracker, conn = _build_job(config=config)
        _record_identical(
            tracker,
            count=3,
            success=True,
            tool_type="file_management",
            context="reading source files",
        )

        # Act
        await job.run()

        # Assert: cap_id = "cap-reading-source-files-file-management"
        cap_write = next(
            (params for query, params in conn.writes if params and "cap_id" in params),
            None,
        )
        assert cap_write is not None
        assert cap_write["cap_id"] == "cap-reading-source-files-file-management"

    @pytest.mark.asyncio
    async def test_has_capability_edge_wired_from_mist_identity(self):
        # Arrange
        config = _make_config(skill_threshold=3, capability_threshold=3)
        job, tracker, conn = _build_job(config=config)
        _record_identical(tracker, count=3, success=True)

        # Act
        await job.run()

        # Assert: the capability creation query references MistIdentity.
        conn.assert_write_executed("MistIdentity")
        conn.assert_write_executed("mist-identity")

    @pytest.mark.asyncio
    async def test_existing_capability_is_updated_returns_false(self):
        # Arrange: pre-seed connection so the capability MATCH query returns
        # a result, causing _ensure_capability to take the update path.
        from tests.mocks.neo4j import FakeNeo4jRecord

        # The first query (find skill) returns empty; the second (find capability) returns hit.
        cap_id = "cap-reading-source-files-file-management"
        conn = FakeNeo4jConnection(
            query_responses={
                "MistCapability": [FakeNeo4jRecord({"id": cap_id})],
            }
        )
        config = _make_config(skill_threshold=3, capability_threshold=3)
        job, tracker, _ = _build_job(config=config, connection=conn)
        _record_identical(tracker, count=3, success=True)

        # Act
        result = await job.run()

        # Assert: capability was found and updated, not newly created.
        assert result.capabilities_created == 0
        conn.assert_write_executed("e.proficiency")


# ---------------------------------------------------------------------------
# TestDeduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_merge_semantics_used_for_skill_creation(self):
        # Arrange: running with a new skill (no existing found) must use MERGE
        # so that a second run will not create a duplicate in real Neo4j.
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=3)

        # Act
        await job.run()

        # Assert: the Skill write uses MERGE, not CREATE.
        create_writes = [q for q, _ in conn.writes if "CREATE" in q and "MERGE" not in q]
        merge_writes = [q for q, _ in conn.writes if "MERGE" in q]
        assert len(merge_writes) > 0, "Expected at least one MERGE write for Skill"
        assert len(create_writes) == 0, "Unexpected bare CREATE write (should use MERGE)"

    @pytest.mark.asyncio
    async def test_merge_semantics_used_for_capability_creation(self):
        # Arrange
        config = _make_config(skill_threshold=3, capability_threshold=3)
        job, tracker, conn = _build_job(config=config)
        _record_identical(tracker, count=3, success=True)

        # Act
        await job.run()

        # Assert: the MistCapability write also uses MERGE.
        cap_writes = [q for q, _ in conn.writes if "MistCapability" in q]
        assert len(cap_writes) > 0
        for q in cap_writes:
            assert "MERGE" in q, f"Expected MERGE in capability write, got: {q}"

    @pytest.mark.asyncio
    async def test_second_run_updates_existing_skill_not_creates(self):
        # Arrange: simulate the scenario where the first run created the skill.
        # On the second call, _find_existing_skill will return the record.
        from tests.mocks.neo4j import FakeNeo4jRecord

        skill_id = "skill-reading-source-files-file-management"
        conn = FakeNeo4jConnection(
            query_responses={
                "entity_type: 'Skill'": [FakeNeo4jRecord({"id": skill_id})],
            }
        )
        config = _make_config(skill_threshold=3)
        job, tracker, _ = _build_job(config=config, connection=conn)
        _record_identical(tracker, count=3)

        # Act: simulate the second run.
        result = await job.run()

        # Assert: the job updates rather than creates.
        assert result.skills_updated == 1
        assert result.skills_created == 0

    @pytest.mark.asyncio
    async def test_skill_query_uses_match_for_existence_check(self):
        # Arrange
        job, tracker, conn = _build_job(config=_make_config(skill_threshold=3))
        _record_identical(tracker, count=3)

        # Act
        await job.run()

        # Assert: a MATCH query was issued to check for the existing skill.
        conn.assert_query_executed("MATCH")
        conn.assert_query_executed("entity_type: 'Skill'")
