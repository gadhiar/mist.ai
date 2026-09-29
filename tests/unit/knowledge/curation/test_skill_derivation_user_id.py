"""Skill derivation must anchor the KNOWS edge on the canonical user node id."""

from datetime import UTC, datetime

import pytest

from backend.knowledge.config import SkillDerivationConfig
from backend.knowledge.curation.skill_derivation import SkillDerivationJob
from backend.knowledge.extraction.tool_usage_tracker import ToolCallRecord, ToolUsageTracker
from backend.knowledge.storage.partitions import USER_ENTITY_ID
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


@pytest.mark.asyncio
async def test_knows_edge_merges_on_the_canonical_user_entity_id():
    cfg = SkillDerivationConfig(
        skill_threshold=3,
        capability_threshold=5,
        lookback_days=7,
        similarity_threshold=0.7,
        window_size=100,
        enabled=True,
    )
    conn = FakeNeo4jConnection()
    tracker = ToolUsageTracker(config=cfg)
    job = SkillDerivationJob(
        tracker=tracker, executor=FakeGraphExecutor(connection=conn), config=cfg
    )
    for i in range(3):
        tracker.record(
            ToolCallRecord(
                tool_name="file_read",
                tool_type="file_management",
                context="reading source files",
                success=True,
                timestamp=datetime.now(UTC),
                session_id="sess-1",
                event_id=f"evt-{i}",
            )
        )

    await job.run()

    knows_writes = [(q, p) for q, p in conn.writes if "KNOWS" in q]
    assert len(knows_writes) == 1
    query, params = knows_writes[0]
    assert "{id: $user_entity_id}" in query
    assert params["user_entity_id"] == USER_ENTITY_ID == "user"
