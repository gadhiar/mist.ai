"""Unit tests for the rebuild target-isolation guard (R1.2)."""

import pytest

from backend.knowledge.eval_isolation import RebuildTargetError, assert_rebuild_target_not_live


class TestRebuildTargetGuard:
    def test_allows_distinct_staging_target(self):
        # Different host:port -> allowed (no raise).
        assert_rebuild_target_not_live(
            target_uri="bolt://mist-neo4j-staging:7687",
            live_uri="bolt://mist-neo4j:7687",
        )

    def test_rejects_target_equal_to_live_host_port(self):
        with pytest.raises(RebuildTargetError, match="live"):
            assert_rebuild_target_not_live(
                target_uri="bolt://mist-neo4j:7687",
                live_uri="bolt://mist-neo4j:7687",
            )

    def test_rejects_host_published_live_port_alias(self):
        # localhost:7687 is the host-published live bolt port -> must be rejected
        # even though the hostname differs from the in-network service name.
        with pytest.raises(RebuildTargetError):
            assert_rebuild_target_not_live(
                target_uri="bolt://localhost:7687",
                live_uri="bolt://localhost:7687",
            )

    def test_rejects_unparseable_target(self):
        with pytest.raises(RebuildTargetError, match="parse"):
            assert_rebuild_target_not_live(
                target_uri="not-a-uri", live_uri="bolt://mist-neo4j:7687"
            )
