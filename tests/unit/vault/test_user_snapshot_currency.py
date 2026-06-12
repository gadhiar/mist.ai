"""C-pattern user snapshot reads only current beliefs (read-path-currency-2).

Post-C1 the graph retains every edge version. Without the canonical 4-arm
filter, the users/<id>-graph-snapshot.md mirror renders superseded versions,
clamped history copies, RETRACT tombstones, and orphaned vault-edit edges as
identical current profile lines.
"""

import pytest

from backend.vault.user_snapshot import query_user_snapshot
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

RENDERED_AT = "2026-06-12T00:00:00+00:00"


def _conn_with_user_node() -> FakeNeo4jConnection:
    # The user-node query must return a row or the neighbor query never runs.
    return FakeNeo4jConnection(query_responses={"AS attrs": [{"display_name": "Raj", "attrs": {}}]})


def _neighbor_calls(conn: FakeNeo4jConnection) -> list[tuple[str, dict]]:
    return [(q, p) for q, p in conn.queries if "rel_type" in q]


class TestSnapshotCurrencyFilter:
    @pytest.mark.asyncio
    async def test_neighbor_query_filters_to_current_beliefs(self):
        conn = _conn_with_user_node()

        await query_user_snapshot(FakeGraphExecutor(conn), "user", RENDERED_AT)

        calls = _neighbor_calls(conn)
        assert calls, "neighbor query must run when the user node exists"
        query, _ = calls[0]
        assert "r.status IS NULL OR r.status <> 'orphaned'" in query
        assert "coalesce(r.is_latest_belief, true)" in query
        assert "r.valid_to IS NULL OR r.valid_to > $now" in query
        assert "r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now" in query

    @pytest.mark.asyncio
    async def test_now_binds_to_rendered_at_for_determinism(self):
        # The function is documented deterministic for tests: currency is
        # evaluated at the same instant the snapshot is stamped, not at a
        # second wall-clock read.
        conn = _conn_with_user_node()

        await query_user_snapshot(FakeGraphExecutor(conn), "user", RENDERED_AT)

        _, params = _neighbor_calls(conn)[0]
        assert params["now"] == RENDERED_AT
