"""Unit tests for GraphExecutor.

Uses FakeNeo4jConnection via DI -- no patching needed.
All tests are async because GraphExecutor wraps sync calls
in run_in_executor.
"""

import pytest

from backend.knowledge.storage.graph_executor import GraphExecutor
from tests.mocks.neo4j import FakeNeo4jConnection

# -------------------------------------------------------------------
# execute_query()
# -------------------------------------------------------------------


class TestExecuteQuery:
    @pytest.mark.asyncio
    async def test_execute_query_delegates_to_connection(self):
        fake_conn = FakeNeo4jConnection(query_results=[{"n": 1}])
        executor = GraphExecutor(connection=fake_conn)

        result = await executor.execute_query("MATCH (n) RETURN n", {"limit": 5})

        assert result == [{"n": 1}]
        assert len(fake_conn.queries) == 1
        query, params = fake_conn.queries[0]
        assert query == "MATCH (n) RETURN n"
        assert params == {"limit": 5}


# -------------------------------------------------------------------
# execute_write()
# -------------------------------------------------------------------


class TestExecuteWrite:
    @pytest.mark.asyncio
    async def test_execute_write_delegates_to_connection(self):
        fake_conn = FakeNeo4jConnection(write_results=[{"created": True}])
        executor = GraphExecutor(connection=fake_conn)

        result = await executor.execute_write("CREATE (n:Test)", {"name": "x"})

        assert result == [{"created": True}]
        assert len(fake_conn.writes) == 1
        query, params = fake_conn.writes[0]
        assert query == "CREATE (n:Test)"
        assert params == {"name": "x"}


# -------------------------------------------------------------------
# sync property
# -------------------------------------------------------------------


class TestSyncProperty:
    def test_sync_property_returns_connection(self):
        fake_conn = FakeNeo4jConnection()
        executor = GraphExecutor(connection=fake_conn)

        assert executor.sync is fake_conn
