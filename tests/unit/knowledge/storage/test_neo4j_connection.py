"""Unit tests for Neo4jConnection.

Neo4jConnection is a leaf dependency wrapping the Neo4j driver.
We patch GraphDatabase.driver here because there is no DI seam --
the driver is created inside connect().
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.errors import Neo4jConnectionError, Neo4jQueryError
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

MODULE = "backend.knowledge.storage.neo4j_connection"


def _make_config(**overrides) -> Neo4jConfig:
    """Build a Neo4jConfig with test defaults."""
    defaults = {
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "test-password",
        "database": "testdb",
    }
    defaults.update(overrides)
    return Neo4jConfig(**defaults)


# -------------------------------------------------------------------
# connect()
# -------------------------------------------------------------------


class TestConnect:
    @patch(f"{MODULE}.GraphDatabase")
    def test_connect_creates_driver_and_verifies_connectivity(self, mock_gdb):
        config = _make_config()
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        conn = Neo4jConnection(config)

        conn.connect()

        mock_gdb.driver.assert_called_once_with(config.uri, auth=(config.username, config.password))
        mock_driver.verify_connectivity.assert_called_once()

    @patch(f"{MODULE}.GraphDatabase")
    def test_connect_raises_neo4j_connection_error_on_failure(self, mock_gdb):
        from neo4j.exceptions import ServiceUnavailable

        mock_gdb.driver.side_effect = ServiceUnavailable("cannot reach host")
        conn = Neo4jConnection(_make_config())

        with pytest.raises(Neo4jConnectionError, match="Failed to connect"):
            conn.connect()


# -------------------------------------------------------------------
# execute_query()
# -------------------------------------------------------------------


class TestExecuteQuery:
    @patch(f"{MODULE}.GraphDatabase")
    def test_execute_query_returns_list_of_dicts(self, mock_gdb):
        # Arrange
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        mock_record = MagicMock()
        mock_record.data.return_value = {"name": "Alice"}
        mock_session.run.return_value = [mock_record]

        conn = Neo4jConnection(_make_config())
        conn.connect()

        # Act
        result = conn.execute_query("MATCH (n) RETURN n.name AS name", {"limit": 1})

        # Assert
        assert result == [{"name": "Alice"}]
        mock_session.run.assert_called_once_with("MATCH (n) RETURN n.name AS name", {"limit": 1})

    @patch(f"{MODULE}.GraphDatabase")
    def test_execute_query_raises_neo4j_query_error_on_failure(self, mock_gdb):
        from neo4j.exceptions import Neo4jError

        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.run.side_effect = Neo4jError("syntax error")

        conn = Neo4jConnection(_make_config())
        conn.connect()

        with pytest.raises(Neo4jQueryError, match="Query execution failed"):
            conn.execute_query("INVALID CYPHER")


# -------------------------------------------------------------------
# execute_write()
# -------------------------------------------------------------------


class TestExecuteWrite:
    @patch(f"{MODULE}.GraphDatabase")
    def test_execute_write_raises_neo4j_query_error_on_failure(self, mock_gdb):
        from neo4j.exceptions import Neo4jError

        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.execute_write.side_effect = Neo4jError("write failed")

        conn = Neo4jConnection(_make_config())
        conn.connect()

        with pytest.raises(Neo4jQueryError, match="Write transaction failed"):
            conn.execute_write("CREATE (n:Test)")


# -------------------------------------------------------------------
# is_connected()
# -------------------------------------------------------------------


class TestIsConnected:
    def test_is_connected_returns_false_when_no_driver(self):
        conn = Neo4jConnection(_make_config())

        assert conn.is_connected() is False


# -------------------------------------------------------------------
# disconnect()
# -------------------------------------------------------------------


class TestDisconnect:
    @patch(f"{MODULE}.GraphDatabase")
    def test_disconnect_closes_driver(self, mock_gdb):
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        conn = Neo4jConnection(_make_config())
        conn.connect()

        conn.disconnect()

        mock_driver.close.assert_called_once()
        assert conn._driver is None
