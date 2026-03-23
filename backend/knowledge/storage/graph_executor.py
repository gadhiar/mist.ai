"""Async boundary layer for synchronous graph database operations.

This module is the SINGLE point in the MIST.AI system where sync-to-async
bridging occurs via ``run_in_executor``.  All async callers (FastAPI handlers,
the extraction pipeline, etc.) go through GraphExecutor instead of touching
the underlying synchronous Neo4j driver directly.

If the project later migrates to an async Neo4j driver, only this module
needs to change -- every caller above remains async, and every implementation
below remains free to be sync or async as appropriate.
"""

import asyncio
import logging
from typing import Any

from backend.interfaces import GraphConnection

logger = logging.getLogger(__name__)


class GraphExecutor:
    """Async wrapper around a synchronous GraphConnection.

    This is the ONLY layer in the system that uses ``run_in_executor``.
    Everything above GraphExecutor is async.  Everything below is sync.
    """

    def __init__(self, connection: GraphConnection) -> None:
        """Initialize the executor with a sync graph connection.

        Args:
            connection: A synchronous GraphConnection implementation
                (e.g. Neo4jConnection).
        """
        self._connection = connection

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Run a read query against the graph database.

        Delegates to the underlying sync ``GraphConnection.execute_query``
        inside the default thread-pool executor so the caller's event loop
        is never blocked.

        Args:
            query: Cypher query string.
            params: Optional parameter dict bound into the query.

        Returns:
            List of result records, each as a dict.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._connection.execute_query, query, params)

    async def execute_write(self, query: str, params: dict[str, Any] | None = None) -> list[dict]:
        """Run a write query against the graph database.

        Delegates to the underlying sync ``GraphConnection.execute_write``
        inside the default thread-pool executor so the caller's event loop
        is never blocked.

        Args:
            query: Cypher query string.
            params: Optional parameter dict bound into the query.

        Returns:
            List of result records, each as a dict.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._connection.execute_write, query, params)

    @property
    def sync(self) -> GraphConnection:
        """Direct access to the underlying sync connection.

        Use this only in contexts that are already synchronous (CLI scripts,
        tests, migration utilities).  Async code should call
        ``execute_query`` / ``execute_write`` instead.
        """
        return self._connection
