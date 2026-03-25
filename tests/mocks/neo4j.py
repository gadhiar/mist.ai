"""Test doubles for Neo4j graph database operations.

These fakes satisfy the GraphConnection protocol from backend.interfaces.
Use them in unit tests instead of real Neo4j connections.
"""


class FakeNeo4jRecord:
    """Simulates a Neo4j record with dict-like access."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data.get(key, default)

    def data(self):
        return self._data


class FakeNeo4jConnection:
    """Test double for Neo4jConnection. Satisfies GraphConnection protocol."""

    def __init__(
        self,
        *,
        query_results=None,
        write_results=None,
        query_responses=None,
        write_errors=None,
        query_errors=None,
    ):
        self._query_results = query_results or []
        self._write_results = write_results or []
        self._query_responses = query_responses or {}
        # Pattern -> exception: raise Neo4jQueryError when pattern appears in write query.
        self._write_errors: dict[str, Exception] = write_errors or {}
        # Pattern -> exception: raise Neo4jQueryError when pattern appears in read query.
        self._query_errors: dict[str, Exception] = query_errors or {}
        self.queries: list[tuple[str, dict | None]] = []
        self.writes: list[tuple[str, dict | None]] = []
        self._connected = True

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def execute_query(self, query, params=None):
        self.queries.append((query, params))
        for pattern, exc in self._query_errors.items():
            if pattern in query:
                raise exc
        for pattern, results in self._query_responses.items():
            if pattern in query:
                return results
        return self._query_results

    def execute_write(self, query, params=None):
        self.writes.append((query, params))
        for pattern, exc in self._write_errors.items():
            if pattern in query:
                raise exc
        return self._write_results

    def assert_query_executed(self, pattern: str):
        """Assert a query matching the pattern was executed."""
        for query, _ in self.queries:
            if pattern in query:
                return
        raise AssertionError(
            f"No query matching '{pattern}' executed. " f"Got: {[q for q, _ in self.queries]}"
        )

    def assert_write_executed(self, pattern: str):
        """Assert a write query matching the pattern was executed."""
        for query, _ in self.writes:
            if pattern in query:
                return
        raise AssertionError(
            f"No write matching '{pattern}' executed. " f"Got: {[q for q, _ in self.writes]}"
        )

    def assert_no_writes(self):
        """Assert no write operations were performed."""
        if self.writes:
            raise AssertionError(f"Expected no writes, but got {len(self.writes)}: {self.writes}")


class FakeGraphExecutor:
    """Async test double for GraphExecutor."""

    def __init__(self, connection=None):
        self._connection = connection or FakeNeo4jConnection()

    async def execute_query(self, query, params=None):
        return self._connection.execute_query(query, params)

    async def execute_write(self, query, params=None):
        return self._connection.execute_write(query, params)

    @property
    def sync(self):
        return self._connection
