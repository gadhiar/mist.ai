"""
Neo4j Database Connection

Handles connection to Neo4j database with automatic retries and health checks.
"""

import logging

from neo4j import Driver, GraphDatabase

from backend.knowledge.config import Neo4jConfig

logger = logging.getLogger(__name__)


class Neo4jConnection:
    """
    Neo4j database connection manager

    Handles connection lifecycle, health checks, and query execution.
    """

    def __init__(self, config: Neo4jConfig):
        """
        Initialize Neo4j connection

        Args:
            config: Neo4j configuration
        """
        self.config = config
        self._driver: Driver | None = None

    def connect(self) -> Driver:
        """
        Establish connection to Neo4j database

        Returns:
            Neo4j driver instance

        Raises:
            Exception: If connection fails
        """
        if self._driver is None:
            logger.info(f"Connecting to Neo4j at {self.config.uri}")

            try:
                self._driver = GraphDatabase.driver(
                    self.config.uri, auth=(self.config.username, self.config.password)
                )

                # Verify connectivity
                self._driver.verify_connectivity()
                logger.info(" Successfully connected to Neo4j")

            except Exception as e:
                logger.error(f" Failed to connect to Neo4j: {e}")
                raise

        return self._driver

    def disconnect(self):
        """Close connection to Neo4j database"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.info("Disconnected from Neo4j")

    def is_connected(self) -> bool:
        """Check if connected to Neo4j"""
        if self._driver is None:
            return False

        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def execute_query(self, query: str, parameters: dict | None = None) -> list:
        """
        Execute a Cypher query

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of query results
        """
        if self._driver is None:
            self.connect()

        with self._driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            return list(result)

    def execute_write(self, query: str, parameters: dict | None = None):
        """
        Execute a write transaction

        Args:
            query: Cypher query string
            parameters: Query parameters
        """
        if self._driver is None:
            self.connect()

        with self._driver.session(database=self.config.database) as session:
            session.execute_write(lambda tx: tx.run(query, parameters or {}))

    def health_check(self) -> dict:
        """
        Perform health check on Neo4j connection

        Returns:
            Health check results
        """
        try:
            # Test basic connectivity
            connected = self.is_connected()

            if not connected:
                return {
                    "status": "unhealthy",
                    "connected": False,
                    "message": "Not connected to Neo4j",
                }

            # Test query execution
            self.execute_query("RETURN 1 AS test")

            return {
                "status": "healthy",
                "connected": True,
                "database": self.config.database,
                "uri": self.config.uri,
                "message": "Neo4j connection is healthy",
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "message": f"Health check failed: {e}",
            }

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
