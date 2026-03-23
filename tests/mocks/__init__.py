"""Shared test fakes and factories.

Import from here for convenience:
    from tests.mocks import FakeNeo4jConnection, FakeLLM, build_test_config
"""

from tests.mocks.config import (  # noqa: F401
    TEST_EVENT_ID,
    TEST_SESSION_ID,
    TEST_USER_ID,
    build_test_config,
)
from tests.mocks.embeddings import FakeEmbeddingGenerator  # noqa: F401
from tests.mocks.neo4j import (  # noqa: F401
    FakeGraphExecutor,
    FakeNeo4jConnection,
    FakeNeo4jRecord,
)
from tests.mocks.ollama import FakeLLM  # noqa: F401
