# MIST.AI Testing Conventions

Project-wide testing standards and reference for the MIST.AI backend test suite.

---

## Philosophy

The goal is NOT 100% code coverage. The goal is **high-signal tests** that guard against
real regressions.

Principles:

- **Every test guards a real regression.** If you cannot describe the bug this test would
  catch, the test has no value.
- **High-signal over high-count.** One test that verifies correct forwarding of a critical
  parameter is worth more than ten tests that assert `result is not None`.
- **Mock only at I/O boundaries.** Internal Python logic runs as-is. Fakes replace Neo4j,
  Ollama, embeddings, and filesystem I/O.
- **Deterministic always.** No randomness, no time-dependence, no execution-order
  dependence.

---

## Test Tiers

### `unit/` -- Fast, Isolated, All I/O Faked

- No external dependencies (no Neo4j, no Ollama, no network).
- All I/O replaced by fakes from `tests/mocks/`.
- Target: entire suite runs in under 30 seconds.
- Run with: `pytest tests/unit/`

### `integration/` -- Real Neo4j + Ollama

- Verifies queries work against real database, LLM responses parse correctly.
- Requires running Neo4j and Ollama instances.
- Run with: `pytest tests/integration/ -v`
- Fixtures in `tests/integration/conftest.py` handle connection setup/teardown.

### `e2e/` -- Not Yet Implemented

- Full pipeline tests (voice input through knowledge storage).
- Will be added when the test suite matures and CI infrastructure supports it.

---

## Directory Mirroring

`tests/unit/` mirrors `backend/` minus the `backend/` prefix.

| Source file | Test file |
|---|---|
| `backend/knowledge/storage/graph_store.py` | `tests/unit/knowledge/storage/test_graph_store.py` |
| `backend/knowledge/extraction/pipeline.py` | `tests/unit/knowledge/extraction/test_pipeline.py` |
| `backend/knowledge/extraction/normalizer.py` | `tests/unit/knowledge/extraction/test_normalizer.py` |
| `backend/knowledge/retrieval/knowledge_retriever.py` | `tests/unit/knowledge/retrieval/test_knowledge_retriever.py` |
| `backend/knowledge/ontologies/v1_0_0.py` | `tests/unit/knowledge/ontologies/test_ontology_v1.py` |

Every test file has a corresponding `__init__.py` in its directory. Create one if missing.

---

## Fixture Organization (Pattern B)

Fixtures are defined in `tests/mocks/fixtures/` and imported into local `conftest.py`
files with explicit imports.

```python
# tests/unit/knowledge/extraction/conftest.py
import pytest

from tests.mocks.ollama import FakeLLM


@pytest.fixture
def fake_llm():
    """A FakeLLM with default empty extraction response."""
    return FakeLLM()
```

Shared fixtures (used across multiple subdirectories) live in `tests/unit/conftest.py`.
Module-specific fixtures live in the module's own `conftest.py`.

Use `# noqa: F401` when re-exporting fixtures that appear unused to the linter.

---

## Mock Factory Reference

| Fake | Location | Protocol | Description |
|---|---|---|---|
| `FakeNeo4jConnection` | `tests/mocks/neo4j.py` | `GraphConnection` | Records queries and writes; returns pre-configured results |
| `FakeGraphExecutor` | `tests/mocks/neo4j.py` | -- | Async wrapper around FakeNeo4jConnection for async callers |
| `FakeNeo4jRecord` | `tests/mocks/neo4j.py` | -- | Dict-like record simulating Neo4j query results |
| `FakeLLM` | `tests/mocks/ollama.py` | `LLMProvider` | Returns configurable responses; pattern-matches on prompt content |
| `FakeEmbeddingGenerator` | `tests/mocks/embeddings.py` | `EmbeddingProvider` | Deterministic 384-dim vectors via SHA-256 hash |
| `build_test_config()` | `tests/mocks/config.py` | -- | Builds `KnowledgeConfig` with test defaults; keyword-only args |

**Test constants** (from `tests/mocks/config`):
- `TEST_USER_ID = "user-test-001"`
- `TEST_SESSION_ID = "session-test-001"`
- `TEST_EVENT_ID = "event-test-001"`

---

## New Module Checklist

When adding tests for a new backend module:

1. **Create test file** mirroring the source path.
   `backend/knowledge/foo/bar.py` -> `tests/unit/knowledge/foo/test_bar.py`

2. **Create `conftest.py`** in the test directory if module-specific fixtures are needed.

3. **Import fixtures** from `tests/mocks/` or parent `conftest.py`.

4. **Define factory functions** for domain objects the module produces or consumes.
   Keyword-only args, sensible defaults, valid output with zero args.

5. **Group tests in classes** by operation type (TestCreate, TestQuery, TestUpdate, etc.).

6. **Use `@pytest.mark.asyncio`** on all async test functions.

7. **Assert side-effect boundaries.** When testing guards/validation, verify downstream
   I/O was NOT triggered (e.g., `fake_connection.assert_no_writes()`).

---

## Async Testing

pytest-asyncio runs in **strict mode** (`asyncio_mode = "strict"` in `pyproject.toml`).

This means:

- Every async test MUST be decorated with `@pytest.mark.asyncio`.
- Undecorated async functions are collected but fail with a clear error.
- Use `FakeGraphExecutor` for async Neo4j operations in tests.

```python
import pytest

@pytest.mark.asyncio
async def test_async_query_returns_results(fake_executor):
    results = await fake_executor.execute_query("MATCH (n) RETURN n")
    assert results == []
```

---

## Ontology Coupling

Changes to `backend/knowledge/ontologies/` affect both extraction and storage.

When modifying ontology definitions:

1. Run extraction tests: `pytest tests/unit/knowledge/extraction/ -v`
2. Run storage tests: `pytest tests/unit/knowledge/storage/ -v`
3. Run ontology tests: `pytest tests/unit/knowledge/ontologies/ -v`

All three must pass. Ontology changes that break extraction or storage indicate a
contract violation.

---

## Retroactive Learning

When code review feedback reveals a pattern issue in a test (e.g., using `MagicMock`
where a fake should be used, or missing side-effect boundary assertions):

1. Fix the flagged test.
2. Search for the same pattern in ALL previously-written tests.
3. Fix proactively. Do not wait for the same feedback on each file.

This prevents the same review comment from appearing across multiple PRs.

---

## Event Store Testing

The event store uses SQLite. Unit tests use **real in-memory SQLite** (`:memory:`) --
no fake needed.

```python
config = build_test_config(event_store_enabled=True, event_store_db_path=":memory:")
```

In-memory SQLite is fast enough for unit tests and eliminates fake/real divergence risk.
Each test gets a fresh database instance.

---

## Progress Tracking

See [TESTING_PROGRESS.md](TESTING_PROGRESS.md) for the living tracker of test
implementation status across sessions.

---

## Reference

- AI test guidance: [tests/CLAUDE.md](tests/CLAUDE.md)
- Design spec: [docs/specs/2026-03-22-testing-foundation-design.md](docs/specs/2026-03-22-testing-foundation-design.md)
- Mock factories: [tests/mocks/](tests/mocks/)
- Fixture definitions: [tests/mocks/fixtures/](tests/mocks/fixtures/)
