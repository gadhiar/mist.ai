# MIST.AI Testing Conventions

Project-wide testing standards and reference for the MIST.AI backend test suite. Read it before
writing or reviewing test code. Run tests in the container: see the root `CLAUDE.md`, Testing.

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
  dependence. A test that passes 99% of the time is broken.

### What Makes a Good Test

1. **Readable first.** A test is documentation: the name, the arrange block and the assertion
   should tell a reader unfamiliar with the module what behaviour is verified.
2. **Single purpose.** One test verifies one behaviour. If its name needs "and", split it.
3. **Explicit assumptions.** If a test depends on a config value, set it in the test or factory
   call. Never rely on implicit defaults from production code. Build config with
   `build_test_config()`, never `KnowledgeConfig.from_env()`, which reads the environment.

Structure each test as arrange / act / assert, separated by blank lines. Name tests in snake_case
by the expected behaviour (`test_raises_neo4j_query_error_on_connection_failure`, not
`test_error_handling`). For parametrized cases, use `pytest.param(..., id="descriptive-id")` so a
failure names its case, and keep one variation per case.

### What to Test

- **Test our reaction to a dependency's output, not the output itself.** Not "the LLM returns
  valid JSON" but our parsing of it and our handling of invalid JSON; not "Neo4j stores data" but
  our Cypher construction, parameter forwarding and connection-failure handling; not "the model
  produces good embeddings" but dimension validation, batching and empty input.
- **Verify forwarding.** When code passes a parsed value downstream, assert it arrived
  (`query, params = fake_connection.writes[0]; assert params["name"] == "Alice"`), not merely that
  nothing raised. `FakeNeo4jConnection` has `assert_write_executed`, `assert_query_executed` and
  `assert_no_writes`.

| Anti-pattern | Instead |
|---|---|
| Manual `__aenter__`/`__aexit__` wiring | `FakeGraphExecutor` |
| `assert mock.called` (tests the mock library) | Assert on domain outcomes |
| Broad `except Exception` in a test | Let it propagate; `pytest.raises` for expected errors |
| `if __name__ == "__main__"` blocks | Remove them |
| `.call_args[0]` index access | Fake assertion helpers, or `.kwargs` |
| Testing private methods | Test through the public API |
| `time.sleep()` | Async fakes or controlled sequencing (`asyncio.Event` in a `ScriptedPass`) |

---

## Test Tiers

### `unit/` -- Fast, Isolated, All I/O Faked

- No external dependencies (no Neo4j, no Ollama, no network).
- All I/O replaced by fakes from `tests/mocks/`.
- Target: entire suite runs in under 30 seconds.
- Run with: `pytest tests/unit/`
- `tests/unit/conftest.py` sets `MIST_EVAL_ISOLATION=1` and unsets `MIST_EVAL_NEO4J_HOSTS` for every unit test, so `Neo4jConnection.connect()` refuses any non-eval endpoint; tests needing a graph inject a fake.

### `integration/` -- Real Neo4j + llama-server

- Verifies queries work against real database, LLM responses parse correctly.
- Requires the Docker stack running (mist-neo4j + mist-llm via
  `docker compose up -d`); the bitemporal currency tests additionally
  target the disposable eval instance and skip cleanly when it is absent.
  That instance is defined in a separate compose file, so BOTH `-f` flags are
  required -- without them the service does not exist and the command fails:
  `docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml --profile eval up -d mist-neo4j-eval`
  (teardown: the same two `-f` flags with `--profile eval rm -sfv mist-neo4j-eval`).
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
| `backend/knowledge/extraction/validator.py` | `tests/unit/knowledge/extraction/test_validator.py` |
| `backend/knowledge/extraction/preprocessor.py` | `tests/unit/knowledge/extraction/test_preprocessor.py` |
| `backend/knowledge/retrieval/knowledge_retriever.py` | `tests/unit/knowledge/retrieval/test_knowledge_retriever.py` |
| `backend/knowledge/ontologies/v1_0_0.py` | `tests/unit/knowledge/ontologies/test_ontology_v1.py` |

Every test file has a corresponding `__init__.py` in its directory. Create one if missing.

When a module's tests outgrow one file, split by aspect as
`test_<module>_<aspect>.py` and keep the directory mirror intact. This is why
some modules have no bare `test_<module>.py`: `extraction/pipeline.py` is covered
by `test_pipeline_curation.py`, `test_pipeline_dedup.py`, and
`test_pipeline_extract_from_event.py`. `ls` the directory before citing a test
path in this file -- every row above is a real path as of 2026-08-03.

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

### Mocking Rules

Mock only at I/O boundaries, with the fakes in `tests/mocks/` (table above).

The LLM boundary is the abstract `StreamingLLMProvider` (`backend/llm/provider.py`), not a
concrete backend: `LlamaServerProvider` is the production implementation, `OllamaProvider` the
alternate, both chosen in `backend/factories.py` and possibly wrapped by
`InstrumentedStreamingLLMProvider`. Fake at the ABC: `FakeLLM` (`tests/mocks/ollama.py`) for
single responses, `FakeStreamingLLMProvider` (`tests/mocks/streaming_llm.py`) for scripted
streaming passes with tool calls and controlled chunk timing. A fake pinned to a concrete backend
rots the next time the backend is swapped.

**Never use an ad-hoc `MagicMock()`.** It accepts any attribute and any call, so a test cannot
catch interface drift: calling a method `StreamingLLMProvider` does not define (say `ainvoke`
instead of `invoke`) fails on `FakeLLM` and silently returns another mock on a `MagicMock`.

**Patching is the last resort**, for code without dependency injection (module-level functions,
legacy code). Define a `MODULE` constant at the top of the test file and build every target from
it, so a moved module is a one-line fix:

```python
MODULE = "backend.llm.llama_server_provider"

with patch(f"{MODULE}.httpx.AsyncClient", return_value=mock_client):
    ...
```

When three or more test files patch the same target, promote it to a shared context manager in
`tests/mocks/`.

---

## Vault Isolation

Any test that exercises a code path that can write to or index a vault (`backend/vault/*.py`,
vault writes through `ConversationHandler`, sidecar indexing, the filewatcher,
`ConventionsLoader`, extraction-pipeline writes; the list is not closed) MUST use the
`isolated_test_vault` fixture instead of touching `mist-memory/`. The real vault is the user's
canonical memory. `GraphRegenerator` no longer qualifies: R1.3 deleted the vault-derived
regenerator, and the surviving homonym raises `NotImplementedError` from both entry points.

```python
def test_vault_write(isolated_test_vault: Path):
    assert (isolated_test_vault / "MIST.md").exists()
```

The fixture (`tests/conftest.py`) copies `tests/fixtures/test-vault/` under pytest's `tmp_path`,
sets `MIST_VAULT_ROOT` to the copy (read by `VaultConfig.from_env()` in
`backend/knowledge/config.py`, its only read site; grep for it), drops the `_config` singleton so
`get_config()` re-reads the variable, and reverts both on teardown.

A test that builds its own config instead of going through `get_config()` (factory wiring, direct
instantiation) must point it at the fixture's path itself:
`build_test_config(vault_root=str(isolated_test_vault), vault_enabled=True, vault_user_id="test-user")`.

The baseline (`MIST.md`, `meta/`, `identity/mist.md`, `users/test-user.md`, two seed sessions,
one seed decision) is generic; its default user is `test-user`.

Forbidden:

- Writing to `mist-memory/` in a test, even for one smoke test.
- Bypassing `MIST_VAULT_ROOT` with a hard-coded path.
- Extending the baseline with test-specific content: write that into the ephemeral copy at arrange
  time.

The V6, V7 and V8 WebSocket gauntlets (`scripts/eval_harness/websocket_gauntlets_runbook.md`)
drive the running backend stack, which mounts `${VAULT_HOST_PATH:-./mist-memory}`
(`docker-compose.yml`). Override `VAULT_HOST_PATH` to a copy of the test-vault baseline before
running one; an out-of-process wrapper is a follow-up.

---

## New Module Checklist

When adding tests for a new backend module:

1. **Create test file** mirroring the source path.
   `backend/knowledge/foo/bar.py` -> `tests/unit/knowledge/foo/test_bar.py`

2. **Create `conftest.py`** in the test directory if module-specific fixtures are needed.

3. **Import fixtures** from `tests/mocks/` or parent `conftest.py`.

4. **Define factory functions** for domain objects the module produces or consumes.
   Keyword-only args, sensible defaults, valid output with zero args, and overrides for
   sensitive fields (thresholds, ids, text that can hit parsing edge cases).

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

## Verification Hazards

From the extraction-cache Phase 1 branch (2026-08-25), where six plan-authored tests passed while
blind to the property their names claimed. These are about how to verify, not what to test.

### Revert a mutation with `Edit`, never `git checkout -- <file>`

Mutation testing (apply a mutant, confirm RED, revert) is the strongest evidence a test is real,
but `git checkout -- <file>` discards all uncommitted changes to the file, not just the mutant,
and mutation testing happens precisely when uncommitted work exists. It cost one task its entire
implementation. Revert with an `Edit` that restores the exact original text; safest is to mutate
only on a clean tree (commit work in progress first). Confirm no mutant marker remains before
running anything else, and never run the suite while mutated source is importable.

### A backslash-sensitive pattern cannot be tested by typing it into a shell

Two agents reproduced a broken embedded `grep` citation by retyping its pattern into a shell; the
shell collapsed `\\` to `\`, so both tested the corrected pattern and found no defect. The real
pattern did not compile. Extract the pattern from the file as bytes (`sed -n 'Np' file | od -c`,
or write it to a file) and feed it with `grep -f`.

### "Collect-clean" is weaker evidence than it looks

`pytest --collect-only` does not execute a constructor call inside a test body, so a `TypeError`
from a newly required keyword argument hides behind a clean collect. If you change a signature,
read the call sites.

### A citation you have not run is not evidence

Write the command, run it, and record what it returned. A citation only ever seen returning zero
is indistinguishable from one that cannot fire, so also run it against something that should
match.

### Derived beats enumerated whenever the enumeration can go stale

A frozen-dataclass `==` between two factories' outputs covers a field added later; per-field
assertions do not. A guard that derives its field set from `dataclasses.fields(...)` extends
itself; a hard-coded list does not. Prove it by adding a member and watching the guard pick it up.

### Say what a guard does and does not catch, and which way it fails

- A guard that can be silently satisfied is worse than one that can be silently bypassed: a
  missed case loses coverage, but a trivially met condition (a source-text count inflated by the
  token appearing in a comment) reports balance while the guarded thing is missing.
- A test that fires on a future change is fine if its docstring says so and names the edit that
  trips it. One that could never fail while presenting itself as proof of something else is a
  trap.

---

## Progress Tracking

See [TESTING_PROGRESS.md](TESTING_PROGRESS.md) for the living tracker of test
implementation status across sessions.

---

## Reference

- Design spec: [docs/specs/2026-03-22-testing-foundation-design.md](docs/specs/2026-03-22-testing-foundation-design.md)
- Mock factories: [tests/mocks/](tests/mocks/)
- Fixture definitions: [tests/mocks/fixtures/](tests/mocks/fixtures/)
