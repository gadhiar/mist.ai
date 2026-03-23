# Testing Progress

Living document tracking test implementation across sessions.
Reference: [docs/specs/2026-03-22-testing-foundation-design.md](docs/specs/2026-03-22-testing-foundation-design.md)

---

## Status

### Completed

- [x] Test infrastructure (conftest, mocks, directory scaffold)
- [x] DI refactor (GraphStore, ExtractionPipeline, Neo4jConnection, OntologyExtractor, Normalizer)
- [x] GraphExecutor async boundary
- [x] Custom exception hierarchy
- [x] Composition root factories
- [x] Unit tests: ontologies (27 tests)
- [x] Unit tests: extraction pipeline stages (54 tests -- confidence, temporal, preprocessor, validator)
- [x] Unit tests: storage (15 tests -- GraphStore, Neo4jConnection, GraphExecutor)
- [x] Unit tests: event store (8 tests)

- [x] Phase 2A curation pipeline (35 tests -- confidence, dedup, conflict, graph writer, pipeline)
- [x] Phase 2A normalizer async migration (3 tests)
- [x] Phase 2B ConversationHandler DI + automatic extraction (5 tests)
- [x] Phase 2B GraphRegenerator migration (2 tests)
- [x] Phase 3 SignalDetector + InternalKnowledgeDeriver (14 tests)
- [x] Phase 4 Tier 1+2: decay, staleness, orphans, reflection, health, scheduler (44 tests)

**Total: 204 tests passing**

### Remaining

- [ ] Unit tests: retrieval (KnowledgeRetriever)
- [ ] Unit tests: normalizer (test_normalizer.py -- additional coverage beyond async migration)
- [ ] Unit tests: ontology extractor (test_ontology_extractor.py)
- [ ] Unit tests: extraction pipeline orchestrator (test_pipeline.py -- additional integration tests)
- [ ] Voice pipeline tests (separate effort, needs FakeVAD/FakeWhisper/FakeTTS)
- [ ] CI structure validation (add when test suite > 20 files)

---

## Decisions & Corrections

### 2026-03-22 Session

1. **Approach D hybrid with B Protocols.** Explicit composition root DI (required constructor params, no hidden construction) with Protocol definitions at I/O boundaries (GraphConnection, EmbeddingProvider, LLMProvider, EventStoreProvider). Combines D's pragmatism with B's production-readiness.

2. **GraphExecutor as single async boundary.** One class wraps sync Neo4j via run_in_executor. Everything above is async, everything below is sync. Future async driver migration changes one file.

3. **Explicit fakes over MagicMock.** FakeNeo4jConnection, FakeLLM, FakeEmbeddingGenerator satisfy Protocol interfaces. MagicMock silently accepts any attribute, defeating Protocol conformance.

4. **ruff D101/D102 ignored for test files.** Test methods use descriptive names instead of docstrings.

5. **Event store tests use real in-memory SQLite.** Fast enough, avoids fake/real divergence.

6. **ConversationHandler DI deferred to Phase 2B.** Would require double-refactoring since Phase 2B rewrites its __init__ and tool setup anyway.

7. **except Exception blocks narrowed.** Custom exception types (Neo4jConnectionError, Neo4jQueryError, etc.) used in all refactored files. Remaining 30+ broad catches in untouched files cleaned up opportunistically.
