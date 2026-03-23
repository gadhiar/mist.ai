# MIST.AI -- Known Issues and Technical Debt

**Created:** 2026-03-22
**Source:** Comprehensive 8-agent backend audit (6 sectional + integration + fix rounds)
**Last Updated:** 2026-03-23

> This file tracks unresolved issues found during the pre-Phase-2 audit.
> Items here are P3 (maintenance risk) -- not blocking, but should be
> addressed before production use. Check items off as they are resolved.

---

## P3: Maintenance Risk / Technical Debt

### Server + Voice (Section A)

- [ ] **server.py:191-193 -- Double-thread pattern.** `run_in_executor` wraps
  `process_complete_audio` which spawns its own daemon thread. The `await`
  resolves before processing completes. No backpressure to client.

- [ ] **server.py:87 -- Broadcast task silently dies.** `asyncio.create_task(broadcast_messages())`
  has no exception handler callback. If `broadcast_messages` raises, messages
  stop being delivered with no indication.

- [ ] **server.py:106-111 -- CORS wildcard.** `allow_origins=["*"]` is acceptable
  for local dev but must be tightened if server is ever network-exposed.

- [ ] **server.py:170 -- No input size validation.** WebSocket `receive_json()`
  accepts arbitrarily large payloads. Could OOM on malicious input.

- [ ] **server.py:97-98 -- Blocking shutdown.** `voice_processor.models.shutdown()`
  calls `thread.join(timeout=2.0)` which blocks the event loop.

- [ ] **voice_processor.py:81-83 -- Blocking torch.hub.load in async context.**
  VAD model download/load blocks the event loop during startup. Should run
  in executor.

- [ ] **voice_processor.py:274-278 -- Recursive thread spawning.** After finishing
  a turn, if `latest_user_input` is set, a new thread is spawned. Unbounded
  thread creation if inputs queue up (mitigated by `generation_lock`).

- [ ] **voice_processor.py:60-61 -- Unused `self.audio_queue`.** Initialized but
  never read anywhere.

- [ ] **voice_processor.py:291-302 -- Deprecated `process_audio_chunk`.** Dead code
  with conditional `import scipy.signal`. Candidate for removal.

- [ ] **config.py:33 -- Unused `max_connections` field.** Defined in `VoiceConfig`
  but never read.

- [ ] **model_manager.py:98 -- No Ollama timeout on warmup.** `ollama.chat()` will
  block indefinitely if Ollama is not running.

- [ ] **model_manager.py:251 -- Misleading `trim_to_last_sentence` regex.** Non-greedy
  pattern only matches when text already ends at sentence boundary. Real
  trimming done by fallback loop. Regex is redundant.

### Chat Module (Section B)

- [ ] **conversation_handler.py -- Unbounded session storage.** `self.sessions` dict
  grows without limit. No TTL, no eviction, no max size. Memory leak over
  long uptime.

- [ ] **conversation_handler.py:373 -- Final LLM call uses `self.llm` (no tools).**
  After tool execution, the final response call uses the raw ChatOllama
  without tool bindings. May confuse LLM when tool-result messages are
  in context but tools are unavailable. Likely intentional but undocumented.

- [x] **conversation_handler.py -- Blocking Neo4j calls in async tool handlers.** -- resolved in Phase 2B: extract_knowledge and extract_knowledge_from_document tools removed. Extraction routed through ExtractionPipeline via GraphExecutor (async). query_knowledge_graph still calls sync GraphStore (separate issue).

- [ ] **conversation_handler.py -- No timeout on tool execution.** `await tool.ainvoke()`
  has no timeout. If Neo4j hangs, entire message handling blocks forever.

- [ ] **conversation_handler.py -- Sequential tool execution.** Multiple tool calls
  are executed in a loop, not with `asyncio.gather`. Adds unnecessary latency.

- [ ] **knowledge_integration.py:89-92 -- Overly broad exception suppression.**
  `contextlib.suppress(BaseException)` around `nest_asyncio.apply()`.
  Should use `Exception` at most.

- [ ] **knowledge_integration.py:100 -- Single-token streaming.** `yield response`
  yields the entire response as one chunk. Clients see no streaming when
  knowledge integration is active.

### Event Store (Section C)

- [ ] **store.py -- Shared connection not thread-safe for concurrent writes.**
  Single `sqlite3.Connection` with `check_same_thread=False`. Concurrent
  `append_turn` calls on the same instance could cause "cannot start a
  transaction within a transaction" errors. Current architecture is
  single-threaded so this is latent.

- [ ] **store.py -- conversation_handler.py still uses naive `datetime.now()`.**
  `_record_turn_event` creates `ConversationTurnEvent` with
  `timestamp=datetime.now()` (local time), while store.py now uses UTC.
  Inconsistency between the two timestamp sources.

- [ ] **schema.sql -- No UNIQUE(session_id, turn_index) constraint.** Duplicate
  turn indices could be inserted without error.

- [ ] **audio_archive.py -- Non-atomic file writes.** `write_bytes()` is not
  atomic. Crash mid-write leaves a corrupt file at the content-addressed
  path. Next call with same data skips it (thinks it exists). Fix: write
  to temp file, then `os.replace`.

- [ ] **models.py -- event_id default_factory always generates UUID.** The
  `store.py:154` check `if not event.event_id` never triggers because
  `default_factory` always produces a truthy UUID. Dead code.

- [ ] **store.py -- Silent JSON swallowing.** `_decode_turn_row` uses
  `contextlib.suppress` for JSON decode failures. Corrupted JSON fields
  silently return None.

### Knowledge Core (Section D)

- [ ] **config.py:20 -- Neo4j password defaults to None.** `os.getenv("NEO4J_PASSWORD")`
  returns None when env var is unset. No validation catches this early --
  fails later with cryptic Neo4j auth error.

- [ ] **config.py -- Class-level os.getenv evaluated at import time.** Default
  values on `Neo4jConfig`, `EmbeddingConfig`, `LLMConfig` are baked in
  when the module is first imported, not when `.from_env()` is called.
  If `.env` is loaded after import, defaults are stale.

- [ ] **config.py -- Global `_config` singleton race condition.** `get_config()`
  uses check-then-set pattern with no lock. Low practical impact (both
  threads produce identical configs) but technically incorrect.

- [ ] **config.py:164 -- `event_store: EventStoreConfig = None` with type-ignore.**
  Should use `field(default_factory=EventStoreConfig)` instead.

- [x] **config.py -- Stale ExtractionConfig fields.** -- resolved in Phase 2B, entity_extractor.py deleted. Dead fields (allowed_nodes, allowed_relationships, additional_instructions) remain in config.py but have no consumers.

- [ ] **EmbeddingGenerator ignores EmbeddingConfig.** Constructor takes raw
  `model_name` string, never reads `device` or `dimension` from config.
  GPU acceleration cannot be enabled through config.

- [ ] **EmbeddingGenerator hardcodes 384 dimension.** Zero-vector fallback
  returns `[0.0] * 384`. Breaks silently if model is changed to a
  different dimension.

- [ ] **ConversationSession name collision.** Two classes with same name in
  `backend.knowledge.models` (live session) and `backend.event_store.models`
  (persistence). No wrong-import bug today, but maintenance hazard.

- [ ] **Multiple independent KnowledgeConfig instances.** `model_manager.py`,
  `graph_regenerator.py`, and `get_config()` each create their own.
  No shared state. Runtime config changes don't propagate.

- [ ] **Two separate config hierarchies never bridged.** `VoiceConfig` (server.py)
  and `KnowledgeConfig` (knowledge system) are completely disconnected.

### Extraction Pipeline (Section E)

- [ ] **pipeline.py -- No per-stage exception handling.** If stages 3-6 throw
  on unexpected data shapes (e.g., LLM returns properties as string
  instead of dict), entire pipeline crashes. Wrap each stage in try/except.

- [x] **entity_extractor.py:59 -- Legacy `node_properties=False`.** -- resolved in Phase 2B, entity_extractor.py deleted

### Storage + Retrieval (Section F)

- [ ] **neo4j_connection.py -- Zero retry logic.** No transient error handling,
  no exponential backoff on any operation.

- [ ] **neo4j_connection.py:43 -- No connection pool configuration.** Missing
  `max_connection_pool_size`, timeouts, retry settings.

- [ ] **neo4j_connection.py -- No reconnection logic.** `connect()` only creates
  driver if `_driver is None`. Stale connections not detected.

- [ ] **neo4j_connection.py:88 -- `execute_query` not read-safe.** Uses
  `session.run()` instead of `session.execute_read()`. No automatic
  retry or read replica routing.

- [ ] **graph_store.py:633 -- FROM_SOURCE direction inconsistency.** Chunks
  created with `(chunk)-[:FROM_SOURCE]->(source)` but queried with
  undirected match `(source)-[:FROM_SOURCE]-(chunk)`.

- [x] **graph_regenerator.py:41-43 -- Redundant connections.** -- resolved in Phase 2B, now uses DI (constructor injection via factories.py)

- [ ] **graph_regenerator.py:327 -- Bulk DETACH DELETE without batching.** Will
  OOM Neo4j on large graphs. Use `CALL { } IN TRANSACTIONS` or iterative
  deletion.

- [ ] **graph_regenerator.py:352-357 -- Shared entity deletion.** Deleting
  entities linked to one conversation can delete entities shared with
  other conversations via MERGE.

- [ ] **graph_regenerator.py:376 -- Empty conversation_history on regeneration.**
  Passes `conversation_history=[]` so regeneration loses context that
  was present during original extraction.

- [ ] **knowledge_retriever.py -- N+1 query pattern.** For each of top 10
  similar entities, a separate `get_entity_neighborhood` call is made.
  Could be batched into 1-2 queries.

- [ ] **knowledge_retriever.py -- Async methods call sync Neo4j.** `_vector_search`
  and `_gather_facts` are `async def` but call synchronous GraphStore
  methods, blocking the event loop.

- [ ] **knowledge_retriever.py:85-87 -- Falsy default handling.** `limit=0` and
  `similarity_threshold=0.0` are treated as falsy, silently reverting
  to defaults.

- [ ] **seed_from_docs.py:107 -- Section title result discarded.** Expression
  is evaluated but never assigned to a variable.

- [ ] **wipe_database.py:66,84 -- Unsanitized constraint/index names in DROP.**
  Names from `SHOW CONSTRAINTS`/`SHOW INDEXES` interpolated into Cypher.
  Low risk (data source is Neo4j itself) but violates defense-in-depth.

### Integration / Cross-Module

- [ ] **KnowledgeIntegration reports enabled=True even when Neo4j is unreachable.**
  GraphStore uses lazy connect. First failure surfaces inside tool execution,
  not at startup. No circuit breaker.

- [x] **ExtractionPipeline outputs ValidationResult, GraphStore expects GraphDocument.** -- resolved in Phase 2B, store_validated_entities added to GraphStore

---

## Resolution Notes

When resolving an item:
1. Check the box
2. Add the commit hash and date on the same line
3. If the item was resolved as part of a larger change, note the context

Example: `- [x] **item** -- resolved in abc1234 (2026-03-25), Phase 2 curation`

---

## Audit Provenance

**Audit date:** 2026-03-22
**Methodology:** 8 Opus agents across 3 rounds
- Round 1: 6 sectional deep-dive agents (A-F)
- Round 2: 1 integration analysis agent
- Round 3: 2 fix agents (P0/P1 + P2)

**Resolved during audit:** 12 issues (2 P0, 4 P1, 6 P2)
**Commits:**
- `1a83569` -- Phase 1A+1B implementation
- `b745556` -- P0/P1/P2 bug fixes
