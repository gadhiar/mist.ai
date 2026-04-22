# MIST.AI Codebase Context

**Last Updated:** 2026-04-22 (Cluster 8 COMPLETE)
**Branch:** main (commits ahead of origin pending push: Phase 6 + Phase 8 + Phase 9 + Phase 10 + Phase 11 + Phase 12)
**Status:** MVP Knowledge Integration — **ALL 8 architectural clusters complete.** Cluster 8 (vault-native memory, ADR-010) all 12 phases shipped (Phase 7 folded into Cluster 2; Cluster 7 absorbed into Phase 10). Workstream `mist-ai-knowledge-integration-mvp-validation` is structurally complete; Phase 4 earned-close gauntlet replay is the only remaining gate before formal close.

---

## Current Status

### Backend
- **Status:** CONTAINERIZED (Docker + CUDA 12.4). All development against the container; Windows native venv is corrupted.
- **Server:** FastAPI WebSocket on port 8001.
- **LLM:** Gemma 4 E4B Q5_K_M dense (carteakey-full recipe) via llama-server (llama.cpp OpenAI-compatible API). Selected 2026-04-16 via gauntlet (ADR-008 revised). Serving at `http://mist-llm:8080`.
- **LLM ctx_size:** 32K configured on llama-server; effective attention window ~8K (Gemma's trained context). Cluster 6's `context_budget.context_window=8192` default respects this.
- **StreamingLLMProvider abstraction:** `LlamaServerProvider` (primary), `OllamaProvider` (fallback), optionally wrapped by `InstrumentedStreamingLLMProvider` (Cluster 5) for JSONL observability.
- **ConversationHandler:** Cluster 3 persona injection + Cluster 6 `ContextBudgetPlanner` + `_build_request` Pydantic-dump helper (Cluster 5). `conversation_max_tokens=1024` (up from 400; Cluster 6 fix for Bug E).
- **ExtractionPipeline:** 6 stages + Stage 1.5 `SubjectScopeClassifier` (Cluster 1) — classifies each utterance as `user-scope | system-scope | third-party | unknown` between pre-processing and ontology extraction. Metadata threaded into `EXTRACTION_USER_TEMPLATE` as a `subject_scope` hint.
- **Voice Pipeline:** VAD -> Whisper -> Gemma 4 E4B -> Chatterbox Turbo TTS with streaming parallelism. ~4-5s TTFA.
- **Audio Transport:** Binary WebSocket frames (MIST protocol: 16-byte header + PCM16), RMS normalization (-20 dBFS), interrupt fade-out.
- **Log Streaming:** WebSocketLogHandler with per-logger gating, token bucket rate limiter, request ID propagation.
- **Persistent Logging:** `./logs/mist-backend.log` at DEBUG level (survives container removal).
- **Debug JSONL Observability:** `DebugJSONLLogger` with 5 record phases (`turn`, `extraction`, `llm_call`, `retrieval_candidates`, `llm_request_raw`). Each gated by its own env var. See Cluster 5 artifacts below.
- **Knowledge Graph:** Extraction + curation pipeline + hybrid retrieval (graph + vector + RRF merge). ADR-009 provenance separation structurally enforced (Cluster 2). MIST identity retrieval injects persona (Cluster 3). Ontology v1.0.0 carries 13 extractable entity types (12 external + MistIdentity) and 25 extractable relationship types (21 original + 4 MIST-scope: IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT, MIST_HAS_PREFERENCE) — Cluster 1.
- **Knowledge Seed:** 32-entity baseline (`mist_admin seed` from `scripts/seed_data.yaml`): 1 MistIdentity + 9 MistTraits + 5 MistCapabilities + 5 MistPreferences + 11 user/technology entities + 1 User + 19 identity relationships + 11 anchor relationships + 32 embeddings.
- **Vault Layer (Cluster 8, in progress):** `backend/vault/` package with `VaultWriter` (serialized `asyncio.Queue` consumer for session-note appends, identity/user upserts), `VaultSidecarIndex` (sqlite-vec `vec0` + FTS5 + RRF hybrid query over two-tier chunks), `VaultFilewatcher` (watchdog daemon thread with 500ms debounce + asyncio bridge + 60s mtime audit job + MIST-write coordination for user-edit detection), Pydantic frontmatter models for the four `mist-*` note types, and `AuthoredBy` 5-state authorship enum. Wired through `VaultConfig` / `SidecarIndexConfig` / `FilewatcherConfig` on `KnowledgeConfig`. **Phase 5 integrated:** single server-owned VaultWriter built and started in `server.py` lifespan, plumbed through `VoiceProcessor -> ModelManager -> KnowledgeIntegration -> ConversationHandler`, with per-turn vault append after event-store write (failure-isolated per ADR-010 Invariant 6). **Phase 6 integrated:** `vault_note_path` is pre-allocated synchronously at `handle_message` Step 0 (via `_get_or_allocate_vault_path`) and threaded through `_extract_knowledge_async` -> `ExtractionPipeline.extract_from_utterance` -> `CurationPipeline.curate_and_store` -> `CurationGraphWriter.write`. Every upserted entity now emits a `DERIVED_FROM` edge to a `:__Provenance__:VaultNote {path}` node (MERGE-idempotent on path). New `VaultNote` ontology node type registered as bridging; `DERIVED_FROM` edge extended to permit `VaultNote` targets and `MistIdentity` sources. The graph is now formally rebuildable from the vault. **Phase 8 integrated:** rebuild-determinism stamps. New `RebuildStamps` frozen dataclass (`ontology_version`, `extraction_version`, `model_hash`) constructed by `build_curation_pipeline` from `KnowledgeConfig` and injected into `CurationGraphWriter`. Every `DERIVED_FROM`->`VaultNote` edge now carries the three stamps + `derived_at` timestamp on both ON CREATE and ON MATCH branches so re-extractions land the current stamps. New config fields `KnowledgeConfig.extraction_version` (default `"2026-04-17-r1"`, env `EXTRACTION_VERSION`) and `KnowledgeConfig.model_hash` (default `"gemma-4-e4b-q5-k-m-carteakey-full-v1"`, env `MIST_MODEL_HASH`). **Phase 9 integrated:** retrieval routing + slug improvement. QueryClassifier extended with a `historical` intent (regex patterns matching "what did we discuss"/"remember when"/"last time"/etc.) routed to the vault sidecar; `hybrid` now produces three-way RRF merges across graph + vector + vault sidecar via `_merge_rrf_three_way`. New `QueryIntentConfig` fields per ADR-010 weight table (`rrf_vault_weight=0.4` hybrid; historical-specific `0.2/0.1/0.7` graph/vector/vault). `KnowledgeRetriever` accepts an optional `vault_sidecar: SidecarIndexProtocol` plumbed top-down through `VoiceProcessor -> ModelManager -> KnowledgeIntegration -> build_conversation_handler -> build_knowledge_retriever`; `_vault_sidecar_retrieve` wraps `query_hybrid` and converts vec0+FTS5 results to `RetrievedFact` rows. Session slug derivation now extracts significant words from the FIRST USER UTTERANCE (stopwords + short tokens filtered, top 5 retained) with a 4-char SHA-256(session_id) suffix for guaranteed per-session uniqueness — produces filenames like `2026-04-22-vault-architecture-mist-a3f1.md` instead of opaque `2026-04-22-<sanitized-session-id>.md`. **Phase 10 integrated:** seed vault bootstrap (absorbs Cluster 7 migration). `mist_admin seed` now extends to `bootstrap_vault_from_seed` (async helper in `backend/knowledge/admin.py`) which calls `VaultWriter.upsert_identity` (rendered from seeded MistTraits/Capabilities/Preferences) and `VaultWriter.upsert_user` (rendered from the seeded user dict via `_build_user_body_markdown`). After the writes, `emit_seed_vault_provenance` MERGE-creates two `:__Provenance__:VaultNote` nodes (one per bootstrap note) and emits per-entity `DERIVED_FROM` edges from each seeded entity (mist-identity + traits/caps/prefs -> identity/mist.md; user + anchor entities -> users/<id>.md). Edges carry `event_id='seed'` literal (no Phase 8 stamps -- seed entities are deterministic via re-run, not extraction-rebuild). New `--no-vault-bootstrap` flag opts out; bootstrap also auto-skips when `config.vault.enabled` is False. Filewatcher + sidecar share the same lifecycle. Phase 11 (CLI subcommands `vault-status` / `vault-reindex` / `vault-rebuild` / `vault-migrate`) is next.
- **Tests:** **1444 unit tests + 1 platform-skipped + 3 xfailed** (vs 1066 Cluster 1 baseline = +378 from Cluster 8 Phase 1+2+3+4+5+6+8+9+10+11+12; Phase 11 alone added +18). Run inside container: `docker compose exec mist-backend python -m pytest tests/unit/`.

### Frontend (Flutter)
- **Status:** IN DEVELOPMENT (unchanged since 2026-04-08). Cluster 1/8 work is backend-focused; frontend touches parked in `mist-ai-frontend-audit-remediation` (status: parked).
- Navigation rail (72/200px), log viewer (filter/search/grouping, 5K ring buffer), binary audio playback via flutter_soloud.

### Code Quality
- Full pre-commit suite: black, ruff (D102 strict), bandit, codespell, AI-slop pattern checker, trim whitespace, fix end-of-files, large file + merge conflict + private key detection.
- AI-slop pattern checker enforces no emoji/unicode-decorative/arrow symbols in new code.
- CI configured via GitHub Actions.

---

## MVP Knowledge Integration — Cluster Status

**Workstream:** `mist-ai-knowledge-integration-mvp-validation` (active, P0). Full detail in the knowledge-vault workstream note at `knowledge-vault/Projects/mist-ai/workstreams/mist-ai-knowledge-integration-mvp-validation.md`.

**Plan artifact:** `~/.claude/plans/cluster-execution-roadmap.md` (canonical) + mirror at `mist.ai/.local/plans/cluster-execution-roadmap.md`.

| Cluster | Scope | Status | Key commit range | Gauntlet artifact |
|---|---|---|---|---|
| 1 | Ontology expansion + subject-scope classifier | COMPLETE 2026-04-21 | 4dc7204 -> 3b10a24 | post-cluster-1-gauntlet-report-2026-04-21.md |
| 2 | Graph provenance separation (ADR-009) | COMPLETE 2026-04-20 | c505c7a -> 8a31fcc | post-cluster-2-gauntlet-report-2026-04-20.md |
| 3 | Identity layer + persona injection + AI-slop filter + dual temperature | COMPLETE 2026-04-21 | f306788 -> 6124e43 | post-cluster-3-gauntlet-report-2026-04-21.md |
| 4 | Deterministic rails (Bugs A, C, G, K) | COMPLETE 2026-04-20 | 4eed4f2 -> 68ffc81 | post-cluster-4-gauntlet-report-2026-04-20.md |
| 5 | Observability (llm_call + retrieval_candidates + llm_request_raw JSONL phases) | COMPLETE 2026-04-21 | 27af364 -> 3c8f0b2 | v6-cluster-5-diagnostic-report-2026-04-21.md |
| 6 | Context budget (ContextBudgetPlanner) + max_tokens=1024 fix | COMPLETE 2026-04-21 | c4c4d71 -> c800e35 | post-cluster-6-gauntlet-report-2026-04-21.md |
| 7 | Existing-data migration | Folds into Cluster 8 phase 10 | — | — |
| 8 | Vault-native memory (ADR-010 implementation, 12-phase) | **COMPLETE** — All 12 phases done (Phase 7 folded into Cluster 2). | 8c06914 -> Phase 11 | — |
| 7 | Existing-data migration | ABSORBED into Phase 10 vault bootstrap | — | — |

**Acceptance status** toward Phase 4 earned close:
- Relationship correctness ≥ 80% — **CLEARED** post-Cluster-1 on targeted probe (11/12 = 92% on v1-mist-scope-inputs.jsonl). V6 spontaneously produces 4 `mist-identity USES X` edges that would have been validator-dropped pre-Cluster-1.
- Post-session retrieval semantic content ≥ 80% — effectively CLEARED on canonical probe post-Cluster-2 (9/10 user-facing facts)
- Emoji violations = 0 — CLEARED post-Cluster-3 (0/46 V4+V5+V6 turns); held post-Cluster-1 (0/47)
- Empty responses < 10% — **CLEARED** post-Cluster-6 (0/30 V6); held post-Cluster-1 (0/30)
- LLMRequest validation errors = 0 — CLEARED post-Cluster-4 (held through Clusters 5, 6, 1)
- Unit tests 900+/900+ green — CLEARED (1066 + 3 xfailed)

All five Phase 4 gates now clear. Only Cluster 8 (ADR-010 vault implementation) remains before earned close.

**Bug status (P1/P2 from 2026-04-17 gauntlet):**
- A (83% NULL provenance) — CLEARED (Cluster 4)
- B (identity drift, emoji leak, AI slop) — CLEARED (Cluster 3)
- C (LLMRequest tool_calls schema) — CLEARED (Cluster 4)
- E (empty LLM responses) — CLEARED (Cluster 6; root cause: max_tokens=400 truncating tool-call JSON)
- G (reserved-namespace guard) — CLEARED (Cluster 4)
- I (LEARNING->USES slippage) — CLEARED (Cluster 1: scope classifier + prompt rebalance)
- J (MIST-tooling attributed to Raj USES) — CLEARED (Cluster 1: validator accepts MistIdentity source on USES/DEPENDS_ON/WORKS_WITH; normalizer forces MistIdentity type on reserved names)
- K (prompt-injection written as fact) — CLEARED (Cluster 4 pre-filter + prompt tightening held through Cluster 1; declarative-framing residual noted as P1)
- N (retrieval returns only provenance plumbing) — STRUCTURALLY RESOLVED (Cluster 2)

---

## Active Work

### Current Focus
1. **Cluster 8 (ADR-010 vault implementation)** — 12-phase roadmap, 4-6 weeks. Sole remaining workstream deliverable before Phase 4 earned close. Move ADR-010 from `proposed` to `accepted` when implementation begins.
2. **Phase 4 earned-close** via `/vault-end-session` + workstream completion once Cluster 8 lands.

### Recently Completed (2026-04-21, end of day)
- **Cluster 1 (Ontology + subject-scope classifier):** 8 commits (`4dc7204` -> `3b10a24`) on main, pushed to origin. Extended validator `RELATIONSHIP_CONSTRAINTS` to accept Organization + MistIdentity as source for USES/DEPENDS_ON/WORKS_WITH. Added 4 new MIST-scope predicates: `IMPLEMENTED_WITH`, `MIST_HAS_CAPABILITY`, `MIST_HAS_TRAIT`, `MIST_HAS_PREFERENCE`. Added `MistIdentity` as extractable entity type (13 total). New `SubjectScopeClassifier` module running as Stage 1.5 AFTER significance + dedup gates, writing `subject_scope` metadata to PreProcessedInput, threaded into extraction user template. Rewrote `EXTRACTION_SYSTEM_PROMPT` removing user-centric bias; 3 user / 3 system / 1 third-party / 1 empty example balance. Normalizer `RESERVED_NAMES` now remaps both id AND entity_type to MistIdentity. Cluster 3 integration: `get_mist_identity_context` UNIONs HAS_* (seed) and MIST_HAS_* (extracted) into one merged set. Cluster 2 integration: `_USER_FACING_REL_TYPES` extended with new edges so multi-hop traversal expands through them. Bug J closure evidence in V6: `mist-identity -[USES]-> lancedb/neo4j/llamacpp/sentence-transformers` landed (all dropped pre-Cluster-1). V1 probe = 11/12 (92%). V6 = 0/30 empty, 0 emoji, 0 Bug C. +44 net new tests (1022 -> 1066).
- **Cluster 6 (Context budget + max_tokens fix):** V6 empty-response rate 53% -> 0%. `LLMConfig.conversation_max_tokens=1024` (up from 400) at all three ConversationHandler invoke sites fixes the "GHOST turn" failure mode (tool-call JSON truncation). `ContextBudgetPlanner` with TokenCounter + HistoryStrategy protocols + SlidingWindowStrategy default provides defense-in-depth. Commits c4c4d71, d354f30, 997517f, c800e35. +32 net new tests.
- **Cluster 5 (Observability):** Three JSONL record phases added to `DebugJSONLLogger` (`llm_call`, `retrieval_candidates`, `llm_request_raw`) each with its own env gate. `InstrumentedStreamingLLMProvider` wraps any concrete provider transparently; `llm_call_context` ContextVar threads caller metadata (`session_id`/`event_id`/`call_site`/`pass_num`). All factories wired. Commits 27af364, f5d0ec4, ab85115, e7ca7e2 + polish 3c8f0b2. +44 net new tests.
- **Cluster 3 (Identity + AI-slop filter + dual temperature):** 6 deliverables — config split (`conversation_temperature=0.7`), slop detector library, pref-no-ai-slop seed, QueryClassifier identity intent at priority 0, `retrieve_mist_context()` + `MistContext` renderer with HARD RULES framing, response post-filter with regen + strip_fixable fallback. Bug B closed: 0 emoji across 46 V4+V5+V6 turns; consulting-voice markdown drift -56%. Commits f306788 -> 6124e43. +90 net new tests.

### Previously Completed (2026-04-20)
- **Cluster 4 (Deterministic rails):** Bug A fix (ON CREATE SET e.provenance='extraction'); Bug C fix (`list[dict[str, Any]]` widening in LLMRequest.messages); Bug G (RESERVED_NAMES table in EntityNormalizer); Bug K two-layer fix (pre-filter + prompt tightening).
- **Cluster 2 (ADR-009 graph provenance separation):** 5 writer sites migrated to `:__Provenance__` base label; retrieval multi-hop filter anchored at `:__Entity__`; `mist_admin graph-reset --include-derived`; `graph-stats` three-section output. Canonical V6 turn-30 probe returns 9/10 user-facing facts vs morning's 0.

### Previously Completed (2026-04-16 -> 2026-04-08)
- Gemma 4 E4B selected as production model via gauntlet (ADR-008 revised)
- Model backend migration: Ollama -> llama-server via `StreamingLLMProvider` abstraction
- Binary WebSocket audio transport (MIST protocol, ~7x bandwidth reduction)
- Personality system (YAML per voice profile)
- FRIDAY default voice profile

### Blockers
None. All upstream gates for Cluster 1 cleared.

---

## Debug Observability Quick-Start

`DebugJSONLLogger` writes structured JSONL records to the path set by `MIST_DEBUG_JSONL`. Three phase-specific gates layered on top:

```bash
# From Git Bash on Windows: MSYS_NO_PATHCONV=1 prefix is REQUIRED
# (see reference_docker_exec_path_mangling memory) or /app/... gets path-translated.
MSYS_NO_PATHCONV=1 docker compose exec -T \
  -e MIST_DEBUG_JSONL=/app/data/ingest/session.jsonl \
  -e MIST_DEBUG_LLM_JSONL=1 \
  -e MIST_DEBUG_RETRIEVAL_JSONL=1 \
  -e MIST_DEBUG_LLM_REQUESTS=1 \
  mist-backend python -m scripts.mist_admin replay \
  /app/data/ingest/v6-inputs.jsonl \
  --session-id diagnostic \
  --output /app/data/ingest/report.jsonl
```

**Emitted phases when gates are open:**
- `phase: "turn"` — per-turn wrapper with event_id/session_id/utterance + retrieval summary + llm_passes + total_turn_ms. (Pre-Cluster-5 infrastructure.)
- `phase: "extraction"` — per-turn extraction stats (entities_count, avg_confidence, graph_writes). (Pre-Cluster-5 infrastructure.)
- `phase: "llm_call"` — full request/response at every provider.invoke(). Content/tool_calls/usage/latency_ms. call_site-tagged: `chat.initial`, `chat.final`, `chat.regen`, `extraction.ontology`, `extraction.internal_derivation`.
- `phase: "retrieval_candidates"` — full graph + vector candidate pools from `KnowledgeRetriever.retrieve()` BEFORE RRF merge + rank truncation. Gate: `MIST_DEBUG_RETRIEVAL_JSONL=1`.
- `phase: "llm_request_raw"` — pre-validation LLMRequest kwargs dump on Pydantic ValidationError. Gate: `MIST_DEBUG_LLM_REQUESTS=1`.

All records carry `ts_iso` + `session_id` + `event_id` for cross-record joins.

---

## Dependency Injection Contract

All classes depending on external systems (Neo4j, LLM, embeddings, event store, vector store, debug logger) accept dependencies as required constructor parameters. Factories in `backend/factories.py` own all wiring with real implementations.

**Factory entry points:**
- `build_conversation_handler(config, llm_provider=None)` — composition root for chat. Reads `DebugJSONLLogger.from_env()` once, logs active phase gates, threads the logger through `build_llm_provider` (wraps with `InstrumentedStreamingLLMProvider` when `llm_call_enabled`) and `build_knowledge_retriever` (forwards `debug_logger`). Returns a `ConversationHandler` with all dependencies wired.
- `build_extraction_pipeline(config, graph_store=None, llm_provider=None, include_curation=True, include_internal_derivation=True)` — extraction + curation + internal derivation pipeline.
- `build_knowledge_retriever(config, graph_store=None, vector_store=None, embedding_provider=None, debug_logger=None)` — hybrid retriever (graph + vector + RRF).
- `build_llm_provider(config, debug_logger=None)` — provider with optional instrumentation.

---

## Architecture Overview

### Docker Stack
```
docker-compose.yml              # 3 services: mist-backend, mist-neo4j, mist-llm
docker-compose.override.yml     # Dev mode volume mounts (backend/tests/scripts/voice_profiles bind-mounted)
docker/backend/Dockerfile       # CUDA 12.4 + Python 3.11 + Chatterbox
```

**Volume mounts** (backend code hot-reloadable):
- `./data:/app/data` (graph snapshots, JSONL diagnostics, event store SQLite)
- `./logs:/app/logs` (persistent logs)
- `./backend:/app/backend`
- `./tests:/app/tests`
- `./scripts:/app/scripts`
- `./voice_profiles:/app/voice_profiles`

### Backend Structure
```
backend/
├── server.py              # WebSocket server (port 8001)
├── voice_processor.py     # Voice pipeline orchestration
├── audio_protocol.py      # MIST binary frame builder
├── log_handler.py         # WebSocketLogHandler
├── request_context.py     # ContextVar propagation
├── sentence_detector.py   # Streaming TTS sentence boundary detection
├── debug_jsonl_logger.py  # Cluster 5: 5-phase JSONL sink with env gates
├── factories.py           # Composition root
├── errors.py              # MistError hierarchy
├── interfaces.py          # Protocols (EmbeddingProvider, VectorStoreProvider, GraphConnection, EventStoreProvider)
├── chat/
│   ├── conversation_handler.py   # Persona + budget + slop + post-filter (Clusters 3, 5, 6)
│   ├── mist_context.py           # Cluster 3 MistContext dataclasses + renderer
│   ├── slop_detector.py          # Cluster 3 pattern catalogue
│   └── context_budget.py         # Cluster 6 planner + TokenCounter + HistoryStrategy
├── llm/
│   ├── provider.py                   # Abstract StreamingLLMProvider ABC
│   ├── llama_server_provider.py      # Primary concrete provider
│   ├── ollama_provider.py            # Fallback concrete provider
│   ├── instrumented_provider.py      # Cluster 5 wrapper + llm_call_context ContextVar
│   └── models.py                     # LLMRequest/LLMResponse/ToolCall Pydantic models
└── knowledge/
    ├── config.py                     # KnowledgeConfig + nested configs
    ├── models.py                     # RetrievalResult, RetrievedFact, QueryIntent, etc.
    ├── embeddings.py                 # EmbeddingGenerator (Sentence Transformers)
    ├── extraction/                   # 6-stage extraction pipeline + ontology constraints + validator
    ├── curation/                     # Dedup + conflict resolver + graph writer + confidence + scheduler
    ├── ingestion/                    # Markdown ingestion for vector store
    ├── retrieval/
    │   ├── knowledge_retriever.py    # Hybrid retrieval + identity-intent routing (Cluster 3)
    │   └── query_classifier.py       # Intent classification (live/relational/factual/hybrid/identity)
    ├── regeneration/                 # Graph regenerator (no-curation replay)
    └── storage/
        ├── neo4j_connection.py
        ├── graph_executor.py         # Async/sync boundary
        └── graph_store.py            # GraphStore + get_mist_identity_context (Cluster 3)
```

### Frontend Structure
Unchanged since 2026-04-08. See `mist_desktop/` directory for Flutter/Riverpod/flutter_soloud structure.

---

## Configuration

### Environment Variables (.env / .env.example)
```bash
# Neo4j
NEO4J_URI=bolt://mist-neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM backend
LLM_BACKEND=llamacpp                       # llamacpp (default) | ollama (fallback)
LLM_SERVER_URL=http://mist-llm:8080
MODELS_DIR=./models                        # Host path; mounted read-only into mist-llm
LLM_MODEL_FILE=unsloth/gemma-4-E4B-it-Q5_K_M.gguf
LLM_CTX_SIZE=32768                         # llama-server ctx_size; effective attention ~8K
LLM_TEMPERATURE=0.0                        # Extraction default
LLM_CONVERSATION_TEMPERATURE=0.7           # Conversation default (Cluster 3 split)
LLM_CONVERSATION_MAX_TOKENS=1024           # Cluster 6 Bug E fix (was hardcoded 400)
MODEL=gemma-4-e4b

# Cluster 6 context budget
MIST_CTX_BUDGET_ENABLED=true
MIST_CTX_BUDGET_WINDOW=8192
MIST_CTX_BUDGET_OUTPUT_RESERVE=512
MIST_CTX_BUDGET_SAFETY=256
MIST_CTX_BUDGET_RETRIEVAL_RATIO=0.4
MIST_CTX_BUDGET_HISTORY_STRATEGY=sliding_window

# Cluster 5 observability (all off by default)
# MIST_DEBUG_JSONL=/app/data/ingest/debug.jsonl
# MIST_DEBUG_LLM_JSONL=1
# MIST_DEBUG_RETRIEVAL_JSONL=1
# MIST_DEBUG_LLM_REQUESTS=1

# Voice / TTS
TTS_ENABLED=true
TTS_ENGINE=chatterbox
VOICE_PROFILE=friday

# Feature flags
ENABLE_KNOWLEDGE_INTEGRATION=true

# Event store (Layer 1)
EVENT_STORE_DB_PATH=/app/data/event_store.db
EVENT_STORE_AUDIO_DIR=/app/data/audio

# Vector store (Layer 2)
VECTOR_STORE_DATA_DIR=/app/data/vector_store
```

### Critical Settings
- `LLM_BACKEND=llamacpp` (Gemma 4 E4B primary; `ollama` available for fallback)
- `LLM_MODEL_FILE=unsloth/gemma-4-E4B-it-Q5_K_M.gguf` (carteakey-full recipe per ADR-008)
- `MIST_CTX_BUDGET_ENABLED=true` (Cluster 6 default; disable only to confirm budget-related regressions)
- Docker data root: `D:\Users\rajga\DockerData` (not default C:)
- Git Bash on Windows: prefix `docker compose exec` with `MSYS_NO_PATHCONV=1` when passing `/app/...` paths in env vars

---

## Tech Stack

### Backend
- Python 3.11+
- FastAPI + Uvicorn (WebSocket server)
- Docker (nvidia/cuda:12.4.0-devel-ubuntu22.04)
- PyTorch 2.6.0+cu124 (flash attention enabled)
- llama-server / llama.cpp (LLM inference — Gemma 4 E4B Q5_K_M via GGUF, OpenAI-compatible API)
- `StreamingLLMProvider` abstraction: `LlamaServerProvider` primary, `OllamaProvider` fallback, `InstrumentedStreamingLLMProvider` decorator
- openai + httpx (LLM client)
- Whisper (STT — base model)
- Chatterbox Turbo (TTS — zero-shot voice cloning, MIT)
- Neo4j 5.x (knowledge graph; `:__Entity__` user-facing, `:__Provenance__` audit-trail per ADR-009)
- LanceDB (vector store Layer 2)
- Sentence Transformers all-MiniLM-L6-v2 (384-dim embeddings)
- SQLite (event store Layer 1)
- Pydantic v2 (LLMRequest/Response + config)

### Frontend
- Flutter 3.24+ / Dart 3.10+
- Riverpod 3.x (state management)
- web_socket_channel, record, flutter_soloud, audioplayers

---

## Development Workflow

### Starting the Stack
```bash
docker compose up -d               # Full stack
docker compose logs -f mist-backend
```

### Running Tests (inside container; native venv is corrupted)
```bash
docker compose exec -T mist-backend python -m pytest tests/unit/                       # Full unit suite
docker compose exec -T mist-backend python -m pytest tests/integration/                # Integration (Neo4j + llama-server must be up)
docker compose exec -T mist-backend python -m pytest tests/unit/chat/ -v               # Targeted
```

### Admin CLI (`mist_admin`)
```bash
# Inside container:
docker compose exec -T mist-backend python -m scripts.mist_admin stack-status
docker compose exec -T mist-backend python -m scripts.mist_admin graph-stats
docker compose exec -T mist-backend python -m scripts.mist_admin graph-reset --include-derived --confirm
docker compose exec -T mist-backend python -m scripts.mist_admin seed
docker compose exec -T mist-backend python -m scripts.mist_admin chat "utterance" --session-id sid
docker compose exec -T mist-backend python -m scripts.mist_admin replay /app/data/ingest/v6-inputs.jsonl --session-id sid --output /app/data/ingest/report.jsonl
```

### Rebuilding
```bash
docker compose build mist-backend          # After Dockerfile/requirements changes
docker compose build --no-cache mist-backend
```

### Code Quality
```bash
python scripts/check_ai_slop.py --critical-only
pre-commit run --all-files
black backend/
ruff check backend/ --fix
cd mist_desktop && dart format . && flutter analyze
```

---

## Testing

### Backend Tests
- **Count:** 1022 unit tests + 3 xfailed
- **Runner:** pytest inside Docker container
- **Command:** `docker compose exec -T mist-backend python -m pytest tests/unit/`
- **Note:** Tests must run inside container

### Integration Reproducers
Landed per-cluster for regression protection:
- `tests/integration/test_cluster_3_reproducers.py` (7 tests) — persona injection, post-filter regen, identity-intent routing, temperature split
- `tests/integration/test_cluster_5_reproducers.py` (6 tests) — all three observability phases emitting end-to-end
- `tests/integration/test_cluster_6_reproducers.py` (4 tests) — budget-driven history pruning, max_tokens config wiring

---

## Gauntlet Workflow (Cluster Validation)

Each cluster validates acceptance via re-running the V4 (5-utterance smoke), V5 (11-utterance breadth), and V6 (30-turn cohesive session) gauntlets against the merged code.

**Canonical protocol:**
1. `mist_admin graph-reset --include-derived --confirm` — wipe graph (snapshots saved to `data/graph_snapshots/`).
2. `mist_admin seed` — restore 32-entity baseline.
3. `mist_admin replay /app/data/ingest/v6-inputs.jsonl --session-id <name> --output /app/data/ingest/<report>.jsonl` with observability env vars set per Debug section above.
4. Analyze the JSONL diagnostic file + per-turn replay output; write a report under `mist.ai/data/ingest/` (gitignored per policy).
5. Compare against the baseline from the prior cluster's gauntlet report.

**Gauntlet input files:** `data/ingest/v4-inputs.jsonl`, `v5-inputs.jsonl`, `v6-inputs.jsonl` (committed).

**Gauntlet reports:** `data/ingest/post-cluster-<N>-gauntlet-report-YYYY-MM-DD.md` (gitignored by `data/ingest/` convention).

---

## Next Steps

### Immediate
1. **Cluster 1 (ontology expansion + subject-scope classifier)** — closes Bugs I, J, partial K. Extraction correctness 44% -> 80% gate.
2. **Cluster 8 (ADR-010 vault implementation)** — 12-phase roadmap, 4-6 weeks. Parallelizable with Cluster 1 if scoped carefully.
3. **Phase 4 gauntlet re-run** after Cluster 1 lands -> path to earned close.

### Short-term (parked workstreams; unpark after MVP close)
1. `mist-ai-context-compression-multi-session` — post-MVP
2. `mist-ai-tool-calling-production-rigor` — post-MVP
3. `mist-ai-frontend-audit-remediation` — parked; Critical + High items in backlog (MIS-77)
4. `mist-ai-mist-personality-growth` — parked (post-MVP)
5. Voice Profiles / Settings screens in Flutter

### Long-term
1. Command Center architecture (orchestrating agentic teams)
2. Vision integration (Gemma 4 vision)
3. GTX 1070 dual-GPU addition (parked post-voice-integration)
4. Mobile app (Flutter iOS/Android)

---

## Known Issues
- GPU contention between llama-server and Chatterbox adds ~1.1x TTS overhead on single GPU
- Binary audio transport implemented but not E2E validated yet (pending manual test)
- 48 P3 items in KNOWN_ISSUES.md from 2026-03-22 audit (opportunistic resolution, tracked in `mist-ai-technical-debt-p3` parked workstream)
- Git Bash on Windows path-mangles unix-absolute paths in env vars passed to `docker compose exec`; prefix with `MSYS_NO_PATHCONV=1`

---

## Important Files

### Documentation
- **CLAUDE.md** — AI integration guidelines (never push to remote)
- **README.md** — Project overview and setup
- **REPOSITORY_STRUCTURE.md** — File organization
- **CONTRIBUTING.md** — Code quality standards
- **KNOWN_ISSUES.md** — 48 P3 items from backend audit
- **TESTING.md** — Test conventions
- **tests/CLAUDE.md** — Backend test AI guidance
- **mist_desktop/test/CLAUDE.md** — Flutter test AI guidance

### Configuration
- **.env** — Environment variables (never commit)
- **.env.example** — All config with defaults
- **.gitattributes** — Line ending normalization (WSL2/Windows)
- **pyproject.toml** — Python tool configuration
- **.pre-commit-config.yaml** — Pre-commit hooks

### Plan Artifacts
- `~/.claude/plans/cluster-execution-roadmap.md` (canonical) + mirror at `mist.ai/.local/plans/cluster-execution-roadmap.md`
- `~/.claude/plans/2026-04-21-cluster-3-identity-layer.md` (completed)

### Vault Artifacts (persistent memory)
- `knowledge-vault/Projects/mist-ai/workstreams/mist-ai-knowledge-integration-mvp-validation.md` — authoritative workstream state
- `knowledge-vault/Projects/mist-ai/sessions/2026-04-21-mist-cluster-6-context-budget.md` — most recent session
- `knowledge-vault/Decisions/ADR-008-revised-model-backend-selection.md`, `ADR-009-graph-provenance-separation.md` (accepted), `ADR-010-memory-storage-architecture.md` (proposed)

---

## Quick Reference

| Area | Status | Notes |
|---|---|---|
| Backend | CONTAINERIZED | Docker + CUDA 12.4, Gemma 4 E4B via llama-server |
| Knowledge integration | 5/8 clusters done | 2, 3, 4, 5, 6 complete; 1 next; 8 after |
| Unit tests | 1022 + 3 xfailed | Run inside container |
| Empty-response rate (V6) | 0/30 (0%) | Post-Cluster-6 validation |
| TTS | Chatterbox Turbo | 0.74x RTF, 3.9GB VRAM |
| Frontend | IN DEV | Parked on knowledge MVP |
| Code Quality | FULL SUITE | black, ruff, bandit, codespell, AI-slop, pre-commit |
| Docker | COMPLETE | 3-service stack (mist-backend, mist-neo4j, mist-llm) |
| CI/CD | CONFIGURED | GitHub Actions |
