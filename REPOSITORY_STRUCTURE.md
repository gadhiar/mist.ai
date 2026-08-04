# MIST.AI - Repository Structure

Python backend repository. Frontend is in a separate repo at `./mist-frontend/` (Tauri 2.x + React 19 + react-three-fiber).

---

## Root Directory Files

### Configuration
- `.env` - Environment variables (Neo4j, model paths, feature flags)
- `.env.example` - Template with all configurable variables
- `.python-version` - Python version (3.11)
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Tool config (black, ruff, bandit, codespell, pytest)
- `docker-compose.yml` + `docker-compose.override.yml` - 3-service stack (backend + Neo4j + llama-server)
- `.pre-commit-config.yaml` - Pre-commit hook chain
- `.gitignore`, `.dockerignore`, `.gitattributes` - VCS / container ignores

### Documentation
- `README.md` - Project overview and quick start
- `CLAUDE.md` - AI integration guide and project rules
- `CODEBASE.md` - Current codebase status, recent changes, active work
- `CONTRIBUTING.md` - Contribution guide and code standards
- `TESTING.md` - Test conventions
- `KNOWN_ISSUES.md` - P3 backlog from audits
- `REPOSITORY_STRUCTURE.md` - This file
- `LICENSE` - MIT license
- `NOTICE` - Attribution notices

### Scripts (`scripts/`)
- `start_dev.py` - Docker compose stack manager (start / stop / restart / logs)
- `mist_admin.py` - Admin CLI (seed, seed-verify, replay, graph-stats, graph-reset, vault-rebuild, etc.)
- `check_ai_slop.py` - AI-slop pattern checker (used by pre-commit)
- `eval_harness/` - Phase 3 evaluator + scorers for V1-V8 gauntlets

---

## Backend Structure

```
backend/
├── server.py                    # FastAPI WebSocket server (port 8001)
├── voice_processor.py           # Voice pipeline orchestration
├── audio_protocol.py            # MIST binary audio frame builder
├── log_handler.py               # WebSocketLogHandler with rate limiter
├── request_context.py           # ContextVar propagation
├── sentence_detector.py         # Streaming TTS sentence boundary detection
├── debug_jsonl_logger.py        # 5-phase JSONL diagnostic sink
├── factories.py                 # Composition root (DI wiring)
├── errors.py                    # MistError hierarchy
├── interfaces.py                # Protocols (Embedding, VectorStore, GraphConnection, etc.)
│
├── chat/                        # Conversation handling
│   ├── conversation_handler.py  # Pass-loop + persona + budget + post-filter
│   ├── knowledge_integration.py # Bridge to voice system
│   ├── mist_context.py          # Identity / persona renderer
│   ├── slop_detector.py         # AI-slop pattern filter
│   └── context_budget.py        # Cluster 6 ContextBudgetPlanner
│
├── llm/                         # LLM provider abstraction
│   ├── provider.py              # Abstract StreamingLLMProvider
│   ├── llama_server_provider.py # Primary (Gemma 4 E4B via llama.cpp)
│   ├── ollama_provider.py       # Fallback
│   ├── instrumented_provider.py # Cluster 5 telemetry wrapper
│   └── models.py                # LLMRequest / LLMResponse / ToolCall (Pydantic)
│
├── knowledge/                   # Knowledge graph system
│   ├── config.py                # KnowledgeConfig + nested configs
│   ├── models.py                # RetrievalResult, QueryIntent, etc.
│   ├── embeddings.py            # Sentence Transformers all-MiniLM-L6-v2
│   ├── extraction/              # 6-stage extraction + subject-scope classifier
│   ├── curation/                # Dedup, ReconciliationEngine (reconciliation.py,
│   │                            #   bitemporal C2; replaced conflict_resolver.py),
│   │                            #   intervals.py, graph writer, regenerator, health
│   ├── ingestion/               # Markdown ingestion for vector store
│   ├── retrieval/               # Hybrid retrieval (graph + vector + vault RRF)
│   ├── regeneration/            # Legacy no-curation replay (quarantined; R1 redesigns)
│   ├── storage/                 # Neo4j executor, graph store, connection
│   ├── eval_isolation.py        # F1 fail-closed eval Neo4j allowlist guard
│   ├── canonical_serialize.py   # F3 wall-clock-free graph form
│   └── extraction_cache.py      # F3 content-addressed extraction cache
│
└── vault/                       # ADR-010 vault layer (Cluster 8)
    ├── conventions.py           # MIST.md auto-load primitive (ADR-014)
    ├── sidecar_index.py         # sqlite-vec + FTS5 over markdown chunks
    ├── filewatcher.py           # watchdog daemon thread, 500ms debounce
    └── (writers, models, etc.)
```

---

## Frontend (Separate Repository)

The MIST frontend lives at `./mist-frontend/` (separate git repo, no remote configured per current intent). Stack: Tauri 2.x shell + React 19 + TypeScript strict + react-three-fiber for 3D composition. See that repository's own documentation for its internal structure.

Integration with this backend is contract-only:
- ADR-016 (LLM-mediated frontend tool calls — backend decides routing)
- ADR-017 (WebSocket message contract — discriminated events, lifecycle, error model)

Both ADRs live in `knowledge-vault/Decisions/` (cross-project ADR home), since the message contract spans both repos.

The Flutter Desktop app at `mist_desktop/` was decommissioned 2026-05-11 after the pivot to the Tauri frontend. Git history at commit `e18c092` preserves the Flutter source code if reference is needed.

---

## Vault Layer (`mist-memory/`)

Per ADR-010 four-layer memory architecture. Filesystem markdown corpus + sidecar index.

```
mist-memory/
├── MIST.md                      # Vault conventions (auto-loaded per ADR-014)
├── seed/                        # Versioned seed source (R1.4 spec 2.0)
│   ├── mist.md                  # Self-model facts + body -- mist_admin.py seed
│   └── user.md                  # User facts + body -- mist_admin.py seed
├── sessions/                    # YYYY-MM-DD-<slug>.md per turn-stream session
├── identity/
│   └── mist.md                  # MIST self-model (traits, prefs, capabilities)
├── users/
│   └── <user-id>.md             # User canonical fact sheet
├── decisions/                   # Per-vault decision notes (DEC-NNN)
└── meta/
    ├── schema.md
    └── changelog.md
```

Sidecar index at `data/vault_sidecar.db` (sqlite-vec `vec0` + FTS5 over heading-block + file-level chunks).

---

## Documentation (`docs/`)

```
docs/
├── decisions/                   # Repo-scoped ADRs (snake_case adr_NNN_*.md)
│   └── adr_008_lancedb_vector_store.md   # the only ADR held in-repo
├── audit/                       # Historical audit reports (Flutter-era; superseded)
├── guides/                      # Setup + reference guides
└── superpowers/specs/           # Phase implementation specs
```

Cross-project ADRs (memory architecture, integration contracts, etc.) live in `knowledge-vault/Decisions/` with the `ADR-NNN-kebab-case.md` convention. See vault for the authoritative cross-project decision set.

Almost every ADR is in the vault, not in-repo. This section previously listed `adr_001_vision.md` and `adr_007_sesame_csm.md` under `docs/decisions/`; neither file has ever existed there. Those decisions are `knowledge-vault/Decisions/ADR-002-mist-vision-and-architecture.md` and `ADR-005-sesame-csm-tts.md`. The two numbering schemes are independent -- an `adr_NNN` in this repo does not correspond to `ADR-NNN` in the vault (in-repo `adr_008_lancedb_vector_store.md` is vault `ADR-007-lancedb-vector-store.md`). `ls docs/decisions/` before citing a repo-scoped ADR path.

---

## Key Features

### 1. Knowledge System (four-layer per ADR-010)
- **Event store** (SQLite, append-only) - raw turn evidence
- **Vault** (mist-memory/ markdown) - canonical, user-editable history
- **Graph** (Neo4j) - MIST's reasoning substrate; typed entities + relationships
- **Sidecar index** (sqlite-vec + FTS5) - hybrid retrieval over vault

### 2. Conversation Pipeline
- LLM-decided tool use (currently `query_knowledge_graph`)
- Pass-loop in `ConversationHandler` with bounded depth
- Context budget planner (Cluster 6)
- AI-slop post-filter with regen / strip fallback
- Persona injection (MistIdentity, Cluster 3)

### 3. Voice Pipeline
- VAD (Silero) -> STT (Whisper) -> LLM (Gemma 4 E4B via llama-server) -> TTS (Chatterbox Turbo)
- Streaming parallelism (~4-5s TTFA)
- Binary WebSocket audio protocol (MIST 16-byte frame header)
- Interrupt fade-out, RMS normalization

### 4. Observability
- DebugJSONLLogger with 5 phases (turn, extraction, llm_call, retrieval_candidates, llm_request_raw)
- Per-phase env-gated emission
- Telemetry events flow to frontend per ADR-017

---

## Integration Points

### Tauri Frontend -> Backend
```
Tauri Frontend (./mist-frontend/)
  -> WebSocket (ws://localhost:8001/ws)
Backend Server (server.py)
  -> binary audio frames OR text messages
Voice / chat pipelines:
  STT -> LLM -> TTS pipelines on voice path
  ConversationHandler pass-loop on text path
  -> discriminated events (state_cycle, stream_token, stream_complete,
     tool_call_*, vad_status, health_status, etc.) per ADR-017
Tauri Frontend renders spatial composition
```

### Backend -> Knowledge Graph
- `KnowledgeRetriever.retrieve()` -> graph + vector + RRF merge -> facts injected into context
- `ExtractionPipeline.extract_from_utterance()` (fire-and-forget) -> typed entities + relationships -> CurationPipeline -> GraphStore -> Neo4j MERGE upserts, anchored by an `EXTRACTED_FROM -> ConversationContext` edge carrying `source_utterance_id`. R1.3 re-anchored provenance here from the retired `DERIVED_FROM -> VaultNote` edge; `DERIVED_FROM` survives in the ontology only for synthesis-sourced entity-to-chunk edges (`graph_writer.py`), not for extraction.

### Backend -> Vault
- `VaultWriter.write_session_note()` at session end. R1.3.1 retired the per-turn `append_turn_to_session` write; the note is rendered whole from the session synthesis, and `tests/unit/vault/test_no_per_turn_vault_write.py` pins that the per-turn path stays gone.
- `VaultSidecarIndex` reindex on filewatcher events (500ms debounce)
- `mist_admin vault-rebuild --confirm` drops and re-indexes the whole sidecar; `mist_admin vault-reindex --scope <path>` re-indexes a single note. `--scope` is a `vault-reindex` flag -- `vault-rebuild` does not parse it.

---

## Environment Variables

See `.env.example` for the full list. Required for runtime:

```bash
# Neo4j
NEO4J_URI=bolt://mist-neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM backend
LLM_BACKEND=llamacpp
LLM_SERVER_URL=http://mist-llm:8080
LLM_MODEL_FILE=unsloth/gemma-4-E4B-it-Q5_K_M.gguf
MODEL=gemma-4-e4b

# Voice / TTS
TTS_ENABLED=true
TTS_ENGINE=chatterbox
VOICE_PROFILE=friday

# Feature flags
ENABLE_KNOWLEDGE_INTEGRATION=true

# Event store + vector store
EVENT_STORE_DB_PATH=/app/data/event_store.db
EVENT_STORE_AUDIO_DIR=/app/data/audio
VECTOR_STORE_DATA_DIR=/app/data/vector_store
```

Optional observability gates: `MIST_DEBUG_JSONL`, `MIST_DEBUG_LLM_JSONL`, `MIST_DEBUG_RETRIEVAL_JSONL`, `MIST_DEBUG_LLM_REQUESTS`.

---

## Usage

### Start the backend stack
```bash
# Via Docker compose
docker compose up -d

# Or via dev script
python scripts/start_dev.py

# Tail logs
docker compose logs -f mist-backend
```

### Run tests (inside container; native Windows venv is corrupted)
```bash
docker compose exec mist-backend python -m pytest tests/unit/ -q
```

### Admin commands
```bash
docker compose exec mist-backend python -m scripts.mist_admin stack-status
docker compose exec mist-backend python -m scripts.mist_admin graph-stats
docker compose exec mist-backend python -m scripts.mist_admin seed
docker compose exec mist-backend python -m scripts.mist_admin chat "utterance" --session-id sid
docker compose exec mist-backend python -m scripts.mist_admin replay /app/data/ingest/v6-inputs.jsonl --session-id sid --output /app/data/ingest/report.jsonl
```

### Start the frontend (separate repo)
```bash
cd mist-frontend
npm install
npm run dev   # Vite dev server on localhost:1420 + Tauri shell window
```

---

## Development Workflow

### Adding new knowledge features
1. Define data models in `backend/knowledge/models.py`
2. Add extraction logic in `backend/knowledge/extraction/`
3. Add retrieval logic in `backend/knowledge/retrieval/`
4. Update `ConversationHandler` tool catalog if needed
5. Add unit tests under `tests/unit/knowledge/`

### Adding new voice features
1. Modify `backend/voice_processor.py` for pipeline changes
2. Update `backend/voice_models/model_manager.py` for model integration
3. Add WebSocket message types to ADR-017 if FE coordination required
4. Emit the new events from `server.py` / `voice_processor.py`

### Adding new vault primitives
1. Extend `backend/vault/` (writer / sidecar / filewatcher) per ADR-010 invariants
2. Update Pydantic schemas if frontmatter shape changes
3. Add CLI surface via `scripts/mist_admin.py` if needed

### Frontend changes
Frontend lives in a separate repo at `./mist-frontend/`. See that repo's contributing guide. Cross-repo coordination happens via ADR-016 / ADR-017 protocol updates.

---

## Tests

- **Unit tests:** `tests/unit/` (>1400 tests as of 2026-05-11). Run inside container.
- **Integration tests:** `tests/integration/` (cluster-scoped reproducers; require live Neo4j + llama-server)
- **Drift guards:** `tests/unit/test_eval_harness_scorers.py` locks scorer frozensets to ontology; `tests/unit/knowledge/extraction/test_validator.py::TestValidatorOntologyConsistency` locks validator to ontology
- **Eval harness:** `scripts/eval_harness/` runs V1-V8 gauntlets; reports in `data/ingest/`

---

## Current Status

See `CODEBASE.md` for live status, active workstreams, recent commits, and active issues.

**Backend:** Production-ready, fully containerized, post-MVP knowledge integration complete (8 clusters), continuous-usage hardening in progress.

**Frontend:** Production-ready Tauri spatial app (separate repo at `./mist-frontend/`); FE/BE integration Wave 1 shipped 2026-05-10, subsequent waves cover tool-call events, cards, graph_subgraph, and visual polish.

**Flutter Desktop:** Decommissioned 2026-05-11. Git history at `e18c092` preserves the Flutter source.
