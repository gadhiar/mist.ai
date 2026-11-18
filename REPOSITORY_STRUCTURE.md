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
- `docker-compose.dev.yml` - Standalone worktree dev stack on the synthetic seed in `tests/fixtures/dev-instance/` (runbook: that directory's `README.md`)
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
- `render_character.py` - Renders one WAV through the character stage, as the realtime downlink would play it
- `wake/` - Tools for MIST's wake model: the guided recorder, the recording manifest check and the evaluation harnesses
  - `eval_wake.py` - The wake model evaluation harness: stage 1 and the addressee gate, offline, over one split
  - `manifest.py` - The wake recording manifest: its check, and the training subset the model records
  - `record.py` - The guided host-side recorder for one headset sitting: 16 kHz clips and their draft manifest entries
  - `spotter_eval.py` - The server spotter's pass bar, measured after the addressee gate: gated tune wakes and gated false wakes per hour over an ambient set, beside the spotter-only figures written as the converter's evaluation
- `eval_harness/` - Phase 3 evaluator + scorers for V1-V8 gauntlets
  - `extraction_report.py` - Dev and held-out extraction gauntlet scores side by side
  - `extraction_retrieval_score.py` - Scores a retrieval run file at k = 5
  - `extraction_retrieval.py` - Builds the retrieval eval graph from the dev turns and runs the queries
  - `entity_alignment.py` - Pairs gold and produced entity ids by id form, for relations scored over aligned ids
  - `ceiling/` - Extraction ceiling runs: an answerer in the model's seat of the unchanged service (see `data/ceiling/README.md`)
    - `store.py` - Append-only, hash-chained answer store and its state-file check
    - `relay.py` - Serves stored answers to the service as llama-server would; logs pending requests
    - `round.py` - One answer round for answerer K: run the probes, offer pending requests, `--add`, `--check`
    - `replay.py` - Replays all stored answers once over every probe, and compares two replays
    - `make_probes.py` - Writes the utterance-only probes file from gold
    - `score_runs.py` - Scores raw runs with a pinned checkout's scorer and classifier: exact and aligned figures, paired lines, the worksheet and pending pairs
    - `adjudicated.py` - Checks an adjudicated worksheet, prints the corrected figures, and checks the aligner's credits against the adjudicated classes
- `daily/` - The daily instance's host tools
  - `instance.py` - Host CLI for a daily instance root (init, release-build, up, status, gaming mode, night extraction, backups, restore drill, promote)
  - `volume_safety.py` - Refuses `down -v`, `rm -v` and `volume rm` before any `instance.py` command runs
  - `gamewatch.py` - The gaming watcher's process lister, game rules and decision
- `backup/` - Backups of the graph, stores and vault, and their restore
  - `scheduler.py` - Runs backups on a schedule inside the compose project; a bad dump never prunes good ones
  - `status.py` - The backup status file: last attempt, its outcome and the last success
  - `verify.py` - Compares a restored graph, stores and vault with the artifact they came from
- `model_bench/` - Model bench host drivers; `scripts/model_bench/README.md` documents both
  - `bench_host.py` - Stops MIST, serves a pinned llama-server as `mist-bench-llm`, restores the stack
  - `speech_bench.py` - Speech suites CLI: runs 1 to 9 and the report; it also designs voices
  - `speech/` - The speech suites (`tts`, `cache`, `stt`, `turn`, `fit`, `lowtier`, `chain`, `game-session`, `listen-pack`, `voice-design`, `report`), their committed inputs, and `report.py`, which writes `numbers.json` and `REPORT.md`
  - `speech/probes/` - The engine code the speech-bench containers run (synthesis, recognition, end of turn, the load generator)

### Container builds (`docker/`)
- `backend/` - The `mist-backend` Dockerfile
- `extraction/` - The extraction service image, its host-1070 compose file and runbook
- `ollama/` - `init-models.sh`
- `speech-bench/` - Dockerfiles and pinned requirements of the four speech-bench images (base, asr, pocket, qwen3)

### Gold sets (`data/gold/`)
- `data/gold/extraction/` - Extraction gold sets (dev lists, held out) and the labelling guide (`LABELLING.md`)
- `data/gold/retrieval/` - Retrieval queries over the dev gold facts, with a README

### Ceiling runs (`data/ceiling/`)
- `data/ceiling/probes.jsonl` - The dev-lists probes as tag and utterance only, made by `make_probes`
- `data/ceiling/answers-<k>.jsonl` - Answerer k's stored answers (k = 1 to 4), once committed
- `data/ceiling/README.md` - The files, the round and replay commands, and the caveats

---

## Backend Structure

```
backend/
├── server.py                    # FastAPI WebSocket server (port 8001)
├── log_handler.py               # WebSocketLogHandler with rate limiter
├── request_context.py           # ContextVar propagation
├── sentence_detector.py         # Streaming TTS sentence boundary detection
├── debug_jsonl_logger.py        # 5-phase JSONL diagnostic sink
├── factories.py                 # Composition root (DI wiring)
├── errors.py                    # MistError hierarchy
├── interfaces.py                # Protocols (Embedding, VectorStore, GraphConnection, etc.)
├── origins.py                   # Browser origin allowlist for HTTP and the WebSocket
├── gpu_yield.py                 # Gaming mode: yield the GPU while the host marker exists
├── extraction_hold.py           # Reads the host's extraction-paused marker for /health
├── user_timezone.py             # The user's zone from MIST_TIMEZONE, one resolver for every reader
│
├── capabilities/                # Tools that act on the world (README.md: adding one)
│   ├── host.py                  # CapabilityHost: admits, decides, runs and logs every call
│   ├── contracts.py             # Capability base class, ToolSpec, outcomes, the turn port
│   ├── config.py                # Capability environment variables, parsed once
│   ├── router.py                # Per-turn routing under the token ceiling (at most 2 caps)
│   ├── decoder.py               # Schema-constrained action decode for a small model
│   ├── schema.py                # Grammar schemas and catalog lines for tools
│   ├── intents.py               # Utterance normalisation for the fast-path grammars
│   ├── policy.py                # Confirmation policy by tool and call source
│   ├── proposals.py             # Cross-turn proposals and the yes-or-no matcher
│   ├── replies.py               # Templated replies, notices and the actions block
│   ├── power.py                 # Power level and model tiers (stopgaps until the router)
│   ├── action_log.py            # One row per call; the per-day usage view
│   ├── db.py                    # capabilities.db, shared by the capability stores
│   ├── events.py                # WebSocket protocol 1.4.0 capability payloads
│   ├── tool_events.py           # Capability fields of tool_call_started/completed
│   ├── secrets.py               # Provider secrets and their redaction
│   ├── clock.py                 # The injectable Clock protocol and SystemClock
│   ├── alarms/                  # Native, offline alarms, timers and the clock
│   │   ├── capability.py        # Alarms, timers and clock capabilities; alarm control
│   │   ├── records.py           # Records, live occurrences and the event trail
│   │   ├── scheduler.py         # The one writer of occurrence transitions
│   │   ├── ringing.py           # Endpoint-neutral ringing and fallback
│   │   ├── notifier.py          # alarm_fired and alarm_state payloads
│   │   ├── intents.py           # Alarm, timer and clock fast-path rows
│   │   └── timeparse.py         # Spoken times and repeat rules
│   ├── lights/                  # Lights over Home Assistant's MCP server
│   │   ├── capability.py        # LightsCapability over a LightsProvider
│   │   ├── home_assistant.py    # The Home Assistant provider
│   │   ├── catalog.py           # Light names and areas from GetLiveContext
│   │   └── intents.py           # Lights fast-path rows
│   ├── mcp/                     # MCP client: transport protocol, supervised connection
│   │   ├── connection.py        # McpConnection: reconnects with backoff
│   │   ├── transport.py         # The McpTransport protocol
│   │   └── sdk_transport.py     # McpTransport over the mcp SDK
│   └── music/                   # Music over the Spotify Web API
│       ├── capability.py        # MusicCapability
│       ├── spotify_api.py       # Web API client and token manager
│       ├── token_store.py       # The Spotify token file
│       ├── authorize.py         # One-time PKCE authorization CLI
│       ├── pkce.py              # PKCE helpers
│       └── intents.py           # Music fast-path rows
│
├── chat/                        # Conversation handling
│   ├── conversation_handler.py  # Pass-loop + persona + budget + post-filter
│   ├── mist_context.py          # Identity / persona renderer
│   ├── slop_detector.py         # AI-slop pattern filter
│   ├── context_budget.py        # Cluster 6 ContextBudgetPlanner
│   ├── retrieval_trace.py       # Per-turn trace of every retrieval candidate and its fate
│   └── turn_scope.py            # Per-turn state and the value objects of the stream-and-commit turn API
│
├── event_store/                 # Append-only SQLite event store
│   ├── usage.py                 # Real-origin turns per local day (ADR-042 use metric)
│   └── status_intervals.py      # Uptime ledger: /health status intervals per local day
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
│   │   ├── run_identity.py      # Run identity of the setup behind each result
│   │   ├── serving_identity.py  # Serving identity an extraction epoch is keyed on
│   │   └── structure_splitter.py # List, table and heading units for Stage 2
│   ├── curation/                # Dedup, ReconciliationEngine (reconciliation.py,
│   │                            #   bitemporal C2; replaced conflict_resolver.py),
│   │                            #   intervals.py, graph writer, regenerator, health
│   ├── ingestion/               # Markdown ingestion for vector store
│   ├── retrieval/               # Hybrid retrieval (graph + vector + vault RRF)
│   ├── regeneration/            # Legacy no-curation replay (quarantined; R1 redesigns)
│   ├── storage/                 # Neo4j executor, graph store, connection
│   │   └── type_labels.py       # Ontology type labels for entity nodes
│   ├── eval_isolation.py        # F1 fail-closed eval Neo4j allowlist guard
│   ├── canonical_serialize.py   # F3 wall-clock-free graph form
│   └── extraction_cache.py      # F3 content-addressed extraction cache
│
├── memory/                      # Memory read path: per-turn recall into a budgeted block
│   ├── config.py                # MemoryConfig (MIST_MEMORY_*), nested as KnowledgeConfig.memory
│   ├── models.py                # Request/result/item types, fact_key, ServingProfile, MemoryStage (trace memory key)
│   └── fact_render.py           # One sentence per fact, with valid-time qualifiers
│
├── realtime/                    # Realtime voice path, one session per WebSocket connection;
│   │                            #   `server.py` serves every connection through it
│   ├── addressee.py             # Addressee gate: whether an utterance holding MIST's name was said to her
│   ├── bargein.py               # Barge-in policy: what user speech means while MIST speaks, holds a reply or thinks
│   ├── budget.py                # Voice latency budget, voice configuration and their loader from the environment
│   ├── capability_sink.py       # Capabilities' event sink over the session registry: house-wide state to every session
│   ├── character.py             # Character stage: the one place the sound of the spoken voice is shaped
│   ├── degraded.py              # `DirectResponder`: answers turns when knowledge integration is off or failed to start
│   ├── engines/                 # Speech engine adapters implementing the protocols in `speech.py`
│   │   ├── audio.py             # Text and audio rules every TTS adapter shares
│   │   ├── kws.py               # Server name spotter: the device's streaming microWakeWord model on LiteRT behind `NameSpotter`
│   │   ├── nemotron_stt.py      # Nemotron streaming English STT behind `StreamingTranscriber`, on the GPU
│   │   ├── parakeet_stt.py      # Parakeet TDT 0.6B v2 (ONNX) STT behind `StreamingTranscriber`, on the CPU
│   │   ├── pocket_tts.py        # Pocket TTS with voice cloning behind `SpeechSynthesizer`
│   │   ├── silero_vad.py        # Silero VAD (ONNX v5) behind `VoiceActivityDetector`, and the ONNX loading it shares
│   │   └── smart_turn.py        # Smart Turn v3.2 (ONNX) behind `TurnEndDetector`
│   ├── errors.py                # Exceptions raised by the realtime voice package
│   ├── ingress.py               # Per-session audio ingress: uplink frames to speech segments, name hits and utterances
│   ├── lanes.py                 # Inference lanes: one worker thread per engine, so model calls never block the event loop
│   ├── playback.py              # Downlink playback: the session's `turn` lane, the pacer that feeds it and the ledger
│   ├── power.py                 # Power level on the realtime path: the gate that turns and voice read
│   ├── protocol.py              # Realtime protocol 2.0.0: the binary frame codec and the control message schemas
│   ├── registry.py              # The process's sessions: routing, log streaming, voice activity and wake arbitration
│   ├── runtime.py               # The composed realtime runtime, and the walk that loads each speech slot from its chain
│   ├── scene.py                 # Scene seam: what the turn controller tells the screen, anchored to the audio clock
│   ├── segmenter.py             # Cuts a reply's Token stream into the text segments TTS synthesizes one call at a time
│   ├── session.py               # One session per WebSocket: uplink routing, voice settings and the outbound lanes
│   ├── speech.py                # Speech engine contracts: the protocols model adapters implement and the values they exchange
│   ├── turn.py                  # Per-session turn controller: one mailbox task owns a voice session's turns
│   └── voice_assets.py          # Voice identities, the private clips MIST's voice is cloned from, and the artefacts built on them
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

## Daily Instance (`deploy/daily/`)

The pinned-release instance for daily use, run by `scripts/daily/instance.py` from an instance
root outside every git work tree.

```
deploy/daily/
├── compose.yml                  # The mist-daily compose project (backend, Neo4j, llama-server, backup, night extraction LLM and service)
├── daily.env.example            # Template for <root>/daily.env
├── README.md                    # Contract, operation and the /health reasons table
├── RESTORE.md                   # Restore drill and promotion
├── restore-drill.yml            # Drill project: restores a backup into a fresh volume
└── windows/
    ├── register-signin-task.ps1    # Sign-in task that starts the instance (-GamingWatch adds the watcher's, -NightExtraction the extraction window task)
    └── unregister-signin-task.ps1  # Removes all three sign-in tasks, for a rollback
```

## Dev Instance Fixture (`tests/fixtures/dev-instance/`)

The synthetic seed `docker-compose.dev.yml` starts from, and that `instance.py init --seed` reads
for a rehearsal root: an invented self-model and a "Dev User".

```
tests/fixtures/dev-instance/
├── README.md                    # Worktree dev stack runbook
├── seed/                        # mist-seed documents for `mist_admin.py seed`
│   ├── mist.md                  # Self-model (__SelfModel__ partition)
│   └── dev-user.md              # The Dev User's facts
└── vault/                       # Copied into an empty dev vault
    ├── MIST.md
    ├── identity/mist.md
    ├── users/dev-user.md
    └── sessions/                # Two synthesis notes
```

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
- VAD (Silero) -> STT (Whisper) -> LLM (Gemma 4 E4B via llama-server) -> TTS (Chatterbox)
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
  -> one RealtimeSession per connection (backend/realtime/session.py)
  -> binary audio frames OR text messages (protocol 2.0.0)
Per-session TurnController (backend/realtime/turn.py):
  voice turns: VAD -> STT -> end of turn -> LLM -> TTS
  typed turns: text -> LLM
  both answered by the ConversationHandler pass-loop
  -> discriminated events (stream_start, stream_token, stream_complete,
     turn_metrics, voice_decision, tool_call_*, etc.) on the session's
     own socket; process-wide events (heartbeat, health_status, ...)
     through the broadcaster
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
VOICE_PROFILE=mist

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
1. Change turn behaviour in `backend/realtime/turn.py` (the per-session `TurnController`) and audio ingest in `backend/realtime/ingress.py`
2. Add a speech engine as an adapter in `backend/realtime/engines/` implementing a protocol in `backend/realtime/speech.py`, named in a `MIST_VOICE_*_CHAIN` and loaded by `load_speech_engines` (`backend/realtime/runtime.py`)
3. Add WebSocket message types to `backend/realtime/protocol.py`, and to ADR-017 if FE coordination is required
4. Send the new events from the session (`backend/realtime/session.py`), on the lane they belong to (`backend/realtime/lanes.py`)

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
- **Memory read path:** `tests/unit/memory/` (fakes in `tests/fakes/memory_fakes.py`); `tests/fixtures/memory_recall/` holds the synthetic recall fixture (facts, conversations, episodes, vault notes, corrections, probes) that `scripts/eval_harness/memory_recall.py` scores

---

## Current Status

See `CODEBASE.md` for live status, active workstreams, recent commits, and active issues.

**Backend:** Production-ready, fully containerized, post-MVP knowledge integration complete (8 clusters), continuous-usage hardening in progress.

**Frontend:** Production-ready Tauri spatial app (separate repo at `./mist-frontend/`); FE/BE integration Wave 1 shipped 2026-05-10, subsequent waves cover tool-call events, cards, graph_subgraph, and visual polish.

**Flutter Desktop:** Decommissioned 2026-05-11. Git history at `e18c092` preserves the Flutter source.
