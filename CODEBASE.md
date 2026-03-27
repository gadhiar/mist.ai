# MIST.AI Codebase Context

**Last Updated:** 2026-03-26
**Branch:** main
**Status:** Voice pipeline optimized (4-5s TTFA), log streaming active, Flutter nav rail + log viewer operational

---

## Current Status

### Backend
- **Status:** CONTAINERIZED (Docker + CUDA 12.4)
- **Server:** FastAPI WebSocket on port 8001
- **Voice Pipeline:** VAD -> Whisper -> Qwen 2.5 7B -> Chatterbox Turbo TTS (pipeline parallelism, ~4-5s TTFA)
- **Log Streaming:** WebSocketLogHandler with per-logger gating, token bucket rate limiter, request ID propagation
- **Persistent Logging:** `./logs/mist-backend.log` at DEBUG level (survives container removal)
- **Knowledge Graph:** Full extraction + curation pipeline (655 tests, 369 new)
- **Configuration:** TTS_ENABLED=true, TTS_ENGINE=chatterbox, VOICE_PROFILE=friday
- **Code Quality:** [COMPLETE] Full suite -- Black, Ruff, Mypy, Bandit, Codespell, AI slop detection
- **Tests:** 655+ unit tests (run inside container), 14 sentence detector tests, token streaming tests

### Frontend
- **Status:** IN DEVELOPMENT
- **Platform:** Flutter desktop (Windows/macOS/Linux)
- **Implementation:** WebSocket client, voice recording, nav rail, log viewer [TESTED]
- **Nav Rail:** Expandable sidebar (72px/200px), 4 destinations (Chat, Logs, Voice Profiles stub, Settings stub)
- **Log Viewer:** Level filter chips, text search (300ms debounce), group by request/component, ring buffer (5,000 entries)
- **Pending:** Knowledge graph visualization, binary WebSocket audio transport
- **Code Quality:** [COMPLETE] Flutter analyzer configured with custom rules

---

## Active Work

### Current Focus
1. Binary WebSocket audio transport (Phase 2 -- 7x bandwidth reduction)
2. Knowledge graph visualization in Flutter
3. Seed knowledge DB, run full e2e knowledge integration test
4. Merge to origin/main when ready to push remotely

### Recently Completed (2026-03-26)
- Merged test/chatterbox-eval to main (Docker, Chatterbox, 10 commits)
- Backend log streaming: WebSocketLogHandler, request ID propagation via contextvars, runtime log level control
- Persistent file logging to ./logs/mist-backend.log (DEBUG level, survives container removal)
- Flutter nav rail + log viewer (expandable sidebar, 4 destinations, filter/search/group, ring buffer)
- Voice pipeline parallelism: LLM producer + TTS consumer via sentence queue
- SentenceBoundaryDetector with abbreviation/decimal/ellipsis/list marker handling (14 tests)
- True Ollama token streaming in KnowledgeIntegration (bypasses ConversationHandler tool chain)
- Duration-based pre-buffering (6s) and sentence coalescing (40 char min)
- Time-to-first-audio improved from 12-19s to ~4-5s
- Eager embedding model loading, CUDA sync removal
- FRIDAY voice profile created (46.2s reference WAV), all 3 profiles migrated to Chatterbox
- Default voice profile switched to friday
- CSM training data deleted (~176GB), Marvel's Avengers deleted
- Start script fixed: health-based waiting, no forced rebuild, Ollama healthcheck
- Missing langchain-ollama dependency fixed

### Previously Completed (2026-03-25)
- Knowledge extraction + curation pipeline merged to main (23 modified + 12 new files)
- Chatterbox Turbo selected after 18-model TTS evaluation (replaces Sesame CSM-1B)
- Docker Compose stack created (backend + Neo4j 5 + Ollama -- 3 services)
- Flash attention confirmed working inside Linux container (PyTorch 2.6+cu124)
- 655 backend tests passing (369 new from knowledge pipeline)

### Known Issues
- Native Windows venv is corrupted -- use Docker container going forward
- GPU contention between Ollama and Chatterbox adds ~1.1x TTS overhead on single GPU
- Log handler uses bare import (`from log_handler import ...`) not relative
- Knowledge DB has not been seeded -- RAG retrieval returns 0 facts
- 43 commits ahead of origin/main (not pushed per policy)
- 48 P3 items in KNOWN_ISSUES.md from 2026-03-22 audit (opportunistic)

### Blockers
None

---

## Recent Changes

### Log Streaming + Nav Rail + Voice Optimization (2026-03-26)
- Backend log streaming via WebSocketLogHandler (structured records, rate limiting, re-entrancy guard)
- Request ID propagation via contextvars with spawn_with_context utility
- Runtime log level control via log_config WebSocket messages
- Flutter nav rail (72px collapsed / 200px expanded, 200ms animation, 4 destinations)
- Log viewer: level filter chips, text search (300ms debounce), group by request/component
- LogEntryTile: monospace, color-coded levels, tap-to-expand, right-click context menu
- Ring buffer (5,000 entries, Queue for O(1)), 100ms batched updates, auto-scroll with 50px threshold
- Voice pipeline parallelism: LLM token streaming -> sentence boundary detection -> TTS consumer
- Pre-buffering (6s) and sentence coalescing (40 char min) for gapless audio
- Time-to-first-audio: 12-19s -> ~4-5s (70-81% reduction)
- 26 files changed, +4,616 lines

### Chatterbox TTS + Docker Stack (2026-03-25)
- Chatterbox Turbo replaces Sesame CSM-1B as primary TTS engine
  - Performance: 0.74x RTF vs 2.3x RTF (CSM)
  - VRAM: 3.9GB vs 10GB (CSM)
  - Zero-shot voice cloning, MIT license
- Docker Compose stack: mist-backend + Neo4j 5 + Ollama
- Dev mode: docker-compose.override.yml mounts code as read-only volumes
- Backend container: nvidia/cuda:12.4.0-devel-ubuntu22.04 + Python 3.11
- PyTorch 2.6.0+cu124 with flash attention inside container

### Knowledge Pipeline Merge (2026-03-25)
- 23 modified files + 12 new files merged to main
- 655 tests total (369 new)
- Full extraction + curation pipeline with automatic knowledge capture

### Previous Milestones
- Phase 1A: Ontology definitions (19 entity types, 30 relationship types)
- Phase 1B: 6-stage extraction pipeline (OntologyConstrainedExtractor)
- Phase 2A: Curation pipeline (dedup, conflict resolution, provenance tracking)
- Phase 2B: ConversationHandler migration (automatic extraction, legacy removal)
- Phase 3: Internal knowledge derivation (MIST self-model, Stage 9)
- Phase 4 Tier 1+2: Periodic curation (decay, staleness, orphans, reflection, health, scheduler)
- Backend audit: P0/P1/P2 fixes, KNOWN_ISSUES.md tracking P3 items
- Code quality merge (PR #4): 16 pre-commit hooks, 7 CI checks
- Removed obsolete React frontend (saved 127MB)
- Flutter migration with Riverpod state management

---

## Architecture Overview

### Docker Stack
```
docker/
├── backend/
│   ├── Dockerfile              # CUDA 12.4 + Python 3.11 + Chatterbox
│   └── .dockerignore           # Excludes CSM training data
└── ollama/
    └── init-models.sh          # First-run model pull
docker-compose.yml              # 3-service stack (backend, neo4j, ollama)
docker-compose.override.yml     # Dev mode volume mounts
```

### Backend Structure
```
backend/
├── server.py              # WebSocket server (port 8001), log handler attachment
├── voice_processor.py     # Voice pipeline orchestration (pipeline parallelism)
├── log_handler.py         # WebSocketLogHandler (rate limiting, re-entrancy guard)
├── request_context.py     # ContextVar propagation + spawn_with_context
├── sentence_detector.py   # Sentence boundary detection for streaming TTS
├── config.py              # Voice system configuration
├── knowledge_config.py    # Knowledge graph configuration
├── voice_models/          # ML model management (Chatterbox adapter)
├── chat/                  # Conversation handling + tool usage
└── knowledge/             # Neo4j knowledge graph system
```

### Frontend Structure
```
mist_desktop/
├── lib/
│   ├── main.dart
│   ├── providers/         # Riverpod state management
│   │   ├── chat_provider.dart
│   │   ├── log_provider.dart        # LogNotifier + LogState (ring buffer)
│   │   └── navigation_provider.dart # Sidebar state
│   ├── services/          # WebSocket, audio services
│   ├── screens/           # UI screens
│   │   ├── chat_screen.dart
│   │   ├── log_screen.dart          # Log viewer with grouping
│   │   └── stub_screen.dart         # Placeholder destinations
│   ├── widgets/           # Reusable components
│   │   ├── app_shell.dart           # Navigation rail + content area
│   │   ├── log_entry_tile.dart      # Monospace log entry display
│   │   └── log_toolbar.dart         # Filter chips, search, controls
│   └── models/            # Data models
│       ├── websocket_message.dart   # Includes log message types
│       └── log_entry.dart           # LogEntry data class
```

### Key Integration Points
- Flutter connects to backend via WebSocket (ws://localhost:8001)
- Backend sends: transcriptions, LLM tokens, audio chunks, structured log entries
- Frontend sends: audio data, text messages, interrupts, log_config (level control)
- Services communicate via Docker container hostnames (mist-neo4j, mist-ollama)

---

## Configuration

### Environment Variables (.env)
```bash
# Neo4j (containerized)
NEO4J_URI=bolt://mist-neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Features
ENABLE_KNOWLEDGE_INTEGRATION=true
TTS_ENABLED=true
TTS_ENGINE=chatterbox

# TTS Parameters
VOICE_PROFILE=friday
TTS_EXAGGERATION=0.5
TTS_TEMPERATURE=0.8
TTS_CFG_WEIGHT=0.5

# Model
MODEL=qwen2.5:7b-instruct
```

### Critical Settings
- TTS_ENABLED=true (Chatterbox enabled by default)
- TTS_ENGINE=chatterbox (legacy CSM still available but not recommended)
- ENABLE_KNOWLEDGE_INTEGRATION=true (knowledge graph active)
- Docker data root: D:\Users\rajga\DockerData (not default C:)
- .env.example has all config with defaults

---

## Tech Stack

### Backend
- Python 3.11+
- FastAPI + Uvicorn (WebSocket server)
- Docker (nvidia/cuda:12.4.0-devel-ubuntu22.04)
- PyTorch 2.6.0+cu124 (flash attention enabled)
- Ollama (LLM inference -- Qwen 2.5 7B)
- Whisper (STT -- base model)
- Chatterbox Turbo (TTS -- zero-shot voice cloning, MIT license)
- Neo4j 5.x (knowledge graph)
- Sentence Transformers (all-MiniLM-L6-v2, 384-dim embeddings)

### Legacy (retained but not active)
- Sesame CSM-1B (replaced by Chatterbox -- imports made lazy, no torchtune dependency)

### Frontend
- Flutter 3.24+ / Dart 3.10+
- Riverpod 3.x (state management)
- web_socket_channel (WebSocket client)
- record (audio recording)
- audioplayers (TTS playback)

---

## Development Workflow

### Starting the Stack
```bash
# Start full stack (backend + Neo4j + Ollama)
docker compose up -d

# Or use dev script
python scripts/start_dev.py

# View logs
docker compose logs -f mist-backend
```

### Running Tests
```bash
# Run tests inside container (native venv is corrupted)
docker compose run --rm --no-deps mist-backend pytest tests/unit/

# Run specific test file
docker compose run --rm --no-deps mist-backend pytest tests/unit/test_specific.py
```

### Building and Rebuilding
```bash
# Rebuild after Dockerfile changes
docker compose build mist-backend

# Full rebuild (no cache)
docker compose build --no-cache mist-backend
```

### Code Quality Checks
```bash
python scripts/check_ai_slop.py --critical-only
pre-commit run --all-files
black backend/
ruff check backend/ --fix
cd mist_desktop && dart format . && flutter analyze
```

### Flutter Development
```bash
cd mist_desktop
flutter test
flutter analyze
flutter run -d windows
```

---

## Testing

### Backend Tests
- **Count:** 655+ unit tests (369 from knowledge pipeline, 14 sentence detector, token streaming tests)
- **Runner:** pytest inside Docker container
- **Command:** `docker compose run --rm --no-deps mist-backend pytest tests/unit/`
- **Note:** Tests must run inside container -- native Windows venv is missing dependencies

### Flutter Tests
```bash
cd mist_desktop
flutter test
flutter analyze
```

### Manual End-to-End Testing
1. Start stack: `docker compose up -d`
2. Run Flutter: `cd mist_desktop && flutter run -d windows`
3. Test voice input and conversation flow
4. Verify TTS audio playback

---

## Next Steps

### Immediate Priorities
1. Binary WebSocket audio transport (Phase 2 -- 7x bandwidth reduction, gapless playback)
2. Knowledge graph visualization in Flutter
3. Seed knowledge DB, run full e2e knowledge integration test
4. Merge to origin/main when ready

### Short-term Goals
1. Voice Profiles screen (currently stub) -- manage/preview voice profiles from Flutter
2. Settings screen (currently stub) -- runtime config from Flutter
3. Address P3 items from KNOWN_ISSUES.md (opportunistic)

### Long-term Goals
1. Command Center architecture (orchestrating agentic teams)
2. Agent spawning and task delegation
3. Model routing strategy (local Qwen vs cloud API)
4. Vision integration (Qwen 2.5 Vision)
5. Meta-reasoning layer for explainability
6. Mobile app (Flutter iOS/Android)

---

## Important Files

### Documentation
- **CLAUDE.md** -- AI integration guidelines
- **README.md** -- Project overview and setup
- **REPOSITORY_STRUCTURE.md** -- File organization
- **CONTRIBUTING.md** -- Code quality standards
- **KNOWN_ISSUES.md** -- 48 P3 items from backend audit

### Configuration
- **.env** -- Environment variables (never commit)
- **.env.example** -- All config with defaults
- **.gitattributes** -- Line ending normalization (WSL2/Windows)
- **pyproject.toml** -- Python tool configuration
- **analysis_options.yaml** -- Flutter/Dart linting
- **.pre-commit-config.yaml** -- Pre-commit hooks
- **docker-compose.yml** -- 3-service stack definition
- **docker-compose.override.yml** -- Dev mode volume mounts

---

## Quick Reference

| Area | Status | Notes |
|------|--------|-------|
| Backend | CONTAINERIZED | Docker + CUDA 12.4 |
| TTS | Chatterbox Turbo | 0.74x RTF, 3.9GB VRAM |
| Knowledge | COMPLETE | 655 tests, Neo4j integrated |
| Frontend | IN DEV | Nav rail + log viewer operational |
| Code Quality | COMPLETE | 16 pre-commit hooks, 7 CI checks |
| Docker | COMPLETE | 3-service Compose stack |
| CI/CD | COMPLETE | GitHub Actions configured |
| Line Endings | FIXED | .gitattributes normalizes to LF |
