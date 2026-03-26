# MIST.AI Codebase Context

**Last Updated:** 2026-03-25
**Branch:** test/chatterbox-eval
**Status:** Knowledge pipeline merged, backend containerized, Chatterbox TTS integrated

---

## Current Status

### Backend
- **Status:** CONTAINERIZED (Docker + CUDA 12.4)
- **Server:** FastAPI WebSocket on port 8001
- **Voice Pipeline:** VAD -> Whisper -> Qwen 2.5 7B -> Chatterbox Turbo TTS
- **Knowledge Graph:** Full extraction + curation pipeline merged to main (655 tests, 369 new)
- **Configuration:** TTS now ENABLED by default (TTS_ENABLED=true, TTS_ENGINE=chatterbox)
- **Code Quality:** [COMPLETE] Full suite -- Black, Ruff, Mypy, Bandit, Codespell, AI slop detection
- **Tests:** 655 unit tests (run inside container)

### Frontend
- **Status:** IN DEVELOPMENT
- **Platform:** Flutter desktop (Windows/macOS/Linux)
- **Implementation:** WebSocket client, voice recording complete
- **Pending:** Audio playback integration testing with Chatterbox TTS
- **Planned:** Knowledge graph visualization
- **Code Quality:** [COMPLETE] Flutter analyzer configured with custom rules

---

## Active Work

### Current Focus
1. Seed knowledge DB, run full end-to-end test
2. Setup FRIDAY voice profile (Chatterbox reference audio)
3. Clean up ~176GB CSM training data (legacy, no longer needed)
4. Merge test/chatterbox-eval to main
5. Knowledge graph visualization in Flutter

### Recently Completed (2026-03-25)
- Knowledge extraction + curation pipeline merged to main (23 modified + 12 new files)
- Chatterbox Turbo selected after 18-model TTS evaluation (replaces Sesame CSM-1B)
- Docker Compose stack created (backend + Neo4j 5 + Ollama -- 3 services)
- Chatterbox adapter implemented in model_manager.py
- CSM imports made lazy (no torchtune dependency when using Chatterbox)
- Voice profile system extended for Chatterbox (tts_engine, reference_audio_path)
- Flash attention confirmed working inside Linux container (PyTorch 2.6+cu124)
- 655 backend tests passing (369 new from knowledge pipeline)

### Known Issues
- Native Windows venv is corrupted -- use Docker container going forward
- Flutter audio playback not tested yet (pending e2e test with Chatterbox)
- ~176GB of legacy CSM training data needs cleanup
- 48 P3 items in KNOWN_ISSUES.md from 2026-03-22 audit (opportunistic)

### Blockers
None

---

## Recent Changes

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
├── server.py              # WebSocket server (port 8001)
├── voice_processor.py     # Voice pipeline orchestration
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
│   ├── services/          # WebSocket, audio services
│   ├── screens/           # UI screens
│   ├── widgets/           # Reusable components
│   └── models/            # Data models
```

### Key Integration Points
- Flutter connects to backend via WebSocket (ws://localhost:8001)
- Backend sends: transcriptions, LLM tokens, audio chunks
- Frontend sends: audio data, text messages, interrupts
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
VOICE_PROFILE=jarvis
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
- **Count:** 655 unit tests (369 new from knowledge pipeline)
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
1. Seed knowledge DB, run full e2e test
2. Setup FRIDAY voice profile (Chatterbox reference audio)
3. Clean up ~176GB legacy CSM training data
4. Merge test/chatterbox-eval to main

### Short-term Goals
1. Knowledge graph visualization in Flutter
2. Flutter audio playback testing with Chatterbox
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
| Frontend | IN DEV | Audio playback pending |
| Code Quality | COMPLETE | 16 pre-commit hooks, 7 CI checks |
| Docker | COMPLETE | 3-service Compose stack |
| CI/CD | COMPLETE | GitHub Actions configured |
| Line Endings | FIXED | .gitattributes normalizes to LF |
