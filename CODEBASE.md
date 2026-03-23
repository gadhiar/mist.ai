# MIST.AI Codebase Context

**Last Updated:** 2026-03-22
**Branch:** fix/pre-rebuild-audit-cleanup (13 commits ahead of origin)
**Status:** Knowledge system Phase 2 complete, Phase 3 next

---

## Current Status

### Backend
- **Status:** PRODUCTION READY
- **Server:** FastAPI WebSocket on port 8001
- **Voice Pipeline:** VAD -> Whisper -> Qwen 2.5 -> Sesame CSM-1B
- **Knowledge Graph:** Full extraction + curation pipeline with automatic knowledge capture
- **Configuration:** TTS currently DISABLED (.env: TTS_ENABLED=false)
- **Code Quality:** [COMPLETE] Full suite - Black, Ruff, Mypy, Bandit, Codespell, AI slop detection
- **Tests:** 146 unit tests passing

### Frontend
- **Status:** IN DEVELOPMENT
- **Platform:** Flutter desktop (Windows/macOS/Linux)
- **Implementation:** WebSocket client, voice recording complete
- **Pending:** Audio playback integration (needs TTS enabled)
- **Planned:** Knowledge graph visualization
- **Code Quality:** [COMPLETE] Flutter analyzer configured with custom rules

---

## Active Work

### Current Focus
1. Phase 4 Tier 3: community detection, centrality, embedding maintenance (in progress)
2. 48 P3 items in KNOWN_ISSUES.md from backend audit (opportunistic)
3. Integration testing with Ollama + Neo4j

### Recently Completed (2026-03-22)
- Phase 1A: Ontology definitions (19 entity types, 30 relationship types)
- Phase 1B: 6-stage extraction pipeline (OntologyConstrainedExtractor)
- Phase 2A: Curation pipeline (dedup, conflict resolution, provenance tracking)
- Phase 2B: ConversationHandler migration (automatic extraction, legacy removal)
- Phase 3: Internal knowledge derivation (MIST self-model, Stage 9)
- Phase 4 Tier 1+2: Periodic curation (decay, staleness, orphans, reflection, health, scheduler)
- Testing foundation: 204 unit tests, DI refactoring, mock factories
- Backend audit: P0/P1/P2 fixes, KNOWN_ISSUES.md tracking P3 items

### Known Issues
- TTS disabled in .env (intentional for development, saves VRAM)
- Flutter audio playback not tested yet (pending TTS enable)

### Blockers
None

---

## Recent Changes

### Post-merge cleanup (2026-03-20)
- Added `.gitattributes` to fix WSL2 line-ending phantom diffs (160 files showing as modified due to CRLF/LF mismatch on /mnt/d/)
- Updated CODEBASE.md to reflect current state on main

### Commits since code/quality merge (6bafae8):
- `6ed1e73` docs: streamline README for public presentation
- `96cc70a` chore: remove internal AI workflow files from tracking
- `4d3c675` chore: fix mypy config, update dependencies, register speech_to_text plugin
- `2b4cd94` docs: reorganize documentation - move historical docs to knowledge-vault
- `0321673` chore: Clean up root directory structure
- `6bafae8` Merge pull request #4 from gadhiar/code/quality

### Code Quality Merge (PR #4, 2025-02-10)
[COMPLETE] Merged code/quality branch into main:

**Quality Tools:**
- Black, Ruff, Mypy, Bandit, Codespell, AI slop checker
- 16 pre-commit hooks configured
- 7 CI checks passing (GitHub Actions)

**Code Fixes:**
- Fixed 228 docstring issues
- Removed 722 emojis/symbols from 16 documentation files
- Fixed undefined logger bug in backend/voice_models/model_manager.py

### Previous Milestones
- Removed obsolete React frontend (saved 127MB)
- Completed Flutter migration with Riverpod state management
- Implemented Flutter WebSocket client and voice recording

---

## Architecture Overview

### Backend Structure
```
backend/
├── server.py              # WebSocket server (port 8001)
├── voice_processor.py     # Voice pipeline orchestration
├── config.py              # Voice system configuration
├── knowledge_config.py    # Knowledge graph configuration
├── voice_models/          # ML model management
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

---

## Configuration

### Environment Variables (.env)
```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Features
ENABLE_KNOWLEDGE_INTEGRATION=true
TTS_ENABLED=false  # Set true to enable audio playback

# Model
MODEL=qwen2.5:7b-instruct
```

### Critical Settings
- TTS_ENABLED: Currently false (saves 2-3GB VRAM during development)
- ENABLE_KNOWLEDGE_INTEGRATION: true (knowledge graph active)

---

## Tech Stack

### Backend
- Python 3.11+
- FastAPI + Uvicorn (WebSocket server)
- Ollama (LLM inference - Qwen 2.5 7B)
- Whisper (STT - base model)
- Sesame CSM-1B (TTS - modified fork)
- Neo4j 5.x (knowledge graph)
- PyTorch 2.4.0 + CUDA 12.4

### Frontend
- Flutter 3.24+ / Dart 3.10+
- Riverpod 3.x (state management)
- web_socket_channel (WebSocket client)
- record (audio recording)
- audioplayers (TTS playback)

---

## Development Workflow

### Starting Work
1. Check this file (CODEBASE.md) for current status
2. Run `git status` and check recent commits
3. Read relevant docs if touching new areas

### Making Changes
1. Update code with proper formatting (black, flutter format)
2. Run checks before committing (pre-commit)
3. Test changes manually or with tests

### Run Code Quality Checks
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
```bash
python test_neo4j_connection.py
python test_conversation_handler.py --mode simple
python test_vector_search.py
```

### Flutter Tests
```bash
cd mist_desktop
flutter test
flutter analyze
```

### Manual Testing
1. Start Neo4j: `neo4j start`
2. Start backend: `python backend/server.py`
3. Run Flutter: `cd mist_desktop && flutter run -d windows`
4. Test voice input and conversation flow

---

## Next Steps

### Immediate Priorities
1. Knowledge/DB failsafe -- persistent memory reliability
2. Command Center architecture design
3. Model routing strategy (local Qwen vs cloud API)

### Short-term Goals
1. Knowledge graph visualization in Flutter
2. Flutter UI polish and error handling
3. Test Flutter audio playback with TTS enabled

### Long-term Goals
1. Agent spawning and task delegation
2. Vision integration (Qwen 2.5 Vision)
3. Meta-reasoning layer for explainability
4. Mobile app (Flutter iOS/Android)

---

## Important Files

### Documentation
- **CLAUDE.md** - AI integration guidelines
- **README.md** - Project overview and setup
- **REPOSITORY_STRUCTURE.md** - File organization
- **CONTRIBUTING.md** - Code quality standards

### Configuration
- **.env** - Environment variables (never commit)
- **.gitattributes** - Line ending normalization (WSL2/Windows)
- **pyproject.toml** - Python tool configuration
- **analysis_options.yaml** - Flutter/Dart linting
- **.pre-commit-config.yaml** - Pre-commit hooks

---

## Quick Reference

| Area | Status | Notes |
|------|--------|-------|
| Backend | PROD READY | TTS disabled for dev |
| Frontend | IN DEV | Audio playback pending |
| Knowledge Graph | COMPLETE | Neo4j integrated |
| Code Quality | COMPLETE | 16 pre-commit hooks, 7 CI checks |
| CI/CD | COMPLETE | GitHub Actions configured |
| Line Endings | FIXED | .gitattributes normalizes to LF |
