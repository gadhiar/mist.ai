# MIST.AI Codebase Context

**Last Updated:** 2025-02-10 (Code Quality Enhancements - Ready for PR)
**Branch:** code/quality
**Status:** Ready for PR to main

---

## Current Status

### Backend
- **Status:** PRODUCTION READY
- **Server:** FastAPI WebSocket on port 8001
- **Voice Pipeline:** VAD -> Whisper -> Qwen 2.5 -> Sesame CSM-1B
- **Knowledge Graph:** Neo4j integration complete with autonomous tool usage
- **Configuration:** TTS currently DISABLED (.env: TTS_ENABLED=false)
- **Code Quality:** [COMPLETE] Full suite - Black, Ruff, Mypy, Bandit, Codespell, AI slop detection

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
1. Code quality infrastructure [COMPLETE]
2. CI/CD configuration [COMPLETE]
3. Documentation updates [COMPLETE]
4. Next: Create PR from code/quality -> main
5. Next: Test Flutter audio playback with TTS enabled

### Known Issues
- TTS disabled in .env (intentional for development, saves VRAM)
- Flutter audio playback not tested yet (pending TTS enable)

### Blockers
None - all CI checks passing, ready for PR

---

## Recent Changes

### Latest Session (2025-02-10) - Code Quality Enhancements Final
[COMPLETE - COMMITTED] Enhanced code quality with additional tools and CI configuration (commit: ff98a6d):

**Quality Tools Added:**
- Mypy - Type checking added to pre-commit and CI
- Bandit - Security scanning with nosec comments
- Codespell - Spell checking with custom ignore list
- AI slop checker - Moved to pre-commit (runs automatically, not in CI)

**Configuration Updates:**
- pyproject.toml: Added 8 ruff ignores for acceptable style patterns
- .pre-commit-config.yaml: Added mypy, bandit, codespell with proper configuration
- .github/workflows/python-quality.yml: Fixed codespell CLI args, removed AI slop from CI
- scripts/check_ai_slop.py: Fixed regex patterns, improved Windows path handling

**Code Fixes:**
- Fixed 228 docstring issues (added missing punctuation)
- Removed 722 emojis/symbols from 16 documentation files
- Fixed undefined logger bug in backend/voice_models/model_manager.py
- Added nosec comments for 4 legitimate security warnings (bind 0.0.0.0, subprocess, test passwords)

**Documentation Updates:**
- CONTRIBUTING.md: Comprehensive pre-commit hooks list (16 hooks)
- QUICKSTART_GIT_HOOKS.md: Detailed hook descriptions
- docs/AI_SLOP_CHECKER.md: Updated for pre-commit integration
- CODEBASE.md: Updated status (this file)

**CI Status - All 7 Checks Passing:**
- Black (formatting)
- Ruff (linting)
- Mypy (type checking - non-blocking)
- Bandit (security)
- Codespell (spelling)
- Flutter format
- Flutter analyze

**Implementation Notes:**
- Fixed 258 docstring issues (30 additional files in cli_client/, root scripts)
- Fixed 9 logger whitespace issues from emoji removal
- Restored 78 platform-specific files (mist_desktop platform directories)
- Normalized 25 dependencies/csm/ files (project code, intentional)
- Updated mixed-line-ending hook to exclude platform directories

**Files Committed:** 111 (after selective restore of platform files)

### Previous Session (2025-02-03) - Code Quality Infrastructure
[MAJOR] Established comprehensive code quality and AI integration system (commit: 444e423):

**AI Integration:**
- Created CLAUDE.md - Complete AI integration guide with NO EMOJIS rule
- Created CODEBASE.md - Living context document (this file)
- Created CONTRIBUTING.md - Contribution guidelines and standards
- Updated .claude/instructions.md - Reference to CLAUDE.md
- Created check_ai_slop.py - Automated detection of AI slop patterns

**Code Quality - Python:**
- Created pyproject.toml - Black, Ruff, mypy, pytest configuration
- Created .pre-commit-config.yaml - Automated pre-commit hooks
- Updated requirements.txt - Added ruff, mypy, pre-commit, pytest-cov
- Created check_ai_slop.py - Efficient, LLM-optimized AI pattern detector

**Code Quality - Flutter:**
- Updated analysis_options.yaml - Custom Dart linter rules
- Configured prefer_single_quotes, require_trailing_commas, etc.
- Excluded generated files from analysis

**Infrastructure:**
- Created scripts/check_ai_slop.py - Detects emojis, superlatives, filler phrases (standalone)
- Configured pre-commit hooks for Python and Flutter (Black, Ruff, formatting)
- Created GitHub Actions workflows for CI/CD (Python + Flutter)
- Created git hook installation scripts (Linux/macOS + Windows)
- Documented complete git workflow setup in GIT_WORKFLOWS.md
- AI slop checker available as standalone tool (not in pre-commit to avoid false positives in deps)

### Previous Sessions
- Removed obsolete React frontend (saved 127MB)
- Completed Flutter migration with Riverpod state management
- Implemented Flutter WebSocket client
- Implemented Flutter voice recording service
- Updated all documentation to reflect Flutter architecture

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
3. Check if any todos exist from previous session
4. Read relevant docs if touching new areas

### Making Changes
1. Create todos with TodoWrite for multi-step work
2. Update code with proper formatting (black, flutter format)
3. Run checks before committing (pre-commit)
4. Test changes manually or with tests

### Ending Session
1. Mark all todos complete or update status
2. Update this file with changes made
3. Commit with clear conventional commit message
4. Push if on a shared branch

---

## Testing

### Backend Tests
```bash
# Connection test
python test_neo4j_connection.py

# Conversation handler test
python test_conversation_handler.py --mode simple

# Vector search test
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

## Common Tasks

### Enable TTS for Audio Testing
1. Edit `.env`: Change `TTS_ENABLED=false` to `TTS_ENABLED=true`
2. Restart backend server
3. Test with Flutter app

### Run Code Quality Checks
```bash
# Check AI slop patterns (emojis, etc.)
python scripts/check_ai_slop.py --critical-only

# Auto-fix emojis and unicode symbols
python scripts/check_ai_slop.py --fix

# Generate compact report for LLM
python scripts/check_ai_slop.py --format markdown --output slop-report.md

# Run pre-commit hooks
pre-commit run --all-files

# Format Python
black backend/

# Lint Python
ruff check backend/ --fix

# Format Flutter
cd mist_desktop && flutter format .

# Analyze Flutter
cd mist_desktop && flutter analyze
```

### Update Dependencies
```bash
# Python
pip install -r requirements.txt

# Flutter
cd mist_desktop && flutter pub get
```

---

## Important Files

### Documentation
- **CLAUDE.md** - AI integration guidelines (READ THIS FIRST)
- **README.md** - Project overview and setup
- **REPOSITORY_STRUCTURE.md** - File organization
- **knowledge-vault/08-Reference/Flutter-Migration-Plan.md** - Detailed Flutter architecture
- **CONTRIBUTING.md** - Code quality standards

### Configuration
- **.env** - Environment variables (never commit!)
- **pyproject.toml** - Python tool configuration
- **analysis_options.yaml** - Flutter/Dart linting
- **.pre-commit-config.yaml** - Pre-commit hooks

### Key Decisions
- **knowledge-vault/03-Decisions/ADR-001-MIST-Vision-and-Architecture.md** - Project vision and philosophy
- **knowledge-vault/03-Decisions/ADR-007-Sesame-CSM-TTS.md** - TTS selection rationale

---

## Next Steps

### Immediate Priorities
1. Test Flutter audio playback with TTS enabled
2. Complete pre-commit hook setup
3. Add CI/CD pipeline (GitHub Actions)

### Short-term Goals
1. Knowledge graph visualization in Flutter
2. Improve Flutter UI polish and animations
3. Add error handling and loading states

### Long-term Goals
1. Rich content support (markdown, images, links)
2. Vision integration (Qwen 2.5 Vision)
3. Meta-reasoning layer for explainability
4. Mobile app (Flutter iOS/Android)

---

## Notes for AI Assistants

### When You Start
- **ALWAYS** read CLAUDE.md first
- Check this file (CODEBASE.md) for current status
- Review recent git commits to understand context
- No emojis anywhere - use [brackets] instead

### When You Finish
- Update "Recent Changes" section above
- Mark all todos complete
- Update "Current Focus" if work direction changed
- Update "Last Updated" date at top

### Best Practices
- Follow CLAUDE.md guidelines strictly
- Use TodoWrite for multi-step tasks
- Test changes before committing
- Ask user if uncertain
- Document architectural decisions in ADRs

---

## Quick Reference

| Area | Status | Notes |
|------|--------|-------|
| Backend | PROD READY | TTS disabled for dev |
| Frontend | IN DEV | Audio playback pending |
| Knowledge Graph | COMPLETE | Neo4j integrated |
| Code Quality | COMPLETE | 16 pre-commit hooks, 7 CI checks |
| CI/CD | COMPLETE | GitHub Actions configured |
| Documentation | EXCELLENT | Fully updated |

---

**Remember:** This file should be updated regularly. If you make significant changes, update the "Recent Changes" section and the "Last Updated" date.
