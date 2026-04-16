# Claude AI Integration Guide for MIST.AI

## CRITICAL RULES - READ FIRST

### 1. NO EMOJIS OR EMOTICONS - EVER

**ABSOLUTE RULE:** Never use emojis, emoticons, or unicode decorative symbols in ANY output.

This includes:
- Code comments
- Documentation files (README, guides, markdown files)
- Commit messages
- File content
- Communication with the user
- Todo items
- Log messages

**Instead use plain text indicators:**
- `[COMPLETE]` or `[DONE]` instead of checkmarks
- `[WARNING]` or `[ALERT]` instead of warning symbols
- `[SUCCESS]` or `[OK]` instead of check marks
- `[ERROR]` or `[FAIL]` instead of X marks
- `[IN PROGRESS]` or `[WORKING]` instead of hourglass
- `->` instead of arrow symbols
- `<->` instead of bidirectional arrows
- `---` or `===` for dividers instead of decorative lines

### 2. Project Identity

**Official Name:** MIST.AI or M.I.S.T (not "Mist.AI", "mist", or "MIST")

**Project Type:** Cognitive architecture research platform
- NOT a ChatGPT replacement
- NOT a simple productivity assistant
- IS a transparent, locally-run AI system with persistent memory

### 3. Communication Style

**Professional & Technical:**
- Objective technical accuracy over emotional validation
- No superlatives (amazing, awesome, fantastic, incredible)
- No filler phrases ("let's dive in", "exciting", "powerful")
- Facts and problem-solving focus
- Disagree when necessary - technical correctness matters

**Direct Communication:**
- Start with the answer, then explain if needed
- No lengthy preambles
- Use clear, concise language
- Monospace-friendly formatting (this is a CLI tool)

---

## Engineering Philosophy

**Always design for the ideal solution.** Never let implementation complexity, cost, time, or difficulty influence the architectural recommendation. Lead with the production-ready, fully optimized architecture. We have agentic teams and the resources to build things right -- recommending a "simpler" approach when a better one exists wastes time building something that needs replacement.

---

## Project Context

### Current Architecture (Updated: 2026-04-08)

**Backend (Python):**
- Python 3.11+
- FastAPI + Uvicorn (WebSocket server)
- Docker Compose (backend + Neo4j 5 + llama-server)
- llama-server (LLM inference - Qwen 2.5 7B via llama.cpp)
- Whisper (STT - base model)
- Chatterbox Turbo (TTS - MIT license, zero-shot voice cloning)
- Neo4j 5.x (knowledge graph)
- PyTorch 2.6.0 + CUDA 12.4 (Linux container)
- Voice pipeline: VAD -> STT (Whisper) -> LLM (Qwen 2.5) -> TTS (Chatterbox Turbo) with streaming parallelism (~4-5s TTFA)
- Log streaming: WebSocketLogHandler with request ID propagation, persistent file logging
- Status: PRODUCTION READY
- Deployment: Docker Compose (nvidia/cuda:12.4.0-devel-ubuntu22.04)

**Frontend (Flutter):**
- Cross-platform desktop (Windows/macOS/Linux)
- WebSocket client connecting to backend
- Voice recording and audio playback
- Navigation rail (4 destinations: Chat, Logs, Voice Profiles stub, Settings stub)
- Log viewer with level filtering, search, grouping, ring buffer
- Riverpod state management
- Status: IN DEVELOPMENT

**Key Technologies:**
- Python 3.11+, FastAPI, llama-server, openai (Python client), Neo4j, PyTorch
- Docker Compose (backend + Neo4j + llama-server), PyTorch 2.6+cu124
- Flutter 3.24+, Dart 3.10+, Riverpod 3.x
- Chatterbox Turbo TTS (MIT license, zero-shot voice cloning)

### Current Branch Status

Branch: `main`
- Backend containerized (Docker Compose: backend + Neo4j + llama-server)
- Chatterbox Turbo TTS (0.74x RTF, 3.9GB VRAM), default voice profile: friday
- Voice pipeline parallelism: LLM token streaming -> sentence detection -> TTS consumer (~4-5s TTFA)
- Log streaming via WebSocket + persistent file logging (./logs/mist-backend.log)
- Flutter nav rail + log viewer operational
- 655+ backend tests, 14 sentence detector tests, token streaming tests
- Flash attention enabled via Linux container (PyTorch 2.6+cu124)
- Native Windows venv corrupted -- use container for all backend work
- 43 commits ahead of origin/main (not pushed)

### Recent Major Changes

1. Voice pipeline optimization: streaming parallelism, 12-19s TTFA down to ~4-5s
2. Backend log streaming with WebSocketLogHandler, request ID propagation
3. Flutter navigation rail + log viewer (filter, search, grouping)
4. Chatterbox TTS integration (replaced Sesame CSM-1B)
5. Docker Compose containerization (backend + Neo4j + llama-server)
6. Knowledge extraction + curation pipeline (655 tests)

---

## Context Management Strategy

### Essential Context Files (Always Read First)

When starting any work session, read these files in order:

1. **CODEBASE.md** - Current status, active work, recent changes
2. **REPOSITORY_STRUCTURE.md** - Project organization and file structure
3. **.env** - Configuration (never commit or expose secrets)
4. **Git status** - Check uncommitted changes and current branch
5. **Recent commits** - Last 3-5 commits to understand recent work

### When Deep Context Needed

For architectural decisions or understanding design rationale:
- `docs/decisions/adr_*.md` - Architecture Decision Records
- `docs/design/*.md` - Design documents and architecture

For specific areas:
- `FLUTTER_MIGRATION_PLAN.md` - Flutter implementation details
- `INTEGRATION_STATUS.md` - Knowledge graph integration status
- `E2E_TEST_GUIDE.md` - Testing procedures

### Documentation Update Requirements

Update these files when making significant changes:

**Always:**
- Update CODEBASE.md after completing features or major changes
- Update git commit messages with clear descriptions

**When Applicable:**
- Update REPOSITORY_STRUCTURE.md when adding new directories/major files
- Create ADR in docs/decisions/ for architectural decisions
- Update relevant guide files when changing workflows

---

## Code Style Guidelines

### Python Code Style

**Formatting:**
- PEP 8 compliant
- Line length: 100 characters (configured in pyproject.toml)
- Use Black formatter (no manual formatting decisions)
- Import order: stdlib, third-party, local (handled by isort/ruff)
- Within a package, use relative imports for intra-package references.
- Use absolute imports for cross-package references.

**Type Hints:**
```python
def function_name(param: str, optional: int = 0) -> ReturnType:
    """Docstring here."""
    pass
```

Use PEP 585/604 syntax (Python 3.11+):
- `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- `str | None` not `Optional[str]`, `int | str` not `Union[int, str]`
- Only import from `typing`: TypeVar, Protocol, Literal, TypedDict

**Docstrings:**
```python
"""
Brief one-line summary.

Longer description if needed. Explain the "why" not the "what".

Args:
    param: Description of parameter
    optional: Description with default behavior

Returns:
    Description of return value

Raises:
    ValueError: When validation fails
"""
```

Use single backticks for inline code references in docstrings (not double backticks).

**File Headers:**
```python
"""
Module-level docstring explaining purpose.

Key classes/functions overview if module is complex.
"""
import statements...
```

### Flutter/Dart Code Style

**Follow Effective Dart:**
- Use `prefer_single_quotes: true`
- Prefer `const` constructors where possible
- Avoid `print()` - use proper logging
- Use `lowerCamelCase` for variables/methods
- Use `UpperCamelCase` for classes/types

**Widget Structure:**
```dart
class WidgetName extends ConsumerWidget {
  const WidgetName({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Read providers at top
    final state = ref.watch(stateProvider);

    // Build UI
    return Widget();
  }
}
```

## Flutter Frontend Guidelines

### Widget Architecture

Prioritize modularization, reusability, and testability when building widgets.

- **Widget modularization**
  - Monolithic single-widget screens are an anti-pattern. Break screens into
    smaller widgets with clear ownership boundaries.
  - Structure widgets so they can be rendered and tested in isolation -- avoid
    deep coupling to global providers or parent context.
  - Keep widget files under ~200 lines. If a widget grows beyond that, extract
    sub-widgets or helper methods.

- **Reusability**
  - Build smaller, composable widgets that can be reused across screens.
  - Prefer passing data and callbacks via constructor parameters over reading
    providers directly -- this makes widgets testable without provider overrides.
  - Extract shared UI patterns (message bubbles, status indicators) into
    dedicated widget files under `widgets/`.

- **Readability and logic isolation**
  - If a widget has complex state logic, extract it into a dedicated Notifier
    or service class rather than inlining it in the widget.
  - Keep `build()` methods focused on rendering. Move side-effect setup
    (listeners, subscriptions) into `initState()` or provider callbacks.

### Riverpod State Management

- **Provider types and when to use them:**
  - `Provider` -- immutable services (WebSocketService, AudioRecordingService)
  - `NotifierProvider` -- mutable state with methods (ChatNotifier)
  - `StreamProvider` -- reactive streams (connectionStatus, isRecording)
  - `FutureProvider` -- one-shot async data

- **Provider DI pattern:** Services are created in `Provider` definitions and
  injected into Notifiers via `ref.read()` in `build()`. This mirrors the
  backend's factory pattern in `backend/factories.py`.

- **Do not over-centralize state.** ChatNotifier currently owns messages,
  processing state, and AI response streaming. As features grow, split into
  focused notifiers (e.g., separate voice state from chat state).

- **Avoid `ref.watch()` in callbacks.** Use `ref.read()` inside event handlers
  and `ref.watch()` only in `build()` methods or provider definitions.

### Service Layer

Services (`lib/services/`) encapsulate platform/external interactions:
- `WebSocketService` -- backend communication (connect, send, receive)
- `AudioRecordingService` -- mic capture and PCM buffer management
- `AudioPlaybackService` -- TTS audio queue and WAV playback
- `SpeechService` -- platform-native STT (legacy, being replaced by backend Whisper)

**Rules:**
- Services must be stateless singletons managed by Riverpod `Provider`.
- Services expose `Stream`s for reactive state, not callbacks.
- Services handle their own `dispose()` via `ref.onDispose()` in the provider.
- No direct `print()` -- use `Logger` from the `logger` package.

### Testability

- New widgets and logic should include corresponding tests.
  See `mist_desktop/test/CLAUDE.md` for conventions and infrastructure.
- Prefer accepting data and callbacks via constructor params over reaching
  directly into providers -- makes widgets testable without full provider trees.
- Extract complex logic (message parsing, audio format conversion, WebSocket
  protocol handling) into pure functions or service methods testable without
  widget rendering.

---

## Git Commit Guidelines

### Commit Message Format

```
type(scope): Brief description (max 72 chars)

Longer description if needed, explaining:
- What changed and why
- Any breaking changes
- Related issues or PRs

[FOOTER with attribution]
Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no functional changes)
- `refactor`: Code refactoring (no feat/fix)
- `test`: Adding/updating tests
- `chore`: Maintenance tasks, dependency updates

### Rules

- NO EMOJIS in commit messages
- Use present tense ("add feature" not "added feature")
- Capitalize first letter of description
- No period at end of subject line
- Blank line between subject and body

---

## Licensing & Attribution

### Main Project
- License: MIT License
- Copyright: Project maintainer
- Attribution: Not required but appreciated

### Third-Party Code

**Sesame CSM TTS (legacy, replaced by Chatterbox Turbo):**
- License: Apache 2.0
- Location: `dependencies/csm/`
- Status: No longer active. Chatterbox Turbo is the current TTS engine.
- Code retained for rollback only
- Must preserve Apache 2.0 license headers in modified files

**Chatterbox Turbo TTS:**
- License: MIT
- Installed via pip (chatterbox-tts package)
- Zero-shot voice cloning from reference WAV
- Adapter: `src/multimodal/tts.py` (ChatterboxTTS class)

**When Using External Code:**
1. Check license compatibility (MIT-compatible licenses preferred)
2. Document in appropriate LICENSE/NOTICE files
3. Preserve original copyright notices
4. Document modifications if you change the code

---

## Key Constraints & Decisions

### Technical Constraints

**Hardware (actual, queried 2026-03-23):**
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB VRAM
- CPU: AMD Ryzen 7 7800X3D 8-Core (16 threads) @ 4.2 GHz
- RAM: ~32 GB
- CUDA 13.1, Driver 591.74
- Backend runs in Docker container (nvidia/cuda:12.4.0-devel-ubuntu22.04)
- Can run 14B quantized models (Q4_K_M) with voice off; 7-12B with voice on
- Windows 11 primary development platform
- Must be air-gapped capable (local-first design)

**Model Choices:**
- LLM: Qwen 2.5 7B Instruct (via llama-server)
- Embeddings: all-MiniLM-L6-v2 (384-dim, fast)
- STT: Whisper base (1.4GB model)
- TTS: Chatterbox Turbo (0.74x RTF, 3.9GB VRAM, zero-shot cloning)

### Design Philosophy

**Transparency:** Every decision the AI makes should be visible
- Show tool calls explicitly
- Log entity extractions
- Visualize knowledge graph retrievals
- No hidden "magic"

**Local-First:** Core functionality works without internet
- llama-server for LLM
- Local Neo4j database
- Offline-capable knowledge system
- Cloud delegation only for strategic decisions

**Privacy:** User controls all data
- No telemetry without explicit consent
- Local storage only
- Air-gapped operation possible
- Export/delete capabilities

---

## Working with AI Code Assistants

### For Claude Code (You!)

**Session Management:**
- Always read CODEBASE.md at start of session
- Use TodoWrite tool to track multi-step tasks
- Mark todos complete immediately after finishing
- Update CODEBASE.md before ending session

**Plan Mode:**
- Use ExitPlanMode for multi-step implementation tasks
- Present clear plan before executing
- Ask clarifying questions if requirements unclear
- One task in_progress at a time in TodoWrite

**Code Changes:**
- Read files before editing them
- Use Edit tool for existing files (not Write)
- Never use emojis (yes, this is repeated intentionally)
- Test changes if possible
- Document breaking changes

### For Other AI Tools

**Cursor/Copilot:**
- Follow same style guidelines
- Read CLAUDE.md before generating code
- No emojis in generated code or comments
- Use check_ai_slop.py to validate output

**ChatGPT/Claude Web:**
- Good for architectural discussions
- Document decisions in ADRs (docs/decisions/)
- Don't copy-paste code without review
- Remove AI filler phrases before committing

## Dependency Injection

### Enforceable rule: No hidden construction in __init__

All classes that depend on external systems (Neo4j, LLM backend, embeddings,
event store) MUST accept dependencies as required constructor parameters.
Factory functions in `backend/factories.py` handle real wiring.

```python
# CORRECT: required params
class GraphStore:
    def __init__(self, connection: GraphConnection, embedding_generator: EmbeddingProvider):
        self.connection = connection
        self.embedding_generator = embedding_generator

# WRONG: hidden construction
class GraphStore:
    def __init__(self, config):
        self.connection = Neo4jConnection(config.neo4j)
```

For tests, bypass factories and pass fakes directly to constructors.

## Error Handling

### Enforceable rule: Use MistError hierarchy

All I/O error handling MUST use specific exception types from
`backend/errors.py`. Never catch bare `Exception` in new code.

```python
# CORRECT
from backend.errors import Neo4jQueryError
try:
    results = connection.execute_query(query)
except Neo4jQueryError as e:
    logger.error("Query failed: %s", e)

# WRONG
try:
    results = connection.execute_query(query)
except Exception as e:
    logger.error(str(e))
```

Available exceptions: Neo4jConnectionError, Neo4jQueryError,
LLMConnectionError, LLMResponseError, ExtractionError,
ExtractionValidationError, NormalizationError, EmbeddingError.

## Async Boundaries

### Enforceable rule: Never call sync Neo4j from async contexts

Use `GraphExecutor` for all async graph operations. GraphStore methods
remain sync. GraphExecutor wraps them for async callers.

```python
# CORRECT: async code uses GraphExecutor
results = await executor.execute_query("MATCH (n) RETURN n")

# WRONG: async code calls sync GraphStore directly
results = graph_store.connection.execute_query("MATCH (n) RETURN n")
```

## Resource Lifetime

Structure operations in phases:
1. Acquire resource -> read/write -> release (short-lived)
2. Do CPU/IO/inference work with no held resources
3. Acquire resource -> write results -> release

Applies to: Neo4j transactions, LLM client calls, GPU tensor
allocations. Never hold a Neo4j transaction open during LLM inference.

## HTTP Response Handling

### Enforceable rule: Check HTTP responses

All HTTP requests (to LLM backend, external services) must either call
`response.raise_for_status()` or explicitly check the status code.
Never silently consume error responses.

## Dataclass vs Pydantic

- `@dataclass(frozen=True)` for ontology and domain objects (immutable)
- `@dataclass(frozen=True, slots=True)` for new internal data structures
- Pydantic `BaseModel` only for WebSocket message schemas or API validation
- Never use raw dicts where a dataclass provides type safety

## Subdirectory Guides

- `tests/CLAUDE.md` -- Backend test conventions and AI guidance
- `mist_desktop/test/CLAUDE.md` -- Flutter frontend test conventions

## Testing

See `TESTING.md` for conventions and `tests/CLAUDE.md` for AI-specific
test guidance. Run tests: `pytest tests/unit/`

---

## Anti-Patterns to Avoid

### "AI Slop" Patterns

These indicate low-quality AI output and must be removed:

**Emojis & Symbols:**
- Any emoji or unicode decorative character
- Checkmarks, X marks, arrows, etc.

**Superlative Language:**
- "Amazing", "incredible", "powerful", "fantastic"
- "Robust", "seamless", "cutting-edge"
- "Revolutionary", "game-changing", "world-class"

**Filler Phrases:**
- "Let's dive in/into"
- "First and foremost"
- "It's worth noting that"
- "At the end of the day"
- "Moving forward"

**Over-Enthusiasm:**
- Excessive exclamation marks
- Overly positive tone inappropriate for technical docs
- Marketing-style language in code comments

**Use check_ai_slop.py to detect these patterns:**
```bash
python scripts/check_ai_slop.py --critical-only  # Fast check
python scripts/check_ai_slop.py --fix            # Auto-fix
```

Full documentation: [docs/AI_SLOP_CHECKER.md](docs/AI_SLOP_CHECKER.md)

---

## Common Tasks & How To Approach Them

### Adding a New Feature

1. Check CODEBASE.md for current status
2. Read relevant ADRs if touching architecture
3. Create TodoWrite plan for multi-step features
4. Implement with proper type hints/documentation
5. Test manually or write tests
6. Update CODEBASE.md with changes
7. Commit with clear message (no emojis!)

### Debugging Issues

1. Check recent git commits for related changes
2. Read relevant code sections completely
3. Check logs/error messages carefully
4. Reproduce issue if possible
5. Fix with minimal changes
6. Document why the bug occurred if not obvious

### Refactoring Code

1. Understand current behavior first
2. Write tests if not present
3. Make incremental changes
4. Test after each change
5. Update documentation if behavior changes
6. Commit frequently with clear messages

### Writing Documentation

1. No emojis or decorative symbols
2. Use clear headings and structure
3. Code examples where helpful
4. Explain "why" not just "what"
5. Keep it up-to-date with code
6. Link to related docs

---

## Error Handling & Edge Cases

### When Things Go Wrong

**Don't:**
- Panic and make hasty changes
- Mark tasks complete if they failed
- Hide errors in logs
- Add workarounds without documentation

**Do:**
- Report errors clearly to user
- Document unexpected behavior
- Add error handling for edge cases
- Update todos to reflect blockers
- Ask user for guidance if stuck

### Uncertainty Handling

**If you're not sure:**
- Use AskUserQuestion tool
- Check existing code for patterns
- Read relevant ADRs or docs
- Propose approach and ask for confirmation
- Don't guess and hope it works

---

## Tool Usage Guidelines

### TodoWrite Best Practices

- Create todos for any multi-step task (3+ steps)
- Use clear, actionable todo descriptions
- Provide both `content` and `activeForm`
- Mark complete IMMEDIATELY after finishing
- Only one todo `in_progress` at a time
- Remove/update todos if plans change

### File Operations

- Always Read before Edit
- Use Edit for existing files (never Write)
- Use Write only for new files
- Glob/Grep for finding files
- Bash only for git, build tools, not file operations

### Agentic Teams (Preferred for Max Effort)

When on max plan/effort, **always prefer dispatching parallel agent teams**
over sequential solo work. This is the primary execution mode for non-trivial
implementation tasks.

**When to use agentic teams:**
- 2+ independent tasks with no shared state or sequential dependencies
- Test writing for multiple components (each test file = independent agent)
- Implementing features across different modules (backend vs frontend)
- Audit, review, or exploration tasks covering different subsystems

**How to dispatch:**
- Use `Agent` tool with multiple concurrent invocations in a single message
- Give each agent a complete, self-contained prompt (agents share no context)
- Use `run_in_background: true` for genuinely independent work
- Name agents for `SendMessage` follow-up if needed

**Role framing (required):**
Every agent prompt MUST open with an expert role definition. Role framing
changes how the agent reasons about quality, trade-offs, and edge cases.

Format: `**Role:** You are a [seniority] [domain] [title] with deep expertise
in [specific technologies/patterns]. You have [relevant experience].`

Examples:
- Implementation: "You are a senior Flutter/Dart architect with deep expertise
  in Riverpod state management, Material 3 design, and desktop app UX."
- Implementation: "You are a senior Python backend engineer with expertise in
  asyncio, threading, and WebSocket server architecture."
- Review: "You are a principal engineer reviewing code for thread safety,
  performance, and production readiness."
- Research: "You are a systems researcher with expertise in distributed
  architectures and protocol design."

Match the role to the task domain. Be specific about technologies -- "Flutter
expert" is weaker than "Flutter desktop architect with Riverpod and Material 3
expertise."

**Rules:**
- Each agent gets a clear scope -- no overlapping file edits
- Agent prompts must include all necessary context (file paths, interfaces, conventions)
- Prefer foreground when results inform next steps; background for independent work
- Review agent output before committing -- agents are trusted but verified

**Anti-patterns:**
- Don't dispatch agents for trivial single-file edits
- Don't have multiple agents edit the same file (merge conflicts)
- Don't use agents when tasks have sequential dependencies

### Task Agent (Explore/Research)

- Use for complex searches requiring multiple rounds
- Use Explore agent for codebase questions
- Specify thoroughness level (quick, medium, very thorough)
- Don't use for simple file path reads

---

## Project-Specific Notes

### Modified Third-Party Code

**Sesame CSM TTS (legacy, replaced by Chatterbox Turbo):**
- License: Apache 2.0
- Location: `dependencies/csm/`
- Status: No longer active. Chatterbox Turbo is the current TTS engine.
- Code retained for rollback only
- Must preserve Apache 2.0 license headers in modified files

**Chatterbox Turbo TTS:**
- License: MIT
- Installed via pip (chatterbox-tts package)
- Zero-shot voice cloning from reference WAV
- Adapter: `src/multimodal/tts.py` (ChatterboxTTS class)

### Empty Directories

Some directories may be empty - DO NOT REMOVE THEM.
They're placeholders for planned features:
- Future test directories
- Planned component directories
- Architecture scaffolding

### .env Security

- Never commit .env file
- Never expose secrets in code/logs
- Use environment variables for all secrets
- Document required env vars in .env.example

---

## Summary - Quick Reference

**Three Golden Rules:**
1. NO EMOJIS EVER (yes, third time stating this)
2. Read CODEBASE.md at session start
3. Update CODEBASE.md at session end

**Code Quality:**
- Format: black (Python), flutter format (Dart)
- Lint: ruff (Python), flutter analyze (Dart)
- Type hints required (Python)
- Docstrings for public APIs

**Communication:**
- Professional, technical, objective
- No AI slop patterns
- Clear and concise
- Run check_ai_slop.py before committing

**Git:**
- Conventional commits
- No emojis in messages
- Clear descriptions
- Test before committing

---

For questions or clarifications, ask the user directly. When in doubt, check existing code for patterns and conventions.

Last Updated: 2026-04-08
