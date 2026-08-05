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

## CODEBASE.md Maintenance Protocol

`CODEBASE.md` is the authoritative in-repo snapshot of current project state -- branch, HEAD, active phase, per-subsystem status, test count, next actions. It is the first thing a fresh session reads (via `/mist-status`), so a stale CODEBASE.md silently misleads every future session. This is not hypothetical: on 2026-07-29 it still named a long-merged feature branch and a HEAD five phases behind reality.

Keep it current as a side effect of routine work -- do NOT treat updating it as a separate task to be deferred:

- **On any status / scan / context-loading pass** (`/mist-status`, session start, "where are we"): if the header (Last Updated / Branch / Status), Current Focus, test count, or any subsystem bullet diverges from `git` or the vault workstream note, reconcile it in the same turn before reporting status.
- **On landing a milestone / merging to `main` / changing the active branch or HEAD:** update the header block and Current Focus before the work is considered done.
- **On adding, removing, or materially changing a subsystem:** update its bullet under Current Status.

Rules:
- **Ground every claim in real state.** Read `git -C "D:\Users\rajga\mist.ai" status` / `log` and the vault workstream note; never copy a hash, count, or version forward without verifying it against the source. If a number cannot be verified (e.g. an ontology rel-type count), flag it rather than guessing.
- **Preserve history.** Demote the prior header entry to a nested `PRIOR ENTRY --` rather than deleting it (the running-history style this file already uses).
- **Docs are TRACKED and PUSHED.** CODEBASE.md and CLAUDE.md live in git and go to origin like any other file. What stays local is the `docs/` tree -- specs, plans, and findings registers -- which is gitignored (`.gitignore:69`), along with `tests/CLAUDE.md` (`.gitignore:73` matches `CLAUDE.md` at any depth, but the two root files predate the rule and remain tracked). **This line previously read "Docs are local-only ... never pushed to origin" while both files were already tracked AND already present on `origin/main`** -- a documented convention contradicted by observable state, which is precisely the defect class the 2026-08-03 reachability work existed to remove. Corrected 2026-08-04 rather than left standing.
- **Schema / convention / structure changes** belong in this CLAUDE.md, not only in CODEBASE.md.

---

## Plan Verification Protocol

Implementation plans and design specs assert things about the codebase -- "X has no caller," "this would create a cycle," "that branch varies per turn." **Those assertions get executed by people and agents who will not re-derive them.** A false one propagates into code, tests that certify it, and a docstring that documents it.

This is not hypothetical. On 2026-08-04, R1.4.6 T0 shipped with **five falsified rationales** on one branch: four in the plan, and a fifth inside the fix for the other four. In every case the CONCLUSION was correct and the STATED REASON was invented rather than checked. Each fell to a few seconds of `grep`. Two of them (`audio_queue` having no writer, `process_audio_chunk` having no caller) were **already documented in `KNOWN_ISSUES.md:114-118`** -- the information was in the repo and nobody looked.

The diagnosis is specific and worth stating, because it makes the rule actionable: **every falsified claim was about code the author had not opened.** The claims about code that HAD been read -- the three deliberate non-resets in `reset_connection_state` -- were correct and survived two independent reviews. Same author, same session, same confidence in the prose. The only variable was whether the file had been read. Uniform confidence over non-uniform verification is what makes the bad claims indistinguishable from the good ones.

### Rules

- **Every causal claim about the codebase carries the command that establishes it.** Not "no cycle exists" but ``no cycle: `grep '^from\|^import' backend/chat/stream_events.py` -> dataclasses, typing only``. If you cannot produce the command, you have not checked -- and you find that out while writing, not two hours later in review.
- **Before mandating work on any attribute or method, grep `KNOWN_ISSUES.md` for it.** It is a backlog of known-dead and known-broken things. It is cheap to search and it is routinely not searched.
- **Run a claim-check pass before dispatching a plan.** One agent, one job: extract every factual assertion about the codebase from the plan and verify each against source. This is mechanical and cheap. It is the same work a whole-branch reviewer does -- but before any implementer builds on a false premise, rather than after commits, task reviews, a fix wave, and follow-up corrections.
- **State unverified claims as unverified.** If a conclusion is right but you have not established why, say so ("do not reset this -- reason not verified") rather than supplying a plausible mechanism. A missing reason is a prompt to check; a wrong reason is a trap.
- **Never collapse two adjacent facts into one label.** This shape produced two of the day's falsified claims on 2026-08-04, hours apart. First: a docstring said `audio_queue` has "no writer -- its only feeder, `process_audio_chunk`, has no caller," merging *`audio_queue` has no writer at all* with *`process_audio_chunk` feeds `vad_processor` and has no caller* -- two true facts, one false sentence. Second: a plan wrote "the V7/V9 negative-control holes" because one auditor covered both files, when **V9 fails closed and does not have the hole**. The trigger is a shared container -- one sentence, one slash-joined label, one agent's scope -- inheriting a property that belongs to only one member. **When two things are named together, verify the claim separately for each**, and if it holds for only one, name only that one.

### Why the fix belongs upstream, not in more review

The reviews on that branch worked -- every finding was caught. **Scoped per-task review structurally CANNOT catch this class**, because the deadness lives outside the diff: you cannot see that `audio_queue` has no producer by reading a diff that adds a consumer to it. Only the whole-branch gate could, and it did. So:

- **Never waive the whole-branch review gate.** 2026-08-03 recorded "no independent review verdict exists for this branch" as not-done rather than waived; 2026-08-04 is the concrete case for why that bar is right.
- **Do not answer this failure mode with more downstream review layers.** That makes the late catch more expensive without moving it earlier.

---

## Project Context

### Current Architecture (Updated: 2026-05-11)

**Backend (Python):**
- Python 3.11+
- FastAPI + Uvicorn (WebSocket server, port 8001)
- Docker Compose (backend + Neo4j 5 + llama-server)
- llama-server (LLM inference - Gemma 4 E4B Q5_K_M via llama.cpp)
- Whisper (STT)
- Chatterbox Turbo (TTS - MIT license, zero-shot voice cloning)
- Neo4j 5.x (knowledge graph)
- PyTorch 2.6.0 + CUDA 12.4 (Linux container)
- Voice pipeline: VAD -> STT (Whisper) -> LLM (Gemma 4 E4B) -> TTS (Chatterbox Turbo) with streaming parallelism (~4-5s TTFA)
- Log streaming: WebSocketLogHandler with request ID propagation, persistent file logging
- Vault layer (ADR-010): mist-memory/ markdown corpus + sqlite-vec sidecar index + watchdog filewatcher
- Status: PRODUCTION READY (continuous-usage hardening + FE/BE integration in progress)
- Deployment: Docker Compose (nvidia/cuda:12.4.0-devel-ubuntu22.04)

**Frontend (separate git repository nested at `./mist-frontend/` (own .git, no remote)):**
- Tauri 2.x cross-platform desktop shell
- React 19 + TypeScript strict
- three.js + @react-three/fiber + drei (spatial 3D composition)
- Vite build
- Native WebSocket via Tauri shell, connecting to backend at `ws://localhost:8001/ws`
- Integration contracts: ADR-016 (LLM-mediated FE tool calls; BE-decided routing) + ADR-017 (WebSocket message contract). Both live in `knowledge-vault/Decisions/`.
- Status: production-ready as of 2026-05-08; FE/BE Wave 1 shipped 2026-05-10 on branch `integration/v1`.
- The Flutter Desktop frontend at `mist_desktop/` was decommissioned 2026-05-11. Git history at commit `e18c092` preserves the Flutter source if reference is needed.

**Key Technologies:**
- Python 3.11+, FastAPI, llama-server, openai (Python client), Neo4j, PyTorch
- Docker Compose (backend + Neo4j + llama-server), PyTorch 2.6+cu124
- Tauri 2.x + React 19 + react-three-fiber + TypeScript (frontend, separate repo)
- Chatterbox Turbo TTS (MIT license, zero-shot voice cloning)

### Current Branch Status

See `CODEBASE.md` for live branch status, active workstreams, recent commits, and outstanding issues.

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
- `docs/decisions/adr_*.md` - Repo-scoped ADRs
- `knowledge-vault/Decisions/ADR-*.md` - Cross-project + integration ADRs (memory architecture, vault layer, FE/BE protocol)
- `docs/superpowers/specs/` - Phase implementation specs

For specific areas:
- `CODEBASE.md` - Recent work, cluster status, current blockers
- `TESTING.md` - Test conventions
- `KNOWN_ISSUES.md` - P3 backlog

### Documentation Update Requirements

Update these files when making significant changes:

**Always:**
- Update CODEBASE.md after completing features or major changes
- Update git commit messages with clear descriptions

**When Applicable:**
- Update REPOSITORY_STRUCTURE.md when adding new directories/major files
- Create ADR (repo-scoped under `docs/decisions/`, cross-project under `knowledge-vault/Decisions/`)
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

### Frontend Code Style

Frontend code lives in a separate git repository nested at `./mist-frontend/` (own .git, no remote). See that repository's own contributing guide and CLAUDE.md (when refreshed post-spatial-app-reframe) for TypeScript / React / Tauri conventions. The two repos coordinate at the protocol layer (ADR-016 + ADR-017), not at the code-style layer.

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
- LLM: Gemma 4 E4B Q5_K_M (via llama-server)
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
- Document decisions in ADRs (docs/decisions/ or knowledge-vault/Decisions/)
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

Frontend test conventions live in the nested mist-frontend repo at `./mist-frontend/`.

## Testing

See `TESTING.md` for conventions and `tests/CLAUDE.md` for AI-specific
test guidance. Run tests inside the backend container:

```bash
docker compose exec mist-backend python -m pytest tests/unit/
```

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
- Implementing features across different modules (backend areas, or backend vs frontend coordination)
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
- Implementation: "You are a senior Python backend engineer with expertise in
  asyncio, threading, and WebSocket server architecture."
- Implementation: "You are a senior knowledge-graph engineer with deep expertise
  in Neo4j Cypher, ontology design, and entity extraction pipelines."
- Review: "You are a principal engineer reviewing code for thread safety,
  performance, and production readiness."
- Research: "You are a systems researcher with expertise in distributed
  architectures and protocol design."

Match the role to the task domain. Be specific about technologies -- "Python
expert" is weaker than "Python asyncio + WebSocket expert with Neo4j driver
experience."

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
- Format: black (Python)
- Lint: ruff (Python)
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

Last Updated: 2026-05-11 (Flutter Desktop decommissioned; mist-frontend/ Tauri repo canonical for FE)
