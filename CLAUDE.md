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

## Project Context

### Current Architecture (Updated: 2025-02-03)

**Backend (Python):**
- WebSocket server (FastAPI, port 8001)
- Voice pipeline: VAD -> STT (Whisper) -> LLM (Qwen 2.5) -> TTS (Sesame CSM-1B)
- Knowledge graph: Neo4j-backed persistent memory
- Autonomous tool usage (MCP-like pattern)
- Status: PRODUCTION READY

**Frontend (Flutter):**
- Cross-platform desktop (Windows/macOS/Linux)
- WebSocket client connecting to backend
- Voice recording and audio playback
- Riverpod state management
- Status: IN DEVELOPMENT

**Key Technologies:**
- Python 3.11+, FastAPI, Ollama, Neo4j, PyTorch
- Flutter 3.24+, Dart 3.10+, Riverpod 3.x
- Modified Sesame CSM TTS (Apache 2.0 licensed fork)

### Current Branch Status

Branch: `feat/frontend`
- React frontend removed (migrated to Flutter)
- Flutter WebSocket integration complete
- Flutter audio recording implemented
- Audio playback pending TTS enablement (TTS_ENABLED=false in .env)
- Knowledge graph visualization planned

### Recent Major Changes

1. Migrated from React/TypeScript to Flutter desktop
2. Removed obsolete React frontend (saved 127MB)
3. Updated all documentation to reflect Flutter
4. Flutter app structure established with Riverpod providers

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

**Type Hints:**
```python
def function_name(param: str, optional: int = 0) -> ReturnType:
    """Docstring here."""
    pass
```

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

**Sesame CSM TTS:**
- License: Apache 2.0
- Location: `dependencies/csm/`
- Modified: Yes - all modifications documented in `dependencies/csm/MODIFICATIONS.md`
- Must preserve Apache 2.0 license headers in modified files

**When Using External Code:**
1. Check license compatibility (MIT-compatible licenses preferred)
2. Document in appropriate LICENSE/NOTICE files
3. Preserve original copyright notices
4. Document modifications if you change the code

---

## Key Constraints & Decisions

### Technical Constraints

**Hardware Limitations:**
- Target: RTX 3070 8GB VRAM (mid-range consumer GPU)
- Cannot run models >13B parameters locally
- Windows 11 primary development platform
- Must be air-gapped capable (local-first design)

**Model Choices:**
- LLM: Qwen 2.5 7B Instruct (via Ollama)
- Embeddings: all-MiniLM-L6-v2 (384-dim, fast)
- STT: Whisper base (1.4GB model)
- TTS: Sesame CSM-1B (high quality, streaming)

### Design Philosophy

**Transparency:** Every decision the AI makes should be visible
- Show tool calls explicitly
- Log entity extractions
- Visualize knowledge graph retrievals
- No hidden "magic"

**Local-First:** Core functionality works without internet
- Ollama for LLM
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

### Task Agent

- Use for complex searches requiring multiple rounds
- Use Explore agent for codebase questions
- Specify thoroughness level (quick, medium, very thorough)
- Don't use for simple file path reads

---

## Project-Specific Notes

### Modified Third-Party Code

**Sesame CSM TTS (dependencies/csm/):**
- Forked from original Apache 2.0 project
- Modifications for streaming, interruption, context
- All changes documented in MODIFICATIONS.md
- Must maintain Apache 2.0 headers in modified files

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

Last Updated: 2025-02-03
