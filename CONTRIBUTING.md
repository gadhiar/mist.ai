# Contributing to MIST.AI

Thank you for your interest in contributing to MIST.AI. This document outlines the code quality standards and development workflow for the backend repository. Frontend contributions live in a separate repo (`./mist-frontend/`); see that repo's own contributing guidance.

---

## Before You Start

### Required Reading
1. **CLAUDE.md** - AI integration guidelines and project rules
2. **CODEBASE.md** - Current status and recent changes
3. **README.md** - Project overview and setup
4. **REPOSITORY_STRUCTURE.md** - Backend layout

### Setup Development Environment

**Backend (this repo):**

```bash
# Clone
git clone https://github.com/gadhiar/mist.ai.git
cd mist.ai

# Container-first: Docker Compose is the canonical dev environment
# (native Windows venv is corrupted; do all backend work in container)
cp .env.example .env
docker compose up -d
docker compose logs -f mist-backend

# Or use the dev script
python scripts/start_dev.py

# Install pre-commit hooks (required for contributions)
pip install pre-commit
pre-commit install
```

**Frontend (separate repo):**

The MIST frontend lives at `./mist-frontend/` (Tauri 2.x + React 19 + r3f). See that repository's own contributing guide for FE conventions.

---

## Code Quality Standards

### The Golden Rule: NO EMOJIS

**Never use emojis, emoticons, or unicode decorative symbols anywhere in the codebase.**

This includes:
- Code comments
- Documentation
- Commit messages
- Variable names
- Log messages

Use plain text alternatives: `[SUCCESS]`, `[WARNING]`, `[ERROR]`, `->`, etc.

### Python Code Style

**Formatting:**
- Line length: 100 characters
- Use Black formatter (no manual formatting decisions)
- Import order: stdlib, third-party, local (handled by isort/ruff)

**Automated Pre-Commit Checks:**
All checks run automatically when you commit (via pre-commit hooks):
- **Black** - Python code formatting
- **Ruff** - Python linting with auto-fix
- **Mypy** - Static type checking (non-blocking)
- **Bandit** - Security vulnerability scanning
- **Codespell** - Spell checking in code and docs
- **AI Slop Checker** - Detects emojis, unicode symbols (auto-fix)
- **File quality** - YAML/JSON validation, trailing whitespace, line endings

**Manual Commands (if needed):**

```bash
# Format code manually
black backend/

# Lint and auto-fix manually
ruff check backend/ --fix

# Type check
mypy backend/

# Security scan
bandit -r backend/ scripts/

# Spell check
codespell

# Check for AI slop
python scripts/check_ai_slop.py --incremental

# Run all pre-commit hooks manually
pre-commit run --all-files
```

**Type Hints:**

All functions should have type hints. Use PEP 585/604 syntax (Python 3.11+):

```python
def process_data(input_str: str, count: int = 0) -> list[str]:
    """Process the input string."""
    pass
```

Use `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`, `str | None` not `Optional[str]`.

**Docstrings:**

Use Google-style docstrings for public APIs:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief one-line description.

    Longer explanation if needed. Explain the "why" not just the "what".

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When input is invalid
    """
    pass
```

---

## Git Workflow

### Commit Message Format

Use conventional commits:

```
type(scope): Brief description (max 72 chars)

Longer explanation if needed:
- What changed
- Why it changed
- Breaking changes or notes

Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Commit Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance, dependencies

**Rules:**
- NO EMOJIS in commit messages
- Use present tense ("add feature" not "added feature")
- Capitalize first letter
- No period at end of subject line
- Blank line between subject and body

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit. All hooks must pass before commit succeeds.

**Python Checks:**
- **Black** - Code formatting (auto-fix)
- **Ruff** - Linting with auto-fix
- **Bandit** - Security scanning (with nosec comments)
- **Codespell** - Spell checking

**General Quality Checks:**
- **AI Slop Checker** - Emoji and unicode symbol detection (auto-fix, runs on changed files only)
- **Trailing whitespace** - Auto-removed
- **End-of-file fixer** - Ensures files end with newline
- **YAML/JSON/TOML validation** - Syntax checking
- **Large file detection** - Prevents commits >1MB
- **Merge conflict markers** - Detects unresolved conflicts
- **Private key detection** - Prevents accidental secret commits
- **Line ending normalization** - Converts to LF

If hooks fail, fix the issues and commit again. Do NOT skip hooks (`--no-verify`) unless explicitly authorized.

**Run manually:**

```bash
# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files
pre-commit run check-ai-slop --all-files
```

---

## Testing

### Python Tests (run inside container)

```bash
# All unit tests
docker compose exec mist-backend python -m pytest tests/unit/

# Specific test module
docker compose exec mist-backend python -m pytest tests/unit/chat/ -v

# Integration tests (requires live Neo4j + llama-server)
docker compose exec mist-backend python -m pytest tests/integration/

# With coverage
docker compose exec mist-backend python -m pytest --cov=backend tests/
```

### Manual Testing

1. Start the stack: `docker compose up -d` (or `python scripts/start_dev.py`)
2. Verify health: `docker compose logs -f mist-backend`
3. Start the frontend (separate repo): `cd mist-frontend && npm run dev`
4. Exercise the voice / chat flow end-to-end

---

## Pull Request Guidelines

### Before Submitting PR

1. Run all code quality checks (pre-commit + tests)
2. Update documentation if needed
3. Update CODEBASE.md with changes
4. Ensure no emojis in any files
5. Confirm CLAUDE.md / audit-report changes stay local (do NOT push per `feedback_no_push_docs`)

### PR Description

```markdown
## Description
Clear description of what this PR does

## Changes
- List of changes made
- Why these changes were needed

## Testing
- How you tested the changes
- Test results or screenshots

## Checklist
- [ ] Code follows style guidelines
- [ ] No emojis in code or docs
- [ ] Tests pass
- [ ] Documentation updated
- [ ] CODEBASE.md updated
```

### Review Process

1. Automated checks must pass (CI/CD via GitHub Actions)
2. Code review by maintainer
3. Address review comments
4. Squash commits if requested
5. Merge when approved

---

## Common Tasks

### Adding a New Python Module

1. Create module in appropriate directory under `backend/`
2. Add docstring at top of file
3. Use type hints for all functions
4. Write unit tests under `tests/unit/`
5. Update imports in `__init__.py` if needed
6. Run `black` and `ruff` before committing

### Updating Dependencies

```bash
# Add to requirements.txt
pip install package-name
pip freeze | grep package-name >> requirements.txt

# After requirements changes, rebuild the container
docker compose build mist-backend
```

### Working with the Knowledge System

See ADR-010 (`knowledge-vault/Decisions/ADR-010-memory-storage-architecture.md`) for the four-layer memory architecture. Changes to extraction / curation / vault writer must respect the six invariants documented there.

---

## AI-Assisted Development

### Using Claude Code (or similar AI tools)

1. AI must read CLAUDE.md first
2. AI must check CODEBASE.md for context
3. AI must follow NO EMOJIS rule
4. AI must use TodoWrite (or equivalent task tracking) for multi-step work
5. AI must update CODEBASE.md after changes

### Reviewing AI-Generated Code

Always review AI-generated code for:
- Emojis or unicode decorative symbols
- Superlative language ("amazing", "incredible")
- Filler phrases ("let's dive in", "moving forward")
- Over-enthusiasm or marketing language
- Missing type hints or docstrings
- Security issues or anti-patterns

**Use the AI slop detector:**

```bash
python scripts/check_ai_slop.py --critical-only  # Fast check for emojis
python scripts/check_ai_slop.py --fix            # Auto-fix emojis/symbols
```

See [AI Slop Checker Documentation](docs/AI_SLOP_CHECKER.md) for complete usage guide.

---

## Documentation Standards

### Markdown Files

- Use ATX-style headers (`#` not underlines)
- No emojis or decorative unicode
- Use code blocks with language specification
- Keep lines under 100 characters where reasonable
- Use `---` for horizontal rules
- Use `[BRACKETS]` for status indicators

### Code Comments

**When to comment:**
- Complex algorithms or logic
- Workarounds or non-obvious solutions
- TODOs (format: `# TODO(author): description`)
- Important constraints or assumptions

**When NOT to comment:**
- Obvious code that's self-explanatory
- Repeating what the code does
- Outdated information

---

## Troubleshooting

### Pre-commit Hook Issues

**Hook fails:**
- Read the error message carefully
- Fix the issue in your code
- Stage the fixes: `git add <files>`
- Commit again

Do NOT use `--no-verify` to skip hooks. Investigate and fix the underlying issue.

### Black and Ruff Conflicts

Black and Ruff should not conflict. If they do:
1. Run Black first: `black backend/`
2. Then run Ruff: `ruff check backend/ --fix`
3. If still conflicts, check pyproject.toml config

---

## Questions or Issues?

- Check CLAUDE.md for guidelines
- Check CODEBASE.md for current status
- Read existing code for patterns
- Ask in issues or discussions
- Contact maintainers if needed

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License (or Apache 2.0 for modifications to legacy Sesame CSM TTS code in `dependencies/csm/`).

---

**Remember:** Quality over quantity. Well-tested, documented, emoji-free code is always preferred over rushed contributions.
