# Contributing to MIST.AI

Thank you for your interest in contributing to MIST.AI. This document outlines the code quality standards and development workflow.

---

## Before You Start

### Required Reading
1. **CLAUDE.md** - AI integration guidelines and project rules
2. **CODEBASE.md** - Current status and recent changes
3. **README.md** - Project overview and setup

### Setup Development Environment

**Python Backend:**
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (IMPORTANT!)
pip install pre-commit
pre-commit install

# Or use the installation script
bash scripts/install-git-hooks.sh     # Linux/macOS
scripts\install-git-hooks.bat          # Windows
```

See [Git Workflows Guide](docs/GIT_WORKFLOWS.md) for complete setup instructions.

**Flutter Frontend:**
```bash
cd mist_desktop
flutter pub get
```

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
- **Flutter format** - Dart code formatting
- **Flutter analyze** - Dart linting
- **File quality** - YAML/JSON validation, trailing whitespace, etc.

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
All functions should have type hints:
```python
def process_data(input_str: str, count: int = 0) -> list[str]:
    """Process the input string."""
    pass
```

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

### Flutter/Dart Code Style

**Formatting:**
- Follow Effective Dart guidelines
- Use single quotes for strings
- Prefer const constructors
- Require trailing commas for better diffs

**Before Committing:**
```bash
cd mist_desktop

# Format code
flutter format .

# Analyze code
flutter analyze

# Run tests
flutter test
```

**Widget Structure:**
```dart
class MyWidget extends ConsumerWidget {
  const MyWidget({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Read providers first
    final state = ref.watch(myProvider);

    // Build UI
    return Container(
      child: Text(state.value),
    );
  }
}
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

Pre-commit hooks run automatically before each commit. **All 16 hooks must pass before commit succeeds.**

**Python Checks:**
- **Black** - Code formatting (auto-fix)
- **Ruff** - Linting with auto-fix
- **Mypy** - Type checking (namespace packages enabled)
- **Bandit** - Security scanning (with nosec comments)
- **Codespell** - Spell checking

**Dart/Flutter Checks:**
- **Dart format** - Code formatting (auto-fix)
- **Flutter analyze** - Linting (info-level warnings non-fatal)

**General Quality Checks:**
- **AI Slop Checker** - Emoji and unicode symbol detection (auto-fix, runs on changed files only)
- **Trailing whitespace** - Auto-removed
- **End-of-file fixer** - Ensures files end with newline
- **YAML/JSON/TOML validation** - Syntax checking
- **Large file detection** - Prevents commits >1MB
- **Merge conflict markers** - Detects unresolved conflicts
- **Private key detection** - Prevents accidental secret commits
- **Line ending normalization** - Converts to LF (Windows: CRLF)

If hooks fail, fix the issues and commit again.

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

### Python Tests

```bash
# Unit tests
pytest tests/

# Specific test
pytest tests/test_knowledge.py -v

# With coverage
pytest --cov=backend tests/
```

### Flutter Tests

```bash
cd mist_desktop

# All tests
flutter test

# Specific test
flutter test test/widget_test.dart

# With coverage
flutter test --coverage
```

### Manual Testing

1. Start Neo4j: `neo4j start`
2. Start backend: `python backend/server.py`
3. Run Flutter app: `cd mist_desktop && flutter run -d windows`
4. Test voice flow and conversation

---

## Pull Request Guidelines

### Before Submitting PR

1. Run all code quality checks
2. Run all tests (passing)
3. Update documentation if needed
4. Update CODEBASE.md with changes
5. Ensure no emojis in any files

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

1. Automated checks must pass (when CI/CD is set up)
2. Code review by maintainer
3. Address review comments
4. Squash commits if requested
5. Merge when approved

---

## Common Tasks

### Adding a New Python Module

1. Create module in appropriate directory
2. Add docstring at top of file
3. Use type hints for all functions
4. Write tests if applicable
5. Update imports in `__init__.py`
6. Run `black` and `ruff` before committing

### Adding a New Flutter Screen

1. Create screen file in `mist_desktop/lib/screens/`
2. Create provider if needed in `providers/`
3. Follow widget naming conventions
4. Use Riverpod for state management
5. Run `flutter format` and `flutter analyze`

### Updating Dependencies

**Python:**
```bash
# Add to requirements.txt
pip install package-name
pip freeze | grep package-name >> requirements.txt

# Update lockfile
pip install -r requirements.txt
```

**Flutter:**
```bash
cd mist_desktop
# Add to pubspec.yaml
flutter pub add package_name
# Or update
flutter pub upgrade
```

---

## AI-Assisted Development

### Using Claude Code (or similar AI tools)

1. AI must read CLAUDE.md first
2. AI must check CODEBASE.md for context
3. AI must follow NO EMOJIS rule
4. AI must use TodoWrite for multi-step work
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
- Stage the fixes: `git add .`
- Commit again

**Skip hooks (emergency only):**
```bash
git commit --no-verify
# Only use when absolutely necessary!
```

### Black and Ruff Conflicts

Black and Ruff should not conflict. If they do:
1. Run Black first: `black backend/`
2. Then run Ruff: `ruff check backend/ --fix`
3. If still conflicts, check pyproject.toml config

### Flutter Analyze Errors

**Fix common issues:**
```bash
# Remove unused imports
flutter pub run import_sorter:main

# Fix formatting
flutter format .

# See detailed errors
flutter analyze --no-fatal-infos
```

---

## Questions or Issues?

- Check CLAUDE.md for guidelines
- Check CODEBASE.md for current status
- Read existing code for patterns
- Ask in issues or discussions
- Contact maintainers if needed

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License (or Apache 2.0 for modifications to CSM TTS code).

---

**Remember:** Quality over quantity. Well-tested, documented, emoji-free code is always preferred over rushed contributions.
